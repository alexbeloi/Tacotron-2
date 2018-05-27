from __future__ import print_function, division, absolute_import
import argparse
import os
import concurrent.futures as futures
import tqdm
import numpy as np
import tensorflow as tf

import preprocessing.audio as audio


def create_dataset(input_dirs, metadata_file, hparams, output_dir, n_jobs=12,
                  examples_per_file=512):
    process_executor = futures.ProcessPoolExecutor(max_workers=n_jobs)
    write_executor = futures.ProcessPoolExecutor(max_workers=n_jobs)

    for input_dir in input_dirs:
        with open(os.path.join(input_dir, metadata_file), encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                wav_path = os.path.join(input_dir,
                                        'wavs',
                                        '{}.wav'.format(parts[0]))
                text = parts[2]
                examples.append(process_executor.submit(make_example,
                                                (wav_path, text, hparams))
                )

    examples_batches = [examples[i:i + examples_per_file]
                       for i in range(0, len(examples), examples_per_file)]
    for i, examples_batch in enumerate(examples_batches):
        output_path = os.path.join(output_dir, 'dataset_%5d.tfrecords')
        results = write_executor.submit(
            write_examples, (examples_batch, output_path))

    for _ in tqdm.tqdm(as_completed(results), total=len(examples_batches)):
        pass


def write_examples(examples, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    for example in examples:
        writer.write(example.SerializeToString())



def make_example(wav_path, text, hparams):
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#Mu-law quantize
	if is_mulaw_quantize(hparams.input_type):
		#[0, quantize_channels)
		out = mulaw_quantize(wav, hparams.quantize_channels)

		#Trim silences
		start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		wav = wav[start: end]
		out = out[start: end]

		constant_values = mulaw_quantize(0, hparams.quantize_channels)
		out_dtype = np.int16

	elif is_mulaw(hparams.input_type):
		#[-1, 1]
		out = mulaw(wav, hparams.quantize_channels)
		constant_values = mulaw(0., hparams.quantize_channels)
		out_dtype = np.float32

	else:
		#[-1, 1]
		out = wav
		constant_values = 0.
		out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
		return None

	#Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	#Ensure time resolution adjustement between audio and mel-spectrogram
	l, r = audio.pad_lr(wav, hparams.fft_size, audio.get_hop_size(hparams))

	#Zero pad for quantized signal
	out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	time_steps = len(out)
	assert time_steps >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert time_steps % audio.get_hop_size(hparams) == 0

    #clean text
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    text_clean = np.asarray(text_to_sequence(text, cleaner_names),
                            dtype=np.int32)

    features = {
        'mel_spectrogram': tf.train.Feature(
            float_list=tf.train.FloatList(mel_spectrogram.T.reshape(-1))),
        'linear_spectrogram': tf.train.Feature(
            float_list=tf.train.FloatList(linear_spectrogram.T.reshape(-1))),
        'audio': tf.train.Feature(
            float_list=tf.train.FloatList(out)),
        'text': tf.train.Feature(
            int64_list=tf.train.Int64List(text_clean)),
        'time_steps': tf.train.Feature(
            int64_list=tf.train.Int64List(time_steps)),
        'mel_frames': tf.train.Feature(
            int64_list=tf.train.Int64List(mel_frames)),
        'linear_frames': tf.train.Feature(
            int64_list=tf.train.Int64List(linear_frames)),
    }

    # create Example
    return tf.train.Example(features=tf.train.Features(features=features))


