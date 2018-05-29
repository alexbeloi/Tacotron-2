import os
import errno
import argparse
import functools
import concurrent.futures as futures
import functools

import tqdm
import tensorflow as tf
import numpy as np

import preprocessing.utils as utils
import utils.audio
import utils.text
import utils.cleaners

import training.hparams as hp


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_featurelist(values):
    return tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=v))
        for v in values
    ])


def _float_featurelist(values):
    return tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=v))
        for v in values
    ])


def create_dataset(input_dir, metadata_file, output_dir, hparams='',
                   max_examples_per_file=128, n_jobs=8):
    modified_hp = hp.hparams.parse(hparams)

    executor = futures.ProcessPoolExecutor(max_workers=n_jobs)
    work_items = []
    with open(metadata_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(input_dir, 'wavs',
                           '{}.wav'.format(parts[0]))
            text = parts[2]
            work_items.append((wav_path, text, modified_hp))

    work_batches = [work_items[i:i + max_examples_per_file] for i in
                    range(0, len(work_items), max_examples_per_file)]

    mkdir_p(output_dir)
    results = []
    for i, work_batch in enumerate(work_batches):
        output_path = os.path.join(output_dir, 'dataset_%03d.tfrecords' % i)
        _partial = functools.partial(_write_examples, work_batch,
                                            output_path)
        results.append(executor.submit(_partial))

    for r in tqdm.tqdm(futures.as_completed(results),
                       total=len(work_batches)):
        r.result()


def _write_examples(items, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    examples = []
    for args in items:
        examples.append(make_example(*args))
    for example in examples:
        writer.write(example.SerializeToString())


def make_example(wav_path, text, hparams):
    try:
        # Load the audio as numpy array
        wav = utils.audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    #rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    #M-AILABS extra silence specific
    if hparams.trim_silence:
        wav = utils.audio.trim_silence(wav, hparams)

    #Mu-law quantize
    if utils.audio.is_mulaw_quantize(hparams.input_type):
        #[0, quantize_channels)
        out = utils.audio.mulaw_quantize(wav, hparams.quantize_channels)

        #Trim silences
        start, end = utils.audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        out = out[start: end]

        constant_values = utils.audio.mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16

    elif utils.audio.is_mulaw(hparams.input_type):
        #[-1, 1]
        out = utils.audio.mulaw(wav, hparams.quantize_channels)
        constant_values = utils.audio.mulaw(0., hparams.quantize_channels)
        out_dtype = np.float32

    else:
        #[-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = utils.audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    #Compute the linear scale spectrogram from the wav
    linear_spectrogram = utils.audio.linearspectrogram(wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    #sanity check
    assert linear_frames == mel_frames

    #Ensure time resolution adjustement between audio and mel-spectrogram
    l, r = utils.audio.pad_lr(wav, hparams.fft_size, utils.audio.get_hop_size(hparams))

    #Zero pad for quantized signal
    out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    time_steps = len(out)
    assert time_steps >= mel_frames * utils.audio.get_hop_size(hparams)

    #time resolution adjustement
    #ensure length of raw audio is multiple of hop size so that we can use
    #transposed convolution to upsample
    out = out[:mel_frames * utils.audio.get_hop_size(hparams)]
    assert time_steps % utils.audio.get_hop_size(hparams) == 0

    #clean text
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    text_clean = np.asarray(utils.text.text_to_sequence(text, cleaner_names),
                            dtype=np.int32)

    context_features = {
        'text_length': _int64_feature([len(text_clean)]),
        'time_steps': _int64_feature([time_steps]),
        'mel_frames': _int64_feature([mel_frames]),
        'linear_frames': _int64_feature([linear_frames]),
    }
    sequence_features = {
        'mel_spectrogram': _float_featurelist(mel_spectrogram.T),
        'linear_spectrogram': _float_featurelist(linear_spectrogram.T),
        'audio': _float_featurelist(out.reshape((-1, 1))),
        'text': _int64_featurelist(text_clean.reshape((-1, 1))),
        'stop_token': _float_featurelist(
            np.zeros((mel_spectrogram.shape[1], 1), dtype=np.float32)),
    }


    # create Example
    return tf.train.SequenceExample(
        context=tf.train.Features(feature=context_features),
        feature_lists=tf.train.FeatureLists(feature_list=sequence_features))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str,
                        help='Path to dataset directory')
    parser.add_argument('metadata_file', type=str,
                        help='Path to metadata file')
    parser.add_argument('output_dir', type=str,
                        help='Path to write tfrecords files')
    parser.add_argument('--hparams', type=str, default='',
                        help='override string for hparams')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_dataset(args.input_dir,
                   args.metadata_file,
                   args.output_dir,
                   args.hparams)
