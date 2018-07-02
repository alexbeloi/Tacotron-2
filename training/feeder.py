import os
import json
import multiprocessing as mp
import tensorflow as tf
import training.hparams as hp
import training.feature_specs as specs


tf_config = os.environ.get('TF_CONFIG', '')
try:
    tf_config_json = json.loads(tf_config)
except json.decoder.JSONDecodeError:
    tf_config_json = {}


def parse_example(example):
    sequence_features = {
        'mel_spectrogram': tf.FixedLenSequenceFeature(
            [hp.hparams.num_mels], dtype=tf.float32),
        'text': tf.FixedLenSequenceFeature(
            [], dtype=tf.int64),
        'stop_token': tf.FixedLenSequenceFeature(
            [], dtype=tf.float32),
    }
    context_features = {
        'text_length': tf.FixedLenFeature([], dtype=tf.int64),
        'mel_frames': tf.FixedLenFeature([], dtype=tf.int64),
    }

    con_feats_parsed, seq_feats_parsed = tf.parse_single_sequence_example(
        example,
        context_features=context_features,
        sequence_features=sequence_features)

    features = {}
    for k, f in seq_feats_parsed.items():
        features['sequence_features.' + k] = f
    for k, f in con_feats_parsed.items():
        features['context_features.' + k] = f

    return features


def input_fn(glob,
             num_epochs=None,
             shuffle=True,
             batch_size=hp.hparams.tacotron_batch_size):
        files = tf.matching_files(glob)
        dataset = tf.data.TFRecordDataset(files,
                                          num_parallel_reads=mp.cpu_count(),
                                          )
        dataset = dataset.shard(
            int(tf_config_json.get('job', {}).get('worker_count', 1)),
            int(tf_config_json.get('task', {}).get('index', 0)))

        dataset = dataset.repeat(num_epochs)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.map(parse_example,
                              num_parallel_calls=mp.cpu_count())
        batch_fn = tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size=batch_size,
            padded_shapes=specs.PADDED_SHAPES,
            padding_values=specs.PADDING_VALUES,
        )
        dataset = dataset.apply(batch_fn)
        dataset = dataset.prefetch(buffer_size=hp.hparams.prefetch_buffer_size)
        return dataset


def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestrings = tf.placeholder(
        shape=None,
        dtype=tf.string,
    )
    features = parse_example(example_bytestrings)
    for k, t in features.items():
        features[k] = tf.expand_dims(t, 0)
    return tf.estimator.export.ServingInputReceiver(
        features,
        example_bytestrings,
    )


# [START serving-function]
def json_serving_input_fn():
    """Build the serving inputs."""
    receiver_inputs = {
        'text': tf.placeholder(shape=[None, None], dtype=tf.int32,
        name='inputs'),
    }
    text = tf.cast(receiver_inputs['text'], tf.int64)
    lengths = tf.map_fn(lambda t: tf.size(t, out_type=tf.int64), text,
                        dtype=tf.int64)
    inputs = {
        'context_features.text_length': lengths,
        'sequence_features.text': text,
    }

    return tf.estimator.export.ServingInputReceiver(inputs, receiver_inputs)
# [END serving-function]


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
}
