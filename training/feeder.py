import tensorflow as tf
import training.hparams as hp
import training.feature_specs as specs


def parse_example(example):
    sequence_features = {
        'mel_spectrogram': tf.FixedLenSequenceFeature(
            [hp.hparams.num_mels], dtype=tf.float32),
        'linear_spectrogram': tf.FixedLenSequenceFeature(
            [hp.hparams.num_freq], dtype=tf.float32),
        'audio': tf.FixedLenSequenceFeature(
            [], dtype=tf.float32),
        'text': tf.FixedLenSequenceFeature(
            [], dtype=tf.int64),
        'stop_token': tf.FixedLenSequenceFeature(
            [], dtype=tf.float32),
    }
    context_features = {
        'text_length': tf.FixedLenFeature([], dtype=tf.int64),
        'time_steps': tf.FixedLenFeature([], dtype=tf.int64),
        'mel_frames': tf.FixedLenFeature([], dtype=tf.int64),
        'linear_frames': tf.FixedLenFeature([], dtype=tf.int64),
    }

    con_feats_parsed, seq_feats_parsed = tf.parse_single_sequence_example(
        example,
        context_features=context_features,
        sequence_features=sequence_features)

    for k, v in con_feats_parsed.items():
        print(k, v)
    for k, v in seq_feats_parsed.items():
        print(k, v)

    return {'context_features': con_feats_parsed,
            'sequence_features': seq_feats_parsed}


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             batch_size=hp.hparams.tacotron_batch_size):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_example)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)
        if num_epochs:
            dataset = dataset.repeat(num_epochs)
        batch_fn = tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size=batch_size,
            padded_shapes=specs.PADDED_SHAPES,
            padding_values=specs.PADDING_VALUES,
        )
        dataset = dataset.apply(batch_fn)

        return dataset


def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    features = parse_example(example_bytestring)
    return tf.estimator.export.ServingInputReceiver(
        features,
        {'example_proto': example_bytestring}
    )


# [START serving-function]
def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in specs.INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
# [END serving-function]


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
}
