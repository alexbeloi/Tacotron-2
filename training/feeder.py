import tensorflow as tf

_batches_per_group = 32

LABEL_COLUMN = 'mel_spectrogram'
INPUT_COLUMNS = ['text']

padded_shapes = {
    'mel_spectrogram': [None],
    'linear_spectrogram': [None],
    'audio': [None],
    'text': [None],
    'text_length': [1],
    'time_steps': [1],
    'mel_frames': [1],
    'linear_frames': [1],
}

padding_values = {
    'mel_spectrogram': 0,
    'linear_spectrogram': 0,
    'audio': 0,
    'text': 0,
}


def parse_example(example):
    features = {}
    features['mel_spectrogram'] = tf.FixedLenFeature([], dtype=tf.float32)
    features['linear_spectrogram'] = tf.FixedLenFeature([], dtype=tf.float32)
    features['audio'] = tf.FixedLenFeature([], dtype=tf.float32)
    features['text'] = tf.FixedLenFeature([], dtype=tf.int64)
    features['text_length'] = tf.FixedLenFeature([], dtype=tf.int64)
    features['time_steps'] = tf.FixedLenFeature([], dtype=tf.int64)
    features['mel_frames'] = tf.FixedLenFeature([], dtype=tf.int64)
    features['linear_frames'] = tf.FixedLenFeature([], dtype=tf.int64)

    return tf.parse_single_example(example, features=features)


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             batch_size=32):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_example)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.padded_batch(batch_size)

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
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
# [END serving-function]


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
}
