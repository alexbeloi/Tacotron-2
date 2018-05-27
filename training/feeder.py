import tensorflow as tf

_batches_per_group = 32

LABEL_COLUMN = 'mel_spectrogram'
INPUT_COLUMNS = ['text']


def parse_example(example):
    features = {}
    features['mel_spectrogram'] = tf.VarLenFeature([], dtype=tf.float32)
    features['linear_spectrogram'] = tf.VarLenFeature([], dtype=tf.float32)
    features['audio'] = tf.VarLenFeature([], dtype=tf.float32)
    features['text'] = tf.VarLenFeature([], dtype=tf.int64)
    features['text_length'] = tf.FixedLenFeature([], dtype=tf.int64)
    features['time_steps'] = tf.FixedLenFeature([], dtype=tf.int64)
    features['mel-frames'] = tf.FixedLenFeature([], dtype=tf.int64)
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
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()

        return iterator


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
