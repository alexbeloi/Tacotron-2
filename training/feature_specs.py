import numpy as np
import tensorflow as tf
import training.hparams as hp


DEFAULT_VALUES = {
    'context_features': {
        'text_length': np.int64(0),
        # 'time_steps': np.int64(0),
        'mel_frames': np.int64(0),
        # 'linear_frames': np.int64(0),
    },
    'sequence_features': {
        'mel_spectrogram': 0.,
        # 'linear_spectrogram': 0.,
        # 'audio': 0.,
        'text': np.int64(0),
        'stop_token': 0.,
    },
}

SPARES_FEATURES = [
    'mel_spectrogram',
    'linear_spectrogram',
    'audio',
    'text',
    'stop_token',
]

PADDED_SHAPES = {
    'context_features': {
        'text_length': [],
        # 'time_steps': [],
        'mel_frames': [],
        # 'linear_frames': [],
    },
    'sequence_features': {
        'mel_spectrogram': [None, hp.hparams.num_mels],
        # 'linear_spectrogram': [None, hp.hparams.num_freq],
        # 'audio': [None],
        'text': [None],
        'stop_token': [None],
    },
}

PADDING_VALUES = {
    'context_features': {
        'text_length': np.int64(0),
        # 'time_steps': np.int64(0),
        'mel_frames': np.int64(0),
        # 'linear_frames': np.int64(0),
    },
    'sequence_features': {
        'mel_spectrogram': 0.,
        # 'linear_spectrogram': 0.,
        # 'audio': 0.,
        'text': np.int64(0),
        'stop_token': 1.,
    },
}

INPUT_COLUMNS = [tf.placeholder(tf.string, [None], name='text')]
