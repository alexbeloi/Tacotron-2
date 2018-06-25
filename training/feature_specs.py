import numpy as np
import tensorflow as tf
import training.hparams as hp


DEFAULT_VALUES = {
    'context_features': {
        'text_length': np.int64(0),
        'mel_frames': np.int64(0),
    },
    'sequence_features': {
        'mel_spectrogram': 0.,
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
    'context_features.text_length': [],
    'context_features.mel_frames': [],
    'sequence_features.mel_spectrogram': [None, hp.hparams.num_mels],
    'sequence_features.text': [None],
    'sequence_features.stop_token': [None],
}

PADDING_VALUES = {
    'context_features.text_length': np.int64(0),
    'context_features.mel_frames': np.int64(0),
    'sequence_features.mel_spectrogram': 0.,
    'sequence_features.text': np.int64(0),
    'sequence_features.stop_token': 1.,
}
