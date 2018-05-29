import tensorflow as tf

import tacotron.models.tacotron as tacotron


def estimator_fn(features,
                 labels,
                 mode=None,
                 params=None,
                 config=None,
                 ):
    hparams = params
    model = tacotron.Tacotron(hparams)

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_evaluating = mode == tf.estimator.ModeKeys.EVAL

    context_features = features['context_features']
    sequence_features = features['sequence_features']

    model.initialize(sequence_features['text'],
                     features['context_features']['text_length'],
                     mel_targets=sequence_features['mel_spectrogram'],
                     stop_token_targets=sequence_features['stop_token'],
                     targets_lengths=context_features['mel_frames'],
                     gta=True,
                     global_step=tf.train.get_global_step(),
                     is_training=is_training,
                     is_evaluating=is_evaluating,
                     )
    model.add_loss()
    model.add_optimizer(tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=model.mel_outputs,
        loss=model.loss,
        train_op=model.optimize,
        eval_metric_ops=None,
    )
