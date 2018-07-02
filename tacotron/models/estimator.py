import tensorflow as tf
import tacotron.models.tacotron as tacotron


def create_summaries(model, hparams, is_training=False, is_evaluating=True):
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    if is_training or is_evaluating:
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('before_loss', model.before_loss)
        tf.summary.scalar('after_loss', model.after_loss)
        tf.summary.scalar('regularization_loss', model.regularization_loss)
        tf.summary.scalar('stop_token_loss', model.stop_token_loss)
        tf.summary.scalar('loss', model.loss)
        if hparams.tacotron_teacher_forcing_mode == 'scheduled':
            tf.summary.scalar('teacher_forcing_ratio', model.ratio)
        if is_training:
            tf.summary.scalar('learning_rate', model.learning_rate)
        alignments = model.alignments
        images = []
        for i in range(hparams.tacotron_batch_size):
            image = tf.expand_dims(
                alignments[i, :, :model.targets_lengths[i]], -1)
            image_resized = tf.image.resize_images(image, [256, 256])
            images.append(image_resized)
        images_stacked = tf.stack(images)

        tf.summary.image('alignments',
                         images_stacked,
                         family='Images')


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

    global_step = tf.train.get_or_create_global_step()
    print(features)
    model.initialize(
        features['sequence_features.text'],
        features['context_features.text_length'],
        mel_targets=features.get('sequence_features.mel_spectrogram'),
        stop_token_targets=features.get('sequence_features.stop_token'),
        targets_lengths=features.get('context_features.mel_frames'),
        global_step=global_step,
        is_training=is_training,
        is_evaluating=is_evaluating,
        )
    if is_training or is_evaluating:
        model.add_loss()
        if is_training:
            model.add_optimizer(global_step)

    create_summaries(model, hparams, is_training=is_training,
                     is_evaluating=is_evaluating)

    export_outputs = {
        'predict': tf.estimator.export.PredictOutput({
            'mels': model.mel_outputs,
            'alignments': model.alignments,
            'stop_token': model.stop_token_prediction,
        }),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=model.mel_outputs,
        loss=model.loss,
        train_op=model.optimize,
        export_outputs=export_outputs,
    )
