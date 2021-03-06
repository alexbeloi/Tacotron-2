import tensorflow as tf
import tacotron.models.tacotron as tacotron


def create_summaries(model, hparams, is_training=False, is_evaluating=True):
    if is_training:
        prefix = 'train/'
    elif is_evaluating:
        prefix = 'eval/'
    else:
        prefix = ''

    tf.summary.histogram(prefix + 'mel_outputs', model.mel_outputs)
    if is_training or is_evaluating:
        tf.summary.histogram(prefix + 'mel_targets', model.mel_targets)
        tf.summary.scalar(prefix + 'before_loss', model.before_loss)
        tf.summary.scalar(prefix + 'after_loss', model.after_loss)
        tf.summary.scalar(prefix + 'stop_token_loss', model.stop_token_loss)
        tf.summary.scalar(prefix + 'loss', model.loss)
        if is_training:
            if hparams.tacotron_teacher_forcing_mode == 'scheduled':
                tf.summary.scalar(
                    prefix + 'teacher_forcing_ratio', model.ratio)
            tf.summary.scalar(prefix + 'learning_rate', model.learning_rate)
            tf.summary.scalar(prefix + 'dropout_rate', model.dropout_rate)
            tf.summary.scalar(prefix + 'zoneout_rate', model.zoneout_rate)

        alignments = model.alignments
        images = []
        for i in range(hparams.tacotron_batch_size):
            image = tf.expand_dims(
                alignments[i, :, :model.targets_lengths[i]], -1)
            image_resized = tf.image.resize_images(image, [256, 256])
            images.append(image_resized)
        images_stacked = tf.stack(images)

        tf.summary.image(prefix + 'alignments',
                         images_stacked)


def estimator_fn(features,
                 labels,
                 mode=None,
                 params=None,
                 ):
    hparams = params
    model = tacotron.Tacotron(hparams)

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_evaluating = mode == tf.estimator.ModeKeys.EVAL

    global_step = tf.train.get_or_create_global_step()

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

    # Create a SummarySaverHook
    eval_summary_hook = tf.train.SummarySaverHook(
                                        save_steps=10,
                                        output_dir=hparams.job_dir + "/eval",
                                        summary_op=tf.summary.merge_all())

    predictions = {
        'mel_output': model.mel_outputs,
        'stop_token': model.stop_token_prediction,
        'alignments': model.alignments,
    }
    export_outputs = {
        'predict': tf.estimator.export.PredictOutput(predictions),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=model.loss,
        train_op=model.optimize,
        export_outputs=export_outputs,
        evaluation_hooks=[eval_summary_hook],
    )
