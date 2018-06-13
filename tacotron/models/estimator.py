import tensorflow as tf
import tacotron.models.tacotron as tacotron


def train_summaries(model, hparams):
    tf.summary.histogram('train/mel_outputs', model.mel_outputs)
    tf.summary.histogram('train/mel_targets', model.mel_targets)
    tf.summary.scalar('train/before_loss', model.before_loss)
    tf.summary.scalar('train/after_loss', model.after_loss)
    # if hparams.predict_linear:
    #     tf.summary.scalar('train/linear_loss', model.linear_loss)
    tf.summary.scalar('train/regularization_loss', model.regularization_loss)
    tf.summary.scalar('train/stop_token_loss', model.stop_token_loss)
    tf.summary.scalar('train/loss', model.loss)
    tf.summary.scalar('train/learning_rate', model.learning_rate)
    if hparams.tacotron_teacher_forcing_mode == 'scheduled':
        tf.summary.scalar('train/teacher_forcing_ratio', model.ratio)
    # gradient_norms = [tf.norm(grad) for grad in model.gradients]
    # tf.summary.histogram('train/gradient_norm', gradient_norms)
    # tf.summary.scalar('train/max_gradient_norm',
    #                    tf.reduce_max(gradient_norms))


def eval_summaries(model, hparams):
    tf.summary.scalar('eval_model/eval_stats/eval_before_loss',
                      model.before_loss)
    tf.summary.scalar('eval_model/eval_stats/eval_after_loss',
                      model.after_loss)
    tf.summary.scalar('eval_model/eval_stats/stop_token_loss',
                      model.stop_token_loss)
    tf.summary.scalar('eval_model/eval_stats/eval_loss', model.loss)
    # if model.linear_loss is not None:
    #     tf.summary.scalar('model/eval_stats/eval_linear_loss',
    #                       model.linear_loss)
    alignments = model.alignments
    images = []
    for i in range(hparams.tacotron_batch_size):
        image = alignments[i, :, :model.targets_lengths[i]]
        image_resized = tf.image.resize_images(image, [256, 256])
        images.append(image_resized)
    images_stacked = tf.stack(images)

    tf.summary.image('alignments',
                     tf.expand_dims(images_stacked, -1),
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
    model.initialize(features['sequence/text'],
                     features['context/text_length'],
                     mel_targets=features['sequence/mel_spectrogram'],
                     stop_token_targets=features['sequence/stop_token'],
                     targets_lengths=features['context/mel_frames'],
                     global_step=global_step,
                     is_training=is_training,
                     is_evaluating=is_evaluating,
                     )
    model.add_loss()
    model.add_optimizer(global_step)

    if is_training:
        print('Training evaluation summaries to graph')
        train_summaries(model, hparams)
    elif is_evaluating:
        print('Adding evaluation summaries to graph')
        eval_summaries(model, hparams)

    eval_summary_hook = tf.train.SummarySaverHook(
        save_steps=1,
        output_dir=params.job_dir,
        summary_op=tf.summary.merge_all())

    evaluation_hooks = [eval_summary_hook]

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=model.mel_outputs,
        loss=model.loss,
        train_op=model.optimize,
        evaluation_hooks=evaluation_hooks,
    )
