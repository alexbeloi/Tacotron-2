import tensorflow as tf
import tacotron.hooks as hooks
import tacotron.models.tacotron as tacotron


def train_summaries(model, hparams):
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('before_loss', model.before_loss)
    tf.summary.scalar('after_loss', model.after_loss)
    if hparams.predict_linear:
        tf.summary.scalar('linear_loss', model.linear_loss)
    tf.summary.scalar('regularization_loss', model.regularization_loss)
    tf.summary.scalar('stop_token_loss', model.stop_token_loss)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    if hparams.tacotron_teacher_forcing_mode == 'scheduled':
        tf.summary.scalar('teacher_forcing_ratio', model.ratio)
    # gradient_norms = [tf.norm(grad) for grad in model.gradients]
    # tf.summary.histogram('gradient_norm', gradient_norms)
    # tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))


def eval_summaries(model, hparams):
    tf.summary.scalar('eval_model/eval_stats/eval_before_loss',
                      model.before_loss)
    tf.summary.scalar('eval_model/eval_stats/eval_after_loss',
                      model.after_loss)
    tf.summary.scalar('eval_model/eval_stats/stop_token_loss',
                      model.stop_token_loss)
    tf.summary.scalar('eval_model/eval_stats/eval_loss', model.loss)
    if model.linear_loss is not None:
        tf.summary.scalar('model/eval_stats/eval_linear_loss',
                          model.linear_loss)
    tf.summary.image('alignment', model.alignment, family='Images')


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

    global_step = tf.train.get_or_create_global_step()
    model.initialize(sequence_features['text'],
                     features['context_features']['text_length'],
                     mel_targets=sequence_features['mel_spectrogram'],
                     stop_token_targets=sequence_features['stop_token'],
                     targets_lengths=context_features['mel_frames'],
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
        test_summaries(model, hparams)

    plots_hook = tf.train.SummarySaverHook(
        save_step=1,
        output_dir=params.job_dir,
        summary_op=tf.summary.merge_all())

    evaluation_hooks = [plot_hook]


    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=model.mel_outputs,
        loss=model.loss,
        train_op=model.optimize,
        evaluation_hooks=evaluation_hooks,
    )
