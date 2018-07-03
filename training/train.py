import argparse
import tensorflow as tf
import training.feeder as feeder
import training.hparams as hp
import training.utils as utils
import tacotron.models.estimator as estimator

tf.logging.set_verbosity(tf.logging.INFO)


def run_experiment(train_files, eval_files, hparams):
    train_dataset = lambda: feeder.input_fn(train_files,
                                            shuffle=True,
                                            )

    eval_dataset = lambda:  feeder.input_fn(eval_files,
                                            shuffle=False,
                                            )

    train_spec = tf.estimator.TrainSpec(train_dataset,
                                        max_steps=hparams.train_steps
                                        )

    exporter = tf.estimator.LatestExporter(
        'tacotron',
        feeder.SERVING_FUNCTIONS[hparams.export_format],
        exports_to_keep=20,
    )

    eval_spec = tf.estimator.EvalSpec(eval_dataset,
                                      steps=hparams.eval_steps,
                                      exporters=[exporter],
                                      name='tacotron-eval',
                                      throttle_secs=hparams.eval_throttle_secs,
                                      )

    run_config = tf.estimator.RunConfig(
        model_dir=hparams.job_dir,
        save_summary_steps=1,
        log_step_count_steps=1,
    )

    _estimator = tf.estimator.Estimator(model_fn=estimator.estimator_fn,
                                        params=hparams,
                                        config=run_config)
    tf.estimator.train_and_evaluate(_estimator, train_spec, eval_spec)


def parse_args():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-files',
        type=str,
        help='GCS or local path glob to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--eval-files', type=str,
        help='GCS or local path glob to validation data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local paths to save run files and output',
        required=True,
    )
    parser.add_argument(
        '--profile',
        type=bool,
        default=False,
        help='Attach profiler to main training loop',
        required=False,
    )
    parser.add_argument(
        '--profile-dir',
        type=str,
        default=None,
        help='Path to write profiler ouputs',
        required=False,
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    hparams = hp.hparams.override_from_dict({'job_dir': args.job_dir})
    if args.profile:
        utils.profile(args.profile_dir)(
            run_experiment)(args.train_files, args.eval_files, hparams)
    else:
        run_experiment(args.train_files, args.eval_files, hparams)
