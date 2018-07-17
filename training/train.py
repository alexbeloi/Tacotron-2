import argparse
import tensorflow as tf
import training.feeder as feeder
import training.hparams as hp
import training.utils as utils
import tacotron.models.estimator as estimator

tf.logging.set_verbosity(tf.logging.INFO)


def _distribution_strategy(hparams):
    distribution = None
    if hparams.distribute:
        num_gpus = hparams.num_gpus
        if num_gpus == 1:
            distribution = tf.contrib.distribute.OneDeviceStrategy(
                device='/gpu:0')
        elif num_gpus > 1:
            distribution = tf.contrib.distribute.MirroredStrategy(
                num_gpus=hparams.num_gpus)
    return distribution


def run_experiment(train_files, eval_files, warm_start_from, hparams):
    train_dataset = lambda: feeder.input_fn(train_files,
                                            shuffle=True,
                                            )

    eval_dataset = lambda:  feeder.input_fn(eval_files,
                                            shuffle=False,
                                            )

    train_spec = tf.estimator.TrainSpec(train_dataset,
                                        max_steps=hparams.train_steps
                                        )

    exporters = [
        exporter = tf.estimator.LatestExporter(
            'tacotron_' + key,
            serving_input_fn,
            exports_to_keep=20,
        ) for key, serving_input_fn in feeder.SERVING_FUNCTIONS.items()
    ]

    eval_spec = tf.estimator.EvalSpec(eval_dataset,
                                      steps=hparams.eval_steps,
                                      exporters=exporters
                                      name='tacotron-eval',
                                      throttle_secs=hparams.eval_throttle_secs,
                                      )

    distribution = _distribution_strategy(hparams)

    run_config = tf.estimator.RunConfig(
        model_dir=hparams.job_dir,
        save_summary_steps=10,
        log_step_count_steps=10,
        train_distribute=distribution,
    )

    _estimator = tf.estimator.Estimator(
        model_fn=estimator.estimator_fn,
        params=hparams,
        config=run_config,
        warm_start_from=warm_start_from,
    )
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
        '--warm-start-from',
        type=str,
        default=None,
        help='GCS or local path to restore from for initialization',
        required=False,
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
    parser.add_argument(
        '--hparams',
        type=str,
        default='',
        help='Comma separated list of "name=value" pairs.',
        required=False,
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    hparams = hp.hparams
    hparams.parse(args.hparams)
    hparams.set_hparam('job_dir', args.job_dir)
    if args.profile:
        utils.profile(args.profile_dir)(
            run_experiment)(args.train_files, args.eval_files,
                            args.warm_start_from,  hparams)
    else:
        run_experiment(args.train_files, args.eval_files,
                       args.warm_start_from, hparams)
