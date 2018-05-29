import argparse
import tensorflow as tf

import training.feeder as feeder
import training.hparams as hparams
import tacotron.models.estimator as estimator


def run_experiment(train_files, eval_files, hparams):
    dataset = feeder.input_fn(train_files,
                                    #hparams.num_epochs,
                                    shuffle=True,
                                    )

    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()

    with tf.Session() as sess:
            print(sess.run(el)) # output: [ 0.42116176  0.40666069]

def parse_args():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-files',
        type=str,
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local paths to save run files and output',
        required=True,
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args.train_files, args.train_files, hparams.hparams)
