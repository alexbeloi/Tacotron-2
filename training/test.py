import argparse
import tensorflow as tf

import training.feeder as feeder
import training.hparams as hparams
import tacotron.models.estimator as estimator


import datetime

def run_experiment(train_files, hparams):
    dataset = feeder.input_fn(train_files,
                                    #hparams.num_epochs,
                                    shuffle=True,
                                    )

    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()

    s = datetime.datetime.now()
    for _ in range(10):
        with tf.Session() as sess:
            sess.run(el)
    e = datetime.datetime.now()
    print((e - s).total_seconds() * 100)

    s = datetime.datetime.now()
    for _ in range(10):
        with tf.Session() as sess:
            sess.run(el)
    e = datetime.datetime.now()
    print((e - s).total_seconds() * 100)

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

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args.train_files, hparams.hparams)
