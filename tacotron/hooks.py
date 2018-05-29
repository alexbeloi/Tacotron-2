import datetime
import tensorflow as tf


class TimeHook(tf.train.SessionRunHook):
    def __init__(self):
        pass

    def begin(self):
        pass

    def before_run(self, run_context):
        self.s = datetime.datetime.now()

    def after_run(self, run_context, run_values):
        print("Elapsed seconds:",
              (datetime.datetime.now() - self.s).total_seconds())
