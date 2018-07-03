import logging
import tempfile
import tensorflow as tf


def profile(output_path=None):
    """Wraps tensorflow func in time, memory, parameter profiler"""
    if output_path is None:
        output_path = tempfile.mkdtemp()
    logging.info('Writing tensorflow profiler outputs to %s' % output_path)

    def _profile(func):
        def profile_and_call(*args, **kwargs):
            builder = tf.profiler.ProfileOptionBuilder
            op = builder(builder.time_and_memory()).order_by(
                'micros').build()
            op2 = builder.trainable_variables_parameter()

            # Collect traces of steps 10~20
            with tf.contrib.tfprof.ProfileContext(output_path,
                                                  trace_steps=range(10, 20),
                                                  dump_steps=[20]) as pctx:
                pctx.add_auto_profiling('op', op, [15, 18, 20])
                pctx.add_auto_profiling('scope', op2, [20])
                func(*args, **kwargs)

        return profile_and_call

    return _profile
