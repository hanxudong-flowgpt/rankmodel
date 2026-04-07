from util.tflog import tflogger as logging
import tensorflow as tf


def restore_from_saved_model(sess, saved_model_dir):
    loaded = tf.saved_model.load(sess=sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=saved_model_dir)
    init_ops = []
    for v in loaded.trainable_variables:
        origin_v = tf.get_variable(v.name)
        logging.info('sm variable:', v.name)
        init_ops.append(tf.assign(origin_v, v))
    sess.run(tf.group(init_ops))
