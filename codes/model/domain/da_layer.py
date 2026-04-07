import tensorflow as tf


def auto_dial(tensor_a, tensor_b, training, alpha, name, epsilon=1e-8):
    return tf.cond(training,
                   lambda: domain_adapt_layer(tensor_a, tensor_b, True, alpha, name, epsilon),
                   lambda: domain_adapt_layer(tensor_a, tensor_b, False, alpha, name, epsilon)
                   )


def domain_adapt_layer(tensor_a, tensor_b, training, alpha, name, epsilon):
    shape = tensor_a.get_shape().as_list()
    moving_mean_a = tf.get_variable(name + "_moving_mean_a", shape[1:], initializer=tf.constant_initializer(0.0),
                                    trainable=False)
    moving_var_a = tf.get_variable(name + "_moving_var_a", shape[1:], initializer=tf.constant_initializer(1.0),
                                   trainable=False)

    shape = tensor_b.get_shape().as_list()
    moving_mean_b = tf.get_variable(name + "_moving_mean_b", shape[1:], initializer=tf.constant_initializer(0.0),
                                    trainable=False)
    moving_var_b = tf.get_variable(name + "_moving_var_b", shape[1:], initializer=tf.constant_initializer(1.0),
                                   trainable=False)

    if training:
        batch_mean_a, batch_var_a = tf.nn.moments(tensor_a, [0])
        batch_mean_b, batch_var_b = tf.nn.moments(tensor_b, [0])

        update_moving_avg_a = tf.assign(moving_mean_a, moving_mean_a * 0.999 + batch_mean_a * 0.001)
        update_moving_var_a = tf.assign(moving_var_a, moving_var_a * 0.999 + batch_var_a * 0.001)
        update_moving_avg_b = tf.assign(moving_mean_b, moving_mean_b * 0.999 + batch_mean_b * 0.001)
        update_moving_var_b = tf.assign(moving_var_b, moving_var_b * 0.999 + batch_var_b * 0.001)

        control_inputs = [update_moving_avg_a, update_moving_var_a, update_moving_avg_b, update_moving_var_b]
    else:
        batch_mean_a, batch_var_a = moving_mean_a, moving_var_a
        batch_mean_b, batch_var_b = moving_mean_b, moving_var_b
        control_inputs = []

    with tf.control_dependencies(control_inputs):
        mean_ab = (1-alpha) * batch_mean_a + alpha * batch_mean_b
        var_ab = (1-alpha) * batch_var_a + alpha * batch_var_b

        mean_ba = (1 - alpha) * batch_mean_b + alpha * batch_mean_a
        var_ba = (1 - alpha) * batch_var_b + alpha * batch_var_a

        da_tensor_a = (tensor_a - mean_ab) / tf.sqrt(var_ab + epsilon)
        da_tensor_b = (tensor_b - mean_ba) / tf.sqrt(var_ba + epsilon)

    return da_tensor_a, da_tensor_b
