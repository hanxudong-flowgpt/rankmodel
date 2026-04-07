import tensorflow as tf
from tensorflow.contrib import layers
from model_ops import ops as base_ops
from model_ops import rtp_ops
from tensorflow.python.layers import core as core_layers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variables as tf_variables


class AdvNet(object):
  def __init__(self, model_config, fg, training_config):
    self.config = model_config
    self.fg = fg
    self.training_config = training_config

  def build_adv_layer(self,
                      name,
                      layer_id,
                      num_outputs,
                      activation_fn=nn.relu,
                      normalizer_fn=None,
                      weights_initializer=initializers.xavier_initializer(),
                      weights_regularizer=None,
                      biases_initializer=init_ops.zeros_initializer(),
                      biases_regularizer=None,
                      reuse=None,
                      trainable=True
                      ):
    with tf.variable_scope(
      name_or_scope="%s_adv_layer_%d" % (name, layer_id),
      partitioner=base_ops.partitioner(self.config.ps_num, mem=8 * 1024 * 1024),
      reuse=tf.AUTO_REUSE) as scope:
      layer = core_layers.Dense(
        units=num_outputs,
        activation=activation_fn,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=scope.name,
        _scope=scope,
        _reuse=reuse)

    return layer


class FlipGradientBuilder(object):
  '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [tf.negative(grad) * l]

    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)

    self.num_calls += 1
    return y

flip_gradient = FlipGradientBuilder()