import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from model.ops import utils
from model.ops import base_ops
from tensorflow.python.ops import standard_ops
from util.tflog import tflogger as logging


class GraphAttenNet(object):
  def __init__(self, model_config, fg, training_config):
    self.config = model_config
    self.fg = fg
    self.training_config = training_config
    self.gat_dnn_hidden_layer = "gat_dnn_hidden_layer"
    self.gat_dnn_hidden_output = "gat_dnn_hidden_output"

  def build_gat_net(self, features, i2i_column):
    with tf.variable_scope(
      name_or_scope="gat_input_from_feature_columns",
      partitioner=base_ops.partitioner(self.config.ps_num, mem=8 * 1024 * 1024),
      reuse=tf.AUTO_REUSE) as scope:
      self.i2i_seq_layer = layers.input_from_feature_columns(self.fg.i2i_features(features),
                                                             i2i_column, scope=scope)  # [?*30, f_num*f_embedding]
    self.item_ids_dense = self.fg.item_id_features(features, length=self.fg.i2i_seq_length)
    self.item_sequence_mask = tf.not_equal(self.item_ids_dense, "0")

    with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.attention_l2_reg)):
      with tf.variable_scope(name_or_scope="Graph-Atten-Layer",
                             partitioner=base_ops.partitioner(self.config.ps_num, mem=64 * 1024)):
        items = tf.split(self.i2i_seq_layer, self.fg.i2i_seq_length, axis=0)  # [?, f_num*f_embedding]*30
        tf.add_to_collection(self.gat_dnn_hidden_output, tf.concat(items, 1, name='input_concat'))
        items_stack = tf.stack(values=items,
                               axis=1)  # [?, 30, f_num*f_embedding] or [h*N, T_q, T_k]
        # self attention
        # self.s_item_vec = attention(queries=items_stack,
        #                             reuse=tf.AUTO_REUSE,
        #                             num_units=self.config.i2i_num_units,
        #                             activation_fn=utils.getActivationFunctionOp(
        #                               self.config.activation_op),
        #                             # activation_fn=utils.getActivationFunctionOp(act_name='tanh'),
        #                             key_masks=self.item_sequence_mask,
        #                             variables_collections=[self.gat_dnn_hidden_layer],
        #                             outputs_collections=[self.gat_dnn_hidden_output])
        # self attention
        self.s_item_vec = multihead_attention(queries=items_stack,
                                              queries_length=None,
                                              keys=items_stack,
                                              num_units=self.config.i2i_num_units,
                                              num_output_units=self.config.i2i_num_units,
                                              activation_fn=utils.getActivationFunctionOp(
                                                self.config.activation_op),
                                              keep_prob=self.config.fc_dropout_keep_prob,
                                              keys_length=None,
                                              is_training=self.config.is_training,
                                              scope="attention",
                                              reuse=tf.AUTO_REUSE,
                                              query_masks=self.item_sequence_mask,
                                              key_masks=self.item_sequence_mask,
                                              variables_collections=[self.gat_dnn_hidden_layer],
                                              outputs_collections=[self.gat_dnn_hidden_output])

    with tf.name_scope("Gat-Summary"):
      base_ops.add_norm2_summary(self.gat_dnn_hidden_layer, summary_prefix="Gat_Norm2/")
      base_ops.add_dense_output_summary(self.gat_dnn_hidden_output,
                                        summary_prefix="Gat_DenseOutput/")

    return self.s_item_vec


def multihead_attention(queries,
                        queries_length,
                        keys,
                        keys_length,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=4,
                        keep_prob=0.8,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        variables_collections=None,
                        outputs_collections=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    num_output_units: A scalar. Output Value size.
    keep_prob: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
    query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
    key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = layers.fully_connected(queries,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="Q"
                               # , biases_initializer=init_ops.constant_initializer(0.5)
                               )  # (N, T_q, C)
    K = layers.fully_connected(keys,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="K"
                               # , biases_initializer=init_ops.constant_initializer(0.5)
                               )  # (N, T_k, C)
    V = layers.fully_connected(keys,
                               num_output_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="V"
                               # , biases_initializer=init_ops.constant_initializer(0.5)
                               )  # (N, T_k, C)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # outputs = tf.Print(outputs, [outputs], message="outputs", summarize=5000)

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Key Masking
    if key_masks is None:
      key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    # key_masks = tf.Print(key_masks, [key_masks], message="key_masks", summarize=5000)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    if query_masks is None:
      query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
    else:
      query_masks = tf.cast(query_masks, dtype=tf.float32)
    # query_masks = tf.Print(query_masks, [query_masks], message="query_masks", summarize=5000)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)
    # Attention vector
    att_vec = outputs

    # Dropouts
    outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

    # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    outputs = tf.reshape(tf.reduce_mean(outputs, 1), [-1, num_output_units])
    # Residual connection
    # outputs += queries

    # Normalize
    # outputs = layers.layer_norm(outputs)  # (N, T_q, C)

  return outputs


def attention(queries,
              num_units=None,
              keys_length=None,
              activation_fn=None,
              scope="attention",
              reuse=None,
              key_masks=None,
              variables_collections=None,
              outputs_collections=None):
  '''Applies attention.
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list()[-1]

    Q = layers.fully_connected(queries,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="Q"
                               # , biases_initializer=init_ops.constant_initializer(0.5)
                               )  # (N, T_q, C)

    K = layers.fully_connected(queries,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="K"
                               # , biases_initializer=init_ops.constant_initializer(0.5)
                               )  # (N, T_q, C)

    gat_atten = layers.fully_connected(Q,
                                       1,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="gat_atten"
                                       # , biases_initializer=init_ops.constant_initializer(0.5)
                                       )  # (N, T_q, 1)

    # atten_w = tf.get_variable(name="gat_atten_w",
    #                           shape=[num_units, 1],
    #                           dtype=tf.float32,
    #                           initializer=initializers.xavier_initializer()
    #                           )  # (C, 1)
    # tf.add_to_collection(name=variables_collections[0], value=atten_w)
    # gat_atten = standard_ops.tensordot(Q, atten_w, [[len(Q.get_shape().as_list()) - 1], [0]])  # (h*N, T_q, 1)
    # Scale
    # gat_atten = gat_atten / (Q.get_shape().as_list()[-1] ** 0.5)  # [?, T_q, 1]

    # Key Masking
    if key_masks is None:
      key_masks = tf.sequence_mask(keys_length, tf.shape(queries)[1])  # (N, T_q)
    # key_masks = tf.Print(key_masks, [key_masks], message="key_masks", summarize=5000)
    key_masks = tf.expand_dims(key_masks, -1)  # (N, T_q, 1)

    paddings = tf.ones_like(gat_atten) * (-2 ** 32 + 1)
    gat_atten = tf.where(key_masks, gat_atten, paddings)  # (N, T_q, 1)

    # Activation
    gat_atten = tf.transpose(gat_atten, [0, 2, 1])  # (N, 1, T_q)
    gat_atten_soft = tf.nn.softmax(gat_atten, name="gat_atten_soft_out")  # (N, 1, T_q)
    # gat_atten_soft = tf.Print(gat_atten_soft, [gat_atten_soft], message="gat_atten_soft_out", summarize=5000)
    # logging.info("outputs_soft size: %s" % outputs_soft.get_shape().as_list())
    gat_atten_out = tf.matmul(gat_atten_soft, K)  # (N, 1, C)
    gat_atten_out = tf.reshape(gat_atten_out, shape=[-1, num_units], name="gat_atten_out")  # (N, C)

    tf.add_to_collection(name=outputs_collections[0], value=gat_atten_soft)
    tf.add_to_collection(name=outputs_collections[0], value=gat_atten_out)

    return gat_atten_out
