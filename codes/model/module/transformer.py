import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from model.ops import utils
from model.ops import base_ops
from tensorflow.python.ops import standard_ops
from util.tflog import tflogger as logging


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
                        scope="multihead_attention",
                        atten_mode="base",
                        first_n_att_weight_report=9,
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        variables_collections=None,
                        outputs_collections=None,
                        atten_weights_collections=None):
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

    if atten_mode == 'cos' or atten_mode == 'ln':
      activation_fn = None

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = layers.fully_connected(queries,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
    K = layers.fully_connected(keys,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="K")  # (N, T_k, C)
    V = layers.fully_connected(keys,
                               num_output_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="V")  # (N, T_k, C)

    def split_last_dimension_then_transpose(tensor, num_heads):
      t_shape = tensor.get_shape().as_list()
      tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
      return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

    Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
    K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
    V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, C'/h)

    if atten_mode == 'cos':
      Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
      K_cos = tf.nn.l2_normalize(K_, dim=-1)

      # Multiplication
      # query-key score matrix
      # each big score matrix is then split into h score matrix with same size
      # w.r.t. different part of the feature
      outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # (h*N, T_q, T_k)
      # [batch_size, num_heads, query_len, key_len]

      # Scale
      outputs = outputs * 20
    elif atten_mode == 'ln':
      # Multiplication
      # query-key score matrix
      # each big score matrix is then split into h score matrix with same size
      # w.r.t. different part of the feature
      Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
      K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
      outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
      # [batch_size, num_heads, query_len, key_len]

      # Scale
      outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
    else:
      # Multiplication
      # query-key score matrix
      # each big score matrix is then split into h score matrix with same size
      # w.r.t. different part of the feature
      outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
      # [batch_size, num_heads, query_len, key_len]

      # Scale
      outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

      # Scale
      outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    query_len = queries.get_shape().as_list()[1]
    key_len = keys.get_shape().as_list()[1]

    key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]),
                        [1, num_heads, query_len, 1])
    paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
    outputs = tf.where(key_masks, outputs, paddings)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]),
                          [1, num_heads, 1, key_len])
    paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
    outputs = tf.where(query_masks, outputs, paddings)

    # Attention vector
    att_vec = outputs

    # Dropouts
    # outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

    # summary
    keys_masks_tmp = tf.reshape(tf.cast(key_masks, tf.float32), [-1, key_len])
    defined_length = tf.constant(first_n_att_weight_report, dtype=tf.float32, name="%s_defined_length" % (scope))
    greater_than_define = tf.cast(tf.greater(tf.reduce_sum(keys_masks_tmp, axis=1), defined_length), tf.float32)
    greater_than_define_exp = tf.tile(tf.expand_dims(greater_than_define, -1), [1, key_len])

    weight = tf.reshape(outputs, [-1, key_len]) * greater_than_define_exp
    weight_mean = tf.reduce_sum(weight, 0) / (tf.reduce_sum(greater_than_define_exp, axis=0) + 1e-5)

    if atten_weights_collections:
      for i in range(key_len):
        for collection in atten_weights_collections:
          tf.add_to_collection(collection, tf.slice(weight_mean, [i], [1], name='%s_atten_weight_%d' % (scope, i)))

    weight_map = tf.reshape(weight, [-1, query_len, key_len])  # BxL1xL2
    greater_than_define_exp_map = tf.reshape(greater_than_define_exp, [-1, query_len, key_len])  # BxL1xL2
    weight_map_mean = tf.reduce_sum(weight_map, 0) / (
        tf.reduce_sum(greater_than_define_exp_map, axis=0) + 1e-5)  # L1xL2
    report_image = tf.expand_dims(tf.expand_dims(weight_map_mean, -1), 0)  # 1xL1xL2x1
    tf.summary.image("%s_attention" % (scope), report_image[:, :80, :80, :])  # 1x10x10x1

    # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    def transpose_then_concat_last_two_dimenstion(tensor):
      tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
      t_shape = tensor.get_shape().as_list()
      num_heads, dim = t_shape[-2:]
      return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, C)

    # Residual connection
    # outputs += queries

    # Normalize
    # outputs = layers.layer_norm(outputs)  # (N, T_q, C)

  return outputs, att_vec