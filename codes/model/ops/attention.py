import tensorflow as tf
from tensorflow.contrib import layers


def multihead_attention(queries,
                        queries_length,
                        keys,
                        keys_length,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
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

    def split_last_dimension_then_transpose(tensor, num_heads):
      t_shape = tensor.get_shape().as_list()
      tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
      return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

    Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
    K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
    V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, C'/h)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
    # [batch_size, num_heads, query_len, key_len]

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    query_len = queries.get_shape().as_list()[1]
    key_len = keys.get_shape().as_list()[1]

    # Key Masking
    if key_masks is None:
      key_masks = tf.sequence_mask(keys_length, key_len)  # (N, T_k)
    key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]),
                        [1, num_heads, query_len, 1])
    paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
    outputs = tf.where(key_masks, outputs, paddings)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    if query_masks is None:
      if queries_length is not None:
        query_masks = tf.sequence_mask(queries_length, query_len)  # (N, T_q)

    if query_masks is not None:
      query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]),
                            [1, num_heads, 1, key_len])
      paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
      outputs = tf.where(query_masks, outputs, paddings)

    # Attention vector
    att_vec = outputs

    # Dropouts
    outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

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


def feedforward(inputs,
                num_units=[2048, 512],
                activation_fn=None,
                scope="feedforward",
                reuse=None,
                variables_collections=None,
                outputs_collections=None):
  '''Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=reuse):
    outputs = layers.fully_connected(inputs,
                                     num_units[0],
                                     activation_fn=activation_fn,
                                     variables_collections=variables_collections,
                                     outputs_collections=outputs_collections)
    outputs = layers.fully_connected(outputs,
                                     num_units[1],
                                     activation_fn=None,
                                     variables_collections=variables_collections,
                                     outputs_collections=outputs_collections)
    outputs += inputs

  return outputs
