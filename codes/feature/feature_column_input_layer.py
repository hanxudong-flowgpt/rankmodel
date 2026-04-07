# coding=utf-8

# Copyright (c) 2019 Pinduoduo Inc. All Rights Reserved

# Author: Zhangyi Chen (yingwuluo@pinduoduo.com)
# Date: Mon Oct 21 14:45:30 CST 2019

'''
Mimic input_from_feature_columns to make serving graphs the same as the trainig graphs
'''

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from feature import feature_column as fc
from tensorflow.contrib.layers.python.layers import feature_column_ops as fc_ops
from tensorflow.python.framework import ops
from tensorflow.python.feature_column.feature_column import _normalize_feature_columns, _DenseColumn, _LazyBuilder, _verify_static_batch_size_equality


# from repeated_column import dense_repeat

def dense_repeat(dense_tensor, n):
    t = tf.squeeze(dense_tensor, [0])
    n_as_i32_tensor = tf.reshape(tf.convert_to_tensor(n, dtype=tf.int32), [1])
    new_shape = tf.concat([n_as_i32_tensor, tf.shape(t)], axis=0)
    return tf.broadcast_to(t, new_shape)


def input_from_one_feature_column(features,
                          feature_columns,
                          weight_collections=None,
                          trainable=True,
                          cols_to_vars=None,
                          scope=None,
                          cols_to_output_tensors=None,
                          from_template=False,
                          use_fast_embeding=False):
  """See input_layer. `scope` is a name or variable scope to use."""

  feature_columns = _normalize_feature_columns(feature_columns)
  for column in feature_columns:
    if not isinstance(column, _DenseColumn):
      raise ValueError(
          'Items of feature_columns must be a _DenseColumn. '
          'You can wrap a categorical column with an '
          'embedding_column or indicator_column. Given: {}'.format(column))
  weight_collections = list(weight_collections or [])
  if ops.GraphKeys.GLOBAL_VARIABLES not in weight_collections:
    weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)
  if ops.GraphKeys.MODEL_VARIABLES not in weight_collections:
    weight_collections.append(ops.GraphKeys.MODEL_VARIABLES)

  def _get_logits():  # pylint: disable=missing-docstring
    builder = _LazyBuilder(features)
    output_tensors = []
    ordered_columns = []
    for column in sorted(feature_columns, key=lambda x: x.name):
      ordered_columns.append(column)
      with variable_scope.variable_scope(
          None, default_name=column._var_scope_name):  # pylint: disable=protected-access
        tensor = column._get_dense_tensor(  # pylint: disable=protected-access
            builder,
            weight_collections=weight_collections,
            trainable=trainable)
        num_elements = column._variable_shape.num_elements()  # pylint: disable=protected-access
        batch_size = array_ops.shape(tensor)[0]
        output_tensor = array_ops.reshape(
            tensor, shape=(batch_size, num_elements))
        output_tensors.append(output_tensor)
        if cols_to_vars is not None:
          # Retrieve any variables created (some _DenseColumn's don't create
          # variables, in which case an empty list is returned).
          cols_to_vars[column] = ops.get_collection(
              ops.GraphKeys.GLOBAL_VARIABLES,
              scope=variable_scope.get_variable_scope().name)
        if cols_to_output_tensors is not None:
          cols_to_output_tensors[column] = output_tensor
    _verify_static_batch_size_equality(output_tensors, ordered_columns)
    return output_tensors[0]

  # If we're constructing from the `make_template`, that by default adds a
  # variable scope with the name of the layer. In that case, we dont want to
  # add another `variable_scope` as that would break checkpoints.
  if from_template:
    return _get_logits()
  else:
    with variable_scope.variable_scope(
        scope, default_name='input_layer', values=features.values()):
      return _get_logits()
