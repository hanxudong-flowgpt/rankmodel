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


# from repeated_column import dense_repeat

def dense_repeat(dense_tensor, n):
    t = tf.squeeze(dense_tensor, [0])
    n_as_i32_tensor = tf.reshape(tf.convert_to_tensor(n, dtype=tf.int32), [1])
    new_shape = tf.concat([n_as_i32_tensor, tf.shape(t)], axis=0)
    return tf.broadcast_to(t, new_shape)


def input_from_feature_columns(
        columns_to_tensors,
        feature_columns,
        repeating_columns=[],
        n=None,
        share_context=None,
        weight_collections=None,
        trainable=True,
        scope=None,
        cols_to_outs=None,
        use_fast_embeding=False):
    """Implementation of `input_from(_sequence)_feature_columns`."""
    columns_to_tensors = columns_to_tensors.copy()
    fc_ops.check_feature_columns(feature_columns)

    if cols_to_outs is not None and not isinstance(cols_to_outs, dict):
        raise ValueError('cols_to_outs must be a dict unless None')
    output_rank = 2

    with variable_scope.variable_scope(scope,
                                       default_name="input_from_feature_columns",
                                       values=columns_to_tensors.values()):
        output_tensors = []
        transformer = fc_ops._Transformer(columns_to_tensors)
        if weight_collections:
            weight_collections = list(set(list(weight_collections) +
                                          [ops.GraphKeys.GLOBAL_VARIABLES]))

        print("\n".join(sorted([e.name for e in set(feature_columns)])))
        for e in repeating_columns:
            print(e.name)
        for column in sorted(set(feature_columns), key=lambda x: x.key):
            with variable_scope.variable_scope(None,
                                               default_name=column.name,
                                               values=columns_to_tensors.values()):
                transformed_tensor = transformer.transform(column)
                try:
                    # pylint: disable=protected-access
                    arguments = column._deep_embedding_lookup_arguments(
                        transformed_tensor)
                    tensor = fc._embeddings_from_arguments(  # pylint: disable=protected-access
                        column,
                        arguments,
                        weight_collections,
                        trainable,
                        output_rank=output_rank,
                        inference=use_fast_embeding)
                    if n is not None and column in repeating_columns:
                        tensor = tf.cond(share_context,
                                         lambda: dense_repeat(tensor, n),
                                         lambda: tensor)

                    output_tensors.append(tensor)

                except NotImplementedError as ee:
                    try:
                        # pylint: disable=protected-access
                        tensor = column._to_dnn_input_layer(
                            transformed_tensor,
                            weight_collections,
                            trainable,
                            output_rank=output_rank)
                        if n is not None and column in repeating_columns:
                            tensor = tf.cond(share_context,
                                             lambda: dense_repeat(tensor, n),
                                             lambda: tensor)
                        output_tensors.append(tensor)
                    except ValueError as e:
                        raise ValueError('Error creating input layer for column: {}.\n'
                                         '{} , {}'.format(column.name, e, ee))
                if cols_to_outs is not None:
                    cols_to_outs[column] = output_tensors[-1]
    return array_ops.concat(output_tensors, output_rank - 1)


def input_from_one_feature_column(
        columns_to_tensors,
        feature_columns,
        repeating_columns=[],
        n=None,
        share_context=None,
        weight_collections=None,
        trainable=True,
        scope=None,
        cols_to_outs=None,
        use_fast_embeding=False):
    """Implementation of `input_from(_sequence)_feature_columns`."""
    assert len(columns_to_tensors) == 1 and len(feature_columns) == 1
    columns_to_tensors = columns_to_tensors.copy()
    fc_ops.check_feature_columns(feature_columns)

    if cols_to_outs is not None and not isinstance(cols_to_outs, dict):
        raise ValueError('cols_to_outs must be a dict unless None')
    output_rank = 2

    with variable_scope.variable_scope(scope,
                                       default_name="input_from_feature_columns",
                                       values=columns_to_tensors.values()):
        output_tensors = []
        transformer = fc_ops._Transformer(columns_to_tensors)
        if weight_collections:
            weight_collections = list(set(list(weight_collections) +
                                          [ops.GraphKeys.GLOBAL_VARIABLES]))

        print("\n".join(sorted([e.name for e in set(feature_columns)])))
        for e in repeating_columns:
            print(e.name)
        for column in sorted(set(feature_columns), key=lambda x: x.key):
            with variable_scope.variable_scope(None,
                                               default_name=column.name,
                                               values=columns_to_tensors.values()):
                transformed_tensor = transformer.transform(column)
                try:
                    # pylint: disable=protected-access
                    arguments = column._deep_embedding_lookup_arguments(
                        transformed_tensor)
                    tensor = fc._embeddings_from_arguments(  # pylint: disable=protected-access
                        column,
                        arguments,
                        weight_collections,
                        trainable,
                        output_rank=output_rank,
                        inference=use_fast_embeding)
                    if n is not None and column in repeating_columns:
                        tensor = tf.cond(share_context,
                                         lambda: dense_repeat(tensor, n),
                                         lambda: tensor)

                    output_tensors.append(tensor)

                except NotImplementedError as ee:
                    try:
                        # pylint: disable=protected-access
                        tensor = column._to_dnn_input_layer(
                            transformed_tensor,
                            weight_collections,
                            trainable,
                            output_rank=output_rank)
                        if n is not None and column in repeating_columns:
                            tensor = tf.cond(share_context,
                                             lambda: dense_repeat(tensor, n),
                                             lambda: tensor)
                        output_tensors.append(tensor)
                    except ValueError as e:
                        raise ValueError('Error creating input layer for column: {}.\n'
                                         '{} , {}'.format(column.name, e, ee))
                if cols_to_outs is not None:
                    cols_to_outs[column] = output_tensors[-1]
    return output_tensors[0]
