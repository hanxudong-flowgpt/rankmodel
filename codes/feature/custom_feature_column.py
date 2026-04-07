from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util.tflog import tflogger as logging
import collections
import math
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python.layers.feature_column import _FeatureColumn, _RealValuedColumn, \
  _LazyBuilderByColumnsToTensor, _DeepEmbeddingLookupArguments, _reshape_real_valued_tensor
from tensorflow.contrib.layers.python.ops import bucketization_op
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_py
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.random_ops import random_normal

class _EmbeddingBucketizedColumn(
  _FeatureColumn,
  fc_core._DenseColumn,  # pylint: disable=protected-access
  collections.namedtuple("_EmbeddingBucketizedColumn", ["source_column",
                                                        "boundaries", "embedding_dimension", "shared_name",
                                                        "max_norm", "trainable", "initializer", "is_local",
                                                        "add_random"])):
  """Represents a bucketization transformation also known as binning.

  Instances of this class are immutable. Values in `source_column` will be
  bucketized based on `boundaries`.
  For example, if the inputs are:
      boundaries = [0, 10, 100]
      source_column = [[-5], [150], [10], [0], [4], [19]]

  then the bucketized feature will be:
      output = [[0], [3], [2], [1], [1], [2]]

  Attributes:
    source_column: A _RealValuedColumn defining dense column.
    boundaries: A list or tuple of floats specifying the boundaries. It has to
      be sorted. [a, b, c] defines following buckets: (-inf., a), [a, b),
      [b, c), [c, inf.)
    embedding_dimension: An integer specifying dimension of the embedding.
    shared_name: (Optional). A string specifying the name of shared
      embedding weights. This will be needed if you want to reference the shared
      embedding separately from the generated `_EmbeddingBucketizedColumn`.
  Raises:
    ValueError: if 'boundaries' is empty or not sorted.
  """

  def __new__(cls, source_column, boundaries, embedding_dimension, shared_name, max_norm, trainable, initializer,
              is_local, add_random):
    if not isinstance(source_column, _RealValuedColumn):
      raise TypeError("source_column must be an instance of _RealValuedColumn. "
                      "source_column: {}".format(source_column))

    if source_column.dimension is None:
      raise ValueError("source_column must have a defined dimension. "
                       "source_column: {}".format(source_column))

    if (not isinstance(boundaries, list) and
          not isinstance(boundaries, tuple)) or not boundaries:
      raise ValueError("boundaries must be a non-empty list or tuple. "
                       "boundaries: {}".format(boundaries))

    # We allow bucket boundaries to be monotonically increasing
    # (ie a[i+1] >= a[i]). When two bucket boundaries are the same, we
    # de-duplicate.
    sanitized_boundaries = []
    for i in range(len(boundaries) - 1):
      if boundaries[i] == boundaries[i + 1]:
        continue
      elif boundaries[i] < boundaries[i + 1]:
        sanitized_boundaries.append(boundaries[i])
      else:
        raise ValueError("boundaries must be a sorted list. "
                         "boundaries: {}".format(boundaries))
    sanitized_boundaries.append(boundaries[len(boundaries) - 1])
    return super(_EmbeddingBucketizedColumn, cls).__new__(cls, source_column,
                                                          tuple(sanitized_boundaries), embedding_dimension, shared_name,
                                                          max_norm, trainable, initializer, is_local, add_random)

  @property
  def name(self):
    return "{}_embedding_bucketized".format(self.source_column.name)

  @property
  def length(self):
    """Returns total number of buckets."""
    return len(self.boundaries) + 1

  @property
  def config(self):
    return self.source_column.config

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def _deep_embedding_lookup_arguments(self, input_tensor):
    if self.initializer is None:
      stddev = 1 / math.sqrt(self.length)
      initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=stddev)
    else:
      initializer = self.initializer
    if not self.is_local:
      return _DeepEmbeddingLookupArguments(
        input_tensor=self.to_sparse_tensor(input_tensor),
        weight_tensor=None,
        vocab_size=self.length,
        dimension=self.embedding_dimension,
        initializer=initializer,
        combiner=None,
        shared_embedding_name=self.shared_name,
        hash_key=None,
        max_norm=self.max_norm,
        trainable=self.trainable,
        origin_feature_tensor=None,
        bucket_size=self.length
      )
    else:
      return _DeepEmbeddingLookupArguments(
        input_tensor=self.to_sparse_tensor(input_tensor),
        weight_tensor=None,
        vocab_size=self.length,
        dimension=self.embedding_dimension,
        initializer=initializer,
        combiner=None,
        shared_embedding_name=self.shared_name,
        hash_key=None,
        max_norm=self.max_norm,
        trainable=self.trainable,
        origin_feature_tensor=None,
        bucket_size=10 # self.length
      )
      '''
      return _DeepEmbeddingLookupArguments(
        input_tensor=self.to_sparse_tensor(input_tensor),
        weight_tensor=None,
        vocab_size=self.length,
        dimension=self.embedding_dimension,
        initializer=initializer,
        combiner=None,
        shared_embedding_name=self.shared_name,
        hash_key=None,
        max_norm=self.max_norm,
        trainable=self.trainable
      )
      '''

  def to_sparse_tensor(self, input_tensor):
    """Creates a SparseTensor from the bucketized Tensor."""
    dimension = self.source_column.dimension
    batch_size = array_ops.shape(input_tensor, name="shape")[0]

    if dimension > 1:
      i1 = array_ops.reshape(
        array_ops.tile(
          array_ops.expand_dims(
            math_ops.range(0, batch_size), 1, name="expand_dims"),
          [1, dimension],
          name="tile"), [-1],
        name="reshape")
      i2 = array_ops.tile(
        math_ops.range(0, dimension), [batch_size], name="tile")
      # Flatten the bucket indices and unique them across dimensions
      # E.g. 2nd dimension indices will range from k to 2*k-1 with k buckets
      bucket_indices = array_ops.reshape(
        input_tensor, [-1], name="reshape") + self.length * i2
    else:
      # Simpler indices when dimension=1
      i1 = math_ops.range(0, batch_size)
      i2 = array_ops.zeros([batch_size], dtype=dtypes.int32, name="zeros")
      bucket_indices = array_ops.reshape(input_tensor, [-1], name="reshape")

    indices = math_ops.to_int64(array_ops.transpose(array_ops.stack((i1, i2))))
    shape = math_ops.to_int64(array_ops.stack([batch_size, dimension]))
    sparse_id_values = sparse_tensor_py.SparseTensor(
      indices, bucket_indices, shape)

    return sparse_id_values

  # pylint: disable=unused-argument
  def _wide_embedding_lookup_arguments(self, input_tensor):
    raise ValueError("Column {} is not supported in linear models. "
                     "Please use sparse_column.".format(self))

  def _transform_feature(self, inputs):
    """Handles cross transformation."""
    # Bucketize the source column.
    if not self.add_random:
      return bucketization_op.bucketize(
        inputs.get(self.source_column),
        boundaries=list(self.boundaries),
        name="bucketize")
    else:
      rawts = inputs.get(self.source_column)
      tbn = np.asarray(self.boundaries[1:])
      if len(tbn)>30:
        # noise =  min(np.median(tbn)-tbn[0],tbn[20:-20].std())/2.
        noise = tbn[10:-10].std() / 10.
        rndts = rawts + random_normal(array_ops.shape(rawts),0,noise)
        logging.info("Adding Random Disturb to %s with noise' stddev=%s; tbn:%s,%s,%s"
                     %(self.name,str(noise),str(np.median(tbn)),str(tbn[0]),str(tbn[10:-10].std())))
        return bucketization_op.bucketize(rndts,
          boundaries=list(self.boundaries),
          name="bucketize")
      else:
        return bucketization_op.bucketize(
          rawts,
          boundaries=list(self.boundaries),
          name="bucketize")

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles sparse column to id conversion."""
    columns_to_tensors[self] = self._transform_feature(
      _LazyBuilderByColumnsToTensor(columns_to_tensors))

  @property
  def _parse_example_spec(self):
    return self.config

  @property
  def _num_buckets(self):
    return self.length * self.source_column.dimension

  @property
  def _variable_shape(self):
    return tensor_shape.TensorShape(
      [self.length * self.source_column.dimension])

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return self._to_dnn_input_layer(
      inputs.get(self), weight_collections, trainable)


def embedding_bucketized_column(source_column, boundaries, embedding_dimension, shared_name=None, max_norm=None,
                                trainable=True, initializer=None, is_local=False, add_random=False):
  """Creates a _EmbeddingBucketizedColumn for discretizing dense input.

  Args:
    source_column: A _RealValuedColumn defining dense column.
    boundaries: A list or tuple of floats specifying the boundaries. It has to
      be sorted.

  Returns:
    A _EmbeddingBucketizedColumn.

  Raises:
    ValueError: if 'boundaries' is empty or not sorted.
  """
  retv = _EmbeddingBucketizedColumn(source_column, boundaries, embedding_dimension, shared_name, max_norm=max_norm,
                                    trainable=trainable, initializer=initializer, is_local=is_local,
                                    add_random=add_random)
  return retv


def _is_variable(v):
  """Returns true if `v` is a variable."""
  return isinstance(v, (variables.Variable,
                        resource_variable_ops.ResourceVariable))


def real_valued_varlen_column(column_name,
                              default_value=None,
                              dtype=dtypes.float32,
                              normalizer=None,
                              is_sparse=False):
  """Creates a `_RealValuedVarLenColumn` for variable-length numeric data.

  Note, this is not integrated with any of the DNNEstimators, except the RNN
  ones DynamicRNNEstimator and the StateSavingRNNEstimator.

  It can either create a parsing config for a SparseTensor (with is_sparse=True)
  or a padded Tensor.
  The (dense_)shape of the result will be [batch_size, None], which can be used
  with is_sparse=False as input into an RNN (see DynamicRNNEstimator or
  StateSavingRNNEstimator) or with is_sparse=True as input into a tree (see
  gtflow).

  Use real_valued_column if the Feature has a fixed length. Use some
  SparseColumn for columns to be embedded / one-hot-encoded.

  Args:
    column_name: A string defining real valued column name.
    default_value: A scalar value compatible with dtype. Needs to be specified
      if is_sparse=False.
    dtype: Defines the type of values. Default value is tf.float32. Needs to be
      convertible to tf.float32.
    normalizer: If not None, a function that can be used to normalize the value
      of the real valued column after default_value is applied for parsing.
      Normalizer function takes the input tensor as its argument, and returns
      the output tensor. (e.g. lambda x: (x - 3.0) / 4.2). Note that for
      is_sparse=False, the normalizer will be run on the values of the
      `SparseTensor`.
    is_sparse: A boolean defining whether to create a SparseTensor or a Tensor.
  Returns:
    A _RealValuedSparseColumn.
  Raises:
    TypeError: if default_value is not a scalar value compatible with dtype.
    TypeError: if dtype is not convertible to tf.float32.
    ValueError: if default_value is None and is_sparse is False.
  """
  if not (dtype.is_integer or dtype.is_floating):
    raise TypeError("dtype must be convertible to float. "
                    "dtype: {}, column_name: {}".format(dtype, column_name))

  if default_value is None and not is_sparse:
    raise ValueError("default_value must be provided when is_sparse=False to "
                     "parse a padded Tensor. "
                     "column_name: {}".format(column_name))
  if isinstance(default_value, list):
    raise ValueError(
        "Only scalar default value. default_value: {}, column_name: {}".format(
            default_value, column_name))
  if default_value is not None:
    if dtype.is_integer:
      default_value = int(default_value)
    elif dtype.is_floating:
      default_value = float(default_value)

  return _RealValuedVarLenColumn(column_name, default_value, dtype, normalizer, is_sparse)


class _RealValuedVarLenColumn(_FeatureColumn, fc_core._DenseColumn, collections.namedtuple(
    "_RealValuedVarLenColumn",
    ["column_name", "default_value", "dtype", "normalizer", "is_sparse"])):
  """Represents a real valued feature column for variable length Features.

  Instances of this class are immutable.
  If is_sparse=False, the dictionary returned by InputBuilder contains a
  ("column_name", Tensor) pair with a Tensor shape of (batch_size, dimension).
  If is_sparse=True, the dictionary contains a ("column_name", SparseTensor)
  pair instead with shape inferred after parsing.
  """

  @property
  def name(self):
    return self.column_name

  @property
  def config(self):
    if self.is_sparse:
      return {self.column_name: parsing_ops.VarLenFeature(self.dtype)}
    else:
      return {self.column_name: parsing_ops.FixedLenSequenceFeature(
          [], self.dtype, allow_missing=True,
          default_value=self.default_value)}

  @property
  def _parse_example_spec(self):
    return self.config

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return self._key_without_properties(["normalizer"])

  @property
  def normalizer_fn(self):
    """Returns the function used to normalize the column."""
    return self.normalizer

  def _normalized_input_tensor(self, input_tensor):
    """Returns the input tensor after custom normalization is applied."""
    if self.normalizer is None:
      return input_tensor
    if self.is_sparse:
      return sparse_tensor_py.SparseTensor(
          input_tensor.indices,
          self.normalizer(input_tensor.values),
          input_tensor.dense_shape)
    else:
      return self.normalizer(input_tensor)

  def insert_transformed_feature(self, columns_to_tensors):
    """Apply transformation and inserts it into columns_to_tensors.

    Args:
      columns_to_tensors: A mapping from feature columns to tensors. 'string'
        key means a base feature (not-transformed). It can have _FeatureColumn
        as a key too. That means that _FeatureColumn is already transformed.
    """
    # Transform the input tensor according to the normalizer function.
    input_tensor = self._normalized_input_tensor(columns_to_tensors[self.name])
    columns_to_tensors[self] = math_ops.to_float(input_tensor)

  # pylint: disable=unused-argument
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    '''
    return _reshape_real_valued_tensor(
        self._to_dense_tensor(input_tensor), output_rank, self.name)
    '''
    return tf.reshape(self._to_dense_tensor(input_tensor), [-1,1])

  def _to_dense_tensor(self, input_tensor):
    if not self.is_sparse:
      return input_tensor
    else:
      return sparse_ops.sparse_reduce_sum(input_tensor, axis=-1, keepdims=True)