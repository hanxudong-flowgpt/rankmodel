import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import sparse_tensor


def put_tags(batch_res):
  from tensorflow.python.framework import ops
  for key, feature in batch_res.iteritems():
    if "target" in key:
      src = 'item:' + key
    else:
      src = 'user:' + key
    print(key, 'come from', src)
    tf.contrib.layers.add_tensor_to_collection(
      ops.GraphKeys.RANK_SERVICE_INPUT, key, feature)
    if isinstance(feature, sparse_tensor.SparseTensor):
      tf.contrib.layers.mark_input_src(feature.values.name, src)
      tf.contrib.layers.mark_input_src(feature.indices.name, src)
      tf.contrib.layers.mark_input_src(feature.dense_shape.name, src)
    else:
      tf.contrib.layers.mark_input_src(feature.name, src)


def mark_output(name, tensor):
  layers.add_tensor_to_collection(ops.GraphKeys.RANK_SERVICE_OUTPUT, name,
                                  tf.identity(tensor, name))


def dropout_in_train(inputs, is_local=False, **kwargs):
  inputs_name_before_dropout = inputs.name
  out = layers.dropout(inputs, **kwargs)
  # if not is_local:
  #   skip_in_inference(inputs_name_before_dropout, out.name)
  return out


def skip_in_inference(block_input, block_out):
  replace_info = {
    "name": '_skip_'.join([block_input, block_out]),
    "op_name": "inference_skip",
    "inputs": [
      block_input
    ],
    "output": block_out,
    "attrs": []
  }
  from tensorflow.python.framework import ops
  utils.update_attr_to_collection(ops.GraphKeys.RANK_SERVICE_REPLACE_OP,
                                  replace_info)
