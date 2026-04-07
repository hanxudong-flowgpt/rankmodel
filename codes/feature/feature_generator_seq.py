from collections import OrderedDict

from util.config.baseconfig import ModelConfig
from feature_generator import FeatureGenerator
import tensorflow as tf
import json

from model.model_column import ModelColumns
import copy

FEATURE_NAME_KEY = "feature_name"
SEQUENCE_NAME_KEY = "sequence_name"
FEATURE_KEY = "features"
SEQUENCE_LENGTH_KEY = "sequence_length"
feature_key = 'features'
value_type_key = 'value_type'
feature_type_key = 'feature_type'
feature_name_key = 'feature_name'
alias_name = 'alias_name'
SKIP_KEY = 'skip'


class FeatureGeneratorSeq(FeatureGenerator):
  def __init__(self, conf_file_path, model_config, mc, training_config=None, cross_file_path=None):
    self._sequence_filter = set(mc.sequence_columns + mc.seq_context_atten_columns + mc.queryseq_columns)
    FeatureGenerator.__init__(self, conf_file_path, model_config, mc, training_config, cross_file_path)
    self._generate_seq_feature_columns()
    self._i2i_column_name = mc.i2i_sequence_columns
    # self._generate_cross_column(cross_file_path)

  def _parse_feature_conf(self, conf_file_path):
    self._feature_conf_map = dict()
    with open(conf_file_path) as f:
      _config = json.load(f)
      feature_conf_list = _config[feature_key]
      feature_conf_list_filtered = []
      for feature_conf in feature_conf_list:
        if "_comment" in feature_conf:
          continue
        if self.train_conf.platform == "local":
          for k,v in feature_conf.items():
            if "hash_bucket_size" == k:
              feature_conf["hash_bucket_size"] = 10
            if type(v) == list:
              for vkv in v:
                if "hash_bucket_size" in vkv:
                  vkv["hash_bucket_size"] = 10
        feature_name = feature_conf[alias_name] if alias_name in feature_conf else feature_conf[feature_name_key]
        if feature_name in self._mc_filter:
          print("feature name:", feature_name)
          if "hash_bucket_size" in feature_conf:
            print(feature_conf["hash_bucket_size"])
          self._feature_conf_map[feature_name] = feature_conf
          if alias_name not in feature_conf:
            feature_conf_list_filtered.append(feature_conf)
        if "opt_seq" == feature_name or "seq" == feature_name or "query_seq" == feature_name:
          if self.train_conf.platform == "local":
            feature_conf_local = copy.deepcopy(feature_conf)
            feature_conf_local[FEATURE_KEY] = []
            for ele_feature_conf in feature_conf[FEATURE_KEY]:
              feature_conf_local[FEATURE_KEY].append(ele_feature_conf)
            feature_conf = feature_conf_local
          sequence_features_filtered = []
          features = feature_conf[FEATURE_KEY]
          self._feature_conf_map[feature_name] = feature_conf
          feature_conf_copy = copy.deepcopy(feature_conf)

          for feature in features:
            # seq_feature_name = feature[FEATURE_NAME_KEY]
            seq_feature_name = feature[alias_name] if alias_name in feature else feature[feature_name_key]
            if self._sequence_filter and seq_feature_name in self._sequence_filter:
              if alias_name not in feature:
                sequence_features_filtered.append(feature)
          feature_conf_copy[FEATURE_KEY] = sequence_features_filtered
          feature_conf_list_filtered.append(feature_conf_copy)

      self._config = {feature_key: feature_conf_list_filtered}

  def _generate_seq_feature_columns(self):
    seq_num = 0
    for name, value in self._feature_conf_map.items():
      if "opt_seq" == name or "seq" == name:
        # if seq_num > 0:
        #   raise RuntimeError("Two sequence is parsed, pls check the lsc_conf.json")
        self.sequence_length = value[SEQUENCE_LENGTH_KEY]
        self._genFeatureColumn(value, name)
        seq_num += 1
      if "query_seq" == name:
        self.queryseq_length = value[SEQUENCE_LENGTH_KEY]
        self._genFeatureColumn(value, name)
        seq_num += 1


  def _genFeatureColumn(self, value, name):
    # key in seq_feature_columns like 'sequence_item_id'.
    self.seq_feature_columns = OrderedDict()
    # Element in feature_name like item_id.
    self.feature_names = []
    features = value[FEATURE_KEY]
    if "query_seq" == name:
      self.queryseq_name = value[SEQUENCE_NAME_KEY]
    if "opt_seq" == name or "seq" == name:
      self.sequence_name = value[SEQUENCE_NAME_KEY]
    for feature in features:
      feature_name = feature[FEATURE_NAME_KEY]
      feature_column_name = feature_name
      self.feature_names.append(feature_name)
      feature_column = self._new_feature(feature_column_name, feature)
      feature_column_name = feature[alias_name] if alias_name in feature else feature[feature_name_key]
      self.seq_feature_columns[feature_column_name] = feature_column
      self._feature_columns[feature_column_name] = feature_column

  def seq_feature_columns_from_name(self, feature_list):
    return [self.seq_feature_columns[fname] for fname in feature_list]

  def sequence_features(self, original_features, keys, length, name,is_context=False):
    """

    :param original_features: A dict which key is string and value is Tensor
    :return: A new tensor pack all element in sequence
    """
    features_list_tensor = OrderedDict()
    sequence_features_tensor = OrderedDict()
    # Init with empty list.
    if is_context:
      feats_name = ['context_type', 'context_time', 'context_st']
    else:
      feats_name = keys
    for feature_name in feats_name:
      feature_tensor_list = []
      for i in range(length):
        feature_name_in_tensor = name + '_' + str(i) + '_' + feature_name
        if feature_name_in_tensor in original_features:
          feature_tensor_list.append(original_features[feature_name_in_tensor])
      if len(feature_tensor_list) > 0:
        features_list_tensor[feature_name] = feature_tensor_list

    for feature_name, tensor_list in features_list_tensor.items():
      if len(tensor_list) == 0:
        continue
      try:
        sequence_features_tensor[feature_name] = tf.sparse_concat(
          sp_inputs=tensor_list,
          axis=0,
          expand_nonconcat_dim=True)
      except:
        sequence_features_tensor[feature_name] = tf.concat(
          values=tensor_list,
          axis=0)
    return sequence_features_tensor

  def item_id_features(self, original_features, length):
    """

    :param original_features: A dict which key is string and value is Tensor
    :return: A new tensor pack all element in sequence
    """
    feature_tensor_list = []
    for i in range(length):
      feature_name_in_tensor = self.sequence_name + '_' + str(i) + '_' + "item_id_d"
      dense_tensor = tf.sparse_tensor_to_dense(original_features[feature_name_in_tensor], default_value="0")
      # dense_tensor = tf.Print(dense_tensor, [dense_tensor], feature_name_in_tensor, summarize=5000)
      dense_tensor = tf.reshape(dense_tensor, [-1, 1])
      feature_tensor_list.append(dense_tensor)

    return tf.concat(values=feature_tensor_list, axis=1)


  def query_id_features(self, original_features, length):
    """

    :param original_features: A dict which key is string and value is Tensor
    :return: A new tensor pack all element in sequence
    """
    feature_tensor_list = []
    for i in range(length):
      feature_name_in_tensor = self.queryseq_name + '_' + str(i) + '_' + "queryseg_norm_d_seqs"
      dense_tensor = tf.sparse_tensor_to_dense(original_features[feature_name_in_tensor], default_value="0")
      dense_tensor = tf.reshape(dense_tensor, [-1, 1])
      feature_tensor_list.append(dense_tensor)

    return tf.concat(values=feature_tensor_list, axis=1)

  @property
  def seq_length(self):
    return self.sequence_length

  def i2i_features(self, original_features):
    """

    :param original_features: A dict which key is string and value is Tensor
    :return: A new tensor pack all element in sequence
    """
    features_list_tensor = OrderedDict()
    sequence_list_tensor = OrderedDict()
    sequence_features_tensor = OrderedDict()
    for feature_name in self._i2i_column_name:
      if 'i2i_seq' in feature_name:
        features_list_tensor[feature_name] = tf.reshape(
          tf.expand_dims(
            tf.slice(original_features[feature_name], [0, 0], [-1, self.i2i_seq_length])
            , -1),
          [-1, 1]
        )  # [?*30, 1]
      else:
        feature_tensor_list = []
        for i in range(self.i2i_seq_length):
          feature_name_in_tensor = self.sequence_name + '_' + str(i) + '_' + feature_name
          # dense_tensor = tf.sparse_tensor_to_dense(original_features[feature_name_in_tensor], default_value="0")
          feature_tensor_list.append(original_features[feature_name_in_tensor])
        sequence_list_tensor[feature_name] = feature_tensor_list

    if len(sequence_list_tensor) > 0:
      for feature_name, tensor_list in sequence_list_tensor.items():
        if len(tensor_list) == 0:
          continue
        try:
          sequence_features_tensor[feature_name] = tf.sparse_concat(
            sp_inputs=tensor_list,
            axis=0,
            expand_nonconcat_dim=True)
        except:
          sequence_features_tensor[feature_name] = tf.concat(
            values=tensor_list,
            axis=0)

      features_list_tensor.update(sequence_features_tensor)
    return features_list_tensor

  @property
  def i2i_seq_length(self):
    return self._feature_column_conf[self._feature_columns[self._i2i_column_name[0]]]['value_dimension']


if __name__ == '__main__':
  fg = FeatureGeneratorSeq('../config/fake_fg_conf.json',
                           model_config=ModelConfig(),
                           mc=ModelColumns("../config/fake_mc_conf.json"))

  origin_features = \
    {"opt_seq_0_item_id_d": tf.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=['a', 'b', 'c'],
                                            dense_shape=[1, 3]),
     "opt_seq_1_item_id_d": tf.SparseTensor(indices=[[0, 0]], values=['1001'],
                                            dense_shape=[1, 1]),
     "opt_seq_0_brand_id": tf.SparseTensor(indices=[[0, 0], [0, 1]], values=['d', 'e'],
                                           dense_shape=[1, 2]),
     "opt_seq_1_brand_id": tf.SparseTensor(indices=[[0, 0]], values=['1002'],
                                           dense_shape=[1, 1]),
     "opt_seq_0_s_nid_pv1_c2c": tf.constant([[1.0, 2.0], [3.0, 4.0]]),
     "opt_seq_1_s_nid_pv1_c2c": tf.constant([[1.0, 2.0], [3.0, 4.0]])
     }
  fg.sequence_features(origin_features)
  print (fg.seq_feature_columns)
