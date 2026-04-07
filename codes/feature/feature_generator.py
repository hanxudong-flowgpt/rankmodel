# coding=utf-8
import json

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.contrib.layers.python.layers.feature_column import _EmbeddingColumn

from util.config.baseconfig import ModelConfig
import custom_feature_column
from model.model_column import ModelColumns
from util.tflog import tflogger as logging
from model.ops import utils

feature_key = 'features'
value_type_key = 'value_type'
feature_type_key = 'feature_type'
feature_name_key = 'feature_name'
alias_name = 'alias_name'
cross_columns = "cross_columns"
SKIP_KEY = 'skip'


class FeatureGenerator:
  def __init__(self, conf_file_path, model_config, mc, training_config=None, cross_file_path=None):
    self.is_local = False
    self.train_conf = training_config
    if training_config is not None:
      self.is_local = training_config.is_local
      self.init_op = training_config.init_op
      self.constant_init = training_config.constant_init
    self._model_config = model_config
    self._mc_filter = mc.all_columns_in_need
    self._parse_feature_conf(conf_file_path)
    self._generate_feature_columns()
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
        if self.train_conf.debug and "hash_bucket_size" in feature_conf:
          feature_conf["hash_bucket_size"] = 10
        feature_name = feature_conf[alias_name] if alias_name in feature_conf else feature_conf[feature_name_key]
        if self._mc_filter and feature_name in self._mc_filter:
          self._feature_conf_map[feature_name] = feature_conf
          if alias_name not in feature_conf:
            feature_conf_list_filtered.append(feature_conf)
      self._config = {feature_key: feature_conf_list_filtered}
      logging.info('[self._config]:%s' % str(self._config))
      print(self._config)

  def _generate_feature_columns(self):
    self._feature_columns = dict()
    self._feature_column_conf = dict()
    shared_feature_map = dict()
    for key, value in self._feature_conf_map.items():
      if value.has_key(SKIP_KEY) and value[SKIP_KEY] == True:
        continue
      if 'shared_name' in value and "boundaries" not in value:
        shared_name = value['shared_name']
        shared_feature_map[shared_name].append(
          value) if shared_name in shared_feature_map else shared_feature_map.setdefault(
          shared_name, [value])
      else:
        feature = self._new_feature(value[feature_name_key], value)
        fname = value[alias_name] if alias_name in value else value[feature_name_key]
        self._feature_columns[fname] = feature
        self._feature_column_conf[feature] = value
    for key, value in shared_feature_map.items():
      sparse_id_columns = []
      for feature_conf in value:
        if "vocabulary_file" in feature_conf:
          if feature_conf[value_type_key] == "Int":
            raise Exception("embedding with vocabulary_file does not support Int type")
          else:
            sparse_id_column = fc.sparse_column_with_vocabulary_file(
              column_name=feature_conf[feature_name_key],
              vocabulary_file=feature_conf["vocabulary_file"],
              num_oov_buckets=feature_conf["num_oov_buckets"],
              vocab_size=feature_conf["vocab_size"]
            )
        else:
          sparse_id_column = layers.sparse_column_with_hash_bucket(
            column_name=feature_conf[feature_name_key],
            hash_bucket_size=feature_conf['hash_bucket_size'])
        sparse_id_columns.append(sparse_id_column)
      combiner = _get_combiner(value[0])
      dimension = value[0]['embedding_dimension']
      shared_embedding_columns = layers.shared_embedding_columns(
        sparse_id_columns,
        dimension,
        combiner,
        key,
        max_norm=None,
        initializer=utils.getInitOp(self.constant_init, self.init_op))
      for feature_conf, column in zip(value, shared_embedding_columns):
        fname = feature_conf[alias_name] if alias_name in feature_conf else feature_conf[feature_name_key]
        self._feature_columns[fname] = column
        self._feature_column_conf[column] = feature_conf

  # @staticmethod
  def _new_feature(self, name, feature_conf):
    """
    新建feature_column，模型训练时使用
    :param name: 特征名称
    :param feature_conf: 特征配置
    :return: 返回一个新建的feature_column
    """
    if not feature_conf.has_key(feature_type_key):
      return
    value_type = feature_conf[value_type_key]
    if "hash_bucket_size" in feature_conf:
        print(feature_conf["feature_name"], feature_conf["hash_bucket_size"])
    if "hash_bucket_size" in feature_conf \
        and "embedding_dimension" not in feature_conf:
      use_hashmap = feature_conf.get("use_hashmap", False)
      if value_type == "Int":
        id_feature = layers.sparse_column_with_integerized_feature(
          column_name=name,
          bucket_size=feature_conf['hash_bucket_size'] if not self.train_conf.debug else 10,
          combiner=_get_combiner(feature_conf),
          # use_hashmap=use_hashmap
        )
      else:
        id_feature = layers.sparse_column_with_hash_bucket(
          column_name=name,
          hash_bucket_size=feature_conf['hash_bucket_size'] if not self.train_conf.debug else 10,
          combiner=_get_combiner(feature_conf),
          # use_hashmap=use_hashmap
        )
      return id_feature
    elif "embedding_dimension" in feature_conf \
        and "shared_name" in feature_conf and "boundaries" not in feature_conf:
      use_hashmap = feature_conf.get("use_hashmap", False)
      if value_type == "Int":
        raise Exception("embedding with vocabulary_file does not support Int type")
      else:
        id_feature = layers.sparse_column_with_hash_bucket(
          column_name=name,
          hash_bucket_size=feature_conf['hash_bucket_size'] if not self.train_conf.debug else 10,
          # use_hashmap=use_hashmap
        )
        return _EmbeddingColumn(
          id_feature,
          dimension=feature_conf['embedding_dimension'],
          combiner=_get_combiner(feature_conf),
          shared_embedding_name=feature_conf["shared_name"],
          max_norm=None,
          initializer=utils.getInitOp(self.constant_init, self.init_op))
    elif "embedding_dimension" in feature_conf \
        and "vocabulary_file" in feature_conf:
      use_hashmap = feature_conf.get("use_hashmap", False)
      if value_type == "Int":
        raise Exception("embedding with vocabulary_file does not support Int type")
      else:
        id_feature = fc.sparse_column_with_vocabulary_file(
          column_name=name,
          vocabulary_file=feature_conf["vocabulary_file"],
          num_oov_buckets=feature_conf["num_oov_buckets"],
          vocab_size=feature_conf["vocab_size"],
        )
        return layers.embedding_column(
          id_feature,
          dimension=feature_conf['embedding_dimension'],
          combiner=_get_combiner(feature_conf),
          max_norm=None,
          initializer=utils.getInitOp(self.constant_init, self.init_op))
    elif "embedding_dimension" in feature_conf and "hash_bucket_size" in feature_conf:
      use_hashmap = feature_conf.get("use_hashmap", False)
      if value_type == "Int":
        return layers.embedding_column(
          sparse_id_column=layers.sparse_column_with_integerized_feature(
            column_name=name,
            bucket_size=feature_conf['hash_bucket_size'] if not self.train_conf.debug else 10,
            combiner=_get_combiner(feature_conf),
            # use_hashmap=use_hashmap
          ),
          dimension=feature_conf['embedding_dimension'],
          combiner=_get_combiner(feature_conf), initializer=utils.getInitOp(self.constant_init, self.init_op))
      else:
        id_feature = layers.sparse_column_with_hash_bucket(
          column_name=name,
          hash_bucket_size=feature_conf['hash_bucket_size'] if not self.train_conf.debug else 10,
          # use_hashmap=use_hashmap
        )
        return layers.embedding_column(
          id_feature,
          dimension=feature_conf['embedding_dimension'],
          combiner=_get_combiner(feature_conf),
          max_norm=None,
          initializer=utils.getInitOp(self.constant_init, self.init_op))
    elif "boundaries" in feature_conf and "embedding_dimension" in feature_conf:
      return custom_feature_column.embedding_bucketized_column(layers.real_valued_column(
        column_name=name,
        dimension=feature_conf.get('dimension', 1),
        default_value=[0.0 for _ in range(int(feature_conf.get('dimension', 1)))]),
        boundaries=[float(b) for b in feature_conf['boundaries'].split(',')],
        embedding_dimension=feature_conf["embedding_dimension"],
        max_norm=None,
        initializer=utils.getInitOp(self.constant_init, self.init_op),
        is_local=self.is_local,
        shared_name=feature_conf.get('shared_name', None),
        add_random=feature_conf.get('add_random', False)
      )
    elif "boundaries" in feature_conf:
      return layers.bucketized_column(
        layers.real_valued_column(
          column_name=name,
          dimension=feature_conf.get('dimension', 1),
          default_value=[0.0 for _ in range(int(feature_conf.get('dimension', 1)))]
        ),
        boundaries=[float(b) for b in feature_conf['boundaries'].split(',')]
      )
    elif "l2_norm" in feature_conf and 'raw_feature' in feature_conf:
      return layers.real_valued_column(
        column_name=name,
        dimension=feature_conf.get('value_dimension', 1),
        default_value=[0.0 for _ in range(int(feature_conf.get('value_dimension', 1)))]
        , normalizer=lambda x: tf.nn.l2_normalize(x, dim=-1)
      )
    else:
      return layers.real_valued_column(
        column_name=name,
        dimension=feature_conf.get('value_dimension', 1),
        default_value=[0.0 for _ in range(int(feature_conf.get('value_dimension', 1)))]
        ,
        # normalizer=lambda x: (
        #   (x - float(feature_conf.get('avg', 0))) / (
        #     0.000001 + float(feature_conf.get('stddev', 1)))) if feature_conf.get(
        #   'batch_norm', False) == False else (layers.batch_norm(
        #   tf.reshape(x, [-1, int(feature_conf.get('value_dimension', 1))])
        #   , is_training=self._model_config.is_training)
        # )
      )

  def _generate_cross_column(self, cross_file_path):
    if cross_file_path is None:
      return
    with open(cross_file_path) as f:
      _config = json.load(f)
      feature_conf_list = _config[feature_key]
      for feature_conf in feature_conf_list:
        if "_comment" in feature_conf:
          continue
        feature = self._new_cross_features(feature_conf)
        if feature is not None:
          self._feature_columns[feature_conf[feature_name_key]] = feature
          self._feature_column_conf[feature] = feature_conf

  def _gen_columns_for_cross_columns(self, str_cross_columns):
    columns = []
    arr_cross_columns = str_cross_columns.split(',', -1)
    for i in range(len(arr_cross_columns)):
      if arr_cross_columns[i] not in self._feature_columns:
        continue
      if isinstance(self._feature_columns[arr_cross_columns[i]],
                    (fc._SparseColumn, fc._CrossedColumn, fc._BucketizedColumn)):

        columns.append(self._feature_columns[arr_cross_columns[i]])
      elif "hash_bucket_size" in self._feature_conf_map[arr_cross_columns[i]]:
        use_hashmap = self._feature_conf_map[arr_cross_columns[i]].get("use_hashmap", False)
        if self._feature_conf_map[arr_cross_columns[i]]["value_type"] == "Int":
          id_feature = layers.sparse_column_with_integerized_feature(
            column_name=self._feature_conf_map[arr_cross_columns[i]][feature_name_key],
            bucket_size=self._feature_conf_map[arr_cross_columns[i]]['hash_bucket_size'] if not self.train_conf.debug else 10,
            combiner=_get_combiner(self._feature_conf_map[arr_cross_columns[i]]),
            # use_hashmap=use_hashmap
          )
        else:
          id_feature = layers.sparse_column_with_hash_bucket(
            column_name=self._feature_conf_map[arr_cross_columns[i]][feature_name_key],
            hash_bucket_size=self._feature_conf_map[arr_cross_columns[i]]['hash_bucket_size'] if not self.train_conf.debug else 10,
            combiner=_get_combiner(self._feature_conf_map[arr_cross_columns[i]]),
            # use_hashmap=use_hashmap
          )
        columns.append(id_feature)
      elif "boundaries" in self._feature_conf_map[arr_cross_columns[i]]:
        bucket_column = layers.bucketized_column(
          layers.real_valued_column(
            column_name=self._feature_conf_map[arr_cross_columns[i]][feature_name_key],
            dimension=self._feature_conf_map[arr_cross_columns[i]].get('dimension', 1),
            default_value=[0.0 for _ in range(int(self._feature_conf_map[arr_cross_columns[i]].get('dimension', 1)))]
          ),
          boundaries=[float(b) for b in self._feature_conf_map[arr_cross_columns[i]]['boundaries'].split(',')]
        )
        columns.append(bucket_column)
    return columns

  def _new_cross_features(self, feature_conf):
    if not feature_conf.has_key(cross_columns) \
        or not feature_conf.has_key("hash_bucket_size"):
      return

    columns = self._gen_columns_for_cross_columns(feature_conf[cross_columns])
    if len(columns) < 2: return
    if "embedding_dimension" in feature_conf:
      return layers.embedding_column(
        layers.crossed_column(
          columns,
          feature_conf["hash_bucket_size"] if not self.train_conf.debug else 10,
          combiner=feature_conf.get('combiner', 'sqrtn'),
          ckpt_to_load_from=None,
          tensor_name_in_ckpt=None,
          hash_key=None
        ),
        dimension=feature_conf['embedding_dimension'],
        combiner=feature_conf.get('combiner', 'sqrtn'),
        max_norm=1)
    else:
      return layers.crossed_column(
        columns,
        feature_conf["hash_bucket_size"] if not self.train_conf.debug else 10,
        combiner=feature_conf.get('combiner', 'sqrtn'),
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        hash_key=None
      )

  @property
  def feature_conf(self):
    return self._config

  @property
  def feature_conf_map(self):
    return self._feature_conf_map

  @property
  def feature_column_conf(self):
    return self._feature_column_conf

  @property
  def feature_columns(self):
    """
    取得所有的feature_columns
    :return: 返回一个list包含所有的feature_column
    """
    return [self._feature_columns[fname] for fname in self._feature_columns.keys()]

  def feature_columns_from_name(self, feature_list):
    """
    取得指定特征名称的feature_columns
    :param feature_list: 特征名称
    :return: 返回一个list，对应feature_list的feature_column
    """
    x = []
    for fname in feature_list:
      if fname in self._feature_columns:
        x.append(self._feature_columns[fname])
    return x
    # [self._feature_columns[fname] for fname in feature_list]


def _get_combiner(feature_conf):
  if "combiner" in feature_conf:
    return feature_conf["combiner"]
  else:
    return "mean"


if __name__ == '__main__':
  fg = FeatureGenerator('../config/lsc_conf.json', model_config=ModelConfig(),
                        mc=ModelColumns("../config/mc_conf.json"))
  print (fg.feature_conf_map)
