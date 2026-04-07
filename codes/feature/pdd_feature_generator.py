import tensorflow as tf
import json, sys, os
from tensorflow.python.ops import init_ops, array_ops
from tensorflow.contrib import layers
import feature_column_input_layer as input_from_fc
from tensorflow.python.lib.io import file_io
from util.tflog import tflogger as logging
from datetime import date
from random import shuffle
from model.ops import utils
from collections import OrderedDict
from custom_feature_column import real_valued_varlen_column

from model.ops import checkpoint_utils
from model.ops import base_ops
from model.ops import utils

j_feature = 'features'
j_feature_type = 'feature_type'
j_value_type = 'value_type'
j_feature_name = 'feature_name'
j_hash_bucket_size = "hash_bucket_size"
j_embedding_sparse = "embedding_sparse"
j_dimension = "dimension"
j_default_value = 'default_value'
j_bucket_size = 'bucket_size'
j_feature_tag = 'tag'
j_split_config = 'split_config'

tf_type_map = {
    'int64': tf.int64,
    'double': tf.float32,
    'string': tf.string
}

tf_act_map = {
    'relu': tf.nn.relu,
    'identity': tf.identity,
    "None": None
}

is_local = tf.flags.FLAGS.platform == 'local'
multi_val_split = tf.flags.FLAGS.multi_val_split
seq_val_split = tf.flags.FLAGS.seq_val_split
field_val_split = '\t'


def merge_dicts(x, y):
    z = {}
    for k, v in x.items():
        if k in z:
            raise ValueError("[merge_dicts] key %s has touched!" % str(k))
        z[k] = v
    for k, v in y.items():
        if k in z:
            raise ValueError("[merge_dicts] key %s has touched!" % str(k))
        z[k] = v
    # z = x.copy()
    # z.update(y)
    return z


def rescale_sequence_embedding(seq_embedding_tensor, seq_mask, sequence_length):
    xx = tf.shape(seq_embedding_tensor)  # (bs, seq_len, embedding_dim)
    logging.info("seq_embedding_tensor:" + str(seq_embedding_tensor))
    logging.info("seq_mask:" + str(seq_mask))
    logging.info("seq_embedding_tensor shape:" + str(xx))
    seq_len = xx[1]
    pad_shape = tf.stack([xx[0], sequence_length - seq_len], axis=0)
    final_tensor = tf.cond(tf.equal(seq_len, sequence_length), lambda: seq_embedding_tensor,
                           lambda: tf.concat([seq_embedding_tensor, tf.broadcast_to(tf.convert_to_tensor("0", dtype=tf.string), pad_shape)], axis=1))
    mask = tf.cond(tf.equal(seq_len, sequence_length), lambda: seq_mask,
                   lambda: tf.concat([seq_mask, tf.broadcast_to(tf.convert_to_tensor(False, dtype=tf.bool), pad_shape)], axis=1))
    return mask, final_tensor


class FeatureGenerator(object):
    def __init__(self, train_config, schema_config, model_column, use_tag=True):
        self.train_config = train_config
        self.json_conf = schema_config
        self.feature_conf_map = {}
        for fea in self.json_conf[j_feature]:
            if j_feature_tag in fea:
                self.feature_conf_map[str(fea[j_feature_tag])] = fea
        self.others_name = []
        for fea in self.json_conf["others"]:
            self.others_name.append(str(fea[j_feature_name]))
        self.model_config = model_column.simple_dict
        self.model_column = model_column
        self.feature_name_tag_dict = {}
        self.feature_tag_name_dict = {}
        self.feature_columns = {}
        self.feature_lengths = {}
        self.feature_norm = {}
        self.feature_tf_tensors = {}
        self.fcfinal_tf_tensors = {}
        self.split_configs = {}
        self.use_tag = use_tag

        self.user_feature_names = self.model_config.get("user_columns", [])
        self.query_feature_names = self.model_config.get('query_columns', [])
        self.item_feature_names = self.model_config.get('item_columns', [])
        self.ui_feature_names = self.model_config.get('ui_columns', [])
        self.sequence_feature_names = self.model_config.get('sequence_columns', [])
        self.cross_feature_names = self.model_config.get('cross_columns', [])
        self.buy_seq_feature_names = self.model_config.get('buy_seq_columns', [])
        self.fav_seq_feature_names = self.model_config.get('fav_seq_columns', [])
        self.clk_seq_feature_names = self.model_config.get('clk_seq_columns', [])
        logging.info("fg -- fav_seq_feature_names: " + str(self.fav_seq_feature_names))
        logging.info("fg -- item_feature_names: " + str(self.item_feature_names))
        logging.info("fg -- user_feature_names: " + str(self.user_feature_names))

        self.generate_feature_columns()
        self.user_columns = self.feature_columns_from_name(self.user_feature_names)
        self.query_columns = self.feature_columns_from_name(self.query_feature_names)
        self.item_columns = self.feature_columns_from_name(self.item_feature_names)
        self.ui_columns = self.feature_columns_from_name(self.ui_feature_names)
        self.sequence_columns = self.feature_columns_from_name(self.sequence_feature_names)
        self.cross_columns = self.feature_columns_from_name(self.cross_feature_names)
        self.clk_seq_columns = self.feature_columns_from_name(self.clk_seq_feature_names)
        self.buy_seq_columns = self.feature_columns_from_name(self.buy_seq_feature_names)
        self.fav_seq_columns = self.feature_columns_from_name(self.fav_seq_feature_names)

        self.feature_spec, self.context_feature_spec, self.doc_feature_spec = self.make_feature_spec()
        self.seq_length = None
        self.sequence_name = None
        self.shared_embed = {}

    def feature_norm_from_name(self, feature_list):
        if feature_list is None:
            return None
        if self.use_tag:
            return [
                self.feature_norm[self.feature_name_tag_dict[fname]]
                for fname in feature_list
            ]
        return [self.feature_norm[fname] for fname in feature_list]

    def feature_length_from_name(self, feature_list):
        if feature_list is None:
            return None
        if self.use_tag:
            return [
                self.feature_lengths[self.feature_name_tag_dict[fname]]
                for fname in feature_list
            ]
        return [self.feature_lengths[fname] for fname in feature_list]

    def feature_columns_from_name(self, feature_list):
        if feature_list is None:
            return None
        if self.use_tag:
            xxx = []
            for fname in feature_list:
                logging.info(fname)
                logging.info(self.feature_name_tag_dict[fname])
                xxx.append(self.feature_columns[self.feature_name_tag_dict[fname]])
            return xxx
        return [self.feature_columns[fname] for fname in feature_list]

    def _generate_feature_column(self, feature):
        if self.use_tag:
            name = str(feature[j_feature_tag])
        else:
            name = feature[j_feature_name]

        if feature[j_feature_type] == 'sparse':
            sparse_col = tf.feature_column.categorical_column_with_hash_bucket(key=name,
                                                               hash_bucket_size=10 if is_local else
                                                               feature.get(j_embedding_sparse)[j_bucket_size])
            embed_col = tf.feature_column.embedding_column(categorical_column=sparse_col,
                                                dimension=4 if is_local else feature.get(j_embedding_sparse)[
                                                    j_dimension],
                                                combiner=self._get_combiner(feature),
                                                initializer=utils.getInitOp(self.train_config.constant_init,
                                                                            self.train_config.init_op))
            f_column = embed_col
        elif feature[j_feature_type] == 'sequence':
            if feature[j_value_type] == 'string':
                sparse_col = tf.feature_column.categorical_column_with_hash_bucket(key=name,
                                                                   hash_bucket_size=10 if is_local else
                                                                   feature.get(j_embedding_sparse)[j_bucket_size])
                embed_col = tf.feature_column.embedding_column(categorical_column=sparse_col,
                                                    dimension=4 if is_local else feature.get(j_embedding_sparse)[
                                                        j_dimension],
                                                    combiner=self._get_combiner(feature),
                                                    initializer=utils.getInitOp(self.train_config.constant_init,
                                                                                self.train_config.init_op))
                f_column = embed_col
            elif feature[j_value_type] == 'double':
                '''
                numeric_col = real_valued_varlen_column(name,
                                                        default_value=None,
                                                        dtype=tf_type_map[feature[j_value_type]],
                                                        normalizer=None,
                                                        is_sparse=True)
                f_column = numeric_col
                '''
                raise ValueError("sequence feature not support double type")
            else:
                raise ValueError("cannot recognize value type: %s in sequence feature<%s>" % (j_feature_type, name))
        elif feature[j_feature_type] == 'dense':
            numeric_col = real_valued_varlen_column(name,
                                                    default_value=None,
                                                    dtype=tf_type_map[feature[j_value_type]],
                                                    normalizer=None,
                                                    is_sparse=True)
            f_column = numeric_col
        else:
            raise ValueError("cannot recognize feature type: %s" % j_feature_type)
        return name, f_column

    def _get_combiner(self, feature_conf):
        if "combiner" in feature_conf:
            return feature_conf.get(j_embedding_sparse)["combiner"]
        else:
            return "mean"

    def generate_feature_columns(self):
        self.shared_embed = {}
        for feature in self.json_conf[j_feature]:
            a = feature[j_feature_name]
            mc_use = False
            for c, name_list in self.model_config.items():
                if a in name_list:
                    mc_use = True
                    break
            if not mc_use:
                continue
            logging.info("get fc for feature:" + feature[j_feature_name])
            self.feature_name_tag_dict[feature[j_feature_name]] = str(feature[j_feature_tag])
            self.feature_tag_name_dict[str(feature[j_feature_tag])] = feature[j_feature_name]
            if 'shared_name' in feature:
                if feature['shared_name'] in self.shared_embed:
                    self.shared_embed[feature['shared_name']] += [feature]
                else:
                    self.shared_embed[feature['shared_name']] = [feature]
            else:
                name, column = self._generate_feature_column(feature)
                self.feature_columns[name] = column
        for shared_name, feature_set in self.shared_embed.items():
            sparse_id_columns = []
            for feature_conf in feature_set:
                if "vocabulary_file" in feature_conf:
                    if feature_conf[j_feature_type] in ("Int", "int"):
                        raise Exception("embedding with vocabulary_file does not support Int type")
                    else:
                        sparse_id_column = tf.feature_column.categorical_column_with_vocabulary_file(
                            column_name=feature_conf[j_feature_name],
                            vocabulary_file=feature_conf["vocabulary_file"],
                            num_oov_buckets=feature_conf["num_oov_buckets"],
                            vocab_size=feature_conf["vocab_size"]
                        )
                else:
                    sparse_id_column = tf.feature_column.categorical_column_with_hash_bucket(
                        key=str(feature_conf[j_feature_tag]) if self.use_tag else feature_conf[j_feature_name],
                        hash_bucket_size=feature_conf.get(j_embedding_sparse)[j_bucket_size])
                sparse_id_columns.append(sparse_id_column)
            combiner = self._get_combiner(feature_set[0])
            dimension = 4 if is_local else feature_set[0].get(j_embedding_sparse)[j_dimension]
            shared_embedding_columns = tf.feature_column.shared_embedding_columns(
                categorical_columns=sparse_id_columns,
                dimension=dimension,
                combiner=combiner,
                # shared_embedding_name=shared_name,
                max_norm=None,
                initializer=utils.getInitOp(self.train_config.constant_init, self.train_config.init_op))
            for feature_conf, column in zip(feature_set, shared_embedding_columns):
                name = str(feature_conf[j_feature_tag]) if self.use_tag else feature_conf[j_feature_name]
                self.feature_columns[name] = column

    def make_feature_spec(self):
        feature_spec = tf.feature_column.make_parse_example_spec(
            self.user_columns + self.item_columns + self.query_columns + self.sequence_columns + self.clk_seq_columns +
            self.fav_seq_columns + self.buy_seq_columns + self.ui_columns + self.cross_columns)
        context_spec = tf.feature_column.make_parse_example_spec(
            self.user_columns + self.query_columns + self.sequence_columns + self.clk_seq_columns + self.fav_seq_columns + self.buy_seq_columns)
        doc_spec = tf.feature_column.make_parse_example_spec(
            self.item_columns + self.ui_columns + self.cross_columns)
        for datas in self.json_conf["others"]:
            if datas[j_value_type] == "int":
                feature_spec[datas[j_feature_name]] = tf.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0)
                context_spec[datas[j_feature_name]] = tf.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0)
            elif datas[j_value_type] == "double":
                feature_spec[datas[j_feature_name]] = tf.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=0.0)
                context_spec[datas[j_feature_name]] = tf.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=0.0)
            else:
                feature_spec[datas[j_feature_name]] = tf.FixedLenFeature(shape=[1], dtype=tf.string, default_value="")
                context_spec[datas[j_feature_name]] = tf.FixedLenFeature(shape=[1], dtype=tf.string, default_value="")

        logging.info("#" * 10 + "{:^20}".format("infer_feature_spec") + "#" * 10)
        logging.info(feature_spec)
        return feature_spec, context_spec, doc_spec

    #def parse_example(self, example):
    #    return tf.parse_example(example, self.feature_spec)

    def parse_share_context(self, context, doc):
        x1 = tf.parse_example(context, self.context_feature_spec)
        x2 = tf.parse_example(doc, self.doc_feature_spec)
        return merge_dicts(x1, x2)

    def parse_csv(self, csv):
        #fea_name_list = ['positive','uid', 'gid', '__id', 'bktid']
        #fea_default_list = [tf.constant(0,tf.int64), "", "", "", 100]
        fea_name_list = []
        fea_default_list = []
        for f in self.json_conf["others"]:
            fea_name_list.append(str(f[j_feature_name]))
            def_val = 0
            if f[j_value_type] == 'string':
                def_val = ""
            elif f[j_value_type] == 'double':
                def_val = 0.0
            fea_default_list.append(f.get(j_default_value, def_val))
        for f in self.json_conf[j_feature]:
            fea_name_list.append(str(f[j_feature_tag]))
            #fea_default_list.append(f.get(j_default_value, "0" if f[j_feature_type] == 'sparse' else 0.0))
            def_val = 0
            if f[j_value_type] == 'string':
                def_val = ""
            elif f[j_value_type] == 'double':
                def_val = ""
            fea_default_list.append(f.get(j_default_value, def_val))

        feas = tf.decode_csv(csv, fea_default_list, field_delim=field_val_split, use_quote_delim=False)
        fea_map = {}
        ccccc = 0
        for a, b in zip(fea_name_list, feas):
            ccccc += 1
            logging.info("%d th feature is %s" % (ccccc, a))
            mc_use = False
            for c, name_list in self.model_config.items():
                if a in self.others_name:
                    mc_use = True
                    break
                if a in self.feature_tag_name_dict and self.feature_tag_name_dict[a] in name_list:
                    mc_use = True
                    break
            if not mc_use:
                continue
            logging.info("decode csv for feature: "+a)
            if a in self.feature_conf_map and self.feature_conf_map[a][j_feature_type] == "sequence":
                '''
                if self.feature_conf_map[a][j_value_type] == "double":
                    seqfs = tf.decode_csv(b, [-1.0] * self.feature_conf_map[a]["sequence_length"], field_delim=seq_val_split)
                elif self.feature_conf_map[a][j_value_type] == "string":
                    seqfs = tf.decode_csv(b, ["0"] * self.feature_conf_map[a]["sequence_length"], field_delim=seq_val_split)
                    seqfs = [tf.string_split(x, '#') for x in seqfs]
                else:
                    raise ValueError("cannot recognize value type: %s " % j_value_type)
                for i in range(self.feature_conf_map[a]["sequence_length"]):
                    fea_map[a+"_"+str(i)] = seqfs[i]
                '''
                fea_map[a] = tf.sparse_fill_empty_rows(tf.string_split(b, seq_val_split, skip_empty=False), "")[0]  # [?,~seq_len]
                '''
                if self.feature_conf_map[a][j_value_type] == "double":
                    fea_map[a] = tf.cast(fea_map[a], tf.float32)
                '''
            elif a in self.others_name:
                #with tf.variable_scope(name_or_scope=a):
                fea_map[a] = tf.reshape(b, [-1, 1])
            elif a not in self.others_name and self.feature_conf_map[a][j_value_type] != 'string':
                with tf.variable_scope(name_or_scope=a):
                    t1 = tf.sparse_fill_empty_rows(tf.string_split(b, multi_val_split, skip_empty=False), "")[0]
                    # fea_map[a] = tf.string_to_number(tf.string_split(b, multi_val_split), tf.float32)
                    fea_map[a] = tf.SparseTensor(indices=t1.indices, values=tf.string_to_number(t1.values, tf.float32), dense_shape=t1.dense_shape)
            else:
                with tf.variable_scope(name_or_scope=a):
                    fea_map[a] = tf.sparse_fill_empty_rows(tf.string_split(b, multi_val_split, skip_empty=False), "")[0]
        #return fname_list,[fea_map[k] for k in fname_list]
        logging.info("#" * 10 + "{:^20}".format("train_feature_spec") + "#" * 10)
        logging.info(fea_map)
        return fea_map

    def sequence_features1(self, original_features, keys, length=None, name=None, is_context=False):
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
            if feature_name not in self.sequence_feature_names:
                raise ValueError("not such sequence feature")
            if self.use_tag:
                feature_name = self.feature_name_tag_dict[feature_name]
            sequence_length = self.feature_conf_map[feature_name]["sequence_length"]
            feature_tensor_list = []
            for i in range(sequence_length):
                feature_name_in_tensor = feature_name + '_' + str(i)
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

    def sequence_features(self, original_features, keys, length=None, name=None, is_context=False):
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
            if feature_name not in self.sequence_feature_names:
                raise ValueError("not such sequence feature")
            if self.use_tag:
                feature_name = self.feature_name_tag_dict[feature_name]
            sequence_length = self.feature_conf_map[feature_name]["sequence_length"]

            tmp = original_features[feature_name]  # [?, ~seq_len]
            paded = tmp  # tf.SparseTensor(indices=tmp.indices, values=tmp.values, dense_shape=[-1, sequence_length])
            sequence_features_tensor[feature_name] = tf.sparse_reshape(paded, [-1, 1])
        '''    
            feature_tensor_list = []
            for i in range(sequence_length):
                feature_name_in_tensor = feature_name + '_' + str(i)
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
        '''
        return sequence_features_tensor

    def build_tf_tensors_share_context(self, examplestr, use_tfrecord_ph, share_context_ph, context_eamplestr, pms):
        #self.feature_tf_tensors = tf.cond(use_tfrecord_ph,
        #                                   lambda: self.parse_share_context(context_eamplestr, examplestr),
        #                                   lambda: self.parse_csv(examplestr))
        if pms:
            feature_tf_tensors = self.parse_share_context(context_eamplestr, examplestr)
        else:
            feature_tf_tensors = self.parse_csv(examplestr)
        ### expand sequence features
        a = [self.fav_seq_feature_names, self.buy_seq_feature_names, self.clk_seq_feature_names]
        b = [self.model_column.config_dict['fav_seq_columns'][0]['sequence_length'],
             self.model_column.config_dict['buy_seq_columns'][0]['sequence_length'],
             self.model_column.config_dict['clk_seq_columns'][0]['sequence_length']]
        for names, seq_len in zip(a, b):
            for name in names:
                tag = self.feature_name_tag_dict[name]
                sparse_id = feature_tf_tensors[tag]
                dense_id = tf.sparse_tensor_to_dense(sparse_id, default_value="0")  # (bs, seq_len)
                # dense_id = tf.expand_dims(dense_id, -1)  # (bs, seq_len, 1)
                mask = tf.not_equal(dense_id, "0")
                mask, dense_id = rescale_sequence_embedding(dense_id, mask, sequence_length=seq_len)
                feature_tf_tensors[tag] = tf.reshape(dense_id, [-1, 1])
                feature_tf_tensors[name+'_mask'] = mask
        ### embedding ids into tensor space
        with tf.variable_scope(
                name_or_scope="input_from_feature_columns",
                partitioner=base_ops.partitioner(self.train_config.ps_num,
                                                 mem=8 * 1024 * 1024), reuse=tf.AUTO_REUSE) as scope:
            fcfinal_tf_tensors = {}
            # document features
            for name in self.item_feature_names + self.ui_feature_names + self.cross_feature_names:
                logging.info("input_from_one_feature_column feature:" + name)
                tag = self.feature_name_tag_dict[name]
                fcfinal_tf_tensors[name] = input_from_fc.input_from_one_feature_column(
                    {tag: feature_tf_tensors[tag]},
                    self.feature_columns_from_name([name]),
                    use_fast_embeding=pms,
                    scope=scope
                )
            # context features
            # batch_size = tf.shape(self.fcfinal_tf_tensors[self.item_feature_names[0]])[0]
            for name in self.user_feature_names + self.query_feature_names + self.sequence_feature_names + self.fav_seq_feature_names + self.buy_seq_feature_names + self.clk_seq_feature_names:
                logging.info("input_from_one_feature_column feature:" + str(name))
                tag = self.feature_name_tag_dict[name]
                fcfinal_tf_tensors[name] = input_from_fc.input_from_one_feature_column(
                    {tag: feature_tf_tensors[tag]},
                    self.feature_columns_from_name([name]),
                    use_fast_embeding=pms,
                    scope=scope
                )
            # other info
            for name, tensor in feature_tf_tensors.items():
                logging.info("origin feature:"+name)
                if name not in fcfinal_tf_tensors:
                    logging.info("origin feature insert into fcfinal:" + name)
                    fcfinal_tf_tensors[name] = tensor
        logging.info("final features:")
        logging.info(fcfinal_tf_tensors)
        return feature_tf_tensors, fcfinal_tf_tensors


def parse_config(config_file):
    return json.load(open(config_file))
    #return json.loads(config_file, object_pairs_hook=OrderedDict);

if __name__ == "__main__":
    f = FeatureGenerator(parse_config("../jsonfile/pdd_ltr_conf_v1.json"),
                         parse_config("../jsonfile/mc_pdd_ltr_ctr_v1.json"), use_tag=True)
    print(f.make_feature_spec())
    print(f.feature_columns)
    print(f.item_feature_names)
    print(f.user_feature_names)
    print(f.query_feature_names)
