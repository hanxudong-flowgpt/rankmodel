# coding:utf-8
#****************************************************************#
# Author: qiaohan.qh@alibaba-inc.com
# Create Date: 2018-03-18 17:50
# Modify Author: qiaohan.qh@alibaba-inc.com
# Modify Date: 2018-03-18 17:50
# Function:
#***************************************************************#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import tensorflow as tf
import json, copy
from collections import OrderedDict
from util.tflog import tflogger as logging
# from graph_io import read_batch_features,read_batch_record_features


def parse_feature_conf(config, mc_filter, sequence_filter, islocal):
    feature_conf_map = dict()
    feature_conf_list = config['features']
    feature_conf_list_filtered = []
    for feature_conf in feature_conf_list:
        if "_comment" in feature_conf:
            continue
        if islocal:
            for k, v in feature_conf.items():
                if "hash_bucket_size" == k:
                    feature_conf["hash_bucket_size"] = 10
                if type(v) == list:
                    for vkv in v:
                        if "hash_bucket_size" in vkv:
                            vkv["hash_bucket_size"] = 10
        feature_name = feature_conf["alias_name"] if "alias_name" in feature_conf else feature_conf["feature_name"]
        if feature_name in mc_filter:
            feature_conf_map[feature_name] = feature_conf
            if "alias_name" not in feature_conf:
                feature_conf_list_filtered.append(feature_conf)

        if "sequence_name" in feature_conf:
            if islocal:
                feature_conf_local = copy.deepcopy(feature_conf)
                feature_conf_local["features"] = []
                for ele_feature_conf in feature_conf["features"]:
                    feature_conf_local["features"].append(ele_feature_conf)
                feature_conf = feature_conf_local
            sequence_features_filtered = []
            features = feature_conf["features"]
            feature_conf_map[feature_name] = feature_conf
            feature_conf_copy = copy.deepcopy(feature_conf)

            for feature in features:
                # seq_feature_name = feature[FEATURE_NAME_KEY]
                seq_feature_name = feature["alias_name"] if "alias_name" in feature else feature["feature_name"]
                if sequence_filter and seq_feature_name in sequence_filter:
                    if "alias_name" not in feature:
                        sequence_features_filtered.append(feature)
            feature_conf_copy["features"] = sequence_features_filtered
            feature_conf_list_filtered.append(feature_conf_copy) 

    print("[feature number] %d" % len(feature_conf_list_filtered))
    return {"features": feature_conf_list_filtered}, feature_conf_map


def get_longseq_expend_fgconf(jsonfile):
    fgconf = json.load(open(jsonfile),object_pairs_hook=OrderedDict)
    fgconf = fgconf["features"]
    expandfgconf = []
    seq_fea = []
    for feaconf in fgconf:
        if "sequence_name" in feaconf:
            # 这里是把seqfeature的每个特征 比如nid category 展开
            # 展开后 和 fg之后的表 对应上了
            seq_fea.append(feaconf["sequence_name"])
            for sfea in feaconf["features"]:
                newsfea = sfea.copy()
                # col_name是训练过程中的name
                newsfea["col_name"] = feaconf["sequence_name"]+"_"+sfea["feature_name"]
                newsfea["force_prefix_name"] = newsfea["feature_name"]
                newsfea["sequence_name"] = feaconf["sequence_name"]
                newsfea["sequence_length"] = feaconf["sequence_length"]
                expandfgconf.append(newsfea)
        else:
            feaconf["col_name"] = feaconf["feature_name"]
            feaconf["force_prefix_name"] = feaconf["feature_name"]
            expandfgconf.append(feaconf)
    return expandfgconf

def get_feature_conf(feablacklist, jsonfile, odps_start_col):
    # 展开feature json 展开后和fg之后的schema对应 名字也完全对应
    expandfgconf = get_longseq_expend_fgconf(jsonfile)

    # 存了和odps fg表里一致的id和feature schema
    odps_column_idx = []
    odps_column_name = []
    # 存了和sample_data一致的feature schema
    feature_name = []
    # 存了和sample_data一致的feature json
    fg_attrs = {}
    # 存了和odps fg表里一致的feature json
    column_attrs = {}

    f_default = []

    column_num = odps_start_col-1
    for feaconf in expandfgconf:
        column_num += 1
        feaconf["use_default_key"] = True
        # 多值特征
        if "combiner" not in feaconf:
            feaconf["combiner"] = "mean"
        if "default_key" not in feaconf:
            if "sequence_name" in feaconf or feaconf["value_type"] == "Double":
                feaconf["default_key"] = "0"
            else:
                feaconf["default_key"] = feaconf["feature_name"]

        odps_column_name.append(feaconf["col_name"])
        odps_column_idx.append(column_num)
        column_attrs[feaconf["col_name"]] = feaconf
        # 这里把seq的conf拆开 和input_fn对应起来了
        # 因此fg_attrs和sample_data完全对应
        if "sequence_name" in feaconf:
            f_default.append(["xxx"])
            for i in xrange(feaconf["sequence_length"]):
                newsfea = feaconf.copy()
                newsfea["feature_name"] = feaconf["sequence_name"] + "_" + str(i) + "_" + feaconf["feature_name"]
                newsfea["expression"] = 'user'+':'+newsfea["feature_name"]
                newsfea["sequence_idx"] = i

                fg_attrs[newsfea["feature_name"]] = newsfea
                feature_name.append(newsfea["feature_name"])
        else:
            fg_attrs[feaconf["feature_name"]] = feaconf
            feature_name.append(feaconf["feature_name"])
            if feaconf["value_type"] == "Double":
                f_default.append([0.0])
            else:
                f_default.append([feaconf["feature_name"] + "_"])

    for i in range(len(feature_name)):
        print(str(i)+'th expanded feature: '+feature_name[i])
        print(fg_attrs[feature_name[i]])
    return feature_name, fg_attrs, column_attrs, odps_column_name, odps_column_idx,f_default


def get_sample_desc(feablacklist, jsonfile, odps_start_col, label_column, embedding_table_config):
    # 解析feature的json
    # feature_name ---- 存了和sample_data一致的feature schema
    # fg_attrs ---- 存了和sample_data一致的feature json
    # column_attrs ---- 存了和odps fg表里一致的feature json
    # odps_column_name ---- 存了和odps fg表里一致的feature schema
    # odps_column_idx ---- 存了和odps fg表里一致的位置id
    # f_default ---- 存了odps fg表里一致的默认值
    feature_name, fg_attrs, column_attrs, odps_column_name, odps_column_idx, f_default = get_feature_conf(feablacklist,jsonfile,odps_start_col-1)
    if type(label_column)==list:
        labels = label_column
    else:
        labels = [label_column-1]
    #desc = get_compound_sample_descriptor(feature_name, feature_column, feature_storage_type, labels)
    fgconf = json.load(open(jsonfile),object_pairs_hook=OrderedDict)
    label_name = fgconf["reserves"] if "reserves" in fgconf else ["label","id"]
    if list!=type(label_name):
        label_name = label_name.split(",")
    col_names = label_name+odps_column_name
    col_default = []

    for i,lname in enumerate(label_name):
        print(str(i)+'th label: '+lname)
        col_default.append(['0'])
    col_default = col_default+f_default

    # appending other info
    # 这里添加上了embedding的信息
    embeddingtable = {}
    for t in embedding_table_config:
        embeddingtable[t["matrix_name"]] = t
    for n in feature_name:
        x = fg_attrs[n]
        if "Double"!=x["value_type"]:
            x["embedding_dimension"] = embeddingtable[x["matrix_name"]]["embed_size"]
            x["hash_bucket_size"] = embeddingtable[x["matrix_name"]]["feature_size"]
            #x["force_prefix_name"] = x["prefix_name"]
            if "shared_embedding_name" not in x and "matrix_name" in x:
                x["shared_embedding_name"] = x["matrix_name"]
    return col_names,col_default,column_attrs,feature_name,fg_attrs


def input_fn(native, slice_count, slice_id, tables, bs, col_names, col_default, feature_conf, fromodps=True,
             num_epochs=1):
    tf.logging.info("table name:%s, slice_count:%d, slice_id:%d" % (tables, slice_count, slice_id))
    tables = tf.constant(tables, dtype=tf.string)
    if native:
        file_queue = tf.train.string_input_producer(tables, num_epochs=num_epochs)
    else:
        # import tensorflow.contrib.global_input as gi
        # partitions = gi.partition_filenames_once(tables, partition_size=slice_count*5)
        # file_queue = gi.global_string_input_producer(partitions, is_chief=slice_id==0, num_epochs=1)
        file_queue = tf.train.string_input_producer(tables, num_epochs=num_epochs)

    if native or not fromodps:
        reader = tf.TextLineReader()
    else:
        tf.logging.info("slice_count:%d, slice_id:%d" % (slice_count, slice_id))
        reader = tf.TableRecordReader(csv_delimiter="\x02",
                                      selected_cols=','.join(col_names),
                                      slice_count=slice_count,
                                      slice_id=slice_id,
                                      num_threads=2,
                                      capacity=2 * bs)

    key, value = reader.read_up_to(file_queue, num_records=bs)
    batch_values = tf.train.batch([value], batch_size=bs, num_threads=2, capacity=2 * bs, enqueue_many=True)
    for name, defa in zip(col_names, col_default):
        tf.logging.info("col_name:%s, default value:%s" % (name, defa))
    listofcol = tf.decode_csv(batch_values, record_defaults=col_default, field_delim="\x02")
    # labels = tf.reshape(tf.cast(v4, tf.int32), [128])
    # features = tf.stack([v1, v2, v3], axis=1)
    tensors_col = []
    for col_name, col_tensor in zip(col_names, listofcol):
        print(col_name, col_tensor)
        if col_name not in feature_conf:
            # reserve odps columns
            tensors_col.append(tf.expand_dims(col_tensor, 1))
            continue
        feaconf = feature_conf[col_name]
        if "sequence_name" in feaconf:
            # col_tensor = tf.squeeze(col_tensor, axis=[1])
            # print "squeeze:",col_name,col_tensor
            if feaconf["value_type"] == "Double":
                tensors = tf.decode_csv(col_tensor, record_defaults=[[0.0]] * feaconf["sequence_length"],
                                        field_delim='\x01', name="seq_input_" + feaconf["sequence_name"])
            else:
                tensors = tf.decode_csv(col_tensor, record_defaults=[["0"]] * feaconf["sequence_length"],
                                        field_delim='\x01', name="seq_input_" + feaconf["sequence_name"])
            # tensors = tf.stack(tensors, 1)
            # tensors = tf.reshape(tensors, [None, feaconf["sequence_length"]])
        elif feaconf["value_type"] == "Double":
            # tensors = tf.reshape(col_tensor, [None,1])
            tensors = [tf.expand_dims(col_tensor, 1)]
        else:
            # print type(col_tensor),col_tensor.get_shape()
            # print tf.reshape(col_tensor, [None]).get_shape()
            # print tf.reshape(col_tensor, [None,1]).get_shape()
            tensors = [tf.string_split(col_tensor, "\x01")]
        print(col_name, tensors)
        tensors_col += tensors
    return tensors_col


def get_tensor(sampledata, feature_name, feature_conf, embedding_table_config_map, native, part_num=20,
                not_allow_hash_conflict=True):
    '''
    tensor to feature_column tensor
    :param sampledata:
    :param feature_name:
    :param feature_conf:
    :param embedding_table_config_map:
    :return:
    '''
    laz_fe = {}
    catecolumns_withsametable = {}
    catecolnames_withsametable = {}
    for x in embedding_table_config_map.keys():
        catecolumns_withsametable[x] = []
        catecolnames_withsametable[x] = []
    rawcolumns = {}
    for i in range(len(sampledata)):
        orginal_tensor = sampledata[i]
        feaconf = feature_conf[feature_name[i]]
        cur_name = feature_name[i]
        laz_fe[cur_name] = orginal_tensor
        # builder = _LazyBuilder(laz_fe)
        if feaconf['value_type'] == 'Double':
            raw_column = tf.contrib.layers.real_valued_column(cur_name)
            rawcolumns[cur_name] = raw_column
        elif feaconf['value_type'] == 'String' and 'matrix_name' in feaconf:
            hash_size = 100 if native else embedding_table_config_map[feaconf['matrix_name']]['feature_size']
            # no conflict embedding ,using use_hashmap=True
            id_column = tf.contrib.layers.sparse_column_with_hash_bucket(cur_name, hash_size,
                                                                         combiner="mean" if "combiner" not in feaconf else
                                                                         feaconf["combiner"],
                                                                         use_hashmap=not_allow_hash_conflict)
            catecolumns_withsametable[feaconf['matrix_name']].append(id_column)
            catecolnames_withsametable[feaconf['matrix_name']].append(cur_name)
        else:
            raise (ValueError(
                feaconf["feature_name"] + ' matrix_name-- ' + feaconf["matrix_name"] + ' not found!'))

    fctensor_map = {}
    for k, v in catecolumns_withsametable.items():
        d = embedding_table_config_map[k]['embed_size']
        uinit = embedding_table_config_map[k]['uinit'] if 'uinit' in embedding_table_config_map[k] else 1e-8
        with tf.variable_scope("fc", partitioner=tf.fixed_size_partitioner(part_num, axis=0),
                               initializer=tf.random_uniform_initializer(minval=-uinit, maxval=uinit)):
            fcs = tf.contrib.layers.shared_embedding_columns(sparse_id_columns=v, dimension=d, shared_embedding_name=k,
                                                             combiner="mean" if "combiner" not in feaconf else feaconf[
                                                                 "combiner"])
            for i in range(len(fcs)):
                fname = catecolnames_withsametable[k][i]
                print(fname)
                fctensor_map[fname] = tf.contrib.layers.input_from_feature_columns(laz_fe, [fcs[i]])
    for k, v in rawcolumns.items():
        fctensor_map[k] = tf.contrib.layers.input_from_feature_columns(laz_fe, [v])

    result_tensor = []
    for i in range(len(sampledata)):
        cur_name = feature_name[i]
        result_tensor.append(fctensor_map[cur_name])
    return result_tensor
