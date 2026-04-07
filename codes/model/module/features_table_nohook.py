# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Python wrappers for feature join."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf

from tensorflow.contrib import lookup
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
import numpy as np


class FeatureTable(object):
  """A generic feature join implementation."""

  def __init__(self, input_tables_conf, server_count=None,
          worker_index=None, worker_count=None, check_any_finish=None):
    """Creates `FeatureTable`.

    Args:
      input_tables_conf: A map containing one or more input table confs.
      server_count: (Optional.)Number of ps for hash tables.
      worker_index: (Optional.)Index of current slice under distributed runtime.
      worker_count: (Optional.)Total amount of slices under distributed runtime.

    Raises:
      InvalidArgumentError: If server_count, worker_index and worker_count are
        neither all non-None nor all None.
    """
    # When server_count, worker_index and worker_count are all None,
    # it means standalone mode.
    self._check_any_finish = check_any_finish
    self._server_count = server_count
    self._worker_index = worker_index
    self._worker_count = worker_count
    self._batch_size = os.environ.get(
        'LOAD_FEATURE_TABLE_BATCH_SIZE', 10000)

    self._hash_table_map = {}
    self._data = []
    # Init hash tables, partitioned among servers.
    for feature_type in input_tables_conf.keys():
      hash_tables = []
      if not self._server_count:
        hash_table = lookup.MutableHashTable(
            key_dtype=dtypes.int64, value_dtype=dtypes.string,
            default_value="",
            name='%s_feature_table' % feature_type)
        hash_tables.append(hash_table)
      else:
        for server_index in xrange(self._server_count):
          with ops.device("/job:ps/task:%d" % server_index):
            hash_table = lookup.MutableHashTable(
                key_dtype=dtypes.int64, value_dtype=dtypes.string,
                default_value="",
                name='%s_feature_table_%d' % (feature_type, server_index))
            hash_tables.append(hash_table)
      self._hash_table_map[feature_type] = hash_tables

    # check args and prepare for standalone mode.
    '''
    if (self._server_count==None and
        self._worker_count==None and
        self._worker_index==None):
      self._server_count = 1
      self._worker_count = 1
      self._worker_index = 0
    elif (self._server_count!=None and
        self._worker_count!=None and
        self._worker_index!=None):
      pass
    else:
      raise errors.InvalidArgumentError(
          None, None, "server_count, worker_count or worker_index is None.")
    '''
    # A map, each load_op is corresponding with a feature_type.
    self._load_ops = self._load_all(input_tables_conf)

    # If tables have been loaded, they will not be loaded again.
    # It is used in the case of failover and restore.

    self._worker_status = []
    for i in range(self._worker_count):
        self._worker_status.append(variable_scope.get_variable(initializer=constant_op.constant(0),
                                                               trainable=False, name="worker%d" % i,
                                                               collections=[ops.GraphKeys.LOCAL_VARIABLES]))
    self._cond = self._worker_status[self._worker_index]
    self._set_cond_op = state_ops.assign(self._cond, 1, use_locking=True)
    self._reset_cond_op = state_ops.assign(self._cond, 0, use_locking=True)
    self._ready_num = reduce(lambda x, y: x+y, self._worker_status)
    self._ever_loaded = math_ops.equal(self._ready_num, self._worker_count)

    # Only when all tables are loaded, it means ready for training.
    # When failover or restore , _ever_loaded is true at the beginning,
    # but hash tables are not restored completely.
    # So, only when both ever_loaded is true and all tables size are not zero,
    # it means really ready for training.
    # This is why we need two-stage-judge here.
    hash_table_ready_list = []
    for tables in self._hash_table_map.values():
      for table in tables:
        hash_table_ready_list.append(math_ops.not_equal(table.size(), 0))
    hash_tables_ready = math_ops.reduce_all(
        array_ops.stack(hash_table_ready_list))
    self._ready = math_ops.logical_and(self._ever_loaded, hash_tables_ready)

    self.table_stats = {}
    for k, v in self._hash_table_map.items():
        self.table_stats[k] = []
        for table in v:
            self.table_stats[k].append(table.size())
    '''
    self.hook = FeatureTableReadyHook(
        self._load_ops, self._set_cond_op, self._reset_cond_op,
        self._worker_status, self._ready, check_wait_secs, self.table_stats, self._ready_num, self._worker_index,
        self._data)
    '''

  def _load_all(self, input_table_confs):
    load_ops = {}
    for feature_type, conf in input_table_confs.iteritems():
      load_ops[feature_type] = self._load_feature_table(feature_type, conf)
    return load_ops

  def _load_feature_table(self, feature_type, conf):
    logging.info(conf)
    logging.info("feature table info: %d,%d,%d" % (self._server_count, self._worker_index, self._worker_count))
    if len(conf) == 3:
        dataset = readers.TableRecordDataset(
            conf[0],
            selected_cols=conf[1],
            record_defaults=conf[2],
            slice_id=self._worker_index,
            slice_count=self._worker_count)
        iterator = dataset.batch(self._batch_size).make_one_shot_iterator()
        id, feature = iterator.get_next()
        '''
        file_queue = tf.train.string_input_producer([conf[0]])
        reader = tf.TableRecordReader(csv_delimiter='\t', slice_count=self._worker_count,
                                      slice_id=self._worker_index, selected_cols=conf[1])
        keys, values = reader.read_up_to(file_queue, num_records=self._batch_size)
        batch_values = tf.train.batch(
            [values],
            batch_size=self._batch_size,
            capacity=self._batch_size * 4,
            enqueue_many=True,
            num_threads=2, allow_smaller_final_batch=True)
        id, feature = tf.decode_csv(
            batch_values,
            record_defaults=conf[2],
            field_delim='\t')
        '''
    else:
        dataset = tf.data.TextLineDataset(
            conf[0].split(",") if type(conf[0]) == str else conf[0])
        iterator = dataset.batch(self._batch_size).make_one_shot_iterator()
        value = iterator.get_next()
        id, feature = tf.decode_csv(value, record_defaults=[[tf.constant(0, dtype=tf.int64)],
                                                            [tf.constant("", dtype=tf.string)]], field_delim='\t')
    self._data.append(id)
    mod_index = math_ops.floormod(
        id, constant_op.constant(self._server_count, dtype=dtypes.int64),
        name='floormod_%s' % feature_type)
    mod_index = math_ops.cast(mod_index, dtypes.int32)

    pt_ids = data_flow_ops.dynamic_partition(
        id, mod_index, self._server_count,
        name='partition_%s' % feature_type)
    pt_features = data_flow_ops.dynamic_partition(
        feature, mod_index, self._server_count,
        name='partition_%s' % feature_type)

    insert_ops = []
    hash_tables = self._hash_table_map[feature_type]
    for i in xrange(self._server_count):
      insert_ops.append(hash_tables[i].insert(pt_ids[i], pt_features[i]))

    return control_flow_ops.group(*insert_ops)

  def join_feature(self, feature_type, ids, feature_num=0, field_delim=':'):
    hash_tables = self._hash_table_map[feature_type]
    features = self._lookup(hash_tables, ids)

    return features

  def _lookup(self, hash_tables, ids):
    # partition ids among servers
    mod_index = math_ops.floormod(
        ids, constant_op.constant(self._server_count, dtype=dtypes.int64),
        name='lookup_floormod')
    mod_index = math_ops.cast(mod_index, dtypes.int32)
    partitioned_ids = data_flow_ops.dynamic_partition(
        ids, mod_index, self._server_count,
        name='lookup_partition_ids')

    # lookup
    partitioned_features = []
    for i in xrange(self._server_count):
      result = hash_tables[i].lookup(partitioned_ids[i])
      partitioned_features.append(result)

    # stitch back together
    origin_indices = math_ops.range(array_ops.size(ids), dtype=dtypes.int32)
    partitioned_indices = data_flow_ops.dynamic_partition(
        origin_indices, mod_index, self._server_count,
        name='lookup_partition_index')
    result = data_flow_ops.dynamic_stitch(
        partitioned_indices, partitioned_features,
        name='lookup_stitch')
    return result

  def get_set_op(self):
      return self._set_cond_op

  def run_load_and_wait4ready(self, session, check_wait_secs):
    # ever_loaded = session.run(self._ever_loaded_op)
    #if not ever_loaded:
    for key, value in self._load_ops.iteritems():
      self._load(session=session, feature_type=key, op=value)

    self._wait_for_ready(session=session, check_wait_secs=check_wait_secs)
    time.sleep(check_wait_secs * 2)
    # session.run(self._after_ready_op)

  def _load(self, session, feature_type, op):
    logging.info("start loading %s feature table" % feature_type)
    t = time.time()
    # if self._worker_id != 1:
    #     return
    while True:
      try:
        # data, _ = session.run([self._data, op])
        session.run(op)
        logging.info("load one batch: %s sec" % str(time.time() - t))
        # np.set_printoptions(threshold='nan')
        # logging.info("table data: %s" % str(data))
      except errors.OutOfRangeError:
        break
    logging.info(
        "load %s feature table done, cost %.2fs"
        % (feature_type, time.time() - t))

  def _wait_for_ready(self, session, check_wait_secs):
    sleep_eval = 1
    while True:
      if self._check_any_finish is not None and self._check_any_finish():
          break
      ready = session.run(self._ready)
      if ready:
        break
      else:
        session.run(self._set_cond_op)
        logging.info(
            'sleep %ds, waiting for feature tables ready' % sleep_eval)
        ready_num = session.run(self._ready_num)
        logging.info('%d worker has finished' % ready_num)

        all_worker_status = session.run(self._worker_status)
        worker_not_ready = []
        for i in range(len(all_worker_status)):
            if 0 == all_worker_status[i]:
                worker_not_ready.append(i)
        logging.info("worker_not_ready : %s" % str(worker_not_ready))

        table_stat = session.run(self.table_stats)
        logging.info(
            'table stat: %s' % str(table_stat))
        time.sleep(sleep_eval)
        if sleep_eval < check_wait_secs:
          sleep_eval = sleep_eval * 2
        if sleep_eval > check_wait_secs:
          sleep_eval = check_wait_secs

