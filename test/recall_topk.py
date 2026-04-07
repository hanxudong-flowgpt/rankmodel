# coding=utf-8
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os
import random
import time
import math
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_util
# from tensorflow.python.platform import tf_logging as logging
# from util.tflog import tflogger as logging
from tensorflow import logging
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS


# 指定当前程序是参数服务器还是计算服务器。
flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
# 指定集群中的参数服务器地址。
flags.DEFINE_string(
    'ps_hosts', 'localhost:2222',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ', )
# 指定集群中的计算服务器地址。
flags.DEFINE_string(
    'worker_hosts', 'localhost:2223',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
# 指定当前程序的任务ID。
flags.DEFINE_integer('task_index', 0, 'Task ID of the worker/replica running the training.')


# user define

flags.DEFINE_string("tableoutput", "", "output table")
flags.DEFINE_string("tableinput", "NO TABLE", "table input")
flags.DEFINE_string("pool_vec_table", "NO TABLE", "pool item vec table")

flags.DEFINE_string("pool_table_selected_columns", "NO TABLE", "")
flags.DEFINE_string("input_table_selected_columns", "NO TABLE", "")

flags.DEFINE_integer("pool_item_number", 1000000, "pool_item_number")
flags.DEFINE_integer("item_vec_dim", 128, "embedding size")
flags.DEFINE_integer("batch_size", 1024, "batch_size")
flags.DEFINE_integer("topk", 50, "topk")


ps_hosts = FLAGS.ps_hosts.split(',')
worker_hosts = FLAGS.worker_hosts.split(',')
number_ps = len(ps_hosts)
number_worker = len(worker_hosts)

input_params = {
    'pool_table_selected_columns': 'item_id,vec',
    'input_table_selected_columns': 'item_id,vec'
}


class TopkRecaller(object):
    def __init__(self, partitioner=None, debug=False):
        # Create model
        # Parameters
        self.item_emb_num = FLAGS.pool_item_number
        self.item_emb_dim = FLAGS.item_vec_dim
        # self.n_class = params["n_class"]
        self.debug = debug
        self.partitioner = partitioner # if tf.fixed_size_partitioner(number_ps)
        # self.neg_sample = params['neg_sample']
        # self.bn_train = params.get('bn_train', True)
        self.batch_size = FLAGS.batch_size
        self.top_k = FLAGS.topk
        self.task_index = FLAGS.task_index
        self.output_table = FLAGS.tableoutput
        self.pool_vec_table = [FLAGS.pool_vec_table]
        self.pool_table_selected_columns = FLAGS.pool_table_selected_columns
        self.input_table = [FLAGS.tableinput]
        self.input_table_selected_columns = FLAGS.input_table_selected_columns

        self.work_device = "/job:worker/task:%d" % self.task_index
        print('self')
        print(vars(self))
        self.build_model()

    def wait_for_ready(self, session, check_wait_secs=20):
        self._check_wait_secs = check_wait_secs
        sleep_eval = 1
        while True:
            ready = session.run(self._ready_op)
            if ready:
                break
            else:
                session.run(self._set_cond_op)
                print(
                    'sleep %ds, waiting for feature tables ready' % sleep_eval)
                ready_num = session.run(self._ready_num)
                print('%d worker has finished' % ready_num)

                all_worker_status = session.run(self._worker_status)
                worker_not_ready = []
                for i in range(len(all_worker_status)):
                    if 0 == all_worker_status[i]:
                        worker_not_ready.append(i)
                print("worker_not_ready : %s" % str(worker_not_ready))

                time.sleep(sleep_eval)
                if sleep_eval < self._check_wait_secs:
                    sleep_eval = sleep_eval * 2
                if sleep_eval > self._check_wait_secs:
                    sleep_eval = self._check_wait_secs

    def init_embedding(self, worker_num, worker_id):
        self._worker_status = []
        for i in range(worker_num):
            self._worker_status.append(variable_scope.get_variable(initializer=constant_op.constant(0),
                                                       trainable=False, name="worker%d" % i))
        self._cond = self._worker_status[worker_id]
        self._set_cond_op = state_ops.assign(self._cond, 1, use_locking=True)
        self._reset_cond_op = state_ops.assign(self._cond, 0, use_locking=True)
        self._ready_num = reduce(lambda x, y: x + y, self._worker_status)
        self._ready_op = math_ops.equal(self._ready_num, worker_num)
        print('init embedding')
        with tf.device(self.work_device):
            filename_queue = tf.train.string_input_producer(self.pool_vec_table, num_epochs=1)
            reader = tf.TableRecordReader(csv_delimiter=':',
                                          selected_cols=self.pool_table_selected_columns,
                                          slice_count=worker_num,
                                          slice_id=worker_id,
                                          num_threads=8,
                                          capacity=4 * self.batch_size)
        # to_read = tf.shape(list(self.item)[0])[0]
        # print("to read:{}", to_read)
        x = int(math.ceil(len(list(self.item_vec)) / worker_num))

        if worker_id == worker_num-1:
            start = (worker_id-1) * x + x
            item_vecs = list(self.item_vec)[start:]
            item_ids = list(self.item_id)[start:]
        else:
            start = worker_id * x
            end = worker_id * x + x
            item_vecs = list(self.item_vec)[start:end]
            item_ids = list(self.item_id)[start:end]
        print("worker id:%d, init embeddings start:%d, end:%d" % (worker_id, start, start+x))
        assign_array = []
        #for i in range(len(item_vecs)):
        #    item_id_tmp = item_ids[i]
        #    item_vec_tmp = item_vecs[i]
        for i in range(len(list(self.item_vec))):
            item_id_tmp = list(self.item_id)[i]
            item_vec_tmp = list(self.item_vec)[i]
            print(str(i) + 'th part:')
            print(item_id_tmp.shape)
            print(item_vec_tmp.shape)

            row_size_id = item_id_tmp.shape[0]
            row_size_vec = item_vec_tmp.shape[0]
            assert row_size_id == row_size_vec, "vec size not equal to vec size:{}, {}".format(row_size_id, row_size_vec)
            key, record_op = reader.read_up_to(filename_queue, num_records=row_size_id)

            default = [tf.constant(['0'], dtype=tf.string) for i in range(0, 2)]
            col_list = tf.decode_csv(record_op, record_defaults=default, field_delim=":")
            item_id = tf.string_to_number(col_list[0], out_type=tf.int64)

            default = [tf.constant([0.0], dtype=tf.float32) for i in range(0, self.item_emb_dim)]
            vec_raw = tf.decode_csv(col_list[1], default, field_delim=',')
            vec = tf.stack(vec_raw, axis=1)
            print('item_id:', item_id, 'vec:', vec)

            assign_init = tf.assign(item_id_tmp, item_id)
            assign_array.append(assign_init)
            assign_init = tf.assign(item_vec_tmp, vec)
            assign_array.append(assign_init)

        return tf.group(*assign_array)

    def build_input(self):
        with tf.device(self.work_device):
            filename_queue = tf.train.string_input_producer(self.input_table, num_epochs=1)
            reader = tf.TableRecordReader(csv_delimiter=':',
                                          selected_cols=self.input_table_selected_columns,
                                          slice_count=1,
                                          slice_id=0,
                                          num_threads=8,
                                          capacity=4 * self.batch_size)
            key, record_op = reader.read_up_to(filename_queue, num_records=self.batch_size)
            default = [tf.constant(['0'], dtype=tf.string) for i in range(0, 2)]
            input_id, input_vec = tf.decode_csv(record_op, record_defaults=default, field_delim=":")

            default = [tf.constant([0.0], dtype=tf.float32) for i in range(0, self.item_emb_dim)]
            vec_raw = tf.decode_csv(input_vec, default, field_delim=',')
            input_vec = tf.stack(vec_raw, axis=1)
        return input_id, input_vec

    def build_model(self):
        input_id, input_vec = self.build_input()
        #with tf.device('/gpu:0'):
        self.item_vec =  tf.get_variable('item_vec', [self.item_emb_num, self.item_emb_dim],
                                         initializer=tf.zeros_initializer(),
                                         partitioner=self.partitioner,
                                         trainable=False)
        self.item_id = tf.get_variable('item_id', [self.item_emb_num],
                                       dtype=tf.int64,
                                       partitioner=self.partitioner,
                                       initializer=tf.zeros_initializer(),
                                       trainable=False)

        # no use
        # tmp_var_to_opt =  tf.get_variable('tmp', [1000], dtype=tf.float32, partitioner=self.partitioner,
        #                           initializer=tf.zeros_initializer())

        out_layer = [None] * len(list(self.item_vec))
        for p in range(len(list(self.item_vec))):
            with ops.colocate_with(list(self.item_vec)[p]):
                out_layer[p] = tf.matmul(input_vec, list(self.item_vec)[p], transpose_b=True)
                print('input_vec:', input_vec)
                # print('item_vec:', list(self.item_vec)[p])
                # out_layer[p] = tf.reshape(out_layer[p], [1024, list(self.item_vec)[p].shape[0]])

        factor = tf.convert_to_tensor(
            [[x * self.top_k * len(list(self.item_vec))] for x in range(self.batch_size)], dtype=tf.int32)
        print('factor:', factor)

        print('output_layer:', out_layer)
        print('factor:', factor)
        value, index = self.get_top_k(out_layer, factor, self.top_k)
        pred_items = tf.nn.embedding_lookup(self.item_id, tf.cast(index, dtype=tf.int64), partition_strategy='div')

        table_writer = tf.TableRecordWriter(self.output_table, slice_id=self.task_index)
        write_to_table = table_writer.write([0, 1],
                                            [input_id, tf.reduce_join(tf.as_string(pred_items), axis=1, separator=",")])

        close_table = table_writer.close()

        self.assign_init = self.init_embedding(worker_num=number_worker, worker_id=self.task_index)
        # self.input_id = input_id
        # self.pred_items = pred_items
        self.write_to_table = write_to_table
        self.close_table = close_table

    def get_top_k(slef, value_list, factor, k=50):
        top_k_of_each = [None] * len(list(value_list))
        for p in range(len(list(value_list))):
            with ops.colocate_with(value_list[p]):
                # (value, index)
                top_k_of_each[p] = tf.nn.top_k(value_list[p], k, sorted=True)
        # value : batchsize * k
        # index : batchsize * k
        # values: batchsize * k * 10
        values = tf.concat([x[0] for x in top_k_of_each], axis=1)

        print('value_list:')
        print(value_list[0].get_shape())
        print(value_list)

        if len(value_list) == 1:
            len_of_each_partition = [0]
        else:
            len_of_each_partition = [int(x.get_shape()[1]) for x in list(value_list)]
        # len_of_each_partition = [len(x[0]) for x in list(value_list)]
        len_of_index_base = [[0]] * len(len_of_each_partition)
        for i in range(1, len(len_of_each_partition)):
            len_of_index_base[i] = [len_of_index_base[i - 1][0] + len_of_each_partition[i - 1]]
        len_of_index_base = [[x] for x in len_of_index_base]
        print('len_of_index_base')
        print(len_of_index_base)
        print(len_of_each_partition)
        print(value_list)

        indices = [x[1] for x in top_k_of_each]
        print('indices before add')
        # print(sess.run([indices]))
        indices = [indices[i] + len_of_index_base[i] for i in range(len(len_of_index_base))]

        indices = tf.concat(indices, axis=1)
        # print(sess.run([indices]))


        print('values')
        print(values)
        # print(sess.run([values]))
        print('indices')
        print(indices)
        # print(sess.run([indices]))
        #values = tf.reshape(values, [1024, k * 10 ])
        new_value, new_index = tf.nn.top_k(values, k, sorted=True)
        # here we need to use the new_index(from 0 to k * 10) to get the true old index in indices
        # I do not find any function, so I need to flatten the index and then use the scatter
        flatten_index = tf.reshape(indices, [-1])
        print('new_index:')
        print(new_index)
        print(factor)
        flatten_new_index = tf.reshape(tf.add(new_index, factor), [-1])
        print("flatten_index:")
        print(flatten_index)
        # print(sess.run(flatten_index))
        print('flatten_new_index')
        print(flatten_new_index)
        # print(sess.run([flatten_new_index]))
        final_index = tf.gather(flatten_index, flatten_new_index)
        final_index = tf.reshape(final_index, [-1, k])
        print('final_index')
        # print(sess.run([final_index]))
        return new_value, final_index






def main(argv=None):
    # 解析flags并通过tf.train.ClusterSpec配置TensorFlow集群。
    # 通过tf.train.ClusterSpec以及当前任务创建tf.train.Server。
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    print(FLAGS.task_index)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。server.join()会
    # 一致停在这条语句上。
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == "worker":
        print("start...")
        recaller = TopkRecaller(partitioner=tf.fixed_size_partitioner(number_ps))
        # training_util.get_global_step()
        # gs = 'global_step:0'
        g = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               hooks=[]) as mon_sess:
            #while not mon_sess.should_stop():
            if FLAGS.task_index == 1:
                try:
                    t = time.time()
                    mon_sess.run(recaller.assign_init)
                    print("load all vectors: %s sec" % str(time.time() - t))
                except tf.errors.OutOfRangeError:
                    print('load base item vector done...')
                    print('waiting other workers...')
                    #recaller.wait_for_ready(mon_sess)
                    #print("all worker load: %s sec" % str(time.time() - t))
                    #print('all workers load base item vector done...')
                    #break
            recaller.wait_for_ready(mon_sess)
            '''
            while not mon_sess.should_stop():
                if FLAGS.task_index > 0:
                    break
                try:
                    for _ in range(5):
                        t = time.time()
                        mon_sess.run(recaller.write_to_table)
                        print("recall one batch: %s sec" % str(time.time() - t))
                    break
                except tf.errors.OutOfRangeError:
                    break
            '''
            while not mon_sess.should_stop():
                try:
                    t = time.time()
                    mon_sess.run(recaller.write_to_table)
                    print("recall one batch: %s sec" % str(time.time() - t))
                except tf.errors.OutOfRangeError:
                    break
            mon_sess.run(recaller.close_table)
            print('get topk item done...')


if __name__ == "__main__":
    tf.app.run(main=main)



