import tensorflow as tf
import time, md5, os
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from flink_tensorflow.log import logger
from flink_tensorflow.kmonitor.KmonClient import get_kmon
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.training import training_util
from tensorflow.core.framework.reader_base_pb2 import TextLineReaderState
import hashlib
import random
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops


class SecondTimer:
    def __init__(self, every_secs=60):
        self._every_secs = every_secs
        self._last_triggered_time = 0

    def should_trigger(self):
        current_time = time.time()
        if current_time >= self._last_triggered_time + self._every_secs:
            self._last_triggered_time = current_time
            return True
        return False


class FIFOQueueShared(session_run_hook.SessionRunHook):

    def __init__(self,
                 filelist,
                 worker_id,
                 worker_num,
                 reader_num,
                 qtype,
                 MAXLIST=60000,
                 MAXWORKER=2000,
                 num_epochs=1,
                 shuffle=False,
                 seed=0,
                 istest=False):

        if 'SQDEBUG' in os.environ:
            multi = int(os.environ['SQDEBUG'])
            if worker_num == 1:
                MAXLIST = worker_num * reader_num * multi * 4
            else:
                MAXLIST = worker_num * reader_num * multi
        logger.info('MAXLIST is [%s]', MAXLIST)
        self.qtype = qtype
        self.qscope = 'DynamicFilelist/' + qtype
        if len(filelist) * num_epochs > MAXLIST:
            logger.info('%s filelist * epochs exceed %s, files will be drop !!!!!!!', qtype, MAXLIST)
        if worker_num > MAXWORKER:
            raise Exception('%s worker_num exceed %s need adjust' % (qtype, MAXWORKER))
        if not qtype:
            raise Exception('%s qscope not empty' % qtype)

        self.filelist = filelist
        self.worker_id = worker_id
        self.worker_num = worker_num
        self.reader_num = reader_num
        self.reader_initops = []
        self._inited = False;

        self.epochs_filelist = filelist * num_epochs
        if shuffle:
            random.seed(seed)
            random.shuffle(self.epochs_filelist)

        self.epochs_filelist = self.epochs_filelist[:MAXLIST]
        left = MAXLIST - len(self.epochs_filelist)
        self.filelist_init = ['%s|%s' % (one[1], one[0]) for one in enumerate(self.epochs_filelist)] + [''] * left
        self.location_init = [-1] * len(self.epochs_filelist) + [-2] * left
        self.readerstate_init = [[''] * reader_num] * MAXWORKER
        mymd5 = hashlib.md5()
        mymd5.update(''.join(self.filelist))
        self.md5_init = mymd5.hexdigest()  # md5.new().hexdigest()
        logger.info('%s md5_init [%s]', self.qtype, self.md5_init)

        with tf.name_scope(self.qscope + '/init'):
            psdev = '/job:localhost/task:0'
            psdev = '/job:ps/task:0' if istest == False else psdev
            with tf.device(psdev):
                self.shared_queue = tf.FIFOQueue(capacity=MAXLIST, dtypes=[tf.string], shared_name='ps_' + self.qtype)
            with tf.variable_scope(self.qscope + '/PsVar', reuse=tf.AUTO_REUSE):
                self.filelist_var = variable_scope.get_variable(name='WholeFileListVar', trainable=False,
                                                                dtype=tf.string, initializer=self.filelist_init,
                                                                collections=[ops.GraphKeys.LOCAL_VARIABLES])
                self.location_var = variable_scope.get_variable(name='FileListAssginVar', trainable=False,
                                                                dtype=tf.int32, initializer=self.location_init,
                                                                collections=[ops.GraphKeys.LOCAL_VARIABLES])
                self.readerstate_var = variable_scope.get_variable(name='ReaderStateVar', trainable=False,
                                                                   dtype=tf.string, initializer=self.readerstate_init,
                                                                   collections=[ops.GraphKeys.LOCAL_VARIABLES])
                self.md5_var = variable_scope.get_variable(name='ReaderMd5Var', trainable=False, dtype=tf.string,
                                                           initializer=self.md5_init,
                                                           collections=[ops.GraphKeys.LOCAL_VARIABLES])

            self.shared_queue_initop = None
            if worker_id == 0:
                unused = tf.equal(self.location_var, [-1] * MAXLIST)
                unused_filelist = tf.boolean_mask(self.filelist_var, unused)
                enop = self.shared_queue.enqueue_many(unused_filelist)
                with tf.control_dependencies([enop]):
                    self.shared_queue_initop = self.shared_queue.close()

            local_capacity = MAXLIST
            local_name = "worker%d_localqueue" % worker_id
            self.local_queue = tf.FIFOQueue(capacity=local_capacity, dtypes=[tf.string], name=local_name,
                                            shared_name=self.qtype)

            used = tf.equal(self.location_var, [worker_id] * MAXLIST)
            used_filelist = tf.boolean_mask(self.filelist_var, used)
            filename, index = tf.decode_csv(used_filelist, [[''], [-1]], field_delim='|')
            self.local_queue_initop = self.local_queue.enqueue_many(filename)

            op0 = tf.assign(self.filelist_var, self.filelist_init)
            op1 = tf.assign(self.location_var, self.location_init)
            op2 = tf.assign(self.readerstate_var, self.readerstate_init)
            op3 = tf.assign(self.md5_var, self.md5_init)
            self.reset_ops = [op0, op1, op2, op3]
            self.queue_stat = [self.shared_queue.size(), self.shared_queue.is_closed(),
                               self.local_queue.size(), self.local_queue.is_closed()]

    def add_reader(self, reader):
        with tf.name_scope(self.qscope + '/addreader'):
            idx = len(self.reader_initops)
            state = self.readerstate_var[self.worker_id, idx]
            op = reader.restore_state(state)
            self.reader_initops.append(op)

    def add_summary(self, output_dir):
        self._output_dir = output_dir

    def update_state(self, reader, idx):
        with tf.name_scope(self.qscope + '/updatestate'):
            def add_one():
                one = self.shared_queue.dequeue()
                filename, index = tf.decode_csv(one, [[''], [-1]], field_delim='|')
                shaped_index = tf.reshape(index, [1, 1])
                assign_used = tf.scatter_nd_update(self.location_var, shaped_index, [self.worker_id])
                with tf.control_dependencies([assign_used]):
                    add_file = self.local_queue.enqueue(filename)
                    with tf.control_dependencies([add_file]):
                        return tf.no_op()

            def save_state():
                state = reader.serialize_state()
                assignop = self.readerstate_var[self.worker_id, idx]
                assign_state = tf.assign(assignop, state)
                with tf.control_dependencies([assign_state]):
                    return tf.no_op()

            op = tf.cond(
                tf.cast(reader.currentwork_finished(), tf.bool),
                lambda: tf.cond(
                    tf.logical_and(tf.equal(self.shared_queue.size(), tf.constant(0, dtype=tf.int32)),
                                   self.shared_queue.is_closed()),
                    lambda: self.local_queue.close(cancel_pending_enqueues=True),
                    add_one
                ),
                lambda: tf.no_op()
            )

            op2 = tf.cond(
                # tf.logical_or(tf.cast(reader.currentwork_finished(), tf.bool), tf.cast(reader.newwork_started(), tf.bool)),
                tf.cast(reader.newwork_started(), tf.bool),
                save_state,
                lambda: tf.no_op()
            )
            with tf.control_dependencies([op, op2]):
                return tf.no_op()

    def printvar(self, sess):
        logger.info('%s filelist_var: %s', self.qtype, sess.run(self.filelist_var)[:len(self.epochs_filelist)])
        logger.info('%s location_var: %s', self.qtype, sess.run(self.location_var)[:len(self.epochs_filelist)])
        logger.info('%s readerstate_var: %s', self.qtype, sess.run(self.readerstate_var)[:self.worker_num])
        logger.info('%s md5_var: %s', self.qtype, sess.run(self.md5_var))
        logger.info('%s queue_stat shared_queue size/close local_queue size/close %s', self.qtype,
                    sess.run(self.queue_stat))

    def begin(self):
        if self.worker_id == 0:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
            self._global_step_tensor = training_util._get_or_create_global_step_read()
            if self._global_step_tensor is None:
                raise RuntimeError("Global step should be created to use StepCounterHook.")

    def after_create_session_before_queue(self, sess, coord):
        logger.info('%s enter after_create_session %s', self.qtype, self)
        if self._inited:
            logger.info('%s enter after_create_session refused', self.qtype)
            return
        self._inited = True
        self._sess = sess
        self._timer = SecondTimer()
        self.client = get_kmon()

        if self.worker_id == 0:
            if self.md5_init != sess.run(self.md5_var):
                logger.info("%s current md5:%s, ps md5:%s" % (self.qtype, self.md5_init, sess.run(self.md5_var)))
                logger.info('%s md5 not equal so clear queue ckeckpoint', self.qtype)
                sess.run(self.reset_ops)
                logger.info("%s current md5:%s, ps md5:%s" % (self.qtype, self.md5_init, sess.run(self.md5_var)))
            else:
                logger.info('%s md5 equal use ckpt_filelist', self.qtype)
        else:
            try_num = 0
            while True:
                if self.md5_init != sess.run(self.md5_var):
                    logger.info("%s current md5:%s, ps md5:%s" % (self.qtype, self.md5_init, sess.run(self.md5_var)))
                    logger.info('%s md5 not equal so sleep 10', self.qtype)
                    tag = {'qtype': self.qtype}
                    self.client.report('tf.md5.diff', self.worker_id, tag)
                    time.sleep(10)
                    try_num = try_num + 1
                    logger.info("try_num: %s", str(try_num))
                    if try_num >= 10:
                        logger.info("try_num >= 10, so break...")
                        break
                else:
                    logger.info('%s md5 equal use ckpt_filelist', self.qtype)
                    break

        if self.reader_initops:
            s = sess.run(self.reader_initops)
            logger.info('%s finish reader_initops', self.qtype)
        if self.local_queue_initop is not None:
            sess.run(self.local_queue_initop)
            logger.info('%s finish local_queue_initop', self.qtype)
        if self.shared_queue_initop is not None:
            sess.run(self.shared_queue_initop)
            logger.info('%s finish shared_queue_initop', self.qtype)
        logger.info('%s finish after_create_session', self.qtype)
        self.printvar(sess)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        if self.worker_id == 0:
            requests = {}
            if self._timer.should_trigger():
                requests['queuestat'] = self.location_var
                requests['readerstat'] = self.readerstate_var
                requests['globalsetp'] = self._global_step_tensor
            return SessionRunArgs(requests)

    def report(self, results):
        if 'queuestat' not in results:
            return
        stat = results['queuestat']
        reader_stat = results['readerstat'][:self.worker_num]
        global_step = results['globalsetp']

        if self.worker_id == 0:
            # for finish info
            count_finished = [0] * self.worker_num
            r_state = TextLineReaderState()
            for idx, reader in enumerate(reader_stat):
                for cont in reader:
                    r_state.ParseFromString(cont)
                    base_state = r_state.reader_base_state
                    finished = base_state.work_finished
                    if r_state.current_work_finished and base_state.work_finished < base_state.work_started:
                        finished += 1
                    count_finished[idx] += finished
                tag = {'qtype': self.qtype, 'idx': idx}
                finished = count_finished[idx]
                # ratio = finished * 1.0 / len(self.epochs_filelist)
                # self.client.report('tf.FileList.Finished', finished, tag)
            tag = {'qtype': self.qtype}
            finished = sum(count_finished)
            ratio = finished * 1.0 / len(self.epochs_filelist)
            self.client.report('tf.FileList.FinishedAll', ratio, tag)
            summary = Summary(value=[Summary.Value(tag='filelist/finish/' + self.qtype, simple_value=ratio)])
            self._summary_writer.add_summary(summary, global_step=global_step)
            logger.info('%s allworker finished[%s] finishRatio[%s]', self.qtype, finished, ratio)

            # for process info
            filter_stat = filter(lambda x: x != -2, stat)
            tag = {'qtype': self.qtype}
            processed = len(filter_stat) - filter_stat.count(-1)
            ratio = processed * 1.0 / len(filter_stat)
            self.client.report('tf.FileList.ProcessedAll', ratio, tag)
            summary = Summary(value=[Summary.Value(tag='filelist/process/' + self.qtype, simple_value=ratio)])
            self._summary_writer.add_summary(summary, global_step=global_step)
            logger.info('%s allworker processed[%s] processRatio[%s]', self.qtype, processed, ratio)

    def after_run(self, run_context, run_values):
        if self.worker_id == 0:
            self.report(run_values.results)

    def end(self, session):
        pass

    def flush(self):
        if self.worker_id == 0:
            requests = {}
            requests['queuestat'] = self.location_var
            requests['readerstat'] = self.readerstate_var
            requests['globalsetp'] = self._global_step_tensor
            self.report(self._sess.run(requests))
            self.client.flush()




