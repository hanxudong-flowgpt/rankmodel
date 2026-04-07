# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from util.tflog import tflogger as logging

FLAGS = tf.flags.FLAGS

class Context(object):
  def __init__(self):
    print('---------')
    print(FLAGS.ps_hosts)
    print('---------')
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    self.cluster_ = tf.train.ClusterSpec({
        'ps': ps_hosts,
        'worker': worker_hosts
    })

    self.algoparam_ = FLAGS.algoparamstr
    self.ps_num_ = len(ps_hosts)
    self.worker_num_ = len(worker_hosts)
    self.job_name_ = FLAGS.job_name
    self.task_index_ = FLAGS.task_index
    self.task_count_ = None
    self.protocol_ = FLAGS.protocol
    if self.is_grpc_plus():
      self._enable_grpc_plus()

  def get_param(self):
    return self.algoparam_

  def get_ps_num(self):
    return self.ps_num_

  def get_worker_num(self):
    return self.worker_num_

  def get_cluster(self):
    return self.cluster_

  def get_job_name(self):
    return self.job_name_

  def get_task_index(self):
    return self.task_index_

  def get_ckp_dir(self):
    return "hdfs://et2prod2/pora/et2prod2/save/miss/"
  def get_task_count(self):
    if not self.task_count_:
      if self.job_name_ == 'ps':
        self.task_count_ = self.ps_num_
      else:
        self.task_count_ = self.worker_num_
    return self.task_count_

  def protocol(self):
    return self.protocol_

  def is_grpc_plus(self):
    return self.protocol_ == "grpc++"

  def _enable_grpc_plus(self):
    os.environ['JOB_NAME'] = FLAGS.job_name
    os.environ['TASK_INDEX'] = str(FLAGS.task_index)

class PsWorkerBox(object):
  def __init__(self):
    self.ctx_ = Context()
    self.cluster_ = self.ctx_.get_cluster()
    self.job_name_ = self.ctx_.get_job_name()
    self.task_index_ = self.ctx_.get_task_index()
    self.task_count_ = self.ctx_.get_task_count()

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    if self.ctx_.is_grpc_plus():
        conf.inter_op_parallelism_threads = 64
        conf.intra_op_parallelism_threads = 20
    else:
        conf.inter_op_parallelism_threads = 64
    self.server_ = tf.train.Server(
        self.cluster_,
        job_name=self.job_name_,
        task_index=self.task_index_,
        config=conf,
        protocol=self.ctx_.protocol()
    )

    self.ps_fn_ = None
    self.worker_fn_ = None
    self.ps_ops_ = None
    self.ps_strategy_ = None

  def task_index(self):
    return self.task_index_

  def task_count(self):
    return self.task_count_

  def server_target(self):
    return self.server_.target

  def set_ps_fn(self, ps_fn):
    self.ps_fn_ = ps_fn

  # worker_fn is a function that will return an object of sub-WorkerClosure.
  def set_worker_fn(self, worker_fn):
    self.worker_fn_ = worker_fn

  def run(self):
    if self.job_name_ == 'ps':
      if self.ps_fn_:
          self.ps_fn_()
      self.server_.join()
    elif self.job_name_ == 'worker':
      if self.worker_fn_:
        with tf.device(self._device_setter()):
          worker_closure = self.worker_fn_(
              self.server_target(), self.task_index(), self.task_count())
          worker_closure.model_fn()
        worker_closure.run_fn()
      else:
        raise ValueError('worker_fn can not be None.')
    else:
      raise ValueError('Only ps-worker job is supported.')

  def _device_setter(self):
    return tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % self.task_index_,
        cluster=self.cluster_,
        ps_ops=self.ps_ops_,
        ps_strategy=self.ps_strategy_)


class OnPai(object):
  def __init__(self, build_graph_fn, create_sess_fn, run_sess_fn, end_sess_fn,
          server_target, task_index, task_count, context, init_confs):
    self._build_graph_fn = build_graph_fn
    self._create_sess_fn = create_sess_fn
    self._run_sess_fn = run_sess_fn
    self._end_sess_fn = end_sess_fn
    self._server_target = server_target
    self._task_id = task_index
    self._task_count = task_count
    self._context = context
    self._init_confs = init_confs

  def model_fn(self):
    print("model fn")
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d'%self._task_id, cluster=self._context.get_cluster())):
      self._build_graph_fn(task_id=self._task_id, context=self._context)
    print("finish model fn")

  def run_fn(self):
    print("start run")
    with self._create_sess_fn(self._server_target, self._task_id) as sess:
      self._run_sess_fn(sess, self._task_id, thread_ind=0)
    self._end_sess_fn(self._task_id)


class PaiBootStrap(object):
  def __init__(self):
    # self._platform = init_confs["platform"]
    # self._init_confs = init_confs
    logging.info("init PaiBootStrap...")

    self.finish_dir = FLAGS.debug_dir + "finishedworkers/"
    if FLAGS.task_index == 0 and not tf.gfile.Exists(self.finish_dir):
      tf.gfile.MkDir(self.finish_dir)
    self.finish_file = FLAGS.debug_dir + "finishedworkers/" + str(FLAGS.task_index)
    if tf.gfile.Exists(self.finish_file):
      tf.gfile.Remove(self.finish_file)

  def buildGraph(self, task_id, context):
    pass

  def createSession(self, target, task_id):
    pass

  def runSession(self, sess, task_id, thread_ind):
    pass

  def endSession(self, task_id):
    pass
  '''
  def finished_workers(self, **kwargs):
    if self.is_on_blink():
      return self._finished_workers_fn()
    else:
      return 0
  '''
  def get_cluster(self):
    return self._context.get_cluster()
  def report_tf_metric(self, metric_name, value, ex_tag=None):
    pass
  def start(self):
    print("start training at pai platform.")
    box = PsWorkerBox()
    self._context = box.ctx_
    box.set_worker_fn(self._pai_worker_fn)
    box.run()
    f = tf.gfile.GFile(self.finish_file, 'w')
    f.write("finish")
    f.close()

  def finishedWorkers(self):
    try:
      files = tf.gfile.ListDirectory(self.finish_dir)
      return [int((x.split('/')[-1]).split('_')[-1]) for x in files]
    except Exception, e:
      logging.info(e)
      return []

  def _pai_worker_fn(self, server_target, task_index, task_count):
    return OnPai(self.buildGraph, self.createSession, self.runSession, self.endSession,
                 server_target, task_index, task_count, self._context, None)