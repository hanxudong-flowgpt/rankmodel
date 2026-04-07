# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import json

FLAGS = tf.flags.FLAGS

class Context(object):
  def __init__(self):
    if "TF_CONFIG" in os.environ:
      tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
      cluster_spec = tf_config.get("cluster", {}) 
      task = tf_config.get("task", {})
      self.job_name = task["type"]
      self.task_index_ = task["index"]
    elif 'CLUSTER_SPEC' in os.environ:
      cluster_spec     = json.loads(os.environ["CLUSTER_SPEC"])
      self.job_name_   = os.environ["JOB_NAME"]
      self.task_index_ = int(os.environ["TASK_INDEX"])
    else:
      cluster_spec = {
          'ps': FLAGS.ps_hosts.split(','),
          'worker': FLAGS.worker_hosts.split(',')
      }
      self.task_index_ = FLAGS.task_index
      self.job_name_   = FLAGS.job_name
    logging.info("ps = {} ; worker= {}".format(cluster_spec["ps"], cluster_spec["worker"]))
    self.cluster_    = tf.train.ClusterSpec(cluster_spec)
    self.algoparam_  = FLAGS.algoparamstr
    self.ps_num_     = len(cluster_spec["ps"])
    self.worker_num_ = len(cluster_spec["worker"])
    self.task_count_ = None
    self.protocol_   = FLAGS.protocol
    self.train_config= json.load(file(FLAGS.train_config))
    self.debug_dir = self.train_config["debug_dir"]
    self.start_day = FLAGS.start_day
    

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
    return self.train_config["checkpoint_dir"]
  def get_inter_op_num(self):
    return self.train_config["inter_op_parallelism_threads"]
  def get_intra_op_num(self):
    return self.train_config["intra_op_parallelism_threads"]
  def get_task_count(self):
    if not self.task_count_:
      if self.job_name_ == 'ps':
        self.task_count_ = self.ps_num_
      else:
        self.task_count_ = self.worker_num_
    return self.task_count_

  def protocol(self):
    return self.protocol_
  def get_finish_file_name(self):
    self.finish_dir_ = self.train_config["debug_dir"] + self.start_day + "finishedworkers/"
    if FLAGS.task_index == 0 and not tf.gfile.Exists(self.finish_dir):
      tf.gfile.MkDir(self.finish_dir)
    self.finish_file_ = self.finish_dir_ + str(self.task_index_)
    if tf.gfile.Exists(self.finish_file_):
      tf.gfile.Remove(self.finish_file_)
    return self.finish_file_

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
    conf.inter_op_parallelism_threads = self.ctx_.get_inter_op_num()
    conf.intra_op_parallelism_threads = self.ctx_.get_intra_op_num()
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


class OnPdi(object):
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


class PdiBootStrap(object):
  def __init__(self):
    pass

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
    print("start training at pdi platform.")
    box = PsWorkerBox()
    self._context = box.ctx_
    self.finish_file = self._context.get_finish_file_name()
    box.set_worker_fn(self._pdi_worker_fn)
    box.run()
    with tf.gfile.GFile(self.finish_file, 'w') as f:
      f.write("finish")

  def finishedWorkers(self):
    files = tf.gfile.ListDirectory(self.finish_dir)
    return [int(x.split('/')[-1]) for x in files]

  def _pdi_worker_fn(self, server_target, task_index, task_count):
    return OnPdi(self.buildGraph, self.createSession, self.runSession, self.endSession,
                 server_target, task_index, task_count, self._context, None)