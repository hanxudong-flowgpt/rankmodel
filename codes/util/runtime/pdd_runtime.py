# -*- coding:utf-8 -*-
import os,json
import tensorflow as tf
from util.tflog import tflogger as logging

FLAGS = tf.flags.FLAGS


class Context(object):
  def __init__(self):
    if "TF_CONFIG" in os.environ and len(os.environ["TF_CONFIG"]) > 20:
      tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
      task = tf_config.get("task", {})
      FLAGS.job_name = task["type"]
      FLAGS.task_index = task["index"]

      cluster_spec = tf_config.get("cluster", {})
      FLAGS.ps_hosts = ','.join(cluster_spec['ps'])
      FLAGS.worker_hosts = ','.join(cluster_spec['worker'])
    elif 'CLUSTER_SPEC' in os.environ and len(os.environ["CLUSTER_SPEC"]) > 20:
      cluster_spec_str = os.environ["CLUSTER_SPEC"]
      cluster_spec = json.loads(cluster_spec_str)
      FLAGS.ps_hosts = ','.join(cluster_spec['ps'])
      FLAGS.worker_hosts = ','.join(cluster_spec['worker'])

      FLAGS.job_name = os.environ["JOB_NAME"]
      FLAGS.task_index = int(os.environ["TASK_INDEX"])
    else:
      ValueError("cannot get cluster config from environ...")
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


class PaiBootStrap(object):
  def __init__(self):
    # self._platform = init_confs["platform"]
    # self._init_confs = init_confs
    logging.info("init PaiBootStrap...")
    # self.box = PsWorkerBox()

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
      task_index=self.task_index_
      # config=conf,
      # protocol=self.ctx_.protocol()
    )

    if FLAGS.task_index == 0 and not tf.gfile.Exists(FLAGS.debug_dir):
      tf.gfile.MkDir(FLAGS.debug_dir)
    self.finish_dir = FLAGS.debug_dir + "finishedworkers/"
    if FLAGS.task_index == 0 and not tf.gfile.Exists(self.finish_dir):
      tf.gfile.MkDir(self.finish_dir)

    # self.finish_dir = FLAGS.debug_dir
    self.finish_file = self.finish_dir + "worker_" + str(FLAGS.task_index)
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

  def report_tf_metric(self, metric_name, value, ex_tag=None):
    pass

  def start(self):
    logging.info("start training at pdd platform.")
    logging.info("finish file: %s" % self.finish_file)
    self._context = self.ctx_

    if self.job_name_ == 'ps':
      self.server_.join()
    elif self.job_name_ == 'worker':
      with tf.device(self._device_setter()):
        self.buildGraph(task_id=self.task_index_, context=self._context)
        logging.info("finish model build")
        logging.info("start run")
        self.sess_ = self.createSession(self.server_.target, self.task_index_)
      self.runSession(self.sess_, self.task_index_, thread_ind=0)
      logging.info("finish run")
      # self.endSession(self.task_index_)
    else:
      raise ValueError('Only ps-worker job is supported.')
    logging.info("ending...")
    f = tf.gfile.GFile(self.finish_file, 'w')
    f.write("thisworkerfinished\n")
    f.close()
    self.sess_.close()
    logging.info("end...")

  def _device_setter(self):
    return tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % self.task_index_, cluster=self.cluster_)


  def finishedWorkers(self):
    #return []
    try:
      files = tf.gfile.ListDirectory(self.finish_dir)
    except:
      files = []
    xxx = []
    for x in files:
      try:
        xx = x.split('/')[-1]
        a = int(xx.split('_')[-1])
        xxx.append(a)
      except:
        continue
    return xxx
