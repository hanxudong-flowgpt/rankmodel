# -*- coding:utf-8 -*-

import os
import sys
from boot_strap import BootStrap
import traceback
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import datetime

class PointwiseCxrBootStrap(BootStrap):
  '''
  need implement three function:
  build_graph --> building tf graph
  create_session --> create ...
  run_session --> run train or predict, print and save model,log to odps
  '''
  def __init__(self, init_confs, modelclass):
    super(PointwiseCxrBootStrap, self).__init__(init_confs)
    if self.is_on_blink(): self._local_mode = (sys.argv[0].count('application') == 0)
    else:
      self._local_mode = False
    self.modelclass = modelclass
  # need override
  def build_graph(self, task_id, context, **kwargs):
    '''
    Re-init config for blink
    tf.app.flags.DEFINE_string("ps_hosts", "", "ps hosts")
    tf.app.flags.DEFINE_string("worker_hosts", "", "worker hosts")
    tf.app.flags.DEFINE_string("job_name", "", "job name")
    tf.app.flags.DEFINE_integer("task_index", "-1", "task index")
    tf.app.flags.DEFINE_string("protocol", "grpc", "default grpc")
    '''
    if self.is_on_blink():
      self._init_confs.update(string2kv(context.get_param(), ';', '='))
      self._init_confs["worker_num"] = context.get_worker_num()
      self._init_confs["ps_num"] = context.get_ps_num()
    else:
      self._init_confs["worker_num"] = len(self._init_confs["worker_hosts"].split(','))
      self._init_confs["ps_num"] = len(self._init_confs["ps_hosts"].split(','))

    self._init_confs['task_id'] = task_id
    self._init_confs['task_index'] = task_id
    if self._init_confs['checkpoint_dir'] == '':
      self._init_confs['save_dir'] = self._init_confs['checkpointDir']
    else:
      self._init_confs['save_dir'] = self._init_confs['checkpoint_dir']

    print("----------------build graph-----------------")
    #self.istrain = tf.placeholder(tf.bool, name="training")
    self.model = self.modelclass(self._init_confs)

    self.oplist_2run,self.resetauc,self.istrain = self.model.forward()
    tf.logging.info("----------------finish building graph-----------------")
      #self.saver = tf.train.Saver()
  # need override
  def create_session(self, target, task_id, **kwargs):
    tf.logging.info("create session with target %s, task_id %d, mode %s" % (target, task_id, self.model.mode))
    hooks = None
    try:
      tracing_hook = tf.train.TracingHook(self.model.debug_dir, 100, min_vtrace_level=1, is_global=False)
      hooks = []
      hooks.append(tracing_hook)
    except Exception,e:
      tf.logging.info('Got Hook error, use no hook')
      tf.logging.info(str(e.__str__()))
      tf.logging.info('**************************')

    self.target = target
    sess = tf.train.MonitoredTrainingSession(
      master=target,
      is_chief=(task_id == 0),
      checkpoint_dir=None if self.model.native else self._init_confs['save_dir'],
      save_checkpoint_secs=None if self.model.native or self.model.mode!="train" else 60*60,
      hooks=hooks,
      chief_only_hooks=None)

    return sess

  def end_session(self, task_id, **kwargs):
    if task_id!=0 or self.model.mode!="train":
      return
    '''
    tf.logging.info("start exporting MetaGraph......")
    g = self.model.get_graph_def_for_rank()
    meta_def = self.model.gen_meta_graph_def(session = tf.Session(target=self.target),graph_def=g)

    from google.protobuf import text_format
    mdef_txt = text_format.MessageToString(meta_def)
    file_name=self._init_confs['save_dir']+"metagraph/20180101/data/0_0"
    writeGFile = gfile.GFile(file_name, mode='w')
    writeGFile.write(mdef_txt)
    writeGFile.flush()
    writeGFile.close()
    tf.logging.info("finish exporting MetaGraph!!!")
    '''

    tf.logging.info("start exporting fg.json ......")
    writeGFile = gfile.GFile(self._init_confs['save_dir']+"fg.json", mode='w')
    writeGFile.write(self.model.get_jsonstr())
    writeGFile.flush()
    writeGFile.close()
    tf.logging.info("finish exporting fg.json !!!")

    try:
      copy_lastest_ckpt = tf.contrib.framework.copy_lastest_ckpt
    except:
      copy_lastest_ckpt = tf.train.copy_lastest_ckpt
    tf.logging.info("start exporting ckpt for rtp ......")
    if self.model.bizdate=="-1":
      copy_lastest_ckpt(self._init_confs['save_dir'], self._init_confs['save_dir']+"ckpt4rtp")
    else:
      copy_lastest_ckpt(self._init_confs['save_dir'], self._init_confs['save_dir']+"ckpt4rtp/"+self.model.bizdate+"/data",False)
    tf.logging.info("finish exporting ckpt for rtp !!!")


  # need override
  def run_session(self, sess, task_id, **kwargs):
    print("run session with task_id %d and mode %s" % (task_id, self.model.mode))
    if self.is_on_blink():
      thread_ind = kwargs['thread_ind']
    else:
      thread_ind = 0

    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    if self.model.mode == "train" or self.model.native:
      self.iterate_train(sess, task_id, thread_ind)
    if self.model.mode == "predict":
      self.iterate_predict(sess, task_id, thread_ind)
    print("finish run session with task_id %d" % (task_id))
    if self.model.odpsWriter is not None:
      sess.run(self.model.odpsWriterClose)
    print("odps writer closed.")

  def iterate_predict(self, mon_sess, task_id, thread_ind):
    tf.logging.info("predict...")
    pred = self.model.get_predict()
    batch_count = 0
    while True:
      try:
        res = mon_sess.run(pred, feed_dict={self.istrain:False})
        tf.logging.info('Worker taskid:{}, threadid:{}, BatchNum:{}'.format(task_id, thread_ind, batch_count))
        self.model.predict_callback(res)
        batch_count += 1
      except tf.errors.ResourceExhaustedError as e:
        tf.logging.info('Got tf.errors.ResourceExhaustedError')
        tf.logging.info('**************************')
        tf.logging.info(str(e.__str__()))
        tf.logging.info('**************************')
        break
      except tf.errors.OutOfRangeError as e:
        tf.logging.info('Got tf.errors.OutOfRangeError')
        tf.logging.info('**************************')
        #tf.logging.info(str(e.__str__()))
        tf.logging.info('**************************')
        break
      except Exception as e:
        tf.logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        break

  def iterate_train(self, mon_sess, task_id, thread_ind):
    tf.logging.info("start iterate_train...")
    batch_count = 0
    pending = self.model.pending if not self.model.native else 1
    while True:
      if batch_count%pending==0:
        tf.logging.info("reset auc")
        prepossum = 0
        prenegsum = 0
        gtpossum = 0
        gtnegsum = 0
        mae = 0
        mse = 0
        eof = 0
        total_loss = 0.0
        mon_sess.run(self.resetauc)
      try:
        # mon_sess.run(tf.convert_to_tensor(stream_vars_auc))
        '''
        if task_id==0 and batch_count%10==9 and tt:
          return
        '''
        auc, prepossum_,prenegsum_,gtpossum_,gtnegsum_,mae_,mse_, _, loss = mon_sess.run(self.oplist_2run, feed_dict={self.istrain:True})
        prepossum += prepossum_
        prenegsum += prenegsum_
        gtpossum += gtpossum_
        gtnegsum += gtnegsum_
        mae += mae_
        mse += mse_
        total_loss += loss

        batch_count+=1

        if batch_count%pending==pending-1:
          tf.logging.info('Worker taskid:{}, threadid:{}, BatchNum:{}, loss:{}, AUC:{}'.format(task_id, thread_ind, batch_count, total_loss/pending, auc))
          #tf.logging.info( "label: {}".format(lab) )
          tf.logging.info( "pcopc: {}".format( (prepossum+prenegsum)/(gtpossum+1e-8)) )
          tf.logging.info( "pos mean: {}".format(prepossum/(gtpossum+1e-8)))
          tf.logging.info( "neg mean: {}".format(prenegsum/(gtnegsum+1e-8)))
          tf.logging.info( "mae: {}".format( mae) )
          tf.logging.info( "rmse: {}".format( np.sqrt(mse)))
          tf.logging.info( "posSUM: {}".format( gtpossum))
          tf.logging.info( "negSUM: {}".format( gtnegsum))
          tf.logging.info( "CTR: {}".format(gtpossum/(gtpossum+gtnegsum)))
          tf.logging.info( "preCTR: {}".format((prenegsum+prepossum)/(gtpossum+gtnegsum)))
      except tf.errors.ResourceExhaustedError as e:
        tf.logging.info('Got tf.errors.ResourceExhaustedError')
        tf.logging.info('**************************')
        tf.logging.info(str(e.__str__()))
        tf.logging.info('**************************')
        break
      except tf.errors.OutOfRangeError as e:
        tf.logging.info('Got tf.errors.OutOfRangeError')
        tf.logging.info('**************************')
        #tf.logging.info(str(e.__str__()))
        tf.logging.info('**************************')
        break
      except Exception as e:
        tf.logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        break
