from __future__ import print_function
import os, json, time, sys

currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + "/..")
from util.runtime.runtime_bootstrap import BootStrap
from model.model_column import ModelColumns, merge_model_column
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
import traceback
from feature.sample_io import build_hdfs_inputs_iterator, build_hdfs_inputs_iterator_shard
from util.pdd_io.hdfs_tony import getFileList
import numpy as np
from util.config.seqconfig import SequenceModelConfig
from feature import parse_column
from util.tflog import tflogger as logging
from model.ops.utils import reset_variables
from model.ops import checkpoint_utils
# from util.tflog import Tfps2Printer
# logging = Tfps2Printer("kongdu_dram_refactor_porsche", tf.flags.FLAGS.protocol, len(tf.FLAGS.worker_hosts.split(',')), len(tf.FLAGS.worker_hosts.split(',')), tf.flags.FLAGS.job_name, tf.flags.FLAGS.task_index)
import tensorflow as tf
from tensorflow.python.saved_model.simple_save import simple_save
from tensorflow.python.lib.io import file_io
from util.hooks import ExportModelHook
from util.scaffold import restore_from_saved_model
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training.training_util import _get_or_create_global_step_read
from feature.pdd_feature_generator import FeatureGenerator

logging.info("welcome...")

class Train(BootStrap):
    def __init__(self):
        logging.info("init TrainBootStrap...")
        super(Train, self).__init__()
        self.models = []
        self.pms_models = []
        self.probs = []
        self.merged = None
        self.is_training = tf.placeholder_with_default(input=tf.constant(False), shape=(), name="training")
        self.use_tfrecord = tf.placeholder_with_default(input=tf.constant(True), shape=(), name='use_tfrecord')
        self.share_context = tf.placeholder_with_default(input=tf.constant(True), shape=(), name='share_context')
        self.context_serving_ph = tf.placeholder_with_default(input=tf.constant(["1", "2"]), shape=[None], name="context")
        self.which_model = tf.placeholder_with_default(input=tf.constant(0), shape=(), name='which_model')
        self.batch_data = None
        self.oss_summary_writer = None
        self.hdfs_summary_writer = None
        self.reset_auc_ops = []
        self.auc_local_vars = []
        self.global_step = None
        self.hooks = []
        self.export_model = None
        self.saver_hook = None
        self.scaffold = None
        logging.info("debug_dir: %s" % tf.flags.FLAGS.debug_dir)
        logging.info("save_dir: %s" % tf.flags.FLAGS.save_dir)

    def buildGraph(self, task_id, context):

        logging.info("******begin********")
        logging.info("run with platform=%s" % str(tf.flags.FLAGS.platform))

        # init config
        '''
        self.train_config = self.parse_config(context.get_param(), platform=tf.flags.FLAGS.platform,
                                              cluster=context.get_cluster(), task_id=task_id,
                                              is_local=(tf.flags.FLAGS.platform == 'local'),
                                              worker_num=context.get_worker_num(), ps_num=context.get_ps_num(),
                                              currentPath=currentPath)
        '''
        job_json = json.load(open(currentPath + "/../" + tf.flags.FLAGS.job_conf))
        self.train_config = self.parse_config_from_dict(job_json["train_conf"],
                                                        platform=tf.flags.FLAGS.platform, cluster=context.get_cluster(),
                                                        task_id=task_id, is_local=(tf.flags.FLAGS.platform == 'local'),
                                                        worker_num=context.get_worker_num(),
                                                        ps_num=context.get_ps_num(), currentPath=currentPath,
                                                        save_dir=tf.flags.FLAGS.save_dir,
                                                        job_conf=tf.flags.FLAGS.job_conf,
                                                        is_training=self.is_training,
                                                        use_tfrecord=self.use_tfrecord,
                                                        share_context=self.share_context
                                                        )
        if tf.flags.FLAGS.train_parts != '':
            self.train_config.train_parts = [tf.flags.FLAGS.train_parts]
        if tf.flags.FLAGS.eval_parts != '':
            self.train_config.eval_parts = [tf.flags.FLAGS.eval_parts]
        # model
        from model import MODELZOO
        Model = MODELZOO[self.train_config.build_model_name]
        # Sets up the global step Tensor
        self.global_step = tf.Variable(
            initial_value=0,
            name="col_global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

        # multi models for multi task
        with open(self.train_config.currentPath + "/../" + self.train_config.job_conf) as f:
            jobconf = json.load(f)
        idx = -1
        self.data_iterators = []
        self.train_itera = []
        for modelconf in jobconf["sources"]:
            idx += 1
            model_config = SequenceModelConfig(ps_num=context.get_ps_num())
            # model_config.updateFromStrMap(modelconf)
            model_config.updateFromMap(modelconf)
            mc_json = json.load(open(self.train_config.currentPath + "/../" + model_config.mc_conf))
            model = Model(self.train_config, model_config, ModelColumns(mc_json, self.train_config.is_local),
                          modelconf["model_name"], idx)
            self.models.append(model)
            self.probs.append(float(modelconf["prob"]))
            logging.info("model %s init done!" % modelconf["model_name"])
            # prepare the features
            if tf.flags.FLAGS.platform == "local":
                file_list_eval = ["../model_debug/fake_data/17"
                    , "../model_debug/fake_data/11"
                    , "../model_debug/fake_data/12"
                    , "../model_debug/fake_data/13"
                    , "../model_debug/fake_data/14"
                    , "../model_debug/fake_data/15"
                    , "../model_debug/fake_data/16"]
                file_list_train = file_list_eval
            else:
                file_list_eval = getFileList(self.train_config.eval_parts, model_config.table_name)
                file_list_train = getFileList(self.train_config.train_parts, model_config.table_name)

            if self.train_config.shard:
                with tf.variable_scope(
                        name_or_scope="%s_input_%s" %
                                      (modelconf["model_name"], "eval"),
                        reuse=tf.AUTO_REUSE):
                    itera1 = build_hdfs_inputs_iterator_shard(file_list_eval,
                                                              self.train_config.num_epochs * 10,
                                                              10 if self.train_config.is_local else self.train_config.eval_bs,
                                                              self.train_config.data_format)
                with tf.variable_scope(
                        name_or_scope="%s_input_%s" % (modelconf["model_name"], 'train'),
                        reuse=tf.AUTO_REUSE):
                    itera2 = build_hdfs_inputs_iterator_shard(file_list_train,
                                                              self.train_config.num_epochs,
                                                              10 if self.train_config.is_local else self.train_config.batch_size,
                                                              self.train_config.data_format)
                self.data_iterators.append(itera1)
                self.data_iterators.append(itera2)
                itera = itera2 if self.train_config.task_id > 0 else itera1
            else:
                bs = self.train_config.batch_size if self.train_config.task_id > 0 else self.train_config.eval_bs
                file_list = file_list_eval if self.train_config.task_id == 0 else file_list_train
                itera = build_hdfs_inputs_iterator(file_list,
                                                   self.train_config.task_id - 1 if self.train_config.task_id > 0 else 0,
                                                   self.train_config.worker_num - 1 if self.train_config.task_id > 0 else 1,
                                                   self.train_config.num_epochs if self.train_config.task_id > 0 else
                                                   self.train_config.num_epochs * 10,
                                                   10 if self.train_config.is_local else bs,
                                                   self.train_config.data_format)
                self.data_iterators.append(itera)
            self.train_itera.append(itera)
            self.models[idx].itera = itera
            logging.info("built dataset for %s done" % model_config.model_name)
        idx = 0
        for modelconf in jobconf["sources"]:
            idx += 1
            model_config = SequenceModelConfig(ps_num=context.get_ps_num())
            # model_config.updateFromStrMap(modelconf)
            model_config.updateFromMap(modelconf)
            mc_json = json.load(open(self.train_config.currentPath + "/../" + model_config.mc_conf))
            model = Model(self.train_config, model_config, ModelColumns(mc_json, self.train_config.is_local),
                          modelconf["model_name"], idx)
            self.pms_models.append(model)
            # self.probs.append(float(modelconf["prob"]))
            logging.info("pms_model %s init done!" % modelconf["model_name"])
            # prepare the features

        '''
        xx = []
        for i in range(len(self.models)):
            x = tf.equal(self.which_model, tf.constant(i))
            xx.append((x, lambda: tf.Print(self.train_itera[i].get_next(), [self.models[i].model_name, self.which_model, tf.constant(i), x])))
        self.batch_data = tf.case(xx, default=xx[0][1], exclusive=True)
        '''
        def switch_model(start, models, callable, which_model):
            if len(models) > 2:
                x = tf.equal(which_model, tf.constant(start))
                return tf.cond(x, lambda: callable(models[0]),
                               lambda: switch_model(start+1, models[start+1:], callable, which_model))
            if len(models) == 2:
                x = tf.equal(which_model, tf.constant(start))
                return tf.cond(x, lambda: callable(models[0]),
                               lambda: callable(models[1]))
            elif len(models) == 1:
                return callable(models[0])
            else:
                raise ValueError("switch data error")
        self.batch_data = switch_model(0, self.models, lambda j: j.itera.get_next(), self.which_model)
        allmc = reduce(lambda a, y: merge_model_column(a, y), [x.mc for x in self.models])
        schema_json = json.load(open(self.train_config.currentPath + "/../" + self.train_config.fg_conf))
        self.fg = FeatureGenerator(self.train_config, schema_json, allmc.simple_dict)

        # model running to train
        self.parsed_feature, self.final_features = self.fg.build_tf_tensors_share_context(
            self.batch_data, self.use_tfrecord, self.share_context, self.context_serving_ph, False)
        for idx in range(len(self.models)):
            model = self.models[idx]
            model.build_main_part(self.final_features, self.is_training)
            if 'item_vecs' not in self.final_features:
                if hasattr(model, 's_item_vecs_map'):
                    self.final_features['item_vecs_da'] = model.s_item_vecs_map
                if hasattr(model, 's_ivec_norm'):
                    self.final_features['item_vecs'] = model.s_ivec_norm
            logging.info("model %s built done!" % model.name)

        if self.models[0].train_op is None:
            loss = switch_model(0, self.models, lambda j: j.loss, self.which_model)
            from tensorflow.contrib import layers
            # with tf.control_dependencies([switch_model(0, self.models, lambda j: tf.assign_add(j.global_step, 1, use_locking=True), self.which_model)]):
            train_op = layers.optimize_loss(
                loss,
                global_step=None,
                #switch_model(0, self.models, lambda j: j.global_step, self.which_model),
                learning_rate=self.train_config.initial_learning_rate,
                optimizer='Adagrad',
                increment_global_step=False)
            # train_op = tf.group([train_op, switch_model(0, self.models, lambda j: tf.assign_add(j.global_step, 1, use_locking=True), self.which_model)])
            for m in self.models:
                m.train_op = train_op
        self.probs = [x / np.sum(self.probs) for x in self.probs]

        # model running in pms
        self.pms_parsed_feature, self.pms_final_features = self.fg.build_tf_tensors_share_context(
            self.batch_data, self.use_tfrecord, self.share_context, self.context_serving_ph, True)
        for idx in range(len(self.pms_models)):
            model = self.pms_models[idx]
            model.build_main_part(self.pms_final_features, self.is_training, train=False)
            if 'item_vecs' not in self.pms_final_features:
                if hasattr(model, 's_item_vecs_map'):
                    self.pms_final_features['item_vecs_da'] = model.s_item_vecs_map
                if hasattr(model, 's_ivec_norm'):
                    self.pms_final_features['item_vecs'] = model.s_ivec_norm
            logging.info("pms model %s built done!" % model.name)
        # init embedding
        init_assignment_map = {
            # "seq_input_from_feature_columns/": "seq_input_from_feature_columns/"
            "input_from_feature_columns/": "input_from_feature_columns/"
            # , "Share/": "Share/"
            # , "Share-User-Vec-Space/": "Share-User-Vec-Space/"
            # , "Recall_CTR/": "Recall_CTR/"
            # , "Recall_CTR-Position-Network/": "Recall_CTR-Position-Network/"
            # , "Recall_CVR/": "Recall_CVR/"
            # , "Recall_CVR-Position-Network/": "Recall_CVR-Position-Network/"
        }
        if self.train_config.is_restore_embedding and self.train_config.embedding_checkpoint_dir is not None:
            logging.info("[init variables] from dir: %s" % self.train_config.embedding_checkpoint_dir)
            checkpoint_utils.init_from_checkpoint(self.train_config.embedding_checkpoint_dir, init_assignment_map)
        # Create Table if not exists
        '''
        if self.train_config.predict:
            if np.max(self.probs) != 1.:
                raise RuntimeError(
                    "Only Support One Model, use [cotrain_one_model_name=CTR/CVR] "
                    "to choose name of model, current probs = %s" % str(self.probs))
            self.odps = myodps.resetOdpsTable(main_training_config.table_name + main_training_config.predict_postfix,
                                              task_id=task_id,
                                              local_mode=local_mode, odps_user=main_training_config.odps_user)
        '''
        """Sets up reset ops."""
        self.reset_auc_ops = []
        self.auc_local_vars = []
        for i, tm in enumerate(self.models):
            rao, localvar = reset_variables(tf.GraphKeys.LOCAL_VARIABLES, [tm.name, 'auc'])
            self.reset_auc_ops.append(rao)
            self.auc_local_vars.append(localvar)

        # Print All Variables to Debug
        all_local_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        for n in all_local_var:
            logging.info("all_local_var:%s" % str(n))
        all_global_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for n in all_global_var:
            logging.info("all_global_var:%s" % str(n))

        self.merged = tf.summary.merge_all()

    def createSession(self, target, task_id):
        tfconf = tf.ConfigProto()  # log_device_placement=True, allow_soft_placement=True)
        # tfconf.gpu_options.allow_growth = True

        # logdir = os.environ['TB_LOG_DIR'] if task_id is 0 else None
        logdir = tf.flags.FLAGS.debug_dir if task_id is 0 else None
        # logdir = None
        # if task_id == 0:
        #    logdir = tf.flags.FLAGS.save_dir + "logdir/"
        # if task_id == 1:
        #    logdir = tf.flags.FLAGS.save_dir + "logdir/train/"

        # saved_model_input = {'examples': self.models[0].serving_input}
        saved_model_input = {'context': self.context_serving_ph,
                             'documents': self.batch_data,
                             # 'vector': self.pms_models[0].s_ivec_norm
                             # 'vector1': self.models[1].s_ivec_norm
                             }
        if self.pms_models[0].s_ivec_norm is not None:
            saved_model_input['vector'] = self.pms_models[0].s_ivec_norm
        saved_model_output = {}
        for i in range(len(self.pms_models)):
            if type(self.pms_models[i].predictions) is list:
                for j in range(len(self.pms_models[i].predictions)):
                    saved_model_output['output%d_%d' % (i, j)] = self.pms_models[i].predictions[j]
            else:
                saved_model_output['output%d' % i] = self.pms_models[i].predictions
            saved_model_output['user_input_layer%d' % i] = self.pms_models[i].user_input_layer
            saved_model_output['item_input_layer%d' % i] = self.pms_models[i].item_input_layer
        '''
        saved_model_output = {'output0': self.models[0].predictions,
                              'output1': self.models[1].predictions,
                              'uvec0': self.models[0].s_uvec_norm,
                              'uvec1': self.models[1].s_uvec_norm,
                              'user_input_layer0': self.models[0].user_input_layer,
                              'user_input_layer1': self.models[1].user_input_layer,
                              'ivec0': self.models[0].s_ivec_norm,
                              'ivec1': self.models[1].s_ivec_norm,
                              'item_input_layer0': self.models[0].item_input_layer,
                              'item_input_layer1': self.models[1].item_input_layer
                              }

        for k, v in self.parsed_feature.items():
            if not isinstance(v, tf.Tensor):
                saved_model_output[k+'_raw'] = tf.sparse_tensor_to_dense(v, default_value="")
            else:
                saved_model_output[k+'_raw'] = v
        # saved_model_output.update(self.final_features)
        '''
        self.export_model = ExportModelHook(self.train_config.save_dir + 'saved_model/',
                                            True,
                                            self.train_config.save_dir + 'saved_model/donelist',
                                            self.train_config.save_dir + 'saved_model/trackerlist',
                                            saved_model_input,
                                            saved_model_output,
                                            is_local=False,
                                            tracker='')
        self.export_model_vec = ExportModelHook(self.train_config.save_dir + 'saved_model_vec/',
                                                True,
                                                self.train_config.save_dir + 'saved_model_vec/donelist',
                                                self.train_config.save_dir + 'saved_model_vec/trackerlist',
                                                saved_model_input,
                                                saved_model_output,
                                                is_local=False,
                                                tracker='')
        local_init_op = tf.group(tf.local_variables_initializer(), [x.initializer for x in self.data_iterators])
        the_scaffold = tf.train.Scaffold(local_init_op=local_init_op)  # , init_fn=init_fn)
        '''
        if self.train_config.need_save_ckp and tf.flags.FLAGS.mode != 'predict':
            self.saver_hook = basic_session_run_hooks.CheckpointSaverHook(self.train_config.save_dir,
                                                                          save_steps=None,
                                                                          save_secs=self.train_config.save_checkpoint_secs,
                                                                          scaffold=the_scaffold)
        '''
        self.scaffold = the_scaffold
        try:
            tracing_hook = tf.train.TracingHook(tf.flags.FLAGS.debug_dir + "trace/" + str(task_id) + "/", 1000,
                                                min_vtrace_level=1, is_global=False)
            self.hooks.append(tracing_hook)
        except Exception as e:
            tf.logging.info('Got Hook error, use no tracehook')
            tf.logging.info(str(e.__str__()))
            tf.logging.info('**************************')

        return tf.train.MonitoredTrainingSession(
            master=target,
            is_chief=(task_id == 0),
            # save_checkpoint_sync=False,
            # config=tfconf,
            checkpoint_dir=self.train_config.save_dir,
            save_checkpoint_secs=self.train_config.save_checkpoint_secs if self.train_config.need_save_ckp and tf.flags.FLAGS.mode == 'train' else None,
            save_summaries_steps=self.train_config.save_summaries_steps if task_id < 1 and tf.flags.FLAGS.mode == 'train' else None,
            # save_summaries_secs=None,
            summary_dir=logdir,
            hooks=self.hooks,
            # chief_only_hooks=[self.saver_hook] if self.saver_hook is not None else None,
            scaffold=the_scaffold)

    def runSession(self, mon_sess, task_id, thread_ind):
        if tf.flags.FLAGS.mode == 'predict':
            self.runPredict(mon_sess, task_id, thread_ind)
        elif tf.flags.FLAGS.mode == 'vec':
            self.getVector(mon_sess, task_id, thread_ind)
        elif tf.flags.FLAGS.mode == 'train':
            self.runTrain(mon_sess, task_id, thread_ind)
        elif tf.flags.FLAGS.mode == 'eval':
            self.runTrain(mon_sess, task_id, thread_ind)
        logging.info("finish run")

    def choose_model(self, model_name=None):
        choice_idx = -1
        if model_name is None:
            choice_idx = np.random.choice(len(self.probs), p=self.probs)
        else:
            for idx, m in enumerate(self.models):
                if model_name == m.name:
                    choice_idx = idx
                    break
        train_op = [self.models[choice_idx].train_op, self.global_step_add]
        return self.models[choice_idx], train_op, choice_idx, self.reset_auc_ops[choice_idx], self.auc_local_vars[
            choice_idx]

    def choose_pms_model(self, model_name=None):
        choice_idx = -1
        if model_name is None:
            choice_idx = np.random.choice(len(self.probs), p=self.probs)
        else:
            for idx, m in enumerate(self.pms_models):
                if model_name == m.name:
                    choice_idx = idx
                    break
        train_op = [self.pms_models[choice_idx].train_op, self.global_step_add]
        return self.pms_models[choice_idx], train_op, choice_idx, self.reset_auc_ops[choice_idx], self.auc_local_vars[
            choice_idx]

    def runTrain(self, mon_sess, task_id, thread_ind):
        localcnt = 0
        taskcnt = [0] * len(self.models)
        timestart = time.time()
        while not mon_sess.should_stop():
            localcnt += 1
            model, train_op, index, reset_auc_ops, localvar = self.choose_model()
            taskcnt[index] += 1
            feed_dict = {}
            try:
                if taskcnt[index] % 150:
                    if self.worker_finish_ratio(self.train_config.worker_num) > 0.8:
                        logging.info("worker_finish end: %s" % str(len(set(self.finishedWorkers()))))
                        break
                if task_id == 0 or task_id == 1:
                    # eval, chief worker
                    feed_dict.update({self.is_training: False, self.use_tfrecord: False, self.share_context: False, self.which_model: index})
                    co_global_step, global_step, loss, metrics, flocalv, labels = mon_sess.run(
                        [self.global_step, model.global_step, model.loss, model.metrics, localvar, model.label],
                        feed_dict=feed_dict)

                    auc, totalauc = metrics['scalar/batch_auc'], metrics['scalar/total_auc']
                    logging.info('%s-Col_Global_Step:%s, Global_Step:%s, loss=%s, totalauc=%s task_id=%s thread=%s' % (
                        model.model_name, str(co_global_step), str(global_step), str(loss),
                        str(totalauc), str(task_id), str(thread_ind)))

                else:  # training, slave worker
                    feed_dict.update({self.is_training: True, self.use_tfrecord: False, self.share_context: False, self.which_model: index})
                    co_global_step, global_step, _, loss, auc, totalauc, flocalv = mon_sess.run(
                        [self.global_step, model.global_step, train_op, model.loss,
                         model.metrics['scalar/batch_auc'], model.metrics['scalar/total_auc'], localvar],
                        feed_dict=feed_dict)
                    # print(seeds)
                    logging.info('%s-Col_Global_Step:%s, Global_Step:%s, loss=%s, totalauc=%s task_id=%s thread=%s' % (
                        model.model_name, str(co_global_step), str(global_step), str(loss),
                        str(totalauc), str(task_id), str(thread_ind)))

                '''
                if taskcnt[index] % self.train_config.save_summaries_steps == 0:
                    logging.info("task local step : %d" % taskcnt[index])
                    if tf.flags.FLAGS.platform == "pai" and task_id in (0, 1):
                        summ = mon_sess.run(self.merged, feed_dict=feed_dict)
                        self.oss_summary_writer.add_summary(summ, co_global_step)
                    elif task_id in (0, 1):
                        logging.info("write summary...")
                        summ = mon_sess.run(self.merged, feed_dict=feed_dict)
                        logging.info("summary : %s" % summ)
                        self.hdfs_summary_writer.add_summary(summ, co_global_step)
                        self.hdfs_summary_writer.flush()
                '''
                # newmark = np.max(flocalv[0][np.array([0, -1])])
                reset_step = self.train_config.auc_reset_step  # * 100 if tf.flags.FLAGS.mode == 'eval' else self.train_config.auc_reset_step
                if taskcnt[index] % reset_step == 500:
                    self.report_tf_metric('total_auc', totalauc, ex_tag={"model_name": model.name})
                    logging.info("reset auc model:%s" % model.name)
                    logging.info("auc_reset_step:%s" % str(self.train_config.auc_reset_step))
                    logging.info("total auc: %s" % str(totalauc))
                    logging.info('reset auc ops run')
                    mon_sess.run(reset_auc_ops, feed_dict=feed_dict)
                    # flocalv = mon_sess.run(reset_auc_ops, feed_dict=feed_dict)
                    # logging.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([x.name for x in localvar]))
                    # logging.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([str(x[index]) for x in flocalv]))
            except (ResourceExhaustedError, OutOfRangeError) as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                # succ = co_trainer.release_model(index)
                # logging.info("Release Model index : %s" % index)
                # logging.info("Release Model index : %s %s" % (index, model.model_name))
                # if not succ:
                #   break
                break
            except Exception as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        if task_id == 0:
            time.sleep(10)
            # gstep = mon_sess.run(_get_or_create_global_step_read())
            # self.scaffold.saver.save(mon_sess._coordinated_creator.tf_sess,
            #                         self.train_config.save_dir + 'sm.ckpt', gstep)
            ts = self.export_model.export_sm(mon_sess._coordinated_creator.tf_sess)
            self.export_model.write_donefile(ts)
            # self.export_model_vec.export_sm(mon_sess._coordinated_creator.tf_sess)

    def runPredict(self, mon_sess, task_id, thread_ind):
        if int(thread_ind) != 0:  # predict with one thread
            logging.info("Skip thread_ind==%s" % str(thread_ind))
            return
        if int(task_id) == 0:  # examples not match
            logging.info("Skip chief worker")
            while True:
                logging.info("worker finished num: %s" % str(len(set(self.finishedWorkers()))))
                time.sleep(60 * 4)  # 4min
                if self.train_config.worker_num == len(set(self.finishedWorkers())) + 1:
                    logging.info("all other worker finished!")
                    return

        predict_step = 100000 * 100000
        localcnt = 0
        fname = os.path.join(tf.flags.FLAGS.tableoutput, "part_%d" % self.train_config.task_id)
        hdfs_file = tf.gfile.Open(fname, 'w')
        while True:
            localcnt += 1
            # model, train_op, index, _, _ = co_trainer.choose_model()
            model, train_op, index, reset_auc_ops, localvar = self.choose_pms_model(self.train_config.predict_model_name)

            feed_dict = {self.is_training: False, self.use_tfrecord: False, self.share_context: False}
            try:
                debugs = [model.user_input_layer, model.item_input_layer]
                for x in ['user_mall_fav_c20', 'query_unigram']:
                    debugs.append(self.final_features[x])
                    #debugs.append(self.parsed_feature[x])
                if "vector" in fname:
                    res = mon_sess.run(model.predict_vector, feed_dict=feed_dict)
                else:
                    dd, res = mon_sess.run([debugs, model.write_to_table], feed_dict=feed_dict)
                    print(dd)
                    # res = mon_sess.run(model.write_to_table, feed_dict=feed_dict)
                hdfs_file.write(res)
                if localcnt % 1000 == 0:
                    logging.info(res)
                logging.info('step_passed=%s step_left=%s' % (str(localcnt), str(predict_step)))
                predict_step -= 1
                if predict_step < 1:
                    break
            except (ResourceExhaustedError, OutOfRangeError) as e:
                logging.info('Got OutOfRangeError run : %s | %s' % (e, traceback.format_exc()))
                break
            except Exception as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

        # while str(task_id) == '0' and self.worker_finish_ratio(self.train_config.worker_num) < 0.99:
        #  logging.info("wait finish %s" % str(len(set(self.finishedWorkers()))))
        #  time.sleep(60)
        '''
        try:
            model, train_op, index, _, _ = self.choose_model(self.train_config.model_name)
            feed_dict = {'training:0': False}
            _ = mon_sess.run(model.odpsWriterClose, feed_dict=feed_dict)
        except Exception as e:
            logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        '''
        hdfs_file.close()
        logging.info("Finish Run, sleep")

    def getVector(self, mon_sess, task_id, thread_ind):
        if int(thread_ind) != 0:  # predict with one thread
            logging.info("Skip thread_ind==%s" % str(thread_ind))
            return
        if int(task_id) == 0:  # examples not match
            logging.info("Skip chief worker")
            while True:
                logging.info("worker finished num: %s" % str(len(set(self.finishedWorkers()))))
                time.sleep(60 * 4)  # 4min
                if self.train_config.worker_num == len(set(self.finishedWorkers())) + 1:
                    logging.info("all other worker finished!")
                    # gstep = mon_sess.run(_get_or_create_global_step_read())
                    # self.scaffold.saver.save(mon_sess._coordinated_creator.tf_sess, self.train_config.save_dir + 'sm.ckpt', gstep)
                    xx = tf.flags.FLAGS.tableoutput.split('/')
                    ts = xx[-2] if xx[-1] == 'vector' else xx[-3]
                    ts = int(ts) + 1
                    self.export_model_vec.export_sm(mon_sess._coordinated_creator.tf_sess, ts)
                    write_status = os.system("hadoop fs -mv %s/%d/vector %s/%d/" %
                                             (self.train_config.save_dir + 'saved_model_vec',
                                              ts-1,
                                              self.train_config.save_dir + 'saved_model_vec',
                                              ts))
                    if write_status == 0:
                        logging.info("###########LOG INFO Sucessfully mv vector")
                        self.export_model_vec.write_donefile(ts)
                    else:
                        logging.info("###########ERROR INFO Failed to mv vector")
                    return

        predict_step = 100000 * 100000
        localcnt = 0
        fname = os.path.join(tf.flags.FLAGS.tableoutput, "part_%d" % self.train_config.task_id)
        hdfs_file = tf.gfile.Open(fname, 'w')
        while True:
            localcnt += 1
            # model, train_op, index, _, _ = co_trainer.choose_model()
            model, train_op, index, reset_auc_ops, localvar = self.choose_model(self.train_config.predict_model_name)

            feed_dict = {self.is_training: False, self.use_tfrecord: False, self.share_context: False}
            try:
                '''
                res = mon_sess.run(model.predict_vector, feed_dict=feed_dict)
                '''

                gid, vec, vaf = mon_sess.run([model.pk, model.s_ivec_norm, model.vec_append_fea], feed_dict=feed_dict)
                if localcnt % 1000 == 0:
                    logging.info(gid)
                    logging.info(type(gid))
                gid = gid.tolist()
                vec = vec.tolist()
                vaf = [x.tolist() for x in vaf]
                if localcnt % 1000 == 0:
                    logging.info(gid)
                    logging.info(vec)
                    logging.info(vaf)
                    logging.info("data num : %d" % len(gid))
                    logging.info("vec dim : %d" % len(vec[0]))
                    for k in range(len(vaf)):
                        logging.info("vec dim vaf : %d" % len(vaf[k][0]))
                    logging.info(type(gid))
                c = []
                from util.feature_pb.vector_pb2 import VectorFeature
                import base64
                #for pk, vv, vvv in zip(gid, vec, vaf):
                for i in range(len(gid)):
                    pk = gid[i]
                    vv = vec[i]
                    #if len(vv) != 128:
                    #    logging.info(len(vv))
                    if long(pk[0]) == 0:
                        logging.info(pk)
                    d = VectorFeature()
                    d.pk = long(pk[0])
                    for v in vv:
                        d.vec.append(float(v))
                    for xx in vaf:
                        for x in xx[i]:
                            if type(x) is str and x in ['', ' ', '  ']:
                                x = -1.0
                            d.vec.append(float(x))

                    pp = d.vec[-2] if d.vec[-2] > 0 else d.vec[-1]
                    ppp = pp if pp > 0 and pp < d.vec[-3] else d.vec[-3]
                    d.vec.append(ppp if ppp > 1.0 else 1.0)
                    if localcnt % 1000 == 0:
                        logging.info(d.pk)
                        logging.info(d.vec)
                    dd = base64.encodestring(d.SerializeToString()).replace("\n", "")
                    c.append(dd)
                res = '\n'.join(c)
                if len(c) != self.train_config.batch_size:
                    logging.info(len(c))

                if localcnt > 1:
                    res = '\n' + res
                hdfs_file.write(res)
                #if localcnt % 1000 == 0:
                #    logging.info(res)
                logging.info('step_passed=%s step_left=%s' % (str(localcnt), str(predict_step)))
                predict_step -= 1
                if predict_step < 1:
                    break
            except (ResourceExhaustedError, OutOfRangeError) as e:
                logging.info('Got OutOfRangeError run : %s | %s' % (e, traceback.format_exc()))
                break
            except Exception as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        # while str(task_id) == '0' and self.worker_finish_ratio(self.train_config.worker_num) < 0.99:
        #  logging.info("wait finish %s" % str(len(set(self.finishedWorkers()))))
        #  time.sleep(60)
        '''
        try:
            model, train_op, index, _, _ = self.choose_model(self.train_config.model_name)
            feed_dict = {'training:0': False}
            _ = mon_sess.run(model.odpsWriterClose, feed_dict=feed_dict)
        except Exception as e:
            logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        '''
        hdfs_file.close()
        logging.info("Finish Run, sleep")


def main(_):
    bootstrap = Train()
    bootstrap.start()


if __name__ == '__main__':
    tf.app.run()
    logging.info("bye...")
