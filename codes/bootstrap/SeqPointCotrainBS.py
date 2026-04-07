import os,json,time,sys
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath+"/..")
from util.runtime.runtime_bootstrap import BootStrap
from model.model_column import ModelColumns, merge_model_column
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
import traceback
from feature.sample_io import build_hdfs_inputs_iterator,build_hdfs_inputs_iterator_shard
from util.pdd_io.hdfs_tony import getFileList
import numpy as np
from model.normal.rank_cxr_model import KonduModel as Model
from util.config.seqconfig import SequenceModelConfig
from feature import parse_column
from util.tflog import tflogger as logging
from model.ops.utils import reset_variables
# from util.tflog import Tfps2Printer
# logging = Tfps2Printer("kongdu_dram_refactor_porsche", tf.flags.FLAGS.protocol, len(tf.FLAGS.worker_hosts.split(',')), len(tf.FLAGS.worker_hosts.split(',')), tf.flags.FLAGS.job_name, tf.flags.FLAGS.task_index)
import tensorflow as tf
from tensorflow.python.saved_model.simple_save import simple_save
from tensorflow.python.lib.io import file_io
from util.hooks import ExportModelHook
from util.scaffold import restore_from_saved_model
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training.training_util import _get_or_create_global_step_read
logging.info("welcome...")


class Train(BootStrap):
    def __init__(self):
        logging.info("init TrainBootStrap...")
        super(Train, self).__init__()
        self.models = []
        self.probs = []
        self.merged = None
        self.is_training = tf.placeholder_with_default(input=tf.constant(False), shape=(), name="training")
        self.use_tfrecord = tf.placeholder_with_default(input=tf.constant(True), shape=(), name='use_tfrecord')
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
        job_json = json.load(open(currentPath+"/../"+tf.flags.FLAGS.job_conf))
        self.train_config = self.parse_config_from_dict(job_json["train_conf"],
                                                        platform=tf.flags.FLAGS.platform, cluster=context.get_cluster(),
                                                        task_id=task_id, is_local=(tf.flags.FLAGS.platform == 'local'),
                                                        worker_num=context.get_worker_num(),
                                                        ps_num=context.get_ps_num(), currentPath=currentPath,
                                                        save_dir=tf.flags.FLAGS.save_dir,
                                                        job_conf=tf.flags.FLAGS.job_conf,
                                                        is_training=self.is_training,
                                                        use_tfrecord=self.use_tfrecord
                                                        )
        if tf.flags.FLAGS.train_parts != '':
            self.train_config.train_parts = [tf.flags.FLAGS.train_parts]
        if tf.flags.FLAGS.eval_parts != '':
            self.train_config.eval_parts = [tf.flags.FLAGS.eval_parts]

        # Sets up the global step Tensor
        self.global_step = tf.Variable(
            initial_value=0,
            name="col_global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

        with tf.name_scope("dont_save_me"):
            self.need_eval = tf.Variable(
                initial_value=False,
                name="need_eval",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
            if task_id == 0:
                self.set_eval_mode = tf.assign(self.need_eval, True)
                self.set_train_mode = tf.assign(self.need_eval, False)
            eval_done = []
            for i in range(context.get_worker_num()):
                eval_done.append(tf.Variable(
                    initial_value=False,
                    name="eval_done_%d" % i,
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES]))
            self.eval_done = eval_done
            self.set_eval_done = tf.assign(eval_done[task_id], True)
            self.reset_eval_done = tf.assign(eval_done[task_id], False)

        # multi models for multi task
        with open(self.train_config.currentPath + "/../" + self.train_config.job_conf) as f:
            jobconf = json.load(f)
        idx = -1
        self.data_iterators = []
        for modelconf in jobconf["sources"]:
            idx += 1
            model_config = SequenceModelConfig(ps_num=context.get_ps_num())
            # model_config.updateFromStrMap(modelconf)
            model_config.updateFromMap(modelconf)
            '''
            mc = ModelColumns(self.train_config.currentPath + "/../" + modelconf["mc_conf"],
                              tf.flags.FLAGS.platform == "local")
            mc_filter = getmcset(mc)
            seq_filter = getseqmcset(mc)
            with open(self.train_config.currentPath + "/../" + self.train_config.fg_conf) as f:
                fgjson = json.load(f)
                fgconf_list, fgconf_map = parse_column.parse_feature_conf(fgjson, mc_filter, seq_filter,
                                                                          tf.flags.FLAGS.platform == "local")
            model_config.fgconf_list, model_config.fgconf_map = fgconf_list, fgconf_map
            
            schema_json = json.load(open(self.train_config.currentPath + "/../" + model_config.lsc_conf))
            fg = FeatureGenerator(schema_json, mc_json)
            '''
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
                file_list_eval = getFileList(self.train_config.eval_parts,model_config.table_name)
                file_list_train = getFileList(self.train_config.train_parts, model_config.table_name)

            if self.train_config.shard:
                with tf.variable_scope(
                        name_or_scope="%s_input_%s" %
                                      (modelconf["model_name"], "eval"),
                        reuse=tf.AUTO_REUSE):
                    itera_eval = build_hdfs_inputs_iterator_shard(file_list_eval,
                                                                  self.train_config.num_epochs,
                                                                  10 if self.train_config.is_local else self.train_config.eval_bs,
                                                                  self.train_config.data_format)
                with tf.variable_scope(
                        name_or_scope="%s_input_%s" % (modelconf["model_name"], 'train'),
                        reuse=tf.AUTO_REUSE):
                    itera_train = build_hdfs_inputs_iterator_shard(file_list_train,
                                                                   self.train_config.num_epochs,
                                                                   10 if self.train_config.is_local else self.train_config.batch_size,
                                                                   self.train_config.data_format)
            else:
                bs = self.train_config.batch_size if self.train_config.task_id > 0 else self.train_config.eval_bs
                # file_list = file_list_eval if self.train_config.task_id == 0 else file_list_train
                itera_eval = build_hdfs_inputs_iterator(file_list_eval,
                                                        self.train_config.task_id - 1,
                                                        self.train_config.worker_num - 1,
                                                        self.train_config.num_epochs,
                                                        10 if self.train_config.is_local else self.train_config.eval_bs,
                                                        self.train_config.data_format)
                itera_train = build_hdfs_inputs_iterator(file_list_train,
                                                         self.train_config.task_id - 1,
                                                         self.train_config.worker_num - 1,
                                                         self.train_config.num_epochs,
                                                         10 if self.train_config.is_local else self.train_config.batch_size,
                                                         self.train_config.data_format)
            logging.info("fg for %s begin ..." % model_config.model_name)
            '''
            batch_data = tf.train.shuffle_batch_join(origin_feature,
                                                      batch_size=self.train_config.batch_size,
                                                      capacity=self.train_config.training_queue_capacity,
                                                      enqueue_many=True,
                                                      allow_smaller_final_batch=True, min_after_dequeue=0)
            '''
            self.data_iterators.append(itera_eval)
            self.data_iterators.append(itera_train)
            self.itera_eval = itera_eval
            batch_data = tf.cond(self.need_eval, lambda: itera_eval.get_next(), lambda: itera_train.get_next())
            # features = fg.parse_csv(batch_data)
            # logging.info("fg succ, %d" % len(features))
            # features = tf.staged(features, capacity=4, num_threads=4)
            # Build Graph
            model.build_main_part(batch_data, self.is_training)
            logging.info("model %s built done!" % model.name)
        self.probs = [x / np.sum(self.probs) for x in self.probs]

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

        local_true_positives = None
        local_true_negatives = None
        local_false_positives = None
        local_false_negatives = None
        for v in self.auc_local_vars:
            if "true_positives" in v.name:
                local_true_positives = v
            elif "true_negatives" in v.name:
                local_true_negatives = v
            elif "false_positives" in v.name:
                local_false_positives = v
            elif "false_negatives" in v.name:
                local_false_negatives = v
        with tf.name_scope("dont_save_me"):
            true_positives = []
            true_nagetives = []
            false_positives = []
            false_nagetives = []
            for i in range(context.get_worker_num()):
                true_positives.append(tf.Variable(
                    initial_value=[0] * self.train_config.auc_number_thre,
                    name="true_positives_%d" % i,
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES]))
                true_nagetives.append(tf.Variable(
                    initial_value=[0] * self.train_config.auc_number_thre,
                    name="true_nagetives_%d" % i,
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES]))
                false_positives.append(tf.Variable(
                    initial_value=[0] * self.train_config.auc_number_thre,
                    name="false_positives_%d" % i,
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES]))
                false_nagetives.append(tf.Variable(
                    initial_value=[0] * self.train_config.auc_number_thre,
                    name="false_nagetives_%d" % i,
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES]))
            self.true_positives = true_positives
            self.true_negatives = true_nagetives
            self.false_positives = false_positives
            self.false_negatives = false_nagetives
            self.clear_auc_vars = tf.group([tf.assign(self.true_positives[task_id], 0, use_locking=True),
                                            tf.assign(self.false_negatives[task_id], 0, use_locking=True),
                                            tf.assign(self.true_negatives[task_id], 0, use_locking=True),
                                            tf.assign(self.false_positives[task_id], 0, use_locking=True)])
            self.set_auc_vars = tf.group([tf.assign_add(self.true_positives[task_id], local_true_positives, use_locking=True),
                                          tf.assign_add(self.false_negatives[task_id], local_false_negatives, use_locking=True),
                                          tf.assign_add(self.true_negatives[task_id], local_true_negatives, use_locking=True),
                                          tf.assign_add(self.false_positives[task_id], local_false_positives, use_locking=True)])

        # Print All Variables to Debug
        all_local_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        for n in all_local_var:
            logging.info("all_local_var:%s" % str(n))
        all_global_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for n in all_global_var:
            logging.info("all_global_var:%s" % str(n))
        # custom a saver that ignores some vars
        savevar = []
        self.initops = []
        allv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        for v in allv:
            if "dont_save_me" in v.name:
                logging.info("do not save : %s" % str(v))
            else:
                savevar.append(v)
        self.saver = tf.train.Saver(var_list=savevar,
                                    sharded=True,
                                    allow_empty=True,
                                    max_to_keep=self.train_config.max_checkpoints_to_keep,
                                    name="custom_saver")
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)
        logging.info("tf.GraphKeys.SAVERS : %s" % str(tf.get_collection(tf.GraphKeys.SAVERS)))

        self.merged = tf.summary.merge_all()

    def createSession(self, target, task_id):
        tfconf = tf.ConfigProto()  # log_device_placement=True, allow_soft_placement=True)
        # tfconf.gpu_options.allow_growth = True

        #logdir = os.environ['TB_LOG_DIR'] if task_id is 0 else None
        logdir = tf.flags.FLAGS.debug_dir if task_id is 0 else None
        #logdir = None
        #if task_id == 0:
        #    logdir = tf.flags.FLAGS.save_dir + "logdir/"
        #if task_id == 1:
        #    logdir = tf.flags.FLAGS.save_dir + "logdir/train/"

        saved_model_input = {'examples': self.models[0].serving_input}
        saved_model_output = {'output': self.models[0].predictions}
        saved_model_output.update(self.models[0].features)
        self.export_model = ExportModelHook(self.train_config.save_dir + 'saved_model/',
                                            True,
                                            self.train_config.save_dir + 'saved_model/donelist',
                                            self.train_config.save_dir + 'saved_model/trackerlist',
                                            saved_model_input,
                                            saved_model_output,
                                            is_local=False,
                                            tracker='')
        local_init_op = tf.group(tf.local_variables_initializer(), [x.initializer for x in self.data_iterators])
        sm_dir = self.train_config.save_dir + 'saved_model/' + tf.flags.FLAGS.predict_saved_model + '/'
        init_fn = None if self.train_config.task_id > 0 or tf.flags.FLAGS.mode == 'train'\
            else lambda scaffold, sess: restore_from_saved_model(sess, sm_dir)
        the_scaffold = tf.train.Scaffold(local_init_op=local_init_op) #, init_fn=init_fn)
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
            save_checkpoint_secs=self.train_config.save_checkpoint_secs if self.train_config.need_save_ckp and tf.flags.FLAGS.mode != 'predict' else None,
            save_summaries_steps=self.train_config.save_summaries_steps if task_id < 1 and tf.flags.FLAGS.mode != 'predict' else None,
            # save_summaries_secs=None,
            summary_dir=logdir,
            hooks=self.hooks,
            # chief_only_hooks=[self.saver_hook] if self.saver_hook is not None else None,
            scaffold=the_scaffold)

    def runSession(self, mon_sess, task_id, thread_ind):
        if tf.flags.FLAGS.mode == 'predict':
            '''
            if self.train_config.platform == "blink":
                self.blinkPredict(mon_sess, task_id, thread_ind)
            elif self.train_config.platform == "pai":
                self.paiPredict(mon_sess, task_id, thread_ind)
            else:
                return
            '''
            self.runPredict(mon_sess, task_id, thread_ind)
        else:
            self.runTrain(mon_sess, task_id, thread_ind)
            logging.info("finish train")

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
        return self.models[choice_idx], train_op, choice_idx, self.reset_auc_ops[choice_idx], self.auc_local_vars[choice_idx]

    def runTrain(self, mon_sess, task_id, thread_ind):
        localcnt = 0
        taskcnt = [0, 0]
        timestart = time.time()
        while not mon_sess.should_stop():
            localcnt += 1
            model, train_op, index, reset_auc_ops, localvar = self.choose_model()
            taskcnt[index] += 1
            feed_dict = {}
            try:
                if task_id == 0:
                    # set eval mode and clear vars, chief worker
                    time.sleep(30)
                    feed_dict.update({self.is_training: False, self.use_tfrecord: False})
                    co_global_step, _, _, _, model_steps = mon_sess.run(
                        [self.global_step, self.set_eval_mode, self.clear_auc_vars, self.reset_eval_done, [m.global_step for m in self.models]],
                        feed_dict=feed_dict)
                    logging.info(
                        'START evalutaion !!! Col_Global_Step:%s-->task:%s, Global_Step:%s' % (
                            str(co_global_step),
                            str([model.model_name for model in self.models]),
                            str(model_steps)))
                    # wait for all workers eval done
                    t1 = time.time()
                    while True:
                        time.sleep(3)
                        worker_hasdone = mon_sess.run(self.eval_done, feed_dict=feed_dict)
                        alldone = True
                        done_workers = []
                        for x in range(1, len(worker_hasdone)):
                            if worker_hasdone[x] == 0:
                                alldone = False
                            else:
                                done_workers.append(x)
                        if alldone or time.time() - t1 > 60 * 30:  # 30 mins
                            break
                    # collet the auc vars
                    TPs, TNs, FNs, FPs, model_steps, co_global_step, _ = mon_sess.run(
                        [self.true_positives, self.true_negatives, self.false_negatives, self.false_positives,
                         [m.global_step for m in self.models], self.global_step, self.set_train_mode], feed_dict=feed_dict)
                    logging.info(
                        'FINISH evalutaion !!! Col_Global_Step:%s-->task:%s, Global_Step:%s, done worker num: %d' % (
                            str(co_global_step), str([model.model_name for model in self.models]),
                            str(model_steps), len(done_workers)))
                    if len(done_workers) > 0:
                        tp = 0
                        tn = 0
                        fn = 0
                        fp = 0
                        for x in done_workers:
                            tp += TPs[x]
                            tn += TNs[x]
                            fn += FNs[x]
                            fp += FPs[x]
                        tpr = tp / (tp + fn)
                        fpr = fp / (tn + fp)
                        for i in range(1, self.train_config.auc_number_thre):

                    # check return
                    if self.worker_finish_ratio(self.train_config.worker_num) > 0.8:
                        time.sleep(10)
                        logging.info("worker_finish end: %s" % str(len(set(self.finishedWorkers()))))
                        gstep = mon_sess.run(_get_or_create_global_step_read())
                        self.scaffold.saver.save(mon_sess._coordinated_creator.tf_sess,
                                                 self.train_config.save_dir + 'sm.ckpt', gstep)
                        self.export_model.end(mon_sess._coordinated_creator.tf_sess)
                        return
                else:  # training, slave worker
                    if localcnt % 50 == 0 and self.worker_finish_ratio(self.train_config.worker_num) > 0.8:
                        logging.info("worker_finish end: %s" % str(len(set(self.finishedWorkers()))))
                        return
                    feed_dict.update({self.is_training: True, self.use_tfrecord: False})
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

                if localcnt % self.train_config.auc_reset_step == 5:
                    self.report_tf_metric('total_auc', totalauc, ex_tag={"model_name": model.name})
                    logging.info("reset auc model:%s" % model.name)
                    logging.info("auc_reset_step:%s" % str(self.train_config.auc_reset_step))
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
                break

    def blinkPredict(self, mon_sess, task_id, thread_ind):
        raise NotImplementedError("cannot predict on blink")

    def paiPredict(self, mon_sess, task_id, thread_ind):
        if int(thread_ind) != 0:  # predict with one thread
            logging.info("Skip thread_ind==%s" % str(thread_ind))
            return

        training_config = self.train_config
        predict_step = training_config.max_step
        localcnt = 0
        while True:
            localcnt += 1
            # model, train_op, index, _, _ = co_trainer.choose_model()
            model, train_op, index, reset_auc_ops, localvar = self.choose_model(self.train_config.model_name)

            feed_dict = {'training:0': False}
            try:
                _ = mon_sess.run(model.write_to_table, feed_dict=feed_dict)
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

        try:
            model, train_op, index, _, _ = self.choose_model(self.train_config.model_name)
            feed_dict = {'training:0': False}
            _ = mon_sess.run(model.odpsWriterClose, feed_dict=feed_dict)
        except Exception as e:
            logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        logging.info("Finish Run, sleep")

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
            model, train_op, index, reset_auc_ops, localvar = self.choose_model(tf.flags.FLAGS.predict_model_name)

            feed_dict = {self.is_training: False, self.use_tfrecord: False}
            try:
                res = mon_sess.run(model.write_to_table, feed_dict=feed_dict)
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


def main(_):
    bootstrap = Train()
    bootstrap.start()


if __name__ == '__main__':
    tf.app.run()
    logging.info("bye...")
