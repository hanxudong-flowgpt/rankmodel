import os,json,time,sys
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath+"/..")
from util.runtime.runtime_bootstrap import BootStrap
from model.model_column import ModelColumns, merge_model_column
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
import traceback
from feature.sample_io import *
import numpy as np
from model.normal.recall_cxr_lookup_model import KonduModel as Model
from util.config.seqconfig import SequenceModelConfig
from feature import parse_column
from util.tflog import tflogger as logging
from model.ops.utils import reset_variables


class Train(BootStrap):
    def __init__(self):
        super(Train, self).__init__()
        self.models = []
        self.probs = []
        self.merged = None
        self.is_training = None
        self.oss_summary_writer = None
        self.hdfs_summary_writer = None
        self.reset_auc_ops = []
        self.auc_local_vars = []
        self.global_step = None
        self.hooks = []

    def mockBlinkRun(self, target, data_stream, task_id, context):
        context.set_param(
            'model_type=baseline,fgconf=lsc_conf_local.json,mcconf=mc_conf_local.json,train_parts=20180227|20180227,eval_parts=20180227,optimizer=Adagrad,initial_learning_rate=0.01,lrcs_init_lr=0.001,lrcs_init_step=1000000,dnn_hidden_units=32,enable_file_dynamic=1')
        self.stepRun(target, data_stream, task_id, context)

    def buildGraph(self, task_id, context):

        logging.info("******begin********")
        logging.info("run with platform=%s" % str(tf.flags.FLAGS.platform))

        # init config
        self.is_training = tf.placeholder(tf.bool, name="training")
        self.train_config = self.parse_config(context.get_param(), platform=tf.flags.FLAGS.platform,
                                              cluster=context.get_cluster(), task_id=task_id,
                                              is_local=(tf.flags.FLAGS.platform == 'local'),
                                              worker_num=context.get_worker_num(), ps_num=context.get_ps_num(),
                                              currentPath=currentPath)

        # prepare the features
        origin_features = {}
        for i in range(len(self.train_config.table_name)):
            if tf.flags.FLAGS.platform == "blink":
                files_list, files_list_dynamic_tain, files_list_dynamic_eval = getHdfsFileList(
                    self.train_config.use_fm_to_hdfs, self.train_config.task_id, self.train_config.worker_num,
                    self.train_config.table_project[i], self.train_config.table_name[i], self.train_config.train_parts,
                    self.train_config.eval_parts)
                origin_feature, fqlist = build_hdfs_inputs_dynamic(self.train_config.table_name[i],
                                                                   self.train_config.predict,
                                                                   files_list_dynamic_tain,
                                                                   files_list_dynamic_eval,
                                                                   self.train_config.task_id,
                                                                   self.train_config.worker_num,
                                                                   self.train_config.num_epochs,
                                                                   self.train_config.num_reader,
                                                                   self.train_config.batch_size)
                for q in fqlist:
                    q.add_summary(self.train_config.save_dir)
                    self.hooks.append(q)
                key = "%s/%s" % (self.train_config.table_project[i], self.train_config.table_name[i])
                origin_features[key] = origin_feature
            elif tf.flags.FLAGS.platform in ("local", "pai") and self.train_config.ps_num > 1:
                parts = self.train_config.train_parts if self.train_config.task_id > 0 else self.train_config.eval_parts
                fn = ["odps://%s/tables/%s/ds=%s" % (self.train_config.table_project[i], self.train_config.table_name[i]
                                                     , part) for part in parts]
                origin_feature = build_odps_inputs(fn, self.train_config.task_id, self.train_config.worker_num,
                                                   self.train_config.num_reader, self.train_config.files_capacity,
                                                   self.train_config.num_epochs, self.train_config.batch_size)
                key = "%s/%s" % (self.train_config.table_project[i], self.train_config.table_name[i])
                origin_features[key] = origin_feature
            elif tf.flags.FLAGS.platform == "local":
                fn = "../model_debug/fake_data/search_offline/lsc_log_wide_wdlfg_cvr"
                #fn2 = "../model_debug/fake_data/search_offline/lsc_log_wide_wdlfg_cvr"
                # self.train_config.table_name[i]
                origin_feature = build_hdfs_inputs([fn], self.train_config.task_id, self.train_config.worker_num,
                                                   self.train_config.files_capacity, self.train_config.num_epochs,
                                                   self.train_config.num_reader, self.train_config.batch_size)
                key = "%s/%s" % (self.train_config.table_project[i], self.train_config.table_name[i])
                origin_features[key] = origin_feature

        # test
        '''
        with open(self.train_config.currentPath + "/../jsonfile/fg.json") as f:
            fgjson = json.load(f)
        generated_features = {}
        for k,v in origin_features.items():
            batch_data_list = tf.train.shuffle_batch_join(v,
                                                          batch_size=self.train_config.batch_size,
                                                          capacity=self.train_config.training_queue_capacity,
                                                          enqueue_many=True,
                                                          allow_smaller_final_batch=True, min_after_dequeue=0)
            import rtp_fg
            batch_data_list[1] = tf.Print(batch_data_list[1], ["here2"])
            features = rtp_fg.parse_genreated_fg(fgjson, batch_data_list[1])
            print("fg succ")
            print features
            features['label'] = tf.reshape(tf.string_to_number(batch_data_list[2], out_type=tf.float32), [-1, 1])
            features['id'] = tf.reshape(batch_data_list[0], [-1, 1])
            features['weight'] = tf.reshape(batch_data_list[3], [-1, 1])
            generated_features[k] = features
        '''

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
        allmc = []

        def getmcset(mc):
            logging.info(mc.all_columns_in_need)
            return mc.all_columns_in_need

        def getseqmcset(mc):
            return set(mc.sequence_columns + mc.seq_context_atten_columns + mc.queryseq_columns)

        for modelconf in jobconf["sources"]:
            idx += 1
            model_config = SequenceModelConfig(ps_num=context.get_ps_num())
            model_config.updateFromStrMap(modelconf)
            mc = ModelColumns(self.train_config.currentPath + "/../" + modelconf["mc_conf"],
                              tf.flags.FLAGS.platform == "local")
            mc_filter = getmcset(mc)
            seq_filter = getseqmcset(mc)
            with open(self.train_config.currentPath + "/../" + self.train_config.fg_conf) as f:
                fgjson = json.load(f)
                fgconf_list, fgconf_map = parse_column.parse_feature_conf(fgjson, mc_filter, seq_filter,
                                                                          tf.flags.FLAGS.platform == "local")
            model_config.fgconf_list, model_config.fgconf_map = fgconf_list, fgconf_map
            self.models.append(Model(self.train_config, model_config, mc, modelconf["model_name"], idx))
            self.probs.append(float(modelconf["prob"]))
            allmc.append(mc)
            logging.info("model %s init done!" % modelconf["model_name"])
        self.probs = [x / np.sum(self.probs) for x in self.probs]

        # feature generate
        mc = reduce(lambda a, y: merge_model_column(a, y, tf.flags.FLAGS.platform == "local"), allmc)
        mc_filter = getmcset(mc)
        seq_filter = getseqmcset(mc)
        with open(self.train_config.currentPath + "/../" + self.train_config.fg_conf) as f:
            fgjson = json.load(f)
            fgconf_list, fgconf_map = parse_column.parse_feature_conf(fgjson, mc_filter, seq_filter,
                                                                      tf.flags.FLAGS.platform == "local")
            logging.info("debug %s" % self.train_config.save_dir + 'fg.json')
            if self.train_config.task_id == 0:
                # json.dump(fgconf_list, file(self.train_config.save_dir + 'fg.json', 'w'), indent=2)
                writeGFile = tf.gfile.GFile(self.train_config.save_dir + 'fg.json', mode='w')
                writeGFile.write(json.dumps(fgconf_list, indent=2))
                writeGFile.flush()
                writeGFile.close()
        generated_features = {}
        for k, v in origin_features.items():
            logging.info("fg for %s begin ..." % k)
            batch_data_list = tf.train.shuffle_batch_join(v,
                                                          batch_size=self.train_config.batch_size,
                                                          capacity=self.train_config.training_queue_capacity,
                                                          enqueue_many=True,
                                                          allow_smaller_final_batch=True, min_after_dequeue=0)
            import rtp_fg
            features = rtp_fg.parse_genreated_fg(fgconf_list, batch_data_list[1])
            logging.info("fg succ, %d" % len(features))
            features['label'] = tf.reshape(tf.string_to_number(batch_data_list[2], out_type=tf.float32), [-1, 1])
            features['id'] = tf.reshape(batch_data_list[0], [-1, 1])
            features['weight'] = tf.reshape(batch_data_list[3], [-1, 1])
            features = tf.staged(features, capacity=4, num_threads=4)
            generated_features[k] = features
            logging.info("fg for %s done ..." % k)

        # Build Graph
        for model in self.models:
            model.build_main_part(generated_features, self.is_training)
            logging.info("model %s built done!" % model.name)
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
        tfconf.gpu_options.allow_growth = True

        if self.train_config.platform != "local":
            try:
                tracing_hook = tf.train.TracingHook(tf.flags.FLAGS.debug_dir + "trace/", 1000, min_vtrace_level=1,
                                                    is_global=False)
                self.hooks.append(tracing_hook)
            except Exception as e:
                tf.logging.info('Got Hook error, use no tracehook')
                tf.logging.info(str(e.__str__()))
                tf.logging.info('**************************')

            mon_sess = tf.train.MonitoredTrainingSession(
                master=target,
                is_chief=(task_id == 0),
                checkpoint_dir=self.train_config.save_dir,
                save_checkpoint_secs=self.train_config.save_checkpoint_secs if self.train_config.need_save_ckp else None,
                save_checkpoint_sync=False,
                config=tfconf,
                save_summaries_steps=None,
                save_summaries_secs=None,
                hooks=self.hooks,
                chief_only_hooks=None)
        else:
            mon_sess = tf.train.MonitoredTrainingSession(
                master=target,
                is_chief=(task_id == 0),
                checkpoint_dir=self.train_config.save_dir,
                save_checkpoint_secs=self.train_config.save_checkpoint_secs if self.train_config.need_save_ckp else None,
                config=tfconf,
                save_summaries_steps=None,
                save_summaries_secs=None,
                hooks=None,
                chief_only_hooks=None)
        if task_id == 0:
            if tf.flags.FLAGS.platform == "pai":
                self.oss_summary_writer = tf.summary.FileWriter(tf.flags.FLAGS.debug_dir + "summary/eval/",
                                                                mon_sess.graph)
            if tf.flags.FLAGS.platform == "blink":
                self.hdfs_summary_writer = tf.summary.FileWriter(self.train_config.save_dir + "summary/eval/",
                                                                 mon_sess.graph)
        elif task_id == 1:
            if tf.flags.FLAGS.platform == "pai":
                self.oss_summary_writer = tf.summary.FileWriter(tf.flags.FLAGS.debug_dir + "summary/train/",
                                                                mon_sess.graph)
            if tf.flags.FLAGS.platform == "blink":
                self.hdfs_summary_writer = tf.summary.FileWriter(self.train_config.save_dir + "summary/train/",
                                                                 mon_sess.graph)
        return mon_sess

    def runSession(self, mon_sess, task_id, thread_ind):
        if self.train_config.predict:
            if self.train_config.platform == "blink":
                self.blinkPredict(mon_sess, task_id, thread_ind)
            elif self.train_config.platform == "pai":
                self.paiPredict(mon_sess, task_id, thread_ind)
            else:
                return
        else:
            self.runTrain(mon_sess, task_id, thread_ind)
            if task_id in (0, 1) and tf.flags.FLAGS.platform == "pai":
                self.oss_summary_writer.close()
            if task_id in (0, 1) and tf.flags.FLAGS.platform == "blink":
                self.hdfs_summary_writer.close()
                # time.sleep(60 * 60 * 10)  # 10 hours

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
        while not mon_sess.should_stop():
            localcnt += 1
            model, train_op, index, reset_auc_ops, localvar = self.choose_model()
            taskcnt[index] += 1
            feed_dict = {}
            try:
                if self.worker_finish_ratio(self.train_config.worker_num) > 0.8:
                    if task_id == 0 or task_id == 1:
                        if len(self.finishedWorkers()) >= self.train_config.worker_num - 2:
                            return
                    else:
                        return
                if task_id == 0 or task_id == 1:
                    # eval, chief worker
                    feed_dict.update({'training:0': False})
                    co_global_step, global_step, loss, metrics, flocalv, labels = mon_sess.run(
                        [self.global_step, model.global_step, model.loss, model.metrics, localvar, model.label],
                        feed_dict=feed_dict)

                    auc, totalauc = metrics['scalar/batch_auc'], metrics['scalar/total_auc']
                    logging.info(
                        '%s-Col_Global_Step:%s, Global_Step:%s, poslabel:%s, loss=%s, auc=%s, totalauc=%s task_id=%s thread=%s' % (
                            model.model_name,
                            str(co_global_step),
                            str(global_step),
                            str(labels.sum()),
                            str(loss), str(auc),
                            str(totalauc), str(task_id),
                            str(thread_ind)))

                else:  # training, slave worker
                    feed_dict.update({'training:0': True})
                    co_global_step, global_step, _, loss, auc, totalauc, flocalv = mon_sess.run(
                        [self.global_step, model.global_step, train_op, model.loss,
                         model.metrics['scalar/batch_auc'], model.metrics['scalar/total_auc'], localvar],
                        feed_dict=feed_dict)
                    logging.info('%s-Col_Global_Step:%s, Global_Step:%s, loss=%s, totalauc=%s task_id=%s thread=%s' % (
                        model.model_name, str(co_global_step), str(global_step), str(loss),
                        str(totalauc), str(task_id), str(thread_ind)))

                # TODO summary write
                if taskcnt[index] % self.train_config.save_summaries_steps == 0:
                    if tf.flags.FLAGS.platform == "pai" and task_id in (0, 1):
                        summ = mon_sess.run(self.merged, feed_dict=feed_dict)
                        self.oss_summary_writer.add_summary(summ, co_global_step)
                    elif tf.flags.FLAGS.platform == "blink" and task_id in (0, 1):
                        summ = mon_sess.run(self.merged, feed_dict=feed_dict)
                        self.hdfs_summary_writer.add_summary(summ, co_global_step)
                # TODO reset all model's auc is not proper
                # newmark = np.max(flocalv[0][np.array([0, -1])])

                if global_step % self.train_config.auc_reset_step == 5:
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
                break  # release all
            except Exception as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

    def blinkPredict(self, mon_sess, task_id, thread_ind):
        if int(thread_ind) != 0:  # predict with one thread
            logging.info("Skip thread_ind==%s" % str(thread_ind))
            return

        training_config = self.train_config
        co_trainer = self.co_trainer
        dummy_feed_dict = {'training:0': False}
        self.tablewriter = myodps.getTableWriter(self.odps,
                                                 self.train_config.table_name + self.train_config.predict_postfix,
                                                 task_id=task_id,
                                                 ds_output='|'.join(self.train_config.eval_parts),
                                                 local_mode=local_mode)

        predict_step = training_config.max_step
        localcnt = 0
        while True:
            localcnt += 1
            model, train_op, index, _, _ = co_trainer.choose_model()
            feed_dict = {'training:0': False}

            try:
                if not training_config.trace_tensors:  # predict without trace tensors, default predict
                    id, prob, y, metrics, cogs, gs, cos, bias = mon_sess.run(
                        [model.id, model.predictions, model.label, model.metrics, co_trainer.global_step_add,
                         model.global_step_add, model.deep_logits, model.pos_logits],
                        feed_dict=feed_dict)
                    # cos = np.squeeze(cos)
                    records = [
                        [str(id[x][0]), str(prob[x][0]) + "_" + str(cos[x][0]) + "_" + str(bias[x][0]), str(y[x][0])]
                        for x in range(len(id))]
                    self.tablewriter.write(task_id, records)

                    logging.info(
                        'size=%s totalauc=%s; step_left=%s' % (
                        str(len(id)), str(metrics['scalar/total_auc']), str(predict_step)))
                else:  # predict with trace tensors, when verify

                    if training_config.trace_tensors_valid:
                        o = []
                        for i in range(len(tensor_list)):
                            o.append(str(i))
                        id, prob, y, cogs, gs, traces, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7] = mon_sess.run(
                            [model.id, model.predictions, model.label, co_trainer.global_step_add,
                             model.global_step_add, model.trace,
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[0]),
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[1]),
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[2]),
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[3]),
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[4]),
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[5]),
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[6]),
                             tf.get_default_graph().get_tensor_by_name(
                                 tensor_list[7]),
                             ],
                            feed_dict=feed_dict)
                        for i in range(len(tensor_list)):
                            output = [[
                                str(id[j][0]), "tensor_" + str(i), ','.join([str(x) for x in o[i][j, :]])] for j in
                                range(len(id))]
                            self.tablewriter.write(task_id, output)
                    else:
                        id, prob, y, cogs, gs, traces, deep_logits = mon_sess.run(
                            [model.source_id, model.source_predictions, model.source_label, co_trainer.global_step_add,
                             model.global_step_add, model.trace, model.source_logits],
                            feed_dict=feed_dict)
                    for kk, vv in traces.items():
                        if vv[0].__class__.__name__ == 'SparseTensorValue':
                            traceinfo = [[
                                str(id[i][0]) + "_trace_" + str(kk), str(prob[i][0]),
                                ','.join([str(x) for x in vv[1][i, :]])] for i in
                                range(len(id))]
                            self.tablewriter.write(task_id, traceinfo)
                    logging.info('size=%s step_left=%s' % (str(len(id)), str(predict_step)))

                predict_step -= 1
                if predict_step < 1:
                    break
            except (ResourceExhaustedError, OutOfRangeError) as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                # succ = co_trainer.release_model(index)
                # logging.info("Release Model index : %s" % index)
                # logging.info("Release Model index : %s %s" % (index, model.model_name))
                # if not succ:
                #   break
                # always break
                break
            except ConnectionError as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                logging.info("Reset table writer")
                self.odps = myodps.resetOdpsTable(training_config.table_name + training_config.predict_postfix,
                                                  task_id=task_id,
                                                  local_mode=local_mode, odps_user=training_config.odps_user)
                self.tablewriter = myodps.getTableWriter(self.odps,
                                                         training_config.table_name + training_config.predict_postfix,
                                                         task_id=task_id,
                                                         ds_output='|'.join(training_config.eval_parts),
                                                         local_mode=local_mode)
            except Exception as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

        notclose = True
        while notclose:
            try:
                if self.tablewriter is not None:
                    self.tablewriter.close()
                notclose = False
            except ConnectionError as e:
                logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                logging.info("Reset table writer when close")
                self.odps = myodps.resetOdpsTable(training_config.table_name + training_config.predict_postfix,
                                                  task_id=task_id,
                                                  local_mode=local_mode, odps_user=training_config.odps_user)
                self.tablewriter = myodps.getTableWriter(self.odps,
                                                         training_config.table_name + training_config.predict_postfix,
                                                         task_id=task_id,
                                                         ds_output='|'.join(training_config.eval_parts),
                                                         local_mode=local_mode)
                if int(task_id) != 0:
                    raise RuntimeError("Reset table writer when close")

        while str(task_id) == '0' and self.worker_finish_ratio(self.train_config.worker_num) < 0.99:
            logging.info("wait finish %s" % str(len(set(self.finishedWorkers()))))
            time.sleep(60)
        logging.info("Finish Run, sleep")

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
        if task_id == 0:
            time.sleep(600)
        try:
            model, train_op, index, _, _ = self.choose_model(self.train_config.model_name)
            feed_dict = {'training:0': False}
            _ = mon_sess.run(model.odpsWriterClose, feed_dict=feed_dict)
        except Exception as e:
            logging.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
        logging.info("Finish Run, sleep")


def main(_):
    bootstrap = Train()
    bootstrap.start()


if __name__ == '__main__':
    tf.app.run()

