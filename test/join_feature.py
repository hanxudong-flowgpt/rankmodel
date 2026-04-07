import os,json,time,sys
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath+"/..")
from util.runtime.runtime_bootstrap import BootStrap
from model.model_column import ModelColumns, merge_model_column
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
import traceback
from feature.sample_io import *
import numpy as np
# from model.pvsample.recall_cxr_pv_model_fast import KonduModel as Model
from util.config.seqconfig import SequenceModelConfig
from feature import parse_column, feature_type
from tensorflow.python.lib.io import file_io
from util.tflog import tflogger as logging
from model.ops.utils import reset_variables
from model.module.features_table_nohook import FeatureTable
from tensorflow.python.ops import control_flow_ops


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
        self.saver = None

    def mockBlinkRun(self, target, data_stream, task_id, context):
        context.set_param(
            'model_type=baseline,fgconf=lsc_conf_local.json,mcconf=mc_conf_local.json,train_parts=20180227|20180227,eval_parts=20180227,optimizer=Adagrad,initial_learning_rate=0.01,lrcs_init_lr=0.001,lrcs_init_step=1000000,dnn_hidden_units=32,enable_file_dynamic=1')
        self.stepRun(target, data_stream, task_id, context)

    '''
    def _sample_itemandfeature(self, feature_table, query, table_name):
        feature_names = []
        for line in open(self.train_config.currentPath + "/../jsonfile/featurejoin/" + table_name):
            feature_names.append(line.strip())
        itemlist = feature_table.join_feature('query2item', query, 100, ',')
        feature_map_list = []
        for itemid in itemlist:
            feature_map = {}
            feature = feature_table.join_feature('itemfeature', tf.string_to_number(itemid, tf.int64), len(feature_names), ',')
            for i in range(len(feature)):
                feature_map[feature_names[i]] = feature[i]
            feature_map_list.append(feature_map)
        return itemlist, feature_map_list
    '''

    def _sample_itemfeature(self, feature_table, itemlist, table_name, fgconf, mode):
        feature_names = []
        for x in json.load(open(self.train_config.currentPath + "/../jsonfile/featurejoin/" + table_name + ".json")):
            feature_names.append(x["column_name"])
        for x in fgconf.keys():
            if x not in feature_names and "expression" in fgconf[x] and "user:" not in fgconf[x]["expression"]:
                raise Exception("[feature join] %s is need by fg but not found in feature table" % x)

        feature_map_list = []
        item_mask_list = []
        for itemid in itemlist:
            feature_map = {}
            feature = feature_table.join_feature('itemfeature_'+mode, tf.string_to_number(itemid, tf.int64), len(feature_names), "\x02")
            item_mask_list.append(tf.reshape(tf.cast(tf.not_equal(feature, ''), tf.float32), [-1, 1]))
            feature = tf.decode_csv(feature, [['']] * len(feature_names), field_delim="\x02", name='decode_item_feature')
            for i in range(len(feature_names)):
                if feature_names[i] in fgconf:
                    feature_map[feature_names[i]] = feature[i]
                    '''
                    if fgconf[feature_names[i]]["value_type"] == "String":
                        feature_map[feature_names[i]] = tf.string_split(feature[i], "\x03")
                    elif fgconf[feature_names[i]]["value_type"] == "Double":
                        feature_map[feature_names[i]] = tf.string_to_number(feature[i], tf.float32)
                    '''
            feature_map_list.append(feature_map)
        return feature_map_list, item_mask_list

    def _get_itemfeature(self, feature_table, itemlist, table_name, mode):
        feature_names = []
        for x in json.load(open(self.train_config.currentPath + "/../jsonfile/featurejoin/" + table_name + ".json")):
            feature_names.append(x["column_name"])

        feature_map_list = []
        item_mask_list = []
        for itemid in itemlist:
            feature_map = {}
            feature = feature_table.join_feature('itemfeature_' + mode, tf.string_to_number(itemid, tf.int64),
                                                 len(feature_names), "\x02")
            item_mask_list.append(tf.reshape(tf.cast(tf.not_equal(feature, ''), tf.float32), [-1, 1]))
            feature = tf.decode_csv(feature, [['']] * len(feature_names), field_delim="\x02",
                                    name='decode_item_feature')
            for i in range(len(feature_names)):
                feature_map[feature_names[i]] = feature[i]
            feature_map_list.append(feature_map)
        return feature_map_list, item_mask_list

    def convert_feature_map(self, fea, fgconf_map):
        feature_list_map = {}
        for features in fea:
            for k, v in features.items():
                if k not in feature_list_map:
                    feature_list_map[k] = [v]
                else:
                    feature_list_map[k].append(v)
        feature_map = {}
        for k, v in feature_list_map.items():
            xx = tf.concat(v, axis=0)
            if fgconf_map[k]["value_type"] == "String":
                feature_map[k] = tf.string_split(xx, "\x03")
            elif fgconf_map[k]["value_type"] == "Double":
                feature_map[k] = tf.string_to_number(xx, tf.float32)
        return feature_map

    def buildGraph(self, task_id, context):
        logging.info("******begin********")
        logging.info("run with platform=%s task_id:%d" % (str(tf.flags.FLAGS.platform), task_id))

        # init config
        self.is_training = tf.placeholder(tf.bool, name="training")
        self.train_config = self.parse_config(context.get_param(), platform=tf.flags.FLAGS.platform,
                                              cluster=context.get_cluster(), task_id=task_id,
                                              is_local=(tf.flags.FLAGS.platform == 'local'),
                                              worker_num=context.get_worker_num(), ps_num=context.get_ps_num(),
                                              currentPath=currentPath)

        print(str(self.train_config))
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

        # prepare the samples
        origin_features = {}
        self.train_config.num_epochs = self.train_config.num_epochs * 3 \
            if self.train_config.task_id in (0, 1) and not self.train_config.predict else self.train_config.num_epochs
        default = [[""]] * 8
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
                                                                   self.train_config.batch_size, default)
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
                                                   self.train_config.num_epochs, self.train_config.batch_size, default)
                key = "%s/%s" % (self.train_config.table_project[i], self.train_config.table_name[i])
                origin_features[key] = origin_feature
            elif tf.flags.FLAGS.platform == "local":
                fn = "../model_debug/fake_data/search_offline/lsc_log_wide_wdlfg_ctr_pv_recall"
                # fn2 = "../model_debug/fake_data/search_offline/lsc_log_wide_wdlfg_cvr"
                # self.train_config.table_name[i]
                origin_feature = build_hdfs_inputs([fn], self.train_config.task_id, self.train_config.worker_num,
                                                   self.train_config.files_capacity, self.train_config.num_epochs,
                                                   self.train_config.num_reader, self.train_config.batch_size, default)
                key = "%s/%s" % (self.train_config.table_project[i], self.train_config.table_name[i])
                origin_features[key] = origin_feature

        # feature table
        if self.train_config.platform in ("pai", "local") and self.train_config.ps_num > 1:
            input_tables_conf = {'itemfeature_train': ["odps://search_offline/tables/target_item_feature/ds=%s"
                                                       % self.train_config.join_feature_parts[0],
                                                       'item_id,features', [tf.constant(0, tf.int64), ""]],
                                 'itemfeature_eval': ["odps://search_offline/tables/target_item_feature/ds=%s"
                                                      % self.train_config.eval_join_feature_parts[0],
                                                      'item_id,features', [tf.constant(0, tf.int64), ""]]}
        elif self.train_config.platform == "blink":
            input_tables_conf = {'itemfeature_train': [
                                     getHdfsFileList(self.train_config.use_fm_to_hdfs, self.train_config.task_id,
                                                     self.train_config.worker_num, "search_offline",
                                                     "target_item_feature",
                                                     self.train_config.join_feature_parts,
                                                     self.train_config.join_feature_parts)[0],
                                     [tf.constant(0, tf.int64), ""]],
                                 'itemfeature_eval': [
                                    getHdfsFileList(self.train_config.use_fm_to_hdfs, self.train_config.task_id,
                                                    self.train_config.worker_num, "search_offline",
                                                    "target_item_feature",
                                                    self.train_config.eval_join_feature_parts,
                                                    self.train_config.eval_join_feature_parts)[0],
                                    [tf.constant(0, tf.int64), ""]]}
        else:
            input_tables_conf = {'itemfeature_train': ["../model_debug/fake_data/search_offline/kd_item_features_fj_0",
                                                 [tf.constant(0, tf.int64), ""]],
                                 'itemfeature_eval': ["../model_debug/fake_data/search_offline/kd_item_features_fj_0",
                                                       [tf.constant(0, tf.int64), ""]]}
        with tf.variable_scope("feature_table"):
            feature_table = FeatureTable(input_tables_conf,
                                         server_count=self.train_config.ps_num,
                                         worker_index=self.train_config.task_id,
                                         worker_count=self.train_config.worker_num,
                                         check_any_finish=self.any_worker_finished)
        self.feature_table = feature_table
        # self.hooks.append(feature_table.ready_hook(check_wait_secs=15))

        def get_feature_list(features):
            fealist = []
            fealist.append(features["id"])
            fealist.append()
            return fealist
        final_features = {}
        for k, v in origin_features.items():
            features = {}
            logging.info("fg for %s begin ..." % k)
            batch_data_list = tf.train.shuffle_batch_join(v,
                                                          batch_size=self.train_config.batch_size,
                                                          capacity=self.train_config.training_queue_capacity,
                                                          enqueue_many=True,
                                                          allow_smaller_final_batch=True, min_after_dequeue=0)

            features["id"] = batch_data_list[0]
            features["pv_feature"] = batch_data_list[1]
            xxlist = tf.decode_csv(batch_data_list[7], record_defaults=[[""]] * 10, field_delim=",")
            position_list = [tf.decode_csv(xx, record_defaults=[[""]] * 3, field_delim="_") for xx in xxlist]
            features['position_list'] = [{"ui_page": x[1], "ui_pos": x[2]} for x in position_list]

            # feature join
            itemlist = tf.decode_csv(batch_data_list[4], record_defaults=[[""]] * 10, field_delim=",")
            # itemlist = tf.string_to_number(itemlist, tf.int64)
            print(itemlist)
            tablename = "kd_item_features_fj" if self.train_config.is_local else "target_item_feature"
            fea, item_mask_list = self._get_itemfeature(feature_table, itemlist, tablename,
                                                           "train" if self.train_config.task_id > 0 else "eval")
            for i in range(len(fea)):
                x = fea[i]
                x["ui_page"] = position_list[i][1]
                x["ui_pos"] = position_list[i][2]
            features['feature_map_list'] = fea
            # features['feature_map'] = self.convert_feature_map(fea, fgconf_map)

            #features['item_list'] = itemlist
            features['item_id_list'] = itemlist
            features['item_mask_list'] = item_mask_list
            features['weight_list'] = tf.decode_csv(batch_data_list[3], record_defaults=[[""]]*10, field_delim=",")
            xxlist = tf.decode_csv(batch_data_list[2], record_defaults=[[""]] * 10, field_delim=",")
            label_list = [tf.decode_csv(xx, record_defaults=[[""]] * 5, field_delim="_")[2] for xx in xxlist]
            features['label_list'] = [tf.reshape(tf.string_to_number(x, out_type=tf.float32), [-1, 1]) for x in label_list]
            features = tf.staged(features, capacity=4, num_threads=4)
            final_features[k] = get_feature_list(features)
            logging.info("fj for %s done ..." % k)
        self.final_features = final_features
        # return

        # Print All Variables to Debug
        all_local_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        for n in all_local_var:
            logging.info("all_local_var:%s" % str(n))
        all_global_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for n in all_global_var:
            logging.info("all_global_var:%s" % str(n))


        savevar = []
        self.initops = []
        allv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        for v in allv:
            if "feature_table" in v.name:
                logging.info("do not save : %s" % str(v))
                if "worker" in v.name:
                    #tf.train.Saver.restore_implicitly(v)
                    logging.info("restore_implicitly : %s" % str(v))
                    # self.initops.append(v.initializer)
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
        tfconf.gpu_options.allow_growth = True

        if self.train_config.platform != "local" or self.train_config.ps_num > 1:
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
                save_checkpoint_secs=self.train_config.save_checkpoint_secs,
                config=tfconf,
                save_summaries_steps=None,
                save_summaries_secs=None,
                hooks=self.hooks,
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
        # mon_sess.run(control_flow_ops.group(*self.initops))
        # mon_sess.run(self.feature_table.ready_hook().after_create_session())
        if not self.any_worker_finished():
            self.feature_table.run_load_and_wait4ready(mon_sess, check_wait_secs=20)
        if self.train_config.predict:
            if self.train_config.platform == "blink":
                self.blinkPredict(mon_sess, task_id, thread_ind)
            elif self.train_config.platform == "pai":
                self.paiPredict(mon_sess, task_id, thread_ind)
            else:
                return
        else:
            # self.runTrain(mon_sess, task_id, thread_ind)
            self.run_fj_train(mon_sess, task_id, thread_ind)
            if task_id in (0, 1) and tf.flags.FLAGS.platform == "pai":
                self.oss_summary_writer.close()
            if task_id in (0, 1) and tf.flags.FLAGS.platform == "blink":
                self.hdfs_summary_writer.close()

    def run_fj_train(self, mon_sess, task_id, thread_ind):
        localcnt = 0
        taskcnt = [0, 0]
        while not mon_sess.should_stop():
            localcnt += 1
            # model, train_op, index, reset_auc_ops, localvar = self.choose_model()
            # taskcnt[index] += 1
            if localcnt > 5:
                break
            try:
                for k,v in self.generated_features.items():
                    logging.info(k)
                    logging.info(mon_sess.run({"samplekey_query": v['samplekey_query'],
                                               "sampled_item_list": v['sampled_item_list'][0],
                                               "sampled_feature_map_list": v['sampled_feature_map_list'][0]}))

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
        time.sleep(600)

    def choose_model(self, model_name=None):
        choice_idx = 0
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
        logging.info("start training...")
        while not mon_sess.should_stop():
            localcnt += 1
            model, train_op, index, reset_auc_ops, localvar = self.choose_model(self.train_config.model_name if self.train_config.model_name!="None" else None)
            taskcnt[index] += 1
            feed_dict = {}
            try:
                mon_sess.run(self.feature_table.get_set_op())
                if self.worker_finish_ratio(self.train_config.worker_num) > 0.8:
                    logging.info("worker_finish end: %s" % str(len(set(self.finishedWorkers()))))
                    if task_id == 0 or task_id == 1:
                        if len(self.finishedWorkers()) >= self.train_config.worker_num - 2:
                            return
                    else:
                        return
                if task_id == 0 or task_id == 1:
                    # training, chief worker
                    feed_dict.update({'training:0': False})
                    co_global_step, global_step, loss, metrics, flocalv = mon_sess.run(
                        [self.global_step, model.global_step, model.loss, model.metrics, localvar],
                        feed_dict=feed_dict)

                    auc, totalauc = metrics['scalar/batch_auc'], metrics['scalar/total_auc']
                    logging.info(
                        '%s-Col_Global_Step:%s, Global_Step:%s, poslabel:%s, loss=%s, auc=%s, totalauc=%s task_id=%s thread=%s' % (
                            model.model_name,
                            str(co_global_step),
                            str(global_step),
                            str(metrics['scalar/gtpossum']),
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

                if localcnt % self.train_config.auc_reset_step == 5:
                    self.report_tf_metric('total_auc', totalauc, ex_tag={"model_name": model.name})
                    logging.info("reset auc model:%s" % model.name)
                    logging.info("auc_reset_step:%s" % str(self.train_config.auc_reset_step))
                    logging.info('reset auc ops run')
                    mon_sess.run(reset_auc_ops, feed_dict=feed_dict)

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
            mon_sess.run(self.feature_table.get_set_op())
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


def main(_):
    bootstrap = Train()
    bootstrap.start()


if __name__ == '__main__':
    tf.app.run()

