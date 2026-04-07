# if not local_mode:
#   from flink_tensorflow.python_sdk.blink_bootstrap import *
#   from model_io.utils import down_load_data
# else:
#   from local_run.local_simulation import *
from util.config import ModelConfig, TrainingConfig
from util.config import MultimodalConfig,SequenceModelConfig
#from config.tiny_model.tiny_model_config import TinyModelConfig

from local_run.util import *
from tensorflow.python.lib.io import file_io
from model_ops.tflog import tflogger as logging
from fg.feature_generator import FeatureGenerator
from fg.feature_generator_seq import FeatureGeneratorSeq
from model.model_column import ModelColumns
from model_ops import queue_ops as myqueue
import json


def getHdfsFileList(use_fm_to_hdfs, task_id, worker_num, table_project, table_name, train_parts, eval_parts):
    if use_fm_to_hdfs == False:
      from model_io.utils import down_load_data
      files_list = down_load_data(task_id,
                                  worker_num,
                                  table_name,
                                  train_parts,
                                  eval_parts,
                                  project=table_project)
    else:
      files_list = []
      '''
      files_list = down_load_data_fm(task_id,
                                     worker_num,
                                     training_config.sample_name,
                                     training_config.train_start_time,
                                     training_config.train_end_time,
                                     training_config.eva_start_time,
                                     training_config.eva_end_time,
                                     training_config.workflow_id,
                                     training_config.cluster_id)
      '''
    logging.info("[input_file_names] (len %s) : %s" % (len(files_list), files_list if len(files_list)<20 else files_list[:20]))

    # Dynamic
    from model_io.utils import down_load_data_dynamic
    files_list_dynamic_tain, files_list_dynamic_eval = down_load_data_dynamic(task_id,
                                                                              worker_num,
                                                                              table_name,
                                                                              train_parts,
                                                                              eval_parts,
                                                                              project=table_project)

    logging.info("[input_file_names_dynamic_train] (len %s) : %s" % (len(files_list_dynamic_tain), files_list_dynamic_tain if len(files_list_dynamic_tain)<20 else files_list_dynamic_tain[:20]))
    logging.info("[input_file_names_dynamic_eval] (len %s) : %s" % (len(files_list_dynamic_eval), files_list_dynamic_eval if len(files_list_dynamic_eval)<20 else files_list_dynamic_eval[:20]))
    return files_list, files_list_dynamic_tain, files_list_dynamic_eval

def getConfig(init_confs, context, task_id, currentPath, local_mode):
  # Context Params
  worker_num = int(context.get_worker_num())
  save_dir = context.get_ckp_dir()
  cluster = context.get_cluster()

  # Training Config
  training_config = TrainingConfig(task_id=task_id, cluster=cluster, worker_num=worker_num,
                                   is_local=local_mode, save_dir=save_dir)
  training_config.updateFromStrMap(init_confs)
  only_predict = training_config.predict
  if not training_config.need_save_ckp:
    training_config.save_checkpoint_secs = None
  if only_predict:
    training_config.train_parts = training_config.eval_parts
  logging.info('[training_config]:%s' % str(training_config))

  # Model Config
  if training_config.model_type.lower() == 'baseline':
    model_config = ModelConfig(ps_num=context.get_ps_num())
  elif training_config.model_type.lower() == 'baseline_seq' \
      or training_config.model_type.lower() == 'baseline_seqatt' \
      or training_config.model_type.lower() == 'baseline_seq_naive' \
      or training_config.model_type.lower() == 'baseline_adv' \
      or training_config.model_type.lower() == 'baseline_recall' \
      or training_config.model_type.lower() == 'baseline_recall_infer' \
      or training_config.model_type.lower() == 'kd_recall_dial' \
      or training_config.model_type.lower() == 'item_static_model' \
      or training_config.model_type.lower() == 'baseline_hierarchical_seqatten':
    model_config = SequenceModelConfig(ps_num=context.get_ps_num())
  elif training_config.model_type.lower() == 'tiny_train' \
      or training_config.model_type.lower() == 'tiny_predict':
    model_config = TinyModelConfig(ps_num=context.get_ps_num())
  elif training_config.model_type.lower() == 'baseline_seqmm':
    model_config = MultimodalConfig(ps_num=context.get_ps_num())
  else:
    model_config = ModelConfig(ps_num=context.get_ps_num())
    raise RuntimeError("Unknown model_type = %s" % str(training_config.model_type))

  model_config.updateFromStrMap(init_confs)
  if training_config.task_id in (0, 1):
    model_config.batch_size = int(model_config.batch_size * 0.7)

  # TODO hack: chief run faster than others when training
  if training_config.task_id == 0 and model_config.num_epochs is not None and not only_predict:
    model_config.num_epochs *= 100
  logging.info('[model_config]:%s' % str(model_config))


  # Model IO
  if training_config.platform == "blink":
    files_list, files_list_dynamic_tain, files_list_dynamic_eval = getHdfsFileList(training_config.use_fm_to_hdfs,
                                                                                    training_config.task_id,
                                                                                    training_config.worker_num,
                                                                                    training_config.table_project,
                                                                                    init_confs["table_name"],
                                                                                    training_config.train_parts,
                                                                                    training_config.eval_parts)
    model_config.input_file_names = files_list
    model_config.input_file_names_dynamic_train = files_list_dynamic_tain
    model_config.input_file_names_dynamic_eval = files_list_dynamic_eval
    if "da_table_name" in init_confs:
        files_list, files_list_dynamic_tain, files_list_dynamic_eval = getHdfsFileList(training_config.use_fm_to_hdfs,
                                                                                       training_config.task_id,
                                                                                       training_config.worker_num,
                                                                                       training_config.table_project,
                                                                                       init_confs["da_table_name"],
                                                                                       training_config.train_parts,
                                                                                       training_config.eval_parts)
        model_config.da_input_file_names = files_list
        model_config.da_input_file_names_dynamic_train = files_list_dynamic_tain
        model_config.da_input_file_names_dynamic_eval = files_list_dynamic_eval
  elif training_config.platform == "pai":
    files_list = ["odps://%s/tables/%s/ds=%s"%(training_config.table_project, init_confs["table_name"],
                                            init_confs["train_parts" if training_config.task_id!=0 else "eval_parts"])]
    model_config.input_file_names = files_list
    if "da_table_name" in init_confs:
        files_list = ["odps://%s/tables/%s/ds=%s"%(training_config.table_project, init_confs["da_table_name"],
                                            init_confs["train_parts" if training_config.task_id!=0 else "eval_parts"])]
        model_config.da_input_file_names = files_list
  else:
    files_list = ['../suffering/rank_model_debug/fake_data/search_offline.%s.txt' % training_config.table_name]
    logging.info("[local train files] : %s" % files_list)
    model_config.input_file_names = files_list


  # MC
  logging.info("[conf_path2] : %s" % (currentPath + "/config/" + init_confs['mcconf']))
  mc = ModelColumns(currentPath + "/config/" + init_confs['mcconf'])
  #mc = [mc, ModelColumns(currentPath + "/config/" + init_confs['mcconf'])] if "da_table_name" in init_confs else mc
  # FG
  conf_path = currentPath + "/config/" + init_confs['fgconf']
  logging.info("[conf_path] : %s" % conf_path)
  cross_path = None
  if 'cross_conf' in init_confs:
    cross_path = currentPath + "/config/" + init_confs["cross_conf"]
    logging.info("[cross_conf_path] : %s" % cross_path)

  if training_config.model_type.lower() == 'baseline':
    fg = FeatureGenerator(conf_path, model_config, mc, training_config, cross_path)
    fg = [fg, FeatureGeneratorSeq(conf_path, model_config, mc, training_config,
                                  cross_path)] if "da_table_name" in init_confs else fg
  elif training_config.model_type.lower() == 'baseline_seq' \
      or training_config.model_type.lower() == 'baseline_seqatt' \
      or training_config.model_type.lower() == 'baseline_seq_naive' \
      or training_config.model_type.lower() == 'baseline_adv' \
      or training_config.model_type.lower() == 'baseline_recall' \
      or training_config.model_type.lower() == 'baseline_recall_infer' \
      or training_config.model_type.lower() == 'kd_recall_dial' \
      or training_config.model_type.lower() == 'item_static_model' \
      or training_config.model_type.lower() == 'baseline_hierarchical_seqatten' \
      or training_config.model_type.lower() == 'tiny_train' \
      or training_config.model_type.lower() == 'tiny_predict' \
      or training_config.model_type.lower() == 'baseline_seqmm':
    fg = FeatureGeneratorSeq(conf_path, model_config, mc, training_config, cross_path)
    fg = [fg, FeatureGeneratorSeq(conf_path, model_config, mc, training_config,
                                  cross_path)] if "da_table_name" in init_confs else fg
  else:
    fg = FeatureGenerator(conf_path, model_config, mc, training_config, cross_path)
    raise RuntimeError("Unknown model_type = %s" % str(training_config.model_type))
  # Copy conf only when training
  if task_id == 0 and not only_predict:
    file_io.write_string_to_file(training_config.save_dir + '/fg.json',
                                 json.dumps(fg[0].feature_conf if type(fg) == list else fg.feature_conf, indent=2))
  return training_config, model_config, fg, mc, only_predict



def getModel(training_config, model_config, fg, mc, local_mode, task_id=0):
  # Dynamic Choosing Model
  if training_config.model_type.lower() == 'baseline':
    from model.base_model import BaseModel as UsingModel
  elif training_config.model_type.lower() == 'baseline_seq':
    from model.base_model_seq import BaseModelSeq as UsingModel
  elif training_config.model_type.lower() == 'baseline_seqatt':
    from model.base_model_seq_atten import BaseModelSeqAtten as UsingModel
  elif training_config.model_type.lower() == 'baseline_seq_naive':
    from model.base_model_seq_naive import BaseModelSeqNaive as UsingModel
  elif training_config.model_type.lower() == "baseline_adv":
    from model.base_model_adv import BaseModelAdv as UsingModel
  elif training_config.model_type.lower() == "baseline_recall":
    from model.base_model_recall import BaseModelRecall as UsingModel
  elif training_config.model_type.lower() == "baseline_recall_infer":
    from model.base_model_recall_infer import BaseModelRecallInfer as UsingModel
  elif training_config.model_type.lower() == 'kd_recall_dial':
    from model.domain.kd_model_recall_dial import KonduModel as UsingModel
  elif training_config.model_type.lower() == "item_static_model":
    from model.util.item_static_model import ItemStaticModel as UsingModel
  elif training_config.model_type.lower() == "baseline_hierarchical_seqatten":
    from model.base_model_opt_seq_atten import BaseModelOptSeqAtten as UsingModel
  elif training_config.model_type.lower() == 'tiny_train':
    from model.tiny_model import TinyModel as UsingModel
  elif training_config.model_type.lower() == 'tiny_predict':
    from model.tiny_model_extract import TinyModelExtract as UsingModel
  elif training_config.model_type.lower() == 'baseline_seqmm':
    from model.base_model_seq_multimodal import BaseModelSeqMultimodal as UsingModel
  else:
    from model.base_model import BaseModel as UsingModel
    raise RuntimeError("Unknown model_type= %s" % training_config.model_type.lower())

  # Local Run Model
  class LocalModel(UsingModel):
    def __init__(self,
                 model_config,
                 training_config,
                 fg,
                 mc,
                 name="CTR"):
      super(LocalModel, self).__init__(
        model_config,
        training_config,
        fg,
        mc,
        name
      )

    def build_inputs(self, model_name=''):
        self.build_training_inputs_local(self.config.input_file_names)
        self.da_features, self.da_label, self.da_id = self.features, self.label, self.id
        self.build_training_inputs_local(self.config.input_file_names)


    def build_training_inputs_local(self, input_file_names):
      with tf.name_scope("%s_Input_Pipeline" % self.name):
        if self.training_config.norepeat_read:
          filename_queue, norepeat_dependency = myqueue.string_input_producer_norepeat(self.config.input_file_names,
                                                                                       workernum=self.training_config.worker_num,
                                                                                       currentid=self.training_config.task_id,
                                                                                       localdevice='/cpu:0',
                                                                                       capacity=self.config.files_capacity,
                                                                                       num_epochs=self.config.num_epochs,
                                                                                       shuffle=False,
                                                                                       seed=None,
                                                                                       norepeat_state_min=self.training_config.norepeat_state_min)
          key, value = self.reader.read_up_to(filename_queue, self.config.batch_size)
          fields = [tf.decode_csv(value,
                                  record_defaults=[[" "]] * 4,
                                  field_delim='\t')
                    for _ in range(1)]
          with tf.control_dependencies(norepeat_dependency):
            self.local_batch_fields = tf.train.batch_join(fields,
                                                          batch_size=self.config.batch_size,
                                                          capacity=self.config.training_queue_capacity,
                                                          enqueue_many=True,
                                                          allow_smaller_final_batch=True)
        else:
          filename_queue = tf.train.string_input_producer(input_file_names,
                                                          shuffle=True,
                                                          capacity=self.config.files_capacity,
                                                          num_epochs=self.config.num_epochs)
          key, value = self.reader.read_up_to(filename_queue, self.config.batch_size)
          fields = [tf.decode_csv(value,
                                  record_defaults=[[" "]] * 4,
                                  field_delim='\t')
                    for _ in range(2)]
          self.local_batch_fields = tf.train.shuffle_batch_join(fields,
                                                                batch_size=self.config.batch_size,
                                                                capacity=self.config.training_queue_capacity,
                                                                enqueue_many=True,
                                                                allow_smaller_final_batch=True, min_after_dequeue=100)
        self.features = fc_to_input_placeholder(self.fg._feature_columns)
        if self.fg.__class__ == FeatureGeneratorSeq:
          self.features.update(fc_to_input_placeholder_seq(self.fg.seq_feature_columns, self.fg))
        self.label = self.features['label']
        self.id = self.features['id']

  if not local_mode:
    model = UsingModel(
      model_config=model_config,
      training_config=training_config,
      fg=fg,
      mc=mc,
      name=training_config.model_name,
      task_id=task_id
    )
  else:
    model = LocalModel(
      model_config=model_config,
      training_config=training_config,
      fg=fg,
      mc=mc,
      name=training_config.model_name
    )
  return model
