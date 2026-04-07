import random

import tensorflow as tf
from tensorflow.python.ops import state_ops, array_ops
from model.ops import base_ops, metrics

from model.model import Model
from feature.feature_generator_seq import FeatureGeneratorSeq


class PvCxrModel(Model):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0, item_num_in_pv=10):
        super(PvCxrModel, self).__init__(train_config, model_config, model_columns, model_name, model_idx)

        # A batch of dict of feature-name & feature-value.
        self.fg = FeatureGeneratorSeq(train_config.currentPath + "/../" + train_config.fg_conf,
                                      model_config,
                                      model_columns,
                                      train_config,
                                      train_config.currentPath + "/../" + model_config.cross_path)

        # A float32 Tensor with shape [batch_size, 1].
        self.id = None
        self.common_features = None
        self.item_num_in_pv = item_num_in_pv
        # list of float32 Tensor with shape [batch_size, 1].
        self.feature_map_list = None
        self.label_list = None
        self.item_id_list = None
        self.item_mask_list = None
        self.weight_list = None
        self.position_list = None
        self.first_ui_input_layer = None
        self.first_position_logits = None

        # float32 Tensor with shape [batch_size * item_num, 1].
        self.feature_map = None
        self.ui_input_layer = None
        self.position_logits = None

        # A float32 Tensor with shape [batch_size * item_num, 1]. after sigmoid
        self.predictions = None
        self.label = None
        self.item_mask = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.loss = None
        self.reg_losses = None

        # A dict of metrics in graph.
        self.metrics = {}

        # Define model variables collection
        self.collections_dnn_hidden_layer = "%s-dnn_hidden_layer" % self.name
        self.collections_dnn_hidden_output = "%s-dnn_hidden_output" % self.name

    def build_inputs(self, generated_features_map):
        self.id = generated_features_map[self.config.table_name]["id"]
        self.common_features = generated_features_map[self.config.table_name]
        self.feature_map = generated_features_map[self.config.table_name]["feature_map"]

        self.label_list = generated_features_map[self.config.table_name]["label_list"]
        self.item_id_list = generated_features_map[self.config.table_name]["item_id_list"]
        self.weight_list = generated_features_map[self.config.table_name]["weight_list"]
        self.feature_map_list = generated_features_map[self.config.table_name]["feature_map_list"]
        self.position_list = generated_features_map[self.config.table_name]["position_list"]
        self.item_mask_list = generated_features_map[self.config.table_name]["item_mask_list"]
        if self.training_config.is_local:
            self.label_list = generated_features_map[self.config.table_name]["label_list"][:2]
            self.item_id_list = generated_features_map[self.config.table_name]["item_id_list"][:2]
            self.weight_list = generated_features_map[self.config.table_name]["weight_list"][:2]
            self.feature_map_list = generated_features_map[self.config.table_name]["feature_map_list"][:2]
            self.position_list = generated_features_map[self.config.table_name]["position_list"][:2]
            self.item_mask_list = generated_features_map[self.config.table_name]["item_mask_list"][:2]
        self.label = tf.concat(self.label_list, axis=0)
        self.item_mask = tf.concat(self.item_mask_list, axis=0)

    def build_model(self):
        pass

    def predictions_op(self):
        pass

    def summary(self):
        with tf.name_scope("%s-Metrics" % self.name):
            print('auc not local with local var')
            worker_device = "/job:worker/task:{}".format(self.training_config.task_id)
            with tf.device(worker_device):
                self.current_auc, self.total_auc = metrics.auc(
                    labels=self.label,
                    predictions=self.predictions,
                    weights=self.item_mask,
                    num_thresholds=2000)
            self.metrics['scalar/loss'] = self.loss
            self.metrics['scalar/reg_loss'] = self.reg_losses

            self.metrics['scalar/batch_auc'] = self.current_auc
            self.metrics['scalar/total_auc'] = self.total_auc


            eff_item_num = tf.reduce_sum(self.item_mask) + 1.0e-12

            self.metrics['scalar/eff_item_num'] = eff_item_num

            self.metrics['scalar/label_mean'] = tf.reduce_sum(self.label * self.item_mask) / eff_item_num
            self.metrics['scalar/predict_mean'] = tf.reduce_sum(self.predictions * self.item_mask) /eff_item_num

            self.metrics['scalar/prepossum'] = tf.reduce_sum(self.label * self.predictions * self.item_mask)
            self.metrics['scalar/prenegsum'] = tf.reduce_sum((1 - self.label) * self.predictions * self.item_mask)
            self.metrics['scalar/gtpossum'] = tf.reduce_sum(self.label * self.item_mask)
            self.metrics['scalar/gtnegsum'] = tf.reduce_sum(tf.cast(tf.equal(self.label, 0), tf.float32) * self.item_mask)
            self.metrics['scalar/mae'] = tf.reduce_sum(tf.abs(self.predictions - self.label) * self.item_mask) / eff_item_num
            self.metrics['scalar/mse'] = tf.reduce_sum((self.predictions - self.label) * self.item_mask *
                                                               (self.predictions - self.label)) / eff_item_num
            self.metrics['scalar/pcopc'] = \
                (self.metrics['scalar/prepossum'] + self.metrics['scalar/prenegsum']) / (
                            self.metrics['scalar/gtpossum'] + tf.constant(1e-16, shape=[]))
            self.metrics['scalar/posmean'] = \
                self.metrics['scalar/prepossum'] / (
                            self.metrics['scalar/gtpossum'] + tf.constant(1e-16, shape=[]))
            self.metrics['scalar/negmean'] = \
                self.metrics['scalar/prenegsum'] / (
                            self.metrics['scalar/gtnegsum'] + tf.constant(1e-16, shape=[]))

            for k, v in self.metrics.items():
                tf.summary.scalar(name=k, tensor=v)

            # self.metrics['scalar/deep_logits_mean'] = tf.reduce_mean(self.deep_logits)
        with tf.name_scope("%s-Summary" % self.name):
            base_ops.add_norm2_summary(self.collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.collections_dnn_hidden_output)
        '''
        with tf.name_scope('%s-Embedding' % self.name):
            base_ops.add_embed_layer_norm(self.user_input_layer, self.sourcefcs["user_column"])
            base_ops.add_embed_layer_norm(self.query_input_layer, self.query_column)
            base_ops.add_embed_layer_norm(self.item_input_layer, self.item_column)
            base_ops.add_embed_layer_norm(self.ui_input_layer, self.ui_column)
        '''
