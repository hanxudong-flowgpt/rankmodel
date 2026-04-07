import random

import tensorflow as tf
from tensorflow.python.ops import state_ops, array_ops
from model.ops import base_ops, metrics

from model.model import Model
from feature.feature_generator_seq import FeatureGeneratorSeq


class DaModel(Model):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        super(DaModel, self).__init__(train_config, model_config, model_columns, model_name, model_idx)

        # A batch of dict of feature-name & feature-value.
        self.source_features = None
        self.target_features = None
        self.common_features = None

        self.fg = FeatureGeneratorSeq(train_config.currentPath + "/../" + train_config.fg_conf,
                                      model_config,
                                      model_columns,
                                      train_config,
                                      train_config.currentPath + "/../" + model_config.cross_path)

        # A float32 Tensor with shape [batch_size, 1].
        self.source_label = None
        # self.target_label = None
        self.source_id = None
        self.source_weight = None
        self.target_id = None
        self.target_weight = None
        '''
        # A float32 Tensor with shape [batch_size, N-dim].
        self.user_input_layer = None

        # A float32 Tensor with shape [batch_size, N-dim].
        self.query_input_layer = None

        # A float32 Tensor with shape [batch_size, N-dim].
        self.item_input_layer = None

        # A float32 Tensor with shape [batch_size, N-dim].
        self.ui_input_layer = None
        '''
        # A float32 Tensor with shape [batch_size, 1].
        self.s_pos_logits = None
        # self.t_pos_logits = None
        self.source_cross_entropy_logits = None
        self.target_cross_entropy_logits = None
        self.final_logit_source = None
        self.final_logit_target = None

        # A float32 Tensor with shape [batch_size, 1]. after sigmoid
        self.source_predictions = None
        self.target_predictions = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.source_loss = None
        self.target_loss = None
        self.reg_losses = None
        self.loss = None

        # A dict of metrics in graph.
        self.metrics = {}

        # Define model variables collection
        self.collections_dnn_hidden_layer = "%s-dnn_hidden_layer" % self.name
        self.collections_dnn_hidden_output = "%s-dnn_hidden_output" % self.name

    def build_inputs(self, generated_features_map):
        self.common_features = generated_features_map[self.config.table_name]

        if "feature_map_list" not in generated_features_map[self.config.table_name]:
            self.source_features = [generated_features_map[self.config.table_name]]
            self.source_label = [generated_features_map[self.config.table_name]["label"]]
            self.source_id = [generated_features_map[self.config.table_name]["id"]]
            self.source_weight = [generated_features_map[self.config.table_name]["weight"]]
        else:
            self.source_features = generated_features_map[self.config.table_name]["feature_map_list"]
            self.source_label = generated_features_map[self.config.table_name]["label_list"]
            self.source_id = generated_features_map[self.config.table_name]["id_list"]
            self.source_weight = generated_features_map[self.config.table_name]["weight_list"]

        if self.config.table_name != self.config.da_table_name:
            self.target_features = [generated_features_map[self.config.da_table_name]]
            self.target_label = [generated_features_map[self.config.da_table_name]["label"]]
            self.target_id = [generated_features_map[self.config.da_table_name]["id"]]
            self.target_weight = [generated_features_map[self.config.da_table_name]["weight"]]
        else:
            self.target_features = generated_features_map[self.config.table_name]["sampled_feature_map_list"]
            # x = generated_features_map[self.config.table_name]["label"]
            # self.target_label = [array_ops.zeros(shape=x.get_shape(), dtype=x.dtype)] * len(self.target_features)
            # self.target_id = [generated_features_map[self.config.table_name]["id"]] * len(self.target_features)
            # self.target_weight = [generated_features_map[self.config.table_name]["weight"]] * len(self.target_features)

    def build_model(self):
        pass

    def predictions_op(self):
        self.source_predictions = [tf.sigmoid(x) for x in self.final_logit_source]
        self.target_predictions = [tf.sigmoid(x) for x in self.final_logit_target]
        self.out4rtp["source_predictions"] = self.source_predictions[0]
        # self.out4rtp["target_predictions"] = self.target_predictions

    def summary(self):
        xx = [self.source_label, self.source_predictions, self.target_label, self.target_predictions]
        with tf.name_scope("%s-Metrics" % self.name):
            self.source_label = tf.concat(self.source_label, axis=0)
            self.source_predictions = tf.concat(self.source_predictions, axis=0)
            self.target_label = tf.concat(self.target_label, axis=0)
            self.target_predictions = tf.concat(self.target_predictions, axis=0)

            print('auc not local with local var')
            worker_device = "/job:worker/task:{}".format(self.training_config.task_id)
            with tf.device(worker_device):
                self.source_current_auc, self.source_total_auc = metrics.auc(
                    labels=self.source_label,
                    predictions=self.source_predictions,
                    num_thresholds=2000)
                self.target_current_auc, self.target_total_auc = metrics.auc(
                    labels=self.target_label,
                    predictions=self.target_predictions,
                    num_thresholds=2000)

            self.metrics['scalar/loss'] = self.loss
            self.metrics['scalar/reg_loss'] = self.reg_losses

            self.metrics['scalar/source/batch_auc'] = self.source_current_auc
            self.metrics['scalar/source/total_auc'] = self.source_total_auc
            self.metrics['scalar/target/batch_auc'] = self.target_current_auc
            self.metrics['scalar/target/total_auc'] = self.target_total_auc

            self.metrics['scalar/source/label_mean'] = tf.reduce_mean(self.source_label)
            self.metrics['scalar/target/label_mean'] = tf.reduce_mean(self.target_label)
            self.metrics['scalar/source/predict_mean'] = tf.reduce_mean(self.source_predictions)
            self.metrics['scalar/target/predict_mean'] = tf.reduce_mean(self.target_predictions)

            self.metrics['scalar/source/prepossum'] = tf.reduce_sum(self.source_label * self.source_predictions)
            self.metrics['scalar/source/prenegsum'] = tf.reduce_sum((1 - self.source_label) * self.source_predictions)
            self.metrics['scalar/source/gtpossum'] = tf.reduce_sum(self.source_label)
            self.metrics['scalar/source/gtnegsum'] = tf.reduce_sum(tf.cast(tf.equal(self.source_label, 0), tf.float32))
            self.metrics['scalar/source/mae'] = tf.reduce_mean(tf.abs(self.source_predictions - self.source_label))
            self.metrics['scalar/source/mse'] = tf.reduce_mean((self.source_predictions - self.source_label) *
                                                               (self.source_predictions - self.source_label))
            self.metrics['scalar/source/pcopc'] = \
                (self.metrics['scalar/source/prepossum'] + self.metrics['scalar/source/prenegsum']) / (
                            self.metrics['scalar/source/gtpossum'] + tf.constant(1e-16, shape=[]))
            self.metrics['scalar/source/posmean'] = \
                self.metrics['scalar/source/prepossum'] / (
                            self.metrics['scalar/source/gtpossum'] + tf.constant(1e-16, shape=[]))
            self.metrics['scalar/source/negmean'] = \
                self.metrics['scalar/source/prenegsum'] / (
                            self.metrics['scalar/source/gtnegsum'] + tf.constant(1e-16, shape=[]))

            self.metrics['scalar/target/prepossum'] = tf.reduce_sum(self.target_label * self.target_predictions)
            self.metrics['scalar/target/prenegsum'] = tf.reduce_sum((1 - self.target_label) * self.target_predictions)
            self.metrics['scalar/target/gtpossum'] = tf.reduce_sum(self.target_label)
            self.metrics['scalar/target/gtnegsum'] = tf.reduce_sum(tf.cast(tf.equal(self.target_label, 0), tf.float32))
            self.metrics['scalar/target/mae'] = tf.reduce_mean(tf.abs(self.target_predictions - self.target_label))
            self.metrics['scalar/target/mse'] = tf.reduce_mean((self.target_predictions - self.target_label) *
                                                               (self.target_predictions - self.target_label))
            self.metrics['scalar/target/pcopc'] = \
                (self.metrics['scalar/target/prepossum'] + self.metrics['scalar/target/prenegsum']) / (
                            self.metrics['scalar/target/gtpossum'] + tf.constant(1e-16, shape=[]))
            self.metrics['scalar/target/posmean'] = \
                self.metrics['scalar/target/prepossum'] / (
                            self.metrics['scalar/target/gtpossum'] + tf.constant(1e-16, shape=[]))
            self.metrics['scalar/target/negmean'] = \
                self.metrics['scalar/target/prenegsum'] / (
                            self.metrics['scalar/target/gtnegsum'] + tf.constant(1e-16, shape=[]))

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
        self.source_label, self.source_predictions, self.target_label, self.target_predictions = xx
