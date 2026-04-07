import random

import tensorflow as tf
from tensorflow.python.ops import state_ops, array_ops
from model.ops import base_ops, metrics

from model.model import Model
from feature.feature_generator_seq import FeatureGeneratorSeq
from feature.pdd_feature_generator import FeatureGenerator
import json


class CxrModelxxx(Model):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        super(CxrModel, self).__init__(train_config, model_config, model_columns, model_name, model_idx)

        # A batch of dict of feature-name & feature-value.
        self.user_vec = None
        self.features = None
        '''
        self.fg = FeatureGeneratorSeq(train_config.currentPath + "/../" + train_config.fg_conf,
                                      model_config,
                                      model_columns,
                                      train_config,
                                      train_config.currentPath + "/../" + model_config.cross_path)
        '''
        schema_json = json.load(open(train_config.currentPath + "/../" + model_config.lsc_conf))
        self.fg = FeatureGenerator(train_config, schema_json, model_columns.simple_dict)
        # A float32 Tensor with shape [batch_size, 1].
        self.label = None
        self.id = None
        self.weight = None
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
        self.position_logits = None
        self.cross_entropy_logits = None
        self.final_logits = None

        # A float32 Tensor with shape [batch_size, 1]. after sigmoid
        self.predictions = None
        self.serving_input = None
        self.context_serving_input = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.loss = None
        self.reg_losses = None

        # A dict of metrics in graph.
        self.metrics = {}

        # Define model variables collection
        self.collections_dnn_hidden_layer = "%s-dnn_hidden_layer" % self.name
        self.collections_dnn_hidden_output = "%s-dnn_hidden_output" % self.name

    def build_inputs(self, sample_data):
        # self.features = self.fg.parse_csv(sample_data)
        self.serving_input = sample_data[0]
        self.context_serving_input = sample_data[1]
        '''
        self.features = tf.cond(self.training_config.use_tfrecord,
                                lambda: tf.cond(self.training_config.share_context,
                                            lambda: self.fg.parse_share_context(self.context_serving_input, self.document_serving_input),
                                            lambda: self.fg.parse_example(self.serving_input)),
                                lambda: self.fg.parse_csv(self.serving_input))
        '''
        #self.features = tf.cond(self.training_config.use_tfrecord,
        #                        lambda: self.fg.parse_share_context(self.context_serving_input, self.serving_input),
        #                        lambda: self.fg.parse_csv(self.serving_input))
        self.label = self.features['positive']
        self.id = self.features['__id']
        '''
        self.features = generated_features_map[self.config.table_name]
        self.label = generated_features_map[self.config.table_name]["label"]
        self.id = generated_features_map[self.config.table_name]["id"]
        self.weight = generated_features_map[self.config.table_name]["weight"]
        '''

    def build_model(self):
        pass

    def predictions_op(self):
        self.predictions = tf.sigmoid(self.final_logits)
        # self.out4rtp["user_vec"] = tf.reduce_mean(self.user_vec, axis=1) + self.position_logits
        self.out4rtp["score"] = self.predictions

    def summary(self):
        with tf.name_scope("%s-Metrics" % self.name):
            print('auc not local with local var')
            worker_device = "/job:worker/task:{}".format(self.training_config.task_id)
            with tf.device(worker_device):
                self.current_auc, self.total_auc = metrics.auc(
                    labels=self.label,
                    predictions=self.predictions,
                    num_thresholds=self.training_config.auc_number_thre)
            self.metrics['scalar/loss'] = self.loss
            self.metrics['scalar/reg_loss'] = self.reg_losses

            self.metrics['scalar/batch_auc'] = self.current_auc
            self.metrics['scalar/total_auc'] = self.total_auc

            self.metrics['scalar/label_mean'] = tf.reduce_mean(self.label)
            self.metrics['scalar/predict_mean'] = tf.reduce_mean(self.predictions)

            self.metrics['scalar/prepossum'] = tf.reduce_sum(self.label * self.predictions)
            self.metrics['scalar/prenegsum'] = tf.reduce_sum((1 - self.label) * self.predictions)
            self.metrics['scalar/gtpossum'] = tf.reduce_sum(self.label)
            self.metrics['scalar/gtnegsum'] = tf.reduce_sum(tf.cast(tf.equal(self.label, 0), tf.float32))
            self.metrics['scalar/mae'] = tf.reduce_mean(tf.abs(self.predictions - self.label))
            self.metrics['scalar/mse'] = tf.reduce_mean((self.predictions - self.label) *
                                                               (self.predictions - self.label))
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
        '''
        with tf.name_scope("%s-Summary" % self.name):
            base_ops.add_norm2_summary(self.collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.collections_dnn_hidden_output)
        
        with tf.name_scope('%s-Embedding' % self.name):
            base_ops.add_embed_layer_norm(self.user_input_layer, self.sourcefcs["user_column"])
            base_ops.add_embed_layer_norm(self.query_input_layer, self.query_column)
            base_ops.add_embed_layer_norm(self.item_input_layer, self.item_column)
            base_ops.add_embed_layer_norm(self.ui_input_layer, self.ui_column)
        '''
