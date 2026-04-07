import random

import tensorflow as tf
from tensorflow.python.ops import state_ops, array_ops
from model.ops import base_ops, metrics
from util.tflog import tflogger as logging
from model.model import Model
from model.ops import utils
from model.ops import attention
from tensorflow.contrib.framework.python.ops import arg_scope


class CxrModel(Model):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        super(CxrModel, self).__init__(train_config, model_config, model_columns, model_name, model_idx)

        # A batch of dict of feature-name & feature-value.
        self.user_vec = None
        self.features = None
        self.s_ivec_norm = None
        self.t_ivec_norm = None
        '''
        self.fg = FeatureGeneratorSeq(train_config.currentPath + "/../" + train_config.fg_conf,
                                      model_config,
                                      model_columns,
                                      train_config,
                                      train_config.currentPath + "/../" + model_config.cross_path)
        '''
        # schema_json = json.load(open(train_config.currentPath + "/../" + model_config.lsc_conf))
        # self.fg = FeatureGenerator(train_config, schema_json, model_columns.simple_dict)
        self.fg = None
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

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.loss = None
        self.reg_losses = None

        # A dict of metrics in graph.
        self.metrics = {}

        # Define model variables collection
        self.collections_dnn_hidden_layer = "%s-dnn_hidden_layer" % self.name
        self.collections_dnn_hidden_output = "%s-dnn_hidden_output" % self.name
        self.attention_collections_dnn_hidden_layer = "%s-attention_dnn_hidden_layer" % self.name
        self.attention_collections_dnn_hidden_output = "%s-attention_dnn_hidden_output" % self.name

    def build_inputs(self, sample_data):
        # self.features = self.fg.parse_csv(sample_data)
        self.features = sample_data
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

    def user_seq_layer(self, user_input_layer, query_input_layer, user_seq_columns, features, seq_len, share_name=None):
        if len(user_seq_columns) == 0:
            return

        user_seq_embeddings = []
        item_sequence_mask = None
        for fname in user_seq_columns:
            padded = features[fname]
            logging.info("fname:" + fname + "  padshape:" + str(padded))
            embedding_array = tf.split(padded, seq_len, axis=0)  # [?, f_embedding]*seq_len
            # reshape_pad = tf.reshape(padded, [-1, seq_len, tf.(padded)[-1]])
            reshape_pad = tf.stack(values=embedding_array, axis=1)  # [?, seq_len, f_embedding] or [h*N, T_q, T_k]
            logging.info("fname:" + fname + "  padreshape:" + str(reshape_pad))
            user_seq_embeddings.append(reshape_pad)
            item_sequence_mask = features[fname+"_mask"]
        print("start user seq attention")
        # item_sequence_mask = tf.sequence_mask(tf.reshape(features["opt_seq_length"], [-1]), fg.sequence_length)
        with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.attention_l2_reg)):
            sc = "%s-User-Layer" % (share_name if share_name is not None else "Share")
            with tf.variable_scope(name_or_scope=sc,
                                   partitioner=base_ops.partitioner(self.config.ps_num, mem=64 * 1024),
                                   reuse=tf.AUTO_REUSE):

                items_stack = tf.concat(values=user_seq_embeddings, axis=2)  # [?, 30, f_num*f_embedding] or [h*N, T_q, T_k]

                if self.config.sequence_item_combiner == 'concat':
                    items_stack = tf.reshape(items_stack,
                                             shape=[-1, items_stack.shape[1] * items_stack.shape[2]])  # (N,L*d)
                elif self.config.sequence_item_combiner == 'mean':
                    items_stack = tf.reduce_mean(items_stack, axis=1)  # (N, d)
                elif self.config.sequence_item_combiner == 'max':
                    items_stack = tf.reduce_max(items_stack, axis=1)  # (N, d)
                elif self.config.sequence_item_combiner == 'transformer':
                    # self attention
                    s_item_vec, stt_vec = attention.multihead_attention(queries=items_stack,
                                                                        queries_length=None,
                                                                        keys=items_stack,
                                                                        num_units=self.config.sa_num_units,
                                                                        num_output_units=self.config.sa_num_output_units,
                                                                        activation_fn=utils.getActivationFunctionOp(
                                                                            self.config.activation_op),
                                                                        keep_prob=self.config.fc_dropout_keep_prob,
                                                                        keys_length=None,
                                                                        is_training=self.is_training,
                                                                        scope="self_attention",
                                                                        reuse=tf.AUTO_REUSE,
                                                                        query_masks=item_sequence_mask,
                                                                        key_masks=item_sequence_mask,
                                                                        variables_collections=[
                                                                            self.attention_collections_dnn_hidden_layer],
                                                                        outputs_collections=[
                                                                            self.attention_collections_dnn_hidden_output],
                                                                        num_heads=self.config.num_heads)

                    item_vec = attention.feedforward(s_item_vec,
                                                     num_units=[self.config.sa_num_output_units // 4,
                                                                self.config.sa_num_output_units],
                                                     activation_fn=utils.getActivationFunctionOp(
                                                         self.config.activation_op),
                                                     scope="sa_feed_forward",
                                                     reuse=tf.AUTO_REUSE,
                                                     variables_collections=[
                                                         self.attention_collections_dnn_hidden_layer],
                                                     outputs_collections=[self.attention_collections_dnn_hidden_output])

                    # user layer attention
                    att = tf.concat([user_input_layer, query_input_layer], axis=1)
                    att = tf.expand_dims(att, 1)
                    ua_item_vec, uatt_vec = attention.multihead_attention(queries=att,
                                                                          queries_length=tf.ones_like(att[:, 0, 0],
                                                                                                      dtype=tf.int32),
                                                                          keys=item_vec,
                                                                          keys_length=None,
                                                                          num_units=self.config.ua_num_units,
                                                                          num_output_units=self.config.ua_num_output_units,
                                                                          activation_fn=utils.getActivationFunctionOp(
                                                                              self.config.activation_op),
                                                                          keep_prob=self.config.fc_dropout_keep_prob,
                                                                          is_training=self.is_training,
                                                                          scope="user_layer_attention",
                                                                          reuse=tf.AUTO_REUSE,
                                                                          key_masks=item_sequence_mask,
                                                                          variables_collections=[
                                                                              self.attention_collections_dnn_hidden_layer],
                                                                          outputs_collections=[
                                                                              self.attention_collections_dnn_hidden_output],
                                                                          num_heads=self.config.num_heads)

                    item_vec = attention.feedforward(ua_item_vec,
                                                     num_units=[self.config.sa_num_output_units // 4,
                                                                self.config.ua_num_output_units],
                                                     activation_fn=utils.getActivationFunctionOp(
                                                         self.config.activation_op),
                                                     scope="ua_feed_forward",
                                                     reuse=tf.AUTO_REUSE,
                                                     variables_collections=[
                                                         self.attention_collections_dnn_hidden_layer],
                                                     outputs_collections=[self.attention_collections_dnn_hidden_output])

                    dec = tf.reshape(item_vec, [-1, self.config.ua_num_output_units])
                    items_stack = dec
                    # self.user_input_layer = tf.concat(values=[self.user_input_layer, self.items_stack], axis=1)
                else:
                    raise ValueError("cannot recognize self.config.sequence_item_combiner: " + str(self.config.sequence_item_combiner))
        return items_stack

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
