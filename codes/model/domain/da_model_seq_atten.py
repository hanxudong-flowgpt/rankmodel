import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope

from model.domain.da_model import DaModel
from model.ops import base_ops
from model.ops import attention
from model.ops import utils


class DaModelSeqAtten(DaModel):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        super(DaModelSeqAtten, self).__init__(train_config, model_config, model_columns, model_name, model_idx)

        # Define model variables collection
        self.attention_collections_dnn_hidden_layer = "attention_dnn_hidden_layer"
        self.attention_collections_dnn_hidden_output = "attention_dnn_hidden_output"

    def user_seq_layer(self, user_input_layer, query_input_layer, user_seq_columns, fg, features, share_name=None):
        if len(user_seq_columns) == 0:
            return
        with tf.variable_scope(
                name_or_scope="seq_input_from_feature_columns",
                partitioner=base_ops.partitioner(self.config.ps_num, mem=8 * 1024 * 1024),
                reuse=tf.AUTO_REUSE) as scope:
            print(user_seq_columns)
            user_seq_layer = layers.input_from_feature_columns(
                fg.sequence_features(features, self.mc.sequence_columns, fg.seq_length, fg.sequence_name, False),
                user_seq_columns, scope=scope)
        print("start user seq attention")
        item_sequence_mask = tf.sequence_mask(tf.reshape(features["opt_seq_length"], [-1]), fg.sequence_length)
        with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.attention_l2_reg)):
            with tf.variable_scope(name_or_scope="%s-User-Layer" % (share_name if share_name is not None else "Share"),
                                   partitioner=base_ops.partitioner(self.config.ps_num, mem=64 * 1024),
                                   reuse=tf.AUTO_REUSE):

                items = tf.split(user_seq_layer, fg.seq_length, axis=0)  # [?, f_num*f_embedding]*30
                items_stack = tf.stack(values=items, axis=1)  # [?, 30, f_num*f_embedding] or [h*N, T_q, T_k]

                items_2d = tf.reshape(items_stack, [-1, tf.shape(items_stack)[2]])
                items_stack = tf.reshape(tf.where(tf.reshape(item_sequence_mask, [-1]),
                                                  items_2d, tf.zeros_like(items_2d)), tf.shape(items_stack))

                if self.config.sequence_item_combiner == 'concat':
                    items_stack = tf.reshape(items_stack,
                                                shape=[-1, items_stack.shape[1] * items_stack.shape[2]])  # (N,L*d)
                elif self.config.sequence_item_combiner == 'mean':
                    items_stack = tf.reduce_mean(items_stack, axis=1)  # (N, d)
                elif self.config.sequence_item_combiner == 'max':
                    items_stack = tf.reduce_max(items_stack, axis=1)  # (N, d)
                else:
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
                                                       variables_collections=[self.attention_collections_dnn_hidden_layer],
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
                                                   variables_collections=[self.attention_collections_dnn_hidden_layer],
                                                   outputs_collections=[self.attention_collections_dnn_hidden_output])

                    dec = tf.reshape(item_vec, [-1, self.config.ua_num_output_units])
                    items_stack = dec
                    #self.user_input_layer = tf.concat(values=[self.user_input_layer, self.items_stack], axis=1)
        return items_stack

    def query_seq_layer(self, queryseq_column, fg, features, share_name=None):
        with tf.variable_scope(
                name_or_scope="queryseq_input_from_feature_columns",
                partitioner=base_ops.partitioner(self.config.ps_num, mem=8 * 1024 * 1024),
                reuse=tf.AUTO_REUSE) as scope:

            queryseq_layer = layers.input_from_feature_columns(fg.sequence_features(features,
                                                                                    self.mc.queryseq_columns,
                                                                                    fg.queryseq_length,
                                                                                    fg.queryseq_name,
                                                                                    False),
                                                                    queryseq_column, scope=scope)

            queryseqs = tf.split(queryseq_layer, fg.queryseq_length, axis=0)
            queryseqs = tf.stack(values=queryseqs, axis=1)

            query_ids_dense = fg.query_id_features(features, length=fg.queryseq_length)  # (N, T)
            query_sequence_mask = tf.not_equal(query_ids_dense, "0")  # (N, T)
            query_sequence_mask_matmul = tf.tile(tf.expand_dims(tf.cast(query_sequence_mask, tf.float32), -1),
                                               [1, 1, tf.shape(queryseqs)[2]])  # (N, T, d)
            queryseqs *= query_sequence_mask_matmul  # (N, T, d)

        with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.attention_l2_reg)):
            with tf.variable_scope(name_or_scope="%s-Query-Layer" % (share_name if share_name is not None else "Share"),
                                   partitioner=base_ops.partitioner(self.config.ps_num, mem=64 * 1024),
                                   reuse=tf.AUTO_REUSE):

                if self.config.queryseq_combiner == "concat":
                    queryseq_layer = tf.reshape(queryseqs, shape=[-1, queryseqs.shape[1] * queryseqs.shape[2]])
                elif self.config.queryseq_combiner == "mean":
                    queryseq_layer = tf.reduce_mean(queryseqs, axis=1)
                elif self.config.queryseq_combiner == "sum":
                    queryseq_layer = tf.reduce_sum(queryseqs, axis=1)
                elif self.config.queryseq_combiner == "max":
                    queryseq_layer = tf.reduce_max(queryseqs, axis=1)
                elif self.config.queryseq_combiner == "CNN_max":
                    pooled_outputs = []
                    for size in self.config.cnn_filter_size:
                        outputs = tf.layers.conv1d(inputs=queryseqs
                                                   , filters=32
                                                   , kernel_size=size
                                                   , strides=1
                                                   , padding='same'
                                                   , activation=utils.getActivationFunctionOp(self.config.activation_op)
                                                   )
                        outputs = tf.layers.max_pooling1d(inputs=outputs
                                                          , pool_size=size
                                                          , strides=2
                                                          , padding='same')
                        pooled_outputs.append(outputs)

                    queryseq_layer = tf.concat(pooled_outputs, axis=1)
                    queryseq_layer = tf.reshape(queryseq_layer, [-1, queryseq_layer.shape[1] * queryseq_layer.shape[2]])

                tf.add_to_collection(self.collections_dnn_hidden_output, queryseq_layer)
                #query_input_layer = tf.concat(values=[query_input_layer, queryseq_layer], axis=1)
                #tf.add_to_collection(self.collections_dnn_hidden_output, self.query_input_layer)
        return queryseq_layer # self.query_input_layer
    def summary(self):
        super(DaModelSeqAtten, self).summary()
        with tf.name_scope("%s-Attention-Summary" % self.name):
          base_ops.add_norm2_summary(self.attention_collections_dnn_hidden_layer, summary_prefix="Attention_Norm2/")
          base_ops.add_dense_output_summary(self.attention_collections_dnn_hidden_output,
                                            summary_prefix="Attention_DenseOutput/")
    '''   
        if self.training_config.trace_prediction_bucket:
          self.summary_single()

    def summary_single(self):
        self.summary_op = []
        with tf.name_scope("%s-Prediction-Summary" % self.name):
          for i in range(len(self.training_config.buckets) - 1):
            self.summary_op.append(
              tf.summary.scalar(
                name="scalar_bucket_%s_%s/label_mean" % (
                  self.training_config.buckets[i], self.training_config.buckets[i + 1]),
                tensor=tf.reduce_mean(self.scalars[i])))
        return self.summary_op

    def loss_op(self, task_label=None, task_num=None):
        pass
    '''