import tensorflow as tf
from util.tflog import tflogger as logging
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
import model.ops.optimizer_ops as myopt
from model.ops import learning_rate_ops

from model.normal.seq_atten_model import SeqAttenModel
from model.module.gat_net import GraphAttenNet
from model.ops import checkpoint_utils
from model.ops import base_ops
from model.ops import rtp_ops
from model.ops import utils

init_assignment_map = {
    "seq_input_from_feature_columns/": "seq_input_from_feature_columns/"
    , "input_from_feature_columns/": "input_from_feature_columns/"
    #, "Share/": "Share/"
    #, "Share-User-Vec-Space/": "Share-User-Vec-Space/"
    #, "Recall_CTR/": "Recall_CTR/"
    , "Recall_CTR-Position-Network/": "Recall_CTR-Position-Network/"
    #, "Recall_CVR/": "Recall_CVR/"
    , "Recall_CVR-Position-Network/": "Recall_CVR-Position-Network/"
}


class KonduModel(SeqAttenModel):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        super(KonduModel, self).__init__(train_config, model_config, model_columns, model_name, model_idx)
        self.write_to_table = None

    def build_model(self):
        print("%s model build begin..." % self.name)
        self.cross_entropy_logits, self.position_logits = self.forward(self.features, self.fg)
        print("%s model build done..." % self.name)
        if self.training_config.is_restore_embedding and self.training_config.embedding_checkpoint_dir is not None:
            logging.info("[init variables] from dir: %s" % self.training_config.embedding_checkpoint_dir)
            checkpoint_utils.init_from_checkpoint(self.training_config.embedding_checkpoint_dir, init_assignment_map)
        self.final_logits = self.cross_entropy_logits # + self.position_logits

    def forward(self, features, fg):
        mc = self.mc
        user_column = fg.feature_columns_from_name(mc.user_columns)
        query_column = fg.feature_columns_from_name(mc.query_columns)
        item_column = fg.feature_columns_from_name(mc.item_columns)
        ui_column = fg.feature_columns_from_name(mc.ui_columns)
        i2i_column = fg.feature_columns_from_name(mc.i2i_sequence_columns)
        queryseq_column = fg.feature_columns_from_name(mc.queryseq_columns)

        user_seq_columns = fg.feature_columns_from_name(mc.sequence_columns)
        GatNet = GraphAttenNet(self.config, fg, self.training_config)
        '''
        title_column = fg.feature_columns_from_name(mc.title_columns)
        i2i_seq_layer = None
        seq_context_columns = fg.feature_columns_from_name(mc.seq_context_columns)
        seq_length_columns = fg.feature_columns_from_name(mc.seq_length_columns)
        seq_context_atten_columns = fg.feature_columns_from_name(mc.seq_context_atten_columns)
        query_atten_columns = fg.feature_columns_from_name(mc.query_atten_columns)
        user_atten_columns = fg.feature_columns_from_name(mc.user_atten_columns)
        personal_columns = fg.feature_columns_from_name(mc.personal_columns)
        '''
        if not (user_column and query_column and item_column):
            raise ValueError("user_column query_column item_column must be defined.")
        if not (features and isinstance(features, dict)):
            raise ValueError("features must be defined and must be a dict.")
        if not self.config.recall_dnn_hidden_units:
            raise ValueError("configuration recall_dnn_hidden_units must be defined.")

        with tf.variable_scope(
                name_or_scope="input_from_feature_columns",
                partitioner=base_ops.partitioner(self.config.ps_num,
                                                 mem=8 * 1024 * 1024), reuse=tf.AUTO_REUSE) as scope:
            logging.info("start user_input_layer...")
            user_input_layer = layers.input_from_feature_columns(features, user_column, scope=scope)
            logging.info("start query_input_layer...")
            # logging.info(query_column)
            query_input_layer = layers.input_from_feature_columns(features, query_column, scope=scope)
            logging.info("start item_input_layer...")
            # logging.info(item_column)
            item_input_layer = tf.feature_column.input_layer(features, item_column)
            if ui_column:
                logging.info("start ui_input_layer...")
                ui_input_layer = layers.input_from_feature_columns(features, ui_column, scope=scope)
            else:
                ui_input_layer = None

            if len(i2i_column) > 0:
                logging.info("start i2i_seq_layer...")
                i2i_seq_layer = GatNet.build_gat_net(features, i2i_column)
                user_input_layer = tf.concat(values=[user_input_layer, i2i_seq_layer], axis=1)

        if user_seq_columns:
            logging.info("start user_seq_layer...")
            if self.config.share_user_layer:
                items_stack = self.user_seq_layer(user_input_layer, query_input_layer, user_seq_columns, fg, features)
            else:
                items_stack = self.user_seq_layer(user_input_layer, query_input_layer, user_seq_columns, fg, features,
                                                  self.name)
            user_input_layer = tf.concat(values=[user_input_layer, items_stack], axis=1)

        if queryseq_column:
            logging.info("start query_seq_layer...")
            if self.config.share_user_layer:
                queryseq = self.query_seq_layer(query_column, fg, features)
            else:
                queryseq = self.query_seq_layer(query_column, fg, features, self.name)
            query_input_layer = tf.concat(values=[query_input_layer, queryseq], axis=1)
            # tf.add_to_collection(self.collections_dnn_hidden_output, query_input_layer)

        with tf.variable_scope(name_or_scope="%s-Param-Space" % self.name,
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):

            if len(query_column) > 0:
                user_input_layer = tf.concat([user_input_layer, query_input_layer], axis=1)
            input_vec = tf.concat([user_input_layer, item_input_layer], axis=1)
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                for layer_id, num_hidden_units in enumerate(self.config.dnn_hidden_units):
                    with variable_scope.variable_scope("hiddenlayer_%d" % layer_id) as hidden_layer_scope:
                        '''
                        item_vec = layers.fully_connected(
                            item_vec,
                            num_hidden_units,
                            utils.getActivationFunctionOp(self.config.activation_op),
                            scope=item_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True,
                                               "is_training": self.is_training}
                        )
                        '''
                        input_vec = layers.fully_connected(
                            input_vec,
                            num_hidden_units,
                            activation_fn=utils.getActivationFunctionOp(self.config.activation_op),
                            scope=hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=None, #layers.batch_norm,
                            normalizer_params=None #{"scale": True, "is_training": self.is_training}
                        )

                output_vec = layers.fully_connected(
                    input_vec,
                    1,
                    activation_fn=None,
                    scope="ouput_linear_layer",
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output],
                )

        with tf.variable_scope(name_or_scope="%s-Position-Network" % self.name,
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            if len(ui_column) > 0:
                user_input_layer = tf.concat([user_input_layer, ui_input_layer], axis=1)
            position_net = user_input_layer

            tf.add_to_collection(self.collections_dnn_hidden_output, position_net)
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                for layer_id, num_hidden_units in enumerate(self.config.position_dnn_hidden_units):
                    with variable_scope.variable_scope("position_hiddenlayer_%d" % layer_id) as dnn_hidden_layer_scope:
                        position_net = layers.fully_connected(
                            position_net,
                            num_hidden_units,
                            utils.getActivationFunctionOp(self.config.activation_op),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=None,  # layers.batch_norm,
                            normalizer_params=None  # {"scale": True, "is_training": self.is_training}
                        )
                        '''
                        position_net = rtp_ops.dropout_in_train(
                            position_net,
                            self.training_config.is_local,
                            keep_prob=self.config.fc_dropout_keep_prob,
                            is_training=self.is_training)
                        '''
        with tf.variable_scope(name_or_scope="%s-Logits" % self.name,
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE) as dnn_logits_scope:
            deep_logits = output_vec

            pos_logits = layers.linear(position_net, 1, variables_collections=[self.collections_dnn_hidden_layer],
                                       outputs_collections=[self.collections_dnn_hidden_output],
                                       biases_initializer=None, scope="pos-net"
                                       )
            # self.logits = self.deep_logits + self.pos_logits
        return deep_logits, pos_logits

    '''
    def build_training_inputs(self, model_name):

        features = self.build_training_inputs_hdfs(model_name, self.config.input_file_names_dynamic_train, self.config.input_file_names_dynamic_eval, None)
        self.features = features
        self.label = features['label']
        self.id = features['id']
        self.weight = features['weight']

        features = self.build_training_inputs_hdfs(model_name+"_da_", self.config.da_input_file_names_dynamic_train, self.config.da_input_file_names_dynamic_eval, None)
        self.da_features = features
        self.da_label = features['label']
        self.da_id = features['id']
        self.da_weight = features['weight']
    '''

    def build_local_inputs(self):
        pass
        # self.features, self.label = self.build_training_inputs_local(self.config.input_file_names)
        # self.da_features, self.da_label = self.build_training_inputs_local(self.config.input_file_names)

    def trace_op(self):
        """
        trace feature input and feature output
        :return:
        """
        # super(BaseModelRecall, self).trace_op()
        if self.training_config.platform == 'pai' and self.training_config.predict:
            self.odpsWriter = tf.TableRecordWriter(tf.flags.FLAGS.tableoutput, slice_id=self.training_config.task_id)
            self.odpsWriterClose = self.odpsWriter.close()
            predict_show = []
            predict_show.append(self.id)
            predict_show.append(tf.as_string(self.predictions))
            vstr = tf.as_string(self.s_ivec_norm if self.config.trace_ivec == "on" else tf.concat([self.s_uvec_norm, self.position_logits], axis=1))
            v = tf.reduce_join(vstr, 1, keep_dims=False, separator=',')
            predict_show.append(v)
            self.write_to_table = self.odpsWriter.write(indices=[0, 1, 2], values=predict_show)

    def loss_op(self, task_label=None, task_num=None):
        with tf.name_scope("Loss_Op"):
            if self.name == "Recall_weighted":
                self.celoss = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(
                        targets=self.label,
                        logits=self.cross_entropy_logits, # + self.position_logits,
                        pos_weight=self.weight))
            else:
                self.celoss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.cross_entropy_logits, # + self.position_logits,
                        labels=self.label))

            # Set up the training ops.
            self.reg_losses = self.reg_loss_op()
            self.loss = self.reg_losses + self.celoss
        return self.loss

    def summary(self):
        super(KonduModel, self).summary()
        with tf.name_scope("%s-Auxi-Loss" % self.name):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("cross_entropy_loss", self.celoss)
            tf.summary.scalar("reg_losses", self.reg_losses)
