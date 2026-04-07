import tensorflow as tf
from util.tflog import tflogger as logging
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope

from cxrmodel import CxrModel
from model.module.gat_net import GraphAttenNet
from model.ops import base_ops
from model.ops import rtp_ops
from model.ops import utils
from feature.input_from_fc import dense_repeat


class KonduModel(CxrModel):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        super(KonduModel, self).__init__(train_config, model_config, model_columns, model_name, model_idx)
        self.write_to_table = None
        self.predict_vector = None

    def build_model(self):
        print("%s model build begin..." % self.name)
        self.cross_entropy_logits, self.position_logits = self.forward(self.features, self.fg)
        print("%s model build done..." % self.name)
        self.final_logits = self.cross_entropy_logits # + self.position_logits

    def forward(self, features, fg):
        # i2i_column = fg.feature_columns_from_name(mc.i2i_sequence_columns)
        # queryseq_column = fg.feature_columns_from_name(mc.queryseq_columns)

        mc = self.mc
        user_column = mc.user_columns
        query_column = mc.query_columns
        item_column = mc.item_columns
        ui_column = mc.ui_columns
        cross_column = mc.cross_columns
        user_seq_columns = mc.sequence_columns
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

        user_input_layer = tf.concat([features[x] for x in user_column], axis=1)
        query_input_layer = tf.concat([features[x] for x in query_column], axis=1)
        item_input_layer = tf.concat([features[x] for x in item_column + cross_column], axis=1)

        if ui_column:
            logging.info("start ui_input_layer...")
            ui_input_layer = tf.concat([features[x] for x in ui_column], axis=1)
        else:
            ui_input_layer = None

        if user_seq_columns:
            logging.info("start user_seq_layer...")
            if self.config.share_user_layer:
                items_stack = self.user_seq_layer(user_input_layer, query_input_layer, user_seq_columns, fg, features)
            else:
                items_stack = self.user_seq_layer(user_input_layer, query_input_layer, user_seq_columns, fg, features,
                                                  self.name)
            user_input_layer = tf.concat(values=[user_input_layer, items_stack], axis=1)

        with tf.variable_scope(name_or_scope="%s-Param-Space" % self.name,
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):

            if len(query_column) > 0:
                user_input_layer = tf.concat([user_input_layer, query_input_layer], axis=1)
            batch_size = tf.shape(item_input_layer)[0]
            user_input_layer = tf.cond(self.training_config.share_context,
                                       lambda: dense_repeat(user_input_layer, batch_size),
                                       lambda: user_input_layer)
            self.user_input_layer = user_input_layer
            self.item_input_layer = item_input_layer
            inputs = [user_input_layer, item_input_layer]
            input_vec = tf.concat(inputs, axis=1)
            self.input_vec = input_vec
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
                            activation_fn=utils.getActivationFunctionOp(self.config.activation_op)
                            if layer_id+1 < len(self.config.dnn_hidden_units) else None,
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
        '''
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
        '''
        predict_show = []
        predict_show.append(self.id)
        predict_show.append(tf.as_string(self.label))
        '''
        for x in ["7", "gid", "goods_id"]:
            if x in self.features:
                predict_show.append(self.features[x])
        '''
        predict_show.append(tf.as_string(self.predictions))
        import os
        fname = os.path.join(tf.flags.FLAGS.tableoutput, "part_%d" % self.training_config.task_id)
        '''
        self.write_to_table = tf.io.write_file(fname, tf.reduce_join(tf.string_join(predict_show, "\t"),
                                                                     axis=None,
                                                                     keep_dims=False,
                                                                     separator="\n"))
        '''
        self.write_to_table = tf.reduce_join(tf.string_join(predict_show, "\t"),
                                                            axis=None,
                                                            keep_dims=False,
                                                            separator="\n")

    def loss_op(self, task_label=None, task_num=None):
        with tf.name_scope("Loss_Op"):
            label = tf.cast(tf.not_equal(self.label, 0.0), tf.float32)
            self.label = label
            if self.name == "Recall_weighted":
                self.celoss = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(
                        targets=self.label,
                        logits=self.final_logits,
                        pos_weight=self.weight))
            else:
                self.celoss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.final_logits,
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
