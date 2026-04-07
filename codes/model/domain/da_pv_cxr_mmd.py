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

from model.domain.da_model_seq_atten import DaModelSeqAtten
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


class KonduModel(DaModelSeqAtten):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        super(KonduModel, self).__init__(train_config, model_config, model_columns, model_name, model_idx)
        self.write_to_table = None
        self.cross_entropy_logits = None
        self.ivec_norm = None
        self.final_logits = None
        self.item_vecs_map = None

    def training_op(self):
        global_step = training_util.get_global_step()
        # Set up the learning rate + decay tensor
        learning_rate_decay_fn = None
        learning_rate = tf.constant(self.training_config.initial_learning_rate)
        if self.training_config.lrwr_using:
            learning_rate_decay_fn = lambda lr, gs: learning_rate_ops.lr_warm_restart(lr, gs,
                                                                                self.training_config.lrwr_init_lr_min,
                                                                                self.training_config.lrwr_periodic_step)
        elif self.training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (self.training_config.num_examples_per_epoch / self.config.batch_size)
            decay_steps = int(num_batches_per_epoch * self.training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                      learning_rate,
                      global_step,
                      decay_steps=decay_steps,
                      decay_rate=self.training_config.learning_rate_decay_factor,
                      staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = lambda lr, gs: lr

        if self.training_config.lrcs_using:
            learning_rate_decay_cs = lambda lr, gs: learning_rate_ops.lr_cold_start(learning_rate_decay_fn(lr, gs), gs,
                                                                              self.training_config.lrcs_init_lr,
                                                                              self.training_config.lrcs_init_step)
        else:
            learning_rate_decay_cs = learning_rate_decay_fn

        with tf.variable_scope(
                            name_or_scope="Optimize",
                            partitioner=base_ops.partitioner(self.config.ps_num,
                                                             mem=self.training_config.embedding_partitino_size),
                            reuse=tf.AUTO_REUSE):
            train_op_vec = []
            loss, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=self.loss,
                    global_step=self.global_step,
                    learning_rate=learning_rate,
                    optimizer=utils.getOptimizer(self.training_config, global_step=global_step,
                                                 learning_rate=learning_rate,
                                                 learning_rate_decay_fn=learning_rate_decay_cs),
                    clip_gradients=self.training_config.clip_gradients,
                    variables=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES),
                    learning_rate_decay_fn=learning_rate_decay_cs,
                    increment_global_step=False,
                    summaries=["learning_rate", "loss"],#myopt.OPTIMIZER_SUMMARIES,
                  )
            train_op_vec.append(loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_op_vec)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_op = state_ops.assign_add(self.global_step, 1).op

    def build_model(self):
        user_column = self.fg.feature_columns_from_name(self.mc.user_columns)
        query_column = self.fg.feature_columns_from_name(self.mc.query_columns)
        item_column = self.fg.feature_columns_from_name(self.mc.item_columns)
        ui_column = self.fg.feature_columns_from_name(self.mc.ui_columns)
        user_seq_columns = self.fg.feature_columns_from_name(self.mc.sequence_columns)
        queryseq_column = self.fg.feature_columns_from_name(self.mc.queryseq_columns)

        with tf.variable_scope(
                name_or_scope="input_from_feature_columns",
                partitioner=base_ops.partitioner(self.config.ps_num,
                                                 mem=8 * 1024 * 1024), reuse=tf.AUTO_REUSE) as scope:
            logging.info("start user_input_layer...")
            user_input_layer = layers.input_from_feature_columns(self.common_features, user_column, scope=scope)
            logging.info("start query_input_layer...")
            if len(query_column) > 0:
                query_input_layer = layers.input_from_feature_columns(self.common_features, query_column, scope=scope)
            else:
                query_input_layer = None
            logging.info("start item_input_layer...")
            # logging.info(item_column)
            item_input_layer = layers.input_from_feature_columns(self.feature_map, item_column, scope=scope)
            if ui_column:
                logging.info("start ui_input_layer...")
                ui_input_layer = layers.input_from_feature_columns(self.feature_map, ui_column, scope=scope)
            else:
                ui_input_layer = None

        if user_seq_columns:
            logging.info("start user_seq_layer...")
            if self.config.share_user_layer:
                items_stack = self.user_seq_layer(user_input_layer, query_input_layer, user_seq_columns, self.fg,
                                                  self.common_features)
            else:
                items_stack = self.user_seq_layer(user_input_layer, query_input_layer, user_seq_columns, self.fg,
                                                  self.common_features, self.name)
            user_input_layer = tf.concat(values=[user_input_layer, items_stack], axis=1)

        if queryseq_column:
            logging.info("start query_seq_layer...")
            if self.config.share_user_layer:
                queryseq = self.query_seq_layer(query_column, self.fg, self.common_features)
            else:
                queryseq = self.query_seq_layer(query_column, self.fg, self.common_features, self.name)
            query_input_layer = tf.concat(values=[query_input_layer, queryseq], axis=1)

        print("%s source model build begin..." % self.name)
        ivec_norm, self.item_vecs_map = self.build_item_vec(item_input_layer)
        self.ivec_norm = ivec_norm
        self.ui_input_layer = ui_input_layer

        print("%s commen model build begin..." % self.name)
        self.user_vec, self.user_input_layer = self.build_common_user_vec(user_input_layer, query_input_layer)
        print("%s commen model build done..." % self.name)

        deep_logits, self.ip = self.build_deep_logits(tf.concat([self.user_vec] * 10, axis=0), self.ivec_norm)
        pos_logits = self.build_position_logits(tf.concat([self.user_input_layer] * 10, axis=0), self.ui_input_layer)
        self.cross_entropy_logits = deep_logits
        self.position_logits = pos_logits
        self.final_logits = pos_logits + deep_logits

        if self.training_config.is_restore_embedding and self.training_config.embedding_checkpoint_dir is not None:
            logging.info("[init variables] from dir: %s" % self.training_config.embedding_checkpoint_dir)
            checkpoint_utils.init_from_checkpoint(self.training_config.embedding_checkpoint_dir, init_assignment_map)

    def build_common_user_vec(self, user_input_layer, query_input_layer):
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

        with tf.variable_scope(name_or_scope="%s-User-Vec-Space" % ("Share" if self.config.share_user_vec else self.name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):

            if query_input_layer is not None:
                user_input_layer = tf.concat([user_input_layer, query_input_layer], axis=1)
            user_vec = user_input_layer
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                for layer_id, num_hidden_units in enumerate(self.config.recall_dnn_hidden_units):
                    with variable_scope.variable_scope("user_hiddenlayer_%d" % layer_id) as user_hidden_layer_scope:
                        user_vec = layers.fully_connected(
                            user_vec,
                            num_hidden_units,
                            utils.getActivationFunctionOp(self.config.activation_op),
                            scope=user_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True,
                                               "is_training": self.is_training}
                        )
                        user_vec = rtp_ops.dropout_in_train(
                            user_vec,
                            self.training_config.is_local,
                            keep_prob=self.config.fc_dropout_keep_prob,
                            is_training=self.is_training
                        )

                user_vec = layers.fully_connected(
                    user_vec,
                    self.config.recall_vec_dim,
                    activation_fn=None,
                    scope="user_linear_layer",
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output]
                )
                user_norm = tf.nn.l2_normalize(user_vec, dim=1)
        return user_norm, user_input_layer

    def build_item_vec(self, item_input_layer):
        with tf.variable_scope(name_or_scope="%s-Item-Vec-Space" % ("Share" if self.config.share_item_vec else self.name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            item_vecs_da = {"item_input_vec":item_input_layer}
            item_vec = item_input_layer
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                for layer_id, num_hidden_units in enumerate(self.config.recall_dnn_hidden_units):
                    with variable_scope.variable_scope("item_hiddenlayer_%d" % layer_id) as item_hidden_layer_scope:
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
                        item_vec = layers.fully_connected(
                            item_vec,
                            num_hidden_units,
                            activation_fn=None,
                            scope=item_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                        )
                        item_vecs_da["item_vec_fc%s_out" % str(layer_id)] = item_vec
                        item_vec = layers.batch_norm(item_vec, scale=True, is_training=self.is_training)
                        item_vecs_da["item_vec_fc%s_bn" % str(layer_id)] = item_vec
                        item_vec = utils.getActivationFunctionOp(self.config.activation_op)(item_vec)
                        item_vecs_da["item_vec_fc%s_acti" % str(layer_id)] = item_vec
                        item_vec = rtp_ops.dropout_in_train(
                            item_vec,
                            self.training_config.is_local,
                            keep_prob=self.config.fc_dropout_keep_prob,
                            is_training=self.is_training
                        )

                item_vec = layers.fully_connected(
                    item_vec,
                    self.config.recall_vec_dim,
                    activation_fn=None,
                    scope="item_linear_layer",
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output],
                )
                item_vecs_da["item_vec_beforenorm"] = item_vec
                item_norm = tf.nn.l2_normalize(item_vec, dim=1)
                item_vecs_da["item_vec_afternorm"] = item_norm
        return item_norm, item_vecs_da

    def build_position_logits(self, user_input_layer, ui_input_layer=None):
        with tf.variable_scope(name_or_scope="%s-Position-Network" % self.name,
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            if ui_input_layer is not None:
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
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True,
                                               "is_training": self.is_training}
                        )
                        position_net = rtp_ops.dropout_in_train(
                            position_net,
                            self.training_config.is_local,
                            keep_prob=self.config.fc_dropout_keep_prob,
                            is_training=self.is_training)

        with tf.variable_scope(name_or_scope="%s-Logits" % self.name,
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                mem=self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE) as dnn_logits_scope:

            pos_logits = layers.linear(position_net, 1, variables_collections=[self.collections_dnn_hidden_layer],
                                       outputs_collections=[self.collections_dnn_hidden_output],
                                       biases_initializer=None, scope="pos-net"
                                       )
            # self.logits = self.deep_logits + self.pos_logits
        return pos_logits

    def build_deep_logits(self, user_norm, item_norm):
        ip = tf.multiply(user_norm, item_norm)
        deep_logits = tf.reduce_sum(ip, axis=1)
        paddings = tf.ones_like(deep_logits)
        deep_logits = tf.multiply(deep_logits, paddings * (5))
        deep_logits = tf.expand_dims(deep_logits, -1)
        return deep_logits, ip

    def trace_op(self):
        """
        trace feature input and feature output
        :return:
        """
        # super(BaseModelRecall, self).trace_op()
        self.trace["item_vec"] = self.ivec_norm
        self.trace["user_vec"] = tf.concat([tf.concat([self.user_vec] * 10, axis=0), self.position_logits], axis=1)

        if self.training_config.platform == 'pai' and self.training_config.predict:
            self.odpsWriter = tf.TableRecordWriter(tf.flags.FLAGS.tableoutput, slice_id=self.training_config.task_id)
            self.odpsWriterClose = self.odpsWriter.close()
            predict_show = []
            predict_show.append(tf.concat([tf.string_join([self.id, tf.reshape(x, [-1, 1]), tf.reshape(tf.as_string(y), [-1, 1])], '#') for x,y in zip(self.item_id_list, self.item_mask_list)], axis=0))
            predict_show.append(tf.as_string(self.predictions))
            vstr = tf.as_string(self.trace["item_vec" if self.config.trace_ivec == "on" else "user_vec"])
            v = tf.reduce_join(vstr, 1, keep_dims=False, separator=',')
            predict_show.append(v)
            self.write_to_table = self.odpsWriter.write(indices=[0, 1, 2], values=predict_show)

    def loss_op(self, task_label=None, task_num=None):
        with tf.name_scope("Loss_Op"):
            if self.name == "Recall_weighted":
                self.celoss = tf.reduce_sum(
                    tf.nn.weighted_cross_entropy_with_logits(
                        targets=self.label,
                        logits=self.final_logits,
                        pos_weight=tf.concat(self.weight_list, axis=0)
                    ) * self.item_mask) / tf.reduce_sum(self.item_mask)
            else:
                self.celoss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.final_logits,
                        labels=self.label) * self.item_mask) / (tf.reduce_sum(self.item_mask) + 1.0e-12)

            # Set up the training ops.
            self.reg_losses = self.reg_loss_op()
            self.loss = self.reg_losses + self.celoss
            # self.loss = tf.Print(self.loss, [x if isinstance(x, tf.Tensor) else x.values for x in self.feature_map_list[0].values()], summarize=1000)
            self.loss = tf.Print(self.loss, [tf.reduce_sum(self.item_mask)/self.training_config.batch_size], summarize=1000)
        return self.loss

    def predictions_op(self):
        self.predictions = tf.sigmoid(self.final_logits)
        # self.out4rtp["uservec"] = tf.reduce_mean(self.user_vec, axis=1)
        deep_logits, ip = self.build_deep_logits(self.user_vec, self.ivec_norm)
        pos_logits = self.build_position_logits(self.user_input_layer, self.ui_input_layer)
        self.out4rtp["score"] = tf.sigmoid(pos_logits + deep_logits)

    def summary(self):
        super(KonduModel, self).summary()
        with tf.name_scope("%s-Auxi-Loss" % self.name):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("cross_entropy_loss", self.celoss)
            tf.summary.scalar("reg_losses", self.reg_losses)
        with tf.name_scope("%s-Item-Vec" % self.name):
            for k, v in self.item_vecs_map.items():
                tf.summary.histogram(k, v * self.item_mask)
                shape = v.get_shape().as_list()
                mean, var = tf.nn.moments(v, range(len(shape)))
                tf.summary.scalar(k + "_mean", tf.squeeze(mean))
                tf.summary.scalar(k + "_var", tf.squeeze(var))
        with tf.name_scope("%s-Inner-Product" % self.name):
            tensor_map = {}
            tensor_map["inner_product"] = self.ip
            tensor_map["final_logits"] = self.final_logits
            tensor_map["cross_entropy_logits"] = self.cross_entropy_logits
            tensor_map["position_logits"] = self.position_logits
            for k, v in tensor_map.items():
                tf.summary.histogram(k, v * self.item_mask)
                shape = v.get_shape().as_list()
                mean, var = tf.nn.moments(v, range(len(shape)))
                tf.summary.scalar(k + "_mean", tf.squeeze(mean))
                tf.summary.scalar(k + "_var", tf.squeeze(var))
