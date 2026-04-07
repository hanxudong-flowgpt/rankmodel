import tensorflow as tf

from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
import ops.optimizer_ops as myopt
from ops import learning_rate_ops
from ops import base_ops
from ops import utils
from util.tflog import tflogger as logging


class Model(object):
    def __init__(self,
                 train_config,
                 model_config,
                 model_columns,
                 model_name="CTR", model_idx=0):
        # Config.
        self.config = model_config
        self.mc = model_columns
        self.training_config = train_config

        # Model Name
        self.name = model_name
        self.model_idx = model_idx

        self.out4rtp = {}
        self.metrics = {}
        self.trace = {}
        self.is_training = None
        self.loss = None
        self.train_op = None
        self.global_step = None
        self.itera = None

    def build_placeholder(self, is_training_placeholder):
        self.is_training = is_training_placeholder

    def build_inputs(self, sample_data):
        pass

    def build_model(self):
        pass

    def get_train_op(self):
        if self.training_config.train_op == 1:
            self.get_train_op1()
        elif self.training_config.train_op == 2:
            self.get_train_op2()
        elif self.training_config.train_op == 3:
            self.get_train_op3()

    def get_train_op1(self):
        from tensorflow.contrib import layers
        self.train_op = layers.optimize_loss(
            self.loss,
            tf.train.get_global_step(),
            # self.global_step,
            learning_rate=self.training_config.initial_learning_rate,
            optimizer='Adagrad')

    def get_train_op2(self):
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
                    summaries=["learning_rate", "loss", "gradient_norm", "global_gradient_norm"],#myopt.OPTIMIZER_SUMMARIES,
                  )
            train_op_vec.append(loss)

            update_ops = self.update_op()
            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_op_vec)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(global_step):
                        # self.train_op = self.global_step_add.op
                        self.train_op = [state_ops.assign_add(global_step, 1, use_locking=True)]
                    with ops.colocate_with(self.global_step):
                        # self.train_op = self.global_step_add.op
                        self.train_op.append(state_ops.assign_add(self.global_step, 1, use_locking=True))

    def get_train_op3(self):
        learning_rate_decay_cs = lambda lr, gs: learning_rate_ops.lr_cold_start(
            self.training_config.initial_learning_rate,
            gs,
            self.training_config.lrcs_init_lr,
            self.training_config.lrcs_init_step)
        learning_rate_decay_cs = None
        with tf.variable_scope(name_or_scope="Optimize_Layer",
                               # partitioner=base_ops.partitioner(self.config.ps_num,
                               #                                  self.training_config.embedding_partitino_size),
                               reuse=tf.AUTO_REUSE):
            gs = tf.train.get_or_create_global_step()
            logging.info("Global_step:{},{}".format(self.name, str(gs)))
            self.train_op, _, _ = myopt.optimize_loss(
                loss=self.loss,
                global_step=self.global_step,
                learning_rate=self.training_config.initial_learning_rate,
                optimizer=utils.getOptimizer(self.training_config,
                                             global_step=gs,
                                             learning_rate=self.training_config.initial_learning_rate,
                                             learning_rate_decay_fn=learning_rate_decay_cs),
                update_ops=self.update_op(),
                clip_gradients=self.training_config.clip_gradients,
                variables=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES),
                # learning_rate_decay_fn=learning_rate_decay_cs,
                increment_global_step=True,
                summaries=myopt.OPTIMIZER_SUMMARIES)

    def mark_output(self):
        with tf.name_scope("%s_Mark_Output" % self.name):
            date = tf.convert_to_tensor(self.training_config.train_parts[0], tf.string)
            xx = tf.cast(tf.not_equal(date, ""), tf.float32)
            for k, v in self.out4rtp.items():
                x = tf.identity(v * xx, name="rank_predict")

    def loss_op(self, task_label=None, task_num=None):
        pass

    def predictions_op(self):
        pass

    def summary(self):
        pass

    def trace_op(self):
        """
        trace feature input and feature output
        :return:
        """
        pass

    @property
    def model_name(self):
        return self.name

    @property
    def model_columns(self):
        return self.mc

    @property
    def model_config(self):
        return self.config

    def setup_cl_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="%s-global_step" % self.name,
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step
        # self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def build_main_part(self, sample_data, is_training_placeholder, train=True):
        """Creates all ops for training and evaluation."""
        self.build_placeholder(is_training_placeholder)
        self.build_inputs(sample_data)
        self.build_model()
        self.predictions_op()
        self.mark_output()
        self.loss_op()
        self.trace_op()
        if train:
            self.setup_cl_global_step()
            self.get_train_op()
            self.summary()

    def update_op(self):
        update_opss = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops = []
        for update_op in update_opss:
            start_name = update_op.name
            arr = update_op.name.split('/')
            if len(arr) > 1:
                start_name = arr[0]
            if start_name.startswith(self.name) or \
                    (start_name.startswith('Share') and not start_name.endswith('_1')):
                update_ops.append(update_op)
        logging.info("update ops: %s" % str(update_ops))
        return update_ops

    def update_op2(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.update_ops = []
        for update_op in update_ops:
            if update_op.name.startswith(self.name):
                self.update_ops.append(update_op)
        logging.info("update ops: %s" % str(self.update_ops))
        return self.update_ops

    def reg_loss_op(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_losses_model = []
        for reg_loss in reg_losses:
            start_name = reg_loss.name
            arr = reg_loss.name.split('/')
            if len(arr) > 1:
                start_name = arr[0]
            if start_name.startswith(self.name) or \
                    (start_name.startswith('Share') and not start_name.endswith('_1')):
                reg_losses_model.append(reg_loss)
        return tf.reduce_sum(reg_losses_model)
