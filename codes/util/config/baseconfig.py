import tensorflow as tf


def mybool(xstr):
    if xstr is None or xstr.lower() == 'off' or xstr.lower() == 'false' or xstr.lower() == '0':
        return False
    else:
        return True


def str2typefunc(demo_value):
    if type(demo_value) == str:
        return lambda x: str(x)
    elif type(demo_value) == bool:
        return lambda x: mybool(x)
    elif type(demo_value) == int:
        return lambda x: int(float(x))
    elif type(demo_value) == float:
        return lambda x: float(x)
    elif type(demo_value) == list:
        if len(demo_value) > 0:
            sonfunc = str2typefunc(demo_value[0])
            return lambda x: [sonfunc(tt) for tt in x.split('|')]
        else:
            return lambda x: x.split('|')
    else:
        print('Unkown type in str2typefunc:', demo_value, type(demo_value))
        return lambda x: str(x)


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self, **kwargs):
        """Sets the default model hyperparameters."""
        # File name of sharded Hdfs file of csv format
        self.table_project = 'search_offline'
        self.table_name = 'lsc_log_wide_wdlfg_ctr'
        self.da_table_name = 'lsc_log_wide_wdlfg_ctr'
        self.cross_path = ""
        self.model_name = "None"

        # fg config
        self.lsc_conf = ""
        self.mc_conf = ""
        self.fgconf_list = []
        self.fgconf_map = {}

        # Number of epochs.
        self.files_capacity = 1000000  # will skip files err if full

        # auxi loss
        self.target_entropy_loss_weight = 0.0
        self.mmd_loss_weight = 0.0

        # If < 1.0, the dropout keep probability applied to FC variables.
        self.fc_dropout_keep_prob = 1

        # Number of hidden units after user/query/item & user/query/ui.
        self.dnn_hidden_units = [512, 256, 128]
        self.personal_dnn_hidden_units = [16]
        self.recall_dnn_hidden_units = [512, 256]
        self.recall_vec_dim = 128
        self.position_dnn_hidden_units = [64, 32, 8]
        self.cnn_filter_size = [1, 3, 5, 7]
        self.share_hidden_units = [256, 64]
        self.activation_op = "lrelu"
        # options as: [relu|tanh|lrelu], use relu if not supported. Used in model_ops/utils.py:getActivationFunctionOp

        self.ps_num = 1

        self.usepos = True
        self.trace_uvec = "on"
        self.trace_ivec = "off"
        if kwargs.has_key("is_training"):
            self.is_training = kwargs["is_training"]
        else:
            try:
                self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
            except KeyError as e:
                self.is_training = tf.placeholder_with_default(input=tf.constant(False), shape=(), name='training')
        if kwargs.has_key("use_tfrecord"):
            self.use_tfrecord = kwargs["use_tfrecord"]
        else:
            try:
                self.use_tfrecord = tf.get_default_graph().get_tensor_by_name("use_tfrecord:0")
            except KeyError as e:
                self.use_tfrecord = tf.placeholder_with_default(input=tf.constant(True), shape=(), name='use_tfrecord')

        # Update other configuration parameters.
        self.__dict__.update(kwargs)
        # self.cvt_func_map = dict([(k, str2typefunc(v)) for k, v in self.__dict__.items() if k != 'cvt_func_map'])

    def __str__(self):
        return str(self.__dict__)
    '''
    def updateFromStrMap(self, str2strmap):
        for k, v in str2strmap.items():
            if k in self.cvt_func_map:
                newv = self.cvt_func_map[k](v)
                print('ModelConfig update %s from %s to %s' % (str(k), str(self.__dict__[k]), str(newv)))
                self.__dict__.update({k: newv})
    '''

    def updateFromMap(self, str2strmap):
        for k, newv in str2strmap.items():
            if k in self.__dict__:
                print('TraingConfig update %s from %s to %s' % (str(k), str(self.__dict__[k]), str(newv)))
                self.__dict__.update({k: newv})



class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self, **kwargs):
        """Sets the default training hyperparameters."""
        # runtime env var
        self.currentPath = None
        self.platform = "blink"
        self.task_id = 0
        self.cluster = None
        self.worker_num = 1
        self.ps_num = 1
        self.is_local = True
        self.job_conf = ""
        self.fg_conf = "jsonfile/lsc_conf.json"
        self.version = "0.2"
        # self.hdfs_base_path = ""
        # Model Type Used
        self.model_type = 'baseline_seqatt'
        self.predict_model_name = 'RANK_CTR'
        self.vec_append_feature = []

        # Read Data Table
        self.table_project = ['search_offline']
        self.table_name = ['lsc_log_wide_wdlfg_ctr']
        self.norepeat_read = True
        self.norepeat_state_min = -1

        # Number of examples per epoch of training data.
        self.files_capacity = 1000000
        self.num_examples_per_epoch = 150 * 10000 * 10000
        self.num_examples_per_epoch_cvr = 20 * 10000 * 10000

        # Number of data buffer capacity.
        self.training_queue_capacity = 16

        # Number of file realder.
        self.num_reader = 2
        self.data_format = 'gz'

        # Batch size.
        self.batch_size = 512
        self.eval_bs = 256

        # Fg Batch params
        self.fg_thread = 4
        self.fg_capacity = 4

        # Number of epochs.
        self.num_epochs = 1
        self.num_epochs_dynamic = 1
        self.shard = False
        self.share_context = None

        # Optimizer for training the model.
        self.optimizer = "Adagrad"
        self.optimizer_use_lock = False
        self.linear_optimizer = 'Ftrl'

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 0.01
        self.momentum = 0.9  # only used with MomentumOptimizer
        self.learning_rate_decay_factor = 0.
        self.num_epochs_per_decay = 1.0
        self.adadelta_rho = 0.95
        self.adadelta_epsilon = 0.000001

        # Params for AdagradDecayOptimizer
        self.decay_step = 1000000
        self.decay_rate = 0.9

        # linear learning rate
        self.linear_learning_rate = 0.1
        self.initial_accumulator_value = 0.1  # more less , more sparse
        self.l1_regularization_strength = 0.8  # more large, more sparse
        self.l2_regularization_strength = 10.0

        # learning rate cold start -- start after create new model
        self.lrcs_using = True
        self.lrcs_init_lr = 0.001
        self.lrcs_init_step = 1000000
        # learning rate warm restart -- periodic repeating
        self.lrwr_using = False
        self.lrwr_init_lr_min = 0.01
        self.lrwr_periodic_step = 10000000

        # Init
        self.init_op = "zero"  # [constant zero xavier]
        self.constant_init = 0.0001
        # restore embedding
        self.train_op = 1
        self.is_restore_embedding = True
        self.embedding_checkpoint_dir = None
        # dnn learning reg
        self.dnn_l2_reg = 1e-6  # 0.0005 0.0001 0.00001
        self.attention_l2_reg = 0.

        # If not None, clip gradients to this value.
        self.clip_gradients = 50.0

        # How many model checkpoints to keep.
        self.need_save_ckp = True
        self.max_checkpoints_to_keep = 10
        self.save_checkpoint_secs = 2400
        self.save_summaries_steps = 100

        self.train_parts = ["20180120", "20180121", "20180122"]
        self.eval_parts = ["20180123"]
        self.join_feature_parts = ["20180123"]
        self.eval_join_feature_parts = ["20180123"]

        self.build_model_name = "shareitem_cxr_model"

        self.max_step = 100000 * 10000

        # Metrics related
        self.auc_reset_step = 500000  # positive numbers related
        self.auc_number_thre = 2000

        # Model Evaluation
        self.debug = False
        self.predict = False
        self.predict_postfix = '_predict'
        self.trace_tensors = False
        self.trace_tensors_valid = False
        self.extract_features = False
        self.odps_user = 'lingyun'

        # exp_version
        self.expr = 'default'

        # save_dir
        self.save_dir = ''

        self.print_variable = False

        # model partition size in bytes
        self.embedding_partitino_size = 8 * 1024 * 1024
        self.dnn_partition_size = 64 * 1024

        self.adv_loss_weight = 0.01

        self.adv_diff_weight = 0.1

        self.use_fm_to_hdfs = False
        self.sample_name = ''
        self.train_start_time = ''
        self.train_end_time = ''
        self.eva_start_time = ''
        self.eva_end_time = ''
        self.workflow_id = ''  # '12'
        self.cluster_id = '19'  # zhangbei:19, shanghai:20, zhangbeihunbu:21

        self.num_thresholds = 21
        self.buckets = [0.0] + [(i + 1) * 1.0 / (self.num_thresholds - 1) for i in range(self.num_thresholds - 2)] + [1.0]
        self.trace_prediction_bucket = False

        # Update other configuration parameters.
        self.__dict__.update(kwargs)

        # Know How to Convert from String
        # self.cvt_func_map = dict([(k, str2typefunc(v)) for k, v in self.__dict__.items()])

    def __str__(self):
        return str(self.__dict__)

    def getLinearOptimizer(self):
        if self.linear_optimizer == 'Ftrl':
            return tf.train.FtrlOptimizer(
                learning_rate=self.initial_learning_rate,
                initial_accumulator_value=self.initial_accumulator_value,  # more less , more sparse
                l1_regularization_strength=self.l1_regularization_strength,  # more large, more sparse
                l2_regularization_strength=self.l2_regularization_strength
            )
    '''
    def updateFromStrMap(self, str2strmap):
        for k, v in str2strmap.items():
            if k in self.cvt_func_map:
                newv = self.cvt_func_map[k](v)
                print('TraingConfig update %s from %s to %s' % (str(k), str(self.__dict__[k]), str(newv)))
                self.__dict__.update({k: newv})
    '''

    def updateFromMap(self, str2strmap):
        for k, newv in str2strmap.items():
            if k in self.__dict__:
                print('TraingConfig update %s from %s to %s' % (str(k), str(self.__dict__[k]), str(newv)))
                self.__dict__.update({k: newv})
