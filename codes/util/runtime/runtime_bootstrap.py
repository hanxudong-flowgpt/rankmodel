#****************************************************************#
# Author: qiaohan.qh@alibaba-inc.com
# Create Date: 2018-03-18 17:50
# Modify Author: qiaohan.qh@alibaba-inc.com
# Modify Date: 2018-03-18 17:50
# Function:
#***************************************************************#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import tensorflow as tf
from util.config.baseconfig import TrainingConfig
from util.tflog import tflogger as logging

tf.flags.DEFINE_string("platform", "blink", "rum time platform")
tf.flags.DEFINE_string("protocol", "grpc", "default grpc")
tf.flags.DEFINE_string("algoparamstr", "", "params used in tf model")
tf.flags.DEFINE_string("job_conf", "", "job config")
tf.flags.DEFINE_string("save_dir", "", "dir for ckpt")


tf.flags.DEFINE_string("debug_dir", "", "dir for debug info while model running")
tf.flags.DEFINE_string("data_format", "tfrecord", "input data format: tfrecord, csv or tfrecord_polaris")
tf.flags.DEFINE_string("start_day", "", "the day corresponding what in hdfs path like YYYY-mm-dd")
tf.flags.DEFINE_string("end_day", "", "the day corresponding what in hdfs path like YYYY-mm-dd")

tf.flags.DEFINE_string("ps_hosts", "127.0.0.1:45678", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "127.0.0.1:45679", "worker hosts")
tf.flags.DEFINE_string("job_name", "", "job name")
tf.flags.DEFINE_integer("task_index", "-1", "task index")

tf.flags.DEFINE_string("train_parts", "", "train_parts")
tf.flags.DEFINE_string("eval_parts", "", "eval_parts")
tf.flags.DEFINE_integer("running_secs", 60*60*200, "max seconds during running")
tf.flags.DEFINE_string("multi_val_split", ";", "splitor of multi value feature")
tf.flags.DEFINE_string("seq_val_split", ";", "splitor of sequence feature")

tf.flags.DEFINE_string("mode", "train", "predict or train")
tf.flags.DEFINE_string("predict_model_name", "None", "which model to predict")
tf.flags.DEFINE_string("predict_saved_model", "None", "which saved model to restore to predict")
tf.flags.DEFINE_string("tableoutput", "", "which hive table or hadoop dir to write predict results into")


logging.info("platform: " + tf.flags.FLAGS.platform)
if tf.flags.FLAGS.platform == "tony":
  from pdd_runtime import PaiBootStrap as Boot
elif tf.flags.FLAGS.platform == "local":
  from pai_runtime import PaiBootStrap as Boot
else:
  raise ValueError("running platform not support")



def string2kv(s, d1, d2):
    kv = {}
    if s is None or s == '':
        return kv
    for ele in s.split(d1):
        pair = ele.split(d2)
        if len(pair) != 2:
            continue
        kv[pair[0]] = pair[1]
    return kv


class BootStrap(Boot):
    def __init__(self):
        logging.info("init BootStrap...")
        super(BootStrap, self).__init__()
        self.train_config = None

    def worker_finish_ratio(self, workernum):
        return len(set(self.finishedWorkers())) * 1.0 / workernum

    def worker_finish_num(self):
        return len(set(self.finishedWorkers()))

    def any_worker_finished(self):
        return len(set(self.finishedWorkers())) > 0

    def parse_config(self, confstr, **kwargs):
        # self.platform = tf.flags.FLAGS.platform
        train_config = TrainingConfig(**kwargs)
        train_config.updateFromStrMap(string2kv(confstr, ',', '='))
        return train_config

    def parse_config_from_dict(self, conf_map, **kwargs):
        # self.platform = tf.flags.FLAGS.platform
        train_config = TrainingConfig(**kwargs)
        train_config.updateFromMap(conf_map)
        return train_config
