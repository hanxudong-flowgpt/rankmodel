# -*- coding: utf-8 -*-
# usage
# pai -name tensorflow -Dscript="file:///path/to/copyhdfs.py"
# -Dcluster="{\"worker\":{\"count\":40}, \"ps\":{\"count\":2}}";

import tensorflow as tf
import numpy as np

# 打印加载feature表的时间和等待时间的info信息，在stderr里可以查看。
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_string('sourceDir', '', 'dir under which the files to copy exist')
flags.DEFINE_string('destDir', '', 'dir to copy to')
flags.DEFINE_integer('threadnum', 40, 'the number of thread to exe cp')


def main(unused_argv):
    destDir = FLAGS.destDir
    sourceDir = FLAGS.sourceDir
    if destDir == "":
        raise Exception("destDir must be given")
    if sourceDir == "" or not tf.gfile.IsDirectory(sourceDir):
        raise Exception("sourceDir wrong")

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    worker_count = len(worker_hosts)
    ps_count = len(ps_hosts)
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):

            try:
                copy_lastest_ckpt = tf.contrib.framework.copy_lastest_ckpt
            except:
                copy_lastest_ckpt = tf.train.copy_lastest_ckpt

            tf.logging.info("start copy ......")
            copy_lastest_ckpt(sourceDir, destDir, generate_timestamp_dir=False,
                      num_threads=FLAGS.threadnum)
            tf.logging.info("finish copy !!!")


if __name__ == '__main__':
    tf.app.run()