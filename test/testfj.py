# -*- coding: utf-8 -*-
# usage
# pai -name tensorflow -Dtables="odps://prj_name/tables/table_name" -Dscript="file:///path/to/tf_test_feature_join_cluster.py"
# -Dcluster="{\"worker\":{\"count\":2}, \"ps\":{\"count\":2}}";

import tensorflow as tf
import numpy as np
import json

# 打印加载feature表的时间和等待时间的info信息，在stderr里可以查看。
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_string('tables', '', 'odps table name')


def sample_itemandfeature(feature_table, query):
    with open("config/fg.json") as f:
        feature_conf = json.load(f)
    itemlist = feature_table.join_feature('query2item', query, 100, ',')
    features = []
    fgfeatures = []
    import rtp_fg
    for itemid in itemlist:
        feature = feature_table.join_feature('itemfeature', tf.string_to_number(itemid, tf.int64), 1, '\t')[0]
        fgfeature = rtp_fg.parse_genreated_fg(feature_conf, feature)
        features.append(feature)
        fgfeatures.append(fgfeature)
    return itemlist, features, fgfeatures
'''
def generate_allfeature(logfeaurelist, otherfeaturelist):
    allf = []
    for lf, of in zip(logfeaurelist, otherfeaturelist):
        allf.append(lf,of)
'''

def main(argv=None):
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
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            tables = FLAGS.tables

            #input_tables_conf = {'user': [tables, 'id,feature', [np.int64(0), ""]]}
            input_tables_conf = {'query2item': ["odps://search_offline/tables/query_item_sample_for_pai/ds=20181226",
                                                'queryseg_norm_encode,item_list', [np.int64(0), ""]],
                                 'itemfeature': ["odps://search_offline/tables/item_features_for_pai/ds=20181225",
                                                 'id,feature', [np.int64(0), ""]]
                                 }

            feature_table = tf.data.FeatureTable(input_tables_conf,
                                                 server_count=ps_count,
                                                 worker_index=FLAGS.task_index,
                                                 worker_count=worker_count)

            dataset = tf.data.TableRecordDataset(tables,
                                                 selected_cols="id",
                                                 record_defaults=[""],#[np.int64(0)],
                                                 slice_id=FLAGS.task_index,
                                                 slice_count=worker_count)
            dataset = dataset.batch(21)
            ids = dataset.make_one_shot_iterator().get_next()[0]

            q = tf.string_split(ids, "_")
            #q = tf.Print(q, ['split', q.values, q.indices])
            q = tf.sparse_slice(q, [0, 2], [21, 1])
            #q = tf.Print(q, ['slice', q])
            q = tf.sparse_to_dense(q.indices, q.dense_shape, q.values, default_value="0")
            #q = tf.Print(q, ['todense', q])
            q = tf.reshape(q, [-1])
            q = tf.string_to_number(q, tf.int64)
            print(q)
            # feature_list = feature_table.join_feature('query2item', ids, 100, ',')
            itemlist, features, fgfeatures = sample_itemandfeature(feature_table, q)
            feature_map = {
                'id': ids,
                'items': itemlist[0],
                'features': features[0],
                'q': q,
                'fgfeatures': [0]
            }

            ready_hook = feature_table.ready_hook(check_wait_secs=10)
            hooks = [ready_hook]

            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   hooks=hooks) as mon_sess:
                print("start training.")
                while not mon_sess.should_stop():
                    try:
                        print(q)
                        print(mon_sess.run(feature_map))
                    except tf.errors.OutOfRangeError:
                        print('end of sequence')
                        break


if __name__ == '__main__':
    tf.app.run()