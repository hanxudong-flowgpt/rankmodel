import tensorflow as tf
from feature_generator import FeatureGenerator
from tensorflow.contrib import layers


def prepare_default():
    return [[tf.constant("", dtype=tf.string)],
            [tf.constant("", dtype=tf.string)],
            [tf.constant(0, dtype=tf.int64)]]


input_file_names = ['/home/boyi.wb/search_model/fg/test_data.csv']
filename_queue = tf.train.string_input_producer(input_file_names,
                                                shuffle=True,
                                                capacity=100,
                                                num_epochs=1)
_, value = tf.TextLineReader().read_up_to(filename_queue, 2)
example_list = [tf.decode_csv(value, record_defaults=prepare_default(), field_delim='\t')
                   for _ in range(2)]
batch_data_list = tf.train.batch_join(example_list,
                                         batch_size=2,
                                         capacity=10000,
                                         enqueue_many=True,
                                         allow_smaller_final_batch=True)

import rtp_fg

fg = FeatureGenerator('./fg_test.json')
features = rtp_fg.parse_genreated_fg(fg.feature_conf, batch_data_list[1])
label = batch_data_list[2]

input_layer = layers.input_from_feature_columns(features, fg.feature_columns)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        step = 0
        while not coord.should_stop():
             f = sess.run(features)
             for k in f.keys():
                 print(k)
                 print(f[k])
             f2 = sess.run(input_layer)
             print(f2)
    except tf.errors.OutOfRangeError:
        print ("done")
    finally:
            coord.request_stop()
    coord.join(threads)
    sess.close()

