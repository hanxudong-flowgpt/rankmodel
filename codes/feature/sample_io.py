import tensorflow as tf
from util.tflog import tflogger as logging


def getHdfsFileList(use_fm_to_hdfs, task_id, worker_num, table_project, table_name, train_parts, eval_parts, base_path='/pora/hdfs-table/'):
    if use_fm_to_hdfs == False:
      from util.io.utils import down_load_data
      files_list = down_load_data(task_id,
                                  worker_num,
                                  table_name,
                                  train_parts,
                                  eval_parts,
                                  project=table_project, base_path=base_path)
    else:
      from util.io.utils import down_load_data_fm
      files_list = []
      '''
      files_list = down_load_data_fm(task_id,
                                     worker_num,
                                     training_config.sample_name,
                                     training_config.train_start_time,
                                     training_config.train_end_time,
                                     training_config.eva_start_time,
                                     training_config.eva_end_time,
                                     training_config.workflow_id,
                                     training_config.cluster_id)
      '''
    logging.info("[input_file_names] (len %s) : %s" % (len(files_list), files_list if len(files_list)<20 else files_list[:20]))

    # Dynamic
    from util.io.utils import down_load_data_dynamic
    files_list_dynamic_tain, files_list_dynamic_eval = down_load_data_dynamic(task_id,
                                                                              worker_num,
                                                                              table_name,
                                                                              train_parts,
                                                                              eval_parts,
                                                                              project=table_project, base_path=base_path)

    logging.info("[input_file_names_dynamic_train] (len %s) : %s" % (len(files_list_dynamic_tain), files_list_dynamic_tain if len(files_list_dynamic_tain)<20 else files_list_dynamic_tain[:20]))
    logging.info("[input_file_names_dynamic_eval] (len %s) : %s" % (len(files_list_dynamic_eval), files_list_dynamic_eval if len(files_list_dynamic_eval)<20 else files_list_dynamic_eval[:20]))
    return files_list, files_list_dynamic_tain, files_list_dynamic_eval


prepared_default = [[tf.constant("", dtype=tf.string)], [tf.constant("", dtype=tf.string)],
                    [tf.constant("0", dtype=tf.string)], [tf.constant("", dtype=tf.string)]]


def build_odps_inputs(input_file_names, slice_id, slice_count, num_reader, files_capacity, num_epochs, batch_size, default=None):
    worker_device = "/job:worker/task:{}".format(slice_id)
    with tf.device(worker_device):
        filename_queue = tf.train.string_input_producer(input_file_names,
                                                        capacity=files_capacity,
                                                        num_epochs=num_epochs,
                                                        shuffle=True,
                                                        seed=None)
        output = []
        for i in range(num_reader):
            logging.info("use odps reader slice_count:%d, slice_id:%d" % (slice_count, slice_id))
            reader = tf.TableRecordReader(csv_delimiter='\t',
                                          selected_cols='id,features,label,type,pv_nid_set,rank_nid_set,negative_nid_set,other_info_1',
                                          slice_count=slice_count,
                                          slice_id=slice_id,
                                          num_threads=8,
                                          capacity=4 * batch_size)
            key, value = reader.read_up_to(filename_queue, batch_size)
            example_list = [tf.decode_csv(value, record_defaults=default if default else prepared_default, field_delim='\t')]
            output.extend(example_list)
    return output


def build_hdfs_inputs_iterator(file_list, bkt_id, bkt_num , num_epochs, batch_size, data_format):
    logging.info("worker num: %d, task id: %d, total file number on all worker: %d" % (bkt_num, bkt_id,
                                                                                       len(file_list)))
    file_list_this = []
    for i in range(len(file_list)):
        if i % bkt_num == bkt_id:
            file_list_this.append(file_list[i])

    logging.info("read hdfs files on this worker: %s" % file_list_this if len(file_list_this) < 20 else
                 file_list_this[:20])
    logging.info("read hdfs files epochs: %d" % num_epochs)
    if num_epochs>1:
        file_list_this = file_list_this * num_epochs
    dataset = tf.data.TextLineDataset(file_list_this,
                                      compression_type='GZIP' if data_format == 'gz' else None,
                                      buffer_size=30 * 1024 * 1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size*2)
    iterator = dataset.make_initializable_iterator()
    return iterator


def build_hdfs_inputs_iterator_shard(file_list_this, num_epochs, batch_size, data_format):
    logging.info("read hdfs files on all worker: %s" % file_list_this if len(file_list_this) < 20 else
                 file_list_this[:20])
    logging.info("read hdfs files epochs: %d" % num_epochs)
    if num_epochs>1:
        file_list_this = file_list_this * num_epochs
    dataset = tf.data.TextLineDataset(file_list_this,
                                      compression_type='GZIP' if data_format == 'gz' else None,
                                      buffer_size=8 * 1024 * 1024, use_dynamic_shard=True,
                                      shuffle_func=tf.data.shuffle_by_dir_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 10)
    iterator = dataset.make_initializable_iterator()
    return iterator


def build_hdfs_inputs1(file_list, task_id, worker_num, files_capacity, num_epochs, num_reader, batch_size, default=None):
    logging.info("worker num: %d, task id: %d, total file number on this worker: %d" % (worker_num, task_id,
                                                                                        len(file_list)))
    logging.info("read hdfs files on this worker: %s" % file_list)
    worker_device = "/job:worker/task:{}".format(task_id)
    with tf.device(worker_device):
        filename_queue = tf.train.string_input_producer(file_list,
                                                        capacity=files_capacity,
                                                        num_epochs=num_epochs,
                                                        shuffle=True,
                                                        seed=None)
    output = []
    for i in range(num_reader):
        reader = tf.TextLineReader()
        key, value = reader.read_up_to(filename_queue, batch_size)
        #example_list = [tf.decode_csv(value, record_defaults=default if default else prepared_default, field_delim='\t')]
        #output.extend(example_list)
        output.append(value)
    return output


def build_hdfs_inputs_dynamic(name, ispredict, file_names_dynamic_train, file_names_dynamic_eval,
                              task_id, worker_num, num_epochs, num_reader, batch_size, default=None):
    #from fifo_queue_shared import FIFOQueueShared
    from flink_tensorflow.queue.FIFOQueueShared import FIFOQueueShared
    train_share_queue = FIFOQueueShared(
        file_names_dynamic_train,
        task_id,
        worker_num,
        num_reader,
        qtype=name + 'train',
        MAXLIST=200000,
        # MAXWORKER=2000,
        num_epochs=num_epochs,
        shuffle=False,
        seed=None)
    '''
    eval_share_queue = FIFOQueueShared(
        file_names_dynamic_eval,
        task_id,
        worker_num,
        num_reader,
        qtype=name + 'eval',
        MAXLIST=200000,
        # MAXWORKER=2000,
        num_epochs=num_epochs+1,
        shuffle=False,
        seed=None)
    '''
    eval_share_queue = FIFOQueueShared(
        file_names_dynamic_eval,
        task_id,
        1,
        num_reader,
        qtype=name + 'evaluate',
        MAXLIST=10000,
        MAXWORKER=1,
        num_epochs=num_epochs,
        shuffle=False,
        seed=None)

    predict_share_queue = FIFOQueueShared(
        file_names_dynamic_train,
        task_id,
        worker_num,
        num_reader,
        qtype=name + 'predict',
        # MAXLIST=2000,
        # MAXWORKER=2000,
        num_epochs=1,
        shuffle=False,
        seed=None)

    share_queues = [train_share_queue, eval_share_queue, predict_share_queue]
    if ispredict:
        share_queue = predict_share_queue
    else:
        if task_id == 0:
            share_queue = eval_share_queue
        else:
            share_queue = train_share_queue

    output = []
    for i in range(num_reader):
        reader = tf.TextLineReader()
        share_queue.add_reader(reader)
        up = share_queue.update_state(reader, i)
        with tf.control_dependencies([up]):
            key, value = reader.read_up_to(share_queue.local_queue, batch_size)
        with tf.control_dependencies([tf.Print(key[0], ['filename key is', key[0]])]):
            example_list = \
                [tf.decode_csv(tf.Print(value, ["here1"]), record_defaults=default if default else prepared_default, field_delim='\t')]
        output.extend(example_list)
    return output, share_queues
