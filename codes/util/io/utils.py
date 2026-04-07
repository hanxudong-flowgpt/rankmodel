import hdfs_file
import hdfs_client
import filesystem
import urllib
import urllib2
import time


def down_load_data(index, worker_num, table_name, train_parts, eval_parts, project='search_offline', base_path='/pora/hdfs-table/'):
  file_list = []
  if index == 0:
    for ev in eval_parts:
      file_list.extend(
        # hdfs_file.get_task_file_list(project=project, table=table_name,
        hdfs_file.get_task_file_list(project=project, table=table_name,
                                     partition="ds=%s" % ev, task_num=worker_num, task_id=index, base_path=base_path))
  else:
    for ds in train_parts:
      file_list.extend(
        hdfs_file.get_task_file_list(project=project, table=table_name,
                                     partition="ds=%s" % ds, task_num=worker_num, task_id=index, base_path=base_path))
  return file_list


def down_load_data_dynamic(index, worker_num, table_name, train_parts, eval_parts, project='search_offline', base_path='/pora/hdfs-table/'):
  file_list_train = []
  for ds in train_parts:
    file_list_train.extend(
      hdfs_file.get_whole_file_list(project=project, table=table_name,
                                   partition="ds=%s" % ds, task_num=worker_num, task_id=index, base_path=base_path))

  file_list_eval = []
  for ds in eval_parts:
    file_list_eval.extend(
      hdfs_file.get_whole_file_list(project=project, table=table_name,
                                   partition="ds=%s" % ds, task_num=worker_num, task_id=index, base_path=base_path))
  return file_list_train, file_list_eval


def down_load_data_fm(index, worker_num, sample_name, train_start_time, train_end_time,
                      eva_start_time, eva_end_time, workflow_id, cluster_id):
  file_list = []
  if index == 0:
    # from flink_tensorflow.python_sdk.smart_io.odps_table_util import *
    file_list.extend(
      get_fm_task_file_list(task_num=worker_num,
                            task_id=index,
                            sample_name=sample_name,  # 'lsc_cvr_sample',
                            start_time=eva_start_time,  # '20180502090000',
                            end_time=eva_end_time,  # '20180502105900',
                            workflow_id=workflow_id,  # '12',
                            cluster_id=cluster_id  # '19'
                           ))
  else:
    # from flink_tensorflow.python_sdk.smart_io.odps_table_util import *
    file_list.extend(
      get_fm_task_file_list(task_num=worker_num,
                            task_id=index,
                            sample_name=sample_name,  # 'lsc_cvr_sample',
                            start_time=train_start_time,  # '20180502090000',
                            end_time=train_end_time,  # '20180502105900',
                            workflow_id=workflow_id,  # '12',
                            cluster_id=cluster_id  # '19'
                            ))
  return file_list


def get_fm_file_list(sample_name, start_time, end_time, workflow_id, cluster_id):
  param = {'appId': '1', 'debugVersion': 'c4test', 'action': 'AsyncService',
           'name': 'WriteFileList', 'env': 'pre'}
  param['sampleName'] = sample_name
  param['dataStartTime'] = start_time
  param['dataEndTime'] = end_time
  param['workflowId'] = workflow_id
  param['clusterId'] = cluster_id
  param_urlencode = urllib.urlencode(param)
  requrl = "http://zbpre.porschemaster.alibaba-inc.com/naruto/api"
  req = urllib2.Request(url=requrl, data=param_urlencode)
  print("request url:", req.get_full_url())
  res_data = urllib2.urlopen(req)
  res = res_data.read()
  print(res)
  time.sleep(10)
  current_path = '/pora/fm/samplecenter/' + workflow_id + "/filelist"
  print("current path:", current_path)
  client = hdfs_client.get_hdfs_client()
  hdfs_fs = filesystem.HDFSFilesystem()
  done = hdfs_fs.exists(current_path)
  if not done:
    print("file not exist!")
    return None
  with client.read(hdfs_path=current_path) as reader:
    content = reader.read()
    file_list = str(content).split('\n')
  return file_list


def get_fm_task_file_list(task_num, task_id, sample_name, start_time, end_time, workflow_id, cluster_id='19'):
  file_list = get_fm_file_list(sample_name=sample_name, start_time=start_time, end_time=end_time,
                               workflow_id=workflow_id, cluster_id=cluster_id)
  return hdfs_file.get_current_task_file_list(file_list=file_list, task_num=task_num, task_id=task_id,
                                              allow_empty=False,
                                              partition=start_time + '-' + end_time
                                              )