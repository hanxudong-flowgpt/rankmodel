import time
import filesystem
import hdfs_client
from util.tflog import tflogger as logging


def get_all_file_list(project, table, partition, allow_empty=False, base_path='/pora/hdfs-table/'):
  current_path = base_path + project + '/' + table + '/' + partition + '/filelist'
  print "current path:", current_path
  client = hdfs_client.get_hdfs_client()
  hdfs_fs = filesystem.HDFSFilesystem()
  done = hdfs_fs.exists(current_path)
  if not done:
    if allow_empty:
      return []
    else:
      print "file not exist!"
      return None
  with client.read(hdfs_path=current_path) as reader:
    content = reader.read().strip()
    file_list = str(content).split('\n')
  return file_list


def get_current_task_file_list(file_list, task_num, task_id, allow_empty, partition):
  task_file_list = []
  if len(file_list) < task_num and not allow_empty:
    logging.info("%s file num less than task num %s %s" % (str(partition), str(len(file_list)), str(task_num)))
  file_list = sorted(file_list)
  for i in range(len(file_list)):
    if task_id == i % task_num:
      task_file_list.append(file_list[i])
  if len(task_file_list) == 0:
    logging.info("%s len(task_file_list)==0" % (str(partition)))
    time.sleep(999999)
  return task_file_list


def get_task_file_list(project, table, partition, task_num, task_id, allow_empty=False, base_path='/pora/hdfs-table/'):
  file_list = get_all_file_list(project=project, table=table, partition=partition, allow_empty=allow_empty, base_path=base_path)
  return get_current_task_file_list(file_list=file_list, task_num=task_num, task_id=task_id, allow_empty=allow_empty,
                                    partition=partition)

def get_whole_file_list(project, table, partition, task_num, task_id, allow_empty=False, base_path='/pora/hdfs-table/'):
  file_list = get_all_file_list(project=project, table=table, partition=partition, allow_empty=allow_empty, base_path=base_path)
  return file_list 



def hdfs_make_dir(origin_dir, dir):
  hdfs_fs = filesystem.HDFSFilesystem()
  done = hdfs_fs.exists(origin_dir + "/" + dir)
  if not done:
    hdfs_fs.mkdir(origin_dir + "/" + dir)
  return origin_dir + "/" + dir
