# coding=utf-8
import os
from datetime import date
from datetime import datetime
from random import shuffle
import tensorflow as tf
import logging
import json

def date_delta(dt, delta):
  date_pattern = '%Y-%m-%d'
  return (date.fromordinal(datetime.strptime(dt, date_pattern).toordinal() + delta)).strftime(date_pattern)

def glob_file_hdfs_one_day(path):
  def _foreach(_path) :
    try:
      for doc in tf.gfile.ListDirectory(_path):
        if "part" in doc:
          yield _path + '/' + doc
    except :
      logging.info("Could not read" +  _path)
  return _foreach(path)
def glob_file_train_hdfs(directory_pattern, **kwargs):
  def _get_data_file_list(key):
    day_str_or_day_lst = kwargs.get(key, None)
    if isinstance(day_str_or_day_lst, str):
      data_list = glob_file_hdfs_one_day(directory_pattern % day_str_or_day_lst)
    elif isinstance(day_str_or_day_lst, list):
      data_list = sum(glob_file_hdfs_one_day(directory_pattern % day) for day in day_str_or_day_lst)
    else:
      data_list = []
    return sorted(data_list)
  train_data_list    = _get_data_file_list("train")
  test_one_data_list     = _get_data_file_list("test_one")
  test_two_data_list = _get_data_file_list("test_two")
  num_worker = kwargs.get("num_worker", 1)
  def _get_worker_data_list(worker_id):
    if num_worker > 10: # 考虑current_day_inf, test_day_inf, test_two_day_inf
      num_worker_on_test_data = min(1, len(test_one_data_list)) + min(1, len(test_two_data_list))
      num_other_worker        = num_worker - num_worker_on_test_data
      avg_size = int(0.99 + len(train_data_list) * 1.0 / num_other_worker)
      if 1 == worker_id:
        if len(test_one_data_list) >= 1:
          shuffle(test_one_data_list)
          return test_one_data_list[:avg_size+1], "test"
        else:
          wid = max(worker_id-num_worker_on_test_data, 0)
          return train_data_list[wid:-1:num_other_worker], "test" if wid == 0 else "train"
      elif 2 == worker_id:
        shuffle(test_two_data_list)
        return test_two_data_list[:avg_size+1], "test"
      else:
        wid = max(worker_id-num_worker_on_test_data, 0)
        return train_data_list[wid:-1:num_other_worker], "test" if wid == 0 else "train"
      pass
    elif num_worker > 0: # 仅仅考虑训练
      return train_data_list[worker_id:num_worker:-1], "train"
  ans = {}
  for wid in range(num_worker):
    pathes, tag = _get_worker_data_list(wid)
    ans[wid] = {
      "pathes": pathes,
      "tag":    tag
    }
  print(json.dumps(ans, indent=True))
  return ans
    

  


if __name__ == '__main__':
  start_day = '2019-10-10'
  glob_file_train_hdfs(
    "hdfs://yiran-data-ns/user/search/rank_data/yaoshi/search/openrank/%s/example/tfrecord/ctr",
    train = start_day,
    test_one = date_delta(start_day,1),
    test_two = date_delta(start_day,2),
    num_worker = 30

  )