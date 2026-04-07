from abc import abstractmethod, ABCMeta
import numpy as np
import tensorflow as tf
import os

def info(x):
    print(x)

class Context(object):

    def __init__(self):
        self.ps_num = ''
        self.worker_num = ''
        self.ckp_dir = ''
        self.work_dir = ''
        self.param = ''
        self.default_fs = ''
        self.http_nn1 = ''
        self.http_nn2 = ''
        self.ps_hosts = ''
        self.worker_hosts = ''
        self.upstream_schema = ''

    def set_ps_num(self, ps_num):
        self.ps_num = ps_num

    def get_ps_num(self):
        return self.ps_num

    def set_worker_num(self, worker_num):
        self.worker_num = worker_num

    def get_worker_num(self):
        return self.worker_num

    def set_ckp_dir(self, ckp_dir):
        self.ckp_dir = ckp_dir

    def get_ckp_dir(self):
        return self.ckp_dir

    def set_work_dir(self, work_dir):
        self.work_dir = work_dir

    def get_work_dir(self):
        return self.work_dir

    def set_param(self, param):
        self.param = param

    def get_param(self):
        return self.param

    def set_default_fs(self, default_fs):
        self.default_fs = default_fs

    def get_default_fs(self):
        return self.default_fs

    def set_http_nn1(self, http_nn1):
        self.http_nn1 = http_nn1

    def get_http_nn1(self):
        return self.http_nn1

    def set_http_nn2(self, http_nn2):
        self.http_nn2 = http_nn2

    def get_http_nn2(self):
        return self.http_nn2

    def set_ps_hosts(self, ps_hosts):
        self.ps_hosts = ps_hosts

    def get_ps_hosts(self):
        return self.ps_hosts

    def set_worker_hosts(self, worker_hosts):
        self.worker_hosts = worker_hosts

    def get_worker_hosts(self):
        return self.worker_hosts

    def set_upstream_schema(self, upstream_schema):
        self.upstream_schema = upstream_schema

    def get_upstream_schema(self):
        return self.upstream_schema


class DataStream(object):

    def __init__(self, path):
        self.path = path
        global file_path
        file_path = path

    def start_service(self, thread_num, batch_size, parse_fn, kwargs):
        self.batch_size = batch_size
        self.parse_fn = parse_fn
        self.kwargs = kwargs
        if self.path != None and os.path.exists(self.path):
            print("Local execute mode")
            with open(self.path, 'r') as f:
                self.content = f.readlines()
            self.cnt = 0
        else:
            print("Stream execute mode")

    def read(self):
        res = []
        while len(res) < self.batch_size:
            line = self.content[self.cnt]
            self.cnt += 1
            l = line.rstrip('\n')
            kv = self.string2kv(l, ',', '=')
            record = []
            upstream_schema = self.kwargs['source_schema']
            schema_list = upstream_schema.split(',')
            for field in schema_list:
                record.append(kv[field])
            res.append(self.parse_fn('\t'.join(record), **self.kwargs))
        return list(map(list, np.transpose(res)))

    def string2kv(self, s, d1, d2):
        kv = {}
        for ele in s.split(d1):
            pair = ele.split(d2)
            if len(pair) != 2:
                continue
            kv[pair[0]] = pair[1]
        return kv

    def output(self, fields_list):
        print(fields_list)
        return None


def get_all_file_list(project, table, partition):
    l = []
    l.append(file_path)
    return l

class BootStrap(object):
    def __init__(self):
        pass
    def finishedWorkers(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def buildGraph(self, task_id, context):
        pass
    @abstractmethod
    def createSession(self, target, task_id):
        pass
    @abstractmethod
    def runSession(self, mon_sess, task_id, thread_ind):
        pass
    def start(self, ckp_dir=None, upstream_schema=None, data_file=None):
        context = Context()
        context.set_ckp_dir(ckp_dir)
        context.set_upstream_schema(upstream_schema)
        context.set_ps_num(1)
        context.set_worker_num(1)
        data_stream = DataStream(data_file)
        #self.run(target=None, data_stream=data_stream, task_id=0, context=context)
        self.buildGraph(task_id=0, context=context)
        sess = self.createSession(target=None, task_id=0)
        self.runSession(sess,task_id=0, thread_ind=0)

class OnBlink(BootStrap):
  def __init__(self, build_graph_fn, create_sess_fn, run_sess_fn, **kwargs):
    self._build_graph_fn = build_graph_fn
    self._create_sess_fn = create_sess_fn
    self._run_sess_fn = run_sess_fn
    self._finished_workers_fn = kwargs['finished_workers_fn']

  def buildGraph(self, task_id, context):
    self._build_graph_fn(task_id, context=context)

  def createSession(self, target, task_id):
    return self._create_sess_fn(target, task_id)

  def runSession(self, mon_sess, task_id, thread_ind):
    self._run_sess_fn(mon_sess, task_id, thread_ind=thread_ind)
