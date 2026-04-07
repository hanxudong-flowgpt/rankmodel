# -*- coding: utf-8 -*-

import shutil
import subprocess
import sys
import time

reload(sys)
sys.setdefaultencoding("utf-8")

try:
    from tensorflow.python.platform import tf_logging as logging
except ImportError:
    import logging
try:
    from hdfs_client import *
except ImportError:
    print "hdfs_client is not included"


def execit(cmd, timeout=60):
    logging.info('%s', cmd)
    execcmd = subprocess.Popen(cmd, shell=True, stdout=sys.stdout.fileno(),
                               stderr=sys.stdout.fileno(), close_fds=True)
    begin_time = time.time()
    while True:
        ret = execcmd.poll()
        if not ret == None:
            out, err = execcmd.communicate()
            if ret == 0:
                return True
            else:
                logging.warn("run cmd %s output: %s error: %s ", cmd, out, err)
                return False
        if not timeout == None:
            if time.time() - begin_time > timeout:
                logging.warn("run cmd %s timeout ", cmd)
                return False
            time.sleep(0.5)


def execit_raise(cmd, timeout=60):
    logging.info("execit_raise cmd %s" % cmd)
    ret = execit(cmd, timeout)
    if not ret:
        raise Exception('exec [%s] failed, ret=[%s]' % (cmd, ret))


class HDFSFilesystem(object):
    def open_file(self, filename):
        new_client = get_hdfs_client()
        return new_client.write(self._remove_prefix(filename),
                                overwrite=True)

    def exists(self, filename):
        new_client = get_hdfs_client()
        try:
            new_client.status(self._remove_prefix(filename))
            return True
        except Exception, e:
            if 'not exist' in str(e):
                return False
            raise

    def rename(self, src, dst):
        new_client = get_hdfs_client()
        new_client.rename(self._remove_prefix(src),
                          self._remove_prefix(dst))

    def copy(self, src, dst):
        while dst.endswith('/'):
            dst = dst[:-1]
        parent = os.path.split(dst)[0]
        execit_raise('%s/bin/hadoop fs -mkdir -p %s' % (
            self._get_hadoop_home(), parent))
        execit_raise('%s/bin/hadoop fs -cp %s %s' % (
            self._get_hadoop_home(), src, dst))

    def rm(self, filename):
        execit_raise('%s/bin/hadoop fs -rmr %s' % (
            self._get_hadoop_home(), filename))

    def put(self, local_file, hdfs_dir):
        execit_raise('%s/bin/hadoop fs -put -f %s %s' % (
            self._get_hadoop_home(), local_file, hdfs_dir), timeout=1000)

    def get(self, hdfs_dir, local_file):
        execit_raise('%s/bin/hadoop fs -get -f %s %s' % (
            self._get_hadoop_home(), hdfs_dir, local_file), timeout=1000)

    def mkdir(self, data_dir):
        # TODO add lock on data_dir
        execit_raise('%s/bin/hadoop fs -mkdir -p %s' % (
            self._get_hadoop_home(), data_dir))

    def _get_hadoop_home(self):
        if os.environ.get('HADOOP_HOME'):
            return os.environ.get('HADOOP_HOME')
        if os.environ.get('HADOOP_HDFS_HOME'):
            return os.environ.get('HADOOP_HDFS_HOME')
        if os.environ.get('HADOOP_INSTALL'):
            return os.environ.get('HADOOP_INSTALL')
        raise Exception('no hadoop home found in env %s' % str(os.environ))

    # for webhdfs client...
    def _remove_prefix(self, path):
        if path.startswith('hdfs://'):
            return '/' + '/'.join(path.split('/')[3:])
        return path


class LocalFilesystem(object):
    def open_file(self, filename):
        return open(filename, "w")

    def exists(self, filename):
        return os.path.exists(filename)

    def rename(self, src, dst):
        shutil.move(src, dst)

    def copy(self, src, dst):
        shutil.copytree(src, dst)

    def rm(self, filename):
        shutil.rmtree(filename)

    def put(self, local_file, remote_dir):
        shutil.copy2(local_file, remote_dir)

    def get(self, remote_dir, local_file):
        shutil.copy2(remote_dir, local_file)

    def mkdir(self, data_dir):
        os.makedirs(data_dir)
