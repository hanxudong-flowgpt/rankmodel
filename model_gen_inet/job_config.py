import yaml
from pprint import pprint
import os, sys
from datetime import datetime, timedelta


class JobConfig:
    def __init__(self, conf_path):
        self.conf = None
        with open(conf_path, "r") as f:
           try:
               self.conf = yaml.safe_load(f)
               pprint(self.conf)
           except yaml.YAMLError as ex:
               print("failed to parse job config: {}".format(conf_path))
               print(ex)

    def hdfs_exists(self, file_path):
        cmd = "hadoop fs -test -e \"{}\"".format(file_path.replace("\\", ""))
        code = os.system(cmd)
        if code == 0:
            return True
        return False

    def ensure_train_valid(self):
        k = "train_and_eval"
        assert self.conf[k] is not None
        te = self.conf["train_and_eval"]
        fields = ["training_path", "graph_path", "op_info_path"]
        for f in fields:
            assert te[f] is not None, "{}.{} is not defined in yaml".format(k, f)
            assert self.hdfs_exists(te[f]), "{} not found in hdfs".format(te[f])

    def ensure_export_valid(self):
        k = "export"
        assert self.conf[k] is not None
        e = self.conf["export"]
        fields = ["graph_path", "serving_sig_path", "done_file_path", "saved_model_dir"]
        for f in fields:
            assert e[f] is not None, "{}.{} is not defined in yaml".format(k, f)
            assert self.hdfs_exists(e[f]), "{} not found in hdfs".format(e[f])

    def ensure_daily_valid(self):
        k = "daily"
        assert self.conf[k] is not None
        e = self.conf[k]
        fields = ["daily_train_path"]
        dt = datetime.now() - timedelta(days=1)
        for f in fields:
            assert e[f] is not None, "{}.{} is not defined in yaml".format(k, f)
            assert self.hdfs_exists(e[f].format(dt)), "{} not found in hdfs".format(e[f].format(dt))

        k = "train_and_eval"
        assert self.conf[k] is not None
        te = self.conf["train_and_eval"]
        fields = ["graph_path", "op_info_path"]
        for f in fields:
            assert te[f] is not None, "{}.{} is not defined in yaml".format(k, f)
            assert self.hdfs_exists(te[f]), "{} not found in hdfs".format(te[f])


if __name__ == "__main__":
    conf = JobConfig("./inet_v5_exp_job_config.yaml")
    conf.ensure_train_valid()
