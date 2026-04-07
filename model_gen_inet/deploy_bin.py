from __future__ import print_function
import os, sys, subprocess, time
from datetime import datetime, timedelta
from job_config import *
import platform
import getpass

src_dir = "./inet"
main_script = "./inet/run.py"
conf_file = "./conf/tony_yyj.xml"
if getpass.getuser() == "search":
    conf_file = "./conf/tony_search.xml"

export = False
daily = False

assert len(sys.argv) == 3

job_config_path = sys.argv[2]
job_config = JobConfig(job_config_path)

working_dir = job_config.conf["working_dir"]

if sys.argv[1] == "daily_train":
    train = 1
    save_model = 1
    daily = True
    job_config.ensure_daily_valid()
    daily_train_path = job_config.conf["daily"]["daily_train_path"].format(
        datetime.now() - timedelta(days=1))
    print("train using data {}".format(daily_train_path))
elif sys.argv[1] == "train":
    train = 1
    save_model = 1
    job_config.ensure_train_valid()
elif sys.argv[1] == "eval":
    train = 0
    save_model = 0
    job_config.ensure_train_valid()
elif sys.argv[1] == "convert":
    export = True
    job_config.ensure_export_valid()
else:
    print("invalid arg {}".format(sys.argv[1]))
    sys.exit(1)

#### submit job
if export:
    conf_file = "./conf/tony_export.xml"
    export_conf = job_config.conf["export"]
    saved_model_dir = export_conf["saved_model_dir"]
    done_file_path = export_conf["done_file_path"]
    serving_sig_path = export_conf["serving_sig_path"]
    serving_graph_path = export_conf["graph_path"]
    
    task_params = [
            "--export 1",
            "--saved_model_dir %s" % saved_model_dir,
            "--working_dir %s" % working_dir,
            "--done_file_path %s" % done_file_path,
            "--graph_path %s" % serving_graph_path,
            "--serving_sig_path %s" % serving_sig_path,
        ]
else:
    train_eval_conf = job_config.conf["train_and_eval"]
    train_path = train_eval_conf["training_path"]
    eval_path = train_eval_conf["eval_path"]
    graph_path = train_eval_conf["graph_path"]
    op_info_path = train_eval_conf["op_info_path"]
    train_path = train_path if train else eval_path

    if train == 0:
        conf_file = "./conf/tony_eval.xml"
        if "eval_graph_path" in train_eval_conf:
            graph_path = train_eval_conf["eval_graph_path"]
            print("!!!! use graph from {} to eval".format(graph_path))

        if getpass.getuser() == "search":
            conf_file = "./conf/tony_eval_search.xml"

    task_params = [
        "--data_format csv",
        "--working_dir %s" % working_dir,
        "--train %s" % train,
        "--save_model %d" % save_model,
        "--input_file_names %s" % (train_path if not daily else daily_train_path),
        "--graph_path %s" % graph_path,
        "--op_info_path %s" % op_info_path
    ]

if getpass.getuser() == "search":
    tony_script = "/data/common/barathrum/1.0.23/start-tony-search.sh"
else:
    tony_script = "/data/common/barathrum/1.0.23/start-tony-hq.sh"

commond = [
    "sh",
    tony_script,
    "-task_params=\"%s\"" % " ".join(task_params),
    "-src_dir=%s" % src_dir,
    "-executes=%s" % main_script,
    "-conf_file=%s" % conf_file, "-python_binary_path=python2"
]

logfile = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(time.time()))
os.system("nohup " + " ".join(commond) + " > log/log.%s &" % logfile)
os.system("tail -f log/log.%s | awk '/completed/{exit;}{print;}'" % logfile)
