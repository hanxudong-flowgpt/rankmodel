import os,sys,subprocess,time

src_dir = "./codes/"
main_script = "./codes/bootstrap/SeqPointCotrainBootStrap.py"
conf_file = "./xmlfile/tony_v1.xml"
# day update
from datetime import datetime, date, timedelta
yesterday = (date.today() + timedelta(days = -1)).strftime("%Y-%m-%d")
input_path_day = yesterday

# train parameters
job_conf = "jsonfile/yyj_exp_pdd_damodel_v1.json"

assert len(sys.argv) == 2

if sys.argv[1] == "train":
    a = 1
elif sys.argv[1] == "eval":
    train = "False"
    save_model = 0
elif sys.argv[1] == "day":
    train = "True"
    save_model = 1
    input_path = input_path_day

else:
    sys.exit(1)

work_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/lsc_v1_workspace/ckpt/'
log_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/lsc_v1_workspace/ckpt/'

#### submit job
task_params=[ "--debug_dir %s" % log_space,
              "--platform tony",
              "--job_conf %s" % job_conf,
              "--save_dir %s" % work_space]
commond = ["/data/common/barathrum/1.0.23/start-tony-hq.sh",
            "-task_params=\"%s\""%" ".join(task_params),
            "-src_dir=%s"%src_dir,
            "-executes=%s"%main_script,
            "-conf_file=%s"%conf_file,
            "-python_binary_path=python2"
          ]

logfile=time.strftime('%Y.%m.%d-%H.%M.%S',time.localtime(time.time()))
os.system( "nohup " +  " ".join(commond) + " > log/log.%s &"%logfile )
time.sleep(5)
os.system( "tail -f log/log.%s "%logfile )
