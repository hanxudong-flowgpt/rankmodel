import os,sys,subprocess,time

assert len(sys.argv) > 1
mode = sys.argv[1]

src_dir = "./codes/"
main_script = "./codes/bootstrap/SeqPointRecallBS.py"
conf_file = "./xmlfile/tony_damodel_v31.xml"
# day update
from datetime import datetime, date, timedelta
yesterday = (date.today() + timedelta(days = -1)).strftime("%Y-%m-%d")
input_path_day = yesterday

# train parameters
job_conf = "jsonfile/yyj_exp_pdd_damodel_v31.json"
if mode == 'predict':
    job_conf = "jsonfile/yyj_exp_pdd_damodel_v31_predict.json"
elif mode == "vec":
    job_conf = "jsonfile/yyj_exp_pdd_damodel_v31_vec.json"

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

#work_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v31_workspace/'
#log_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v31_workspace/'

work_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v31_workspace_2/'
log_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v31_workspace_2/'


#### submit job

def eprint(x):
    sys.stderr.writelines(x+'\n')

def submit_job(train_parts, timestamp):
    task_params = ["--debug_dir %s" % log_space,
                   "--platform tony",
                   "--mode %s" % "train" if mode == "train" else "predict",
                   "--job_conf %s" % job_conf,
                   "--save_dir %s" % work_space]
    if train_parts:
        task_params.append("--train_parts %s" % train_parts)
        task_params.append("--eval_parts %s" % train_parts)
    if mode == 'predict':
        task_params.append("--tableoutput %s/" % work_space+"predict/"+train_parts)
    if mode == 'vec':
        task_params.append("--tableoutput %s/" % work_space+"vector/"+timestamp)
    commond = ["sh /data/common/barathrum/1.0.23/start-tony.sh",
                "-task_params=\"%s\""%" ".join(task_params),
                "-src_dir=%s"%src_dir,
                "-executes=%s"%main_script,
                "-conf_file=%s"%conf_file,
                "-python_binary_path=python2"
              ]
    logfile=time.strftime('%Y.%m.%d-%H.%M.%S',time.localtime(time.time()))
    os.system("nohup "+" ".join(commond) + " > log/log.%s &"%logfile)
    time.sleep(20)
    # os.system( "tail -f log/log.%s "%logfile )
    for line in open("log/log.%s"%logfile):
        if '.tony/application_' in line:
            return line.split('/')[-1], 'log/log.%s' % logfile
    return None, 'log/log.%s' % logfile

if __name__ == "__main__":
    pt = None
    if len(sys.argv) > 2:
        pt = sys.argv[2]
    ts = None
    if len(sys.argv) > 3:
        ts = sys.argv[3]
    _, logf = submit_job(pt, ts)
    os.system("tail -f %s " % logf)