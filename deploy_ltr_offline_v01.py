import os,sys,subprocess,time
from subprocess import Popen, PIPE

src_dir = "./codes/"
main_script = "./codes/bootstrap/SeqPointCotrainBootStrap.py"
conf_file = "./xmlfile/tony_ltr_v0.xml"

# day update
from datetime import datetime, date, timedelta
assert len(sys.argv) == 2
mode = sys.argv[1]

# train parameters
job_conf = "jsonfile/yyj_exp_pdd_ltr_v1.json"


work_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/ltr_ckpt/mainse/ltr_offline_v01_workspace/'
log_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/ltr_ckpt/mainse/ltr_offline_v01_workspace/'

def eprint(x):
    sys.stderr.writelines(x+'\n')

def submit_job(train_parts):
    task_params = ["--debug_dir %s" % log_space,
                   "--platform tony",
                   "--mode %s" % mode,
                   "--train_parts %s" % train_parts,
                   "--eval_parts %s" % train_parts,
                   "--job_conf %s" % job_conf,
                   "--multi_val_split ,",
                   "--seq_val_split ,",
                   "--save_dir %s" % work_space]
    if mode == 'predict':
        task_params.append("--tableoutput %s/" % work_space+"predict/"+train_parts)
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
    _, logf = submit_job('2019-11-28')
    os.system("tail -f %s " % logf)