import os,sys,subprocess,time
from subprocess import Popen, PIPE

src_dir = "./codes/"
main_script = "./codes/bootstrap/SeqPointCotrainBootStrap.py"
conf_file = "./xmlfile/tony_ltr_v1.xml"

# day update
from datetime import datetime, date, timedelta
assert len(sys.argv) == 2
start_time = sys.argv[1]

# train parameters
job_conf = "jsonfile/yyj_exp_pdd_ltr_v1.json"


source_root = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/flinkdata/pureData'
work_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/ltr_ckpt/mainse/ltr_offline_v1_workspace/'
log_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/ltr_ckpt/mainse/ltr_offline_v1_workspace/'

def eprint(x):
    sys.stderr.writelines(x+'\n')

def submit_job(train_parts):
    task_params = ["--debug_dir %s" % log_space,
                  "--platform tony",
                  "--train_parts %s" % train_parts,
                  "--eval_parts %s" % train_parts,
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
    os.system("nohup "+" ".join(commond) + " > log/log.%s &"%logfile)
    time.sleep(20)
    # os.system( "tail -f log/log.%s "%logfile )
    for line in open("log/log.%s"%logfile):
        if '.tony/application_' in line:
            return line.split('/')[-1], 'log/log.%s' % logfile
    return None, 'log/log.%s' % logfile


year, month, day, hour, minute = [int(x) for x in start_time.split('-')]
date_time = datetime(year=year, month=month, day=day, hour=hour, minute=minute)
while True:
    files_list = []
    resp = Popen("hadoop fs -ls -R %s/%d-%d-%d/hr=%d/min=%d" %
                 (source_root, date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute),
                 shell=True, stdout=PIPE, stderr=PIPE).stdout.readlines()
    for i in range(len(resp)):
        parts = resp[i].strip().split()
        if len(parts) == 8 and parts[7].split('/')[-1].startswith("part"):
            files_list += [parts[7]]
    if len(files_list) == 0:
        eprint('dir %d-%d-%d/hr=%d/min=%d file dump not starting...' %
              (date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute))
        time.sleep(60*5)
        continue
    else:
        for file_name in files_list:
            if file_name.endswith('in-progress') or file_name.endswith('_COPYING_'):
                eprint('dir %d-%d-%d/hr=%d/min=%d file dump not end...' %
                      (date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute))
                time.sleep(60)
                continue
        eprint('part %d-%d-%d/hr=%d/min=%d start train...' %
              (date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute))
        os.system("hadoop fs -rm -r %s" % log_space + "finishedworkers/")
        appname, logf = submit_job('%d-%d-%d/hr=%d/min=%d' %
                                  (date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute))
        eprint('log file for this round: %s and appname: %s' % (logf, appname))
        t0 = time.time()
        while True:
            if time.time() - t0 > 60 * 60:
                eprint('train for 1 hour, we have to stop...')
                os.system("yarn application -kill %s" % appname)
                break
            time.sleep(60)
            resp = Popen("hadoop fs -cat %s" % log_space + "finishedworkers/worker_0",
                         shell=True, stdout=PIPE, stderr=PIPE).stdout.readlines()
            if len(resp) > 0 and 'finish' in resp[-1]:
                eprint('finish train')
                os.system("yarn application -kill %s" % appname)
                break
            else:
                eprint('part %d-%d-%d/hr=%d/min=%d still train...' %
                      (date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute))
        date_time += timedelta(minutes=10)