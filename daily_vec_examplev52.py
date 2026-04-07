import os,sys,subprocess,time
from subprocess import Popen, PIPE

# day update
from datetime import datetime, date, timedelta
assert len(sys.argv) == 3
start_time = sys.argv[1]
dep_file = sys.argv[2]
log_space = None
for line in open(dep_file):
    if 'log_space=' in line or 'log_space =' in line:
        log_space = line.split('=')[-1].strip()[1:-1]
print(log_space)
assert log_space is not None

def eprint(x):
    sys.stderr.writelines(x+'\n')


def submit_job(train_parts, mode):
    logfile=str(int(time.time()))
    os.system("python2.7 %s %s %s %s"%(dep_file, mode, train_parts, logfile))
    time.sleep(20)
    # os.system( "tail -f log/log.%s "%logfile )
    for line in open("log/log.%s"%logfile):
        if '.tony/application_' in line:
            return line.split('/')[-1], 'log/log.%s' % logfile
    return None, 'log/log.%s' % logfile


year, month, day = [int(x) for x in start_time.split('-')]
date_time = datetime(year=year, month=month, day=day, hour=6, minute=0)
while True:
    files_list = []
    resp = Popen("hadoop fs -ls -R hdfs://hq3-data-ns/apps/nothive/warehouse/search/polaris_inet_data/yangyouji/domain_adap/v52/%d-%02d-%02d/domain/all_domain_v5/csv/event/ctvr/" %
                 (date_time.year, date_time.month, date_time.day),
                 shell=True, stdout=PIPE, stderr=PIPE).stdout.readlines()
    for i in range(len(resp)):
        parts = resp[i].strip().split()
        if len(parts) == 8 and parts[7].split('/')[-1].startswith("part"):
            files_list += [parts[7]]
    if len(files_list) == 0:
        eprint('dir %d-%02d-%02d file dump not starting...' %
              (date_time.year, date_time.month, date_time.day))
        time.sleep(60*10) # 10 min
        continue
    else:
        for file_name in files_list:
            if '.' in file_name:
                eprint('dir %d-%02d-%02d file dump not end...' %
                      (date_time.year, date_time.month, date_time.day))
                time.sleep(60 * 10)  # 10 min
                continue
        eprint('part %d-%02d-%02d start train...' %
              (date_time.year, date_time.month, date_time.day))
        appname, logf = submit_job('%d-%02d-%02d' %
                                  (date_time.year, date_time.month, date_time.day), 'train')
        eprint('log file for this round: %s and appname: %s' % (logf, appname))
        t0 = time.time()
        while True:
            if time.time() - t0 > 5 * 60 * 60:
                eprint('train for 5 hour, we have to stop...')
                os.system("yarn application -kill %s" % appname)
                break
            time.sleep(60 * 10)  # 10 min
            resp = Popen("hadoop fs -cat %s" % log_space + "finishedworkers/worker_0",
                         shell=True, stdout=PIPE, stderr=PIPE).stdout.readlines()
            if len(resp) > 0 and 'finish' in resp[-1]:
                eprint('finish train')
                os.system("yarn application -kill %s" % appname)
                break
            else:
                eprint('part %d-%02d-%02d still train...' %
                      (date_time.year, date_time.month, date_time.day))
        ## vec mode
        appname, logf = submit_job('%d-%02d-%02d' %
                                   (date_time.year, date_time.month, date_time.day), 'vec')
        eprint('log file for this round: %s and appname: %s' % (logf, appname))
        t0 = time.time()
        while True:
            if time.time() - t0 > 2 * 60 * 60:
                eprint('train for 2 hour, we have to stop...')
                os.system("yarn application -kill %s" % appname)
                break
            time.sleep(60 * 10)  # 10 min
        date_time += timedelta(days=1)