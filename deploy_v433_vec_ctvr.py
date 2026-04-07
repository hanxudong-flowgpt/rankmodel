import os,sys,subprocess,time
import xml.etree.ElementTree as ET


def set_xml_value(et, name, value):
    for x in et.getroot():
        if x[0].text == name:
            x[1].text = value
            return
    raise ValueError("set xml:%s not found in xml" % name)


def get_xml_value(et, name):
    for x in et.getroot():
        if x[0].text == name:
            return x[1].text
    raise ValueError("get xml:%s not found in xml" % name)


src_dir = "./codes/"
main_script = "./codes/bootstrap/SeqPointCorunVecRecallBS.py"
conf_file = "./xmlfile/tony_pmlimage_12p120w_smallps.xml"
tony_et = ET.parse(conf_file)
# day update
from datetime import datetime, date, timedelta
yesterday = (date.today() + timedelta(days = -1)).strftime("%Y-%m-%d")
input_path_day = yesterday

assert len(sys.argv) > 1

mode = sys.argv[1]

# train parameters
job_conf = "jsonfile/yyj_exp_pdd_damodel_v433_ctvr.json"
if mode == 'predict':
    job_conf = "jsonfile/yyj_exp_pdd_damodel_v4_predict.json"
elif mode == "vec":
    job_conf = "jsonfile/yyj_exp_pdd_damodel_v4_vec.json"

work_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v433_ctvr_workspace/'
log_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v433_ctvr_workspace/'
if mode == 'predict':
    #work_space += "testpredict/"
    #log_space += "testpredict/"
    set_xml_value(tony_et, "tony.worker.instances", "2")
set_xml_value(tony_et, "tony.tensorboard.log.dir", work_space)
os.system("hadoop fs -rm -r %s" % log_space + "finishedworkers")
#### submit job

def eprint(x):
    sys.stderr.writelines(x+'\n')

def submit_job(train_parts, timestamp):
    if train_parts == 'pms' or mode == "eval":
        set_xml_value(tony_et, "tony.worker.instances", "2")
    task_params = ["--debug_dir %s" % log_space,
                   "--platform tony",
                   "--mode %s" % mode,
                   "--job_conf %s" % job_conf,
                   "--save_dir %s" % work_space]
    if train_parts:
        task_params.append("--train_parts %s" % train_parts)
        task_params.append("--eval_parts %s" % train_parts)
    if mode == 'predict':
        task_params.append("--tableoutput %s/" % work_space+"predict/"+train_parts)
    if mode == 'vec':
        task_params.append("--tableoutput %s/" % work_space+"saved_model_vec/"+timestamp+"/vector/")
    tony_et.write("tony.xml", encoding="utf-8", xml_declaration=True)
    commond = ["sh /data/common/barathrum/1.0.23/start-tony.sh",
                "-task_params=\"%s\""%" ".join(task_params),
                "-src_dir=%s"%src_dir,
                "-executes=%s"%main_script,
                "-conf_file=tony.xml",
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
    ts = str(int(time.time()))
    if len(sys.argv) > 3:
        ts = sys.argv[3]
    _, logf = submit_job(pt, ts)
    os.system("tail -f %s " % logf)