import os,sys,subprocess,time
import xml.etree.ElementTree as ET

# day update
from datetime import datetime, date, timedelta
yesterday = (date.today() + timedelta(days = -1)).strftime("%Y-%m-%d")
input_path_day = yesterday

assert len(sys.argv) > 1

mode = sys.argv[1]


ps_num = 1
worker_num = 2

main_script = "bootstrap/SeqPointCorunVecRecallBS.py"

# train parameters
job_conf = "jsonfile/hxd_flowrec_v512_ctvr.json"
if mode == 'predict':
    job_conf = "jsonfile/yyj_exp_pdd_damodel_v4_predict.json"
elif mode == "vec":
    job_conf = "jsonfile/yyj_exp_pdd_damodel_v511_vec.json"

work_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v512_ctvr_workspace/'
log_space = 'hdfs://hq3-data-ns/user/search/rank_data/yangyouji/inet_ckpt/mainse/damodel_v512_ctvr_workspace/'

os.system("rm -rf %s" % log_space + "finishedworkers")
#### submit job

def eprint(x):
    sys.stderr.writelines(x+'\n')

def submit_job(train_parts, timestamp):
    ''''
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
    commond = ["sh /data/common/barathrum/1.0.23/start-tony.sh",
                "-task_params=\"%s\""%" ".join(task_params),
                "-src_dir=%s"%src_dir,
                "-executes=%s"%main_script,
                "-conf_file=tony.xml",
                "-python_binary_path=python2"
              ]
    '''
    ps_hosts = ",".join(["127.0.0.1:%d"%(12000+i) for i in range(ps_num)])
    worker_hosts = ",".join(["127.0.0.1:%d"%(13000+i) for i in range(worker_num)])
    
    logfile=timestamp #time.strftime('%Y.%m.%d-%H.%M.%S',time.localtime(time.time()))
    algoparamstr = ""
    for i in range(ps_num):
        print('start ps%d'%i)
        commond = ["python codes/bootstrap/SeqPointCotrainBootStrap.py",
                    "--platform=local",
                    "--debug_dir=%s" % log_space,
                    "--save_dir=%s" % work_space,
                    "--algoparamstr=%s" % algoparamstr,
                    "--mode=%s" % mode,
                    "--job_conf=%s" % job_conf,
                    "--task_index=%d" % i,
                    "--job_name=ps",
                    "--worker_hosts=%s" % worker_hosts,
                    "--ps_hosts=1%s" % ps_hosts
                ]
        os.system("nohup "+" ".join(commond) + " > %slog/ps%d.log.%s &"%(log_space, i,logfile))
        time.sleep(5)
        # os.system( "tail -f log/log.%s "%logfile )
    #algoparamstr="batch_size=3,save_dir=${debug_dir}/ckpt/,job_conf=jsonfile/yyj_exp_pdd_damodel_v1.json,train_parts=20181223,eval_parts=20181224,attention_l2_reg=1e-6,dnn_l2_reg=1e-6,optimizer=AdagradDecay,decay_step=3000000,decay_rate=0.95,max_checkpoints_to_keep=8,save_checkpoint_secs=3600,init_op=constant,model_type=kd_recall_dial,enable_file_dynamic=on,share_user_vec=off,share_item_vec=on"
    for i in range(worker_num):
        print('start worker%d'%i)
        commond = ["python ./bootstrap/SeqPointCotrainBootStrap.py",
                    "--platform=local",
                    "--debug_dir=%s" % log_space,
                    "--save_dir=%s" % work_space,
                    "--algoparamstr=%s" % algoparamstr,
                    "--mode=%s" % mode,
                    "--job_conf=%s" % job_conf,
                    "--task_index=%d" % i,
                    "--job_name=worker",
                    "--worker_hosts=%s" % worker_hosts,
                    "--ps_hosts=1%s" % ps_hosts
                ]
        os.system("nohup "+" ".join(commond) + " > %slog/worker%d.log.%s &"%(log_space, i,logfile))
        time.sleep(5)
        # os.system( "tail -f log/log.%s "%logfile )
    return None, 'log/log.%s' % logfile

if __name__ == "__main__":
    pt = None
    if len(sys.argv) > 2:
        pt = sys.argv[2]
    ts = str(int(time.time()))
    if len(sys.argv) > 3:
        ts = sys.argv[3]
    _, logf = submit_job(pt, ts)
    #if len(sys.argv) <= 3:
    #    os.system("tail -f %s " % logf)