#!/usr/bin/env bash


# example : sh pvmodel_predict_uvec.sh kd_hx_dram_fastpv_v2 kd_test_fastpv_v2ctr_uvec 20190525 20190523 search_algo_quality_dev kdquality
jobname=$1
outtablename=$2
bizdate=$3
yes_2day=$4
project=$5
osskey=$6

#jobname='kd_hx_dram_fastpv_v2'
#postfix='_fastpv_v2ctr_pvvec'
#project='search_offline'
#bizdate='20190211'
#yes_2day='20190209'

#ads_dev='/apsarapangu/disk11/cuiqing/console/bin/odpscmd --config=/home/cuiqing.cq/work/console/conf/odps_config.ini.ads_train_dev'
ODPSCMD="odpscmd"

runLCMD() {
    local cmdstr=$(cat "$@")
    echo "$cmdstr"
    $ODPSCMD -e "$cmdstr"
}
#set odps.task.required.cluster.features=fuxi_gpu;
workerNum=200
psNum=40
gpu=0
workerCpu=800
workerMem=18000
serverCpu=1000
serverMem=24000



entryFile='bootstrap/FastPvSeqPointCotrainBootStrap.py'

tables="odps://search_offline/tables/kd_hx_eval_4auc_sample_ctr_v1_pv/ds=${bizdate}"
tables="${tables},odps://search_offline/tables/target_item_feature/ds=${yes_2day}"

algoparamstr="model_name=Recall_CTR,batch_size=1000,predict=on,need_save_ckp=false,save_dir=hdfs://na61storage/pora/na61prod/tfonblink/release/save/kdhx/${jobname}/ckpt4rtp/${yes_2day}/data/,table_name=kd_hx_eval_4auc_sample_ctr_v1_pv,table_project=search_offline,job_conf=jsonfile/kd_pv_cotrain_conf_recall_v1_preduvec.json,train_parts=${bizdate},join_feature_parts=${yes_2day},eval_parts=${bizdate},eval_join_feature_parts=${yes_2day},attention_l2_reg=1e-6,dnn_l2_reg=1e-6,optimizer=AdagradDecay,decay_step=3000000,decay_rate=0.95,max_checkpoints_to_keep=8,save_checkpoint_secs=3600,init_op=constant"

$ODPSCMD -e "create table if not exists ${jobname}_uvec(id string, predict string, label string) partitioned by(ds string) lifecycle 7;"
tableoutput="odps://${project}/tables/${jobname}_uvec/ds=${bizdate}"

runLCMD <<HERE
pai -name tensorflow140_lite_beta_kd -project algo_public_dev -DuseSparseClusterSchema=true
-Dbuckets='oss://kongdu/?role_arn=acs:ram::1218002761110343:role/${osskey}&host=cn-hangzhou.oss-internal.aliyun-inc.com'
-Dscript="odps://${project}/resources/kd_recall_model.tar.gz"
-DentryFile="./${entryFile}"
-Dtables="${tables}"
-Doutputs="${tableoutput}"
-Dcluster="{\"worker\":{\"count\":$workerNum,\"cpu\":$workerCpu,\"memory\":$workerMem}, \"ps\":{\"count\":$psNum,\"cpu\":$serverCpu,\"memory\":$serverMem}}"
-DuserDefinedParameters=" --protocol=grpc --platform='pai' --debug_dir='oss://kongdu/${jobname}/predpvvec/' --tableoutput='${tableoutput}' --algoparamstr='${algoparamstr}'";

HERE

$ODPSCMD -e "create table if not exists ${outtablename}(id string, item_id bigint, vector string, posnet double, predict double) partitioned by(ds string) lifecycle 7;"
$ODPSCMD -e "insert overwrite table ${outtablename} partition(ds=${bizdate}) select max(id), max(item_id), max(vector), max(posnet), max(predict)  from (select id, split_part(id, '_', 1) as rn, split_part(id, '#', 2) as item_id, split_part(label, ',', 1, 128) as vector, split_part(label, ',', 129) as posnet, predict from ${project}.${jobname}_uvec where ds=${bizdate})ta group by rn;"
