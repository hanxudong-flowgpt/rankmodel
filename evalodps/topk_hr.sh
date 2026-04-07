#!/usr/bin/env bash

# sh topk_hr.sh 20190525 20190523 search_offline.kd_online_ctr_uvec_top10query search_offline.kd_online_ivec_top10query kd_online_ctr_top10query_top3k search_algo_quality_dev

bizdate=$1
yesterday=$2
uvectable=$3
ivectable=$4
outtablename=$5
project=$6

ODPSCMD="odpscmd"
cur_date="`date +%Y%m%d%H%m%s`"

runLCMD() {
    local cmdstr=$(cat "$@")
    echo "$cmdstr"
    $ODPSCMD -e "$cmdstr"
}
#set odps.task.required.cluster.features=fuxi_gpu;
workerNum=100
psNum=400
gpu=0
workerCpu=1200
workerMem=4000
serverCpu=1200
serverMem=32000

$ODPSCMD -e "create table if not exists ${outtablename}(id string,vector string) partitioned by(ds string) lifecycle 7;"
$ODPSCMD -e "create table if not exists uvec_table_${cur_date} like ${uvectable} lifecycle 7;"
$ODPSCMD -e "create table if not exists ivec_table_${cur_date} like ${ivectable} lifecycle 7;"

$ODPSCMD -e "insert overwrite table uvec_table_${cur_date} partition(ds=${bizdate}) select \`(ds)?+.+\` from ${uvectable} where ds=${bizdate}"
$ODPSCMD -e "insert overwrite table ivec_table_${cur_date} partition(ds=${yesterday}) select \`(ds)?+.+\` from ${ivectable} where ds=${yesterday}"

tableinput="odps://${project}/tables/uvec_table_${cur_date}/ds=${bizdate}"
pool_vec_table="odps://${project}/tables/ivec_table_${cur_date}/ds=${yesterday}"
tableoutput="odps://${project}/tables/${outtablename}/ds=${bizdate}"


itemcnt_info="`$ODPSCMD -e "select count(1) from ivec_table_${cur_date} where ds=${yesterday};" ` "
echo $itemcnt_info
itemcnt="`echo $itemcnt_info | awk -F"----+" '{split($3,dd," ");print dd[3]}'`"
echo "itemcnt ##############"$itemcnt"####################"
if [ $(echo "$itemcnt < 100000 "|bc) = 1 ]; then
    echo -e 'too few items!'
    exit 1
fi
echo "start..."
#outputs=odps://search_algo_quality_dev/tables/kd_hx_2u1i_v2_top10query_res

runLCMD <<HERE
pai -name tensorflow140_lite -project algo_public_dev
    -Dscript='odps://${project}/resources/kd_topk.tar.gz'
    -DentryFile=tf2_dist_for_item_topk.py
    -Dtables="${tableinput},${pool_vec_table}"
    -Doutputs="${tableoutput}"
    -DenableDynamicCluster=true
    -Dcluster="{\"worker\":{\"count\":$workerNum,\"cpu\":$workerCpu,\"memory\":$workerMem}, \"ps\":{\"count\":$psNum,\"cpu\":$serverCpu,\"memory\":$serverMem}}"
    -DuserDefinedParameters=" --item_emb_num=${itemcnt} --checkpointDir=hdfs://na61storage/pora/na61hunbu/train/gul_match/kongdu/topK/m_calibrate_mp_2_mlp_1913_t2_top_k/${bizdate} --model_type=base_match --item_vec_dim=128 --user_vec_dim=128 --ui_recall_group_dim=16 --group_operations=max|mean --selected_columns=item_id,vector --features_extract_from_table=id,vector --top_k=50";
HERE
