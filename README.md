### 模型代码

##### 本地调试环境准备(数据)
* 在~目录下新建mycodes目录并进入
* git clone 本repo
* mkdir -p model_debug/ckpt
* mkdir -p model_debug/fake_data
* mkdir -p model_debug/log
* tunnel down 命令下载样本到model_debug/fake_data

##### 本地调试环境准备(docker)
* mac上docker下载安装
* docker下载 docker pull reg.docker.alibaba-inc.com/qiaohan/debugtf:rtpop
* 运行docker: sudo docker run -v ~/mycodes/:/opt/mycodes -it reg.docker.alibaba-inc.com/qiaohan/debugtf:rtpop bash
* docker中cd 到 /opt/mycodes即可看到代码目录和model_debug

##### pai平台运行方法
* 注意pai job运行在odps上，一般都是张北机房，所以hdfs最好也用张北的，不然会很慢
* 建议参数，worker cpu不少于1800，server mem不少于22000，server数量50左右

```bash
bizdate=$1
yesterday=$2

ODPSCMD="/opt/taobao/tbdpapp/odpswrapper/odpswrapper.py"

runLCMD() {
    local cmdstr=$(cat "$@")
    echo "$cmdstr"    
    $ODPSCMD -e "$cmdstr"
}
#set odps.task.required.cluster.features=fuxi_gpu;
workerNum=50
psNum=40
gpu=0
workerCpu=200
workerMem=12000
serverCpu=800
serverMem=24000

#entryFile='main_cotrain_predict_pai.py'
entryFile='da_cotrain_predict.py'
jobname='kd_hx_2u1i_v0'
poraid='16720'
tables="odps://search_offline/tables/lsc_log_wide_wdlfg_ctr/ds=${bizdate},odps://search_offline/tables/lsc_log_wide_wdlfg_cvr/ds=${bizdate},odps://search_offline/tables/lsc_log_wide_fg_ctr_negative/ds=${bizdate}"
tables="${tables},odps://search_offline/tables/lsc_log_wide_wdlfg_ctr/ds=${yesterday},odps://search_offline/tables/lsc_log_wide_wdlfg_cvr/ds=${yesterday},odps://search_offline/tables/lsc_log_wide_fg_ctr_negative/ds=${yesterday}"
#algoparamstr="cotrain_conf=cotrain_conf_recall_v3.json,train_parts=${bizdate},eval_parts=${bizdate},optimizer=AdagradDecay,decay_step=3000000,decay_rate=0.95,max_checkpoints_to_keep=8,save_checkpoint_secs=2400,init_op=constant,save_dir=hdfs://et2prod2/pora/et2prod2/save/kdtfmodel/${jobname}/"
algoparamstr="save_dir=hdfs://et2prod2/pora/et2prod2/save/${poraid}/${jobname}/,cotrain_conf=kd_da_cotrain_conf_recall_v2.json,train_parts=20181223,eval_parts=20181224,attention_l2_reg=1e-6,dnn_l2_reg=1e-6,optimizer=AdagradDecay,decay_step=3000000,decay_rate=0.95,max_checkpoints_to_keep=8,save_checkpoint_secs=3600,init_op=constant,model_type=kd_recall_dial,enable_file_dynamic=on,share_user_vec=off,share_item_vec=on"

runLCMD <<HERE
pai -name tensorflow140_lite -project algo_public_dev -DuseSparseClusterSchema=true
-Dbuckets='oss://kongdu/?role_arn=acs:ram::1218002761110343:role/kdsearch&host=cn-hangzhou.oss-internal.aliyun-inc.com' 
-Dscript="odps://search_offline_dev/resources/kd_search_model.tar.gz" 
-DentryFile="./${entryFile}" 
-Dtables="${tables}"
-Dcluster="{\"worker\":{\"count\":$workerNum,\"cpu\":$workerCpu,\"memory\":$workerMem}, \"ps\":{\"count\":$psNum,\"cpu\":$serverCpu,\"memory\":$serverMem}}" 
-DuserDefinedParameters=" --protocol=grpc --platform='pai' --debug_dir='oss://kongdu/${jobname}/' --algoparamstr='${algoparamstr}'";

HERE
```