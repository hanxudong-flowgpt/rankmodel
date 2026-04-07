git clone http://pdd_search_deploy:yKdZeMBeQV1Pexi5hfxz@gitlab.hutaojie.com/yangyouji/da-model.git
cd da-model
git checkout yyj_dev
cd codes

hadoop fs -mkdir -p hdfs://searchhdfs/data/search/yangyouji/ltroffline_v11/ckpt/finishedworkers/
hadoop fs -chmod -R 777 hdfs://searchhdfs/data/search/yangyouji/ltroffline_v11

python bootstrap/SeqPointCotrainBootStrap.py \
--job_conf jsonfile/yyj_exp_pdd_ltr_v1.json \
--debug_dir hdfs://searchhdfs/data/search/yangyouji/ltroffline_v11/ckpt/ \
--save_dir hdfs://searchhdfs/data/search/yangyouji/ltroffline_v11/ckpt/ \
--platform tony $@