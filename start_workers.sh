debug_dir="../model_debug/"
algoparamstr="batch_size=3,save_dir=${debug_dir}/ckpt/,job_conf=jsonfile/yyj_exp_pdd_damodel_v1.json,train_parts=20181223,eval_parts=20181224,attention_l2_reg=1e-6,dnn_l2_reg=1e-6,optimizer=AdagradDecay,decay_step=3000000,decay_rate=0.95,max_checkpoints_to_keep=8,save_checkpoint_secs=3600,init_op=constant,model_type=kd_recall_dial,enable_file_dynamic=on,share_user_vec=off,share_item_vec=on"


echo 'strat worker2'
python bootstrap/SeqPointCotrainBootStrap.py --debug_dir=../model_debug/ --platform=local --algoparamstr=${algoparamstr} --task_index=2 --job_name=worker --worker_hosts=127.0.0.1:2222,127.0.0.1:3333,127.0.0.1:4444 --ps_hosts=127.0.0.1:1111 > ${debug_dir}log/worker.2 2>&1 &


echo 'strat worker1'
python bootstrap/SeqPointCotrainBootStrap.py --debug_dir=../model_debug/ --platform=local --algoparamstr=${algoparamstr} --task_index=1 --job_name=worker --worker_hosts=127.0.0.1:2222,127.0.0.1:3333,127.0.0.1:4444 --ps_hosts=127.0.0.1:1111 > ${debug_dir}log/worker.1 2>&1 &


echo 'start worker0'
python bootstrap/SeqPointCotrainBootStrap.py --debug_dir=../model_debug/ --platform=local --algoparamstr=${algoparamstr} --task_index=0 --job_name=worker --worker_hosts=127.0.0.1:2222,127.0.0.1:3333,127.0.0.1:4444 --ps_hosts=127.0.0.1:1111 > ${debug_dir}log/worker.0 2>&1 &
