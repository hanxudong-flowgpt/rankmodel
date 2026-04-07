algoparamstr=""
debug_dir="../model_debug/"
echo 'start ps0'
python bootstrap/SeqPointCotrainBootStrap.py --platform=local --debug_dir=${debug_dir} --algoparamstr=${algoparamstr} --task_index=0 --job_name=ps --worker_hosts=127.0.0.1:2222,127.0.0.1:3333,127.0.0.1:4444 --ps_hosts=127.0.0.1:1111 > ${debug_dir}log/ps.0 2>&1 &