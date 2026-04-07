import json
schema = open('../online-training/model_gen_inet/imagese/imagese_v51/inet_imagese_v51.schema').read()

names = schema.split(',')
for line in open("../model_debug/fake_data/13"):
    xx = {}
    for i in range(len(names)):
        xx[names[i]] = line.split('\t')[i]
    print xx['user_session_goods_ranked_list']
    if xx['split_query_ner_brand'] in ('0', '-1', ''):
        continue
    with open('sample.json', 'w') as f:
        json.dump(xx, f, indent=0, separators=(',', ':'))
    break