import json

fea_conf = json.load(open('../jsonfile/pdd_damodel_conf_v42.json'))
#fea_names = []
fea_conf_map = {}
tags = []
fnames = []
for fconf in fea_conf["others"] + fea_conf["features"]:
    #fea_names.append(fconf['feature_name'])
    fea_conf_map[fconf['feature_name']] = fconf
    fnames.append(fconf['feature_name'])
    if 'tag' in fconf:
        tags.append(fconf['tag'])
    else:
        tags.append(0)


mc = json.load(open('../jsonfile/mc_pdd_damodel_rank_ctr_v432.json'))
feature_need = [x["column_name"] for x in mc['user_columns']+mc['query_columns']+mc['item_columns']+mc['cross_columns']]

examples = []
for line in open("part_v42.txt"):
    examples.append({})
    datas = [vv.strip() for vv in line.strip().split('\t')]
    for k, v in zip(fnames, datas):
        if k in feature_need:
            examples[-1][k] = v

json.dump(examples, open("final_examples.json", "w"))
