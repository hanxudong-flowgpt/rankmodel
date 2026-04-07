import json
conf1 = json.load(open("pdd_damodel_conf_v32.json"))
conf2 = json.load(open("pdd_damodel_conf_v50.json"))

mc_conf = json.load(open("mc_pdd_damodel_ctr_v43.json"))

fnames = []
for k, v in mc_conf.items():
    for x in v:
        fnames.append(x["column_name"])

map1 = {}
for x in conf1["features"]:
    if "feature_name" in x and x["feature_name"] in fnames:
        map1[x["feature_name"]] = x

map2 = {}
for x in conf2["features"]:
    if "feature_name" in x and x["feature_name"] in fnames:
        map2[x["feature_name"]] = x


assert map1.__len__() == map2.__len__()

for k, v in map1.items():
    if "shared_name" in v and "shared_name" not in map2[k]:
        print v
    if "shared_name" not in v and "shared_name" in map2[k]:
        print v
    if "shared_name" in v and "shared_name" in map2[k] and v["shared_name"] != map2[k]["shared_name"]:
        print v