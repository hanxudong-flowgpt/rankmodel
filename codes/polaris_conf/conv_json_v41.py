import json

# schema = open('fgv41.schema').read()
fg_conf_list = json.load(open('featurev41.json'))["features"]
fg_conf = {}
for fg in fg_conf_list:
    if "feature_tag" not in fg:
        print(fg)
        print("feature_tag not found")
        if fg["tag"] == -1:
            fg["tag"] = int(fg["feature_name"])
            fg["feature_name"] = "-1"
            fg["feature_tag"] = "-1"
        continue
    # print("%s found feature_tag" % fg["feature_tag"])
    fg_conf[fg["feature_tag"]] = fg
# name2tag = json.load(open('../online-training/model_gen_inet/imagese/imagese_v51/name_tag_map.json'))

features = []

for tagxxx in open('fgv42.schema'):
    f_name = tagxxx.split(":")[-1].strip()
    if f_name in fg_conf:
        fea_map = fg_conf[f_name]
        '''
        if fea_map["feature_type"] == "id_feature":
            fea_map["feature_type"] = "sparse"
            fea_map["value_type"] = "string"
            fea_map["embedding_sparse"] = {"bucket_size": fea_map["hash_bucket_size"],
                                           "dimension": fea_map["embedding_dimension"],
                                           "combiner": "mean", "name_scope": ""}
            fea_map.pop("hash_bucket_size")
            fea_map.pop("embedding_dimension")
        elif fea_map["feature_type"] == "sequence":
            print(f_name)
            fea_map["embedding_sparse"] = {"bucket_size": fea_map["hash_bucket_size"],
                                           "dimension": fea_map["embedding_dimension"],
                                           "combiner": "mean", "name_scope": ""}
            fea_map.pop("hash_bucket_size")
            fea_map.pop("embedding_dimension")
        elif fea_map["value_type"] == "Double":
            fea_map["feature_type"] = "dense"
            fea_map["value_type"] = "double"
        else:
            print("%s not recognize feature json" % f_name)

        fea_map["tag"] = int(f_name)
        fea_map["embedding_sparse"]["name_scope"] = f_name
        '''
    else:
        print("%s not found json" % f_name)
        fea_map = {
            "tag": int(f_name),
            "feature_name": "-1",
            "feature_tag": "-1",
            "feature_type": "sparse",
            "value_type": "string",
            "embedding_sparse": {"bucket_size": 10,"dimension": 16, "combiner": "mean", "name_scope": ""}
        }
    features.append(fea_map)
json.dump({"features": features}, open('featurev42.json', 'w'))