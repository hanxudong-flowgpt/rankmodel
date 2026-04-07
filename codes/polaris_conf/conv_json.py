import json

schema = open('../online-training/model_gen_inet/imagese/imagese_v51/inet_imagese_v51.schema').read()
fg_conf_list = json.load(open('../online-training/model_gen_inet/imagese/imagese_v51/inet_imagese_v51.json'))["features"]
fg_conf = {}
for fg in fg_conf_list:
    if "feature_name" not in fg:
        continue
    fg_conf[fg["feature_name"]] = fg
name2tag = json.load(open('../online-training/model_gen_inet/imagese/imagese_v51/name_tag_map.json'))

features = []

for f_name in schema.split(','):
    if f_name in fg_conf:
        fea_map = fg_conf[f_name]
        if fea_map["feature_type"] == "id_feature":
            fea_map["feature_type"] = "sparse"
            fea_map["value_type"] = "string"
            fea_map["embedding_sparse"] = {"bucket_size": fea_map["hash_bucket_size"],
                                           "dimension": fea_map["embedding_dimension"],
                                           "combiner": "mean", "name_scope": ""}
            fea_map.pop("hash_bucket_size")
            fea_map.pop("embedding_dimension")
        elif fea_map["feature_type"] == "sequence":
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
        if f_name in name2tag:
            fea_map["tag"] = name2tag[f_name]
            fea_map["embedding_sparse"]["name_scope"] = str(name2tag[f_name])
        else:
            print("%s not found tag" % f_name)
    else:
        print("%s not found json" % f_name)
        fea_map = {
            "tag":-1,
            "feature_name": f_name,
            "feature_type": "sparse",
            "value_type": "string",
            "embedding_sparse": {"bucket_size": 10,"dimension": 16, "combiner": "mean", "name_scope": ""}
        }
    features.append(fea_map)
json.dump({"features": features}, open('test.json', 'w'))