import data_pb2
import conf_pb2
import json
import base64
import sys
from google.protobuf import text_format

fea_conf_pb = text_format.Parse(open("domain_v4_pms.conf").read(),conf_pb2.FeatureConf())
tag2type = {}
for x in fea_conf_pb.feature:
    if "@user" not in x.__getattribute__("def") and "@context" not in x.__getattribute__("def"):
        continue
    if "null" in x.__getattribute__("def"):
        print x.__getattribute__("def")
    types = x.__getattribute__("def").split(">>")[0].split("->")[-1]
    for a, b in zip(x.tag, types.split(",")):
        tag2type[int(a)] = b.strip()

idx = int(sys.argv[1])

def conv_features(fea):
    ser = fea.SerializeToString()
    return base64.encodestring(ser).replace("\n", "")


fea_conf = json.load(open('../jsonfile/pdd_damodel_conf_v32.json'))
fea_names = []
fea_conf_map = {}
for fconf in fea_conf["others"] + fea_conf["features"]:
    fea_names.append(fconf['feature_name'])
    fea_conf_map[fconf['feature_name']] = fconf
# context_fea = ['ltr_query_cat_r_list', 'user_tag_gndr', 'user_tag_age', 'user_tag_buy_power', 'user_tag_plat']
context_fea = []
req_set = []
for line in open("dav4_sample.txt"):
    datas = line.split('\t')
    if len(datas) != len(fea_names):
        print(len(datas))
        print(len(fea_names))
        raise ValueError("xx")
    context_f = data_pb2.Features()
    doc_f = data_pb2.Features()
    for k, v in zip(fea_names, datas):
        if k in context_fea:
            ff = context_f.feature.add()
        else:
            ff = doc_f.feature.add()

        ff.tag = fea_conf_map[k]["tag"] if "tag" in fea_conf_map[k] else 0
        if ff.tag not in tag2type:
            # print ff.tag
            continue
        if ff.tag == 1088:
            print v
            print len(v)
        for x in v.split(';'):
            if "int64" in tag2type[ff.tag]:
                try:
                    ff.int64_value.append(long(x))
                except Exception, e:
                    print "%d not format succ"%ff.tag
            elif "bytes" in tag2type[ff.tag]:
                ff.bytes_value.append(x)
            elif "float" in tag2type[ff.tag]:
                try:
                    ff.float_value.append(float(x))
                except Exception, e:
                    print "%d not format succ"%ff.tag
            else:
                raise ValueError("%d %s" % (ff.tag, tag2type[ff.tag]))

    context_f_bytes = conv_features(doc_f)
    doc_f_bytes = conv_features(context_f)
    req = {"request_id": "xxdds", "trace_enabled_module": ["fg_module", "tensor_inference_module"]}
    req["examples"] = {"context": context_f_bytes, "documents": [doc_f_bytes]}
    req_set.append(req)
    break
#json.dump("req1.json", req_set[0])
json.dump(req_set[idx], open("req_dav4.json", "w"))
with open("examples.txt", "w") as f:
    for req in req_set:
        f.write(base64.decodestring(req["examples"]["context"])+"\n")
