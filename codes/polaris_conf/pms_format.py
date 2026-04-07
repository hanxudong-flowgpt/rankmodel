import data_pb2
import conf_pb2
import json
import base64
import sys
from google.protobuf import text_format
import hashlib

idx = int(sys.argv[1])

def conv_features(fea):
    ser = fea.SerializeToString()
    #return ser
    return base64.encodestring(ser).replace("\n", "")


fea_conf = json.load(open('../jsonfile/pdd_damodel_conf_v42.json'))
#fea_names = []
fea_conf_map = {}
tags = []
for fconf in fea_conf["others"] + fea_conf["features"]:
    #fea_names.append(fconf['feature_name'])
    fea_conf_map[fconf['feature_name']] = fconf
    if 'tag' in fconf:
        tags.append(fconf['tag'])
    else:
        tags.append(0)

fea_conf_pb = text_format.Parse(open("domain_v4_pms.conf").read(), conf_pb2.FeatureConf())
tag2type = {}
tag2finalname = {}
context_fea = []
doc_fea = []

for x in fea_conf_pb.feature:
    #if "@item" not in x.__getattribute__("def") and "@user" not in x.__getattribute__("def") and "@context" not in x.__getattribute__("def") and "@query" not in x.__getattribute__("def"):
    if '@' not in x.__getattribute__("def"):
        continue
    if "@item" in x.__getattribute__("def"):
        print 'doc:', x.tag
    if "null" in x.__getattribute__("def"):
        print x.__getattribute__("def")

    types = x.__getattribute__("def").split(">>")[0].split("->")[-1]
    for a, b in zip(x.tag, types.split(",")):
        tag2type[int(a)] = b.strip()
        if "@item" not in x.__getattribute__("def"):
            context_fea.append(int(a))
        else:
            doc_fea.append(int(a))
        #if a not in tags:
        #    print a,b
    names = x.__getattribute__("def").split("=")[0]
    for a, b in zip(x.tag, names.split(",")):
        tag2finalname[int(a)] = b.strip()
        #if a not in tags:
        #    print a,b
# context_fea = ['ltr_query_cat_r_list', 'user_tag_gndr', 'user_tag_age', 'user_tag_buy_power', 'user_tag_plat']
req_set = []
fname2features = []
allfname2features = []
for line in open("part_v42.txt"):
    touched_tags = []
    datas = [vv for vv in line.strip().split('\t')]
    fname2features.append({})
    allfname2features.append({})
    if len(datas) != len(tags):
        print(len(datas))
        print(len(tags))
        raise ValueError("xx")
    #for a, k in zip(datas, tags):
    #    fname2features[-1][k] = a
        # print(a.encode('utf-8'))
    '''
    datas.append(fname2features[-1][540])
    tags.append(1)
    fname2features[-1][1] = fname2features[-1][540]
    '''
    context_f = data_pb2.Features()
    doc_f = data_pb2.Features()
    for k, vv in zip(tags, datas):
        allfname2features[-1][k] = vv
        if k not in tag2type:
            # print ff.tag
            continue
        v = vv #.encode('utf-8')
        if k in context_fea:
            ff = context_f.feature.add()
        else:
            ff = doc_f.feature.add()
        fname2features[-1][k] = vv
        ff.tag = k
        touched_tags.append(ff.tag)
        for x in v.split(';'):
            if "int64" in tag2type[ff.tag]:
                try:
                    ff.int64_value.append(long(x))
                except Exception, e:
                    print "%d not format succ"%ff.tag
            elif "bytes" in tag2type[ff.tag]:
                ff.bytes_value.append(x)
            elif "float" in tag2type[ff.tag]:
                #print "%d is float"%ff.tag
                try:
                    ff.float_value.append(float(x))
                except Exception, e:
                    print "%d not format succ"%ff.tag
            else:
                raise ValueError("%d %s" % (ff.tag, tag2type[ff.tag]))

    context_f_bytes = conv_features(context_f)
    doc_f_bytes = conv_features(doc_f)
    req = {"request_id": "xxdds", "trace_enabled_module": ["fg_module", "tensor_inference_module"]}
    req["examples"] = {"context": context_f_bytes, "documents": [doc_f_bytes]}
    req_set.append(req)
    # assert len(touched_tags) == len(tag2type.keys())
    for x in tag2type.keys():
        if x not in touched_tags:
            print x
    #break
#json.dump("req1.json", req_set[0])
context_f_bytes = req_set[idx]["examples"]["context"]
doc_f_bytes = []
doc_f_bytes.append(req_set[idx]["examples"]["documents"][0])
doc_f_bytes.append(req_set[0]["examples"]["documents"][0])
doc_f_bytes.append(req_set[1]["examples"]["documents"][0])
doc_f_bytes.append(req_set[2]["examples"]["documents"][0])
doc_f_bytes.append(req_set[3]["examples"]["documents"][0])
doc_f_bytes.append(req_set[4]["examples"]["documents"][0])
doc_f_bytes.append(req_set[5]["examples"]["documents"][0])
doc_f_bytes.append(req_set[6]["examples"]["documents"][0])
doc_f_bytes.append(req_set[7]["examples"]["documents"][0])


req = {"request_id": "xxdds", "trace_enabled_module": ["fg_module", "tensor_inference_module"]}
req["examples"] = {"context": context_f_bytes, "documents": doc_f_bytes}
json.dump(req, open("req_dav4.json", "w"))
json.dump(fname2features[idx], open("debug.json", "w"))
xxxx = {}
for k, v in fname2features[idx].items():
    xxxx[tag2finalname[k]] = v
json.dump(xxxx, open("debug2.json", "w"))

with open("req_dav4.fea", "w") as f:
    f.write(str(fname2features[idx]))
with open("req_dav4.allfea", "w") as f:
    for i in range(400):
        f.write(str(allfname2features[i]) + '\n')
with open("examples.txt", "w") as f:
    for req in req_set:
        f.write(req["examples"]["documents"][0]+"\n")
print context_fea, doc_fea
print len(context_fea), len(doc_fea)
