import json
import math
import sys, os

def edit_distance(s1, s2):
    m = len(s1)+1
    n = len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def check_fc_config(config_path, schema_path):
    schema = None
    with open(schema_path, "r") as f:
        schema = f.read().strip().split(",")
    print "{} fields in schema".format(len(schema))
    
    f = open(config_path, "r")
    config = json.load(f)
    f.close()

    print "{} fields in schema".format(len(schema))
    
    f = open(config_path, "r")
    fc_config = json.load(f)
    f.close()

    #f = open("./config/pdd_mc_conf.json", "r")
    #mc_config = json.load(f)
    #f.close()

    required_cols = []
    
    # shared embedding must have same hash bucket and dimension
    shared_name_dict = {}
    conf_dict = {}
    
    all_pass = True
    for e in config["features"]:
        if "feature_name" in e:
            #print e
            fc_name = e["feature_name"]
            conf_dict[fc_name] = e

            if "shared_name" in e:
                dim = e.get("embedding_dimension", -1)
                hash_bucket_size = e.get("hash_bucket_size", -1)
                shared_name = e["shared_name"]
                if shared_name not in shared_name_dict:
                    shared_name_dict[shared_name] = (hash_bucket_size, dim)
                else:
                    dim_config = shared_name_dict[shared_name]
                    if dim_config[0] != hash_bucket_size:
                        print "\t{} hash_bucket_size inconsistent".format(fc_name)
                    
                    if dim_config[1] != dim:
                        print "\t{} embedding_dimension inconsistent".format(fc_name)

            if fc_name not in schema:
                print "lsc {} not found in input schema".format(fc_name)
                all_pass = False
                for f in schema:
                    if fc_name in f:
                        print "\tdo you mean: {}".format(f)
                    if edit_distance(fc_name, f) < 3:
                        print "\tdo you mean: {}".format(f)

    for col_g in ["item_columns", "cross_columns", "user_columns"]:
        for e in config[col_g]:
            #if "column_name" in e:
                #print e
                #fc_name = e["column_name"]
            fc_name = e
            required_cols.append(fc_name)
            if fc_name not in schema:
                print "mc {} not found in input schema".format(fc_name)
                all_pass = False
                for f in schema:
                    if fc_name in f:
                        print "\tdo you mean: {}".format(f)
                    if edit_distance(fc_name, f) < 3:
                        print "\tdo you mean: {}".format(f)

    # print def of used columns here
    print("\n{}\t{}\t{}\t{}\t{}\t{}".format("feature_name", "shared_name", "bucket_size", "embedding_dimension", "total_param_size", "min_ps_num"))
    all_param_size = 0
    i = 1
    for col_g in ["{}_columns".format(e) for e in ["item", "cross", "user"]]:
        for e in config[col_g]:
            #if "column_name" in e:
                #fc_name = e["column_name"]
            fc_name = e
            conf = conf_dict[fc_name]
            shared_name = conf.get("shared_name", "")
            bucket_size = conf.get("hash_bucket_size", None)
            d = conf.get("embedding_dimension", None)
            param_size = bucket_size * d * 4
            all_param_size = all_param_size + param_size
            min_ps_num = math.ceil(float(param_size) / (2 * 2 << 29))
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i, fc_name, shared_name, bucket_size, d, sizeof_fmt(param_size), min_ps_num))
            i += 1
    print("\nestimated model size: {}\n".format(sizeof_fmt(all_param_size)))

    if all_pass:
        print "all fields in config valid"
    
    print "required cols: "
    print "[", ",".join(['"{}"'.format(e) for e in required_cols]), "]"


if __name__ == "__main__":
    config_path = sys.argv[1]
    schema_path = sys.argv[2]
    check_fc_config(config_path, schema_path)
