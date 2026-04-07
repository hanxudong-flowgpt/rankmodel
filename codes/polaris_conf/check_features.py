import json

feature1 = json.load(open("debug2.json"))

hive_schema = []
for line in open('maidian_hive.schema'):
    f_name = line.split()[0]
    hive_schema.append(f_name[1:-1])

for k, v in feature1.items():
    if k not in hive_schema:
        print k