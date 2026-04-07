import os, sys, subprocess, time, datetime
from subprocess import Popen, PIPE

assert len(sys.argv) == 6

source_root = sys.argv[1]
target_root = sys.argv[2]
day = sys.argv[3]
hour = sys.argv[4]
minute = sys.argv[5]

y, m, d = day.split('-')
time1 = datetime.datetime(int(y), int(m), int(d), int(hour), int(minute))
time2 = time1 + datetime.timedelta(hours=1)
day = '%d-%d-%d' % (time2.year, time2.month, time2.day)
hour = str(time2.hour)

print('here is: %s/hr=%s/min=%s' % (day, hour, minute))
ks_param="-Dipc.client.fallback-to-simple-auth-allowed=true -Ddfs.nameservices=temp-data-ns,yiran-data-ns,yiran2-data-ns,yiran3-data-ns,yiran4-data-ns,yiran5-data-ns,yiranhq-data-ns,backuphq-data-ns,hq2-data-ns,hq3-data-ns,lg8-data-ns,lg9-data-ns,hq4-data-ns,hq5-data-ns,lg10-data-ns,lg11-data-ns,lg12-data-ns,kingso-hdfs,searchhdfs -Ddfs.ha.namenodes.kingso-hdfs=nn1,nn2 -Ddfs.ha.namenodes.searchhdfs=nn1,nn2 -Ddfs.client.failover.proxy.provider.kingso-hdfs=org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider -Ddfs.client.failover.proxy.provider.searchhdfs=org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider -Ddfs.namenode.rpc-address.kingso-hdfs.nn1=st3-kingso-hdfs-nn-1.prod.yiran.com:8020 -Ddfs.namenode.rpc-address.kingso-hdfs.nn2=st3-kingso-hdfs-nn-2.prod.yiran.com:8020 -Ddfs.namenode.rpc-address.searchhdfs.nn1=st5-search-hdfs-nn-2.prod.yiran.com:8020 -Ddfs.namenode.rpc-address.searchhdfs.nn2=st5-search-hdfs-nn-1.prod.yiran.com:8020"

t1 = time.time()
while True:
    if time.time() - t1 > 60 * 60 * 2:
        print('two hours gone, break...')
        break
    files_list = []
    resp = Popen("hadoop fs %s -ls -R %s/%s/hr=%s/min=%s" % (ks_param, source_root, day, hour, minute), shell=True, stdout=PIPE, stderr=PIPE).stdout.readlines()
    for i in range(len(resp)):
        parts = resp[i].strip().split()
        if len(parts) == 8 and parts[7].split('/')[-1].startswith("part"):
            files_list += [parts[7]]
    if len(files_list) == 0:
        print('dir %s/hr=%s/min=%s file dump not starting...' % (day, hour, minute))
        time.sleep(60*5)
        continue
    else:
        for file_name in files_list:
            if file_name.endswith('in-progress') or file_name.endswith('_COPYING_'):
                print('dir %s/hr=%s/min=%s file dump start, but not end...' % (day, hour, minute))
                time.sleep(60)
                continue
        print('dir %s/hr=%s/min=%s start copying...' % (day, hour, minute))
        # os.system("hadoop fs -mkdir -p %s/%s/hr=%s/min=%s" % (target_root, day, hour, minute))
        os.system("hadoop distcp %s %s/%s/hr=%s/min=%s/ %s/%s/hr=%s/min=%s/" %
                  (ks_param, source_root, day, hour, minute, target_root, day, hour, minute))
        print(resp)
        print('dir %s/hr=%s/min=%s end copy...' % (day, hour, minute))
        break