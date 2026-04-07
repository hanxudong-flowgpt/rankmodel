#!/usr/bin/env python
"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import os
from hdfs import InsecureClient

nn1 = os.getenv('HDFS_NN1')  # 'http://hdpet2gpum1.et2.tbsite.net:50070' #os.getenv('HDFS_NN1')
nn2 = os.getenv('HDFS_NN2')  # 'http://hdpet2gpum2.et2.tbsite.net:50070'#os.getenv('HDFS_NN2')
defaultFS = os.getenv('HDFS_DEFAULT_FS')  # 'hdfs://hdpet2gpu'#os.getenv('HDFS_DEFAULT_FS')
username = os.getenv('HADOOP_USER_NAME')  # 'admin'


def get_hdfs_client():
    print "nn1:", nn1
    print "nn2:", nn2
    print "defaultFS:", defaultFS
    print "username:", username
    try:
        client = InsecureClient(nn1, user=username)
        client.list('')
    except:
        client = InsecureClient(nn2, user=username)
    return client


# return HDFS full path prefix, HDFS://hdpdev
def get_defaultFS():
    return defaultFS


client = get_hdfs_client()
