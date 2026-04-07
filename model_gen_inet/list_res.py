from __future__ import print_function
import requests
import os, sys
import subprocess
import re

MONITOR_URL = "http://st3-mhd-rm-2.prod.yiran.com:8088/proxy/{}/monitor"

def list_apps():
    cmd = "yarn application -list | grep TensorFlowApplication"
    raw_app_list = os.popen(cmd).read()
    app_list = []
    for line in raw_app_list.split("\n"):
        parts = line.split("\t")
        if len(parts[0]) > 0:
            app_list.append(parts[0])

    return app_list

def get_resource_usage(app_id):
    resp = requests.get(MONITOR_URL.format(app_id))
    usage = []
    if resp is None:
        print("failed to get resource usage with app id {}".format(app_id))
        print(resp.text) 
    try:
        data = resp.json()
        
        if "allApplyResources" in data:
            for e in data["allApplyResources"]:
                #if e["containerJobName"] == "worker#0":
                core = e["containerCoreLimit"]
                job_type = e["containerJobType"]
                mem_limit = e["containerMemLimit"]
                num = int(e["containerNumber"])
                total_cores = core * num

                usage.append((job_type, num, core, mem_limit, total_cores))
            return usage
    except Exception as ex:
        print(ex)
    
    print("failed to get resource usage with app id {}".format(app_id))
    return None



if __name__ == "__main__":
    app_list = list_apps()

    for e in app_list:
        usage = get_resource_usage(e)
        print("{}: {}".format(e, usage))
