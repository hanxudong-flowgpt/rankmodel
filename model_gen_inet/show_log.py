from __future__ import print_function
import requests
import os, sys
import subprocess
import re

MONITOR_URL = "http://st3-mhd-rm-2.prod.yiran.com:8088/proxy/{}/monitor"

def get_container_id(app_id, label):
    resp = requests.get(MONITOR_URL.format(app_id))
    if resp is None:
        print("failed to get monitor info of {}".format(app_id))
        print(resp.text) 
    try:
        data = resp.json()
        
        if "allContainerBriefs" in data:
            for e in data["allContainerBriefs"]:
                #if e["containerJobName"] == "worker#0":
                if e["containerJobName"] == label:
                    container_id = e["containerId"]
                    print("container id: {}".format(container_id))
                    return container_id
    except Exception as ex:
        #print(ex)
        print(resp.text)
    
    print("failed to get container id with label {}".format(label))
    return None


def get_latest_log_file():
    cmd = "ls -tp log/log* | grep -v /$ | head -1"
    latest_log_file_name = os.popen(cmd).read().strip()
    return "./{}".format(latest_log_file_name)


def get_app_id(log_file_name):
    with open(log_file_name) as f:
        for line in f:
            result = re.findall("ResourceManager web address for application: (.+)?$", line)
            if len(result) > 0:
                print(result[0])
                return result[0].split("/")[-1]


def get_container_log(app_id, label):
    '''
    get whole container log and distill stdout part
    '''
    container_id = get_container_id(app_id, label)
    if container_id is None:
        return

    log_file_name = "./log/{}_{}.log".format(app_id, label).replace("#", "")
    cmd = "yarn logs -applicationId {} -containerId {} > {}".format(app_id, container_id, log_file_name)
    print(cmd)
    subprocess.call(cmd, shell=True)
    stderr_log_file_name = "./log/{}_stderr_{}.log".format(label, app_id).replace("#", "")
    
    stderr_f = open(stderr_log_file_name, 'w+')

    stderr_head = "LogType:container_stderr"
    stderr_tail = "End of LogType:container_stderr"

    with open(log_file_name, 'r') as f:
        stderr_part = False
        for line in f:
            if line[:len(stderr_head)] == stderr_head:
                stderr_part = True

            if line[:len(stderr_tail)] == stderr_tail:
                break

            if stderr_part:
                stderr_f.write(line)

    stderr_f.close()
    print("{} stderr log written to {}".format(label, stderr_log_file_name))
    os.system("grep auc {} | tail -50".format(stderr_log_file_name))


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        app_id = sys.argv[1]
        label = sys.argv[2]
    else:
        #print("usage: python show_log.py app_id task_id")
        #sys.exit(1)
        print("no parameter passed, use app_id in the latest log file")
        log_file_name = get_latest_log_file()
        app_id = get_app_id(log_file_name)
        print(app_id)
        print("get worker 0 log")
        label = "worker#0"
    
    get_container_log(app_id, label)

    #container_id = get_container_id(app_id, label)

    #if container_id is not None:
    #    get_container_log(app_id, container_id)
    #else:
    #    print("failed to get container id")
