

def getFileList(daylist, base_path):
    from subprocess import Popen, PIPE
    files_list = []
    for day in daylist:
        thepath = base_path % day
        resp = Popen("hadoop fs -ls -R %s" % thepath, shell=True, stdout=PIPE, stderr=PIPE).stdout.readlines()
        for i in range(len(resp)):
            parts = resp[i].strip().split()
            if len(parts) == 8 and (parts[7].split('/')[-1].startswith("part") or parts[7].split('/')[-1].endswith("_0")):
                files_list += [parts[7]]
    return files_list
