from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import subprocess

if __name__ == "__main__":
    args = sys.argv[1:]
    for i, e in enumerate(args):
        if e.find("hdfs:") == 0:
            args[i] = '"{}"'.format(args[i])
    cwd = os.path.dirname(__file__)
    print("cwd: {}".format(cwd))

    bin_path = os.path.join(cwd, "load_run.bin_v1")
    os.system("chmod +x {}".format(bin_path))
    cmd = [bin_path]
    cmd.extend(args)
    print("command: {}".format(cmd))
    os.system("{} {}".format(bin_path, " ".join(cmd[1:])))

