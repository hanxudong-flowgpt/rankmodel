import os, json, time, sys
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + "/..")
from model import MODELZOO

print MODELZOO