#!/usr/bin/env python3

import os
import subprocess as pc
import time


os.system('kubectl delete -f kubepod.yml')
while True:
    time.sleep(4)
    try:
        os.system('kubectl create -f kubepod.yml')
    except:
        continue
    break
