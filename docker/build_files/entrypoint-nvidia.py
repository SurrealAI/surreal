#!/usr/bin/env python
import os
import sys
import shlex


def init():
    os.system('Xdummy > /dev/null 2>&1 &')


def _run_cmd_list(args):
    os.system(' '.join(map(shlex.quote, args)))


init()
_run_cmd_list(sys.argv[1:])
