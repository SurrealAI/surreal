#!/usr/bin/env python
import os
import sys
import shlex


def init():
    os.system('Xdummy > /dev/null 2>&1 &')
    # Compiles everything
    os.system('python -c "import mujoco_py"')


def _run_cmd_list(args):
    if len(args) == 1:
        os.system(args[0])
    else:  # docker run
        os.system(' '.join(map(shlex.quote, args)))


init()
_run_cmd_list(sys.argv[1:])
