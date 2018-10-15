#!/usr/bin/env python
import os
import sys
import shlex
import shutil


def init():
    os.system('/usr/bin/Xorg -noreset +extension GLX '
              '+extension RANDR +extension RENDER -logfile /etc/fakeX/10.log '
              '-config /etc/fakeX/xorg.conf :10 > /dev/null 2>&1 &')
    if os.path.exists('/etc/secrets/mjkey.txt'):
        shutil.copy('/etc/secrets/mjkey.txt', '/root/.mujoco/mjkey.txt')


def _run_cmd_list(args):
    if len(args) == 1:
        os.system(args[0])
    else:  # docker run
        os.system(' '.join(map(shlex.quote, args)))


init()
_run_cmd_list(sys.argv[1:])
