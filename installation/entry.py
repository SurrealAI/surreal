#!/usr/bin/env python
import os
import sys
import argparse
import shutil
import glob
import errno

parser = argparse.ArgumentParser()
parser.add_argument('--cmd', type=str, nargs='+', help='run arbitrary command')
parser.add_argument('--bash', type=str, default='', help='bash script')
parser.add_argument('--py', type=str, default='', help='python script')

args = parser.parse_args()


def f_copy(fsrc, fdst):
    """
    If exist, remove. Supports both dir and file. Supports glob wildcard.
    """
    for f in glob.glob(fsrc):
        try:
            shutil.copytree(f, fdst)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                shutil.copy(f, fdst)


def init():
    os.system('/usr/bin/Xorg -noreset +extension GLX '
              '+extension RANDR +extension RENDER -logfile /etc/fakeX/10.log '
              '-config /etc/fakeX/xorg.conf :10 > /dev/null 2>&1 &')
    assert os.path.exists('/code/mjkey.txt'), 'missing Mujoco `mjkey.txt`'
    f_copy('/code/mjkey.txt', '/root/.mujoco/')


init()


if args.cmd:
    os.system(' '.join(args.cmd))
elif args.py:
    assert args.py.endswith('.py')
    os.system('python ' + args.py)
elif args.bash:
    os.system('/bin/bash ' + args.bash)
else:
    print('No args given to /mylibs/entry.py')
