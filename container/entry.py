#!/usr/bin/env python
"""
ENV variables
- mujoco_key_text: plain text environment value
- repo_surreal: /mylibs/surreal/surreal
- repo_tensorplex

"""
import os
import sys
import shlex
import argparse
import shutil
import glob
import errno

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

def _run_cmd_list(args):
    if len(args) == 1:
        os.system(args[0])
    else:  # docker run
        os.system(' '.join(map(shlex.quote, args)))

init()
_run_cmd_list(sys.argv[1:])