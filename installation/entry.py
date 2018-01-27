#!/usr/bin/env python
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cmd', type=str, nargs='+', help='run arbitrary command')
parser.add_argument('--bash', type=str, default='', help='bash script')
parser.add_argument('--py', type=str, default='', help='python script')

args = parser.parse_args()

if args.cmd:
    os.system(' '.join(args.cmd))
elif args.py:
    assert args.py.endswith('.py')
    os.system('python ' + args.py)
elif args.bash:
    os.system('/bin/bash ' + args.bash)
else:
    print('No args given to /mylibs/entry.py')
