"""
Simple python client. The official client is poorly documented and crashes
on import. Except for `watch`, other info can simply be parsed from stdout.
"""
import shlex
import subprocess as pc
from surreal.kube.yaml_util import YamlList
import os
import os.path as path


def run_process(cmd):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd.strip())
    proc = pc.Popen(cmd, stdout=pc.PIPE, stderr=pc.PIPE)
    out, err = proc.communicate()
    return out.decode('utf-8'), err.decode('utf-8'), proc.returncode


if __name__ == '__main__':
    out, err, code = run_process('kubectl config view')
    print('OUT', out)
    print('-'*30)
    print(err)
