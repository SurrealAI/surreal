"""
Simple python client. The official client is poorly documented and crashes
on import. Except for `watch`, other info can simply be parsed from stdout.

~/.surreal.yml
-
"""
import sys
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


def print_err(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


class Kubectl(object):
    def __init__(self, surreal_yml='~/.surreal.yml'):
        surreal_yml = path.expanduser(surreal_yml)
        assert path.exists(surreal_yml)
        self.config = YamlList.from_file(surreal_yml)[0]
        self.git_config = {
            'GIT_USER': self.config.git.user,
            'GIT_TOKEN': self.config.git.token,
            'GIT_SNAPSHOT_BRANCH': self.config.git['snapshot-branch']
        }

    def run(self, cmd):
        return run_process('kubectl ' + cmd)

    def run_verbose(self, cmd):
        out, err, retcode = self.run(cmd)
        out, err = out.strip(), err.strip()
        if retcode != 0:
            print_err('Kube command error:', retcode)
            print_err('*' * 20, 'stderr', '*' * 20)
            print_err(err)
            print_err('*' * 20, 'stdout', '*' * 20)
            print_err(out)
        elif out:
            print(out)

    def create(self, yaml_file, context=None, **context_kwargs):
        """
        kubectl create -f dummy.yml

        Args:
            yaml_file:
            context: see `YamlList`
            **context_kwargs: see `YamlList`
        """
        if context or context_kwargs:
            yaml_list = YamlList.from_template_file(
                yaml_file,
                context=context,
                **context_kwargs)
            with yaml_list.temp_file() as temp:
                self.run_verbose('create -f "{}"'.format(temp))
        else:
            self.run_verbose('create -f "{}"'.format(yaml_file))

    def create_with_git(self, yaml_file):
        self.create(yaml_file, context=self.git_config)


if __name__ == '__main__':
    kube = Kubectl()
    kube.create('~/Dropbox/Portfolio/Kurreal-demo/kpod_gcloud.yml')


