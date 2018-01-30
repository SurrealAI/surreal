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
from surreal.kube.git_snapshot import push_snapshot
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
    def __init__(self, surreal_yml='~/.surreal.yml', dry_run=False):
        surreal_yml = path.expanduser(surreal_yml)
        assert path.exists(surreal_yml)
        self.config = YamlList.from_file(surreal_yml)[0]
        self.git_config = {
            'GIT_USER': self.config.git.user,
            'GIT_TOKEN': self.config.git.token,
            'GIT_SNAPSHOT_BRANCH': self.config.git['snapshot-branch']
        }
        self.dry_run = dry_run

    def run(self, cmd):
        if self.dry_run:
            print('kubectl ' + cmd)
            return '', '', 0
        else:
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
            if self.dry_run:
                print(yaml_list)
            with yaml_list.temp_file() as temp:
                self.run_verbose('create -f "{}"'.format(temp))
        else:
            self.run_verbose('create -f "{}"'.format(yaml_file))

    def create_with_git(self, yaml_file, snapshot=True, **context_kwargs):
        """
        First create a snapshot of the git repos, upload to github
        Then create Kube objects with the git info
        Args:
            context_kwargs: for extra context
        """
        if snapshot:
            for repo_path in self.config.git.get('snapshot-repos', []):
                push_snapshot(
                    snapshot_branch=self.config.git['snapshot-branch'],
                    repo_path=path.expanduser(repo_path)
                )
        self.create(yaml_file, context=self.git_config, **context_kwargs)


if __name__ == '__main__':
    kube = Kubectl(dry_run=True)
    kube.create_with_git('~/Dropbox/Portfolio/Kurreal-demo/kpod_gcloud.yml')


