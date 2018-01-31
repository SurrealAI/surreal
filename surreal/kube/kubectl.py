"""
Simple python client. The official client is poorly documented and crashes
on import. Except for `watch`, other info can simply be parsed from stdout.

~/.surreal.yml
-
"""
import sys
import time
import subprocess as pc
import shlex
from surreal.kube.yaml_util import YamlList, JinjaYaml, file_content
from surreal.kube.git_snapshot import push_snapshot
import os
import os.path as path


def run_process(cmd):
    # if isinstance(cmd, str):  # useful for shell=False
    #     cmd = shlex.split(cmd.strip())
    proc = pc.Popen(cmd, stdout=pc.PIPE, stderr=pc.PIPE, shell=True)
    out, err = proc.communicate()
    return out.decode('utf-8'), err.decode('utf-8'), proc.returncode


def print_err(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


class Kubectl(object):
    def __init__(self, surreal_yml='~/.surreal.yml', dry_run=False):
        surreal_yml = path.expanduser(surreal_yml)
        assert path.exists(surreal_yml)
        self.config = YamlList.from_file(surreal_yml)[0]
        self.dry_run = dry_run
        self._loop_start_time = None

    def run(self, cmd):
        if self.dry_run:
            print('kubectl ' + cmd)
            return '', '', 0
        else:
            out, err, retcode = run_process('kubectl ' + cmd)
            return out.strip(), err.strip(), retcode

    def _print_err_return(self, out, err, retcode):
        print_err('error code:', retcode)
        print_err('*' * 20, 'stderr', '*' * 20)
        print_err(err)
        print_err('*' * 20, 'stdout', '*' * 20)
        print_err(out)

    def run_verbose(self, cmd):
        out, err, retcode = self.run(cmd)
        if retcode != 0:
            print_err('Command `{}`'.format(cmd))
            self._print_err_return(out, err, retcode)
        elif out:
            print(out)

    def run_event_loop(self, func, *args, poll_interval=1, **kwargs):
        """
        Run a function repeatedly until it returns True
        """
        self._loop_start_time = time.time()
        while True:
            if func(*args, **kwargs):
                break
            time.sleep(poll_interval)

    def _create_loop(self, yaml_file):
        """
        Useful for restarting a kube service.
        Resource might be in the process of deletion. Wait until deletion completes
        """
        yaml_file = path.expanduser(yaml_file)
        out, err, retcode = self.run('create -f "{}"'.format(yaml_file))
        if retcode:
            # TODO: very hacky check, should run checks on names instead
            if 'is being deleted' in err:
                if True or time.time() - self._loop_start_time > 30:
                    print_err('old resource being deleted, waiting ...')
                    print_err(err)
                    self._loop_start_time = time.time()
                return False
            else:
                print_err('create encounters an error that is not `being deleted`')
                self._print_err_return(out, err, retcode)
                return True
        else:
            print(out)
            return True

    def create(self, yaml_file, context=None, **context_kwargs):
        """
        kubectl create -f dummy.yml

        Args:
            yaml_file:
            context: see `YamlList`
            **context_kwargs: see `YamlList`
        """
        with JinjaYaml.from_file(yaml_file).render_temp_file(
                context=context,
                **context_kwargs
        ) as temp:
            if self.dry_run:
                print(file_content(temp))
            else:
                self.run_event_loop(self._create_loop, temp, poll_interval=5)

    def get_secret_file(self, yaml_key):
        """
        To be passed to Jinja2 engine.
        Hack yaml to support multiline key files like Mujoco.
        https://stackoverflow.com/questions/3790454/in-yaml-how-do-i-break-a-string-over-multiple-lines
        """
        fpath = self.config[yaml_key]
        return file_content(fpath)

    def create_with_git(self, yaml_file, snapshot=True, context=None):
        """
        First create a snapshot of the git repos, upload to github
        Then create Kube objects with the git info
        Args:
            context: for extra context variables
        """
        repo_paths = self.config.git.get('snapshot_repos', [])
        repo_paths = [path.expanduser(p) for p in repo_paths]
        if snapshot and not self.dry_run:
            for repo_path in repo_paths:
                push_snapshot(
                    snapshot_branch=self.config.git.snapshot_branch,
                    repo_path=repo_path
                )
        repo_names = [path.basename(path.normpath(p)).lower()
                      for p in repo_paths]
        git_config = {
            'GIT_USER': self.config.git.user,
            'GIT_TOKEN': self.config.git.token,
            'GIT_SNAPSHOT_BRANCH': self.config.git.snapshot_branch,
            'GIT_REPOS': repo_names,
        }
        if context is None:
            context = {}
        git_config.update(context)
        self.create(yaml_file, context=git_config)


class SurrealKube(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    kube = Kubectl(dry_run=0)
    kube.create_with_git('~/Dropbox/Portfolio/Kurreal_demo/kfinal_gcloud.yml',
             snapshot=0,
             context={'MUJOCO_KEY_TEXT': kube.get_secret_file('mujoco_key_path')})


