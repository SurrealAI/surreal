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
import json
import os
import os.path as path
from surreal.kube.yaml_util import YamlList, JinjaYaml, file_content
from surreal.kube.git_snapshot import push_snapshot
from surreal.utils.ezdict import EzDict


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
        self._created_yaml = None  # will be set after create()

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
        print_err('*' * 46)

    def run_verbose(self, cmd, print_out=True, raise_on_error=False):
        out, err, retcode = self.run(cmd)
        if retcode != 0:
            self._print_err_return(out, err, retcode)
            msg = 'Command `kubectl {}` fails'.format(cmd)
            if raise_on_error:
                raise RuntimeError(msg)
            else:
                print_err(msg)
        elif out and print_out:
            print(out)
        return out, err, retcode

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
            self._created_yaml = YamlList.from_file(temp)
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

    def create_surreal(self,
                       yaml_file,
                       snapshot=True,
                       mujoco=True,
                       context=None):
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
        if mujoco:
            git_config['MUJOCO_KEY_TEXT'] = self.get_secret_file('mujoco_key_path')
        git_config.update(context)
        self.create(yaml_file, context=git_config)

    def _get_selectors(self, labels, fields):
        """
        Helper for list_resources and list_jsonpath
        """
        cmd= ' '
        if labels:
            cmd += '--selector ' + shlex.quote(labels) + ' '
        if fields:
            cmd += ' --field-selector ' + shlex.quote(fields) + ' '
        return cmd

    def query_resources(self, resource, output_format, labels=None, fields=None):
        """
        Query all items in the resource with `output_format`
        JSONpath: https://kubernetes.io/docs/reference/kubectl/jsonpath/
        label selectors: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/

        Args:
            resource: pod, service, deployment, etc.
            output_format: https://kubernetes.io/docs/reference/kubectl/overview/#output-options
              - custom-columns=<spec>
              - custom-columns-file=<filename>
              - json: returns a dict
              - jsonpath=<template>
              - jsonpath-file=<filename>
              - name: list
              - wide
              - yaml: returns a dict
            labels: label selector syntax, comma separated as logical AND. E.g:
              - equality: mylabel=production
              - inequality: mylabel!=production
              - set: mylabel in (group1, group2)
              - set exclude: mylabel notin (group1, group2)
              - don't check value, only check key existence: mylabel
              - don't check value, only check key nonexistence: !mylabel
            fields: field selector, similar to label selector but operates on the
              pod fields, such as `status.phase=Running`
              fields can be found from `kubectl get pod <mypod> -o yaml`

        Returns:
            dict if output format is yaml or json
            list if output format is name
            string from stdout otherwise
        """
        cmd = 'get ' + resource
        cmd += self._get_selectors(labels, fields)
        if '=' in output_format:
            # quoting the part after jsonpath=<...>
            prefix, arg = output_format.split('=', 1)
            output_format = prefix + '=' + shlex.quote(arg)
        cmd += '-o ' + output_format
        out, _, _ = self.run_verbose(cmd, print_out=False, raise_on_error=True)
        if output_format == 'yaml':
            return EzDict.loads_yaml(out)
        elif output_format == 'json':
            return EzDict.loads_json(out)
        elif output_format == 'name':
            return out.split('\n')
        else:
            return out

    def query_jsonpath(self, resource, jsonpath, labels=None, fields=None):
        """
        Query items in the resource with jsonpath
        https://kubernetes.io/docs/reference/kubectl/jsonpath/
        This method is an extension of list_resources()
        Args:
            resource:
            jsonpath: make sure you escape dot if resource key string contains dot.
              key must be enclosed in *single* quote!!
              e.g. {.metadata.labels['kubernetes\.io/hostname']}
              you don't have to do the range over items, we take care of it
            labels: see `list_resources`
            fields:

        Returns:
            a list of returned jsonpath values
        """
        jsonpath = '{range .items[*]}' + jsonpath + '{"\\n\\n"}{end}'
        output_format = "jsonpath=" + jsonpath
        out = self.query_resources(
            resource=resource,
            output_format=output_format,
            labels=labels,
            fields=fields
        )
        return out.split('\n\n')


if __name__ == '__main__':
    kube = Kubectl(dry_run=0)
    if 0:
        kube.create_surreal('~/Dropbox/Portfolio/Kurreal_demo/kfinal_gcloud.yml',
                 snapshot=0,
                 context={'MUJOCO_KEY_TEXT': kube.get_secret_file('mujoco_key_path')})
    else:
        import pprint
        pp = pprint.pprint
        # 3 different ways to get a list of node names
        pp(kube.query_jsonpath('nodes', '{.metadata.name}'))
        pp(kube.query_jsonpath('nodes', "{.metadata.labels['kubernetes\.io/hostname']}"))
        pp(kube.query_resources('nodes', 'name'))
        y = kube.query_resources('nodes', 'yaml')
        pp(y.items[0].metadata)
        # print(YamlList(y).to_string())
        print(kube.query_jsonpath('pods', '{.metadata.name}', labels='mytype=transient_component'))
        print(kube.query_resources('pods', 'name', labels='mytype=persistent_component'))
        y = kube.query_resources('pods', 'yaml')
        pp(y.items[0].status)

