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
import functools
import os
import re
import os.path as path
from pkg_resources import parse_version
from collections import OrderedDict
from surreal.kube.yaml_util import YamlList, JinjaYaml, file_content
from surreal.kube.git_snapshot import push_snapshot
from surreal.utils.ezdict import EzDict
import surreal.utils as U
from symphony.cluster.kubecluster import KubeCluster, KubeNFSVolume, KubeGitVolume
from symphony.experiment import ProcessConfig, ProcessGroupConfig, ExperimentConfig
import itertools



SURREAL_YML_VERSION = '0.0.2'  # force version check


def run_process(cmd):
    # if isinstance(cmd, str):  # useful for shell=False
    #     cmd = shlex.split(cmd.strip())
    proc = pc.Popen(cmd, stdout=pc.PIPE, stderr=pc.PIPE, shell=True)
    out, err = proc.communicate()
    return out.decode('utf-8'), err.decode('utf-8'), proc.returncode


def print_err(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


_DNS_RE = re.compile('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')


def check_valid_dns(name):
    """
    experiment name is used as namespace, which must conform to DNS format
    """
    if not _DNS_RE.match(name):
        raise ValueError(name + ' must be a valid DNS name with only lower-case '
            'letters, 0-9 and hyphen. No underscore or dot allowed.')


class Kubectl(object):
    NONAGENT_COMPONENTS = ['learner', 'ps', 'replay',
                           'tensorplex', 'loggerplex', 'tensorboard']

    def __init__(self, surreal_yml='~/.surreal.yml', dry_run=False):
        surreal_yml = U.f_expand(surreal_yml)
        assert U.f_exists(surreal_yml)
        # persistent config in the home dir that contains git access token
        self.config = YamlList.from_file(surreal_yml)[0]
        self._check_version()
        self.folder = U.f_expand(self.config.local_kurreal_folder)
        self.dry_run = dry_run
        self._loop_start_time = None

    def get_path(self, subfolder, file_name):
        "file under local_kurreal_folder"
        return U.f_join(self.folder, subfolder, file_name)

    def _check_version(self):
        """
        Check ~/.surreal.yml `version` key
        """
        assert 'version' in self.config, 'surreal yml version not specified.'
        if parse_version(SURREAL_YML_VERSION) != parse_version(self.config.version):
            raise ValueError('version incompatible, please check the latest '
                             'sample.surreal.yml and make sure ~/.surreal.yml is '
                             + SURREAL_YML_VERSION)

    def run(self, cmd, program='kubectl'):
        cmd = program + ' ' + cmd
        if self.dry_run:
            print(cmd)
            return '', '', 0
        else:
            out, err, retcode = run_process(cmd)
            if 'could not find default credentials' in err:
                print("Please try `gcloud container clusters get-credentials mycluster` "
                      "to fix credential error")
            return out.strip(), err.strip(), retcode

    def run_raw(self, cmd, program='kubectl', print_cmd=False):
        """
        Raw os.system calls

        Returns:
            error code
        """
        cmd = program + ' ' + cmd
        if self.dry_run:
            print(cmd)
        else:
            if print_cmd:
                print(cmd)
            return os.system(cmd)

    def _print_err_return(self, out, err, retcode):
        print_err('error code:', retcode)
        print_err('*' * 20, 'stderr', '*' * 20)
        print_err(err)
        print_err('*' * 20, 'stdout', '*' * 20)
        print_err(out)
        print_err('*' * 46)

    def run_verbose(self, cmd,
                    print_out=True,
                    raise_on_error=False,
                    program='kubectl'):
        out, err, retcode = self.run(cmd, program=program)
        if retcode != 0:
            self._print_err_return(out, err, retcode)
            msg = 'Command `{} {}` fails'.format(program, cmd)
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

    # TODO remove this
    def _create_loop(self, yaml_file, namespace):
        """
        Useful for restarting a kube service.
        Resource might be in the process of deletion. Wait until deletion completes
        """
        yaml_file = U.f_expand(yaml_file)
        out, err, retcode = self.run('create -f "{}" --namespace {}'
                                     .format(yaml_file, namespace))
        if retcode:
            # TODO: very hacky check, should run checks on names instead
            if 'is being deleted' in err:
                if time.time() - self._loop_start_time > 30:
                    print_err('old resource being deleted, waiting ...')
                    print_err(err)
                    self._loop_start_time = time.time()
                return False
            else:
                if 'AlreadyExists' in err:
                    print('Warning: some components already exist')
                else:
                    print_err('create encounters an error that is not `being deleted`')
                self._print_err_return(out, err, retcode)
                return True
        else:
            print(out)
            return True

    def create(self,
               namespace,
               jinja_template,
               rendered_path,
               context=None,
               check_experiment_exists=True,
               **context_kwargs):
        """
        kubectl create namespace <experiment_name>
        kubectl create -f kurreal.yml --namespace <experiment_name>

        Args:
            jinja_template: Jinja kurreal_template.yml
            context: see `YamlList`
            **context_kwargs: see `YamlList`

        Returns:
            path for the rendered kurreal.yml in the experiment folder
        """
        check_valid_dns(namespace)
        if check_experiment_exists and U.f_exists(rendered_path):
            raise FileExistsError(rendered_path
                                  + ' already exists, cannot run `create`.')
        JinjaYaml.from_file(jinja_template).render_file(
            rendered_path, context=context, **context_kwargs
        )
        if self.dry_run:
            print(file_content(rendered_path))
        else:
            self.run('create namespace ' + namespace)
            self.run_event_loop(
                self._create_loop,
                rendered_path,
                namespace=namespace,
                poll_interval=5
            )

    def _yamlify_label_string(self, label_string):
        if not label_string:
            return ''
        assert (':' in label_string or '=' in label_string), \
            'label spec should look like <labelname>=<labelvalue>'
        if ':' in label_string:
            label_spec = label_string.split(':', 1)
        else:
            label_spec = label_string.split('=', 1)
        # the space after colon is necessary for valid yaml
        return '{}: {}'.format(*label_spec)

    def _parse_label_strings(self, selector_string):
        """
        Receives input like:
            surreal-node=agent,cloud.google.com/gke-accelerator=nvidia-tesla-k80
        splits by ',' and then parse by = into dictionary
        i.e.:
            surreal-node=agent,cloud.google.com/gke-accelerator=nvidia-tesla-k80
            =>
            {
                'surreal-node': 'agent',
                'cloud.google.com/gke-accelerator': 'nvidia-tesla-k80'
            }
        """
        label_strings = selector_string.split(',')
        di = {}
        for label_string in label_strings:
            if label_string == '':
                continue
            assert(label_string.count('=') == 1), 'invalid label {}'.format(label_string)
            k, v = label_string.split('=')
            di[k] = v
        return di

    @property
    def username(self):
        assert 'username' in self.config, 'must specify username in ~/.surreal.yml'
        return self.config.username

    def prefix_username(self, experiment_name):
        """
        Set boolean flag `prefix_experiment_with_username` in ~/.surreal.yml.
        Will prefix the experiment name unless it is already manually prefixed.
        """
        assert 'prefix_experiment_with_username' in self.config
        if self.config.prefix_experiment_with_username:
            prefix = self.username + '-'
            if not experiment_name.startswith(prefix):
                experiment_name = prefix + experiment_name
        return experiment_name

    def strip_username(self, experiment_name):
        """
        Will remove the "<username>-" prefix
        """
        assert 'prefix_experiment_with_username' in self.config
        if self.config.prefix_experiment_with_username:
            prefix = self.username + '-'
            if experiment_name.startswith(prefix):
                experiment_name = experiment_name[len(prefix):]
        return experiment_name

    def get_remote_experiment_folder(self, experiment_name):
        """
        actual experiment folder will be <mount_path>/<root_subfolder>/<experiment_name>/
        """
        # DON'T use U.f_join because we don't want to expand the path locally
        root_subfolder = self.config.fs.experiment_root_subfolder
        assert not root_subfolder.startswith('/'), \
            'experiment_root_subfolder should not start with "/". ' \
            'Actual experiment folder path will be ' \
            '<mount_path>/<root_subfolder>/<experiment_name>/'
        return os.path.join(
            self.config.fs.mount_path,
            self.config.fs.experiment_root_subfolder,
            experiment_name
        )

    def create_surreal(self,
                       experiment_name,
                       jinja_template,
                       rendered_path,
                       agent_pod_type,
                       nonagent_pod_type,
                       cmd_dict,
                       snapshot=True,
                       mujoco=True,
                       check_experiment_exists=True):
        """
        First create a snapshot of the git repos, upload to github
        Then create Kube objects with the git info
        Args:
            experiment_name: will also be used as hostname for DNS
            jinja_template: kurreal_template.yml file path
            rendered_path: rendered yaml file path
            agent_pod_type: key to spec defined in `pod_types` section of .surreal.yml
            nonagent_pod_type: key to spec defined in `pod_types` section of .surreal.yml
            cmd_dict: dict of commands to be run on each container
            snapshot: True to take a snapshot of git repo and upload
            mujoco: True to copy mujoco key into the generated yaml
            prefix_user_name: True to prefix experiment name (and host name)
                as <myusername>-<experiment_name>
            check_experiment_exists: check if the Kube yaml has already been generated.
        """
        check_valid_dns(experiment_name)
        if check_experiment_exists and U.f_exists(rendered_path):
            raise FileExistsError(rendered_path
                      + ' already exists, cannot run `create_surreal`.')
        C = self.config
        repo_paths = C.git.get('snapshot_repos', [])
        repo_paths = [U.f_expand(p) for p in repo_paths]
        if snapshot and not self.dry_run:
            for repo_path in repo_paths:
                push_snapshot(
                    snapshot_branch=C.git.snapshot_branch,
                    repo_path=repo_path
                )
        repo_names = [path.basename(path.normpath(p)).lower()
                      for p in repo_paths]

        # Read pod specifications
        assert agent_pod_type in C.pod_types, \
            'agent pod type not found in `pod_types` section in ~/.surreal.yml'
        assert nonagent_pod_type in C.pod_types, \
            'nonagent pod type not found in `pod_types` section in ~/.surreal.yml'
        agent_pod_spec = C.pod_types[agent_pod_type]
        nonagent_pod_spec = C.pod_types[nonagent_pod_type]
        agent_resource_request = agent_pod_spec.get('resource_request', {})
        nonagent_resource_request = nonagent_pod_spec.get('resource_request', {})
        agent_resource_limit = agent_pod_spec.get('resource_limit', {})
        nonagent_resource_limit = nonagent_pod_spec.get('resource_limit', {})

        c = KubeCluster(dry_run=False)

        learner = ProcessConfig('learner', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['learner']])
        replay = ProcessConfig('replay', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['replay']])
        ps = ProcessConfig('ps', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['ps']])
        tensorboard = ProcessConfig('tensorboard', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['tensorboard']])
        tensorplex = ProcessConfig('tensorplex', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['tensorplex']])
        loggerplex = ProcessConfig('loggerplex', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['loggerplex']])

        nonagent = ProcessGroupConfig('nonagent')
        nonagent.add_process(learner, replay, ps, tensorboard, tensorplex, loggerplex)

        agents = []
        for i, arg in enumerate(cmd_dict['agent']):
            agent_p = ProcessConfig('agent-{}'.format(i), container_image=agent_pod_spec.image, args=['--cmd', arg])
            agents.append(agent_p)

        evals = []
        for i, arg in enumerate(cmd_dict['eval']):
            eval_p = ProcessConfig('eval-{}'.format(i), container_image=agent_pod_spec.image, args=['--cmd', arg])
            evals.append(eval_p)

        # TODO: Use kubernete's load-balancing instead
        for proc in itertools.chain(agents, evals):
            proc.requests('ps-frontend')
            proc.requests('collector-frontend')
        
        ps.provides('ps-frontend')
        ps.reserves('ps-backend')
        ps.requests('paramter-publish')
        
        replay.provides('collector-frontend')
        replay.provides('sampler-frontend')
        replay.reserves('collector-backend')
        replay.reserves('sampler-backend')
        
        learner.requests('sampler-frontend')
        learner.provides('paramter-publish')
        learner.reserves('prefetch-queue')

        tensorplex.provides('tensorplex')
        loggerplex.provides('loggerplex')
        for proc in itertools.chain(agents, evals, [ps, replay, learner]):
            proc.requests('tensorplex')
            proc.requests('loggerplex')

        tensorboard.exposes(tensorboard=6006)

        exp = ExperimentConfig(experiment_name)
        exp.add_process_group(nonagent)
        exp.add_process(*agents)
        exp.add_process(*evals)

        # Use kube-specific configs
        exp.use_kube()

        # Mount volumes
        # TODO: support other fs
        
        if not C.fs.type.lower() in ['nfs']:
            raise NotImplementedError('Unsupported file server type: "{}". '
              'Supported options are [nfs]'.format(C.fs.type)) 
        nfs_server = C.fs.server
        nfs_server_path = C.fs.path_on_server
        nfs_mount_path = C.fs.mount_path

        if mujoco:
            mjkey = file_content(C.mujoco_key_path)

        for proc in exp.processes.values():
            # Mount nfs
            proc.kube.mount_nfs(server=nfs_server, path=nfs_server_path, mount_path=nfs_mount_path)

            # mount git
            # This needs fixing, currently it has a lot of assumptions, 
            # including that direcotry name of the repo locally should equal to cloud
            # which can be false
            for git_repo in repo_names:
                # if git_repo == 'surreal':
                #     git_repo_name_github='Surreal'
                # elif git_repo == 'tensorplex':
                #     git_repo_name_github='Tensorplex'
                # else:
                #     git_repo_name_github=git_repo
                repository = 'https://{}:{}@github.com/SurrealAI/{}'.format(
                    C.git.user, C.git.token, git_repo)
                revision = C.git.snapshot_branch
                mount_path = '/mylibs/{}'.format(git_repo)
                proc.kube.mount_git_repo(repository=repository, revision=revision, mount_path=mount_path)
                env_key = 'repo_{}'.format(git_repo.replace('-', '_'))
                env_val = '/mylibs/{0}/{0}'.format(git_repo)
                proc.kube.set_env(env_key, env_val) # TODO: is this env needed currently?

            # Add mujoco key: TODO: handle secret properly
            proc.kube.set_env('mujoco_key_text', mjkey)
            proc.kube.image_pull_policy('Always')

        agent_selector = agent_pod_spec.get('selector', {})
        for proc in itertools.chain(agents, evals):
            proc.kube.resource_request(cpu=1.5)
            proc.kube.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
            proc.kube.restart_policy('Never')
            # required services
            for k, v in agent_selector.items():
                proc.kube.node_selector(key=k, value=v)
            proc.kube.resource_request(**agent_resource_request)
            proc.kube.resource_limit(**agent_resource_limit)

        learner.kube.set_env('DISABLE_MUJOCO_RENDERING', "1")
        learner.kube.resource_request(**nonagent_resource_request)
        if 'nvidia.com/gpu' in nonagent_resource_limit: 
        # Note/TODO: We are passing resource limits as kwargs, so '/' cannot happen here
        # Should we change this?
            nonagent_resource_limit['gpu'] = nonagent_resource_limit['nvidia.com/gpu']
            del nonagent_resource_limit['nvidia.com/gpu']
        learner.kube.resource_limit(**nonagent_resource_limit)
        # learner.kube.resource_limit(gpu=0)
        
        non_agent_selector = nonagent_pod_spec.get('selector', {})
        for k, v in non_agent_selector.items():
            nonagent.kube.node_selector(key=k, value=v)
        nonagent.kube.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
        nonagent.kube.image_pull_policy('Always')

        c.launch(exp)

    def create_tensorboard(self,
                           remote_path,
                           jinja_template,
                           rendered_path,
                           tensorboard_pod_type,
                           ):
        """
        Create a standalone pod under namespace "tb-remote-folder". The "remote-folder"
        part is determined from the last two level of folders in remote_path,
        and replace any special character with "-"

        Args:
            remote_path: absolute path to the experiment folder
                "tensorboard" should be a subfolder under remote_path
            jinja_template: tensorboard_template.yml file path
            rendered_path: rendered yaml file path
            tensorboard_pod_type: key to spec defined in
                `pod_types` section of .surreal.yml

        Returns:
            newly created namespace for the tensorboard pod
        """
        remote_parts = U.f_split_path(remote_path, normpath=True)
        namespace = 'tb-' + '-'.join(remote_parts[-2:])
        namespace = namespace.replace('.', '-').replace('_', '-')
        C = self.config
        context = {
            'TENSORBOARD_CMD':
                'tensorboard --logdir {} --port 6006'.format(remote_path)
        }
        # Mount file system from
        if C.fs.type.lower() in ['temp', 'temporary', 'emptydir']:
            context['FS_TYPE'] = 'emptyDir'
            context['FS_SERVER'] = None
            context['FS_PATH_ON_SERVER'] = None
        elif C.fs.type.lower() in ['localhost', 'hostpath']:
            context['FS_TYPE'] = 'hostPath'
            context['FS_SERVER'] = None
            context['FS_PATH_ON_SERVER'] = C.fs.path_on_server
        elif C.fs.type.lower() in ['nfs']:
            context['FS_TYPE'] = 'nfs'
            context['FS_SERVER'] = C.fs.server
            context['FS_PATH_ON_SERVER'] = C.fs.path_on_server
        else:
            raise NotImplementedError('Unsupported file server type: "{}". '
                                      'Supported options are [emptyDir, hostPath, nfs]'.format(C.fs.type))
        context['FS_MOUNT_PATH'] = C.fs.mount_path

        assert tensorboard_pod_type in C.pod_types, \
            'tensorboard pod type not found in `pod_types` section in ~/.surreal.yml'
        agent_pod_spec = C.pod_types[tensorboard_pod_type]
        context['AGENT_IMAGE'] = agent_pod_spec.image
        context['AGENT_SELECTOR'] = agent_pod_spec.get('selector', {})
        context['AGENT_RESOURCE_REQUEST'] = \
            agent_pod_spec.get('resource_request', {})
        context['AGENT_RESOURCE_LIMIT'] = \
            agent_pod_spec.get('resource_limit', {})

        self.create(
            namespace=namespace,
            jinja_template=jinja_template,
            rendered_path=rendered_path,
            context=context,
            check_experiment_exists=False,
        )
        return namespace

    def delete(self, namespace, yaml_path=None):
        """
        kubectl delete -f kurreal.yml --namespace <experiment_name>
        kubectl delete namespace <experiment_name>

        Notes:
            Delete a namespace will automatically delete all resources under it.

        Args:
            namespace
            yaml_path: if None, delete the namespace.
        """
        check_valid_dns(namespace)
        if yaml_path:
            if not U.f_exists(yaml_path) and not self.dry_run:
                raise FileNotFoundError(yaml_path + ' does not exist, cannot stop.')
            self.run_verbose(
                'delete -f "{}" --namespace {}'
                    .format(yaml_path, namespace),
                print_out=True, raise_on_error=False
            )
        self.run_verbose(
            'delete namespace {}'.format(namespace),
            print_out=True, raise_on_error=False
        )

    def current_context(self):
        out, err, retcode = self.run_verbose(
            'config current-context', print_out=False, raise_on_error=True
        )
        return out

    def current_namespace(self):
        """
        Parse from `kubectl config view`
        """
        config = self.config_view()
        current_context = self.current_context()
        if self.dry_run:
            return 'dummy-namespace'
        for context in config['contexts']:
            if context['name'] == current_context:
                return context['context']['namespace']
        raise RuntimeError('INTERNAL: current context not found')

    def list_namespaces(self):
        all_names = self.query_resources('namespace', output_format='name')
        # names look like namespace/<actual_name>, need to postprocess
        return [n.split('/')[-1] for n in all_names]

    def set_namespace(self, namespace):
        """
        https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
        After this call, all subsequent `kubectl` will default to the namespace
        """
        check_valid_dns(namespace)
        _, _, retcode = self.run_verbose(
            'config set-context $(kubectl config current-context) --namespace='
            + namespace,
            print_out=True, raise_on_error=False
        )
        if not self.dry_run and retcode == 0:
            print('successfully switched to namespace `{}`'.format(namespace))

    def _deduplicate_with_order(self, seq):
        """
        https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
        deduplicate list while preserving order
        """
        return list(OrderedDict.fromkeys(seq))

    def fuzzy_match_namespace(self, name, max_matches=10):
        """
        Fuzzy match namespace, precedence from high to low:
        1. exact match of <prefix + name>, if prefix option is turned on in ~/.surreal.yml
        2. exact match of <name> itself
        3. starts with <prefix + name>, sorted alphabetically
        4. starts with <name>, sorted alphabetically
        5. contains <name>, sorted alphabetically
        Up to `max_matches` number of matches

        Returns:
            - string if the matching is exact
            - OR list of fuzzy matches
        """
        all_names = self.list_namespaces()
        prefixed_name = self.prefix_username(name)
        if prefixed_name in all_names:
            return prefixed_name
        if name in all_names:
            return name
        # fuzzy matching
        matches = []
        matches += sorted([n for n in all_names if n.startswith(prefixed_name)])
        matches += sorted([n for n in all_names if n.startswith(name)])
        matches += sorted([n for n in all_names if name in n])
        matches = self._deduplicate_with_order(matches)
        return matches[:max_matches]

    def label_nodes(self, old_labels, new_label_name, new_label_value):
        """
        Select nodes that comply with `old_labels` spec, and assign them
        a set of new nodes: `label:value`
        https://kubernetes.io/docs/concepts/configuration/assign-pod-node/
        """
        node_names = self.query_resources('node', 'name', labels=old_labels)
        for node_name in node_names:
            new_label_string = shlex.quote('{}={}'.format(
                new_label_name, new_label_value
            ))
            # no need for `kubectl label nodes` because the `names` returned
            # will have fully qualified name "nodes/my-node-name`
            self.run_verbose('label --overwrite {} {}'.format(
                node_name, new_label_string
            ))

    def _get_selectors(self, labels, fields):
        """
        Helper for list_resources and list_jsonpath
        """
        labels, fields = labels.strip(), fields.strip()
        cmd= ' '
        if labels:
            cmd += '--selector ' + shlex.quote(labels)
        if fields:
            cmd += ' --field-selector ' + shlex.quote(fields)
        return cmd

    def query_resources(self, resource,
                        output_format,
                        names=None,
                        labels='',
                        fields='',
                        namespace=''):
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
            names: list of names to get resource, mutually exclusive with
                label and field selectors. Should only specify one.
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
        if names and (labels or fields):
            raise ValueError('names and (labels or fields) are mutually exclusive')
        cmd = 'get ' + resource
        cmd += self._get_ns_cmd(namespace)
        if names is None:
            cmd += self._get_selectors(labels, fields)
        else:
            assert isinstance(names, (list, tuple))
            cmd += ' ' + ' '.join(names)
        if '=' in output_format:
            # quoting the part after jsonpath=<...>
            prefix, arg = output_format.split('=', 1)
            output_format = prefix + '=' + shlex.quote(arg)
        cmd += ' -o ' + output_format
        out, _, _ = self.run_verbose(cmd, print_out=False, raise_on_error=True)
        if output_format == 'yaml':
            return EzDict.loads_yaml(out)
        elif output_format == 'json':
            return EzDict.loads_json(out)
        elif output_format == 'name':
            return out.split('\n')
        else:
            return out

    def query_jsonpath(self, resource,
                       jsonpath,
                       names=None,
                       labels='',
                       fields='',
                       namespace=''):
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
        if '{' not in jsonpath:
            jsonpath = '{' + jsonpath + '}'
        jsonpath = '{range .items[*]}' + jsonpath + '{"\\n\\n"}{end}'
        output_format = "jsonpath=" + jsonpath
        out = self.query_resources(
            resource=resource,
            names=names,
            output_format=output_format,
            labels=labels,
            fields=fields,
            namespace=namespace
        )
        return out.split('\n\n')

    def config_view(self):
        """
        kubectl config view
        Generates a yaml of context and cluster info
        """
        out, err, retcode = self.run_verbose(
            'config view', print_out=False, raise_on_error=True
        )
        return EzDict.loads_yaml(out)

    def external_ip(self, pod_name, namespace=''):
        """
        Returns:
            "<ip>:<port>"
        """
        tb = self.query_resources('svc', 'yaml',
                                  names=[pod_name], namespace=namespace)
        conf = tb.status.loadBalancer
        if not ('ingress' in conf and 'ip' in conf.ingress[0]):
            return ''
        ip = conf.ingress[0].ip
        port = tb.spec.ports[0].port
        return '{}:{}'.format(ip, port)

    def _get_ns_cmd(self, namespace):
        if namespace:
            return ' --namespace ' + namespace
        else:
            return ''

    def _get_logs_cmd(self, pod_name, container_name,
                      follow, since=0, tail=-1, namespace=''):
        return 'logs {} {} {} --since={} --tail={}{}'.format(
            pod_name,
            container_name,
            '--follow' if follow else '',
            since,
            tail,
            self._get_ns_cmd(namespace)
        )

    def logs(self, pod_name,
             container_name='',
             since=0,
             tail=100,
             namespace=''):
        """
        kubectl logs <pod_name> <container_name> --follow --since= --tail=
        https://kubernetes-v1-4.github.io/docs/user-guide/kubectl/kubectl_logs/

        Returns:
            stdout string
        """
        out, err, retcode = self.run_verbose(
            self._get_logs_cmd(
                pod_name, container_name, follow=False,
                since=since, tail=tail, namespace=namespace
            ),
            print_out=False,
            raise_on_error=False
        )
        if retcode != 0:
            return ''
        else:
            return out

    def describe(self, pod_name, namespace=''):
        cmd = 'describe pod ' + pod_name + self._get_ns_cmd(namespace)
        return self.run_verbose(cmd, print_out=True, raise_on_error=False)

    def print_logs(self, pod_name,
                   container_name='',
                   follow=False,
                   since=0,
                   tail=100,
                   namespace=''):
        """
        kubectl logs <pod_name> <container_name>
        No error checking, no string caching, delegates to os.system
        """
        cmd = self._get_logs_cmd(
            pod_name, container_name, follow=follow,
            since=since, tail=tail, namespace=namespace
        )
        self.run_raw(cmd)

    def logs_surreal(self, component_name, is_print=False,
                     follow=False, since=0, tail=100, namespace=''):
        """
        Args:
            component_name: can be agent-N, learner, ps, replay, tensorplex, tensorboard

        Returns:
            stdout string if is_print else None
        """
        if is_print:
            log_func = functools.partial(self.print_logs, follow=follow)
        else:
            log_func = self.logs
        log_func = functools.partial(
            log_func, since=since, tail=tail, namespace=namespace
        )
        if component_name in self.NONAGENT_COMPONENTS:
            return log_func('nonagent', component_name)
        else:
            return log_func(component_name)

    def exec_surreal(self, component_name, cmd, namespace=''):
        """
        kubectl exec -ti

        Args:
            component_name: can be agent-N, learner, ps, replay, tensorplex, tensorboard
            cmd: either a string command or a list of command args

        Returns:
            stdout string if is_print else None
        """
        if U.is_sequence(cmd):
            cmd = ' '.join(map(shlex.quote, cmd))
        ns_cmd = self._get_ns_cmd(namespace)
        if component_name in self.NONAGENT_COMPONENTS:
            return self.run_raw(
                'exec -ti nonagent -c {}{} -- {}'.format(component_name, ns_cmd, cmd)
            )
        else:
            return self.run_raw(
                'exec -ti {}{} -- {}'.format(component_name, ns_cmd, cmd)
            )

    def scp_surreal(self, src_file, dest_file, namespace=''):
        """
        https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#cp
        kurreal cp /my/local/file learner:/remote/file mynamespace
        is the same as
        kubectl cp /my/local/file mynamespace/nonagent:/remote/file -c learner
        """
        def _split(f):
            if ':' in f:
                pod, path = f.split(':')
                container = None
                if pod in self.NONAGENT_COMPONENTS:
                    container = pod
                    pod = 'nonagent'
            else:
                pod, path, container = None, f, None
            if pod and namespace:
                pod = namespace + '/' + pod
            if pod:
                path = pod + ':' + path
            return pod, path, container

        src_pod, src_path, src_container = _split(src_file)
        dest_pod, dest_path, dest_container = _split(dest_file)
        assert bool(src_pod) != bool(dest_pod), \
            'one of "src_file" and "dest_file" must be remote and the other local.'
        container = src_container or dest_container  # at least one is None
        cmd = 'cp {} {}'.format(src_path, dest_path)
        if container:
            cmd += ' -c ' + container
        self.run_raw(cmd, print_cmd=True)

    def gcloud_get_config(self, config_key):
        """
        Returns: value of the gcloud config
        https://cloud.google.com/sdk/gcloud/reference/config/get-value
        for more complex outputs, add --format="json" to gcloud command
        """
        out, _, _ = self.run_verbose(
            'config get-value ' + config_key,
            print_out=False,
            raise_on_error=True,
            program='gcloud'
        )
        return out.strip()

    def gcloud_zone(self):
        """
        Returns: current gcloud zone
        """
        return self.gcloud_get_config('compute/zone')

    def gcloud_project(self):
        """
        Returns: current gcloud project
        https://cloud.google.com/sdk/gcloud/reference/config/get-value
        for more complex outputs, add --format="json" to gcloud command
        """
        return self.gcloud_get_config('project')

    def gcloud_configure_ssh(self):
        """
        Refresh the ssh settings for the current gcloud project
        populate SSH config files with Host entries from each instance
        https://cloud.google.com/sdk/gcloud/reference/compute/config-ssh
        """
        return self.run_raw('compute config-ssh', program='gcloud')

    def gcloud_url(self, node_name):
        """
        Returns: current gcloud project
        https://cloud.google.com/sdk/gcloud/reference/config/get-value
        for more complex outputs, add --format="json" to gcloud command
        """
        return '{}.{}.{}'.format(
            node_name, self.gcloud_zone(), self.gcloud_project()
        )

    def gcloud_ssh_node(self, node_name):
        """
        Don't forget to run gcloud_config_ssh() first
        """
        url = self.gcloud_url(node_name)
        return self.run_raw(
            'ssh -o StrictHostKeyChecking=no ' + url,
            program='',
            print_cmd=True
        )

    def gcloud_ssh_fs(self):
        """
        ssh into the file system server specified in ~/.surrreal.yml
        """
        return self.gcloud_ssh_node(self.config.fs.server)

    def capture_tensorboard(self, namespace):
        node_path = self.config.capture_tensorboard.node_path
        library_path = self.config.capture_tensorboard.library_path
        url = 'http://{}'.format(self.external_ip('tensorboard',namespace=namespace))
        save_path = os.path.join(os.getcwd(), '{}.jpg'.format(namespace))
        WIDTH = 5000
        HEIGHT = 3300
        cmd = [node_path, library_path, url, save_path, str(WIDTH), str(HEIGHT)]
        print('>> {:s}'.format(' '.join(cmd)))
        p = pc.Popen(cmd)
        return p


if __name__ == '__main__':
    import pprint
    pp = pprint.pprint
    kube = Kubectl(dry_run=0)
    # kube.scp_surreal('/my/local', 'learner:/my/remote', 'myNS')
    # kube.scp_surreal('/my/local', 'agent-0:/my/remote', 'myNS')
    # kube.scp_surreal('/my/local', 'agent-0:/my/remote', '')
    # kube.scp_surreal('loggerplex:/my/remote', '/my/local', 'NS')
    # print(kube.gcloud_url('surreal-shared-fs-vm'))
    # 3 different ways to get a list of node names
    # pp(kube.query_jsonpath('nodes', '{.metadata.name}'))
    # pp(kube.query_jsonpath('nodes', "{.metadata.labels['kubernetes\.io/hostname']}"))
    # pp(kube.query_resources('nodes', 'name'))
    # yaml for pods
    # pp(kube.query_resources('pods', 'json', fields='metadata.name=agent-0').dumps_yaml())
