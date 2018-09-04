import os
import argparse
import itertools
import re
from copy import copy
from pathlib import Path
from symphony.commandline import SymphonyParser
from symphony.engine import SymphonyConfig, Cluster
from symphony.kube import KubeCluster, GKEMachineDispatcher
from symphony.addons import DockerBuilder, clean_images
from benedict import BeneDict
import surreal
import subprocess
from surreal.kube.generate_command import CommandGenerator
import surreal.utils as U
import pkg_resources


def _process_labels(label_string):
    """
    mylabel1=myvalue1,mylabel2=myvalue2
    """
    assert '=' in label_string
    label_pairs = label_string.split(',')
    return [label_pair.split('=') for label_pair in label_pairs]


def _merge_setting_dictionaries(customize, base):
    di = copy(base)
    for key in di:
        if isinstance(di[key], dict):
            if key in customize:
                di[key] = _merge_setting_dictionaries(customize[key], di[key])
        else:
            if key in customize:
                di[key] = customize[key]
    return di


class KurrealParser(SymphonyParser):
    def create_cluster(self):
        return Cluster.new('kube')

    def setup(self):
        super().setup()
        self.docker_build_settings = {}
        self.config = BeneDict()
        self.load_config()
        self._setup_create()
        self._setup_create_dev()
        # self._setup_tensorboard()
        # self._setup_docker_clean()

        # Secondary nfs related support
        # self._setup_get_videos()
        # self._setup_get_config()
        # self._setup_get_tensorboard()

    def load_config(self, surreal_yml='~/.surreal.yml'):
        surreal_yml = U.f_expand(surreal_yml)
        if not U.f_exists(surreal_yml):
            raise ValueError('Cannot find surreal config file at {}'.format(surreal_yml))
        self.config = BeneDict.load_yaml_file(surreal_yml)
        SymphonyConfig().set_username(self.username)
        SymphonyConfig().set_experiment_folder(self.folder)

        if 'docker_build_settings' in self.config:
            for setting in self.config['docker_build_settings']:
                self.docker_build_settings[setting['name']] = setting

    @property
    def folder(self):
        return U.f_expand(self.config.local_kurreal_folder)

    @property
    def username(self):
        assert 'username' in self.config, 'must specify username in ~/.surreal.yml'
        return self.config.username

    # def _setup_get_videos(self):
    #     parser = self.add_subparser('get-videos', aliases=['gv'])
    #     parser.add_argument('experiment_names', nargs='*', type=str, metavar='experiment_name',
    #                         help='experiments to retrieve videos for, '
    #                         'none to retrieve your own running experiments')
    #     parser.add_argument('--last', type=int, default=5, metavar='last_n_videos',
    #                         help='Number of most recent videos, -1 to get all')
    #     parser.add_argument('save_folder', type=str,
    #                         help='save_videos in [save_folder]/experiment_name')

    # def _setup_get_config(self):
    #     parser = self.add_subparser('get-config', aliases=['gc'])
    #     parser.add_argument('experiment_name', type=str,
    #                         help='experiments to retrieve videos for, '
    #                              'none to retrieve your own running experiments')
    #     parser.add_argument('-o', '--output-file', type=str,
    #                         help='save remote config to a specified local file path')

    # def _setup_get_tensorboard(self):
    #     parser = self.add_subparser('get-tensorboard', aliases=['gt'])
    #     parser.add_argument('experiment_name', type=str,
    #                         help='experiments to retrieve tensorboard for, '
    #                              'none to retrieve your own running experiments')
    #     parser.add_argument('-s', '--subfolder', type=str, default='',
    #                         help='retrieve only a subfolder under the "tensorboard" folder. '
    #                              'currently valid folders are agent, eval, learner, replay')
    #     parser.add_argument('-o', '--output-folder', type=str,
    #                         help='save remote TB folder to a specified local folder path')

    def _setup_docker_clean(self):
        parser = self.add_subparser('docker-clean', aliases=['dc'])

    def _setup_tensorboard(self):
        parser = self.add_subparser('tensorboard', aliases=['tb'])
        self._add_experiment_name(parser, required=False, positional=True)
        parser.set_defaults(service_name='tensorboard')
        parser.add_argument(
            '-u', '--url-only',
            action='store_true',
            help='only show the URL without opening the browser.'
        )

    def _setup_create(self):
        parser = self.add_subparser('create', aliases=['c'])
        self._add_experiment_name(parser)
        parser.add_argument(
            'setting_name',
            type=str,
            help='the setting in .surreal.yml that specifies how an'
                 'experiment should be run')
        parser.add_argument(
            'algorithm',
            type=str,
            help='ddpg / ppo or the'
                 'location of algorithm python script **in the docker container**'
                 'home dir, i.e. /root/ on the pod'
        )
        parser.add_argument(
            '--num_agent',
            type=int,
            default=None,
            help='number of agents to run in parallel.'
        )
        parser.add_argument(
            '--num_eval',
            type=int,
            default=None,
            help='number of evals to run in parallel.'
        )
        parser.add_argument(
            '--batch_agent',
            type=int,
            default=None,
            help='put how many agent on each agent machine'
        )
        parser.add_argument(
            '--batch_eval',
            type=int,
            default=None,
            help='put how many eval on each eval machine'
        )

        parser.add_argument(
            '-f', '--force',
            action='store_true',
            help='force overwrite an existing kurreal.yml file '
                 'if its experiment folder already exists.'
        )
        self._add_dry_run(parser)

    # ==================== helpers ====================
    def _add_dry_run(self, parser):
        parser.add_argument(
            '-dr', '--dry-run',
            action='store_true',
            help='print the kubectl command without actually executing it.'
        )

    def _process_experiment_name(self, experiment_name):
        """
        experiment_name will be used as DNS, so must not have underscore or dot
        """
        new_name = experiment_name.lower().replace('.', '-').replace('_', '-')
        if new_name != experiment_name:
            print('experiment name string has been fixed: {} -> {}'
                  .format(experiment_name, new_name))
        return new_name

    SUPPORTED_MODES = ['basic', 'batch']
    def action_create(self, args):
        """
        Spin up a multi-node distributed Surreal experiment.
        Put any command line args that pass to the config script after "--"
        """
        setting_name = args.setting_name

        if not setting_name in self.config.creation_settings:
            raise KeyError('Cannot find setting {}'.format(setting_name))
        setting = self.config.creation_settings[setting_name]
        mode = setting['mode']
        if mode not in self.SUPPORTED_MODES:
            raise ValueError('Unknown mode {}'.format(mode) +
                             'available options are : {}'.format(
                                 ', '.join(self.SUPPORTED_MODES)
                             ))
        if mode == 'basic':
            self.create_basic(
                setting=setting,
                experiment_name=args.experiment_name,
                algorithm_args=args.remainder,
                custom_input=vars(args),
                force=args.force,
                dry_run=args.dry_run,
                )
        elif mode == 'batch':
            self.create_batch(
                setting=setting,
                algorithm_args=args.remainder,
                experiment_name=args.experiment_name,
                custom_input=vars(args),
                force=args.force,
                dry_run=args.dry_run,
                )

    DEFAULT_SETTING_BASIC = {
        'algorithm': 'ppo',
        'num_agents': 2,
        'num_evals': 3,
        'compute_additional_args': True,
        'agent': {
            'image': 'surreal-cpu-image',  # TODO
            'node_pool': 'surreal-default-cpu-nodepool',  # TODO
            'build_image': None
        },
        'nonagent': {
            'image': 'surreal-cpu-image',  # TODO
            'node_pool': 'surreal-default-cpu-nodepool',  # TODO
            'build_image': None
        },
    }
    def create_basic(self, *,
                     setting,
                     experiment_name,
                     algorithm_args,
                     custom_input,
                     force,
                     dry_run):
        setting = _merge_setting_dictionaries(setting,
                                              self.DEFAULT_SETTING_BASIC)
        setting = _merge_setting_dictionaries(custom_input, setting)
        setting = BeneDict(setting)

    DEFAULT_SETTING_BATCH = {
        'algorithm': 'ppo',
        'num_agents': 16,
        'num_evals': 8,
        'compute_additional_args': True,
        'agent_batch': 8,
        'eval_batch': 8,
        'agent': {
            'image': 'surreal-cpu-image',  # TODO
            'node_pool': 'surreal-default-cpu-nodepool',  # TODO
            'build_image': None
        },
        'nonagent': {
            'image': 'surreal-cpu-image',  # TODO
            'node_pool': 'surreal-default-cpu-nodepool',  # TODO
            'build_image': None
        },
    }
    def create_batch(self, *,
                     setting,
                     experiment_name,
                     algorithm_args,
                     custom_input,
                     force,
                     dry_run):
        setting = _merge_setting_dictionaries(setting,
                                              self.DEFAULT_SETTING_BATCH)
        setting = _merge_setting_dictionaries(custom_input, setting)
        setting = BeneDict(setting)

    def action_create_dev(self, args):
        """
        << internal dev only >>
        """
        assert not args.has_remainder, \
            'create_dev cannot have "--". Use --env and --gpu'

        if args.env:
            env = args.env
        else:
            env = 'gym:HalfCheetah-v2'
        config_command = ['--env', env]

        if args.num_gpus is None:  # nargs=?, num gpu should be 1 when omitted
            num_gpus = 1
        else:
            num_gpus = args.num_gpus

        if args.gpu_type == 'k80':
            POD_TYPES = {
                0: 'nonagent-cpu',
                1: 'nonagent-gpu',
                2: 'nonagent-2k80-16cpu',
                4: 'nonagent-4k80-32cpu',
            }
        elif args.gpu_type == 'p100':
            POD_TYPES = {
                0: 'nonagent-cpu',
                1: 'nonagent-gpu-p100',
                4: 'nonagent-gpu-4p100',
            }
        elif args.gpu_type == 'v100':
            POD_TYPES = {
                0: 'nonagent-cpu',
                1: 'nonagent-gpu-v100',
                4: 'nonagent-gpu-4v100',
            }
        else:
            raise ValueError('Unknown GPU type: {}'.format(args.gpu_type))
        if num_gpus not in POD_TYPES:
            raise ValueError('invalid number of GPUs, choose from {}'
                             .format(list(POD_TYPES.keys())))
        nonagent_pod_type = POD_TYPES[num_gpus]
        config_command += ["--num-gpus", str(num_gpus)]
        config_command += ["--num-agents", str(args.num_agents)]
        # '/mylibs/surreal/surreal/surreal/main/ddpg_configs.py'
        config_py = 'surreal/surreal/main/' + args.config_file

        if args.batch_agent > 1:
            agent_pod_type = 'agent-mj-batch'
            nonagent_pod_type = 'nonagent-mj-batch'
            if args.gpu_type == 'p100':
                nonagent_pod_type = 'nonagent-mj-batch-p100'
            # if args.gpu_type == 'v100':
            #     nonagent_pod_type = 'nonagent-mj-batch-v100'
            eval_pod_type = 'agent-mj-batch'
            config_command += ["--agent-num-gpus", '1']
            num_evals = 8
        else:
            agent_pod_type = 'agent'
            eval_pod_type = 'agent'
            num_evals = 1

        self._create_helper(
            config_py=config_py,
            experiment_name=args.experiment_name,
            num_agents=args.num_agents,
            config_command=config_command,
            agent_pod_type=agent_pod_type,
            nonagent_pod_type=nonagent_pod_type,
            eval_pod_type=eval_pod_type,
            restore=False,
            restore_folder=None,
            force=args.force,
            dry_run=args.dry_run,
            colocate_agent=args.colocate_agent,
            batch_agent=args.batch_agent,
            num_evals=num_evals,
            has_eval=not args.no_eval,
        )

    def _create_helper(self, *,
                       config_py,
                       experiment_name,
                       num_agents,
                       config_command,
                       agent_pod_type,
                       nonagent_pod_type,
                       eval_pod_type,
                       restore,
                       restore_folder,
                       force,
                       num_evals,
                       colocate_agent=1,
                       batch_agent=1,
                       dry_run=False,
                       has_eval=True):
        if colocate_agent > 1 and batch_agent > 1:
            raise ValueError('Cannot colocate and batch at the same time')
        if config_py.startswith('/'):
            config_py = config_py
        else:
            config_py = U.f_join('/mylibs', config_py)

        remote_experiment_folder = self.get_remote_experiment_folder(experiment_name)

        cmd_gen = CommandGenerator(
            num_agents=num_agents,
            experiment_folder=remote_experiment_folder,  # TODO: fixme
            config_py=config_py,
            config_command=config_command,
            restore=restore,
            restore_folder=restore_folder,
            batch_agent=batch_agent,
            num_evals=num_evals,
        )
        cmd_dict = cmd_gen.generate()
        print('  agent_pod_type:', agent_pod_type)
        print('  nonagent_pod_type:', nonagent_pod_type)
        print('  eval_pod_type:', eval_pod_type)

        self.create_surreal(
            experiment_name,
            agent_pod_type=agent_pod_type,
            nonagent_pod_type=nonagent_pod_type,
            eval_pod_type=eval_pod_type,
            cmd_dict=cmd_dict,
            force=force,
            dry_run=dry_run,
            colocate_agent=colocate_agent,
            batch_agent=batch_agent,
            has_eval=has_eval,
        )

    def create_surreal(self,
                       experiment_name,
                       agent_pod_type,
                       nonagent_pod_type,
                       cmd_dict,
                       eval_pod_type,
                       colocate_agent=1,
                       batch_agent=1,
                       force=False,
                       dry_run=False,
                       has_eval=True):
        """
        Then create a surreal experiment
        Args:
            experiment_name: will also be used as hostname for DNS
            agent_pod_type: key to spec defined in `pod_types` section of .surreal.yml
            nonagent_pod_type: key to spec defined in `pod_types` section of .surreal.yml
            cmd_dict: dict of commands to be run on each container
            force: check if the Kube yaml has already been generated.
            dry_run: only print yaml, do not actually launch
        """
        C = self.config
        cluster = Cluster.new('kube')
        exp = cluster.new_experiment(experiment_name)

        # Read pod specifications
        assert agent_pod_type in C.pod_types, \
            'agent pod type not found in `pod_types` section in ~/.surreal.yml'
        assert nonagent_pod_type in C.pod_types, \
            'nonagent pod type not found in `pod_types` section in ~/.surreal.yml'
        if eval_pod_type is None: eval_pod_type = agent_pod_type
        assert eval_pod_type in C.pod_types, \
            'eval pod type not found in `pod_types` section in ~/.surreal.yml'

        agent_pod_spec = copy(C.pod_types[agent_pod_type])
        nonagent_pod_spec = copy(C.pod_types[nonagent_pod_type])
        eval_pod_spec = copy(C.pod_types[eval_pod_type])

        json_path = 'cluster_definition.tf.json'  # always use slash
        filepath = pkg_resources.resource_filename(__name__, json_path)
        dispatcher = GKEMachineDispatcher(filepath)

        agent_node_pool = agent_pod_spec["node_pool"]
        nonagent_node_pool = nonagent_pod_spec["node_pool"]
        eval_node_pool = eval_pod_spec["node_pool"]

        # agent_resource_request = agent_pod_spec.get('resource_request', {})
        # nonagent_resource_request = nonagent_pod_spec.get('resource_request', {})
        # eval_resource_request = eval_pod_spec.get('resource_request', {})

        # agent_resource_limit = agent_pod_spec.get('resource_limit', {})
        # nonagent_resource_limit = nonagent_pod_spec.get('resource_limit', {})
        # eval_resource_limit = eval_pod_spec.get('resource_limit', {})

        images_to_build = {}
        # defer to build last, so we don't build unless everythingpasses
        if 'build_image' in agent_pod_spec:
            image_name = agent_pod_spec['build_image']
            images_to_build[image_name] = agent_pod_spec['image']
            # Use experiment_name as tag
            agent_pod_spec['image'] = '{}:{}'.format(agent_pod_spec['image'], exp.name)
        if 'build_image' in nonagent_pod_spec:
            image_name = nonagent_pod_spec['build_image']
            images_to_build[image_name] = nonagent_pod_spec['image']
            nonagent_pod_spec['image'] = '{}:{}'.format(nonagent_pod_spec['image'], exp.name)
        if has_eval:
            if 'build_image' in eval_pod_spec:
                image_name = eval_pod_spec['build_image']
                images_to_build[image_name] = eval_pod_spec['image']
                eval_pod_spec['image'] = '{}:{}'.format(eval_pod_spec['image'], exp.name)

        nonagent = exp.new_process_group('nonagent')
        learner = nonagent.new_process('learner', container_image=nonagent_pod_spec.image, args=[cmd_dict['learner']])
        replay = nonagent.new_process('replay', container_image=nonagent_pod_spec.image, args=[cmd_dict['replay']])
        ps = nonagent.new_process('ps', container_image=nonagent_pod_spec.image, args=[cmd_dict['ps']])
        tensorboard = nonagent.new_process('tensorboard', container_image=nonagent_pod_spec.image, args=[cmd_dict['tensorboard']])
        tensorplex = nonagent.new_process('tensorplex', container_image=nonagent_pod_spec.image, args=[cmd_dict['tensorplex']])
        loggerplex = nonagent.new_process('loggerplex', container_image=nonagent_pod_spec.image, args=[cmd_dict['loggerplex']])

        agents = []
        agent_pods = []
        if colocate_agent > 1:
            assert len(cmd_dict['agent']) % colocate_agent == 0
            for i in range(int(len(cmd_dict['agent']) / colocate_agent)):
                agent_pods.append(exp.new_process_group('agent-pg-{}'.format(i)))
            for i, arg in enumerate(cmd_dict['agent']):
                pg_index = int(i / colocate_agent)
                agent_p = agent_pods[pg_index].new_process('agent-{}'.format(i), container_image=agent_pod_spec.image, args=[arg])
                agents.append(agent_p)
        elif batch_agent > 1:
            for i, arg in enumerate(cmd_dict['agent-batch']):
                agent_p = exp.new_process('agents-{}'.format(i), container_image=agent_pod_spec.image, args=[arg])
                agent_pods.append(agent_p)
                agents.append(agent_p)
        else:
            for i, arg in enumerate(cmd_dict['agent']):
                agent_p = exp.new_process('agent-{}'.format(i), container_image=agent_pod_spec.image, args=[arg])
                agent_pods.append(agent_p)
                agents.append(agent_p)
        evals = []
        if has_eval:
            if batch_agent > 1:
                for i, arg in enumerate(cmd_dict['eval-batch']):
                    eval_p = exp.new_process('evals-{}'.format(i), container_image=eval_pod_spec.image, args=[arg])
                    evals.append(eval_p)
            else:
                for i, arg in enumerate(cmd_dict['eval']):
                    eval_p = exp.new_process('eval-{}'.format(i), container_image=eval_pod_spec.image, args=[arg])
                    evals.append(eval_p)

        for proc in itertools.chain(agents, evals):
            proc.connects('ps-frontend')
            proc.connects('collector-frontend')

        ps.binds('ps-frontend')
        ps.binds('ps-backend')
        ps.connects('parameter-publish')

        replay.binds('collector-frontend')
        replay.binds('sampler-frontend')
        replay.binds('collector-backend')
        replay.binds('sampler-backend')

        learner.connects('sampler-frontend')
        learner.binds('parameter-publish')
        learner.binds('prefetch-queue')

        tensorplex.binds('tensorplex')
        loggerplex.binds('loggerplex')

        for proc in itertools.chain(agents, evals, [ps, replay, learner]):
            proc.connects('tensorplex')
            proc.connects('loggerplex')

        tensorboard.exposes({'tensorboard': 6006})

        if not C.fs.type.lower() in ['nfs']:
            raise NotImplementedError('Unsupported file server type: "{}". '
                                      'Supported options are [nfs]'.format(C.fs.type))
        nfs_server = C.fs.server
        nfs_server_path = C.fs.path_on_server
        nfs_mount_path = C.fs.mount_path

        for proc in exp.list_all_processes():
            # Mount nfs
            proc.mount_nfs(server=nfs_server, path=nfs_server_path, mount_path=nfs_mount_path)

        # resource_limit_gpu(agent_resource_limit)
        # agent_selector = agent_pod_spec.get('selector', {})
        for proc in agents:
            # required services
            # proc.resource_request(**agent_resource_request)
            # proc.resource_limit(**agent_resource_limit)
            proc.image_pull_policy('Always')

        for proc_g in agent_pods:
            # proc_g.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
            proc_g.restart_policy('Never')
            dispatcher.assign_to_nodepool(proc_g, agent_node_pool,
                                          process_group=proc_g, exclusive=True)
            # for k, v in agent_selector.items():
            #     proc_g.node_selector(key=k, value=v)

        # resource_limit_gpu(eval_resource_limit)
        # eval_selector = eval_pod_spec.get('selector', {})
        for eval_p in evals:
            # eval_p.resource_request(**eval_resource_request)
            # eval_p.resource_limit(**eval_resource_limit)
            dispatcher.assign_to_nodepool(eval_p, eval_node_pool, exclusive=True)
            eval_p.image_pull_policy('Always')
            # eval_p.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
            eval_p.restart_policy('Never')
            # for k, v in eval_selector.items():
            #     eval_p.node_selector(key=k, value=v)

        learner.set_env('DISABLE_MUJOCO_RENDERING', "1")
        # learner.resource_request(**nonagent_resource_request)

        # resource_limit_gpu(nonagent_resource_limit)
        # learner.resource_limit(**nonagent_resource_limit)

        # non_agent_selector = nonagent_pod_spec.get('selector', {})
        # for k, v in non_agent_selector.items():
        #     nonagent.node_selector(key=k, value=v)
        # nonagent.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
        nonagent.image_pull_policy('Always')

        dispatcher.assign_to_nodepool(learner,
                                      nonagent_node_pool,
                                      process_group=nonagent,
                                      exclusive=True)

        for name, repo in images_to_build.items():
            builder = DockerBuilder.from_dict(self.docker_build_settings[name])
            builder.build()
            builder.tag(repo, exp.name)
            builder.push(repo, exp.name)

        cluster.launch(exp, force=force, dry_run=dry_run)

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

    def action_docker_clean(self, args):
        """
        Cleans all docker images used to create experiments
        """
        images_to_clean = {}
        for pod_type_name, pod_type in self.config.pod_types.items():
            if 'image' in pod_type and ':' not in pod_type['image']:
                images_to_clean[pod_type['image']] = True
        images_to_clean = ['{}:*'.format(x) for x in images_to_clean.keys()]
        clean_images(images_to_clean)

def main():
    KurrealParser().main()


if __name__ == '__main__':
    main()
