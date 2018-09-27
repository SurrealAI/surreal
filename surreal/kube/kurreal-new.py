import os
import argparse
import itertools
import shlex
from copy import copy
from pathlib import Path
from symphony.commandline import SymphonyParser
from symphony.engine import SymphonyConfig, Cluster
from symphony.kube import GKEMachineDispatcher
from symphony.addons import DockerBuilder, clean_images
from benedict import BeneDict
from surreal.launch import (
    CommandGeneratorBasic,
    CommandGeneratorBatched,
    setup_network,
    SurrealDockerBuilder
)
import surreal.utils as U
import pkg_resources


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
            raise ValueError('Cannot find surreal config file at {}'
                             .format(surreal_yml))
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
                 'location of algorithm python script **in the docker '
                 'container**'
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

    SUPPORTED_MODES = ['basic']

    def action_create(self, args):
        """
            Spin up a multi-node distributed Surreal experiment.
            Put any command line args that pass to the config script after "--"
        """
        setting_name = args.setting_name

        if not setting_name in self.config.creation_settings:
            raise KeyError('Cannot find setting {}'.format(setting_name))
        settings = self.config.creation_settings[setting_name]
        mode = settings['mode']
        if mode not in self.SUPPORTED_MODES:
            raise ValueError('Unknown mode {}'.format(mode) +
                             'available options are : {}'.format(
                                 ', '.join(self.SUPPORTED_MODES)
                             ))
        if mode == 'basic':
            self.create_basic(
                settings=settings,
                experiment_name=args.experiment_name,
                algorithm_args=args.remainder,
                input_args=vars(args),
                force=args.force,
                dry_run=args.dry_run,
                )

    def _find_executable(self, name):
        """
            Finds the .py file corresponding to the algorithm specified

        Args:
            name: ddpg / ppo / <path in container to compatible .py file>
        """
        if name == 'ddpg':
            return '/mylibs/surreal/surreal/surreal/main/ddpg_configs.py'
        elif name == 'ppo':
            return '/mylibs/surreal/surreal/surreal/main/ppo_configs.py'
        else:
            return name

    DEFAULT_SETTING_BASIC = {
        'algorithm': 'ddpg',
        'num_agents': 2,
        'num_evals': 1,
        'agent_batch': 1,
        'eval_batch': 1,
        'restore_folder': None,
        'agent': {
            'image': 'surreal-cpu-image',  # TODO
            'node_pool': 'surreal-default-cpu-nodepool',  # TODO
            'cpu': None,
            'memory': None,
            'gpu': None,
            'build_image': None
        },
        'nonagent': {
            'image': 'surreal-cpu-image',  # TODO
            'node_pool': 'surreal-default-cpu-nodepool',  # TODO
            'cpu': None,
            'memory': None,
            'gpu': None,
            'build_image': None
        },
    }

    def create_basic(self, *,
                     settings,
                     experiment_name,
                     algorithm_args,
                     input_args,
                     force,
                     dry_run):
        setting = _merge_setting_dictionaries(settings,
                                              self.DEFAULT_SETTING_BASIC)
        setting = _merge_setting_dictionaries(input_args, setting)
        setting = BeneDict(setting)

        cluster = Cluster.new('kube')
        exp = cluster.new_experiment(experiment_name)

        image_builder = SurrealDockerBuilder(
            build_settings=self.config.docker_build_settings,
            images_requested={
                'agent': setting.agent.image,
                'nonagent': setting.nonagent.image
            },
            tag=experiment_name,
            push=True)
        agent_image = image_builder.images_provided['agent']
        nonagent_image = image_builder.images_provided['nonagent']
        # defer to build last, so we don't build unless everything passes

        # TODO: experiment_folder,
        # TODO: restore_folder
        # TODO: CommandGenerator
        algorithm_args += [
            "--num-agents",
            str(settings.num_agents * settings.agent_batch),
            ]

        algorithm_args += ["--restore_folder",
                           shlex.quote(settings.restore_folder)]

        # self.experiment_folder = experiment_folder
        # --experiment-folder <experiment_folder>
        executable = self._find_executable(setting.algorithm)
        cmd_gen = CommandGeneratorBasic(
            num_agents=settings.num_agents,
            num_evals=settings.num_evals,
            executable=executable)
        cmd_dict = cmd_gen.generate_command

        nonagent = exp.new_process_group('nonagent')
        learner = nonagent.new_process(
            'learner',
            container_image=nonagent_image,
            args=[cmd_dict['learner']])
        # Because learner and everything are bundled together
        # We only need to claim resources for learner
        dispatcher.assign_to_nodepool(learner,
                                      nonagent_node_pool,
                                      process_group=nonagent,
                                      exclusive=True)
        # For dm_control
        learner.set_env('DISABLE_MUJOCO_RENDERING', "1")

        replay = nonagent.new_process(
            'replay',
            container_image=nonagent_image,
            args=[cmd_dict['replay']])

        ps = nonagent.new_process(
            'ps',
            container_image=nonagent_image,
            args=[cmd_dict['ps']])

        tensorboard = nonagent.new_process(
            'tensorboard',
            container_image=nonagent_image,
            args=[cmd_dict['tensorboard']])

        tensorplex = nonagent.new_process(
            'tensorplex',
            container_image=nonagent_image,
            args=[cmd_dict['tensorplex']])

        loggerplex = nonagent.new_process(
            'loggerplex',
            container_image=nonagent_image,
            args=[cmd_dict['loggerplex']])
        nonagent.image_pull_policy('Always')

        agents = []
        for i, arg in enumerate(cmd_dict['agent']):
            if settings.agent_batch == 1:
                agent_name = 'agent-{}'.format(i)
            else:
                agent_name = 'agents-{}'.format(i)
            agent = exp.new_process(
                agent_name,
                container_image=agent_image,
                args=[cmd_gen.get_command(agent_name)])

            agent.image_pull_policy('Always')
            agent.restart_policy('Never')
            dispatcher.assign_to_nodepool(agent,
                                          agent_node_pool,
                                          process_group=agent,
                                          exclusive=True)

            agents.append(agent)

        evals = []
        for i, arg in enumerate(cmd_dict['eval']):
            if settings.eval_batch == 1:
                eval_name = 'eval-{}'.format(i)
            else:
                eval_name = 'evals-{}'.format(i)
            eval_p = exp.new_process(
                eval_name,
                container_image=agent_image,
                args=[cmd_gen.get_command(eval_name)])
            dispatcher.assign_to_nodepool(eval_p,
                                          eval_node_pool,
                                          exclusive=True)
            agent.image_pull_policy('Always')
            agent.restart_policy('Never')

            evals.append(eval_p)

        setup_network(agents=agents,
                      evals=evals,
                      learner=learner,
                      replay=replay,
                      ps=ps,
                      tensorboard=tensorboard,
                      tensorplex=tensorplex,
                      loggerplex=loggerplex)

        # TODO: nfs support
        # if not C.fs.type.lower() in ['nfs']:
        #     raise NotImplementedError('Unsupported file server type: "{}". '
        #                               'Supported options are [nfs]'.format(C.fs.type))
        # nfs_server = C.fs.server
        # nfs_server_path = C.fs.path_on_server
        # nfs_mount_path = C.fs.mount_path
        # for proc in exp.list_all_processes():
        #     proc.mount_nfs(server=nfs_server, path=nfs_server_path, mount_path=nfs_mount_path)

        image_builder.build()
        cluster.launch(exp, force=force, dry_run=dry_run)

    # DEFAULT_SETTING_BATCH = {
    #     'algorithm': 'ppo',
    #     'num_agents': 16,
    #     'num_evals': 8,
    #     'compute_additional_args': True,
    #     'agent_batch': 8,
    #     'eval_batch': 8,
    #     'agent': {
    #         'image': 'surreal-cpu-image',  # TODO
    #         'node_pool': 'surreal-default-cpu-nodepool',  # TODO
    #         'cpu': None,
    #         'memory': None,
    #         'gpu': None,
    #         'build_image': None
    #     },
    #     'nonagent': {
    #         'image': 'surreal-cpu-image',  # TODO
    #         'node_pool': 'surreal-default-cpu-nodepool',  # TODO
    #         'cpu': None,
    #         'memory': None,
    #         'gpu': None,
    #         'build_image': None
    #     },
    # }

    # def create_batch(self, *,
    #                  setting,
    #                  experiment_name,
    #                  algorithm_args,
    #                  input_args,
    #                  force,
    #                  dry_run):
    #     setting = _merge_setting_dictionaries(setting,
    #                                           self.DEFAULT_SETTING_BATCH)
    #     setting = _merge_setting_dictionaries(input_args, setting)
    #     setting = BeneDict(setting)

    def get_remote_experiment_folder(self, experiment_name):
        """
            Actual experiment folder will be
            <mount_path>/<root_subfolder>/<experiment_name>/
        """
        # DON'T use U.f_join because we don't want to expand the path locally
        root_subfolder = self.config.fs.experiment_root_subfolder
        # TODO: remove assert
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
