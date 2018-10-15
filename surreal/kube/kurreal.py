import os
import shlex
from copy import copy
from symphony.commandline import SymphonyParser
from symphony.engine import SymphonyConfig, Cluster
from symphony.kube import GKEDispatcher
from symphony.addons import clean_images
from benedict import BeneDict
from surreal.launch import (
    SurrealDockerBuilder,
    CommandGenerator,
    setup_network,
)
import surreal.utils as U


def _merge_setting_dictionaries(customize, base):
    di = copy(base)
    for key in di:
        if isinstance(di[key], dict):
            if key in customize:
                di[key] = _merge_setting_dictionaries(customize[key], di[key])
        else:
            if key in customize and customize[key] is not None:
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
        self._setup_tensorboard()
        # self._setup_docker_clean()

        # Secondary nfs related support
        # self._setup_get_videos()
        # self._setup_get_config()
        # self._setup_get_tensorboard()

    def load_config(self):
        surreal_yml_path = U.get_config_file()
        if not U.f_exists(surreal_yml_path):
            raise ValueError('Cannot find surreal config file at {}'
                             .format(surreal_yml_path))
        self.config = BeneDict.load_yaml_file(surreal_yml_path)
        SymphonyConfig().set_username(self.username)
        SymphonyConfig().set_experiment_folder(self.folder)

        if 'docker_build_settings' in self.config:
            for setting in self.config['docker_build_settings']:
                self.docker_build_settings[setting['name']] = setting

    @property
    def folder(self):
        return U.f_expand(self.config.kurreal_metadata_folder)

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
        parser.add_argument(
            'setting_name',
            type=str,
            help='the setting in .surreal.yml that specifies how an'
                 'experiment should be run')
        self._add_experiment_name(parser)
        parser.add_argument(
            '--algorithm',
            type=str,
            help='ddpg / ppo or the'
                 'location of algorithm python script **in the docker '
                 'container**'
        )
        parser.add_argument(
            '--num_agents',
            type=int,
            default=None,
            help='number of agent pods to run in parallel.'
        )
        parser.add_argument(
            '--num_evals',
            type=int,
            default=None,
            help='number of eval pods to run in parallel.'
        )
        parser.add_argument(
            '--agent-batch',
            type=int,
            default=None,
            help='put how many agent on each agent pod'
        )
        parser.add_argument(
            '--eval-batch',
            type=int,
            default=None,
            help='put how many eval on each eval pod'
        )
        parser.add_argument(
            '--env',
            type=str,
            default=None,
            help='What environment to run'
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
            return '/mylibs/surreal/surreal/main/ddpg_configs.py'
        elif name == 'ppo':
            return '/mylibs/surreal/surreal/main/ppo_configs.py'
        else:
            return name

    DEFAULT_SETTING_BASIC = {
        'algorithm': 'ddpg',
        'num_agents': 2,
        'num_evals': 1,
        'agent_batch': 1,
        'eval_batch': 1,
        'restore_folder': None,
        'env': 'gym:HalfCheetah-v2',
        'agent': {
            'image': 'surreal-cpu-image',  # TODO
            'build_image': None,
            'scheduling': {
                'assign_to': 'node_pool',
                'node_pool_name': None,
                'cpu': None,
                'memory_m': None,
                'gpu_type': None,
                'gpu_count': None,
            }
        },
        'nonagent': {
            'image': 'surreal-cpu-image',  # TODO
            'build_image': None,
            'scheduling': {
                'assign_to': 'node_pool',
                'node_pool_name': None,
                'cpu': None,
                'memory_m': None,
                'gpu_type': None,
                'gpu_count': None,
            }
        },
    }

    def create_basic(self, *,
                     settings,
                     experiment_name,
                     algorithm_args,
                     input_args,
                     force,
                     dry_run):
        settings = _merge_setting_dictionaries(settings,
                                               self.DEFAULT_SETTING_BASIC)
        settings = _merge_setting_dictionaries(input_args, settings)
        settings = BeneDict(settings)

        cluster = self.create_cluster()
        if 'mount_secrets' in self.config:
            secrets = self.config.mount_secrets
        else:
            secrets = None
        exp = cluster.new_experiment(experiment_name, secrets=secrets)

        image_builder = SurrealDockerBuilder(
            build_settings=self.docker_build_settings,
            images_requested={
                'agent': {
                    'identifier': settings.agent.image,
                    'build_config': settings.agent.build_image
                },
                'nonagent': {
                    'identifier': settings.nonagent.image,
                    'build_config': settings.nonagent.build_image
                },
            },
            tag=experiment_name,
            push=True)
        agent_image = image_builder.images_provided['agent']
        nonagent_image = image_builder.images_provided['nonagent']
        # defer to build last, so we don't build unless everything passes

        algorithm_args += [
            "--num-agents",
            str(settings.num_agents * settings.agent_batch),
            ]
        # TODO: restore_functionalities
        if settings.restore_folder is not None:
            algorithm_args += ["--restore_folder",
                               shlex.quote(settings.restore_folder)]
        experiment_folder = self.get_remote_experiment_folder(experiment_name)
        algorithm_args += ["--experiment-folder",
                           experiment_folder]
        algorithm_args += ["--env", settings.env]
        executable = self._find_executable(settings.algorithm)
        cmd_gen = CommandGenerator(
            num_agents=settings.num_agents,
            num_evals=settings.num_evals,
            executable=executable,
            config_commands=algorithm_args)

        nonagent = exp.new_process_group('nonagent')
        learner = nonagent.new_process(
            'learner',
            container_image=nonagent_image,
            args=[cmd_gen.get_command('learner')])
        # Because learner and everything are bundled together

        # json_path = 'cluster_definition.tf.json'  # always use slash
        # filepath = pkg_resources.resource_filename(__name__, json_path)
        json_path = self.config.cluster_definition
        dispatcher = GKEDispatcher(json_path)
        # We only need to claim resources for learner
        dispatcher.assign_to(learner,
                             process_group=nonagent,
                             **settings.nonagent.scheduling)
        # For dm_control
        learner.set_env('DISABLE_MUJOCO_RENDERING', "1")

        replay = nonagent.new_process(
            'replay',
            container_image=nonagent_image,
            args=[cmd_gen.get_command('replay')])

        ps = nonagent.new_process(
            'ps',
            container_image=nonagent_image,
            args=[cmd_gen.get_command('ps')])

        tensorboard = nonagent.new_process(
            'tensorboard',
            container_image=nonagent_image,
            args=[cmd_gen.get_command('tensorboard')])

        tensorplex = nonagent.new_process(
            'tensorplex',
            container_image=nonagent_image,
            args=[cmd_gen.get_command('tensorplex')])

        loggerplex = nonagent.new_process(
            'loggerplex',
            container_image=nonagent_image,
            args=[cmd_gen.get_command('loggerplex')])
        nonagent.image_pull_policy('Always')

        agents = []
        for i in range(settings.num_agents):
            if settings.agent_batch == 1:
                agent_name = 'agent-{}'.format(i)
            else:
                agent_name = 'agents-{}'.format(i)
            agent = exp.new_process(
                agent_name,
                container_image=agent_image,
                args=[cmd_gen.get_command(agent_name)])

            agent.image_pull_policy('Always')
            # We only need to claim resources for learner
            dispatcher.assign_to(agent,
                                 **settings.agent.scheduling)
            agents.append(agent)

        evals = []
        for i in range(settings.num_evals):
            if settings.eval_batch == 1:
                eval_name = 'eval-{}'.format(i)
            else:
                eval_name = 'evals-{}'.format(i)
            eval_p = exp.new_process(
                eval_name,
                container_image=agent_image,
                args=[cmd_gen.get_command(eval_name)])
            dispatcher.assign_to(eval_p, **settings.agent.scheduling)
            eval_p.image_pull_policy('Always')

            evals.append(eval_p)

        setup_network(agents=agents,
                      evals=evals,
                      learner=learner,
                      replay=replay,
                      ps=ps,
                      tensorboard=tensorboard,
                      tensorplex=tensorplex,
                      loggerplex=loggerplex)

        if 'nfs' in self.config:
            print('NFS mounted')
            nfs_server = self.config.nfs.servername
            nfs_server_path = self.config.nfs.path_on_server
            nfs_mount_path = self.config.nfs.mount_path
            for proc in exp.list_all_processes():
                proc.mount_nfs(server=nfs_server,
                               path=nfs_server_path,
                               mount_path=nfs_mount_path)
        else:
            print('NFS not mounted')

        image_builder.build()
        cluster.launch(exp, force=force, dry_run=dry_run)

    def get_remote_experiment_folder(self, experiment_name):
        """
            Actual experiment folder will be
            <mount_path>/<root_subfolder>/<experiment_name>/
        """
        # DON'T use U.f_join because we don't want to expand the path locally
        directory = self.config.kurreal_results_folder
        return os.path.join(directory, experiment_name)

    def action_tensorboard(self, args):
        self.action_visit(args)

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
