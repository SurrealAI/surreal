import os
import re
import shlex
import subprocess
from copy import copy
from pathlib import Path
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
        self._setup_docker_clean()

        # Secondary nfs related support
        self._setup_get_videos()
        self._setup_get_config()
        self._setup_get_tensorboard()

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
        return U.f_expand(self.config.kube_metadata_folder)

    @property
    def username(self):
        assert 'username' in self.config, 'must specify username in ~/.surreal.yml'
        return self.config.username

    def _setup_get_videos(self):
        parser = self.add_subparser('get-videos', aliases=['gv'])
        parser.add_argument('experiment_names', nargs='*', type=str, metavar='experiment_name',
                            help='experiments to retrieve videos for, '
                            'none to retrieve your own running experiments')
        parser.add_argument('--last', type=int, default=5, metavar='last_n_videos',
                            help='Number of most recent videos, -1 to get all')
        parser.add_argument('--save_folder', type=str, default='.',
                            help='save_videos in [save_folder]/experiment_name')

    def _setup_get_config(self):
        parser = self.add_subparser('get-config', aliases=['gc'])
        parser.add_argument('experiment_name', type=str,
                            help='experiments to retrieve videos for, '
                                 'none to retrieve your own running experiments')
        parser.add_argument('-o', '--output-file', type=str,
                            help='save remote config to a specified local file path')

    def _setup_get_tensorboard(self):
        parser = self.add_subparser('get-tensorboard', aliases=['gt'])
        parser.add_argument('experiment_name', type=str,
                            help='experiments to retrieve tensorboard for, '
                                 'none to retrieve your own running experiments')
        parser.add_argument('-s', '--subfolder', type=str, default='',
                            help='retrieve only a subfolder under the "tensorboard" folder. '
                                 'currently valid folders are agent, eval, learner, replay')
        parser.add_argument('-o', '--output-folder', type=str,
                            help='save remote TB folder to a specified local folder path')

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
            return 'surreal-ddpg'
        elif name == 'ppo':
            return 'surreal-ppo'
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
            'image': 'surrealai/surreal-nvidia:v0.1',
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
            'image': 'surrealai/surreal-nvidia:v0.1',
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
            tag=exp.name,
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
        experiment_folder = self.get_remote_experiment_folder(exp.name)
        algorithm_args += ["--experiment-folder",
                           str(experiment_folder)]
        algorithm_args += ["--env", str(settings.env)]
        algorithm_args += ["--agent-batch", str(settings.agent_batch)]
        algorithm_args += ["--eval-batch", str(settings.eval_batch)]
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
            nfs_server_path = self.config.nfs.fs_location
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
        directory = self.config.kube_results_folder
        return os.path.join(directory, experiment_name)

    def action_tensorboard(self, args):
        self.action_visit(args)

    def action_docker_clean(self, args):
        """
            Cleans all docker images used to create experiments
        """
        potential_images = []
        for name, settings in self.config.creation_settings.items():
            if settings.mode == 'basic':
                if 'agent' in settings and 'image' in settings.agent:
                    potential_images.append(settings.agent.image)
                if 'nonagent' in settings and 'image' in settings.nonagent:
                    potential_images.append(settings.nonagent.image)
            else:
                raise ValueError('Unsupported creation setting mode {}'
                                 .format(settings.mode) + ' for creation'
                                 + ' setting ' + name)

        potential_images = list(set(potential_images))
        potential_images = [x for x in potential_images if ':' not in x]
        images_to_clean = ['{}:*'.format(x) for x in potential_images]
        clean_images(images_to_clean)

    def action_get_videos(self, args):
        self._check_nfs_retrieve_settings()
        remote_folder = Path(self.config.nfs.results_folder)
        todos = []
        if len(args.experiment_names) == 0:
            experiments = self.cluster.list_experiments()
            for experiment in experiments:
                if re.match(self.username, experiment):
                    todos.append(experiment)
        else:
            todos = args.experiment_names

        print('Fetching videos for:')
        print('\n'.join(['\t' + x for x in todos]))

        save_folder = os.path.expanduser(args.save_folder)
        save_last = args.last

        for experiment_name in todos:
            self._get_video_for_experiment(
                experiment_name, save_folder, remote_folder, save_last
            )

    def _get_video_for_experiment(self,
                                  experiment_name,
                                  save_folder,
                                  remote_folder,
                                  save_last=-1):
        # Find remote path
        remote_folder = remote_folder / experiment_name / 'videos'
        # parse existing files
        results = self._gcloud_nfs_exec('ls -1 {}'.format(remote_folder))
        video_files = results.strip().split('\n')
        video_episodes = [x[len('video_eps_'):] for x in video_files]
        video_episodes = [int(x[:len(x) - len('.mp4')]) for x in video_episodes]
        video_episodes = sorted(video_episodes, reverse=True)
        if save_last > 0:
            save_last = min(save_last, len(video_episodes))
            video_episodes = video_episodes[:save_last]
        filenames = ['video_eps_{}.mp4'.format(x) for x in video_episodes]
        # local path
        local_folder = Path(os.path.expanduser(save_folder)) / experiment_name
        local_folder.mkdir(exist_ok=True, parents=True)
        # download
        for filename in filenames:
            print('$> get {}'.format(str(remote_folder / filename)))
            self._gcloud_download(remote_folder / filename, local_folder / filename)
        return filenames

    def action_get_config(self, args):
        """
        Download remote config.yml in the experiment folder
        """
        self._check_nfs_retrieve_settings()
        remote_folder = Path(self.config.nfs.results_folder)
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = 'config.yml'
        experiment_name = args.experiment_name
        remote_config_file = (remote_folder / experiment_name / 'config.yml')
        print('Downloading', remote_config_file)
        self._gcloud_download(remote_config_file, output_file)

    def action_get_tensorboard(self, args):
        """
        Download remote config.yml in the experiment folder
        """
        self._check_nfs_retrieve_settings()
        remote_folder = Path(self.config.nfs.results_folder)
        experiment_name = args.experiment_name
        if args.output_folder:
            output_folder = args.output_folder
        else:
            output_folder = experiment_name
        remote_tb_folder = (remote_folder / experiment_name
                            / 'tensorboard' / args.subfolder)
        print('Downloading', remote_tb_folder)
        self._gcloud_download(remote_tb_folder, output_folder)

    def _gcloud_download(self, remote_path, local_path):
        cmd = "gcloud compute scp --recurse {}:'{}' '{}'".format(
            self.config.nfs.servername, remote_path, local_path
        )
        os.system(cmd)

    def _gcloud_nfs_exec(self, command):
        return subprocess.check_output(
            "gcloud compute ssh {} -- '{}'".format(self.config.nfs.servername, command),
            shell=True
        ).decode('utf-8').replace('\r\n', '\n')


    def _check_nfs_retrieve_settings(self):
        if 'nfs' not in self.config:
            raise ValueError('nfs field not found in .surreal.yml, aborting')
        if 'servername' not in self.config.nfs:
            raise ValueError('nfs:servername field not found in .surreal.yml, aborting')
        if 'results_folder' not in self.config.nfs:
            raise ValueError('nfs:results_folder not found .surreal.yml, aborting')


def main():
    KurrealParser().main()


if __name__ == '__main__':
    main()
