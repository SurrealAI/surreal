import os
import argparse
import itertools
from pkg_resources import parse_version
from symphony.commandline import SymphonyParser
from symphony.engine import SymphonyConfig, Cluster
from symphony.kube import KubeCluster
from symphony.addons import DockerBuilder
from benedict import BeneDict
import surreal
from surreal.kube.generate_command import CommandGenerator
import surreal.utils as U


SURREAL_YML_VERSION = '0.0.3'  # force version check


def _process_labels(label_string):
    """
    mylabel1=myvalue1,mylabel2=myvalue2
    """
    assert '=' in label_string
    label_pairs = label_string.split(',')
    return [label_pair.split('=') for label_pair in label_pairs]


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

    def _check_version(self):
        """
        Check ~/.surreal.yml `version` key
        """
        assert 'version' in self.config, 'surreal yml version not specified.'
        if parse_version(SURREAL_YML_VERSION) != parse_version(self.config.version):
            raise ValueError('version incompatible, please check the latest '
                             'sample.surreal.yml and make sure ~/.surreal.yml is '
                             + SURREAL_YML_VERSION)

    def load_config(self, surreal_yml='~/.surreal.yml'):
        surreal_yml = U.f_expand(surreal_yml)
        if not U.f_exists(surreal_yml):
            raise ValueError('Cannot find surreal config file at {}'.format(surreal_yml))
        self.config = BeneDict.load_yaml_file(surreal_yml)
        self._check_version()
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

    def _setup_create(self):
        parser = self.add_subparser('create', aliases=['c'])
        self._add_experiment_name(parser)
        parser.add_argument(
            'config_py',
            type=str,
            help='location of python script **in the Kube pod** that contains the '
                 'runnable config. If the path does not start with /, defaults to '
                 'home dir, i.e. /root/ on the pod'
        )
        parser.add_argument(
            'num_agents',
            type=int,
            help='number of agents to run in parallel.'
        )
        self._add_dry_run(parser)
        self._add_create_args(parser)


    def _setup_create_dev(self):
        parser = self.add_subparser('create-dev', aliases=['cd'])
        self._add_experiment_name(parser)
        parser.add_argument('num_agents', type=int)
        parser.add_argument('-e', '--env', default='cheetah')
        parser.add_argument(
            '-g', '--gpu', '--num-gpus',
            dest='num_gpus',
            type=int,
            nargs='?',
            default=0
        )
        parser.add_argument(
            '--gpu-type',
            dest='gpu_type',
            type=str,
            default='k80'
        )
        parser.add_argument('-f', '--force', action='store_true')
        parser.add_argument(
            '-c', '--config_file',
            default='ddpg_configs.py',
            help='which config file in surreal/main to use'
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
            print_err('experiment name string has been fixed: {} -> {}'
                      .format(experiment_name, new_name))
        return new_name

    def _add_create_args(self, parser):
        """
        Used in create(), restore(), resume()
        """
        parser.add_argument(
            '-at', '--agent-pod-type',
            default='agent',
            help='key in ~/.surreal.yml `pod_types` section that describes spec for agent pod. '
                 'Default: "agent"'
        )
        parser.add_argument(
            '-nt', '--nonagent-pod-type',
            default='nonagent-cpu',
            help='key in ~/.surreal.yml `pod_types` section that describes spec for '
                 'nonagent pod with multiple containers: learner, ps, tensorboard, etc. '
                 'Default: "nonagent-cpu"'
        )
        parser.add_argument(
            '-f', '--force',
            action='store_true',
            help='force overwrite an existing kurreal.yml file '
                 'if its experiment folder already exists.'
        )

    def action_create(self, args):
        """
        Spin up a multi-node distributed Surreal experiment.
        Put any command line args that pass to the config script after "--"
        """
        self._create_helper(
            config_py=args.config_py,
            experiment_name=args.experiment_name,
            num_agents=args.num_agents,
            config_command=args.remainder,  # cmd line remainder after "--"
            agent_pod_type=args.agent_pod_type,
            nonagent_pod_type=args.nonagent_pod_type,
            restore=False,
            restore_folder=None,
            force=args.force,
            dry_run=args.dry_run
        )

    def action_create_dev(self, args):
        """
        << internal dev only >>
        """
        assert not args.has_remainder, \
            'create_dev cannot have "--". Use --env and --gpu'
        ENV_ALIAS = {
            # dm_control:cartpole-swingup
            'humanoid': 'dm_control:humanoid-walk',
            'ca': 'dm_control:cartpole-balance',
            'cartpole': 'dm_control:cartpole-balance',
            'ch': 'dm_control:cheetah-run',
            'cheetah': 'dm_control:cheetah-run',
            'hopper': 'dm_control:hopper-hop',
            'mj': 'mujocomanip:SawyerLiftEnv'
        }
        if args.env:
            env = args.env
        else:
            env = 'cheetah'
        config_command = ['--env', ENV_ALIAS[env]]

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
            }
        else:
            raise ValueError('Unknown GPU type: {}'.format(args.gpu_type))
        if num_gpus not in POD_TYPES:
            raise ValueError('invalid number of GPUs, choose from {}'
                             .format(list(POD_TYPES.keys())))
        nonagent_pod_type = POD_TYPES[num_gpus]
        config_command += ["--num-gpus", str(num_gpus)]
        # '/mylibs/surreal/surreal/surreal/main/ddpg_configs.py'
        config_py = 'surreal/surreal/main/' + args.config_file

        self._create_helper(
            config_py=config_py,
            experiment_name=args.experiment_name,
            num_agents=args.num_agents,
            config_command=config_command,
            agent_pod_type='agent',
            nonagent_pod_type=nonagent_pod_type,
            restore=False,
            restore_folder=None,
            force=args.force,
            dry_run=args.dry_run,
        )

    def _create_helper(self, *,
                       config_py,
                       experiment_name,
                       num_agents,
                       config_command,
                       agent_pod_type,
                       nonagent_pod_type,
                       restore,
                       restore_folder,
                       force,
                       dry_run=False):
        if config_py.startswith('/'):
            config_py = config_py
        else:
            config_py = U.f_join('/mylibs', config_py)

        remote_experiment_folder = self.get_remote_experiment_folder(experiment_name)

        cmd_gen = CommandGenerator(
            num_agents=num_agents,
            experiment_folder=remote_experiment_folder, # TODO: fixme
            config_py=config_py,
            config_command=config_command,
            service_url=None,
            restore=restore,
            restore_folder=restore_folder,
        )
        cmd_dict = cmd_gen.generate()
        print('  agent_pod_type:', agent_pod_type)
        print('  nonagent_pod_type:', nonagent_pod_type)

        self.create_surreal(
            experiment_name,
            agent_pod_type=agent_pod_type,
            nonagent_pod_type=nonagent_pod_type,
            cmd_dict=cmd_dict,
            force=force,
            dry_run=dry_run,
        )

    def create_surreal(self,
                       experiment_name,
                       agent_pod_type,
                       nonagent_pod_type,
                       cmd_dict,
                       force=False,
                       dry_run=False):
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
        agent_pod_spec = C.pod_types[agent_pod_type]
        nonagent_pod_spec = C.pod_types[nonagent_pod_type]
        agent_resource_request = agent_pod_spec.get('resource_request', {})
        nonagent_resource_request = nonagent_pod_spec.get('resource_request', {})
        agent_resource_limit = agent_pod_spec.get('resource_limit', {})
        nonagent_resource_limit = nonagent_pod_spec.get('resource_limit', {})

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

        nonagent = exp.new_process_group('nonagent')
        learner = nonagent.new_process('learner', container_image=nonagent_pod_spec.image, args=[cmd_dict['learner']])
        replay = nonagent.new_process('replay', container_image=nonagent_pod_spec.image, args=[cmd_dict['replay']])
        ps = nonagent.new_process('ps', container_image=nonagent_pod_spec.image, args=[cmd_dict['ps']])
        tensorboard = nonagent.new_process('tensorboard', container_image=nonagent_pod_spec.image, args=[cmd_dict['tensorboard']])
        tensorplex = nonagent.new_process('tensorplex', container_image=nonagent_pod_spec.image, args=[cmd_dict['tensorplex']])
        loggerplex = nonagent.new_process('loggerplex', container_image=nonagent_pod_spec.image, args=[cmd_dict['loggerplex']])

        agents = []
        for i, arg in enumerate(cmd_dict['agent']):
            agent_p = exp.new_process('agent-{}'.format(i), container_image=agent_pod_spec.image, args=[arg])
            agents.append(agent_p)
        # TODO: make command generator return list
        evals = []
        for i, arg in enumerate(cmd_dict['eval']):
            eval_p = exp.new_process('eval-{}'.format(i), container_image=agent_pod_spec.image, args=[arg])
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

        agent_selector = agent_pod_spec.get('selector', {})
        for proc in itertools.chain(agents, evals):
            proc.resource_request(cpu=1.5)
            proc.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
            proc.restart_policy('Never')
            # required services
            for k, v in agent_selector.items():
                proc.node_selector(key=k, value=v)
            proc.resource_request(**agent_resource_request)
            proc.resource_limit(**agent_resource_limit)

        learner.set_env('DISABLE_MUJOCO_RENDERING', "1")
        learner.resource_request(**nonagent_resource_request)
        if 'nvidia.com/gpu' in nonagent_resource_limit:
            nonagent_resource_limit['gpu'] = nonagent_resource_limit['nvidia.com/gpu']
            del nonagent_resource_limit['nvidia.com/gpu']
        learner.resource_limit(**nonagent_resource_limit)

        non_agent_selector = nonagent_pod_spec.get('selector', {})
        for k, v in non_agent_selector.items():
            nonagent.node_selector(key=k, value=v)
        nonagent.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
        nonagent.image_pull_policy('Always')

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

def main():
    KurrealParser().main()


if __name__ == '__main__':
    main()
