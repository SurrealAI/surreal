import argparse
import surreal
import webbrowser
import re
import itertools
from collections import OrderedDict
from surreal.kube.kubectl import *
from surreal.kube.generate_command import *
import surreal.utils as U
from symphony.commandline import SymphonyParser
from symphony.engine import SymphonyConfig, Cluster
from symphony.kube import *


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
        self.kube = Kubectl() # TODO: remove
        self.load_config()
        self._setup_create()
        self._setup_create_dev()
        self._setup_restore()
        self._setup_resume()
        self._setup_download_experiment()

    def load_config(self, surreal_yml='~/.surreal.yml'):
        surreal_yml = U.f_expand(surreal_yml)
        assert U.f_exists(surreal_yml)
        self.config = YamlList.from_file(surreal_yml)[0]
        SymphonyConfig().set_username(self.username)
        SymphonyConfig().set_experiment_folder(self.folder)

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
        parser.add_argument('-nos', '--no-snapshot', action='store_true')
        parser.add_argument('-f', '--force', action='store_true')
        parser.add_argument(
            '-c', '--config_file',
            default='ddpg_configs.py',
            help='which config file in surreal/main to use'
        )
        self._add_dry_run(parser)

    def _setup_restore(self): # TODO: fix
        parser = self.add_subparser('restore', aliases=[])
        parser.add_argument(
            '-new', '--new',
            dest='experiment_name',
            type=self._process_experiment_name,
            required=True,
            help='Start a new experiment from the old checkpoint. '
                 'experiment name will be used as namespace for DNS. '
                 'Should only contain lower case letters, digits, and hypen. '
                 'Underscores and dots are not allowed and will be converted to hyphen.'
        )
        parser.add_argument(
            '-old', '--old',
            dest='restore_experiment',
            required=True,
            help="old experiment name to restore from. "
                 "you can also give full path to the folder on the shared FS: "
                 "'/fs/experiments/myfriend/.../'"
        )
        self._add_restore_args(parser)
        self._add_create_args(parser)

    def _setup_resume(self): # TODO: fix
        parser = self.add_subparser('resume', aliases=['continue'])
        self._add_experiment_name(parser)
        self._add_restore_args(parser)
        self._add_create_args(parser)  # --force is automatically turned on

    def _setup_download_experiment(self):
        parser = self.add_subparser('download-experiment',
                                    aliases=['de', 'download'])
        parser.add_argument(
            'remote_experiment_subfolder',
            help='remote subfolder under <fs_mount_path>/<root_subfolder>.'
        )
        parser.add_argument(
            '-a', '--absolute-path',
            action='store_true',
            help='use absolute remote path instead of '
                 '<fs_mount_path>/<root_subfolder>/<remote_folder>'
        )
        parser.add_argument(
            '-o', '--output-path',
            default='.',
            help='local folder path to download to'
        )
        parser.add_argument(
            '-m', '--match-fuzzy',
            action='store_true',
            help='enable fuzzy matching with the currently running namespaces'
        )


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
            '-nos', '--no-snapshot',
            action='store_true',
            help='Unless this flag is on, Surreal will take snapshot of the specified '
                 'git repos in ~/.surreal.yml and upload to your remote Github branch.'
        )
        parser.add_argument(
            '-f', '--force',
            action='store_true',
            help='force overwrite an existing kurreal.yml file '
                 'if its experiment folder already exists.'
        )

    def _add_restore_args(self, parser):
        """
        Used in restore(), resume()
        If remainders (cmd line args after "--") are specified,
        override the saved launch cmd args
        """
        parser.add_argument(
            '--config-py',
            default=None,
            help='If unspecified, defaults to the saved launch command. '
                 'location of python script **in the Kube pod** that contains the '
                 'runnable config. If the path does not start with /, defaults to '
                 'home dir, i.e. /root/ on the pod'
        )
        parser.add_argument(
            '-n', '--num-agents',
            type=int,
            default=None,
            help='If unspecified, defaults to the saved launch command. '
                 'number of agents to run in parallel.'
        )
        # the following should not be managed by Kurreal, should be set in config.py
        # session_config.checkpoint.learner.restore_target
        # parser.add_argument(
        #     '--best',
        #     action='store_true',
        #     help='restore from the best checkpoint, otherwise from history'
        # )
        # parser.add_argument(
        #     '-t', '--target',
        #     default='0',
        #     help='see "Checkpoint" class. Restore target can be one of '
        #          'the following semantics:\n'
        #          '- int: 0 for the last (or best), 1 for the second last (or best), etc.'
        #          '- global steps of the ckpt file, the suffix string right before ".ckpt"'
        # )

    @staticmethod
    def _find_kurreal_template(template_name):
        """
        https://stackoverflow.com/questions/20298729/pip-installing-data-files-to-the-wrong-place
        make sure the path is consistent with MANIFEST.in
        """
        return U.f_join(surreal.__path__[0], 'kube', template_name)

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
            no_snapshot=args.no_snapshot,
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
            'ca': 'dm_control:cartpole-balance',
            'cartpole': 'dm_control:cartpole-balance',
            'ch': 'dm_control:cheetah-run',
            'cheetah': 'dm_control:cheetah-run',
            'hopper': 'dm_control:hopper-hop',
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
            no_snapshot=args.no_snapshot,
            force=args.force,
            dry_run=args.dry_run,
        )

    def action_restore(self, args): # TODO: Jim: You may need to fix it.
        """
        Restore experiment with the saved CommandGenerator and checkpoint
        Put any command line args that pass to the config script after "--"
        """
        kube = self.kube
        if '/' in args.restore_experiment:  # full path on remote shared FS
            saved = None
            restore_folder = args.restore_experiment
        else:
            # yaml save of "kurreal create" command line args
            saved = CommandGenerator.get_yaml(
                kube.get_path(kube.strip_username(args.restore_experiment),
                              'launch_commands.yml')
            )
            restore_folder = kube.get_remote_experiment_folder(
                kube.strip_username(args.restore_experiment)
            )

        # "kurreal restore" args take precedence unless unspecified
        if args.config_py:
            config_py = args.config_py
        else:
            assert saved, 'No saved launch, must specify --config-py'
            config_py = saved.config_py
        if args.num_agents:
            num_agents = args.num_agents
        else:
            assert saved, 'No saved launch, must specify --num-agents'
            num_agents = saved.num_agents
        if args.has_remainder:
            config_command = args.remainder
        else:
            assert saved, 'No saved launch, must specify -- <config commands>'
            config_command = saved.config_command

        prefix_restored = kube.prefix_username(args.restore_experiment)
        prefix_current = kube.prefix_username(args.experiment_name)
        if prefix_restored == prefix_current:
            print('Resume at experiment folder "{}"'.format(restore_folder))
        else:
            print('Restore from remote experiment folder "{}"\n'
                  'and create new experiment "{}"'
                  .format(restore_folder, prefix_current))

        self._create_helper(
            config_py=config_py,
            experiment_name=args.experiment_name,
            num_agents=num_agents,
            config_command=config_command,
            agent_pod_type=args.agent_pod_type,
            nonagent_pod_type=args.nonagent_pod_type,
            restore=True,
            restore_folder=restore_folder,
            no_snapshot=args.no_snapshot,
            force=args.force,
        )

    def action_resume(self, args): # TODO: Jim you may need to fix this
        args.force = True  # always override the generated kurreal.yml
        args.restore_experiment = args.experiment_name
        self.action_restore(args)

    def action_download_experiment(self, args): # TODO: Jim you may need to fix this
        # TODO:
        """
        Same as `kurreal scp learner:<mount_path>/<root_subfolder>/experiment-folder .`
        """
        kube = self.kube
        remote_subfolder = args.remote_experiment_subfolder
        if not args.absolute_path:
            if args.match_fuzzy:
                assert '/' not in remote_subfolder, \
                    "fuzzy match does not allow '/' in experiment name"
                remote_subfolder = self._interactive_find_ns(remote_subfolder)
                remote_subfolder = kube.strip_username(remote_subfolder)
            remote_path = kube.get_remote_experiment_folder(remote_subfolder)
        else:
            assert not args.match_fuzzy, \
                'cannot fuzzy match when --absolute-path is turned on.'
            remote_path = remote_subfolder
        # the experiment folder will be unpacked if directly scp to "."
        output_path = U.f_join(args.output_path, U.f_last_part_in_path(remote_path))
        kube.scp_surreal(
            'learner:' + remote_path, output_path
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
                       no_snapshot,
                       force,
                       dry_run):
        if config_py.startswith('/'):
            config_py = config_py
        else:
            config_py = U.f_join('/root', config_py)

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
            snapshot=not no_snapshot,
            agent_pod_type=agent_pod_type,
            nonagent_pod_type=nonagent_pod_type,
            cmd_dict=cmd_dict,
            force=force,
        )

    def create_surreal(self,
                       experiment_name,
                       agent_pod_type,
                       nonagent_pod_type,
                       cmd_dict,
                       snapshot=True,
                       mujoco=True,
                       force=False,
                       dry_run=False):
        """
        First create a snapshot of the git repos, upload to github
        Then create Kube objects with the git info
        Args:
            experiment_name: will also be used as hostname for DNS
            rendered_path: rendered yaml file path
            agent_pod_type: key to spec defined in `pod_types` section of .surreal.yml
            nonagent_pod_type: key to spec defined in `pod_types` section of .surreal.yml
            cmd_dict: dict of commands to be run on each container
            snapshot: True to take a snapshot of git repo and upload
            mujoco: True to copy mujoco key into the generated yaml
            prefix_user_name: True to prefix experiment name (and host name)
                as <myusername>-<experiment_name>
            force: check if the Kube yaml has already been generated.
            dry_run: only print yaml, do not actually launch
        """
        C = self.config
        repo_paths = C.git.get('snapshot_repos', [])
        repo_paths = [U.f_expand(p) for p in repo_paths]
        if snapshot and not dry_run:
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

        cluster = Cluster.new('kube')

        exp = cluster.new_experiment(experiment_name)

        nonagent = exp.new_process_group('nonagent')
        learner = nonagent.new_process('learner', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['learner']])
        replay = nonagent.new_process('replay', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['replay']])
        ps = nonagent.new_process('ps', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['ps']])
        tensorboard = nonagent.new_process('tensorboard', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['tensorboard']])
        tensorplex = nonagent.new_process('tensorplex', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['tensorplex']])
        loggerplex = nonagent.new_process('loggerplex', container_image=nonagent_pod_spec.image, args=['--cmd', cmd_dict['loggerplex']])
        learner.binds('myserver')
        replay.connects('myserver')

        agents = []
        for i, arg in enumerate(cmd_dict['agent']):
            agent_p = exp.new_process('agent-{}'.format(i), container_image=agent_pod_spec.image, args=['--cmd', arg])
            agents.append(agent_p)

        evals = []
        for i, arg in enumerate(cmd_dict['eval']):
            eval_p = exp.new_process('eval-{}'.format(i), container_image=agent_pod_spec.image, args=['--cmd', arg])
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

        if mujoco:
            mjkey = file_content(C.mujoco_key_path)

        for proc in exp.list_all_processes():
            # Mount nfs
            proc.mount_nfs(server=nfs_server, path=nfs_server_path, mount_path=nfs_mount_path)

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
                proc.mount_git_repo(repository=repository, revision=revision, mount_path=mount_path)
                env_key = 'repo_{}'.format(git_repo.replace('-', '_'))
                env_val = '/mylibs/{0}/{0}'.format(git_repo)
                proc.set_env(env_key, env_val)

            # Add mujoco key: TODO: handle secret properly
            proc.set_env('mujoco_key_text', mjkey)
            proc.image_pull_policy('Always')

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
        # Note/TODO: We are passing resource limits as kwargs, so '/' cannot happen here
        # Should we change this?
            nonagent_resource_limit['gpu'] = nonagent_resource_limit['nvidia.com/gpu']
            del nonagent_resource_limit['nvidia.com/gpu']
        learner.resource_limit(**nonagent_resource_limit)

        non_agent_selector = nonagent_pod_spec.get('selector', {})
        for k, v in non_agent_selector.items():
            nonagent.node_selector(key=k, value=v)
        nonagent.add_toleration(key='surreal', operator='Exists', effect='NoExecute')
        nonagent.image_pull_policy('Always')

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
