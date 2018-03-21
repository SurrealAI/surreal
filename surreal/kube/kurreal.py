import argparse
import surreal
import webbrowser
import re
from collections import OrderedDict
from surreal.kube.kubectl import *
from surreal.kube.generate_command import *


def _process_labels(label_string):
    """
    mylabel1=myvalue1,mylabel2=myvalue2
    """
    assert '=' in label_string
    label_pairs = label_string.split(',')
    return [label_pair.split('=') for label_pair in label_pairs]


class KurrealParser:
    def __init__(self):
        self._master_parser = argparse.ArgumentParser()
        self._add_dry_run(self._master_parser)

        self._subparsers = self._master_parser.add_subparsers(
            help='kurreal action commands',
            dest='kurreal_action'  # will store to parser.subcommand_name
        )
        self._subparsers.required = True

    def setup_master(self):
        """
        Main function that returns the configured parser
        """
        self._setup_create()
        self._setup_create_dev()
        self._setup_restore()
        self._setup_resume()
        self._setup_delete()
        self._setup_delete_batch()
        self._setup_log()
        self._setup_namespace()
        self._setup_tensorboard()
        self._setup_create_tensorboard()
        self._setup_list()
        self._setup_pod()
        self._setup_describe()
        self._setup_exec()
        self._setup_scp()
        self._setup_download_experiment()
        self._setup_ssh()
        self._setup_ssh_node()
        self._setup_ssh_nfs()
        self._setup_configure_ssh()
        self._setup_capture_tensorboard()
        return self._master_parser

    def _add_subparser(self, name, aliases, **kwargs):
        method_name = 'kurreal_' + name.replace('-', '_')
        raw_method = getattr(Kurreal, method_name)  # Kurreal.kurreal_create()

        def _kurreal_func(args):
            """
            Get function that processes parsed args and runs kurreal actions
            """
            kurreal_object = Kurreal(args)
            raw_method(kurreal_object, args)

        parser = self._subparsers.add_parser(
            name,
            help=raw_method.__doc__,
            aliases=aliases,
            **kwargs
        )
        self._add_dry_run(parser)
        parser.set_defaults(func=_kurreal_func)
        return parser

    def _setup_create(self):
        parser = self._add_subparser('create', aliases=['c'])
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
        self._add_create_args(parser)

    def _setup_create_dev(self):
        parser = self._add_subparser('create-dev', aliases=['cd'])
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

    def _setup_delete(self):
        parser = self._add_subparser('delete', aliases=['d'])
        self._add_experiment_name(parser, nargs='?')
        parser.add_argument(
            '-f', '--force',
            action='store_true',
            help='force delete, do not show confirmation message.'
        )

    def _setup_delete_batch(self):
        parser = self._add_subparser('delete-batch', aliases=['db'])
        parser.add_argument('experiment_name', type=str)
        parser.add_argument(
            '-f', '--force',
            action='store_true',
            help='force delete, do not show confirmation message.'
        )

    def _setup_restore(self):
        parser = self._add_subparser('restore', aliases=[])
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

    def _setup_resume(self):
        parser = self._add_subparser('resume', aliases=['continue'])
        self._add_experiment_name(parser)
        self._add_restore_args(parser)
        self._add_create_args(parser)  # --force is automatically turned on

    def _setup_log(self):
        parser = self._add_subparser('log', aliases=['logs', 'l'])
        self._add_component_arg(parser)
        self._add_namespace(parser, positional=True)
        parser.add_argument(
            '-f', '--follow',
            action='store_true',
            help='if the logs should be streamed.'
        )
        parser.add_argument(
            '-s', '--since',
            default='0',
            help='only show logs newer than a relative duration like 5s, 2m, 3h.'
        )
        parser.add_argument(
            '-t', '--tail',
            type=int,
            default=100,
            help='Only show the most recent lines of log. -1 to show all log lines.'
        )

    def _setup_namespace(self):
        parser = self._add_subparser(
            'namespace',
            aliases=['ns', 'exp', 'experiment']
        )
        # no arg to get the current namespace
        self._add_experiment_name(parser, nargs='?')

    def _setup_list(self):
        parser = self._add_subparser('list', aliases=['ls'])
        parser.add_argument(
            'resource',
            choices=['ns', 'namespace', 'namespaces',
                     'e', 'exp', 'experiment', 'experiments',
                     'p', 'pod', 'pods',
                     'no', 'node', 'nodes',
                     's', 'svc', 'service', 'services'],
            default='ns',
            nargs='?',
            help='list experiment, pod, and node'
        )
        self._add_namespace(parser, positional=True)
        parser.add_argument(
            '-a', '--all',
            action='store_true',
            help='show all resources from all namespace.'
        )

    def _setup_pod(self):
        "save as 'kurreal list pod'"
        parser = self._add_subparser('pod', aliases=['p', 'pods'])
        self._add_namespace(parser, positional=True)
        parser.add_argument(
            '-a', '--all',
            action='store_true',
            help='show all pods from all namespace.'
        )

    def _setup_tensorboard(self):
        parser = self._add_subparser('tensorboard', aliases=['tb'])
        self._add_namespace(parser, positional=True)
        parser.add_argument(
            '-u', '--url-only',
            action='store_true',
            help='only show the URL without opening the browser.'
        )

    def _setup_create_tensorboard(self):
        parser = self._add_subparser('create-tensorboard', aliases=['ctb'])
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
            '-p', '--pod-type',
            default='tensorboard',
            help='pod type for the tensorboard pod (specified in ~/.surreal.yml). '
                 'please use the smallest compute instance possible.'
        )

    def _setup_describe(self):
        parser = self._add_subparser('describe', aliases=['des'])
        parser.add_argument(
            'pod_name',
            help="should be either 'agent-<N>' or 'nonagent'"
        )
        self._add_namespace(parser, positional=True)

    def _setup_exec(self):
        """
        Actual exec commands must be added after "--"
        will throw error if no "--" in command args
        """
        parser = self._add_subparser('exec', aliases=['x'])
        self._add_component_arg(parser)
        self._add_namespace(parser, positional=True)

    def _setup_scp(self):
        parser = self._add_subparser('scp', aliases=['cp'])
        parser.add_argument(
            'src_file',
            help='source file or folder. "<component>:/file/path" denotes remote.'
        )
        parser.add_argument(
            'dest_file',
            help='destination file or folder. "<component>:/file/path" denotes remote.'
        )
        self._add_namespace(parser, positional=True)

    def _setup_download_experiment(self):
        parser = self._add_subparser('download-experiment',
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

    def _setup_ssh(self):
        parser = self._add_subparser('ssh', aliases=[])
        self._add_component_arg(parser)
        self._add_namespace(parser, positional=True)

    def _setup_ssh_node(self):
        parser = self._add_subparser('ssh-node', aliases=['sshnode'])
        parser.add_argument('node_name', help='gcloud only')

    def _setup_ssh_nfs(self):
        parser = self._add_subparser('ssh-nfs', aliases=['sshnfs'])

    def _setup_configure_ssh(self):
        parser = self._add_subparser('configure-ssh', aliases=['configssh'])

    def _setup_label(self):
        """
        Shouldn't manually label if you are using kube autoscaling
        """
        parser = self._add_subparser('label', aliases=[])
        parser.add_argument(
            'old_labels',
            help='select nodes according to their old labels'
        )
        parser.add_argument(
            'new_labels',
            type=_process_labels,
            help='mark the selected nodes with new labels in format '
                 '"mylabel1=myvalue1,mylabel2=myvalue2"'
        )

    def _setup_capture_tensorboard(self):
        parser = self._add_subparser('capture-tensorboard', aliases=['cptb'])
        parser.add_argument(
            'experiment_prefix',
            help='capture tensorboard screenshot for all prefix matched experiments,\
                  one can also use regex'  
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

    def _add_experiment_name(self, parser, nargs=None):
        parser.add_argument(
            'experiment_name',
            type=self._process_experiment_name,
            nargs=nargs,
            help='experiment name will be used as namespace for DNS. '
                 'Should only contain lower case letters, digits, and hypen. '
                 'Underscores and dots are not allowed and will be converted to hyphen.'
        )

    def _add_component_arg(self, parser):
        nonagent_str = ', '.join(map('"{}"'.format, Kubectl.NONAGENT_COMPONENTS))
        parser.add_argument(
            'component_name',
            help="should be either agent-<N> or one of [{}]".format(nonagent_str)
        )

    def _add_namespace(self, parser, positional=True):
        help='run the command in the designated namespace'
        if positional:
            parser.add_argument(
                'namespace',
                type=self._process_experiment_name,
                nargs='?',
                default='',
                help=help,
            )
        else:
            parser.add_argument(
                '-ns', '--ns', '--namespace',
                dest='namespace',
                type=self._process_experiment_name,
                default='',
                help=help,
            )

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


class Kurreal:
    def __init__(self, args):
        self.kube = Kubectl(dry_run=args.dry_run)

    @staticmethod
    def _find_kurreal_template(template_name):
        """
        https://stackoverflow.com/questions/20298729/pip-installing-data-files-to-the-wrong-place
        make sure the path is consistent with MANIFEST.in
        """
        return U.f_join(surreal.__path__[0], 'kube', template_name)

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
                       force):
        kube = self.kube
        if config_py.startswith('/'):
            config_py = config_py
        else:
            config_py = U.f_join('/root', config_py)

        prefixed_name = kube.prefix_username(experiment_name)
        stripped_name = kube.strip_username(experiment_name)
        experiment_folder = kube.get_remote_experiment_folder(stripped_name)
        rendered_path = kube.get_path(stripped_name, 'kurreal.yml')
        U.f_mkdir_in_path(rendered_path)

        cmd_gen = CommandGenerator(
            num_agents=num_agents,
            experiment_folder=experiment_folder,
            config_py=config_py,
            config_command=config_command,
            service_url=prefixed_name + '.surreal',
            restore=restore,
            restore_folder=restore_folder,
        )
        # local subfolder of kurreal.yml will strip away "<username>-" prefix
        cmd_dict = cmd_gen.generate(
            kube.get_path(stripped_name, 'launch_commands.yml'))
        # follow the printing from cmd_gen.generate()
        print('  agent_pod_type:', agent_pod_type)
        print('  nonagent_pod_type:', nonagent_pod_type)

        kube.create_surreal(
            prefixed_name,
            jinja_template=self._find_kurreal_template('kurreal_template.yml'),
            rendered_path=rendered_path,
            snapshot=not no_snapshot,
            agent_pod_type=agent_pod_type,
            nonagent_pod_type=nonagent_pod_type,
            cmd_dict=cmd_dict,
            check_experiment_exists=not force,
        )
        # switch to the experiment namespace just created
        kube.set_namespace(prefixed_name)

    def kurreal_create(self, args):
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
        )

    def kurreal_create_dev(self, args):
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
        )

    def kurreal_restore(self, args):
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

    def kurreal_resume(self, args):
        args.force = True  # always override the generated kurreal.yml
        args.restore_experiment = args.experiment_name
        self.kurreal_restore(args)

    def _interactive_find_ns(self, name, max_matches=10):
        """
        Find partial match of namespace, ask user to verify before switching to
        ns or delete experiment.
        Used in:
        - kurreal delete
        - kurreal ns
        Disabled when --force
        """
        matches = self.kube.fuzzy_match_namespace(name, max_matches=max_matches)
        if isinstance(matches, str):
            return matches  # exact match
        if len(matches) == 0:
            print_err('namespace `{}` not found. '
                      'Please run `kurreal list ns` and check for typos.'.format(name))
            return None
        elif len(matches) == 1:
            match = matches[0]
            print_err('No exact match. Fuzzy match only finds one candidate: "{}"'
                      .format(match))
            return match
        prompt = '\n'.join(['{}) {}'.format(i, n) for i, n in enumerate(matches)])
        prompt = ('Cannot find exact match. Fuzzy matching: \n'
                  '{}\nEnter your selection 0-{} (enter to select 0, q to quit): '
                  .format(prompt, len(matches) - 1))
        ans = input(prompt)
        if not ans.strip():  # blank
            ans = '0'
        try:
            ans = int(ans)
        except ValueError:  # cannot convert to int, quit
            print_err('aborted')
            return None
        if ans >= len(matches):
            raise IndexError('must enter a number between 0 - {}'.format(len(matches)-1))
        return matches[ans]

    def _kurreal_delete(self, experiment_name, force, dry_run):
        """
        Stop an experiment, delete corresponding pods, services, and namespace.
        If experiment_name is omitted, default to deleting the current namespace.
        """
        kube = self.kube
        if experiment_name:
            to_delete = experiment_name
            if force:
                assert to_delete in kube.list_namespaces(), \
                    'namespace `{}` not found. ' \
                    'Run without --force to fuzzy match the name.'.format(to_delete)
            else:  # fuzzy match namespace to delete
                to_delete = self._interactive_find_ns(to_delete)
                if to_delete is None:
                    return
        else:
            to_delete = kube.current_namespace()

        assert to_delete not in ['default', 'kube-public', 'kube-system'], \
            'cannot delete reserved namespaces: default, kube-public, kube-system'
        if not force and not dry_run:
            ans = input('Confirm delete {}? <enter>=yes,<n>=no: '.format(to_delete))
            if ans not in ['', 'y', 'yes', 'Y']:
                print('aborted')
                return

        kube.delete(
            namespace=to_delete,
            # don't need to specify yaml cause deleting a namespace
            # auto-deletes all resources under it
            # yaml_path=kube.get_path(kube.strip_username(to_delete), 'kurreal.yml'),
            yaml_path=None
        )
        print('deleting all resources under namespace "{}"'.format(to_delete))

    def kurreal_delete(self, args):
        """
        Stop an experiment, delete corresponding pods, services, and namespace.
        If experiment_name is omitted, default to deleting the current namespace.
        """
        self._kurreal_delete(args.experiment_name, args.force, args.dry_run)

    def kurreal_delete_batch(self, args):
        """
        Stop an experiment, delete corresponding pods, services, and namespace.
        If experiment_name is omitted, default to deleting the current namespace.
        Matches all possible experiments
        """
        out, _, _ = self.kube.run_verbose('get namespace -o name',
                        print_out=False,raise_on_error=True)
        namespaces = [x.strip()[len('namespaces/'):] for x in out.split()]
        cwd = os.getcwd()
        processes = []
        for namespace in namespaces:
            if re.match(args.experiment_name, namespace):
                self._kurreal_delete(namespace, args.force, args.dry_run)

    def kurreal_namespace(self, args):
        """
        `kurreal ns`: show the current namespace/experiment
        `kurreal ns <namespace>`: switch context to another namespace/experiment
        """
        kube = self.kube
        name = args.experiment_name
        if name:
            name = self._interactive_find_ns(name)
            if name is None:
                return
            kube.set_namespace(name)
        else:
            print(kube.current_namespace())

    def _get_namespace(self, args):
        "Returns: <fuzzy-matched-name>"
        name = args.namespace
        if not name:
            return ''
        name = self._interactive_find_ns(name)
        if not name:
            sys.exit(1)
        return name

    def kurreal_list(self, args):
        """
        List resource information: namespace, pods, nodes, services
        """
        run = lambda cmd: \
            self.kube.run_verbose(cmd, print_out=True, raise_on_error=False)
        if args.all:
            ns_cmd = ' --all-namespaces'
        elif args.namespace:
            ns_cmd = ' --namespace ' + self._get_namespace(args)
        else:
            ns_cmd = ''
        if args.resource in ['ns', 'namespace', 'namespaces',
                             'e', 'exp', 'experiment', 'experiments']:
            run('get namespace')
        elif args.resource in ['p', 'pod', 'pods']:
            run('get pods -o wide' + ns_cmd)
        elif args.resource in ['no', 'node', 'nodes']:
            run('get nodes -o wide' + ns_cmd)
        elif args.resource in ['s', 'svc', 'service', 'services']:
            run('get services -o wide' + ns_cmd)
        else:
            raise ValueError('INTERNAL ERROR: invalid kurreal list choice.')

    def kurreal_pod(self, args):
        "same as 'kurreal list pod'"
        args.resource = 'pod'
        self.kurreal_list(args)

    def kurreal_log(self, args):
        """
        Show logs of Surreal components: agent-<N>, learner, ps, etc.
        https://kubernetes-v1-4.github.io/docs/user-guide/kubectl/kubectl_logs/
        """
        self.kube.logs_surreal(
            args.component_name,
            is_print=True,
            follow=args.follow,
            since=args.since,
            tail=args.tail,
            namespace=self._get_namespace(args)
        )

    def kurreal_exec(self, args):
        """
        Exec command on a Surreal component: agent-<N>, learner, ps, etc.
        kubectl exec -ti <component> -- <command>
        """
        if not args.has_remainder:
            raise RuntimeError(
                'please enter your command after "--". '
                'One and only one "--" must be present. \n'
                'Example: kurreal exec learner [optional-namespace] -- ls -alf /fs/'
            )
        commands = args.remainder
        if len(commands) == 1:
            commands = commands[0]  # don't quote the singleton string
        self.kube.exec_surreal(
            args.component_name,
            commands,
            namespace=self._get_namespace(args)
        )

    def kurreal_scp(self, args):
        """
        https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#cp
        kurreal cp /my/local/file learner:/remote/file mynamespace
        is the same as
        kubectl cp /my/local/file mynamespace/nonagent:/remote/file -c learner
        """
        self.kube.scp_surreal(
            args.src_file, args.dest_file, self._get_namespace(args)
        )

    def kurreal_download_experiment(self, args):
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

    def kurreal_ssh(self, args):
        """
        Interactive /bin/bash into the pod
        kubectl exec -ti <component> -- /bin/bash
        """
        self.kube.exec_surreal(
            args.component_name,
            '/bin/bash',
            namespace=self._get_namespace(args)
        )

    def kurreal_ssh_node(self, args):
        """
        GCloud only, ssh into gcloud nodes.
        Run `kurreal list node` to get the node name.
        Run with --configure-ssh if ssh config is outdated
        """
        errcode = self.kube.gcloud_ssh_node(args.node_name)
        if errcode != 0:
            print_err('GCloud ssh aliases might be outdated. Try:\n'
                      'kurreal configure-ssh && '
                      'kurreal ssh-node ' + args.node_name)

    def kurreal_ssh_nfs(self, args):
        """
        GCloud only, ssh into gcloud NFS.
        Its server address should be specified in ~/.surreal.yml
        Run with --configure-ssh if ssh config is outdated
        """
        errcode = self.kube.gcloud_ssh_fs()
        if errcode != 0:
            print_err('GCloud ssh aliases might be outdated. Try:\n'
                      'kurreal configure-ssh && kurreal ssh-nfs')

    def kurreal_configure_ssh(self, args):
        errcode = self.kube.gcloud_configure_ssh()
        if errcode == 0:
            print('GCloud ssh configured successfully')

    def kurreal_describe(self, args):
        """
        Same as `kubectl describe pod <pod_name>`
        """
        self.kube.describe(args.pod_name, namespace=self._get_namespace(args))

    def kurreal_tensorboard(self, args):
        """
        Open tensorboard in your default browser.
        """
        url = self.kube.external_ip(
            'tensorboard',
            namespace=self._get_namespace(args)
        )
        if url:
            url = 'http://' + url
            print(url)
            if not args.url_only:
                webbrowser.open(url)
        else:
            print_err('Tensorboard does not yet have an external IP.')

    def kurreal_create_tensorboard(self, args):
        """
        Create a single pod that displays tensorboard of an old experiment.
        After the service is up and running, run `kurreal tb` to open the external
        URL in your browser
        """
        kube = self.kube
        remote_subfolder = args.remote_experiment_subfolder
        rendered_name = (remote_subfolder.replace('/', '-')
                         .replace('.', '-').replace('_', '-'))
        rendered_path = kube.get_path('kurreal-tensorboard', rendered_name + '.yml')
        U.f_mkdir_in_path(rendered_path)
        if not args.absolute_path:
            remote_path = kube.get_remote_experiment_folder(remote_subfolder)
        else:
            remote_path = remote_subfolder
        namespace = kube.create_tensorboard(
            remote_path=remote_path,
            jinja_template=self._find_kurreal_template('tensorboard_template.yml'),
            rendered_path=rendered_path,
            tensorboard_pod_type=args.pod_type
        )
        print('Creating standalone tensorboard pod. ')
        print('Please run `kurreal tb` to open the tensorboard URL in your browser '
              'when the service is up and running. '
              'You can check service by `kurreal list service`')
        print('  remote_path:', remote_path)
        print('  tensorboard_pod_type:', args.pod_type)
        # switch to the standalone pod namespace just created
        kube.set_namespace(namespace)

    def kurreal_capture_tensorboard(self, args):
        print('############### \n '
            'If this command fails, check that your surreal.yml contains\n'
            'capture_tensorboard:'
            '  node_path: ...'
            '  library_path: ...'
            '###############')
        pattern = args.experiment_prefix
        out, _, _ = self.kube.run_verbose('get namespace -o name',
                                print_out=False,raise_on_error=True)
        # out is in format namespaces/[namespace_name]
        namespaces = [x.strip()[len('namespaces/'):] for x in out.split()]
        cwd = os.getcwd()
        processes = []
        for namespace in namespaces:
            if re.match(pattern, namespace):
                job = self.kube.capture_tensorboard(namespace)
                processes.append(job)
                # My computer cannot do everything at the same time unfortunately
                job.wait()
        # for process in processes:
        #     process.wait()

    def kurreal_label(self, args):
        """
        Label nodes in node pools
        """
        for label, value in args.new_labels:
            self.kube.label_nodes(args.old_labels, label, value)

    def kurreal_label_gcloud(self, args):
        """
        NOTE: you don't need this for autoscale

        Add default labels for GCloud cluster.
        Note that you have to create the node-pools with the exact names:
        "agent-pool" and "nonagent-pool-cpu"
        gcloud container node-pools create agent-pool-cpu -m n1-standard-2 --num-nodes=8

        Command to check whether the labeling is successful:
        kubectl get node -o jsonpath="{range .items[*]}{.metadata.labels['surreal-node']}{'\n---\n'}{end}"
        """
        kube = self.kube
        kube.label_nodes('cloud.google.com/gke-nodepool=agent-pool',
                         'surreal-node', 'agent-pool')
        kube.label_nodes('cloud.google.com/gke-nodepool=nonagent-pool',
                         'surreal-node', 'nonagent-pool')


def main():
    parser = KurrealParser().setup_master()
    assert sys.argv.count('--') <= 1, \
        'command line can only have at most one "--"'
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        remainder = sys.argv[idx+1:]
        sys.argv = sys.argv[:idx]
        has_remainder = True  # even if remainder itself is empty
    else:
        remainder = []
        has_remainder = False
        
    args = parser.parse_args()
    args.remainder = remainder
    args.has_remainder = has_remainder
    args.func(args)


if __name__ == '__main__':
    main()
