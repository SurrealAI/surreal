import argparse
import surreal
import webbrowser
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
        self._setup_delete()
        self._setup_log()
        self._setup_namespace()
        self._setup_tensorboard()
        self._setup_list()
        self._setup_describe()
        self._setup_exec()
        self._setup_ssh()
        self._setup_ssh_node()
        self._setup_ssh_nfs()
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
        parser.add_argument('-nos', '--no-snapshot', action='store_true')
        parser.add_argument('-f', '--force', action='store_true')
        parser.add_argument('-g', '--gpu', action='store_true')
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

    def _setup_restore(self):
        parser = self._add_subparser('restore', aliases=[])
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

    def _setup_log(self):
        parser = self._add_subparser('log', aliases=['logs', 'l'])
        self._add_component_arg(parser)
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
                     's', 'service', 'services'],
            default='ns',
            nargs='?',
            help='list experiment, pod, and node'
        )

    def _setup_tensorboard(self):
        parser = self._add_subparser('tensorboard', aliases=['tb'])
        parser.add_argument(
            '-u', '--url-only',
            action='store_true',
            help='only show the URL without opening the browser.'
        )

    def _setup_describe(self):
        parser = self._add_subparser('describe', aliases=['des'])
        parser.add_argument(
            'pod_name',
            help="should be either 'agent-<N>' or 'nonagent'"
        )

    def _setup_exec(self):
        parser = self._add_subparser('exec', aliases=['x'])
        self._add_component_arg(parser)
        parser.add_argument(
            'commands',
            nargs=argparse.REMAINDER,
            help="command to be executed in the pod. You don't have to quote it."
        )

    def _setup_ssh(self):
        parser = self._add_subparser('ssh', aliases=[])
        self._add_component_arg(parser)

    def _setup_ssh_node(self):
        parser = self._add_subparser('ssh-node', aliases=['sshnode'])
        parser.add_argument('node_name', help='gcloud only')
        parser.add_argument(
            '-c', '--configure-ssh',
            action='store_true',
            help='update ssh configs first if you cannot ssh into the node. '
                 'reconfigure every time you switch project or add new nodes.'
        )

    def _setup_ssh_nfs(self):
        parser = self._add_subparser('ssh-nfs', aliases=['sshnfs'])
        parser.add_argument(
            '-c', '--configure-ssh',
            action='store_true',
            help='update ssh configs first if you cannot ssh into the node. '
                 'reconfigure every time you switch project or add new nodes.'
        )

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

    def _add_create_args(self, parser):
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


class Kurreal:
    def __init__(self, args):
        self.kube = Kubectl(dry_run=args.dry_run)

    @staticmethod
    def _find_kurreal_template():
        """
        https://stackoverflow.com/questions/20298729/pip-installing-data-files-to-the-wrong-place
        make sure the path is consistent with MANIFEST.in
        """
        return U.f_join(surreal.__path__[0], 'kube', 'kurreal_template.yml')

    def kurreal_create(self, args):
        """
        Spin up a multi-node distributed Surreal experiment.
        Put any command line args that pass to the config script after "--"
        """
        kube = self.kube
        if args.config_py.startswith('/'):
            config_py = args.config_py
        else:
            config_py = U.f_join('/root', args.config_py)

        prefixed_name = kube.prefix_username(args.experiment_name)
        stripped_name = kube.strip_username(args.experiment_name)

        cmd_gen = CommandGenerator(
            num_agents=args.num_agents,
            experiment_folder=kube.get_remote_experiment_folder(stripped_name),
            config_py=config_py,
            config_command=args.remainder,
            service_url=prefixed_name + '.surreal',
            restore_ckpt=False,
        )
        # local subfolder of kurreal.yml will strip away "<username>-" prefix
        cmd_dict = cmd_gen.generate(
            kube.get_path(stripped_name, 'launch_commands.yml'))
        rendered_path = kube.get_path(stripped_name, 'kurreal.yml')
        kube.create_surreal(
            prefixed_name,
            jinja_template=self._find_kurreal_template(),
            rendered_path=rendered_path,
            snapshot=not args.no_snapshot,
            agent_pod_type=args.agent_pod_type,
            nonagent_pod_type=args.nonagent_pod_type,
            cmd_dict=cmd_dict,
            check_experiment_exists=not args.force,
        )
        # switch to the experiment namespace just created
        kube.set_namespace(prefixed_name)

    def kurreal_create_dev(self, args):
        """
        << internal dev only >>
        """
        if args.remainder:
            config_command = args.remainder
        else:
            config_command = ['--env', 'dm_control:cheetah-run']

        if args.gpu:
            nonagent_pod_type = 'nonagent-gpu'
            config_command += ["--gpu", "0"]
        else:
            nonagent_pod_type = 'nonagent-cpu'

        args.agent_pod_type = 'agent'
        args.nonagent_pod_type = nonagent_pod_type
        # '/mylibs/surreal/surreal/surreal/main/ddpg_configs.py'
        args.config_py = 'surreal/surreal/main/' + args.config_file
        args.remainder = config_command
        self.kurreal_create(args)

    def kurreal_restore(self, args):
        """
        Restore experiment with the saved CommandGenerator and checkpoint
        Put any command line args that pass to the config script after "--"
        """
        kube = self.kube

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

    def kurreal_delete(self, args):
        """
        Stop an experiment, delete corresponding pods, services, and namespace.
        If experiment_name is omitted, default to deleting the current namespace.
        """
        kube = self.kube
        if args.experiment_name:
            to_delete = args.experiment_name
            if args.force:
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
        if not args.force and not args.dry_run:
            ans = input('Confirm delete {}? <enter>=yes,<n>=no: '.format(to_delete))
            if ans not in ['', 'y', 'yes', 'Y']:
                print('aborted')
                return

        kube.delete(
            yaml_path=kube.get_path(kube.strip_username(to_delete), 'kurreal.yml'),
            namespace=kube.prefix_username(to_delete)
        )

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

    def kurreal_list(self, args):
        """
        List resource information: namespace, pods, nodes, services
        """
        run = lambda cmd: \
            self.kube.run_verbose(cmd, print_out=True, raise_on_error=False)
        if args.resource in ['ns', 'namespace', 'namespaces',
                             'e', 'exp', 'experiment', 'experiments']:
            run('get namespace')
        elif args.resource in ['p', 'pod', 'pods']:
            run('get pods -o wide')
        elif args.resource in ['no', 'node', 'nodes']:
            run('get nodes -o wide')
        elif args.resource in ['s', 'service', 'services']:
            run('get services -o wide')
        else:
            raise ValueError('INTERNAL ERROR: invalid kurreal list choice.')

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
            tail=args.tail
        )

    def kurreal_exec(self, args):
        """
        Exec command on a Surreal component: agent-<N>, learner, ps, etc.
        kubectl exec -ti <component> -- <command>
        """
        if len(args.commands) == 1:
            args.commands = args.commands[0]  # don't quote the singleton string
        self.kube.exec_surreal(args.component_name, args.commands)

    def kurreal_ssh(self, args):
        """
        Interactive /bin/bash into the pod
        kubectl exec -ti <component> -- /bin/bash
        """
        self.kube.exec_surreal(args.component_name, '/bin/bash')

    def kurreal_ssh_node(self, args):
        """
        GCloud only, ssh into gcloud nodes.
        Run `kurreal list node` to get the node name.
        Run with --configure-ssh if ssh config is outdated
        """
        kube = self.kube
        if args.configure_ssh:
            kube.gcloud_configure_ssh()
            print('GCloud ssh configured successfully')
        kube.gcloud_ssh_node(args.node_name)

    def kurreal_ssh_nfs(self, args):
        """
        GCloud only, ssh into gcloud NFS.
        Its server address should be specified in ~/.surreal.yml
        Run with --configure-ssh if ssh config is outdated
        """
        kube = self.kube
        if args.configure_ssh:
            kube.gcloud_configure_ssh()
            print('GCloud ssh configured successfully')
        kube.gcloud_ssh_fs()

    def kurreal_describe(self, args):
        """
        Same as `kubectl describe pod <pod_name>`
        """
        self.kube.describe(args.pod_name)

    def kurreal_tensorboard(self, args):
        """
        Open tensorboard in your default browser.
        """
        url = self.kube.external_ip('tensorboard')
        if url:
            url = 'http://' + url
            print(url)
            if not args.url_only:
                webbrowser.open(url)
        else:
            print_err('Tensorboard does not yet have an external IP.')


def main():
    parser = KurrealParser().setup_master()
    assert sys.argv.count('--') <= 1, 'command line can only have at most one "--"'
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        remainder = sys.argv[idx+1:]
        sys.argv = sys.argv[:idx]
    else:
        remainder = []
        
    args = parser.parse_args()
    args.remainder = remainder
    args.func(args)


if __name__ == '__main__':
    main()
