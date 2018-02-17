import argparse
import surreal
import webbrowser
from surreal.kube.kubectl import *
from surreal.kube.generate_command import *


def _add_dry_run(parser):
    parser.add_argument(
        '-dr', '--dry-run',
        action='store_true',
        help='print the kubectl command without actually executing it.'
    )


def _process_experiment_name(experiment_name):
    """
    experiment_name will be used as DNS, so must not have underscore or dot
    """
    new_name = experiment_name.lower().replace('.', '-').replace('_', '-')
    if new_name != experiment_name:
        print_err('experiment name string has been fixed: {} -> {}'
                  .format(experiment_name, new_name))
    return new_name


def _add_experiment_name(parser, nargs=None):
    parser.add_argument(
        'experiment_name',
        type=_process_experiment_name,
        nargs=nargs,
        help='experiment name will be used as namespace for DNS. '
             'Should only contain lower case letters, digits, and hypen. '
             'Underscores and dots are not allowed and will be converted to hyphen.'
    )


def _add_component_arg(parser):
    nonagent_str = ', '.join(map('"{}"'.format, Kubectl.NONAGENT_COMPONENTS))
    parser.add_argument(
        'component_name',
        help="should be either agent-<N> or one of [{}]".format(nonagent_str)
    )


def _process_labels(label_string):
    """
    mylabel1=myvalue1,mylabel2=myvalue2
    """
    assert '=' in label_string
    label_pairs = label_string.split(',')
    return [label_pair.split('=') for label_pair in label_pairs]


def setup_parser():
    parser = argparse.ArgumentParser()
    _add_dry_run(parser)

    subparsers = parser.add_subparsers(
        help='kurreal action commands',
        dest='kurreal_action'  # will store to parser.subcommand_name
    )
    subparsers.required = True

    def _add_subparser(name, parse_func, **kwargs):
        parser = subparsers.add_parser(
            name,
            help=parse_func.__doc__,
            **kwargs
        )
        _add_dry_run(parser)
        parser.set_defaults(func=parse_func)
        return parser

    create_parser = _add_subparser('create', kurreal_create, aliases=['c'])
    _add_experiment_name(create_parser)
    create_parser.add_argument(
        'config_py',
        type=str,
        help='location of python script **in the Kube pod** that contains the '
             'runnable config. If the path does not start with /, defaults to '
             'home dir, i.e. /root/ on the pod'
    )
    create_parser.add_argument(
        'num_agents',
        type=int,
        help='number of agents to run in parallel.'
    )
    create_parser.add_argument(
        '-at', '--agent-pod-type',
        default='agent',
        help='key in ~/.surreal.yml `pod_types` section that describes spec for agent pod. '
             'Default: "agent"'
    )
    create_parser.add_argument(
        '-nt', '--nonagent-pod-type',
        default='nonagent-cpu',
        help='key in ~/.surreal.yml `pod_types` section that describes spec for '
             'nonagent pod with multiple containers: learner, ps, tensorboard, etc. '
             'Default: "nonagent-cpu"'
    )
    create_parser.add_argument(
        '-sn', '--snapshot',
        action='store_true',
        help='take a snapshot of the specified Git repos in ~/.surreal.yml '
             'and upload to your remote branch'
    )
    create_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='force overwrite an existing kurreal.yml file '
             'if its experiment folder already exists.'
    )

    delete_parser = _add_subparser('delete', kurreal_delete, aliases=['d'])
    _add_experiment_name(delete_parser, nargs='?')
    delete_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='force delete, do not show confirmation message.'
    )

    logs_parser = _add_subparser('log', kurreal_logs, aliases=['logs', 'l'])
    _add_component_arg(logs_parser)
    logs_parser.add_argument(
        '-f', '--follow',
        action='store_true',
        help='if the logs should be streamed.'
    )
    logs_parser.add_argument(
        '-s', '--since',
        default='0',
        help='only show logs newer than a relative duration like 5s, 2m, 3h.'
    )
    logs_parser.add_argument(
        '-t', '--tail',
        type=int,
        default=100,
        help='Only show the most recent lines of log. -1 to show all log lines.'
    )

    exec_parser = _add_subparser('exec', kurreal_exec, aliases=['x'])
    _add_component_arg(exec_parser)
    exec_parser.add_argument(
        'commands',
        nargs=argparse.REMAINDER,
        help="command to be executed in the pod. You don't have to quote it."
    )

    ssh_parser = _add_subparser('ssh', kurreal_ssh, aliases=['login'])
    _add_component_arg(ssh_parser)

    describe_parser = _add_subparser('describe', kurreal_describe, aliases=['des'])
    describe_parser.add_argument(
        'pod_name',
        help="should be either 'agent-<N>' or 'nonagent'"
    )

    namespace_parser = _add_subparser('ns', kurreal_namespace,
                                      aliases=['exp', 'experiment'])
    # no arg to get the current namespace
    _add_experiment_name(namespace_parser, nargs='?')

    list_parser = _add_subparser('list', kurreal_list, aliases=['ls'])
    list_parser.add_argument(
        'resource',
        choices=['ns', 'namespace', 'namespaces',
                 'e', 'exp', 'experiment', 'experiments',
                 'p', 'pod', 'pods',
                 'no', 'node', 'nodes',
                 's', 'service', 'services'],
        help='list experiment, pod, and node'
    )

    tb_parser = _add_subparser('tb', kurreal_tb, aliases=['tensorboard'])
    tb_parser.add_argument(
        '-u', '--url-only',
        action='store_true',
        help='only show the URL without opening the browser.'
    )

    # ===== internal dev only =====
    create_dev_parser = _add_subparser('create-dev', kurreal_create_dev,
                                       aliases=['cdev', 'devc', 'cd', 'dev-create'])
    _add_experiment_name(create_dev_parser)
    create_dev_parser.add_argument('num_agents', type=int)
    create_dev_parser.add_argument('-nos', '--no-snapshot', action='store_true')
    create_dev_parser.add_argument('-f', '--force', action='store_true')
    create_dev_parser.add_argument('-g', '--gpu', action='store_true')
    create_dev_parser.add_argument('-c', '--config_file',
                                   default='ddpg_configs.py',
                                   help='which config file in surreal/main to use')

    # you don't need labeling for kube autoscaling
    # label_parser = _add_subparser('label', kurreal_label)
    # label_parser.add_argument(
    #     'old_labels',
    #     help='select nodes according to their old labels'
    # )
    # label_parser.add_argument(
    #     'new_labels',
    #     type=_process_labels,
    #     help='mark the selected nodes with new labels in format '
    #          '"mylabel1=myvalue1,mylabel2=myvalue2"'
    # )

    # label_gcloud_parser = _add_subparser('label-gcloud', kurreal_label_gcloud)

    return parser


def _find_kurreal_template():
    """
    https://stackoverflow.com/questions/20298729/pip-installing-data-files-to-the-wrong-place
    make sure the path is consistent with MANIFEST.in
    """
    return U.f_join(surreal.__path__[0], 'kube', 'kurreal_template.yml')


def kurreal_create(args):
    """
    Spin up a multi-node distributed Surreal experiment.
    Put any command line args that pass to the config script after "--"
    """
    kube = Kubectl(dry_run=args.dry_run)
    if args.config_py.startswith('/'):
        config_py = args.config_py
    else:
        config_py = U.f_join('/root', args.config_py)
    args.experiment_name = kube.get_experiment_name(args.experiment_name)
    cmd_gen = CommandGenerator(
        config_py,
        config_command=' '.join(args.remainder),
        service_url=args.experiment_name + '.surreal'
    )
    cmd_dict = cmd_gen.generate(args.num_agents)
    kube.create_surreal(
        args.experiment_name,
        jinja_template=_find_kurreal_template(),
        snapshot=args.snapshot,
        agent_pod_type=args.agent_pod_type,
        nonagent_pod_type=args.nonagent_pod_type,
        cmd_dict=cmd_dict,
        check_experiment_exists=not args.force,
    )
    # switch to the experiment namespace just created
    kurreal_namespace(args)


def kurreal_create_dev(args):
    """
    CommandGenerator('/mylibs/surreal/surreal/surreal/main/ddpg_configs.py',
    config_command="--env 'dm_control:cheetah-run' --savefile /experiment/",
    service_url=experiment_name + '.surreal')
    """
    kube = Kubectl(dry_run=args.dry_run)
    if len(args.remainder) > 0:
        config_command = args.remainder
    else:
        config_command = ['--env', "'dm_control:cheetah-run'"]

    if args.gpu:
        nonagent_pod_type = 'nonagent-gpu'
        config_command += ["--gpu", "0"]
    else:
        nonagent_pod_type = 'nonagent-cpu'

    config_command += ["--savefile",
       "/fs/experiments/{}/{}".format(kube.config.username, args.experiment_name)]

    args.experiment_name = kube.get_experiment_name(args.experiment_name)
    cmd_gen = CommandGenerator(
        # '/mylibs/surreal/surreal/surreal/main/ddpg_configs.py',
        'surreal/surreal/main/' + args.config_file,
        config_command=' '.join(config_command),
        service_url=args.experiment_name + '.surreal'
    )
    cmd_dict = cmd_gen.generate(args.num_agents)
    kube.create_surreal(
        args.experiment_name,
        jinja_template=_find_kurreal_template(),
        snapshot=not args.no_snapshot,
        agent_pod_type='agent',
        nonagent_pod_type=nonagent_pod_type,
        cmd_dict=cmd_dict,
        check_experiment_exists=not args.force,
    )
    kurreal_namespace(args)


def kurreal_delete(args):
    """
    Stop an experiment, delete corresponding pods, services, and namespace.
    If experiment_name is omitted, default to deleting the current namespace.
    """
    kube = Kubectl(dry_run=args.dry_run)
    if args.experiment_name:
        to_delete = args.experiment_name
    else:
        to_delete = kube.current_namespace()
    assert to_delete not in ['default', 'kube-public', 'kube-system'], \
        'cannot delete reserved namespaces: default, kube-public, kube-system'
    if not args.force:
        ans = input('Confirm delete {}? <enter>=yes,<n>=no: '.format(to_delete))
        if ans not in ['', 'y', 'yes', 'Y']:
            print('Aborted')
            return
    kube.delete(to_delete)


def kurreal_namespace(args):
    """
    `kurreal ns`: show the current namespace/experiment
    `kurreal ns <namespace>`: switch context to another namespace/experiment
    """
    kube = Kubectl(dry_run=args.dry_run)
    if args.experiment_name:
        kube.set_namespace(args.experiment_name)
    else:
        print(kube.current_namespace())


def kurreal_list(args):
    """
    List resource information: namespace, pods, nodes, services
    """
    kube = Kubectl(dry_run=args.dry_run)
    run = lambda cmd: kube.run_verbose(cmd, print_out=True, raise_on_error=False)
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


def kurreal_label(args):
    """
    Label nodes in node pools
    """
    kube = Kubectl(dry_run=args.dry_run)
    for label, value in args.new_labels:
        kube.label_nodes(args.old_labels, label, value)


def kurreal_label_gcloud(args):
    """
    NOTE: you don't need this for autoscale

    Add default labels for GCloud cluster.
    Note that you have to create the node-pools with the exact names:
    "agent-pool" and "nonagent-pool-cpu"
    gcloud container node-pools create agent-pool-cpu -m n1-standard-2 --num-nodes=8

    Command to check whether the labeling is successful:
    kubectl get node -o jsonpath="{range .items[*]}{.metadata.labels['surreal-node']}{'\n---\n'}{end}"
    """
    kube = Kubectl(dry_run=args.dry_run)
    kube.label_nodes('cloud.google.com/gke-nodepool=agent-pool',
                     'surreal-node', 'agent-pool')
    kube.label_nodes('cloud.google.com/gke-nodepool=nonagent-pool',
                     'surreal-node', 'nonagent-pool')


def kurreal_logs(args):
    """
    Show logs of Surreal components: agent-<N>, learner, ps, etc.
    https://kubernetes-v1-4.github.io/docs/user-guide/kubectl/kubectl_logs/
    """
    kube = Kubectl(dry_run=args.dry_run)
    kube.logs_surreal(
        args.component_name,
        is_print=True,
        follow=args.follow,
        since=args.since,
        tail=args.tail
    )


def kurreal_exec(args):
    """
    Exec command on a Surreal component: agent-<N>, learner, ps, etc.
    kubectl exec -ti <component> -- <command>
    """
    kube = Kubectl(dry_run=args.dry_run)
    if len(args.commands) == 1:
        args.commands = args.commands[0]  # don't quote the singleton string
    kube.exec_surreal(args.component_name, args.commands)


def kurreal_ssh(args):
    """
    Interactive /bin/bash into the pod
    kubectl exec -ti <component> -- /bin/bash
    """
    kube = Kubectl(dry_run=args.dry_run)
    kube.exec_surreal(args.component_name, '/bin/bash')


def kurreal_describe(args):
    """
    Same as `kubectl describe pod <pod_name>`
    """
    kube = Kubectl(dry_run=args.dry_run)
    kube.describe(args.pod_name)


def kurreal_tb(args):
    """
    Open tensorboard in your default browser.
    """
    kube = Kubectl(dry_run=args.dry_run)
    url = kube.external_ip('tensorboard')
    if url:
        url = 'http://' + url
        print(url)
        if not args.url_only:
            webbrowser.open(url)
    else:
        print_err('Tensorboard does not yet have an external IP.')


def main():
    parser = setup_parser()
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
