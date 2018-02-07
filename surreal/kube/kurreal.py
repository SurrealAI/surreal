import argparse
import surreal
from surreal.kube.kubectl import *
from surreal.kube.generate_command import *
from surreal.kube.yaml_util import *
from surreal.utils.serializer import string_hash
from pkg_resources import resource_filename


def _add_dry_run(parser):
    parser.add_argument(
        '-dr', '--dry-run',
        action='store_true',
        help='print the rendered yaml and not actually execute it.'
    )


def _process_experiment_name(experiment_name):
    """
    experiment_name will be used as DNS, so must not have underscore or dot
    """
    return experiment_name.lower().replace('.', '-').replace('_', '-')


def _add_experiment_name(parser, nargs=None):
    parser.add_argument(
        'experiment_name',
        type=_process_experiment_name,
        nargs=nargs,
        help='experiment name will be used as namespace for DNS. '
             'Should only contain lower case letters, digits, and hypen. '
             'Underscores and dots are not allowed and will be converted to hyphen.'
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

    # parser.add_argument('config', help='file name of the config')
    # parser.add_argument('--service-url', type=str, help='override domain name for parameter server and replay server. (Used when they are on the same machine)')
    subparsers = parser.add_subparsers(
        help='kurreal actions',
        dest='subcommand_name'  # will store to parser.dest
    )

    create_parser = subparsers.add_parser('create')
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
    _add_dry_run(create_parser)
    create_parser.add_argument(
        '-sn', '--snapshot',
        action='store_true',
        help='upload a snapshot of the git repos (specified in ~/.surreal.yml).'
    )
    create_parser.add_argument(
        '--force',
        action='store_true',
        help='force overwrite an existing kurreal.yml file '
             'if its experiment folder already exists.'
    )
    create_parser.set_defaults(func=kurreal_create)

    stop_parser = subparsers.add_parser('stop')
    _add_experiment_name(stop_parser)
    _add_dry_run(stop_parser)
    stop_parser.set_defaults(func=kurreal_stop)

    label_parser = subparsers.add_parser('label')
    _add_dry_run(label_parser)
    label_parser.add_argument(
        'old_labels',
        help='select nodes according to their old labels'
    )
    label_parser.add_argument(
        'new_labels',
        type=_process_labels,
        help='mark the selected nodes with new labels in format '
             '"mylabel1=myvalue1,mylabel2=myvalue2"'
    )
    label_parser.set_defaults(func=kurreal_label)

    label_gcloud_parser = subparsers.add_parser(
        'label-gcloud',
        help=kurreal_label_gcloud.__doc__,
    )
    _add_dry_run(label_gcloud_parser)
    label_gcloud_parser.set_defaults(func=kurreal_label_gcloud)

    logs_parser = subparsers.add_parser('logs')
    logs_parser.add_argument(
        'component_name',
        help="must be either agent-<N> or one of "
             "'learner', 'ps', 'replay', 'tensorplex', 'tensorboard'"
    )
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
    _add_dry_run(logs_parser)
    logs_parser.set_defaults(func=kurreal_logs)

    namespace_parser = subparsers.add_parser('ns')
    # no arg to get the current namespace
    _add_experiment_name(namespace_parser, nargs='?')
    _add_dry_run(namespace_parser)
    namespace_parser.set_defaults(func=kurreal_namespace)

    list_parser = subparsers.add_parser('list')
    _add_dry_run(list_parser)
    list_parser.add_argument(
        'resource',
        choices=['ns', 'namespace', 'p', 'pod', 'no', 'node', 's', 'service'],
        help='list experiment, pod, and node'
    )
    list_parser.set_defaults(func=kurreal_list)

    debug_create_parser = subparsers.add_parser('debug-create')
    _add_experiment_name(debug_create_parser)
    _add_dry_run(debug_create_parser)
    debug_create_parser.add_argument('-sn', '--snapshot', action='store_true')
    debug_create_parser.add_argument('num_agents', type=int)
    debug_create_parser.set_defaults(func=kurreal_debug_create)

    return parser


def _find_kurreal_template():
    """
    https://stackoverflow.com/questions/20298729/pip-installing-data-files-to-the-wrong-place
    make sure the path is consistent with MANIFEST.in
    """
    return U.f_join(surreal.__path__[0], 'kube', 'kurreal_template.yml')


def kurreal_create(args, remainder):
    """
    CommandGenerator('/mylibs/surreal/surreal/surreal/main/ddpg_configs.py',
    config_command="--env 'dm_control:cheetah-run' --savefile /experiment/",
    service_url=service_url)
    """
    kube = Kubectl(dry_run=args.dry_run)
    if args.config_py.startswith('/'):
        config_py = args.config_py
    else:
        config_py = U.f_join('/root', args.config_py)
    cmd_gen = CommandGenerator(
        config_py,
        config_command=' '.join(remainder),
        service_url=args.experiment_name + '.surreal'
    )
    cmd_dict = cmd_gen.generate(args.num_agents)
    kube.create_surreal(
        args.experiment_name,
        jinja_template=_find_kurreal_template(),
        snapshot=args.snapshot,
        check_file_exists=not args.force,
        NONAGENT_HOST_NAME=args.experiment_name,
        # TODO change to NFS
        FILE_SERVER='temp',
        PATH_ON_SERVER='/',
        CMD_DICT=cmd_dict
    )
    # switch to the experiment namespace just created
    kurreal_namespace(args, remainder)


def kurreal_debug_create(args, _):
    kube = Kubectl(dry_run=args.dry_run)
    cmd_gen = CommandGenerator(
        # '/mylibs/surreal/surreal/surreal/main/ddpg_configs.py',
        'surreal/surreal/main/ddpg_configs.py',
        config_command="--env 'dm_control:cheetah-run' --savefile /experiment/",
        service_url=args.experiment_name + '.surreal'
    )
    cmd_dict = cmd_gen.generate(args.num_agents)
    kube.create_surreal(
        args.experiment_name,
        jinja_template=_find_kurreal_template(),
        snapshot=args.snapshot,
        check_file_exists=False,
        NONAGENT_HOST_NAME=args.experiment_name,
        # TODO change to NFS
        FILE_SERVER='temp',
        PATH_ON_SERVER='/',
        CMD_DICT=cmd_dict
    )
    kurreal_namespace(args, _)


def kurreal_stop(args, _):
    kube = Kubectl(dry_run=args.dry_run)
    kube.stop(args.experiment_name)


def kurreal_namespace(args, _):
    """
    If no arg specified to `kurreal ns`, show the current namespace
    """
    kube = Kubectl(dry_run=args.dry_run)
    if args.experiment_name:
        kube.set_namespace(args.experiment_name)
    else:
        print(kube.current_namespace())


def kurreal_list(args, _):
    kube = Kubectl(dry_run=args.dry_run)
    run = lambda cmd: kube.run_verbose(cmd, print_out=True, raise_on_error=False)
    if args.resource in ['ns', 'namespace']:
        run('get namespace')
    elif args.resource in ['p', 'pod']:
        run('get pods -o wide')
    elif args.resource in ['no', 'node']:
        run('get nodes -o wide')
    elif args.resource in ['s', 'service']:
        run('get services -o wide')
    else:
        raise ValueError('INTERNAL ERROR: invalid kurreal list choice.')


def kurreal_label(args, _):
    kube = Kubectl(dry_run=args.dry_run)
    for label, value in args.new_labels:
        kube.label_nodes(args.old_labels, label, value)


def kurreal_label_gcloud(args, _):
    """
    Add default labels for GCloud cluster.
    Note that you have to create the node-pools with the exact names:
    "agent-pool" and "nonagent-pool"
    gcloud container node-pools create agent-pool -m n1-standard-2 --num-nodes=8

    Command to check whether the labeling is successful:
    kubectl get node -o jsonpath="{range .items[*]}{.metadata.labels['surreal-node']}{'\n---\n'}{end}"
    """
    kube = Kubectl(dry_run=args.dry_run)
    kube.label_nodes('cloud.google.com/gke-nodepool=agent-pool',
                     'surreal-node', 'agent-pool')
    kube.label_nodes('cloud.google.com/gke-nodepool=nonagent-pool',
                     'surreal-node', 'nonagent-pool')


def kurreal_logs(args, _):
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


def kurreal_tb(args, _):
    """
    Open tensorboard in your default browser.
    -ip, --ip to just print IP
    """
    kube = Kubectl(dry_run=args.dry_run)



def main():
    parser = setup_parser()
    args, remainder = parser.parse_known_args()
    if '--' in remainder:
        remainder.remove('--')
    args.func(args, remainder)


if __name__ == '__main__':
    main()
