import os
from tensorplex import Tensorplex


def tensorplex_parser_setup(parser):
    pass


def run_tensorplexserver_main(args, config):
    folder = os.path.join(config.session_config.folder, 'tensorboard')
    tensorplex_config = config.session_config.tensorplex

    tensorplex = Tensorplex(
        folder,
        max_processes=tensorplex_config.max_processes,
    )

    (tensorplex
        .register_normal_group('stats')
        .register_indexed_group('agents', tensorplex_config.agent_bin_size)
        .register_indexed_group('eval', 4)
        .register_indexed_group('replay', 99)
    )

    tensorplex.start_server(
        port=tensorplex_config.port,
    )