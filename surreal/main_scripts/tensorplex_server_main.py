import os
from tensorplex import Tensorplex
from surreal.session.tensorplex_adapter import *


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
        .register_normal_group(STATS_GROUP_NAME)
        .register_indexed_group(AGENT_GROUP_NAME, tensorplex_config.agent_bin_size)
        .register_combined_group(EVAL_GROUP_NAME, lambda tag: 'all')
    )

    tensorplex.start_server(
        port=tensorplex_config.port,
    )