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

    """
        Tensorboard categories:
            learner/replay/eval: algorithmic level, e.g. reward, ... 
            ***-core: low level metrics, i/o speed, computation time, etc.
            ***-system: Metrics derived from raw metric data in core, i.e. exp_in/exp_out
    """
    (tensorplex
        .register_normal_group('learner')
        .register_indexed_group('agent', tensorplex_config.agent_bin_size)
        .register_indexed_group('eval', 4)
        .register_indexed_group('replay', 10)
    )

    tensorplex.start_server(
        port=tensorplex_config.port,
    )