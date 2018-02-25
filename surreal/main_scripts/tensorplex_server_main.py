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
        .register_indexed_group('learner', 1)
        .register_indexed_group('learner_system', 1)
        .register_indexed_group('learner_core', 1)
        .register_indexed_group('agent', tensorplex_config.agent_bin_size)
        .register_indexed_group('agent_system', tensorplex_config.agent_bin_size)
        .register_indexed_group('agent_core', tensorplex_config.agent_bin_size)
        .register_indexed_group('eval', 4)
        .register_indexed_group('eval_system', 4)
        .register_indexed_group('eval_core', 4)
        .register_indexed_group('replay', 10)
        .register_indexed_group('replay_system', 10)
        .register_indexed_group('replay_core', 10)
    )

    tensorplex.start_server(
        port=tensorplex_config.port,
    )