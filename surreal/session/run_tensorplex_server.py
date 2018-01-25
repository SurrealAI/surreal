# """
# Only 1 command line arg: JSON string of session_config
# """
# import os
# import sys
# import json
# from tensorplex import Tensorplex
# from surreal.session.config import Config
# from surreal.session.tensorplex_adapter import *
# from surreal.session.default_configs import BASE_SESSION_CONFIG


# config = Config(json.loads(sys.argv[1]))
# config.extend(BASE_SESSION_CONFIG)
# folder = config.folder
# config = config.tensorplex


# tensorplex = Tensorplex(
#     folder,
#     max_processes=config.max_processes,
# )

# (tensorplex
#     .register_normal_group(STATS_GROUP_NAME)
#     .register_indexed_group(AGENT_GROUP_NAME, config.agent_bin_size)
#     .register_combined_group(EVAL_GROUP_NAME, lambda tag: 'all')
# )

# tensorplex.start_server(
#     port=config.port,
# )
