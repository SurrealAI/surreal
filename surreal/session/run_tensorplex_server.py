"""
Only 1 command line arg: JSON string of session_config
"""
import os
import sys
import json
from tensorplex import TensorplexServer, NumberedGroup
from surreal.session.config import Config
from surreal.session.tensorplex_adapter import (
    AGENT_GROUP_NAME, NONAGENT_GROUP_NAME
)
from surreal.session.default_configs import BASE_SESSION_CONFIG


config = Config(json.loads(sys.argv[1]))
config.extend(BASE_SESSION_CONFIG)
config = config.tensorplex


tensorplex = TensorplexServer(
    config.folder,
    normal_groups=[NONAGENT_GROUP_NAME],
    numbered_groups=[NumberedGroup(name=AGENT_GROUP_NAME,
                                   N=config.num_agents,
                                   bin_size=config.agent_bin_size)]
)
tensorplex.start_server(
    host=config.host,
    port=config.port,
)
