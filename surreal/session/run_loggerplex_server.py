"""
Only 1 command line arg: JSON string of session_config
"""
import os
import sys
import json
from tensorplex import LoggerplexServer
from surreal.session.config import Config
from surreal.session.default_configs import BASE_SESSION_CONFIG


config = Config(json.loads(sys.argv[1]))
config.extend(BASE_SESSION_CONFIG)
config = config.tensorplex


loggerplex = LoggerplexServer(
    os.path.join(config.folder, 'log'),
    overwrite=config.log_overwrite,
    debug=config.log_debug
)
loggerplex.start_server(
    host=config.host,
    port=config.port,
)
