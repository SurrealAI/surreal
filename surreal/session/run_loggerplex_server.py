# """
# Only 1 command line arg: JSON string of session_config
# """
# import os
# import sys
# import json
# from tensorplex import Loggerplex
# from surreal.session.config import Config
# from surreal.session.default_configs import BASE_SESSION_CONFIG


# config = Config(json.loads(sys.argv[1]))
# config.extend(BASE_SESSION_CONFIG)
# folder = config.folder
# config = config.loggerplex


# loggerplex = Loggerplex(
#     os.path.join(folder, 'log'),
#     level=config.level,
#     overwrite=config.overwrite,
#     show_level=config.show_level,
#     time_format=config.time_format
# )
# loggerplex.start_server(config.port)
