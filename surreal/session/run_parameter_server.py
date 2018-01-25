# """
# Only 1 command line arg: JSON string of session_config
# """
# import sys
# import json
# from surreal.session.config import Config
# from surreal.session.default_configs import BASE_SESSION_CONFIG
# from surreal.distributed.ps import ParameterServer


# config = Config(json.loads(sys.argv[1]))
# config.extend(BASE_SESSION_CONFIG)

# server = ParameterServer(
#     publish_host=config.ps.publish_host,
#     publish_port=config.ps.publish_port,
#     agent_port=config.ps.port,
# )
# server.run_loop()

