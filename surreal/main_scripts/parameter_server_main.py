from surreal.distributed.ps import ParameterServer

def parameterserver_parser_setup(parser):
    pass

def run_parameterserver_main(args, config):
    folder = config.session_config.folder
    ps_config = config.session_config.ps

    server = ParameterServer(
        publish_host=ps_config.publish_host,
        publish_port=ps_config.publish_port,
        agent_port=ps_config.port,
    )

    server.run_loop()