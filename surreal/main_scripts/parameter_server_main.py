from surreal.distributed.ps import ShardedParameterServer

def parameterserver_parser_setup(parser):
    pass

def run_parameterserver_main(args, config):
    folder = config.session_config.folder
    ps_config = config.session_config.ps

    server = ShardedParameterServer(config=config)

    server.launch()
    server.join()