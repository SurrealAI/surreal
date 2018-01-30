import os.path as path
from tensorplex import Loggerplex

def loggerplex_parser_setup(parser):
    pass

def run_loggerplexserver_main(args, config):
    folder = config.session_config.folder
    loggerplex_config = config.session_config.loggerplex

    loggerplex = Loggerplex(
        path.join(folder, 'log'),
        level=loggerplex_config.level,
        overwrite=loggerplex_config.overwrite,
        show_level=loggerplex_config.show_level,
        time_format=loggerplex_config.time_format
    )
    loggerplex.start_server(loggerplex_config.port)
