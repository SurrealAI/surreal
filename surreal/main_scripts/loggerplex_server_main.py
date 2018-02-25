import os.path as path
from tensorplex import Loggerplex
from surreal.kube.tar_snapshot import tar_kurreal_repo


def loggerplex_parser_setup(parser):
    pass


def run_loggerplexserver_main(args, config):
    """
    Loggerplex is also responsible for creating a tar.gz snapshot of
    the mounted kurreal git repos
    """
    folder = config.session_config.folder
    loggerplex_config = config.session_config.loggerplex

    tar_kurreal_repo(folder)

    loggerplex = Loggerplex(
        path.join(folder, 'logs'),
        level=loggerplex_config.level,
        overwrite=loggerplex_config.overwrite,
        show_level=loggerplex_config.show_level,
        time_format=loggerplex_config.time_format
    )
    loggerplex.start_server(loggerplex_config.port)
