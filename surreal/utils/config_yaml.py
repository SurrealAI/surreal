import os
from surreal.utils.filesys import f_expand


def get_config_file():
    """
    Get the path to
    """
    path = '~/.surreal.yml'
    if 'SURREAL_CONFIG_PATH' in os.environ:
        path = os.environ['SURREAL_CONFIG_PATH']
    return f_expand(path)
