import os
import subprocess


def tensorboard_parser_setup(parser):
    pass


def run_tensorboardwrapper_main(args, config):
    folder = os.path.join(config.session_config.folder, 'tensorboard')
    tensorplex_config = config.session_config.tensorplex
    cmd = ['tensorboard',
           '--logdir', folder,
           '--port', str(tensorplex_config.tensorboard_port)]
    subprocess.call(cmd)
