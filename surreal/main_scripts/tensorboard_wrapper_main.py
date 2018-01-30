import os.path as path
import subprocess

def tensorboard_parser_setup(parser):
    pass

def run_tensorboardwrapper_main(args, config):
    folder = config.session_config.folder
    tensorplex_config = config.session_config.tensorplex
    cmd = ['tensorboard', '--logdir', folder, '--port', str(tensorplex_config.tensorboard_port)]
    subprocess.call(cmd)
    

    