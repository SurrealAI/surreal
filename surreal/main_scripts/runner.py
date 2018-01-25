"""
    This file provides the centiralized entry point for all surreal components.
"""

import argparse
import sys
import os.path as path
import importlib
from surreal.main_scripts.agent_main import agent_parser_setup, run_agent_main
from surreal.main_scripts.eval_main import eval_parser_setup, run_eval_main
from surreal.main_scripts.learner_main import learner_parser_setup, run_learner_main
from surreal.main_scripts.replay_main import replay_parser_setup, run_replay_main
from surreal.main_scripts.parameter_server_main import parameterserver_parser_setup, run_parameterserver_main
from surreal.main_scripts.tensorplex_server_main import tensorplex_parser_setup, run_tensorplexserver_main
from surreal.main_scripts.tensorboard_wrapper_main import tensorboard_parser_setup, run_tensorboardwrapper_main
from surreal.main_scripts.loggerplex_server_main import loggerplex_parser_setup, run_loggerplexserver_main
from surreal.session import Config, BASE_LEARNER_CONFIG, BASE_ENV_CONFIG, BASE_SESSION_CONFIG


"""
    def *_parser_setup(parser):
        @parser: argparse.Argumentparser() that parses subcommands 
"""

"""
    def run_*_main(args, config):
        @args: parsed arguments with specifications defined in parser_setup
        @config: config that passes validate_config(config)
"""

def setup_parser():
    """
        Sets up the high level argument parser and delegate corresponding sub-parsers to every component
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='file name of the config')
    
    subparsers = parser.add_subparsers(help='process type', dest='subcommand_name')

    agent_parser = subparsers.add_parser('agent')
    agent_parser.set_defaults(function=run_agent_main)
    agent_parser_setup(agent_parser)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.set_defaults(function=run_eval_main)
    eval_parser_setup(eval_parser)

    learner_parser = subparsers.add_parser('learner')
    learner_parser.set_defaults(function=run_learner_main)
    learner_parser_setup(learner_parser)

    replay_parser = subparsers.add_parser('replay')
    replay_parser.set_defaults(function=run_replay_main)
    replay_parser_setup(replay_parser)

    parameterserver_parser = subparsers.add_parser('ps')
    parameterserver_parser.set_defaults(function=run_parameterserver_main)
    parameterserver_parser_setup(parameterserver_parser)

    tensorplex_parser = subparsers.add_parser('tensorplex')
    tensorplex_parser.set_defaults(function=run_tensorplexserver_main)
    tensorplex_parser_setup(tensorplex_parser)

    loggerplex_parser = subparsers.add_parser('loggerplex')
    loggerplex_parser.set_defaults(function=run_loggerplexserver_main)
    loggerplex_parser_setup(loggerplex_parser)

    tensorboard_parser = subparsers.add_parser('tensorboard')
    tensorboard_parser.set_defaults(function=run_tensorboardwrapper_main)
    tensorboard_parser_setup(tensorboard_parser)

    

    return parser

def validate_config_module(config_module):
    """
        Validates that the config module provides the proper files
    """
    pass

def load_config(pathname):
    """
        Load the python module specified in pathname
    """

    pathname = path.expanduser(pathname)
    pathname = path.abspath(pathname)
    
    if not path.isfile(pathname):
        raise ValueError('{} must be a valid file.'.format(pathname))
    
    directory, file = path.split(pathname)
    
    basename, ext = path.splitext(file)
    if not ext == '.py':
        raise ValueError('{} must be a valid python module.'.format(pathname))

    sys.path.append(directory)
    config_module = importlib.import_module(basename)

    validate_config_module(config_module)

    config_module.learner_config = Config(config_module.learner_config).extend(BASE_LEARNER_CONFIG)
    config_module.env_config = Config(config_module.env_config).extend(BASE_ENV_CONFIG)
    config_module.session_config = Config(config_module.session_config).extend(BASE_SESSION_CONFIG)

    return config_module

def main():
    parser = setup_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    args.function(args, config)

if __name__ == '__main__':
    main()
