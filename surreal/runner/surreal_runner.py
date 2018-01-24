import argparse
import sys
import os.path as path
import importlib
from surreal.session import Config, BASE_LEARNER_CONFIG, BASE_ENV_CONFIG, BASE_SESSION_CONFIG

def agent_parser_setup(parser):
    parser.add_argument('id', type=int, help='agent id')

def run_agent_main(args, config):
    pass

def eval_parser_setup(parser):
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--render', action='store_true', default=False)

def run_eval_main(args, config):
    pass

def learner_parser_setup(parser):
    pass

def run_learner_main(args, config):
    pass

def replay_parser_setup(parser):
    pass

def run_replay_main(args, config):
    pass

def parameterserver_parser_setup(parser):
    pass

def run_parameterserver_main(args, config):
    pass

def tensorplex_parser_setup(parser):
    pass

def run_tensorplexserver_main(args, config):
    pass

def loggerplex_parser_setup(parser):
    pass

def run_loggerplexserver_main(args, config):
    pass

def setup_parser():
    """
        Sets up the high level argument parser and delegate corresponding sub-parsers to every component
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='process type', dest='subcommand_name')

    agent_parser = subparsers.add_parser('agent')
    agent_parser.set_defaults(function=run_agent_main)
    agent_parser_setup(agent_parser)

    eval_parser = subparsers.add_parser('eval')
    agent_parser.set_defaults(function=run_eval_main)
    eval_parser_setup(eval_parser)

    learner_parser = subparsers.add_parser('learner')
    agent_parser.set_defaults(function=run_learner_main)
    learner_parser_setup(learner_parser)

    replay_parser = subparsers.add_parser('replay-server')
    agent_parser.set_defaults(function=run_replay_main)
    replay_parser_setup(replay_parser)

    parameterserver_parser = subparsers.add_parser('parameter-server')
    parameterserver_parser.set_defaults(function=run_parameterserver_main)
    parameterserver_parser_setup(parameterserver_parser)

    tensorplex_parser = subparsers.add_parser('tensorplex-server')
    tensorplex_parser.set_defaults(function=run_tensorplexserver_main)
    tensorplex_parser_setup(tensorplex_parser)

    loggerplex_parser = subparsers.add_parser('loggerplex-server')
    loggerplex_parser.set_defaults(function=run_loggerplexserver_main)
    loggerplex_parser_setup(loggerplex_parser)

    parser.add_argument('config', help='file name of the config')

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
    if not path.isfile(pathname):
        raise ValueError('{} must be a valid file.'.fomrat(pathname))
    pathname = path.abspath(pathname)
    directory, file = path.split(pathname)
    
    basename, ext = path.splitext(file)
    if not ext == '.py':
        raise ValueError('{} must be a valid python module.'.fomrat(pathname))

    sys.path.append(directory)
    config_module = importlib.import_module(basename)

    validate_config_module(config_module)

    config_module.learner_config= Config(config_module.learner_config).extend(BASE_LEARNER_CONFIG)
    config_module.env_config= Config(config_module.env_config).extend(BASE_ENV_CONFIG)
    config_module.session_config= Config(config_module.session_config).extend(BASE_SESSION_CONFIG)

    return config_module

def main():
    parser = setup_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    args.function(args, config)

if __name__ == '__main__':
    main()
