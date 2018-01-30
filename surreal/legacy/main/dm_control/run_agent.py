from surreal.main.basic_boilerplate import *
import importlib
import sys

def print_usage():
    print('Usage: python run_[agent|learner|eval|replay].py [config_package_name] [...]')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        exit(0)
    
    config_pkg = importlib.import_module(sys.argv[1])

    learner_config = Config(config_pkg.learner_config).extend(BASE_LEARNER_CONFIG)
    env_config = Config(config_pkg.env_config).extend(BASE_ENV_CONFIG)
    session_config = Config(config_pkg.session_config).extend(BASE_SESSION_CONFIG)


    run_agent_main(learner_config=learner_config,
                   env_config=env_config,
                   session_config=session_config)