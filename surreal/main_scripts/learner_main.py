from pathlib import Path
from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.learner import learner_factory


def learner_parser_setup(parser):
    parser.add_argument(
        '--restore',
        action='store_true',
        help='restore from checkpoint'
    )
    parser.add_argument(
        '--restore-folder',
        default=None,
        help='load from folder other than the current experiment folder. '
             'set None to load from this experiment folder.'
    )


def run_learner_main(args, config):
    session_config, learner_config, env_config = \
        config.session_config, config.learner_config, config.env_config
    env, env_config = make_env(env_config, 'learner')
    del env  # Does not work for dm_control as they don't clean up

    if args.restore:
        session_config.checkpoint.restore = True
        session_config.checkpoint.restore_folder = args.restore_folder

    folder = Path(session_config.folder)
    folder.mkdir(exist_ok=True, parents=True)
    config.dump_file(str(folder / 'config.yml'))

    learner_class = learner_factory(learner_config.algo.learner_class)
    learner = learner_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config
    )
    learner.main_loop()