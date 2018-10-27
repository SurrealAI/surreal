import os
import sys
import psutil
import subprocess


# Currently planned tests
# DDPG dm_control
# DDPG mujocomanip
# PPO dm_control
# PPO mujocomanip


def _setup_env():
    """
    Setup the necessary environment variables
    """
    os.environ["SYMPH_PS_BACKEND_PORT"] = "7006"
    os.environ["SYMPH_PARAMETER_PUBLISH_PORT"] = "7001"
    os.environ["SYMPH_SAMPLER_FRONTEND_ADDR"] = "7004"
    os.environ["SYMPHONY_PARAMETER_SERVER_HOST"] = "127.0.0.1"
    os.environ["SYMPH_TENSORPLEX_HOST"] = "127.0.0.1"
    os.environ["SYMPH_TENSORPLEX_PORT"] = "7009"
    os.environ["SYMPH_LOGGERPLEX_HOST"] = "127.0.0.1"
    os.environ["SYMPH_LOGGERPLEX_PORT"] = "7003"
    os.environ["SYMPH_COLLECTOR_FRONTEND_HOST"] = "127.0.0.1"
    os.environ["SYMPH_COLLECTOR_FRONTEND_PORT"] = "7005"
    os.environ["SYMPH_PS_FRONTEND_HOST"] = "127.0.0.1"
    os.environ["SYMPH_PS_FRONTEND_PORT"] = "7008"
    os.environ["SYMPH_SAMPLER_FRONTEND_HOST"] = "127.0.0.1"
    os.environ["SYMPH_SAMPLER_FRONTEND_PORT"] = "7003"
    os.environ["SYMPH_SAMPLER_BACKEND_HOST"] = "127.0.0.1"
    os.environ["SYMPH_SAMPLER_BACKEND_PORT"] = "7002"
    os.environ["SYMPH_PARAMETER_PUBLISH_HOST"] = "127.0.0.1"
    os.environ["SYMPH_PARAMETER_PUBLISH_PORT"] = "7001"
    os.environ["SYMPH_COLLECTOR_BACKEND_HOST"] = "127.0.0.1"
    os.environ["SYMPH_COLLECTOR_BACKEND_PORT"] = "7007"
    os.environ["SYMPH_PREFETCH_QUEUE_HOST"] = "127.0.0.1"
    os.environ["SYMPH_PREFETCH_QUEUE_PORT"] = "7000"


def integration_test(temp_path,
                     config_path,
                     launcher,
                     env='gym:HalfCheetah-v2',
                     additional_args=None):
    if additional_args is None:
        additional_args = []
    print("Making temp directory...")
    os.makedirs(temp_path, exist_ok=True)
    print("Setting up experiment launcher...")
    args = [
        '--unit-test',
        '--num-agents',
        '1',
        '--env',
        'gym:HalfCheetah-v2',
        # 'robosuite:SawyerLift',
        # 'dm_control:cartpole-balance',
        '--experiment-folder',
        str(temp_path)] + additional_args

    print("Setting up environment variables...")
    _setup_env()

    subprocesses = []

    for module in ['replay', 'ps', 'eval-0']:  # tensorboard,
        subprocesses.append(subprocess.Popen([sys.executable,
                                              '-u',
                                              config_path,
                                              module,
                                              '--'] + args))
        print(module + '=' * 20 + 'done')
    print('Supplementary components launched')

    launcher.setup(args)

    print('Launcher setup')

    agent = launcher.setup_agent(0)
    agent.main_setup()

    print('Agent setup')

    learner = launcher.setup_learner()
    learner.main_setup()

    print('Learner setup')

    for i in range(2):
        print('Iteration {}'.format(i))
        for j in range(2):
            agent.main_loop()
        learner.main_loop()

    for subprocess_ in subprocesses:
        print(psutil.Process(subprocess_.pid).status())
        # assert(psutil.Process(subprocess_.pid).status() == 'running')

    parent = psutil.Process()
    for child in parent.children(recursive=True):
        child.terminate()

    print('Finished testing.')
