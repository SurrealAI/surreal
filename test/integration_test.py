import os
import signal
import sys
import psutil
import subprocess
from surreal.main.ddpg_configs import DDPGLauncher
from surreal.main.ppo_configs import PPOLauncher


# Currently planned tests
# DDPG dm_control
# DDPG mujocomanip
# PPO dm_control
# PPO mujocomanip


def _setup_env():
    """
    Setup the necessary environment variables
    """
    # TODO: set environment vairables like:
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
    # _SYMPHONY_PARAMETER_SERVER_HOST = '127.0.0.1'
    # _SYMPHONY_PARAMETER_SERVER_PORT = 7001
    # Refer to launch logs in ~/kurreal


def test_ddpg(tmpdir):
    print("Making temp directory...")
    temp_path = os.path.join(tmpdir, "test_ddpg")
    os.makedirs(temp_path, exist_ok=True)
    print("Setting up experiment launcher...")
    args = [
            '--num-agents',
            '1',
            '--env',
            'dm_control:cartpole-balance',
            '--experiment-folder',
            str(temp_path)]

    print("Setting up environment variables...")
    _setup_env()

    subprocesses = []

    for module in ['eval-0', 'replay', 'ps', 'tensorboard']:
        subprocesses.append(subprocess.Popen([sys.executable, '../surreal/main/ddpg_configs.py', module, '--'] + args))

    agent_launcher = DDPGLauncher()
    agent_launcher.setup(args)
    agent_launcher.run_agent(0, iterations=10)

    learner_launcher = DDPGLauncher()
    learner_launcher.setup(args)
    learner_launcher.run_learner(iterations=1)


    for subprocess_ in subprocesses:
        parent = psutil.Process(subprocess_.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

    """
    launcher.setup(args)


    print("Running...")

    launcher.launch('agent-0')
    launcher.launch('eval-0')
    launcher.launch('replay')
    launcher.launch('learner')
    launcher.launch('ps')
    launcher.launch('tensorboard')
    """

    """
    Fork out all the necessary processes
    """

    # Run agent for several iterations
    # Run learner for several iterations

    # Tell every process to shut down

    # exit

def test_ppo(tmpdir):
    temp_path = tmpdir.mkdir("test_ddpg")
    launcher = PPOLauncher()
    # TODO: same as DDPG

if __name__ == '__main__':
    print('BEGIN DDPG TEST')
    test_ddpg("/tmp/surreal")
    print('PASSED')
    self = psutil.Process()
    self.kill()
