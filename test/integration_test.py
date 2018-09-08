import os
import sys
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
    # _SYMPHONY_PARAMETER_SERVER_HOST = '127.0.0.1'
    # _SYMPHONY_PARAMETER_SERVER_PORT = 7001
    # Refer to launch logs in ~/kurreal


def test_ddpg(tmpdir):
    print("Making temp directory...")
    temp_path = os.path.join(tmpdir, "test_ddpg")
    os.makedirs(temp_path, exist_ok=True)
    print("Setting up experiment launcher...")
    launcher = DDPGLauncher()
    args = [
            '--num-agents',
            '1',
            '--env',
            'dm_control:cartpole-balance',
            '--experiment-folder',
            str(temp_path)]

    print("Setting up environment variables...")
    _setup_env()

    subprocess.check_call([sys.executable, '../surreal/main/ddpg_configs.py', 'agent-0', '--'] + args)
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
    test_ddpg("/tmp/surreal")
