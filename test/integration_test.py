import os
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
    # _SYMPHONY_PARAMETER_SERVER_HOST = '127.0.0.1'
    # _SYMPHONY_PARAMETER_SERVER_PORT = 7001
    # Refer to launch logs in ~/kurreal


def test_ddpg(tmpdir):
    print("Making temp directory...")
    temp_path = os.path.join(tmpdir, "test_ddpg")
    os.makedirs(temp_path, exist_ok=True)
    print("Setting up experiment launcher...")
    launcher = DDPGLauncher()
    args = ['--num-agents',
            '1',
            '--env',
            'dm_control:cartpole-balance',
            '--experiment-folder',
            str(temp_path)]
    launcher.setup(args)

    print("Setting up environment variables...")
    _setup_env()

    print("Running...")

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
