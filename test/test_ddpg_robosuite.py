import os
import psutil
from surreal.main.ddpg_configs import DDPGLauncher
from surreal.test_helpers import integration_test


if __name__ == '__main__':
    print('BEGIN DDPG-Gym TEST')
    integration_test('/tmp/surreal/ddpg',
                     os.path.join(os.path.dirname(__file__),
                                  '../surreal/main/ddpg_configs.py'),
                     DDPGLauncher(),
                     'robosuite:SawyerLift')
    print('PASSED')
    self = psutil.Process()
    self.kill()
