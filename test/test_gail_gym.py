import os
import psutil
from surreal.main.gail_configs import GAILLauncher
from surreal.test_helpers import integration_test


if __name__ == '__main__':
    print('BEGIN GAIL-Gym TEST')
    integration_test('/tmp/surreal/gail',
                     os.path.join(os.path.dirname(__file__),
                                  '../surreal/main/gail_configs.py'),
                     GAILLauncher(),
                     'gym:HalfCheetah-v2',
                     demo_path='../../surreal_demos',)
    print('PASSED')
    self = psutil.Process()
    self.kill()
