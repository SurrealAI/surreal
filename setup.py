import os
from setuptools import setup, find_packages


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


setup(
    name='Surreal',
    version='0.0.1',
    author='Stanford Vision and Learning Lab',
    url='http://github.com/StanfordVL/Surreal',
    description='Stanford University Repository for Reinforcement Algorithms',
    # long_description=read('README.rst'),
    keywords=['Reinforcement Learning',
              'Deep Learning',
              'Distributed Computing'],
    license='GPLv3',
    packages=[
        package for package in find_packages() if package.startswith("surreal")
    ],
    entry_points={
        'console_scripts': [
            'kurreal=surreal.kube.kurreal:main',
            'turreal=surreal.tmux.turreal:main',
            'surreal-ddpg=surreal.main.ddpg_configs:main',
            'surreal-ppo=surreal.main.ppo_configs:main',
            'surreal-default-config=surreal.main.generate_default_config:main',
        ]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3"
    ],
    include_package_data=True,
    zip_safe=False
)
