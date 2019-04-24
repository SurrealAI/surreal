import os
from setuptools import setup, find_packages


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


setup(
    name='Surreal',
    version='0.2.1',
    author='Stanford Vision and Learning Lab',
    url='http://github.com/StanfordVL/Surreal',
    description='Stanford University Repository for Reinforcement Algorithms',
    long_description=read('README.rst'),
    keywords=['Reinforcement Learning',
              'Deep Learning',
              'Distributed Computing'],
    license='GPLv3',
    packages=[
        package for package in find_packages() if package.startswith("surreal")
    ],
    entry_points={
        'console_scripts': [
            'surreal-kube=surreal.kube.surreal_kube:main',
            'surreal-tmux=surreal.tmux.surreal_tmux:main',
            'surreal-subproc=surreal.subproc.surreal_subproc:main',
            'surreal-ddpg=surreal.main.ddpg_configs:main',
            'surreal-ppo=surreal.main.ppo_configs:main',
            'surreal-default-config=surreal.main.generate_default_config:main',
        ]
    },
    install_requires=[
        "gym",
        "robosuite",
        "mujoco-py<1.50.2,>=1.50.1",
        "tabulate",
        "tensorflow",
        "tensorboardX",
        "imageio",
        "pygame",
        "benedict",
        "nanolog",
        "psutil",
        "tabulate",
        "imageio",
        "caraml>=0.10.0",
        "symphony>=0.9.1",
        "torchx==0.9",
        "tensorplex",
        "cloudwise>=0.1.1",
        "Cython<0.29",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3"
    ],
    include_package_data=True,
    zip_safe=False
)
