<img src=".README_images/surreal-logo.png" width=15% align="right" />

# **[SURREAL](https://surreal.stanford.edu)**
## Open-Source Distributed Reinforcement Learning Framework

_Stanford Vision and Learning Lab_

[SURREAL](https://surreal.stanford.edu) is a fully integrated framework that runs state-of-the-art distributed reinforcement learning (RL) algorithms.


<div align="center">
<img src=".README_images/iconic-features.png" width=40% />
</div>


- **Scalability**. RL algorithms are data hungry by nature. Even the simplest Atari games, like Breakout, typically requires up to a billion frames to learn a good solution. To accelerate training significantly, SURREAL parallelizes the environment simulation and learning. The system can easily scale to thousands of CPUs and hundreds of GPUs.


- **Flexibility**. SURREAL unifies distributed on-policy and off-policy learning into a single algorithmic formulation. The key is to separate experience generation from learning. Parallel actors generate massive amount of experience data, while a _single, centralized_ learner performs model updates. Each actor interacts with the environment independently, which allows them to diversify the exploration for hard long-horizon robotic tasks. They send the experiences to a centralized buffer, which can be instantiated as a FIFO queue for on-policy mode and replay memory for off-policy mode. 

<!--<img src=".README_images/distributed.png" alt="drawing" width="500" />-->

- **Reproducibility**. RL algorithms are notoriously hard to reproduce \[Henderson et al., 2017\], due to multiple sources of variations like algorithm implementation details, library dependencies, and hardware types. We address this by providing an _end-to-end integrated pipeline_ that replicates our full cluster hardware and software runtime setup.

<!--<img src=".README_images/pipeline.png" alt="drawing" height="250" />-->

## Quick Start

### Install and Run SURREAL Locally
* Create a python environment. Surreal runs on python 3, we recommend using conda or a virtual environment. For example:
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n surreal python>=3.5
source activate surreal
```

* If you run into issues when installing on Ubuntu, you can check [our docker file](docker/Dockerfile-nvidia).

* Install [pytorch](https://pytorch.org/get-started/locally/).

* Install surreal
(WIP) Surreal is not released yet, see below for instructions
TODO: allow `pip install surreal to work`
```bash
pip install surreal
```

* (Temporary): Install Robotics Suite and Symphony from master branch. Install surreal by `pip install -e .` on branch `refactor4release`.

* Setup `.surreal.yml`. Run the following command to setup the surreal config file at `~/.surreal.yml`.
```bash
surreal-default-config
```
Then follow [this guide](yaml_config.md) to setup the config file.

* Install environments
You can install RL environments depending on your need. The supported environments are [gym](https://github.com/openai/gym), [DeepMind Control Suite](https://github.com/deepmind/dm_control), and [Surreal Robotics Suite](https://github.com/StanfordVL/MujocoManipulation/tree/refactor4release).
  - TODO(optional): step by step installation commands

* Install Tmux
We use tmux to manage experiments locally. You need to install it if you don't already have it.
On Mac
```bash
brew install tmux
```
On linux
```bash
sudo apt-get install tmux
```

* Create an experiment
You are now ready to create an experiment. Run
```bash
turreal create <experiment_name>
```
If you setup your `.surreal.yml` as default (you need fiels `tmux_preamble_cmds` and `tmux_results_folder` to be properly set), you will see experiment outputs in `~/turreal/experiment_name` and see tensorboard output at `localhost:6006`. You can choose one of the two pre-installed surreal algorithms by using the `--algorithm` flag.
```bash
turreal create --algorithm ppo <experiment_name> # Runs Surreal-PPO
turreal create --algorithm ddpg <experiment_name> # Runs Surreal-DDPG
```
If you have a GPU and installed pytorch with GPU compatibility, you can use
```
turreal create ... -- --num-gpus 1
```
to use a GPU for learner training.

Use `turreal p` to check the status of each process.
```bash
turreal p
>Group  Name         status
>       tensorplex   live
>       learner      live
>       agent-0      live
>       loggerplex   live
>       agent-1      live
>       tensorboard  live
>       replay       live
>       eval-0       live
>       ps           live
```

Use `turreal logs` to inspect logs of different components.
```bash
turreal logs learner
> ...
```

You can use `turreal delete` to terminate the experiment.
```bash
turreal delete
> Terminating
```
TODO (optional): more documentation on turreal

### Develop Algorithms Locally
If you want to develop algorithms locally using Surreal. You should create a `my_algorithm.py` file. Run
```bash
turreal create --algorithm <path to my_algorithm.py>
```
to launch your own algorithm. You can refer to [ddpg_configs.py](../) or [ppo_configs.py](../surreal/main/ppo_configs.py) for reference.


### Deploy on the cloud

For how to deploy on the Kubernetes engine with Google Cloud, please refer to the [detailed installation guide](docs/installation.md)

## Benchmarking

- Scalability of Surreal-PPO with up to 1024 actors on Surreal Robotics Suite.

![](.README_images/scalability-robotics.png)

- Training curves of 16 actors on OpenAI Gym tasks for 3 hours, compared to other baselines. 

<img src=".README_images/performance-gym.png" width=60% />


## Citations
Please cite our CORL paper if you use this repository in your publications:

```
@inproceedings{corl2018surreal,
  title={SURREAL: Open-Source Reinforcement Learning Framework and Robot Manipulation Benchmark},
  author={Fan, Linxi and Zhu, Yuke and Zhu, Jiren and Liu, Zihua and Zeng, Orien and Gupta, Anchit and Creus-Costa, Joan and Savarese, Silvio and Fei-Fei, Li},
  booktitle={Conference on Robot Learning},
  year={2018}
}
```


