# Install and Run SURREAL Locally
This guide will allow you to setup Surreal on your local machine. Multiple processes are launched by the `surreal-subproc` commandline interface.

[Requirements](#requirements)  
[Install Surreal](#install-surreal)  
[Create and run an experiment](#Create-and-run-an-experiment)  
[Develop Algorithms Locally](#Develop-Algorithms-Locally)

## Requirements
* This guide is written and tested primarily on Mac OS X and Ubuntu 16.04. If you run into issues when installing, you can check [our docker file](docker/Dockerfile-nvidia) (adapted from the image for [mujoco_py](https://github.com/openai/mujoco-py)). For linux users, the dependencies that we needed on top of a nvidia image is listed as follows.
```bash
sudo apt-get update
sudo apt-get install \
    curl \
    git \
    cmake \
    unzip \
    bzip2 \
    wget \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
```

* Mujoco is the physics simulator for our environments. First download MuJoCo 1.5.0 ([Linux](https://www.roboti.us/download/mjpro150_linux.zip) and [Mac OS X](https://www.roboti.us/download/mjpro150_osx.zip)) and place the `mjpro150` folder and your license key `mjkey.txt` in `~/.mujoco`. You can obtain a license key from [here](https://www.roboti.us/license.html).
   - For Linux, you will need to install some packages to build `mujoco-py` (sourced from [here](https://github.com/openai/mujoco-py/blob/master/Dockerfile), with a couple missing packages added). If using `apt`, the required installation command is:
     ```sh
     $ sudo apt install libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
             libosmesa6-dev software-properties-common net-tools wget \
             xpra xserver-xorg-dev libglfw3-dev patchelf

## Install Surreal
1. **(Optional) Setup python environment.** First, create a python environment. Surreal is developed on python 3.5+. We recommend using [conda](https://conda.io/docs/user-guide/install/index.html) or a [virtual environment](https://virtualenv.pypa.io/en/stable/) for the dependencies of Surreal. For example, we usually use conda.
```bash
conda create -n surreal python>=3.5
source activate surreal
```

2. **Install pytorch.** Install [pytorch](https://pytorch.org/get-started/locally/) following the official guide. Optionally, setup cuda if you have GPUs enabled.

3. **Install Surreal.** Installing surreal through pip would install the surreal library and all its dependencies.
```bash
pip install surreal
```

4. **Configure Surreal.** Setup `.surreal.yml`. Run the following command to setup the surreal config file at `~/.surreal.yml`. 
```bash
surreal-default-config
```

This will generate a config file at `~/.surreal.yml`. Optionally, you can put it in another location and specify `SURREAL_CONFIG_PATH`. 

Experiments are automatically prepended with your username. Specify it in the config. You can turn this behavior off by setting `prefix_experiment_with_username = False`.

```yaml
username: <your_username>
```

Every time an experiment is created using `surreal-subproc`, result data will be written to `<subproc_results_folder>/<experiment_name>`. `subproc_results_folder` is specified in the config. (e.g. You can put all your experiment results in `~/surreal/subproc/`)

```yaml
subproc_results_folder: <put path here> # e.g. ~/surreal/subproc/
```

If you want to know more about the config and other fields, refer to [this guide](yaml_config.md). For now, we have what we need to setup local experiments.


## Launch an Experiment
You are now ready to launch an experiment. Run

```bash
surreal-subproc <experiment_name> --num-agents 4
```

If you have your own virtualenv, you can activate it before running subproc:

```bash
source activate mypythonenv
surreal-subproc <experiment_name> --num-agents 4
```

If you setup your `.surreal.yml` as default, you need to make sure the `subproc_results_folder` field is properly set. You will see experiment outputs in `~/<your_subproc_folder>/experiment_name` and see tensorboard output at `localhost:6006`. You can choose one of the two pre-installed surreal algorithms by using the `--algorithm` flag.

```bash
surreal-subproc --algorithm ppo <experiment_name> # Runs Surreal-PPO
surreal-subproc --algorithm ddpg <experiment_name> # Runs Surreal-DDPG
```

**Consuming GPU.** `surreal-subproc` respects `CUDA_VISIBLE_DEVICES` for using GPU acceleration for rendering and neural network acceleration. You can control what GPUs are used with
```bash
CUDA_VISIBLE_DEVICES=0,1 surreal-subproc <experiment_name> --num-agents 4
```

When there is only one GPU present, the launcher will assign agents, evals and learner to this GPU. When there is more than one. The launcher will assign learner to one GPU and evenly distribute agents and evals to the remaining GPUs.

Use Ctrl-C to stop a running distributed experiment. `surreal-subproc` has mechanism to capture the interruption signal and terminate all child processes gracefully.
