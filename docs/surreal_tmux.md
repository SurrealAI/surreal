# Install and Run SURREAL Locally
This guide will allow you to setup Surreal on your local machine. Multiple processes are managed through tmux using the `surreal-tmux` commandline interface.  
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
     ```
     Note that for older versions of Ubuntu (e.g., 14.04) there's no libglfw3 package, in which case you need to `export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin` before proceeding to the next step.

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

When running experiments locally, you may want to setup environment variables (e.g. activate a virtual python environment) before running the python script. You can do this by specifying one command in each line of 
`tmux_preamble_cmds`.

```yaml
tmux_preamble_cmds:
  - 'source activate surreal'
```

Every time an experiment is created on tmux, result data will be written to `<tmux_results_folder>/<experiment_name>`. `tmux_results_folder` is specified in the config. (e.g. You can put all your experiment results in `~/surreal/tmux/`)

```yaml
tmux_results_folder: <put path here> # ~/surreal/tmux/
```

If you want to know more about the config and other fields, refer to [this guide](yaml_config.md). For now, we have what we need to setup local experiments.

5. **Install Tmux.** We build upon tmux to manage experiments locally. You need to install tmux if you don't already have it.  
On Mac
```bash
brew install tmux
```
On linux
```bash
sudo apt-get install tmux
```

## Create and run an Experiment
You are now ready to create an experiment. Run
```bash
surreal-tmux create <experiment_name>
```
If you setup your `.surreal.yml` as default (you need fiels `tmux_preamble_cmds` and `tmux_results_folder` to be properly set), you will see experiment outputs in `~/surreal-tmux/experiment_name` and see tensorboard output at `localhost:6006`. You can choose one of the two pre-installed surreal algorithms by using the `--algorithm` flag.
```bash
surreal-tmux create --algorithm ppo <experiment_name> # Runs Surreal-PPO
surreal-tmux create --algorithm ddpg <experiment_name> # Runs Surreal-DDPG
```

**Consuming GPU.** `surreal-tmux create ...` respects `CUDA_VISIBLE_DEVICES` for using GPU acceleration for rendering and neural network acceleration. You can control what GPUs are used with
```bash
export CUDA_VISIBLE_DEVICES=0,1
surreal-tmux create ...
```
When there is only one GPU present, the launcher will assign agents, evals and learner to this GPU. When there is more than one. The launcher will assign learner to one GPU and evenly distribute agents and evals to the remaining GPUs.

Use `surreal-tmux p` (`p` is a short hand for `list-processes`) to check the status of each process.
```bash
surreal-tmux p
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

Use `surreal-tmux logs` to inspect logs of different components.
```bash
surreal-tmux logs learner
> ...
```

You can use `surreal-tmux delete` to terminate the experiment.
```bash
surreal-tmux delete
> Terminating
```
