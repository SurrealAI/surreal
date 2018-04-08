# Onboarding installation guide for new Surreal Team members
This guide helps you setup surreal and run experiments, both locally and remotely. This guide should work on Mac or a linux machine. If you meet problems, you can ask Jim (jimfanspire@gmail.com) or Jiren (jirenz@stanford.edu).

## Setup python env
* Depending on your preference, setup a python 3 (3.5/3.6) environment, say named `<surreal_venv>`
* We will assume that `<surreal_venv>` is activated in this guide
* It is highly recommended that you use conda
* Our own tradition is `<surreal_venv> = surreal`

## Surreal and Tensorplex 
We will install necessary libraries so we can start an experiment locally.
* Clone these repos [Surreal](https://github.com/StanfordVL/Surreal) and [Tensorplex](https://github.com/StanfordVL/Tensorplex). We will refer to their path as `<surreal_path>` and `<tensorplex_path>`. We will need these paths later
* Install some dependencies. Go to `<surreal_path>`, run
```bash
pip install -e .
pip install -r container/requirements.txt
```
* Install pytorch: our computation backend. 
```bash
conda install pytorch torchvision -c pytorch # or refer to http://pytorch.org
```
* Install [dm_control](https://github.com/deepmind/dm_control), a set of benchmarking enviroments. Besides running the following command, check [dm_control's](https://github.com/deepmind/dm_control) readme.md 
```bash
pip install git+git://github.com/deepmind/dm_control.git
```
* Install the tensorplex library (later we will make it pip install-able). Go to `<tensorplex_path>`, run
```bash
pip install -e .
```
* [Mujoco](http://www.mujoco.org) is a physical simulator. We need to set it up for use
```bash
mkdir ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mjpro150_linux.zip
unzip mjpro150_linux.zip
rm mjpro150_linux.zip
wget https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
rm mjpro131_linux.zip
```
* Also, you need to put the liscense file `mjkey.txt` into `~/.mujoco`. Ask Jim or Jiren (or anyone you know working on surreal) if you don't have it.

## Jupyter
To run an experiment locally, we use jupyter notebook to manage the processes. Here is a short guide for setup.
* If you don't have jupyter-notebook, install jupyter-notebook.
```bash
pip install jupyter
```
* Register ipython kernel for `<surreal_venv>`. So that we can run experiments locally. Refer to [this doc](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for registering ipython kernel
```bash
pip install ipykernel
python -m ipykernel install --user --name other-env 
```

## Docker - Kubernetes - Google Cloud
To run an experiment remotely, we deploy the framework into docker containers. Here are the setup guides
* Ask Yuke to add you to the surreal-dev google cloud project.
* (Skip this unless you are adding new dependencies to the codebase, which should be rare) Install [docker](https://www.docker.com)
* Install [google cloud commandline tools](https://cloud.google.com/sdk/). 
* To talk to kubernetes, we have a kurreal wrapper that orchestrates experiments. Copy `<surreal_path>/surreal/kube/sample.surreal.yml` to `~/.surreal.yml`. Update your `~/.surreal.yml` following the comments in the file.
* Install `kubectl`. This is the commandline tool to talk to a kubernetes cluster. You can use [the official guide](https://kubernetes.io/docs/tasks/tools/install-kubectl/) or `gcloud components install kubectl`
* Now we need to configure google cloud config to our project/cluster/zone. Do the following
```bash
gcloud config set project surreal-dev-188523
gcloud config set container/cluster kurreal-1
gcloud config set compute/zone us-west1-b
gcloud container clusters get-credentials kurreal-1
```

# Run an experiment on the cloud
Here we are going to create a default experiment (using DDPG by default) on dm_control cheetah with 4 agents using k80 GPU:  
* !! Important: kurreal's git snapshot functionality is non-atomic  
* !! when kurreal is doing git operatoins for you, don't ctrl-C 
```bash
kurreal create-dev [experiment-name] 4 --gpu
```
* If everything runs fine, you will see an experiment running. We can inspect the actor/leaners (named pods in kubernetes's terminology) by using (p is for pods)
```bash
kurreal p
```
* Use the following command to open the tensorboard for an experiment
```bash
kurreal tb
```
* Each experiment is under a specific namespace. 
```bash
kurreal ns
```
tells you what is your current namespace
```bash
kurreal ls
```
lists all namespaces, and 
```bash
kurreal ns [namespace name]
```
swtiches your current namespace
* Use the following command to delete an experiment
```bash
kurreal delete
```






