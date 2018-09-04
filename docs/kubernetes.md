# Using surreal with Kubernetes
Kubernetes is recommended if you want to use surreal on complex environment with many (possibly GPU accelerated) agents/leaners. This guide shows you how to setup an experiment on Kubernetes. There are several levels of customization that can be achieved:

## Basic Usage
For running surreal's algorithms (DDPG \ PPO) on already supported environments. You are in this mode if:
* You don't have custom agent / learner logic
* You don't want to specify agent / learner config beyond what is currently customizable in `ddpg_configs.py` or `ppo_configs.py`
* You are creating an environment already registered in `make_env`.
TODO: link to *_config files

In other words:
* You don't need to build a custom docker image
* You don't need to change the scheduling logic in the provided scripts

In this case, you can run surreal on a google cloud kubernetes engine cluster using existing public images. Here are the steps:
* Follow the instructions here (TODO!!) to setup the cluster. You will obtain `<cluster_name>.tf.json`
* Install surreal
* Create `~/.surreal.yml` or a yml file at `SURREAL_CONFIG_PATH`.
* Setup necessary information in `~/.surreal.yml`, see guide here (TODO!!)
* Run `kurreal create ...` (TODO: more details)

## Basic Development
For running (possibly your own) agent / learner on (possibly your own) environments. You are in this mode if:
* You have your own agent / learner class (or altered version of surreal DDPG/PPO) or your own environment
* You are still using the agent / learner / replay system

In other words:
* You need to build a custom docker image
* You don't need to change the scheduling logic in the provided scripts

In this case, you can run surreal on a google cloud kubernetes engine cluster using your own images. Here are the steps
* Follow the instructions here (TODO!!) to setup the cluster. You will obtain `<cluster_name>.tf.json`
* 
* Install surreal
* Create `~/.surreal.yml` or a yml file at `SURREAL_CONFIG_PATH`.
* Setup necessary information in `.surreal.yml`, see guide here (TODO!! different guide)
* 
* Install docker
* Pull the surreal base docker image 
* Setup your `main.py` file
* Setup docker build process
* Configure kurreal to push your built image and refer to them in `.surreal.yml`
* 
* Run `kurreal create ...` (TODO: more details)


## Custom Development
For running your own algorithm on custom scheduling logic, using some of surreal's toolchains and components. You are in this mdoe if:
* You want to add a component to the distributed system
* You want to change the node scheduling of existing / new components

In other words:
* You need to build a custom docker image
* You need to change the scheduling logic in the provided scripts

Similar to Basic Development, but you may want to use `SymphonyParser` directly, see `symphony` documentation (TODO: link)
