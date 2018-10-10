# Creation Settings
Creating a surreal experiment in kurreal involves a lot of settings, e.g. the number of agents, the number of evals, and where to schedule each pod. Creation setting helps declare and configure these settings in an intuitive way.

## Specifying a setting
When you are creating an experiment, you write
```bash
kurreal create <experiment_name> <settings_name> [settings_override_args] -- config_args
```
`settings` name help `kurreal` locate a sub-dictionary in `creation_settings` field of `.surreal.yml`. This dictionary decides the behavior of create.
```yaml
creation_settings:
    <settings_name>:
        mode: <mode>
        [settings_key]: [settings_val] 
        ...
```
The most important filed is `mode`. `mode` decides what kind of experiment you are creating, e.g. whether ES or RL should be run. For each `mode`, there are specific config kyes and vals to specify specific behaviors, e.g. how many agents to use in RL, how to schedule pods, and what environment to use. See [below](#basic).

## Customizing settings
For each mode, the settings are decided in the following way:
* The default setting for the mode is loaded
* Settings specified in `.surreal.yml` overrides default settings
* Key-val pairs specified by commandline. See mode specific documentation to see what are supported

# Creation modes
Currentonly there is only basic mode.

## Basic
This is the basic agent / learner / replay setup of surreal. Default settings are listed as follows
```yaml
creation_settings:
    <settings_name>:
        mode: basic
        # Number of agent pods
        num_agent: 2
        # Number of eval pods
        num_eval: 1
        # Number of agent processes per pod
        agent_batch: 1
        # Number of eval processes per pod
        eval_batch: 1
        # RL algorithm to use (ddpg / ppo) 
        # or path to a .py excecutable file in the container
        # The .py excecutable should at least support the interface of surreal/main/ddpg.py and surreal/main/ppo.py
        algorithm: ppo
        # Environment to use, see TODO for details
        env: gym:HalfCheetah-v2
        agent:
            # The docker image to use for agent and eval
            image: <agent_image>
            # When build_image is not None and has value <build_settings_name>, build docker image according to build settings and push to <agent_image>:experiment_name
            build_image: None
            # See TODO
            scheduling: {<symphony scheduling kwargs>}
        nonagent:
            # The docker image to use for nonagent (learner + replay + ps + logging etc.)
            image: <nonagent_image>
            # When build_image is not None and has value <build_settings_name>, build docker image according to build settings and push to <nonagent_image>:experiment_name
            build_image: None
            # See TODO
            scheduling: {<symphony scheduling kwargs>}
```

When creating an experiment using `kurreal`, the following commandline arguments are allowed:
```bash
--num_agent 2
--num_eval 1
--algorithm ppo
--agent_batch: 8
--eval_batch: 8
--env: dm_control:cartpole-balance
```
TODO: symphony_scheduling kwargs
TODO: .surreal.yml file