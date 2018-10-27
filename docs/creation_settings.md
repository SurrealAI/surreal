# Creation Settings
Creating a surreal experiment in surreal-kube involves a lot of settings, e.g. the number of agents, the number of evals, and where to schedule each pod. Creation setting helps declare and configure these settings in an intuitive way.

## Specifying a setting
When you are creating an experiment, you write
```bash
surreal-kube create <experiment_name> <settings_name> [settings_override_args] -- config_args
```
`settings` name help `surreal-kube` locate a sub-dictionary in `creation_settings` field of `.surreal.yml`. This dictionary decides the behavior of create.
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
        # RL algorithm to use (ddpg / ppo) 
        # or path to a .py excecutable file in the container
        # The .py excecutable should at least support the interface of surreal/main/ddpg.py and surreal/main/ppo.py
        algorithm: ppo
        # Number of agent containers
        num_agents: 2
        # Number of eval pods
        num_evals: 1
        # Number of agent processes per pod
        agent_batch: 1
        # Number of eval processes per pod
        eval_batch: 1
        # Environment to use
        env: gym:HalfCheetah-v2
        agent:
            # The docker image to use for agent and eval
            image: <agent_image>
            # When build_image is not None and has value <build_settings_name>, build docker image according to build settings and push to <agent_image>:experiment_name
            build_image: None
            # schedulign specifications
            scheduling: {<symphony scheduling kwargs>}
        nonagent:
            # The docker image to use for nonagent (learner + replay + ps + logging etc.)
            image: <nonagent_image>
            # When build_image is not None and has value <build_settings_name>, build docker image according to build settings and push to <nonagent_image>:experiment_name
            build_image: None
            # scheduling specifications
            scheduling: {<symphony scheduling kwargs>}
```
* `algorithm` is the RL algorithm to use. It can be `ddpg` or `ppo`, in which case our `surreal-kube` commandline knows where the executables are located. If you wrote your own algorithm (follow the example of [ddpg](../surreal/main/ddpg_configs.py) and [ppo](../surreal/main/ppo_configs.py) to do so), provide the path to `<you_algorithm>.py` so our launcher can properly provide arguments to the containers' entrypoint (`python -u` will be prepended before your provided path if it ends with `.py`, otherwise, we assume that you are providing an executable and will call it directly).
* `num_agents`, `num_evals`, `agent_batch`, and `eval_batch` controls how many agent / evals there are in an experiment. The total number of agents is computed by `num_agents x agent_batch`. The total number of evaluators is computed by `num_evals x eval_batch`. Agents in the same batch are launched in the same container. This setup allows us to run 16 agents on a single GPU, maximizing resource usage. 
* `env` is the name of the enrivonment to run our algorithms on. 
* `agent` / `nonagent`: specifies deployment related information on the cluster. `agent` is defined for each agent container (`agent_batch` processes). `nonagent` is defined for the `nonagent` container, which includes learner, replay, parameter server, tensorplex, loggerplex and tensorboard.
    - `image` and `build_image` define the container image to run on the cluster. You can provide `image:repo:tag, build_image: null` to pull from an existing image. Or you can provide `image:repo, build_image: <image-build-setting-name>`. When you run an experiment that requires this image, this will trigger an image build , push it to `repo:<experiment-name>` and use it for your experiment. For more about image build settings, see [documentation of Symphony docker builder](https://github.com/SurrealAI/symphony/blob/master/docs/docker_builder.md).
    - `scheduling` defines how much compute resource to allocate to the `agent`, resp. `nonagent`, container. See `symphony.GKEDispatcher.assign_to` ([documented here](https://github.com/SurrealAI/symphony/blob/master/docs/kubernetes.md#dispatcher)) for details. You need to provide all kwargs other than `process` and `process_group`.
* When creating an experiment using `surreal-kube`, the following commandline arguments are allowed:
```bash
--num_agents 2
--num_evals 1
--algorithm ppo
--agent_batch: 8
--eval_batch: 8
--env dm_control:cartpole-balance
```
