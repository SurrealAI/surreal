# Creation Settings
The creasion_settings field in `.surreal.yml` configures the settings for `kurreal create`. 
```
creation_settings:
    <setting_name>:
        mode: <mode>
        [config_key]: [config_val] 
        ...
```

For each `mode`, there are specific config kyes and vals to be provided

# Basic
This is the basic agent / learner / replay setup of surreal
```
creation_settings:
    <setting_name>:
        mode: basic
        num_agent: 2
        num_eval: 1
        algorithm: ppo
        agent:
            image: <agent_image>
            node_pool:
            build_image:
        nonagent:
            image: <nonagent_image>
            node_pool:
            build_image:
        ...
```
supported commandline arguments
```
--num_agent 2
--num_eval 1
--algorithm
```

# Batched
This is similar to basic, but multiple agent / eval are run on a single machine. This is how you could share a GPU with 5 agents
```
creation_settings:
    <setting_name>:
        mode: batched
        num_agent: 8
        agent_batch_size: 8
        num_eval: 8
        eval_batch_size: 8
        algorithm: 
        agent:
            image: <agent_image>
            node_pool:
            build_image:
        nonagent:
            image: <nonagent_image>
            node_pool:
            build_image:
        ...
```
Supported commandline arguments
```
--num_agent 16
--agent_batch 8
--num_eval 8
--eval_batch 8
--algorithm
```