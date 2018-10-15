# Kurreal user guide
`kurreal` stands for Surreal on Kubernetes. It is a commandline for launching and managing Surreal experiments on Kubernetes, powered by [Symphony](https://github.com/SurrealAI/symphony). Check symphony docs to see how to creat a commandline like `kurreal`.

(Configs)[#configs]  
(Create Experiments)[#create-experiments]  
(Monitor Experiments)[#monitor-experiments]  
(Manage Multiple Experiments)[#manage-multiple-experiments]  
(Delete Experiments)[#delete-experiments]  
(Miscellaneous)[#miscellaneous]  

---

## Configs
Using kurreal requires you to configure `.surreal.yml` correctly. Here is a template that we will be using. See [documentation](yaml_config.md) for details.

## Create Experiments
If you followed the guide in [installation](installation.md), you can launch an experiment with 
```bash
kurreal create cpu-experiment my-first-experiment
```
Here `cpu-experiment` refers to the `creation_settings: cpu-experiment:`  in `.surreal.yml`. `my-first-experiment` is the name of the experiment. For details of the creation settings in each mode, see [documentation](creation_settings.md). 

There are several ways to customize `kurrea create` (or `kurreal c`).

* Duplicate experiment names are by default blocked to prevent overwriting existing data. Override this behavior by using `-f`
```bash
kurreal create -f cpu-experiment my-first-experiment
> Creating...
kurreal create cpu-experiment my-first-experiment
> Error
kurreal create -f cpu-experiment my-first-experiment
> Creating...
```

* You can override specific creation setting fields.
```bash
kurreal create cpu-experiment my-first-experiment --algorithm ddpg --num-agents 100
```
See [documentation](creation_settings.md) for details.

* You can provide algorithm specific configs after `--`. This is especially useful when you are using your own entrypoint.
```bash
kurreal create cpu-experiment my-first-experiment --algorithm my_ppo -- --use-alternative-loss
```
TODO: add documentation about algorithm specific settings

## Monitor Experiments
After creating the experiment, you can use `kurreal list-processes` (or `kurreal lsp` or `kurreal p`) to list all running processes in the current experiment.
```bash
kurreal p
# TODO: Add output
```

You can always use `kubectl` to supplement `kurreal`. For example, to check autoscaling related issues, you can do
```bash
kubectl describe pod agent-0
```

To see logs of an experiment, use `kurreal logs`. 
```bash
kurreal logs learner
> DDPG learner starting
```

Surreal experiments expose tensorboard. So you can open tensorboard using `kurreal tensorboard` (or `kurreal tb`).
```bash
kurreal tb
> Opening browser to ...:6006
```

## Manage Multiple Experiments
Each experiment is a kubernetes namespace. You can switch between experiments. Use `kurreal switch-experiment` (or `kurreal se`). You can find out what are the running experiments by `kurreal list-experiment` (or `kurreal lse`). 
```bash
kurreal se
> [Your current experiment name]
kurreal lse
> jim-cartpole-1
> jim-cheetah-1
> jim-stacking-1
kurreal se jim-stacking-1
kurreal p
> processes in jim-stacking-1
```

## Delete Experiments
To delete an experiment, you can use `kurreal delete` (or `kurreal d`). There is also `kurreal delete-batch` (or `kurreal db`).
```
kurreal delete
> Deleting current experiment
kurreal d jim-cartpole
> Deleting jim's experiment cartpole
kurreal db jim
> Deleting all of jim's experiments
```

## Miscellaneous 
There are some helper functions in `kurreal` as well.

If you are developing using surreal, you may have a log of docker images built. Use `kurreal docker-clean` to clean them up.
```bash
kurreal dc
> Cleaning docker images
```

TODO: add nfs support related, `get-video`, `get-experiment`, etc.


