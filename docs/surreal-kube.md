# surreal-kube user guide
`surreal-kube` stands for Surreal on Kubernetes. It is a commandline for launching and managing Surreal experiments on Kubernetes, powered by [Symphony](https://github.com/SurrealAI/symphony). Check symphony docs to see how to creat a commandline like `surreal-kube`.

[Configs](#configs)  
[Create Experiments](#create-experiments)  
[Monitor Experiments](#monitor-experiments)  
[Manage Multiple Experiments](#manage-multiple-experiments)  
[Delete Experiments](#delete-experiments)  
[Miscellaneous](#miscellaneous)  

---

## Configs
Using surreal-kube requires you to configure `.surreal.yml` correctly. Here is a template that we will be using. See [documentation](yaml_config.md) for details.

## Create Experiments
If you followed the guide in [installation](installation.md), you can launch an experiment with 
```bash
surreal-kube create cpu-gym my-first-experiment
```
Here `cpu-gym` refers to the `creation_settings: cpu-gym:`  in `.surreal.yml`. `my-first-experiment` is the name of the experiment. For details of the creation settings in each mode, see [documentation](creation_settings.md). 

There are several ways to customize `surreal-kube create` (or `surreal-kube c`).

* Duplicate experiment names are by default blocked to prevent overwriting existing data. Override this behavior by using `-f`
```bash
surreal-kube create -f cpu-gym my-first-experiment
> Creating...
surreal-kube create cpu-gym my-first-experiment
> Error
surreal-kube create -f cpu-gym my-first-experiment
> Creating...
```

* You can override specific creation setting fields.
```bash
surreal-kube create cpu-gym my-first-experiment --algorithm ddpg --num-agents 100
```
See [documentation](creation_settings.md) for details.

* You can provide algorithm specific configs after `--`. This is especially useful when you are using your own entrypoint.
```bash
surreal-kube create cpu-gym my-first-experiment --algorithm my_ppo.py -- --use-alternative-loss
```

## Monitor Experiments
After creating the experiment, you can use `surreal-kube list-processes` (or `surreal-kube lsp` or `surreal-kube p`) to list all running processes in the current experiment.
```bash
surreal-kube p
```

You can always use `kubectl` to supplement `surreal-kube`. For example, to check autoscaling related issues, you can do
```bash
kubectl describe pod agent-0
```

To see logs of an experiment, use `surreal-kube logs`. 
```bash
surreal-kube logs learner
> DDPG learner starting
```

Surreal experiments expose tensorboard. So you can open tensorboard using `surreal-kube tensorboard` (or `surreal-kube tb`).
```bash
surreal-kube tb
> Opening browser to ...:6006
```

## Manage Multiple Experiments
Each experiment is a kubernetes namespace. You can switch between experiments. Use `surreal-kube switch-experiment` (or `surreal-kube se`). You can find out what are the running experiments by `surreal-kube list-experiment` (or `surreal-kube lse`). 
```bash
surreal-kube se
> [Your current experiment name]
surreal-kube lse
> jim-cartpole-1
> jim-cheetah-1
> jim-stacking-1
surreal-kube se jim-stacking-1
surreal-kube p
> processes in jim-stacking-1
```

## Delete Experiments
To delete an experiment, you can use `surreal-kube delete` (or `surreal-kube d`). There is also `surreal-kube delete-batch` (or `surreal-kube db`).
```
surreal-kube delete
> Deleting current experiment
surreal-kube d jim-cartpole
> Deleting jim's experiment cartpole
surreal-kube db jim
> Deleting all of jim's experiments
```

## Miscellaneous 
There are some helper functions in `surreal-kube` as well.

If you are developing using surreal, you may have a log of docker images built. Use `surreal-kube docker-clean` to clean them up.
```bash
surreal-kube dc
> Cleaning docker images
```

If you setup the `nfs` fields in `.surreal.yml` following [the tutorial](surreal_kube_gke.md#create-the-cluster). You can use `get-video`, `get-config`, `get-tensorboard` to get the coresponding information from the nfs.
```
surreal-kube get-video <experiment_name>
surreal-kube get-config <experiment_name>
surreal-kube get-tensorboard <experiment_name>
```

