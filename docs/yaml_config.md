# Setup Surreal Config File
This guides shows you how to setup the config yaml file of surreal. 

## Generating the config file
By default the config file is  `~/.surreal.yml`. You can also point to another location through the `SURREAL_CONFIG_PATH` environment variable. You can generate a default template to `~/.surreal.yml` by running
```bash
surreal-default-config
```
after installing surreal.

---
Follow the following guide to setup the necessary fields in the config
## General
* `username`. Username is prefixed to every experiment to avoid naming collisiong during collaboration.
* `prefix_experiment_with_username`. This boolean flag controls this naming behavior.

## Tmux
* `tmux_results_folder`. This is the path that experiments ran on tmux would be write to. For example, the experiment `foo` would be writing to directory `<tmux_results_folder>/foo`.

## NFS
* `nfs`. When running surreal on Kubernetes. It is highly recommended to use a network file system with Kubernetes. The `nfs` block in `.surreal.yml` configures nfs related information.

### Retrieving Data From the NFS
* `hostname`. Hostname of the nfs on your machine. Define this to use our helpers for retrieving data from the nfs system.
# `results_folder`. Where experiment results are stored on the nfs server. Experiment data utilities retrieve data from `<results_folder>/<experiment_name>`.

### Mounting the NFS on Kubernetes.
When running an experiment on kubernetes, you can mount a nfs to all containers so every process can access a shared file system.
* `servername`. Name of the nfs server in the perspective of nodes in the kubernetes cluster.
* `path_on_server`. Server directory serving as the file system.
* `mount_path`. Where the nfs is mounted in the containers.

## Kubernetes
* `kurreal_metadata_folder`. This is the path that experiments launched to kubernetes store their metadata (actual experiments happen in the cloud). 
* `cluster_definition`. After creating a kubernetes cluster with cloudwise, you will obtain a `.tf.json` file detailing the setup of the cluster. Specify its location at `cluster_definition` to allow `kurreal` commandline interface to properly schedule your workload.
* `kurreal_results_folder`. Where do experiments save results. Experiments write results to `<kurreal_results_folder>/<experiment_name>` in the container.
* `creation_settings`. Configures how experiments are launched on Kubernetes. See [documentation](creation_settings.md) for details.
* `mount_secrets`. Mount the listed files as secrets. These files would be available in `/etc/secrets` on every container. One example is to use it to mount the mujoco liscense. 
```yaml
mount_secrets:
  - ~/.mujoco/mjkey.txt
```
* `docker_build_settings`. Configures customly built docker images TODO.
