# Setup Surreal Config File
This guides shows you how to setup the config yaml file of surreal. 

## Generating the config file
By default the config file is  `~/.surreal.yml`. You can also point to another location through the `SURREAL_CONFIG_PATH` environment variable. You can generate a default template to `~/.surreal.yml` by running
```bash
surreal-default-config
```
after installing surreal.

---

The following guide allows you to setup the necessary fields in the config. Depending on your need, there are several parts that you need to setup.

[General](#general)  
[Tmux](#tmux)  
[NFS (optional, recommended)](#nfs)  
[Kubernetes](#Kubernetes)  

## General
* `username`. Username is prefixed to every experiment to avoid naming collisiong during collaboration.
* `prefix_experiment_with_username`. This boolean flag controls this naming behavior.

## Tmux
* `tmux_results_folder`. This is the path that experiments ran on tmux would be write to. For example, the experiment `foo` would be writing to directory `<tmux_results_folder>/foo`.
* `tmux_preamble_cmds`. This is a list of commands to run in every tmux pane before everything else. This is the prefect place to put `source activate <my_virtual_environment>`.

## NFS
* `nfs`. When running surreal on Kubernetes. It is highly recommended to use a network file system with Kubernetes. The `nfs` block in `.surreal.yml` configures nfs related information.

### Retrieving Data From the NFS
* `hostname`. Hostname of the nfs on your machine. Define this to use our helpers for retrieving data from the nfs system.
* `results_folder`. Where experiment results are stored on the nfs server. Experiment data utilities retrieve data from `<results_folder>/<experiment_name>`.

### Mounting the NFS on Kubernetes.
When running an experiment on kubernetes, you can mount a nfs to all containers so every process can access a shared file system.

* `servername`. Name of the nfs server in the perspective of nodes in the kubernetes cluster.
* `path_on_server`. Server directory serving as the file system.
* `mount_path`. Where the nfs is mounted in the containers.

### Example
Suppose that you created a single node file server on Google Cloud and named it `surreal-fs-server`. You configred the server's `/data` directory to serve as the file system. On containers, you mount the file system to directory `/fs`. And you want to store the experiment outputs to `/fs/experiments/my_username`. You can use the following configuration. In your own `~/.ssh/config`, you can configure host name `surrealfs`. This would result in the following settings.
```yaml
kurreal_results_folder: /fs/experiments/my_username
nfs:
  hostname: surrealfs
  servername: surreal-fs-server
  path_on_server: /data
  mount_path: /fs
  results_folder: /data/experiments/my_username
```
```bash
# ~/.ssh/config
Host surrealfs
  Hostname <ip or surreal-fs-server>
  User <your username to access google cloud vm>
  IdentityFile <your secret key to access google cloud vm>
```
Note: This solution does not deal with user name mapping in nfs well. If there are permission issues, set the permissions on your `/data/experiments` folder to be `777`.

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
* `docker_build_settings`. Configures customly built docker images. See below

### Docker Build Settings
When you are developing a distributed containerized application, it is very natural that you will build docker images and then deploy them. This process is sped up by `symphony.DockerBuilder` which allows you build docker images using files at different location on your file system. 

`docker_build_settings` in your `.surreal.yml` allows you to specify how docker images are build. Take the following file as an example. Let's assume that you have your own fork of Surreal at `~/my_surreal`. You have a docker registry accessible from your Kubernetes cluster at `my-registry`. The following setting would allow `kurreal` to automatically build an image for you whenever you are launching an experiment.
```yaml
# .surreal.yml
creation_settings:
  contrib:
    mode: basic
    agent:
      image: my-registry/contrib-image
      build_image: contrib-image
    nonagent:
      ...

docker_build_settings:
  - name: contrib-image
    ...
```
Here `agent: build_image` field equals, `contrib-image`, which tells `kurreal` to look at the dictionary in `docker_build_settings` with `name: contrib-image`. The remaining fields in the `contrib-image` build settings are passed to `symphony.DockerBuilder`, see[documentation here](https://github.com/SurrealAI/symphony/blob/master/docs/docker_builder.md). Long story short, it builds a docker image using `~/surreal/docker/Dockerfile-contribute` in a directory containing a copy of `~/my_surreal`.
```yaml
docker_build_settings:
  - name: contrib-image
    temp_directory: ~/symph_temp/contrib
    verbose: true
    dockerfile: ~/surreal/docker/Dockerfile-contribute
    context_directories:
      - name: surreal 
        path: ~/my_surreal
        force_update: true
```
The dockerfile then copies the custom Surreal library and installs it.
```Dockerfile
TODO: From
# ~/surreal/docker/Dockerfile-contribute
COPY surreal /mylibs/surreal-dev
RUN pip install -e -U /mylibs/surreal-dev
```
After building, the image is pushed to `my-registry/contrib-image:<experiment_name>`. This is why we don't need to specify tag in the `agent:image` setting field
```
settings:
  contrib:
    agent:
      image: my-registry/contrib-image
```
