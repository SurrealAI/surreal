# Contributing to Surreal

## Develop Locally
To develop surreal, first create a fork. And then obtain an editable installation.
```bash
git clone https://github.com/SurrealAI/Surreal.git
cd Surreal
pip install -e .
```
Setup the dependencies as in [tmux tutorial](surreal_tmux.md). Then you are good to go. You may need to write your own version of `ppo_configs.py` or `ddpg_configs.py` (but still compatible to the commandline interface), in this case, you can point to it when launching the experiment 
```bash
surreal-tmux create --algorithm <path to my_algorithm.py> -- [arguments parsed by my_algorithm.py]
```
Refer to [ddpg_configs.py](../) or [ppo_configs.py](../surreal/main/ppo_configs.py) for reference.

To test your implementation, you can use our integration test script.
```bash
python test/test_ppo_gym.py
...
```

## Develop on Kubernetes
If you want to try your customized Surreal on Kubernets, you need to build your version of surreal into a different docker image. This requires some setup. We assume that you already have Surreal locally installed with all the dependencies.

### Preparations
1. Install [Docker](https://www.docker.com)
2. Obtain the Surreal base image. This image take ~10G space.
```bash
docker pull surrealai/surreal-nvidia:v0.0
```
3. Prepare your own docker registry to host your custom images. You need to put your custom images in a registry that is accessible from your Kubernetes cluster. For clarity we suppose that we are pushing to a repository called `better-ddpg`. You have two options:
    - You can use dockerhub if you don't mind your image being public. So we can be pushing our built images to `foobar/better-ddpg`.
    - If you are using Google Cloud with project-id, you can push-pull from Google Cloud's container registry: `us.gcr.io/<project-id>/<repo>:tag`. So if your project id is `myproject-123456`. The repository to push to is `us.gcr.io/myproject-123456/better-ddpg`. You do need to configure your own docker to have the credentials to read/write from the google hosted registry, use `gcloud auth configure docker` (see [documentation](https://cloud.google.com/sdk/gcloud/reference/auth/configure-docker) for details). Google Cloud's docker registry is private but by default accessible by your kubernetes cluster.

### Building images automatically
Constantly building and uploading new docker images is quite tedious. We provide docker builder (powered by [Symphony](https://github.com/SurrealAI/symphony)) that puts together a docker build environment from multiple locations in your file system. Everytime you use `surreal-kube` to create an experiment, we will build a new image automatically containing the latest code. Here is how to configure it (a skeleton of these configs are provided by default in the generated `.surreal.yml`). This example assumes that you are using the google cloud registry.

* In your creation setting, specify repository for `agent:image` and `nonagent:image` but no tag. 
```yaml
creation_settings
  contrib:
    ...
    agent:
      image: us.gcr.io/myproject-123456/better-ddpg # <my-registry>/<repo-name>
      build_image: contrib-image
      ...
    nonagent:
      image: us.gcr.io/myproject-123456/better-ddpg # <my-registry>/<repo-name>
      build_image: contrib-image
      ...
```
* You may have noticed that `build_image` is no longer `null` here. This field specifies the image build process defined in `docker_build_settings`. These build settings are defined in `docker_build_settings` section of the config. `build_image: contrib-image` refers to the `name: contrib-image` attribute of the first entry in `docker_build_settings`.
```yaml
docker_build_settings:
  - name: contrib-image
    temp_directory: <~/symph_temp/contrib or anywhere you want image build to happen>
    verbose: true
    dockerfile: <path to your surreal fork>/docker/Dockerfile-contribute
    context_directories:
      - name: surreal
        path: <path to your surreal fork>
        force_update: true
```

* The syntax of `docker_build_settings` is detailed in [Symphony Docker builder documentation](https://github.com/SurrealAI/symphony/blob/master/docs/docker_builder.md). It tells the builder where to collect the files needed to build your image. In this case, to build the `contrib` image, the docker builder should use Dockerfile in `<surreal_repo>/docker/Dockerfile-contribute`. It also copies your surreal fork into the context of docker build, under the folder named `surreal` (specified by `context_directories.name = surreal`).

* Now we can look at the Dockerfile `Dockerfile-contribute`. It is very straightforward. It uses the base Surreal image, uninstalls the existing version of Surreal and installs your version.

```
FROM surrealai/surreal-nvidia:v0.0
# ~/surreal/docker/Dockerfile-contribute
COPY surreal /mylibs/surreal-dev
RUN pip uninstall -y surreal
RUN pip install -e /mylibs/surreal-dev
```

* When you run `surreal-kube create contrib ...`. The builder will build the image and push it to `us.gcr.io/myproject-123456/better-ddpg:<experiment_name>`. The `surreal-kube` logic would also tell Kubernetes to pull the correct image. Now you have a simple an automated deployment process.
