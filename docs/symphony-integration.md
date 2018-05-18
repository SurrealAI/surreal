Folks, we are ready to roll out the update that migrates surreal to use symphony.

# Main features
Surreal will be powered by Symphony
The build process is updated and now you will build container images locally

# ACTION ITEMS
Clone symphony and do pip install -e .
After merging, Update your .surreal.yml according to the [surreal repo]/containers/sample.surreal.yml
At your convenience, get docker, and run these commands (takes ~ 15G of disk space):
```
gcloud auth configure-docker
docker pull us.gcr.io/surreal-dev-188523/surreal-base-cpu-new:latest
docker pull us.gcr.io/surreal-dev-188523/surreal-base-gpu-new:latest
```

# Symphony features
* Symphony shouldn’t change your usual workflow. Most actions like create dev remain the same. Some naming schemes are now following the symphony convention.
* `create-dev` is unchanged
* `delete`, `delete-batch`, `ssh`, `exec`, `scp`, `log (logs)`, are unchanged
* `ls`, `lse` list all experiments
* `p`, `lsp` list all processes with upgraded ui
* `exp [new_exp]` lists current experiment or switches namespace
* `kurreal visit tensorboard` now leads you to tensorboard

# New build process
* The new build process is configured to do the following:
When you create an experiment, the required images are determined by the pod spec: 
```
# in .surreal.yml
pod_types:
  agent:
# kurreal handles tag now
    image: us.gcr.io/surreal-dev-188523/surreal-cpu-new 
    selector:
      surreal-node: agent
    resource_request:
      cpu: 1.5
    build_image: cpu-image
```
When pod_type agent is referred, the build_image tag tells kurreal to look for cpu-image the spec in your config's docker_build_settings
In your updated .surreal.yml config:
```
docker_build_settings:
  - name: cpu-image # kurreal refers to this name when launching
    # set it to your favorite,
    # a subdirectory will hold all build files
    temp_directory: ~/symph_temp/cpu
    # verbose: true (default: prints docker cmdline output)
    dockerfile: ~/surreal/Surreal/container/DockerfileCPU
    context_directories:
# These paths will be copied to the build temp dir
      - name: surreal
        path: ~/surreal/Surreal
        force_update: true 
# Directory ~/surreal/Surreal will be put under folder 
# surreal in the build context for docker build
# force_update means copy a new version every time
      - name: tensorplex
        path: ~/surreal/Tensorplex
        force_update: false # Only copy when not found in build_directory
      - name: mujoco
        path: ~/surreal/MujocoManipulation
        force_update: false 
      - name: torchx
        path: ~/surreal/TorchX
        force_update: false
```
* Note the docker file field: cpu-image is pointed to `[surreal_repo]/container/DockerfileCPU` and gpu-image is pointed to `[surreal_repo]/container/DockerfileGPU`. You can edit these files in your development to alter the build process of your images, say to add pip install .... 
* The image is pushed to our google cloud's container registry with tag being your experiment name. So all your code files will be persisted. 
* Addtionally, mjkey.txt is now built into our base images. So you don't need to link to it locally.