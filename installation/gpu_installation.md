# Guide to installing gpu and using it for surreal

# Running on a vm

In general, following the [google official documentation](https://cloud.google.com/compute/docs/gpus/add-gpus) is enough.

If you are a surreal-dev team member running on a google vm with nfs mounted (default on /mnt). Do the installation script
```
sudo /mnt/gpu_installation_google.sh
sudo /mnt/gpu_performance.sh
echo 'export DISABLE_MUJOCO_RENDERING=1' >> ~/.bashrc
```
Note that the default installation provided by google overwrites the original opengl installation in the image so glfw will throw errors as we attemp to render dm_control. We need to disable rendering for dm_control.

A potential work around is to install through nvidia run file where you can configure whether or not to install opengl bindings. However this is not officially supported by google and one needs to manage the dependency manually. TODO: see if [this guide](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07#install-nvidia-graphics-driver-via-runfile) is sufficient 

# Running gpu instances on a kubernetes engine
Resources:
* [google contianer engines on gpu](https://cloud.google.com/kubernetes-engine/docs/concepts/gpus)
* [Kubernetes on GPU support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
* As the GPU functionality is in beta, we need to use an alpha cluster to access the functionalities.
* [Latest GKE versions](https://cloud.google.com/kubernetes-engine/release-notes)
* [GKE alpha clusters](https://cloud.google.com/kubernetes-engine/docs/concepts/alpha-clusters)

## Instructions
Command to create cluster
```
gcloud  beta container clusters create [cluster-name] --enable-kubernetes-alpha --accelerator type=nvidia-tesla-k80,count=1 --cluster-version 1.9.2-gke.0 --zone [zone]
```
Add custom commands as needed, if you have the money, you can replace k80 with p100

Linking kubectl to the cloud
```
gcloud container clusters get-credentials [cluster-name]
```

Create a daemonset which installs dependencies on every new gpu-node
```
kubectl create -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/k8s-1.9/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

When creating pods running on gpu nodes, use the following template
```
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: foo-bar
    resources:
      limits:
       nvidia.com/gpu: 1
    image: [foo-bar] 
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-k80 # or nvidia-tesla-k80
```
where [foo-bar] is an image built from nvidia provided base images, like nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04