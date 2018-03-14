# Internal Setup

This guide is for internal dev only. It contains mostly one-time commands to configure the GCloud cluster, docker image, and workarounds for dependencies like DM control.

For contributors: you don't need to read this doc unless you need to change the cluster configuration, docker image, or other low-level infrastructure. 

Please consult with the core dev team before you make any disruptive changes.


# GCloud setup

### Create NFS
* there is one running called surreal-shared-fs-vm
* Ask Yuke for permission
* Go to google cloud launcher
* Search for ["single node file server"](https://console.cloud.google.com/launcher/details/click-to-deploy-images/singlefs).
* Follow the instructions
* Copy the mount command in the completed page
* Set "startupscript" field of vm to be
```
apt-get -y install nfs-common
mount -t nfs surreal-shared-fs-vm:/data /mnt
```

### Mound nfs on kube pods
In .yml file
```
containers:
  - name: ...
    ...
    volumeMounts:
    - mountPath: /mnt
      name: nfs-volume
  volumes:
  - name: nfs-volume
    nfs:
      server: surreal-shared-fs-vm
      path: /data
```


# GPU for Surreal

## Running on a vm

In general, following the [google official documentation](https://cloud.google.com/compute/docs/gpus/add-gpus) is enough.

If you are a surreal-dev team member running on a google vm with nfs mounted (default on /mnt). Do the installation script
```
sudo /mnt/gpu_installation_google.sh
sudo /mnt/gpu_performance.sh
echo 'export DISABLE_MUJOCO_RENDERING=1' >> ~/.bashrc
```
Note that the default installation provided by google overwrites the original opengl installation in the image so glfw will throw errors as we attemp to render dm_control. We need to disable rendering for dm_control.

A potential work around is to install through nvidia run file where you can configure whether or not to install opengl bindings. However this is not officially supported by google and one needs to manage the dependency manually. TODO: see if [this guide](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07#install-nvidia-graphics-driver-via-runfile) is sufficient 

## Running gpu instances on a kubernetes engine
Resources:
* [google contianer engines on gpu](https://cloud.google.com/kubernetes-engine/docs/concepts/gpus)
* [Kubernetes on GPU support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
* As the GPU functionality is in beta, we need to use an alpha cluster to access the functionalities.
* [Latest GKE versions](https://cloud.google.com/kubernetes-engine/release-notes)
* [GKE alpha clusters](https://cloud.google.com/kubernetes-engine/docs/concepts/alpha-clusters)



## Creating a GPU cluster
This guide shows how to create an alpha cluster and run gpu-accelerated ku/surreal on it.

First we create the cluster. You can replace k80 with p100:
```
gcloud beta container clusters create [cluster-name] --enable-kubernetes-alpha --cluster-version 1.9.2-gke.0 --zone [zone] --num-nodes=1
```
```
gcloud beta container clusters create kurreal-1 --cluster-version 1.9.2-gke.1 --zone us-west1-b --num-nodes=2 -m n1-standard-1
```

Next we opt in for beta-features (tainting)
```
gcloud config set container/use_v1_api_client false
```

Next we set up some context to link local kubectl on your laptop to the remote GCloud. 
```
gcloud config set container/cluster [cluster-name]
gcloud container clusters get-credentials [cluster-name]
```

If this line runs successfully the cluster is setup
```
kubectl cluster-info
```

To create the nodes, run
```
gcloud beta container node-pools create agent-pool -m n1-standard-2 --node-labels surreal-node=agent --enable-autoscaling --min-nodes 0 --max-nodes 500 --num-nodes 2 --node-taints surreal=true:NoExecute


gcloud beta container node-pools create nonagent-pool-cpu -m n1-highmem-8 --node-labels surreal-node=nonagent-cpu --enable-autoscaling --min-nodes 0 --max-nodes 100 --num-nodes 2 --node-taints surreal=true:NoExecute

gcloud beta container node-pools create nonagent-pool-gpu -m n1-highmem-8 --accelerator type=nvidia-tesla-k80,count=1 --node-labels surreal-node=nonagent-gpu --enable-autoscaling --min-nodes 0 --max-nodes 100 --num-nodes 2 --node-taints surreal=true:NoExecute

gcloud beta container node-pools create nonagent-pool-gpu-2k80-16cpu -m n1-standard-16 --accelerator type=nvidia-tesla-k80,count=2 --node-labels surreal-node=nonagent-gpu-2k80-16cpu --enable-autoscaling --min-nodes 0 --max-nodes 100 --num-nodes 1 --node-taints surreal=true:NoExecute

gcloud beta container node-pools create nonagent-pool-gpu-4k80-32cpu -m n1-standard-32 --accelerator type=nvidia-tesla-k80,count=4 --node-labels surreal-node=nonagent-gpu-4k80-32cpu --enable-autoscaling --min-nodes 0 --max-nodes 100 --num-nodes 1 --node-taints surreal=true:NoExecute

gcloud beta container node-pools create nonagent-pool-gpu-4k80-16cpu -m n1-standard-16 --accelerator type=nvidia-tesla-k80,count=4 --node-labels surreal-node=nonagent-gpu-4k80-16cpu --enable-autoscaling --min-nodes 0 --max-nodes 100 --num-nodes 1 --node-taints surreal=true:NoExecute

gcloud beta container node-pools create nonagent-pool-gpu-1p100-16cpu -m n1-standard-16 --accelerator type=nvidia-tesla-p100,count=1 --node-labels surreal-node=nonagent-gpu --enable-autoscaling --min-nodes 0 --max-nodes 100 --num-nodes 1 --node-taints surreal=true:NoExecute
```

Create a daemonset to install dependencies (e.g. nvidia drivers) on every new gpu-node
```
kubectl create -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/k8s-1.9/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

Check to see the daemons are running by 
```
kubectl get pods --namespace=kube-system | grep nvidia
>>>nvidia-driver-installer-ng5vv ...
>>>nvidia-gpu-device-plugin-n8stz ...
```

Must tag nodes in order to assign pods to their corresponding node pool:
```python
kube.label_nodes('cloud.google.com/gke-nodepool=agent-pool',
                            'surreal-node', 'agent-pool')
kube.label_nodes('cloud.google.com/gke-nodepool=nonagent-pool',
                    'surreal-node', 'nonagent-pool')
```
The rest are the same for the non-GPU case

To clean up the cluster:
```
gcloud container clusters delete [cluster-name]
gcloud config unset container/cluster
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
 
 
# `dm_control` on Ubuntu

Workaround for [`dm_control` issue](https://github.com/deepmind/dm_control/issues/21).

### Install Xdummy
```
# Install the Xdummy driver
sudo apt-get install xserver-xorg-video-dummy
```

### Running the X server
One time set up
```
# Running xorg server
# https://xpra.org/trac/wiki/Xdummy
mkdir ~/.fakeX/
touch ~/.fakeX/10.log
wget -O ~/.fakeX/xorg.conf http://xpra.org/xorg.conf
```
Keep this process running when you run experiments. It provides a fake display (:10) 
```
Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ~/.fakeX/10.log -config ~/.fakeX/xorg.conf :10
```

*TODO: make the process a daemon*

### Set the DISPLAY variable
The Xserver can be accessed by setting "DISPLAY=:10"  
This script sets the DISPLAY variable for the current shell and writes to .bashrc for all future logins.
```
export DISPLAY=:10
echo '' >> ~/.bashrc 
echo '# Set display variable for X server' >> ~/.bashrc
echo 'export DISPLAY=:10' >> ~/.bashrc 
```

### Compile latest glfw from source 
An issues we will meet was fixed after the latest release in 2016
```
# See https://github.com/glfw/glfw/issues/1004
sudo apt-get install libglfw3
sudo apt-get remove libglfw3
# So we have all depencencies
cd ~
git clone https://github.com/glfw/glfw.git
cd glfw
cmake -DBUILD_SHARED_LIBS=ON .
make && sudo make install
```
