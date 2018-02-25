# Installation


# One-time setup

`~/.surreal.yml`

git generate access token

decide which branch is the temp branch to snapshot your code

# Walk through

## Google Cloud cluster
If you want GPU, go to [creating a GPU cluster](# Creating a GPU cluster)
```
gcloud container node-pools create agent-pool --cluster mycluster -m n1-standard-2 --num-nodes=8

gcloud container node-pools create nonagent-pool --cluster mycluster -m n1-highmem-8 --num-nodes=1
```

Must label nodes in order to assign agent/non-agent to their corresponding node pool

Also need to put the label config in `~/.surreal.yml`

```python
kube.label_nodes('cloud.google.com/gke-nodepool=agent-pool',
                            'surreal-node', 'agent-pool')
kube.label_nodes('cloud.google.com/gke-nodepool=nonagent-pool',
                    'surreal-node', 'nonagent-pool')
```

## Pod YAML

## Kurreal API

## TO create a nfs on google
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

## Connect to Tensorboard

```bash
autossh -N -L localhost:9006:10.138.0.33:6006 gke-mycluster-nonagent-pool-0b0a9484-l3kg.us-west1-b.<gcloud-url>
```



## To mound nfs on kube
In .yaml file
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

# Creating a GPU cluster
This guide shows how to create an alpha cluster and run gpu-accelerated ku/surreal on it.

First we create the cluster
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

Next we set up some context
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

Run the daemon to install nvidia drivers
```
kubectl create -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/k8s-1.9/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

Check to see the daemons are running by 
```
kubectl get pods --namespace=kube-system | grep nvidia
>>>nvidia-driver-installer-ng5vv ...
>>>nvidia-gpu-device-plugin-n8stz ...
```

Next we tag the nodes, 
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