# Installation

## Docker

## Minikube

[Official instructions](https://kubernetes.io/docs/tasks/tools/install-minikube/).

First install Virtualbox driver.

```bash
sudo apt-get update
sudo apt-get install virtualbox
```

Then [install](https://github.com/kubernetes/minikube/releases) Minikube executable

```bash
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.25.0/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
```

`minikube-setenv.sh` to communicate with local docker.
`minikube-mount.sh` to mount hostDir

# One-time setup

`~/.surreal.yml`

git generate access token

decide which branch is the temp branch to snapshot your code

# Walk through

## Google Cloud cluster

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
* Search for single node file server
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


