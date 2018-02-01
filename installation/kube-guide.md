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


