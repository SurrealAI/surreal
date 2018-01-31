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

`~/.surreal` config file

git generate access token

decide which branch is the temp branch to snapshot your code

# Walk through

## Pod YAML

## Kurreal API


