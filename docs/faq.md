# FAQ
## Local Tmux Mode

## Kubernetes Mode
* Terraform install fails.
    - If you are seeing error: `... API has not been used in project...`: during `terraform apply`, go to the Kubernetes Engine tab and/or Compute Engine tab on your google cloud console to enable their APIs.

* GPU nodes are not scaling up.
    - Check if the driver installation daemon set is running (see [documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)). Make sure you have run 
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

* Agents/learner pending for a long time. 
    - Because autoscaling is enabled, it takes a while for machines to spin up once a new experiment is launched. This takes 5 ~ 10 minutes for CPU nodes and as long as 15 minutes for GPU nodes. To debug this, you can run `kubectl describe pod [agent-0/eval-1/agents-3/nonagent`] and look at the activities in the bottom. If you see "Pod triggered scale up", then in a while the machines would spin up and the workloads will be scheduled. If you see "Pod won't fit even if a node is created", something is wrong about scheduling. You can also refer to the cluster autoscaler documentation [here](https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md).

* Nodes not scaling down after no experiments are running. 
    - Again, you can refer to the cluster autoscaler documentation [here](https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md). By default, a node will not be scaled down until it has been idle for 10 minutes, so be patient. Checking the output of `kubectl get configmap cluster-autoscaler-status -n kube-system -o yaml` can also be helpful.
