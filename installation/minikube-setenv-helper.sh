
# minikube automatically link to host docker 
# note that this has to be run in every console, preferably added to ~/.bashrc 
_MINIKUBE_RUNNING=`minikube status | grep Running`
if [ ! -z "$_MINIKUBE_RUNNING" ];
then
    eval $(minikube docker-env)
fi
