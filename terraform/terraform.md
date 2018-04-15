# WARNING: This is here temporarily, will be moved.

# Use Terraform to Setup a Surreal Cluster
[Terraform is a tool for building, changing, and versioning infrastructure safely and efficiently.](https://www.terraform.io/intro/index.html) You can use it to setup the cloud kubernetes cluster easily. Here are the instructions:

# Install terraform
Follow the instructions on the [official website](https://www.terraform.io/intro/getting-started/install.html) and put `terraform` executable under your `$PATH`. 

# Index
Depending on what provider you have, there are different instructions and templates to use. Here are the supported platforms. We welcome contributions if you setup Surreal on another platform.  
* [Google Cloud](#google-cloud)
* Amazon AWS (TODO)
* Microsoft Azure (TODO)

# Google Cloud 
## Setup credential
You need to go to [Google Developers Console](https://console.developers.google.com/) and create a service account key (json format) for a Compute Engine default service account. [See "Authentication JSON File" section of this guide](https://www.terraform.io/docs/providers/google/). Link to the file in google-cloud.tf's `credential` variable so terraform knows where to find it.

## Use terraform to setup a Google Kubernetes Engine cluster
Create a new directory, copy `google-cloud.tf` to it, do 
```
terraform init
terraform plan
```
Terraform will tell you what needs to be done to satisfy the specifications. If you are happy with that, do 
```
terraform apply
```
and confirm. Terraform would setup the kubernetes cluster for you.

