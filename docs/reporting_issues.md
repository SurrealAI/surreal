# Reporting issues
* When you are reporting issues, be sure to include the stack trace, plus related information. 
* We understand that it is hard to provide a simple test case for a distributed system. We appreciate it if you can narrow the issue down.
* For something related to Kubernetes or Google Cloud, especially when some workloads fail to schedule, remember that `<kube_metadata_folder>/<experiment_name>` contains information about how the experiment is launched. These files, plus the `.tf.json` cluster definition can be very helpful.