# Setting up an NFS on the Cloud
* [Google Cloud](#google-cloud)   
* [AWS](#aws)  
* [Azure](#azure)  

---

# Google Cloud
The general guide for file servers on Google Cloud is [here](https://cloud.google.com/solutions/filers-on-compute-engine). We used a single node file server, [documentation here](https://cloud.google.com/solutions/filers-on-compute-engine#single-node-file-server). Follow the setup guide to create the network file system. If you created a single node file server, you can use its name (say it is `my-nfs-server`) as seen in your Compute Engine Console for the file to configure `kurreal`
```yaml
nfs:
  servername: my-nfs-server
```

# AWS
Stay tuned

# Azure
Stay tuned