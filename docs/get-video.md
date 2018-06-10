 # DEPRECATED

**Now we use `gcloud compute ssh` and `gcloud compute scp` instead of plain SSH, so no need for configuring ssh.**
 
To use `kurreal gv` functionality, you need to do some preparations

## Installation
* Setup ssh connection to surrealfs
In `~/.ssh/config` I have following settings to allow me to do `ssh surrealfs`
```
Host surrealfs
  Hostname 35.227.164.158
  User jirenz
  IdentityFile ...
```
* Add the following line to your `.surreal.yml`
```
nfs_host: surrealfs # or whatever you do ssh into to access nfs
```
* Install `fabric`
```
pip install fabric
```

## Usage
* Find a directory that you want to store the videos
```
kurreal gv .
```
It will get videos of all your running experiments.

You can also do 
```
kurreal gv experiment_1 experiment_2
```
to fetch videos for older environments

For other options see `--help`