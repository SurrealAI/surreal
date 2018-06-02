To use `kurreal gv` functionality, you need to do some preparations

# Installation
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

# Usage
* Find a directory that you want to store the videos
```
kurreal gv .
```
It will get videos of all your running experiments.