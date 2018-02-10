# Guide to Mounting ssh server onto your local machine (Linux/ OS X)

* (OS X only) Download FUSE and sshfs from [this site](https://osxfuse.github.io).
* (Linux only) sudo apt-get install sshfs

* Setup in '~/.ssh/config'
```
Host surrealfs
  Hostname 35.227.164.158
  User [your user name]
  IdentityFile [your identity file]
```

## To set up the server
```
mkdir ~/surrealfsroot
sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 surrealfs:/data ~/surrealfsroot
ln -s ~/surrealfsroot/[your user name] ~/surrealfs
```

## To mount
```
sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 surrealfs:/data ~/surrealfsroot
```

## To unmount
```
umount -f ~/surrealfsroot
```