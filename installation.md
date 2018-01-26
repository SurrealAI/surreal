This is an installation guide written under Ubuntu 16. 

# Create VM
Create a Ubuntu 16 VM with custom boot disk image surreal-dev-base-cpu-image. Then go to [user setup](#user-setup)

# Install system Dependencies

* Ubuntu:
```
sudo apt-get install xserver-xorg-video-dummy cmake libglfw3 xorg-dev unzip bzip2 git libglew1.13
```

* Debian:
```
sudo apt-get install xserver-xorg-video-dummy cmake libglfw3 xorg-dev unzip bzip2 git libglew2.0
```

* Compile latest glfw

Compile the latest glfw library from source. We did install it earlier because this gives us all the dependencies.
```
sudo apt-get remove libglfw3
cd ~
git clone https://github.com/glfw/glfw.git
cd glfw
cmake -DBUILD_SHARED_LIBS=ON .
make && sudo make install
```

* Set up X11
```
git clone https://github.com/StanfordVL/Surreal.git
sudo mkdir /etc/fakeX
sudo cp Surreal/installation/xorg.conf /etc/fakeX/
sudo touch /etc/fakeX/10.log
sudo cp Surreal/installation/xorg.service /etc/systemd/system/
sudo systemctl enable xorg
```

* (Don't use) Set up mujoco on a shared machine. Danger! Downloading and running stuff with sudo
```
sudo mkdir /etc/mujoco
cd /etc/mujoco
sudo wget https://www.roboti.us/download/mjpro150_linux.zip
sudo unzip mjpro150_linux.zip
sudo rm mjpro150_linux.zip
sudo wget https://www.roboti.us/download/mjpro131_linux.zip
sudo unzip mjpro131_linux.zip
sudo rm mjpro131_linux.zip
```
And then add the mjkey.txt

<a name="user-setup"></a>
# If you are using images, start from here

# User level setup
* Set the DISPLAY variable
```
export DISPLAY=:10
echo '' >> ~/.bashrc 
echo '# Set display variable for X server' >> ~/.bashrc
echo 'export DISPLAY=:10' >> ~/.bashrc 
```

* Set up mujoco (Note: not for surreal-dev users)
```
mkdir ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mjpro150_linux.zip
unzip mjpro150_linux.zip
rm mjpro150_linux.zip
wget https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
rm mjpro131_linux.zip
echo "Put your liscense file (mjkey.txt) under ~/.mujoco/mjkey.txt"
```

* Surreal-dev users, please use
```
cd ~
ln -s /etc/mujoco .mujoco
```



# Python, Conda
* We will use conda for environment management
```
cd ~
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh 
source ~/.bashrc 
```

* Get the installation specifications from the repos
```
cd ~
git clone https://github.com/StanfordVL/Surreal.git
git clone https://github.com/StanfordVL/Tensorplex.git
```

* Create a new virtual env for surreal
```
cd ~
conda create --name surreal python=3.5 --file Surreal/installation/surreal-all-spec-file.txt
source activate surreal
pip install -r Surreal/installation/surreal-all-requirements.txt
python -m ipykernel install --user --name surreal
```

* Install dm_control. This will fail if you don't have mujoco installed
```
pip install git+git://github.com/deepmind/dm_control.git
```

# Install the Surreal packages
```
cd ~/Surreal
pip install -e .
cd ~/Tensorplex
pip install -e .
```

Install pytorch
# CPU
```
conda install pytorch-cpu torchvision -c pytorch
```
# GPU
```
conda install pytorch torchvision -c pytorch
```


