To run dm_control on the cloud, we need the following operations

# Install Xdummy
```
# Install the Xdummy driver
sudo apt-get install xserver-xorg-video-dummy
```

# Running the X server
One time set up
```
# Running xorg server
# https://xpra.org/trac/wiki/Xdummy
mkdir ~/.fakeX/
touch ~/.fakeX/10.log
wget -O ~/.fakeX/xorg.conf http://xpra.org/xorg.conf
```
Keep this process running when you run experiments. It provides a fake display (:10) 
```
Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ~/.fakeX/10.log -config ~/.fakeX/xorg.conf :10
```
# TODO: make the process a daemon


# Set the DISPLAY variable
The Xserver can be accessed by setting "DISPLAY=:10"
This script sets the DISPLAY variable for the current shell and writes to .bashrc for all future logins.
```
export DISPLAY=:10
echo '' >> ~/.bashrc 
echo '# Set display variable for X server'
echo 'export DISPLAY=:10' >> ~/.bashrc 
```

# Compile latest glfw from source 
```
# See https://github.com/glfw/glfw/issues/1004
sugo apt-get remove libglfw3
cd ~
git clone https://github.com/glfw/glfw.git
cd glfw
cmake -DBUILD_SHARED_LIBS=ON .
make && sudo make install
```
