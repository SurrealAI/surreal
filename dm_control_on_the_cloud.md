To run dm_control on the cloud, we need the following operations

* Install Xdummy
# Install the Xdummy driver
sudo apt-get install xserver-xorg-video-dummy

* Running the X server
# Running xorg server
# https://xpra.org/trac/wiki/Xdummy

mkdir ~/.fakeX/
touch ~/.fakeX/10.log
wget -O ~/.fakeX/xorg.conf http://xpra.org/xorg.conf
Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ~/.fakeX/10.log -config ~/.fakeX/xorg.conf :10

# TODO: make the process a daemon

* Set the DISPLAY variable
export DISPLAY=:10
echo '' >> ~/.bashrc 
echo '# Set display variable for X server'
echo 'export DISPLAY=:10' >> ~/.bashrc 

* Compile latest glfw from source 
# https://github.com/glfw/glfw/issues/1004
sugo apt-get remove libglfw3
cd ~
git clone https://github.com/glfw/glfw.git
cd glfw
cmake -DBUILD_SHARED_LIBS=ON .
make && sudo make install

