# WIP: this is for dm_control environments

FROM ubuntu:16.04

LABEL author="Stanford Surreal team"

# Base dependency
RUN mkdir /mylibs
WORKDIR /mylibs
RUN apt-get update && apt-get install -y cmake unzip bzip2 curl git wget libglfw3 libglew1.13 libglu1-mesa-dev libglew-dev libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xserver-xorg-video-dummy xorg-dev patchelf

RUN apt-get -y remove libglfw3 \
	&& git clone https://github.com/glfw/glfw.git
WORKDIR /mylibs/glfw
RUN cmake -DBUILD_SHARED_LIBS=ON . \
	&& make \
	&& make install
WORKDIR /
RUN rm -r /mylibs/glfw

# mujoco
RUN mkdir /root/.mujoco
WORKDIR /root/.mujoco
RUN wget https://www.roboti.us/download/mjpro150_linux.zip \
    && unzip mjpro150_linux.zip \
    && rm mjpro150_linux.zip
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mjpro150/bin

# python deps
WORKDIR /mylibs
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& bash Miniconda3-latest-Linux-x86_64.sh -p /mylibs/miniconda -b
RUN rm /mylibs/Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/mylibs/miniconda/bin:${PATH}

# pytorch
RUN conda install --yes pytorch-cpu torchvision -c pytorch
COPY requirements.txt /mylibs/
RUN pip install -r /mylibs/requirements.txt

# DM control suite
RUN pip install git+git://github.com/deepmind/dm_control.git

# required for imageio
RUN conda install -y ffmpeg -c conda-forge

# fake display to work around the GLFW problem
RUN mkdir /etc/fakeX \
	&& touch /etc/fakeX/10.log
COPY build_files/xorg.conf /etc/fakeX/
COPY build_files/xorg.service /etc/systemd/system/
RUN systemctl enable xorg
ENV DISPLAY=:10

# TODO: remove (by deprecating) lines below
RUN pip install \
    git+git://github.com/SurrealAI/Tensorplex.git \
    git+git://github.com/SurrealAI/TorchX.git

COPY mujoco /mylibs/mujoco
RUN pip install -e /mylibs/mujoco

COPY surreal /mylibs/surreal
RUN pip install -e /mylibs/surreal

COPY build_files/entrypoint-ubuntu.py /usr/local/bin/entrypoint.py
ENTRYPOINT ["/usr/local/bin/entrypoint.py"]
