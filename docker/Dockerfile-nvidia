FROM nvidia/cuda@sha256:4df157f2afde1cb6077a191104ab134ed4b2fd62927f27b69d788e8e79a45fa1
# Adapted from mujoco-py image

# Comments from mujoco-py image docker file
# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.


LABEL author="Stanford Surreal team"

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    cmake \
    unzip \
    bzip2 \
    wget \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY build_files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY build_files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# python deps
WORKDIR /mylibs
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -p /mylibs/miniconda -b
RUN rm /mylibs/Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/mylibs/miniconda/bin:${PATH}
RUN conda install python=3.5

# pytorch
RUN conda install --yes pytorch torchvision cuda90 -c pytorch
RUN conda install --yes ffmpeg -c conda-forge
# Fake a mujoco key
RUN touch /root/.mujoco/mjkey.txt 
COPY build_files/requirements.txt /mylibs/
RUN pip install -r /mylibs/requirements.txt

COPY surreal /mylibs/surreal
RUN pip install -e /mylibs/surreal

COPY build_files/entrypoint-nvidia.py /usr/local/bin/entrypoint.py
ENTRYPOINT ["/usr/local/bin/entrypoint.py"]

