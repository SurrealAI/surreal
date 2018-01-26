FROM ubuntu:16.04

RUN apt-get update && apt-get -y install xserver-xorg-video-dummy cmake libglfw3 xorg-dev unzip bzip2 curl git libglew1.13 wget

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
	&& bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b \
	&& rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

RUN apt-get -y remove libglfw3 \
	&& git clone https://github.com/glfw/glfw.git
WORKDIR glfw
RUN cmake -DBUILD_SHARED_LIBS=ON . \
	&& make \
	&& make install
WORKDIR /
RUN rm -r /glfw

RUN mkdir /etc/fakeX \
	&& touch /etc/fakeX/10.log
COPY xorg.conf /etc/fakeX/
COPY xorg.service /etc/systemd/system/
RUN systemctl enable xorg

RUN mkdir /etc/mujoco
WORKDIR /etc/mujoco
RUN wget https://www.roboti.us/download/mjpro150_linux.zip
RUN unzip mjpro150_linux.zip
RUN rm mjpro150_linux.zip
RUN wget https://www.roboti.us/download/mjpro131_linux.zip
RUN unzip mjpro131_linux.zip
RUN rm mjpro131_linux.zip
WORKDIR /

ENV DISPLAY=:10

COPY surreal-all-spec-file.txt /surreal-all-spec-file.txt
RUN conda install python=3.5
RUN conda install --file /surreal-all-spec-file.txt

COPY surreal-all-requirements.txt /surreal-all-requirements.txt
RUN pip install -r /surreal-all-requirements.txt \
	&& pip install git+git://github.com/deepmind/dm_control.git

# Note pytorch not installed

