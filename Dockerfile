# Start with cuDNN base image
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Yohei Kikuta <yohei-kikuta@cookpad.com>

# Install git, wget, bc and dependencies
RUN apt-get update && apt-get install -y \
  git \
  iproute2 \
  wget \
  python3.5 \
  python3-pip \
  python3-dev

# Install tensorflow and basics
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Install Keras and its dependencies
RUN pip3 install h5py \
    keras

RUN useradd docker -u 1001 -s /bin/bash -m
USER docker

# Set alias for python3.5
RUN echo "alias python=python3" >> $HOME/.bashrc && \
    echo "alias pip=pip3" >> $HOME/.bashrc

# Set working directory
WORKDIR /work

ENTRYPOINT ["/bin/bash"]
