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
ENV TF_PYTHON_URL https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl
RUN pip3 install $TF_PYTHON_URL \
    jupyter \
    pandas \
    scikit-learn \
    matplotlib \
    scikit-image \
    pillow \
    tqdm

# Install Keras and its dependencies
RUN pip3 install h5py \
    keras

# Install miscs
# RUN pip3 install awscli akagi libmysqlclient-dev

#### Miscellaneous items for Kikuta
#RUN apt-get install -y vim && \
#    git config --global core.editor 'vim -c "set fenc=utf-8"' && \
#    pip3 install jupyterthemes

# assume ubuntu user id is 1000.
RUN useradd docker -u 1000 -s /bin/bash -m
USER docker

# Set alias for python3.5
RUN echo "alias python=python3" >> $HOME/.bashrc && \
    echo "alias pip=pip3" >> $HOME/.bashrc

# Set working directory
WORKDIR /work

ENTRYPOINT ["/bin/bash"]
