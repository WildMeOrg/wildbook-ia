FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as org.wildme.wbia.base

MAINTAINER Wild Me <dev@wildme.org>

ENV LC_ALL C.UTF-8

ENV LANG C.UTF-8

# Install package updates and dependencies
RUN set -ex \
 && apt update \
 && apt upgrade -y \
 && apt dist-upgrade -y \
 && apt clean \
 && apt autoclean \
 && apt autoremove -y \
 && apt install -y \
        # Install requirements - Pegged to Version 3.7 to support TensorFlow 1.15.3
        ca-certificates \
        build-essential \
        pkg-config \
        python3.7 \
        python3.7-dev \
        python3.7-gdbm \
        python3-pip \
        # Install developer tools
        curl \
        git \
        htop \
        locate \
        tmux \
        unzip \
        vim \
        wget \
        xvfb \
 && rm -rf /var/lib/apt/lists/*

# Install Docker CE
RUN set -ex \
 && curl -fsSL https://get.docker.com -o get-docker.sh \
 && sh get-docker.sh

# Create wbia source location
RUN mkdir -p /wbia

# Create virtualenv location
RUN mkdir -p /virtualenv

# Install Python Virtual Environments
RUN set -ex \
 && /usr/bin/pip3 install virtualenv

# Create virtualenvs for Python3
RUN set -ex \
 && virtualenv -p $(which python3.7) /virtualenv/env3

# Install PyPI packages
RUN set -ex \
 && /virtualenv/env3/bin/pip install --upgrade pip \
 && /virtualenv/env3/bin/pip install --upgrade cmake ninja scikit-build setuptools

# Install CNMeM
RUN set -ex \
 && . /virtualenv/env3/bin/activate \
 && git clone https://github.com/NVIDIA/cnmem.git /tmp/cnmem \
 && cd /tmp/cnmem/ \
 && git checkout v1.0.0 \
 && mkdir -p /tmp/cnmem/build \
 && cd /tmp/cnmem/build \
 && cmake .. \
 && make -j4 \
 && make install \
 && cd .. \
 && rm -rf /tmp/cnmem

# Install NVTOP
RUN set -ex \
 && . /virtualenv/env3/bin/activate \
 && apt update \
 && apt install -y libncurses5-dev libncursesw5-dev \
 && git clone https://github.com/Syllo/nvtop.git /tmp/nvtop \
 && cd /tmp/nvtop/ \
 && mkdir -p /tmp/nvtop/build \
 && cd /tmp/nvtop/build \
 && cmake .. \
 && make -j4 \
 && make install \
 && cd .. \
 && rm -rf /tmp/nvtop \
 && rm -rf /var/lib/apt/lists/*

# Set CUDA-specific environment paths
ENV PATH "/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:/virtualenv/env3/lib:${LD_LIBRARY_PATH}"

ENV CUDA_HOME "/usr/local/cuda"
