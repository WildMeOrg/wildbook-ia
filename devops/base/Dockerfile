FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as org.wildme.wbia.base

MAINTAINER Wild Me <dev@wildme.org>

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

# Install requirements
RUN apt-get update \
 && apt-get install -y \
        ca-certificates \
        build-essential \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install developer tools
RUN apt-get update \
 && apt-get install -y \
        git \
        tmux \
        htop \
        ipython \
        ipython3 \
        python3-gdbm \
        graphviz \
        vim \
        unzip \
        xvfb \
 && rm -rf /var/lib/apt/lists/*

# Install Docker CE
RUN set -ex \
 && apt-get update \
 && apt-get install -y curl \
 && curl -fsSL https://get.docker.com -o get-docker.sh \
 && sh get-docker.sh \
 && apt-get purge -y curl \
 && rm -rf /var/lib/apt/lists/*
 
# Create wbia source location
RUN mkdir -p /wbia

# Create virtualenv location
RUN mkdir -p /virtualenv

# Install cmake and ninja
RUN /usr/bin/pip3 install cmake ninja

# Install CNMeM
RUN git clone https://github.com/NVIDIA/cnmem.git /wbia/cnmem \
 && cd /wbia/cnmem/ \
 && git checkout v1.0.0 \
 && mkdir -p /wbia/cnmem/build \
 && cd /wbia/cnmem/build \
 && cmake .. \
 && make -j4 \
 && make install \
 && cd .. \
 && rm -rf /wbia/cnmem

# Install virtualenv PyPI package
RUN /usr/bin/pip3 install \
    virtualenv==15.2.0

# Create virtualenvs for Python3
RUN virtualenv -p $(which python3) /virtualenv/env3

# Set CUDA-specific environment paths
ENV PATH "/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV CUDA_HOME "/usr/local/cuda"
