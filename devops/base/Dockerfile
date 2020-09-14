FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as org.wildme.wbia.base

MAINTAINER Wild Me <dev@wildme.org>

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
        ipython \
        ipython3 \
        python3-gdbm \
        git \
        graphviz \
        htop \
        tmux \
        unzip \
        vim \
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
RUN git clone https://github.com/NVIDIA/cnmem.git /tmp/cnmem \
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
 && apt-get update \
 && apt-get install -y slibncurses5-dev libncursesw5-dev \
 && git clone https://github.com/Syllo/nvtop.git /tmp/nvtop \
 && cd /tmp/nvtop/ \
 && mkdir -p /tmp/nvtop/build \
 && cd /tmp/nvtop/build \
 && cmake .. \
 && make -j4 \
 && make install \
 && cd .. \
 && rm -rf /tmp/nvtop \
 && apt-get purge -y slibncurses5-dev libncursesw5-dev \
 && rm -rf /var/lib/apt/lists/*

# Install virtualenv PyPI package
RUN /usr/bin/pip3 install \
    virtualenv==15.2.0

# Create virtualenvs for Python3
RUN virtualenv -p $(which python3) /virtualenv/env3

# Set CUDA-specific environment paths
ENV PATH "/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV CUDA_HOME "/usr/local/cuda"
