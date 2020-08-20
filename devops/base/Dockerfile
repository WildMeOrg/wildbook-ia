FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as org.wildme.wbia.base

MAINTAINER Wild Me <dev@wildme.org>

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

# Install apt packages
RUN apt-get update \
 && apt-get install -y \
        ca-certificates \
        build-essential \
        pkg-config \
        libtbb2 \
        libtbb-dev \
        git \
        tmux \
        locate \
        htop \
        ipython \
        ipython3 \
        python \
        python-dev \
        python-pip \
        python3 \
        python3-dev \
        python3-pip \
        python3-gdbm \
        graphviz \
        libeigen3-dev \
        libgraphviz-dev \
        libfreetype6-dev \
        libgdal-dev \
        libgl1-mesa-glx \
        libgoogle-glog-dev \
        libharfbuzz-dev \
        libhdf5-dev \
        liblapack-dev \
        liblapacke-dev \
        libleptonica-dev \
        libopenblas-dev \
        libtbb-dev \
        libtesseract-dev \
        vim \
        libv4l-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libavresample-dev \
        ffmpeg \
        libpng-dev \
        libjpeg-dev \
        libopenjp2-7-dev \
        libopenexr22 \
        libtiff-dev \
        libwebp-dev \
        libdc1394-22-dev \
        unzip \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        tesseract-ocr \
        tesseract-ocr-eng \
        liblz4-dev \
        libhdf5-serial-dev \
        xvfb \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://get.docker.com -o get-docker.sh \
 && sh get-docker.sh

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
