ARG WBIA_UBUNTU_IMAGE=nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04

FROM ${WBIA_UBUNTU_IMAGE} as org.wildme.wbia.base

MAINTAINER Wild Me <dev@wildme.org>

ENV LC_ALL C.UTF-8

ENV LANG C.UTF-8

# Set up xvfb for running gui doctests
ENV DISPLAY :1

USER root

# Fix for Arm64 builds on GitHub actions
RUN set -ex \
 && ln -s /usr/bin/dpkg-split /usr/sbin/dpkg-split \
 && ln -s /usr/bin/dpkg-deb /usr/sbin/dpkg-deb \
 && ln -s /bin/rm /usr/sbin/rm \
 && ln -s /bin/tar /usr/sbin/tar

# Install package updates and dependencies
RUN set -ex \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        apt-utils \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get upgrade -y \
 # && apt-get dist-upgrade -y \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Install build requirements
        ca-certificates \
        build-essential \
        lsb-release \
        pkg-config \
        # Install Python 3.7
        python3.7 \
        python3.7-dev \
        python3.7-gdbm \
        python3.7-distutils \
        python3-pip \
        python3-setuptools \
        python3-venv \
        # Install OpenCV
        libopencv-dev \
        # Install dependencies for PyFLANN
        libboost-all-dev \
        libopenmpi-dev \
        libomp5 \
        libomp-dev \
        libhdf5-openmpi-dev \
        liblz4-dev \
        # Install dependencies for PyHesaff
        libgdal-dev \
        # Install dependencies for PyDarknet
        libtbb-dev \
        # Install dependencies for vtool
        libeigen3-dev \
        # Install dependencies for Python packages
        libgeos-dev \
        libgdal-dev \
        libproj-dev \
        graphviz \
        graphviz-dev \
        python3-pyqt5 \
        # Install dependencies for pgloader
        libssl-dev \
        # Install handy developer tools
        coreutils \
        curl \
        git \
        gnupg \
        gosu \
        htop \
        locate \
        netcat \
        nvtop \
        pgloader \
        postgresql \
        rsync \
        tmux \
        unzip \
        vim \
        wget \
        xvfb \
 && apt-get upgrade \
 && apt-get clean \
 && apt-get autoclean \
 && apt-get autoremove -y \
 && rm -rf /var/cache/apt \
 && rm -rf /var/lib/apt/lists/*

RUN set -ex \
 update-alternatives --set hdf5.pc /usr/lib/*-linux-gnu/pkgconfig/hdf5-openmpi.pc

RUN unlink /etc/localtime

RUN ln -s /usr/share/zoneinfo/America/Los_Angeles /etc/localtime

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
 && virtualenv --system-site-packages -p $(which python3.7) /virtualenv/env3

# Install PyPI packages
RUN set -ex \
 && /virtualenv/env3/bin/pip install --upgrade \
        pip \
 && /virtualenv/env3/bin/pip install --upgrade \
        # Install build requirements
        cmake \
        ninja \
        scikit-build \
        setuptools \
        setuptools_scm[toml] \
        cython \
        # Install handy developer tools
        ipython

RUN apt-get remove -y \
        python3-setuptools

 # Install wait-for
RUN set -ex \
    && curl -s https://raw.githubusercontent.com/eficode/wait-for/v2.0.0/wait-for > /usr/local/bin/wait-for \
    && chmod a+x /usr/local/bin/wait-for \
    # test it works
    && wait-for google.com:80 -- echo "success"

# Set CUDA-specific environment paths
ENV PATH "/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:/virtualenv/env3/lib:${LD_LIBRARY_PATH}"

ENV CUDA_HOME "/usr/local/cuda"

ENV CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda"

CMD ["/bin/bash", "-c", "Xvfb :1 -screen 0 1024x768x16 &>/tmp/xvfb.log & /bin/bash"]
