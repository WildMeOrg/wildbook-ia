ARG WBIA_UBUNTU_IMAGE=nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

FROM ${WBIA_UBUNTU_IMAGE} as org.wildme.wbia.base

MAINTAINER Wild Me <dev@wildme.org>

ENV LC_ALL C.UTF-8

ENV LANG C.UTF-8

# Set up xvfb for running gui doctests
ENV DISPLAY :1

USER root

# Install package updates and dependencies
RUN set -ex \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 # && apt-get upgrade -y \
 # && apt-get dist-upgrade -y \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        # Install build requirements
        ca-certificates \
        build-essential \
        lsb-release \
        pkg-config \
        # Install Python 3.7
        python3.7 \
        python3.7-dev \
        python3.7-gdbm \
        python3-pip \
        python3-pyqt5 \
        python3-setuptools \
        python3-venv \
        # Install OpenCV
        libopencv-dev \
        # Install dependencies for NVTOP
        libncurses5-dev \
        libncursesw5-dev \
        # Install dependencies for PyHesaff & PyFLANN
        libboost-all-dev \
        libopenmpi-dev \
        libomp-dev \
        libhdf5-openmpi-dev \
        libgdal-dev \
        libgflags-dev \
        libeigen3-dev \
        liblz4-dev \
        libssl-dev \
        libgraphviz-dev \
        # Install run-time tools
        graphviz \
        gosu \
        xvfb \
        # Install handy developer tools
        curl \
        pgloader \
        git \
        gnupg \
        htop \
        locate \
        netcat \
        rsync \
        tmux \
        unzip \
        vim \
        wget \
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

# Install NVTOP
RUN set -ex \
 && . /virtualenv/env3/bin/activate \
 && git clone https://github.com/Syllo/nvtop.git /tmp/nvtop \
 && cd /tmp/nvtop/ \
 && mkdir -p /tmp/nvtop/build \
 && cd /tmp/nvtop/build \
 && cmake .. \
 && make -j4 \
 && make install \
 && cd .. \
 && rm -rf /tmp/nvtop

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
