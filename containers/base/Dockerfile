FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04 as org.wildme.wbia.base
# wbia == WildBook IA

MAINTAINER Wild Me <dev@wildme.org>

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

# Create runtime user "wbia" (aka WildBook IA)
RUN set -x \
    && addgroup --system wbia \
    && adduser --system --group wbia

# Install base system dependencies via APT packages
# ??? How many of these are NTH vs required?
# ??? Why pin to specific versions? Do we not trust upstream?
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential=12.4ubuntu1 \
        pkg-config=0.29.1-0ubuntu2 \
        libtbb2=2017~U7-8 \
        libtbb-dev=2017~U7-8 \
        cmake \
        git \
        wget \
        tmux=2.6-3 \
        locate=4.6.0+git+20170828-2 \
        htop=2.1.0-3 \
        ipython=5.5.0-1 \
        ipython3=5.5.0-1 \
        python=2.7.15~rc1-1 \
        python-dev=2.7.15~rc1-1 \
        python-pip=9.0.1-2.3~ubuntu1.18.04.1 \
        python3=3.6.7-1~18.04 \
        python3-dev=3.6.7-1~18.04 \
        python3-pip=9.0.1-2.3~ubuntu1.18.04.1 \
        python3-gdbm=3.6.9-1~18.04 \
        graphviz=2.40.1-2 \
        libeigen3-dev=3.3.4-4 \
        libgraphviz-dev=2.40.1-2 \
        libfreetype6-dev=2.8.1-2ubuntu2 \
        libgdal-dev=2.2.3+dfsg-2 \
        libgl1-mesa-glx \
        libgoogle-glog-dev=0.3.5-1 \
        libharfbuzz-dev=1.7.2-1ubuntu1 \
        libhdf5-dev=1.10.0-patch1+docs-4 \
        liblapack-dev=3.7.1-4ubuntu1 \
        liblapacke-dev=3.7.1-4ubuntu1 \
        libleptonica-dev=1.75.3-3 \
        libopenblas-dev=0.2.20+ds-4 \
        libtbb-dev=2017~U7-8 \
        libtesseract-dev=4.00~git2288-10f4998a-2 \
        vim=2:8.0.1453-1ubuntu1.3 \
        libv4l-dev=1.14.2-1 \
        libavcodec-dev=7:3.4.6-0ubuntu0.18.04.1 \
        libavformat-dev=7:3.4.6-0ubuntu0.18.04.1 \
        libswscale-dev=7:3.4.6-0ubuntu0.18.04.1 \
        libavresample-dev=7:3.4.6-0ubuntu0.18.04.1 \
        ffmpeg=7:3.4.6-0ubuntu0.18.04.1 \
        libpng-dev=1.6.34-1ubuntu0.18.04.2 \
        libjpeg-dev=8c-2ubuntu8 \
        libopenjp2-7-dev=2.3.0-2build0.18.04.1 \
        libopenexr22=2.2.0-11.1ubuntu1 \
        libtiff-dev=4.0.9-5 \
        libwebp-dev=0.6.1-2 \
        libdc1394-22-dev=2.2.5-1 \
        unzip=6.0-21ubuntu1 \
        libxvidcore-dev=2:1.3.5-1 \
        libx264-dev=2:0.152.2854+gite9a5903-2 \
        libgtk-3-dev=3.22.30-1ubuntu4 \
        libatlas-base-dev=3.10.3-5 \
        gfortran=4:7.4.0-1ubuntu2.3 \
        tesseract-ocr=4.00~git2288-10f4998a-2 \
        tesseract-ocr-eng=4.00~git24-0e00fe6-1.2 \
    && rm -rf /var/lib/apt/lists/*
# RUN set -x \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends \
#         ca-certificates \
#         build-essential=12.4ubuntu1 \
#         pkg-config=0.29.1-0ubuntu2 \
#         libtbb2=2017~U7-8 \
#         libtbb-dev=2017~U7-8 \
#         cmake \
#         git \
#         wget \
#         locate=4.6.0+git+20170828-2 \
#         htop=2.1.0-3 \
#         python-dev=2.7.15~rc1-1 \
#         python-pip=9.0.1-2.3~ubuntu1.18.04.1 \
#         python3 \
#         python3-dev \
#         python3-pip \
#         python3-gdbm \
#         graphviz=2.40.1-2 \
#         libeigen3-dev=3.3.4-4 \
#         libgraphviz-dev=2.40.1-2 \
#         libfreetype6-dev=2.8.1-2ubuntu2 \
#         libgdal-dev=2.2.3+dfsg-2 \
#         libgl1-mesa-glx \
#         libgoogle-glog-dev=0.3.5-1 \
#         libharfbuzz-dev=1.7.2-1ubuntu1 \
#         libhdf5-dev=1.10.0-patch1+docs-4 \
#         liblapack-dev=3.7.1-4ubuntu1 \
#         liblapacke-dev=3.7.1-4ubuntu1 \
#         libleptonica-dev=1.75.3-3 \
#         libopenblas-dev=0.2.20+ds-4 \
#         libtbb-dev=2017~U7-8 \
#         libtesseract-dev=4.00~git2288-10f4998a-2 \
#         libv4l-dev=1.14.2-1 \
#         libavcodec-dev=7:3.4.6-0ubuntu0.18.04.1 \
#         libavformat-dev=7:3.4.6-0ubuntu0.18.04.1 \
#         libswscale-dev=7:3.4.6-0ubuntu0.18.04.1 \
#         libavresample-dev=7:3.4.6-0ubuntu0.18.04.1 \
#         ffmpeg=7:3.4.6-0ubuntu0.18.04.1 \
#         libpng-dev=1.6.34-1ubuntu0.18.04.2 \
#         libjpeg-dev=8c-2ubuntu8 \
#         libopenjp2-7-dev=2.3.0-2build0.18.04.1 \
#         libopenexr22=2.2.0-11.1ubuntu1 \
#         libtiff-dev=4.0.9-5 \
#         libwebp-dev=0.6.1-2 \
#         libdc1394-22-dev=2.2.5-1 \
#         unzip \
#         libxvidcore-dev=2:1.3.5-1 \
#         libx264-dev=2:0.152.2854+gite9a5903-2 \
#         libgtk-3-dev=3.22.30-1ubuntu4 \
#         libatlas-base-dev=3.10.3-5 \
#         gfortran=4:7.4.0-1ubuntu2.3 \
#         tesseract-ocr=4.00~git2288-10f4998a-2 \
#         tesseract-ocr-eng=4.00~git24-0e00fe6-1.2 \
#     && rm -rf /var/lib/apt/lists/*

# Install Docker
# ??? DinD... is this wise?
RUN set -x \
    && wget -O /tmp/get-docker.sh https://get.docker.com \
    && sh /tmp/get-docker.sh \
    && rm -rf /tmp/get-docker.sh \
    && rm -rf /var/lib/apt/lists/* 

# Install CNMeM, a memory manager for CUDA
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

# Set CUDA-specific environment paths
ENV PATH "/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME "/usr/local/cuda"

# Create the working directory
RUN set -x \
    && mkdir -p /runtime\
    && chown wbia:wbia /runtime \
    && chmod 755 /runtime

# Change to the runtime user
USER wbia

WORKDIR /runtime
