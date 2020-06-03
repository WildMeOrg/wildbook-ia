FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 as com.wildme.wildbook-image-curation.base

MAINTAINER Jason Parham <parham@wildme.org>

# Install apt packages
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake=3.5.1-1ubuntu3 \
    git=1:2.7.4-0ubuntu1.3 \
    tmux=2.1-3build1 \
    ipython=2.4.1-1 \
    ipython3=2.4.1-1 \
    python=2.7.12-1~16.04 \
    python-dev=2.7.12-1~16.04 \
    python-pip=8.1.1-2ubuntu0.4 \
    python3=3.5.1-3 \
    python3-dev=3.5.1-3 \
    python3-pip=8.1.1-2ubuntu0.4 \
    graphviz=2.38.0-12ubuntu2.1 \
    graphviz-dev=2.38.0-12ubuntu2.1 \
    libeigen3-dev=3.3~beta1-2 \
    libfreetype6-dev=2.6.1-0.1ubuntu2.3 \
    libgdal-dev=1.11.3+dfsg-3build2 \
    libgl1-mesa-glx=17.2.8-0ubuntu0~16.04.1 \
    libgoogle-glog-dev=0.3.4-0.1 \
    libharfbuzz-dev=1.0.1-1ubuntu0.1 \
    libhdf5-dev=1.8.16+docs-4ubuntu1 \
    liblapack-dev=3.6.0-2ubuntu2 \
    liblapacke-dev=3.6.0-2ubuntu2 \
    libleptonica-dev=1.73-1 \
    libopenblas-dev=0.2.18-1ubuntu1 \
    libtbb-dev=4.4~20151115-0ubuntu3 \
    libtesseract-dev=3.04.01-4 \
    && rm -rf /var/lib/apt/lists/*

# Install CNMeM
RUN git clone https://github.com/NVIDIA/cnmem.git /src/cnmem \
    && cd /src/cnmem/ \
    && git checkout v1.0.0 \
    && mkdir -p /src/cnmem/build \
    && cd /src/cnmem/build \
    && cmake .. \
    && make -j4 \
    && make install \
    && cd .. \
    && rm -rf /src/cnmem

# Install virtualenv PyPI package
RUN /usr/bin/pip3 install \
    virtualenv==15.2.0

# Create virtualenv location
RUN mkdir -p /virtualenv

# Create virtualenvs for Python2 and Python3
RUN    virtualenv -p $(which python2) /virtualenv/env2 \
    && virtualenv -p $(which python3) /virtualenv/env3

# Set CUDA-specific environment paths
ENV PATH "/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV CUDA_HOME "/usr/local/cuda"

##########################################################################################
FROM com.wildme.wildbook-image-curation.base as com.wildme.wildbook-image-curation.build

RUN mkdir -p /root/.ssh

# Add deploy keys and SSH keys for Github with this container
ADD dev/keys/deploy-wbia /root/.ssh/deploy-wbia

ADD dev/keys/deploy-ibeis_cnn /root/.ssh/deploy-ibeis_cnn

ADD dev/keys/deploy-ibeis-flukematch-module /root/.ssh/deploy-ibeis-flukematch-module

ADD dev/keys/deploy-ibeis-curvrank-module /root/.ssh/deploy-ibeis-curvrank-module

RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN echo """\n\
Host wbia.github.com\n\
HostName github.com\n\
IdentityFile /root/.ssh/deploy-wbia\n\
User git\n\n\
Host ibeis_cnn.github.com\n\
HostName github.com\n\
IdentityFile /root/.ssh/deploy-ibeis_cnn\n\
User git\n\n\
Host ibeis-flukematch-module.github.com\n\
HostName github.com\n\
IdentityFile /root/.ssh/deploy-ibeis-flukematch-module\n\
User git\n\n\
Host ibeis-curvrank-module.github.com\n\
HostName github.com\n\
IdentityFile /root/.ssh/deploy-ibeis-curvrank-module\n\
User git\
""" >> /root/.ssh/config

RUN chmod -R 600 /root/.ssh/

# Clone
RUN    git clone https://github.com/opencv/opencv.git               /wbia/opencv \
    && git clone https://github.com/opencv/opencv_contrib.git       /wbia/opencv_contrib \
    && git clone https://github.com/Theano/libgpuarray.git          /wbia/libgpuarray \
    && git clone https://github.com/Theano/Theano.git               /wbia/Theano \
    && git clone https://github.com/Lasagne/Lasagne.git             /wbia/Lasagne \
    && git clone https://github.com/networkx/networkx.git           /wbia/networkx \
    && git clone https://github.com/WildbookOrg/utool.git           /wbia/utool \
    && git clone https://github.com/WildbookOrg/ubelt.git           /wbia/ubelt

RUN    git clone git@wbia.github.com:WildbookOrg/wbia.git         /wbia/wbia \
    && git clone git@ibeis_cnn.github.com:WildbookOrg/ibeis_cnn.git /wbia/ibeis_cnn \
    && git clone git@ibeis-flukematch-module.github.com:WildbookOrg/ibeis-flukematch-module.git /wbia/ibeis-flukematch-module \
    && git clone git@ibeis-curvrank-module.github.com:WildbookOrg/ibeis-curvrank-module.git /wbia/ibeis-curvrank-module

# Build and install utool and ubelt
RUN cd /wbia/utool \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/ubelt \
    && /virtualenv/env3/bin/pip install -e .

# Clone all other public wbia repositories and update
RUN cd /wbia/wbia \
    && /virtualenv/env3/bin/python super_setup.py --pull \
    && /virtualenv/env3/bin/python super_setup.py --checkout next \
    && /virtualenv/env3/bin/python super_setup.py --pull

# Install Numpy to Python2 environment for OpenCV build
RUN /virtualenv/env2/bin/pip install \
    numpy==1.14.2

# Install basic Python3 dependencies (and ones that were missing in IBEIS install)
RUN /virtualenv/env3/bin/pip install \
    numpy==1.14.2

# Install OpenCV
RUN    cd /wbia/opencv \
    && git checkout 3.4.0 \
    && cd /wbia/opencv_contrib \
    && git checkout 3.4.0 \
    && mkdir -p /wbia/opencv/build \
    && cd /wbia/opencv/build \
    && cmake -j4 \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/virtualenv/env3 \
        -D BUILD_TBB=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_java=OFF \
        -D WITH_MATLAB=OFF \
        -D WITH_TBB=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUBLAS=1 \
        -D WITH_GDAL=ON \
        -D PYTHON2_LIBRARY=/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so \
        -D PYTHON2_INCLUDE_DIR=/virtualenv/env2/include/python2.7 \
        -D PYTHON2_EXECUTABLE=/virtualenv/env2/bin/python \
        -D PYTHON2_PACKAGES_PATH=/virtualenv/env2/lib/python2.7/site-packages \
        -D PYTHON2_NUMPY_INCLUDE_DIRS=/virtualenv/env2/lib/python2.7/site-packages/numpy/core/include \
        -D PYTHON3_LIBRARY=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5m.so \
        -D PYTHON3_INCLUDE_DIR=/virtualenv/env3/include/python3.5m \
        -D PYTHON3_EXECUTABLE=/virtualenv/env3/bin/python \
        -D PYTHON3_PACKAGES_PATH=/virtualenv/env3/lib/python3.5/site-packages \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=/virtualenv/env3/lib/python3.5/site-packages/numpy/core/include \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=/wbia/opencv_contrib/modules \
        .. \
    && make -j4 \
    && make install \
    && cd .. \
    && rm -rf /wbia/opencv/build

RUN /virtualenv/env3/bin/pip install \
    cython==0.28.1

# Install libgpuarray (pygpu)
RUN    cd /wbia/libgpuarray \
    && git checkout 04c2892 \
    && mkdir -p /wbia/libgpuarray/build \
    && cd /wbia/libgpuarray/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/virtualenv/env3 \
    && make -j4 \
    && make install \
    && cd .. \
    && /virtualenv/env3/bin/python setup.py build_ext -L /virtualenv/env3/lib -I /virtualenv/env3/include \
    && /virtualenv/env3/bin/pip install -e . \
    && rm -rf /wbia/libgpuarray/build

# Install basic Python3 dependencies (and ones that were missing in IBEIS install)
RUN /virtualenv/env3/bin/pip install \
    ipython==6.2.1 \
    pyqt5==5.10.1 \
    futures_actors==0.0.5 \
    scikit-image==0.13.1 \
    mako \
    boto

# Install other PyPI dependencies
RUN    /virtualenv/env3/bin/pip install git+https://github.com/cameronbwhite/Flask-CAS.git@10ee70466ac9e71cec3602c1cd46f0566618f67e \
    && /virtualenv/env3/bin/pip install git+https://github.com/pwaller/pyfiglet.git@6dabdb0e720b5a61d81ff819faf0ad86127275fc \
    && /virtualenv/env3/bin/pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"

# Checkout specific versions of each repository
RUN    cd /wbia/wbia \
    && git checkout next \
    && /virtualenv/env3/bin/python super_setup.py --pull \
    # Double pull is important because super_setup.py could have just been updated as well
    && /virtualenv/env3/bin/python super_setup.py --pull \
    # Checkout specific version of wbia
    && git checkout 2cfb0ac4 \
    # Checkout specific version of all other repos
    && cd /wbia/Theano \
    && git checkout 61b514a \
    && cd /wbia/Lasagne \
    && git checkout 37ca134 \
    && cd /wbia/networkx \
    && git checkout networkx-2.1 \
    && cd /wbia/utool \
    && git checkout 6ec8e7b \
    && cd /wbia/vtool \
    && git checkout 123ef03 \
    && cd /wbia/dtool \
    && git checkout fcf7345 \
    && cd /wbia/ubelt \
    && git checkout c0eb1c7  \
    && cd /wbia/detecttools \
    && git checkout f885410 \
    && cd /wbia/plottool \
    && git checkout c3be3ec \
    && cd /wbia/guitool \
    && git checkout 5107bd0 \
    && cd /wbia/flann \
    && git checkout 6146453 \
    && cd /wbia/hesaff \
    && git checkout 28e8ef5 \
    && cd /wbia/ibeis_cnn \
    && git checkout 09d166c \
    && cd /wbia/pydarknet \
    && git checkout f36e935 \
    && cd /wbia/ibeis-flukematch-module \
    && git checkout fd1e028 \
    && cd /wbia/ibeis-curvrank-module \
    && git checkout 99b6115 \
    && cd /wbia/pyrf \
    && git checkout 8802d3a

# Install Python dependencies
RUN    cd /wbia/Theano \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/Lasagne \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/networkx \
    && /virtualenv/env3/bin/pip install -e .

# Build Python repositories with external codebases
RUN . /virtualenv/env3/bin/activate \
    && cd /wbia/flann \
    && /bin/bash unix_build.sh \
    && cd /wbia/hesaff \
    && /bin/bash unix_build.sh \
    && cd /wbia/ibeis-flukematch-module \
    && /bin/bash unix_build.sh \
    && cd /wbia/ibeis-curvrank-module \
    && /bin/bash unix_build.sh \
    && cd /wbia/pydarknet \
    && /bin/bash unix_build.sh \
    && cd /wbia/pyrf \
    && /bin/bash unix_build.sh \
    && cd /wbia/vtool \
    && /bin/bash unix_build.sh

# Install Python repositories
RUN    cd /wbia/detecttools \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/dtool \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/ibeis_cnn \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/guitool \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/hesaff \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/ibeis-flukematch-module \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/ibeis-curvrank-module \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/plottool \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/pydarknet \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/pyrf \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/ubelt \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/vtool \
    && /virtualenv/env3/bin/pip install -e . \
    && cd /wbia/wbia \
    && /virtualenv/env3/bin/pip install -e .

##########################################################################################
FROM com.wildme.wildbook-image-curation.base as com.wildme.wildbook-image-curation.install

COPY --from=com.wildme.wildbook-image-curation.build /virtualenv /virtualenv

COPY --from=com.wildme.wildbook-image-curation.build /wbia /wbia

RUN    ln -s /virtualenv/env3/lib/libgpuarray.so     /usr/lib/libgpuarray.so \
    && ln -s /virtualenv/env3/lib/libgpuarray.so.3   /usr/lib/libgpuarray.so.3 \
    && ln -s /virtualenv/env3/lib/libgpuarray.so.3.0 /usr/lib/libgpuarray.so.3.0

RUN echo """ \
[global]\n\
floatX = float32\n\
device = cuda0\n\
openmp = True\n\
allow_gc = True\n\
optimizer = fast_run\n\
enable_initial_driver_test = False\n\n\
[cuda]\n\
root = /usr/local/cuda/\n\n\
[lib]\n\
cnmem = 0.01\n\n\
[dnn]\n\
enabled = True\n\n\
[dnn.conv]\n\
algo_fwd = time_once\n\
algo_bwd_data = time_once\n\
algo_bwd_filter = time_once\n\n\
[nvcc]\n\
fastmath = True\n\
optimizer_including = cudnn\n\n\
[blas]\n\
ldflags = -llapack -lblas\n\
""" >> /root/.theanorc

RUN mkdir /data

RUN /virtualenv/env3/bin/python /wbia/wbia/dev.py --set-workdir /data

##########################################################################################
FROM com.wildme.wildbook-image-curation.install as com.wildme.wildbook-image-curation.test

RUN /virtualenv/env3/bin/python /wbia/wbia/reset_dbs.py --reset-all

VOLUME /data

ENTRYPOINT ["/virtualenv/env3/bin/python", "/wbia/wbia/run_tests.py"]

##########################################################################################
FROM com.wildme.wildbook-image-curation.install as com.wildme.wildbook-image-curation.deploy

ENTRYPOINT ["/virtualenv/env3/bin/python", "/wbia/wbia/dev.py"]

CMD ["--dbdir", "/data/container", "--web", "--port", "5000", "--web-deterministic-ports"]

# Ports for the frontend web server
EXPOSE 5000

# Ports for the backend job engine
EXPOSE 51381

EXPOSE 51382

EXPOSE 51383

EXPOSE 51384

EXPOSE 51385

STOPSIGNAL SIGTERM
