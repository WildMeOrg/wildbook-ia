FROM wildme.azurecr.io/ibeis/base:latest as org.wildme.ibeis.dependencies

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

# Clone other Github repositories
RUN git clone https://github.com/opencv/opencv.git                        /ibeis/opencv \
 && git clone https://github.com/opencv/opencv_contrib.git                /ibeis/opencv_contrib \
 && git clone https://github.com/Theano/libgpuarray.git                   /ibeis/libgpuarray \
 && git clone https://github.com/Theano/Theano.git                        /ibeis/Theano \
 && git clone https://github.com/Lasagne/Lasagne.git                      /ibeis/Lasagne \
 && git clone https://github.com/networkx/networkx.git                    /ibeis/networkx

# Install basic Python3 dependencies (and ones that were missing in IBEIS install)
RUN /virtualenv/env3/bin/pip install \
    numpy==1.17.0 \
    ipython==5.5.0

# Install OpenCV
RUN . /virtualenv/env3/bin/activate \
 && cd /ibeis/opencv \
 && git checkout 3.4.4 \
 && cd /ibeis/opencv_contrib \
 && git checkout 3.4.4 \
 && rm -rf /ibeis/opencv/build \
 && mkdir -p /ibeis/opencv/build \
 && cd /ibeis/opencv/build \
 && cmake \
     -D CMAKE_C_COMPILER=gcc \
     -D CMAKE_CXX_COMPILER=g++ \
     -D CMAKE_BUILD_TYPE=RELEASE \
     -D CMAKE_INSTALL_PREFIX=/virtualenv/env3 \
     -D ENABLE_PRECOMPILED_HEADERS=OFF \
     -D BUILD_DOCS=OFF \
     -D BUILD_EXAMPLES=ON \
     -D BUILD_opencv_java=OFF \
     -D BUILD_opencv_python3=ON \
     -D BUILD_NEW_PYTHON_SUPPORT=ON \
     -D INSTALL_C_EXAMPLES=OFF \
     -D INSTALL_PYTHON_EXAMPLES=ON \
     -D BUILD_JPEG=ON \
     -D BUILD_HDR=ON \
     -D WITH_MATLAB=OFF \
     -D WITH_TBB=ON \
     -D WITH_CUDA=ON \
     -D WITH_CUBLAS=1 \
     -D WITH_EIGEN=ON \
     -D WITH_AVFOUNDATION=ON \
     -D WITH_JPEG=ON \
     -D WITH_HDR=ON \
     -D WITH_V4L=ON \
     -D WITH_GDAL=ON \
     -D PYTHON_EXECUTABLE=/virtualenv/env3/bin/python \
     -D PYTHON_INCLUDE_DIR=/virtualenv/env3/include/python3.6m \
     -D PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
     -D PYTHON_PACKAGES_PATH=/virtualenv/env3/lib/python3.6/site-packages \
     -D PYTHON_NUMPY_INCLUDE_DIRS=/virtualenv/env3/lib/python3.6/site-packages/numpy/core/include \
     -D PYTHON3_EXECUTABLE=/virtualenv/env3/bin/python \
     -D PYTHON3_INCLUDE_DIR=/virtualenv/env3/include/python3.6m \
     -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
     -D PYTHON3_PACKAGES_PATH=/virtualenv/env3/lib/python3.6/site-packages \
     -D PYTHON3_NUMPY_INCLUDE_DIRS=/virtualenv/env3/lib/python3.6/site-packages/numpy/core/include \
     -D ENABLE_FAST_MATH=1 \
     -D CUDA_FAST_MATH=1 \
     -D OPENCV_ENABLE_NONFREE=ON \
     -D OPENCV_EXTRA_MODULES_PATH=/ibeis/opencv_contrib/modules \
     .. \
 && make -j2 \
 && make install \
 && ldconfig \
 && cd .. \
 && rm -rf /ibeis/opencv/build

RUN ln -s /virtualenv/env3/python/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so /virtualenv/env3/lib/python3.6/site-packages/cv2.so

RUN /virtualenv/env3/bin/pip install \
    cython==0.28.1

# Install libgpuarray (pygpu)
RUN cd /ibeis/libgpuarray \
 && git checkout 04c2892 \
 && mkdir -p /ibeis/libgpuarray/build \
 && cd /ibeis/libgpuarray/build \
 && cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/virtualenv/env3 \
    .. \
 && make -j4 \
 && make install \
 && cd .. \
 && /virtualenv/env3/bin/python setup.py build_ext -L /virtualenv/env3/lib -I /virtualenv/env3/include \
 && /virtualenv/env3/bin/pip install -e . \
 && rm -rf /ibeis/libgpuarray/build

# Install basic Python3 dependencies (and ones that were missing in IBEIS install)
RUN /virtualenv/env3/bin/pip install \
    ipython==6.2.1 \
    pyqt5==5.10.1 \
    Pillow==6.2.1 \
    futures_actors==0.0.5 \
    scikit-image==0.15.0 \
    mako==1.0.7 \
    boto==2.49.0 \
    colorama==0.3.9 \
    jupyter==1.0.0 \
    ansi2html==1.5.2 \
    shapely==1.6.4.post2 \
    pyshp==2.1.0 \
    setproctitle==1.1.10 \
    matplotlib==3.0.2 \
    torch==1.0.1.post2 \
    torchvision==0.2.1 \
    flask_swagger==0.2.13 \
    PyQt5-sip==4.19.13 \
    sip==4.19.8 \
    tqdm==4.26.0 \
    pytesseract==0.2.6 \
    line-profiler==2.1.2 \
    gdm==0.8.2 \
    prometheus-client==0.5.0 \
    docker==4.0.2

RUN /virtualenv/env3/bin/pip install git+https://github.com/aleju/imgaug \
 && /virtualenv/env3/bin/pip install git+https://github.com/cameronbwhite/Flask-CAS.git@10ee70466ac9e71cec3602c1cd46f0566618f67e \
 && /virtualenv/env3/bin/pip install git+https://github.com/pwaller/pyfiglet.git@6dabdb0e720b5a61d81ff819faf0ad86127275fc \
 && /virtualenv/env3/bin/pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"

# Install Python dependencies
RUN cd /ibeis/Theano \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/Lasagne \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/networkx \
 && /virtualenv/env3/bin/pip install -e .
