FROM wildme/wbia-base:latest as org.wildme.wbia.dependencies

ARG OPENCV_VERSION=3.4.10

# Install OpenCV
RUN set -ex \
 && cd /tmp \
 && git clone https://github.com/opencv/opencv.git \
 && git clone https://github.com/opencv/opencv_contrib.git \
 && cd /tmp/opencv \
 && git checkout $OPENCV_VERSION \
 && cd /tmp/opencv_contrib \
 && git checkout $OPENCV_VERSION \
 && rm -rf /tmp/opencv/build \
 && mkdir -p /tmp/opencv/build \
 && cd /tmp/opencv/build \
 && cmake -G "Unix Makefiles" \
        -D CMAKE_C_COMPILER=gcc \
        -D CMAKE_CXX_COMPILER=g++ \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/virtualenv/env3 \
     ############################################################
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_SHARED_LIBS=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_java=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_CREATE_DISTRIB=ON \
        -D BUILD_JPEG=ON \
        -D BUILD_HDR=ON \
        -D BUILD_TIFF=ON \
        -D WITH_MATLAB=OFF \
        -D WITH_TBB=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUBLAS=1 \
        -D WITH_EIGEN=ON \
        -D WITH_AVFOUNDATION=ON \
        -D WITH_JPEG=ON \
        -D WITH_HDR=ON \
        -D WITH_V4L=OFF \
        -D WITH_GDAL=ON \
        -D WITH_WIN32UI=OFF \
        -D WITH_QT=OFF \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
     ############################################################
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
     ############################################################
        .. \
 && make -j9 \
 && make install \
 && ldconfig \
 && cd .. \
 && rm -rf /tmp/*

RUN ln -s /virtualenv/env3/lib/python3.7/site-packages/cv2/python-3.7/cv2.cpython-37-x86_64-linux-gnu.so /virtualenv/env3/lib/python3.7/site-packages/cv2.so

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate && python3 -c "import cv2; print(cv2.getBuildInformation())"'


# Install libgpuarray (pygpu)
RUN set -ex \
 && . /virtualenv/env3/bin/activate \
 && git clone https://github.com/Theano/libgpuarray.git /wbia/libgpuarray \
 && cd /wbia/libgpuarray \
 && git checkout 04c2892 \
 && mkdir -p /wbia/libgpuarray/build \
 && cd /wbia/libgpuarray/build \
 && cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/virtualenv/env3 \
        .. \
 && make -j4 \
 && make install \
 && cd .. \
 && /virtualenv/env3/bin/python setup.py build_ext -L /virtualenv/env3/lib -I /virtualenv/env3/include \
 && /virtualenv/env3/bin/pip install -e . \
 && rm -rf /wbia/libgpuarray/build

# Install pygraphviz (for developer use)
RUN set -ex \
 && /virtualenv/env3/bin/pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
