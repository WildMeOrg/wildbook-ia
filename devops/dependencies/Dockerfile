ARG WBIA_BASE_IMAGE=wildme/wbia-base:latest
ARG OPENCV_VERSION=3.4.11

FROM $WBIA_BASE_IMAGE as org.wildme.wbia.dependencies

# Install OpenCV
RUN set -ex \
 && . /virtualenv/env3/bin/activate \
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
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D BUILD_SHARED_LIBS=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_opencv_freetype=OFF \
        -D BUILD_opencv_hdf=OFF \
        -D BUILD_opencv_java=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_CREATE_DISTRIB=ON \
        -D BUILD_ZLIB=ON \
        -D BUILD_JPEG=ON \
        -D BUILD_WEBP=ON \
        -D BUILD_PNG=ON \
        -D BUILD_TIFF=ON \
        -D BUILD_JASPER=ON \
        -D BUILD_OPENEXR=ON \
        -D WITH_MATLAB=OFF \
        -D WITH_TBB=OFF \
        -D WITH_CUDA=OFF \
        -D WITH_CUBLAS=0 \
        -D WITH_EIGEN=ON \
        -D WITH_1394=OFF \
        -D WITH_FFMPEG=OFF \
        -D WITH_GSTREAMER=OFF \
        -D WITH_V4L=OFF \
        -D WITH_AVFOUNDATION=OFF \
        -D WITH_TESSERACT=OFF \
        -D WITH_HDR=ON \
        -D WITH_GDAL=OFF \
        -D WITH_WIN32UI=OFF \
        -D WITH_QT=OFF \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON3_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
        -D PYTHON3_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
        -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        .. \
 && make -j4 \
 && make install \
 && ldconfig \
 && cd .. \
 && rm -rf /tmp/*

# RUN ln -s /virtualenv/env3/lib/python3.7/site-packages/cv2/python-3.7/cv2.cpython-37-x86_64-linux-gnu.so /virtualenv/env3/lib/python3.7/site-packages/cv2.so

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate && python3 -c "import cv2; print(cv2.getBuildInformation()); print(cv2.__version__)"'

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

# Fix install pygraphviz (for developer use)
RUN set -ex \
 && /virtualenv/env3/bin/pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
