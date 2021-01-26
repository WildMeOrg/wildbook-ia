
export CODE=/home/$( whoami )/code

export VENV=/home/$( whoami )/virtualenv/wildme3.7

export LC_ALL=C.UTF-8

export LANG=C.UTF-8

export OPENCV_VERSION=3.4.11

# Delete any old venvs
# rm -rf ~/virtualenv/test*
# rm -rf ~/virtualenv/wildme*

# Delete all old code
# rm -rf ~/code/brambox
# rm -rf ~/code/detecttools
# rm -rf ~/code/dtool
# rm -rf ~/code/flann
# rm -rf ~/code/guitool
# rm -rf ~/code/hesaff
# rm -rf ~/code/ibeis
# rm -rf ~/code/ibeis_cnn
# rm -rf ~/code/ibeis-*
# rm -rf ~/code/Lasagne
# rm -rf ~/code/libgpuarray
# rm -rf ~/code/lightnet
# rm -rf ~/code/networkx
# rm -rf ~/code/opencv
# rm -rf ~/code/opencv_contrib
# rm -rf ~/code/plottool
# rm -rf ~/code/pydarknet
# rm -rf ~/code/pyrf
# rm -rf ~/code/Theano
# rm -rf ~/code/ubelt
# rm -rf ~/code/utool
# rm -rf ~/code/vtool
# rm -rf ~/code/wbia-*

# sudo apt update
# sudo apt install \
#     ca-certificates \
#     build-essential \
#     pkg-config \
#     python3.7 \
#     python3.7-dev \
#     python3.7-gdbm \
#     python3-pip \
#     libncurses5-dev \
#     libncursesw5-dev \
#     libgflags-dev \
#     libgflags-doc \
#     libeigen3-dev \
#     libgtk2.0-dev \
#     libhdf5-serial-dev \
#     liblz4-dev \
#     graphviz \
#     libgraphviz-dev \
#     libopenblas-dev \
#     curl \
#     git \
#     htop \
#     locate \
#     tmux \
#     unzip \
#     vim \
#     wget

pip3 install virtualenv

virtualenv -p $(which python3.7) ${VENV}

source ${VENV}/bin/activate

pip install --upgrade pip
pip install --upgrade \
    'cmake!=3.18.2' \
    ninja \
    scikit-build \
    'setuptools>=42' \
    'setuptools_scm[toml]>=3.4' \
    cython \
    numpy \
    ipython

# Install OpenCV
cd ${CODE}
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd ${CODE}/opencv
git checkout $OPENCV_VERSION
cd ${CODE}/opencv_contrib
git checkout $OPENCV_VERSION
rm -rf ${CODE}/opencv/build
mkdir -p ${CODE}/opencv/build
cd ${CODE}/opencv/build

cmake -G "Unix Makefiles" \
    -D CMAKE_C_COMPILER=gcc \
    -D CMAKE_CXX_COMPILER=g++ \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${VIRTUAL_ENV} \
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
    -D OPENCV_EXTRA_MODULES_PATH=${CODE}/opencv_contrib/modules \
    ..

make -j20
make install

python3 -c "import cv2; print(cv2.getBuildInformation()); print(cv2.__version__)"

cd ${CODE}
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
git checkout 04c2892
mkdir -p build
cd build

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=${VIRTUAL_ENV} \
    ..

make -j20
make install

cd ..
python setup.py build_ext -L ${VIRTUAL_ENV}/lib -I ${VIRTUAL_ENV}/include
pip install -e .

pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"

cp -r ${VIRTUAL_ENV}/lib/python3.7/site-packages/cv2 /tmp/cv2

cd ${CODE}
git clone --branch develop https://github.com/WildMeOrg/wildbook-ia.git
git clone --branch develop https://github.com/WildMeOrg/wbia-utool.git
git clone --branch develop https://github.com/WildMeOrg/wbia-vtool.git
git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pyhesaff.git
git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pyflann.git
git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pydarknet.git
git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pyrf.git
git clone --branch develop https://github.com/WildMeOrg/wbia-deprecate-tpl-brambox
git clone --branch develop https://github.com/WildMeOrg/wbia-deprecate-tpl-lightnet
git clone --recursive --branch develop https://github.com/WildMeOrg/wbia-plugin-cnn.git
git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-flukematch.git
git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-finfindr.git
git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-deepsense.git
git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-pie.git

cd ${CODE}
git clone --recursive --branch develop https://github.com/WildMeOrg/wbia-plugin-curvrank.git
cd wbia-plugin-curvrank/wbia_curvrank
git fetch origin
git checkout develop

cd ${CODE}
git clone --recursive --branch develop https://github.com/WildMeOrg/wbia-plugin-kaggle7.git
cd wbia-plugin-kaggle7/wbia_kaggle7
git fetch origin
git checkout develop

cd ${CODE}
git clone --recursive --branch develop https://github.com/WildMeOrg/wbia-plugin-lca.git

cd ${CODE}/wbia-utool
./run_developer_setup.sh

cd ${CODE}/wbia-vtool
./run_developer_setup.sh

cd ${CODE}/wbia-tpl-pyhesaff
./run_developer_setup.sh

cd ${CODE}/wbia-tpl-pyflann
./run_developer_setup.sh

cd ${CODE}/wbia-tpl-pydarknet
./run_developer_setup.sh

cd ${CODE}/wbia-tpl-pyrf
./run_developer_setup.sh

cd ${CODE}/wbia-deprecate-tpl-brambox
pip install -e .

cd ${CODE}/wbia-deprecate-tpl-lightnet
pip install -r develop.txt
pip install -e .

cd ${CODE}/wildbook-ia
./run_developer_setup.sh

cd ${CODE}/wbia-plugin-cnn
./run_developer_setup.sh

cd ${CODE}/wbia-plugin-pie
./run_developer_setup.sh

cd ${CODE}/wbia-plugin-finfindr
pip install -e .

cd ${CODE}/wbia-plugin-deepsense
pip install -e .

cd ${CODE}/wbia-plugin-kaggle7
pip install -e .

cd ${CODE}/wbia-plugin-lca
pip install -e .

cd ${CODE}/wbia-plugin-flukematch
./unix_build.sh
pip install -e .

cd ${CODE}/wbia-plugin-curvrank
./unix_build.sh
pip install -e .

pip uninstall -y \
    opencv-python \
    opencv-python-headless

pip uninstall -y \
    dtool-ibeis \
    guitool-ibeis \
    plottool-ibeis \
    utool-ibeis \
    vtool-ibeis

pip uninstall -y \
    dtool \
    guitool \
    plottool \
    utool \
    vtool

pip uninstall -y \
    lightnet \
    brambox

pip uninstall -y \
    tensorflow \
    tensorflow-gpu \
    tensorflow-estimator \
    tensorboard \
    tensorboard-plugin-wit \
    keras

pip install \
    tensorflow-gpu==1.15.4 \
    keras==2.2.5

rm -rf ${VIRTUAL_ENV}/lib/python3.7/site-packages/cv2*
cp -r /tmp/cv2 ${VIRTUAL_ENV}/lib/python3.7/site-packages/cv2
rm -rf /tmp/cv2

python -c "import wbia;            from wbia.__main__ import smoke_test; smoke_test()"
python -c "import wbia_cnn;        from wbia_cnn.__main__ import main;   main()"
python -c "import wbia_pie;        from wbia_pie.__main__ import main;   main()"
python -c "import wbia_flukematch; from wbia_flukematch.plugin import *"
python -c "import wbia_curvrank;   from wbia_curvrank._plugin  import *"
python -c "import wbia_finfindr;   from wbia_finfindr._plugin  import *"
python -c "import wbia_kaggle7;    from wbia_kaggle7._plugin   import *"
python -c "import wbia_deepsense;  from wbia_deepsense._plugin import *"

find ${CODE}/wbia* -name '*.a' -print0 | xargs -0 -i /bin/bash -c 'echo {} && ld -d {}'
find ${CODE}/wbia* -name '*.so' -print0 | xargs -0 -i /bin/bash -c 'echo {} && ld -d {}'
find ${CODE}/wildbook* -name '*.a' -print0 | xargs -0 -i /bin/bash -c 'echo {} && ld -d {}'
find ${CODE}/wildbook* -name '*.so' -print0 | xargs -0 -i /bin/bash -c 'echo {} && ld -d {}'

python -m wbia.dev --set-workdir /data/ibeis
