#!/bin/bash

set -ex

export CUR_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pip install -r requirements/build.txt

if command -v yum &> /dev/null
then
    yum install -y \
        epel-release \
        yum-utils

    yum-config-manager --enable pgdg12

    yum install -y \
        geos-devel \
        gdal-devel \
        proj-devel \
        graphviz \
        graphviz-devel \
        wget \
        postgresql12-server \
        postgresql12 \
        qtbase5-dev \
        qt5-qtbase-devel \
        qt5-qmake \
        coreutils
else
    apt-get install -y \
        pgloader \
        libgeos-dev \
        libgdal-dev \
        libproj-dev \
        graphviz \
        graphviz-dev \
        postgresql \
        libopencv-dev \
        qt5-qmake \
        qtbase5-dev \
        qtchooser \
        qtbase5-dev-tools \
        qttools5-dev-tools \
        qtchooser \
        coreutils
fi

pip install --global-option=build_ext --global-option="-I/usr/include/graphviz/" --global-option="-L/usr/lib/graphviz/" pygraphviz
pip uninstall -y pyqt5
pip install --upgrade pyqt5
