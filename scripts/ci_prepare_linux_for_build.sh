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
        postgresql12
else
    apt-get install -y \
        pgloader \
        libgeos-dev \
        libgdal-dev \
        libproj-dev \
        graphviz \
        graphviz-dev \
        postgresql
fi

pip install --global-option=build_ext --global-option="-I/usr/include/graphviz/" --global-option="-L/usr/lib/graphviz/" pygraphviz
