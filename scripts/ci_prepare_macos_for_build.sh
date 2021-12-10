#!/bin/bash

set -ex

export CUR_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJ_DIR=$(brew --prefix proj)

brew install \
    geos \
    gdal \
    proj \
    graphviz \
    postgresql \
    pyqt@5 \
    coreutils

pip install --global-option=build_ext --global-option="-I$(echo $(brew --prefix graphviz)/include/)" --global-option="-L$(echo $(brew --prefix graphviz)/lib/)" pygraphviz

pip install -r requirements.txt
