#!/bin/bash

set -ex

PYTHON_BIN=$(which python3)
PIP_BIN=$(which pip3)

# Force-clean pip environment before reinstall
$PYTHON_BIN -m pip uninstall -y pip setuptools wheel || true

# Reinstall specific versions cleanly
$PYTHON_BIN -m ensurepip --upgrade
$PYTHON_BIN -m pip install --upgrade --force-reinstall \
  'pip==24.0' 'setuptools==59.5.0' 'wheel==0.38.4'

# Confirm what pip we're running
$PYTHON_BIN -m pip --version

# Ensure clean pip cache to avoid versioning bugs
$PIP_BIN cache purge || true

# Install build deps (omegaconf==2.0.6 already pinned in requirements/build.txt)
$PYTHON_BIN -m pip install -r requirements/build.txt

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
    echo "Skipping apt installs – handled in testing.yml"
fi

$PYTHON_BIN -m pip install --global-option=build_ext \
  --global-option="-I/usr/include/graphviz/" \
  --global-option="-L/usr/lib/graphviz/" pygraphviz

$PYTHON_BIN -m pip uninstall -y pyqt5
$PYTHON_BIN -m pip install --upgrade pyqt5
