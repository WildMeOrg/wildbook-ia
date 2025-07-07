#!/bin/bash

set -ex

# Force use of the GitHub Actions Python (3.8 or 3.9), not system /usr/bin/python3
#!/bin/bash
set -ex

# Use correct Python installed by actions/setup-python
PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"
PIP_BIN="${PIP_BIN:-$(which pip3)}"

$PYTHON_BIN --version  # Debug
$PIP_BIN --version     # Debug

# Downgrade pip, setuptools, and wheel to versions that work with old metadata specs
$PYTHON_BIN -m pip install --upgrade --force-reinstall \
  --ignore-installed \
  'pip<24.1' 'setuptools==59.5.0' 'wheel==0.38.4'
    
export CUR_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Workaround fairseq + hydra-core omegaconf dependency hell
echo "Appending omegaconf==2.0.6 to requirements/build.txt if not present..."
grep -qxF 'omegaconf==2.0.6' requirements/build.txt || echo 'omegaconf==2.0.6' >> requirements/build.txt

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

$PYTHON_BIN -m pip install --global-option=build_ext \
  --global-option="-I/usr/include/graphviz/" \
  --global-option="-L/usr/lib/graphviz/" pygraphviz

$PYTHON_BIN -m pip uninstall -y pyqt5
$PYTHON_BIN -m pip install --upgrade pyqt5
