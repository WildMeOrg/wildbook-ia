#!/bin/bash

# Install essential dependencies
set -ex

export TMPDIR=/home/$USER/tmp
mkdir -p $TMPDIR
cd /wbia

# Clone necessary repo
git clone https://github.com/WildMeOrg/wbia-plugin-pie.git

# Upgrade necessary libraries
/virtualenv/env3/bin/pip install --upgrade Cython==3.0.0
/virtualenv/env3/bin/pip install --upgrade gast

# Run developer setup
/bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-pie \
 && /bin/bash run_developer_setup.sh'

# Install required dependencies
/virtualenv/env3/bin/pip install --no-cache-dir \
    numpy==1.22.4 \
    scipy==1.7.3 \
    werkzeug==0.15.6 \
    flask==1.1.4 \
    markupsafe==1.1.1 \
    keras==2.2.5 \
    opencv-python==4.6.0.66 \
    ultralytics==8.3.84 \
    IPython==8.12.3 \
    six==1.16.0

# Verify installations
/virtualenv/env3/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
/virtualenv/env3/bin/python -c "import scipy; print('SciPy version:', scipy.__version__)"
/virtualenv/env3/bin/python -c "import six; print('six version:', six.__version__)"

# Run the plugin
/virtualenv/env3/bin/python -c "import wbia_pie; from wbia_pie.__main__ import main; main()"
