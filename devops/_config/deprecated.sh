#!/bin/bash

# Install Tensorflow
set -ex

export TMPDIR=/home/$USER/tmp
mkdir $TMPDIR
cd /wbia \

git clone https://github.com/WildMeOrg/wbia-plugin-pie.git

/virtualenv/env3/bin/pip install --upgrade Cython
/virtualenv/env3/bin/pip install --upgrade gast

/bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-pie \
 && /bin/bash run_developer_setup.sh'

/virtualenv/env3/bin/pip install --no-cache-dir \
    'tensorflow-gpu==2.2.0' \
    'keras==3.1.1'

/virtualenv/env3/bin/pip install --no-cache-dir --upgrade \
    numpy

/virtualenv/env3/bin/python -c "import wbia_pie; from wbia_pie.__main__ import main; main()"
