#!/bin/bash

# Install Tensorflow
set -ex

export TMPDIR=/home/$USER/tmp
mkdir $TMPDIR
export PIP_CACHE_DIR=/home/$USER/tmp
/virtualenv/env3/bin/pip cache dir
du -h
cd /wbia \

git clone https://github.com/WildMeOrg/wbia-plugin-pie.git -b patch-1

/virtualenv/env3/bin/pip install --cache-dir $TMPDIR --upgrade Cython
/virtualenv/env3/bin/pip install --cache-dir $TMPDIR --upgrade gast

/virtualenv/env3/bin/pip cache purge
du -h
/bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-pie \
 && /bin/bash run_developer_setup.sh'
du -h
# /virtualenv/env3/bin/pip install --cache-dir $TMPDIR \
#    'tensorflow-gpu==1.15.5' \
#    'keras==2.2.5'

/virtualenv/env3/bin/pip install --cache-dir $TMPDIR --upgrade \
    numpy

/virtualenv/env3/bin/python -c "import wbia_pie; from wbia_pie.__main__ import main; main()"
