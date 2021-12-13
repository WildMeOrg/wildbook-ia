#!/bin/bash

# Install Tensorflow
set -ex

cd /wbia \

git clone https://github.com/WildMeOrg/wbia-plugin-pie.git

/bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-pie \
 && /bin/bash run_developer_setup.sh'

/virtualenv/env3/bin/pip install \
    'tensorflow-gpu==1.15.5' \
    'keras==2.2.5'

/virtualenv/env3/bin/pip install --upgrade \
    numpy

/virtualenv/env3/bin/python -c "import wbia_pie; from wbia_pie.__main__ import main; main()"
