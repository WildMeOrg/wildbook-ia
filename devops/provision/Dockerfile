FROM wildme/wbia-dependencies:latest as org.wildme.wbia.provision

MAINTAINER Wild Me <dev@wildme.org>

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Clone IBEIS VSTS repositories
RUN git clone --branch next https://github.com/WildbookOrg/brambox                     /wbia/brambox \
 && git clone --branch master https://github.com/WildbookOrg/flann                     /wbia/flann \
 && git clone --branch master https://github.com/WildbookOrg/hesaff                    /wbia/hesaff \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-flukematch-module     /wbia/ibeis-flukematch-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-curvrank-module       /wbia/ibeis-curvrank-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-deepsense-module      /wbia/ibeis-deepsense-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-finfindr-module       /wbia/ibeis-finfindr-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-kaggle7-module        /wbia/ibeis-kaggle7-module \
 && git clone --branch next https://github.com/WildbookOrg/wbia-2d-orientation-module /wbia/wbia-2d-orientation-module \
 && git clone --branch master https://github.com/WildbookOrg/wbia                     /wbia/wbia \
 && git clone --branch next https://github.com/WildbookOrg/ibeis_cnn                   /wbia/ibeis_cnn \
 && git clone --branch next https://github.com/WildbookOrg/lightnet                    /wbia/lightnet \
 && git clone --branch next https://github.com/WildbookOrg/pydarknet                   /wbia/pydarknet \
 && git clone --branch next https://github.com/WildbookOrg/pyrf                        /wbia/pyrf \
 && git clone --branch next https://github.com/WildbookOrg/utool                       /wbia/utool \
 && git clone --branch master https://github.com/WildbookOrg/vtool                     /wbia/vtool

RUN cd /wbia/ibeis-curvrank-module/ibeis_curvrank \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/ibeis_curvrank \
 && git fetch vs \
 && git checkout functional-wbia

RUN cd /wbia/ibeis-kaggle7-module/wbia_kaggle7 \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/whale-identification-2018 \
 && git fetch vs \
 && git checkout next

RUN cd /wbia/wbia-2d-orientation-module/wbia_2d_orientation \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/2D-Orientation-v2 \
 && git fetch vs \
 && git checkout plugin

# Build and install utool and ubelt
RUN cd /wbia/utool \
 && /virtualenv/env3/bin/pip install -e .

RUN /virtualenv/env3/bin/pip install ubelt

# TODO Temporarily disabled: Checkout latest version of all other repos
#RUN cd /wbia/wbia \
# && /virtualenv/env3/bin/python super_setup.py pull \
# && /virtualenv/env3/bin/python super_setup.py pull

ADD ./_config/types_c.h /virtualenv/env3/include/opencv2/core/types_c.h

RUN . /virtualenv/env3/bin/activate \
 && pip install scikit-build

# Build Python repositories with external codebases
# TODO remove flann checkout v2.0.0 once the "next" branch is working
# TODO remove hesaff checkout v2.0.0 once the "next" branch is working
# TODO remove vtool checkout v2.0.0 once the "master" branch is working
RUN . /virtualenv/env3/bin/activate \
 && cd /wbia/flann \
 && git checkout v2.0.0 \
 && /bin/bash unix_build.sh \
 && cd /wbia/hesaff \
 && git checkout v2.0.0 \
 && /bin/bash unix_build.sh \
 && cd /wbia/lightnet \
 && /virtualenv/env3/bin/pip install -r requirements.txt \
 && /virtualenv/env3/bin/pip install -r develop.txt \
 && cd /wbia/ibeis-flukematch-module \
 && /bin/bash unix_build.sh \
 && cd /wbia/ibeis-curvrank-module \
 && /bin/bash unix_build.sh \
 && cd /wbia/pydarknet \
 && /bin/bash unix_build.sh \
 && cd /wbia/pyrf \
 && /bin/bash unix_build.sh \
 && cd /wbia/vtool \
 && git checkout v2.0.0 \
 && /bin/bash unix_build.sh

# Install Python repositories
RUN . /virtualenv/env3/bin/activate \
 && cd /wbia/ibeis_cnn \
 && pip install -e . \
 && cd /wbia/hesaff \
 && pip install -e . \
 && cd /wbia/brambox \
 && pip install -e . \
 && cd /wbia/lightnet \
 && pip install -e . \
 && cd /wbia/pydarknet \
 && pip install -e . \
 && cd /wbia/pyrf \
 && pip install -e . \
 && cd /wbia/utool \
 && pip install -e . \
 && cd /wbia/vtool \
 && pip install -e . \
 && cd /wbia/wbia \
 && pip install -e . \
 && cd /wbia/ibeis-flukematch-module \
 && pip install -e . \
 && cd /wbia/ibeis-curvrank-module \
 && pip install -e . \
 && cd /wbia/ibeis-deepsense-module \
 && pip install -e . \
 && cd /wbia/ibeis-finfindr-module \
 && pip install -e . \
 && cd /wbia/ibeis-kaggle7-module \
 && pip install -e . \
 && cd /wbia/wbia-2d-orientation-module \
 && pip install -e .
