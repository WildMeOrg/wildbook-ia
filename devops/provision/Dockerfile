FROM wildme/ibeis-dependencies:latest as org.wildme.ibeis.provision

MAINTAINER Wild Me <dev@wildme.org>

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Clone IBEIS VSTS repositories
RUN git clone --branch next https://github.com/WildbookOrg/brambox                     /ibeis/brambox \
 && git clone --branch next https://github.com/WildbookOrg/flann                       /ibeis/flann \
 && git clone --branch next https://github.com/WildbookOrg/guitool                     /ibeis/guitool \
 && git clone --branch next https://github.com/WildbookOrg/hesaff                      /ibeis/hesaff \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-flukematch-module     /ibeis/ibeis-flukematch-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-curvrank-module       /ibeis/ibeis-curvrank-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-deepsense-module      /ibeis/ibeis-deepsense-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-finfindr-module       /ibeis/ibeis-finfindr-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-kaggle7-module        /ibeis/ibeis-kaggle7-module \
 && git clone --branch next https://github.com/WildbookOrg/ibeis-2d-orientation-module /ibeis/ibeis-2d-orientation-module \
 && git clone --branch master https://github.com/WildbookOrg/ibeis                     /ibeis/ibeis \
 && git clone --branch next https://github.com/WildbookOrg/ibeis_cnn                   /ibeis/ibeis_cnn \
 && git clone --branch next https://github.com/WildbookOrg/lightnet                    /ibeis/lightnet \
 && git clone --branch next https://github.com/WildbookOrg/plottool                    /ibeis/plottool \
 && git clone --branch next https://github.com/WildbookOrg/pydarknet                   /ibeis/pydarknet \
 && git clone --branch next https://github.com/WildbookOrg/pyrf                        /ibeis/pyrf \
 && git clone --branch next https://github.com/WildbookOrg/ubelt                       /ibeis/ubelt \
 && git clone --branch next https://github.com/WildbookOrg/utool                       /ibeis/utool \
 && git clone --branch next https://github.com/WildbookOrg/vtool                       /ibeis/vtool

RUN cd /ibeis/ibeis-curvrank-module/ibeis_curvrank \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/ibeis_curvrank \
 && git fetch vs \
 && git checkout functional-ibeis

RUN cd /ibeis/ibeis-kaggle7-module/ibeis_kaggle7 \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/whale-identification-2018 \
 && git fetch vs \
 && git checkout next

RUN cd /ibeis/ibeis-2d-orientation-module/ibeis_2d_orientation \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/2D-Orientation-v2 \
 && git fetch vs \
 && git checkout plugin

# Build and install utool and ubelt
RUN cd /ibeis/utool \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ubelt \
 && /virtualenv/env3/bin/pip install -e .

# TODO Temporarily disabled: Checkout latest version of all other repos
#RUN cd /ibeis/ibeis \
# && /virtualenv/env3/bin/python super_setup.py pull \
# && /virtualenv/env3/bin/python super_setup.py pull

ADD ./_config/types_c.h /virtualenv/env3/include/opencv2/core/types_c.h

RUN . /virtualenv/env3/bin/activate \
 && pip install scikit-build

# Build Python repositories with external codebases
# TODO remove flann checkout v2.0.0 once the "next" branch is working
# TODO remove hesaff checkout v2.0.0 once the "next" branch is working
RUN . /virtualenv/env3/bin/activate \
 && cd /ibeis/flann \
 && git checkout v2.0.0 \
 && /bin/bash unix_build.sh \
 && cd /ibeis/hesaff \
 && git checkout v2.0.0 \
 && /bin/bash unix_build.sh \
 && cd /ibeis/lightnet \
 && /virtualenv/env3/bin/pip install -r requirements.txt \
 && /virtualenv/env3/bin/pip install -r develop.txt \
 && cd /ibeis/ibeis-flukematch-module \
 && /bin/bash unix_build.sh \
 && cd /ibeis/ibeis-curvrank-module \
 && /bin/bash unix_build.sh \
 && cd /ibeis/pydarknet \
 && /bin/bash unix_build.sh \
 && cd /ibeis/pyrf \
 && /bin/bash unix_build.sh \
 && cd /ibeis/vtool \
 && /bin/bash dev/unix_build.sh

# Install Python repositories
RUN . /virtualenv/env3/bin/activate \
 && cd /ibeis/ibeis_cnn \
 && pip install -e . \
 && cd /ibeis/guitool \
 && pip install -e . \
 && cd /ibeis/hesaff \
 && pip install -e . \
 && cd /ibeis/brambox \
 && pip install -e . \
 && cd /ibeis/lightnet \
 && pip install -e . \
 && cd /ibeis/plottool \
 && pip install -e . \
 && cd /ibeis/pydarknet \
 && pip install -e . \
 && cd /ibeis/pyrf \
 && pip install -e . \
 && cd /ibeis/utool \
 && pip install -e . \
 && cd /ibeis/ubelt \
 && pip install -e . \
 && cd /ibeis/vtool \
 && pip install -e . \
 && cd /ibeis/ibeis \
 && pip install -e . \
 && cd /ibeis/ibeis-flukematch-module \
 && pip install -e . \
 && cd /ibeis/ibeis-curvrank-module \
 && pip install -e . \
 && cd /ibeis/ibeis-deepsense-module \
 && pip install -e . \
 && cd /ibeis/ibeis-finfindr-module \
 && pip install -e . \
 && cd /ibeis/ibeis-kaggle7-module \
 && pip install -e . \
 && cd /ibeis/ibeis-2d-orientation-module \
 && pip install -e .
