FROM wildme/ibeis-dependencies:latest as org.wildme.ibeis.provision

MAINTAINER Wild Me <dev@wildme.org>

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Clone IBEIS VSTS repositories
RUN git clone --branch next https://github.com/WildbookOrg/brambox                     /ibeis/brambox \
 && git clone --branch next https://github.com/WildbookOrg/detecttools                 /ibeis/detecttools \
 && git clone --branch next https://github.com/WildbookOrg/dtool                       /ibeis/dtool \
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

# Build Python repositories with external codebases
RUN . /virtualenv/env3/bin/activate \
 && cd /ibeis/flann \
 && /bin/bash unix_build.sh \
 && cd /ibeis/hesaff \
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
 && /bin/bash unix_build.sh

# Install Python repositories
RUN cd /ibeis/detecttools \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/dtool \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis_cnn \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/guitool \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/hesaff \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/brambox \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/lightnet \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/plottool \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/pydarknet \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/pyrf \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/utool \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ubelt \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/vtool \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis-flukematch-module \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis-curvrank-module \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis-deepsense-module \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis-finfindr-module \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis-kaggle7-module \
 && /virtualenv/env3/bin/pip install -e . \
 && cd /ibeis/ibeis-2d-orientation-module \
 && /virtualenv/env3/bin/pip install -e .
