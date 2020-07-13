FROM wildme/wbia-dependencies:latest as org.wildme.wbia.provision

MAINTAINER Wild Me <dev@wildme.org>

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Clone IBEIS VSTS repositories
RUN git clone --branch master https://github.com/WildbookOrg/wbia-tpl-brambox           /wbia/brambox \
 && git clone --branch master https://github.com/WildbookOrg/wbia-tpl-pyflann         /wbia/flann \
 && git clone --branch master https://github.com/WildbookOrg/wbia-tpl-pyhesaff        /wbia/hesaff \
 && git clone --branch master https://github.com/WildbookOrg/wbia-plugin-flukematch     /wbia/wbia-plugin-flukematch \
 && git clone --branch master https://github.com/WildbookOrg/wbia-plugin-curvrank       /wbia/wbia-plugin-curvrank \
 && git clone --branch master https://github.com/WildbookOrg/wbia-plugin-deepsense      /wbia/wbia-plugin-deepsense \
 && git clone --branch master https://github.com/WildbookOrg/wbia-plugin-finfindr       /wbia/wbia-plugin-finfindr \
 && git clone --branch master https://github.com/WildbookOrg/wbia-plugin-kaggle7        /wbia/wbia-plugin-kaggle7 \
 && git clone --branch master https://github.com/WildbookOrg/wbia-plugin-2d-orientation /wbia/wbia-plugin-2d-orientation \
 && git clone --branch develop https://github.com/WildbookOrg/wildbook-ia               /wbia/wildbook-ia \
 && git clone --branch master https://github.com/WildbookOrg/wbia-plugin-cnn            /wbia/wbia-plugin-cnn \
 && git clone --branch master https://github.com/WildbookOrg/wbia-tpl-lightnet          /wbia/lightnet \
 && git clone --branch master https://github.com/WildbookOrg/wbia-tpl-pydarknet         /wbia/pydarknet \
 && git clone --branch master https://github.com/WildbookOrg/wbia-tpl-pyrf              /wbia/pyrf \
 && git clone --branch master https://github.com/WildbookOrg/wbia-utool                 /wbia/utool \
 && git clone --branch master https://github.com/WildbookOrg/wbia-vtool               /wbia/vtool

RUN cd /wbia/wbia-plugin-curvrank/wbia_curvrank \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/wbia-tpl-curvrank \
 && git fetch vs \
 # FIXME (11-Jun-12020) Looking to change the branch to `master`
 && git checkout functional-ibeis

RUN cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/wbia-tpl-kaggle7 \
 && git fetch vs \
 && git checkout master

RUN cd /wbia/wbia-plugin-2d-orientation/wbia_2d_orientation \
 && git init \
 && git remote add vs https://github.com/WildbookOrg/wbia-tpl-2d-orientation \
 && git fetch vs \
 && git checkout master

# Build and install utool and ubelt
RUN cd /wbia/utool \
 && /virtualenv/env3/bin/pip install -e .

RUN /virtualenv/env3/bin/pip install ubelt

# TODO Temporarily disabled: Checkout latest version of all other repos
#RUN cd /wbia/wbia \
# && /virtualenv/env3/bin/python super_setup.py pull \
# && /virtualenv/env3/bin/python super_setup.py pull

RUN . /virtualenv/env3/bin/activate \
 && pip install scikit-build

# Build Python repositories with external codebases
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/flann \
 && /bin/bash run_developer_setup.sh'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/hesaff \
 && /bin/bash run_developer_setup.sh'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/brambox \
 && /virtualenv/env3/bin/pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/lightnet \
 && /virtualenv/env3/bin/pip install -r requirements.txt \
 && /virtualenv/env3/bin/pip install -r develop.txt'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-flukematch \
 && /bin/bash unix_build.sh'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-curvrank \
 && /bin/bash unix_build.sh'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/pyrf \
 && /bin/bash unix_build.sh'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/vtool \
 && /bin/bash run_developer_setup.sh'

# Install Python repositories
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-cnn \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/hesaff \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/lightnet \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/pydarknet \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/pyrf \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/utool \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/vtool \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-flukematch \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-curvrank \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-deepsense \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-finfindr \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-kaggle7 \
 && pip install -e .'
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-2d-orientation \
 && pip install -e .'

# Set up xvfb for running gui doctests
ENV DISPLAY :1

CMD ["/bin/bash", "-c", "Xvfb :1 -screen 0 1024x768x16 &>/tmp/xvfb.log & /bin/bash"]
