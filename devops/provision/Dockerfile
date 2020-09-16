FROM wildme/wbia-dependencies:latest as org.wildme.wbia.provision

MAINTAINER Wild Me <dev@wildme.org>

# Set up xvfb for running gui doctests
ENV DISPLAY :1

# Clone WBIA repository
RUN set -ex \
 && cd /wbia \
 && git clone --branch build-process-merge https://github.com/WildbookOrg/wildbook-ia.git

# Clone WBIA toolkit repositories
RUN set -ex \
 && cd /wbia \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-utool.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-vtool.git

# Clone WBIA third-party toolkit repositories
RUN set -ex \
 && cd /wbia \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-tpl-pyhesaff.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-tpl-pyflann.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-tpl-pydarknet.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-tpl-pyrf.git \
 # Depricated
 && git clone --branch develop https://github.com/WildbookOrg/wbia-deprecate-tpl-brambox \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-deprecate-tpl-lightnet

# Clone first-party WBIA plug-in repositories
RUN set -ex \
 && cd /wbia \
 && git clone --recursive --branch develop https://github.com/WildbookOrg/wbia-plugin-cnn.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-plugin-flukematch.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-plugin-finfindr.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-plugin-deepsense.git \
 && git clone --branch develop https://github.com/WildbookOrg/wbia-plugin-pie.git

# Clone third-party WBIA plug-in repositories
RUN set -ex \
 && cd /wbia \
 && git clone --recursive --branch develop https://github.com/WildbookOrg/wbia-plugin-curvrank.git \
 && cd /wbia/wbia-plugin-curvrank/wbia_curvrank \
 && git fetch origin \
 && git checkout develop 

RUN set -ex \
 && cd /wbia \
 && git clone --recursive --branch develop https://github.com/WildbookOrg/wbia-plugin-kaggle7.git \
 && cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git fetch origin \
 && git checkout develop

# git clone --recursive --branch develop https://github.com/WildbookOrg/wbia-plugin-2d-orientation.git
# git clone --recursive --branch develop https://github.com/WildbookOrg/wbia-plugin-lca.git

# WBIA Toolkits

# Build and install wbia-utool
RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-utool \
 && /bin/bash run_developer_setup.sh'

# Build and install wbia-vtool
RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-vtool \
 && /bin/bash run_developer_setup.sh'

# WBIA third-party toolkits

# Build and install wbia-pyhesaff
RUN set -ex \
 && apt update \
 && apt install -y \
     libhdf5-serial-dev \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
     && cd /wbia/wbia-tpl-pyhesaff \
     && /bin/bash run_developer_setup.sh' \
 && rm -rf /var/lib/apt/lists/*

# Build and install wbia-pyflann
RUN set -ex \
 && apt update \
 && apt install -y \
     liblz4-dev \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
     && cd /wbia/wbia-tpl-pyflann \
     && /bin/bash run_developer_setup.sh' \
 && rm -rf /var/lib/apt/lists/*

# Build and install wbia-pydarknet
RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
     && cd /wbia/wbia-tpl-pydarknet \
     && /bin/bash run_developer_setup.sh'

# Build and install wbia-pyrf
RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
     && cd /wbia/wbia-tpl-pyrf \
     && /bin/bash run_developer_setup.sh'

# Build and install brambox
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-deprecate-tpl-brambox \
 && pip install -e .'

# Build and install lightnet
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-deprecate-tpl-lightnet \
 && /virtualenv/env3/bin/pip install -r develop.txt \
 && pip install -e .'

# Wildbook IA

# Build and install wildbook-ia
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && pip install -e .'

# WBIA third-party plug-ins

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-cnn \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-flukematch \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-finfindr \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-deepsense \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-pie \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-curvrank \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-kaggle7 \
 && pip install -e .'

CMD ["/bin/bash", "-c", "Xvfb :1 -screen 0 1024x768x16 &>/tmp/xvfb.log & /bin/bash"]
