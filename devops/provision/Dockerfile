ARG WBIA_BASE_IMAGE=wildme/wbia-base:latest

FROM ${WBIA_BASE_IMAGE} as org.wildme.wbia.provision

# Wildbook IA version
ARG VCS_URL="https://github.com/WildMeOrg/wildbook-ia"

ARG VCS_REF="fix-ubuntu1804-pub-file-url"

# Clone WBIA repository
RUN set -ex \
 && cd /wbia \
 && git clone --branch ${VCS_REF} ${VCS_URL}

# Clone WBIA toolkit repositories
RUN set -ex \
 && cd /wbia \
 && git clone https://github.com/WildMeOrg/wbia-utool.git \
 && git clone https://github.com/WildMeOrg/wbia-vtool.git

# Clone WBIA third-party toolkit repositories
RUN set -ex \
 && cd /wbia \
 && git clone https://github.com/WildMeOrg/wbia-tpl-pyhesaff.git \
 && git clone https://github.com/WildMeOrg/wbia-tpl-pyflann.git \
 && git clone https://github.com/WildMeOrg/wbia-tpl-pydarknet.git \
 # Depricated
 && git clone https://github.com/WildMeOrg/wbia-deprecate-tpl-brambox \
 && git clone https://github.com/WildMeOrg/wbia-deprecate-tpl-lightnet

# Clone first-party WBIA plug-in repositories
RUN set -ex \
 && cd /wbia \
 && git clone --recursive https://github.com/WildMeOrg/wbia-plugin-cnn.git

RUN set -ex \
 && cd /wbia \
 && git clone https://github.com/WildMeOrg/wbia-plugin-orientation.git \
 && git clone https://github.com/WildMeOrg/wbia-plugin-flukematch.git \
 && git clone https://github.com/WildMeOrg/wbia-plugin-finfindr.git \
 && git clone https://github.com/WildMeOrg/wbia-plugin-deepsense.git \
 && git clone https://github.com/WildMeOrg/wbia-plugin-pie-v2.git \
 && git clone https://github.com/WildMeOrg/wbia-plugin-tbd.git


RUN set -ex \
 && cd /wbia \
 && git clone --recursive https://github.com/WildMeOrg/wbia-plugin-blend.git \
 && cd /wbia/wbia-plugin-blend/wbia_blend \
 && git fetch origin \
 && git checkout main

# Clone third-party WBIA plug-in repositories
RUN set -ex \
 && cd /wbia \
 && git clone --recursive https://github.com/WildMeOrg/wbia-plugin-curvrank.git \
 && cd /wbia/wbia-plugin-curvrank/wbia_curvrank_v2 \
 && git fetch origin \
 && git checkout main

RUN set -ex \
 && cd /wbia \
 && git clone --recursive https://github.com/WildMeOrg/wbia-plugin-kaggle7.git \
 && cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git fetch origin \
 && git checkout main

RUN set -ex \
 && cd /wbia \
 && git clone --recursive https://github.com/WildMeOrg/wbia-plugin-lca.git \
 && cd /wbia/wbia-plugin-lca/wbia_lca \
 && git fetch origin \
 && git checkout main

# WBIA Toolkits
RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-utool \
 && /bin/bash run_developer_setup.sh'

RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-vtool \
 && /bin/bash run_developer_setup.sh'

# WBIA third-party toolkits
RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-tpl-pyhesaff \
 && /bin/bash run_developer_setup.sh'

RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-tpl-pyflann \
 && /bin/bash run_developer_setup.sh'

RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-tpl-pydarknet \
 && /bin/bash run_developer_setup.sh'


RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-deprecate-tpl-brambox \
 && /virtualenv/env3/bin/pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-deprecate-tpl-lightnet \
 && /virtualenv/env3/bin/pip install -r develop.txt \
 && /virtualenv/env3/bin/pip install -e .'

# Wildbook IA
# && /bin/bash run_developer_setup.sh \

#RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
# && cd /wbia/wildbook-ia \'
# &&  /bin/bash clean.sh'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && python setup.py clean'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && ls -l \
 && pip install -r requirements.txt'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && python setup.py develop'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd cd /wbia/wildbook-ia \
 && /virtualenv/env3/bin/pip install -e .'
 
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && /virtualenv/env3/bin/pip install -r requirements/postgres.txt'

# RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
# && cd /wbia/wildbook-ia \
# && /virtualenv/env3/bin/pip install -r requirements/postgres.txt'
 
# WBIA third-party plug-ins
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-cnn \
 && /bin/bash run_developer_setup.sh'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-pie-v2 \
 && /bin/bash run_developer_setup.sh'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-blend \
 && /virtualenv/env3/bin/pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-finfindr \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-deepsense \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-kaggle7 \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-lca \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-orientation \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-flukematch \
 && ./unix_build.sh \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-curvrank \
 && ./unix_build.sh \
 && pip install -e .'

# Ensure no previous or deprecated repositories are installed
# Ensure that opencv-python is un-installed since we want to use our built version
# Ensure specific version of keras and tensorflow-gpu are installed (for Pie support)
RUN set -ex \
 && /virtualenv/env3/bin/pip uninstall -y \
     dtool-ibeis \
     guitool-ibeis \
     plottool-ibeis \
     utool-ibeis \
     vtool-ibeis \
     dtool \
     guitool \
     plottool \
     utool \
     vtool \
     lightnet \
     brambox \
 && /virtualenv/env3/bin/pip uninstall -y \
     opencv-python \
     opencv-contrib-python \
     opencv-python-headless \
     opencv-contrib-python-headless \
     tensorflow \
     tensorflow-gpu \
     tensorflow-estimator \
     tensorboard \
     tensorboard-plugin-wit \
     keras \
     torch \
     torchvision \
     torchaudio \
 && /virtualenv/env3/bin/pip install \
     pynvml \
 && /virtualenv/env3/bin/pip install --upgrade \
     opencv-contrib-python-headless \
 && /virtualenv/env3/bin/pip install --upgrade --pre \
     torch==1.12.1 \
     torchvision \
     torchaudio \
     --extra-index-url https://download.pytorch.org/whl/nightly/cu113 \
 && /virtualenv/env3/bin/pip install \
     MarkupSafe==2.0.1 \
     Flask==2.1.0

# Wildbook IA problematic dependencies
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && pip install -r requirements/problematic.txt'

# Ensure that we are running the local installed copy, not from PyPI
RUN set -ex \
 && /virtualenv/env3/bin/pip uninstall -y \
     wildbook-ia \
 && cd /wbia/wildbook-ia \
 && /virtualenv/env3/bin/pip install -e .
