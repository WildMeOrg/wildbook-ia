ARG WBIA_DEPENDENCIES_IMAGE=wildme/wbia-dependencies:latest

FROM ${WBIA_DEPENDENCIES_IMAGE} as org.wildme.wbia.provision

# Wildbook IA version
ARG VCS_URL="https://github.com/WildMeOrg/wildbook-ia"

ARG VCS_REF="develop"

# Set up xvfb for running gui doctests
ENV DISPLAY :1

# Make a copy of our build OpenCV
RUN set -ex \
 && rm -rf /tmp/cv2 \
 && mv /virtualenv/env3/lib/python3.7/site-packages/cv2 /tmp/cv2

# Clone WBIA repository
RUN set -ex \
 && cd /wbia \
 && git clone --branch ${VCS_REF} ${VCS_URL}

# Clone WBIA toolkit repositories
RUN set -ex \
 && cd /wbia \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-utool.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-vtool.git

# Clone WBIA third-party toolkit repositories
RUN set -ex \
 && cd /wbia \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pyhesaff.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pyflann.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pydarknet.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-tpl-pyrf.git \
 # Depricated
 && git clone --branch develop https://github.com/WildMeOrg/wbia-deprecate-tpl-brambox \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-deprecate-tpl-lightnet

# Clone first-party WBIA plug-in repositories
RUN set -ex \
 && cd /wbia \
 && git clone --recursive --branch develop https://github.com/WildMeOrg/wbia-plugin-cnn.git

RUN set -ex \
 && cd /wbia \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-flukematch.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-finfindr.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-deepsense.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-pie.git \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-lca.git

RUN set -ex \
 && cd /wbia \
 && git clone --branch develop https://github.com/WildMeOrg/wbia-plugin-orientation.git

# git clone --recursive --branch develop https://github.com/WildMeOrg/wbia-plugin-2d-orientation.git

# Clone third-party WBIA plug-in repositories
RUN set -ex \
 && cd /wbia \
 && git clone --recursive --branch develop-curvrank-v1 https://github.com/WildMeOrg/wbia-plugin-curvrank.git /wbia/wbia-plugin-curvrank-v1 \
 && cd /wbia/wbia-plugin-curvrank-v1/wbia_curvrank \
 && git fetch origin \
 && git checkout develop

RUN set -ex \
 && cd /wbia \
 && git clone --recursive --branch develop-curvrank-v2 https://github.com/WildMeOrg/wbia-plugin-curvrank.git /wbia/wbia-plugin-curvrank-v2 \
 && cd /wbia/wbia-plugin-curvrank-v2/wbia_curvrank_v2 \
 && git fetch origin \
 && git checkout develop

RUN set -ex \
 && cd /wbia \
 && git clone --recursive --branch develop https://github.com/WildMeOrg/wbia-plugin-kaggle7.git \
 && cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git fetch origin \
 && git checkout develop

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

RUN set -ex \
 && /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-tpl-pyrf \
 && /bin/bash run_developer_setup.sh'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-deprecate-tpl-brambox \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-deprecate-tpl-lightnet \
 && /virtualenv/env3/bin/pip install -r develop.txt \
 && pip install -e .'

# Wildbook IA
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wildbook-ia \
 && /bin/bash run_developer_setup.sh'

# WBIA third-party plug-ins
RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-cnn \
 && /bin/bash run_developer_setup.sh'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-pie \
 && /bin/bash run_developer_setup.sh'

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
 && cd /wbia/wbia-plugin-curvrank-v1 \
 && ./unix_build.sh \
 && pip install -e .'

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && cd /wbia/wbia-plugin-curvrank-v2 \
 && ./unix_build.sh \
 && pip install -e .'

# Ensure no previous or deprecated repositories are installed
# Ensure that opencv-python is un-installed since we want to use our built version
# Ensure specific version of keras and tensorflow-gpu are installed (for Pie support)
RUN set -ex \
 && /virtualenv/env3/bin/pip uninstall -y \
        opencv-python \
        opencv-python-headless \
 && /virtualenv/env3/bin/pip uninstall -y \
        dtool-ibeis \
        guitool-ibeis \
        plottool-ibeis \
        utool-ibeis \
        vtool-ibeis \
 && /virtualenv/env3/bin/pip uninstall -y \
        dtool \
        guitool \
        plottool \
        utool \
        vtool \
 && /virtualenv/env3/bin/pip uninstall -y \
        lightnet \
        brambox \
 && /virtualenv/env3/bin/pip uninstall -y \
        tensorflow \
        tensorflow-gpu \
        tensorflow-estimator \
        tensorboard \
        tensorboard-plugin-wit \
        keras \
 && /virtualenv/env3/bin/pip install \
        'tensorflow-gpu==1.15.4' \
        'keras==2.2.5' \
        'h5py<3.0.0' \
 && /virtualenv/env3/bin/pip install \
        'jedi==0.17.2'

RUN set -ex \
 && /virtualenv/env3/bin/pip freeze | grep wbia \
 && /virtualenv/env3/bin/pip freeze | grep wildbook

# Restore copy of our build OpenCV
# When opencv-python is installed and subsequently un-installed is clobbers our version
RUN set -ex \
 && rm -rf /virtualenv/env3/lib/python3.7/site-packages/cv2* \
 && mv /tmp/cv2 /virtualenv/env3/lib/python3.7/site-packages/cv2 \
 && rm -rf /tmp/cv2

CMD ["/bin/bash", "-c", "Xvfb :1 -screen 0 1024x768x16 &>/tmp/xvfb.log & /bin/bash"]
