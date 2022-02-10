ARG WBIA_BASE_IMAGE=wildme/wbia-base:latest

ARG WBIA_PROVISION_IMAGE=wildme/wbia-provision:latest

FROM ${WBIA_PROVISION_IMAGE} as org.wildme.wbia.latest

# Grab latest code
RUN set -ex \
 && git config --global user.email "dev@wildme.org" \
 && git config --global user.name "Wild Me" \
 && find /wbia/wbia* -name '.git' -type d -print0 | xargs -0 -i /bin/bash -c \
    'cd {} && cd .. && echo $(pwd) && git reset --hard origin/main && git pull' \
 && cd /wbia/wbia-plugin-curvrank/wbia_curvrank_v2 \
 && git reset --hard origin/main \
 && git pull \
 && cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git reset --hard origin/main \
 && git pull \
 && cd /wbia/wbia-plugin-lca/wbia_lca \
 && git reset --hard origin/main \
 && git pull \
 && cd /wbia/wbia-plugin-orientation/ \
 && git reset --hard origin/main \
 && git pull \
 && cd /wbia/wildbook-ia/ \
 && git reset --hard origin/main \
 && git pull

##########################################################################################

FROM ${WBIA_BASE_IMAGE} as org.wildme.wbia.install

ARG VERSION="3.7.2"

ARG VCS_URL="https://github.com/WildMeOrg/wildbook-ia"

ARG VCS_REF="main"

ARG BUILD_DATE="2020"

LABEL autoheal=true

LABEL org.opencontainers.image.source https://github.com/WildMeOrg/wildbook-ia

LABEL org.wildme.vendor="Wild Me" \
      org.wildme.url="https://wildme.org" \
      org.wildme.name="Wildbook IA" \
      org.wildme.description="Wildbook's Image Analysis (WBIA) backend service supporting machine learning for wildlife conservation" \
      org.wildme.version=${VERSION} \
      org.wildme.vcs-url=${VCS_URL} \
      org.wildme.vcs-ref=${VCS_REF} \
      org.wildme.build-date=${BUILD_DATE} \
      org.wildme.docker.schema-version="1.0"

ENV HOST_USER root

ENV HOST_UID 0

ENV AWS_ACCESS_KEY_ID ACCESS_KEY

ENV AWS_SECRET_ACCESS_KEY SECRET_KEY

WORKDIR /data/db

COPY --from=org.wildme.wbia.latest /virtualenv /virtualenv

COPY --from=org.wildme.wbia.latest /wbia /wbia

# theano configuration file
COPY ./_config/theanorc /root/.theanorc

# setup script for python development
COPY ./_config/setup.sh /bin/setup

# embed script for python development
COPY ./_config/embed.sh /bin/embed

# embed script for python development
COPY ./_config/embedpg.sh /bin/embedpg

# (non-root) bash shell script for python development
COPY ./_config/shell.sh /bin/shell

# entrypoint
COPY ./_config/entrypoint.sh /bin/entrypoint

# Python health check
COPY ./_config/healthcheck.py /bin/healthcheck.py

# Update locate database
RUN updatedb

# Temporary fix for ARM64
ENV LD_PRELOAD "${LD_PRELOAD}:/virtualenv/env3/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1"

ENV LD_PRELOAD "${LD_PRELOAD}:/virtualenv/env3/lib/python3.7/site-packages/scikit_image.libs/libgomp-d22c30c5.so.1.0.0"

# Fix PyQt5
RUN set -ex \
 && apt-get update \
 && apt-get remove -y \
     python3-pyqt5* \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
     qt5-qmake \
     qt5-default \
 && /virtualenv/env3/bin/pip uninstall -y \
     pyqt5 \
 && /virtualenv/env3/bin/pip install --upgrade \
     pyqt5 \
 && apt-get clean \
 && apt-get autoclean \
 && apt-get autoremove -y \
 && rm -rf /var/cache/apt \
 && rm -rf /var/lib/apt/lists/*

# Run smoke tests
RUN set -ex \
 && mkdir -p /data \
 && /virtualenv/env3/bin/python -m wbia.dev --set-workdir /data --preload-exit \
 && /virtualenv/env3/bin/python -c "import wbia;             from wbia.__main__ import smoke_test; smoke_test()" \
 && /virtualenv/env3/bin/python -c "import wbia_cnn;         from wbia_cnn.__main__ import main;         main()" \
 && /virtualenv/env3/bin/python -c "import wbia_pie_v2;      from wbia_pie_v2.__main__ import main;      main()" \
 && /virtualenv/env3/bin/python -c "import wbia_orientation; from wbia_orientation.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import wbia_flukematch;  from wbia_flukematch.plugin   import *" \
 && /virtualenv/env3/bin/python -c "import wbia_curvrank_v2; from wbia_curvrank_v2._plugin import *" \
 && /virtualenv/env3/bin/python -c "import wbia_finfindr;    from wbia_finfindr._plugin    import *" \
 && /virtualenv/env3/bin/python -c "import wbia_kaggle7;     from wbia_kaggle7._plugin     import *" \
 && /virtualenv/env3/bin/python -c "import wbia_lca;         from wbia_lca._plugin         import *" \
 && /virtualenv/env3/bin/python -c "import wbia_deepsense;   from wbia_deepsense._plugin   import *" \
 && find /wbia/wbia* -name '*.a' -print      | grep -v "cpython-37m-x86_64-linux-gnu" | grep -v "cpython-37m-aarch64-linux-gnu.so" | xargs -i /bin/bash -c 'echo {} && ld -d {}' \
 && find /wbia/wbia* -name '*.so' -print     | grep -v "cpython-37m-x86_64-linux-gnu" | grep -v "cpython-37m-aarch64-linux-gnu.so" | xargs -i /bin/bash -c 'echo {} && ld -d {}' \
 && find /wbia/wildbook* -name '*.a' -print  | grep -v "cpython-37m-x86_64-linux-gnu" | grep -v "cpython-37m-aarch64-linux-gnu.so" | xargs -i /bin/bash -c 'echo {} && ld -d {}' \
 && find /wbia/wildbook* -name '*.so' -print | grep -v "cpython-37m-x86_64-linux-gnu" | grep -v "cpython-37m-aarch64-linux-gnu.so" | xargs -i /bin/bash -c 'echo {} && ld -d {}'

# Undo Fix
ENV LD_PRELOAD ""

HEALTHCHECK --interval=10s --timeout=5s --retries=60 --start-period=1h \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

STOPSIGNAL SIGTERM

# Port for the web server
EXPOSE 5000

ENTRYPOINT ["/bin/entrypoint", "/virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --logdir /data/db/logs --web --containerized --production"]

CMD []

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.depricated

COPY ./_config/deprecated.sh /bin/deprecated

RUN if [ "$(uname -m)" != "aarch64" ] ; then deprecated ; fi
