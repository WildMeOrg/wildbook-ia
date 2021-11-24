ARG WBIA_BASE_IMAGE=wildme/wbia-base:latest

ARG WBIA_PROVISION_IMAGE=wildme/wbia-provision:latest

FROM ${WBIA_PROVISION_IMAGE} as org.wildme.wbia.latest

# Grab latest code
RUN set -ex \
 && git config --global user.email "dev@wildme.org" \
 && git config --global user.name "Wild Me" \
 && find /wbia/wbia* -name '.git' -type d -print0 | xargs -0 -i /bin/bash -c \
    'cd {} && cd .. && echo $(pwd) && git reset --hard origin/develop && git pull' \
 && cd /wbia/wbia-plugin-curvrank-v2/wbia_curvrank_v2 \
 && git reset --hard origin/develop \
 && git pull \
 && cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git reset --hard origin/develop \
 && git pull \
 && cd /wbia/wbia-plugin-lca/wbia_lca \
 && git reset --hard origin/develop \
 && git pull \
 && cd /wbia/wbia-plugin-orientation/ \
 && git reset --hard origin/develop \
 && git pull \
 && cd /wbia/wildbook-ia/ \
 && git reset --hard origin/main \
 && git pull

##########################################################################################

FROM ${WBIA_BASE_IMAGE} as org.wildme.wbia.install

LABEL autoheal=true

ARG VERSION="3.7.2"

ARG VCS_URL="https://github.com/WildMeOrg/wildbook-ia"

ARG VCS_REF="main"

ARG BUILD_DATE="2020"

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

WORKDIR /wbia

COPY --from=org.wildme.wbia.latest /virtualenv /virtualenv

COPY --from=org.wildme.wbia.latest /wbia /wbia

# Add theano configuration file
ADD ./_config/theanorc /root/.theanorc

# Add setup script for python development
ADD ./_config/setup.sh /bin/setup

# Add embed script for python development
ADD ./_config/embed.sh /bin/embed

# Add embed script for python development
ADD ./_config/embedpg.sh /bin/embedpg

# Add (non-root) bash shell script for python development
ADD ./_config/shell.sh /bin/shell

# Add entrypoint
ADD ./_config/entrypoint.sh /bin/entrypoint

# Add Python health check
ADD ./_config/healthcheck.py /bin/healthcheck.py

ENV LD_PRELOAD "/usr/lib/aarch64-linux-gnu/libgomp.so.1:${LD_PRELOAD}"

ENV LD_PRELOAD "/virtualenv/env3/lib/python3.7/site-packages/scikit_image.libs/libgomp-d22c30c5.so.1.0.0:${LD_PRELOAD}"

ENV LD_PRELOAD "/virtualenv/env3/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1:${LD_PRELOAD}"

# Run smoke tests
RUN set -ex \
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

# Specify database directory
RUN set -x \
 && mkdir -p /data \
 && /virtualenv/env3/bin/python -m wbia.dev --set-workdir /data

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

STOPSIGNAL SIGTERM

# Port for the web server
EXPOSE 5000

ENTRYPOINT ["/bin/entrypoint", "/virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --logdir /data/db/logs --web --containerized --production"]

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.configure

CMD []
