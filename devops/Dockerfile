ARG WBIA_BASE_IMAGE=wildme/wbia-base:latest

ARG WBIA_PROVISION_IMAGE=wildme/wbia-provision:latest

FROM ${WBIA_PROVISION_IMAGE} AS org.wildme.wbia.latest

# Trim unnecessary VCS metadata and caches before handing off to final stage to
# keep multi-stage COPY operations within GitHub runner disk limits.
RUN set -ex \
 && find /wbia -mindepth 1 -maxdepth 4 -type d -name '.git' -prune -exec rm -rf {} + \
 && find /wbia -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && rm -rf /wbia/wildbook-ia/.cache /wbia/wildbook-ia/.mypy_cache || true
#  && cd /wbia/wbia-plugin-curvrank/wbia_curvrank_v2 \
# && git reset --hard origin/main \
# && git pull \

# (Removed non-deterministic git update step; build now uses code baked into provision image + build context only.)

##########################################################################################

FROM ${WBIA_BASE_IMAGE} AS org.wildme.wbia.install

ARG VERSION="3.7.2"

ARG VCS_URL="https://github.com/WildMeOrg/wildbook-ia"

ARG VCS_REF="main"

ARG BUILD_DATE="2023"

LABEL autoheal="true" \
    org.opencontainers.image.source="https://github.com/WildMeOrg/wildbook-ia" \
    org.wildme.vendor="Wild Me" \
    org.wildme.url="https://wildme.org" \
    org.wildme.name="Wildbook IA" \
    org.wildme.description="Wildbook's Image Analysis (WBIA) backend service supporting machine learning for wildlife conservation" \
    org.wildme.version="${VERSION}" \
    org.wildme.vcs-url="${VCS_URL}" \
    org.wildme.vcs-ref="${VCS_REF}" \
    org.wildme.build-date="${BUILD_DATE}" \
    org.wildme.docker.schema-version="1.0"

ENV HOST_USER=root HOST_UID=0
# NOTE: AWS credentials should be provided at runtime (e.g. -e or IAM role), not baked into the image.
# ARG AWS_ACCESS_KEY_ID
# ARG AWS_SECRET_ACCESS_KEY

WORKDIR /data/db

COPY --from=org.wildme.wbia.latest /virtualenv /virtualenv

COPY --from=org.wildme.wbia.latest /wbia /wbia

# theano configuration file
COPY devops/_config/theanorc /root/.theanorc

# setup script for python development
COPY devops/_config/setup.sh /bin/setup

# embed script for python development
COPY devops/_config/embed.sh /bin/embed

# embed script for python development
COPY devops/_config/embedpg.sh /bin/embedpg

# (non-root) bash shell script for python development
COPY devops/_config/shell.sh /bin/shell

# entrypoint
COPY devops/_config/entrypoint.sh /bin/entrypoint

# Python health check
COPY devops/_config/healthcheck.py /bin/healthcheck.py

# (removed updatedb: utility not installed in base image and non-essential)

# Temporary fix for ARM64
#ENV LD_PRELOAD "${LD_PRELOAD}:/virtualenv/env3/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1"

#ENV LD_PRELOAD "${LD_PRELOAD}:/virtualenv/env3/lib/python3.7/site-packages/scikit_image.libs/libgomp-d22c30c5.so.1.0.0"

# Fix PyQt5 and install OpenCV development headers for pyhesaff build
RUN set -ex \
 && apt-get update \
 && apt-get remove -y \
     python3-pyqt5* \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
     qtbase5-dev \
     qtchooser \
     qt5-qmake \
     qtbase5-dev-tools \
     libopencv-dev \
 && /virtualenv/env3/bin/pip uninstall -y \
     pyqt5 \
 && /virtualenv/env3/bin/pip install --no-cache-dir --upgrade \
     pyqt5 \
 && apt-get clean \
 && apt-get autoclean \
 && apt-get autoremove -y \
 && rm -rf /var/cache/apt \
 && rm -rf /var/lib/apt/lists/*

# Fix Lasagne (conditional: only if plugin exists)
RUN set -ex; \
 if [ -d /wbia/wbia-plugin-cnn ]; then \
     . /virtualenv/env3/bin/activate; \
     cd /wbia/wbia-plugin-cnn; \
     /bin/bash run_developer_setup.sh || echo "[warn] wbia-plugin-cnn setup script failed"; \
 else \
     echo "[skip] wbia-plugin-cnn not present"; \
 fi

# Fix OpenCV
RUN set -ex \
 && /virtualenv/env3/bin/pip uninstall -y \
     opencv-python \
     opencv-contrib-python \
     opencv-python-headless \
     opencv-contrib-python-headless \
 && /virtualenv/env3/bin/pip install --no-cache-dir --upgrade \
     opencv-contrib-python-headless \
 && /virtualenv/env3/bin/pip uninstall -y \
     nvidia_cublas_cu11

# Run smoke tests
# && /virtualenv/env3/bin/python -c "import wbia;             from wbia.__main__ import smoke_test; smoke_test()" \
# && /virtualenv/env3/bin/python -c "import wbia_flukematch;  from wbia_flukematch.plugin   import *" \  -- commented out due to Lasagne package dependency issues
#  && /virtualenv/env3/bin/python -c "import wbia_finfindr;    from wbia_finfindr.__main__ import main;      main()" \
#  && /virtualenv/env3/bin/python -c "import wbia_curvrank_v2; from wbia_curvrank_v2._plugin import *" \
#&& /virtualenv/env3/bin/python -c "import wbia_finfindr;    from wbia_finfindr._plugin    import *" \
# && /virtualenv/env3/bin/python -c "import wbia_whaleridgefindr;    from wbia_whaleridgefindr._plugin     import *" \


COPY devops/smoke_plugins.py /tmp/smoke_plugins.py

# Run smoke tests (moved all inline heredoc Python to external script to avoid heredoc parse issues)
RUN set -ex \
 && mkdir -p /data \
 && /virtualenv/env3/bin/python -m wbia.dev --set-workdir /data --preload-exit \
 && /virtualenv/env3/bin/python /tmp/smoke_plugins.py \
 && for pattern in /wbia/wbia* /wbia/wildbook*; do \
            find $pattern -name '*.a' -print | grep -v 'cpython-37m-' | xargs -r -i /bin/bash -c 'echo {} && ld -d {}' || true; \
            find $pattern -name '*.so' -print | grep -v 'cpython-37m-' | xargs -r -i /bin/bash -c 'echo {} && ld -d {}' || true; \
        done || true


# Undo Fix
ARG LIGHT_MODE=0
RUN set -ex \
 && if [ "$LIGHT_MODE" = "1" ]; then \
      echo "[light-mode] pruning torch CUDA libraries (final stage)"; \
      /virtualenv/env3/bin/python - <<'PY' || true
try:
    import torch, os, glob
    libdir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    for p in glob.glob(os.path.join(libdir, 'libtorch_cuda*')):
        try:
            os.remove(p); print('[light-mode] removed', p)
        except OSError as e:
            print('[light-mode] failed to remove', p, e)
except Exception as e:
    print('[light-mode] torch not importable or cleanup issue:', e)
PY \
    ; fi

ENV LD_PRELOAD=""

HEALTHCHECK --interval=10s --timeout=5s --retries=60 --start-period=1h \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

STOPSIGNAL SIGTERM

# Port for the web server
EXPOSE 5000

ENTRYPOINT ["/bin/entrypoint", "/virtualenv/env3/bin/python", "-m", "wbia.dev", "--dbdir", "/data/db", "--logdir", "/data/db/logs", "--web", "--containerized", "--production"]

CMD []

##########################################################################################
FROM org.wildme.wbia.install AS org.wildme.wbia.depricated

COPY devops/_config/deprecated.sh /bin/deprecated

RUN if [ "$(uname -m)" != "aarch64" ] ; then deprecated ; fi
