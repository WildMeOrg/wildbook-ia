ARG WBIA_FINAL_IMAGE=wildme/wbia:latest

FROM ${WBIA_FINAL_IMAGE} as org.wildme.wbia.latest

COPY ./ /wbia/wildbook-ia

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && pip uninstall -y wildbook-ia \
 && cd /wbia/wildbook-ia \
 && /bin/bash run_developer_setup.sh \
 && pip install -r /wbia/wildbook-ia/requirements/tests.txt \
 && pip install -r /wbia/wildbook-ia/requirements/runtime.txt \
 && pip install -r /wbia/wildbook-ia/requirements/postgres.txt \
 && pip install -e .'

# Temporary fix for ARM64
ENV LD_PRELOAD "/virtualenv/env3/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1"

RUN set -ex \
 && /virtualenv/env3/bin/python -c "import wbia; from wbia.__main__ import smoke_test; smoke_test()"

# Undo Fix
ENV LD_PRELOAD ""

 # Update locate database
RUN updatedb
