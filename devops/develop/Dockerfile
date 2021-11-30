ARG WBIA_FINAL_IMAGE=wildme/wbia:latest

FROM ${WBIA_FINAL_IMAGE} as org.wildme.wbia.final

COPY ./ /wbia/wildbook-ia

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && pip uninstall -y wildbook-ia \
 && cd /wbia/wildbook-ia \
 && /bin/bash run_developer_setup.sh \
 && pip install -r /tmp/code/requirements/tests.txt \
 && pip install -r /tmp/code/requirements/runtime.txt \
 && pip install -r /tmp/code/requirements/postgres.txt \
 && pip install -e .'
