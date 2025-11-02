ARG WBIA_FINAL_IMAGE=wildme/wbia:latest

FROM ${WBIA_FINAL_IMAGE} AS org.wildme.wbia.latest

COPY ./ /wbia/wildbook-ia

RUN /bin/bash -xc '. /virtualenv/env3/bin/activate \
 && pip uninstall -y wildbook-ia \
 && cd /wbia/wildbook-ia \
 && /bin/bash run_developer_setup.sh \
 && pip install --upgrade pip setuptools-scm \
 && pip install --no-cache-dir -r /wbia/wildbook-ia/requirements/tests.txt \
 && pip install --no-cache-dir -r /wbia/wildbook-ia/requirements/runtime.txt \
 && pip install --no-cache-dir -r /wbia/wildbook-ia/requirements/postgres.txt \
 && pip install --no-cache-dir -e .'

COPY devops/smoke_plugins.py /tmp/smoke_plugins.py

RUN set -ex \
 && /virtualenv/env3/bin/python -c "import wbia; from wbia.__main__ import smoke_test; smoke_test()" \
 && /virtualenv/env3/bin/python /tmp/smoke_plugins.py || true
