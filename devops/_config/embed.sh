#!/bin/bash

set -ex

source /bin/setup

# If running as root, don't try to switch groups
if [ "${HOST_USER}" = "root" ]; then
    exec /virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --cmd "$@"
else
    exec gosu ${HOST_USER}:docker /virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --cmd "$@"
fi
