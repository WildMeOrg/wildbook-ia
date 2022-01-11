#!/bin/bash

set -ex

source /bin/setup

exec gosu ${HOST_USER}:${HOST_USER} /virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --cmd $@
