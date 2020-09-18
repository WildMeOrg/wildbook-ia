#!/bin/bash

set -ex

exec gosu ${HOST_USER}:${HOST_USER} /virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --cmd $@
