#!/bin/bash

set -ex

source /bin/setup

exec gosu ${HOST_USER}:docker /virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --db-uri $DB_URI --cmd $@
