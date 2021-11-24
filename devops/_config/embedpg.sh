#!/bin/bash

set -ex

setup

exec gosu ${HOST_USER}:${HOST_USER} /virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --db-uri $DB_URI --cmd $@
