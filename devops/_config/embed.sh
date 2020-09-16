#!/bin/bash

set -ex

# Executes as `wbia` with PID 1
exec gosu wbia:wbia /virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --cmd $@
