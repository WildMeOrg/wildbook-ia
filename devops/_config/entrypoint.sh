#!/bin/bash

set -ex

adduser --uid ${HOST_UID} --system --no-create-home --group ${HOST_USER}

exec gosu ${HOST_USER}:${HOST_USER} $@
