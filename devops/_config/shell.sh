#!/bin/bash

set -ex

addgroup --system ${HOST_USER}

adduser --uid ${HOST_UID} --system --group ${HOST_USER}

exec gosu ${HOST_USER}:${HOST_USER} /bin/bash
