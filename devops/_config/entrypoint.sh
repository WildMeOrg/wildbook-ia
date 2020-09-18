#!/bin/bash

set -ex

addgroup --system ${HOST_USER}

adduser --uid ${HOST_UID} --system --group ${HOST_USER}

if [ ! -d "/data/db" ]
then
   mkdir -p /data/db
   chown ${HOST_USER}:${HOST_USER} /data/db
   chmod 750 /data/db
fi

exec gosu ${HOST_USER}:${HOST_USER} $@
