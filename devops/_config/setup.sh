#!/bin/bash

if [ "${HOST_USER}" != "root" ]; then
    # addgroup --system --non-unique --gid ${HOST_UID} ${HOST_USER}
    adduser --uid ${HOST_UID} --system --group --force-badname ${HOST_USER}
fi

if [ ! -d "/data/db" ]; then
   mkdir -p /data/db
   chown ${HOST_USER}:${HOST_USER} /data/db
   chmod 750 /data/db
fi

rm -rf /cache

if [ "${HOST_USER}" == "root" ]; then
    export HOST_CACHE=/root/.cache
else
    export HOST_CACHE=/home/${HOST_USER}/.cache
fi

rm -rf ${HOST_CACHE}

ln -s /cache ${HOST_CACHE}

mkdir /cache

chown ${HOST_USER}:${HOST_USER} ${HOST_CACHE}

chown ${HOST_USER}:${HOST_USER} /cache

# Hotfixes!

# PermissionError: [Errno 13] Permission denied: '/wbia/wbia-plugin-pie/wbia_pie/examples/manta-demo/db_localised'
chown -R ${HOST_USER}:${HOST_USER} /wbia/wbia-plugin-pie/

# Web error wbia.control.controller_inject.WebMatchThumbException, old symlinks expecting /data/docker to exist
ln -s /data/db /data/docker
