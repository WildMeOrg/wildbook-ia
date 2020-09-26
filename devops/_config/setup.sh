#!/bin/bash

if [ "${HOST_USER}" != "root" ]; then
    # addgroup --system --non-unique --gid ${HOST_UID} ${HOST_USER}
    adduser --uid ${HOST_UID} --system --group --force-badname ${HOST_USER}
fi

addgroup --system --gid 999 docker

usermod -aG docker ${HOST_USER}

if [ ! -d "/data/db" ]; then
   mkdir -p /data/db/
   chown ${HOST_USER}:${HOST_USER} /data/db/
   chmod 750 /data/db/
fi

if [ ! -d "/cache" ]; then
   mkdir -p /cache
   chown ${HOST_USER}:${HOST_USER} /cache
   chmod 750 /cache
fi

rm -rf ~/.cache

ln -s -T /cache ~/.cache

chown ${HOST_USER}:${HOST_USER} ~/.cache/

# Hotfixes!

# PermissionError: [Errno 13] Permission denied: '/wbia/wbia-plugin-pie/wbia_pie/examples/manta-demo/db_localised'
chown -R ${HOST_USER}:${HOST_USER} /wbia/wbia-plugin-pie/

# Web error wbia.control.controller_inject.WebMatchThumbException, old symlinks expecting /data/docker to exist
if [ ! -d "/data/docker" ]; then
    ln -s -T /data/db /data/docker
fi
