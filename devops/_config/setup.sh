#!/bin/bash

if [ "${HOST_USER}" != "root" ]; then
    # addgroup --system --non-unique --gid ${HOST_UID} ${HOST_USER}
    adduser --uid ${HOST_UID} --system --group --force-badname ${HOST_USER}
fi

addgroup --system --gid 999 docker

usermod -aG docker ${HOST_USER}

if [ ! -d "/data/db" ]; then
   mkdir -p /data/db
   chown ${HOST_USER}:${HOST_USER} /data/db
   chmod 750 /data/db
fi

cd ~/

rm -rf .cache

ln -s /cache .cache

chown ${HOST_USER}:${HOST_USER} .cache

chown ${HOST_USER}:${HOST_USER} /cache

# Hotfixes!

# PermissionError: [Errno 13] Permission denied: '/wbia/wbia-plugin-pie/wbia_pie/examples/manta-demo/db_localised'
chown -R ${HOST_USER}:${HOST_USER} /wbia/wbia-plugin-pie/

# Web error wbia.control.controller_inject.WebMatchThumbException, old symlinks expecting /data/docker to exist
rm -rf /data/docker

rm -rf /data/db/db

cd /data

ln -s /data/db docker
