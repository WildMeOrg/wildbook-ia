#!/bin/bash

# DOCKER_SOCKET=/var/run/docker.sock

if [ "${HOST_USER}" != "root" ]; then
    # addgroup --system --non-unique --gid ${HOST_UID} ${HOST_USER}
    adduser --uid ${HOST_UID} --system --group --force-badname ${HOST_USER}
fi

# if [ -S ${DOCKER_SOCKET} ]; then
#     DOCKER_GID=$(stat -c '%g' ${DOCKER_SOCKET})
#     DOCKER_GROUP=docker

#     addgroup --system --gid ${DOCKER_GID} ${DOCKER_GROUP}
#     usermod -aG ${DOCKER_GROUP} ${HOST_USER}

#     sg ${DOCKER_GROUP} -c "/bin/bash"
# fi

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

# Allow Tensorflow to use GPU memory more dynamically
export TF_FORCE_GPU_ALLOW_GROWTH=true
