#!/bin/bash

# Fix for Arm64 LD_PRELOAD on libgomp
if [ "$(uname -m)" == "aarch64" ]; then
    export LD_PRELOAD="$(locate libgomp | grep ".so" | xargs | sed -e 's/ /:/g'):${LD_PRELOAD}"
fi

if [ "${HOST_USER}" != "root" ]; then
    # addgroup --system --non-unique --gid ${HOST_UID} ${HOST_USER}
    if id "${HOST_USER}" >/dev/null 2>&1; then
        echo "Using existing user ${HOST_USER}"
    else
        echo "Adding new user ${HOST_USER}"
        adduser --uid ${HOST_UID} --system --group --force-badname ${HOST_USER}
    fi
    export HOME_FOLDER=$(eval echo ~${HOST_USER})
else
    export HOME_FOLDER=${HOME}
fi

DOCKER_SOCKET=/var/run/docker.sock

if [ -S ${DOCKER_SOCKET} ]; then
    DOCKER_GID=$(stat -c '%g' ${DOCKER_SOCKET})
    DOCKER_GROUP=docker
    groupmod -g ${DOCKER_GID} ${DOCKER_GROUP}

    # addgroup -q --system --gid ${DOCKER_GID} ${DOCKER_GROUP}
    # usermod -aG ${DOCKER_GROUP} ${HOST_USER}
    # sg ${DOCKER_GROUP} -c "/bin/bash"
fi

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

rm -rf ${HOME_FOLDER}/.cache
ln -s -T /cache ${HOME_FOLDER}/.cache
chown ${HOST_USER}:${HOST_USER} ${HOME_FOLDER}/.cache/

if [ "${HOST_USER}" != "root" ]; then
    cp /root/.theanorc ${HOME_FOLDER}/.theanorc
fi

chown ${HOST_USER}:${HOST_USER} ${HOME_FOLDER}/.theanorc
# Hotfixes!

# PermissionError: [Errno 13] Permission denied: '/wbia/wbia-plugin-pie/wbia_pie/examples/manta-demo/db_localised'
if [ -d "/wbia/wbia-plugin-pie" ]; then
    chown -R ${HOST_USER}:${HOST_USER} /wbia/wbia-plugin-pie
fi

# Web error wbia.control.controller_inject.WebMatchThumbException, old symlinks expecting /data/docker to exist
if [ ! -d "/data/docker" ]; then
    ln -s -T /data/db /data/docker
fi

# Allow Tensorflow to use GPU memory more dynamically
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Comment out localhost from /etc/hosts for development
sed 's/^\([^#].*localhost\)/# \1/' /etc/hosts >/etc/hosts.new
cat /etc/hosts.new >/etc/hosts
