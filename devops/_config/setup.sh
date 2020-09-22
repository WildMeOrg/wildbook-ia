#!/bin/bash

if [ "${HOST_USER}" != "root" ]; then
    addgroup --system ${HOST_USER}
    adduser --uid ${HOST_UID} --system --group ${HOST_USER}
fi

if [ ! -d "/data/db" ]; then
   mkdir -p /data/db
   chown ${HOST_USER}:${HOST_USER} /data/db
   chmod 750 /data/db
fi

rm -rf /config

if [ "${HOST_USER}" == "root" ]; then
    export HOST_CONFIG=/root/.config
else
    export HOST_CONFIG=/home/${HOST_USER}/.config
fi

rm -rf ${HOST_CONFIG}

ln -s /config ${HOST_CONFIG}

mkdir /config

chown ${HOST_USER}:${HOST_USER} ${HOST_CONFIG}

chown ${HOST_USER}:${HOST_USER} /config

# Hotfixes!

# PermissionError: [Errno 13] Permission denied: '/wbia/wbia-plugin-pie/wbia_pie/examples/manta-demo/db_localised'
chown -R ${HOST_USER}:${HOST_USER} /wbia/wbia-plugin-pie/
