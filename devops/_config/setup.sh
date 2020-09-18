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
ln -s ${HOME}/.config/ /config
chown ${HOST_USER}:${HOST_USER} /config
