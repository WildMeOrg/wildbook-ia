#!/bin/bash

export DB_NAME="carnivore"  # e.g. `manta` or `jaguar`
export DB_PATH="/data/ibeis/ACW_Master"
export WBIA_CACHE_PATH="/data/ibeis/cache"

export DB_PORT=5011

export WB_DOMAIN="africancarnivore.wildbook.org"

export HOST_UID=1001
export HOST_USER="jason.parham"

export POSTGRES_PASSWORD="8kzu8Br3YMcdVLbJPmZGCkTuXM6j6n7L"
export DB_PASSWORD="pZXUt3kysvcRPQCwCpcrACtPMC5aBQuM"

export AWS_ACCESS_KEY_ID="AKIAJPDDJLGOCKGVSZAA"
export AWS_SECRET_ACCESS_KEY="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"

##########

export DB_USER="wbia"
export POSTGRES_USER="postgres"
export POSTGRES_DB="db"

export DB_DIR="/data/db"
export WBIA_CACHE_DIR="/cache"

export DB_PATH_POSTGRES="${DB_PATH}/_ibsdb_postgres"
export DB_DEVOPS="${DB_PATH}/_devops"
export DB_ENV="${DB_DEVOPS}/.env"

export COMPOSE_MIGRATE_YAML="${DB_DEVOPS}/docker-compose.migrate.yml"
export COMPOSE_RUNTIME_YAML="${DB_DEVOPS}/docker-compose.yml"

##########

# sudo apt install python3-pip
# sudo pip3 install docker-compose
# sudo docker swarm init

docker --version
docker-compose --version

docker pull postgres:latest
docker pull wildme/wbia:develop
docker rm -f autoheal

##########

cd ${DB_PATH}

sudo rm -rf ${DB_PATH_POSTGRES}
sudo rm -rf ${DB_DEVOPS}
sudo rm -rf ${DB_PATH}/_ibsdb/_ibeis_database_backup_*.sqlite3
sudo rm -rf ${DB_PATH}/_ibsdb/_ibeis_staging_backup_*.sqlite3
sudo rm -rf ${DB_PATH}/_ibsdb/_ibeis_backups/

find . -name '*.sqlite*' -print0 | du -ch --files0-from=- | sort -h

mkdir -p ${DB_PATH_POSTGRES}
mkdir -p ${DB_DEVOPS}

##########

wget -O ${DB_DEVOPS}/init-db.sh https://raw.githubusercontent.com/WildMeOrg/wildbook-ia/develop/.dockerfiles/init-db.sh

echo """
COMPOSE_PROJECT_NAME=${DB_NAME}

HOST_UID=${HOST_UID}
HOST_USER=\"${HOST_USER}\"

POSTGRES_USER=\"${POSTGRES_USER}\"
POSTGRES_DB=\"${POSTGRES_DB}\"
POSTGRES_PASSWORD=\"${POSTGRES_PASSWORD}\"

DB_NAME=\"${DB_NAME}\"
DB_USER=\"${DB_USER}\"
DB_PASSWORD=\"${DB_PASSWORD}\"
DB_URI=\"postgresql://${DB_USER}:${DB_PASSWORD}@${POSTGRES_DB}/${DB_NAME}\"
DB_DIR=\"${DB_DIR}\"

AWS_ACCESS_KEY_ID=\"${AWS_ACCESS_KEY_ID}\"
AWS_SECRET_ACCESS_KEY=\"${AWS_SECRET_ACCESS_KEY}\"

WB_DOMAIN=\"${WB_DOMAIN}\"
""" >> ${DB_ENV}

echo """version: \"3\"
services:
  db:
    image: postgres:latest
    volumes:
      - ${DB_PATH_POSTGRES}:/var/lib/postgresql/data
      - ${DB_DEVOPS}/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
    env_file: ${DB_ENV}
  wbia:
    image: wildme/wbia:develop
    depends_on:
      - \"db\"
    shm_size: '1g'
    command: [\"sleep\", \"infinity\"]
    volumes:
      - ${DB_PATH}:${DB_DIR}
    env_file: ${DB_ENV}
""" >> ${COMPOSE_MIGRATE_YAML}

echo """version: \"3\"
services:
  db:
    image: postgres:latest
    volumes:
      - ${DB_PATH_POSTGRES}:/var/lib/postgresql/data
    env_file: ${DB_ENV}
  wbia:
    image: wildme/wbia:develop
    depends_on:
      - \"db\"
    command: [\"--db-uri\", \"\$DB_URI\", \"--https\", \"--container-name\", \"\$DB_NAME\", \"--wildbook-target\", \"\$WB_DOMAIN\"]
    ports:
      - \"${DB_PORT}:5000\"
    volumes:
      - ${DB_PATH}:${DB_DIR}
      - ${WBIA_CACHE_PATH}:${WBIA_CACHE_DIR}
    env_file: ${DB_ENV}
    deploy:
    resources:
      reservations:
        devices:
          - capabilities:
            - gpu
    restart: unless-stopped
""" >> ${COMPOSE_RUNTIME_YAML}

ls -al ${DB_DEVOPS}

##########

cd ${DB_DEVOPS}

docker-compose -f ${COMPOSE_MIGRATE_YAML} down

docker-compose -f ${COMPOSE_MIGRATE_YAML} up -d

# HOTFIX FOR UBELT
docker-compose -f ${COMPOSE_MIGRATE_YAML} exec wbia bash -c \
  '/virtualenv/env3/bin/pip install opencv-python'

docker-compose -f ${COMPOSE_MIGRATE_YAML} exec wbia bash -c \
  '/virtualenv/env3/bin/wbia-migrate-sqlite-to-postgres -v --db-dir ${DB_DIR} --db-uri ${DB_URI}'

# touch /data/ibeis/ACW_Master/_ibsdb/_ibeis_cache/featcache.sqlite

# sudo watch -n 30 "du -sh ${DB_PATH_POSTGRES}"

docker-compose -f ${COMPOSE_MIGRATE_YAML} exec wbia bash -c \
  '/virtualenv/env3/bin/wbia-compare-databases -v --db-dir ${DB_DIR} --pg-uri ${DB_URI} --check-pc 1 --check-max -1'

docker-compose -f ${COMPOSE_MIGRATE_YAML} down

##########

docker rm -f ${DB_NAME}

cd ${DB_PATH}

find . -name '*.sqlite*' -print | grep -v "_ibsdb/_ibeis_backups" | grep -v "_ibsdb_postgres" | grep -v ".migrated" | xargs -i /bin/bash -c 'mv {} {}.migrated'  # echo {}
find . -name '*.sqlite*' -print | grep -v ".migrated"

##########

cd ${DB_DEVOPS}

docker-compose up -d

docker logs --follow ${DB_NAME}_wbia_1

##########

docker run -d \
    --name autoheal \
    --restart=always \
    -e AUTOHEAL_CONTAINER_LABEL=autoheal \
    -e AUTOHEAL_INTERVAL=15 \
    -e AUTOHEAL_START_PERIOD=3600 \
    -e AUTOHEAL_DEFAULT_STOP_TIMEOUT=60 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    willfarrell/autoheal
