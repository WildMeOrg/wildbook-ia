#!/bin/bash

export DB_NAME="jaguar"
export DB_PORT=6013

export POSTGRES_PASSWORD="< INSERT POSTGRES_PASSWORD >"
export DB_PASSWORD="< INSERT DB_PASSWORD >"

##########

export HOST_UID=1000
export HOST_USER="wildme"

export DB_USER="wbia"
export POSTGRES_USER="postgres"
export POSTGRES_DB="db"

export DB_PATH="/data/${DB_NAME}"
export DB_DIR="/data/db"
export WBIA_CACHE_PATH="/data/cache"
export WBIA_CACHE_DIR="/cache"

export DB_POSTGRES="${DB_PATH}/_ibsdb_postgres"
export DB_SCRIPTS="${DB_PATH}/_scripts"
export DB_ENV="${DB_SCRIPTS}/env.wbia.${DB_NAME}"

export COMPOSE_MIGRATE_YAML="${DB_SCRIPTS}/docker-compose.migrate.yml"
export COMPOSE_RUNTIME_YAML="${DB_SCRIPTS}/docker-compose.yml"

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

sudo rm -rf ${DB_POSTGRES}
sudo rm -rf ${DB_SCRIPTS}
sudo rm -rf ${DB_PATH}/_ibsdb/_ibeis_database_backup_*.sqlite3
sudo rm -rf ${DB_PATH}/_ibsdb/_ibeis_staging_backup_*.sqlite3
sudo rm -rf ${DB_PATH}/_ibsdb/_ibeis_backups/

find . -name '*.sqlite*' -print0 | du -ch --files0-from=- | sort -h

mkdir -p ${DB_POSTGRES}
mkdir -p ${DB_SCRIPTS}

##########

wget -O ${DB_SCRIPTS}/init-db.sh https://raw.githubusercontent.com/WildMeOrg/wildbook-ia/develop/.dockerfiles/init-db.sh

echo """
COMPOSE_PROJECT_NAME=${DB_NAME}
""" >> ${DB_SCRIPTS}/.env

echo """
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

AWS_ACCESS_KEY_ID=\"< INSERT AWS_ACCESS_KEY_ID >\"
AWS_SECRET_ACCESS_KEY=\"< INSERT AWS_SECRET_ACCESS_KEY >\"
""" >> ${DB_ENV}

echo """version: \"3\"
services:
  db:
    image: postgres:latest
    volumes:
      - ${DB_POSTGRES}:/var/lib/postgresql/data
      - ${DB_SCRIPTS}/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
    env_file: ${DB_ENV}
  wbia:
    image: wildme/wbia:develop
    depends_on:
      - \"db\"
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
      - ${DB_POSTGRES}:/var/lib/postgresql/data
    env_file: ${DB_ENV}
  wbia:
    image: wildme/wbia:develop
    depends_on:
      - \"db\"
    args: [\"--db-uri\", \"\${DB_URI}\", \"--https\", \"--container-name\", \"\${DB_NAME}\"]
    ports:
      - \"${DB_PORT}:5000\"
    volumes:
      - ${DB_PATH}:${DB_DIR}
      - ${WBIA_CACHE_PATH}:${WBIA_CACHE_DIR}
    env_file: ${DB_ENV}
    restart: unless-stopped
""" >> ${COMPOSE_RUNTIME_YAML}

ls -al ${DB_SCRIPTS}

##########

cd ${DB_SCRIPTS}

docker-compose -f ${COMPOSE_MIGRATE_YAML} down

docker-compose -f ${COMPOSE_MIGRATE_YAML} up -d

# HOTFIX FOR UBELT
docker-compose -f ${COMPOSE_MIGRATE_YAML} exec wbia bash -c \
  '/virtualenv/env3/bin/pip install opencv-python'

docker-compose -f ${COMPOSE_MIGRATE_YAML} exec wbia bash -c \
  '/virtualenv/env3/bin/wbia-migrate-sqlite-to-postgres -v --db-dir ${DB_DIR} --db-uri ${DB_URI}'

# sudo watch -n 30 "du -sh ${DB_POSTGRES}"

docker-compose -f ${COMPOSE_MIGRATE_YAML} exec wbia shell -c \
  '/virtualenv/env3/bin/wbia-compare-databases -v --db-dir ${DB_DIR} --pg-uri ${DB_URI} --check-pc 1 --check-max -1'

docker-compose -f ${COMPOSE_MIGRATE_YAML} down

##########

docker rm -f ${DB_NAME}

cd ${DB_PATH}

find . -name '*.sqlite*' -print | grep -v "_ibsdb/_ibeis_backups" | xargs -i /bin/bash -c 'echo {}'  # && mv {} {}.migrated'

##########

cd ${DB_SCRIPTS}

docker-compose up -d

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
