# Deploying the application with Postgres

These instructions are for deploying an existing application from SQLite databases to a Postgres database.

## Structure

Root the project

- `$PROJECT` - workspace for the project, rooted by project name (e.g. `mantas`)
  - `docker-compose.yml` - compose configuration
  - `.env` - environment variables for both `docker-compose.yml` files and containers
  - `dbdir` - contains `_ibsdb`, `smart_patrol`, etc.)
  - `pgdata` - postgres database)
  - `init-db.sh` - database initialization script (see `wildbook-ia/.dockerfiles/init-db.sh`)

Note, our use of the `.env` file in this documentation is two fold. The file is first implicitly used by `docker-compose` to fill in `${var}` variables within the `docker-compose.yml` file. It is secondly used within the `docker-compose.yml` within the `env_file` setting to pass variables into the container as environment variables. This could be pulled into two separate files if you so choose.

## Setup and Configuration

Configure your scenario for this documentation. These environment variables help to keep this documentation generic, but are not required.

```
export PROJECT="mantas"
export PROJECT_ROOT="/external/wb-1329/$PROJECT"
mkdir -p $PROJECT_ROOT
cd $PROJECT_ROOT
# Downlaod tip of develop's init-db.sh script
wget https://raw.githubusercontent.com/WildMeOrg/wildbook-ia/develop/.dockerfiles/init-db.sh
# Make a space for the postgres database (container fixes ownership on startup)
mkdir pgdata
```

Make sure there aren't any random databases laying around that aren't backup (i.e. have `backup` in the filename). For example, the flukebook-testing data had a cache database file named `finfindr.old.sqlite`, which is going to be picked up by the migration script unless removed.

The main part of the configuration is within the `.env` file:

```
# Postgres container's 'postgres' role password
# See image's documentation for details
POSTGRES_PASSWORD="<required>"

# Our postgres database settings
DB_NAME="<required>"
DB_USER="wbia"
DB_PASSWORD="<required>"

# The URI format is "postgresql://DB_USER:DB_PASSWORD@DB_HOST/DB_NAME"
WBIA_DB_URI="postgresql://${DB_USER}:${DB_PASSWORD}@db/${DB_NAME}"
# The database directory (i.e. dbdir) as mounted in the container
WBIA_DB_DIR="/data/db"
```

## Migration

`$PROJECT_ROOT/docker-compose.in-migration.yml`:

```
version: "2"
services:
  db:
    image: postgres:10
    volumes:
      - ./pgdata:/var/lib/postgresql/data
      # See `wildbook-ia/.dockerfiles/init-db.sh` for original
      - ./init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
    env_file: ./.env
  app:
    image: wildme/wildbook-ia:latest
    command: ["sleep", "infinity"]
    volumes:
      # mount to the default dbdir location
      - ./dbdir:/data/db
    env_file: ./.env
```

Run the migration:

```
export COMPOSE_CONFIG="docker-compose.in-migration.yml"
docker-compose -f ${COMPOSE_CONFIG} up -d
date && docker-compose -f ${COMPOSE_CONFIG} exec app /bin/bash -c 'wbia-migrate-sqlite-to-postgres -v --db-dir $WBIA_DB_DIR --db-uri $WBIA_DB_URI' && date
```

Run the comparison and verification over all rows (100% check):

```
date && docker-compose -f ${COMPOSE_CONFIG} exec app /bin/bash -c 'wbia-compare-databases -v --db-dir $WBIA_DB_DIR --pg-uri $WBIA_DB_URI --check-pc 1 --check-max -1' && date
```

## Run the Application

Change to the runtime version of the docker-compose file

`$PROJECT_ROOT/docker-compose.yml`:

```
version: "2.4"
services:
  db:
    image: postgres:10
    volumes:
      - ./pgdata:/var/lib/postgresql/data
      - ./init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
    env_file: ./.env
  app:
    image: wildme/wildbook-ia:latest
    # FIXME: change docker-entrypoint.sh to allow for additive arguments
    command: ["wait-for", "db:5432", "--", "python3", "-m", "wbia.dev", "--dbdir", "${WBIA_DB_DIR}", "--logdir", "/data/logs/", "--web", "--port", "5000", "--web-deterministic-ports", "--containerized", "--cpudark", "--production", "--db-uri", "${WBIA_DB_URI}"]
    ports:
      # Generically exposed, update as needed
      - "5000"
    volumes:
      # mount to the default dbdir location
      - ./dbdir:/data/db
    env_file: ./.env
```

The `--db-uri` option flag is set via the `$WBIA_DB_URI` environment variable.

Run the service with:

```
docker-compose up -d
```
