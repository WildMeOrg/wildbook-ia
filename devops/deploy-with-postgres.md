# Deploying the application with PostgreSQL

These instructions are for deploying an existing application from SQLite databases to a PostgreSQL database.

See `devops/migrate.sh` for a full script to migrate from SQLite3 with Docker -> PostgreSQL with Docker Compose

## Structure

- `$DB_NAME` - Name for the project (e.g. `mantas` or `jaguar`)
  - `${DB_PATH}` - The WBIA database path, contains `_ibsdb`, `smart_patrol`, etc.
  - `${DB_POSTGRES}` - The WBIA PostgreSQL database data path
  - `${DB_DEVOPS}` - Docker compose DevOps files
    - `docker-compose.yml` - compose configuration
    - `docker-compose.yml` - compose configuration
    - `.env` - environment variables for both `docker-compose.yml` files and containers
    - `init-db.sh` - database initialization script (see `wildbook-ia/.dockerfiles/init-db.sh`)

Note, our use of the `.env` file in this documentation is two fold. The file is first implicitly used by `docker-compose` to fill in `${var}` variables within the `docker-compose.yml` file. It is secondly used within the `docker-compose.yml` within the `env_file` setting to pass variables into the container as environment variables. This could be pulled into two separate files if you so choose.

## Notes

Make sure there aren't any random databases laying around that aren't backup (i.e. have `backup` in the filename). For example, the database file named `finfindr.old.sqlite` will be picked up by the migration script unless removed.

## Run the service with:

```
cd ${DB_DEVOPS}
docker-compose up -d
```
