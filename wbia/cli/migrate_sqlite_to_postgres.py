# -*- coding: utf-8 -*-
import logging
import re
import sys
from pathlib import Path

import click
import sqlalchemy

from wbia.dtool.copy_sqlite_to_postgres import (
    copy_sqlite_to_postgres,
    SqliteDatabaseInfo,
    PostgresDatabaseInfo,
    compare_databases,
)


logger = logging.getLogger('wbia')


@click.command()
@click.option(
    '--db-dir', required=True, type=click.Path(exists=True), help='database location'
)
@click.option(
    '--db-uri',
    required=True,
    help='Postgres connection URI (e.g. postgres://user:pass@host)',
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Delete all tables in the public schema in postgres',
)
def main(db_dir, db_uri, force):
    """"""
    # Set up logging
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    logger.info(f'using {db_dir} ...')

    # Create the database if it doesn't exist
    engine = sqlalchemy.create_engine(db_uri)
    try:
        engine.connect()
    except sqlalchemy.exc.OperationalError as e:
        m = re.search(r'database "([^"]*)" does not exist', str(e))
        if m:
            dbname = m.group(1)
            engine = sqlalchemy.create_engine(db_uri.rsplit('/', 1)[0])
            logger.info(f'Creating "{dbname}"...')
            engine.execution_options(isolation_level='AUTOCOMMIT').execute(
                f'CREATE DATABASE {dbname}'
            )
        else:
            raise
    finally:
        engine.dispose()

    # Check that the database hasn't already been migrated.
    db_infos = [
        SqliteDatabaseInfo(Path(db_dir)),
        PostgresDatabaseInfo(db_uri),
    ]
    differences = compare_databases(*db_infos)

    if not differences:
        logger.info('Database already migrated')
        sys.exit(0)

    # Make sure there are no tables in the public schema in postgresql
    # because we're using it as the workspace for the migration
    if 'public' in db_infos[1].get_schema():
        table_names = [
            t for schema, t in db_infos[1].get_table_names() if schema == 'public'
        ]
        if not force:
            click.echo(
                f'Tables in public schema in postgres database: {", ".join(table_names)}'
            )
            click.echo('Use --force to remove the tables in public schema')
            sys.exit(1)
        else:
            click.echo(f'Dropping all tables in public schema: {", ".join(table_names)}')
            for table_name in table_names:
                db_infos[1].engine.execute(f'DROP TABLE {table_name}')

    # Migrate
    copy_sqlite_to_postgres(Path(db_dir), db_uri)

    # Verify the migration
    differences = compare_databases(*db_infos)

    if differences:
        logger.info(f'Databases {db_infos[0]} and {db_infos[1]} are different:')
        for line in differences:
            logger.info(line)
        sys.exit(1)
    else:
        logger.info(f'Database {db_infos[0]} successfully migrated to {db_infos[1]}')

    sys.exit(0)


if __name__ == '__main__':
    main()
