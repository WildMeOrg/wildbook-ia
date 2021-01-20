# -*- coding: utf-8 -*-
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


@click.command()
@click.option(
    '--db-dir', required=True, type=click.Path(exists=True), help='database location'
)
@click.option(
    '--db-uri',
    required=True,
    help='Postgres connection URI (e.g. postgres://user:pass@host)',
)
def main(db_dir, db_uri):
    """"""
    click.echo(f'using {db_dir} ...')

    # Create the database if it doesn't exist
    engine = sqlalchemy.create_engine(db_uri)
    try:
        engine.connect()
    except sqlalchemy.exc.OperationalError as e:
        m = re.search(r'database "([^"]*)" does not exist', str(e))
        if m:
            dbname = m.group(1)
            engine = sqlalchemy.create_engine(db_uri.rsplit('/', 1)[0])
            click.echo(f'Creating "{dbname}"...')
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
        click.echo('Database already migrated')
        sys.exit(0)

    # Migrate
    copy_sqlite_to_postgres(Path(db_dir), db_uri)

    # Verify the migration
    differences = compare_databases(*db_infos)

    if differences:
        click.echo(f'Databases {db_infos[0]} and {db_infos[1]} are different:')
        for line in differences:
            click.echo(line)
        sys.exit(1)
    else:
        click.echo(f'Database {db_infos[0]} successfully migrated to {db_infos[1]}')

    sys.exit(0)


if __name__ == '__main__':
    main()
