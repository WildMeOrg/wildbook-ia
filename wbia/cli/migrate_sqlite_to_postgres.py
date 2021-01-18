# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import click

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

    # TODO Check that the database hasn't already been migrated.

    # TODO Create the database if it doesn't exist

    # Migrate
    copy_sqlite_to_postgres(Path(db_dir), db_uri)

    # Verify the migration
    db_infos = [
        SqliteDatabaseInfo(Path(db_dir)),
        PostgresDatabaseInfo(db_uri),
    ]
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
