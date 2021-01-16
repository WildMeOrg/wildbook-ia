# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import click

from wbia.dtool.copy_sqlite_to_postgres import copy_sqlite_to_postgres


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

    # TODO Verify the migration

    sys.exit(0)


if __name__ == '__main__':
    main()
