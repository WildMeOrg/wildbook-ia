# -*- coding: utf-8 -*-
import sys
import click

from wbia.dtool.copy_sqlite_to_postgres import (
    SqliteDatabaseInfo,
    PostgresDatabaseInfo,
    compare_databases,
)


@click.command()
@click.option(
    '--db-dir',
    multiple=True,
    help='SQLite database(s) location, can be a directory or sqlite:////path.sqlite3',
)
@click.option(
    '--db-uri',
    multiple=True,
    help='Postgres connection URI (e.g. postgresql://user:pass@host)',
)
def main(db_dir, db_uri):
    if len(db_dir) + len(db_uri) != 2:
        raise click.BadParameter('exactly 2 db_dir or db_uri must be given')
    db_infos = []
    for db_dir_ in db_dir:
        db_infos.append(SqliteDatabaseInfo(db_dir_))
    for db_uri_ in db_uri:
        db_infos.append(PostgresDatabaseInfo(db_uri_))
    differences = compare_databases(*db_infos)
    if differences:
        click.echo(f'Databases {db_infos[0]} and {db_infos[1]} are different:')
        for line in differences:
            click.echo(line)
        sys.exit(1)
    else:
        click.echo(f'Databases {db_infos[0]} and {db_infos[1]} are the same')


if __name__ == '__main__':
    main()
