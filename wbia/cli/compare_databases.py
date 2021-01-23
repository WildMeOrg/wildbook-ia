# -*- coding: utf-8 -*-
import logging
import sys
import click

from wbia.dtool.copy_sqlite_to_postgres import (
    SqliteDatabaseInfo,
    PostgresDatabaseInfo,
    compare_databases,
    DEFAULT_CHECK_PC,
    DEFAULT_CHECK_MIN,
    DEFAULT_CHECK_MAX,
)


logger = logging.getLogger('wbia')


@click.command()
@click.option(
    '--db-dir',
    multiple=True,
    help='SQLite databases location',
)
@click.option(
    '--sqlite-uri',
    multiple=True,
    help='SQLite database URI (e.g. sqlite:////path.sqlite3)',
)
@click.option(
    '--pg-uri',
    multiple=True,
    help='Postgres connection URI (e.g. postgresql://user:pass@host)',
)
@click.option(
    '--check-pc',
    type=float,
    default=DEFAULT_CHECK_PC,
    help=f'Percentage of table to check, default {DEFAULT_CHECK_PC} ({int(DEFAULT_CHECK_PC * 100)}% of the table)',
)
@click.option(
    '--check-max',
    type=int,
    default=DEFAULT_CHECK_MAX,
    help=f'Maximum number of rows to check, default {DEFAULT_CHECK_MAX} (0 for no limit)',
)
@click.option(
    '--check-min',
    type=int,
    default=DEFAULT_CHECK_MIN,
    help=f'Minimum number of rows to check, default {DEFAULT_CHECK_MIN}',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    default=False,
    help='Show debug messages',
)
def main(db_dir, sqlite_uri, pg_uri, check_pc, check_max, check_min, verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.addHandler(logging.StreamHandler())

    if len(db_dir) + len(sqlite_uri) + len(pg_uri) != 2:
        raise click.BadParameter('exactly 2 db_dir or sqlite_uri or pg_uri must be given')
    db_infos = []
    for db_dir_ in db_dir:
        db_infos.append(SqliteDatabaseInfo(db_dir_))
    for sqlite_uri_ in sqlite_uri:
        db_infos.append(SqliteDatabaseInfo(sqlite_uri_))
    for pg_uri_ in pg_uri:
        db_infos.append(PostgresDatabaseInfo(pg_uri_))
    exact = not (sqlite_uri and pg_uri)
    differences = compare_databases(
        *db_infos,
        exact=exact,
        check_pc=check_pc,
        check_max=check_max,
        check_min=check_min,
    )
    if differences:
        click.echo(f'Databases {db_infos[0]} and {db_infos[1]} are different:')
        for line in differences:
            click.echo(line)
        sys.exit(1)
    else:
        click.echo(f'Databases {db_infos[0]} and {db_infos[1]} are the same')


if __name__ == '__main__':
    main()
