# -*- coding: utf-8 -*-
import os
import sys
import typing
from pathlib import Path

import click

# XXX (10-Jan-12021) Significant changes were required to copy_sqlite_to_postgres and copy_sqlite_to_postgres.
#     The changes are scoped for now. The implementation here will replace the one found in the following imports.
# from wbia.dtool.sql_control import sqlite_uri_to_postgres_uri_schema
# from wbia.dtool.copy_sqlite_to_postgres import copy_sqlite_to_postgres
import tempfile
from wbia.dtool.copy_sqlite_to_postgres import (
    add_rowids,
    after_pgloader,
    before_pgloader,
    remove_rowids,
    run_pgloader,
)
from wbia.dtool.sql_control import create_engine


MAIN_DB_FILENAME = '_ibeis_database.sqlite3'
STAGING_DB_FILENAME = '_ibeis_staging.sqlite3'
CACHE_DIRECTORY_NAME = '_ibeis_cache'


def get_sqlite_db_paths(db_dir: Path):
    """Generates a sequence of sqlite database file paths.
    The sequence will end with staging and the main database.

    """
    base_loc = (db_dir / '_ibsdb').resolve()
    main_db = base_loc / MAIN_DB_FILENAME
    staging_db = base_loc / STAGING_DB_FILENAME
    cache_directory = base_loc / CACHE_DIRECTORY_NAME

    # churn over the cache databases
    for matcher in [
        cache_directory.glob('**/*.sqlite'),
        cache_directory.glob('**/*.sqlite3'),
    ]:
        for f in matcher:
            if 'backup' in f.name:
                continue
            yield f.resolve()

    if staging_db.exists():
        # doesn't exist in test databases
        yield staging_db
    yield main_db


def sqlite_uri_to_postgres_uri_and_schema(sqlite_uri, base_postgres_uri, db_name=None):
    """
    Converts a sqlite URI to a URI and schema namespace for connecting to Postgres.

    Args:
        sqlite_uri: sqlite uri (e.g. sqlite:///foo/bar/baz.sqlite)
        base_postgres_uri: postgres uri
        db_name: (optional) database name (defaults to the sqlite database name)

    Returns:
        uri, namespace: a connection uri and schema namespace

    """
    from wbia.init.sysres import get_workdir

    workdir = Path(get_workdir()).resolve()
    base_sqlite_uri = f'sqlite:///{workdir}'
    namespace = None

    # Can we discover the database name?
    if not sqlite_uri.startswith(base_sqlite_uri) and db_name is None:
        raise ValueError(
            "Can't determine the database name within "
            f"'{sqlite_uri}' because it's not within the workdir."
        )

    # Remove sqlite:///{workdir} from uri
    # -> /NAUT_test/_ibsdb/_ibeis_cache/chipcache4.sqlite
    sqlite_db_path = sqlite_uri[len(base_sqlite_uri) :]

    # Find the database name and namespace
    # e.g. ['', 'NAUT_test', '_ibsdb', '_ibeis_cache', 'chipcache4.sqlite']
    sqlite_db_path_parts = sqlite_db_path.lower().split(os.path.sep)
    if len(sqlite_db_path_parts) > 2:
        # e.g. naut_test
        if db_name is None:
            db_name = sqlite_db_path_parts[1]
        # e.g. chipcache4
        namespace = os.path.splitext(sqlite_db_path_parts[-1])[0]
        if namespace == '_ibeis_staging':
            namespace = 'staging'
        elif namespace == '_ibeis_database':
            namespace = 'public'
        # e.g. postgresql://wbia@db/naut_test
        uri = f'{base_postgres_uri}/{db_name}'
    else:
        raise ValueError(
            'Unable to determine the database name from the given input: '
            f'sqlite_uri={sqlite_uri} sqlite_db_path={sqlite_db_path} sqlite_db_path_parts={sqlite_db_path_parts}'
        )

    return uri, namespace


def copy_sqlite_to_postgres(
    db_dir: Path, base_postgres_uri: str, db_name: typing.Optional[str] = None
) -> None:
    """Copies all the sqlite databases into a single postgres database

    Args:
        db_dir: the colloquial dbdir (i.e. directory containing '_ibsdb', 'smart_patrol', etc.)
        base_postgres_uri: a postgres connection uri without the database name
        db_name: explicitly name the database (defaults to derived value from ``db_dir``)

    """
    # Done within a temporary directory for writing pgloader configuration files
    with tempfile.TemporaryDirectory() as tempdir:
        for sqlite_db_path in get_sqlite_db_paths(db_dir):
            # XXX logger.info(...)
            click.echo(f'working on {sqlite_db_path} ...')

            sqlite_uri = f'sqlite:///{sqlite_db_path}'
            # create new tables with sqlite built-in rowid column
            sqlite_engine = create_engine(sqlite_uri)

            try:
                add_rowids(sqlite_engine)
                uri, schema = sqlite_uri_to_postgres_uri_and_schema(
                    sqlite_uri, base_postgres_uri, db_name
                )
                engine = create_engine(uri)
                before_pgloader(engine, schema)
                run_pgloader(sqlite_db_path, uri, tempdir)
                after_pgloader(engine, schema)
            finally:
                remove_rowids(sqlite_engine)


@click.command()
@click.option(
    '--db-dir', required=True, type=click.Path(exists=True), help='database location'
)
@click.option(
    '--postgres-base-uri', required=True, help='URI (e.g. postgres://user:pass@host)'
)
@click.option(
    '--new-database-name',
    help='different database name (default: derived from --db-dir value)',
)
def main(db_dir, postgres_base_uri, new_database_name):
    """"""
    click.echo(f'using {db_dir} ...')

    # TODO Check that the database hasn't already been migrated.

    # TODO Create the database if it doesn't exist

    # Migrate
    copy_sqlite_to_postgres(Path(db_dir), postgres_base_uri, new_database_name)

    # TODO Verify the migration

    sys.exit(0)


if __name__ == '__main__':
    main()
