# -*- coding: utf-8 -*-
import logging
import re
import subprocess
import sys
from pathlib import Path

import click
import sqlalchemy

from wbia.dtool.copy_sqlite_to_postgres import (
    PostgresDatabaseInfo,
    SqliteDatabaseInfo,
    compare_databases,
    copy_sqlite_to_postgres,
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
    '-v',
    '--verbose',
    is_flag=True,
    default=False,
    help='Show debug messages',
)
@click.option(
    '--num-procs',
    type=int,
    default=6,
    help='number of migration processes to concurrently run',
)
def main(db_dir, db_uri, verbose, num_procs):
    """"""
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    logger.info(f'running {num_procs} concurrent processes')
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

    # Migrate
    problems = {}
    with click.progressbar(length=100000, show_eta=True) as bar:
        for path, exc, db_size, total_size in copy_sqlite_to_postgres(
            Path(db_dir),
            db_uri,
            num_procs=num_procs,
        ):
            if exc is not None:
                logger.info(f'\nfailed while processing {str(path)}\n{exc}')
                problems[path] = exc
            else:
                logger.info(f'\nfinished processing {str(path)}')
            bar.update(int(db_size / total_size * bar.length))

    # Report problems
    for path, exc in problems.items():
        logger.info('*' * 60)
        logger.info(f'There was a problem migrating {str(path)}')
        logger.exception(exc)
        if hasattr(exc, '__cause__'):
            # __cause__ is the formated traceback on a multiprocess exception
            logger.info('-' * 30)
            logger.info(exc.__cause__)
        if isinstance(exc, subprocess.CalledProcessError):
            logger.info('-' * 30)
            logger.info(exc.stdout.decode())

    # Verify the migration
    db_infos = [
        SqliteDatabaseInfo(Path(db_dir)),
        PostgresDatabaseInfo(db_uri),
    ]
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
