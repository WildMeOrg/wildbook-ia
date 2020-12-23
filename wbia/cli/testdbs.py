# -*- coding: utf-8 -*-
"""Simple script to set up testdb0 & testdb1 on the filesystem"""
from pathlib import Path
from shutil import rmtree

import click

from wbia.dbio import ingest_database
from wbia.init.sysres import (
    ensure_nauts,
    ensure_pz_mtest,
    ensure_testdb2,
    ensure_wilddogs,
    get_workdir,
)


@click.command()
@click.option('-r', '--force-replace', is_flag=True, help='replace if database exists')
def main(force_replace):
    """Initializes the testdb0 & testdb1 testing directories on the filesystem"""
    workdir = Path(get_workdir())

    dbs = {
        # <name>: <factory>
        'testdb1': lambda: ingest_database.ingest_standard_database('testdb1'),
        'PZ_MTEST': ensure_pz_mtest,
        'NAUT_test': ensure_nauts,
        'wd_peter2': ensure_wilddogs,
        'testdb2': ensure_testdb2,
    }

    for db in dbs:
        loc = (workdir / db).resolve()
        if loc.exists():
            if force_replace:
                rmtree(loc)
            else:
                raise RuntimeError(f'{db} already exists at {loc}, aborting')
        factory = dbs[db]
        factory()
        click.echo(f'{db} created at {loc}')


if __name__ == '__main__':
    main()
