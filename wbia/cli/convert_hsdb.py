# -*- coding: utf-8 -*-
"""Script to convert hotspotter database (HSDB) to a WBIA compatible database"""
import sys

import click

from wbia.dbio.ingest_hsdb import convert_hsdb_to_wbia, is_hsdb, is_succesful_convert


@click.command()
@click.option(
    '--db-dir', required=True, type=click.Path(exists=True), help='database location'
)
def main(db_dir):
    """Convert hotspotter database (HSDB) to a WBIA compatible database"""
    click.echo(f'⏳ working on {db_dir}')
    if is_hsdb(db_dir):
        click.echo('✅ confirmed hotspotter database')
    else:
        click.echo('❌ not a hotspotter database')
        sys.exit(1)
    if is_succesful_convert(db_dir):
        click.echo('✅ already converted hotspotter database')
        sys.exit(0)

    convert_hsdb_to_wbia(
        db_dir,
        ensure=True,
        verbose=True,
    )

    if is_succesful_convert(db_dir):
        click.echo('✅ successfully converted database')
    else:
        click.echo('❌ unsuccessfully converted... further investigation necessary')
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
