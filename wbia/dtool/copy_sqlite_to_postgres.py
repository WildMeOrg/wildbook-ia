# -*- coding: utf-8 -*-
"""
Copy sqlite database into a postgresql database using pgloader (from
apt-get)
"""
import os
import re
import subprocess
import tempfile

import sqlalchemy
from wbia.dtool.sql_control import (
    create_engine,
    sqlite_uri_to_postgres_uri_schema,
)


def get_sqlite_db_paths(parent_dir):
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        for filename in filenames:
            if (
                filename.endswith('.sqlite') or filename.endswith('.sqlite3')
            ) and 'backup' not in filename:
                yield os.path.join(dirpath, filename)


def add_rowids(engine):
    connection = engine.connect()
    create_table_stmts = connection.execute(
        """\
        SELECT name, sql FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%' AND type = 'table'
        """
    ).fetchall()
    for table, stmt in create_table_stmts:
        # Create a new table with suffix "_with_rowid"
        new_table = f'{table}_with_rowid'
        stmt = re.sub(
            r'CREATE TABLE [^ ]* \(',
            f'CREATE TABLE {new_table} (rowid INTEGER NOT NULL UNIQUE, ',
            stmt,
        )
        connection.execute(stmt)
        connection.execute(f'INSERT INTO {new_table} SELECT rowid, * FROM {table}')


def remove_rowids(engine):
    connection = engine.connect()
    create_table_stmts = connection.execute(
        """\
        SELECT name, sql FROM sqlite_master
        WHERE name LIKE '%_with_rowid'"""
    ).fetchall()
    for table, stmt in create_table_stmts:
        connection.execute(f'DROP TABLE {table}')


def before_pgloader(engine, schema):
    connection = engine.connect()
    for domain, base_type in (
        ('dict', 'json'),
        ('list', 'json'),
        ('ndarray', 'bytea'),
        ('numpy', 'bytea'),
    ):
        try:
            connection.execute(f'CREATE DOMAIN {domain} AS {base_type}')
        except sqlalchemy.exc.ProgrammingError:
            # sqlalchemy.exc.ProgrammingError:
            # (psycopg2.errors.DuplicateObject) type "dict" already
            # exists
            pass


def run_pgloader(sqlite_db_path, postgres_uri, tempdir):
    # create the pgloader source file
    fname = os.path.join(tempdir, 'sqlite.load')
    with open(fname, 'w') as f:
        f.write(
            f"""\
LOAD DATABASE
    FROM '{sqlite_db_path}'
    INTO {postgres_uri}

  WITH include drop,
       create tables,
       create indexes,
       reset sequences

    SET work_mem to '16MB',
        maintenance_work_mem to '512 MB'

  CAST type uuid to uuid using sql-server-uniqueidentifier-to-uuid

  INCLUDING ONLY TABLE NAMES LIKE '%_with_rowid';
"""
        )
    subprocess.check_output(['pgloader', fname])


def after_pgloader(engine, schema):
    connection = engine.connect()
    connection.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
    table_pkeys = connection.execute(
        """\
        SELECT table_name, column_name
        FROM information_schema.table_constraints
        NATURAL JOIN information_schema.constraint_column_usage
        WHERE table_schema = 'public'
        AND constraint_type = 'PRIMARY KEY'"""
    ).fetchall()
    for (table_name, pkey) in table_pkeys:
        new_table_name = table_name.replace('_with_rowid', '')
        # Rename tables from "images_with_rowid" to "images"
        connection.execute(f'ALTER TABLE {table_name} RENAME TO {new_table_name}')
        # Create sequences for rowid fields
        for column_name in ('rowid', pkey):
            seq_name = f'{new_table_name}_{column_name}_seq'
            connection.execute(f'CREATE SEQUENCE {seq_name}')
            connection.execute(
                f"SELECT setval('{seq_name}', (SELECT max({column_name}) FROM {new_table_name}))"
            )
            connection.execute(
                f"ALTER TABLE {new_table_name} ALTER COLUMN {column_name} SET DEFAULT nextval('{seq_name}')"
            )
            connection.execute(
                f'ALTER SEQUENCE {seq_name} OWNED BY {new_table_name}.{column_name}'
            )
        # Set schema / namespace to "_ibeis_database" for example
        connection.execute(f'ALTER TABLE {new_table_name} SET SCHEMA {schema}')


def copy_sqlite_to_postgres(parent_dir):
    with tempfile.TemporaryDirectory() as tempdir:
        for sqlite_db_path in get_sqlite_db_paths(parent_dir):
            # create new tables with sqlite built-in rowid column
            sqlite_engine = create_engine(f'sqlite:///{sqlite_db_path}')

            try:
                add_rowids(sqlite_engine)
                uri, schema = sqlite_uri_to_postgres_uri_schema(
                    f'sqlite:///{os.path.realpath(sqlite_db_path)}'
                )
                engine = create_engine(uri)
                before_pgloader(engine, schema)
                run_pgloader(sqlite_db_path, uri, tempdir)
                after_pgloader(engine, schema)
            finally:
                remove_rowids(sqlite_engine)
