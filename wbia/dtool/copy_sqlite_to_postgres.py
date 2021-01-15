# -*- coding: utf-8 -*-
"""
Copy sqlite database into a postgresql database using pgloader (from
apt-get)
"""
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import sqlalchemy

from wbia.dtool.sql_control import create_engine


logger = logging.getLogger('wbia')


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


def get_schema_name_from_uri(uri: str):
    """Derives the schema name from a sqlite URI (e.g. sqlite:///foo/bar/baz.sqlite)"""
    from wbia.init.sysres import get_workdir

    workdir = Path(get_workdir()).resolve()
    base_sqlite_uri = f'sqlite:///{workdir}'
    db_path = Path(uri[len(base_sqlite_uri) :])
    name = db_path.stem  # filename without extension

    # special names
    if name == '_ibeis_staging':
        name = 'staging'
    elif name == '_ibeis_database':
        name = 'public'

    return name


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

  CAST type uuid to uuid using wbia-uuid-bytes-to-uuid

  INCLUDING ONLY TABLE NAMES LIKE '%_with_rowid';
"""
        )
    wbia_uuid_loader = os.path.join(tempdir, 'wbia_uuid_loader.lisp')
    # Copied from the built-in sql-server-uniqueidentifier-to-uuid
    # transform in pgloader 3.6.2
    # Prior to 3.6.2, the transform was for uuid in big-endian order
    with open(wbia_uuid_loader, 'w') as f:
        f.write(
            """\
(in-package :pgloader.transforms)

(defmacro arr-to-bytes-rev (from to array)
  `(loop for i from ,to downto ,from
    with res = 0
    do (setf (ldb (byte 8 (* 8 (- i,from))) res) (aref ,array i))
    finally (return res)))

(defun wbia-uuid-bytes-to-uuid (id)
  (declare (type (or null (array (unsigned-byte 8) (16))) id))
  (when id
    (let ((uuid
        (make-instance 'uuid:uuid
               :time-low (arr-to-bytes-rev 0 3 id)
               :time-mid (arr-to-bytes-rev 4 5 id)
               :time-high (arr-to-bytes-rev 6 7 id)
               :clock-seq-var (aref id 8)
               :clock-seq-low (aref id 9)
               :node (uuid::arr-to-bytes 10 15 id))))
      (princ-to-string uuid))))
"""
        )
    subprocess.check_output(['pgloader', '--load-lisp-file', wbia_uuid_loader, fname])


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
        if '_with_rowid' in table_name:
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


def copy_sqlite_to_postgres(db_dir: Path, postgres_uri: str) -> None:
    """Copies all the sqlite databases into a single postgres database

    Args:
        db_dir: the colloquial dbdir (i.e. directory containing '_ibsdb', 'smart_patrol', etc.)
        postgres_uri: a postgres connection uri without the database name

    """
    # Done within a temporary directory for writing pgloader configuration files
    with tempfile.TemporaryDirectory() as tempdir:
        for sqlite_db_path in get_sqlite_db_paths(db_dir):
            logger.info(f'working on {sqlite_db_path} ...')

            sqlite_uri = f'sqlite:///{sqlite_db_path}'
            # create new tables with sqlite built-in rowid column
            sqlite_engine = create_engine(sqlite_uri)

            try:
                add_rowids(sqlite_engine)
                schema_name = get_schema_name_from_uri(sqlite_uri)
                engine = create_engine(postgres_uri)
                before_pgloader(engine, schema_name)
                run_pgloader(sqlite_db_path, postgres_uri, tempdir)
                after_pgloader(engine, schema_name)
            finally:
                remove_rowids(sqlite_engine)
