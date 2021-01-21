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


class SqliteDatabaseInfo:
    def __init__(self, db_dir_or_db_uri):
        self.engines = {}
        self.metadata = {}
        self.db_dir = None
        if str(db_dir_or_db_uri).startswith('sqlite:///'):
            self.db_uri = db_dir_or_db_uri
            schema = get_schema_name_from_uri(self.db_uri)
            engine = sqlalchemy.create_engine(self.db_uri)
            self.engines[schema] = engine
            self.metadata[schema] = sqlalchemy.MetaData(bind=engine)
        else:
            self.db_dir = Path(db_dir_or_db_uri)
            for db_path in get_sqlite_db_paths(self.db_dir):
                db_uri = f'sqlite:///{db_path}'
                schema = get_schema_name_from_uri(db_uri)
                engine = sqlalchemy.create_engine(db_uri)
                self.engines[schema] = engine
                self.metadata[schema] = sqlalchemy.MetaData(bind=engine)

    def __str__(self):
        if self.db_dir:
            return f'<SqliteDatabaseInfo db_dir={self.db_dir}>'
        return f'<SqliteDatabaseInfo db_uri={self.db_uri}>'

    def get_schema(self):
        return sorted(self.engines.keys())

    def get_table_names(self):
        for schema in sorted(self.metadata.keys()):
            metadata = self.metadata[schema]
            metadata.reflect()
            for table in sorted(metadata.tables):
                yield schema, table

    def get_total_rows(self, schema, table_name):
        result = self.engines[schema].execute(f'SELECT count(*) FROM {table_name}')
        return result.fetchone()[0]

    def get_random_rows(self, schema, table_name, max_rows=100):
        table = self.metadata[schema].tables[table_name]
        stmt = (
            sqlalchemy.select([sqlalchemy.column('rowid'), table])
            .order_by(sqlalchemy.text('random()'))
            .limit(max_rows)
        )
        return self.engines[schema].execute(stmt)

    def get_row(self, schema, table_name, rowid_):
        table = self.metadata[schema].tables[table_name]
        rowid = sqlalchemy.column('rowid')
        stmt = sqlalchemy.select([rowid, table]).where(rowid == rowid_)
        result = self.engines[schema].execute(stmt)
        return result.fetchone()


class PostgresDatabaseInfo:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        self.engine = sqlalchemy.create_engine(db_uri)
        self.metadata = sqlalchemy.MetaData(bind=self.engine)

    def __str__(self):
        return f'<PostgresDatabaseInfo db_uri={self.db_uri}>'

    def get_schema(self):
        schemas = self.engine.execute(
            'SELECT DISTINCT table_schema FROM information_schema.tables'
        )
        return sorted(
            [s[0] for s in schemas if s[0] not in ('pg_catalog', 'information_schema')]
        )

    def get_table_names(self):
        for schema in self.get_schema():
            self.metadata.reflect(schema=schema)
        return [tuple(table.split('.', 1)) for table in sorted(self.metadata.tables)]

    def get_total_rows(self, schema, table_name):
        result = self.engine.execute(f'SELECT count(*) FROM {schema}.{table_name}')
        return result.fetchone()[0]

    def get_random_rows(self, schema, table_name, max_rows=100):
        table = self.metadata.tables[f'{schema}.{table_name}']
        stmt = (
            sqlalchemy.select([table])
            .order_by(sqlalchemy.text('random()'))
            .limit(max_rows)
        )
        return self.engine.execute(stmt)

    def get_row(self, schema, table_name, rowid):
        table = self.metadata.tables[f'{schema}.{table_name}']
        stmt = sqlalchemy.select([table]).where(table.c['rowid'] == rowid)
        result = self.engine.execute(stmt)
        return result.fetchone()


def rows_equal(row1, row2):
    for e1, e2 in zip(row1, row2):
        if type(e1) != type(e2):
            return False
        if isinstance(e1, float):
            if abs(e1 - e2) >= 0.0000000001:
                return False
        elif e1 != e2:
            return False
    return True


def compare_databases(db_info1, db_info2, exact=True):
    messages = []

    # Compare schema
    schema1 = set(db_info1.get_schema())
    schema2 = set(db_info2.get_schema())
    if len(schema2) < len(schema1):
        # Make sure db_info1 is the one with the smaller set
        db_info2, db_info1 = db_info1, db_info2
        schema2, schema1 = schema1, schema2

    if exact and schema1 != schema2:
        messages.append(f'Schema difference: {schema1} != {schema2}')
        return messages
    if not exact and not schema1:
        messages.append(f'Schema in {db_info1} is empty')
        return messages
    if not exact and schema1.difference(schema2):
        messages.append(f'Schema difference: {schema1} not in {schema2}')
        return messages

    # Compare tables
    tables1 = list(db_info1.get_table_names())
    tables2 = list(db_info2.get_table_names())
    if not exact:
        tables2 = [(schema, table) for schema, table in tables2 if schema in schema1]
    if tables1 != tables2:
        messages.append(
            'Table names difference: '
            f'Only in {db_info1}={set(tables1).difference(tables2)} '
            f'Only in {db_info2}={set(tables2).difference(tables1)}'
        )
        return messages

    # Compare number of rows
    for schema, table in tables1:
        total1 = db_info1.get_total_rows(schema, table)
        total2 = db_info2.get_total_rows(schema, table)
        if total1 != total2:
            messages.append(
                f'Total number of rows in "{schema}.{table}" difference: {total1} != {total2}'
            )

    # Compare data
    for schema, table in tables1:
        for row in db_info1.get_random_rows(schema, table):
            row2 = db_info2.get_row(schema, table, row[0])
            if not rows_equal(row, row2):
                messages.append(
                    f'Table "{schema}.{table}" data difference: {row} != {row2}'
                )

    return messages


def get_sqlite_db_paths(db_dir: Path):
    """Generates a sequence of sqlite database file paths.
    The sequence will end with staging and the main database.

    """
    base_loc = (db_dir / '_ibsdb').resolve()
    main_db = base_loc / MAIN_DB_FILENAME
    staging_db = base_loc / STAGING_DB_FILENAME
    cache_directory = base_loc / CACHE_DIRECTORY_NAME

    # churn over the cache databases
    for f in cache_directory.glob('*.sqlite'):
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
        name = 'main'

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
        # Change "REAL" type to "DOUBLE" because "REAL" in postgresql
        # only can only store 6 digits and so we'd lose precision
        stmt = re.sub('REAL', 'DOUBLE', stmt)
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


def drop_schema(engine, schema_name):
    connection = engine.connect()
    connection.execute(f'DROP SCHEMA {schema_name} CASCADE')


def copy_sqlite_to_postgres(db_dir: Path, postgres_uri: str) -> None:
    """Copies all the sqlite databases into a single postgres database

    Args:
        db_dir: the colloquial dbdir (i.e. directory containing '_ibsdb', 'smart_patrol', etc.)
        postgres_uri: a postgres connection uri without the database name

    """
    # Done within a temporary directory for writing pgloader configuration files
    pg_info = PostgresDatabaseInfo(postgres_uri)
    pg_schema = pg_info.get_schema()
    with tempfile.TemporaryDirectory() as tempdir:
        for sqlite_db_path in get_sqlite_db_paths(db_dir):
            logger.info(f'working on {sqlite_db_path} ...')

            sqlite_uri = f'sqlite:///{sqlite_db_path}'
            sl_info = SqliteDatabaseInfo(sqlite_uri)
            if not compare_databases(sl_info, pg_info, exact=False):
                logger.info(f'{sl_info} already migrated to {pg_info}')
                continue

            sqlite_engine = create_engine(sqlite_uri)
            schema_name = get_schema_name_from_uri(sqlite_uri)
            remove_rowids(sqlite_engine)
            engine = create_engine(postgres_uri)
            if schema_name in pg_schema:
                logger.warning(f'Dropping schema "{schema_name}"')
                drop_schema(engine, schema_name)
            try:
                # create new tables with sqlite built-in rowid column
                add_rowids(sqlite_engine)
                before_pgloader(engine, schema_name)
                run_pgloader(sqlite_db_path, postgres_uri, tempdir)
                after_pgloader(engine, schema_name)
            finally:
                remove_rowids(sqlite_engine)
