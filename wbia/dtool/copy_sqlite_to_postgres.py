# -*- coding: utf-8 -*-
"""
Copy sqlite database into a postgresql database using pgloader (from
apt-get)
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sqlalchemy

from wbia.dtool.sql_control import create_engine


logger = logging.getLogger('wbia')


MAIN_DB_FILENAME = '_ibeis_database.sqlite3'
STAGING_DB_FILENAME = '_ibeis_staging.sqlite3'
CACHE_DIRECTORY_NAME = '_ibeis_cache'
DEFAULT_CHECK_PC = 0.1
DEFAULT_CHECK_MAX = 100
DEFAULT_CHECK_MIN = 10


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

    def get_random_rows(
        self,
        schema,
        table_name,
        percentage=DEFAULT_CHECK_PC,
        max_rows=DEFAULT_CHECK_MAX,
        min_rows=DEFAULT_CHECK_MIN,
    ):
        total_rows = self.get_total_rows(schema, table_name)
        rows_to_check = max(int(total_rows * percentage), min_rows)
        if max_rows > 0:
            rows_to_check = min(rows_to_check, max_rows)
        table = self.metadata[schema].tables[table_name]
        stmt = (
            sqlalchemy.select([sqlalchemy.column('rowid'), table])
            .order_by(sqlalchemy.text('random()'))
            .limit(rows_to_check)
        )
        return self.engines[schema].execute(stmt)

    def get_row(self, schema, table_name, rowid_):
        table = self.metadata[schema].tables[table_name]
        rowid = sqlalchemy.column('rowid')
        stmt = sqlalchemy.select([rowid, table]).where(rowid == rowid_)
        result = self.engines[schema].execute(stmt)
        return result.fetchone()

    def get_column(self, schema, table_name, column_name='rowid'):
        table = self.metadata[schema].tables[table_name]
        if column_name == 'rowid':
            column = sqlalchemy.column('rowid')
        else:
            column = table.c[column_name]
        stmt = sqlalchemy.select(columns=[column], from_obj=table).order_by(column)
        return self.engines[schema].execute(stmt)


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

    def get_random_rows(
        self,
        schema,
        table_name,
        percentage=DEFAULT_CHECK_PC,
        max_rows=DEFAULT_CHECK_MAX,
        min_rows=DEFAULT_CHECK_MIN,
    ):
        total_rows = self.get_total_rows(schema, table_name)
        rows_to_check = max(int(total_rows * percentage), min_rows)
        if max_rows > 0:
            rows_to_check = min(rows_to_check, max_rows)
        table = self.metadata.tables[f'{schema}.{table_name}']
        stmt = (
            sqlalchemy.select([table])
            .order_by(sqlalchemy.text('random()'))
            .limit(rows_to_check)
        )
        return self.engine.execute(stmt)

    def get_row(self, schema, table_name, rowid):
        table = self.metadata.tables[f'{schema}.{table_name}']
        stmt = sqlalchemy.select([table]).where(table.c['rowid'] == rowid)
        result = self.engine.execute(stmt)
        return result.fetchone()

    def get_column(self, schema, table_name, column_name='rowid'):
        table = self.metadata.tables[f'{schema}.{table_name}']
        column = table.c[column_name]
        stmt = sqlalchemy.select(column).order_by(column)
        return self.engine.execute(stmt)


def rows_equal(row1, row2):
    """Check the rows' values for equality"""
    for e1, e2 in zip(row1, row2):
        if not _complex_type_equality_check(e1, e2):
            return False
    return True


def _complex_type_equality_check(v1, v2):
    """Returns True on the equality of ``v1`` and ``v2``; otherwise False"""
    if type(v1) != type(v2):
        return False
    if isinstance(v1, float):
        if abs(v1 - v2) >= 0.0000000001:
            return False
    elif isinstance(v1, np.ndarray):
        if (v1 != v2).any():  # see ndarray.any() for details
            return False
    elif v1 != v2:
        return False
    return True


def compare_databases(
    db_info1,
    db_info2,
    exact=True,
    check_pc=DEFAULT_CHECK_PC,
    check_min=DEFAULT_CHECK_MIN,
    check_max=DEFAULT_CHECK_MAX,
):
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

    # Compare rowid
    for schema, table in tables1:
        rowid1 = list(db_info1.get_column(schema, table))
        rowid2 = list(db_info2.get_column(schema, table))
        if rowid1 != rowid2:
            messages.append(
                f'Row ids in "{schema}.{table}" difference: '
                f'Only in {db_info1}={set(rowid1).difference(rowid2)} '
                f'Only in {db_info2}={set(rowid2).difference(rowid1)}'
            )
            return messages

    # Compare data
    for schema, table in tables1:
        n_rows = 0
        for row in db_info1.get_random_rows(
            schema, table, percentage=check_pc, min_rows=check_min, max_rows=check_max
        ):
            n_rows += 1
            row2 = db_info2.get_row(schema, table, row[0])
            if not rows_equal(row, row2):
                messages.append(
                    f'Table "{schema}.{table}" data difference: {row} != {row2}'
                )
        logger.debug(f'Compared {n_rows} rows in {schema}.{table}')

    return messages


def get_sqlite_db_paths(db_dir: Path):
    """Generates a sequence of sqlite database file paths.
    The sequence will end with staging and the main database.

    """
    base_loc = (db_dir / '_ibsdb').resolve()
    main_db = base_loc / MAIN_DB_FILENAME
    staging_db = base_loc / STAGING_DB_FILENAME
    cache_directory = base_loc / CACHE_DIRECTORY_NAME
    paths = []

    # churn over the cache databases
    for f in cache_directory.glob('*.sqlite'):
        if 'backup' in f.name:
            continue
        paths.append(f.resolve())

    if staging_db.exists():
        # doesn't exist in test databases
        paths.append(staging_db)
    paths.append(main_db)

    # Sort databases by file size, smallest first
    paths.sort(key=lambda a: a.stat().st_size)
    return paths


def get_schema_name_from_uri(uri: str):
    """Derives the schema name from a sqlite URI (e.g. sqlite:///foo/bar/baz.sqlite)"""
    db_path = Path(uri[len('sqlite:///') :])
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

  CAST type uuid to uuid using wbia-uuid-bytes-to-uuid,
       type ndarray to ndarray using byte-vector-to-bytea,
       type numpy to numpy using byte-vector-to-bytea

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
    proc = subprocess.run(
        ['pgloader', '--load-lisp-file', wbia_uuid_loader, fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logger.debug(proc.stdout.decode())
    proc.check_returncode()


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
            temp_db_path = Path(tempdir) / sqlite_db_path.name
            shutil.copy(sqlite_db_path, temp_db_path)

            logger.debug('*' * 60)  # b/c pgloader debug output can be lengthy
            logger.info(f'working on {sqlite_db_path} as {temp_db_path} ...')

            sqlite_uri = f'sqlite:///{temp_db_path}'
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
                run_pgloader(temp_db_path, postgres_uri, tempdir)
                after_pgloader(engine, schema_name)
            finally:
                remove_rowids(sqlite_engine)
