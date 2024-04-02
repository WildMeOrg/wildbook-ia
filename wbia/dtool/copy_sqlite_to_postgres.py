# -*- coding: utf-8 -*-
"""
Copy sqlite database into a postgresql database using pgloader (from
apt-get)
"""
import logging
import re
import shutil
import subprocess
import tempfile
import traceback
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, wraps
from pathlib import Path

import numpy as np
import sqlalchemy

from wbia.dtool.sql_control import create_engine

logger = logging.getLogger('wbia.dtool')


MAIN_DB_FILENAME = '_ibeis_database.sqlite3'
STAGING_DB_FILENAME = '_ibeis_staging.sqlite3'
CACHE_DIRECTORY_NAME = '_ibeis_cache'
DEFAULT_CHECK_PC = 0.1
DEFAULT_CHECK_MAX = 100
DEFAULT_CHECK_MIN = 10


def _modified_date(f: Path) -> str:
    """Get the modified date of the file"""
    return str(f.stat().st_mtime)


def _sqlite_uri_to_path(uri: str) -> Path:
    """Convert a SQLite URI to a pathlib.Path"""
    return Path(uri.split(':')[-1]).resolve()


def _sqlite_path_to_uri(path: Path) -> str:
    """Convert a pathlib.Path to a SQLite database to a URI"""
    return f'sqlite:///{str(path)}'


def get_sqlite_db_paths(db_dir: Path):
    """Generates a sequence of sqlite database file paths and sizes.
    The sequence is sorted by database size, smallest first.

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
        p = f.resolve()
        paths.append((p, p.stat().st_size))

    if staging_db.exists():
        # doesn't exist in test databases
        paths.append((staging_db, staging_db.stat().st_size))
    paths.append((main_db, main_db.stat().st_size))

    # Sort databases by file size, smallest first
    paths.sort(key=lambda a: a[1])
    return paths


def get_schema_name_from_uri(uri: str):
    """Derives the schema name from a sqlite URI (e.g. sqlite:///foo/bar/baz.sqlite)"""
    db_path = _sqlite_uri_to_path(uri)
    name = db_path.stem  # filename without extension

    # special names
    if name == '_ibeis_staging':
        name = 'staging'
    elif name == '_ibeis_database':
        name = 'main'

    return name


class Timer:
    class TimerError(Exception):
        """A custom exception used to report errors in use of Timer class"""

    def __init__(self):
        self._start_time = None
        self._elapsed_time = None

    def start(self):
        """Start a new timer"""
        import time

        if self._start_time is not None:
            raise self.TimerError('Timer is running. Use .stop() to stop it')

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        import time

        if self._start_time is None:
            raise self.TimerError('Timer is not running. Use .start() to start it')

        self._elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

    def report(self):
        return f'Elapsed time: {self._elapsed_time:0.4f} seconds'


# ##############################
#   Information Structs
# ##############################


class SqliteDatabaseInfo:
    def __init__(self, db_dir_or_db_uri):
        self.engines = {}
        self.metadata = {}
        self.db_dir = None
        # This is used to compare the last recorded migration
        # from the sqlite database file (i.e. the modified date)
        # to the record in postgres.
        self.database_modified_dates = {}

        if str(db_dir_or_db_uri).startswith('sqlite://'):
            self.db_uri = db_dir_or_db_uri
            schema = get_schema_name_from_uri(self.db_uri)
            engine = sqlalchemy.create_engine(self.db_uri)
            self.engines[schema] = engine
            self.metadata[schema] = sqlalchemy.MetaData(bind=engine)
            self.database_modified_dates[schema] = _modified_date(
                _sqlite_uri_to_path(self.db_uri)
            )
        else:
            self.db_dir = Path(db_dir_or_db_uri)
            for db_path, _ in get_sqlite_db_paths(self.db_dir):
                db_uri = _sqlite_path_to_uri(db_path)
                schema = get_schema_name_from_uri(db_uri)
                engine = sqlalchemy.create_engine(db_uri)
                self.engines[schema] = engine
                self.metadata[schema] = sqlalchemy.MetaData(bind=engine)
                self.database_modified_dates[schema] = _modified_date(db_path)

    def __str__(self):
        if self.db_dir:
            return f'<SqliteDatabaseInfo db_dir={self.db_dir}>'
        return f'<SqliteDatabaseInfo db_uri={self.db_uri}>'

    def get_schema(self):
        return sorted(self.engines.keys())

    def get_table_names(self, normalized=False):
        for schema in sorted(self.metadata.keys()):
            metadata = self.metadata[schema]
            metadata.reflect()
            for table in sorted(metadata.tables, key=lambda a: a.lower()):
                if normalized:
                    yield schema.lower(), table.lower()
                else:
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
        table = self.metadata[schema].tables[table_name]
        stmt = sqlalchemy.select([sqlalchemy.column('rowid'), table])
        if percentage < 1 or max_rows > 0:
            total_rows = self.get_total_rows(schema, table_name)
            rows_to_check = max(int(total_rows * percentage), min_rows)
            if max_rows > 0:
                rows_to_check = min(rows_to_check, max_rows)
            stmt = stmt.order_by(sqlalchemy.text('random()')).limit(rows_to_check)
        return self.engines[schema].execute(stmt)

    def get_rows(self, schema, table_name, rowids):
        table = self.metadata[schema].tables[table_name]
        rowid = sqlalchemy.column('rowid')
        stmt = sqlalchemy.select([rowid, table])
        if rowids:
            stmt = stmt.where(rowid.in_(rowids))
        result = self.engines[schema].execute(stmt)
        return result.fetchall()

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
        # Load the metadata
        for schema in self.get_schema():
            self.metadata.reflect(schema=schema)
        self.metadata.reflect()
        # create sqlite to postgres migration table
        self.__sqlite_to_postgres_migration_table

    def __str__(self):
        return f'<PostgresDatabaseInfo db_uri={self.db_uri}>'

    def get_schema(self):
        schemas = self.engine.execute(
            'SELECT DISTINCT table_schema FROM information_schema.tables'
        )
        return sorted(
            s[0]
            for s in schemas
            if s[0] not in ('public', 'pg_catalog', 'information_schema')
        )

    def get_table_names(self, normalized=False):
        """Returns table names as tuples of (schema, table)"""
        for schema in self.get_schema():
            self.metadata.reflect(schema=schema)
        for table in sorted(self.metadata.tables, key=lambda a: a.lower()):
            if '.' not in table:
                # This is a table in the 'public' schema and should be ignored.
                continue
            if normalized:
                yield tuple(table.lower().split('.', 1))
            else:
                yield tuple(table.split('.', 1))

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
        table = self.metadata.tables[f'{schema}.{table_name}']
        stmt = sqlalchemy.select([table])
        if percentage < 1 or max_rows > 0:
            total_rows = self.get_total_rows(schema, table_name)
            rows_to_check = max(int(total_rows * percentage), min_rows)
            if max_rows > 0:
                rows_to_check = min(rows_to_check, max_rows)
            stmt = stmt.order_by(sqlalchemy.text('random()')).limit(rows_to_check)
        return self.engine.execute(stmt)

    def get_rows(self, schema, table_name, rowids):
        table = self.metadata.tables[f'{schema}.{table_name}']
        stmt = sqlalchemy.select([table])
        if rowids:
            stmt = stmt.where(table.c['rowid'].in_(rowids))
        result = self.engine.execute(stmt)
        return result.fetchall()

    def get_column(self, schema, table_name, column_name='rowid'):
        table = self.metadata.tables[f'{schema}.{table_name}']
        column = table.c[column_name]
        stmt = sqlalchemy.select(column).order_by(column)
        return self.engine.execute(stmt)

    @property
    def __sqlite_to_postgres_migration_table(self):
        """schema name to sqlite database file modified date mapping in SQL"""
        table_name = 'sqlite_to_postgres_migration'
        if table_name not in self.metadata.tables:
            table = sqlalchemy.schema.Table(
                table_name,
                self.metadata,
                sqlalchemy.Column('name', sqlalchemy.Text, primary_key=True),
                sqlalchemy.Column('modified_date', sqlalchemy.Text),
            )
            try:
                table.create(checkfirst=True)
            except (sqlalchemy.exc.IntegrityError, sqlalchemy.exc.ProgrammingError):
                # table created between check and creation
                table = self.metadata.tables[table_name]
        else:
            table = self.metadata.tables[table_name]
        return table

    def get_sqlite_db_modified_date(self, name: str) -> str:
        """schema name to a sqlite database file modified_date mapping"""
        # This is used to compare the last recorded migration
        # from the sqlite database file (i.e. the modified date)
        # to the record in postgres.

        # Query the table for values
        t = self.__sqlite_to_postgres_migration_table
        query = sqlalchemy.select(t.c.modified_date).where(t.c.name == name)
        try:
            mod_date = self.engine.execute(query).fetchone()[0]
        except TypeError:  # NoneType
            mod_date = None
        return mod_date

    def stamp_migration(self, name: str, modified_date: str):
        """Stamp the database as having migrated ``name``
        with sqlite database file ``modified_date``

        """
        t = self.__sqlite_to_postgres_migration_table
        from sqlalchemy.dialects.postgresql import insert

        stmt = insert(t).values(name=name, modified_date=modified_date)
        stmt = stmt.on_conflict_do_update(
            index_elements=['name'],
            set_=dict(modified_date=stmt.excluded.modified_date),
        )
        self.engine.execute(stmt)


# ##############################
#   Comparison
# ##############################


def rows_equal(row1, row2):
    """Check the rows' values for equality"""
    for e1, e2 in zip(row1, row2):
        if not _complex_type_equality_check(e1, e2):
            return False
    return True


def _complex_type_equality_check(v1, v2):
    """Returns True on the equality of ``v1`` and ``v2``; otherwise False"""
    if type(v1) is not type(v2):
        if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            # We have some float in integer fields in sqlite, for example in
            # annotmatch, the annotmatch_posixtime_modified field has values
            # like 1607396181.67946
            return int(v1) == int(v2)
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
        messages.append(f'differences A->B: {schema1.difference(schema2)}')
        messages.append(f'differences B->A: {schema2.difference(schema1)}')
        return messages
    if not exact and not schema1:
        messages.append(f'Schema in {db_info1} is empty')
        return messages
    if not exact and schema1.difference(schema2):
        messages.append(f'Schema difference: {schema1} not in {schema2}')
        messages.append(f'differences A->B: {schema1.difference(schema2)}')
        messages.append(f'differences B->A: {schema2.difference(schema1)}')
        return messages

    # Compare tables
    tables1 = list(db_info1.get_table_names(normalized=True))
    tables2 = list(db_info2.get_table_names(normalized=True))
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
    tables1 = list(db_info1.get_table_names())
    tables2 = list(db_info2.get_table_names())
    normalized_schema1 = [s.lower() for s in schema1]
    if not exact:
        tables2 = [
            (schema, table) for schema, table in tables2 if schema in normalized_schema1
        ]
    table_total = {}
    for (schema1, table1), (schema2, table2) in zip(tables1, tables2):
        total1 = db_info1.get_total_rows(schema1, table1)
        table_total[f'{schema1}.{table1}'] = total1
        total2 = db_info2.get_total_rows(schema2, table2)
        if total1 != total2:
            messages.append(
                f'Total number of rows in "{schema2}.{table2}" difference: {total1} != {total2}'
            )

    # Compare rowid
    for (schema1, table1), (schema2, table2) in zip(tables1, tables2):
        rowid1 = list(db_info1.get_column(schema1, table1))
        rowid2 = list(db_info2.get_column(schema2, table2))
        if rowid1 != rowid2:
            messages.append(
                f'Row ids in "{schema2}.{table2}" difference: '
                f'Only in {db_info1}={set(rowid1).difference(rowid2)} '
                f'Only in {db_info2}={set(rowid2).difference(rowid1)}'
            )
            return messages

    # Compare data
    for (schema1, table1), (schema2, table2) in zip(tables1, tables2):
        rows1 = list(
            db_info1.get_random_rows(
                schema1,
                table1,
                percentage=check_pc,
                min_rows=check_min,
                max_rows=check_max,
            )
        )
        if table_total[f'{schema1}.{table1}'] == len(rows1):
            rowids = None
        else:
            rowids = [row[0] for row in rows1]
        rows2 = {row[0]: row for row in db_info2.get_rows(schema2, table2, rowids)}
        for row in rows1:
            rowid = row[0]
            row2 = rows2[rowid]
            if not rows_equal(row, row2):
                messages.append(
                    f'Table "{schema2}.{table2}" data difference: {row} != {row2}'
                )
        logger.debug(f'Compared {len(rows1)} rows in {schema2}.{table2}')

    return messages


# ##############################
#   SQLite to Postgres Migration/Copy
# ##############################


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
        connection.execute(f'DROP TABLE {table}')
        connection.execute(f'ALTER TABLE {new_table} RENAME TO {table}')


def before_pgloader(engine, schema):
    connection = engine.connect()
    connection.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
    connection.execute(f"SET SCHEMA '{schema}'")

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


PGLOADER_CONFIG_TEMPLATE = """\
LOAD DATABASE
    FROM {sqlite_uri}
    INTO {postgres_uri}

  WITH create tables,
       create indexes,
       reset no sequences,
       workers = 2, concurrency = 2,
       batch rows = 128,
       prefetch rows = 512

    SET work_mem to '64MB',
        maintenance_work_mem to '512MB',
        search_path to '{schema_name}'

  CAST type uuid to uuid using wbia-uuid-bytes-to-uuid,
       type integer to bigint using wbia-integer-to-string,
       type ndarray to ndarray using byte-vector-to-bytea,
       type numpy to numpy using byte-vector-to-bytea;
"""
# Copied from the built-in sql-server-uniqueidentifier-to-uuid
# transform in pgloader 3.6.2
# Prior to 3.6.2, the transform was for uuid in big-endian order
TRANSFORMS_LISP = """\
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

(defun wbia-integer-to-string (integer-string)
  (declare (type (or null string fixnum float) integer-string))
  (when integer-string
    (princ-to-string
     (typecase integer-string
       (integer integer-string)
       (float   (floor integer-string))
       (string  (let ((with-decimal (position #\\. integer-string)))
                  (handler-case
                    (if with-decimal
                      (parse-integer integer-string :start 0
                                     :end with-decimal)
                      (parse-integer integer-string :start 0))
                    (condition (c)
                      (declare (ignore c))
                        (if with-decimal
                          (parse-integer integer-string :start 1
                                         :end with-decimal)
                          (parse-integer integer-string :start 1
                                         :end (- (length integer-string) 1)))))))))))
"""


def run_pgloader(sqlite_uri: str, postgres_uri: str) -> subprocess.CompletedProcess:
    """Configure and run ``pgloader``.
    If there is a problem this will raise a ``CalledProcessError``
    from ``Process.check_returncode``.

    """
    schema_name = get_schema_name_from_uri(sqlite_uri)

    # Do all this within a self-cleaning temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        td = Path(tempdir)
        pgloader_config = td / f'wbia_{schema_name}.load'
        with pgloader_config.open('w') as fb:
            fb.write(PGLOADER_CONFIG_TEMPLATE.format(**locals()))

        wbia_transforms = td / 'wbia_transforms.lisp'
        with wbia_transforms.open('w') as fb:
            fb.write(TRANSFORMS_LISP)

        try:
            from subprocess import DEVNULL  # Python 3
        except ImportError:
            import os

            DEVNULL = open(os.devnull, 'r+b', 0)

        proc = subprocess.run(
            [
                'pgloader',
                '--verbose',
                '--load-lisp-file',
                str(wbia_transforms),
                str(pgloader_config),
            ],
            stdin=DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        proc.check_returncode()  # raises subprocess.CalledProcessError
        logger.debug(proc.stdout.decode())
        return proc


def after_pgloader(sqlite_engine, pg_engine, schema):
    # Some "NOT NULL" weren't migrated by pgloader for some reason
    connection = pg_engine.connect()
    connection.execute(f"SET SCHEMA '{schema}'")
    sqlite_metadata = sqlalchemy.MetaData(bind=sqlite_engine)
    sqlite_metadata.reflect()
    for table in sqlite_metadata.tables.values():
        for column in table.c:
            if not column.primary_key and not column.nullable:
                connection.execute(
                    f'ALTER TABLE {table.name} ALTER COLUMN {column.name} SET NOT NULL'
                )

    table_pkeys = connection.execute(
        f"""\
        SELECT table_name, column_name
        FROM information_schema.table_constraints
        NATURAL JOIN information_schema.constraint_column_usage
        WHERE table_schema = '{schema}'
        AND constraint_type = 'PRIMARY KEY'"""
    ).fetchall()
    exclude_sequences = set()
    for (table_name, pkey) in table_pkeys:
        # Create sequences for rowid fields
        for column_name in ('rowid', pkey):
            seq_name = f'{table_name}_{column_name}_seq'
            exclude_sequences.add(seq_name)
            connection.execute(f'CREATE SEQUENCE {seq_name}')
            connection.execute(
                f"SELECT setval('{seq_name}', (SELECT max({column_name}) FROM {table_name}))"
            )
            connection.execute(
                f"ALTER TABLE {table_name} ALTER COLUMN {column_name} SET DEFAULT nextval('{seq_name}')"
            )
            connection.execute(
                f'ALTER SEQUENCE {seq_name} OWNED BY {table_name}.{column_name}'
            )

    # Reset all sequences except the ones we just created (doing it here
    # instead of in pgloader because it causes a fatal error in pgloader
    # when pgloader runs in parallel:
    #
    # Asynchronous notification "seqs" (payload: "0") received from
    # server process with PID 28472.)
    sequences = connection.execute(
        f"""\
        SELECT table_name, column_name, column_default
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
        AND column_default LIKE 'nextval%%'"""
    ).fetchall()
    for table_name, column_name, column_default in sequences:
        seq_name = re.sub(r"nextval\('([^']*)'.*", r'\1', column_default)
        if seq_name not in exclude_sequences:
            connection.execute(
                f"SELECT setval('{seq_name}', (SELECT max({column_name}) FROM {table_name}))"
            )


def drop_schema(engine, schema_name):
    connection = engine.connect()
    connection.execute(f'DROP SCHEMA {schema_name} CASCADE')


def _use_copy_of_sqlite_database(f):
    """Makes a copy of the sqlite database given as the first argument in URI form"""

    @wraps(f)
    def wrapper(*args):
        uri = args[0]
        db = _sqlite_uri_to_path(uri)
        with tempfile.TemporaryDirectory() as tempdir:
            temp_db = Path(tempdir) / db.name
            logger.debug(f'Copying {str(db)} to {str(temp_db)} ({db.stat().st_size} B)')
            shutil.copy2(db, temp_db)
            new_uri = _sqlite_path_to_uri(temp_db)
            return f(new_uri, *args[1:])

    return wrapper


@_use_copy_of_sqlite_database
def migrate(sqlite_uri: str, postgres_uri: str):
    logger.info(f'\nworking on {sqlite_uri} ...')
    schema_name = get_schema_name_from_uri(sqlite_uri)
    sl_info = SqliteDatabaseInfo(sqlite_uri)
    pg_info = PostgresDatabaseInfo(postgres_uri)
    sl_engine = create_engine(sqlite_uri)
    pg_engine = create_engine(postgres_uri)
    timer = Timer()

    # Add sqlite built-in rowid column to tables
    logger.debug(f'({schema_name}) adding rowids to database')
    timer.start()
    add_rowids(sl_engine)
    timer.stop()
    logger.debug(f'({schema_name}) added rowids to database ... {timer.report()}')

    logger.debug(f'({schema_name}) running pre-pgloader operations')
    timer.start()
    before_pgloader(pg_engine, schema_name)
    timer.stop()
    logger.debug(f'({schema_name}) ran pre-pgloader operations ... {timer.report()}')

    logger.debug(f'({schema_name}) running pgloader ...')
    timer.start()
    run_pgloader(sqlite_uri, postgres_uri)
    timer.stop()
    logger.debug(f'({schema_name}) ran pgloader ... {timer.report()}')

    logger.debug(f'({schema_name}) running post-pgloader operations')
    timer.start()
    after_pgloader(sl_engine, pg_engine, schema_name)
    timer.stop()
    logger.debug(f'({schema_name}) ran post-pgloader operations ... {timer.report()}')

    # Record in postgres having migrated the sqlite database in its current state
    logger.debug(f'({schema_name}) recorded the migration')
    pg_info.stamp_migration(schema_name, sl_info.database_modified_dates[schema_name])


def _by_modification_date(path: Path, pg_info: PostgresDatabaseInfo) -> bool:
    """Filter out the SQLite database files that have unchanged or not-yet-migrated paths"""
    name = get_schema_name_from_uri(_sqlite_path_to_uri(path))
    current_mod_date = _modified_date(path)
    recorded_mod_date = pg_info.get_sqlite_db_modified_date(name)
    if current_mod_date != recorded_mod_date:
        return True
    else:
        logger.info(f'Skipping already migrated SQLite database at {str(path)}')
        return False


def copy_sqlite_to_postgres(
    db_dir: Path,
    postgres_uri: str,
    num_procs: int = 6,
) -> typing.Generator[typing.Tuple[Path, Exception, int, int], None, None]:
    """Copies all the sqlite databases into a single postgres database

    Args:
        db_dir: the colloquial dbdir (i.e. directory containing '_ibsdb', 'smart_patrol', etc.)
        postgres_uri: a postgres connection uri without the database name
        num_procs: number of concurrent processes to use

    """
    executor = ProcessPoolExecutor(max_workers=num_procs)
    mod_date_filter = partial(
        _by_modification_date, pg_info=PostgresDatabaseInfo(postgres_uri)
    )
    sqlite_dbs = dict(
        filter(lambda x: mod_date_filter(x[0]), dict(get_sqlite_db_paths(db_dir)).items())
    )
    total_size = sum(sqlite_dbs.values())

    if num_procs <= 1:
        # serial migration
        for path in sqlite_dbs:
            try:
                migrate(_sqlite_path_to_uri(path), postgres_uri)
            except Exception as e:
                exc = e
                traceback.print_exc()
            else:
                exc = None
            db_size = sqlite_dbs[path]
            yield (path, exc, db_size, total_size)
    else:
        migration_futures_to_paths = {
            executor.submit(migrate, _sqlite_path_to_uri(p), postgres_uri): p
            for p in sqlite_dbs
        }
        for future in as_completed(migration_futures_to_paths):
            path = migration_futures_to_paths[future]
            db_size = sqlite_dbs[path]
            yield (path, future.exception(), db_size, total_size)
