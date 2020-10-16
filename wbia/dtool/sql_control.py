# -*- coding: utf-8 -*-
"""
Interface into SQL for the IBEIS Controller

TODO; need to use some sort of sticky bit so
sql files are created with reasonable permissions.
"""
import logging
import collections
import os
import parse
import re
import threading
import uuid
from collections.abc import Mapping, MutableMapping
from os.path import join, exists

import six
import sqlalchemy
import utool as ut
from deprecated import deprecated
from sqlalchemy.sql import text, ClauseElement

from wbia.dtool import lite
from wbia.dtool.dump import dumps


print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger('wbia')


READ_ONLY = ut.get_argflag(('--readonly-mode', '--read-only', '--readonly'))
VERBOSE_SQL = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql', '--verbsql'))
NOT_QUIET = not (ut.QUIET or ut.get_argflag('--quiet-sql'))

VERBOSE = ut.VERBOSE
VERYVERBOSE = ut.VERYVERBOSE

TIMEOUT = 600  # Wait for up to 600 seconds for the database to return from a locked state

SQLColumnRichInfo = collections.namedtuple(
    'SQLColumnRichInfo', ('column_id', 'name', 'type_', 'notnull', 'dflt_value', 'pk')
)


# FIXME (31-Jul-12020) Duplicate definition of wbia.constants.METADATA_TABLE
#       Use this definition as the authority because it's within the context of its use.
METADATA_TABLE_NAME = 'metadata'
# Defines the columns used within the metadata table.
METADATA_TABLE_COLUMNS = {
    # Dictionary of metadata column names pair with:
    #  - is_coded_data: bool showing if the value is a data type (True) or string (False)
    # <column-name>: <info-dict>
    'dependson': dict(is_coded_data=True),
    'docstr': dict(is_coded_data=False),
    'relates': dict(is_coded_data=True),
    'shortname': dict(is_coded_data=True),
    'superkeys': dict(is_coded_data=True),
    'extern_tables': dict(is_coded_data=True),
    'dependsmap': dict(is_coded_data=True),
    'primary_superkey': dict(is_coded_data=True),
    'constraint': dict(is_coded_data=False),
}
METADATA_TABLE_COLUMN_NAMES = list(METADATA_TABLE_COLUMNS.keys())


def _unpacker(results):
    """ HELPER: Unpacks results if unpack_scalars is True. """
    if not results:  # Check for None or empty list
        results = None
    else:
        assert len(results) <= 1, 'throwing away results! { %r }' % (results,)
        results = results[0]
    return results


def tuplize(list_):
    """ Converts each scalar item in a list to a dimension-1 tuple """
    tup_list = [item if ut.isiterable(item) else (item,) for item in list_]
    return tup_list


def sanitize_sql(db, tablename_, columns=None):
    """ Sanatizes an sql tablename and column. Use sparingly """
    tablename = re.sub('[^a-zA-Z_0-9]', '', tablename_)
    valid_tables = db.get_table_names()
    if tablename not in valid_tables:
        logger.info('tablename_ = %r' % (tablename_,))
        logger.info('valid_tables = %r' % (valid_tables,))
        raise Exception(
            'UNSAFE TABLE: tablename=%r. '
            'Column names and table names should be different' % tablename
        )
    if columns is None:
        return tablename
    else:

        def _sanitize_sql_helper(column):
            column_ = re.sub('[^a-zA-Z_0-9]', '', column)
            valid_columns = db.get_column_names(tablename)
            if column_ not in valid_columns:
                raise Exception(
                    'UNSAFE COLUMN: must be all lowercase. '
                    'tablename={}, column={}, valid_columns={} column_={}'.format(
                        tablename, column, valid_columns, column_
                    )
                )
                return None
            else:
                return column_

        columns = [_sanitize_sql_helper(column) for column in columns]
        columns = [column for column in columns if columns is not None]

        return tablename, columns


def dev_test_new_schema_version(
    dbname, sqldb_dpath, sqldb_fname, version_current, version_next=None
):
    """
    HACK

    hacky function to ensure that only developer sees the development schema
    and only on test databases
    """
    TESTING_NEW_SQL_VERSION = version_current != version_next
    if TESTING_NEW_SQL_VERSION:
        logger.info('[sql] ATTEMPTING TO TEST NEW SQLDB VERSION')
        devdb_list = [
            'PZ_MTEST',
            'testdb1',
            'testdb2',
            'testdb_dst2',
            'emptydatabase',
        ]
        testing_newschmea = ut.is_developer() and dbname in devdb_list
        # testing_newschmea = False
        # ut.is_developer() and ibs.get_dbname() in ['PZ_MTEST', 'testdb1']
        if testing_newschmea:
            # Set to true until the schema module is good then continue tests
            # with this set to false
            testing_force_fresh = True or ut.get_argflag('--force-fresh')
            # Work on a fresh schema copy when developing
            dev_sqldb_fname = ut.augpath(sqldb_fname, '_develop_schema')
            sqldb_fpath = join(sqldb_dpath, sqldb_fname)
            dev_sqldb_fpath = join(sqldb_dpath, dev_sqldb_fname)
            ut.copy(sqldb_fpath, dev_sqldb_fpath, overwrite=testing_force_fresh)
            # Set testing schema version
            # ibs.db_version_expected = '1.3.6'
            logger.info('[sql] TESTING NEW SQLDB VERSION: %r' % (version_next,))
            # logger.info('[sql] ... pass --force-fresh to reload any changes')
            return version_next, dev_sqldb_fname
        else:
            logger.info('[ibs] NOT TESTING')
    return version_current, sqldb_fname


@six.add_metaclass(ut.ReloadingMetaclass)
class SQLDatabaseController(object):
    """
    Interface to an SQL database
    """

    class Metadata(Mapping):
        """Metadata is an attribute of the ``SQLDatabaseController`` that
        facilitates easy usages by internal and exteral users.
        Each metadata attributes represents a table (i.e. an instance of ``TableMetadata``).
        Each ``TableMetadata`` instance has metadata names as attributes.
        The ``TableMetadata`` can also be adapated to a dictionary for compatability.

        The the ``database`` attribute is a special case that results
        in a ``DatabaseMetadata`` instance rather than ``TableMetadata``.
        This primarily give access to the version and initial UUID,
        respectively as ``database.version`` and ``database.init_uuid``.

        Args:
            ctrlr (SQLDatabaseController): parent controller object

        """

        class DatabaseMetadata(MutableMapping):
            """Special metadata for database information"""

            __fields = (
                'version',
                'init_uuid',
            )

            def __init__(self, ctrlr):
                self.ctrlr = ctrlr

            @property
            def version(self):
                stmt = text(
                    f'SELECT metadata_value FROM {METADATA_TABLE_NAME} WHERE metadata_key = :key'
                )
                try:
                    return self.ctrlr.executeone(stmt, {'key': 'database_version'})[0]
                except TypeError:  # NoneType
                    return None

            @version.setter
            def version(self, value):
                if not value:
                    raise ValueError(value)
                stmt = text(
                    f'INSERT OR REPLACE INTO {METADATA_TABLE_NAME} (metadata_key, metadata_value)'
                    'VALUES (:key, :value)'
                )
                params = {'key': 'database_version', 'value': value}
                self.ctrlr.executeone(stmt, params)

            @property
            def init_uuid(self):
                stmt = text(
                    f'SELECT metadata_value FROM {METADATA_TABLE_NAME} WHERE metadata_key = :key'
                )
                try:
                    value = self.ctrlr.executeone(stmt, {'key': 'database_init_uuid'})[0]
                except TypeError:  # NoneType
                    return None
                if value is not None:
                    value = uuid.UUID(value)
                return value

            @init_uuid.setter
            def init_uuid(self, value):
                if not value:
                    raise ValueError(value)
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                stmt = text(
                    f'INSERT OR REPLACE INTO {METADATA_TABLE_NAME} (metadata_key, metadata_value) '
                    'VALUES (:key, :value)'
                )
                params = {'key': 'database_init_uuid', 'value': value}
                self.ctrlr.executeone(stmt, params)

            # collections.abc.MutableMapping abstract methods

            def __getitem__(self, key):
                try:
                    return getattr(self, key)
                except AttributeError as exc:
                    raise KeyError(*exc.args)

            def __setitem__(self, key, value):
                if key not in self.__fields:
                    raise AttributeError(key)
                setattr(self, key, value)

            def __delitem__(self, key):
                raise RuntimeError(f"'{key}' cannot be deleted")

            def __iter__(self):
                for name in self.__fields:
                    yield name

            def __len__(self):
                return len(self.__fields)

        class TableMetadata(MutableMapping):
            """Metadata on a particular SQL table"""

            def __init__(self, ctrlr, table_name):
                super().__setattr__('ctrlr', ctrlr)
                super().__setattr__('table_name', table_name)

            def _get_key_name(self, name):
                """Because keys are `<table-name>_<name>`"""
                return '_'.join([self.table_name, name])

            def update(self, **kwargs):
                """Update or insert the value into the metadata table with the given keyword arguments of metadata field names"""
                for keyword, value in kwargs.items():
                    if keyword not in METADATA_TABLE_COLUMN_NAMES:
                        # ignore unknown keywords
                        continue
                    setattr(self, keyword, value)

            def __getattr__(self, name):
                # Query the database for the value represented as name
                key = '_'.join([self.table_name, name])
                statement = text(
                    'SELECT metadata_value '
                    f'FROM {METADATA_TABLE_NAME} '
                    'WHERE metadata_key = :key'
                )
                try:
                    value = self.ctrlr.executeone(statement, {'key': key})[0]
                except TypeError:  # NoneType
                    return None
                if METADATA_TABLE_COLUMNS[name]['is_coded_data']:
                    value = eval(value)
                return value

            def __getattribute__(self, name):
                return super().__getattribute__(name)

            def __setattr__(self, name, value):
                try:
                    info = METADATA_TABLE_COLUMNS[name]
                except KeyError:
                    # This prevents setting of any attributes outside of the known names
                    raise AttributeError

                # Delete the record if given None
                if value is None:
                    return self.__delattr__(name)

                if info['is_coded_data']:
                    # Treat the data as code.
                    value = repr(value)
                key = self._get_key_name(name)

                # Insert or update the record
                # FIXME postgresql (4-Aug-12020) 'insert or replace' is not valid for postgresql
                statement = text(
                    f'INSERT OR REPLACE INTO {METADATA_TABLE_NAME} '
                    f'(metadata_key, metadata_value) VALUES (:key, :value)'
                )
                params = {
                    'key': key,
                    'value': value,
                }
                self.ctrlr.executeone(statement, params)

            def __delattr__(self, name):
                if name not in METADATA_TABLE_COLUMN_NAMES:
                    # This prevents deleting of any attributes outside of the known names
                    raise AttributeError

                # Insert or update the record
                statement = text(
                    f'DELETE FROM {METADATA_TABLE_NAME} where metadata_key = :key'
                )
                params = {'key': self._get_key_name(name)}
                self.ctrlr.executeone(statement, params)

            def __dir__(self):
                return METADATA_TABLE_COLUMN_NAMES

            # collections.abc.MutableMapping abstract methods

            def __getitem__(self, key):
                try:
                    return self.__getattr__(key)
                except AttributeError as exc:
                    raise KeyError(*exc.args)

            def __setitem__(self, key, value):
                try:
                    setattr(self, key, value)
                except AttributeError as exc:
                    raise KeyError(*exc.args)

            def __delitem__(self, key):
                try:
                    setattr(self, key, None)
                except AttributeError as exc:
                    raise KeyError(*exc.args)

            def __iter__(self):
                for name in METADATA_TABLE_COLUMN_NAMES:
                    yield name

            def __len__(self):
                return len(METADATA_TABLE_COLUMN_NAMES)

        def __init__(self, ctrlr):
            super().__setattr__('ctrlr', ctrlr)

        def __getattr__(self, name):
            # If the table exists pass back a ``TableMetadata`` instance
            if name == 'database':
                value = self.DatabaseMetadata(self.ctrlr)
            else:
                if name not in self.ctrlr.get_table_names():
                    raise AttributeError(f'not a valid tablename: {name}')
                value = self.TableMetadata(self.ctrlr, name)
            return value

        def __getattribute__(self, name):
            return super().__getattribute__(name)

        def __setattr__(self, name, value):
            # This is inaccessible since any changes
            # to a TableMetadata instance would make on-demand mutations.
            raise NotImplementedError

        def __delattr__(self, name):
            # no-op
            pass

        def __dir__(self):
            # List all available tables, plus 'database'
            raise NotImplementedError

        # collections.abc.Mapping abstract methods

        def __getitem__(self, key):
            try:
                return self.__getattr__(key)
            except AttributeError as exc:
                raise KeyError(*exc.args)

        def __iter__(self):
            for name in self.ctrlr.get_table_names():
                yield name
            yield 'database'

        def __len__(self):
            return len(self.ctrlr.get_table_names()) + 1  # for 'database'

    def __init_engine(self):
        """Create the SQLAlchemy Engine"""
        self._engine = sqlalchemy.create_engine(
            self.uri,
            # The echo flag is a shortcut to set up SQLAlchemy logging
            echo=False,
        )

    @classmethod
    def from_uri(cls, uri, readonly=READ_ONLY, timeout=TIMEOUT):
        """Creates a controller instance from a connection URI

        Args:
            uri (str): connection string or uri
            timeout (int): connection timeout in seconds

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> sqldb_dpath = ut.ensure_app_resource_dir('dtool')
            >>> sqldb_fname = u'test_database.sqlite3'
            >>> path = os.path.join(sqldb_dpath, sqldb_fname)
            >>> db_uri = 'sqlite:///{}'.format(os.path.realpath(path))
            >>> db = SQLDatabaseController.from_uri(db_uri)
            >>> db.print_schema()
            >>> print(db)
            >>> db2 = SQLDatabaseController.from_uri(db_uri, readonly=True)
            >>> db.add_table('temptable', (
            >>>     ('rowid',               'INTEGER PRIMARY KEY'),
            >>>     ('key',                 'TEXT'),
            >>>     ('val',                 'TEXT'),
            >>> ),
            >>>     superkeys=[('key',)])
            >>> db2.print_schema()
        """
        self = cls.__new__(cls)
        self.uri = uri
        self.timeout = timeout
        self.metadata = self.Metadata(self)
        self.readonly = readonly

        self.__init_engine()
        # Create a _private_ SQLAlchemy metadata instance
        # TODO (27-Sept-12020) Develop API to expose elements of SQLAlchemy.
        #      The MetaData is unbound to ensure we don't accidentally misuse it.
        self._sa_metadata = sqlalchemy.MetaData()

        # Reflect all known tables
        self._sa_metadata.reflect(bind=self._engine)

        self._tablenames = None
        # FIXME (31-Jul-12020) rename to private attribute
        self.thread_connections = {}
        self._connection = None

        if not self.readonly:
            # Ensure the metadata table is initialized.
            self._ensure_metadata_table()

        # TODO (31-Jul-12020) Move to Operations code.
        #      Optimization is going to depends on the operational deployment of this codebase.
        # Optimize the database
        self.optimize()

        return self

    def connect(self):
        """Create a connection for the instance or use the existing connection"""
        self._connection = self._engine.connect()
        return self._connection

    @property
    def connection(self):
        """Create a connection or reuse the existing connection"""
        # TODO (31-Jul-12020) Grab the correct connection for the thread.
        if self._connection is not None:
            conn = self._connection
        else:
            conn = self.connect()
        return conn

    def _create_connection(self):
        path = self.uri.replace('sqlite://', '')
        if not exists(path):
            logger.info('[sql] Initializing new database: %r' % (self.uri,))
            if self.readonly:
                raise AssertionError('Cannot open a new database in readonly mode')
        # Open the SQL database connection with support for custom types
        # lite.enable_callback_tracebacks(True)
        # self.fpath = ':memory:'

        # References:
        # http://stackoverflow.com/questions/10205744/opening-sqlite3-database-from-python-in-read-only-mode
        uri = self.uri
        if self.readonly:
            uri += '?mode=ro'
        engine = sqlalchemy.create_engine(
            uri,
            echo=False,
        )
        connection = engine.connect()

        # Keep track of what thead this was started in
        threadid = threading.current_thread()
        self.thread_connections[threadid] = connection

        return connection, uri

    def close(self):
        self.connection.close()
        self.thread_connections = {}

    # def reconnect(db):
    #     # Call this if we move into a new thread
    #     assert db.fname != ':memory:', 'cant reconnect to mem'
    #     connection, uri = db._create_connection()
    #     db.connection = connection
    #     db.cur = db.connection.cursor()

    def thread_connection(self):
        threadid = threading.current_thread()
        if threadid in self.thread_connections:
            connection = self.thread_connections[threadid]
        else:
            connection, uri = self._create_connection()
        return connection

    @profile
    def _ensure_metadata_table(self):
        """
        Creates the metadata table if it does not exist

        We need this to be done every time so that the update code works
        correctly.
        """
        try:
            orig_table_kw = self.get_table_autogen_dict(METADATA_TABLE_NAME)
        except (sqlalchemy.exc.OperationalError, NameError):
            orig_table_kw = None

        meta_table_kw = ut.odict(
            [
                ('tablename', METADATA_TABLE_NAME),
                (
                    'coldef_list',
                    [
                        ('metadata_rowid', 'INTEGER PRIMARY KEY'),
                        ('metadata_key', 'TEXT'),
                        ('metadata_value', 'TEXT'),
                    ],
                ),
                (
                    'docstr',
                    """
                  The table that stores permanently all of the metadata about the
                  database (tables, etc)""",
                ),
                ('superkeys', [('metadata_key',)]),
                ('dependson', None),
            ]
        )
        if meta_table_kw != orig_table_kw:
            # Don't execute a write operation if we don't have to
            self.add_table(**meta_table_kw)
        # METADATA_TABLE_NAME,
        #    superkeys=[('metadata_key',)],
        # IMPORTANT: Yes, we want this line to be tabbed over for the
        # schema auto-generation
        # Ensure that a version number exists
        self.get_db_version(ensure=True)
        # Ensure that an init UUID exists
        self.get_db_init_uuid(ensure=True)

    def get_db_version(self, ensure=True):
        version = self.metadata.database.version
        if version is None and ensure:
            BASE_DATABASE_VERSION = '0.0.0'
            version = BASE_DATABASE_VERSION
            colnames = ['metadata_key', 'metadata_value']
            params_iter = zip(['database_version'], [version])
            # We don't care to find any, because we know there is no version

            def get_rowid_from_superkey(x):
                return [None] * len(x)

            self.add_cleanly(
                METADATA_TABLE_NAME, colnames, params_iter, get_rowid_from_superkey
            )
        return version

    def get_db_init_uuid(self, ensure=True):
        """
        Get the database initialization (creation) UUID

        CommandLine:
            python -m dtool.sql_control get_db_init_uuid

        Example:
            >>> # ENABLE_DOCTEST
            >>> import uuid
            >>> import os
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> # Check random database gets new UUID on init
            >>> db = SQLDatabaseController.from_uri('sqlite:///')
            >>> uuid_ = db.get_db_init_uuid()
            >>> print('New Database: %r is valid' % (uuid_, ))
            >>> assert isinstance(uuid_, uuid.UUID)
            >>> # Check existing database keeps UUID
            >>> sqldb_dpath = ut.ensure_app_resource_dir('dtool')
            >>> sqldb_fname = u'test_database.sqlite3'
            >>> path = os.path.join(sqldb_dpath, sqldb_fname)
            >>> db_uri = 'sqlite:///{}'.format(os.path.realpath(path))
            >>> db1 = SQLDatabaseController.from_uri(db_uri)
            >>> uuid_1 = db1.get_db_init_uuid()
            >>> db2 = SQLDatabaseController.from_uri(db_uri)
            >>> uuid_2 = db2.get_db_init_uuid()
            >>> print('Existing Database: %r == %r' % (uuid_1, uuid_2, ))
            >>> assert uuid_1 == uuid_2
        """
        db_init_uuid = self.metadata.database.init_uuid
        if db_init_uuid is None and ensure:
            db_init_uuid = uuid.uuid4()
            self.metadata.database.init_uuid = db_init_uuid
        return db_init_uuid

    def reboot(self):
        logger.info('[sql] reboot')
        self.connection.close()
        del self.connection
        # Re-initialize the engine
        # ??? May be better to use the `dispose()` method?
        self.__init_engine()
        self.connection = self._engine.connect()

    def backup(self, backup_filepath):
        """
        backup_filepath = dst_fpath
        """
        # Create a brand new conenction to lock out current thread and any others
        connection, uri = self._create_connection()
        # Start Exclusive transaction, lock out all other writers from making database changes
        transaction = connection.begin()
        connection.isolation_level = 'EXCLUSIVE'
        connection.execute('BEGIN EXCLUSIVE')
        # Assert the database file exists, and copy to backup path
        path = self.uri.replace('sqlite://', '')
        if exists(path):
            ut.copy(path, backup_filepath)
        else:
            raise IOError(
                'Could not backup the database as the URI does not exist: %r' % (uri,)
            )
        # Commit the transaction, releasing the lock
        transaction.commit()
        # Close the connection
        connection.close()

    def optimize(self):
        # http://web.utk.edu/~jplyon/sqlite/SQLite_optimization_FAQ.html#pragma-cache_size
        # http://web.utk.edu/~jplyon/sqlite/SQLite_optimization_FAQ.html
        if VERBOSE_SQL:
            logger.info('[sql] running sql pragma optimizions')
        # self.connection.execute('PRAGMA cache_size = 0;')
        # self.connection.execute('PRAGMA cache_size = 1024;')
        # self.connection.execute('PRAGMA page_size = 1024;')
        # logger.info('[sql] running sql pragma optimizions')
        self.connection.execute('PRAGMA cache_size = 10000;')  # Default: 2000
        self.connection.execute('PRAGMA temp_store = MEMORY;')
        self.connection.execute('PRAGMA synchronous = OFF;')
        # self.connection.execute('PRAGMA synchronous = NORMAL;')
        # self.connection.execute('PRAGMA synchronous = FULL;')  # Default
        # self.connection.execute('PRAGMA parser_trace = OFF;')
        # self.connection.execute('PRAGMA busy_timeout = 1;')
        # self.connection.execute('PRAGMA default_cache_size = 0;')

    def shrink_memory(self):
        logger.info('[sql] shrink_memory')
        self.connection.commit()
        self.connection.execute('PRAGMA shrink_memory;')
        self.connection.commit()

    def vacuum(self):
        logger.info('[sql] vaccum')
        transaction = self.connection.begin()
        self.connection.execute('VACUUM;')
        transaction.commit()

    def integrity(self):
        logger.info('[sql] vaccum')
        self.connection.commit()
        self.connection.execute('PRAGMA integrity_check;')
        self.connection.commit()

    def squeeze(self):
        logger.info('[sql] squeeze')
        self.shrink_memory()
        self.vacuum()

    # ==============
    # API INTERFACE
    # ==============

    def get_row_count(self, tblname):
        fmtdict = {
            'tblname': tblname,
        }
        operation_fmt = 'SELECT COUNT(*) FROM {tblname}'
        count = self._executeone_operation_fmt(operation_fmt, fmtdict)[0]
        return count

    def get_all_rowids(self, tblname, **kwargs):
        """ returns a list of all rowids from a table in ascending order """
        operation = text(f'SELECT rowid FROM {tblname} ORDER BY rowid ASC')
        return self.executeone(operation, **kwargs)

    def get_all_col_rows(self, tblname, colname):
        """ returns a list of all rowids from a table in ascending order """
        fmtdict = {
            'colname': colname,
            'tblname': tblname,
        }
        operation_fmt = 'SELECT {colname} FROM {tblname} ORDER BY rowid ASC'
        return self._executeone_operation_fmt(operation_fmt, fmtdict)

    def get_all_rowids_where(self, tblname, where_clause, params, **kwargs):
        """
        returns a list of rowids from a table in ascending order satisfying a
        condition
        """
        fmtdict = {
            'tblname': tblname,
            'where_clause': where_clause,
        }
        operation_fmt = """
        SELECT rowid
        FROM {tblname}
        WHERE {where_clause}
        ORDER BY rowid ASC
        """
        return self._executeone_operation_fmt(operation_fmt, fmtdict, params, **kwargs)

    def check_rowid_exists(self, tablename, rowid_iter, eager=True, **kwargs):
        rowid_list1 = self.get(tablename, ('rowid',), rowid_iter)
        exists_list = [rowid is not None for rowid in rowid_list1]
        return exists_list

    def _add(self, tblname, colnames, params_iter, unpack_scalars=True, **kwargs):
        """ ADDER NOTE: use add_cleanly """
        columns = ', '.join(colnames)
        column_params = ', '.join(f':{col}' for col in colnames)
        parameterized_values = [
            {col: val for col, val in zip(colnames, params)} for params in params_iter
        ]
        stmt = text(f'INSERT INTO {tblname} ({columns}) VALUES ({column_params})')
        return self.executemany(stmt, parameterized_values, unpack_scalars=unpack_scalars)

    def add_cleanly(
        self,
        tblname,
        colnames,
        params_iter,
        get_rowid_from_superkey,
        superkey_paramx=(0,),
        **kwargs,
    ):
        """
        ADDER Extra input:
        the first item of params_iter must be a superkey (like a uuid),

        Does not add None values. Does not add duplicate values.
        For each None input returns None ouptut.
        For each duplicate input returns existing rowid

        Args:
            tblname (str): table name to add into

            colnames (tuple of strs): columns whos values are specified in params_iter

            params_iter (iterable): an iterable of tuples where each tuple corresonds to a row

            get_rowid_from_superkey (func): function that tests if a row needs
                to be added. It should return None for any new rows to be inserted.
                It should return the existing rowid if one exists

            superkey_paramx (tuple of ints): indices of tuples in params_iter which
                correspond to superkeys. defaults to (0,)

        Returns:
            iterable: rowid_list_ -- list of newly added or previously added rowids

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> db = SQLDatabaseController.from_uri('sqlite:///')
            >>> db.add_table('dummy_table', (
            >>>     ('rowid',               'INTEGER PRIMARY KEY'),
            >>>     ('key',                 'TEXT'),
            >>>     ('superkey1',           'TEXT'),
            >>>     ('superkey2',           'TEXT'),
            >>>     ('val',                 'TEXT'),
            >>> ),
            >>>     superkeys=[('key',), ('superkey1', 'superkey2')],
            >>>     docstr='')
            >>> db.print_schema()
            >>> tblname = 'dummy_table'
            >>> colnames = ('key', 'val')
            >>> params_iter = [('spam', 'eggs'), ('foo', 'bar')]
            >>> # Find a useable superkey
            >>> superkey_colnames = db.get_table_superkey_colnames(tblname)
            >>> superkey_paramx = None
            >>> for superkey in superkey_colnames:
            >>>    if all(k in colnames for k in superkey):
            >>>        superkey_paramx = [colnames.index(k) for k in superkey]
            >>>        superkey_colnames = ut.take(colnames, superkey_paramx)
            >>>        break
            >>> def get_rowid_from_superkey(superkeys_list):
            >>>     return db.get_where_eq(tblname, ('rowid',), zip(superkeys_list), superkey_colnames)
            >>> rowid_list_ = db.add_cleanly(
            >>>     tblname, colnames, params_iter, get_rowid_from_superkey, superkey_paramx)
            >>> print(rowid_list_)
        """
        # ADD_CLEANLY_1: PREPROCESS INPUT
        # eagerly evaluate for superkeys
        params_list = list(params_iter)
        # Extract superkeys from the params list (requires eager eval)
        superkey_lists = [
            [None if params is None else params[x] for params in params_list]
            for x in superkey_paramx
        ]
        # ADD_CLEANLY_2: PREFORM INPUT CHECKS
        # check which parameters are valid
        # and not any(ut.flag_None_items(params))
        isvalid_list = [params is not None for params in params_list]
        # Check for duplicate inputs
        isunique_list = ut.flag_unique_items(list(zip(*superkey_lists)))
        # Check to see if this already exists in the database
        # superkey_params_iter = list(zip(*superkey_lists))
        # get_rowid_from_superkey functions take each list separately here
        rowid_list_ = get_rowid_from_superkey(*superkey_lists)
        isnew_list = [rowid is None for rowid in rowid_list_]
        if VERBOSE_SQL and not all(isunique_list):
            logger.info('[WARNING]: duplicate inputs to db.add_cleanly')
        # Flag each item that needs to added to the database
        needsadd_list = list(map(all, zip(isvalid_list, isunique_list, isnew_list)))
        # ADD_CLEANLY_3.1: EXIT IF CLEAN
        if not any(needsadd_list):
            return rowid_list_  # There is nothing to add. Return the rowids
        # ADD_CLEANLY_3.2: PERFORM DIRTY ADDITIONS
        dirty_params = ut.compress(params_list, needsadd_list)
        if ut.VERBOSE:
            logger.info(
                '[sql] adding %r/%r new %s'
                % (len(dirty_params), len(params_list), tblname)
            )
        # Add any unadded parameters to the database
        try:
            self._add(tblname, colnames, dirty_params, **kwargs)
        except Exception as ex:
            nInput = len(params_list)  # NOQA
            ut.printex(
                ex,
                key_list=[
                    'dirty_params',
                    'needsadd_list',
                    'superkey_lists',
                    'nInput',
                    'rowid_list_',
                ],
            )
            raise
        # TODO: We should only have to preform a subset of adds here
        # (at the positions where rowid_list was None in the getter check)
        rowid_list = get_rowid_from_superkey(*superkey_lists)

        # ADD_CLEANLY_4: SANITY CHECK AND RETURN
        assert len(rowid_list) == len(params_list), 'failed sanity check'
        return rowid_list

    def rows_exist(self, tblname, rowids):
        """
        Checks if rowids exist. Yields True if they do
        """
        operation = 'SELECT count(1) FROM {tblname} WHERE rowid=?'.format(tblname=tblname)
        for rowid in rowids:
            yield bool(self.connection.execute(operation, (rowid,)).fetchone()[0])

    def get_where_eq(
        self,
        tblname,
        colnames,
        params_iter,
        where_colnames,
        unpack_scalars=True,
        op='AND',
        **kwargs,
    ):
        """Executes a SQL select where the given parameters match/equal
        the specified where columns.

        Args:
            tblname (str): table name
            colnames (tuple[str]): sequence of column names
            params_iter (list[list]): a sequence of a sequence with parameters,
                                      where each item in the sequence is used in a SQL execution
            where_colnames (list[str]): column names to match for equality against the same index
                                        of the param_iter values
            op (str): SQL boolean operator (e.g. AND, OR)
            unpack_scalars (bool): [deprecated] use to unpack a single result from each query
                                   only use with operations that return a single result for each query
                                   (default: True)

        """
        equal_conditions = [f'{c}=:{c}' for c in where_colnames]
        where_conditions = f' {op} '.upper().join(equal_conditions)
        params = [dict(zip(where_colnames, p)) for p in params_iter]
        return self.get_where(
            tblname,
            colnames,
            params,
            where_conditions,
            unpack_scalars=unpack_scalars,
            **kwargs,
        )

    def get_where_eq_set(
        self,
        tblname,
        colnames,
        params_iter,
        where_colnames,
        unpack_scalars=True,
        eager=True,
        op='AND',
        **kwargs,
    ):
        params_iter_ = list(params_iter)
        params_length = len(params_iter_)

        if params_length > 0:
            args = (
                tblname,
                params_length,
            )
            logger.info('Using sql_control.get_where_eq_set() for %r on %d params' % args)

        if params_length == 0:
            return []

        assert len(where_colnames) == 1
        assert len(params_iter_[0]) == 1
        where_colname = where_colnames[0]
        where_set = list(set(ut.flatten(params_iter_)))

        where_set_str = ['%r' % (where_value,) for where_value in where_set]

        operation_fmt = """
        SELECT {colnames}
        FROM {tblname}
        WHERE {where_colname} IN ( {where_set} )
        """
        fmtdict = {
            'tblname': tblname,
            'colnames': ', '.join(colnames),
            'where_colname': where_colname,
            'where_set': ', '.join(where_set_str),
        }
        return self._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)

    def get_where(
        self,
        tblname,
        colnames,
        params_iter,
        where_clause,
        unpack_scalars=True,
        eager=True,
        **kwargs,
    ):
        """
        Interface to do a SQL select with a where clause

        Args:
            tblname (str): table name
            colnames (tuple[str]): sequence of column names
            params_iter (list[dict]): a sequence of dicts with parameters,
                                      where each item in the sequence is used in a SQL execution
            where_clause (str): conditional statement used in the where clause
            unpack_scalars (bool): [deprecated] use to unpack a single result from each query
                                   only use with operations that return a single result for each query
                                   (default: True)

        """
        if not isinstance(colnames, (tuple, list)):
            raise TypeError('colnames must be a sequence type of strings')

        # Build and execute the query
        columns = ', '.join(colnames)
        stmt = f'SELECT {columns} FROM {tblname}'
        if where_clause is None:
            val_list = self.executeone(text(stmt), **kwargs)
        elif '?' in where_clause:
            raise ValueError(
                "Statements cannot use '?' parameterization, "
                "use ':name' parameters instead."
            )
        else:
            stmt += f' WHERE {where_clause}'
            val_list = self.executemany(
                text(stmt),
                params_iter,
                unpack_scalars=unpack_scalars,
                eager=eager,
                **kwargs,
            )
        return val_list

    def exists_where_eq(
        self,
        tblname,
        params_iter,
        where_colnames,
        op='AND',
        unpack_scalars=True,
        eager=True,
        **kwargs,
    ):
        """ hacked in function for nicer templates """
        andwhere_clauses = [colname + '=?' for colname in where_colnames]
        where_clause = (' %s ' % (op,)).join(andwhere_clauses)
        fmtdict = {
            'tblname': tblname,
            'where_clauses': where_clause,
        }
        operation_fmt = ut.codeblock(
            """
            SELECT EXISTS(
            SELECT 1
            FROM {tblname}
            WHERE {where_clauses}
            LIMIT 1)
            """
        )
        val_list = self._executemany_operation_fmt(
            operation_fmt,
            fmtdict,
            params_iter=params_iter,
            unpack_scalars=unpack_scalars,
            eager=eager,
            **kwargs,
        )
        return val_list

    def get_rowid_from_superkey(
        self, tblname, params_iter=None, superkey_colnames=None, **kwargs
    ):
        """ getter which uses the constrained superkeys instead of rowids """
        where_clause = ' AND '.join([colname + '=?' for colname in superkey_colnames])
        return self.get_where(tblname, ('rowid',), params_iter, where_clause, **kwargs)

    def get(
        self,
        tblname,
        colnames,
        id_iter=None,
        id_colname='rowid',
        eager=True,
        assume_unique=False,
        **kwargs,
    ):
        """Get rows of data by ID

        Args:
            tblname (str): table name to get from
            colnames (tuple of str): column names to grab from
            id_iter (iterable): iterable of search keys
            id_colname (str): column to be used as the search key (default: rowid)
            eager (bool): use eager evaluation
            assume_unique (bool): default False. Experimental feature that could result in a 10x speedup
            unpack_scalars (bool): default True

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> depc.clear_all()
            >>> rowids = depc.get_rowids('notch', [1, 2, 3])
            >>> table = depc['notch']
            >>> db = table.db
            >>> table.print_csv()
            >>> # Break things to test set
            >>> colnames = ('dummy_annot_rowid',)
            >>> got_data = db.get('notch', colnames, id_iter=rowids)
            >>> assert got_data == [1, 2, 3]
        """
        logger.debug(
            '[sql]'
            + ut.get_caller_name(list(range(1, 4)))
            + ' db.get(%r, %r, ...)' % (tblname, colnames)
        )
        if not isinstance(colnames, (tuple, list)):
            raise TypeError('colnames must be a sequence type of strings')

        # ??? Getting a single column of unique values that is matched on rowid?
        #     And sorts the results after the query?
        # ??? This seems oddly specific for a generic method.
        #     Perhaps the logic should be in its own method?
        if (
            assume_unique
            and id_iter is not None
            and id_colname == 'rowid'
            and len(colnames) == 1
        ):
            id_iter = list(id_iter)
            columns = ', '.join(colnames)
            ids_listing = ', '.join(map(str, id_iter))
            operation = f'SELECT {columns} FROM {tblname} WHERE rowid in ({ids_listing}) ORDER BY rowid ASC'
            results = self.connection.execute(operation).fetchall()
            import numpy as np

            # ??? Why order the results if they are going to be sorted here?
            sortx = np.argsort(np.argsort(id_iter))
            results = ut.take(results, sortx)
            if kwargs.get('unpack_scalars', True):
                results = ut.take_column(results, 0)
            return results
        else:
            if id_iter is None:
                where_clause = None
                params_iter = []
            else:
                where_clause = id_colname + ' = :id'
                params_iter = [{'id': id} for id in id_iter]

            return self.get_where(
                tblname, colnames, params_iter, where_clause, eager=eager, **kwargs
            )

    def set(
        self,
        tblname,
        colnames,
        val_iter,
        id_iter,
        id_colname='rowid',
        duplicate_behavior='error',
        duplcate_auto_resolve=True,
        **kwargs,
    ):
        """
        setter

        CommandLine:
            python -m dtool.sql_control set

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> depc.clear_all()
            >>> rowids = depc.get_rowids('notch', [1, 2, 3])
            >>> table = depc['notch']
            >>> db = table.db
            >>> table.print_csv()
            >>> # Break things to test set
            >>> colnames = ('dummy_annot_rowid',)
            >>> val_iter = [(9003,), (9001,), (9002,)]
            >>> orig_data = db.get('notch', colnames, id_iter=rowids)
            >>> db.set('notch', colnames, val_iter, id_iter=rowids)
            >>> new_data = db.get('notch', colnames, id_iter=rowids)
            >>> assert new_data == [x[0] for x in val_iter]
            >>> assert new_data != orig_data
            >>> table.print_csv()
            >>> depc.clear_all()
        """
        if not isinstance(colnames, (tuple, list)):
            raise TypeError('colnames must be a sequence type of strings')

        val_list = list(val_iter)  # eager evaluation
        id_list = list(id_iter)  # eager evaluation

        logger.debug('[sql] SETTER: ' + ut.get_caller_name())
        logger.debug('[sql] * tblname=%r' % (tblname,))
        logger.debug('[sql] * val_list=%r' % (val_list,))
        logger.debug('[sql] * id_list=%r' % (id_list,))
        logger.debug('[sql] * id_colname=%r' % (id_colname,))

        if duplicate_behavior == 'error':
            try:
                has_duplicates = ut.duplicates_exist(id_list)

                if duplcate_auto_resolve:
                    # Check if values being set are equivalent
                    if has_duplicates:

                        debug_dict = ut.debug_duplicate_items(id_list)
                        key_list = list(debug_dict.keys())
                        assert len(key_list) > 0, 'has_duplicates sanity check failed'

                        pop_list = []
                        for key in key_list:
                            index_list = debug_dict[key]
                            assert len(index_list) > 1
                            value_list = ut.take(val_list, index_list)
                            assert all(
                                value == value_list[0] for value in value_list
                            ), 'Passing a non-unique list of ids with different set values'
                            pop_list += index_list[1:]

                        for index in sorted(pop_list, reverse=True):
                            del id_list[index]
                            del val_list[index]
                        logger.debug(
                            '[!set] Auto Resolution: Removed %d duplicate (id, value) pairs from the database operation'
                            % (len(pop_list),)
                        )

                    has_duplicates = ut.duplicates_exist(id_list)

                assert not has_duplicates, 'Passing a not-unique list of ids'
            except Exception as ex:

                ut.printex(
                    ex,
                    'len(id_list) = %r, len(set(id_list)) = %r'
                    % (len(id_list), len(set(id_list))),
                )
                ut.print_traceback()
                raise
        elif duplicate_behavior == 'filter':
            # Keep only the first setting of every row
            isunique_list = ut.flag_unique_items(id_list)
            id_list = ut.compress(id_list, isunique_list)
            val_list = ut.compress(val_list, isunique_list)
        else:
            raise AssertionError(
                (
                    'unknown duplicate_behavior=%r. '
                    'known behaviors are: error and filter'
                )
                % (duplicate_behavior,)
            )

        # Check for incongruity between values and identifiers
        try:
            num_val = len(val_list)
            num_id = len(id_list)
            assert num_val == num_id, 'list inputs have different lengths'
        except AssertionError as ex:
            ut.printex(ex, key_list=['num_val', 'num_id'])
            raise

        # Execute the SQL updates for each set of values
        assignments = ', '.join([f'{col} = :e{i}' for i, col in enumerate(colnames)])
        where_condition = f'{id_colname} = :id'
        stmt = text(f'UPDATE {tblname} SET {assignments} WHERE {where_condition}')
        for i, id in enumerate(id_list):
            params = {'id': id}
            params.update({f'e{e}': p for e, p in enumerate(val_list[i])})
            self.connection.execute(stmt, **params)

    def delete(self, tblname, id_list, id_colname='rowid', **kwargs):
        """Deletes rows from a SQL table (``tblname``) by ID,
        given a sequence of IDs (``id_list``).
        Optionally a different ID column can be specified via ``id_colname``.

        """
        stmt = text(f'DELETE FROM {tblname} WHERE {id_colname} = :id')
        for id in id_list:
            self.connection.execute(stmt, id=id)

    def delete_rowids(self, tblname, rowid_list, **kwargs):
        """ deletes the the rows in rowid_list """
        self.delete(tblname, rowid_list, id_colname='rowid', **kwargs)

    # ==============
    # CORE WRAPPERS
    # ==============

    def _executeone_operation_fmt(
        self, operation_fmt, fmtdict, params=None, eager=True, **kwargs
    ):
        if params is None:
            params = []
        operation = operation_fmt.format(**fmtdict)
        return self.executeone(operation, params, eager=eager, **kwargs)

    @profile
    def _executemany_operation_fmt(
        self,
        operation_fmt,
        fmtdict,
        params_iter,
        unpack_scalars=True,
        eager=True,
        dryrun=False,
        **kwargs,
    ):
        operation = operation_fmt.format(**fmtdict)
        if dryrun:
            logger.info('Dry Run')
            logger.info(operation)
            return
        return self.executemany(
            operation, params_iter, unpack_scalars=unpack_scalars, eager=eager, **kwargs
        )

    # =========
    # SQLDB CORE
    # =========

    def executeone(self, operation, params=(), eager=True, verbose=VERBOSE_SQL):
        """Executes the given ``operation`` once with the given set of ``params``"""
        if not isinstance(operation, ClauseElement):
            raise TypeError(
                "'operation' needs to be a sqlalchemy textual sql instance "
                "see docs on 'sqlalchemy.sql:text' factory function; "
                f"'operation' is a '{type(operation)}'"
            )
        # FIXME (12-Sept-12020) Allows passing through '?' (question mark) parameters.
        results = self.connection.execute(operation, params)

        # BBB (12-Sept-12020) Retaining insertion rowid result
        # FIXME postgresql (12-Sept-12020) This won't work in postgres.
        #       Maybe see if ResultProxy.inserted_primary_key will work
        if 'insert' in operation.text.lower():
            # BBB (12-Sept-12020) Retaining behavior to unwrap single value rows.
            return [results.lastrowid]
        elif not results.returns_rows:
            return None
        else:
            values = list(
                [
                    # BBB (12-Sept-12020) Retaining behavior to unwrap single value rows.
                    row[0] if len(row) == 1 else row
                    for row in results
                ]
            )
            if not values:  # empty list
                values = None
            return values

    def executemany(self, operation, params_iter, unpack_scalars=True, **kwargs):
        """Executes the given ``operation`` once for each item in ``params_iter``

        Args:
            operation (str): SQL operation
            params_iter (sequence): a sequence of sequences
                                    containing parameters in the sql operation
            unpack_scalars (bool): [deprecated] use to unpack a single result from each query
                                   only use with operations that return a single result for each query
                                   (default: True)

        """
        if not isinstance(operation, ClauseElement):
            raise TypeError(
                "'operation' needs to be a sqlalchemy textual sql instance "
                "see docs on 'sqlalchemy.sql:text' factory function; "
                f"'operation' is a '{type(operation)}'"
            )

        results = []
        with self.connection.begin():
            for params in params_iter:
                value = self.executeone(operation, params)
                # Should only be used when the user wants back on value.
                # Let the error bubble up if used wrong.
                # Deprecated... Do not depend on the unpacking behavior.
                if unpack_scalars:
                    value = _unpacker(value)
                results.append(value)
        return results

    def print_dbg_schema(self):
        logger.info(
            '\n\nCREATE'.join(dumps(self.connection, schema_only=True).split('CREATE'))
        )

    # =========
    # SQLDB METADATA
    # =========
    def get_metadata_items(self):
        r"""
        Returns:
            list: metadata_items

        CommandLine:
            python -m dtool.sql_control --exec-get_metadata_items

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> db = testdata_depc()['notch'].db
            >>> metadata_items = db.get_metadata_items()
            >>> result = ('metadata_items = %s' % (ut.repr2(sorted(metadata_items)),))
            >>> print(result)
        """
        metadata_rowids = self.get_all_rowids(METADATA_TABLE_NAME)
        metadata_items = self.get(
            METADATA_TABLE_NAME, ('metadata_key', 'metadata_value'), metadata_rowids
        )
        return metadata_items

    @deprecated('Use the metadata property instead')
    def set_metadata_val(self, key, val):
        """
        key must be given as a repr-ed string
        """
        fmtkw = {
            'tablename': METADATA_TABLE_NAME,
            'columns': 'metadata_key, metadata_value',
        }
        op_fmtstr = 'INSERT OR REPLACE INTO {tablename} ({columns}) VALUES (?, ?)'
        operation = op_fmtstr.format(**fmtkw)
        params = [key, val]
        self.executeone(operation, params, verbose=False)

    @deprecated('Use metadata property instead')
    def get_metadata_val(self, key, eval_=False, default=None):
        """
        val is the repr string unless eval_ is true
        """
        where_clause = 'metadata_key=?'
        colnames = ('metadata_value',)
        params_iter = [(key,)]
        vals = self.get_where(METADATA_TABLE_NAME, colnames, params_iter, where_clause)
        assert len(vals) == 1, 'duplicate keys in metadata table'
        val = vals[0]
        if val is None:
            if default == ut.NoParam:
                assert val is not None, 'metadata_table key=%r does not exist' % (key,)
            else:
                val = default
        # if key.endswith('_constraint') or
        if key.endswith('_docstr'):
            # Hack eval off for constriant and docstr
            return val
        try:
            if eval_ and val is not None:
                # eventually we will not have to worry about
                # mid level representations by default, for now flag it
                val = eval(val, {}, {})
        except Exception as ex:
            ut.printex(ex, keys=['key', 'val'])
            raise
        return val

    # ==============
    # SCHEMA MODIFICATION
    # ==============

    def add_column(self, tablename, colname, coltype):
        if VERBOSE_SQL:
            logger.info(
                '[sql] add column=%r of type=%r to tablename=%r'
                % (colname, coltype, tablename)
            )
        fmtkw = {
            'tablename': tablename,
            'colname': colname,
            'coltype': coltype,
        }
        op_fmtstr = 'ALTER TABLE {tablename} ADD COLUMN {colname} {coltype}'
        operation = op_fmtstr.format(**fmtkw)
        self.executeone(operation, [], verbose=False)

    def __make_unique_constraint(self, column_or_columns):
        """Creates a SQL ``CONSTRAINT`` clause for ``UNIQUE`` column data"""
        if not isinstance(column_or_columns, (list, tuple)):
            columns = [column_or_columns]
        else:
            # Cast as list incase it's a tuple, b/c tuple + list = error
            columns = list(column_or_columns)
        constraint_name = '_'.join(['unique'] + columns)
        columns_listing = ', '.join(columns)
        return f'CONSTRAINT {constraint_name} UNIQUE ({columns_listing})'

    def __make_column_definition(self, name: str, definition: str) -> str:
        """Creates SQL for the given column `name` and type, default & constraint (i.e. `definition`)."""
        if not name:
            raise ValueError(f'name cannot be an empty string paired with {definition}')
        elif not definition:
            raise ValueError(f'definition cannot be an empty string paired with {name}')
        return f'{name} {definition}'

    def _make_add_table_sqlstr(
        self, tablename: str, coldef_list: list, sep=' ', **metadata_keyval
    ):
        """Creates the SQL for a CREATE TABLE statement

        Args:
            tablename (str): table name
            coldef_list (list): list of tuples (name, type definition)
            sep (str): clause separation character(s) (default: space)
            kwargs: metadata specifications

        Returns:
            str: operation

        """
        if not coldef_list:
            raise ValueError(f'empty coldef_list specified for {tablename}')

        # Check for invalid keyword arguments
        bad_kwargs = set(metadata_keyval.keys()) - set(METADATA_TABLE_COLUMN_NAMES)
        if len(bad_kwargs) > 0:
            raise TypeError(f'got unexpected keyword arguments: {bad_kwargs}')

        logger.debug('[sql] schema ensuring tablename=%r' % tablename)
        logger.debug(
            ut.func_str(self.add_table, [tablename, coldef_list], metadata_keyval)
        )

        # Create the main body of the CREATE TABLE statement with column definitions
        # coldef_list = [(<column-name>, <definition>,), ...]
        body_list = [self.__make_column_definition(c, d) for c, d in coldef_list]

        # Make a list of constraints to place on the table
        # superkeys = [(<column-name>, ...), ...]
        constraint_list = [
            self.__make_unique_constraint(x) for x in metadata_keyval.get('superkeys', [])
        ]
        constraint_list = ut.unique_ordered(constraint_list)

        comma = ',' + sep
        table_body = comma.join(body_list + constraint_list)
        return text(f'CREATE TABLE IF NOT EXISTS {tablename} ({sep}{table_body}{sep})')

    def add_table(self, tablename=None, coldef_list=None, **metadata_keyval):
        """
        add_table

        Args:
            tablename (str):
            coldef_list (list):
            constraint (list or None):
            docstr (str):
            superkeys (list or None): list of tuples of column names which
                uniquely identifies a rowid
        """
        operation = self._make_add_table_sqlstr(tablename, coldef_list, **metadata_keyval)
        self.executeone(operation, [], verbose=False)

        self.metadata[tablename].update(**metadata_keyval)
        if self._tablenames is not None:
            self._tablenames.add(tablename)

    def modify_table(
        self,
        tablename=None,
        colmap_list=None,
        tablename_new=None,
        drop_columns=[],
        add_columns=[],
        rename_columns=[],
        # transform_columns=[],
        # constraint=None, docstr=None, superkeys=None,
        **metadata_keyval,
    ):
        """
        function to modify the schema - only columns that are being added,
        removed or changed need to be enumerated

        Args:
           tablename (str): tablename
           colmap_list (list): of tuples (orig_colname, new_colname, new_coltype, convert_func)
               orig_colname - the original name of the column, None to append, int for index
               new_colname - the new column name ('' for same, None to delete)
               new_coltype - New Column Type. None to use data unmodified
               convert_func - Function to convert data from old to new
           constraint (str):
           superkeys (list)
           docstr (str)
           tablename_new (?)

        Example:
            >>> # DISABLE_DOCTEST
            >>> def loc_zip_map(x):
            ...     return x
            >>> db.modify_table(const.CONTRIBUTOR_TABLE, (
            >>>         # orig_colname,             new_colname,      new_coltype, convert_func
            >>>         # a non-needed, but correct mapping (identity function)
            >>>         ('contrib_rowid',      '',                    '',               None),
            >>>         # for new columns, function is ignored (TYPE CANNOT BE EMPTY IF ADDING)
            >>>         (None,                 'contrib_loc_address', 'TEXT',           None),
            >>>         # adding a new column at index 4 (if index is invalid, None is used)
            >>>         (4,                    'contrib_loc_address', 'TEXT',           None),
            >>>         # for deleted columns, type and function are ignored
            >>>         ('contrib_loc_city',    None,                 '',               None),
            >>>         # for renamed columns, type and function are ignored
            >>>         ('contrib_loc_city',   'contrib_loc_town',    '',       None),
            >>>         ('contrib_loc_zip',    'contrib_loc_zip',     'TEXT',   loc_zip_map),
            >>>         # type not changing, only NOT NULL provision
            >>>         ('contrib_loc_country', '',                   'TEXT NOT NULL',  None),
            >>>     ),
            >>>     superkeys=[('contributor_rowid',)],
            >>>     constraint=[],
            >>>     docstr='Used to store the contributors to the project'
            >>> )
        """
        # assert colmap_list is not None, 'must specify colmaplist'
        assert tablename is not None, 'tablename must be given'

        if VERBOSE_SQL or ut.VERBOSE:
            logger.info('[sql] schema modifying tablename=%r' % tablename)
            logger.info(
                '[sql] * colmap_list = ' + 'None'
                if colmap_list is None
                else ut.repr2(colmap_list)
            )

        if colmap_list is None:
            colmap_list = []

        # Augment colmap_list using convience mappings
        for drop_col in drop_columns:
            colmap_list += [(drop_col, None, '', None)]

        for add_col, add_type in add_columns:
            colmap_list += [(None, add_col, add_type, None)]

        for old_col, new_col in rename_columns:
            colmap_list += [(old_col, new_col, None, None)]

        coldef_list = self.get_coldef_list(tablename)
        colname_list = ut.take_column(coldef_list, 0)
        coltype_list = ut.take_column(coldef_list, 1)

        colname_original_list = colname_list[:]
        colname_dict = {colname: colname for colname in colname_list}
        colmap_dict = {}

        insert = False
        for colmap in colmap_list:
            (src, dst, type_, map_) = colmap
            if src is None or isinstance(src, int):
                # Add column
                assert (
                    dst is not None and len(dst) > 0
                ), 'New column name must be valid in colmap=%r' % (colmap,)
                assert (
                    type_ is not None and len(type_) > 0
                ), 'New column type must be specified in colmap=%r' % (colmap,)
                if isinstance(src, int) and (src < 0 or len(colname_list) <= src):
                    src = None
                if src is None:
                    colname_list.append(dst)
                    coltype_list.append(type_)
                else:
                    if insert:
                        logger.info(
                            '[sql] WARNING: multiple index inserted add '
                            'columns, may cause alignment issues'
                        )
                    colname_list.insert(src, dst)
                    coltype_list.insert(src, type_)
                    insert = True
            else:
                # Modify column
                try:
                    assert (
                        src in colname_list
                    ), 'Unkown source colname=%s in tablename=%s' % (src, tablename)
                except AssertionError as ex:
                    ut.printex(ex, keys=['colname_list'])
                index = colname_list.index(src)
                if dst is None:
                    # Drop column
                    assert (
                        src is not None and len(src) > 0
                    ), 'Deleted column name  must be valid'
                    del colname_list[index]
                    del coltype_list[index]
                    del colname_dict[src]
                elif len(src) > 0 and len(dst) > 0 and src != dst:
                    # Rename column
                    colname_list[index] = dst
                    colname_dict[src] = dst
                    # Check if type should change as well
                    if (
                        type_ is not None
                        and len(type_) > 0
                        and type_ != coltype_list[index]
                    ):
                        coltype_list[index] = type_
                elif len(type_) > 0 and type_ != coltype_list[index]:
                    # Change column type
                    if len(dst) == 0:
                        dst = src
                    coltype_list[index] = type_
                elif map_ is not None:
                    # Simply map function across table's data
                    if len(dst) == 0:
                        dst = src
                    if len(type_) == 0:
                        type_ = coltype_list[index]
                else:
                    # Identity, this can be ommited as it is automatically done
                    if len(dst) == 0:
                        dst = src
                    if type_ is None or len(type_) == 0:
                        type_ = coltype_list[index]
            if map_ is not None:
                colmap_dict[src] = map_

        coldef_list = list(zip(colname_list, coltype_list))
        tablename_orig = tablename
        tablename_temp = tablename_orig + '_temp' + ut.random_nonce(length=8)
        metadata_keyval2 = metadata_keyval.copy()
        for suffix in METADATA_TABLE_COLUMN_NAMES:
            if suffix not in metadata_keyval2 or metadata_keyval2[suffix] is None:
                val = getattr(self.metadata[tablename_orig], suffix)
                metadata_keyval2[suffix] = val

        self.add_table(tablename_temp, coldef_list, **metadata_keyval2)

        # Copy data
        src_list = []
        dst_list = []

        for name in colname_original_list:
            if name in colname_dict.keys():
                src_list.append(name)
                dst_list.append(colname_dict[name])

        if len(src_list) > 0:
            data_list_ = self.get(tablename, tuple(src_list))
        else:
            data_list_ = []
        # Run functions across all data for specified callums
        data_list = [
            tuple(
                [
                    colmap_dict[src_](d) if src_ in colmap_dict.keys() else d
                    for d, src_ in zip(data, src_list)
                ]
            )
            for data in data_list_
        ]
        # Add the data to the database

        def get_rowid_from_superkey(x):
            return [None] * len(x)

        self.add_cleanly(tablename_temp, dst_list, data_list, get_rowid_from_superkey)
        if tablename_new is None:
            # Drop original table
            self.drop_table(tablename)
            # Rename temp table to original table name
            self.rename_table(tablename_temp, tablename)
        else:
            # Rename new table to new name
            self.rename_table(tablename_temp, tablename_new)

    def rename_table(self, tablename_old, tablename_new):
        logger.info(
            '[sql] schema renaming tablename=%r -> %r' % (tablename_old, tablename_new)
        )
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        operation = text(f'ALTER TABLE {tablename_old} RENAME TO {tablename_new}')
        self.executeone(operation, [])

        # Rename table's metadata
        key_old_list = [
            tablename_old + '_' + suffix for suffix in METADATA_TABLE_COLUMN_NAMES
        ]
        key_new_list = [
            tablename_new + '_' + suffix for suffix in METADATA_TABLE_COLUMN_NAMES
        ]
        id_iter = [key for key in key_old_list]
        val_iter = [(key,) for key in key_new_list]
        colnames = ('metadata_key',)
        self.set(
            METADATA_TABLE_NAME, colnames, val_iter, id_iter, id_colname='metadata_key'
        )

    def drop_table(self, tablename):
        logger.info('[sql] schema dropping tablename=%r' % tablename)
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        operation = text(f'DROP TABLE IF EXISTS {tablename}')
        self.executeone(operation, [])

        # Delete table's metadata
        key_list = [tablename + '_' + suffix for suffix in METADATA_TABLE_COLUMN_NAMES]
        self.delete(METADATA_TABLE_NAME, key_list, id_colname='metadata_key')

    def drop_all_tables(self):
        """
        DELETES ALL INFO IN TABLE
        """
        self._tablenames = None
        for tablename in self.get_table_names():
            if tablename != 'metadata':
                self.drop_table(tablename)
        self._tablenames = None

    # ==============
    # CONVINENCE
    # ==============

    def dump_tables_to_csv(self, dump_dir=None):
        """ Convenience: Dumps all csv database files to disk """
        if dump_dir is None:
            dump_dir = join(self.dir_, 'CSV_DUMP')
        ut.ensuredir(dump_dir)
        for tablename in self.get_table_names():
            table_fname = tablename + '.csv'
            table_fpath = join(dump_dir, table_fname)
            table_csv = self.get_table_csv(tablename)
            ut.writeto(table_fpath, table_csv)

    def get_schema_current_autogeneration_str(self, autogen_cmd=''):
        """Convenience: Autogenerates the most up-to-date database schema

        CommandLine:
            python -m dtool.sql_control --exec-get_schema_current_autogeneration_str

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> db = depc[tablename].db
            >>> result = db.get_schema_current_autogeneration_str('')
            >>> print(result)
        """
        db_version_current = self.get_db_version()
        # Define what tab space we want to save
        tab1 = ' ' * 4
        line_list = []
        # autogen_cmd = 'python -m dtool.DB_SCHEMA --test-test_dbschema
        # --force-incremental-db-update --dump-autogen-schema'
        # File Header
        line_list.append(ut.TRIPLE_DOUBLE_QUOTE)
        line_list.append('AUTOGENERATED ON ' + ut.timestamp('printable'))
        line_list.append('AutogenCommandLine:')
        # TODO: Fix autogen command
        line_list.append(ut.indent(autogen_cmd, tab1))
        line_list.append(ut.TRIPLE_DOUBLE_QUOTE)
        line_list.append('# -*- coding: utf-8 -*-')
        # line_list.append('from wbia import constants as const')
        line_list.append('\n')
        line_list.append('# =======================')
        line_list.append('# Schema Version Current')
        line_list.append('# =======================')
        line_list.append('\n')
        line_list.append('VERSION_CURRENT = %s' % ut.repr2(db_version_current))
        line_list.append('\n')
        line_list.append('def update_current(db, ibs=None):')
        # Function content
        first = True
        for tablename in sorted(self.get_table_names()):
            if first:
                first = False
            else:
                line_list.append('%s' % '')
            line_list += self.get_table_autogen_str(tablename)
            pass

        line_list.append('')
        return '\n'.join(line_list)

    def get_table_constraints(self, tablename):
        """
        TODO: use coldef_list with table_autogen_dict instead
        """
        constraint = self.metadata[tablename].constraint
        return None if constraint is None else constraint.split(';')

    def get_coldef_list(self, tablename):
        """
        Returns:
            list of (str, str) : each tuple is (col_name, col_type)
        """
        column_list = self.get_columns(tablename)

        coldef_list = []
        for column in column_list:
            col_name = column.name
            col_type = str(column[2])
            if column[5] == 1:
                col_type += ' PRIMARY KEY'
            elif column[3] == 1:
                col_type += ' NOT NULL'
            elif column[4] is not None:
                default_value = six.text_type(column[4])
                # HACK: add parens if the value contains parens in the future
                # all default values should contain parens
                LEOPARD_TURK_HACK = True
                if LEOPARD_TURK_HACK and '(' not in default_value:
                    col_type += ' DEFAULT %s' % default_value
                else:
                    col_type += ' DEFAULT (%s)' % default_value
            coldef_list.append((col_name, col_type))
        return coldef_list

    @profile
    def get_table_autogen_dict(self, tablename):
        r"""
        Args:
            tablename (str):

        Returns:
            dict: autogen_dict

        CommandLine:
            python -m dtool.sql_control get_table_autogen_dict

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> db = SQLDatabaseController.from_uri('sqlite:///')
            >>> tablename = 'dummy_table'
            >>> db.add_table(tablename, (
            >>>     ('rowid', 'INTEGER PRIMARY KEY'),
            >>>     ('value1', 'TEXT'),
            >>>     ('value2', 'TEXT NOT NULL'),
            >>>     ('value3', 'TEXT DEFAULT 1'),
            >>>     ('time_added', "INTEGER DEFAULT (CAST(STRFTIME('%s', 'NOW', 'UTC') AS INTEGER))")
            >>> ))
            >>> autogen_dict = db.get_table_autogen_dict(tablename)
            >>> result = ut.repr2(autogen_dict, nl=2)
            >>> print(result)
        """
        autogen_dict = ut.odict()
        autogen_dict['tablename'] = tablename
        autogen_dict['coldef_list'] = self.get_coldef_list(tablename)
        autogen_dict['docstr'] = self.get_table_docstr(tablename)
        autogen_dict['superkeys'] = self.get_table_superkey_colnames(tablename)
        autogen_dict['dependson'] = self.metadata[tablename].dependson
        return autogen_dict

    def get_table_autogen_str(self, tablename):
        r"""
        Args:
            tablename (str):

        Returns:
            str: quoted_docstr

        CommandLine:
            python -m dtool.sql_control get_table_autogen_str

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> db = SQLDatabaseController.from_uri('sqlite:///')
            >>> tablename = 'dummy_table'
            >>> db.add_table(tablename, (
            >>>     ('rowid', 'INTEGER PRIMARY KEY'),
            >>>     ('value', 'TEXT'),
            >>>     ('time_added', "INTEGER DEFAULT (CAST(STRFTIME('%s', 'NOW', 'UTC') AS INTEGER))")
            >>> ))
            >>> result = '\n'.join(db.get_table_autogen_str(tablename))
            >>> print(result)
        """
        line_list = []
        tab1 = ' ' * 4
        tab2 = ' ' * 8
        line_list.append(tab1 + 'db.add_table(%s, [' % (ut.repr2(tablename),))
        # column_list = db.get_columns(tablename)
        # colnamerepr_list = [ut.repr2(six.text_type(column[1]))
        #                     for column in column_list]
        autogen_dict = self.get_table_autogen_dict(tablename)
        coldef_list = autogen_dict['coldef_list']
        max_colsize = max(32, 2 + max(map(len, ut.take_column(coldef_list, 0))))
        # for column, colname_repr in zip(column_list, colnamerepr_list):
        for col_name, col_type in coldef_list:
            name_part = ('%s,' % ut.repr2(col_name)).ljust(max_colsize)
            type_part = ut.repr2(col_type)
            line_list.append(tab2 + '(%s%s),' % (name_part, type_part))
        line_list.append(tab1 + '],')
        superkeys = self.get_table_superkey_colnames(tablename)
        docstr = self.get_table_docstr(tablename)
        # Append metadata values
        specially_handled_table_metakeys = [
            'docstr',
            'superkeys',
            # 'constraint',
            'dependsmap',
        ]

        def quote_docstr(docstr):
            if docstr is None:
                return None
            import textwrap

            wraped_docstr = '\n'.join(textwrap.wrap(ut.textblock(docstr)))
            indented_docstr = ut.indent(wraped_docstr.strip(), tab2)
            _TSQ = ut.TRIPLE_SINGLE_QUOTE
            quoted_docstr = _TSQ + '\n' + indented_docstr + '\n' + tab2 + _TSQ
            return quoted_docstr

        line_list.append(tab2 + 'docstr=%s,' % quote_docstr(docstr))
        line_list.append(tab2 + 'superkeys=%s,' % (ut.repr2(superkeys),))
        # Hack out docstr and superkeys for now
        for suffix in METADATA_TABLE_COLUMN_NAMES:
            if suffix in specially_handled_table_metakeys:
                continue
            key = tablename + '_' + suffix
            val = getattr(self.metadata[tablename], suffix)
            logger.info(key)
            if val is not None:
                line_list.append(tab2 + '%s=%s,' % (suffix, ut.repr2(val)))
        dependsmap = self.metadata[tablename].dependsmap
        if dependsmap is not None:
            _dictstr = ut.indent(ut.repr2(dependsmap, nl=1), tab2)
            depends_map_dictstr = ut.align(_dictstr.lstrip(' '), ':')
            # hack for formatting
            depends_map_dictstr = depends_map_dictstr.replace(tab1 + '}', '}')
            line_list.append(tab2 + 'dependsmap=%s,' % (depends_map_dictstr,))
        line_list.append(tab1 + ')')
        return line_list

    def dump_schema(self):
        """
        Convenience: Dumps all csv database files to disk NOTE: This function
        is semi-obsolete because of the auto-generated current schema file.
        Use dump_schema_current_autogeneration instead for all purposes except
        for parsing out the database schema or for consice visual
        representation.
        """
        app_resource_dir = ut.get_app_resource_dir('wbia')
        dump_fpath = join(app_resource_dir, 'schema.txt')
        with open(dump_fpath, 'w') as file_:
            for tablename in sorted(self.get_table_names()):
                file_.write(tablename + '\n')
                column_list = self.get_columns(tablename)
                for column in column_list:
                    col_name = str(column[1]).ljust(30)
                    col_type = str(column[2]).ljust(10)
                    col_null = str(
                        ('ALLOW NULL' if column[3] == 1 else 'NOT NULL')
                    ).ljust(12)
                    col_default = str(column[4]).ljust(10)
                    col_key = str(('KEY' if column[5] == 1 else ''))
                    col = (col_name, col_type, col_null, col_default, col_key)
                    file_.write('\t%s%s%s%s%s\n' % col)
        ut.view_directory(app_resource_dir)

    def get_table_names(self, lazy=False):
        """ Conveinience: """
        if not lazy or self._tablenames is None:
            result = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tablename_list = result.fetchall()
            self._tablenames = {str(tablename[0]) for tablename in tablename_list}
        return self._tablenames

    @property
    def tablenames(self):
        return self.get_table_names()

    def has_table(self, tablename, colnames=None, lazy=True):
        """ checks if a table exists """
        # if not lazy or self._tablenames is None:
        return tablename in self.get_table_names(lazy=lazy)

    @profile
    def get_table_superkey_colnames(self, tablename):
        """
        get_table_superkey_colnames
        Actually resturns a list of tuples. need to change the name to
        get_table_superkey_colnames_list

        Args:
            tablename (str):

        Returns:
            list: superkeys

        CommandLine:
            python -m dtool.sql_control --test-get_table_superkey_colnames
            python -m wbia --tf get_table_superkey_colnames --tablename=contributors
            python -m wbia --tf get_table_superkey_colnames --db PZ_Master0 --tablename=annotations
            python -m wbia --tf get_table_superkey_colnames --db PZ_Master0 --tablename=contributors  # NOQA

        Example0:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> db = depc['chip'].db
            >>> superkeys = db.get_table_superkey_colnames('chip')
            >>> result = ut.repr2(superkeys, nl=False)
            >>> print(result)
            [('dummy_annot_rowid', 'config_rowid')]
        """
        assert tablename in self.get_table_names(
            lazy=True
        ), 'tablename=%r is not a part of this database' % (tablename,)
        superkeys = self.metadata[tablename].superkeys
        if superkeys is None:
            superkeys = []
        return superkeys

    def get_table_primarykey_colnames(self, tablename):
        columns = self.get_columns(tablename)
        primarykey_colnames = tuple(
            [name for (column_id, name, type_, notnull, dflt_value, pk) in columns if pk]
        )
        return primarykey_colnames

    def get_table_docstr(self, tablename):
        r"""
        CommandLine:
            python -m dtool.sql_control --exec-get_table_docstr

        Example0:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> db = depc[tablename].db
            >>> result = db.get_table_docstr(tablename)
            >>> print(result)
            Used to store individual chip features (ellipses)
        """
        return self.metadata[tablename].docstr

    def get_columns(self, tablename):
        """
        get_columns

        Args:
            tablename (str): table name

        Returns:
            column_list : list of tuples with format:
                (
                    column_id  : id of the column
                    name       : the name of the column
                    type_      : the type of the column
                    notnull    : 0 or 1 if the column can contains null values
                    dflt_value : the default value
                    pk         : 0 or 1 if the column partecipate to the primary key
                )

        References:
            http://stackoverflow.com/questions/17717829/how-to-get-column-names-from-a-table-in-sqlite-via-pragma-net-c
            http://stackoverflow.com/questions/1601151/how-do-i-check-in-sqlite-whether-a-table-exists

        CommandLine:
            python -m dtool.sql_control --exec-get_columns
            python -m dtool.sql_control --exec-get_columns --tablename=contributors
            python -m dtool.sql_control --exec-get_columns --tablename=nonexist

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> db = depc[tablename].db
            >>> colrichinfo_list = db.get_columns(tablename)
            >>> result = ('colrichinfo_list = %s' % (ut.repr2(colrichinfo_list, nl=1),))
            >>> print(result)
            colrichinfo_list = [
                (0, 'keypoint_rowid', 'INTEGER', 0, None, 1),
                (1, 'chip_rowid', 'INTEGER', 1, None, 0),
                (2, 'config_rowid', 'INTEGER', 0, '0', 0),
                (3, 'kpts', 'NDARRAY', 0, None, 0),
                (4, 'num', 'INTEGER', 0, None, 0),
            ]
        """
        # check if the table exists first. Throws an error if it does not exist.
        self.connection.execute('SELECT 1 FROM ' + tablename + ' LIMIT 1')
        result = self.connection.execute("PRAGMA TABLE_INFO('" + tablename + "')")
        colinfo_list = result.fetchall()
        colrichinfo_list = [SQLColumnRichInfo(*colinfo) for colinfo in colinfo_list]
        return colrichinfo_list

    def get_column_names(self, tablename):
        """ Conveinience: Returns the sql tablename columns """
        column_list = self.get_columns(tablename)
        column_names = ut.lmap(six.text_type, ut.take_column(column_list, 1))
        return column_names

    def get_column(self, tablename, name):
        """ Conveinience: """
        table, (column,) = sanitize_sql(self, tablename, (name,))
        return self.executeone(text(f'SELECT {column} FROM {table} ORDER BY rowid ASC'))

    def get_table_as_pandas(
        self, tablename, rowids=None, columns=None, exclude_columns=[]
    ):
        """
        aid = 30
        db = ibs.staging
        rowids = ut.flatten(ibs.get_review_rowids_from_single([aid]))
        tablename = 'reviews'
        exclude_columns = 'review_user_confidence review_user_identity'.split(' ')
        logger.info(db.get_table_as_pandas(tablename, rowids, exclude_columns=exclude_columns))

        db = ibs.db
        rowids = ut.flatten(ibs.get_annotmatch_rowids_from_aid([aid]))
        tablename = 'annotmatch'
        exclude_columns = 'annotmatch_confidence annotmatch_posixtime_modified annotmatch_reviewer'.split(' ')
        logger.info(db.get_table_as_pandas(tablename, rowids, exclude_columns=exclude_columns))
        """
        if rowids is None:
            rowids = self.get_all_rowids(tablename)
        column_list, column_names = self.get_table_column_data(
            tablename, rowids=rowids, columns=columns, exclude_columns=exclude_columns
        )
        import pandas as pd

        index = pd.Index(rowids, name='rowid')
        df = pd.DataFrame(ut.dzip(column_names, column_list), index=index)
        return df

    # TODO (25-Sept-12020) Deprecate once ResultProxy can be exposed,
    #      because it will allow result access by index or column name.
    def get_table_column_data(
        self, tablename, columns=None, exclude_columns=[], rowids=None
    ):
        """
        Grabs a table of information

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> db = depc[tablename].db
            >>> column_list, column_names = db.get_table_column_data(tablename)
        """
        if columns is None:
            all_column_names = self.get_column_names(tablename)
            column_names = ut.setdiff(all_column_names, exclude_columns)
        else:
            column_names = columns
        if rowids is not None:
            column_list = [
                self.get(tablename, (name,), rowids, unpack_scalars=True)
                for name in column_names
            ]
        else:
            column_list = [self.get_column(tablename, name) for name in column_names]
        return column_list, column_names

    def make_json_table_definition(self, tablename):
        r"""
        VERY HACKY FUNC RIGHT NOW. NEED TO FIX LATER

        Args:
            tablename (?):

        Returns:
            ?: new_transferdata

        CommandLine:
            python -m wbia --tf sql_control.make_json_table_definition

        CommandLine:
            python -m utool --tf iter_module_doctestable --modname=dtool.sql_control
            --include_inherited=True
            python -m dtool.sql_control --exec-make_json_table_definition

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> db = depc[tablename].db
            >>> table_def = db.make_json_table_definition(tablename)
            >>> result = ('table_def = %s' % (ut.repr2(table_def, nl=True),))
            >>> print(result)
            table_def = {
                'keypoint_rowid': 'INTEGER',
                'chip_rowid': 'INTEGER',
                'config_rowid': 'INTEGER',
                'kpts': 'NDARRAY',
                'num': 'INTEGER',
            }
        """
        new_transferdata = self.get_table_new_transferdata(tablename)
        (
            column_list,
            column_names,
            extern_colx_list,
            extern_superkey_colname_list,
            extern_superkey_colval_list,
            extern_tablename_list,
            extern_primarycolnames_list,
        ) = new_transferdata
        dependsmap = self.metadata[tablename].dependsmap

        richcolinfo_list = self.get_columns(tablename)
        table_dict_def = ut.odict([(r.name, r.type_) for r in richcolinfo_list])
        if dependsmap is not None:
            for key, val in dependsmap.items():
                if val[0] == tablename:
                    del table_dict_def[key]
                elif val[1] is None:
                    del table_dict_def[key]
                else:
                    # replace with superkey
                    del table_dict_def[key]
                    _deptablecols = self.get_columns(val[0])
                    superkey = val[2]
                    assert len(superkey) == 1, 'unhandled'
                    colinfo = {_.name: _ for _ in _deptablecols}[superkey[0]]
                    table_dict_def[superkey[0]] = colinfo.type_
        # json_def_str = ut.repr2(table_dict_def, aligned=True)
        return table_dict_def

    def get_table_new_transferdata(self, tablename, exclude_columns=[]):
        """
        CommandLine:
            python -m dtool.sql_control --test-get_table_column_data
            python -m dtool.sql_control --test-get_table_new_transferdata
            python -m dtool.sql_control --test-get_table_new_transferdata:1

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> db = depc[tablename].db
            >>> tablename_list = db.get_table_names()
            >>> colrichinfo_list = db.get_columns(tablename)
            >>> for tablename in tablename_list:
            ...     new_transferdata = db.get_table_new_transferdata(tablename)
            ...     column_list, column_names, extern_colx_list, extern_superkey_colname_list, extern_superkey_colval_list, extern_tablename_list, extern_primarycolnames_list = new_transferdata
            ...     print('tablename = %r' % (tablename,))
            ...     print('colnames = ' + ut.repr2(column_names))
            ...     print('extern_colx_list = ' + ut.repr2(extern_colx_list))
            ...     print('extern_superkey_colname_list = ' + ut.repr2(extern_superkey_colname_list))
            ...     print('L___')

        Example:
            >>> # SLOW_DOCTEST
            >>> # xdoctest: +REQUIRES(module:wbia)
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb('testdb1')
            >>> db = ibs.db
            >>> exclude_columns = []
            >>> tablename_list = ibs.db.get_table_names()
            >>> for tablename in tablename_list:
            ...     new_transferdata = db.get_table_new_transferdata(tablename)
            ...     column_list, column_names, extern_colx_list, extern_superkey_colname_list, extern_superkey_colval_list, extern_tablename_list, extern_primarycolnames_list = new_transferdata
            ...     print('tablename = %r' % (tablename,))
            ...     print('colnames = ' + ut.repr2(column_names))
            ...     print('extern_colx_list = ' + ut.repr2(extern_colx_list))
            ...     print('extern_superkey_colname_list = ' + ut.repr2(extern_superkey_colname_list))
            ...     print('L___')

        Example:
            >>> # SLOW_DOCTEST
            >>> # xdoctest: +REQUIRES(module:wbia)
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb('testdb1')
            >>> db = ibs.db
            >>> exclude_columns = []
            >>> tablename = ibs.const.IMAGE_TABLE
            >>> new_transferdata = db.get_table_new_transferdata(tablename)
            >>> column_list, column_names, extern_colx_list, extern_superkey_colname_list, extern_superkey_colval_list, extern_tablename_list, extern_primarycolnames_list = new_transferdata
            >>> dependsmap = db.metadata[tablename].dependsmap
            >>> print('tablename = %r' % (tablename,))
            >>> print('colnames = ' + ut.repr2(column_names))
            >>> print('extern_colx_list = ' + ut.repr2(extern_colx_list))
            >>> print('extern_superkey_colname_list = ' + ut.repr2(extern_superkey_colname_list))
            >>> print('dependsmap = %s' % (ut.repr2(dependsmap, nl=True),))
            >>> print('L___')
            >>> tablename = ibs.const.ANNOTATION_TABLE
            >>> new_transferdata = db.get_table_new_transferdata(tablename)
            >>> column_list, column_names, extern_colx_list, extern_superkey_colname_list, extern_superkey_colval_list, extern_tablename_list, extern_primarycolnames_list = new_transferdata
            >>> dependsmap = db.metadata[tablename].dependsmap
            >>> print('tablename = %r' % (tablename,))
            >>> print('colnames = ' + ut.repr2(column_names))
            >>> print('extern_colx_list = ' + ut.repr2(extern_colx_list))
            >>> print('extern_superkey_colname_list = ' + ut.repr2(extern_superkey_colname_list))
            >>> print('dependsmap = %s' % (ut.repr2(dependsmap, nl=True),))
            >>> print('L___')
        """
        import utool

        with utool.embed_on_exception_context:
            all_column_names = self.get_column_names(tablename)
            isvalid_list = [name not in exclude_columns for name in all_column_names]
            column_names = ut.compress(all_column_names, isvalid_list)
            column_list = [
                self.get_column(tablename, name)
                for name in column_names
                if name not in exclude_columns
            ]

            extern_colx_list = []
            extern_tablename_list = []
            extern_superkey_colname_list = []
            extern_superkey_colval_list = []
            extern_primarycolnames_list = []
            dependsmap = self.metadata[tablename].dependsmap
            if dependsmap is not None:
                for colname, dependtup in six.iteritems(dependsmap):
                    assert len(dependtup) == 3, 'must be 3 for now'
                    (
                        extern_tablename,
                        extern_primary_colnames,
                        extern_superkey_colnames,
                    ) = dependtup
                    if extern_primary_colnames is None:
                        # INFER PRIMARY COLNAMES
                        extern_primary_colnames = self.get_table_primarykey_colnames(
                            extern_tablename
                        )
                    if extern_superkey_colnames is None:

                        def get_standard_superkey_colnames(tablename_):
                            try:
                                # FIXME: Rectify duplicate code
                                superkeys = self.get_table_superkey_colnames(tablename_)
                                if len(superkeys) > 1:
                                    primary_superkey = self.metadata[
                                        tablename_
                                    ].primary_superkey
                                    self.get_table_superkey_colnames('contributors')
                                    if primary_superkey is None:
                                        raise AssertionError(
                                            (
                                                'tablename_=%r has multiple superkeys=%r, '
                                                'but no primary superkey.'
                                                ' A primary superkey is required'
                                            )
                                            % (tablename_, superkeys)
                                        )
                                    else:
                                        index = superkeys.index(primary_superkey)
                                        superkey_colnames = superkeys[index]
                                elif len(superkeys) == 1:
                                    superkey_colnames = superkeys[0]
                                else:
                                    logger.info(self.get_table_csv_header(tablename_))
                                    self.print_table_csv(
                                        'metadata', exclude_columns=['metadata_value']
                                    )
                                    # Execute hack to fix contributor tables
                                    if tablename_ == 'contributors':
                                        # hack to fix contributors table
                                        constraint_str = self.metadata[
                                            tablename_
                                        ].constraint
                                        parse_result = parse.parse(
                                            'CONSTRAINT superkey UNIQUE ({superkey})',
                                            constraint_str,
                                        )
                                        superkey = parse_result['superkey']
                                        assert (
                                            superkey == 'contributor_tag'
                                        ), 'hack failed1'
                                        assert (
                                            self.metadata['contributors'].superkey is None
                                        ), 'hack failed2'
                                        self.metadata['contributors'].superkey = [
                                            (superkey,)
                                        ]
                                        return (superkey,)
                                    else:
                                        raise NotImplementedError(
                                            'Cannot Handle: len(superkeys) == 0. '
                                            'Probably a degenerate case'
                                        )
                            except Exception as ex:
                                ut.printex(
                                    ex,
                                    'Error Getting superkey colnames',
                                    keys=['tablename_', 'superkeys'],
                                )
                                raise
                            return superkey_colnames

                        try:
                            extern_superkey_colnames = get_standard_superkey_colnames(
                                extern_tablename
                            )
                        except Exception as ex:
                            ut.printex(
                                ex,
                                'Error Building Transferdata',
                                keys=['tablename_', 'dependtup'],
                            )
                            raise
                        # INFER SUPERKEY COLNAMES
                    colx = ut.listfind(column_names, colname)
                    extern_rowids = column_list[colx]
                    superkey_column = self.get(
                        extern_tablename, extern_superkey_colnames, extern_rowids
                    )
                    extern_colx_list.append(colx)
                    extern_superkey_colname_list.append(extern_superkey_colnames)
                    extern_superkey_colval_list.append(superkey_column)
                    extern_tablename_list.append(extern_tablename)
                    extern_primarycolnames_list.append(extern_primary_colnames)

            new_transferdata = (
                column_list,
                column_names,
                extern_colx_list,
                extern_superkey_colname_list,
                extern_superkey_colval_list,
                extern_tablename_list,
                extern_primarycolnames_list,
            )
        return new_transferdata

    # def import_table_new_transferdata(tablename, new_transferdata):
    #    pass

    def merge_databases_new(self, db_src, ignore_tables=None, rowid_subsets=None):
        r"""
        Copies over all non-rowid properties into another sql table. handles
        annotated dependenceis.
        Does not handle external files
        Could handle dependency tree order, but not yet implemented.

        FINISHME

        Args:
            db_src (SQLController): merge data from db_src into db

        CommandLine:
            python -m dtool.sql_control --test-merge_databases_new:0
            python -m dtool.sql_control --test-merge_databases_new:2

        Example0:
            >>> # DISABLE_DOCTEST
            >>> # xdoctest: +REQUIRES(module:wbia)
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> import wbia
            >>> #ibs_dst = wbia.opendb(dbdir='testdb_dst')
            >>> ibs_src = wbia.opendb(db='testdb1')
            >>> # OPEN A CLEAN DATABASE
            >>> ibs_dst = wbia.opendb(dbdir='test_sql_merge_dst1', allow_newdir=True, delete_ibsdir=True)
            >>> ibs_src.ensure_contributor_rowids()
            >>> # build test data
            >>> db = ibs_dst.db
            >>> db_src = ibs_src.db
            >>> rowid_subsets = None
            >>> # execute function
            >>> db.merge_databases_new(db_src)

        Example1:
            >>> # DISABLE_DOCTEST
            >>> # xdoctest: +REQUIRES(module:wbia)
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> import wbia
            >>> ibs_src = wbia.opendb(db='testdb2')
            >>> # OPEN A CLEAN DATABASE
            >>> ibs_dst = wbia.opendb(dbdir='test_sql_merge_dst2', allow_newdir=True, delete_ibsdir=True)
            >>> ibs_src.ensure_contributor_rowids()
            >>> # build test data
            >>> db = ibs_dst.db
            >>> db_src = ibs_src.db
            >>> ignore_tables = ['lblannot', 'lblimage', 'image_lblimage_relationship', 'annotation_lblannot_relationship', 'keys']
            >>> rowid_subsets = None
            >>> # execute function
            >>> db.merge_databases_new(db_src, ignore_tables=ignore_tables)

        Example2:
            >>> # DISABLE_DOCTEST
            >>> # xdoctest: +REQUIRES(module:wbia)
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> import wbia
            >>> ibs_src = wbia.opendb(db='testdb2')
            >>> # OPEN A CLEAN DATABASE
            >>> ibs_src.fix_invalid_annotmatches()
            >>> ibs_dst = wbia.opendb(dbdir='test_sql_subexport_dst2', allow_newdir=True, delete_ibsdir=True)
            >>> ibs_src.ensure_contributor_rowids()
            >>> # build test data
            >>> db = ibs_dst.db
            >>> db_src = ibs_src.db
            >>> ignore_tables = ['lblannot', 'lblimage', 'image_lblimage_relationship', 'annotation_lblannot_relationship', 'keys']
            >>> # execute function
            >>> aid_subset = [1, 2, 3]
            >>> rowid_subsets = {ANNOTATION_TABLE: aid_subset,
            ...                  NAME_TABLE: ibs_src.get_annot_nids(aid_subset),
            ...                  IMAGE_TABLE: ibs_src.get_annot_gids(aid_subset),
            ...                  ANNOTMATCH_TABLE: [],
            ...                  GSG_RELATION_TABLE: [],
            ...                  }
            >>> db.merge_databases_new(db_src, ignore_tables=ignore_tables, rowid_subsets=rowid_subsets)
        """
        verbose = True
        veryverbose = True
        # Check version consistency
        version_dst = self.metadata.database.version
        version_src = db_src.metadata.database.version
        assert (
            version_src == version_dst
        ), 'cannot merge databases that have different versions'
        # Get merge tablenames
        all_tablename_list = self.get_table_names()
        # always ignore the metadata table.
        ignore_tables_ = ['metadata']
        if ignore_tables is None:
            ignore_tables = []
        ignore_tables_ += ignore_tables
        tablename_list = [
            tablename
            for tablename in all_tablename_list
            if tablename not in ignore_tables_
        ]
        # Reorder tablenames based on dependencies.
        # the tables with dependencies are merged after the tables they depend on
        dependsmap_list = [
            self.metadata[tablename].dependsmap for tablename in tablename_list
        ]
        dependency_digraph = {
            tablename: []
            if dependsmap is None
            else ut.get_list_column(dependsmap.values(), 0)
            for dependsmap, tablename in zip(dependsmap_list, tablename_list)
        }

        def find_depth(tablename, dependency_digraph):
            """
            depth first search to find root self cycles are counted as 0 depth
            will break if a true cycle exists
            """
            depth_list = [
                find_depth(depends_tablename, dependency_digraph)
                if depends_tablename != tablename
                else 0
                for depends_tablename in dependency_digraph[tablename]
            ]
            depth = 0 if len(depth_list) == 0 else max(depth_list) + 1
            return depth

        order_list = [
            find_depth(tablename, dependency_digraph) for tablename in tablename_list
        ]
        sorted_tablename_list = ut.sortedby(tablename_list, order_list)
        # ================================
        # Merge each table into new database
        # ================================
        tablename_to_rowidmap = {}  # TODO
        # old_rowids_to_new_roids
        for tablename in sorted_tablename_list:
            if verbose:
                logger.info('\n[sqlmerge] Merging tablename=%r' % (tablename,))
            # Collect the data from the source table that will be merged in
            new_transferdata = db_src.get_table_new_transferdata(tablename)
            # FIXME: This needs to pass back sparser output
            (
                column_list,
                column_names,
                # These fields are for external data dependencies. We need to find what the
                # new rowids will be in the destintation database
                extern_colx_list,
                extern_superkey_colname_list,
                extern_superkey_colval_list,
                extern_tablename_list,
                extern_primarycolnames_list,
            ) = new_transferdata
            # FIXME: extract the primary rowid column a little bit nicer
            assert column_names[0].endswith('_rowid')
            old_rowid_list = column_list[0]
            column_names_ = column_names[1:]
            column_list_ = column_list[1:]

            # +=================================================
            # WIP: IF SUBSET REQUSTED FILTER OUT INVALID ROWIDS
            if rowid_subsets is not None and tablename in rowid_subsets:
                valid_rowids = set(rowid_subsets[tablename])
                isvalid_list = [rowid in valid_rowids for rowid in old_rowid_list]
                valid_old_rowid_list = ut.compress(old_rowid_list, isvalid_list)
                valid_column_list_ = [
                    ut.compress(col, isvalid_list) for col in column_list_
                ]
                valid_extern_superkey_colval_list = [
                    ut.compress(col, isvalid_list) for col in extern_superkey_colval_list
                ]
                logger.info(
                    ' * filtered number of rows from %d to %d.'
                    % (len(valid_rowids), len(valid_old_rowid_list))
                )
            else:
                logger.info(' * no filtering requested')
                valid_extern_superkey_colval_list = extern_superkey_colval_list
                valid_old_rowid_list = old_rowid_list
                valid_column_list_ = column_list_
            # if len(valid_old_rowid_list) == 0:
            #    continue
            # L=================================================

            # ================================
            # Resolve external superkey lookups
            # ================================
            if len(extern_colx_list) > 0:
                if verbose:
                    logger.info(
                        '[sqlmerge] %s has %d externaly dependant columns to resolve'
                        % (tablename, len(extern_colx_list))
                    )
                modified_column_list_ = valid_column_list_[:]
                new_extern_rowid_list = []

                # Find the mappings from the old tables rowids to the new tables rowids
                for tup in zip(
                    extern_colx_list,
                    extern_superkey_colname_list,
                    valid_extern_superkey_colval_list,
                    extern_tablename_list,
                    extern_primarycolnames_list,
                ):
                    (
                        colx,
                        extern_superkey_colname,
                        extern_superkey_colval,
                        extern_tablename,
                        extern_primarycolname,
                    ) = tup
                    source_colname = column_names_[colx - 1]
                    if veryverbose or verbose:
                        if veryverbose:
                            logger.info('[sqlmerge] +--')
                            logger.info(
                                (
                                    '[sqlmerge] * resolving source_colname=%r \n'
                                    '                 via extern_superkey_colname=%r ...\n'
                                    '                 -> extern_primarycolname=%r. colx=%r'
                                )
                                % (
                                    source_colname,
                                    extern_superkey_colname,
                                    extern_primarycolname,
                                    colx,
                                )
                            )
                        elif verbose:
                            logger.info(
                                '[sqlmerge] * resolving %r via %r -> %r'
                                % (
                                    source_colname,
                                    extern_superkey_colname,
                                    extern_primarycolname,
                                )
                            )
                    _params_iter = list(zip(extern_superkey_colval))
                    new_extern_rowids = self.get_rowid_from_superkey(
                        extern_tablename,
                        _params_iter,
                        superkey_colnames=extern_superkey_colname,
                    )
                    num_Nones = sum(ut.flag_None_items(new_extern_rowids))
                    if verbose:
                        logger.info(
                            '[sqlmerge] * there were %d none items' % (num_Nones,)
                        )
                    # ut.assert_all_not_None(new_extern_rowids)
                    new_extern_rowid_list.append(new_extern_rowids)

                for colx, new_extern_rowids in zip(
                    extern_colx_list, new_extern_rowid_list
                ):
                    modified_column_list_[colx - 1] = new_extern_rowids
            else:
                modified_column_list_ = valid_column_list_

            # ================================
            # Merge into db with add_cleanly
            # ================================
            superkey_colnames_list = self.get_table_superkey_colnames(tablename)
            try:
                superkey_paramxs_list = [
                    [column_names_.index(str(superkey)) for superkey in superkey_colnames]
                    for superkey_colnames in superkey_colnames_list
                ]
            except Exception as ex:
                ut.printex(ex, keys=['column_names_', 'superkey_colnames_list'])
                raise
            if len(superkey_colnames_list) > 1:
                # FIXME: Rectify duplicate code
                primary_superkey = self.metadata[tablename].primary_superkey
                if primary_superkey is None:
                    raise AssertionError(
                        (
                            'tablename=%r has multiple superkey_colnames_list=%r, '
                            'but no primary superkey. '
                            'A primary superkey is required'
                        )
                        % (tablename, superkey_colnames_list)
                    )
                else:
                    superkey_index = superkey_colnames_list.index(primary_superkey)
                    superkey_paramx = superkey_paramxs_list[superkey_index]
                    superkey_colnames = superkey_colnames_list[superkey_index]
            elif len(superkey_colnames_list) == 1:
                superkey_paramx = superkey_paramxs_list[0]
                superkey_colnames = superkey_colnames_list[0]
            else:
                superkey_paramx = superkey_paramxs_list[0]
                superkey_colnames = superkey_colnames_list[0]
                # def get_referenced_table():
                #     # TODO use foreign keys to infer this data instead of hacks
                #     pass
                # logger.info('superkey_paramxs_list = %r' % (superkey_paramxs_list, ))
                # logger.info('superkey_colnames_list = %r' % (superkey_colnames_list, ))
                # raise ValueError('Cannot merge %r' % (tablename, ))

            params_iter = list(zip(*modified_column_list_))

            def get_rowid_from_superkey(*superkey_column_list):
                superkey_params_iter = zip(*superkey_column_list)
                rowid = self.get_rowid_from_superkey(
                    tablename, superkey_params_iter, superkey_colnames=superkey_colnames
                )
                return rowid

            # TODO: allow for cetrain databases to take precidence over another
            # basically allow insert or replace
            new_rowid_list = self.add_cleanly(
                tablename,
                column_names_,
                params_iter,
                get_rowid_from_superkey=get_rowid_from_superkey,
                superkey_paramx=superkey_paramx,
            )
            # TODO: Use mapping generated here for new rowids
            old_rowids_to_new_roids = dict(
                zip(valid_old_rowid_list, new_rowid_list)  # NOQA
            )
            tablename_to_rowidmap[tablename] = old_rowids_to_new_roids

    def get_table_csv(self, tablename, exclude_columns=[], rowids=None, truncate=False):
        """
        Converts a tablename to csv format

        Args:
            tablename (str):
            exclude_columns (list):

        Returns:
            str: csv_table

        CommandLine:
            python -m dtool.sql_control --test-get_table_csv
            python -m dtool.sql_control --exec-get_table_csv --tablename=contributors

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> depc.clear_all()
            >>> rowids = depc.get_rowids('notch', [1, 2, 3])
            >>> table = depc['notch']
            >>> db = table.db
            >>> ut.exec_funckw(db.get_table_csv, globals())
            >>> tablename = 'notch'
            >>> csv_table = db.get_table_csv(tablename, exclude_columns, truncate=True)
            >>> print(csv_table)
        """
        # =None, column_list=[], header='', column_type=None
        column_list, column_names = self.get_table_column_data(
            tablename, exclude_columns=exclude_columns, rowids=rowids
        )
        # remove column prefix for more compact csvs
        column_lbls = [name.replace(tablename[:-1] + '_', '') for name in column_names]
        header = self.get_table_csv_header(tablename)
        # truncate = True
        if truncate:
            column_list = [
                [ut.trunc_repr(col) for col in column] for column in column_list
            ]

        csv_table = ut.make_csv_table(column_list, column_lbls, header, comma_repl=';')
        csv_table = ut.ensure_unicode(csv_table)
        # csv_table = ut.make_csv_table(column_list, column_lbls, header, comma_repl='<comma>')
        return csv_table

    def print_table_csv(self, tablename, exclude_columns=[], truncate=False):
        logger.info(
            self.get_table_csv(
                tablename, exclude_columns=exclude_columns, truncate=truncate
            )
        )

    def get_table_csv_header(db, tablename):
        coldef_list = db.get_coldef_list(tablename)
        header_constraints = '# CONSTRAINTS: %r' % db.get_table_constraints(tablename)
        header_name = '# TABLENAME: %r' % tablename
        header_types = ut.indentjoin(coldef_list, '\n# ')
        docstr = db.get_table_docstr(tablename)
        if docstr is None:
            docstr = ''
        header_doc = ut.indentjoin(ut.unindent(docstr).split('\n'), '\n# ')
        header = (
            header_doc + '\n' + header_name + header_types + '\n' + header_constraints
        )
        return header

    def print_schema(self):
        for tablename in self.get_table_names():
            logger.info(self.get_table_csv_header(tablename) + '\n')

    def view_db_in_external_reader(self):
        known_readers = ['sqlitebrowser', 'sqliteman']
        sqlite3_reader = known_readers[0]
        os.system(sqlite3_reader + ' ' + self.uri)
        # ut.cmd(sqlite3_reader, sqlite3_db_fpath)
        pass

    def set_db_version(self, version):
        # Do things properly, get the metadata_rowid (best because we want to assert anyway)
        metadata_key_list = ['database_version']
        params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
        where_clause = 'metadata_key=?'
        # list of relationships for each image
        metadata_rowid_list = self.get_where(
            METADATA_TABLE_NAME,
            ('metadata_rowid',),
            params_iter,
            where_clause,
            unpack_scalars=True,
        )
        assert (
            len(metadata_rowid_list) == 1
        ), 'duplicate database_version keys in database'
        id_iter = ((metadata_rowid,) for metadata_rowid in metadata_rowid_list)
        val_list = ((_,) for _ in [version])
        self.set(METADATA_TABLE_NAME, ('metadata_value',), val_list, id_iter)

    def get_sql_version(self):
        """ Conveinience """
        self.connection.execute('SELECT sqlite_version()')
        sql_version = self.connection.fetchone()
        logger.info('[sql] SELECT sqlite_version = %r' % (sql_version,))
        # The version number sqlite3 module. NOT the version of SQLite library.
        logger.info('[sql] sqlite3.version = %r' % (lite.version,))
        # The version of the SQLite library
        logger.info('[sql] sqlite3.sqlite_version = %r' % (lite.sqlite_version,))
        return sql_version

    def __getitem__(self, key):
        if not self.has_table(key):
            raise KeyError('Choose on of: ' + str(self.tablenames))
        table = SQLTable(self, name=key)
        return table


@six.add_metaclass(ut.ReloadingMetaclass)
class SQLTable(ut.NiceRepr):
    """
    convinience object for dealing with a specific table

    table = db
    table = SQLTable(db, 'annotmatch')
    """

    def __init__(table, db, name):
        table.db = db
        table.name = name
        table._setup_column_methods()

    def get(table, colnames, id_iter, id_colname='rowid', eager=True):
        return table.db.get(
            table.name, colnames, id_iter=id_iter, id_colname=id_colname, eager=eager
        )

    def _setup_column_methods(table):
        def _make_getter(column):
            def _getter(table, rowids):
                table.get(column, rowids)

            return _getter

        for column in table.db.get_column_names(table.name):
            getter = _make_getter(column)
            ut.inject_func_as_method(table, getter, '{}'.format(column))

    def number_of_rows(table):
        return table.db.get_row_count(table.name)

    def as_pandas(table, rowids=None, columns=None):
        return table.db.get_table_as_pandas(table.name, rowids=rowids, columns=columns)

    def rowids(table):
        return table.db.get_all_rowids(table.name)

    def delete(table, rowids):
        table.db.delete_rowids(table.name, rowids)

    def clear(table):
        rowids = table.rowids()
        table.delete(rowids)

    def __nice__(table):
        return table.name + ', n=' + str(table.number_of_rows())
