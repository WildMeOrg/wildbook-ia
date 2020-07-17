# -*- coding: utf-8 -*-
"""
Interface into SQL for the IBEIS Controller

TODO; need to use some sort of sticky bit so
sql files are created with reasonable permissions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import parse
import re
import threading
from functools import partial
from os.path import join, exists, dirname, basename

import six
import utool as ut
from six.moves import map, zip, cStringIO

from wbia.dtool import sqlite3 as lite

print, rrr, profile = ut.inject2(__name__)


METADATA_TABLE = 'metadata'

READ_ONLY = ut.get_argflag(('--readonly-mode', '--read-only', '--readonly'))
VERBOSE_SQL = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql', '--verbsql'))
NOT_QUIET = not (ut.QUIET or ut.get_argflag('--quiet-sql'))

VERBOSE = ut.VERBOSE
VERYVERBOSE = ut.VERYVERBOSE
COPY_TO_MEMORY = ut.get_argflag(('--copy-db-to-memory'))
# AUTODUMP       = ut.get_argflag('--auto-dump')

TIMEOUT = 600  # Wait for up to 600 seconds for the database to return from a locked state

SQLColumnRichInfo = collections.namedtuple(
    'SQLColumnRichInfo', ('column_id', 'name', 'type_', 'notnull', 'dflt_value', 'pk')
)


def _unpacker(results_):
    """ HELPER: Unpacks results if unpack_scalars is True. """
    if len(results_) == 0:
        results = None
    else:
        assert len(results_) <= 1, 'throwing away results! { %r }' % (results_,)
        results = results_[0]
    return results


def tuplize(list_):
    """ Converts each scalar item in a list to a dimension-1 tuple """
    tup_list = [item if ut.isiterable(item) else (item,) for item in list_]
    return tup_list


def flattenize(list_):
    """
    maps flatten to a tuplized list

    Weird function. DEPRICATE

    Example:
        >>> # DISABLE_DOCTEST
        >>> list_ = [[1, 2, 3], [2, 3, [4, 2, 1]], [3, 2], [[1, 2], [3, 4]]]
        >>> import utool
        >>> from itertools import zip
        >>> val_list1 = [(1, 2), (2, 4), (5, 3)]
        >>> id_list1  = [(1,),     (2,),   (3,)]
        >>> out_list1 = utool.flattenize(zip(val_list1, id_list1))

        >>> val_list2 = [1, 4, 5]
        >>> id_list2  = [(1,),     (2,),   (3,)]
        >>> out_list2 = utool.flattenize(zip(val_list2, id_list2))

        >>> val_list3 = [1, 4, 5]
        >>> id_list3  = [1, 2, 3]
        >>> out_list3 = utool.flattenize(zip(val_list3, id_list3))

        out_list4 = list(zip(val_list3, id_list3))
        %timeit utool.flattenize(zip(val_list1, id_list1))
        %timeit utool.flattenize(zip(val_list2, id_list2))
        %timeit utool.flattenize(zip(val_list3, id_list3))
        %timeit list(zip(val_list3, id_list3))

        100000 loops, best of 3: 14 us per loop
        100000 loops, best of 3: 16.5 us per loop
        100000 loops, best of 3: 18 us per loop
        1000000 loops, best of 3: 1.18 us per loop
    """
    tuplized_iter = map(tuplize, list_)
    flatenized_list = list(map(ut.flatten, tuplized_iter))
    return flatenized_list


# =======================
# SQL Context Class
# =======================


class SQLExecutionContext(object):
    """
    Context manager for transactional database calls

    FIXME: hash out details. I don't think anybody who programmed this
    knows what is going on here. So much for fine grained control.

    Referencs:
        http://stackoverflow.com/questions/9573768/understand-sqlite-multi-module-envs

    """

    def __init__(
        context,
        db,
        operation,
        nInput=None,
        auto_commit=True,
        start_transaction=False,
        keepwrap=False,
        verbose=VERBOSE_SQL,
        tablename=None,
    ):
        context.tablename = None
        context.auto_commit = auto_commit
        context.db = db
        context.operation = operation
        context.nInput = nInput
        context.start_transaction = start_transaction
        context.operation_type = get_operation_type(operation)
        context.verbose = verbose
        context.is_insert = context.operation_type.startswith('INSERT')
        context.keepwrap = keepwrap
        context.cur = None
        context.connection = None

    def __enter__(context):
        """ Checks to see if the operating will change the database """
        # ut.printif(lambda: '[sql] Callers: ' + ut.get_caller_name(range(3, 6)), DEBUG)
        if context.nInput is not None:
            context.operation_lbl = '[sql] execute nInput=%d optype=%s: ' % (
                context.nInput,
                context.operation_type,
            )
        else:
            context.operation_lbl = '[sql] executeone optype=%s: ' % (
                context.operation_type
            )
        # Start SQL Transaction

        context.connection = context.db.connection
        try:
            context.cur = context.connection.cursor()  # HACK in a new cursor
        except lite.ProgrammingError:
            # Get connection for new thread
            context.connection = context.db.thread_connection()
            context.cur = context.connection.cursor()

        # context.cur = context.db.cur  # OR USE DB CURSOR??
        if context.start_transaction:
            # context.cur.execute('BEGIN', ())
            try:
                context.cur.execute('BEGIN')
            except lite.OperationalError:
                context.connection.rollback()
                context.cur.execute('BEGIN')
        if context.verbose or VERBOSE_SQL:
            print(context.operation_lbl)
            if context.verbose:
                print('[sql] operation=\n' + context.operation)
        # Comment out timeing code
        # if __debug__:
        #    if NOT_QUIET and (VERBOSE_SQL or context.verbose):
        #        context.tt = ut.tic(context.operation_lbl)
        return context

    # @profile
    def execute_and_generate_results(context, params):
        """ helper for context statment """
        try:
            context.cur.execute(context.operation, params)
        except lite.Error as ex:
            print('Reporting SQLite Error')
            print('params = ' + ut.repr2(params, truncate=not ut.VERBOSE))
            ut.printex(ex, 'sql.Error', keys=['params'])
            if (
                hasattr(ex, 'message')
                and ex.message.find('probably unsupported type') > -1
            ):
                print(
                    'ERR REPORT: given param types = ' + ut.repr2(ut.lmap(type, params))
                )
                if context.tablename is None:
                    if context.operation_type.startswith('SELECT'):
                        tablename = ut.str_between(
                            context.operation, 'FROM', 'WHERE'
                        ).strip()
                    else:
                        tablename = context.operation_type.split(' ')[-1]
                else:
                    tablename = context.tablename
                try:
                    coldef_list = context.db.get_coldef_list(tablename)
                    print('ERR REPORT: expected types = %s' % (ut.repr4(coldef_list),))
                except Exception:
                    pass
            raise
        return context._results_gen()

    # @profile
    def _results_gen(context):
        """ HELPER - Returns as many results as there are.
        Careful. Overwrites the results once you call it.
        Basically: Dont call this twice.
        """
        if context.is_insert:
            # The sqlite3_last_insert_rowid(D) interface returns the
            # <b> rowid of the most recent successful INSERT </b>
            # into a rowid table in D
            context.cur.execute('SELECT last_insert_rowid()', ())
        # Wraping fetchone in a generator for some pretty tight calls.
        while True:
            result = context.cur.fetchone()
            if not result:
                return
            if context.keepwrap:
                # Results are always returned wraped in a tuple
                yield result
            else:
                # Here unpacking is conditional
                # FIXME: can this if be removed?
                yield result[0] if len(result) == 1 else result

    def __exit__(context, type_, value, trace):
        """ Finalization of an SQLController call """
        if trace is not None:
            # An SQLError is a serious offence.
            print('[sql] FATAL ERROR IN QUERY CONTEXT')
            print('[sql] operation=\n' + context.operation)
            DUMP_ON_EXCEPTION = False
            if DUMP_ON_EXCEPTION:
                # Dump on error
                context.db.dump()
            print('[sql] Error in context manager!: ' + str(value))
            # return a falsey value on error
            return False
        else:
            # Commit the transaction
            if context.auto_commit:
                context.connection.commit()
            else:
                print('no commit %r' % context.operation_lbl)


def get_operation_type(operation):
    """
    Parses the operation_type from an SQL operation
    """
    operation = ' '.join(operation.split('\n')).strip()
    operation_type = operation.split(' ')[0].strip()
    if operation_type.startswith('SELECT'):
        operation_args = ut.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('INSERT'):
        operation_args = ut.str_between(operation, operation_type, '(').strip()
    elif operation_type.startswith('DROP'):
        operation_args = ''
    elif operation_type.startswith('ALTER'):
        operation_args = ''
    elif operation_type.startswith('UPDATE'):
        operation_args = ut.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('DELETE'):
        operation_args = ut.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('CREATE'):
        operation_args = ut.str_between(operation, operation_type, '(').strip()
    else:
        operation_args = None
    operation_type += ' ' + operation_args.replace('\n', ' ')
    return operation_type.upper()


def sanitize_sql(db, tablename_, columns=None):
    """ Sanatizes an sql tablename and column. Use sparingly """
    tablename = re.sub('[^a-zA-Z_0-9]', '', tablename_)
    valid_tables = db.get_table_names()
    if tablename not in valid_tables:
        print('tablename_ = %r' % (tablename_,))
        print('valid_tables = %r' % (valid_tables,))
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
        print('[sql] ATTEMPTING TO TEST NEW SQLDB VERSION')
        devdb_list = [
            'PZ_MTEST',
            'testdb1',
            'testdb0',
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
            print('[sql] TESTING NEW SQLDB VERSION: %r' % (version_next,))
            # print('[sql] ... pass --force-fresh to reload any changes')
            return version_next, dev_sqldb_fname
        else:
            print('[ibs] NOT TESTING')
    return version_current, sqldb_fname


@six.add_metaclass(ut.ReloadingMetaclass)
class SQLDatabaseController(object):
    """
    Interface to an SQL database
    """

    @profile
    def __init__(
        db,
        sqldb_dpath='.',
        sqldb_fname='database.sqlite3',
        text_factory=six.text_type,
        inmemory=None,
        fpath=None,
        readonly=None,
        always_check_metadata=True,
        timeout=TIMEOUT,
    ):
        """ Creates db and opens connection

        Args:
            sqldb_dpath (unicode):  directory path string(default = '.')
            sqldb_fname (unicode): (default = 'database.sqlite3')
            text_factory (type): (default = unicode)
            inmemory (None): (default = None)
            fpath (str):  file path string(default = None)
            readonly (bool): (default = False)

        CommandLine:
            python -m dtool.sql_control --exec-__init__

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> ut.exec_funckw(SQLDatabaseController.__init__, locals())
            >>> sqldb_dpath = ut.ensure_app_resource_dir('dtool')
            >>> sqldb_fname = u'test_database.sqlite3'
            >>> readonly = False
            >>> db = SQLDatabaseController(sqldb_dpath, sqldb_fname)
            >>> db.print_schema()
            >>> print(db)
            >>> db2 = SQLDatabaseController(sqldb_dpath, sqldb_fname, readonly=True)
            >>> db.add_table('temptable', (
            >>>     ('rowid',               'INTEGER PRIMARY KEY'),
            >>>     ('key',                 'TEXT'),
            >>>     ('val',                 'TEXT'),
            >>> ),
            >>>     superkeys=[('key',)])
            >>> db2.print_schema()
        """
        # standard metadata table keys for each docstr
        # TODO: generalize the places that use this so to add a new cannonical
        # metadata field it is only necessary to append to this list.
        if readonly is None:
            readonly = READ_ONLY
            # HACK

        db.timeout = timeout
        db._tablenames = None
        db.readonly = readonly
        db.table_metadata_keys = [
            # 'constraint',
            'dependson',
            'docstr',
            'relates',
            'shortname',
            'superkeys',
            'extern_tables',
            'dependsmap',
            'primary_superkey',
        ]
        # Get SQL file path
        if fpath is None:
            db.dir_ = sqldb_dpath
            db.fname = sqldb_fname
            db.fpath = join(db.dir_, db.fname)
        else:
            db.fpath = fpath
            db.dir_ = dirname(db.fpath)
            db.fname = basename(db.fpath)

        db.text_factory = text_factory

        is_new = not exists(db.fpath)

        db.thread_connections = {}

        # Create connection
        connection, uri = db._create_connection()
        db.connection = connection
        db.uri = uri

        # Get a cursor which will preform sql commands / queries / executions
        db.cur = db.connection.cursor()
        # db.connection.isolation_level = None  # turns sqlite3 autocommit off
        # db.connection.isolation_level = lite.IMMEDIATE  # turns sqlite3 autocommit off
        if inmemory is True or (inmemory is None and COPY_TO_MEMORY):
            db.squeeze()
            db._copy_to_memory()
            db.connection.text_factory = text_factory

        db.USE_FOREIGN_HACK = False
        if db.USE_FOREIGN_HACK:
            db.cur.execute('PRAGMA foreign_keys = ON')
            # db.cur.execute("PRAGMA foreign_keys;")
            # print(db.cur.fetchall())

        # Optimize the database (if anything is set)
        if is_new:
            db.optimize()

        if not is_new:
            # Check for old database versions
            try:
                db.get_db_version(ensure=False)
            except lite.OperationalError:
                always_check_metadata = True

        if not db.readonly:
            if is_new or always_check_metadata:
                # TODO: make this happen lazilly
                db._ensure_metadata_table()

    def _create_connection(db):
        if db.fname == ':memory:':
            uri = None
            connection = lite.connect(
                ':memory:', detect_types=lite.PARSE_DECLTYPES, timeout=db.timeout
            )
        else:
            assert exists(db.dir_), '[sql] db.dir_=%r does not exist!' % db.dir_
            if not exists(db.fpath):
                print('[sql] Initializing new database: %r' % (db.fname,))
                if db.readonly:
                    raise AssertionError('Cannot open a new database in readonly mode')
            # Open the SQL database connection with support for custom types
            # lite.enable_callback_tracebacks(True)
            # db.fpath = ':memory:'

            if six.PY3:
                # References:
                # http://stackoverflow.com/questions/10205744/opening-sqlite3-database-from-python-in-read-only-mode
                uri = 'file:' + db.fpath
                if db.readonly:
                    uri += '?mode=ro'
                connection = lite.connect(
                    uri, uri=True, detect_types=lite.PARSE_DECLTYPES, timeout=db.timeout
                )
            else:
                import os

                if db.readonly:
                    assert not ut.WIN32, 'cannot open readonly on windows.'
                    flag = os.O_RDONLY  # if db.readonly else os.O_RDWR
                    fd = os.open(db.fpath, flag)
                    uri = '/dev/fd/%d' % fd
                    connection = lite.connect(
                        uri, detect_types=lite.PARSE_DECLTYPES, timeout=db.timeout
                    )
                    os.close(fd)
                else:
                    uri = db.fpath
                    connection = lite.connect(
                        uri, detect_types=lite.PARSE_DECLTYPES, timeout=db.timeout
                    )

        # Keep track of what thead this was started in
        threadid = threading.current_thread()
        db.thread_connections[threadid] = connection
        connection.text_factory = db.text_factory

        return connection, uri

    def get_fpath(db):
        return db.fpath

    def close(db):
        db.cur = None
        db.connection.close()
        db.thread_connections = {}

    # def reconnect(db):
    #     # Call this if we move into a new thread
    #     assert db.fname != ':memory:', 'cant reconnect to mem'
    #     connection, uri = db._create_connection()
    #     db.connection = connection
    #     db.cur = db.connection.cursor()

    def thread_connection(db):
        threadid = threading.current_thread()
        if threadid in db.thread_connections:
            connection = db.thread_connections[threadid]
        else:
            connection, uri = db._create_connection()
        return connection

    @profile
    def _ensure_metadata_table(db):
        """
        Creates the metadata table if it does not exist

        We need this to be done every time so that the update code works
        correctly.
        """
        try:
            orig_table_kw = db.get_table_autogen_dict(METADATA_TABLE)
        except (lite.OperationalError, NameError):
            orig_table_kw = None

        meta_table_kw = ut.odict(
            [
                ('tablename', METADATA_TABLE),
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
            db.add_table(**meta_table_kw)
        # METADATA_TABLE,
        #    superkeys=[('metadata_key',)],
        # IMPORTANT: Yes, we want this line to be tabbed over for the
        # schema auto-generation
        # Ensure that a version number exists
        db.get_db_version(ensure=True)
        # Ensure that an init UUID exists
        db.get_db_init_uuid(ensure=True)

    def get_db_version(db, ensure=True):
        version = db.get_metadata_val('database_version', default=None)
        if version is None and ensure:
            BASE_DATABASE_VERSION = '0.0.0'
            version = BASE_DATABASE_VERSION
            colnames = ['metadata_key', 'metadata_value']
            params_iter = zip(['database_version'], [version])
            # We don't care to find any, because we know there is no version

            def get_rowid_from_superkey(x):
                return [None] * len(x)

            db.add_cleanly(METADATA_TABLE, colnames, params_iter, get_rowid_from_superkey)
        return version

    def get_db_init_uuid(db, ensure=True):
        """
        Get the database initialization (creation) UUID

        CommandLine:
            python -m dtool.sql_control get_db_init_uuid

        Example:
            >>> # ENABLE_DOCTEST
            >>> import uuid
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> # Check random database gets new UUID on init
            >>> db = SQLDatabaseController(sqldb_fname=':memory:')
            >>> uuid_ = db.get_db_init_uuid()
            >>> print('New Database: %r is valid' % (uuid_, ))
            >>> assert isinstance(uuid_, uuid.UUID)
            >>> # Check existing database keeps UUID
            >>> sqldb_dpath = ut.ensure_app_resource_dir('dtool')
            >>> sqldb_fname = u'test_database.sqlite3'
            >>> readonly = False
            >>> db1 = SQLDatabaseController(sqldb_dpath, sqldb_fname)
            >>> uuid_1 = db1.get_db_init_uuid()
            >>> db2 = SQLDatabaseController(sqldb_dpath, sqldb_fname)
            >>> uuid_2 = db2.get_db_init_uuid()
            >>> print('Existing Database: %r == %r' % (uuid_1, uuid_2, ))
            >>> assert uuid_1 == uuid_2
        """
        import uuid

        db_init_uuid_str = db.get_metadata_val('database_init_uuid', default=None)
        if db_init_uuid_str is None and ensure:
            db_init_uuid_str = six.text_type(uuid.uuid4())
            colnames = ['metadata_key', 'metadata_value']
            params_iter = zip(['database_init_uuid'], [db_init_uuid_str])
            # We don't care to find any, because we know there is no version

            def get_rowid_from_superkey(x):
                return [None] * len(x)

            db.add_cleanly(METADATA_TABLE, colnames, params_iter, get_rowid_from_superkey)
        db_init_uuid = uuid.UUID(db_init_uuid_str)
        return db_init_uuid

    def _copy_to_memory(db):
        """
        References:
            http://stackoverflow.com/questions/3850022/python-sqlite3-load-existing-db-file-to-memory
        """
        if NOT_QUIET:
            print('[sql] Copying database into RAM')
        tempfile = cStringIO()
        for line in db.connection.iterdump():
            tempfile.write('%s\n' % line)
        db.connection.close()
        tempfile.seek(0)
        # Create a database in memory and import from tempfile
        db.connection = lite.connect(
            ':memory:', detect_types=lite.PARSE_DECLTYPES, timeout=db.timeout
        )
        db.connection.cursor().executescript(tempfile.read())
        db.connection.commit()
        db.connection.row_factory = lite.Row

    # @ut.memprof
    def reboot(db):
        print('[sql] reboot')
        db.cur.close()
        del db.cur
        db.connection.close()
        del db.connection
        db.connection = lite.connect(
            db.fpath, detect_types=lite.PARSE_DECLTYPES, timeout=db.timeout
        )
        db.connection.text_factory = db.text_factory
        db.cur = db.connection.cursor()

    def backup(db, backup_filepath):
        """
        backup_filepath = dst_fpath
        """
        # Create a brand new conenction to lock out current thread and any others
        connection, uri = db._create_connection()
        # Start Exclusive transaction, lock out all other writers from making database changes
        connection.isolation_level = 'EXCLUSIVE'
        connection.execute('BEGIN EXCLUSIVE')
        # Assert the database file exists, and copy to backup path
        if exists(db.fpath):
            ut.copy(db.fpath, backup_filepath)
        else:
            raise IOError(
                'Could not backup the database as the URI does not exist: %r' % (uri,)
            )
        # Commit the transaction, releasing the lock
        connection.commit()
        # Close the connection
        connection.close()

    def optimize(db):
        # http://web.utk.edu/~jplyon/sqlite/SQLite_optimization_FAQ.html#pragma-cache_size
        # http://web.utk.edu/~jplyon/sqlite/SQLite_optimization_FAQ.html
        if VERBOSE_SQL:
            print('[sql] running sql pragma optimizions')
        # db.cur.execute('PRAGMA cache_size = 0;')
        # db.cur.execute('PRAGMA cache_size = 1024;')
        # db.cur.execute('PRAGMA page_size = 1024;')
        # print('[sql] running sql pragma optimizions')
        db.cur.execute('PRAGMA cache_size = 10000;')  # Default: 2000
        db.cur.execute('PRAGMA temp_store = MEMORY;')
        db.cur.execute('PRAGMA synchronous = OFF;')
        # db.cur.execute('PRAGMA synchronous = NORMAL;')
        # db.cur.execute('PRAGMA synchronous = FULL;')  # Default
        # db.cur.execute('PRAGMA parser_trace = OFF;')
        # db.cur.execute('PRAGMA busy_timeout = 1;')
        # db.cur.execute('PRAGMA default_cache_size = 0;')

    def shrink_memory(db):
        print('[sql] shrink_memory')
        db.connection.commit()
        db.cur.execute('PRAGMA shrink_memory;')
        db.connection.commit()

    def vacuum(db):
        print('[sql] vaccum')
        db.connection.commit()
        db.cur.execute('VACUUM;')
        db.connection.commit()

    def integrity(db):
        print('[sql] vaccum')
        db.connection.commit()
        db.cur.execute('PRAGMA integrity_check;')
        db.connection.commit()

    def squeeze(db):
        print('[sql] squeeze')
        db.shrink_memory()
        db.vacuum()

    # ==============
    # API INTERFACE
    # ==============

    def get_row_count(db, tblname):
        fmtdict = {
            'tblname': tblname,
        }
        operation_fmt = 'SELECT COUNT(*) FROM {tblname}'
        count = db._executeone_operation_fmt(operation_fmt, fmtdict)[0]
        return count

    def get_all_rowids(db, tblname, **kwargs):
        """ returns a list of all rowids from a table in ascending order """
        fmtdict = {
            'tblname': tblname,
        }
        operation_fmt = 'SELECT rowid FROM {tblname} ORDER BY rowid ASC'
        return db._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)

    def get_all_col_rows(db, tblname, colname):
        """ returns a list of all rowids from a table in ascending order """
        fmtdict = {
            'colname': colname,
            'tblname': tblname,
        }
        operation_fmt = 'SELECT {colname} FROM {tblname} ORDER BY rowid ASC'
        return db._executeone_operation_fmt(operation_fmt, fmtdict)

    def get_all_rowids_where(db, tblname, where_clause, params, **kwargs):
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
        return db._executeone_operation_fmt(operation_fmt, fmtdict, params, **kwargs)

    def check_rowid_exists(db, tablename, rowid_iter, eager=True, **kwargs):
        rowid_list1 = db.get(tablename, ('rowid',), rowid_iter)
        exists_list = [rowid is not None for rowid in rowid_list1]
        return exists_list

    def _add(db, tblname, colnames, params_iter, **kwargs):
        """ ADDER NOTE: use add_cleanly """
        fmtdict = {
            'tblname': tblname,
            'erotemes': ', '.join(['?'] * len(colnames)),
            'params': ',\n'.join(colnames),
        }
        operation_fmt = """
        INSERT INTO {tblname}(
        rowid,
        {params}
        ) VALUES (NULL, {erotemes})
        """
        rowid_list = db._executemany_operation_fmt(
            operation_fmt, fmtdict, params_iter=params_iter, **kwargs
        )
        return rowid_list

    def add_cleanly(
        db,
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
            >>> db = SQLDatabaseController(sqldb_fname=':memory:')
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
            print('[WARNING]: duplicate inputs to db.add_cleanly')
        # Flag each item that needs to added to the database
        needsadd_list = list(map(all, zip(isvalid_list, isunique_list, isnew_list)))
        # ADD_CLEANLY_3.1: EXIT IF CLEAN
        if not any(needsadd_list):
            return rowid_list_  # There is nothing to add. Return the rowids
        # ADD_CLEANLY_3.2: PERFORM DIRTY ADDITIONS
        dirty_params = ut.compress(params_list, needsadd_list)
        if ut.VERBOSE:
            print(
                '[sql] adding %r/%r new %s'
                % (len(dirty_params), len(params_list), tblname)
            )
        # Add any unadded parameters to the database
        try:
            db._add(tblname, colnames, dirty_params, **kwargs)
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

    def rows_exist(db, tblname, rowids):
        """
        Checks if rowids exist. Yields True if they do
        """
        operation = 'SELECT count(1) FROM {tblname} WHERE rowid=?'.format(tblname=tblname)
        for rowid in rowids:
            yield bool(db.cur.execute(operation, (rowid,)).fetchone()[0])

    def get_where_eq(
        db,
        tblname,
        colnames,
        params_iter,
        where_colnames,
        unpack_scalars=True,
        eager=True,
        op='AND',
        **kwargs,
    ):
        """ hacked in function for nicer templates

        unpack_scalars = True
        kwargs = {}

        Kwargs:
            verbose:
        """
        andwhere_clauses = [colname + '=?' for colname in where_colnames]
        logicop_ = ' %s ' % (op,)
        where_clause = logicop_.join(andwhere_clauses)
        return db.get_where(
            tblname,
            colnames,
            params_iter,
            where_clause,
            unpack_scalars=unpack_scalars,
            eager=eager,
            **kwargs,
        )

    def get_where_eq_set(
        db,
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
            print('Using sql_control.get_where_eq_set() for %r on %d params' % args)

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
        return db._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)

    @profile
    def get_where(
        db,
        tblname,
        colnames,
        params_iter,
        where_clause,
        unpack_scalars=True,
        eager=True,
        **kwargs,
    ):
        """
        """
        assert isinstance(colnames, tuple), 'colnames must be a tuple'

        if where_clause is None:
            operation_fmt = """
            SELECT {colnames}
            FROM {tblname}
            """
            fmtdict = {
                'tblname': tblname,
                'colnames': ', '.join(colnames),
            }
            val_list = db._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)
        else:
            operation_fmt = """
            SELECT {colnames}
            FROM {tblname}
            WHERE {where_clauses}
            """
            fmtdict = {
                'tblname': tblname,
                'colnames': ', '.join(colnames),
                'where_clauses': where_clause,
            }
            val_list = db._executemany_operation_fmt(
                operation_fmt,
                fmtdict,
                params_iter=params_iter,
                unpack_scalars=unpack_scalars,
                eager=eager,
                **kwargs,
            )
        return val_list

    def exists_where_eq(
        db,
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
        val_list = db._executemany_operation_fmt(
            operation_fmt,
            fmtdict,
            params_iter=params_iter,
            unpack_scalars=unpack_scalars,
            eager=eager,
            **kwargs,
        )
        return val_list

    def get_rowid_from_superkey(
        db, tblname, params_iter=None, superkey_colnames=None, **kwargs
    ):
        """ getter which uses the constrained superkeys instead of rowids """
        where_clause = ' AND '.join([colname + '=?' for colname in superkey_colnames])
        return db.get_where(tblname, ('rowid',), params_iter, where_clause, **kwargs)

    def get(
        db,
        tblname,
        colnames,
        id_iter=None,
        id_colname='rowid',
        eager=True,
        assume_unique=False,
        **kwargs,
    ):
        """ getter

        Args:
            tblname (str): table name to get from
            colnames (tuple of str): column names to grab from
            id_iter (iterable): iterable of search keys
            id_colname (str): column to be used as the search key (default: rowid)
            eager (bool): use eager evaluation
            unpack_scalars (bool): default True
            id_colname (bool): default False. Experimental feature that could result in a 10x speedup

        CommandLine:
            python -m dtool.sql_control get

        Ignore:
            tblname = 'annotations'
            colnames = ('name_rowid',)
            id_iter = aid_list
            #id_iter = id_iter[0:20]
            id_colname = 'rowid'
            eager = True
            db = ibs.db

            x1 = db.get(tblname, colnames, id_iter, assume_unique=True)
            x2 = db.get(tblname, colnames, id_iter, assume_unique=False)
            x1 == x2
            %timeit  db.get(tblname, colnames, id_iter, assume_unique=True)
            %timeit  db.get(tblname, colnames, id_iter, assume_unique=False)

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
        if VERBOSE_SQL:
            print(
                '[sql]'
                + ut.get_caller_name(list(range(1, 4)))
                + ' db.get(%r, %r, ...)' % (tblname, colnames,)
            )
        assert isinstance(colnames, tuple), 'must specify column names TUPLE to get from'
        # if isinstance(colnames, six.string_types):
        #    colnames = (colnames,)

        if (
            assume_unique
            and id_iter is not None
            and id_colname == 'rowid'
            and len(colnames) == 1
        ):
            id_iter = list(id_iter)
            operation_fmt = """
                SELECT {colnames}
                FROM {tblname}
                WHERE rowid in ({id_repr})
                ORDER BY rowid ASC
                """
            fmtdict = {
                'tblname': tblname,
                'colnames': ', '.join(colnames),
                'id_repr': ','.join(map(str, id_iter)),
            }
            operation = operation_fmt.format(**fmtdict)
            results = db.cur.execute(operation).fetchall()
            import numpy as np

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
                where_clause = id_colname + '=?'
                params_iter = [(_rowid,) for _rowid in id_iter]

            return db.get_where(
                tblname, colnames, params_iter, where_clause, eager=eager, **kwargs
            )

    def set(
        db,
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
            >>> val_iter = [9003, 9001, 9002]
            >>> orig_data = db.get('notch', colnames, id_iter=rowids)
            >>> db.set('notch', colnames, val_iter, id_iter=rowids)
            >>> new_data = db.get('notch', colnames, id_iter=rowids)
            >>> assert new_data == val_iter
            >>> assert new_data != orig_data
            >>> table.print_csv()
            >>> depc.clear_all()
        """
        assert isinstance(colnames, tuple)
        # if isinstance(colnames, six.string_types):
        #    colnames = (colnames,)
        val_list = list(val_iter)  # eager evaluation
        id_list = list(id_iter)  # eager evaluation

        if VERBOSE_SQL or (NOT_QUIET and VERYVERBOSE):
            print('[sql] SETTER: ' + ut.get_caller_name())
            print('[sql] * tblname=%r' % (tblname,))
            print('[sql] * val_list=%r' % (val_list,))
            print('[sql] * id_list=%r' % (id_list,))
            print('[sql] * id_colname=%r' % (id_colname,))

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
                        print(
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

        try:
            num_val = len(val_list)
            num_id = len(id_list)
            assert num_val == num_id, 'list inputs have different lengths'
        except AssertionError as ex:
            ut.printex(ex, key_list=['num_val', 'num_id'])
            raise
        fmtdict = {
            'tblname_str': tblname,
            'assign_str': ',\n'.join(['%s=?' % name for name in colnames]),
            'where_clause': (id_colname + '=?'),
        }
        operation_fmt = """
            UPDATE {tblname_str}
            SET {assign_str}
            WHERE {where_clause}
            """

        # TODO: The flattenize can be removed if we pass in val_lists instead
        params_iter = flattenize(list(zip(val_list, id_list)))

        # params_iter = list(zip(val_list, id_list))
        return db._executemany_operation_fmt(
            operation_fmt, fmtdict, params_iter=params_iter, **kwargs
        )

    def delete(db, tblname, id_list, id_colname='rowid', **kwargs):
        """
        deleter. USE delete_rowids instead
        """
        fmtdict = {
            'tblname': tblname,
            'rowid_str': (id_colname + '=?'),
        }
        operation_fmt = """
            DELETE
            FROM {tblname}
            WHERE {rowid_str}
            """
        params_iter = ((_rowid,) for _rowid in id_list)
        return db._executemany_operation_fmt(
            operation_fmt, fmtdict, params_iter=params_iter, **kwargs
        )

    def delete_rowids(db, tblname, rowid_list, **kwargs):
        """ deletes the the rows in rowid_list """
        fmtdict = {
            'tblname': tblname,
            'rowid_str': ('rowid=?'),
        }
        operation_fmt = """
            DELETE
            FROM {tblname}
            WHERE {rowid_str}
            """
        params_iter = ((_rowid,) for _rowid in rowid_list)
        return db._executemany_operation_fmt(
            operation_fmt, fmtdict, params_iter=params_iter, **kwargs
        )

    # ==============
    # CORE WRAPPERS
    # ==============

    def _executeone_operation_fmt(
        db, operation_fmt, fmtdict, params=None, eager=True, **kwargs
    ):
        if params is None:
            params = []
        operation = operation_fmt.format(**fmtdict)
        return db.executeone(operation, params, eager=eager, **kwargs)

    @profile
    def _executemany_operation_fmt(
        db,
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
            print('Dry Run')
            print(operation)
            return
        return db.executemany(
            operation, params_iter, unpack_scalars=unpack_scalars, eager=eager, **kwargs
        )

    # =========
    # SQLDB CORE
    # =========

    def executeone(db, operation, params=(), eager=True, verbose=VERBOSE_SQL):
        contextkw = dict(nInput=1, verbose=verbose)
        with SQLExecutionContext(db, operation, **contextkw) as context:
            try:
                result_iter = context.execute_and_generate_results(params)
                result_list = list(result_iter)
            except Exception as ex:
                ut.printex(ex, key_list=[(str, 'operation'), 'params'])
                # ut.sys.exit(1)
                raise
        return result_list

    # @ut.memprof
    @profile
    def executemany(
        db,
        operation,
        params_iter,
        verbose=VERBOSE_SQL,
        unpack_scalars=True,
        nInput=None,
        eager=True,
        keepwrap=False,
        showprog=False,
    ):
        """
        if unpack_scalars is True only a single result must be returned for each query.
        """
        # --- ARGS PREPROC ---
        # Aggresively compute iterator if the nInput is not given
        if nInput is None:
            if isinstance(params_iter, (list, tuple)):
                nInput = len(params_iter)
            else:
                if VERBOSE_SQL:
                    print(
                        '[sql!] WARNING: aggressive eval of params_iter because nInput=None'
                    )
                params_iter = list(params_iter)
                nInput = len(params_iter)
        else:
            if VERBOSE_SQL:
                print('[sql] Taking params_iter as iterator')

        # Do not compute executemany without params
        if nInput == 0:
            if VERBOSE_SQL:
                print(
                    '[sql!] WARNING: dont use executemany'
                    'with no params use executeone instead.'
                )
            return []
        # --- SQL EXECUTION ---
        contextkw = {
            'nInput': nInput,
            'start_transaction': True,
            'verbose': verbose,
            'keepwrap': keepwrap,
        }
        with SQLExecutionContext(db, operation, **contextkw) as context:
            if eager:
                if showprog:
                    if isinstance(showprog, six.string_types):
                        lbl = showprog
                    else:
                        lbl = 'sqlread'
                    prog = ut.ProgPartial(
                        adjust=True, length=nInput, freq=1, lbl=lbl, bs=True
                    )
                    params_iter = prog(params_iter)
                results_iter = [
                    list(context.execute_and_generate_results(params))
                    for params in params_iter
                ]
                if unpack_scalars:
                    # list of iterators
                    _unpacker_ = partial(_unpacker)
                    results_iter = list(map(_unpacker_, results_iter))
                # Eager evaluation
                results_list = list(results_iter)
            else:

                def _tmpgen(context):
                    # Temporary hack to turn off eager_evaluation
                    for params in params_iter:
                        # Eval results per query yeild per iter
                        results = list(context.execute_and_generate_results(params))
                        if unpack_scalars:
                            yield _unpacker(results)
                        else:
                            yield results

                results_list = _tmpgen(context)
        return results_list

    # def commit(db):
    #    db.connection.commit()

    def dump(db, file_=None, **kwargs):
        if file_ is None or isinstance(file_, six.string_types):
            db.dump_to_fpath(file_, **kwargs)
        else:
            db.dump_to_file(file_, **kwargs)

    def dump_to_fpath(db, dump_fpath, **kwargs):
        if dump_fpath is None:
            # Default filepath
            version_str = 'v' + db.get_db_version()
            if kwargs.get('schema_only', False):
                version_str += '.schema_only'
            dump_fname = db.fname + '.' + version_str + '.dump.txt'
            dump_fpath = join(db.dir_, dump_fname)
        with open(dump_fpath, 'w') as file_:
            db.dump_to_file(file_, **kwargs)

    def dump_to_string(db, **kwargs):
        # string_file = cStringIO.StringIO()
        string_file = cStringIO()
        db.dump_to_file(string_file, **kwargs)
        retstr = string_file.getvalue()
        return retstr

    def dump_to_file(
        db, file_, auto_commit=True, schema_only=False, include_metadata=True
    ):
        VERBOSE_SQL = True
        if VERBOSE_SQL:
            print('[sql.dump_to_file] file_=%r' % (file_,))
        if auto_commit:
            db.connection.commit()
            # db.commit(verbose=False)
        for line in db.connection.iterdump():
            if schema_only and line.startswith('INSERT'):
                if not include_metadata or 'metadata' not in line:
                    continue
            to_write = '%s\n' % line
            # Ensure python2 writes in bytes
            if six.PY2:
                if isinstance(to_write, unicode):  # NOQA
                    to_write = to_write.encode('utf8')
            try:
                file_.write(to_write)
            except UnicodeEncodeError:
                raise

    def dump_to_stdout(db, **kwargs):
        import sys

        file_ = sys.stdout
        kwargs['schema_only'] = kwargs.get('schema_only', True)
        db.dump(file_, **kwargs)

    def print_dbg_schema(db):
        print('\n\nCREATE'.join(db.dump_to_string(schema_only=True).split('CREATE')))

    # =========
    # SQLDB METADATA
    # =========
    def get_metadata_items(db):
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
        metadata_rowids = db.get_all_rowids(METADATA_TABLE)
        metadata_items = db.get(
            METADATA_TABLE, ('metadata_key', 'metadata_value'), metadata_rowids
        )
        return metadata_items

    def set_metadata_val(db, key, val):
        """
        key must be given as a repr-ed string
        """
        fmtkw = {'tablename': METADATA_TABLE, 'columns': 'metadata_key, metadata_value'}
        op_fmtstr = 'INSERT OR REPLACE INTO {tablename} ({columns}) VALUES (?, ?)'
        operation = op_fmtstr.format(**fmtkw)
        params = [key, val]
        db.executeone(operation, params, verbose=False)

    # def get_metadata_val(db, key, eval_=False, default=ut.NoParam):
    @profile
    def get_metadata_val(db, key, eval_=False, default=None):
        """
        val is the repr string unless eval_ is true
        """
        where_clause = 'metadata_key=?'
        colnames = ('metadata_value',)
        params_iter = [(key,)]
        vals = db.get_where(METADATA_TABLE, colnames, params_iter, where_clause)
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

    def add_column(db, tablename, colname, coltype):
        if VERBOSE_SQL:
            print(
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
        db.executeone(operation, [], verbose=False)

    def _make_add_table_sqlstr(
        db, tablename=None, coldef_list=None, sep=' ', **metadata_keyval
    ):
        r"""
        TODO: Foreign keys and indexed columns

        Args:
            tablename (None): (default = None)
            coldef_list (list): (default = None)

        Returns:
            str: operation

        CommandLine:
            python -m dtool.sql_control _make_add_table_sqlstr

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.sql_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> db = depc[tablename].db
            >>> autogen_dict = db.get_table_autogen_dict(tablename)
            >>> coldef_list = autogen_dict['coldef_list']
            >>> operation = db._make_add_table_sqlstr(tablename, coldef_list)
            >>> print(operation)
        """
        if len(coldef_list) == 0 or coldef_list is None:
            raise AssertionError('table %s is not given any columns' % (tablename,))
        bad_kwargs = set(metadata_keyval.keys()) - set(db.table_metadata_keys)
        assert (
            len(bad_kwargs) == 0
        ), 'keyword args specified that are not metadata keys=%r' % (bad_kwargs,)
        assert tablename is not None, 'tablename must be given'
        if ut.DEBUG2:
            print('[sql] schema ensuring tablename=%r' % tablename)
        if ut.VERBOSE:
            print('')
            _args = [tablename, coldef_list]
            print(ut.func_str(db.add_table, _args, metadata_keyval))
            print('')
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        constraint_list = []
        superkeys = metadata_keyval.get('superkeys', None)
        try:
            has_superkeys = superkeys is not None and len(superkeys) > 0
            if has_superkeys:
                # Add in superkeys to constraints
                constraint_fmtstr = 'CONSTRAINT superkey UNIQUE ({colnames_str})'
                assert isinstance(
                    superkeys, list
                ), 'must be list got %r, superkeys=%r' % (type(superkeys), superkeys)
                for superkey_colnames in superkeys:
                    assert isinstance(
                        superkey_colnames, tuple
                    ), 'must be list of tuples got list of %r' % (
                        type(superkey_colnames,)
                    )
                    colnames_str = ','.join(superkey_colnames)
                    unique_constraint = constraint_fmtstr.format(
                        colnames_str=colnames_str
                    )
                    constraint_list.append(unique_constraint)
                constraint_list = ut.unique_ordered(constraint_list)
        except Exception as ex:
            ut.printex(ex, keys=locals().keys())
            raise

        # ASSERT VALID TYPES
        for name, type_ in coldef_list:
            assert (
                isinstance(name, six.string_types) and len(name) > 0
            ), 'cannot have empty name. name=%r, type_=%r' % (name, type_)
            assert (
                isinstance(type_, six.string_types) and len(type_) > 0
            ), 'cannot have empty type. name=%r, type_=%r' % (name, type_)

        body_list = []
        foreign_body = []
        # False
        for (name, type_) in coldef_list:
            if db.USE_FOREIGN_HACK and name.endswith('_rowid'):
                # HACK THAT ONLY WORKS WITH IBEIS STRUCTURE
                foreign_table = name[: -len('_rowid')]
                if foreign_table != tablename and foreign_table + 's' != tablename:
                    if False:
                        foreign_body.append(
                            'FOREIGN KEY (%s) REFERENCES %s (%s)'
                            % (name, foreign_table, name)
                        )
                        coldef_line = '%s %s' % (name, type_)
                    else:
                        # alternative way to do this in SQLITE 3
                        coldef_line = '%s %s REFERENCES %s' % (
                            name,
                            type_,
                            foreign_table,
                        )
                    body_list.append(coldef_line)
                else:
                    coldef_line = '%s %s' % (name, type_)
                    body_list.append(coldef_line)
            else:
                # else:
                coldef_line = '%s %s' % (name, type_)
                body_list.append(coldef_line)

        if db.USE_FOREIGN_HACK:
            body_list += foreign_body

        sep_ = ',' + sep
        table_body = sep_.join(body_list + constraint_list)
        fmtkw = {
            'table_body': table_body,
            'tablename': tablename,
            'sep': sep,
        }
        if sep == ' ':
            op_fmtstr = 'CREATE TABLE IF NOT EXISTS {tablename} ({table_body})'
        else:
            op_fmtstr = 'CREATE TABLE IF NOT EXISTS {tablename} ({sep}{table_body}{sep})'
        operation = op_fmtstr.format(**fmtkw)
        return operation

    def add_table(db, tablename=None, coldef_list=None, **metadata_keyval):
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
        operation = db._make_add_table_sqlstr(tablename, coldef_list, **metadata_keyval)
        db.executeone(operation, [], verbose=False)

        # Handle table metdata
        for suffix in db.table_metadata_keys:
            if suffix in metadata_keyval and metadata_keyval[suffix] is not None:
                val = metadata_keyval[suffix]
                if suffix in ['docstr']:
                    db.set_metadata_val(tablename + '_' + suffix, val)
                else:
                    db.set_metadata_val(tablename + '_' + suffix, repr(val))
        if db._tablenames is not None:
            db._tablenames.add(tablename)

    def modify_table(
        db,
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
            print('[sql] schema modifying tablename=%r' % tablename)
            print(
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

        coldef_list = db.get_coldef_list(tablename)
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
                        print(
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
        for suffix in db.table_metadata_keys:
            if suffix not in metadata_keyval2 or metadata_keyval2[suffix] is None:
                val = db.get_metadata_val(tablename_orig + '_' + suffix, eval_=True)
                metadata_keyval2[suffix] = val

        db.add_table(tablename_temp, coldef_list, **metadata_keyval2)

        # Copy data
        src_list = []
        dst_list = []

        for name in colname_original_list:
            if name in colname_dict.keys():
                src_list.append(name)
                dst_list.append(colname_dict[name])

        if len(src_list) > 0:
            data_list_ = db.get(tablename, tuple(src_list))
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

        db.add_cleanly(tablename_temp, dst_list, data_list, get_rowid_from_superkey)
        if tablename_new is None:
            # Drop original table
            db.drop_table(tablename)
            # Rename temp table to original table name
            db.rename_table(tablename_temp, tablename)
        else:
            # Rename new table to new name
            db.rename_table(tablename_temp, tablename_new)

    def rename_table(db, tablename_old, tablename_new):
        if ut.VERBOSE:
            print(
                '[sql] schema renaming tablename=%r -> %r'
                % (tablename_old, tablename_new)
            )
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        fmtkw = {
            'tablename_old': tablename_old,
            'tablename_new': tablename_new,
        }
        op_fmtstr = 'ALTER TABLE {tablename_old} RENAME TO {tablename_new}'
        operation = op_fmtstr.format(**fmtkw)
        db.executeone(operation, [], verbose=False)

        # Rename table's metadata
        key_old_list = [tablename_old + '_' + suffix for suffix in db.table_metadata_keys]
        key_new_list = [tablename_new + '_' + suffix for suffix in db.table_metadata_keys]
        id_iter = [(key,) for key in key_old_list]
        val_iter = [(key,) for key in key_new_list]
        colnames = ('metadata_key',)
        # print('Setting metadata_key from %s to %s' % (ut.repr2(id_iter), ut.repr2(val_iter)))
        db.set(METADATA_TABLE, colnames, val_iter, id_iter, id_colname='metadata_key')

    def drop_table(db, tablename):
        if VERBOSE_SQL:
            print('[sql] schema dropping tablename=%r' % tablename)
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        fmtkw = {
            'tablename': tablename,
        }
        op_fmtstr = 'DROP TABLE IF EXISTS {tablename}'
        operation = op_fmtstr.format(**fmtkw)
        db.executeone(operation, [], verbose=False)

        # Delete table's metadata
        key_list = [tablename + '_' + suffix for suffix in db.table_metadata_keys]
        db.delete(METADATA_TABLE, key_list, id_colname='metadata_key')

    def drop_all_tables(db):
        """
        DELETES ALL INFO IN TABLE
        """
        db._tablenames = None
        for tablename in db.get_table_names():
            if tablename != 'metadata':
                db.drop_table(tablename)
        db._tablenames = None

    # ==============
    # CONVINENCE
    # ==============

    def dump_tables_to_csv(db, dump_dir=None):
        """ Convenience: Dumps all csv database files to disk """
        if dump_dir is None:
            dump_dir = join(db.dir_, 'CSV_DUMP')
        ut.ensuredir(dump_dir)
        for tablename in db.get_table_names():
            table_fname = tablename + '.csv'
            table_fpath = join(dump_dir, table_fname)
            table_csv = db.get_table_csv(tablename)
            ut.writeto(table_fpath, table_csv)

    def get_schema_current_autogeneration_str(db, autogen_cmd=''):
        """ Convenience: Autogenerates the most up-to-date database schema

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
        db_version_current = db.get_db_version()
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
        # line_list.append('from __future__ import absolute_import, division, print_function')
        line_list.append(
            'from __future__ import absolute_import, division, print_function, unicode_literals'
        )
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
        for tablename in sorted(db.get_table_names()):
            if first:
                first = False
            else:
                line_list.append('%s' % '')
            line_list += db.get_table_autogen_str(tablename)
            pass

        line_list.append('')
        return '\n'.join(line_list)

    def get_table_constraints(db, tablename):
        """
        TODO: use coldef_list with table_autogen_dict instead
        """
        constraint = db.get_metadata_val(tablename + '_constraint', default=None)
        return None if constraint is None else constraint.split(';')

    def get_coldef_list(db, tablename):
        """
        Returns:
            list of (str, str) : each tuple is (col_name, col_type)
        """
        column_list = db.get_columns(tablename)

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
    def get_table_autogen_dict(db, tablename):
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
            >>> db = SQLDatabaseController(sqldb_fname=':memory:')
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
        autogen_dict['coldef_list'] = db.get_coldef_list(tablename)
        autogen_dict['docstr'] = db.get_table_docstr(tablename)
        autogen_dict['superkeys'] = db.get_table_superkey_colnames(tablename)
        autogen_dict['dependson'] = db.get_metadata_val(
            tablename + '_dependson', eval_=True, default=None
        )
        return autogen_dict

    def get_table_autogen_str(db, tablename):
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
            >>> db = SQLDatabaseController(sqldb_fname=':memory:')
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
        autogen_dict = db.get_table_autogen_dict(tablename)
        coldef_list = autogen_dict['coldef_list']
        max_colsize = max(32, 2 + max(map(len, ut.take_column(coldef_list, 0))))
        # for column, colname_repr in zip(column_list, colnamerepr_list):
        for col_name, col_type in coldef_list:
            name_part = ('%s,' % ut.repr2(col_name)).ljust(max_colsize)
            type_part = ut.repr2(col_type)
            line_list.append(tab2 + '(%s%s),' % (name_part, type_part,))
        line_list.append(tab1 + '],')
        superkeys = db.get_table_superkey_colnames(tablename)
        docstr = db.get_table_docstr(tablename)
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
        for suffix in db.table_metadata_keys:
            if suffix in specially_handled_table_metakeys:
                continue
            key = tablename + '_' + suffix
            val = db.get_metadata_val(key, eval_=True, default=None)
            print(key)
            if val is not None:
                line_list.append(tab2 + '%s=%s,' % (suffix, ut.repr2(val)))
        # FIXME: are we depricating dependsmap?
        dependsmap = db.get_metadata_val(
            tablename + '_dependsmap', eval_=True, default=None
        )
        if dependsmap is not None:
            _dictstr = ut.indent(ut.repr2(dependsmap, nl=1), tab2)
            depends_map_dictstr = ut.align(_dictstr.lstrip(' '), ':')
            # hack for formatting
            depends_map_dictstr = depends_map_dictstr.replace(tab1 + '}', '}')
            line_list.append(tab2 + 'dependsmap=%s,' % (depends_map_dictstr,))
        line_list.append(tab1 + ')')
        return line_list

    def dump_schema(db):
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
            for tablename in sorted(db.get_table_names()):
                file_.write(tablename + '\n')
                column_list = db.get_columns(tablename)
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

    def get_table_names(db, lazy=False):
        """ Conveinience: """
        if not lazy or db._tablenames is None:
            db.cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tablename_list = db.cur.fetchall()
            db._tablenames = {str(tablename[0]) for tablename in tablename_list}
        return db._tablenames

    @property
    def tablenames(db):
        return db.get_table_names()

    def has_table(db, tablename, colnames=None, lazy=True):
        """ checks if a table exists """
        # if not lazy or db._tablenames is None:
        return tablename in db.get_table_names(lazy=lazy)

    @profile
    def get_table_superkey_colnames(db, tablename):
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
        assert tablename in db.get_table_names(
            lazy=True
        ), 'tablename=%r is not a part of this database' % (tablename,)
        superkey_colnames_list_repr = db.get_metadata_val(
            tablename + '_superkeys', default=None
        )
        # These asserts might not be valid, but in that case this function needs
        # to be rewritten under a different name
        # assert len(superkeys) == 1, 'INVALID DEVELOPER ASSUMPTION IN
        # SQLCONTROLLER. MORE THAN 1 SUPERKEY'
        if superkey_colnames_list_repr is None:
            superkeys = []
            pass
        else:
            # superkey_colnames = superkey_colnames_str.split(';')
            # with ut.EmbedOnException():
            if superkey_colnames_list_repr.find(';') > -1:
                # SHOW NOT HAPPEN
                # hack for old metadata superkey_val format
                superkeys = [tuple(map(str, superkey_colnames_list_repr.split(';')))]
            else:
                # new evalable format
                locals_ = {}
                globals_ = {}
                superkeys = eval(superkey_colnames_list_repr, globals_, locals_)
        # superkeys = [
        #    None if superkey_colname is None else str(superkey_colname)
        #    for superkey_colname in superkey_colnames
        # ]
        superkeys = list(map(tuple, superkeys))
        return superkeys

    def get_table_primarykey_colnames(db, tablename):
        columns = db.get_columns(tablename)
        primarykey_colnames = tuple(
            [name for (column_id, name, type_, notnull, dflt_value, pk,) in columns if pk]
        )
        return primarykey_colnames

    def get_table_docstr(db, tablename):
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
        docstr = db.get_metadata_val(tablename + '_docstr')
        # where_clause = 'metadata_key=?'
        # colnames = ('metadata_value',)
        # data = [(tablename + '_docstr',)]
        # docstr = db.get_where(const.METADATA_TABLE, colnames, data, where_clause)[0]
        return docstr

    def get_columns(db, tablename):
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
        db.cur.execute('SELECT 1 FROM ' + tablename + ' LIMIT 1')
        db.cur.execute("PRAGMA TABLE_INFO('" + tablename + "')")
        colinfo_list = db.cur.fetchall()
        colrichinfo_list = [SQLColumnRichInfo(*colinfo) for colinfo in colinfo_list]
        return colrichinfo_list

    def get_column_names(db, tablename):
        """ Conveinience: Returns the sql tablename columns """
        column_list = db.get_columns(tablename)
        column_names = ut.lmap(six.text_type, ut.take_column(column_list, 1))
        return column_names

    def get_column(db, tablename, name):
        """ Conveinience: """
        _table, (_column,) = sanitize_sql(db, tablename, (name,))
        column_vals = db.executeone(
            operation="""
            SELECT %s
            FROM %s
            ORDER BY rowid ASC
            """
            % (_column, _table)
        )
        return column_vals

    def get_table_as_pandas(db, tablename, rowids=None, columns=None, exclude_columns=[]):
        """
        aid = 30
        db = ibs.staging
        rowids = ut.flatten(ibs.get_review_rowids_from_single([aid]))
        tablename = 'reviews'
        exclude_columns = 'review_user_confidence review_user_identity'.split(' ')
        print(db.get_table_as_pandas(tablename, rowids, exclude_columns=exclude_columns))

        db = ibs.db
        rowids = ut.flatten(ibs.get_annotmatch_rowids_from_aid([aid]))
        tablename = 'annotmatch'
        exclude_columns = 'annotmatch_confidence annotmatch_posixtime_modified annotmatch_reviewer'.split(' ')
        print(db.get_table_as_pandas(tablename, rowids, exclude_columns=exclude_columns))
        """
        if rowids is None:
            rowids = db.get_all_rowids(tablename)
        column_list, column_names = db.get_table_column_data(
            tablename, rowids=rowids, columns=columns, exclude_columns=exclude_columns
        )
        import pandas as pd

        index = pd.Index(rowids, name='rowid')
        df = pd.DataFrame(ut.dzip(column_names, column_list), index=index)
        return df

    def get_table_column_data(
        db, tablename, columns=None, exclude_columns=[], rowids=None
    ):
        """
        Grabs a table of information

        CommandLine:
            python -m dtool.sql_control --test-get_table_column_data

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
            all_column_names = db.get_column_names(tablename)
            column_names = ut.setdiff(all_column_names, exclude_columns)
        else:
            column_names = columns
        if rowids is not None:
            column_list = [
                db.get(tablename, (name,), rowids, unpack_scalars=True)
                for name in column_names
            ]
        else:
            column_list = [db.get_column(tablename, name) for name in column_names]
        return column_list, column_names

    def make_json_table_definition(db, tablename):
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
        new_transferdata = db.get_table_new_transferdata(tablename)
        (
            column_list,
            column_names,
            extern_colx_list,
            extern_superkey_colname_list,
            extern_superkey_colval_list,
            extern_tablename_list,
            extern_primarycolnames_list,
        ) = new_transferdata
        dependsmap = db.get_metadata_val(
            tablename + '_dependsmap', eval_=True, default=None
        )
        # dependson = db.get_metadata_val(tablename + '_dependson', eval_=True, default=None)

        richcolinfo_list = db.get_columns(tablename)
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
                    _deptablecols = db.get_columns(val[0])
                    superkey = val[2]
                    assert len(superkey) == 1, 'unhandled'
                    colinfo = {_.name: _ for _ in _deptablecols}[superkey[0]]
                    table_dict_def[superkey[0]] = colinfo.type_
        # json_def_str = ut.repr2(table_dict_def, aligned=True)
        return table_dict_def
        # table_obj_def =

    def get_table_new_transferdata(db, tablename, exclude_columns=[]):
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
            >>> dependsmap = db.get_metadata_val(tablename + '_dependsmap', eval_=True, default=None)
            >>> print('tablename = %r' % (tablename,))
            >>> print('colnames = ' + ut.repr2(column_names))
            >>> print('extern_colx_list = ' + ut.repr2(extern_colx_list))
            >>> print('extern_superkey_colname_list = ' + ut.repr2(extern_superkey_colname_list))
            >>> print('dependsmap = %s' % (ut.repr2(dependsmap, nl=True),))
            >>> print('L___')
            >>> tablename = ibs.const.ANNOTATION_TABLE
            >>> new_transferdata = db.get_table_new_transferdata(tablename)
            >>> column_list, column_names, extern_colx_list, extern_superkey_colname_list, extern_superkey_colval_list, extern_tablename_list, extern_primarycolnames_list = new_transferdata
            >>> dependsmap = db.get_metadata_val(tablename + '_dependsmap', eval_=True, default=None)
            >>> print('tablename = %r' % (tablename,))
            >>> print('colnames = ' + ut.repr2(column_names))
            >>> print('extern_colx_list = ' + ut.repr2(extern_colx_list))
            >>> print('extern_superkey_colname_list = ' + ut.repr2(extern_superkey_colname_list))
            >>> print('dependsmap = %s' % (ut.repr2(dependsmap, nl=True),))
            >>> print('L___')
        """
        import utool

        with utool.embed_on_exception_context:
            all_column_names = db.get_column_names(tablename)
            isvalid_list = [name not in exclude_columns for name in all_column_names]
            column_names = ut.compress(all_column_names, isvalid_list)
            column_list = [
                db.get_column(tablename, name)
                for name in column_names
                if name not in exclude_columns
            ]

            extern_colx_list = []
            extern_tablename_list = []
            extern_superkey_colname_list = []
            extern_superkey_colval_list = []
            extern_primarycolnames_list = []
            dependsmap = db.get_metadata_val(
                tablename + '_dependsmap', eval_=True, default=None
            )
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
                        extern_primary_colnames = db.get_table_primarykey_colnames(
                            extern_tablename
                        )
                    if extern_superkey_colnames is None:

                        def get_standard_superkey_colnames(tablename_):
                            try:
                                # FIXME: Rectify duplicate code
                                superkeys = db.get_table_superkey_colnames(tablename_)
                                if len(superkeys) > 1:
                                    # primary_superkey =
                                    # db.get_metadata_val(tablename_ +
                                    #                    '_primary_superkey',
                                    #                    eval_=True)
                                    primary_superkey = db.get_metadata_val(
                                        tablename_ + '_primary_superkey', eval_=True
                                    )
                                    db.get_table_superkey_colnames('contributors')
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
                                    print(db.get_table_csv_header(tablename_))
                                    db.print_table_csv(
                                        'metadata', exclude_columns=['metadata_value']
                                    )
                                    # Execute hack to fix contributor tables
                                    if tablename_ == 'contributors':
                                        # hack to fix contributors table
                                        constraint_str = db.get_metadata_val(
                                            tablename_ + '_constraint'
                                        )
                                        parse_result = parse.parse(
                                            'CONSTRAINT superkey UNIQUE ({superkey})',
                                            constraint_str,
                                        )
                                        superkey = parse_result['superkey']
                                        assert (
                                            superkey == 'contributor_tag'
                                        ), 'hack failed1'
                                        assert None is db.get_metadata_val(
                                            'contributors_superkey'
                                        ), 'hack failed2'
                                        if True:
                                            db.set_metadata_val(
                                                'contributors_superkeys',
                                                "[('" + superkey + "',)]",
                                            )
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
                    superkey_column = db.get(
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

    def merge_databases_new(db, db_src, ignore_tables=None, rowid_subsets=None):
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
        version_dst = db.get_metadata_val('database_version')
        version_src = db_src.get_metadata_val('database_version')
        assert (
            version_src == version_dst
        ), 'cannot merge databases that have different versions'
        # Get merge tablenames
        all_tablename_list = db.get_table_names()
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
            db.get_metadata_val(tablename + '_dependsmap', eval_=True, default=None)
            for tablename in tablename_list
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
                print('\n[sqlmerge] Merging tablename=%r' % (tablename,))
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
                print(
                    ' * filtered number of rows from %d to %d.'
                    % (len(valid_rowids), len(valid_old_rowid_list))
                )
            else:
                print(' * no filtering requested')
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
                    print(
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
                            print('[sqlmerge] +--')
                            print(
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
                            print(
                                '[sqlmerge] * resolving %r via %r -> %r'
                                % (
                                    source_colname,
                                    extern_superkey_colname,
                                    extern_primarycolname,
                                )
                            )
                    _params_iter = list(zip(extern_superkey_colval))
                    new_extern_rowids = db.get_rowid_from_superkey(
                        extern_tablename,
                        _params_iter,
                        superkey_colnames=extern_superkey_colname,
                    )
                    num_Nones = sum(ut.flag_None_items(new_extern_rowids))
                    if verbose:
                        print('[sqlmerge] * there were %d none items' % (num_Nones,))
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
            superkey_colnames_list = db.get_table_superkey_colnames(tablename)
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
                primary_superkey = db.get_metadata_val(
                    tablename + '_primary_superkey', eval_=True, default=None
                )
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
                # print('superkey_paramxs_list = %r' % (superkey_paramxs_list, ))
                # print('superkey_colnames_list = %r' % (superkey_colnames_list, ))
                # raise ValueError('Cannot merge %r' % (tablename, ))

            params_iter = list(zip(*modified_column_list_))

            def get_rowid_from_superkey(*superkey_column_list):
                superkey_params_iter = zip(*superkey_column_list)
                rowid = db.get_rowid_from_superkey(
                    tablename, superkey_params_iter, superkey_colnames=superkey_colnames
                )
                return rowid

            # TODO: allow for cetrain databases to take precidence over another
            # basically allow insert or replace
            new_rowid_list = db.add_cleanly(
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

    def get_table_csv(db, tablename, exclude_columns=[], rowids=None, truncate=False):
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
        column_list, column_names = db.get_table_column_data(
            tablename, exclude_columns=exclude_columns, rowids=rowids
        )
        # remove column prefix for more compact csvs
        column_lbls = [name.replace(tablename[:-1] + '_', '') for name in column_names]
        header = db.get_table_csv_header(tablename)
        # truncate = True
        if truncate:
            column_list = [
                [ut.trunc_repr(col) for col in column] for column in column_list
            ]

        csv_table = ut.make_csv_table(column_list, column_lbls, header, comma_repl=';')
        csv_table = ut.ensure_unicode(csv_table)
        # csv_table = ut.make_csv_table(column_list, column_lbls, header, comma_repl='<comma>')
        return csv_table

    def print_table_csv(db, tablename, exclude_columns=[], truncate=False):
        print(
            db.get_table_csv(
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

    def print_schema(db):
        for tablename in db.get_table_names():
            print(db.get_table_csv_header(tablename) + '\n')

    def view_db_in_external_reader(db):
        import os

        known_readers = ['sqlitebrowser', 'sqliteman']
        sqlite3_reader = known_readers[0]
        sqlite3_db_fpath = db.get_fpath()
        os.system(sqlite3_reader + ' ' + sqlite3_db_fpath)
        # ut.cmd(sqlite3_reader, sqlite3_db_fpath)
        pass

    def set_db_version(db, version):
        # Do things properly, get the metadata_rowid (best because we want to assert anyway)
        metadata_key_list = ['database_version']
        params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
        where_clause = 'metadata_key=?'
        # list of relationships for each image
        metadata_rowid_list = db.get_where(
            METADATA_TABLE,
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
        db.set(METADATA_TABLE, ('metadata_value',), val_list, id_iter)

    def get_sql_version(db):
        """ Conveinience """
        db.cur.execute('SELECT sqlite_version()')
        sql_version = db.cur.fetchone()
        print('[sql] SELECT sqlite_version = %r' % (sql_version,))
        # The version number sqlite3 module. NOT the version of SQLite library.
        print('[sql] sqlite3.version = %r' % (lite.version,))
        # The version of the SQLite library
        print('[sql] sqlite3.sqlite_version = %r' % (lite.sqlite_version,))
        return sql_version

    def __getitem__(db, key):
        if not db.has_table(key):
            raise KeyError('Choose on of: ' + str(db.tablenames))
        table = SQLTable(db, name=key)
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.sql_control
        python -m dtool.sql_control --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
