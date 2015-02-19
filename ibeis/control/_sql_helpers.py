from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import re
from . import __SQLITE3__ as lite
from os.path import split, splitext, join, exists
#import six
import logging
import datetime
from ibeis import params
from ibeis import constants as const
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[sql-helpers]')

# =======================
# Helper Functions
# =======================
PRINT_SQL = utool.get_argflag(('--print-sql', '--verbose-sql'))
#AUTODUMP = utool.get_argflag('--auto-dump')
NOT_QUIET = not (utool.QUIET or utool.get_argflag('--quiet-sql'))


def _results_gen(cur, get_last_id=False):
    """ HELPER - Returns as many results as there are.
    Careful. Overwrites the results once you call it.
    Basically: Dont call this twice.
    """
    if get_last_id:
        # The sqlite3_last_insert_rowid(D) interface returns the
        # <b> rowid of the most recent successful INSERT </b>
        # into a rowid table in D
        cur.execute('SELECT last_insert_rowid()', ())
    # Wraping fetchone in a generator for some pretty tight calls.
    while True:
        result = cur.fetchone()
        if not result:
            raise StopIteration()
        else:
            # Results are always returned wraped in a tuple
            result_ = result[0] if len(result) == 1 else result
            #if get_last_id and result == 0:
            #    result = None
            yield result_


def _unpacker(results_):
    """ HELPER: Unpacks results if unpack_scalars is True """
    results = None if len(results_) == 0 else results_[0]
    assert len(results_) < 2, 'throwing away results! { %r }' % (results_)
    return results


def compare_string_versions(a, b):
    a = map(int, a.strip().split('.'))
    b = map(int, b.strip().split('.'))
    while len(a) < 3:
        a.append(0)
    while len(b) < 3:
        b.append(0)
    if a[0] < b[0]:
        return -1
    elif a[0] > b[0]:
        return 1
    else:
        if a[1] < b[1]:
            return -1
        elif a[1] > b[1]:
            return 1
        else:
            if a[2] < b[2]:
                return -1
            elif a[2] > b[2]:
                return 1
    # return 0 - identical
    raise AssertionError('[!update_schema_version] Two version numbers are the same along the update path')


# =========================
# Database Backup Functions
# =========================


def ensure_daily_database_backup(db_dir, db_fname, backup_dir, max_keep=60):
    # Keep 60 days worth of database backups
    database_backup(db_dir, db_fname, backup_dir, max_keep=max_keep, manual=False)


def database_backup(db_dir, db_fname, backup_dir, max_keep=60, manual=True):
    # Keep 60 days worth of database backups
    fname, ext = splitext(db_fname)
    src_fpath = join(db_dir, db_fname)
    now = datetime.datetime.now()
    if manual:
        now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
    else:
        now_str = now.strftime('%Y_%m_%d_00_00_00')
    #dst_fpath = join(backup_dir, '%s_backup_%s%s' % (fname, now_str, ext))
    dst_fname = ''.join((fname, '_backup_', now_str, ext))
    dst_fpath = join(backup_dir, dst_fname)
    if exists(src_fpath) and not exists(dst_fpath):
        print('[ensure_daily_database_backup] Daily backup of database: %r -> %r' % (src_fpath, dst_fpath, ))
        utool.copy(src_fpath, dst_fpath)
        # Clean-up old database backups
        path_list = sorted(utool.glob(backup_dir, '*%s' % ext))
        if path_list > max_keep:
            path_delete_list = path_list[:-1 * max_keep]
            for path_delete in path_delete_list:
                print('[ensure_daily_database_backup] Deleting old backup %r' % path_delete)
                utool.remove_file(path_delete, verbose=False)


# ========================
# Schema Updater Functions
# ========================


@profile
def ensure_correct_version(ibs, db, version_expected, schema_spec,
                           dobackup=True, autogenerate=False,
                           verbose=ut.NOT_QUIET):
    """
    FIXME: AN SQL HELPER FUNCTION SHOULD BE AGNOSTIC TO CONTROLER OBJECTS

    ensure_correct_version

    Args:
        ibs (IBEISController):
        db (SQLController):
        version_expected (str): version you want to be at
        schema_spec (module): schema module
        dobackup (bool):
        autogenerate (bool):

    Example:
        >>> from ibeis.control._sql_helpers import *  # NOQA
        >>> ibs = '?'
        >>> db = ibs.db
        >>> version_expected = ibs.db_version_expected
        >>> schema_spec = DB_SCHEMA
        >>> dobackup = True
        >>> autogenerate = False
        >>> result = ensure_correct_version(ibs, db, version_expected, schema_spec, dobackup, autogenerate)
        >>> print(result)

    Args:
        schema_spec (module): module of schema specifications
    """

    want_base_version = version_expected == const.BASE_DATABASE_VERSION
    if want_base_version:
        print('[SQL_] base version expected... returning')
        # Nothing to do. autogenerated file is pointless
        return
    db_versions = schema_spec.VALID_VERSIONS
    version     = db.get_db_version()
    # NEW DATABASE CONDITION
    force_incremental = params.args.force_incremental_db_update
    is_base_version = (version == const.BASE_DATABASE_VERSION)
    can_skip = is_base_version and not force_incremental
    # <DEBUG>
    #if utool.get_flag('--verbsql') or utool.VERBOSE or True:
    #    key_list = locals().keys()
    #    keystr_list = sorted(ut.parse_locals_keylist(locals(), key_list))
    #    print('KEYLIST:' + ut.indentjoin(keystr_list, '\n * '))
    # </DEBUG>
    #+-----------------------------------
    # SKIP TO CURRENT VERSION IF POSSIBLE
    #+-----------------------------------
    if can_skip:
        current_schema_exists = (schema_spec.UPDATE_CURRENT is not None and
                                 schema_spec.VERSION_CURRENT is not None)
        if current_schema_exists:
            # check to see if more than the metadata table exists
            is_newdb = db.get_table_names() == [const.METADATA_TABLE]
            current_schema_compatible = (
                is_newdb and schema_spec.VERSION_CURRENT <= version_expected)
            if current_schema_compatible:
                # Since this is a new database, we do not have to worry about backinng up the
                # current database.  The subsequent update functions (if needed) will handle
                # this for us.
                if verbose:
                    print('[_SQL] New database and a current schema found')
                schema_spec.UPDATE_CURRENT(db, ibs=ibs)
                db.set_db_version(schema_spec.VERSION_CURRENT)
                if verbose:
                    print('[_SQL] Database version updated (skipped) to %r ' % (schema_spec.VERSION_CURRENT))
            else:
                print('[_SQL] Current database is not compatible, updating incrementally...')
        else:
            print('[_SQL] New database but current version not exported, updating incrementally...')
    #+--------------------------------------
    # INCREMENTAL UPDATE TO EXPECTED VERSION
    #+--------------------------------------
    # Check version again for sanity's sake, update if exported current is behind expected
    version = db.get_db_version()
    if verbose:
        print('[_SQL.%s] Database version: %r | Expected version: %r ' %
                (ut.get_caller_name(), version, version_expected))
    if version < version_expected:
        print('[_SQL] Database version behind, updating...')
        update_schema_version(ibs, db, db_versions, version, version_expected,
                              dobackup=dobackup)
        db.set_db_version(version_expected)
        print('[_SQL] Database version updated (incrementally) to %r' %
                   (version_expected))
    elif version > version_expected:
        msg = (('[_SQL] ERROR: '
                'Expected database version behind. expected: %r. got: %r') %
               (version_expected, version))
        raise AssertionError(msg)
    #+--------------------------------------
    # AUTOGENERATE CURRENT SCHEMA
    #+--------------------------------------
    if autogenerate:
        if schema_spec.UPDATE_CURRENT > version_expected:
            print('[_SQL] WARNING: autogenerated version is going backwards')
        print('[_SQL] DUMPING  AUTOGENERATED CURRENT SCHEMA')
        # Auto-generate the version skip schema file
        schema_spec_dir, schema_spec_filename = split(schema_spec.__file__)
        schema_spec_filename = splitext(schema_spec_filename)[0]
        # HACK TO GET AUTOGEN COMMAND
        # FIXME: Make this autogen command a bit more sane
        # and not completely coupled with ibeis
        autogen_cmd = 'python -m ibeis.control.%s --test-test_%s --force-incremental-db-update --dump-autogen-schema' % (schema_spec_filename, schema_spec_filename.lower())
        db.dump_schema_current_autogeneration(schema_spec_dir, '%s_CURRENT.py' % schema_spec_filename, autogen_cmd)


@profile
def update_schema_version(ibs, db, db_versions, version, version_target,
                          dobackup=True, clearbackup=False):
    """
    version_target = version_expected
    clearbackup = False
    FIXME: AN SQL HELPER FUNCTION SHOULD BE AGNOSTIC TO CONTROLER OBJECTS
    """
    db_fpath = db.fpath
    if dobackup:
        db_dpath, db_fname = split(db_fpath)
        db_fname_noext, ext = splitext(db_fname)
        db_backup_fname = ''.join((db_fname_noext, '_backup', '_v', version, ext))
        db_backup_fpath = join(db_dpath, db_backup_fname)
        count = 0
        # TODO MAKE UTOOL THAT DOES THIS (there might be one in util_logging)
        while utool.checkpath(db_backup_fpath, verbose=True):
            db_backup_fname = ''.join((db_fname_noext, '_backup', '_v', version, '_copy', str(count), ext))
            db_backup_fpath = join(db_dpath, db_backup_fname)
            count += 1
        utool.copy(db_fpath, db_backup_fpath)
    valid_versions = sorted(db_versions.keys(), compare_string_versions)
    try:
        start_index = valid_versions.index(version) + 1
    except IndexError:
        raise AssertionError('[!update_schema_version] The current database version is unknown')
    try:
        end_index = valid_versions.index(version_target) + 1
    except IndexError:
        raise AssertionError('[!update_schema_version] The target database version is unknown')

    try:
        print('Update path: %r ' % (valid_versions[start_index:end_index]))
        for index in range(start_index, end_index):
            next_version = valid_versions[index]
            print('Updating database to version: %r' % (next_version))
            pre, update, post = db_versions[next_version]
            if pre is not None:
                pre(db, ibs=ibs)
            if update is not None:
                update(db, ibs=ibs)
            if post is not None:
                post(db, ibs=ibs)
    except Exception as ex:
        if dobackup:
            msg = 'The database update failed, rolled back to the original version.'
            utool.printex(ex, msg, iswarning=True)
            utool.remove_file(db_fpath)
            utool.copy(db_backup_fpath, db_fpath)
            if clearbackup:
                utool.remove_file(db_backup_fpath)
            # Why are we using the logging module when utool does it for us?
            logging.exception(msg)
            raise
        else:
            utool.printex(ex, (
                'The database update failed, and no backup was made.'),
                iswarning=False)
            raise
    if dobackup and clearbackup:
        utool.remove_file(db_backup_fpath)


# =======================
# SQL Context Class
# =======================


class SQLExecutionContext(object):
    """
    Context manager for transactional database calls

    FIXME: hash out details. I don't think anybody who programmed this
    knows what is going on here. So much for fine grained control.

    Referencs:
        http://stackoverflow.com/questions/9573768/understanding-python-sqlite-mechanics-in-multi-module-enviroments

    """
    def __init__(context, db, operation, nInput=None, auto_commit=True,
                 start_transaction=False, verbose=PRINT_SQL):
        context.auto_commit = auto_commit
        context.db = db  # Reference to sqldb
        context.operation = operation
        context.nInput = nInput
        context.start_transaction = start_transaction
        #context.__dict__.update(locals())  # Too mystic?
        context.operation_type = get_operation_type(operation)  # Parse the optype
        context.verbose = verbose

    def __enter__(context):
        """ Checks to see if the operating will change the database """
        #utool.printif(lambda: '[sql] Callers: ' + utool.get_caller_name(range(3, 6)), DEBUG)
        if context.nInput is not None:
            context.operation_lbl = ('[sql] execute nInput=%d optype=%s: '
                                       % (context.nInput, context.operation_type))
        else:
            context.operation_lbl = '[sql] executeone optype=%s: ' % (context.operation_type)
        # Start SQL Transaction
        context.cur = context.db.connection.cursor()  # HACK in a new cursor
        #context.cur = context.db.cur  # OR USE DB CURSOR??
        if context.start_transaction:
            #context.cur.execute('BEGIN', ())
            context.cur.execute('BEGIN')
        if context.verbose or PRINT_SQL:
            print(context.operation_lbl)
            if context.verbose:
                print('[sql] operation=\n' + context.operation)
        # Comment out timeing code
        if __debug__:
            if NOT_QUIET and (PRINT_SQL or context.verbose):
                context.tt = utool.tic(context.operation_lbl)
        return context

    # --- with SQLExecutionContext: statment code happens here ---

    def execute_and_generate_results(context, params):
        """ helper for context statment """
        try:
            #print(context.cur.rowcount)
            #print(id(context.cur))
            context.cur.execute(context.operation, params)
            #print(context.cur.rowcount)
        except lite.Error as ex:
            utool.printex(ex, 'sql.Error', keys=['params'])
            #print('[sql.Error] %r' % (type(ex),))
            #print('[sql.Error] params=<%r>' % (params,))
            raise
        is_insert = context.operation_type.startswith('INSERT')
        return _results_gen(context.cur, get_last_id=is_insert)

    def __exit__(context, type_, value, trace):
        """ Finalization of an SQLController call """
        #print('exit context')
        if __debug__:
            if NOT_QUIET and (PRINT_SQL or context.verbose):
                utool.toc(context.tt)
        if trace is not None:
            # An SQLError is a serious offence.
            print('[sql] FATAL ERROR IN QUERY CONTEXT')
            print('[sql] operation=\n' + context.operation)
            context.db.dump()  # Dump on error
            print('[sql] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
        else:
            # Commit the transaction
            if context.auto_commit:
                #print('commit %r' % context.operation_lbl)
                context.db.connection.commit()
                #if context.start_transaction:
                #    context.cur.execute('COMMIT')
            else:
                print('no commit %r' % context.operation_lbl)
                #context.db.commit(verbose=False)
                #context.cur.commit()
            #context.cur.close()


def get_operation_type(operation):
    """
    Parses the operation_type from an SQL operation
    """
    operation = ' '.join(operation.split('\n')).strip()
    operation_type = operation.split(' ')[0].strip()
    if operation_type.startswith('SELECT'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('INSERT'):
        operation_args = utool.str_between(operation, operation_type, '(').strip()
    elif operation_type.startswith('DROP'):
        operation_args = ''
    elif operation_type.startswith('ALTER'):
        operation_args = ''
    elif operation_type.startswith('UPDATE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('DELETE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('CREATE'):
        operation_args = utool.str_between(operation, operation_type, '(').strip()
    else:
        operation_args = None
    operation_type += ' ' + operation_args.replace('\n', ' ')
    return operation_type.upper()


def sanatize_sql(db, tablename, columns=None):
    """ Sanatizes an sql tablename and column. Use sparingly """
    tablename = re.sub('[^a-z_0-9]', '', tablename)
    valid_tables = db.get_table_names()
    if tablename not in valid_tables:
        raise Exception('UNSAFE TABLE: tablename=%r' % tablename)
    if columns is None:
        return tablename
    else:
        def _sanitize_sql_helper(column):
            column = re.sub('[^a-z_0-9]', '', column)
            valid_columns = db.get_column_names(tablename)
            if column not in valid_columns:
                raise Exception('UNSAFE COLUMN: tablename=%r column=%r' %
                                (tablename, column))
                return None
            else:
                return column

        columns = [_sanitize_sql_helper(column) for column in columns]
        columns = [column for column in columns if columns is not None]

        return tablename, columns


def get_nth_test_schema_version(schema_spec, n=-1, autogenerate=False):
    """
    Args:
        schema_spec (module): schema module to get nth version of
        n (int): version index (-1 is the latest)
    """
    from ibeis.control import SQLDatabaseControl as sqldbc
    dbname = schema_spec.__name__
    print('[_SQL] getting n=%r-th version of %r' % (n, dbname))
    version_expected = list(schema_spec.VALID_VERSIONS.keys())[n]
    cachedir = ut.ensure_app_resource_dir('ibeis_test')
    db_fname = 'test_%s.sqlite3' % dbname
    utool.delete(join(cachedir, db_fname))
    db = sqldbc.SQLDatabaseController(cachedir, db_fname, text_factory=unicode)
    ensure_correct_version(
        None,
        db,
        version_expected,
        schema_spec,
        autogenerate=autogenerate
    )
    return db
