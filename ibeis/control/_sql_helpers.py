from __future__ import absolute_import, division, print_function
from ibeis import constants as const
from ibeis import params
from ibeis.control import __SQLITE3__ as lite
from os.path import split, splitext, join, exists
import datetime
import logging
import re
import utool as ut
import distutils
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[sql-helpers]')

# =======================
# Helper Functions
# =======================
VERBOSE_SQL    = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql'))
#AUTODUMP = ut.get_argflag('--auto-dump')
NOT_QUIET = not (ut.QUIET or ut.get_argflag('--quiet-sql'))


def default_decor(func):
    return profile(func)
    #return func


@default_decor
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
    assert len(results_) < 2, 'throwing away results! { %r }' % (results_,)
    return results


def compare_string_versions(a, b):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control._sql_helpers import *  # NOQA
        >>> a = '1.1.1'
        >>> b = '1.0.0'
        >>> result1 = compare_string_versions(a, b)
        >>> result2 = compare_string_versions(b, a)
        >>> result3 = compare_string_versions(a, a)
        >>> result = ', '.join(map(str, [result1, result2, result3]))
        >>> print(result)
        1, -1, 0
    """
    va = distutils.version.LooseVersion(a)
    vb = distutils.version.LooseVersion(b)
    if va > vb:
        return 1
    elif va < vb:
        return -1
    elif va == vb:
        return 0
    #a = map(int, a.strip().split('.'))
    #b = map(int, b.strip().split('.'))
    #while len(a) < 3:
    #    a.append(0)
    #while len(b) < 3:
    #    b.append(0)
    #if a[0] < b[0]:
    #    return -1
    #elif a[0] > b[0]:
    #    return 1
    #else:
    #    if a[1] < b[1]:
    #        return -1
    #    elif a[1] > b[1]:
    #        return 1
    #    else:
    #        if a[2] < b[2]:
    #            return -1
    #        elif a[2] > b[2]:
    #            return 1
    #        elif a[2] == b[2]:
    #            return 0
    ## return 0 - identical
    raise AssertionError('[!update_schema_version] Two version numbers are the same along the update path')


def fix_metadata_consistency(db):
    """
    duct tape function

    db.print_table_csv('metadata')
    """
    db.print_table_csv('metadata')
    metadata_items = db.get_metadata_items()
    new_metadata_items = []
    for key, val in metadata_items:
        # FIX OLDSTYLE SUPERKEYS
        if key.endswith('_constraint'):
            # the constraint metadata keys were all depricated
            continue
        if key.endswith('_superkeys'):
            if val.find(';') > -1:
                newval = repr([tuple(val.split(';'))])
            elif not val.startswith('['):
                newval = repr([(val,)])
            else:
                continue
            new_metadata_items.append((key, newval))
        else:
            print('--')
            print(key)
            print(val)
    for key, val in new_metadata_items:
        db.set_metadata_val(key, val)


# =========================
# Database Backup Functions
# =========================


MAX_KEEP = 2048


def revert_to_backup(ibs):
    r"""
    Args:
        db_dir (?):

    CommandLine:
        python -m ibeis.control._sql_helpers --exec-revert_to_backup

    Example:
        >>> # SCRIPT
        >>> from ibeis.control._sql_helpers import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='GZ_Master1')
        >>> result = revert_to_backup(ibs)
        >>> print(result)
    """
    db_path = ibs.get_db_core_path()
    ibs.disconnect_sqldatabase()
    backup_dir = ibs.backupdir
    ut.move(db_path, ut.get_nonconflicting_path(db_path + 'revertfrom.%d.orig'))
    # Carefull may invalidate the cache
    fname, ext = splitext(db_path)
    path_list = sorted(ut.glob(backup_dir, '*%s' % ext))
    previous_backup = path_list[-1]
    ut.copy(previous_backup, db_path)


def ensure_daily_database_backup(db_dir, db_fname, backup_dir, max_keep=MAX_KEEP):
    database_backup(db_dir, db_fname, backup_dir, max_keep=max_keep, manual=False)


def database_backup(db_dir, db_fname, backup_dir, max_keep=MAX_KEEP, manual=True):
    fname, ext = splitext(db_fname)
    src_fpath = join(db_dir, db_fname)
    #now = datetime.datetime.now()
    now = datetime.datetime.utcnow()
    if manual:
        now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
    else:
        now_str = now.strftime('%Y_%m_%d_00_00_00')
    #dst_fpath = join(backup_dir, '%s_backup_%s%s' % (fname, now_str, ext))
    dst_fname = ''.join((fname, '_backup_', now_str, ext))
    dst_fpath = join(backup_dir, dst_fname)
    if exists(src_fpath) and not exists(dst_fpath):
        print('[ensure_daily_database_backup] Daily backup of database: %r -> %r' % (src_fpath, dst_fpath, ))
        ut.copy(src_fpath, dst_fpath)
        # Clean-up old database backups
        path_list = sorted(ut.glob(backup_dir, '*%s' % ext))
        if len(path_list) > max_keep:
            path_delete_list = path_list[:-1 * max_keep]
            for path_delete in path_delete_list:
                print('[ensure_daily_database_backup] Deleting old backup %r' % path_delete)
                ut.remove_file(path_delete, verbose=False)


# ========================
# Schema Updater Functions
# ========================


@profile
def ensure_correct_version(ibs, db, version_expected, schema_spec,
                           dobackup=True,
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

    Example:
        >>> from ibeis.control._sql_helpers import *  # NOQA
        >>> ibs = '?'
        >>> db = ibs.db
        >>> version_expected = ibs.db_version_expected
        >>> schema_spec = DB_SCHEMA
        >>> dobackup = True
        >>> result = ensure_correct_version(ibs, db, version_expected, schema_spec, dobackup)
        >>> print(result)

    Args:
        schema_spec (module): module of schema specifications
    """
    #print('[SQL_] ensure_correct_version')
    force_incremental = params.args.force_incremental_db_update
    want_base_version = version_expected == const.BASE_DATABASE_VERSION
    if want_base_version:
        print('[SQL_] base version expected... returning')
        # Nothing to do. autogenerated file is pointless
        return
    version = db.get_db_version()
    # NEW DATABASE CONDITION
    is_base_version = (version == const.BASE_DATABASE_VERSION)
    # <DEBUG>
    #if ut.get_flag('--verbsql') or ut.VERBOSE or True:
    #    key_list = locals().keys()
    #    keystr_list = sorted(ut.parse_locals_keylist(locals(), key_list))
    #    print('KEYLIST:' + ut.indentjoin(keystr_list, '\n * '))
    # </DEBUG>
    #+-----------------------------------
    # SKIP TO CURRENT VERSION IF POSSIBLE
    #+-----------------------------------
    can_skip = is_base_version and not force_incremental
    if can_skip:
        #print('[SQL_] we can skip')
        # Check to see if a prebuilt current schema_spec module exists
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
        update_schema_version(ibs, db, schema_spec, version, version_expected,
                              dobackup=dobackup)
        db.set_db_version(version_expected)
        print('[_SQL] Database version updated (incrementally) to %r' %
                   (version_expected))
    elif version > version_expected:
        msg = (('[_SQL] ERROR: '
                'Expected database version behind. expected: %r. got: %r') %
               (version_expected, version))
        raise AssertionError(msg)


@profile
def update_schema_version(ibs, db, schema_spec, version, version_target,
                          dobackup=True, clearbackup=False):
    """
    version_target = version_expected
    clearbackup = False
    FIXME: AN SQL HELPER FUNCTION SHOULD BE AGNOSTIC TO CONTROLER OBJECTS
    """
    print('[_SQL] update_schema_version')
    db_fpath = db.fpath
    if dobackup:
        db_dpath, db_fname = split(db_fpath)
        db_fname_noext, ext = splitext(db_fname)
        db_backup_fname = ''.join((db_fname_noext, '_backup', '_v', version, ext))
        db_backup_fpath = join(db_dpath, db_backup_fname)
        count = 0
        # TODO MAKE UTOOL THAT DOES THIS (there might be one in util_logging)
        while ut.checkpath(db_backup_fpath, verbose=True):
            db_backup_fname = ''.join((db_fname_noext, '_backup', '_v', version, '_copy', str(count), ext))
            db_backup_fpath = join(db_dpath, db_backup_fname)
            count += 1
        ut.copy(db_fpath, db_backup_fpath)

    legacy_update_funcs = schema_spec.LEGACY_UPDATE_FUNCTIONS
    for legacy_version, func in legacy_update_funcs:
        if compare_string_versions(version, legacy_version) == -1:
            func(db)
    db_versions = schema_spec.VALID_VERSIONS
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
            ut.printex(ex, msg, iswarning=True)
            ut.remove_file(db_fpath)
            ut.copy(db_backup_fpath, db_fpath)
            if clearbackup:
                ut.remove_file(db_backup_fpath)
            # Why are we using the logging module when ut does it for us?
            logging.exception(msg)
            raise
        else:
            ut.printex(ex, (
                'The database update failed, and no backup was made.'),
                iswarning=False)
            raise
    if dobackup and clearbackup:
        ut.remove_file(db_backup_fpath)


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
                 start_transaction=False, verbose=VERBOSE_SQL):
        context.auto_commit = auto_commit
        context.db = db  # Reference to sqldb
        context.operation = operation
        context.nInput = nInput
        context.start_transaction = start_transaction
        #context.__dict__.update(locals())  # Too mystic?
        context.operation_type = get_operation_type(operation)  # Parse the optype
        context.verbose = verbose

    @default_decor
    def __enter__(context):
        """ Checks to see if the operating will change the database """
        #ut.printif(lambda: '[sql] Callers: ' + ut.get_caller_name(range(3, 6)), DEBUG)
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
        if context.verbose or VERBOSE_SQL:
            print(context.operation_lbl)
            if context.verbose:
                print('[sql] operation=\n' + context.operation)
        # Comment out timeing code
        if __debug__:
            if NOT_QUIET and (VERBOSE_SQL or context.verbose):
                context.tt = ut.tic(context.operation_lbl)
        return context

    # --- with SQLExecutionContext: statment code happens here ---

    @default_decor
    def execute_and_generate_results(context, params):
        """ helper for context statment """
        try:
            #print(context.cur.rowcount)
            #print(id(context.cur))
            context.cur.execute(context.operation, params)
            #print(context.cur.rowcount)
        except lite.Error as ex:
            ut.printex(ex, 'sql.Error', keys=['params'])
            #print('[sql.Error] %r' % (type(ex),))
            #print('[sql.Error] params=<%r>' % (params,))
            raise
        is_insert = context.operation_type.startswith('INSERT')
        return _results_gen(context.cur, get_last_id=is_insert)

    @default_decor
    def __exit__(context, type_, value, trace):
        """ Finalization of an SQLController call """
        #print('exit context')
        if __debug__:
            if NOT_QUIET and (VERBOSE_SQL or context.verbose):
                ut.toc(context.tt)
        if trace is not None:
            # An SQLError is a serious offence.
            print('[sql] FATAL ERROR IN QUERY CONTEXT')
            print('[sql] operation=\n' + context.operation)
            DUMP_ON_EXCEPTION = False
            if DUMP_ON_EXCEPTION:
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


def sanitize_sql(db, tablename, columns=None):
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


def autogenerate_nth_schema_version(schema_spec, n=-1):
    r"""
    dumps, prints, or diffs autogen schema based on command line

    Args:
        schema_spec (?):
        n (int):

    CommandLine:
        python -m ibeis.control._sql_helpers --test-autogenerate_nth_schema_version

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control._sql_helpers import *  # NOQA
        >>> from ibeis.control import DB_SCHEMA
        >>> # build test data
        >>> schema_spec = DB_SCHEMA
        >>> n = 1
        >>> # execute function
        >>> tablename = autogenerate_nth_schema_version(schema_spec, n)
        >>> # verify results
        >>> result = str(tablename)
        >>> print(result)
    """
    import utool as ut
    print('[_SQL] AUTOGENERATING CURRENT SCHEMA')
    db = get_nth_test_schema_version(schema_spec, n=n)
    # Auto-generate the version skip schema file
    schema_spec_dir, schema_spec_fname = split(schema_spec.__file__)
    schema_spec_fname = splitext(schema_spec_fname)[0]
    # HACK TO GET AUTOGEN COMMAND
    # FIXME: Make this autogen command a bit more sane and not completely
    # coupled with ibeis
    autogen_cmd = ut.codeblock(
        '''
        python -m ibeis.control.{schema_spec_fname} --test-autogen_{funcname} --force-incremental-db-update --write
        python -m ibeis.control.{schema_spec_fname} --test-autogen_{funcname} --force-incremental-db-update
        '''
    ).format(schema_spec_fname=schema_spec_fname, funcname=schema_spec_fname.lower())
    autogen_text = db.get_schema_current_autogeneration_str(autogen_cmd)

    autogen_fname = '%s_CURRENT.py' % schema_spec_fname
    autogen_fpath = join(schema_spec_dir, autogen_fname)

    dowrite = ut.get_argflag(('-w', '--write', '--dump-autogen-schema'))
    show_diff = ut.get_argflag('--diff')
    num_context_lines = ut.get_argval('--diff', type_=int, default=None)
    show_diff = show_diff or num_context_lines is not None
    dowrite = dowrite and not show_diff

    if dowrite:
        ut.write_to(autogen_fpath, autogen_text)
    else:
        if show_diff:
            if ut.checkpath(autogen_fpath, verbose=True):
                prev_text = ut.read_from(autogen_fpath)
                textdiff = ut.util_str.get_textdiff(prev_text, autogen_text, num_context_lines=num_context_lines)
                ut.print_difftext(textdiff)
        else:
            ut.util_print.print_python_code(autogen_text)
        print('\nL___\n...would write to: %s' % autogen_fpath)

    print(' Run with -n=%r to get a specific schema version by index. -1 == latest')
    print(' Run with --write to autogenerate latest schema version')
    print(' Run with --diff or --diff=<numcontextlines> to see the difference between current and requested')
    return db


def get_nth_test_schema_version(schema_spec, n=-1):
    """
    Gets a fresh and empty test version of a schema

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
    ut.delete(join(cachedir, db_fname))
    db = sqldbc.SQLDatabaseController(cachedir, db_fname, text_factory=unicode)
    ensure_correct_version(
        None, db, version_expected, schema_spec, dobackup=False)
    return db

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control._sql_helpers
        python -m ibeis.control._sql_helpers --allexamples
        python -m ibeis.control._sql_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
