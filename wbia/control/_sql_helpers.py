# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import split, splitext, join, exists
import six
import datetime
import distutils
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)

VERBOSE_SQL = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql', '--verbsql'))
NOT_QUIET = not (ut.QUIET or ut.get_argflag('--quiet-sql'))


def compare_string_versions(a, b):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control._sql_helpers import *  # NOQA
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
    raise AssertionError(
        '[!update_schema_version] Two version numbers are '
        'the same along the update path'
    )


def _devcheck_backups():
    from wbia import dtool as dt

    dbdir = ut.truepath('~/work/PZ_Master1/_ibsdb')
    sorted(ut.glob(join(dbdir, '_wbia_backups'), '*staging_back*.sqlite3'))
    fpaths = sorted(ut.glob(join(dbdir, '_wbia_backups'), '*database_back*.sqlite3'))
    for fpath in fpaths:
        db = dt.SQLDatabaseController(fpath=fpath)
        print('fpath = %r' % (fpath,))
        num_edges = len(db.executeone('SELECT rowid from annotmatch'))
        print('num_edges = %r' % (num_edges,))
        num_names = len(db.executeone('SELECT DISTINCT name_rowid from annotations'))
        print('num_names = %r' % (num_names,))
        # df = db.get_table_as_pandas('annotations', columns=['annot_rowid',
        #                                                     'name_rowid'])


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
        python -m wbia.control._sql_helpers --exec-revert_to_backup

    Example:
        >>> # SCRIPT
        >>> from wbia.control._sql_helpers import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='elephants')
        >>> result = revert_to_backup(ibs)
        >>> print(result)
    """
    db_path = ibs.get_db_core_path()
    staging_path = ibs.get_db_staging_path()

    ibs.disconnect_sqldatabase()
    backup_dir = ibs.backupdir

    # Core database
    fname, ext = splitext(db_path)
    db_path_ = '%s_revert.sqlite3' % (fname,)
    ut.move(db_path, db_path_)
    fpath, fname = split(fname)
    path_list = sorted(ut.glob(backup_dir, '%s_*%s' % (fname, ext,)))
    assert len(path_list) > 0
    previous_backup = path_list[-1]
    copy_database(previous_backup, db_path)

    # Staging database
    fname, ext = splitext(staging_path)
    staging_path_ = '%s_revert.sqlite3' % (fname,)
    ut.move(staging_path, staging_path_)
    fpath, fname = split(fname)
    path_list = sorted(ut.glob(backup_dir, '%s_*%s' % (fname, ext,)))
    assert len(path_list) > 0
    previous_backup = path_list[-1]
    copy_database(previous_backup, staging_path)

    # Delete the cache
    ut.delete(ibs.cachedir)


def ensure_daily_database_backup(db_dir, db_fname, backup_dir, max_keep=MAX_KEEP):
    database_backup(db_dir, db_fname, backup_dir, max_keep=max_keep, manual=False)


def get_backupdir(db_dir, db_fname):
    r"""
    CommandLine:
        python -m _sql_helpers get_backupdir --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control._sql_helpers import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> db_dir = ibs.get_ibsdir()
        >>> db_fname = ibs.sqldb_fname
        >>> backup_dir = ibs.backupdir
        >>> result = get_backupdir(db_dir, db_fname)
    """
    pass


def get_backup_fpaths(ibs):
    fname, ext = splitext(ibs.sqldb_fname)
    backups = sorted(ut.glob(ibs.backupdir, '*%s' % ext))
    # backup_info = [ut.get_file_info(fpath) for fpath in backups]
    modified = [ut.get_file_info(fpath)['last_modified'] for fpath in backups]
    unixtimes = [ut.util_time.exiftime_to_unixtime(tag) for tag in modified]
    backups = ut.sortedby(backups, unixtimes)
    return backups
    # backup_uuids = [ut.get_file_uuid(fpath) for fpath in backups]
    # backup_hashes = [ut.get_file_hash(fpath) for fpath in backups]
    # backup_bytes = [ut.get_file_nBytes(fpath) for fpath in backups]
    pass


def copy_database(src_fpath, dst_fpath):
    from wbia import dtool

    # Load database and ask it to copy itself, which enforces an exclusive
    # blocked lock for all processes potentially writing to the database
    timeout = 12 * 60 * 60  # Allow a lock of up to 12 hours for a database backup routine
    db = dtool.SQLDatabaseController(
        fpath=src_fpath, text_factory=six.text_type, inmemory=False, timeout=timeout
    )
    db.backup(dst_fpath)


def database_backup(db_dir, db_fname, backup_dir, max_keep=MAX_KEEP, manual=True):
    """
    >>> db_dir = ibs.get_ibsdir()
    >>> db_fname = ibs.sqldb_fname
    >>> backup_dir = ibs.backupdir
    >>> max_keep = MAX_KEEP
    >>> manual = False
    """
    fname, ext = splitext(db_fname)
    src_fpath = join(db_dir, db_fname)
    # now = datetime.datetime.now()
    now = datetime.datetime.utcnow()
    if manual:
        now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
    else:
        now_str = now.strftime('%Y_%m_%d_00_00_00')
    # dst_fpath = join(backup_dir, '%s_backup_%s%s' % (fname, now_str, ext))
    dst_fname = ''.join((fname, '_backup_', now_str, ext))
    dst_fpath = join(backup_dir, dst_fname)
    if exists(src_fpath) and not exists(dst_fpath):
        print(
            '[ensure_daily_database_backup] Daily backup of database: %r -> %r'
            % (src_fpath, dst_fpath,)
        )
        copy_database(src_fpath, dst_fpath)
        # Clean-up old database backups
        remove_old_backups(backup_dir, ext, max_keep)


def remove_old_backups(backup_dir, ext, max_keep):
    path_list = sorted(ut.glob(backup_dir, '*%s' % ext))
    if len(path_list) > max_keep:
        path_delete_list = path_list[: -1 * max_keep]
        for path_delete in path_delete_list:
            print('[ensure_daily_database_backup] Deleting old backup %r' % path_delete)
            ut.remove_file(path_delete, verbose=False)


# ========================
# Schema Updater Functions
# ========================


@profile
def ensure_correct_version(
    ibs, db, version_expected, schema_spec, dobackup=True, verbose=ut.NOT_QUIET
):
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
        >>> # DISABLE_DOCTEST
        >>> from wbia.control._sql_helpers import *  # NOQA
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
    from wbia import constants as const
    from wbia import params

    # print('[SQL_] ensure_correct_version')
    force_incremental = params.args.force_incremental_db_update
    want_base_version = version_expected == const.BASE_DATABASE_VERSION
    if want_base_version:
        print('[SQL_] base version expected... returning')
        # Nothing to do. autogenerated file is pointless
        return
    version = db.get_db_version()
    # NEW DATABASE CONDITION
    is_base_version = version == const.BASE_DATABASE_VERSION
    # <DEBUG>
    # if ut.get_flag('--verbsql') or ut.VERBOSE or True:
    #    key_list = locals().keys()
    #    keystr_list = sorted(ut.parse_locals_keylist(locals(), key_list))
    #    print('KEYLIST:' + ut.indentjoin(keystr_list, '\n * '))
    # </DEBUG>
    # +-----------------------------------
    # SKIP TO CURRENT VERSION IF POSSIBLE
    # +-----------------------------------
    can_skip = is_base_version and not force_incremental
    if can_skip:
        # print('[SQL_] we can skip')
        # Check to see if a prebuilt current schema_spec module exists
        current_schema_exists = (
            schema_spec.UPDATE_CURRENT is not None
            and schema_spec.VERSION_CURRENT is not None
        )
        if current_schema_exists:
            # check to see if more than the metadata table exists
            is_newdb = db.get_table_names() == [const.METADATA_TABLE]
            current_schema_compatible = (
                is_newdb and schema_spec.VERSION_CURRENT <= version_expected
            )
            if current_schema_compatible:
                # Since this is a new database, we do not have to worry about backinng up the
                # current database.  The subsequent update functions (if needed) will handle
                # this for us.
                if verbose:
                    print('[_SQL] New database and a current schema found')
                schema_spec.UPDATE_CURRENT(db, ibs=ibs)
                db.set_db_version(schema_spec.VERSION_CURRENT)
                if verbose:
                    print(
                        '[_SQL] Database version updated (skipped) to %r '
                        % (schema_spec.VERSION_CURRENT)
                    )
            else:
                print(
                    '[_SQL] Current database is not compatible, updating incrementally...'
                )
        else:
            print(
                '[_SQL] New database but current version not exported, updating incrementally...'
            )
    # +--------------------------------------
    # INCREMENTAL UPDATE TO EXPECTED VERSION
    # +--------------------------------------
    # Check version again for sanity's sake, update if exported current is behind expected
    version = db.get_db_version()
    if verbose:
        print(
            '[_SQL.%s] Database version: %r | Expected version: %r '
            % (ut.get_caller_name(), version, version_expected)
        )
    if version < version_expected:
        print('[_SQL] Database version behind, updating...')
        update_schema_version(
            ibs, db, schema_spec, version, version_expected, dobackup=dobackup
        )
        db.set_db_version(version_expected)
        print(
            '[_SQL] Database version updated (incrementally) to %r' % (version_expected)
        )
    elif version > version_expected:
        msg = (
            '[_SQL] ERROR: ' 'Expected database version behind. expected: %r. got: %r'
        ) % (version_expected, version)
        raise AssertionError(msg)


@profile
def update_schema_version(
    ibs, db, schema_spec, version, version_target, dobackup=True, clearbackup=False
):
    """
    version_target = version_expected
    clearbackup = False
    FIXME: AN SQL HELPER FUNCTION SHOULD BE AGNOSTIC TO CONTROLER OBJECTS
    """

    def _check_superkeys():
        all_tablename_list = db.get_table_names()
        # always ignore the metadata table.
        ignore_tables_ = ['metadata']
        tablename_list = [
            tablename
            for tablename in all_tablename_list
            if tablename not in ignore_tables_
        ]
        for tablename in tablename_list:
            superkey_colnames_list = db.get_table_superkey_colnames(tablename)
            # some tables seem to only have old constraints and aren't
            # properly updated to superkeys... weird.
            old_constraints = db.get_table_constraints(tablename)
            assert (
                len(superkey_colnames_list) > 0 or len(old_constraints) > 0
            ), 'ERROR UPDATING DATABASE, SUPERKEYS of %s DROPPED!' % (tablename,)

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
            db_backup_fname = ''.join(
                (db_fname_noext, '_backup', '_v', version, '_copy', str(count), ext)
            )
            db_backup_fpath = join(db_dpath, db_backup_fname)
            count += 1
        copy_database(db_fpath, db_backup_fpath)

    legacy_update_funcs = schema_spec.LEGACY_UPDATE_FUNCTIONS
    for legacy_version, func in legacy_update_funcs:
        if compare_string_versions(version, legacy_version) == -1:
            func(db)
    db_versions = schema_spec.VALID_VERSIONS
    import functools

    _key = functools.cmp_to_key(compare_string_versions)
    valid_versions = sorted(db_versions.keys(), key=_key)
    print('valid_versions = %r' % (valid_versions,))
    try:
        start_index = valid_versions.index(version) + 1
    except IndexError:
        raise AssertionError(
            '[!update_schema_version]' ' The current database version is unknown'
        )
    try:
        end_index = valid_versions.index(version_target) + 1
    except IndexError:
        raise AssertionError(
            '[!update_schema_version]' ' The target database version is unknown'
        )

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
            _check_superkeys()
    except Exception as ex:
        if dobackup:
            msg = 'The database update failed, rolled back to the original version.'
            ut.printex(ex, msg, iswarning=True)
            ut.remove_file(db_fpath)
            copy_database(db_backup_fpath, db_fpath)
            if clearbackup:
                ut.remove_file(db_backup_fpath)
            raise
        else:
            ut.printex(
                ex,
                ('The database update failed, and no backup was made.'),
                iswarning=False,
            )
            raise
    if dobackup and clearbackup:
        ut.remove_file(db_backup_fpath)


def autogenerate_nth_schema_version(schema_spec, n=-1):
    r"""
    dumps, prints, or diffs autogen schema based on command line

    Args:
        n (int):

    CommandLine:
        python -m wbia.control._sql_helpers --test-autogenerate_nth_schema_version

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control._sql_helpers import *  # NOQA
        >>> from wbia.control import DB_SCHEMA
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
    # coupled with wbia
    autogen_cmd = ut.codeblock(
        """
        python -m wbia.control.{schema_spec_fname} --test-autogen_{funcname} --force-incremental-db-update --write
        python -m wbia.control.{schema_spec_fname} --test-autogen_{funcname} --force-incremental-db-update --diff=1
        python -m wbia.control.{schema_spec_fname} --test-autogen_{funcname} --force-incremental-db-update
        """
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
                textdiff = ut.util_str.get_textdiff(
                    prev_text, autogen_text, num_context_lines=num_context_lines
                )
                ut.print_difftext(textdiff)
        else:
            ut.util_print.print_python_code(autogen_text)
        print('\nL___\n...would write to: %s' % autogen_fpath)

    print(' Run with -n=%r to get a specific schema version by index. -1 == latest')
    print(' Run with --write to autogenerate latest schema version')
    print(
        ' Run with --diff or --diff=<numcontextlines> to see the difference between current and requested'
    )
    return db


def get_nth_test_schema_version(schema_spec, n=-1):
    """
    Gets a fresh and empty test version of a schema

    Args:
        schema_spec (module): schema module to get nth version of
        n (int): version index (-1 is the latest)
    """
    from wbia.dtool.sql_control import SQLDatabaseController

    dbname = schema_spec.__name__
    print('[_SQL] getting n=%r-th version of %r' % (n, dbname))
    version_expected = list(schema_spec.VALID_VERSIONS.keys())[n]
    cachedir = ut.ensure_app_resource_dir('wbia_test')
    db_fname = 'test_%s.sqlite3' % dbname
    ut.delete(join(cachedir, db_fname))
    db = SQLDatabaseController(cachedir, db_fname, text_factory=six.text_type)
    ensure_correct_version(None, db, version_expected, schema_spec, dobackup=False)
    return db


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control._sql_helpers
        python -m wbia.control._sql_helpers --allexamples
        python -m wbia.control._sql_helpers --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
