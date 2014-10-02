"""
sysres.py == system_resources
Module for dealing with system resoureces in the context of IBEIS
"""
from __future__ import absolute_import, division, print_function
import os
from os.path import exists, join, realpath
import utool
from utool import util_cache, util_list
from ibeis import constants
from ibeis import params

# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[sysres]', DEBUG=False)

WORKDIR_CACHEID   = 'work_directory_cache_id'
DEFAULTDB_CAHCEID = 'cached_dbdir'
LOGDIR_CACHEID = utool.logdir_cacheid
__APPNAME__ = 'ibeis'


def _ibeis_cache_dump():
    util_cache.global_cache_dump(appname=__APPNAME__)


def _ibeis_cache_write(key, val):
    """ Writes to global IBEIS cache """
    print('[sysres] set %s=%r' % (key, val))
    util_cache.global_cache_write(key, val, appname=__APPNAME__)


def _ibeis_cache_read(key, **kwargs):
    """ Reads from global IBEIS cache """
    return util_cache.global_cache_read(key, appname=__APPNAME__, **kwargs)


# Specific cache getters / setters

def set_default_dbdir(dbdir):
    printDBG('[sysres] SETTING DEFAULT DBDIR: %r' % dbdir)
    _ibeis_cache_write(DEFAULTDB_CAHCEID, dbdir)


def get_default_dbdir():
    dbdir = _ibeis_cache_read(DEFAULTDB_CAHCEID, default=None)
    printDBG('[sysres] READING DEFAULT DBDIR: %r' % dbdir)
    return dbdir


def get_syncdir():
    # TODO: Allow dirs in syncdir to count as in workdir
    secret = 'AFETDAKURTJ6WH3PXYOTJDBO3KBC2KJJP'  # NOQA


def get_workdir(allow_gui=True):
    """ Returns the work directory set for this computer.  If allow_gui is true,
    a dialog will ask a user to specify the workdir if it does not exist. """
    work_dir = _ibeis_cache_read(WORKDIR_CACHEID, default='.')
    if work_dir is not '.' and exists(work_dir):
        return work_dir
    if allow_gui:
        work_dir = set_workdir()
        return get_workdir(allow_gui=False)
    return None


def set_workdir(work_dir=None, allow_gui=True):
    """ Sets the workdirectory for this computer """
    if work_dir is None and allow_gui:
        work_dir = guiselect_workdir()
    if work_dir is None or not exists(work_dir):
        raise AssertionError('invalid workdir=%r' % work_dir)
    _ibeis_cache_write(WORKDIR_CACHEID, work_dir)


def set_logdir(log_dir):
    utool.ensuredir(log_dir)
    utool.stop_logging()
    _ibeis_cache_write(LOGDIR_CACHEID, log_dir)
    utool.start_logging(appname=__APPNAME__)


def get_rawdir():
    """ Returns the standard raw data directory """
    workdir = get_workdir()
    rawdir = utool.truepath(join(workdir, '../raw'))
    return rawdir


def guiselect_workdir():
    """ Prompts the user to specify a work directory """
    import guitool
    guitool.ensure_qtapp()
    # Gui selection
    work_dir = guitool.select_directory('Work dir not currently set.' +
                                        'Select a work directory')
    # Make sure selection is ok
    if not exists(work_dir):
        try_again = guitool.user_option(
            paremt=None,
            msg='Directory %r does not exist.' % work_dir,
            title='get work dir failed',
            options=['Try Again'],
            use_cache=False)
        if try_again == 'Try Again':
            return guiselect_workdir()
    return work_dir


def get_dbalias_dict():
    dbalias_dict = {}
    if utool.is_developer():
        # For jon's convinience
        dbalias_dict.update({
            'NAUTS':            'NAUT_Dan',
            'WD':               'WD_Siva',
            'LF':               'LF_all',
            'GZ':               'GZ_ALL',
            'MOTHERS':          'PZ_MOTHERS',
            'FROGS':            'Frogs',
            'TOADS':            'WY_Toads',
            'SEALS':            'Seals',

            'OXFORD':           'Oxford_Buildings',
            'PARIS':            'Paris_Buildings',

            'JAG_KELLY':        'JAG_Kelly',
            'JAG_KIERYN':       'JAG_Kieryn',
            'WILDEBEAST':       'Wildebeast',
            'WDOGS':            'WD_Siva',

            'PZ':               'PZ_FlankHack',
            'PZ2':              'PZ-Sweatwater',
            'PZ_MARIANNE':      'PZ_Marianne',
            'PZ_DANEXT_TEST':   'PZ_DanExt_Test',
            'PZ_DANEXT_ALL':    'PZ_DanExt_All',

            'LF_ALL':           'LF_all',
            'WS_HARD':          'WS_hard',
            'SONOGRAMS':        'sonograms',

        })
        dbalias_dict['JAG'] = dbalias_dict['JAG_KELLY']
    return dbalias_dict


def db_to_dbdir(db, allow_newdir=False, extra_workdirs=[], use_sync=False):
    """ Implicitly gets dbdir. Searches for db inside of workdir """
    if utool.VERBOSE:
        print('[sysres] db_to_dbdir: db=%r, allow_newdir=%r' % (db, allow_newdir))

    work_dir = get_workdir()
    dbalias_dict = get_dbalias_dict()

    workdir_list = []
    for extra_dir in extra_workdirs:
        if exists(extra_dir):
            workdir_list.append(extra_dir)
    if use_sync:
        sync_dir = join(work_dir, '../sync')
        if exists(sync_dir):
            workdir_list.append(sync_dir)
    workdir_list.append(work_dir)  # TODO: Allow multiple workdirs

    # Check all of your work directories for the database
    for _dir in workdir_list:
        dbdir = realpath(join(_dir, db))
        # Use db aliases
        if not exists(dbdir) and db.upper() in dbalias_dict:
            dbdir = join(_dir, dbalias_dict[db.upper()])
        if exists(dbdir):
            break

    # Create the database if newdbs are allowed in the workdir
    #print('allow_newdir=%r' % allow_newdir)
    if allow_newdir:
        utool.ensuredir(dbdir, verbose=True)

    # Complain if the implicit dbdir does not exist
    if not exists(dbdir):
        print('!!!')
        print('[sysres] WARNING: db=%r not found in work_dir=%r' %
              (db, work_dir))
        fname_list = os.listdir(work_dir)
        lower_list = [fname.lower() for fname in fname_list]
        index = util_list.listfind(lower_list, db.lower())
        if index is not None:
            print('[sysres] WARNING: db capitalization seems to be off')
            if not utool.STRICT:
                print('[sysres] attempting to fix it')
                db = fname_list[index]
                dbdir = join(work_dir, db)
                print('[sysres] dbdir=%r' % dbdir)
                print('[sysres] db=%r' % db)
        if not exists(dbdir):
            msg = '[sysres!] ERROR: Database does not exist and allow_newdir=False'
            print('<!!!>')
            print(msg)
            print('[sysres!] Here is a list of valid dbs: ' +
                  utool.indentjoin(fname_list, '\n  * '))
            print('[sysres!] dbdir=%r' % dbdir)
            print('[sysres!] db=%r' % db)
            print('[sysres!] work_dir=%r' % work_dir)
            print('</!!!>')
            raise AssertionError(msg)
        print('!!!')
    return dbdir


def get_args_dbdir(defaultdb=None, allow_newdir=False, db=None, dbdir=None):
    """ Machinery for finding a database directory """
    #if not utool.QUIET and utool.VERBOSE:
    printDBG('[sysres] get_args_dbdir: parsing commandline for dbdir')
    printDBG('[sysres] defaultdb=%r, allow_newdir=%r' % (defaultdb, allow_newdir))
    printDBG('[sysres] db=%r, dbdir=%r' % (db, dbdir))
    if dbdir is None:
        printDBG('[sysres] use command line dbdir')
        dbdir = params.args.dbdir
    if db is None:
        printDBG('[sysres] use command line db')
        db = params.args.db
    if dbdir == 'None' or db == 'None':
        print('Forcing no dbdir')
        # If specified as the string none, the user forces no db
        return None
    # Force absolute path
    if dbdir is not None:
        dbdir = realpath(dbdir)
        printDBG('[sysres] realpath dbdir: %r' % dbdir)
    # Invalidate bad values
    if dbdir is None or dbdir in ['', ' ', '.']:  # or not exists(dbdir):
        dbdir = None
        printDBG('[sysres] Invalidate dbdir: %r' % dbdir)
    # Fallback onto args.db
    if dbdir is None:
        printDBG('[sysres] Trying cache')
        # Try a cached / commandline / default db
        if db is None and defaultdb == 'cache' and not params.args.nocache_db:
            dbdir = get_default_dbdir()
            #if not utool.QUIET and utool.VERBOSE:
            printDBG('[sysres] Loading dbdir from cache.')
            printDBG('[sysres] dbdir=%r' % (dbdir,))
        elif db is not None:
            dbdir = db_to_dbdir(db, allow_newdir=allow_newdir)
        elif defaultdb is not None:
            dbdir = db_to_dbdir(defaultdb, allow_newdir=allow_newdir)
    printDBG('[sysres] return get_args_dbdir: dbdir=%r' % (dbdir,))
    return dbdir


def is_ibeisdb(path):
    """ Checks to see if path contains the IBEIS internal dir """
    return exists(join(path, constants.PATH_NAMES._ibsdb))


def is_hsdb(dbdir):
    return is_hsdbv4(dbdir) or is_hsdbv3(dbdir)


def is_hsdbv4(dbdir):
    has4 = (exists(join(dbdir, '_hsdb')) and
            exists(join(dbdir, '_hsdb', 'name_table.csv')) and
            exists(join(dbdir, '_hsdb', 'image_table.csv')) and
            exists(join(dbdir, '_hsdb', 'chip_table.csv')))
    return has4


def is_hsdbv3(dbdir):
    has3 = (exists(join(dbdir, '.hs_internals')) and
            exists(join(dbdir, '.hs_internals', 'name_table.csv')) and
            exists(join(dbdir, '.hs_internals', 'image_table.csv')) and
            exists(join(dbdir, '.hs_internals', 'chip_table.csv')))
    return has3


def get_hsinternal(hsdb_dir):
    internal_dir = join(hsdb_dir, '_hsdb')
    if not is_hsdbv4(hsdb_dir):
        internal_dir = join(hsdb_dir, '.hs_internals')
    return internal_dir


def is_hsinternal(dbdir):
    return exists(join(dbdir, '.hs_internals'))


def get_ibsdb_list(workdir=None):
    import numpy as np
    if workdir is None:
        workdir = get_workdir()
    dbname_list = os.listdir(workdir)
    dbpath_list = np.array([join(workdir, name) for name in dbname_list])
    is_ibs_list = np.array(list(map(is_ibeisdb, dbpath_list)))
    ibsdb_list  = dbpath_list[is_ibs_list].tolist()
    return ibsdb_list


def ensure_pz_mtest():
    """ Ensures that you have the PZ_MTEST dataset """
    from ibeis import sysres
    import utool
    workdir = sysres.get_workdir()
    mtest_zipped_url = 'https://www.dropbox.com/s/xdae2yvsp57l4t2/PZ_MTEST.zip'
    mtest_dir = utool.grab_zipped_url(mtest_zipped_url, ensure=True, download_dir=workdir)
    print('have mtest_dir=%r' % (mtest_dir,))
