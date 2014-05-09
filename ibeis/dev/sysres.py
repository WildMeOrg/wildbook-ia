"""
Module for dealing with system resoureces in the context of IBEIS
"""
from __future__ import absolute_import, division, print_function
import os
import sys
from os.path import exists, join, realpath, dirname
import utool
from utool import util_cache, util_list, util_cplat
from ibeis.dev import params

WORKDIR_CACHEID   = 'work_directory_cache_id'
DEFAULTDB_CAHCEID = 'cached_dbdir'


def _ibeis_cache_dump():
    util_cache.global_cache_dump(appname='ibeis')


def _ibeis_cache_write(key, val):
    """ Writes to global IBEIS cache """
    util_cache.global_cache_write(key, val, appname='ibeis')


def _ibeis_cache_read(key, **kwargs):
    """ Reads from global IBEIS cache """
    return util_cache.global_cache_read(key, appname='ibeis', **kwargs)


# Specific cache getters / setters

def set_default_dbdir(dbdir):
    print('seting default database directory to: %r' % dbdir)
    _ibeis_cache_write(DEFAULTDB_CAHCEID, dbdir)


def get_default_dbdir():
    return _ibeis_cache_read(DEFAULTDB_CAHCEID, default=None)


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
    computer_name = util_cplat.get_computer_name()
    if computer_name in ['Hyrule', 'Ooo', 'BakerStreet']:
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


def db_to_dbdir(db, allow_newdir=False):
    """ Implicitly gets dbdir. Searches for db inside of workdir """
    work_dir = get_workdir()
    dbalias_dict = get_dbalias_dict()

    workdir_list = [work_dir]  # TODO: Allow multiple workdirs
    sync_dir = join(work_dir, '../sync')
    if exists(sync_dir):
        workdir_list.append(sync_dir)

    # Check all of your work directories for the database
    for _dir in workdir_list:
        dbdir = realpath(join(_dir, db))
        # Use db aliases
        if not exists(dbdir) and db.upper() in dbalias_dict:
            dbdir = join(_dir, dbalias_dict[db.upper()])

    # Create the database if newdbs are allowed in the workdir
    if not exists(dbdir) and allow_newdir:
        if exists(dirname(dbdir)):
            print('[workdir] MAKE DIR: %r' % dbdir)
            os.mkdir(dbdir)

    # Complain if the implicit dbdir does not exist
    if not exists(dbdir):
        print('!!!')
        print('[workdir] WARNING: db=%r not found in work_dir=%r' %
              (db, work_dir))
        fname_list = os.listdir(work_dir)
        lower_list = [fname.lower() for fname in fname_list]
        index = util_list.listfind(lower_list, db.lower())
        if index is not None:
            print('[workdir] WARNING: db capitalization seems to be off')
            if not '--strict' in sys.argv:
                print('[workdir] attempting to fix it')
                db = fname_list[index]
                dbdir = join(work_dir, db)
                print('[workdir] dbdir=%r' % dbdir)
                print('[workdir] db=%r' % db)
        if not exists(dbdir):
            print('[workdir] Valid DBs:')
            print('\n'.join(fname_list))
            print('[workdir] dbdir=%r' % dbdir)
            print('[workdir] db=%r' % db)
            print('[workdir] work_dir=%r' % work_dir)

            raise AssertionError('[workdir] FATAL ERROR. Cannot load database')
        print('!!!')
    return dbdir


def get_args_dbdir(defaultdb=None, allow_newdir=False):
    """ Machinery for finding a database directory """
    if not utool.QUIET and utool.VERBOSE:
        print('[sysres] parsing commandline for dbdir')
        print('[sysres] defaultdb=%r, allow_newdir=%r' % (defaultdb, allow_newdir))
    dbdir = params.args.dbdir
    db = params.args.db
    # Force absolute path
    if dbdir is not None:
        dbdir = realpath(dbdir)
    # Invalidate bad values
    if dbdir is None or dbdir in ['', ' ', '.'] or not exists(dbdir):
        dbdir = None
    # Fallback onto args.db
    if dbdir is None:
        # Try a cached / commandline / default db
        if db is None and defaultdb == 'cache' and not params.args.nocache_db:
            dbdir = get_default_dbdir()
            if not utool.QUIET and utool.VERBOSE:
                print('[sysres] Loading dbdir from cache.')
                print('[sysres] dbdir=%r' % (dbdir,))
        elif db is not None:
            dbdir = db_to_dbdir(db, allow_newdir=allow_newdir)
        elif defaultdb is not None:
            dbdir = db_to_dbdir(defaultdb, allow_newdir=allow_newdir)
    return dbdir
