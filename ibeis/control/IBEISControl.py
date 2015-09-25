# -*- coding: utf-8 -*-
"""
Note:
    THERE ARE FUNCTIONS THAT ARE INJECTED INTO THE CONTROLLER
    THAT ARE NOT DEFINED IN THIS MODULE.

    functions in the IBEISController have been split up into several submodules.
    look at the modules listed in autogenmodname_list to see the full list of
    functions that will be injected into an IBEISController object

TODO:
    Module Licence and docstring
"""
# TODO: rename annotation annotations
# TODO: make all names consistent
from __future__ import absolute_import, division, print_function
# Python
import six
import atexit
import requests
import weakref
from six.moves import zip
from os.path import join, split
import utool as ut
import ibeis  # NOQA
from ibeis.init import sysres
from ibeis import constants as const
#from ibeis import params
from ibeis.control import accessor_decors, controller_inject
from ibeis.control.accessor_decors import (default_decorator, )
import xml.etree.ElementTree as ET
# Import modules which define injectable functions
# Older manual ibeiscontrol functions
from ibeis import ibsfuncs
from ibeis.model.hots import pipeline

# NOTE: new plugin code needs to be hacked in here currently
# this is not a long term solution.

# <Pyinstaller hacks>
from ibeis.control import _autogen_featweight_funcs  # NOQA
from ibeis.control import _autogen_party_funcs  # NOQA
from ibeis.control import _autogen_annotmatch_funcs  # NOQA
from ibeis.control import manual_ibeiscontrol_funcs  # NOQA
from ibeis.control import manual_wildbook_funcs  # NOQA
from ibeis.control import manual_meta_funcs  # NOQA
from ibeis.control import manual_lbltype_funcs  # NOQA   # DEPRICATE
from ibeis.control import manual_lblannot_funcs  # NOQA  # DEPRICATE
from ibeis.control import manual_lblimage_funcs  # NOQA  # DEPRICATE
from ibeis.control import manual_image_funcs  # NOQA
from ibeis.control import manual_encounter_funcs  # NOQA
from ibeis.control import manual_egrelate_funcs  # NOQA
from ibeis.control import manual_garelate_funcs  # NOQA
from ibeis.control import manual_annot_funcs  # NOQA
from ibeis.control import manual_name_funcs  # NOQA
from ibeis.control import manual_species_funcs  # NOQA
from ibeis.control import manual_annotgroup_funcs  # NOQA
#from ibeis.control import manual_dependant_funcs  # NOQA
from ibeis.control import manual_chip_funcs  # NOQA
from ibeis.control import manual_feat_funcs  # NOQA
# </Pyinstaller hacks>


# Shiny new way to inject external functions
autogenmodname_list = [
    ('ibeis.control', '_autogen_featweight_funcs'),
    ('ibeis.control', '_autogen_party_funcs'),
    ('ibeis.control', '_autogen_annotmatch_funcs'),
    # ('ibeis.control', '_autogen_annot_funcs'),
    ('ibeis.control', 'manual_ibeiscontrol_funcs'),
    ('ibeis.control', 'manual_wildbook_funcs'),
    ('ibeis.control', 'manual_meta_funcs'),
    ('ibeis.control', 'manual_lbltype_funcs'),   # DEPRICATE
    ('ibeis.control', 'manual_lblannot_funcs'),  # DEPRICATE
    ('ibeis.control', 'manual_lblimage_funcs'),  # DEPRICATE
    ('ibeis.control', 'manual_image_funcs'),
    ('ibeis.control', 'manual_encounter_funcs'),
    ('ibeis.control', 'manual_egrelate_funcs'),
    ('ibeis.control', 'manual_garelate_funcs'),
    ('ibeis.control', 'manual_annot_funcs'),
    ('ibeis.control', 'manual_name_funcs'),
    ('ibeis.control', 'manual_species_funcs'),
    ('ibeis.control', 'manual_annotgroup_funcs'),
    # ('ibeis.control', 'manual_dependant_funcs'),
    ('ibeis.control', 'manual_chip_funcs'),
    ('ibeis.control', 'manual_feat_funcs'),
]

WITH_CNN = ut.get_argflag(('--with-cnn', '--withcnn', '--cnn'))

# HACK, don't include cnn unless its already there due to theano stuff
pluginmodname_list = []
import sys
if 'ibeis_cnn' in sys.modules or WITH_CNN:
    pluginmodname_list = [
        ('ibeis_cnn', '_plugin'),
    ]


def make_explicit_imports_for_pyinstaller():
    #making actual imports pyinstaller
    print('\n'.join(['from %s import %s  # NOQA' % pkgtup for pkgtup in autogenmodname_list]))

INJECTED_MODULES = []

for pkgname, modname in autogenmodname_list + pluginmodname_list:
    try:
        exec('from %s import %s' % (pkgname, modname, ), globals(), locals())
        module = eval(modname)
        INJECTED_MODULES.append(module)
    except ImportError as ex:
        ut.printex(ex, 'Cannot load package=%r, module=%r' % (pkgname, modname, ))

# Inject utool functions
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[ibs]')


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


__ALL_CONTROLLERS__ = []  # Global variable containing all created controllers
__IBEIS_CONTROLLER_CACHE__ = {}


def request_IBEISController(
        dbdir=None, ensure=True, wbaddr=None, verbose=ut.VERBOSE,
        use_cache=True, request_dbversion=None):
    r"""
    Alternative to directory instantiating a new controller object. Might
    return a memory cached object

    Args:
        dbdir     (str): databse directory
        ensure    (bool):
        wbaddr    (None):
        verbose   (bool):
        use_cache (bool): use the global ibeis controller cache. (default=True)
        request_dbversion (str): developer flag. Do not use.

    Returns:
        IBEISController: ibs

    CommandLine:
        python -m ibeis.control.IBEISControl --test-request_IBEISController

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> dbdir = 'testdb1'
        >>> ensure = True
        >>> wbaddr = None
        >>> verbose = True
        >>> use_cache = False
        >>> ibs = request_IBEISController(dbdir, ensure, wbaddr, verbose, use_cache)
        >>> result = str(ibs)
        >>> print(result)
    """
    # TODO: change name from new to request
    global __IBEIS_CONTROLLER_CACHE__
    if use_cache and dbdir in __IBEIS_CONTROLLER_CACHE__:
        if verbose:
            print('[request_IBEISController] returning cached controller')
        ibs = __IBEIS_CONTROLLER_CACHE__[dbdir]
    else:
        # Convert hold hotspotter dirs if necessary
        from ibeis.dbio import ingest_hsdb
        if ingest_hsdb.check_unconverted_hsdb(dbdir):
            ibs = ingest_hsdb.convert_hsdb_to_ibeis(dbdir, ensure=ensure, wbaddr=wbaddr, verbose=verbose)
        else:
            ibs = IBEISController(
                dbdir=dbdir, ensure=ensure, wbaddr=wbaddr, verbose=verbose,
                request_dbversion=request_dbversion)
        __IBEIS_CONTROLLER_CACHE__[dbdir] = ibs
    return ibs


@atexit.register
def __cleanup():
    """ prevents flann errors (not for cleaning up individual objects) """
    global __ALL_CONTROLLERS__
    global __IBEIS_CONTROLLER_CACHE__
    try:
        del __ALL_CONTROLLERS__
        del __IBEIS_CONTROLLER_CACHE__
    except NameError:
        print('cannot cleanup IBEISController')
        pass


#
#
#-----------------
# IBEIS CONTROLLER
#-----------------

@six.add_metaclass(ut.ReloadingMetaclass)
class IBEISController(object):
    """
    IBEISController docstring

    NameingConventions:
        chip  - cropped region of interest in an image, maps to one animal
        cid   - chip unique id
        gid   - image unique id (could just be the relative file path)
        name  - name unique id
        eid   - encounter unique id
        aid   - region of interest unique id
        annot - an annotation i.e. region of interest for a chip
        theta - angle of rotation for a chip
    """

    #
    #
    #-------------------------------
    # --- CONSTRUCTOR / PRIVATES ---
    #-------------------------------

    def __init__(ibs, dbdir=None, ensure=True, wbaddr=None, verbose=True,
                 request_dbversion=None):
        """ Creates a new IBEIS Controller associated with one database """
        #if verbose and ut.VERBOSE:
        print('\n[ibs.__init__] new IBEISController')
        # an dict to hack in temporary state
        ibs.const = const
        ibs.allow_override = 'override+warn'
        # observer_weakref_list keeps track of the guibacks connected to this controller
        ibs.observer_weakref_list = []
        # not completely working decorator cache
        ibs.table_cache = None
        ibs._initialize_self()
        ibs._init_dirs(dbdir=dbdir, ensure=ensure)
        # _send_wildbook_request will do nothing if no wildbook address is specified
        ibs._send_wildbook_request(wbaddr)
        ibs._init_sql(request_dbversion=request_dbversion)
        ibs._init_config()
        print('[ibs.__init__] END new IBEISController\n')

    def reset_table_cache(ibs):
        ibs.table_cache = accessor_decors.init_tablecache()

    def clear_table_cache(ibs, tablename=None):
        print('[ibs] clearing table_cache[%r]' % (tablename,))
        if tablename is None:
            ibs.reset_table_cache()
        else:
            del ibs.table_cache[tablename]

    def get_cachestats_str(ibs):
        """
        Returns info about the underlying SQL cache memory
        """
        total_size_str = ut.get_object_size_str(ibs.table_cache, lbl='size(table_cache): ')
        total_size_str = '\nlen(table_cache) = %r' % (len(ibs.table_cache))
        table_size_str_list = [ut.get_object_size_str(val, lbl='size(table_cache[%s]): ' % (key,))
                               for key, val in six.iteritems(ibs.table_cache)]
        cachestats_str = total_size_str + ut.indentjoin(table_size_str_list, '\n  * ')
        return cachestats_str

    def print_cachestats_str(ibs):
        cachestats_str = ibs.get_cachestats_str()
        print('IBEIS Controller Cache Stats:')
        print(cachestats_str)
        return cachestats_str

    #def inject_module_plugin(ibs, module):
    #    global INJECTED_MODULES
    #    if module not in INJECTED_MODULES:
    #        INJECTED_MODULES.append(module)
    #    ut.inject_instance(
    #        ibs, classtype=module.CLASS_INJECT_KEY,
    #        allow_override=ibs.allow_override, strict=False)

    def _initialize_self(ibs):
        """
        For utools auto reload
        Called after reload
        Injects code from development modules into the controller
        """
        if ut.VERBOSE:
            print('[ibs] _initialize_self()')
        ibs.reset_table_cache()

        for module in INJECTED_MODULES:
            ut.inject_instance(
                ibs, classtype=module.CLASS_INJECT_KEY,
                allow_override=ibs.allow_override, strict=False)
        ut.inject_instance(ibs, classtype=ibsfuncs.CLASS_INJECT_KEY,
                           allow_override=ibs.allow_override, strict=True)
        assert hasattr(ibs, 'get_database_species'), 'issue with ibsfuncs'

        #ut.inject_instance(ibs, classtype=('IBEISController', 'autogen_featweight'),
        #                   allow_override=ibs.allow_override, strict=False)
        #ut.inject_instance(ibs, classtype=('IBEISController', 'manual'),
        #                   allow_override=ibs.allow_override, strict=False)
        ibs.register_controller()

    def _on_reload(ibs):
        """
        For utools auto reload.
        Called before reload
        """
        # Reloading breaks flask, turn it off
        controller_inject.GLOBAL_APP_ENABLED = False
        # Only warn on first load. Overrideing while reloading is ok
        ibs.allow_override = True
        ibs.unregister_controller()
        # Reload dependent modules
        for module in INJECTED_MODULES:
            module.rrr()
        ibsfuncs.rrr()
        pass

    # We should probably not implement __del__
    # see: https://docs.python.org/2/reference/datamodel.html#object.__del__
    #def __del__(ibs):
    #    ibs.cleanup()

    # ------------
    # SELF REGISTRATION
    # ------------

    def register_controller(ibs):
        """ registers controller with global list """
        ibs_weakref = weakref.ref(ibs)
        __ALL_CONTROLLERS__.append(ibs_weakref)

    def unregister_controller(ibs):
        ibs_weakref = weakref.ref(ibs)
        try:
            __ALL_CONTROLLERS__.remove(ibs_weakref)
            pass
        except ValueError:
            pass

    # ------------
    # OBSERVER REGISTRATION
    # ------------

    def cleanup(ibs):
        """ call on del? """
        print('[ibs.cleanup] Observers (if any) notified [controller killed]')
        for observer_weakref in ibs.observer_weakref_list:
            observer_weakref().notify_controller_killed()

    @default_decorator
    def register_observer(ibs, observer):
        print('[register_observer] Observer registered: %r' % observer)
        observer_weakref = weakref.ref(observer)
        ibs.observer_weakref_list.append(observer_weakref)

    @default_decorator
    def remove_observer(ibs, observer):
        print('[remove_observer] Observer removed: %r' % observer)
        ibs.observer_weakref_list.remove(observer)

    @default_decorator
    def notify_observers(ibs):
        print('[notify_observers] Observers (if any) notified')
        for observer_weakref in ibs.observer_weakref_list:
            observer_weakref().notify()

    # ------------

    def _init_rowid_constants(ibs):
        ibs.UNKNOWN_LBLANNOT_ROWID = 0  # ADD TO CONSTANTS
        ibs.UNKNOWN_NAME_ROWID     = ibs.UNKNOWN_LBLANNOT_ROWID  # ADD TO CONSTANTS
        ibs.UNKNOWN_SPECIES_ROWID  = ibs.UNKNOWN_LBLANNOT_ROWID  # ADD TO CONSTANTS
        ibs.MANUAL_CONFIG_SUFFIX = 'MANUAL_CONFIG'
        ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)
        # duct_tape.fix_compname_configs(ibs)
        # duct_tape.remove_database_slag(ibs)
        # duct_tape.fix_nulled_yaws(ibs)
        lbltype_names    = const.KEY_DEFAULTS.keys()
        lbltype_defaults = const.KEY_DEFAULTS.values()
        lbltype_ids = ibs.add_lbltype(lbltype_names, lbltype_defaults)
        ibs.lbltype_ids = dict(zip(lbltype_names, lbltype_ids))

    @default_decorator
    def _init_sql(ibs, request_dbversion=None):
        """ Load or create sql database """
        from ibeis.other import duct_tape  # NOQA
        ibs._init_sqldbcore(request_dbversion=request_dbversion)
        ibs._init_sqldbcache()
        # ibs.db.dump_schema()
        # ibs.db.dump()
        ibs._init_rowid_constants()

    #@ut.indent_func
    def _init_sqldbcore(ibs, request_dbversion=None):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis  # NOQA
            >>> #ibs = ibeis.opendb('PZ_MTEST')
            >>> #ibs = ibeis.opendb('PZ_Master0')
            >>> ibs = ibeis.opendb('testdb1')
            >>> #ibs = ibeis.opendb('PZ_Master0')

        Ignore:
            aid_list = ibs.get_valid_aids()
            #ibs.update_annot_visual_uuids(aid_list)
            vuuid_list = ibs.get_annot_visual_uuids(aid_list)
            aid_list2 =  ibs.get_annot_aids_from_visual_uuid(vuuid_list)
            assert aid_list2 == aid_list
            # v1.3.0 testdb1:264us, PZ_MTEST:3.93ms, PZ_Master0:11.6s
            %timeit ibs.get_annot_aids_from_visual_uuid(vuuid_list)
            # v1.3.1 testdb1:236us, PZ_MTEST:1.83ms, PZ_Master0:140ms

            ibs.print_encounter_table(exclude_columns=['encounter_uuid'])

        """
        from ibeis.control import _sql_helpers
        from ibeis.control import SQLDatabaseControl as sqldbc
        from ibeis.control import DB_SCHEMA
        # Before load, ensure database has been backed up for the day
        if not ut.get_argflag('--nobackup'):
            try:
                _sql_helpers.ensure_daily_database_backup(ibs.get_ibsdir(),
                                                          ibs.sqldb_fname,
                                                          ibs.backupdir)
            except IOError as ex:
                ut.printex(ex, (
                    'Failed making daily backup. '
                    'Run with --nobackup to disable'))
                raise
        # IBEIS SQL State Database
        #ibs.db_version_expected = '1.1.1'
        if request_dbversion is None:
            ibs.db_version_expected = '1.4.5'
        else:
            ibs.db_version_expected = request_dbversion
        # TODO: add this functionality to SQLController
        new_version, new_fname = sqldbc.dev_test_new_schema_version(
            ibs.get_dbname(), ibs.get_ibsdir(),
            ibs.sqldb_fname, ibs.db_version_expected, version_next='1.4.5')
        ibs.db_version_expected = new_version
        ibs.sqldb_fname = new_fname
        ibs.db = sqldbc.SQLDatabaseController(ibs.get_ibsdir(), ibs.sqldb_fname,
                                              text_factory=const.__STR__,
                                              inmemory=False, )
        # Ensure correct schema versions
        _sql_helpers.ensure_correct_version(
            ibs,
            ibs.db,
            ibs.db_version_expected,
            DB_SCHEMA,
            verbose=ut.VERBOSE,
        )
        #print(ibs.sqldbcache_fname)
        #import sys
        #sys.exit(1)

    #@ut.indent_func
    def _init_sqldbcache(ibs):
        """ Need to reinit this sometimes if cache is ever deleted """
        from ibeis.control import _sql_helpers
        from ibeis.control import SQLDatabaseControl as sqldbc
        from ibeis.control import DBCACHE_SCHEMA
        # IBEIS SQL Features & Chips database
        ibs.dbcache_version_expected = '1.0.4'
        # Test a new schema if developer
        new_version, new_fname = sqldbc.dev_test_new_schema_version(
            ibs.get_dbname(), ibs.get_cachedir(),
            ibs.sqldbcache_fname, ibs.dbcache_version_expected,
            version_next='1.0.4')
        ibs.dbcache_version_expected = new_version
        ibs.sqldbcache_fname = new_fname
        # Create cache sql database
        ibs.dbcache = sqldbc.SQLDatabaseController(
            ibs.get_cachedir(), ibs.sqldbcache_fname,
            text_factory=const.__STR__)
        _sql_helpers.ensure_correct_version(
            ibs,
            ibs.dbcache,
            ibs.dbcache_version_expected,
            DBCACHE_SCHEMA,
            dobackup=False,  # Everything in dbcache can be regenerated.
            verbose=ut.VERBOSE,
        )

    def _close_sqldbcache(ibs):
        ibs.dbcache.close()
        ibs.dbcache = None

    def disconnect_sqldatabase(ibs):
        print('disconnecting from sql database')
        ibs.dbcache.close()
        ibs.dbcache = None
        ibs.db.close()
        ibs.db = None

    @default_decorator
    def clone_handle(ibs, **kwargs):
        ibs2 = IBEISController(dbdir=ibs.get_dbdir(), ensure=False)
        if len(kwargs) > 0:
            ibs2.update_query_cfg(**kwargs)
        #if ibs.qreq is not None:
        #    ibs2._prep_qreq(ibs.qreq.qaids, ibs.qreq.daids)
        return ibs2

    @default_decorator
    def backup_database(ibs):
        from ibeis.control import _sql_helpers
        _sql_helpers.database_backup(ibs.get_ibsdir(), ibs.sqldb_fname,
                                     ibs.backupdir)

    @default_decorator
    def _send_wildbook_request(ibs, wbaddr, payload=None):
        if wbaddr is None:
            return
        try:
            if payload is None:
                response = requests.get(wbaddr)
            else:
                response = requests.post(wbaddr, data=payload)
        # except requests.MissingSchema:
        #     print('[ibs._send_wildbook_request] Invalid URL: %r' % wbaddr)
        #     return None
        except requests.ConnectionError:
            print('[ibs.wb_reqst] Could not connect to Wildbook server at %r' %
                  wbaddr)
            return None
        return response

    @default_decorator
    def _init_dirs(ibs, dbdir=None, dbname='testdb_1',
                   workdir='~/ibeis_workdir', ensure=True):
        """
        Define ibs directories
        """
        PATH_NAMES = const.PATH_NAMES
        REL_PATHS = const.REL_PATHS

        if not ut.QUIET:
            print('[ibs._init_dirs] ibs.dbdir = %r' % dbdir)
        if dbdir is not None:
            workdir, dbname = split(dbdir)
        ibs.workdir  = ut.truepath(workdir)
        ibs.dbname = dbname
        ibs.sqldb_fname = PATH_NAMES.sqldb
        ibs.sqldbcache_fname = PATH_NAMES.sqldbcache

        # Make sure you are not nesting databases
        assert PATH_NAMES._ibsdb != ut.dirsplit(ibs.workdir), \
            'cannot work in _ibsdb internals'
        assert PATH_NAMES._ibsdb != dbname,\
            'cannot create db in _ibsdb internals'
        ibs.dbdir    = join(ibs.workdir, ibs.dbname)
        # All internal paths live in <dbdir>/_ibsdb
        # TODO: constantify these
        # so non controller objects (like in score normalization) have access to
        # these
        ibs._ibsdb      = join(ibs.dbdir, REL_PATHS._ibsdb)
        ibs.trashdir    = join(ibs.dbdir, REL_PATHS.trashdir)
        ibs.cachedir    = join(ibs.dbdir, REL_PATHS.cache)
        ibs.backupdir   = join(ibs.dbdir, REL_PATHS.backups)
        ibs.chipdir     = join(ibs.dbdir, REL_PATHS.chips)
        ibs.imgdir      = join(ibs.dbdir, REL_PATHS.images)
        ibs.treesdir    = join(ibs.dbdir, REL_PATHS.trees)
        ibs.uploadsdir  = join(ibs.dbdir, REL_PATHS.uploads)
        # All computed dirs live in <dbdir>/_ibsdb/_ibeis_cache
        ibs.thumb_dpath = join(ibs.dbdir, REL_PATHS.thumbs)
        ibs.flanndir    = join(ibs.dbdir, REL_PATHS.flann)
        ibs.qresdir     = join(ibs.dbdir, REL_PATHS.qres)
        ibs.bigcachedir = join(ibs.dbdir, REL_PATHS.bigcache)
        ibs.distinctdir = join(ibs.dbdir, REL_PATHS.distinctdir)
        if ensure:
            ibs.ensure_directories()
        assert dbdir is not None, 'must specify database directory'

    def ensure_directories(ibs):
        """
        Makes sure the core directores for the controller exist
        """
        _verbose = ut.VERBOSE
        ut.ensuredir(ibs._ibsdb)
        ut.ensuredir(ibs.cachedir,    verbose=_verbose)
        ut.ensuredir(ibs.backupdir,   verbose=_verbose)
        ut.ensuredir(ibs.workdir,     verbose=_verbose)
        ut.ensuredir(ibs.imgdir,      verbose=_verbose)
        ut.ensuredir(ibs.chipdir,     verbose=_verbose)
        ut.ensuredir(ibs.flanndir,    verbose=_verbose)
        ut.ensuredir(ibs.qresdir,     verbose=_verbose)
        ut.ensuredir(ibs.bigcachedir, verbose=_verbose)
        ut.ensuredir(ibs.thumb_dpath, verbose=_verbose)
        ut.ensuredir(ibs.distinctdir, verbose=_verbose)
        ibs.get_smart_patrol_dir()

    #
    #
    #--------------
    # --- DIRS ----
    #--------------

    @register_api('/api/core/dbname/', methods=['GET'])
    def get_dbname(ibs):
        """
        Returns:
            list_ (list): database name

        RESTful:
            Method: GET
            URL:    /api/core/dbname/
        """
        return ibs.dbname

    def get_logdir(ibs):
        return ut.get_logging_dir(appname='ibeis')

    def get_dbdir(ibs):
        """
        Returns:
            list_ (list): database dir with ibs internal directory """
        #return join(ibs.workdir, ibs.dbname)
        return ibs.dbdir

    def get_db_core_path(ibs):
        """
        Returns:
            path (str): path of the sqlite3 core database file """
        return ibs.db.fpath

    def get_db_cache_path(ibs):
        """
        Returns:
            path (str): path of the sqlite3 cache database file """
        return ibs.dbcache.fpath

    def get_trashdir(ibs):
        return ibs.trashdir

    def get_ibsdir(ibs):
        """
        Returns:
            list_ (list): ibs internal directory """
        return ibs._ibsdb

    def get_chipdir(ibs):
        return ibs.chipdir

    def get_probchip_dir(ibs):
        return join(ibs.get_cachedir(), 'prob_chips')

    def get_fig_dir(ibs):
        """
        Returns:
            list_ (list): ibs internal directory """
        return join(ibs._ibsdb, 'figures')

    def get_imgdir(ibs):
        """
        Returns:
            list_ (list): ibs internal directory """
        return ibs.imgdir

    def get_treesdir(ibs):
        """
        Returns:
            list_ (list): ibs internal directory """
        return ibs.treesdir

    def get_uploadsdir(ibs):
        """
        Returns:
            list_ (list): ibs internal directory """
        return ibs.uploadsdir

    def get_thumbdir(ibs):
        """
        Returns:
            list_ (list): database directory where thumbnails are cached """
        return ibs.thumb_dpath

    def get_workdir(ibs):
        """
        Returns:
            list_ (list): directory where databases are saved to """
        return ibs.workdir

    def get_cachedir(ibs):
        """
        Returns:
            list_ (list): database directory of all cached files """
        return ibs.cachedir

    def get_match_thumbdir(ibs):
        match_thumb_dir = ut.unixjoin(ibs.get_cachedir(), 'match_thumbs')
        ut.ensuredir(match_thumb_dir)
        return match_thumb_dir

    def get_ibeis_resource_dir(ibs):
        """ returns the global resource dir in .config or AppData or whatever """
        resource_dir = sysres.get_ibeis_resource_dir()
        return resource_dir

    def get_global_species_scorenorm_cachedir(ibs, species_text, ensure=True):
        """

        Args:
            species_text (str):
            ensure       (bool):

        Returns:
            str: species_cachedir

        CommandLine:
            python -m ibeis.control.IBEISControl --test-get_global_species_scorenorm_cachedir

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis  # NOQA
            >>> ibs = ibeis.opendb('testdb1')
            >>> species_text = ibeis.const.Species.ZEB_GREVY
            >>> ensure = True
            >>> species_cachedir = ibs.get_global_species_scorenorm_cachedir(species_text, ensure)
            >>> resourcedir = ibs.get_ibeis_resource_dir()
            >>> result = ut.relpath_unix(species_cachedir, resourcedir)
            >>> print(result)
            scorenorm/zebra_grevys

        """
        scorenorm_cachedir = join(ibs.get_ibeis_resource_dir(),
                                  const.PATH_NAMES.scorenormdir)
        species_cachedir = join(scorenorm_cachedir, species_text)
        if ensure:
            ut.ensurepath(scorenorm_cachedir)
            ut.ensuredir(species_cachedir)
        return species_cachedir

    def get_local_species_scorenorm_cachedir(ibs, species_text, ensure=True):
        """
        """
        scorenorm_cachedir = join(ibs.get_cachedir(),
                                  const.PATH_NAMES.scorenormdir)
        species_cachedir = join(scorenorm_cachedir, species_text)
        if ensure:
            ut.ensuredir(scorenorm_cachedir)
            ut.ensuredir(species_cachedir)
        return species_cachedir

    def get_global_distinctiveness_modeldir(ibs, ensure=True):
        """
        Returns:
            global_distinctdir (str): ibs internal directory
        """
        global_distinctdir = sysres.get_global_distinctiveness_modeldir(ensure=ensure)
        return global_distinctdir

    def get_local_distinctiveness_modeldir(ibs):
        """
        Returns:
            distinctdir (str): ibs internal directory """
        return ibs.distinctdir

    def get_detect_modeldir(ibs):
        return join(sysres.get_ibeis_resource_dir(), 'detectmodels')

    def get_detectimg_cachedir(ibs):
        """
        Returns:
            detectimgdir (str): database directory of image resized for
                detections
        """
        return join(ibs.cachedir, const.PATH_NAMES.detectimg)

    def get_flann_cachedir(ibs):
        """
        Returns:
            flanndir (str): database directory where the FLANN KD-Tree is
                stored
        """
        return ibs.flanndir

    def get_qres_cachedir(ibs):
        """
        Returns:
            qresdir (str): database directory where query results are stored """
        return ibs.qresdir

    def get_neighbor_cachedir(ibs):
        neighbor_cachedir = ut.unixjoin(ibs.get_cachedir(), 'neighborcache2')
        return neighbor_cachedir

    def get_big_cachedir(ibs):
        """
        Returns:
            bigcachedir (str): database directory where aggregate results are
                stored
        """
        return ibs.bigcachedir

    def get_smart_patrol_dir(ibs, ensure=True):
        """
        Args:
            ensure (bool):

        Returns:
            str smart_patrol_dpath

        CommandLine:
            python -m ibeis.control.IBEISControl --test-get_smart_patrol_dir

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis
            >>> # build test data
            >>> ibs = ibeis.opendb('testdb1')
            >>> ensure = True
            >>> # execute function
            >>> smart_patrol_dpath = ibs.get_smart_patrol_dir(ensure)
            >>> # verify results
            >>> ut.assertpath(smart_patrol_dpath, verbose=True)
        """
        smart_patrol_dpath = join(ibs.dbdir, const.PATH_NAMES.smartpatrol)
        if ensure:
            ut.ensuredir(smart_patrol_dpath)
        return smart_patrol_dpath

    #
    #
    #------------------
    # --- DETECTION ---
    #------------------

    @default_decorator
    @accessor_decors.getter_1to1
    @register_api('/api/core/detect_random_forest/', methods=['PUT', 'GET'])
    def detect_random_forest(ibs, gid_list, species, **kwargs):
        """
        Runs animal detection in each image. Adds annotations to the database
        as they are found.

        Args:
            gid_list (list): list of image ids to run detection on
            species (str): string text of the species to identify

        Returns:
            aids_list (list): list of lists of annotation ids detected in each
                image

        CommandLine:
            python -m ibeis.control.IBEISControl --test-detect_random_forest --show

        RESTful:
            Method: PUT
            URL:    /api/core/detect_random_forest/

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis
            >>> # build test data
            >>> ibs = ibeis.opendb('testdb1')
            >>> gid_list = ibs.get_valid_gids()[0:2]
            >>> species = ibeis.const.Species.ZEB_PLAIN
            >>> # execute function
            >>> aids_list = ibs.detect_random_forest(gid_list, species)
            >>> # Visualize results
            >>> if ut.show_was_requested():
            >>>     import plottool as pt
            >>>     from ibeis.viz import viz_image
            >>>     for fnum, gid in enumerate(gid_list):
            >>>         viz_image.show_image(ibs, gid, fnum=fnum)
            >>>     pt.show_if_requested()
            >>> # Remove newly detected annotations
            >>> ibs.delete_annots(ut.flatten(aids_list))
        """
        # TODO: Return confidence here as well
        print('[ibs] detecting using random forests')
        from ibeis.model.detect import randomforest  # NOQA
        if isinstance(gid_list, int):
            gid_list = [gid_list]
        print('TYPE:' + str(type(gid_list)))
        print('GID_LIST:' + ut.truncate_str(str(gid_list)))
        detect_gen = randomforest.detect_gid_list_with_species(
            ibs, gid_list, species, **kwargs)
        # ibs.cfg.other_cfg.ensure_attr('detect_add_after', 1)
        # ADD_AFTER_THRESHOLD = ibs.cfg.other_cfg.detect_add_after
        print('TYPE:' + str(type(detect_gen)))
        aids_list = []
        for gid, (gpath, result_list) in zip(gid_list, detect_gen):
            aids = []
            for result in result_list:
                # Ideally, species will come from the detector with confidences
                # that actually mean something
                bbox = (result['xtl'], result['ytl'],
                        result['width'], result['height'])
                (aid,) = ibs.add_annots(
                    [gid], [bbox], notes_list=['rfdetect'],
                    species_list=[species], quiet_delete_thumbs=True,
                    detect_confidence_list=[result['confidence']])
                aids.append(aid)
            aids_list.append(aids)
        return aids_list

    #
    #
    #-----------------------------
    # --- ENCOUNTER CLUSTERING ---
    #-----------------------------

    def _parse_smart_xml(back, xml_path, nTotal, offset=1):
        # Storage for the patrol encounters
        xml_dir, xml_name = split(xml_path)
        encounter_info_list = []
        last_photo_number = None
        last_encounter_info = None
        # Parse the XML file for the information
        patrol_tree = ET.parse(xml_path)
        namespace = '{http://www.smartconservationsoftware.org/xml/1.1/patrol}'
        # Load all waypoint elements
        element = './/%swaypoints' % (namespace, )
        waypoint_list = patrol_tree.findall(element)
        if len(waypoint_list) == 0:
            # raise IOError('There are no observations (waypoints) in this Patrol XML file: %r' % (xml_path, ))
            print('There are no observations (waypoints) in this Patrol XML file: %r' % (xml_path, ))
        for waypoint in waypoint_list:
            # Get the relevant information about the waypoint
            waypoint_id   = int(waypoint.get('id'))
            waypoint_lat  = float(waypoint.get('y'))
            waypoint_lon  = float(waypoint.get('x'))
            waypoint_time = waypoint.get('time')
            waypoint_info = [
                xml_name,
                waypoint_id,
                (waypoint_lat, waypoint_lon),
                waypoint_time,
            ]
            if None in waypoint_info:
                raise IOError('The observation (waypoint) is missing information: %r' % (waypoint_info, ))
            # Get all of the waypoint's observations (we expect only one
            # normally)
            element = './/%sobservations' % (namespace, )
            observation_list = waypoint.findall(element)
            # if len(observation_list) == 0:
            #     raise IOError('There are no observations in this waypoint, waypoint_id: %r' % (waypoint_id, ))
            for observation in observation_list:
                # Filter the observations based on type, we only care
                # about certain types
                categoryKey = observation.attrib['categoryKey']
                if categoryKey.startswith('animals.liveanimals') or categoryKey.startswith('animals.problemanimal'):
                    # Get the photonumber attribute for the waypoint's
                    # observation
                    element = './/%sattributes[@attributeKey="photonumber"]' % (namespace, )
                    photonumber = observation.find(element)
                    if photonumber is not None:
                        element = './/%ssValue' % (namespace, )
                        # Get the value for photonumber
                        sValue  = photonumber.find(element)
                        if sValue is None:
                            raise IOError('The photonumber sValue is missing from photonumber, waypoint_id: %r' % (waypoint_id, ))
                        # Python cast the value
                        try:
                            photo_number = int(float(sValue.text)) - offset
                        except ValueError:
                            # raise IOError('The photonumber sValue is invalid, waypoint_id: %r' % (waypoint_id, ))
                            print('[ibs]     Skipped Invalid Observation with photonumber: %r, waypoint_id: %r' % (sValue.text, waypoint_id, ))
                            continue
                        # Check that the photo_number is within the acceptable bounds
                        if photo_number >= nTotal:
                            raise IOError('The Patrol XML file is looking for images that do not exist (too few images given)')
                        # Keep track of the last waypoint that was processed
                        # becuase we only have photono, which indicates start
                        # indices and doesn't specify the end index.  The
                        # ending index is extracted as the next waypoint's
                        # photonum minus 1.
                        if last_photo_number is not None and last_encounter_info is not None:
                            encounter_info = last_encounter_info + [(last_photo_number, photo_number)]
                            encounter_info_list.append(encounter_info)
                        last_photo_number = photo_number
                        last_encounter_info = waypoint_info
                    else:
                        # raise IOError('The photonumber value is missing from waypoint, waypoint_id: %r' % (waypoint_id, ))
                        print('[ibs]     Skipped Empty Observation with "categoryKey": %r, waypoint_id: %r' % (categoryKey, waypoint_id, ))
                else:
                    print('[ibs]     Skipped Incompatible Observation with "categoryKey": %r, waypoint_id: %r' % (categoryKey, waypoint_id, ))
        # Append the last photo_number
        if last_photo_number is not None and last_encounter_info is not None:
            encounter_info = last_encounter_info + [(last_photo_number, nTotal)]
            encounter_info_list.append(encounter_info)
        return encounter_info_list

    #@ut.indent_func('[ibs.compute_encounters]')
    def compute_encounters_smart(ibs, gid_list, smart_xml_fpath):
        """
        Function to load and process a SMART patrol XML file
        """
        # Get file and copy to ibeis database folder
        xml_dir, xml_name = split(smart_xml_fpath)
        dst_xml_path = join(ibs.get_smart_patrol_dir(), xml_name)
        ut.copy(smart_xml_fpath, dst_xml_path, overwrite=True)
        # Process the XML File
        print('[ibs] Processing Patrol XML file: %r' % (dst_xml_path, ))
        try:
            encounter_info_list = ibs._parse_smart_xml(dst_xml_path, len(gid_list))
        except Exception as e:
            ibs.delete_images(gid_list)
            print('[ibs] ERROR: Parsing Patrol XML file failed, rolling back by deleting %d images...' % (len(gid_list, )))
            raise e
        if len(gid_list) > 0:
            # Sanity check
            assert len(encounter_info_list) > 0, (
                'Trying to added %d images, but the Patrol  XML file has no observations' % (len(gid_list), ))
        # Display the patrol encounters
        for index, encounter_info in enumerate(encounter_info_list):
            smart_xml_fname, smart_waypoint_id, gps, local_time, range_ = encounter_info
            start, end = range_
            gid_list_ = gid_list[start:end]
            print('[ibs]     Found Patrol Encounter: %r' % (encounter_info, ))
            print('[ibs]         GIDs: %r' % (gid_list_, ))
            if len(gid_list_) == 0:
                print('[ibs]         SKIPPING EMPTY ENCOUNTER')
                continue
            # Add the GPS data to the iamges
            gps_list  = [ gps ] * len(gid_list_)
            ibs.set_image_gps(gid_list_, gps_list)
            # Create a new encounter
            enctext = '%s Waypoint %03d' % (xml_name.replace('.xml', ''), index + 1, )
            eid = ibs.add_encounters(enctext)
            # Add images to the encounters
            eid_list = [eid] * len(gid_list_)
            ibs.set_image_eids(gid_list_, eid_list)
            # Set the encounter's smart fields
            ibs.set_encounter_smart_xml_fnames([eid], [smart_xml_fname])
            ibs.set_encounter_smart_waypoint_ids([eid], [smart_waypoint_id])
            # Set the encounter's time based on the images
            unixtime_list = sorted(ibs.get_image_unixtime(gid_list_))
            start_time = unixtime_list[0]
            end_time = unixtime_list[-1]
            ibs.set_encounter_start_time_posix([eid], [start_time])
            ibs.set_encounter_end_time_posix([eid], [end_time])
        # Complete
        print('[ibs] ...Done processing Patrol XML file')

    #@ut.indent_func('[ibs.compute_encounters]')
    def compute_encounters(ibs):
        """
        Clusters ungrouped images into encounters

        CommandLine:
            python -m ibeis.control.IBEISControl --test-compute_encounters

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.control import *  # NOQA
            >>> import ibeis  # NOQA
            >>> # build test data
            >>> ibs = ibeis.opendb('testdb1')
            >>> ibs.compute_encounters()
            >>> ibs.update_special_encounters()
            >>> # Now we want to remove some images from a non-special encounter
            >>> nonspecial_eids = [i for i in ibs.get_valid_eids() if i not in ibs.get_special_eids()]
            >>> images_to_remove = ibs.get_encounter_gids(nonspecial_eids[0:1])[0][0:1]
            >>> ibs.unrelate_images_and_encounters(images_to_remove,nonspecial_eids[0:1] * len(images_to_remove))
            >>> ibs.update_special_encounters()
            >>> ungr_eid = ibs.get_encounter_eids_from_text(const.UNGROUPED_IMAGES_ENCTEXT)
            >>> ungr_gids = ibs.get_encounter_gids([ungr_eid])[0]
            >>> #Now let's make sure that when we recompute encounters, our non-special eid remains the same
            >>> print('PRE COMPUTE: Encounters are %r' % ibs.get_valid_eids())
            >>> print('Containing: %r' % ibs.get_encounter_gids(ibs.get_valid_eids()))
            >>> ibs.compute_encounters()
            >>> print('COMPUTE: New encounters are %r' % ibs.get_valid_eids())
            >>> print('Containing: %r' % ibs.get_encounter_gids(ibs.get_valid_eids()))
            >>> ibs.update_special_encounters()
            >>> print('UPDATE SPECIAL: New encounters are %r' % ibs.get_valid_eids())
            >>> print('Containing: %r' % ibs.get_encounter_gids(ibs.get_valid_eids()))
            >>> assert(images_to_remove[0] not in ibs.get_encounter_gids(nonspecial_eids[0:1])[0])
        """
        from ibeis.model.preproc import preproc_encounter
        print('[ibs] Computing and adding encounters.')
        #gid_list = ibs.get_valid_gids(require_unixtime=False, reviewed=False)
        # only cluster ungrouped images
        gid_list = ibs.get_ungrouped_gids()
        with ut.Timer('computing encounters'):
            flat_eids, flat_gids = preproc_encounter.ibeis_compute_encounters(
                ibs, gid_list)
        valid_eids = ibs.get_valid_eids()
        eid_offset = 0 if len(valid_eids) == 0 else max(valid_eids)
        # This way we can make sure that manually separated encounters
        flat_eids_offset = [eid + eid_offset for eid in flat_eids]
        # remain untouched, and ensure that new encounters are created
        enctext_list = ['Encounter ' + str(eid) for eid in flat_eids_offset]
        #print('enctext_list: %r; flat_gids: %r' % (enctext_list, flat_gids))
        print('[ibs] Finished computing, about to add encounter.')
        ibs.set_image_enctext(flat_gids, enctext_list)
        # HACK TO UPDATE ENCOUNTER POSIX TIMES
        # CAREFUL THIS BLOWS AWAY SMART DATA
        ibs.update_encounter_info(ibs.get_valid_eids())
        print('[ibs] Finished computing and adding encounters.')

    #
    #
    #-----------------------
    # --- IDENTIFICATION ---
    #-----------------------

    @default_decorator
    @register_api('/api/core/get_current_log_text/', methods=['GET'])
    def get_current_log_text(ibs):
        """

        Example:
            >>> # WEB_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import time
            >>> import ibeis
            >>> import requests
            >>> # Start up the web instance
            >>> web_instance = ibeis.opendb_in_background(db='testdb1', web=True, browser=False)
            >>> time.sleep(.5)
            >>> baseurl = 'http://127.0.1.1:5000'
            >>> resp = requests.get(baseurl + '/api/core/get_current_log_text/')
            >>> print(resp)
            >>> web_instance.terminate()
            >>> json_dict = resp.json()
            >>> text = json_dict['response']
            >>> print(text)
        """
        text = ut.get_current_log_text()
        return text

    @default_decorator
    @register_api('/api/core/recognition_query_aids/', methods=['GET'])
    def get_recognition_query_aids(ibs, is_known, species=None):
        """
        DEPCIRATE

        RESTful:
            Method: GET
            URL:    /api/core/recognition_query_aids/
        """
        qaid_list = ibs.get_valid_aids(is_known=is_known, species=species)
        return qaid_list

    @register_api('/api/core/query_chips_simple_dict/', methods=['GET'])
    def query_chips_simple_dict(ibs, *args, **kwargs):
        r"""
        Runs query_chips, but returns a json compatible dictionary

        Args:
            same as query_chips

        RESTful:
            Method: GET
            URL:    /api/core/query_chips_simple_dict/

        SeeAlso:
            query_chips

        CommandLine:
            python -m ibeis.control.IBEISControl --test-query_chips_simple_dict:0
            python -m ibeis.control.IBEISControl --test-query_chips_simple_dict:1

            python -m ibeis.control.IBEISControl --test-query_chips_simple_dict:0 --humpbacks

        Example:
            >>> # WEB_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb(defaultdb='testdb1')
            >>> #qaid = ibs.get_valid_aids()[0:3]
            >>> qaids = ibs.get_valid_aids()
            >>> daids = ibs.get_valid_aids()
            >>> dict_list = ibs.query_chips_simple_dict(qaids, daids, return_cm=True)
            >>> qgids = ibs.get_annot_image_rowids(qaids)
            >>> qnids = ibs.get_annot_name_rowids(qaids)
            >>> for dict_, qgid, qnid in zip(dict_list, qgids, qnids):
            >>>     dict_['qgid'] = qgid
            >>>     dict_['qnid'] = qnid
            >>>     dict_['dgid_list'] = ibs.get_annot_image_rowids(dict_['daid_list'])
            >>>     dict_['dnid_list'] = ibs.get_annot_name_rowids(dict_['daid_list'])
            >>>     dict_['dgname_list'] = ibs.get_image_gnames(dict_['dgid_list'])
            >>>     dict_['qgname'] = ibs.get_image_gnames(dict_['qgid'])
            >>> result  = ut.list_str(dict_list, nl=2, precision=2, hack_liststr=True)
            >>> result = result.replace('u\'', '"').replace('\'', '"')
            >>> print(result)

        Example:
            >>> # WEB_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import time
            >>> import ibeis
            >>> import requests
            >>> # Start up the web instance
            >>> web_instance = ibeis.opendb_in_background(db='testdb1', web=True, browser=False)
            >>> time.sleep(.5)
            >>> baseurl = 'http://127.0.1.1:5000'
            >>> data = dict(qaid_list=[1])
            >>> resp = requests.get(baseurl + '/api/core/query_chips_simple_dict/', data=data)
            >>> print(resp)
            >>> web_instance.terminate()
            >>> json_dict = resp.json()
            >>> cmdict_list = json_dict['response']
            >>> assert 'score_list' in cmdict_list[0]

        """
        kwargs['return_cm_simple_dict'] = True
        return ibs.query_chips(*args, **kwargs)

    @register_api('/api/core/query_chips_dict/', methods=['GET'])
    def query_chips_dict(ibs, *args, **kwargs):
        """
        Runs query_chips, but returns a json compatible dictionary

        RESTful:
            Method: GET
            URL:    /api/core/query_chips_dict/
        """
        kwargs['return_cm_dict'] = True
        return ibs.query_chips(*args, **kwargs)

    @register_api('/api/core/query_chips/', methods=['GET'])
    def query_chips(ibs, qaid_list=None,
                    daid_list=None,
                    cfgdict=None,
                    use_cache=None,
                    use_bigcache=None,
                    qreq_=None,
                    return_request=False,
                    verbose=pipeline.VERB_PIPELINE,
                    save_qcache=None,
                    prog_hook=None,
                    return_cm=False,
                    return_cm_dict=False,
                    return_cm_simple_dict=False,
                    ):
        r"""
        Submits a query request to the hotspotter recognition pipeline. Returns
        a list of QueryResult objects.

        Note:
            In the future the QueryResult objects will be replaced by ChipMatch
            objects

        Args:
            qaid_list (list): a list of annotation ids to be submitted as queries
            daid_list (list): a list of annotation ids used as the database that will be searched
            cfgdict (dict): dictionary of configuration options used to create a new QueryRequest if not already specified
            use_cache (bool):
            use_bigcache (bool):
            qreq_ (QueryRequest): optional, a QueryRequest object that overrides all previous settings
            return_request (bool): returns the request which will be created if one is not already specified
            verbose (bool): default=False, turns on verbose printing
            return_cm (bool): default=False, if true converts QueryResult objects into serializable ChipMatch2 objects (in the future this will be defaulted to True)

        Returns:
            list: a list of QueryResult objects containing the matching annotations, scores, and feature matches

        Returns(2):
            tuple: (qres_list, qreq_) - a list of query results and optionally the QueryRequest object used

        CommandLine:
            python -m ibeis.control.IBEISControl --test-query_chips

        RESTful:
            Method: PUT
            URL:    /api/core/query_chips/

        Example:
            >>> # SLOW_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb('testdb1')
            >>> qaids = ibs.get_valid_aids()[0:1]
            >>> qres = ibs.query_chips(qaids)[0]
            >>> assert qres.qaid == qaids[0]

        """
        # The qaid and daid objects are allowed to be None if qreq_ is
        # specified
        if qaid_list is None:
            qaid_list = qreq_.get_external_qaids()
        if daid_list is None:
            if qreq_ is not None:
                daid_list = qreq_.get_external_daids()
            else:
                daid_list = ibs.get_valid_aids()

        qaid_list, was_scalar = ut.wrap_iterable(qaid_list)

        # Wrapped call to the main entrypoint in the API to the hotspotter
        # pipeline
        _res = ibs._query_chips4(
            qaid_list, daid_list, cfgdict=cfgdict, use_cache=use_cache,
            use_bigcache=use_bigcache, qreq_=qreq_,
            return_request=return_request, verbose=verbose,
            save_qcache=save_qcache,
            prog_hook=prog_hook)
        if return_request:
            qaid2_qres, qreq_ = _res
        else:
            qaid2_qres = _res

        # Return a list of query results instead of that awful dictionary
        # that will be depricated in future version of hotspotter.
        qres_list = [qaid2_qres[qaid] for qaid in qaid_list]

        if return_cm or return_cm_dict or return_cm_simple_dict:
            from ibeis.model.hots import chip_match
            cm_list = [
                chip_match.ChipMatch2.from_qres(qres) for qres in qres_list]
            # Convert to cm_list
            if return_cm_simple_dict:
                qres_list = [cm.to_simple_dict() for cm in cm_list]
            elif return_cm_dict:
                qres_list = [cm.to_dict() for cm in cm_list]
            else:
                qres_list = cm_list

        if was_scalar:
            # hack for scalar input
            assert len(qres_list) == 1
            qres_list = qres_list[0]

        if return_request:
            return qres_list, qreq_
        else:
            return qres_list

    @register_api('/api/core/query_chips4/', methods=['PUT'])
    def _query_chips4(ibs, qaid_list, daid_list,
                      use_cache=None,
                      use_bigcache=None,
                      return_request=False,
                      cfgdict=None,
                      qreq_=None,
                      verbose=pipeline.VERB_PIPELINE,
                      save_qcache=None,
                      prog_hook=None):
        """
        submits a query request
        main entrypoint in the IBIES API to the hotspotter pipeline

        CommandLine:
            python -m ibeis.control.IBEISControl --test-_query_chips4

        RESTful:
            Method: PUT
            URL:    /api/core/query_chips4/

        Example:
            >>> # SLOW_DOCTEST
            >>> #from ibeis.all_imports import *  # NOQA
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> qaid_list = [1]
            >>> daid_list = [1, 2, 3, 4, 5]
            >>> ibs = ibeis.test_main(db='testdb1')
            >>> qres = ibs._query_chips4(qaid_list, daid_list, use_cache=False)[1]

        #>>> qreq_ = mc4.get_ibeis_query_request(ibs, qaid_list, daid_list)
        #>>> qreq_.load_indexer()
        #>>> qreq_.load_query_vectors()
        #>>> qreq = ibs.qreq
        """
        from ibeis.model.hots import match_chips4 as mc4
        # Check fo empty queries
        try:
            assert len(daid_list) > 0, 'there are no database chips'
            assert len(qaid_list) > 0, 'there are no query chips'
        except AssertionError as ex:
            ut.printex(ex, 'Impossible query request', iswarning=True,
                       keys=['qaid_list', 'daid_list'])
            if ut.SUPER_STRICT:
                raise
            qaid2_qres = {qaid: None for qaid in qaid_list}
            if return_request:
                return qaid2_qres, qreq_
            else:
                return qaid2_qres

        # Check for consistency
        if qreq_ is not None:
            ut.assert_lists_eq(
                qreq_.get_external_qaids(), qaid_list,
                'qaids do not agree with qreq_', verbose=True)
            ut.assert_lists_eq(
                qreq_.get_external_daids(), daid_list,
                'daids do not agree with qreq_', verbose=True)

        # Send query to hotspotter (runs the query)
        _res = mc4.submit_query_request(
            ibs,  qaid_list, daid_list, use_cache, use_bigcache,
            return_request=return_request, cfgdict=cfgdict, qreq_=qreq_,
            verbose=verbose, save_qcache=save_qcache, prog_hook=prog_hook)

        if return_request:
            qaid2_qres, qreq_ = _res
            return qaid2_qres, qreq_
        else:
            qaid2_qres = _res
            return qaid2_qres

    #_query_chips = _query_chips3
    _query_chips = _query_chips4

    @default_decorator
    @register_api('/api/core/query_encounter/', methods=['PUT'])
    def query_encounter(ibs, qaid_list, eid, **kwargs):
        """
        _query_chips wrapper

        RESTful:
            Method: PUT
            URL:    /api/core/query_encounter/
        """
        daid_list = ibs.get_encounter_aids(eid)  # encounter database chips
        qaid2_qres = ibs._query_chips4(qaid_list, daid_list, **kwargs)
        # HACK IN ENCOUNTER INFO
        for qres in six.itervalues(qaid2_qres):
            qres.eid = eid
        return qaid2_qres

    @default_decorator
    @register_api('/api/core/query_exemplars/', methods=['PUT'])
    def query_exemplars(ibs, qaid_list, **kwargs):
        """
        Queries vs the exemplars

        RESTful:
            Method: PUT
            URL:    /api/core/query_exemplars/
        """
        daid_list = ibs.get_valid_aids(is_exemplar=True)
        assert len(daid_list) > 0, 'there are no exemplars'
        return ibs._query_chips4(qaid_list, daid_list, **kwargs)

    @default_decorator
    @register_api('/api/core/query_all/', methods=['PUT'])
    def query_all(ibs, qaid_list, **kwargs):
        """
        DEPRICATED

        Queries vs all valid chip ~~the exemplars~~

        RESTful:
            Method: PUT
            URL:    /api/core/query_all/
        """
        daid_list = ibs.get_valid_aids()
        qaid2_qres = ibs._query_chips4(qaid_list, daid_list, **kwargs)
        return qaid2_qres

    @default_decorator
    @register_api('/api/core/has_species_detector/', methods=['GET'])
    def has_species_detector(ibs, species_text):
        """
        TODO: extend to use non-constant species

        RESTful:
            Method: GET
            URL:    /api/core/has_species_detector/
        """
        # FIXME: infer this
        return species_text in const.SPECIES_WITH_DETECTORS

    @default_decorator
    @register_api('/api/core/species_with_detectors/', methods=['GET'])
    def get_species_with_detectors(ibs):
        """
        RESTful:
            Method: GET
            URL:    /api/core/species_with_detectors/
        """
        # FIXME: infer this
        return const.SPECIES_WITH_DETECTORS


if __name__ == '__main__':
    """
    Issue when running on windows:
    python ibeis/control/IBEISControl.py
    python -m ibeis.control.IBEISControl --verbose --very-verbose --veryverbose --nodyn --quietclass

    CommandLine:
        python -m ibeis.control.IBEISControl
        python -m ibeis.control.IBEISControl --allexamples
        python -m ibeis.control.IBEISControl --allexamples --noface --nosrc
    """
    #from ibeis.control import IBEISControl
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
