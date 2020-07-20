# -*- coding: utf-8 -*-
"""
This module contains the definition of IBEISController. This object
allows access to a single database. Construction of this object should be
done using wbia.opendb().

TODO:
    Module Licence and docstring

    load plugin logic:
        - known plugin list - plugin_register.txt / dirs/symlinks in plugin folder
        - disabled flags
        - try import && register
        - except flag errored
        - init db
        - check versioning / update
        - (determine plugin import ordering?)
        - inject and initialize plugins

Note:
    There are functions that are injected into the controller that are not
      defined in this module.
    Functions in the IBEISController have been split up into several
      submodules.
    look at the modules listed in autogenmodname_list to see the full list of
      functions that will be injected into an IBEISController object

    Recently, these functions have been enumerated in
      wbia.control._autogen_explicit_controller.py,
      and explicitly added to the

    controller using subclassing.
    This submodule only provides function headers, the source code still
      resides in the injected modules.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
from wbia import dtool
import atexit
import weakref
import utool as ut
import ubelt as ub
from six.moves import zip
from os.path import join, split
from wbia.init import sysres
from wbia.dbio import ingest_hsdb
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__)

# Import modules which define injectable functions

# tuples represent conditional imports with the flags in the first part of the
# tuple and the modname in the second
AUTOLOAD_PLUGIN_MODNAMES = [
    'wbia.annotmatch_funcs',
    'wbia.tag_funcs',
    'wbia.annots',
    'wbia.images',
    'wbia.other.ibsfuncs',
    'wbia.other.detectfuncs',
    'wbia.other.detectcore',
    'wbia.other.detectgrave',
    'wbia.other.detecttrain',
    'wbia.init.filter_annots',
    'wbia.control.manual_featweight_funcs',
    'wbia.control._autogen_party_funcs',
    'wbia.control.manual_annotmatch_funcs',
    'wbia.control.manual_wbiacontrol_funcs',
    'wbia.control.manual_wildbook_funcs',
    'wbia.control.manual_meta_funcs',
    'wbia.control.manual_lbltype_funcs',  # DEPRICATE
    'wbia.control.manual_lblannot_funcs',  # DEPRICATE
    'wbia.control.manual_lblimage_funcs',  # DEPRICATE
    'wbia.control.manual_image_funcs',
    'wbia.control.manual_imageset_funcs',
    'wbia.control.manual_gsgrelate_funcs',
    'wbia.control.manual_garelate_funcs',
    'wbia.control.manual_annot_funcs',
    'wbia.control.manual_part_funcs',
    'wbia.control.manual_name_funcs',
    'wbia.control.manual_review_funcs',
    'wbia.control.manual_test_funcs',
    'wbia.control.manual_species_funcs',
    'wbia.control.manual_annotgroup_funcs',
    # 'wbia.control.manual_dependant_funcs',
    'wbia.control.manual_chip_funcs',
    'wbia.control.manual_feat_funcs',
    # 'wbia.algo.hots.query_request',
    'wbia.control.docker_control',
    'wbia.web.apis_detect',
    'wbia.web.apis_engine',
    'wbia.web.apis_query',
    'wbia.web.apis_sync',
    'wbia.web.apis',
    'wbia.core_images',
    'wbia.core_annots',
    'wbia.core_parts',
    'wbia.algo.smk.vocab_indexer',
    'wbia.algo.smk.smk_pipeline',
    # (('--no-cnn', '--nocnn'), 'wbia_cnn'),
    (('--no-cnn', '--nocnn'), 'wbia_cnn._plugin'),
    # (('--no-fluke', '--nofluke'), 'ibeis_flukematch.plugin'),
    # (('--no-curvrank', '--nocurvrank'), 'ibeis_curvrank._plugin'),
    # 'wbia_plugin_identification_example',
]

if ut.get_argflag('--deepsense'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-deepsense', '--nodeepsense'), 'wbia_deepsense._plugin'),
    ]

if ut.get_argflag('--finfindr'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-finfindr', '--nofinfindr'), 'wbia_finfindr._plugin'),
    ]

if ut.get_argflag('--kaggle7') or ut.get_argflag('--kaggleseven'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (
            ('--no-kaggle7', '--nokaggle7', '--no-kaggleseven', '--nokaggleseven'),
            'wbia_kaggle7._plugin',
        ),
    ]

if ut.get_argflag('--orient2d'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-2d-orient', '--no2dorient'), 'wbia_2d_orientation._plugin'),
    ]


"""
# Should import
python -c "import wbia"
# Should not import
python -c "import wbia" --no-cnn
UTOOL_NO_CNN=True python -c "import wbia"
"""

for modname in ut.ProgIter(
    AUTOLOAD_PLUGIN_MODNAMES,
    'loading plugins',
    enabled=ut.VERYVERBOSE,
    adjust=False,
    freq=1,
):
    if isinstance(modname, tuple):
        flag, modname = modname
        if ut.get_argflag(flag):
            continue
    try:
        # ut.import_modname(modname)
        ub.import_module_from_name(modname)
    except ImportError:
        if 'wbia_cnn' in modname:
            import warnings

            warnings.warn('Unable to load plugin: {!r}'.format(modname))
        else:
            raise


# NOTE: new plugin code needs to be hacked in here currently
# this is not a long term solution.  THE Long term solution is to get these
# working (which are partially integrated)
#     python -m wbia dev_autogen_explicit_imports
#     python -m wbia dev_autogen_explicit_injects

# Ensure that all injectable modules are imported before constructing the
# class instance

# Explicit Inject Subclass
try:
    if ut.get_argflag('--dyn'):
        raise ImportError
    else:
        """
        python -m wbia dev_autogen_explicit_injects
        """
        from wbia.control import _autogen_explicit_controller

        BASE_CLASS = _autogen_explicit_controller.ExplicitInjectIBEISController
except ImportError:
    BASE_CLASS = object


register_api = controller_inject.get_wbia_flask_api(__name__)


__ALL_CONTROLLERS__ = []  # Global variable containing all created controllers
__IBEIS_CONTROLLER_CACHE__ = {}
CORE_DB_UUID_INIT_API_RULE = '/api/core/db/uuid/init/'


def request_IBEISController(
    dbdir=None,
    ensure=True,
    wbaddr=None,
    verbose=ut.VERBOSE,
    use_cache=True,
    request_dbversion=None,
    request_stagingversion=None,
    force_serial=False,
    asproxy=None,
    check_hsdb=True,
):
    r"""
    Alternative to directory instantiating a new controller object. Might
    return a memory cached object

    Args:
        dbdir     (str): databse directory
        ensure    (bool):
        wbaddr    (None):
        verbose   (bool):
        use_cache (bool): use the global wbia controller cache.
            Make sure this is false if calling from a Thread. (default=True)
        request_dbversion (str): developer flag. Do not use.
        request_stagingversion (str): developer flag. Do not use.

    Returns:
        IBEISController: ibs

    CommandLine:
        python -m wbia.control.IBEISControl --test-request_IBEISController

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> dbdir = 'testdb1'
        >>> ensure = True
        >>> wbaddr = None
        >>> verbose = True
        >>> use_cache = False
        >>> ibs = request_IBEISController(dbdir, ensure, wbaddr, verbose,
        >>>                               use_cache)
        >>> result = str(ibs)
        >>> print(result)
    """
    global __IBEIS_CONTROLLER_CACHE__

    if use_cache and dbdir in __IBEIS_CONTROLLER_CACHE__:
        if verbose:
            print('[request_IBEISController] returning cached controller')
        ibs = __IBEIS_CONTROLLER_CACHE__[dbdir]
        if force_serial:
            assert ibs.force_serial, 'set use_cache=False in wbia.opendb'
    else:
        # Convert hold hotspotter dirs if necessary
        if check_hsdb and ingest_hsdb.check_unconverted_hsdb(dbdir):
            ibs = ingest_hsdb.convert_hsdb_to_wbia(
                dbdir, ensure=ensure, wbaddr=wbaddr, verbose=verbose
            )
        else:
            ibs = IBEISController(
                dbdir=dbdir,
                ensure=ensure,
                wbaddr=wbaddr,
                verbose=verbose,
                force_serial=force_serial,
                request_dbversion=request_dbversion,
                request_stagingversion=request_stagingversion,
            )
        __IBEIS_CONTROLLER_CACHE__[dbdir] = ibs
    return ibs


@atexit.register
def __cleanup():
    """
    prevents flann errors (not for cleaning up individual objects)
    """
    global __ALL_CONTROLLERS__
    global __IBEIS_CONTROLLER_CACHE__
    try:
        del __ALL_CONTROLLERS__
        del __IBEIS_CONTROLLER_CACHE__
    except NameError:
        print('cannot cleanup IBEISController')
        pass


# -----------------
# IBEIS CONTROLLER
# -----------------


@six.add_metaclass(ut.ReloadingMetaclass)
class IBEISController(BASE_CLASS):
    """
    IBEISController docstring

    NameingConventions:
        chip  - cropped region of interest in an image, maps to one animal
        cid   - chip unique id
        gid   - image unique id (could just be the relative file path)
        name  - name unique id
        imgsetid   - imageset unique id
        aid   - region of interest unique id
        annot - an annotation i.e. region of interest for a chip
        theta - angle of rotation for a chip
    """

    # -------------------------------
    # --- CONSTRUCTOR / PRIVATES ---
    # -------------------------------

    @profile
    def __init__(
        ibs,
        dbdir=None,
        ensure=True,
        wbaddr=None,
        verbose=True,
        request_dbversion=None,
        request_stagingversion=None,
        force_serial=None,
    ):
        """ Creates a new IBEIS Controller associated with one database """
        # if verbose and ut.VERBOSE:
        print('\n[ibs.__init__] new IBEISController')

        ibs.dbname = None
        # an dict to hack in temporary state
        ibs.const = const
        ibs.readonly = None
        ibs.depc_image = None
        ibs.depc_annot = None
        ibs.depc_part = None
        # ibs.allow_override = 'override+warn'
        ibs.allow_override = True
        if force_serial is None:
            if ut.get_argflag(('--utool-force-serial', '--force-serial', '--serial')):
                force_serial = True
            else:
                force_serial = not ut.in_main_process()
        if const.CONTAINERIZED:
            force_serial = True
        ibs.force_serial = force_serial
        # observer_weakref_list keeps track of the guibacks connected to this
        # controller
        ibs.observer_weakref_list = []
        # not completely working decorator cache
        ibs.table_cache = None
        ibs._initialize_self()
        ibs._init_dirs(dbdir=dbdir, ensure=ensure)
        # _send_wildbook_request will do nothing if no wildbook address is
        # specified
        ibs._send_wildbook_request(wbaddr)
        ibs._init_sql(
            request_dbversion=request_dbversion,
            request_stagingversion=request_stagingversion,
        )
        ibs._init_config()
        if not ut.get_argflag('--noclean') and not ibs.readonly:
            # ibs._init_burned_in_species()
            ibs._clean_species()
        ibs.job_manager = None

        # Hack for changing the way chips compute
        # by default use serial because warpAffine is weird with multiproc
        is_mac = 'macosx' in ut.get_plat_specifier().lower()
        ibs._parallel_chips = not ibs.force_serial and not is_mac

        ibs.containerized = const.CONTAINERIZED
        ibs.production = const.PRODUCTION

        print('[ibs.__init__] CONTAINERIZED: %s\n' % (ibs.containerized,))
        print('[ibs.__init__] PRODUCTION: %s\n' % (ibs.production,))

        # Hack to store HTTPS flag (deliver secure content in web)
        ibs.https = const.HTTPS

        print('[ibs.__init__] END new IBEISController\n')

    def reset_table_cache(ibs):
        ibs.table_cache = accessor_decors.init_tablecache()

    def clear_table_cache(ibs, tablename=None):
        print('[ibs] clearing table_cache[%r]' % (tablename,))
        if tablename is None:
            ibs.reset_table_cache()
        else:
            try:
                del ibs.table_cache[tablename]
            except KeyError:
                pass

    def show_depc_graph(ibs, depc, reduced=False):
        depc.show_graph(reduced=reduced)

    def show_depc_image_graph(ibs, **kwargs):
        """
        CommandLine:
            python -m wbia.control.IBEISControl --test-show_depc_image_graph --show
            python -m wbia.control.IBEISControl --test-show_depc_image_graph --show --reduced

        Example:
            >>> # SCRIPT
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia  # NOQA
            >>> ibs = wbia.opendb('testdb1')
            >>> reduced = ut.get_argflag('--reduced')
            >>> ibs.show_depc_image_graph(reduced=reduced)
            >>> ut.show_if_requested()
        """
        ibs.show_depc_graph(ibs.depc_image, **kwargs)

    def show_depc_annot_graph(ibs, *args, **kwargs):
        """
        CommandLine:
            python -m wbia.control.IBEISControl --test-show_depc_annot_graph --show
            python -m wbia.control.IBEISControl --test-show_depc_annot_graph --show --reduced

        Example:
            >>> # SCRIPT
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia  # NOQA
            >>> ibs = wbia.opendb('testdb1')
            >>> reduced = ut.get_argflag('--reduced')
            >>> ibs.show_depc_annot_graph(reduced=reduced)
            >>> ut.show_if_requested()
        """
        ibs.show_depc_graph(ibs.depc_annot, *args, **kwargs)

    def show_depc_annot_table_input(ibs, tablename, *args, **kwargs):
        """
        CommandLine:
            python -m wbia.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=vsone
            python -m wbia.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=neighbor_index
            python -m wbia.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=feat_neighbs --testmode

        Example:
            >>> # SCRIPT
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia  # NOQA
            >>> ibs = wbia.opendb('testdb1')
            >>> tablename = ut.get_argval('--tablename')
            >>> ibs.show_depc_annot_table_input(tablename)
            >>> ut.show_if_requested()
        """
        ibs.depc_annot[tablename].show_input_graph()

    def get_cachestats_str(ibs):
        """
        Returns info about the underlying SQL cache memory
        """
        total_size_str = ut.get_object_size_str(
            ibs.table_cache, lbl='size(table_cache): '
        )
        total_size_str = '\nlen(table_cache) = %r' % (len(ibs.table_cache))
        table_size_str_list = [
            ut.get_object_size_str(val, lbl='size(table_cache[%s]): ' % (key,))
            for key, val in six.iteritems(ibs.table_cache)
        ]
        cachestats_str = total_size_str + ut.indentjoin(table_size_str_list, '\n  * ')
        return cachestats_str

    def print_cachestats_str(ibs):
        cachestats_str = ibs.get_cachestats_str()
        print('IBEIS Controller Cache Stats:')
        print(cachestats_str)
        return cachestats_str

    def _initialize_self(ibs):
        """
        Injects code from plugin modules into the controller

        Used in utools auto reload.  Called after reload.
        """
        if ut.VERBOSE:
            print('[ibs] _initialize_self()')
        ibs.reset_table_cache()
        ut.util_class.inject_all_external_modules(
            ibs,
            controller_inject.CONTROLLER_CLASSNAME,
            allow_override=ibs.allow_override,
        )
        assert hasattr(ibs, 'get_database_species'), 'issue with ibsfuncs'
        assert hasattr(ibs, 'get_annot_pair_timedelta'), 'issue with annotmatch_funcs'
        ibs.register_controller()

    def _on_reload(ibs):
        """
        For utools auto reload (rrr).
        Called before reload
        """
        # Reloading breaks flask, turn it off
        controller_inject.GLOBAL_APP_ENABLED = False
        # Only warn on first load. Overrideing while reloading is ok
        ibs.allow_override = True
        ibs.unregister_controller()
        # Reload dependent modules
        ut.reload_injected_modules(controller_inject.CONTROLLER_CLASSNAME)

    def load_plugin_module(ibs, module):
        ut.inject_instance(
            ibs,
            classkey=module.CLASS_INJECT_KEY,
            allow_override=ibs.allow_override,
            strict=False,
            verbose=False,
        )

    # We should probably not implement __del__
    # see: https://docs.python.org/2/reference/datamodel.html#object.__del__
    # def __del__(ibs):
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

    def register_observer(ibs, observer):
        print('[register_observer] Observer registered: %r' % observer)
        observer_weakref = weakref.ref(observer)
        ibs.observer_weakref_list.append(observer_weakref)

    def remove_observer(ibs, observer):
        print('[remove_observer] Observer removed: %r' % observer)
        ibs.observer_weakref_list.remove(observer)

    def notify_observers(ibs):
        print('[notify_observers] Observers (if any) notified')
        for observer_weakref in ibs.observer_weakref_list:
            observer_weakref().notify()

    # ------------

    def _init_rowid_constants(ibs):
        # ADD TO CONSTANTS

        # THIS IS EXPLICIT IN CONST, USE THAT VERSION INSTEAD
        # ibs.UNKNOWN_LBLANNOT_ROWID = const.UNKNOWN_LBLANNOT_ROWID
        # ibs.UNKNOWN_NAME_ROWID     = ibs.UNKNOWN_LBLANNOT_ROWID
        # ibs.UNKNOWN_SPECIES_ROWID  = ibs.UNKNOWN_LBLANNOT_ROWID

        # ibs.MANUAL_CONFIG_SUFFIX = 'MANUAL_CONFIG'
        # ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)
        # duct_tape.fix_compname_configs(ibs)
        # duct_tape.remove_database_slag(ibs)
        # duct_tape.fix_nulled_yaws(ibs)
        lbltype_names = const.KEY_DEFAULTS.keys()
        lbltype_defaults = const.KEY_DEFAULTS.values()
        lbltype_ids = ibs.add_lbltype(lbltype_names, lbltype_defaults)
        ibs.lbltype_ids = dict(zip(lbltype_names, lbltype_ids))

    @profile
    def _init_sql(ibs, request_dbversion=None, request_stagingversion=None):
        """ Load or create sql database """
        from wbia.other import duct_tape  # NOQA

        # LOAD THE DEPENDENCY CACHE BEFORE THE MAIN DATABASE SO THAT ANY UPDATE
        # CALLS TO THE CORE DATABASE WILL HAVE ACCESS TO THE CACHE DATABASES IF
        # THEY ARE NEEDED.  THIS IS A DECISION MADE ON 8/16/16 BY JP AND JC TO
        # ALLOW FOR COLUMN DATA IN THE CORE DATABASE TO BE MIGRATED TO THE CACHE
        # DATABASE DURING A POST UPDATE FUNCTION ROUTINE, WHICH HAS TO BE LOADED
        # FIRST AND DEFINED IN ORDER TO MAKE THE SUBSEQUENT WRITE CALLS TO THE
        # RELEVANT CACHE DATABASE
        ibs._init_depcache()
        ibs._init_sqldbcore(request_dbversion=request_dbversion)
        ibs._init_sqldbstaging(request_stagingversion=request_stagingversion)
        # ibs.db.dump_schema()
        # ibs.db.dump()
        ibs._init_rowid_constants()

    def _needs_backup(ibs):
        needs_backup = not ut.get_argflag('--nobackup')
        if ibs.get_dbname() == 'PZ_MTEST':
            needs_backup = False
        if dtool.sql_control.READ_ONLY:
            needs_backup = False
        return needs_backup

    @profile
    def _init_sqldbcore(ibs, request_dbversion=None):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia  # NOQA
            >>> #ibs = wbia.opendb('PZ_MTEST')
            >>> #ibs = wbia.opendb('PZ_Master0')
            >>> ibs = wbia.opendb('testdb1')
            >>> #ibs = wbia.opendb('PZ_Master0')

        Ignore:
            aid_list = ibs.get_valid_aids()
            #ibs.update_annot_visual_uuids(aid_list)
            vuuid_list = ibs.get_annot_visual_uuids(aid_list)
            aid_list2 =  ibs.get_annot_aids_from_visual_uuid(vuuid_list)
            assert aid_list2 == aid_list
            # v1.3.0 testdb1:264us, PZ_MTEST:3.93ms, PZ_Master0:11.6s
            %timeit ibs.get_annot_aids_from_visual_uuid(vuuid_list)
            # v1.3.1 testdb1:236us, PZ_MTEST:1.83ms, PZ_Master0:140ms

            ibs.print_imageset_table(exclude_columns=['imageset_uuid'])
        """
        from wbia.control import _sql_helpers
        from wbia.control import DB_SCHEMA

        # Before load, ensure database has been backed up for the day
        backup_idx = ut.get_argval('--loadbackup', type_=int, default=None)
        sqldb_fpath = None
        if backup_idx is not None:
            backups = _sql_helpers.get_backup_fpaths(ibs)
            print('backups = %r' % (backups,))
            sqldb_fpath = backups[backup_idx]
            print('CHOSE BACKUP sqldb_fpath = %r' % (sqldb_fpath,))
        if backup_idx is None and ibs._needs_backup():
            try:
                _sql_helpers.ensure_daily_database_backup(
                    ibs.get_ibsdir(), ibs.sqldb_fname, ibs.backupdir
                )
            except IOError as ex:
                ut.printex(
                    ex, ('Failed making daily backup. ' 'Run with --nobackup to disable'),
                )
                import utool

                utool.embed()
                raise
        # IBEIS SQL State Database
        # ibs.db_version_expected = '1.1.1'
        if request_dbversion is None:
            ibs.db_version_expected = '2.0.0'
        else:
            ibs.db_version_expected = request_dbversion
        # TODO: add this functionality to SQLController
        if backup_idx is None:
            new_version, new_fname = dtool.sql_control.dev_test_new_schema_version(
                ibs.get_dbname(),
                ibs.get_ibsdir(),
                ibs.sqldb_fname,
                ibs.db_version_expected,
                version_next='2.0.0',
            )
            ibs.db_version_expected = new_version
            ibs.sqldb_fname = new_fname
        if sqldb_fpath is None:
            assert backup_idx is None
            sqldb_fpath = join(ibs.get_ibsdir(), ibs.sqldb_fname)
            readonly = None
        else:
            readonly = True
        ibs.db = dtool.SQLDatabaseController(
            fpath=sqldb_fpath,
            text_factory=six.text_type,
            inmemory=False,
            readonly=readonly,
            always_check_metadata=False,
        )
        ibs.readonly = ibs.db.readonly

        if backup_idx is None:
            # Ensure correct schema versions
            _sql_helpers.ensure_correct_version(
                ibs,
                ibs.db,
                ibs.db_version_expected,
                DB_SCHEMA,
                verbose=ut.VERBOSE,
                dobackup=not ibs.readonly,
            )
        # import sys
        # sys.exit(1)

    @profile
    def _init_sqldbstaging(ibs, request_stagingversion=None):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia  # NOQA
            >>> #ibs = wbia.opendb('PZ_MTEST')
            >>> #ibs = wbia.opendb('PZ_Master0')
            >>> ibs = wbia.opendb('testdb1')
            >>> #ibs = wbia.opendb('PZ_Master0')

        Ignore:
            aid_list = ibs.get_valid_aids()
            #ibs.update_annot_visual_uuids(aid_list)
            vuuid_list = ibs.get_annot_visual_uuids(aid_list)
            aid_list2 =  ibs.get_annot_aids_from_visual_uuid(vuuid_list)
            assert aid_list2 == aid_list
            # v1.3.0 testdb1:264us, PZ_MTEST:3.93ms, PZ_Master0:11.6s
            %timeit ibs.get_annot_aids_from_visual_uuid(vuuid_list)
            # v1.3.1 testdb1:236us, PZ_MTEST:1.83ms, PZ_Master0:140ms

            ibs.print_imageset_table(exclude_columns=['imageset_uuid'])
        """
        from wbia.control import _sql_helpers
        from wbia.control import STAGING_SCHEMA

        # Before load, ensure database has been backed up for the day
        backup_idx = ut.get_argval('--loadbackup-staging', type_=int, default=None)
        sqlstaging_fpath = None
        if backup_idx is not None:
            backups = _sql_helpers.get_backup_fpaths(ibs)
            print('backups = %r' % (backups,))
            sqlstaging_fpath = backups[backup_idx]
            print('CHOSE BACKUP sqlstaging_fpath = %r' % (sqlstaging_fpath,))
        # HACK
        if backup_idx is None and ibs._needs_backup():
            try:
                _sql_helpers.ensure_daily_database_backup(
                    ibs.get_ibsdir(), ibs.sqlstaging_fname, ibs.backupdir
                )
            except IOError as ex:
                ut.printex(
                    ex, ('Failed making daily backup. ' 'Run with --nobackup to disable'),
                )
                raise
        # IBEIS SQL State Database
        if request_stagingversion is None:
            ibs.staging_version_expected = '1.1.1'
        else:
            ibs.staging_version_expected = request_stagingversion
        # TODO: add this functionality to SQLController
        if backup_idx is None:
            new_version, new_fname = dtool.sql_control.dev_test_new_schema_version(
                ibs.get_dbname(),
                ibs.get_ibsdir(),
                ibs.sqlstaging_fname,
                ibs.staging_version_expected,
                version_next='1.1.1',
            )
            ibs.staging_version_expected = new_version
            ibs.sqlstaging_fname = new_fname
        if sqlstaging_fpath is None:
            assert backup_idx is None
            sqlstaging_fpath = join(ibs.get_ibsdir(), ibs.sqlstaging_fname)
            readonly = None
        else:
            readonly = True
        ibs.staging = dtool.SQLDatabaseController(
            fpath=sqlstaging_fpath,
            text_factory=six.text_type,
            inmemory=False,
            readonly=readonly,
            always_check_metadata=False,
        )
        ibs.readonly = ibs.staging.readonly

        if backup_idx is None:
            # Ensure correct schema versions
            _sql_helpers.ensure_correct_version(
                ibs,
                ibs.staging,
                ibs.staging_version_expected,
                STAGING_SCHEMA,
                verbose=ut.VERBOSE,
            )
        # import sys
        # sys.exit(1)

    @profile
    def _init_depcache(ibs):
        # Initialize dependency cache for images
        image_root_getters = {}
        ibs.depc_image = dtool.DependencyCache(
            root_tablename=const.IMAGE_TABLE,
            default_fname=const.IMAGE_TABLE + '_depcache',
            cache_dpath=ibs.get_cachedir(),
            controller=ibs,
            get_root_uuid=ibs.get_image_uuids,
            root_getters=image_root_getters,
        )
        ibs.depc_image.initialize()

        """ Need to reinit this sometimes if cache is ever deleted """
        # Initialize dependency cache for annotations
        annot_root_getters = {
            'name': ibs.get_annot_names,
            'species': ibs.get_annot_species,
            'yaw': ibs.get_annot_yaws,
            'viewpoint_int': ibs.get_annot_viewpoint_int,
            'viewpoint': ibs.get_annot_viewpoints,
            'bbox': ibs.get_annot_bboxes,
            'verts': ibs.get_annot_verts,
            'image_uuid': lambda aids: ibs.get_image_uuids(
                ibs.get_annot_image_rowids(aids)
            ),
            'theta': ibs.get_annot_thetas,
            'occurrence_text': ibs.get_annot_occurrence_text,
        }
        ibs.depc_annot = dtool.DependencyCache(
            # root_tablename='annot',   # const.ANNOTATION_TABLE
            root_tablename=const.ANNOTATION_TABLE,
            default_fname=const.ANNOTATION_TABLE + '_depcache',
            cache_dpath=ibs.get_cachedir(),
            controller=ibs,
            get_root_uuid=ibs.get_annot_visual_uuids,
            root_getters=annot_root_getters,
        )
        # backwards compatibility
        ibs.depc = ibs.depc_annot
        # TODO: root_uuids should be specified as the
        # base_root_uuid plus a hash of the attributes that matter for the
        # requested computation.
        ibs.depc_annot.initialize()

        # Initialize dependency cache for parts
        part_root_getters = {}
        ibs.depc_part = dtool.DependencyCache(
            root_tablename=const.PART_TABLE,
            default_fname=const.PART_TABLE + '_depcache',
            cache_dpath=ibs.get_cachedir(),
            controller=ibs,
            get_root_uuid=ibs.get_part_uuids,
            root_getters=part_root_getters,
        )
        ibs.depc_part.initialize()

    def _close_depcache(ibs):
        ibs.depc_image.close()
        ibs.depc_image = None
        ibs.depc_annot.close()
        ibs.depc_annot = None
        ibs.depc_part.close()
        ibs.depc_part = None

    def disconnect_sqldatabase(ibs):
        print('disconnecting from sql database')
        ibs._close_depcache()
        ibs.db.close()
        ibs.db = None
        ibs.staging.close()
        ibs.staging = None

    def clone_handle(ibs, **kwargs):
        ibs2 = IBEISController(dbdir=ibs.get_dbdir(), ensure=False)
        if len(kwargs) > 0:
            ibs2.update_query_cfg(**kwargs)
        # if ibs.qreq is not None:
        #    ibs2._prep_qreq(ibs.qreq.qaids, ibs.qreq.daids)
        return ibs2

    def backup_database(ibs):
        from wbia.control import _sql_helpers

        _sql_helpers.database_backup(ibs.get_ibsdir(), ibs.sqldb_fname, ibs.backupdir)
        _sql_helpers.database_backup(
            ibs.get_ibsdir(), ibs.sqlstaging_fname, ibs.backupdir
        )

    def _send_wildbook_request(ibs, wbaddr, payload=None):
        import requests

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
            print('[ibs.wb_reqst] Could not connect to Wildbook server at %r' % wbaddr)
            return None
        return response

    def _init_dirs(
        ibs, dbdir=None, dbname='testdb_1', workdir='~/wbia_workdir', ensure=True
    ):
        """
        Define ibs directories
        """
        PATH_NAMES = const.PATH_NAMES
        REL_PATHS = const.REL_PATHS

        if not ut.QUIET:
            print('[ibs._init_dirs] ibs.dbdir = %r' % dbdir)
        if dbdir is not None:
            workdir, dbname = split(dbdir)
        ibs.workdir = ut.truepath(workdir)
        ibs.dbname = dbname
        ibs.sqldb_fname = PATH_NAMES.sqldb
        ibs.sqlstaging_fname = PATH_NAMES.sqlstaging

        # Make sure you are not nesting databases
        assert PATH_NAMES._ibsdb != ut.dirsplit(
            ibs.workdir
        ), 'cannot work in _ibsdb internals'
        assert PATH_NAMES._ibsdb != dbname, 'cannot create db in _ibsdb internals'
        ibs.dbdir = join(ibs.workdir, ibs.dbname)
        # All internal paths live in <dbdir>/_ibsdb
        # TODO: constantify these
        # so non controller objects (like in score normalization) have access
        # to these
        ibs._ibsdb = join(ibs.dbdir, REL_PATHS._ibsdb)
        ibs.trashdir = join(ibs.dbdir, REL_PATHS.trashdir)
        ibs.cachedir = join(ibs.dbdir, REL_PATHS.cache)
        ibs.backupdir = join(ibs.dbdir, REL_PATHS.backups)
        ibs.logsdir = join(ibs.dbdir, REL_PATHS.logs)
        ibs.chipdir = join(ibs.dbdir, REL_PATHS.chips)
        ibs.imgdir = join(ibs.dbdir, REL_PATHS.images)
        ibs.uploadsdir = join(ibs.dbdir, REL_PATHS.uploads)
        # All computed dirs live in <dbdir>/_ibsdb/_wbia_cache
        ibs.thumb_dpath = join(ibs.dbdir, REL_PATHS.thumbs)
        ibs.flanndir = join(ibs.dbdir, REL_PATHS.flann)
        ibs.qresdir = join(ibs.dbdir, REL_PATHS.qres)
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
        ut.ensuredir(ibs.cachedir, verbose=_verbose)
        ut.ensuredir(ibs.backupdir, verbose=_verbose)
        ut.ensuredir(ibs.logsdir, verbose=_verbose)
        ut.ensuredir(ibs.workdir, verbose=_verbose)
        ut.ensuredir(ibs.imgdir, verbose=_verbose)
        ut.ensuredir(ibs.chipdir, verbose=_verbose)
        ut.ensuredir(ibs.flanndir, verbose=_verbose)
        ut.ensuredir(ibs.qresdir, verbose=_verbose)
        ut.ensuredir(ibs.bigcachedir, verbose=_verbose)
        ut.ensuredir(ibs.thumb_dpath, verbose=_verbose)
        ut.ensuredir(ibs.distinctdir, verbose=_verbose)
        ibs.get_smart_patrol_dir()

    # --------------
    # --- DIRS ----
    # --------------

    @register_api('/api/core/db/name/', methods=['GET'])
    def get_dbname(ibs):
        """
        Returns:
            list_ (list): database name

        RESTful:
            Method: GET
            URL:    /api/core/db/name/
        """
        return ibs.dbname

    def get_db_name(ibs):
        """ Alias for ibs.get_dbname(). """
        return ibs.get_dbname()

    @register_api(CORE_DB_UUID_INIT_API_RULE, methods=['GET'])
    def get_db_init_uuid(ibs):
        """
        Returns:
            UUID: The SQLDatabaseController's initialization UUID

        RESTful:
            Method: GET
            URL:    /api/core/db/uuid/init/
        """
        return ibs.db.get_db_init_uuid()

    def get_logdir_local(ibs):
        return ibs.logsdir

    def get_logdir_global(ibs, local=False):
        if const.CONTAINERIZED:
            return ibs.get_logdir_local()
        else:
            return ut.get_logging_dir(appname='wbia')

    def get_dbdir(ibs):
        """ database dir with ibs internal directory """
        return ibs.dbdir

    def get_db_core_path(ibs):
        return ibs.db.fpath

    def get_db_staging_path(ibs):
        return ibs.staging.fpath

    def get_db_cache_path(ibs):
        return ibs.dbcache.fpath

    def get_shelves_path(ibs):
        engine_slot = const.ENGINE_SLOT
        engine_slot = str(engine_slot).lower()
        if engine_slot in ['none', 'null', '1', 'default']:
            engine_shelve_dir = 'engine_shelves'
        else:
            engine_shelve_dir = 'engine_shelves_%s' % (engine_slot,)
        return join(ibs.get_cachedir(), engine_shelve_dir)

    def get_trashdir(ibs):
        return ibs.trashdir

    def get_ibsdir(ibs):
        """ ibs internal directory """
        return ibs._ibsdb

    def get_chipdir(ibs):
        return ibs.chipdir

    def get_probchip_dir(ibs):
        return join(ibs.get_cachedir(), 'prob_chips')

    def get_fig_dir(ibs):
        """ ibs internal directory """
        return join(ibs._ibsdb, 'figures')

    def get_imgdir(ibs):
        """ ibs internal directory """
        return ibs.imgdir

    def get_uploadsdir(ibs):
        """ ibs internal directory """
        return ibs.uploadsdir

    def get_thumbdir(ibs):
        """ database directory where thumbnails are cached """
        return ibs.thumb_dpath

    def get_workdir(ibs):
        """ directory where databases are saved to """
        return ibs.workdir

    def get_cachedir(ibs):
        """ database directory of all cached files """
        return ibs.cachedir

    def get_match_thumbdir(ibs):
        match_thumb_dir = ut.unixjoin(ibs.get_cachedir(), 'match_thumbs')
        ut.ensuredir(match_thumb_dir)
        return match_thumb_dir

    def get_wbia_resource_dir(ibs):
        """ returns the global resource dir in .config or AppData or whatever """
        resource_dir = sysres.get_wbia_resource_dir()
        return resource_dir

    def get_detect_modeldir(ibs):
        return join(sysres.get_wbia_resource_dir(), 'detectmodels')

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
            qresdir (str): database directory where query results are stored
        """
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
            python -m wbia.control.IBEISControl --test-get_smart_patrol_dir

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia
            >>> # build test data
            >>> ibs = wbia.opendb('testdb1')
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

    # ------------------
    # --- WEB CORE ----
    # ------------------

    @register_api('/log/current/', methods=['GET'])
    def get_current_log_text(ibs):
        r"""
        CommandLine:
            python -m wbia.control.IBEISControl --exec-get_current_log_text
            python -m wbia.control.IBEISControl --exec-get_current_log_text --domain http://52.33.105.88

        Example:
            >>> # xdoctest: +REQUIRES(--web)
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia
            >>> import wbia.web
            >>> with wbia.opendb_bg_web('testdb1', start_job_queue=False, managed=True) as web_ibs:
            ...     resp = web_ibs.send_wbia_request('/log/current/', 'get')
            >>> print('\n-------Logs ----: \n' )
            >>> print(resp)
            >>> print('\nL____ END LOGS ___\n')
        """
        text = ut.get_current_log_text()
        return text

    @register_api('/api/core/db/info/', methods=['GET'])
    def get_dbinfo(ibs):
        from wbia.other import dbinfo

        locals_ = dbinfo.get_dbinfo(ibs)
        return locals_['info_str']
        # return ut.repr2(dbinfo.get_dbinfo(ibs), nl=1)['infostr']

    # --------------
    # --- MISC ----
    # --------------

    def copy_database(ibs, dest_dbdir):
        # TODO: rectify with rsync, script, and merge script.
        from wbia.init import sysres

        sysres.copy_wbiadb(ibs.get_dbdir(), dest_dbdir)

    def dump_database_csv(ibs):
        dump_dir = join(ibs.get_dbdir(), 'CSV_DUMP')
        ibs.db.dump_tables_to_csv(dump_dir=dump_dir)
        ibs.db.dump_to_fpath(dump_fpath=join(dump_dir, '_ibsdb.dump'))

    def get_database_icon(ibs, max_dsize=(None, 192), aid=None):
        r"""
        Args:
            max_dsize (tuple): (default = (None, 192))

        Returns:
            None: None

        CommandLine:
            python -m wbia.control.IBEISControl --exec-get_database_icon --show
            python -m wbia.control.IBEISControl --exec-get_database_icon --show --db Oxford

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.control.IBEISControl import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb(defaultdb='testdb1')
            >>> icon = ibs.get_database_icon()
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> pt.imshow(icon)
            >>> ut.show_if_requested()
        """
        # if ibs.get_dbname() == 'Oxford':
        #    pass
        # else:
        import vtool as vt

        if hasattr(ibs, 'force_icon_aid'):
            aid = ibs.force_icon_aid
        if aid is None:
            species = ibs.get_primary_database_species()
            # Use a url to get the icon
            url = {
                ibs.const.TEST_SPECIES.GIR_MASAI: 'http://i.imgur.com/tGDVaKC.png',
                ibs.const.TEST_SPECIES.ZEB_PLAIN: 'http://i.imgur.com/2Ge1PRg.png',
                ibs.const.TEST_SPECIES.ZEB_GREVY: 'http://i.imgur.com/PaUT45f.png',
            }.get(species, None)
            if url is not None:
                icon = vt.imread(ut.grab_file_url(url), orient='auto')
            else:
                # HACK: (this should probably be a db setting)
                # use an specific aid to get the icon
                aid = {'Oxford': 73, 'seaturtles': 37}.get(ibs.get_dbname(), None)
                if aid is None:
                    # otherwise just grab a random aid
                    aid = ibs.get_valid_aids()[0]
        if aid is not None:
            icon = ibs.get_annot_chips(aid)
        icon = vt.resize_to_maxdims(icon, max_dsize)
        return icon

    def _custom_ibsstr(ibs):
        # typestr = ut.type_str(type(ibs)).split('.')[-1]
        typestr = ibs.__class__.__name__
        dbname = ibs.get_dbname()
        # hash_str = hex(id(ibs))
        # ibsstr = '<%s(%s) at %s>' % (typestr, dbname, hash_str, )
        hash_str = ibs.get_db_init_uuid()
        ibsstr = '<%s(%s) with UUID %s>' % (typestr, dbname, hash_str,)
        return ibsstr

    def __str__(ibs):
        return ibs._custom_ibsstr()

    def __repr__(ibs):
        return ibs._custom_ibsstr()

    def __getstate__(ibs):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> import wbia
            >>> from six.moves import cPickle as pickle
            >>> ibs = wbia.opendb('testdb1')
            >>> ibs_dump = pickle.dumps(ibs)
            >>> ibs2 = pickle.loads(ibs_dump)
        """
        # Hack to allow for wbia objects to be pickled
        state = {
            'dbdir': ibs.get_dbdir(),
            'machine_name': ut.get_computer_name(),
        }
        return state

    def __setstate__(ibs, state):
        # Hack to allow for wbia objects to be pickled
        import wbia

        dbdir = state['dbdir']
        machine_name = state.pop('machine_name')
        try:
            assert (
                machine_name == ut.get_computer_name()
            ), 'wbia objects can only be picked and unpickled on the same machine'
        except AssertionError as ex:
            iswarning = ut.checkpath(dbdir)
            ut.printex(ex, iswarning=iswarning)
            if not iswarning:
                raise
        ibs2 = wbia.opendb(dbdir=dbdir, web=False)
        ibs.__dict__.update(**ibs2.__dict__)

    def predict_ws_injury_interim_svm(ibs, aids):
        from wbia.scripts import classify_shark

        return classify_shark.predict_ws_injury_interim_svm(ibs, aids)

    def get_web_port_via_scan(
        ibs, url_base='127.0.0.1', port_base=5000, scan_limit=100, verbose=True
    ):
        import requests

        api_rule = CORE_DB_UUID_INIT_API_RULE
        target_uuid = ibs.get_db_init_uuid()
        for candidate_port in range(port_base, port_base + scan_limit + 1):
            candidate_url = 'http://%s:%s%s' % (url_base, candidate_port, api_rule)
            try:
                response = requests.get(candidate_url)
            except (requests.ConnectionError):
                if verbose:
                    print('Failed to find IA server at %s' % (candidate_url,))
                continue
            print('Found IA server at %s' % (candidate_url,))
            try:
                response = ut.from_json(response.text)
                candidate_uuid = response.get('response')
                assert candidate_uuid == target_uuid
                return candidate_port
            except (AssertionError):
                if verbose:
                    print('Invalid response from IA server at %s' % (candidate_url,))
                continue

        return None


if __name__ == '__main__':
    """
    Issue when running on windows:
    python wbia/control/IBEISControl.py
    python -m wbia.control.IBEISControl --verbose --very-verbose --veryverbose --nodyn --quietclass

    CommandLine:
        python -m wbia.control.IBEISControl
        python -m wbia.control.IBEISControl --allexamples
        python -m wbia.control.IBEISControl --allexamples --noface --nosrc
    """
    # from wbia.control import IBEISControl
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
