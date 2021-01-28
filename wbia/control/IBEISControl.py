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
import logging
import six
from wbia import dtool
import atexit
import weakref
import utool as ut
import ubelt as ub
from six.moves import zip
from os.path import join, split
from wbia.init import sysres
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject
from wbia.dtool.dump import dump
from pathlib import Path


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


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
    'wbia.research.metrics',
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
    (('--no-cnn', '--nocnn'), 'wbia_cnn._plugin'),
]


if ut.get_argflag('--flukematch'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-flukematch', '--noflukematch'), 'wbia_flukematch.plugin'),
    ]

if ut.get_argflag('--curvrank'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-curvrank', '--nocurvrank'), 'wbia_curvrank._plugin'),
    ]

if ut.get_argflag('--curvrank-v2'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-curvrank-v2', '--nocurvrankv2'), 'wbia_curvrank_v2._plugin'),
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


if ut.get_argflag('--orient'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-orient', '--noorient'), 'wbia_orientation._plugin'),
    ]

if ut.get_argflag('--pie'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-pie', '--nopie'), 'wbia_pie._plugin'),
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
        >>> from wbia.init.sysres import get_workdir
        >>> dbdir = '/'.join([get_workdir(), 'testdb1'])
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
    dbdir = str(dbdir)

    if use_cache and dbdir in __IBEIS_CONTROLLER_CACHE__:
        if verbose:
            logger.info('[request_IBEISController] returning cached controller')
        ibs = __IBEIS_CONTROLLER_CACHE__[dbdir]
        if force_serial:
            assert ibs.force_serial, 'set use_cache=False in wbia.opendb'
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
        logger.info('cannot cleanup IBEISController')
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
        self,
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
        logger.info('\n[ibs.__init__] new IBEISController')

        self.dbname = None
        # an dict to hack in temporary state
        self.const = const
        self.readonly = None
        self.depc_image = None
        self.depc_annot = None
        self.depc_part = None
        # self.allow_override = 'override+warn'
        self.allow_override = True
        if force_serial is None:
            if ut.get_argflag(('--utool-force-serial', '--force-serial', '--serial')):
                force_serial = True
            else:
                force_serial = not ut.in_main_process()
        # if const.CONTAINERIZED:
        #     force_serial = True
        self.force_serial = force_serial
        # observer_weakref_list keeps track of the guibacks connected to this
        # controller
        self.observer_weakref_list = []
        # not completely working decorator cache
        self.table_cache = None
        self._initialize_self()
        self._init_dirs(dbdir=dbdir, ensure=ensure)

        # Set the base URI to be used for all database connections
        self.__init_base_uri()

        # _send_wildbook_request will do nothing if no wildbook address is
        # specified
        self._send_wildbook_request(wbaddr)
        self._init_sql(
            request_dbversion=request_dbversion,
            request_stagingversion=request_stagingversion,
        )
        self._init_config()
        if not ut.get_argflag('--noclean') and not self.readonly:
            # self._init_burned_in_species()
            self._clean_species()
        self.job_manager = None

        # Hack for changing the way chips compute
        # by default use serial because warpAffine is weird with multiproc
        is_mac = 'macosx' in ut.get_plat_specifier().lower()
        self._parallel_chips = not self.force_serial and not is_mac

        self.containerized = const.CONTAINERIZED
        self.production = const.PRODUCTION

        logger.info('[ibs.__init__] CONTAINERIZED: %s\n' % (self.containerized,))
        logger.info('[ibs.__init__] PRODUCTION: %s\n' % (self.production,))

        # Hack to store HTTPS flag (deliver secure content in web)
        self.https = const.HTTPS

        logger.info('[ibs.__init__] END new IBEISController\n')

    def __init_base_uri(self) -> None:
        """Initialize the base URI that is used for all database connections.
        This sets the ``_base_uri`` attribute.
        This influences the ``is_using_postgres`` property.

        One of the following conditions is met in order to set the uri value:

        - ``--db-uri`` is set to a Postgres URI on the commandline
        - only db-dir is set, and thus we assume a sqlite connection

        """
        self._is_using_postgres_db = False

        uri = sysres.get_wbia_db_uri(self.dbdir)
        if uri:
            if not uri.startswith('postgresql://'):
                raise RuntimeError(
                    "invalid use of '--db-uri'; only supports postgres uris; "
                    f"uri = '{uri}'"
                )
            # Capture that we are using postgres
            self._is_using_postgres_db = True
        else:
            # Assume a sqlite database
            uri = f'sqlite:///{self.get_ibsdir()}'
        self._base_uri = uri

    @property
    def is_using_postgres_db(self) -> bool:
        """Indicates whether this controller is using postgres as the database"""
        return self._is_using_postgres_db

    @property
    def base_uri(self):
        """Base database URI without a specific database name"""
        return self._base_uri

    def make_cache_db_uri(self, name):
        """Given a name of the cache produce a database connection URI"""
        if self.is_using_postgres_db:
            # When using postgres, the base-uri is a connection to a single database
            # that is used for all database needs and scoped using namespace schemas.
            return self._base_uri
        return f'sqlite:///{self.get_cachedir()}/{name}.sqlite'

    def reset_table_cache(self):
        self.table_cache = accessor_decors.init_tablecache()

    def clear_table_cache(self, tablename=None):
        logger.info('[ibs] clearing table_cache[%r]' % (tablename,))
        if tablename is None:
            self.reset_table_cache()
        else:
            try:
                del self.table_cache[tablename]
            except KeyError:
                pass

    def show_depc_graph(self, depc, reduced=False):
        depc.show_graph(reduced=reduced)

    def show_depc_image_graph(self, **kwargs):
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
        self.show_depc_graph(self.depc_image, **kwargs)

    def show_depc_annot_graph(self, *args, **kwargs):
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
        self.show_depc_graph(self.depc_annot, *args, **kwargs)

    def show_depc_annot_table_input(self, tablename, *args, **kwargs):
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
        self.depc_annot[tablename].show_input_graph()

    def get_cachestats_str(self):
        """
        Returns info about the underlying SQL cache memory
        """
        total_size_str = ut.get_object_size_str(
            self.table_cache, lbl='size(table_cache): '
        )
        total_size_str = '\nlen(table_cache) = %r' % (len(self.table_cache))
        table_size_str_list = [
            ut.get_object_size_str(val, lbl='size(table_cache[%s]): ' % (key,))
            for key, val in six.iteritems(self.table_cache)
        ]
        cachestats_str = total_size_str + ut.indentjoin(table_size_str_list, '\n  * ')
        return cachestats_str

    def print_cachestats_str(self):
        cachestats_str = self.get_cachestats_str()
        logger.info('IBEIS Controller Cache Stats:')
        logger.info(cachestats_str)
        return cachestats_str

    def _initialize_self(self):
        """
        Injects code from plugin modules into the controller

        Used in utools auto reload.  Called after reload.
        """
        if ut.VERBOSE:
            logger.info('[ibs] _initialize_self()')
        self.reset_table_cache()
        ut.util_class.inject_all_external_modules(
            self,
            controller_inject.CONTROLLER_CLASSNAME,
            allow_override=self.allow_override,
        )
        assert hasattr(self, 'get_database_species'), 'issue with ibsfuncs'
        assert hasattr(self, 'get_annot_pair_timedelta'), 'issue with annotmatch_funcs'
        self.register_controller()

    def _on_reload(self):
        """
        For utools auto reload (rrr).
        Called before reload
        """
        # Reloading breaks flask, turn it off
        controller_inject.GLOBAL_APP_ENABLED = False
        # Only warn on first load. Overrideing while reloading is ok
        self.allow_override = True
        self.unregister_controller()
        # Reload dependent modules
        ut.reload_injected_modules(controller_inject.CONTROLLER_CLASSNAME)

    def load_plugin_module(self, module):
        ut.inject_instance(
            self,
            classkey=module.CLASS_INJECT_KEY,
            allow_override=self.allow_override,
            strict=False,
            verbose=False,
        )

    # We should probably not implement __del__
    # see: https://docs.python.org/2/reference/datamodel.html#object.__del__
    # def __del__(self):
    #    self.cleanup()

    # ------------
    # SELF REGISTRATION
    # ------------

    def register_controller(self):
        """ registers controller with global list """
        ibs_weakref = weakref.ref(self)
        __ALL_CONTROLLERS__.append(ibs_weakref)

    def unregister_controller(self):
        ibs_weakref = weakref.ref(self)
        try:
            __ALL_CONTROLLERS__.remove(ibs_weakref)
            pass
        except ValueError:
            pass

    # ------------
    # OBSERVER REGISTRATION
    # ------------

    def cleanup(self):
        """ call on del? """
        logger.info('[self.cleanup] Observers (if any) notified [controller killed]')
        for observer_weakref in self.observer_weakref_list:
            observer_weakref().notify_controller_killed()

    def register_observer(self, observer):
        logger.info('[register_observer] Observer registered: %r' % observer)
        observer_weakref = weakref.ref(observer)
        self.observer_weakref_list.append(observer_weakref)

    def remove_observer(self, observer):
        logger.info('[remove_observer] Observer removed: %r' % observer)
        self.observer_weakref_list.remove(observer)

    def notify_observers(self):
        logger.info('[notify_observers] Observers (if any) notified')
        for observer_weakref in self.observer_weakref_list:
            observer_weakref().notify()

    # ------------

    def _init_rowid_constants(self):
        # ADD TO CONSTANTS

        # THIS IS EXPLICIT IN CONST, USE THAT VERSION INSTEAD
        # self.UNKNOWN_LBLANNOT_ROWID = const.UNKNOWN_LBLANNOT_ROWID
        # self.UNKNOWN_NAME_ROWID     = self.UNKNOWN_LBLANNOT_ROWID
        # self.UNKNOWN_SPECIES_ROWID  = self.UNKNOWN_LBLANNOT_ROWID

        # self.MANUAL_CONFIG_SUFFIX = 'MANUAL_CONFIG'
        # self.MANUAL_CONFIGID = self.add_config(self.MANUAL_CONFIG_SUFFIX)
        # duct_tape.fix_compname_configs(ibs)
        # duct_tape.remove_database_slag(ibs)
        # duct_tape.fix_nulled_yaws(ibs)
        lbltype_names = const.KEY_DEFAULTS.keys()
        lbltype_defaults = const.KEY_DEFAULTS.values()
        lbltype_ids = self.add_lbltype(lbltype_names, lbltype_defaults)
        self.lbltype_ids = dict(zip(lbltype_names, lbltype_ids))

    @profile
    def _init_sql(self, request_dbversion=None, request_stagingversion=None):
        """ Load or create sql database """
        from wbia.other import duct_tape  # NOQA

        # LOAD THE DEPENDENCY CACHE BEFORE THE MAIN DATABASE SO THAT ANY UPDATE
        # CALLS TO THE CORE DATABASE WILL HAVE ACCESS TO THE CACHE DATABASES IF
        # THEY ARE NEEDED.  THIS IS A DECISION MADE ON 8/16/16 BY JP AND JC TO
        # ALLOW FOR COLUMN DATA IN THE CORE DATABASE TO BE MIGRATED TO THE CACHE
        # DATABASE DURING A POST UPDATE FUNCTION ROUTINE, WHICH HAS TO BE LOADED
        # FIRST AND DEFINED IN ORDER TO MAKE THE SUBSEQUENT WRITE CALLS TO THE
        # RELEVANT CACHE DATABASE
        self._init_depcache()
        self._init_sqldbcore(request_dbversion=request_dbversion)
        self._init_sqldbstaging(request_stagingversion=request_stagingversion)
        # self.db.dump_schema()
        self._init_rowid_constants()

    def _needs_backup(self):
        needs_backup = not ut.get_argflag('--nobackup')
        if self.get_dbname() == 'PZ_MTEST':
            needs_backup = False
        if dtool.sql_control.READ_ONLY:
            needs_backup = False
        return needs_backup

    @profile
    def _init_sqldbcore(self, request_dbversion=None):
        """Initializes the *main* database object"""
        # FIXME (12-Jan-12021) Disabled automatic schema upgrade
        DB_VERSION_EXPECTED = '2.0.0'

        if self.is_using_postgres_db:
            uri = self._base_uri
        else:
            uri = f'{self.base_uri}/{self.sqldb_fname}'
        fname = Path(self.sqldb_fname).stem  # filename without extension
        self.db = dtool.SQLDatabaseController(uri, fname)

        # BBB (12-Jan-12021) Disabled the ability to make the database read-only
        self.readonly = False

        # Upgrade the database
        from wbia.control._sql_helpers import ensure_correct_version
        from wbia.control import DB_SCHEMA

        ensure_correct_version(
            self,
            self.db,
            DB_VERSION_EXPECTED,
            DB_SCHEMA,
            verbose=True,
            dobackup=not self.readonly,
        )

    @profile
    def _init_sqldbstaging(self, request_stagingversion=None):
        """Initializes the *staging* database object"""
        # FIXME (12-Jan-12021) Disabled automatic schema upgrade
        DB_VERSION_EXPECTED = '1.2.0'

        if self.is_using_postgres_db:
            uri = self._base_uri
        else:
            uri = f'{self.base_uri}/{self.sqlstaging_fname}'
        fname = Path(self.sqlstaging_fname).stem  # filename without extension
        self.staging = dtool.SQLDatabaseController(uri, fname)

        # BBB (12-Jan-12021) Disabled the ability to make the database read-only
        self.readonly = False

        # Upgrade the database
        from wbia.control._sql_helpers import ensure_correct_version
        from wbia.control import STAGING_SCHEMA

        ensure_correct_version(
            self,
            self.staging,
            DB_VERSION_EXPECTED,
            STAGING_SCHEMA,
            verbose=True,
            dobackup=not self.readonly,
        )

    @profile
    def _init_depcache(self):
        # Initialize dependency cache for images
        image_root_getters = {}
        self.depc_image = dtool.DependencyCache(
            self,
            const.IMAGE_TABLE,
            self.get_image_uuids,
            root_getters=image_root_getters,
        )
        self.depc_image.initialize()

        # Need to reinit this sometimes if cache is ever deleted
        # Initialize dependency cache for annotations
        annot_root_getters = {
            'name': self.get_annot_names,
            'species': self.get_annot_species,
            'yaw': self.get_annot_yaws,
            'viewpoint_int': self.get_annot_viewpoint_int,
            'viewpoint': self.get_annot_viewpoints,
            'bbox': self.get_annot_bboxes,
            'verts': self.get_annot_verts,
            'image_uuid': lambda aids: self.get_image_uuids(
                self.get_annot_image_rowids(aids)
            ),
            'theta': self.get_annot_thetas,
            'occurrence_text': self.get_annot_occurrence_text,
        }
        self.depc_annot = dtool.DependencyCache(
            self,
            const.ANNOTATION_TABLE,
            self.get_annot_visual_uuids,
            root_getters=annot_root_getters,
        )
        # backwards compatibility
        self.depc = self.depc_annot
        # TODO: root_uuids should be specified as the
        # base_root_uuid plus a hash of the attributes that matter for the
        # requested computation.
        self.depc_annot.initialize()

        # Initialize dependency cache for parts
        part_root_getters = {}
        self.depc_part = dtool.DependencyCache(
            self,
            const.PART_TABLE,
            self.get_part_uuids,
            root_getters=part_root_getters,
        )
        self.depc_part.initialize()

    def _close_depcache(self):
        self.depc_image.close()
        self.depc_image = None
        self.depc_annot.close()
        self.depc_annot = None
        self.depc_part.close()
        self.depc_part = None

    def disconnect_sqldatabase(self):
        logger.info('disconnecting from sql database')
        self._close_depcache()
        self.db.close()
        self.db = None
        self.staging.close()
        self.staging = None

    def clone_handle(self, **kwargs):
        ibs2 = IBEISController(dbdir=self.get_dbdir(), ensure=False)
        if len(kwargs) > 0:
            ibs2.update_query_cfg(**kwargs)
        # if self.qreq is not None:
        #    ibs2._prep_qreq(self.qreq.qaids, self.qreq.daids)
        return ibs2

    def backup_database(self):
        from wbia.control import _sql_helpers

        _sql_helpers.database_backup(self.get_ibsdir(), self.sqldb_fname, self.backupdir)
        _sql_helpers.database_backup(
            self.get_ibsdir(), self.sqlstaging_fname, self.backupdir
        )

    def daily_backup_database(self):
        from wbia.control import _sql_helpers

        _sql_helpers.database_backup(
            self.get_ibsdir(), self.sqldb_fname, self.backupdir, False
        )
        _sql_helpers.database_backup(
            self.get_ibsdir(),
            self.sqlstaging_fname,
            self.backupdir,
            False,
        )

    def _send_wildbook_request(self, wbaddr, payload=None):
        import requests

        if wbaddr is None:
            return
        try:
            if payload is None:
                response = requests.get(wbaddr)
            else:
                response = requests.post(wbaddr, data=payload)
        # except requests.MissingSchema:
        #     logger.info('[ibs._send_wildbook_request] Invalid URL: %r' % wbaddr)
        #     return None
        except requests.ConnectionError:
            logger.info(
                '[ibs.wb_reqst] Could not connect to Wildbook server at %r' % wbaddr
            )
            return None
        return response

    def _init_dirs(
        self, dbdir=None, dbname='testdb_1', workdir='~/wbia_workdir', ensure=True
    ):
        """
        Define ibs directories
        """
        PATH_NAMES = const.PATH_NAMES
        REL_PATHS = const.REL_PATHS

        if not ut.QUIET:
            logger.info('[self._init_dirs] self.dbdir = %r' % dbdir)
        if dbdir is not None:
            workdir, dbname = split(dbdir)
        self.workdir = ut.truepath(workdir)
        self.dbname = dbname
        self.sqldb_fname = PATH_NAMES.sqldb
        self.sqlstaging_fname = PATH_NAMES.sqlstaging

        # Make sure you are not nesting databases
        assert PATH_NAMES._ibsdb != ut.dirsplit(
            self.workdir
        ), 'cannot work in _ibsdb internals'
        assert PATH_NAMES._ibsdb != dbname, 'cannot create db in _ibsdb internals'
        self.dbdir = join(self.workdir, self.dbname)
        # All internal paths live in <dbdir>/_ibsdb
        # TODO: constantify these
        # so non controller objects (like in score normalization) have access
        # to these
        self._ibsdb = join(self.dbdir, REL_PATHS._ibsdb)
        self.trashdir = join(self.dbdir, REL_PATHS.trashdir)
        self.cachedir = join(self.dbdir, REL_PATHS.cache)
        self.backupdir = join(self.dbdir, REL_PATHS.backups)
        self.logsdir = join(self.dbdir, REL_PATHS.logs)
        self.chipdir = join(self.dbdir, REL_PATHS.chips)
        self.imgdir = join(self.dbdir, REL_PATHS.images)
        self.uploadsdir = join(self.dbdir, REL_PATHS.uploads)
        # All computed dirs live in <dbdir>/_ibsdb/_wbia_cache
        self.thumb_dpath = join(self.dbdir, REL_PATHS.thumbs)
        self.flanndir = join(self.dbdir, REL_PATHS.flann)
        self.qresdir = join(self.dbdir, REL_PATHS.qres)
        self.bigcachedir = join(self.dbdir, REL_PATHS.bigcache)
        self.distinctdir = join(self.dbdir, REL_PATHS.distinctdir)
        if ensure:
            self.ensure_directories()
        assert dbdir is not None, 'must specify database directory'

    def ensure_directories(self):
        """
        Makes sure the core directores for the controller exist
        """
        _verbose = ut.VERBOSE
        ut.ensuredir(self._ibsdb)
        ut.ensuredir(self.cachedir, verbose=_verbose)
        ut.ensuredir(self.backupdir, verbose=_verbose)
        ut.ensuredir(self.logsdir, verbose=_verbose)
        ut.ensuredir(self.workdir, verbose=_verbose)
        ut.ensuredir(self.imgdir, verbose=_verbose)
        ut.ensuredir(self.chipdir, verbose=_verbose)
        ut.ensuredir(self.flanndir, verbose=_verbose)
        ut.ensuredir(self.qresdir, verbose=_verbose)
        ut.ensuredir(self.bigcachedir, verbose=_verbose)
        ut.ensuredir(self.thumb_dpath, verbose=_verbose)
        ut.ensuredir(self.distinctdir, verbose=_verbose)
        self.get_smart_patrol_dir()

    # --------------
    # --- DIRS ----
    # --------------

    @register_api('/api/core/db/name/', methods=['GET'])
    def get_dbname(self):
        """
        Returns:
            list_ (list): database name

        RESTful:
            Method: GET
            URL:    /api/core/db/name/
        """
        return self.dbname

    def get_db_name(self):
        """ Alias for self.get_dbname(). """
        return self.get_dbname()

    @register_api(CORE_DB_UUID_INIT_API_RULE, methods=['GET'])
    def get_db_init_uuid(self):
        """
        Returns:
            UUID: The SQLDatabaseController's initialization UUID

        RESTful:
            Method: GET
            URL:    /api/core/db/uuid/init/
        """
        return self.db.get_db_init_uuid()

    def get_logdir_local(self):
        return self.logsdir

    def get_logdir_global(self, local=False):
        if const.CONTAINERIZED:
            return self.get_logdir_local()
        else:
            return ut.get_logging_dir(appname='wbia')

    def get_dbdir(self):
        """ database dir with ibs internal directory """
        return self.dbdir

    def get_db_core_path(self):
        return self.db.uri

    def get_db_staging_path(self):
        return self.staging.uri

    def get_db_cache_path(self):
        return self.dbcache.uri

    def get_shelves_path(self):
        engine_slot = const.ENGINE_SLOT
        engine_slot = str(engine_slot).lower()
        if engine_slot in ['none', 'null', '1', 'default']:
            engine_shelve_dir = 'engine_shelves'
        else:
            engine_shelve_dir = 'engine_shelves_%s' % (engine_slot,)
        return join(self.get_cachedir(), engine_shelve_dir)

    def get_trashdir(self):
        return self.trashdir

    def get_ibsdir(self):
        """ ibs internal directory """
        return self._ibsdb

    def get_chipdir(self):
        return self.chipdir

    def get_probchip_dir(self):
        return join(self.get_cachedir(), 'prob_chips')

    def get_fig_dir(self):
        """ ibs internal directory """
        return join(self._ibsdb, 'figures')

    def get_imgdir(self):
        """ ibs internal directory """
        return self.imgdir

    def get_uploadsdir(self):
        """ ibs internal directory """
        return self.uploadsdir

    def get_thumbdir(self):
        """ database directory where thumbnails are cached """
        return self.thumb_dpath

    def get_workdir(self):
        """ directory where databases are saved to """
        return self.workdir

    def get_cachedir(self):
        """ database directory of all cached files """
        return self.cachedir

    def get_match_thumbdir(self):
        match_thumb_dir = ut.unixjoin(self.get_cachedir(), 'match_thumbs')
        ut.ensuredir(match_thumb_dir)
        return match_thumb_dir

    def get_wbia_resource_dir(self):
        """ returns the global resource dir in .config or AppData or whatever """
        resource_dir = sysres.get_wbia_resource_dir()
        return resource_dir

    def get_detect_modeldir(self):
        return join(sysres.get_wbia_resource_dir(), 'detectmodels')

    def get_detectimg_cachedir(self):
        """
        Returns:
            detectimgdir (str): database directory of image resized for
                detections
        """
        return join(self.cachedir, const.PATH_NAMES.detectimg)

    def get_flann_cachedir(self):
        """
        Returns:
            flanndir (str): database directory where the FLANN KD-Tree is
                stored
        """
        return self.flanndir

    def get_qres_cachedir(self):
        """
        Returns:
            qresdir (str): database directory where query results are stored
        """
        return self.qresdir

    def get_neighbor_cachedir(self):
        neighbor_cachedir = ut.unixjoin(self.get_cachedir(), 'neighborcache2')
        return neighbor_cachedir

    def get_big_cachedir(self):
        """
        Returns:
            bigcachedir (str): database directory where aggregate results are
                stored
        """
        return self.bigcachedir

    def get_smart_patrol_dir(self, ensure=True):
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
        smart_patrol_dpath = join(self.dbdir, const.PATH_NAMES.smartpatrol)
        if ensure:
            ut.ensuredir(smart_patrol_dpath)
        return smart_patrol_dpath

    # ------------------
    # --- WEB CORE ----
    # ------------------

    @register_api('/log/current/', methods=['GET'])
    def get_current_log_text(self):
        r"""

        Example:
            >>> # xdoctest: +REQUIRES(--web-tests)
            >>> import wbia
            >>> with wbia.opendb_with_web('testdb1') as (ibs, client):
            ...     resp = client.get('/log/current/')
            >>> resp.json
            {'status': {'success': True, 'code': 200, 'message': '', 'cache': -1}, 'response': None}

        """
        text = ut.get_current_log_text()
        return text

    @register_api('/api/core/db/info/', methods=['GET'])
    def get_dbinfo(self):
        from wbia.other import dbinfo

        locals_ = dbinfo.get_dbinfo(self)
        return locals_['info_str']
        # return ut.repr2(dbinfo.get_dbinfo(self), nl=1)['infostr']

    # --------------
    # --- MISC ----
    # --------------

    def copy_database(self, dest_dbdir):
        # TODO: rectify with rsync, script, and merge script.
        from wbia.init import sysres

        sysres.copy_wbiadb(self.get_dbdir(), dest_dbdir)

    def dump_database_csv(self):
        dump_dir = join(self.get_dbdir(), 'CSV_DUMP')
        self.db.dump_tables_to_csv(dump_dir=dump_dir)
        with open(join(dump_dir, '_ibsdb.dump'), 'w') as fp:
            dump(self.db.connection, fp)

    def get_database_icon(self, max_dsize=(None, 192), aid=None):
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
            >>> icon = self.get_database_icon()
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> pt.imshow(icon)
            >>> ut.show_if_requested()
        """
        # if self.get_dbname() == 'Oxford':
        #    pass
        # else:
        import vtool as vt

        if hasattr(self, 'force_icon_aid'):
            aid = self.force_icon_aid
        if aid is None:
            species = self.get_primary_database_species()
            # Use a url to get the icon
            url = {
                self.const.TEST_SPECIES.GIR_MASAI: 'http://i.imgur.com/tGDVaKC.png',
                self.const.TEST_SPECIES.ZEB_PLAIN: 'http://i.imgur.com/2Ge1PRg.png',
                self.const.TEST_SPECIES.ZEB_GREVY: 'http://i.imgur.com/PaUT45f.png',
            }.get(species, None)
            if url is not None:
                icon = vt.imread(ut.grab_file_url(url), orient='auto')
            else:
                # HACK: (this should probably be a db setting)
                # use an specific aid to get the icon
                aid = {'Oxford': 73, 'seaturtles': 37}.get(self.get_dbname(), None)
                if aid is None:
                    # otherwise just grab a random aid
                    aid = self.get_valid_aids()[0]
        if aid is not None:
            icon = self.get_annot_chips(aid)
        icon = vt.resize_to_maxdims(icon, max_dsize)
        return icon

    def _custom_ibsstr(self):
        # typestr = ut.type_str(type(ibs)).split('.')[-1]
        typestr = self.__class__.__name__
        dbname = self.get_dbname()
        # hash_str = hex(id(ibs))
        # ibsstr = '<%s(%s) at %s>' % (typestr, dbname, hash_str, )
        hash_str = self.get_db_init_uuid()
        ibsstr = '<%s(%s) with UUID %s>' % (typestr, dbname, hash_str)
        return ibsstr

    def __str__(self):
        return self._custom_ibsstr()

    def __repr__(self):
        return self._custom_ibsstr()

    def __getstate__(self):
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
            'dbdir': self.get_dbdir(),
            'machine_name': ut.get_computer_name(),
        }
        return state

    def __setstate__(self, state):
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
        self.__dict__.update(**ibs2.__dict__)

    def predict_ws_injury_interim_svm(self, aids):
        from wbia.scripts import classify_shark

        return classify_shark.predict_ws_injury_interim_svm(self, aids)

    def get_web_port_via_scan(
        self, url_base='127.0.0.1', port_base=5000, scan_limit=100, verbose=True
    ):
        import requests

        api_rule = CORE_DB_UUID_INIT_API_RULE
        target_uuid = self.get_db_init_uuid()
        for candidate_port in range(port_base, port_base + scan_limit + 1):
            candidate_url = 'http://%s:%s%s' % (url_base, candidate_port, api_rule)
            try:
                response = requests.get(candidate_url)
            except (requests.ConnectionError):
                if verbose:
                    logger.info('Failed to find IA server at %s' % (candidate_url,))
                continue
            logger.info('Found IA server at %s' % (candidate_url,))
            try:
                response = ut.from_json(response.text)
                candidate_uuid = response.get('response')
                assert candidate_uuid == target_uuid
                return candidate_port
            except (AssertionError):
                if verbose:
                    logger.info(
                        'Invalid response from IA server at %s' % (candidate_url,)
                    )
                continue

        return None
