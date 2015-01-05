"""
TODO: Module Licence and docstring

functions in the IBEISController have been split up into several submodules.

look at the modules listed in autogenmodname_list to see the full list of
functions that will be injected into an IBEISController object
"""
# TODO: rename annotation annotations
# TODO: make all names consistent
from __future__ import absolute_import, division, print_function
# Python
import six
import atexit
import requests
import weakref
import lockfile
from six.moves import zip
from os import system
from os.path import join, exists, split
# UTool
import utool as ut  # NOQA
# IBEIS
import ibeis  # NOQA
from ibeis import constants as const
from ibeis import params
from ibeis.control import accessor_decors
from ibeis.control.accessor_decors import (default_decorator, )
# Import modules which define injectable functions
# Older manual ibeiscontrol functions
from ibeis import ibsfuncs
from ibeis.model.hots import pipeline
#from ibeis.control import controller_inject


# Shiny new way to inject external functions
autogenmodname_list = [
    '_autogen_featweight_funcs',
    #'_autogen_annot_funcs',
    'manual_ibeiscontrol_funcs',
    'manual_meta_funcs',
    'manual_lbltype_funcs',
    'manual_lblannot_funcs',
    'manual_lblimage_funcs',
    'manual_image_funcs',
    'manual_annot_funcs',
    'manual_name_species_funcs',
    'manual_dependant_funcs',
]

INJECTED_MODULES = []

for modname in autogenmodname_list:
    exec('from ibeis.control import ' + modname, globals(), locals())
    module = eval(modname)
    INJECTED_MODULES.append(module)
# Inject utool functions
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[ibs]')


__ALL_CONTROLLERS__ = []  # Global variable containing all created controllers
__IBEIS_CONTROLLER_CACHE__ = {}


def request_IBEISController(dbdir=None, ensure=True, wbaddr=None, verbose=ut.VERBOSE, use_cache=True):
    r"""
    Alternative to directory instantiating a new controller object. Might
    return a memory cached object

    Args:
        dbdir     (str):
        ensure    (bool):
        wbaddr    (None):
        verbose   (bool):
        use_cache (bool):

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
        ibs = IBEISController(dbdir=dbdir, ensure=ensure, wbaddr=wbaddr, verbose=verbose)
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

    def __init__(ibs, dbdir=None, ensure=True, wbaddr=None, verbose=True):
        """ Creates a new IBEIS Controller associated with one database """
        if verbose and ut.VERBOSE:
            print('[ibs.__init__] new IBEISController')
        # an dict to hack in temporary state
        ibs.temporary_state = {}
        ibs.allow_override = 'override+warn'
        # observer_weakref_list keeps track of the guibacks connected to this controller
        ibs.observer_weakref_list = []
        # not completely working decorator cache
        ibs.table_cache = None
        ibs._initialize_self()
        ibs._init_dirs(dbdir=dbdir, ensure=ensure)
        # _init_wb will do nothing if no wildbook address is specified
        ibs._init_wb(wbaddr)
        ibs._init_sql()
        ibs._init_config()

    def reset_table_cache(ibs):
        ibs.table_cache = accessor_decors.init_tablecache()

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
        # duct_tape.fix_nulled_viewpoints(ibs)
        lbltype_names    = const.KEY_DEFAULTS.keys()
        lbltype_defaults = const.KEY_DEFAULTS.values()
        lbltype_ids = ibs.add_lbltype(lbltype_names, lbltype_defaults)
        ibs.lbltype_ids = dict(zip(lbltype_names, lbltype_ids))

    @default_decorator
    def _init_sql(ibs):
        """ Load or create sql database """
        from ibeis.dev import duct_tape  # NOQA
        ibs._init_sqldbcore()
        ibs._init_sqldbcache()
        # ibs.db.dump_schema()
        # ibs.db.dump()
        ibs._init_rowid_constants()

    #@ut.indent_func
    def _init_sqldbcore(ibs):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis  # NOQA
            >>> ibs = ibeis.opendb('testdb1')
        """
        from ibeis.control import _sql_helpers
        from ibeis.control import SQLDatabaseControl as sqldbc
        from ibeis.control import DB_SCHEMA
        # Before load, ensure database has been backed up for the day
        _sql_helpers.ensure_daily_database_backup(ibs.get_ibsdir(), ibs.sqldb_fname, ibs.backupdir)
        # IBEIS SQL State Database
        #ibs.db_version_expected = '1.1.1'
        ibs.db_version_expected = '1.3.0'
        ## TODO: add this functionality to SQLController
        #testing_newschmea = ut.is_developer() and ibs.get_dbname() in ['PZ_MTEST', 'testdb1', 'testdb0']
        ##testing_newschmea = False
        ##ut.is_developer() and ibs.get_dbname() in ['PZ_MTEST', 'testdb1']
        #if testing_newschmea:
        #    # Set to true until the schema module is good then continue tests with this set to false
        #    testing_force_fresh = True or ut.get_argflag('--force-fresh')
        #    # Work on a fresh schema copy when developing
        #    dev_sqldb_fname = ut.augpath(ibs.sqldb_fname, '_develop_schema')
        #    sqldb_fpath = join(ibs.get_ibsdir(), ibs.sqldb_fname)
        #    dev_sqldb_fpath = join(ibs.get_ibsdir(), dev_sqldb_fname)
        #    ut.copy(sqldb_fpath, dev_sqldb_fpath, overwrite=testing_force_fresh)
        #    # Set testing schema version
        #    ibs.db_version_expected = '1.3.0'
        ibs.db = sqldbc.SQLDatabaseController(ibs.get_ibsdir(), ibs.sqldb_fname,
                                              text_factory=const.__STR__,
                                              inmemory=False)
        # Ensure correct schema versions
        _sql_helpers.ensure_correct_version(
            ibs,
            ibs.db,
            ibs.db_version_expected,
            DB_SCHEMA,
            autogenerate=params.args.dump_autogen_schema
        )

    #@ut.indent_func
    def _init_sqldbcache(ibs):
        """ Need to reinit this sometimes if cache is ever deleted """
        from ibeis.control import _sql_helpers
        from ibeis.control import SQLDatabaseControl as sqldbc
        from ibeis.control import DBCACHE_SCHEMA
        # IBEIS SQL Features & Chips database
        ibs.dbcache_version_expected = '1.0.3'
        ibs.dbcache = sqldbc.SQLDatabaseController(
            ibs.get_cachedir(), ibs.sqldbcache_fname, text_factory=const.__STR__)
        _sql_helpers.ensure_correct_version(
            ibs,
            ibs.dbcache,
            ibs.dbcache_version_expected,
            DBCACHE_SCHEMA,
            dobackup=False,  # Everything in dbcache can be regenerated.
            autogenerate=params.args.dump_autogen_schema
        )

    def _close_sqldbcache(ibs):
        ibs.dbcache.close()
        ibs.dbcache = None

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
        _sql_helpers.database_backup(ibs.get_ibsdir(), ibs.sqldb_fname, ibs.backupdir)

    @default_decorator
    def _init_wb(ibs, wbaddr):
        if wbaddr is None:
            return
        #TODO: Clean this up to use like ut and such
        try:
            response = requests.get(wbaddr)
        # except requests.MissingSchema:
        #     print('[ibs._init_wb] Invalid URL: %r' % wbaddr)
        #     return None
        except requests.ConnectionError:
            print('[ibs._init_wb] Could not connect to Wildbook server at %r' % wbaddr)
            return None
        return response

    @default_decorator
    def _init_dirs(ibs, dbdir=None, dbname='testdb_1', workdir='~/ibeis_workdir', ensure=True):
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
        # All computed dirs live in <dbdir>/_ibsdb/_ibeis_cache
        ibs.thumb_dpath = join(ibs.dbdir, REL_PATHS.thumbs)
        ibs.flanndir    = join(ibs.dbdir, REL_PATHS.flann)
        ibs.qresdir     = join(ibs.dbdir, REL_PATHS.qres)
        ibs.bigcachedir = join(ibs.dbdir, REL_PATHS.bigcache)
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

    #
    #
    #--------------
    # --- DIRS ----
    #--------------

    def get_dbname(ibs):
        """
        Returns:
            list_ (list): database name """
        return ibs.dbname

    def get_logdir(ibs):
        return ut.get_logging_dir(appname='ibeis')

    def get_dbdir(ibs):
        """
        Returns:
            list_ (list): database dir with ibs internal directory """
        #return join(ibs.workdir, ibs.dbname)
        return ibs.dbdir

    def get_trashdir(ibs):
        return ibs.trashdir

    def get_ibsdir(ibs):
        """
        Returns:
            list_ (list): ibs internal directory """
        return ibs._ibsdb

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

    def get_ibeis_resource_dir(ibs):
        from ibeis.dev import sysres
        return sysres.get_ibeis_resource_dir()

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
        scorenorm_cachedir = join(ibs.get_ibeis_resource_dir(), 'scorenorm')
        species_cachedir = join(scorenorm_cachedir, species_text)
        if ensure:
            ut.ensurepath(scorenorm_cachedir)
            ut.ensuredir(species_cachedir)
        return species_cachedir

    def get_local_species_scorenorm_cachedir(ibs, species_text, ensure=True):
        """
        """
        scorenorm_cachedir = join(ibs.get_cachedir(), 'scorenorm')
        species_cachedir = join(scorenorm_cachedir, species_text)
        if ensure:
            ut.ensuredir(scorenorm_cachedir)
            ut.ensuredir(species_cachedir)
        return species_cachedir

    def get_detect_modeldir(ibs):
        from ibeis.dev import sysres
        return join(sysres.get_ibeis_resource_dir(), 'detectmodels')

    def get_detectimg_cachedir(ibs):
        """
        Returns:
            list_ (list): database directory of image resized for detections """
        return join(ibs.cachedir, const.PATH_NAMES.detectimg)

    def get_flann_cachedir(ibs):
        """
        Returns:
            list_ (list): database directory where the FLANN KD-Tree is stored """
        return ibs.flanndir

    def get_qres_cachedir(ibs):
        """
        Returns:
            list_ (list): database directory where query results are stored """
        return ibs.qresdir

    def get_big_cachedir(ibs):
        """
        Returns:
            list_ (list): database directory where aggregate results are stored """
        return ibs.bigcachedir

    #
    #
    #----------------
    # --- Configs ---
    #----------------

    # @default_decorator
    def export_to_wildbook(ibs):
        """
            Exports identified chips to wildbook

            Legacy:
                import ibeis.dbio.export_wb as wb
                print('[ibs] exporting to wildbook')
                eid_list = ibs.get_valid_eids()
                addr = "http://127.0.0.1:8080/wildbook-4.1.0-RELEASE"
                #addr = "http://tomcat:tomcat123@127.0.0.1:8080/wildbook-5.0.0-EXPERIMENTAL"
                ibs._init_wb(addr)
                wb.export_ibeis_to_wildbook(ibs, eid_list)

                # compute encounters
                # get encounters by id
                # get ANNOTATIONs by encounter id
                # submit requests to wildbook
                return None
        """
        raise NotImplementedError()

    @default_decorator
    def wildbook_signal_eid_list(ibs, eid_list=None, set_shipped_flag=True):
        """ Exports specified encounters to wildbook """
        def _send(eid):
            encounter_uuid = ibs.get_encounter_uuid(eid)
            addr_ = addr % (hostname, encounter_uuid)
            response = ibs._init_wb(addr_)
            print(addr_, response)
            return response is not None
        # Setup
        wildbook_tomcat_path = '/var/lib/tomcat7/webapps/wildbook/'
        if exists(wildbook_tomcat_path):
            wildbook_properties_path  = 'WEB-INF/classes/bundles/'
            wildbook_properties_path_ = join(wildbook_tomcat_path, wildbook_properties_path)
            src_config = 'commonConfiguration.properties.default'
            dst_config = 'commonConfiguration.properties'
            print('[ibs.wildbook_signal_eid_list()] Wildbook properties=%r' % (wildbook_properties_path_, ))
             # Configuration
            hostname = '127.0.0.1'
            addr = "http://%s:8080/wildbook/OccurrenceCreateIBEIS?ibeis_encounter_id=%s"
            # With a lock file, modify the configuration with the new settings
            with lockfile.LockFile(join(ibs.get_cachedir(), 'wildbook.lock')):
                # Update the Wildbook configuration to see *THIS* ibeis database
                with open(join(wildbook_properties_path_, src_config), 'r') as f:
                    content = f.read()
                    content = content.replace('__IBEIS_DB_PATH__', ibs.get_dbdir())
                    content = content.replace('__IBEIS_IMAGE_PATH__', ibs.get_imgdir())
                    content = '"%s"' % (content, )
                # Write to the configuration
                print('[ibs.wildbook_signal_eid_list()] To update the Wildbook configuration, we need sudo privaleges')
                command = ['sudo', 'echo', content, '>', dst_config]
                # ut.cmd(command, sudo=True)
                system(' '.join(command))
                # with open(join(wildbook_properties_path_, dst_config), 'w') as f:
                #     f.write(content)
                # Call Wildbook url to signal update
                print('[ibs.wildbook_signal_eid_list()] shipping eid_list = %r to wildbook' % (eid_list, ))
                if eid_list is None:
                    eid_list = ibs.get_valid_eids()
                status_list = [ _send(eid) for eid in eid_list ]
                if set_shipped_flag:
                    for eid, status in zip(eid_list, status_list):
                        val = 1 if status else 0
                        ibs.set_encounter_shipped_flags([eid], [val])
                return status_list
        else:
            raise AssertionError('Wildbook is not installed on this machine')

    #
    #
    #------------------
    # --- DETECTION ---
    #------------------

    @default_decorator
    def detect_random_forest(ibs, gid_list, species, **kwargs):
        """ Runs animal detection in each image """
        # TODO: Return confidence here as well
        print('[ibs] detecting using random forests')
        from ibeis.model.detect import randomforest  # NOQ
        detect_gen = randomforest.detect_gid_list_with_species(ibs, gid_list, species, **kwargs)
        # ibs.cfg.other_cfg.ensure_attr('detect_add_after', 1)
        # ADD_AFTER_THRESHOLD = ibs.cfg.other_cfg.detect_add_after
        for gid, (gpath, result_list) in zip(gid_list, detect_gen):
            for result in result_list:
                # Ideally, species will come from the detector with confidences that actually mean something
                bbox = (result['xtl'], result['ytl'], result['width'], result['height'])
                ibs.add_annots([gid], [bbox], notes_list=['rfdetect'],
                               species_list=[species],
                               detect_confidence_list=[result['confidence']])

    #
    #
    #-----------------------------
    # --- ENCOUNTER CLUSTERING ---
    #-----------------------------

    @ut.indent_func('[ibs.compute_encounters]')
    def compute_encounters(ibs):
        """
        Clusters images into encounters

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
            >>> print("PRE COMPUTE: Encounters are %r" % ibs.get_valid_eids())
            >>> print("Containing: %r" % ibs.get_encounter_gids(ibs.get_valid_eids()))
            >>> ibs.compute_encounters()
            >>> print("COMPUTE: New encounters are %r" % ibs.get_valid_eids())
            >>> print("Containing: %r" % ibs.get_encounter_gids(ibs.get_valid_eids()))
            >>> ibs.update_special_encounters()
            >>> print("UPDATE SPECIAL: New encounters are %r" % ibs.get_valid_eids())
            >>> print("Containing: %r" % ibs.get_encounter_gids(ibs.get_valid_eids()))
            >>> assert(images_to_remove[0] not in ibs.get_encounter_gids(nonspecial_eids[0:1])[0])
        """
        from ibeis.model.preproc import preproc_encounter
        print('[ibs] Computing and adding encounters.')
        #gid_list = ibs.get_valid_gids(require_unixtime=False, reviewed=False)
        gid_list = ibs.get_ungrouped_gids()
        flat_eids, flat_gids = preproc_encounter.ibeis_compute_encounters(ibs, gid_list)
        valid_eids = ibs.get_valid_eids()
        eid_offset = 0 if len(valid_eids) == 0 else max(valid_eids)
        flat_eids_offset = [eid + eid_offset for eid in flat_eids]  # This way we can make sure that manually separated encounters
        # remain untouched, and ensure that new encounters are created
        enctext_list = ['Encounter ' + str(eid) for eid in flat_eids_offset]
        print("enctext_list: %r; flat_gids: %r" % (enctext_list, flat_gids))
        print('[ibs] Finished computing, about to add encounter.')
        ibs.set_image_enctext(flat_gids, enctext_list)
        print('[ibs] Finished computing and adding encounters.')

    #
    #
    #-----------------------
    # --- IDENTIFICATION ---
    #-----------------------

    @default_decorator
    def get_recognition_database_aids(ibs, eid=None, is_exemplar=True, species=None):
        """
        DEPRECATE or refactor

        Returns:
            daid_list (list): testing recognition database annotations
        """
        if 'daid_list' in ibs.temporary_state:
            daid_list = ibs.temporary_state['daid_list']
        else:
            daid_list = ibs.get_valid_aids(eid=eid, species=species, is_exemplar=is_exemplar)
        return daid_list

    @default_decorator
    def get_recognition_query_aids(ibs, is_known, species=None):
        qaid_list = ibs.get_valid_aids(is_known=is_known, species=species)
        return qaid_list

    def query_chips(ibs, qaid_list,
                    daid_list=None,
                    cfgdict=None,
                    use_cache=None,
                    use_bigcache=None,
                    qreq_=None,
                    return_request=False,
                    verbose=pipeline.VERB_PIPELINE):
        r"""
        Args:
            qaid_list (list):
            daid_list (list):
            cfgdict (None):
            use_cache (None):
            use_bigcache (None):
            qreq_ (QueryRequest):  hyper-parameters
            return_request (bool):
            verbose (bool):

        Returns:
            tuple: (qres_list, qreq_)

        CommandLine:
            python -m ibeis.control.IBEISControl --test-query_chips

        Example:
            >>> # SLOW_DOCTEST
            >>> from ibeis.control.IBEISControl import *  # NOQA
            >>> import ibeis  # NOQA
            >>> ibs = ibeis.opendb('testdb1')
            >>> qaids = ibs.get_valid_aids()[0:1]
            >>> qres = ibs.query_chips(qaids)[0]
            >>> assert qres.qaid == qaids[0]
        """
        if daid_list is None:
            daid_list = ibs.get_valid_aids()

        res = ibs._query_chips4(
            qaid_list, daid_list, cfgdict=cfgdict, use_cache=use_cache,
            use_bigcache=use_bigcache, qreq_=qreq_,
            return_request=return_request, verbose=verbose)

        with ut.EmbedOnException():

            if return_request:
                qaid2_qres, qreq_ = res
            else:
                qaid2_qres = res

            qres_list = [qaid2_qres[qaid] for qaid in qaid_list]

        if return_request:
            return qres_list, qreq_
        else:
            return qres_list

    def _query_chips4(ibs, qaid_list, daid_list,
                      use_cache=None,
                      use_bigcache=None,
                      return_request=False,
                      cfgdict=None,
                      qreq_=None,
                      verbose=pipeline.VERB_PIPELINE):
        """
        main entrypoint to submitting a query request

        CommandLine:
            python -m ibeis.control.IBEISControl --test-_query_chips4

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

        # Actually run query
        if qreq_ is not None:
            import numpy as np
            assert np.all(qreq_.get_external_qaids() == qaid_list)
            assert np.all(qreq_.get_external_daids() == daid_list)

        res = mc4.submit_query_request(
            ibs,  qaid_list, daid_list, use_cache, use_bigcache,
            return_request=return_request, cfgdict=cfgdict, qreq_=qreq_,
            verbose=verbose)

        if return_request:
            qaid2_qres, qreq_ = res
            return qaid2_qres, qreq_
        else:
            qaid2_qres = res
            return qaid2_qres

    #_query_chips = _query_chips3
    _query_chips = _query_chips4

    @default_decorator
    def query_encounter(ibs, qaid_list, eid, **kwargs):
        """ _query_chips wrapper """
        daid_list = ibs.get_encounter_aids(eid)  # encounter database chips
        qaid2_qres = ibs._query_chips4(qaid_list, daid_list, **kwargs)
        # HACK IN ENCOUNTER INFO
        for qres in six.itervalues(qaid2_qres):
            qres.eid = eid
        return qaid2_qres

    @default_decorator
    def query_exemplars(ibs, qaid_list, **kwargs):
        """ Queries vs the exemplars """
        daid_list = ibs.get_valid_aids(is_exemplar=True)
        assert len(daid_list) > 0, 'there are no exemplars'
        return ibs._query_chips4(qaid_list, daid_list, **kwargs)

    @default_decorator
    def query_all(ibs, qaid_list, **kwargs):
        """ Queries vs the exemplars """
        daid_list = ibs.get_valid_aids()
        qaid2_qres = ibs._query_chips4(qaid_list, daid_list, **kwargs)
        return qaid2_qres


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
