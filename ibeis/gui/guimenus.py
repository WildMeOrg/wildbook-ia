"""
This module defines all of the menu items in the main GUI
as well as their callbacks in guiback
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import functools
from ibeis import constants as const
import guitool
ut.noinject(__name__, '[guimenus]', DEBUG=False)


class DummyBack(object):
    def __init__(self):
        print('using dummy back')
        pass
    def __getattr__(self, name):
        #print(name)
        if name.startswith('_'):
            return self.__dict__[name]
        import mock
        mock.Mock()
        return mock.Mock()


def setup_dummy_menus():
    r"""
    CommandLine:
        python -m ibeis.gui.guimenus --test-setup_dummy_menus

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.guimenus import *  # NOQA
        >>> result = setup_dummy_menus()
        >>> print(result)
    """
    #import unittest
    import guitool
    guitool.ensure_qapp()  # must be ensured before any embeding
    mainwin = guitool.QtGui.QMainWindow()
    back = DummyBack()
    import mock
    mainwin.expand_names_tree = mock.Mock
    setup_menus(mainwin, back)
    mainwin.show()
    mainwin.resize(600, 100)
    #ut.embed()
    guitool.qtapp_loop(mainwin, frequency=100)
    #guitool.qtapp_loop(mainwin, frequency=100, ipy=ut.inIPython())


def setup_menus(mainwin, back=None):
    print('[guimenus] creating menus')
    mainwin.menubar = guitool.newMenubar(mainwin)
    if back is None:
        back = DummyBack()
    setup_file_menu(mainwin, back)
    setup_view_menu(mainwin, back)
    setup_actions_menu(mainwin, back)
    #setup_batch_menu(mainwin, back)
    #setup_checks_menu(mainwin, back)
    setup_option_menu(mainwin, back)
    setup_refresh_menu(mainwin, back)
    #setup_wildbook_menu(mainwin, back)
    setup_web_menu(mainwin, back)
    setup_help_menu(mainwin, back)
    setup_developer_menu(mainwin, back)


def setup_file_menu(mainwin, back):
    """ FILE MENU """
    mainwin.menuFile = guitool.newMenu(mainwin, mainwin.menubar, 'menuFile', 'File')
    menu = mainwin.menuFile
    menu.newAction(
        name='actionNew_Database',
        text='New Database',
        tooltip='Create a new folder to use as a database.',
        shortcut='Ctrl+N',
        slot_fn=back.new_database)
    menu.newAction(
        name='actionOpen_Database',
        text='Open Database',
        tooltip='Opens a different database directory.',
        shortcut='Ctrl+O',
        slot_fn=back.open_database)
    menu.addSeparator()
    menu.newAction(
        name='actionBackup_Database',
        tooltip='Backup the current main database.',
        text='Backup Database',
        shortcut='Ctrl+B',
        slot_fn=back.backup_database)
    menu.newAction(
        name='actionExport_Database',
        tooltip='Dumps and exports database as csv tables.',
        text='Export Database',
        shortcut='Ctrl+S',
        slot_fn=back.export_database)
    menu.addSeparator()
    menu.newAction(
        name='actionImport_Img_file',
        text='Import Images (select file(s))',
        slot_fn=back.import_images_from_file)
    menu.newAction(
        name='actionImport_Img_dir',
        text='Import Images (select directory)',
        shortcut='Ctrl+I',
        slot_fn=back.import_images_from_dir)
    menu.addSeparator()
    #menu.newAction(
    #    name='actionImport_Img_file_with_smart',
    #    text='Import Images (select file(s)) with smart Patrol XML',
    #    slot_fn=back.import_images_from_file_with_smart)
    menu.newAction(
        name='actionImport_Img_dir_with_smart',
        text='Import Images (select directory) with smart Patrol XML',
        slot_fn=back.import_images_from_dir_with_smart)
    menu.addSeparator()
    menu.newAction(
        name='actionQuit',
        text='Quit',
        slot_fn=back.quit)


def setup_view_menu(mainwin, back):
    mainwin.menuView = guitool.newMenu(mainwin, mainwin.menubar, 'menuView', 'View')
    menu = mainwin.menuView
    #mainwin.menuActions.addSeparator()
    #mainwin.menuActions.addSeparator()
    #mainwin.menuActions.newAction(
    #    name='actionNext',
    #    text='Select Next',
    #    shortcut='Ctrl+N',
    #    slot_fn=back.select_next)
    #mainwin.menuActions.newAction(
    #    name='actionPrev',
    #    text='Select Previous',
    #    shortcut='Ctrl+P',
    #    slot_fn=back.select_prev)
    menu.addSeparator()
    menu.newAction(
        name='actionExpandNamesTree',
        text='Expand Names Tree',
        slot_fn=mainwin.expand_names_tree)
    menu.addSeparator()
    menu.newAction(
        name='toggleThumbnails',
        text='Toggle Thumbnails',
        slot_fn=back.toggle_thumbnails)
    menu.newAction(
        name='actionLayout_Figures',
        text='Layout Figures',
        tooltip='Organizes windows in a grid',
        shortcut='Ctrl+L',
        slot_fn=back.layout_figures)
    pass


def setup_actions_menu(mainwin, back):
    """ ACTIONS MENU """
    mainwin.menuActions = guitool.newMenu(mainwin, mainwin.menubar, 'menuActions', 'Actions')
    menu = mainwin.menuActions
    menu.newAction(
        name='actionCompute_Encounters',
        text='Group Encounters',
        #shortcut='Ctrl+2',
        slot_fn=back.compute_encounters)
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionDetect',
        text='Run Detection',
        #shortcut='Ctrl+3',
        slot_fn=back.run_detection)
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionQuery',
        text='Query Single Annotation',
        shortcut='Q',
        slot_fn=functools.partial(back.compute_queries, use_visual_selection=True))
    menu.newAction(
        name='actionBatchIntraEncounterQueries',
        text='Query: Intra Encounter',
        slot_fn=functools.partial(back.compute_queries, daids_mode=const.INTRA_ENC_KEY),
    )
    menu.newAction(
        name='actionBatchVsExemplarQueries',
        text='Query: vs Exemplars',
        slot_fn=functools.partial(back.compute_queries, daids_mode=const.VS_EXEMPLARS_KEY),
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionSetExemplarsFromQualityAndViewpoint',
        text='Set Exemplars from Quality and Viewpoint',
        slot_fn=back.set_exemplars_from_quality_and_viewpoint,
        tooltip=ut.textblock(
            '''
            Uses the quality and viewpoint column to pick the best N exemplars
            per viewpoint, per name.
            ''')
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionRunMergeChecks',
        text='Run Merge Checks (Exemplars vs Exemplars)',
        slot_fn=back.run_merge_checks)
    menu.addSeparator()  # ---------
    if not const.SIMPLIFY_INTERFACE:
        menu.newAction(
            name='actionBatchConsecutiveLocationSpeciesRename',
            text='Consecutive Location+Species Rename',
            slot_fn=back.batch_rename_consecutive_via_species,
            tooltip=ut.textblock(
                '''
                Renames ALL the names in the database to
                {other_cfg.location_for_names}_{species_code}_{num}
                ''')
        )

    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionShipProcessedEncounters',
        text='Ship Processed Encounters',
        tooltip='''This action will ship to WildBook any encounters that have
                    been marked as processed.  This can also be used to send
                    processed encounters that failed to ship correctly.''',
        #shortcut='Ctrl+5',
        slot_fn=back.send_unshipped_processed_encounters)
    menu.addSeparator()  # ---------
    menu.newAction(
        text='Override All Annotation Species',
        slot_fn=back.override_all_annotation_species)
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionDeleteAllEncounters',
        text='Delete All Encounters',
        slot_fn=back.delete_all_encounters)
    menu.newAction(
        name='actionDelete_Image',
        text='Delete Image',
        slot_fn=back.delete_image)
    menu.newAction(
        name='actionDelete_ANNOTATION',
        text='Delete Annotation',
        shortcut='Ctrl+Del',
        slot_fn=back.delete_annot)
    menu.addSeparator()


def setup_batch_menu(mainwin, back):
    """ BATCH MENU """
    mainwin.menuBatch = guitool.newMenu(mainwin, mainwin.menubar, 'menuBatch', 'Batch')
    menu = mainwin.menuBatch
    menu


def setup_option_menu(mainwin, back):
    """ OPTIONS MENU """
    mainwin.menuOptions = guitool.newMenu(mainwin, mainwin.menubar, 'menuOptions', 'Options')
    menu = mainwin.menuOptions
    menu.newAction(
        name='actionToggleQueryMode',
        text='Toggle Query Mode: ----',
        tooltip='Changes behavior of Actions->Query',
        slot_fn=functools.partial(back.set_daids_mode, 'toggle'))
    menu.addSeparator()
    menu.newAction(
        name='actionPreferences',
        text='Edit Preferences',
        tooltip='Changes algorithm parameters and program behavior.',
        shortcut='Ctrl+P',
        slot_fn=back.edit_preferences)


def setup_checks_menu(mainwin, back):
    mainwin.menuChecks = guitool.newMenu(mainwin, mainwin.menubar, 'menuChecks', 'Checks')
    pass


def setup_help_menu(mainwin, back):
    """ HELP MENU """
    mainwin.menuHelp = guitool.newMenu(mainwin, mainwin.menubar, 'menuHelp', 'Help')
    menu = mainwin.menuHelp
    #from ibeis.control import DB_SCHEMA_CURRENT
    #version = DB_SCHEMA_CURRENT.VERSION_CURRENT
    menu.newAction(
        name='actionAbout',
        text='About',
        slot_fn=back.show_about_message)
    menu.newAction(
        name='actionDBInfo',
        text='Database Info',
        slot_fn=back.display_dbinfo),
    #menu.newAction(
    #    name='actionView_Docs',
    #    text='View Documentation',
    #    slot_fn=back.view_docs)
    # ---
    menu.addSeparator()
    # ---
    menu.newAction(
        text='View Logs',
        slot_fn=back.view_logs)
    mainwin.viewDirectoryMenu = guitool.newMenu(mainwin, menu, 'viewDirectoryMenu', 'View Directories')
    mainwin.viewDirectoryMenu.newAction(
        text='View Log Directory',
        slot_fn=back.view_log_dir)
    mainwin.viewDirectoryMenu.newAction(
        text='View Database Directory',
        slot_fn=back.view_database_dir)
    mainwin.viewDirectoryMenu.newAction(
        text='View Application Files Directory',
        slot_fn=back.view_app_files_dir)
    # ---
    menu.addSeparator()
    # ---
    menu.newAction(
        text='Run Integrity Checks',
        slot_fn=back.run_integrity_checks)
    menu.newAction(
        text='Fix/Clean Database Integrity',
        slot_fn=back.fix_and_clean_database)


def setup_web_menu(mainwin, back):
    mainwin.menuWeb = guitool.newMenu(mainwin, mainwin.menubar, 'menuWeb', 'Web')
    menu = mainwin.menuWeb
    menu.newAction(
        text='Startup Web Interface',
        slot_fn=back.start_web_server_parallel)
    menu.newAction(
        text='Shutdown Web Interface',
        slot_fn=back.kill_web_server_parallel)
    menu.addSeparator()
    menu.newAction(
        text='Startup Wildbook',
        slot_fn=back.startup_wildbook)
    menu.newAction(
        text='Shutdown Wildbook',
        slot_fn=back.shutdown_wildbook)
    menu.addSeparator()
    menu.newAction(
        text='Browse Wildbook',
        slot_fn=back.browse_wildbook)
    menu.newAction(
        text='Force Wildbook Name Change',
        slot_fn=back.force_wildbook_namechange)
    menu.addSeparator()
    menu.newAction(
        text='Install Wildbook',
        slot_fn=back.install_wildbook)


#def setup_wildbook_menu(mainwin, back):
#    mainwin.menuWildbook = guitool.newMenu(mainwin, mainwin.menubar, 'menuWildbook', 'Wildbook')
#    menu = mainwin.menuWildbook


def setup_developer_menu(mainwin, back):
    """ DEV MENU """
    mainwin.menuDev = guitool.newMenu(mainwin, mainwin.menubar, 'menuDev', 'Dev')
    menu = mainwin.menuDev
    menu.newAction(
        name='actionDeveloper_mode',
        text='Developer IPython',
        shortcut='Ctrl+Shift+I',
        slot_fn=back.dev_mode)
    menu.newAction(
        text='Set Work Directory',
        slot_fn=back.set_workdir)
    # TESTS
    mainwin.menuTests = guitool.newMenu(mainwin, menu, 'menuTests', 'Tests')
    mainwin.menuTests.newAction(
        text='Run IBEIS Tests',
        slot_fn=back.run_tests)
    mainwin.menuTests.newAction(
        text='Run Utool Tests',
        slot_fn=back.run_utool_tests)
    mainwin.menuTests.newAction(
        text='Run Vtool Tests',
        slot_fn=back.run_vtool_tests)
    mainwin.menuTests.newAction(
        text='Assert Modules',
        slot_fn=back.assert_modules)
    # --- TESTS --
    menu.addSeparator()
    menu.newAction(
        name='actionDeveloper_DumpDB',
        text='Dump SQL Database',
        slot_fn=back.dev_dumpdb)
    menu.addSeparator()
    menu.newAction(
        name='actionLocalizeImages',
        text='Localize Images',
        slot_fn=back.localize_images)
    menu.addSeparator()
    menu.newAction(
        name='export_learning_data',
        text='Export learning data',
        slot_fn=back.dev_export_annotations)
    menu.newAction(
        name='actionTrainWithEncounters',
        text='Train RF with Open Encounter',
        slot_fn=back.train_rf_with_encounter)
    menu.addSeparator()  # ---------
    adv_ieq_menu = mainwin.menuAdvancedIEQuery = guitool.newMenu(mainwin, menu, 'menuAdvancedIEQuery', 'Advanced Intra Encounter Queries')
    adv_exq_menu = mainwin.menuAdvancedEXQuery = guitool.newMenu(mainwin, menu, 'menuAdvancedEXQuery', 'Advanced Vs Exemplar Queries')
    menu.addSeparator()  # ---------
    adv_ieq_menu.newAction(
        name='actionBatchUnknownIntraEncounterQueries',
        text='Query: Unknown Intra Encounter',
        slot_fn=functools.partial(back.compute_queries, query_is_known=False, daids_mode=const.INTRA_ENC_KEY),
    )
    adv_exq_menu.newAction(
        name='actionBatchUnknownVsExemplarQueries',
        text='Query: Unknowns vs Exemplars',
        slot_fn=functools.partial(back.compute_queries, query_is_known=False, daids_mode=const.VS_EXEMPLARS_KEY),
    )
    adv_exq_menu.newAction(
        name='actionNameVsExemplarsQuery',
        text='Query: Names vs Exemplar',
        slot_fn=functools.partial(back.compute_queries,
                                  use_prioritized_name_subset=True,
                                  daids_mode=const.VS_EXEMPLARS_KEY,
                                  cfgdict=dict(can_match_samename=False, use_k_padding=False)),
    )
    adv_exq_menu.newAction(
        name='actionNameVsExemplarsMode3',
        text='Query: Names vs Exemplar + Ori Hack + Scale + No Affine',
        slot_fn=functools.partial(back.compute_queries,
                                  use_prioritized_name_subset=True,
                                  daids_mode=const.VS_EXEMPLARS_KEY,
                                  cfgdict=dict(can_match_samename=False, use_k_padding=False,
                                               affine_invariance=False, scale_max=150, augment_queryside_hack=True)),
    )
    adv_ieq_menu.newAction(
        name='actionQueryInEncMode1',
        text='Query: Names Intra Encounter With OriAugment',
        slot_fn=functools.partial(back.compute_queries, daids_mode=const.INTRA_ENC_KEY,
                                  use_prioritized_name_subset=True,
                                  cfgdict=dict(augment_queryside_hack=True, can_match_samename=False, use_k_padding=False)),
    )
    adv_exq_menu.newAction(
        name='actionQueryVsExempMode2',
        text='Query: Names VsExamplar With OriAugment',
        slot_fn=functools.partial(back.compute_queries, daids_mode=const.VS_EXEMPLARS_KEY,
                                  use_prioritized_name_subset=True,
                                  cfgdict=dict(augment_queryside_hack=True, can_match_samename=False, use_k_padding=False)),
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='takeScreenshot',
        text='Take Screenshot',
        shortcut='Ctrl+]',
        slot_fn=back.take_screenshot)
    setup_depricated_menu(mainwin, back)


def setup_refresh_menu(mainwin, back):
    mainwin.menuRefresh = guitool.newMenu(mainwin, mainwin.menubar, 'menuRefresh', 'Refresh')
    menu = mainwin.menuRefresh
    # ---------
    menu.newAction(
        name='actionDeveloper_CLS',
        text='Refresh Tables',
        shortcut='Ctrl+Shift+C',
        slot_fn=back.dev_cls)
    # ---------
    menu.newAction(
        name='actionUpdateSpecialEncounters',
        text='Refresh Special Encounters',
        slot_fn=back.update_special_encounters)
    # ---------
    menu.newAction(
        name='actionReconnectController',
        text='Reconnect Controller',
        slot_fn=back.reconnect_controller)
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionPrecomputeANNOTATIONFeatures',
        text='Precompute Chips/Features',
        #shortcut='Ctrl+Return',
        slot_fn=back.compute_feats)
    menu.newAction(
        name='actionPrecomputeThumbnails',
        text='Precompute Thumbnails',
        slot_fn=back.compute_thumbs)
    # ---------
    menu.addSeparator()
    # ---------
    menu.newAction(
        name='actionRedownload_Detection_Models',
        text='Redownload Detection Models',
        slot_fn=back.redownload_detection_models)
    # ---------
    menu.addSeparator()
    # ---------
    menu.newAction(
        name='actionDelete_Precomputed_Results',
        text='Delete Cached Query Results',
        slot_fn=back.delete_queryresults_dir)
    menu.newAction(
        name='actionDelete_Cache_Directory',
        text='Delete Database Cache',
        slot_fn=back.delete_cache)
    menu.newAction(
        name='actionDelete_global_preferences',
        text='Delete Global Preferences',
        slot_fn=back.delete_global_prefs)
    menu.newAction(
        name='actionDeleteThumbnails',
        text='Delete Thumbnails',
        slot_fn=back.delete_thumbnails)


def setup_depricated_menu(mainwin, back):
    #mainwin.menuDepr = guitool.newMenu(mainwin, mainwin.menubar, 'menuDepr', 'Depricated')
    mainwin.menuDepr = guitool.newMenu(mainwin, mainwin.menuDev, 'menuDepr', 'Depricated')
    menu = mainwin.menuDepr
    menu.addSeparator()  # ---------
    #menu.newAction(
    #    name='actionCompute_Queries',
    #    text='Query: Old Style',
    #    tooltip='''This might take anywhere from a coffee break to an
    #                overnight procedure depending on how many ANNOTATIONs you\'ve
    #                made. It queries each chip and saves the result which
    #                allows multiple queries to be rapidly inspected later.''',
    #    #shortcut='Ctrl+4',
    #    slot_fn=back.compute_queries)
    menu.addSeparator()  # ---------
    menu.newAction(
        text='Query: Incremental',
        slot_fn=back.incremental_query
    )
    menu.newAction(
        text='Import Cropped Images As Annotations (select file(s))',
        slot_fn=back.import_images_as_annots_from_file)
    menu.addSeparator()
    menu.newAction(
        text='Developer Reload',
        shortcut='Ctrl+Shift+R',
        slot_fn=back.dev_reload)
    menu.newAction(
        text='Reviewed All Encounter Images',
        slot_fn=back.encounter_reviewed_all_images)

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.gui.guimenus --test-setup_dummy_menus
        python -m ibeis.gui.guimenus
        python -m ibeis.gui.guimenus --allexamples
        python -m ibeis.gui.guimenus --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
