# -*- coding: utf-8 -*-
"""
This module defines all of the menu items in the main GUI
as well as their callbacks in guiback
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import functools
from wbia import constants as const
import wbia.guitool as gt

ut.noinject(__name__, '[guimenus]')


class DummyBack(object):
    def __init__(self):
        print('using dummy back')
        pass

    def __getattr__(self, name):
        # print(name)
        if name.startswith('_'):
            return self.__dict__[name]
        import mock

        mock.Mock()
        return mock.Mock()


def setup_dummy_menus():
    r"""
    CommandLine:
        python -m wbia.gui.guimenus --test-setup_dummy_menus

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.guimenus import *  # NOQA
        >>> result = setup_dummy_menus()
        >>> print(result)
    """
    # import unittest
    import wbia.guitool as gt

    gt.ensure_qapp()  # must be ensured before any embeding
    mainwin = gt.QtWidgets.QMainWindow()
    back = DummyBack()
    import mock

    mainwin.expand_names_tree = mock.Mock
    setup_menus(mainwin, back)
    mainwin.show()
    mainwin.resize(600, 100)
    # ut.embed()
    gt.qtapp_loop(mainwin, frequency=100)
    # gt.qtapp_loop(mainwin, frequency=100, ipy=ut.inIPython())


def setup_menus(mainwin, back=None):
    if ut.VERBOSE:
        print('[guimenus] creating menus')
    mainwin.menubar = gt.newMenubar(mainwin)
    if back is None:
        back = DummyBack()
    setup_file_menu(mainwin, back)
    setup_view_menu(mainwin, back)
    setup_actions_menu(mainwin, back)
    # setup_batch_menu(mainwin, back)
    # setup_checks_menu(mainwin, back)
    setup_option_menu(mainwin, back)
    setup_refresh_menu(mainwin, back)
    # setup_wildbook_menu(mainwin, back)
    setup_web_menu(mainwin, back)
    setup_help_menu(mainwin, back)
    setup_developer_menu(mainwin, back)
    setup_zebra_menu(mainwin, back)


def setup_file_menu(mainwin, back):
    """ FILE MENU """
    mainwin.menuFile = mainwin.menubar.newMenu('File')
    menu = mainwin.menuFile
    menu.newAction(
        name='actionNew_Database',
        text='New Database',
        tooltip='Create a new folder to use as a database.',
        shortcut='Ctrl+N',
        triggered=back.new_database,
    )
    menu.newAction(
        name='actionOpen_Database',
        text='Open Database',
        tooltip='Opens a different database folder.',
        shortcut='Ctrl+O',
        triggered=back.open_database,
    )
    menu.addSeparator()
    menu.newAction(
        name='actionBackup_Database',
        tooltip='Backup the current main database.',
        text='Backup Database',
        shortcut='Ctrl+B',
        triggered=back.backup_database,
    )
    menu.newAction(
        name='actionExport_Database',
        tooltip='Dumps and exports database as csv tables.',
        text='Export As CSV',
        triggered=back.export_database_as_csv,
    )
    menu.newAction(
        name='actionDuplicate_Database',
        tooltip='Creates a duplicate of the database',
        text='Duplicate Database',
        triggered=back.make_database_duplicate,
    )
    menu.addSeparator()
    menu.newAction(
        name='actionImport_Img_file',
        text='Import Images (select file(s))',
        triggered=back.import_images_from_file,
    )
    menu.newAction(
        name='actionImport_Img_dir',
        text='Import Images (select directory)',
        shortcut='Ctrl+I',
        triggered=back.import_images_from_dir,
    )
    menu.addSeparator()
    # menu.newAction(
    #    name='actionImport_Img_file_with_smart',
    #    text='Import Images (select file(s)) with smart Patrol XML',
    #    triggered=back.import_images_from_file_with_smart)
    menu.newAction(
        name='actionImport_Img_dir_with_smart',
        text='Import Images (select directory) with smart Patrol XML',
        triggered=back.import_images_from_dir_with_smart,
    )
    menu.addSeparator()
    menu.newAction(
        name='actionImport_Img_dir_from_encouters_1',
        text='Import Images (select folder(s)) from Encounters (1 level)',
        triggered=back.import_images_from_encounters_1,
    )
    menu.newAction(
        name='actionImport_Img_dir_from_encouters_2',
        text='Import Images (select folder(s)) from Encounters (2 levels)',
        triggered=back.import_images_from_encounters_2,
    )
    menu.addSeparator()
    menu.newAction(name='actionQuit', text='Quit', triggered=back.quit)


def setup_view_menu(mainwin, back):
    mainwin.menuView = mainwin.menubar.newMenu('View')
    menu = mainwin.menuView
    menu.addSeparator()
    menu.newAction(
        name='actionExpandNamesTree',
        text='Expand Names Tree',
        triggered=mainwin.expand_names_tree,
    )
    menu.addSeparator()
    menu.newAction(
        name='toggleThumbnails',
        text='Toggle Thumbnails',
        triggered=back.toggle_thumbnails,
    )
    menu.newAction(
        name='toggleOutput',
        text='Toggle Output Log',
        triggered=back.toggle_output_widget,
    )
    # menu.newAction(
    #     name='actionLayout_Figures',
    #     text='Layout Figures',
    #     tooltip='Organizes windows in a grid',
    #     shortcut='Ctrl+L',
    #     triggered=back.layout_figures)
    pass


def setup_actions_menu(mainwin, back):
    """ ACTIONS MENU """
    mainwin.menuActions = mainwin.menubar.newMenu('Actions')
    menu = mainwin.menuActions
    menu.newAction(
        name='actionCompute_Occurrences',
        text='Group Occurrences',
        # shortcut='Ctrl+2',
        triggered=back.do_group_occurrence_step,
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionDetect',
        text='Run Detection',
        # shortcut='Ctrl+3',
        triggered=back.run_detection_step,
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionQuery',
        text='Query Single Annotation',
        shortcut='Q',
        triggered=functools.partial(back.compute_queries, use_visual_selection=True),
    )
    menu.newAction(
        name='actionBatchIntraOccurrenceQueries',
        text='Query: Intra Occurrence',
        triggered=functools.partial(
            back.compute_queries, daids_mode=const.INTRA_OCCUR_KEY
        ),
    )
    menu.newAction(
        name='actionBatchVsExemplarQueries',
        text='Query: vs Exemplars',
        triggered=functools.partial(
            back.compute_queries, daids_mode=const.VS_EXEMPLARS_KEY
        ),
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionSetExemplarsFromQualityAndViewpoint',
        text='Set Exemplars from Quality and Viewpoint',
        triggered=back.set_exemplars_from_quality_and_viewpoint_,
        tooltip=ut.textblock(
            """
            Uses the quality and viewpoint column to pick the best N exemplars
            per viewpoint, per name.
            """
        ),
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionAdvancedID',
        text='Advanced ID Interface',
        triggered=back.show_advanced_id_interface,
        shortcut='Ctrl+G',
    )
    menu.newAction(
        name='actionRunMergeChecks',
        text='Run Merge Checks (Exemplars vs Exemplars)',
        triggered=back.run_merge_checks,
    )
    mainwin.mergeMenu = menu.newMenu('Other Merge Checks')
    mainwin.mergeMenu.newAction(
        name='actionRunMergeChecks2',
        text='Run Merge Checks (multitons)',
        triggered=back.run_merge_checks_multitons,
    )
    menu.addSeparator()  # ---------
    if not const.SIMPLIFY_INTERFACE:
        menu.newAction(
            name='actionBatchConsecutiveLocationSpeciesRename',
            text='Consecutive Location+Species Rename',
            triggered=back.batch_rename_consecutive_via_species_,
            tooltip=ut.textblock(
                """
                Renames ALL the names in the database to
                {other_cfg.location_for_names}_{species_code}_{num}
                """
            ),
        )

    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionShipProcessedOccurrences',
        text='Ship Processed Occurrences',
        tooltip="""This action will ship to WildBook any occurrences that have
                    been marked as processed.  This can also be used to send
                    processed imagesets that failed to ship correctly.""",
        # shortcut='Ctrl+5',
        triggered=back.send_unshipped_processed_imagesets,
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        text='Override All Annotation Species',
        triggered=back.override_all_annotation_species,
    )
    menu.newAction(text='Rename Species', triggered=back.update_species_nice_name)
    menu.newAction(text='Delete Species', triggered=back.delete_selected_species)
    menu.addSeparator()  # ---------
    menu.newAction(
        name='actionDeleteAllImageSets',
        text='Delete All ImageSets',
        triggered=back.delete_all_imagesets,
    )
    menu.newAction(
        name='actionDelete_Image', text='Delete Image', triggered=back.delete_image
    )
    menu.newAction(
        name='actionDelete_ANNOTATION',
        text='Delete Annotation',
        shortcut='Ctrl+Del',
        triggered=back.delete_annot,
    )
    menu.addSeparator()


def setup_batch_menu(mainwin, back):
    """ BATCH MENU """
    mainwin.menuBatch = mainwin.menubar.newMenu('Batch')
    menu = mainwin.menuBatch
    menu


def setup_option_menu(mainwin, back):
    """ OPTIONS MENU """
    mainwin.menuOptions = mainwin.menubar.newMenu('Options')
    menu = mainwin.menuOptions
    mainwin.actionToggleQueryMode = menu.newAction(
        name='actionToggleQueryMode',
        text='Toggle Query Mode: ----',
        tooltip='Changes behavior of Actions->Query',
        triggered=functools.partial(back.set_daids_mode, 'toggle'),
    )
    menu.addSeparator()
    menu.newAction(
        name='actionPreferences',
        text='Edit Preferences',
        tooltip='Changes algorithm parameters and program behavior.',
        shortcut='Ctrl+P',
        triggered=back.edit_preferences,
    )


def setup_checks_menu(mainwin, back):
    mainwin.menuChecks = mainwin.menubar.newMenu('Checks')
    pass


def setup_help_menu(mainwin, back):
    """ HELP MENU """
    mainwin.menuHelp = mainwin.menubar.newMenu('Help')
    menu = mainwin.menuHelp
    # from wbia.control import DB_SCHEMA_CURRENT
    # version = DB_SCHEMA_CURRENT.VERSION_CURRENT
    menu.newAction(name='actionAbout', text='About', triggered=back.show_about_message)
    menu.newAction(
        name='actionDBInfo', text='Database Info', triggered=back.display_dbinfo
    ),
    # menu.newAction(
    #    name='actionView_Docs',
    #    text='View Documentation',
    #    triggered=back.view_docs)
    # ---
    menu.addSeparator()
    # ---
    menu.newAction(text='View Global Logs', triggered=back.view_logs_global)
    mainwin.viewDirectoryMenu = menu.newMenu('View Directories')
    mainwin.viewDirectoryMenu.newAction(
        text='View Local Log Directory', triggered=back.view_log_dir_local
    )
    mainwin.viewDirectoryMenu.newAction(
        text='View Global Log Directory', triggered=back.view_log_dir_global
    )
    mainwin.viewDirectoryMenu.newAction(
        text='View Database Directory', triggered=back.view_database_dir
    )
    mainwin.viewDirectoryMenu.newAction(
        text='View Application Files Directory', triggered=back.view_app_files_dir
    )
    # ---
    menu.addSeparator()
    # ---
    menu.newAction(text='Run Integrity Checks', triggered=back.run_integrity_checks)
    menu.newAction(
        text='Fix/Clean Database Integrity', triggered=back.fix_and_clean_database
    )


def setup_web_menu(mainwin, back):
    mainwin.menuWeb = mainwin.menubar.newMenu('Web')
    menu = mainwin.menuWeb
    menu.newAction(text='Startup Web Interface', triggered=back.start_web_server_parallel)
    menu.newAction(text='Shutdown Web Interface', triggered=back.kill_web_server_parallel)
    menu.addSeparator()
    menu.newAction(text='Startup Wildbook', triggered=back.startup_wildbook)
    menu.newAction(text='Shutdown Wildbook', triggered=back.shutdown_wildbook)
    menu.addSeparator()
    menu.newAction(text='Browse Wildbook', triggered=back.browse_wildbook)
    menu.newAction(
        text='Force Wildbook Name Change', triggered=back.force_wildbook_namechange
    )
    menu.addSeparator()
    menu.newAction(text='Install Wildbook', triggered=back.install_wildbook)


# def setup_wildbook_menu(mainwin, back):
#    mainwin.menuWildbook = mainwin.menubar.newMenu('Wildbook')
#    menu = mainwin.menuWildbook


def setup_developer_menu(mainwin, back):
    """ DEV MENU """
    mainwin.menuDev = mainwin.menubar.newMenu('Dev')
    menu = mainwin.menuDev
    menu.newAction(text='Download Demo Data', triggered=back.ensure_demodata)
    menu.newAction(
        name='actionMakeIPythonNotebook',
        text='Launch IPython Notebook',
        triggered=back.launch_ipy_notebook,
    )
    menu.newAction(
        name='actionDeveloper_mode',
        text='Developer IPython',
        shortcut='Ctrl+Shift+I',
        triggered=back.dev_mode,
    )
    menu.newAction(text='Graph Interface', triggered=back.make_qt_graph_interface)
    menu.newAction(text='Set Work Directory', triggered=back.set_workdir)
    # --- TESTS --
    menu.addSeparator()
    menu.newAction(
        name='actionLocalizeImages',
        text='Localize Images',
        triggered=back.localize_images,
    )
    menu.addSeparator()
    menu.newAction(
        name='export_learning_data',
        text='Export learning data',
        triggered=back.dev_export_annotations,
    )
    menu.newAction(
        name='actionTrainWithImageSets',
        text='Train RF with Open ImageSet',
        triggered=back.train_rf_with_imageset,
    )
    menu.addSeparator()  # ---------
    adv_ieq_menu = mainwin.menuAdvancedIEQuery = menu.newMenu(
        'Advanced Intra Occurrence Queries'
    )
    adv_exq_menu = mainwin.menuAdvancedEXQuery = menu.newMenu(
        'Advanced Vs Exemplar Queries'
    )
    menu.addSeparator()  # ---------
    adv_ieq_menu.newAction(
        name='actionBatchUnknownIntraImageSetQueries',
        text='Query: Unknown Intra Occurrence',
        triggered=functools.partial(
            back.compute_queries, query_is_known=False, daids_mode=const.INTRA_OCCUR_KEY
        ),
    )
    adv_exq_menu.newAction(
        name='actionBatchUnknownVsExemplarQueries',
        text='Query: Unknowns vs Exemplars',
        triggered=functools.partial(
            back.compute_queries, query_is_known=False, daids_mode=const.VS_EXEMPLARS_KEY,
        ),
    )
    adv_exq_menu.newAction(
        name='actionNameVsExemplarsQuery',
        text='Query: Names vs Exemplar',
        triggered=functools.partial(
            back.compute_queries,
            use_prioritized_name_subset=True,
            daids_mode=const.VS_EXEMPLARS_KEY,
            cfgdict=dict(can_match_samename=False, use_k_padding=False),
        ),
    )
    adv_exq_menu.newAction(
        name='actionNameVsExemplarsMode3',
        text='Query: Names vs Exemplar + Ori Hack + Scale + No Affine',
        triggered=functools.partial(
            back.compute_queries,
            use_prioritized_name_subset=True,
            daids_mode=const.VS_EXEMPLARS_KEY,
            cfgdict=dict(
                can_match_samename=False,
                use_k_padding=False,
                affine_invariance=False,
                scale_max=150,
                query_rotation_heuristic=True,
            ),
        ),
    )
    adv_ieq_menu.newAction(
        name='actionQueryInEncMode1',
        text='Query: Names Intra Occurrence With OriAugment',
        triggered=functools.partial(
            back.compute_queries,
            daids_mode=const.INTRA_OCCUR_KEY,
            use_prioritized_name_subset=True,
            cfgdict=dict(
                query_rotation_heuristic=True,
                can_match_samename=False,
                use_k_padding=False,
            ),
        ),
    )
    adv_exq_menu.newAction(
        name='actionQueryVsExempMode2',
        text='Query: Names VsExamplar With OriAugment',
        triggered=functools.partial(
            back.compute_queries,
            daids_mode=const.VS_EXEMPLARS_KEY,
            use_prioritized_name_subset=True,
            cfgdict=dict(
                query_rotation_heuristic=True,
                can_match_samename=False,
                use_k_padding=False,
            ),
        ),
    )
    menu.addSeparator()  # ---------
    menu.newAction(
        name='takeScreenshot',
        text='Take Screenshot',
        # shortcut='Ctrl+]',
        shortcut='Ctrl+K',
        triggered=back.take_screenshot,
    )
    setup_depricated_menu(mainwin, back)


def setup_refresh_menu(mainwin, back):
    mainwin.menuRefresh = mainwin.menubar.newMenu('Refresh')
    menu = mainwin.menuRefresh
    # ---------
    menu.newAction(
        name='actionDeveloper_CLS',
        text='Refresh Tables',
        shortcut='Ctrl+Shift+C',
        triggered=back.dev_cls,
    )
    # ---------
    menu.newAction(
        name='actionUpdateSpecialImageSets',
        text='Refresh Special ImageSets',
        triggered=back.update_special_imagesets_,
    )
    # ---------
    menu.newAction(
        name='actionReconnectController',
        text='Reconnect Controller',
        triggered=back.reconnect_controller,
    )
    menu.addSeparator()  # ---------
    # ---------
    menu.newAction(
        name='actionRedownload_Detection_Models',
        text='Redownload Detection Models',
        triggered=back.redownload_detection_models,
    )
    # ---------
    menu.addSeparator()
    # ---------
    menu.newAction(
        name='actionDelete_Precomputed_Results',
        text='Delete Cached Query Results',
        triggered=back.delete_queryresults_dir,
    )
    menu.newAction(
        name='actionDelete_Cache_Directory',
        text='Delete Database Cache',
        triggered=back.delete_cache,
    )
    menu.newAction(
        name='actionDelete_global_preferences',
        text='Delete Global Preferences',
        triggered=back.delete_global_prefs,
    )
    menu.newAction(
        name='actionDeleteThumbnails',
        text='Delete Thumbnails',
        triggered=back.delete_thumbnails,
    )


def setup_depricated_menu(mainwin, back):
    # mainwin.menuDepr = mainwin.menubar.newMenu('Depricated')
    mainwin.menuDepr = mainwin.menuDev.newMenu('Depricated')
    menu = mainwin.menuDepr
    menu.addSeparator()  # ---------
    # menu.newAction(
    #    name='actionCompute_Queries',
    #    text='Query: Old Style',
    #    tooltip='''This might take anywhere from a coffee break to an
    #                overnight procedure depending on how many ANNOTATIONs you\'ve
    #                made. It queries each chip and saves the result which
    #                allows multiple queries to be rapidly inspected later.''',
    #    #shortcut='Ctrl+4',
    #    triggered=back.compute_queries)
    menu.addSeparator()  # ---------
    # menu.newAction(
    #     text='Query: Incremental',
    #     triggered=back.incremental_query
    # )
    menu.newAction(
        text='Import Cropped Images As Annotations (select file(s))',
        triggered=back.import_images_as_annots_from_file,
    )
    menu.addSeparator()
    menu.newAction(
        text='Developer Reload', shortcut='Ctrl+Shift+R', triggered=back.dev_reload
    )

    # TESTS
    mainwin.menuTests = menu.newMenu('Tests')
    mainwin.menuTests.newAction(text='Run IBEIS Tests', triggered=back.run_tests)
    mainwin.menuTests.newAction(text='Run Utool Tests', triggered=back.run_utool_tests)
    mainwin.menuTests.newAction(text='Run Vtool Tests', triggered=back.run_vtool_tests)
    mainwin.menuTests.newAction(text='Assert Modules', triggered=back.assert_modules)
    menu.newAction(text='Update Source Install', triggered=back.update_source_install)


def setup_zebra_menu(mainwin, back):
    mainwin.menuDev = mainwin.menubar.newMenu('Zebra')
    menu = mainwin.menuDev
    menu.newAction(
        name='processImagesetAsCameraTrapImages',
        text='Process ImageSet as Camera Trap Images',
        triggered=back.filter_imageset_as_camera_trap,
    )


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.gui.guimenus --test-setup_dummy_menus
        python -m wbia.gui.guimenus
        python -m wbia.gui.guimenus --allexamples
        python -m wbia.gui.guimenus --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
