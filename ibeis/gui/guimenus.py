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
        return None


def setup_menus(mainwin, back=None):
    print('[guimenus] creating menus')
    mainwin.menubar = guitool.newMenubar(mainwin)
    if back is None:
        back = DummyBack()
    setup_file_menu(mainwin, back)
    setup_actions_menu(mainwin, back)
    setup_batch_menu(mainwin, back)
    setup_option_menu(mainwin, back)
    setup_help_menu(mainwin, back)
    setup_developer_menu(mainwin, back)


def setup_file_menu(mainwin, back):
    """ FILE MENU """
    mainwin.menuFile = guitool.newMenu(mainwin, mainwin.menubar, 'menuFile', 'File')
    mainwin.menuFile.newAction(
        name='actionNew_Database',
        text='New Database',
        tooltip='Create a new folder to use as a database.',
        shortcut='Ctrl+N',
        slot_fn=back.new_database)
    mainwin.menuFile.newAction(
        name='actionOpen_Database',
        text='Open Database',
        tooltip='Opens a different database directory.',
        shortcut='Ctrl+O',
        slot_fn=back.open_database)
    mainwin.menuFile.addSeparator()
    mainwin.menuFile.newAction(
        name='actionBackup_Database',
        tooltip='Backup the current main database.',
        text='Backup Database',
        shortcut='Ctrl+B',
        slot_fn=back.backup_database)
    mainwin.menuFile.newAction(
        name='actionExport_Database',
        tooltip='Dumps and exports database as csv tables.',
        text='Export Database',
        shortcut='Ctrl+S',
        slot_fn=back.export_database)
    mainwin.menuFile.addSeparator()
    mainwin.menuFile.newAction(
        name='actionImport_Img_file',
        text='Import Images (select file(s))',
        shortcut=None,
        slot_fn=back.import_images_from_file)
    mainwin.menuFile.newAction(
        name='actionImport_Img_dir',
        text='Import Images (select directory)',
        shortcut='Ctrl+I',
        slot_fn=back.import_images_from_dir)
    mainwin.menuFile.newAction(
        name='actionImport_Img_AsAnnot_file',
        text='Import Cropped Images As Annotations (select file(s))',
        shortcut=None,
        slot_fn=back.import_images_as_annots_from_file)
    mainwin.menuFile.addSeparator()
    mainwin.menuFile.newAction(
        name='actionImport_Img_file_with_smart',
        text='Import Images (select file(s)) with smart Patrol XML',
        shortcut=None,
        slot_fn=back.import_images_from_file_with_smart)
    mainwin.menuFile.newAction(
        name='actionImport_Img_dir_with_smart',
        text='Import Images (select directory) with smart Patrol XML',
        shortcut=None,
        slot_fn=back.import_images_from_dir_with_smart)
    mainwin.menuFile.addSeparator()
    mainwin.menuFile.newAction(
        name='actionLocalizeImages',
        text='Localize Images',
        shortcut=None,
        slot_fn=back.localize_images)
    mainwin.menuFile.addSeparator()
    mainwin.menuFile.newAction(
        name='actionQuit',
        text='Quit',
        shortcut='',
        slot_fn=back.quit)


def setup_actions_menu(mainwin, back):
    """ ACTIONS MENU """
    mainwin.menuActions = guitool.newMenu(mainwin, mainwin.menubar, 'menuActions', 'Actions')
    mainwin.menuActions.newAction(
        name='actionQuery',
        text='Query Single Annotation',
        shortcut='Q',
        slot_fn=functools.partial(back.compute_queries, use_visual_selection=True))
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
    mainwin.menuActions.addSeparator()
    mainwin.menuActions.newAction(
        name='actionDeleteAllEncounters',
        text='Delete All Encounters',
        shortcut='',
        slot_fn=back.delete_all_encounters)
    mainwin.menuActions.newAction(
        name='actionDelete_Image',
        text='Delete Image',
        shortcut='',
        slot_fn=back.delete_image)
    mainwin.menuActions.newAction(
        name='actionDelete_ANNOTATION',
        text='Delete Annotation',
        shortcut='Ctrl+Del',
        slot_fn=back.delete_annot)
    mainwin.menuActions.addSeparator()
    mainwin.menuActions.newAction(
        name='actionTrainWithEncounters',
        text='Train RF with Open Encounter',
        shortcut='',
        slot_fn=back.train_rf_with_encounter)
    mainwin.menuActions.addSeparator()
    mainwin.menuActions.newAction(
        name='toggleThumbnails',
        text='Toggle Thumbnails',
        shortcut='',
        slot_fn=back.toggle_thumbnails)


def setup_batch_menu(mainwin, back):
    """ BATCH MENU """
    mainwin.menuBatch = guitool.newMenu(mainwin, mainwin.menubar, 'menuBatch', 'Batch')
    mainwin.menuBatch.newAction(
        name='actionCompute_Encounters',
        text='Cluster Encounters',
        #shortcut='Ctrl+2',
        slot_fn=back.compute_encounters)
    mainwin.menuBatch.addSeparator()  # ---------
    mainwin.menuBatch.newAction(
        name='actionDetect',
        text='Run Detection',
        #shortcut='Ctrl+3',
        slot_fn=back.run_detection)
    mainwin.menuBatch.addSeparator()  # ---------
    mainwin.menuBatch.newAction(
        name='actionCompute_Queries',
        text='Compute Old Style Queries',
        tooltip='''This might take anywhere from a coffee break to an
                    overnight procedure depending on how many ANNOTATIONs you\'ve
                    made. It queries each chip and saves the result which
                    allows multiple queries to be rapidly inspected later.''',
        #shortcut='Ctrl+4',
        slot_fn=back.compute_queries)
    mainwin.menuBatch.newAction(
        name='actionComputeIncremental_Queries',
        text='Compute Incremental Queries',
        slot_fn=back.incremental_query
    )
    mainwin.menuBatch.addSeparator()  # ---------
    mainwin.menuBatch.newAction(
        name='actionBatchIntraEncounterQueries',
        text='Compute Batch IntraEncounter Queries',
        slot_fn=functools.partial(back.compute_queries, query_mode=const.INTRA_ENC_KEY),
        tooltip=ut.textblock(
            '''
            all-encounter-annots VS all-encounter-annots (nonjunk)
            ''')
    )
    mainwin.menuBatch.newAction(
        name='actionBatchVsExemplarQueries',
        text='Compute Batch VsExemplar Queries',
        slot_fn=functools.partial(back.compute_queries, query_mode=const.VS_EXEMPLARS_KEY),
        tooltip=ut.textblock(
            '''
            all-encounter-annots VS all-exemplar-annots (nonjunk)
            ''')
    )
    mainwin.menuBatch.addSeparator()  # ---------
    mainwin.menuBatch.newAction(
        name='actionBatchUnknownIntraEncounterQueries',
        text='Compute Batch Unknown IntraEncounter Queries',
        slot_fn=functools.partial(back.compute_queries, query_is_known=False, query_mode=const.INTRA_ENC_KEY),
        tooltip=ut.textblock(
            '''
            unnamed-encounter-annots VS all-encounter-annots (nonjunk)
            ''')
    )
    mainwin.menuBatch.newAction(
        name='actionBatchUnknownVsExemplarQueries',
        text='Compute Batch Unknown VsExemplar Queries',
        slot_fn=functools.partial(back.compute_queries, query_is_known=False, query_mode=const.VS_EXEMPLARS_KEY),
        tooltip=ut.textblock(
            '''
            unnamed-encounter-annots VS all-exemplar-annots (nonjunk)
            ''')
    )
    mainwin.menuBatch.addSeparator()  # ---------
    mainwin.menuBatch.newAction(
        name='actionSetExemplarsFromQualityAndViewpoint',
        text='Set Exemplars from Quality and Viewpoint',
        slot_fn=back.set_exemplars_from_quality_and_viewpoint,
        tooltip=ut.textblock(
            '''
            Uses the quality and viewpoint column to pick the best N exemplars
            per viewpoint, per name.
            ''')
    )
    mainwin.menuBatch.newAction(
        name='actionBatchConsecutiveLocationSpeciesRename',
        text='Consecutive Location+Species Rename',
        slot_fn=back.batch_rename_consecutive_via_species,
        tooltip=ut.textblock(
            '''
            Renames ALL the names in the database to
            {other_cfg.location_for_names}_{species_code}_{num}
            ''')
    )

    mainwin.menuBatch.addSeparator()  # ---------
    mainwin.menuBatch.newAction(
        name='actionShipProcessedEncounters',
        text='Ship Processed Encounters',
        tooltip='''This action will ship to WildBook any encounters that have
                    been marked as processed.  This can also be used to send
                    processed encounters that failed to ship correctly.''',
        #shortcut='Ctrl+5',
        slot_fn=back.send_unshipped_processed_encounters)
    mainwin.menuBatch.addSeparator()  # ---------
    mainwin.menuBatch.newAction(
        name='actionEncounterImagesReviewed',
        text='Reviewed All Encounter Images',
        shortcut='',
        slot_fn=back.encounter_reviewed_all_images)
    mainwin.menuBatch.newAction(
        name='actionPrecomputeANNOTATIONFeatures',
        text='Precompute Chips/Features',
        #shortcut='Ctrl+Return',
        slot_fn=back.compute_feats)
    mainwin.menuBatch.newAction(
        name='actionPrecomputeThumbnails',
        text='Precompute Thumbnails',
        shortcut='',
        slot_fn=back.compute_thumbs)
    mainwin.menuBatch.addSeparator()  # ---------


def setup_option_menu(mainwin, back):
    """ OPTIONS MENU """
    mainwin.menuOptions = guitool.newMenu(mainwin, mainwin.menubar, 'menuOptions', 'Options')
    mainwin.menuOptions.newAction(
        name='actionLayout_Figures',
        text='Layout Figures',
        tooltip='Organizes windows in a grid',
        shortcut='Ctrl+L',
        slot_fn=back.layout_figures)
    mainwin.menuOptions.newAction(
        name='actionToggleQueryMode',
        text='Toggle Query Mode: ----',
        tooltip='Changes behavior of Actions->Query',
        slot_fn=functools.partial(back.set_query_mode, 'toggle'))
    mainwin.menuOptions.addSeparator()
    mainwin.menuOptions.newAction(
        name='actionPreferences',
        text='Edit Preferences',
        tooltip='Changes algorithm parameters and program behavior.',
        shortcut='Ctrl+P',
        slot_fn=back.edit_preferences)


def setup_help_menu(mainwin, back):
    """ HELP MENU """
    mainwin.menuHelp = guitool.newMenu(mainwin, mainwin.menubar, 'menuHelp', 'Help')
    about_msg = 'IBEIS = Image Based Ecological Information System\nhttp://ibeis.org/'
    mainwin.menuHelp.newAction(
        name='actionAbout',
        text='About',
        shortcut='',
        slot_fn=guitool.msg_event('About', about_msg))
    #mainwin.menuHelp.newAction(
    #    name='actionView_Docs',
    #    text='View Documentation',
    #    shortcut='',
    #    slot_fn=back.view_docs)
    # ---
    mainwin.menuHelp.addSeparator()
    # ---
    mainwin.menuHelp.newAction(
        name='actionView_DBDir',
        text='View Database Directory',
        shortcut='',
        slot_fn=back.view_database_dir)
    mainwin.menuHelp.newAction(
        name='actionView_App_Files_Dir',
        text='View Application Files Directory',
        shortcut='',
        slot_fn=back.view_app_files_dir)
    mainwin.menuHelp.newAction(
        name='actionViewLogsDir',
        text='View Log Directory',
        shortcut='',
        slot_fn=back.view_log_dir)
    # ---
    mainwin.menuHelp.addSeparator()
    # ---
    mainwin.menuHelp.newAction(
        name='actionRedownload_Detection_Models',
        text='Redownload Detection Models',
        shortcut='',
        slot_fn=back.redownload_detection_models)
    mainwin.menuHelp.newAction(
        name='actionFixCleanDatabase',
        text='Fix/Clean Database',
        shortcut='',
        slot_fn=back.fix_and_clean_database)
    mainwin.menuHelp.newAction(
        name='actionConsistencyCheck',
        text='Run Consistency Checks',
        shortcut='',
        slot_fn=back.run_consistency_checks)
    mainwin.menuHelp.addSeparator()
    mainwin.menuHelp.newAction(
        name='actionDelete_Precomputed_Results',
        text='Delete Cached Query Results',
        shortcut='',
        slot_fn=back.delete_queryresults_dir)
    mainwin.menuHelp.newAction(
        name='actionDelete_Cache_Directory',
        text='Delete Database Cache',
        shortcut='',
        slot_fn=back.delete_cache)
    mainwin.menuHelp.newAction(
        name='actionDelete_global_preferences',
        text='Delete Global Preferences',
        shortcut='',
        slot_fn=back.delete_global_prefs)
    mainwin.menuHelp.newAction(
        name='actionDeleteThumbnails',
        text='Delete Thumbnails',
        shortcut='',
        slot_fn=back.delete_thumbnails)


def setup_developer_menu(mainwin, back):
    """ DEV MENU """
    mainwin.menuDev = guitool.newMenu(mainwin, mainwin.menubar, 'menuDev', 'Dev')
    mainwin.menuDev.newAction(
        name='actionExpandNamesTree',
        text='Expand Names Tree',
        slot_fn=mainwin.expand_names_tree)
    mainwin.menuDev.newAction(
        name='actionDeveloper_reload',
        text='Developer Reload',
        shortcut='Ctrl+Shift+R',
        slot_fn=back.dev_reload)
    mainwin.menuDev.newAction(
        name='actionDevRunTests',
        text='Run Developer Tests',
        slot_fn=back.run_tests)
    mainwin.menuDev.newAction(
        name='actionDeveloper_mode',
        text='Developer IPython',
        shortcut='Ctrl+Shift+I',
        slot_fn=back.dev_mode)
    mainwin.menuDev.newAction(
        name='actionDeveloper_CLS',
        text='Refresh Tables',
        shortcut='Ctrl+Shift+C',
        slot_fn=back.dev_cls)
    mainwin.menuDev.newAction(
        name='actionDeveloper_DumpDB',
        text='Dump SQL Database',
        slot_fn=back.dev_dumpdb)
    mainwin.menuDev.newAction(
        name='export_learning_data',
        text='Export learning data',
        slot_fn=back.dev_export_annotations)
