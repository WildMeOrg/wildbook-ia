from guitool.guitool_components import newMenu, newMenubar, msg_event


class DummyBack(object):
    def __init__(self):
        pass
    def __getattr__(self, name):
        print(name)
        if name.startswith('_'):
            return self.__dict__[name]
        return None


def setup_menus(ibswin):
    print('[guimenus] creating menus')
    ibswin.menubar = newMenubar(ibswin)
    back = ibswin.back
    if back is None:
        back = DummyBack()
    setup_file_menu(ibswin, back)
    setup_actions_menu(ibswin, back)
    setup_batch_menu(ibswin, back)
    setup_option_menu(ibswin, back)
    setup_help_menu(ibswin, back)
    setup_developer_menu(ibswin, back)


def setup_file_menu(ibswin, back):
    """ FILE MENU """
    ibswin.menuFile = newMenu(ibswin, ibswin.menubar, 'menuFile', 'File')
    ibswin.menuFile.newAction(
        name='actionNew_Database',
        text='New Database',
        tooltip='Create a new folder to use as a database.',
        shortcut='Ctrl+N',
        slot_fn=back.new_database)
    ibswin.menuFile.newAction(
        name='actionOpen_Database',
        text='Open Database',
        tooltip='Opens a different database directory.',
        shortcut='Ctrl+O',
        slot_fn=back.open_database)
    ibswin.menuFile.addSeparator()
    ibswin.menuFile.newAction(
        name='actionSave_Database',
        tooltip='Saves csv tables',
        text='Save Database',
        shortcut='Ctrl+S',
        slot_fn=back.save_database)
    ibswin.menuFile.addSeparator()
    ibswin.menuFile.newAction(
        name='actionImport_Img_file',
        text='Import Images (select file(s))',
        shortcut=None,
        slot_fn=back.import_images_from_file)
    ibswin.menuFile.newAction(
        name='actionImport_Img_dir',
        text='Import Images (select directory)',
        shortcut='Ctrl+I',
        slot_fn=back.import_images_from_dir)
    ibswin.menuFile.addSeparator()
    ibswin.menuFile.newAction(
        name='actionQuit',
        text='Quit',
        shortcut='',
        slot_fn=back.quit)


def setup_actions_menu(ibswin, back):
    """ ACTIONS MENU """
    ibswin.menuActions = newMenu(ibswin, ibswin.menubar, 'menuActions', 'Actions')
    ibswin.menuActions.newAction(
        name='actionAdd_ROI',
        text='Add ROI',
        shortcut='A',
        slot_fn=back.add_roi)
    ibswin.menuActions.newAction(
        name='actionQuery',
        text='Query',
        shortcut='Q',
        slot_fn=back.query)
    ibswin.menuActions.addSeparator()
    ibswin.menuActions.newAction(
        name='actionReselect_ROI',
        text='Reselect ROI Bbox',
        shortcut='R',
        slot_fn=back.reselect_roi)
    ibswin.menuActions.newAction(
        name='actionReselect_Ori',
        text='Reselect ROI Orientation',
        shortcut='O',
        slot_fn=back.reselect_ori)
    ibswin.menuActions.addSeparator()
    ibswin.menuActions.newAction(
        name='actionNext',
        text='Select Next',
        shortcut='N',
        slot_fn=back.select_next)
    ibswin.menuActions.newAction(
        name='actionPrev',
        text='Select Previous',
        shortcut='P',
        slot_fn=back.select_prev)
    ibswin.menuActions.addSeparator()
    ibswin.menuActions.newAction(
        name='actionDelete_ROI',
        text='Delete ROI',
        shortcut='Ctrl+Del',
        slot_fn=back.delete_roi)
    ibswin.menuActions.newAction(
        name='actionDelete_Image',
        text='Trash Image',
        shortcut='',
        slot_fn=back.delete_image)


def setup_batch_menu(ibswin, back):
    """ BATCH MENU """
    ibswin.menuBatch = newMenu(ibswin, ibswin.menubar, 'menuBatch', 'Batch')
    ibswin.menuBatch.newAction(
        name='actionPrecomputeROIFeatures',
        text='Precompute Chips/Features',
        shortcut='Ctrl+Return',
        slot_fn=back.precompute_feats)
    ibswin.menuBatch.newAction(
        name='actionPrecompute_Queries',
        text='Precompute Queries',
        tooltip='''This might take anywhere from a coffee break to an
                    overnight procedure depending on how many ROIs you\'ve
                    made. It queries each chip and saves the result which
                    allows multiple queries to be rapidly inspected later.''',
        shortcut='',
        slot_fn=back.precompute_queries)
    ibswin.menuBatch.newAction(
        name='actionDetect_Grevys_Quick',
        text='Detect Grevys Quick',
        slot_fn=back.detect_grevys_quick)
    ibswin.menuBatch.newAction(
        name='actionDetect_Grevys_Fine',
        text='Detect Grevys Fine',
        slot_fn=back.detect_grevys_fine)
    ibswin.menuBatch.addSeparator()
    ibswin.menuBatch.newAction(
        name='actionCompute_Encounters',
        text='Compute Encounters',
        shortcut='Ctrl+E',
        slot_fn=back.compute_encounters)
    ibswin.menuBatch.addSeparator()


def setup_option_menu(ibswin, back):
    """ OPTIONS MENU """
    ibswin.menuOptions = newMenu(ibswin, ibswin.menubar, 'menuOptions', 'Options')
    ibswin.menuOptions.newAction(
        name='actionLayout_Figures',
        text='Layout Figures',
        tooltip='Organizes windows in a grid',
        shortcut='Ctrl+L',
        slot_fn=back.layout_figures)
    ibswin.menuOptions.addSeparator()
    ibswin.menuOptions.newAction(
        name='actionPreferences',
        text='Edit Preferences',
        tooltip='Changes algorithm parameters and program behavior.',
        shortcut='Ctrl+P',
        slot_fn=back.edit_preferences)


def setup_help_menu(ibswin, back):
    """ HELP MENU """
    ibswin.menuHelp = newMenu(ibswin, ibswin.menubar, 'menuHelp', 'Help')
    about_msg = 'IBEIS = Image Based Ecological Information System'
    ibswin.menuHelp.newAction(
        name='actionAbout',
        text='About',
        shortcut='',
        slot_fn=msg_event('About', about_msg))
    ibswin.menuHelp.newAction(
        name='actionView_Docs',
        text='View Documentation',
        shortcut='',
        slot_fn=back.view_docs)
    # ---
    ibswin.menuHelp.addSeparator()
    # ---
    ibswin.menuHelp.newAction(
        name='actionView_DBDir',
        text='View Database Directory',
        shortcut='',
        slot_fn=back.view_database_dir)
    # ---
    ibswin.menuHelp.addSeparator()
    # ---
    ibswin.menuHelp.newAction(
        name='actionDelete_Precomputed_Results',
        text='Delete Cached Query Results',
        shortcut='',
        slot_fn=back.delete_queryresults_dir)
    ibswin.menuHelp.newAction(
        name='actionDelete_computed_directory',
        text='Delete computed directory',
        shortcut='',
        slot_fn=back.delete_cache)
    ibswin.menuHelp.newAction(
        name='actionDelete_global_preferences',
        text='Delete Global Preferences',
        shortcut='',
        slot_fn=back.delete_global_prefs)


def setup_developer_menu(ibswin, back):
    """ DEV MENU """
    ibswin.menuDev = newMenu(ibswin, ibswin.menubar, 'menuDev', 'Dev')
    ibswin.menuDev.newAction(
        name='actionDeveloper_reload',
        text='Developer Reload',
        shortcut='Ctrl+Shift+R',
        slot_fn=back.dev_reload)
    ibswin.menuDev.newAction(
        name='actionDeveloper_mode',
        text='Developer IPython',
        shortcut='Ctrl+Shift+I',
        slot_fn=back.dev_mode)
    ibswin.menuDev.newAction(
        name='actionDeveloper_CLS',
        text='CLS',
        shortcut='Ctrl+Shift+C',
        slot_fn=back.dev_cls)
    ibswin.menuDev.newAction(
        name='actionDeveloper_DumpDB',
        text='Dump SQL Database',
        slot_fn=back.dev_dumpdb)
