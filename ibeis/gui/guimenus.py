from guitool.guitool_components import newMenu, newMenubar, msg_event


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
    mainwin.menubar = newMenubar(mainwin)
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
    mainwin.menuFile = newMenu(mainwin, mainwin.menubar, 'menuFile', 'File')
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
        name='actionSave_Database',
        tooltip='Saves csv tables',
        text='Save Database',
        shortcut='Ctrl+S',
        slot_fn=back.save_database)
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
    mainwin.menuFile.addSeparator()
    mainwin.menuFile.newAction(
        name='actionQuit',
        text='Quit',
        shortcut='',
        slot_fn=back.quit)


def setup_actions_menu(mainwin, back):
    """ ACTIONS MENU """
    mainwin.menuActions = newMenu(mainwin, mainwin.menubar, 'menuActions', 'Actions')
    mainwin.menuActions.newAction(
        name='actionAdd_ROI',
        text='Add ROI',
        shortcut='A',
        slot_fn=back.add_roi)
    mainwin.menuActions.newAction(
        name='actionQuery',
        text='Query',
        shortcut='Q',
        slot_fn=back.query)
    mainwin.menuActions.addSeparator()
    mainwin.menuActions.newAction(
        name='actionReselect_ROI',
        text='Reselect ROI Bbox',
        shortcut='R',
        slot_fn=back.reselect_roi)
    mainwin.menuActions.newAction(
        name='actionReselect_Ori',
        text='Reselect ROI Orientation',
        shortcut='O',
        slot_fn=back.reselect_ori)
    mainwin.menuActions.addSeparator()
    mainwin.menuActions.newAction(
        name='actionNext',
        text='Select Next',
        shortcut='N',
        slot_fn=back.select_next)
    mainwin.menuActions.newAction(
        name='actionPrev',
        text='Select Previous',
        shortcut='P',
        slot_fn=back.select_prev)
    mainwin.menuActions.addSeparator()
    mainwin.menuActions.newAction(
        name='actionDelete_ROI',
        text='Delete ROI',
        shortcut='Ctrl+Del',
        slot_fn=back.delete_roi)
    mainwin.menuActions.newAction(
        name='actionDelete_Image',
        text='Trash Image',
        shortcut='',
        slot_fn=back.delete_image)


def setup_batch_menu(mainwin, back):
    """ BATCH MENU """
    mainwin.menuBatch = newMenu(mainwin, mainwin.menubar, 'menuBatch', 'Batch')
    mainwin.menuBatch.newAction(
        name='actionPrecomputeROIFeatures',
        text='Precompute Chips/Features',
        shortcut='Ctrl+Return',
        slot_fn=back.precompute_feats)
    mainwin.menuBatch.newAction(
        name='actionPrecompute_Queries',
        text='Precompute Queries',
        tooltip='''This might take anywhere from a coffee break to an
                    overnight procedure depending on how many ROIs you\'ve
                    made. It queries each chip and saves the result which
                    allows multiple queries to be rapidly inspected later.''',
        shortcut='',
        slot_fn=back.precompute_queries)
    mainwin.menuBatch.newAction(
        name='actionDetect_Grevys_Quick',
        text='Detect Grevys Quick',
        slot_fn=back.detect_grevys_quick)
    mainwin.menuBatch.newAction(
        name='actionDetect_Grevys_Fine',
        text='Detect Grevys Fine',
        slot_fn=back.detect_grevys_fine)
    mainwin.menuBatch.addSeparator()
    mainwin.menuBatch.newAction(
        name='actionCompute_Encounters',
        text='Compute Encounters',
        shortcut='Ctrl+E',
        slot_fn=back.compute_encounters)
    mainwin.menuBatch.addSeparator()


def setup_option_menu(mainwin, back):
    """ OPTIONS MENU """
    mainwin.menuOptions = newMenu(mainwin, mainwin.menubar, 'menuOptions', 'Options')
    mainwin.menuOptions.newAction(
        name='actionLayout_Figures',
        text='Layout Figures',
        tooltip='Organizes windows in a grid',
        shortcut='Ctrl+L',
        slot_fn=back.layout_figures)
    mainwin.menuOptions.addSeparator()
    mainwin.menuOptions.newAction(
        name='actionPreferences',
        text='Edit Preferences',
        tooltip='Changes algorithm parameters and program behavior.',
        shortcut='Ctrl+P',
        slot_fn=back.edit_preferences)


def setup_help_menu(mainwin, back):
    """ HELP MENU """
    mainwin.menuHelp = newMenu(mainwin, mainwin.menubar, 'menuHelp', 'Help')
    about_msg = 'IBEIS = Image Based Ecological Information System'
    mainwin.menuHelp.newAction(
        name='actionAbout',
        text='About',
        shortcut='',
        slot_fn=msg_event('About', about_msg))
    mainwin.menuHelp.newAction(
        name='actionView_Docs',
        text='View Documentation',
        shortcut='',
        slot_fn=back.view_docs)
    # ---
    mainwin.menuHelp.addSeparator()
    # ---
    mainwin.menuHelp.newAction(
        name='actionView_DBDir',
        text='View Database Directory',
        shortcut='',
        slot_fn=back.view_database_dir)
    # ---
    mainwin.menuHelp.addSeparator()
    # ---
    mainwin.menuHelp.newAction(
        name='actionDelete_Precomputed_Results',
        text='Delete Cached Query Results',
        shortcut='',
        slot_fn=back.delete_queryresults_dir)
    mainwin.menuHelp.newAction(
        name='actionDelete_computed_directory',
        text='Delete computed directory',
        shortcut='',
        slot_fn=back.delete_cache)
    mainwin.menuHelp.newAction(
        name='actionDelete_global_preferences',
        text='Delete Global Preferences',
        shortcut='',
        slot_fn=back.delete_global_prefs)


def setup_developer_menu(mainwin, back):
    """ DEV MENU """
    mainwin.menuDev = newMenu(mainwin, mainwin.menubar, 'menuDev', 'Dev')
    mainwin.menuDev.newAction(
        name='actionDeveloper_reload',
        text='Developer Reload',
        shortcut='Ctrl+Shift+R',
        slot_fn=back.dev_reload)
    mainwin.menuDev.newAction(
        name='actionDeveloper_mode',
        text='Developer IPython',
        shortcut='Ctrl+Shift+I',
        slot_fn=back.dev_mode)
    mainwin.menuDev.newAction(
        name='actionDeveloper_CLS',
        text='CLS',
        shortcut='Ctrl+Shift+C',
        slot_fn=back.dev_cls)
    mainwin.menuDev.newAction(
        name='actionDeveloper_DumpDB',
        text='Dump SQL Database',
        slot_fn=back.dev_dumpdb)
