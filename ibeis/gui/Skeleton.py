from __future__ import absolute_import, division, print_function
import sys
from os.path import expanduser
sys.path.append(expanduser('~/code/ibeis'))
import utool
utool.inject_colored_exceptions()
from PyQt4 import QtCore, QtGui
import utool  # NOQA
from ibeis.gui.frontend_helpers import *  # NOQA


class Ui_mainSkel(object):
    def setupUi(ui, front):
        setup_ui(ui, front, front.back)

    def postsetupUI(ui):
        print('[skel] Calling Postsetup')
        for func in ui.postsetup_fns:
            func()

    def retranslateUi(ui, front):
        print('[skel] Calling Retranslate')
        for func in ui.retranslatable_fns:
            func()


def newEncounterTabs(front, parent, suffix=None):
    if suffix is None:
        tab_text = 'database'
        suffix = ''
    else:
        tab_text = 'encounter' + suffix
    tabWidget = newTabbedTabWidget(front, parent,
                                   'tablesView' + suffix,
                                   'tablesTabWidget' + suffix,
                                   tab_text,
                                   vstretch=10)
    tabWidget.newTabbedTable('gids_view' + suffix,
                             'gids_TBL' + suffix,
                             'Image Table' + suffix)
    tabWidget.newTabbedTable('rids_view' + suffix,
                             'rids_TBL' + suffix,
                             'ROI Table' + suffix)
    tabWidget.newTabbedTable('nids_view' + suffix,
                             'nids_TBL' + suffix,
                             'Name Table' + suffix)
    tabWidget.newTabbedTable('qres_view' + suffix,
                             'qres_TBL' + suffix,
                             'Query Result Table' + suffix)


def setup_ui(ui, front, back):
    ui.retranslatable_fns = []  # A list of retranslatable functions
    ui.connect_fns = []  # A list of signals / slots to connect
    ui.postsetup_fns = []

    back = front.back

    setup_main_layout(ui, front, back)

    # ENCOUNTER SUPERTABS
    encountersTabWidget  = newTabWidget(front, ui.splitter, 'encountersTabWidget', vstretch=10)
    newEncounterTabs(front, encountersTabWidget, suffix=None)
    #newEncounterTabs(front, encountersTabWidget, suffix='1')
    #newEncounterTabs(front, encountersTabWidget, suffix='2')

    #ui.database_view = ui.encountersTabWidget.newTabView('database_view'

    # TAB TABLES
    #ui.tablesTabWidget  = newTabWidget(front, , 'tablesTabWidget', vstretch=10)

    # Split Panes
    ui.progressBar = newProgressBar(ui.splitter, visible=False)
    ui.outputEdit  = newOutputEdit(ui.splitter, visible=False)

    # Menus
    setup_file_menu(ui, front, back)
    setup_actions_menu(ui, front, back)
    setup_batch_menu(ui, front, back)
    setup_option_menu(ui, front, back)
    setup_help_menu(ui, front, back)
    setup_developer_menu(ui, front, back)

    ui.postsetupUI()

    ui.tablesTabWidget.setCurrentIndex(0)
    QtCore.QMetaObject.connectSlotsByName(front)
    ui.retranslateUi(front)


def setup_file_menu(ui, front, back):
    """ FILE MENU """
    ui.menuFile = newMenu(front, ui.menubar, 'menuFile', 'File')
    ui.menuFile.newAction(
        name='actionNew_Database',
        text='New Database',
        tooltip='Create a new folder to use as a database.',
        shortcut='Ctrl+N',
        slot_fn=back.new_database)
    ui.menuFile.newAction(
        name='actionOpen_Database',
        text='Open Database',
        tooltip='Opens a different database directory.',
        shortcut='Ctrl+O',
        slot_fn=back.open_database)
    ui.menuFile.addSeparator()
    ui.menuFile.newAction(
        name='actionSave_Database',
        tooltip='Saves csv tables',
        text='Save Database',
        shortcut='Ctrl+S',
        slot_fn=back.save_database)
    ui.menuFile.addSeparator()
    ui.menuFile.newAction(
        name='actionImport_Img_file',
        text='Import Images (select file(s))',
        shortcut=None,
        slot_fn=back.import_images_from_file)
    ui.menuFile.newAction(
        name='actionImport_Img_dir',
        text='Import Images (select directory)',
        shortcut='Ctrl+I',
        slot_fn=back.import_images_from_dir)
    ui.menuFile.addSeparator()
    ui.menuFile.newAction(
        name='actionQuit',
        text='Quit',
        shortcut='',
        slot_fn=back.quit)


def setup_actions_menu(ui, front, back):
    """ ACTIONS MENU """
    ui.menuActions = newMenu(front, ui.menubar, 'menuActions', 'Actions')
    ui.menuActions.newAction(
        name='actionAdd_ROI',
        text='Add ROI',
        shortcut='A',
        slot_fn=back.add_roi)
    ui.menuActions.newAction(
        name='actionQuery',
        text='Query',
        shortcut='Q',
        slot_fn=back.query)
    ui.menuActions.addSeparator()
    ui.menuActions.newAction(
        name='actionReselect_ROI',
        text='Reselect ROI Bbox',
        shortcut='R',
        slot_fn=back.reselect_roi)
    ui.menuActions.newAction(
        name='actionReselect_Ori',
        text='Reselect ROI Orientation',
        shortcut='O',
        slot_fn=back.reselect_ori)
    ui.menuActions.addSeparator()
    ui.menuActions.newAction(
        name='actionNext',
        text='Select Next',
        shortcut='N',
        slot_fn=back.select_next)
    ui.menuActions.newAction(
        name='actionPrev',
        text='Select Previous',
        shortcut='P',
        slot_fn=back.select_prev)
    ui.menuActions.addSeparator()
    ui.menuActions.newAction(
        name='actionDelete_ROI',
        text='Delete ROI',
        shortcut='Ctrl+Del',
        slot_fn=back.delete_roi)
    ui.menuActions.newAction(
        name='actionDelete_Image',
        text='Trash Image',
        shortcut='',
        slot_fn=back.delete_image)


def setup_batch_menu(ui, front, back):
    """ BATCH MENU """
    ui.menuBatch = newMenu(front, ui.menubar, 'menuBatch', 'Batch')
    ui.menuBatch.newAction(
        name='actionPrecomputeROIFeatures',
        text='Precompute Chips/Features',
        shortcut='Ctrl+Return',
        slot_fn=back.precompute_feats)
    ui.menuBatch.newAction(
        name='actionPrecompute_Queries',
        text='Precompute Queries',
        tooltip='''This might take anywhere from a coffee break to an
                    overnight procedure depending on how many ROIs you\'ve
                    made. It queries each chip and saves the result which
                    allows multiple queries to be rapidly inspected later.''',
        shortcut='',
        slot_fn=back.precompute_queries)
    ui.menuBatch.addSeparator()
    ui.menuBatch.newAction(
        name='actionCompute_Encounters',
        text='Compute Encounters',
        shortcut='Ctrl+E',
        slot_fn=back.compute_encounters)
    ui.menuBatch.addSeparator()


def setup_option_menu(ui, front, back):
    """ OPTIONS MENU """
    ui.menuOptions = newMenu(front, ui.menubar, 'menuOptions', 'Options')
    ui.menuOptions.newAction(
        name='actionLayout_Figures',
        text='Layout Figures',
        tooltip='Organizes windows in a grid',
        shortcut='Ctrl+L',
        slot_fn=back.layout_figures)
    ui.menuOptions.addSeparator()
    ui.menuOptions.newAction(
        name='actionPreferences',
        text='Edit Preferences',
        tooltip='Changes algorithm parameters and program behavior.',
        shortcut='Ctrl+P',
        slot_fn=back.edit_preferences)


def setup_help_menu(ui, front, back):
    """ HELP MENU """
    ui.menuHelp = newMenu(front, ui.menubar, 'menuHelp', 'Help')
    ui.menuHelp.newAction(
        name='actionAbout',
        text='About',
        shortcut='',
        slot_fn=msg_event('About', 'IBEIS = Image Based Ecological Information System'))
    ui.menuHelp.newAction(
        name='actionView_Docs',
        text='View Documentation',
        shortcut='',
        slot_fn=back.view_docs)
    # ---
    ui.menuHelp.addSeparator()
    # ---
    ui.menuHelp.newAction(
        name='actionView_DBDir',
        text='View Database Directory',
        shortcut='',
        slot_fn=back.view_database_dir)
    # ---
    ui.menuHelp.addSeparator()
    # ---
    ui.menuHelp.newAction(
        name='actionDelete_Precomputed_Results',
        text='Delete Cached Query Results',
        shortcut='',
        slot_fn=back.delete_queryresults_dir)
    ui.menuHelp.newAction(
        name='actionDelete_computed_directory',
        text='Delete computed directory',
        shortcut='',
        slot_fn=back.delete_cache)
    ui.menuHelp.newAction(
        name='actionDelete_global_preferences',
        text='Delete Global Preferences',
        shortcut='',
        slot_fn=back.delete_global_prefs)


def setup_developer_menu(ui, front, back):
    """ DEV MENU """
    ui.menuDev = newMenu(front, ui.menubar, 'menuDev', 'Dev')
    ui.menuDev.newAction(
        name='actionDeveloper_reload',
        text='Developer Reload',
        shortcut='Ctrl+Shift+R',
        slot_fn=back.dev_reload)
    ui.menuDev.newAction(
        name='actionDeveloper_mode',
        text='Developer IPython',
        shortcut='Ctrl+Shift+I',
        slot_fn=back.dev_mode)


def setup_main_layout(ui, front, back):
    initMainWidget(front, 'mainSkel', size=(1000, 600), title='IBEIS - No Database Opened')
    ui.centralwidget, ui.verticalLayout = newCentralLayout(front)
    ui.splitter = newVerticalSplitter(ui.centralwidget, ui.verticalLayout)
    ui.menubar = newMenubar(front, 'menubar')

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    front = QtGui.QMainWindow()
    front.ui = Ui_mainSkel()
    front.ui.setupUi(front)
    front.show()
    sys.exit(app.exec_())
