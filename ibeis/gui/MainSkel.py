from __future__ import absolute_import, division, print_function
import sys
from os.path import expanduser
sys.path.append(expanduser('~/code/ibeis'))
import utool
utool.inject_colored_exceptions()
from PyQt4 import QtCore, QtGui
try:
    from ibeis.gui.frontend_helpers import *  # NOQA
except ImportError:
    try:
        from .frontend_helpers import *  # NOQA
    except ValueError:
        from frontend_helpers import *  # NOQA


class Ui_mainSkel(object):
    def setupUi(ui, mainSkel):
        ui.retranslatable_fns = []  # A list of retranslatable functions
        ui.connect_fns = []  # A list of signals / slots to connect

        front = mainSkel

        initMainWidget(front, 'mainSkel', size=(1000, 600), title='IBEIS - No Database Opened')
        ui.centralwidget, ui.verticalLayout = newCentralLayout(front)
        ui.splitter = newSplitter(ui.centralwidget, ui.verticalLayout)

        # TAB TABLES
        ui.tablesTabWidget, newTab = newTabWidget(front, ui.splitter, 'tablesTabWidget', vstretch=10)

        # IMG TAB
        ui.image_view, ui.gids_TBL = newTab('img_view', 'gids_TBL', 'Image Table')

        # ROI TAB
        ui.roi_view, ui.rids_TBL   = newTab('roi_view', 'rids_TBL', 'ROI Table')

        # NAME TAB
        ui.name_view, ui.nids_TBL  = newTab('name_view', 'nids_TBL', 'Name Table')

        # RECOGNITION TAB
        ui.qres_view, ui.qres_TBL  = newTab('qres_view', 'qres_TBL', 'Query Result Table')

        ui.progressBar = newProgressBar(ui.splitter, visible=False)
        #ui.outputEdit  = newOutputEdit(ui.splitter)

        ui.menubar = newMenubar(front)

        # Menus
        ui.menuFile,    newFileAction    = newMenu(front, ui.menubar, 'menuFile', 'File')
        ui.menuActions, newActionsAction = newMenu(front, ui.menubar, 'menuActions', 'Actions')
        ui.menuBatch,   newBatchActions  = newMenu(front, ui.menubar, 'menuBatch', 'Batch')
        ui.menuOptions, newOptionsAction = newMenu(front, ui.menubar, 'menuOptions', 'Options')
        ui.menuHelp,    newHelpAction    = newMenu(front, ui.menubar, 'menuHelp', 'Help')

        #
        #
        # FILE MENU
        ui.actionNew_Database = newFileAction(
            name='actionNew_Database',
            text='New Database',
            tooltip='Create a new folder to use as a database.',
            shortcut='Ctrl+N')
        ui.actionOpen_Database = newFileAction(
            name='actionOpen_Database',
            text='Open Database',
            tooltip='Opens a different database directory.',
            shortcut='Ctrl+O')
        # ---
        ui.menuFile.addSeparator()
        # ---
        ui.actionSave_Database = newFileAction(
            name='actionSave_Database',
            tooltip='Saves csv tables',
            text='Save Database',
            shortcut='Ctrl+S')
        # ---
        ui.menuFile.addSeparator()
        # ---
        ui.actionImport_Img_file = newFileAction(
            name='actionImport_Img_file',
            text='Import Images (select file(s))',
            shortcut=None)
        ui.actionImport_Img_dir = newFileAction(
            name='actionImport_Img_dir',
            text='Import Images (select directory)',
            shortcut='Ctrl+I')
        # ---
        ui.menuFile.addSeparator()
        # ---
        ui.actionQuit = newFileAction(
            name='actionQuit',
            text='Quit',
            shortcut='')

        #
        #
        # ACTIONS MENU
        ui.actionAdd_ROI = newActionsAction(
            name='actionAdd_ROI', text='Add ROI', shortcut='A')
        ui.actionQuery = newActionsAction(
            name='actionQuery', text='Query', shortcut='Q')
        # ---
        ui.menuActions.addSeparator()
        # ---
        ui.actionReselect_ROI = newActionsAction(
            name='actionReselect_ROI',
            text='Reselect ROI Bbox',
            shortcut='R')
        ui.actionReselect_Ori = newActionsAction(
            name='actionReselect_Ori',
            text='Reselect ROI Orientation',
            shortcut='O')
        # ---
        ui.menuActions.addSeparator()
        # ---
        ui.actionNext = newActionsAction(
            name='actionNext',
            text='Select Next',
            shortcut='N')
        ui.actionPrev = newActionsAction(
            name='actionPrev',
            text='Select Previous',
            shortcut='P')
        # ---
        ui.menuActions.addSeparator()
        # ---
        ui.actionDelete_ROI = newActionsAction(
            name='actionDelete_ROI',
            text='Delete ROI',
            shortcut='Ctrl+Del')
        ui.actionDelete_Image = newActionsAction(
            name='actionDelete_Image',
            text='Trash Image',
            shortcut='')

        #
        #
        # BATCH MENU
        #
        ui.actionPrecomputeROIFeatures = newBatchActions(
            name='actionPrecomputeROIFeatures',
            text='Precompute Chips/Features',

            shortcut='Ctrl+Return')
        #
        ui.actionPrecompute_Queries = newBatchActions(
            name='actionPrecompute_Queries',
            text='Precompute Queries',
            tooltip='''This might take anywhere from a coffee break to an
                     overnight procedure depending on how many ROIs you\'ve
                     made. It queries each chip and saves the result which
                     allows multiple queries to be rapidly inspected later.''',
            shortcut='')
        # ---
        ui.menuBatch.addSeparator()
        #
        #

        #
        #
        # OPTIONS MENU
        ui.actionLayout_Figures = newOptionsAction(
            name='actionLayout_Figures',
            text='Layout Figures',
            tooltip='Organizes windows in a grid',
            shortcut='Ctrl+L')
        # ---
        ui.menuOptions.addSeparator()
        # ---
        ui.actionPreferences = newOptionsAction(
            name='actionPreferences',
            text='Edit Preferences',
            tooltip='Changes algorithm parameters and program behavior.',
            shortcut='Ctrl+P')

        #
        #
        # HELP MENU
        ui.actionAbout = newHelpAction(
            name='actionAbout',
            text='About',
            shortcut='')
        ui.actionView_Docs = newHelpAction(
            name='actionView_Docs',
            text='View Documentation',
            shortcut='')
        # ---
        ui.menuHelp.addSeparator()
        # ---
        ui.actionView_DBDir = newHelpAction(
            name='actionView_DBDir',
            text='View Database Directory',
            shortcut='')
        # ---
        ui.menuHelp.addSeparator()
        # ---
        ui.actionDelete_Precomputed_Results = newHelpAction(
            name='actionDelete_Precomputed_Results',
            text='Delete Cached Query Results',
            shortcut='')
        ui.actionDelete_computed_directory = newHelpAction(
            name='actionDelete_computed_directory',
            text='Delete computed directory',
            shortcut='')
        ui.actionDelete_global_preferences = newHelpAction(
            name='actionDelete_global_preferences',
            text='Delete Global Preferences',
            shortcut='')
        # ---
        ui.menuHelp.addSeparator()
        # ---
        ui.actionDeveloper_Reload = newHelpAction(
            name='actionDeveloper_reload',
            text='Developer Reload',
            shortcut='Ctrl+Shift+R')

        ui.menubar.addAction(ui.menuFile.menuAction())
        ui.menubar.addAction(ui.menuActions.menuAction())
        ui.menubar.addAction(ui.menuBatch.menuAction())
        ui.menubar.addAction(ui.menuOptions.menuAction())
        ui.menubar.addAction(ui.menuHelp.menuAction())

        ui.retranslateUi(mainSkel)
        ui.tablesTabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(mainSkel)

    def retranslateUi(ui, mainSkel):
        print('[skel] Calling Retranslate')
        for func in ui.retranslatable_fns:
            func()

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    mainSkel = QtGui.QMainWindow()
    mainSkel.ui = Ui_mainSkel()
    mainSkel.ui.setupUi(mainSkel)
    mainSkel.show()
    sys.exit(app.exec_())
