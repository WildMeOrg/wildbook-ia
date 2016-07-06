from __future__ import absolute_import, division, print_function
import sys
from os.path import expanduser
sys.path.append(expanduser('~/code/ibeis'))
import utool
utool.inject_colored_exceptions()
from PyQt4 import QtCore, QtGui
import utool  # NOQA
from ibeis.gui.frontend_helpers import *  # NOQA


QTRANSLATE = QtWidgets.QApplication.translate
QUTF8      = QtWidgets.QApplication.UnicodeUTF8


class Ui_mainSkel(object):
    def setupUi(ui, front):
        setup_ui(ui, front, front.back)
        ui.postsetupUI()
        ui.tablesTabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(front)
        ui.retranslateUi(front)
        ui.connectUi()

    def postsetupUI(ui):
        #print('[skel] Calling Postsetup')
        for func in ui.postsetup_fns:
            func()

    def retranslateUi(ui, front):
        #print('[skel] Calling Retranslate')
        for key, text in ui.retranslate_dict.iteritems():
            obj, setter = key
            frontname = front.objectName()
            #print('TRANSLATE %s.%s to %r' % (objname, setter,
            #                                 text))
            qtext = QTRANSLATE(frontname, text, None, QUTF8)
            getattr(obj, setter)(qtext)

    def connectUi(ui):
        # Connect all signals from GUI
        for key, slot_fn in ui.connection_dict.iteritems():
            obj, attr = key
            key_sig = (obj.objectName(), attr, slot_fn.func_name)
            # Do not connect signals more than once
            if key_sig in ui.connected_signals:
                ui.connected_signals
                continue
            ui.connected_signals.add(key_sig)
            #print('CONNECT %s.%s to %r' % (obj.objectName(), attr,
            #                               slot_fn.func_name))
            getattr(obj, attr).connect(slot_fn)

    def ensureImageSetTab(ui, front, imagesettext):
        """ Ensure imageset tab for specific imagesettext """
        parent = ui.imagesetsTabWidget
        # ImageSetText Sanitization
        if imagesettext == '' or imagesettext == 'None':
            imagesettext = None
        if imagesettext not in ui.imagesettext_dict:
            # Create the imageset tab
            tabWidget = newImageSetTabs(front, parent, imagesettext=imagesettext)
            ui.imagesettext_dict[imagesettext] = tabWidget
        ui.retranslateUi(front)

    def deleteImageSetTab(ui, front, imagesettext):
        """ Delete imageset tab for specific imagesettext """
        # ImageSetText Sanitization
        if imagesettext == '' or imagesettext == 'None':
            imagesettext = None
        try:  # Remove the imageset tab
            tabWiget = ui.imagesettext_dict[imagesettext]
            ui.deleteImageSetTab(front, imagesettext)
            del tabWiget
        except KeyError:
            pass
        ui.retranslateUi(front)


def setup_ui(ui, front, back):
    ui.imagesettext_dict = {}
    ui.connected_signals = set()
    ui.connection_dict  = {}  # dict of signal / slots to connect
    ui.retranslate_dict = {}
    ui.retranslatable_fns = []  # A list of retranslatable functions
    ui.postsetup_fns      = []

    back = front.back

    setup_main_layout(ui, front, back)

    # IMAGESET SUPERTABS
    ui.imagesetsTabWidget = newTabWidget(front, ui.splitter,
                                          'imagesetsTabWidget', vstretch=10)
    ui.ensureImageSetTab(front, imagesettext=None)

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


def newImageSetTabs(front, parent, imagesettext=None):
    if imagesettext is None or imagesettext == 'None' or imagesettext == '':
        tab_text = 'database'
        imagesettext = ''
    else:
        tab_text = str(imagesettext)
    tabWidget = newTabbedTabWidget(front, parent,
                                   'tablesView' + imagesettext,
                                   'tablesTabWidget' + imagesettext,
                                   tab_text,
                                   vstretch=10)
    tabWidget.newTabbedTable('gids', imagesettext, 'Image Table',
                             clicked_slot_fn=front.gids_tbl_clicked,
                             pressed_slot_fn=front.rowid_tbl_pressed,
                             changed_slot_fn=front.gids_tbl_changed)
    tabWidget.newTabbedTable('rids', imagesettext, 'ROI Table',
                             clicked_slot_fn=front.rids_tbl_clicked,
                             pressed_slot_fn=front.rowid_tbl_pressed,
                             changed_slot_fn=front.rids_tbl_changed)
    tabWidget.newTabbedTable('nids', imagesettext, 'Name Table',
                             clicked_slot_fn=front.nids_tbl_clicked,
                             pressed_slot_fn=front.rowid_tbl_pressed,
                             changed_slot_fn=front.nids_tbl_clicked)
    tabWidget.newTabbedTable('qres', imagesettext, 'Query Result Table',
                             clicked_slot_fn=front.qres_tbl_clicked,
                             pressed_slot_fn=front.rowid_tbl_pressed,
                             changed_slot_fn=front.qres_tbl_changed)


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
    ui.menuBatch.newAction(
        name='actionDetect_Grevys_Quick',
        text='Detect Grevys Quick',
        slot_fn=back.detect_grevys_quick)
    ui.menuBatch.newAction(
        name='actionDetect_Grevys_Fine',
        text='Detect Grevys Fine',
        slot_fn=back.detect_grevys_fine)
    ui.menuBatch.addSeparator()
    ui.menuBatch.newAction(
        name='actionCompute_ImageSets',
        text='Compute ImageSets',
        shortcut='Ctrl+E',
        slot_fn=back.compute_occurrences)
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
    about_msg = 'IBEIS = Image Based Ecological Information System'
    ui.menuHelp.newAction(
        name='actionAbout',
        text='About',
        shortcut='',
        slot_fn=msg_event('About', about_msg))
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
    ui.menuDev.newAction(
        name='actionDeveloper_CLS',
        text='CLS',
        shortcut='Ctrl+Shift+C',
        slot_fn=back.dev_cls)
    ui.menuDev.newAction(
        name='actionDeveloper_DumpDB',
        text='Dump SQL Database',
        slot_fn=back.dev_dumpdb)


def setup_main_layout(ui, front, back):
    default_title = 'IBEIS - No Database Opened'
    initMainWidget(front, 'mainSkel', size=(1000, 600), title=default_title)
    ui.centralwidget, ui.verticalLayout = newCentralLayout(front)
    ui.splitter = newVerticalSplitter(ui.centralwidget, ui.verticalLayout)
    ui.menubar = newMenubar(front, 'menubar')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    front = QtWidgets.QMainWindow()
    front.ui = Ui_mainSkel()
    front.ui.setupUi(front)
    front.show()
    sys.exit(app.exec_())
