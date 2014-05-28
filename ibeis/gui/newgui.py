#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import guitool
from itertools import izip
from guitool import slot_, checks_qt_error
from ibeis.gui import newgui_views
from ibeis.gui.newgui_views import IBEISTableView, EncView
from ibeis.gui.newgui_models import IBEISTableModel, EncModel
from guitool.APITableModel import ChangingModelLayout
from PyQt4 import QtGui, QtCore
from ibeis.gui import guiheaders as gh
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
from guitool.guitool_components import newMenu, newMenubar  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


def msg_event(title, msg):
    """ Returns a message event slot """
    return lambda: guitool.msgbox(title, msg)


#############################
###### Window Widgets #######
#############################


#VIEWCLASS_DICT = {
#    gh.IMAGE_TABLE     : newgui_views.ImageView,
#    gh.ROI_TABLE       : newgui_views.ROIView,
#    gh.NAME_TABLE      : newgui_views.NameView,
#    gh.ENCOUNTER_TABLE : newgui_views.EncView,
#}
#def make_modelview(ibswin, tblname):
#    ViewClass = VIEWCLASS_DICT[tblname]
#    header = ibswin.header_dict[tblname]
#    # TODO Unify these models:
#    if tblname == gh.ENCOUNTER_TABLE:
#        model = EncModel(header, parent=ibswin)
#    else:
#        model = IBEISTableModel(header, parent=ibswin)
#    view = ViewClass(parent=ibswin)
#    view.setModel(model)
#    return model, view
#ibswin._image_model, ibswin._image_view = make_modelview(ibswin, gh.IMAGE_TABLE)
#ibswin._roi_model,   ibswin._roi_view   = make_modelview(ibswin, gh.ROI_TABLE)
#ibswin._name_model,  ibswin._name_view  = make_modelview(ibswin, gh.NAME_TABLE)
#ibswin._enc_model,  ibswin._enc_view  = make_modelview(ibswin, gh.ENCOUNTER_TABLE)


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


class DummyBack(object):
    def __init__(self):
        pass
    def __getattr__(self, name):
        print(name)
        if name.startswith('_'):
            return self.__dict__[name]
        return None


class IBEISGuiWidget(QtGui.QMainWindow):
    @checks_qt_error
    def __init__(ibswin, back=None, ibs=None, parent=None):
        QtGui.QMainWindow.__init__(ibswin, parent)
        ibswin.ibs = ibs
        ibswin.back = DummyBack()
        ibswin._init_layout()
        ibswin._connect_signals_and_slots()
        ibswin.connect_ibeis_control(ibswin.ibs)

    @checks_qt_error
    def _init_layout(ibswin):
        # Menus
        #back = ibswin.back
        ibswin.centralwidget = QtGui.QWidget(ibswin)
        parent = ibswin.centralwidget

        ibswin.setCentralWidget(ibswin.centralwidget)
        ibswin.menubar = newMenubar(ibswin)
        setup_file_menu(ibswin, ibswin.back)
        #setup_actions_menu(ibswin, back)
        #setup_batch_menu(ibswin, back)
        #setup_option_menu(ibswin, back)
        #setup_help_menu(ibswin, back)
        #setup_developer_menu(ibswin, back)

        ibswin.vlayout = QtGui.QVBoxLayout(parent)
        ibswin.hsplitter = guitool.newHorizontalSplitter(parent)
        # Tabes Tab
        ibswin._tab_table_wgt = QtGui.QTabWidget(parent)
        # Models
        ibswin._image_model = IBEISTableModel(parent=parent)
        ibswin._roi_model   = IBEISTableModel(parent=parent)
        ibswin._name_model  = IBEISTableModel(parent=parent)
        ibswin._enc_model   = EncModel(parent=parent)
        # Views
        ibswin._image_view = IBEISTableView(parent=parent)
        ibswin._roi_view   = IBEISTableView(parent=parent)
        ibswin._name_view  = IBEISTableView(parent=parent)
        ibswin._enc_view   = EncView(parent=parent)
        # Add models to views
        ibswin._image_view.setModel(ibswin._image_model)
        ibswin._roi_view.setModel(ibswin._roi_model)
        ibswin._name_view.setModel(ibswin._name_model)
        ibswin._enc_view.setModel(ibswin._enc_model)
        # Add Tabes to Tables Tab
        view_list = [ibswin._image_view,
                      ibswin._roi_view,
                      ibswin._name_view]
        tblname_list = [gh.IMAGE_TABLE,
                        gh.ROI_TABLE,
                        gh.NAME_TABLE]
        for view, tblname in izip(view_list, tblname_list):
            ibswin._tab_table_wgt.addTab(view, tblname)
        # Encs Tabs
        ibswin.enc_tabwgt = newgui_views.EncoutnerTabWidget(parent=ibswin)
        # Add Other elements to the view
        ibswin.vlayout.addWidget(ibswin.enc_tabwgt)
        ibswin.vlayout.addWidget(ibswin.hsplitter)
        ibswin.hsplitter.addWidget(ibswin._enc_view)
        ibswin.hsplitter.addWidget(ibswin._tab_table_wgt)

    @checks_qt_error
    def connect_ibeis_control(ibswin, ibs):
        print('[newgui] connecting ibs control')
        if ibs is not None:
            ibs.delete_invalid_eids()
            print('[newgui] Connecting valid ibs=%r'  % ibs.get_dbname())
            ibswin.ibs = ibs
            header_dict = gh.make_ibeis_headers_dict(ibswin.ibs)
            model_list = [ibswin._image_model,
                          ibswin._roi_model,
                          ibswin._name_model,
                          ibswin._enc_model]
            tblname_list = [gh.IMAGE_TABLE,
                            gh.ROI_TABLE,
                            gh.NAME_TABLE,
                            gh.ENCOUNTER_TABLE]
            with ChangingModelLayout(model_list):
                for model, tblname in izip(model_list, tblname_list):
                    model._init_headers(**header_dict[tblname])
        print('[newgui] invalid ibs')
        ibswin.refresh_state()

    @checks_qt_error
    def refresh_state(ibswin):
        print('Refresh State')
        title = 'No Database Opened'
        if ibswin.ibs is not None:
            title = ibsfuncs.get_title(ibswin.ibs)
            model_list = [ibswin._image_model,
                          ibswin._roi_model,
                          ibswin._name_model]
            tblname_list = [gh.IMAGE_TABLE,
                            gh.ROI_TABLE,
                            gh.NAME_TABLE]
            for index, (model, tblname) in enumerate(izip(model_list, tblname_list)):
                nRows = len(model.ider())
                ibswin._tab_table_wgt.setTabText(index, tblname + str(nRows))
        ibswin.setWindowTitle(title)

    @checks_qt_error
    def _change_enc(ibswin, eid):
        ibswin._image_view._change_enc(eid)
        ibswin._roi_view._change_enc(eid)
        ibswin._name_view._change_enc(eid)

    @checks_qt_error
    def _update_enc_tab_name(ibswin, eid, enctext):
        ibswin.enc_tabwgt._update_enc_tab_name(eid, enctext)

    @checks_qt_error
    def _connect_signals_and_slots(ibswin):
        ibswin._image_view.doubleClicked.connect(ibswin.on_doubleclick_image)
        ibswin._roi_view.doubleClicked.connect(ibswin.on_doubleclick_roi)
        ibswin._name_view.doubleClicked.connect(ibswin.on_doubleclick_name)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_image(ibswin, qtindex):
        row = qtindex.row()
        model = qtindex.model()
        gid = model._get_row_id(row)
        print("Image Selected, %r (ENC %r)" % (gid, model.eid))
        print('img')

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_roi(ibswin, qtindex):
        print('roi')
        row = qtindex.row()
        model = qtindex.model()
        rid = model._get_row_id(row)
        print("ROI Selected, %r (ENC %r)" % (rid, model.eid))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_name(ibswin, qtindex):
        print('name')
        model = qtindex.model()
        row = qtindex.row()
        nid = model._get_row_id(row)
        print("Name Selected, %r (ENC %r)" % (nid, model.eid))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_encounter(ibswin, qtindex):
        print('name')
        row = qtindex.row()
        model = qtindex.model()
        eid = model._get_row_id(row)
        enctext = ibswin.ibs.get_encounter_enctext(eid)
        ibswin.enc_tabwgt._add_enc_tab(eid, enctext)
        print("Name Selected, %r (ENC %r)" % (eid, model.eid))


if __name__ == '__main__':
    import ibeis
    import guitool  # NOQA
    import sys
    ibeis._preload(mpl=False, par=False)
    print('app')

    guitool.ensure_qtapp()

    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')

    dbdir2 = ibeis.sysres.db_to_dbdir('GZ_ALL')

    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    #ibs2 = IBEISControl.IBEISController(dbdir=dbdir2)

    ibswin = IBEISGuiWidget(ibs=ibs)

    if '--cmd' in sys.argv:
        guitool.qtapp_loop(qwin=ibswin, ipy=True)
        exec(utool.ipython_execstr())
    else:
        guitool.qtapp_loop(qwin=ibswin)
