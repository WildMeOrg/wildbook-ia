#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import guitool
from itertools import izip
from PyQt4 import QtGui, QtCore
from guitool import slot_, checks_qt_error
from guitool.APITableModel import ChangingModelLayout
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui import guimenus
from ibeis.gui import newgui_views
from ibeis.gui.newgui_models import IBEISTableModel, EncModel
from ibeis.gui.newgui_views import IBEISTableView, EncView
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


#############################
###### Window Widgets #######
#############################

#class IBEISMainWindow(QtGui.QMainWindow):

class IBEISGuiWidget(QtGui.QMainWindow):
    @checks_qt_error
    def __init__(ibswin, back=None, ibs=None, parent=None):
        QtGui.QMainWindow.__init__(ibswin, parent)
        ibswin.ibs = ibs
        ibswin.back = back
        ibswin._init_layout()
        ibswin._connect_signals_and_slots()
        ibswin.connect_ibeis_control(ibswin.ibs)

    @checks_qt_error
    def _init_layout(ibswin):
        """ Layout the widgets, menus, and containers """
        # Menus
        ibswin.setUnifiedTitleAndToolBarOnMac(False)
        ibswin.resize(900, 600)
        ibswin.centralwidget = QtGui.QWidget(ibswin)
        parent = ibswin
        root = ibswin.centralwidget

        ibswin.setCentralWidget(ibswin.centralwidget)
        guimenus.setup_menus(ibswin)

        ibswin.vlayout = QtGui.QVBoxLayout(root)
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
        else:
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
