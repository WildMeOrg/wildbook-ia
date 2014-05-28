#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import guitool
from itertools import izip  # noqa
from PyQt4 import QtGui, QtCore
from guitool import slot_, checks_qt_error, ChangingModelLayout  # NOQA
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui.guiheaders import (IMAGE_TABLE, ROI_TABLE, NAME_TABLE,
                                  ENCOUNTER_TABLE)
from ibeis.gui import guimenus
from ibeis.gui.newgui_models import IBEISTableModel, EncModel
from ibeis.gui.newgui_views import IBEISTableView, EncView
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


#############################
###### Tab Widgets #######
#############################


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(enc_tabwgt, parent=None):
        QtGui.QTabWidget.__init__(enc_tabwgt, parent)
        enc_tabwgt.ibswin = parent
        enc_tabwgt.setTabsClosable(True)
        enc_tabwgt.setMaximumSize(9999, guitool.get_cplat_tab_height())
        enc_tabwgt.tabbar = enc_tabwgt.tabBar()
        enc_tabwgt.tabbar.setMovable(True)
        enc_tabwgt.setStyleSheet('border: none;')
        enc_tabwgt.tabbar.setStyleSheet('border: none;')

        enc_tabwgt.tabCloseRequested.connect(enc_tabwgt._close_tab)
        enc_tabwgt.currentChanged.connect(enc_tabwgt._on_change)

        enc_tabwgt.eid_list = []
        enc_tabwgt._add_enc_tab(None, 'Recognition Database')

    def _on_change(enc_tabwgt, index):
        """ Switch to the current encounter tab """
        if 0 <= index and index < len(enc_tabwgt.eid_list):
            eid = enc_tabwgt.eid_list[index]
            enc_tabwgt.ibswin._change_enc(eid)

    def _close_tab(enc_tabwgt, index):
        if enc_tabwgt.eid_list[index] is not None:
            enc_tabwgt.eid_list.pop(index)
            enc_tabwgt.removeTab(index)

    def _add_enc_tab(enc_tabwgt, eid, enctext):
        if eid not in enc_tabwgt.eid_list:
            # tab_name = str(eid) + ' - ' + str(enctext)
            tab_name = str(enctext)
            enc_tabwgt.addTab(QtGui.QWidget(), tab_name)

            enc_tabwgt.eid_list.append(eid)
            index = len(enc_tabwgt.eid_list) - 1
        else:
            index = enc_tabwgt.eid_list.index(eid)

        enc_tabwgt.setCurrentIndex(index)
        enc_tabwgt._on_change(index)

    def _update_enc_tab_name(enc_tabwgt, eid, enctext):
        for index, _id in enumerate(enc_tabwgt.eid_list):
            if eid == _id:
                enc_tabwgt.setTabText(index, enctext)


#############################
###### Window Widgets #######
#############################

#class IBEISMainWindow(QtGui.QMainWindow):
class IBEISGuiWidget(QtGui.QMainWindow):
    #@checks_qt_error
    def __init__(ibswin, back=None, ibs=None, parent=None):
        QtGui.QMainWindow.__init__(ibswin, parent)
        ibswin.ibs = ibs
        ibswin.back = back
        ibswin._init_layout()
        ibswin._connect_signals_and_slots()
        ibswin.connect_ibeis_control(ibswin.ibs)

    #@checks_qt_error
    def _init_layout(ibswin):
        """ Layout the widgets, menus, and containers """
        # Define the abstract item models and views for the tables
        ibswin.modelview_defs = [
            (IMAGE_TABLE,     IBEISTableModel, IBEISTableView),
            (ROI_TABLE,       IBEISTableModel, IBEISTableView),
            (NAME_TABLE,      IBEISTableModel, IBEISTableView),
            (ENCOUNTER_TABLE, EncModel,        EncView),
        ]
        # Sturcutres that will hold models and views
        ibswin.models       = {}
        ibswin.views        = {}
        ibswin.tblname_list = [IMAGE_TABLE, ROI_TABLE, NAME_TABLE]
        ibswin.super_tblname_list = ibswin.tblname_list + [ENCOUNTER_TABLE]
        # Menus
        ibswin.resize(900, 600)
        ibswin.setUnifiedTitleAndToolBarOnMac(False)
        ibswin.centralwidget = QtGui.QWidget(ibswin)
        ibswin.setCentralWidget(ibswin.centralwidget)
        parent = ibswin
        root = ibswin.centralwidget
        guimenus.setup_menus(ibswin)
        # Layout
        ibswin.vlayout = QtGui.QVBoxLayout(root)
        ibswin.hsplitter = guitool.newHorizontalSplitter(parent)
        # Tables Tab
        ibswin._tab_table_wgt = QtGui.QTabWidget(parent)
        # Create models and views
        for tblname, ModelClass, ViewClass in ibswin.modelview_defs:
            ibswin.models[tblname] = ModelClass(parent=parent)
            ibswin.views[tblname]  = ViewClass(parent=parent)
            ibswin.views[tblname].setModel(ibswin.models[tblname])
        # Add Image, ROI, and Names as tabs
        for tblname in ibswin.tblname_list:
            ibswin._tab_table_wgt.addTab(ibswin.views[tblname], tblname)
        # Encs Tabs
        ibswin.enc_tabwgt = EncoutnerTabWidget(parent=ibswin)
        # Add Other elements to the view
        ibswin.vlayout.addWidget(ibswin.enc_tabwgt)
        ibswin.vlayout.addWidget(ibswin.hsplitter)
        ibswin.hsplitter.addWidget(ibswin.views[ENCOUNTER_TABLE])
        ibswin.hsplitter.addWidget(ibswin._tab_table_wgt)

    #@checks_qt_error
    def _connect_signals_and_slots(ibswin):
        tblslots = {
            IMAGE_TABLE     : ibswin.on_doubleclick_image,
            ROI_TABLE       : ibswin.on_doubleclick_roi,
            NAME_TABLE      : ibswin.on_doubleclick_name,
            ENCOUNTER_TABLE : ibswin.on_doubleclick_encounter,
        }
        for tblname, slot in tblslots.iteritems():
            view = ibswin.views[tblname]
            view.doubleClicked.connect(slot)
            model = ibswin.models[tblname]
            model._rows_updated.connect(ibswin.on_rows_updated)

    def change_model_context_gen(ibswin, tblnames=None):
        """
        Loops over tablenames emitting layoutChanged at the end for each
        """
        if tblnames is None:
            tblnames = ibswin.super_tblname_list
        model_list = [ibswin.models[tblname] for tblname in tblnames]
        with ChangingModelLayout(model_list):
            for tblname in tblnames:
                yield tblname

    def update_tables(ibswin, tblnames=None):
        """ forces changing models """
        for tblname in ibswin.change_model_context_gen(tblnames=tblnames):
            model = ibswin.models[tblname]
            model._update()

    def connect_ibeis_control(ibswin, ibs):
        """ Connects a new ibscontroler to the models """
        print('[newgui] connecting ibs control')
        if ibs is None:
            print('[newgui] invalid ibs')
            title = 'No Database Opened'
        else:
            print('[newgui] Connecting valid ibs=%r' % ibs.get_dbname())
            # Give the frontend the new control
            ibswin.ibs = ibs
            # Update the api models to use the new control
            header_dict = gh.make_ibeis_headers_dict(ibswin.ibs)
            print('[newgui] Calling model _update_headers')
            for tblname in ibswin.change_model_context_gen(ibswin.super_tblname_list):
                model = ibswin.models[tblname]
                header = header_dict[tblname]
                model._update_headers(**header)
            #ibs.delete_invalid_eids()
            title = ibsfuncs.get_title(ibswin.ibs)
        ibswin.setWindowTitle(title)

    def _change_enc(ibswin, eid):
        for tblname in ibswin.change_model_context_gen(tblnames=ibswin.tblname_list):
            ibswin.views[tblname]._change_enc(eid)

    def _update_enc_tab_name(ibswin, eid, enctext):
        ibswin.enc_tabwgt._update_enc_tab_name(eid, enctext)

    #------------
    # SLOTS
    #------------

    @slot_(str, int)
    def on_rows_updated(ibswin, tblname, nRows):
        print('Rows updated in tblname=%r, nRows=%r' % (str(tblname), nRows))
        if tblname == ENCOUNTER_TABLE:
            return
        tblname = str(tblname)
        index = ibswin._tab_table_wgt.indexOf(ibswin.views[tblname])
        ibswin._tab_table_wgt.setTabText(index, tblname + ' ' + str(nRows))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_image(ibswin, qtindex):
        row   = qtindex.row()
        model = qtindex.model()
        gid = model._get_row_id(row)
        ibswin.back.select_gid(gid, model.eid)
        print("Image Selected, %r (ENC %r)" % (gid, model.eid))
        print('img')

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_roi(ibswin, qtindex):
        print('roi')
        row   = qtindex.row()
        model = qtindex.model()
        rid = model._get_row_id(row)
        ibswin.back.select_rid(rid, model.eid)
        print("ROI Selected, %r (ENC %r)" % (rid, model.eid))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_name(ibswin, qtindex):
        print('name')
        model = qtindex.model()
        row   = qtindex.row()
        nid = model._get_row_id(row)
        ibswin.back.select_nid(nid, model.eid)
        print("Name Selected, %r (ENC %r)" % (nid, model.eid))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_encounter(ibswin, qtindex):
        print('encounter')
        row   = qtindex.row()
        model = qtindex.model()
        eid = model._get_row_id(row)
        enctext = ibswin.ibs.get_encounter_enctext(eid)
        ibswin.enc_tabwgt._add_enc_tab(eid, enctext)
        print("Encounter Selected, %r" % (eid))


if __name__ == '__main__':
    import ibeis
    import sys
    ibeis._preload(mpl=False, par=False)
    guitool.ensure_qtapp()
    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')
    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    ibswin = IBEISGuiWidget(ibs=ibs)

    if '--cmd' in sys.argv:
        guitool.qtapp_loop(qwin=ibswin, ipy=True)
        exec(utool.ipython_execstr())
    else:
        guitool.qtapp_loop(qwin=ibswin)
