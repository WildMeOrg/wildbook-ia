#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import guitool
from itertools import izip  # noqa
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
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
    def __init__(enc_tabwgt, parent=None, horizontalStretch=1):
        QtGui.QTabWidget.__init__(enc_tabwgt, parent)
        enc_tabwgt.ibswgt = parent
        enc_tabwgt.setTabsClosable(True)
        enc_tabwgt.setMaximumSize(9999, guitool.get_cplat_tab_height())
        enc_tabwgt.tabbar = enc_tabwgt.tabBar()
        enc_tabwgt.tabbar.setMovable(True)
        enc_tabwgt.setStyleSheet('border: none;')
        enc_tabwgt.tabbar.setStyleSheet('border: none;')
        sizePolicy = guitool.newSizePolicy(enc_tabwgt, horizontalStretch=horizontalStretch)
        enc_tabwgt.setSizePolicy(sizePolicy)

        enc_tabwgt.tabCloseRequested.connect(enc_tabwgt._close_tab)
        enc_tabwgt.currentChanged.connect(enc_tabwgt._on_change)

        enc_tabwgt.eid_list = []
        enc_tabwgt._add_enc_tab(None, 'Recognition Database')

    def _on_change(enc_tabwgt, index):
        """ Switch to the current encounter tab """
        if 0 <= index and index < len(enc_tabwgt.eid_list):
            eid = enc_tabwgt.eid_list[index]
            enc_tabwgt.ibswgt._change_enc(eid)

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

class IBEISMainWindow(QtGui.QMainWindow):
    def __init__(mainwin, back=None, ibs=None, parent=None):
        QtGui.QMainWindow.__init__(mainwin, parent)
        # Menus
        mainwin.setUnifiedTitleAndToolBarOnMac(False)
        guimenus.setup_menus(mainwin, back)
        # Central Widget
        mainwin.ibswgt = IBEISGuiWidget(back=back, ibs=ibs, parent=mainwin)
        mainwin.setCentralWidget(mainwin.ibswgt)
        #
        mainwin.resize(900, 600)


CLASS_IBEISGUIWidget = QtGui.QWidget


class IBEISGuiWidget(CLASS_IBEISGUIWidget):
    #@checks_qt_error
    def __init__(ibswgt, back=None, ibs=None, parent=None):
        CLASS_IBEISGUIWidget.__init__(ibswgt, parent)
        ibswgt.ibs = ibs
        ibswgt.back = back
        # Define the abstract item models and views for the tables
        ibswgt.modelview_defs = [
            (IMAGE_TABLE,     IBEISTableModel, IBEISTableView),
            (ROI_TABLE,       IBEISTableModel, IBEISTableView),
            (NAME_TABLE,      IBEISTableModel, IBEISTableView),
            (ENCOUNTER_TABLE, EncModel,        EncView),
        ]
        # Sturcutres that will hold models and views
        ibswgt.models       = {}
        ibswgt.views        = {}
        ibswgt.tblname_list = [IMAGE_TABLE, ROI_TABLE, NAME_TABLE]
        ibswgt.super_tblname_list = ibswgt.tblname_list + [ENCOUNTER_TABLE]
        # Create and layout components
        ibswgt._init_components()
        ibswgt._init_layout()
        # Connect signals and slots
        ibswgt._connect_signals_and_slots()
        # Connect the IBEIS control
        ibswgt.connect_ibeis_control(ibswgt.ibs)

    #@checks_qt_error
    def _init_components(ibswgt):
        """ Defines gui components """
        # Layout
        ibswgt.vlayout = QtGui.QVBoxLayout(ibswgt)
        ibswgt.hsplitter = guitool.newSplitter(ibswgt, Qt.Horizontal, verticalStretch=18)
        ibswgt.vsplitter = guitool.newSplitter(ibswgt, Qt.Vertical)
        # Tables Tab
        ibswgt._tab_table_wgt = guitool.newTabWidget(ibswgt, horizontalStretch=81)
        # Create models and views
        for tblname, ModelClass, ViewClass in ibswgt.modelview_defs:
            ibswgt.models[tblname] = ModelClass(parent=ibswgt)
            ibswgt.views[tblname]  = ViewClass(parent=ibswgt)
            ibswgt.views[tblname].setModel(ibswgt.models[tblname])
        # Add Image, ROI, and Names as tabs
        for tblname in ibswgt.tblname_list:
            ibswgt._tab_table_wgt.addTab(ibswgt.views[tblname], tblname)
        # Custom Encounter Tab Wiget
        ibswgt.enc_tabwgt = EncoutnerTabWidget(parent=ibswgt, horizontalStretch=19)
        # Other components
        ibswgt.outputLog   = guitool.newOutputLog(ibswgt, pointSize=8, visible=True, verticalStretch=6)
        ibswgt.progressBar = guitool.newProgressBar(ibswgt, visible=False, verticalStretch=1)
        ibswgt.status_wgt, ibswgt.status_vlayout = guitool.newWidget(ibswgt, verticalStretch=6)

    def _init_layout(ibswgt):
        """ Lays out the defined components """
        # Add elements to the layout
        ibswgt.vlayout.addWidget(ibswgt.enc_tabwgt)
        ibswgt.vlayout.addWidget(ibswgt.vsplitter)
        # Vertical
        ibswgt.vsplitter.addWidget(ibswgt.hsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.status_wgt)
        # Horizontal Upper
        ibswgt.hsplitter.addWidget(ibswgt.views[ENCOUNTER_TABLE])
        ibswgt.hsplitter.addWidget(ibswgt._tab_table_wgt)
        # Horizontal Lower
        ibswgt.status_vlayout.addWidget(ibswgt.outputLog)
        ibswgt.status_vlayout.addWidget(ibswgt.progressBar)

    #@checks_qt_error
    def _connect_signals_and_slots(ibswgt):
        tblslots = {
            IMAGE_TABLE     : ibswgt.on_doubleclick_image,
            ROI_TABLE       : ibswgt.on_doubleclick_roi,
            NAME_TABLE      : ibswgt.on_doubleclick_name,
            ENCOUNTER_TABLE : ibswgt.on_doubleclick_encounter,
        }
        for tblname, slot in tblslots.iteritems():
            view = ibswgt.views[tblname]
            view.doubleClicked.connect(slot)
            model = ibswgt.models[tblname]
            model._rows_updated.connect(ibswgt.on_rows_updated)

    def change_model_context_gen(ibswgt, tblnames=None):
        """
        Loops over tablenames emitting layoutChanged at the end for each
        """
        if tblnames is None:
            tblnames = ibswgt.super_tblname_list
        model_list = [ibswgt.models[tblname] for tblname in tblnames]
        with ChangingModelLayout(model_list):
            for tblname in tblnames:
                yield tblname

    def update_tables(ibswgt, tblnames=None):
        """ forces changing models """
        for tblname in ibswgt.change_model_context_gen(tblnames=tblnames):
            model = ibswgt.models[tblname]
            model._update()

    def connect_ibeis_control(ibswgt, ibs):
        """ Connects a new ibscontroler to the models """
        print('[newgui] connecting ibs control')
        if ibs is None:
            print('[newgui] invalid ibs')
            title = 'No Database Opened'
        else:
            print('[newgui] Connecting valid ibs=%r' % ibs.get_dbname())
            # Give the frontend the new control
            ibswgt.ibs = ibs
            # Update the api models to use the new control
            header_dict = gh.make_ibeis_headers_dict(ibswgt.ibs)
            print('[newgui] Calling model _update_headers')
            for tblname in ibswgt.change_model_context_gen(ibswgt.super_tblname_list):
                model = ibswgt.models[tblname]
                header = header_dict[tblname]
                model._update_headers(**header)
            #ibs.delete_invalid_eids()
            title = ibsfuncs.get_title(ibswgt.ibs)
        ibswgt.setWindowTitle(title)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            CLASS_IBEISGUIWidget.setWindowTitle(ibswgt, title)

    def _change_enc(ibswgt, eid):
        for tblname in ibswgt.change_model_context_gen(tblnames=ibswgt.tblname_list):
            ibswgt.views[tblname]._change_enc(eid)

    def _update_enc_tab_name(ibswgt, eid, enctext):
        ibswgt.enc_tabwgt._update_enc_tab_name(eid, enctext)

    #------------
    # SLOTS
    #------------

    @slot_(str, int)
    def on_rows_updated(ibswgt, tblname, nRows):
        printDBG('Rows updated in tblname=%r, nRows=%r' % (str(tblname), nRows))
        if tblname == ENCOUNTER_TABLE:
            return
        tblname = str(tblname)
        index = ibswgt._tab_table_wgt.indexOf(ibswgt.views[tblname])
        ibswgt._tab_table_wgt.setTabText(index, tblname + ' ' + str(nRows))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_image(ibswgt, qtindex):
        row   = qtindex.row()
        model = qtindex.model()
        gid = model._get_row_id(row)
        ibswgt.back.select_gid(gid, model.eid)
        print("Image Selected, %r (ENC %r)" % (gid, model.eid))
        print('img')

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_roi(ibswgt, qtindex):
        print('roi')
        row   = qtindex.row()
        model = qtindex.model()
        rid = model._get_row_id(row)
        ibswgt.back.select_rid(rid, model.eid)
        print("ROI Selected, %r (ENC %r)" % (rid, model.eid))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_name(ibswgt, qtindex):
        print('name')
        model = qtindex.model()
        row   = qtindex.row()
        nid = model._get_row_id(row)
        ibswgt.back.select_nid(nid, model.eid)
        print("Name Selected, %r (ENC %r)" % (nid, model.eid))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_encounter(ibswgt, qtindex):
        print('encounter')
        row   = qtindex.row()
        model = qtindex.model()
        eid = model._get_row_id(row)
        enctext = ibswgt.ibs.get_encounter_enctext(eid)
        ibswgt.enc_tabwgt._add_enc_tab(eid, enctext)
        print("Encounter Selected, %r" % (eid))


if __name__ == '__main__':
    import ibeis
    import sys
    ibeis._preload(mpl=False, par=False)
    guitool.ensure_qtapp()
    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')
    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    ibswgt = IBEISGuiWidget(ibs=ibs)

    if '--cmd' in sys.argv:
        guitool.qtapp_loop(qwin=ibswgt, ipy=True)
        exec(utool.ipython_execstr())
    else:
        guitool.qtapp_loop(qwin=ibswgt)
