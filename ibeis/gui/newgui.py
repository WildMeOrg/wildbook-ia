#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from itertools import izip  # noqa
import functools  # NOQA
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QSizePolicy
from guitool import slot_, checks_qt_error, ChangeLayoutContext  # NOQA
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui import guimenus
from ibeis.gui.guiheaders import (
    IMAGE_TABLE, ROI_TABLE, NAME_TABLE, ENCOUNTER_TABLE)
from ibeis.gui.models_and_views import (
    IBEISTableModel, IBEISTableView, EncTableModel, EncTableView)
import guitool
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


#############################
###### Tab Widgets #######
#############################


class APITabWidget(QtGui.QTabWidget):
    def __init__(tabwgt, parent=None, horizontalStretch=1):
        QtGui.QTabWidget.__init__(tabwgt, parent)
        tabwgt.ibswgt = parent
        sizePolicy = guitool.newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
        tabwgt.setSizePolicy(sizePolicy)
        tabwgt.currentChanged.connect(tabwgt.setCurrentIndex)

    def setCurrentIndex(tabwgt, index):
        print('Set current Index: %r ' % index)
        tblname = tabwgt.ibswgt.tblname_list[index]
        print(tblname)
        #model = tabwgt.ibswgt.models[tblname]
        #with ChangeLayoutContext([model]):
        #    QtGui.QTabWidget.setCurrentIndex(tabwgt, index)


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(enc_tabwgt, parent=None, horizontalStretch=1):
        QtGui.QTabWidget.__init__(enc_tabwgt, parent)
        enc_tabwgt.ibswgt = parent
        enc_tabwgt.setTabsClosable(True)
        enc_tabwgt.setMaximumSize(9999, guitool.get_cplat_tab_height())
        enc_tabwgt.tabbar = enc_tabwgt.tabBar()
        enc_tabwgt.tabbar.setMovable(False)
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
            (ENCOUNTER_TABLE, EncTableModel,     EncTableView),
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
        #ibswgt.hsplitter = guitool.newWidget(ibswgt, Qt.Horizontal, verticalStretch=18)
        #ibswgt.vsplitter = guitool.newWidget(ibswgt)
        #
        # Tables Tab
        ibswgt._tab_table_wgt = APITabWidget(ibswgt, horizontalStretch=81)
        #guitool.newTabWidget(ibswgt, horizontalStretch=81)
        # Create models and views
        for tblname, ModelClass, ViewClass in ibswgt.modelview_defs:
            ibswgt.models[tblname] = ModelClass(parent=ibswgt)
            ibswgt.views[tblname]  = ViewClass(parent=ibswgt)
        # Connect models and views
        for tblname in ibswgt.super_tblname_list:
            ibswgt.views[tblname].setModel(ibswgt.models[tblname])
        # Add Image, ROI, and Names as tabs
        for tblname in ibswgt.tblname_list:
            ibswgt._tab_table_wgt.addTab(ibswgt.views[tblname], tblname)
        # Custom Encounter Tab Wiget
        ibswgt.enc_tabwgt = EncoutnerTabWidget(parent=ibswgt, horizontalStretch=19)
        # Other components
        ibswgt.outputLog   = guitool.newOutputLog(ibswgt, pointSize=8, visible=True, verticalStretch=6)
        ibswgt.progressBar = guitool.newProgressBar(ibswgt, visible=False, verticalStretch=1)
        # New widget has black magic (for implicit layouts) in it
        ibswgt.status_wgt  = guitool.newWidget(ibswgt, Qt.Vertical,
                                               verticalStretch=6,
                                               horizontalSizePolicy=QSizePolicy.Maximum)

        ibswgt.statusBar = QtGui.QHBoxLayout(ibswgt)
        _NEWLBL = functools.partial(guitool.newLabel, ibswgt)
        ibswgt.statusLabel_list = [
            _NEWLBL(''),
            _NEWLBL('Status Bar'),
            _NEWLBL(''),
            _NEWLBL(''),
        ]

        ibswgt.buttonBar = QtGui.QHBoxLayout(ibswgt)
        _NEWBUT = functools.partial(guitool.newButton, ibswgt)
        #_SEP = lambda: None
        ibswgt.button_list = [
            _NEWBUT('Import Images (from file)', lambda *args: None),
            _NEWBUT('Import Images (from dir)', lambda *args: None),
            _NEWBUT('Import Images (from dir with size filter)', lambda *args: None),

            #_SEP(),

            _NEWBUT('Filter Images (GIST)', lambda *args: None),

            #_SEP(),

            _NEWBUT('Compute {algid} Encounters', lambda *args: None),

            #_SEP(),

            _NEWBUT('Run {species}-{algid} Detector', ibswgt.back.detect_grevys_quick),
            _NEWBUT('Review {species}-Detections', lambda *args: None),

            #_SEP(),

            _NEWBUT('Individual Recognition (who are these?)', ibswgt.back.precompute_queries),
            _NEWBUT('Review Individual Matches', lambda *args: None),

            #_SEP(),

            _NEWBUT('DELETE ALL ENCOUNTERS', ibswgt.back.delete_all_encounters),
        ]

    def _init_layout(ibswgt):
        """ Lays out the defined components """
        # Add elements to the layout
        ibswgt.vlayout.addWidget(ibswgt.enc_tabwgt)
        ibswgt.vlayout.addWidget(ibswgt.vsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.hsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.status_wgt)
        # Horizontal Upper
        ibswgt.hsplitter.addWidget(ibswgt.views[ENCOUNTER_TABLE])
        ibswgt.hsplitter.addWidget(ibswgt._tab_table_wgt)
        # Horizontal Lower
        ibswgt.status_wgt.addWidget(ibswgt.outputLog)
        ibswgt.status_wgt.addWidget(ibswgt.progressBar)
        ibswgt.status_wgt.addLayout(ibswgt.buttonBar)
        ibswgt.status_wgt.addLayout(ibswgt.statusBar)
        # Add statusbar
        for widget in ibswgt.statusLabel_list:
            ibswgt.statusBar.addWidget(widget)
        # Add buttonbar
        for button in ibswgt.button_list:
            ibswgt.buttonBar.addWidget(button)

    def set_status_label(ibswgt, index, text):
        printDBG('set_status_label[%r] = %r' % (index, text))
        ibswgt.statusLabel_list[index].setText(text)

    def changing_models_gen(ibswgt, tblnames=None):
        """
        Loops over tablenames emitting layoutChanged at the end for each
        """
        if tblnames is None:
            tblnames = ibswgt.super_tblname_list
        model_list = [ibswgt.models[tblname] for tblname in tblnames]
        #model_list = [ibswgt.models[tblname] for tblname in tblnames if ibswgt.views[tblname].isVisible()]
        with ChangeLayoutContext(model_list):
            for tblname in tblnames:
                yield tblname

    def update_tables(ibswgt, tblnames=None):
        """ forces changing models """
        for tblname in ibswgt.changing_models_gen(tblnames=tblnames):
            model = ibswgt.models[tblname]
            model._update()

    def connect_ibeis_control(ibswgt, ibs):
        """ Connects a new ibscontroler to the models """
        print('[newgui] connecting ibs control')
        if ibs is None:
            print('[newgui] invalid ibs')
            title = 'No Database Opened'
            ibswgt.setWindowTitle(title)
        else:
            print('[newgui] Connecting valid ibs=%r' % ibs.get_dbname())
            # Give the frontend the new control
            ibswgt.ibs = ibs
            # Update the api models to use the new control
            header_dict = gh.make_ibeis_headers_dict(ibswgt.ibs)
            title = ibsfuncs.get_title(ibswgt.ibs)
            ibswgt.setWindowTitle(title)
            print('[newgui] Calling model _update_headers')
            _flag = ibswgt._tab_table_wgt.blockSignals(True)
            for tblname in ibswgt.changing_models_gen(ibswgt.super_tblname_list):
                model = ibswgt.models[tblname]
                header = header_dict[tblname]
                model._update_headers(**header)
            ibswgt._tab_table_wgt.blockSignals(_flag)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            CLASS_IBEISGUIWidget.setWindowTitle(ibswgt, title)

    def _change_enc(ibswgt, eid):
        for tblname in ibswgt.changing_models_gen(tblnames=ibswgt.tblname_list):
            ibswgt.views[tblname]._change_enc(eid)
        try:
            ibswgt.button_list[3].setText('QUERY(eid=%r)' % eid)
        except Exception:
            pass

    def _update_enc_tab_name(ibswgt, eid, enctext):
        ibswgt.enc_tabwgt._update_enc_tab_name(eid, enctext)

    #@checks_qt_error
    def _connect_signals_and_slots(ibswgt):
        for tblname in ibswgt.super_tblname_list:
            tblview = ibswgt.views[tblname]
            tblview.doubleClicked.connect(ibswgt.on_doubleclick)
            tblview.clicked.connect(ibswgt.on_click)
            tblview.contextMenuClicked.connect(ibswgt.on_contextMenuClicked)
            model = ibswgt.models[tblname]
            model._rows_updated.connect(ibswgt.on_rows_updated)

    #------------
    # SLOTS
    #------------

    @slot_(str, int)
    def on_rows_updated(ibswgt, tblname, nRows):
        """ When the rows are updated change the tab names """
        printDBG('Rows updated in tblname=%r, nRows=%r' % (str(tblname), nRows))
        if tblname == ENCOUNTER_TABLE:  # Hack
            return
        tblname = str(tblname)
        index = ibswgt._tab_table_wgt.indexOf(ibswgt.views[tblname])
        ibswgt._tab_table_wgt.setTabText(index, tblname + ' ' + str(nRows))

    @slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuClicked(ibswgt, qtindex, pos):
        print('[newgui] contextmenu')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex.row())
        tblview = ibswgt.views[model.name]
        if model.name == ENCOUNTER_TABLE:
            eid = id_
            guitool.popup_menu(tblview, pos, [
                ('delete encounter', lambda: ibswgt.back.delete_encounter(eid)),
            ])
        else:
            eid = model.eid
            if model.name == IMAGE_TABLE:
                gid = id_
                guitool.popup_menu(tblview, pos, [
                    ('view hough image', lambda: ibswgt.back.show_hough(gid)),
                    ('delete image', lambda: ibswgt.back.delete_image(gid)),
                ])
            elif model.name == ROI_TABLE:
                rid = id_
                guitool.popup_menu(tblview, pos, [
                    ('delete roi', lambda: ibswgt.back.delete_roi(rid)),


                ])
    @slot_(QtCore.QModelIndex)
    def on_click(ibswgt, qtindex):
        print('on_click')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex.row())
        if model.name == ENCOUNTER_TABLE:
            print('clicked encounter')
        else:
            eid = model.eid
            if model.name == IMAGE_TABLE:
                gid = id_
                ibswgt.back.select_gid(gid, eid, show=False)
            elif model.name == ROI_TABLE:
                rid = id_
                ibswgt.back.select_rid(rid, eid, show=False)
            elif model.name == NAME_TABLE:
                nid = id_
                ibswgt.back.select_nid(nid, eid, show=False)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick(ibswgt, qtindex):
        print('on_doubleclick')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex.row())
        if model.name == ENCOUNTER_TABLE:
            eid = id_
            enctext = ibswgt.ibs.get_encounter_enctext(eid)
            ibswgt.enc_tabwgt._add_enc_tab(eid, enctext)
            ibswgt.back.select_eid(eid)
        else:
            eid = model.eid
            if model.name == IMAGE_TABLE:
                gid = id_
                ibswgt.back.select_gid(gid, eid)
            elif model.name == ROI_TABLE:
                rid = id_
                ibswgt.back.select_rid(rid, eid)
            elif model.name == NAME_TABLE:
                nid = id_
                ibswgt.back.select_nid(nid, eid)

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
