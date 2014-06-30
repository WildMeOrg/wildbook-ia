#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from itertools import izip  # noqa
import functools  # NOQA
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QSizePolicy
from guitool import signal_, slot_, checks_qt_error, ChangeLayoutContext  # NOQA
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui import guimenus
from ibeis.viz.interact import interact_annotations2
from ibeis import constants
from ibeis.gui.guiheaders import (
    IMAGE_TABLE, ANNOTATION_TABLE, NAME_TABLE, NAMES_TREE, ENCOUNTER_TABLE)
from ibeis.gui.models_and_views import (
    IBEISTableModel, IBEISTableView, IBEISTreeView, EncTableModel, EncTableView,
    IBEISTableWidget, IBEISTreeWidget, EncTableWidget)
import guitool
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


IBEIS_WIDGET_BASE = QtGui.QWidget


#############################
###### Tab Widgets #######
#############################


class APITabWidget(QtGui.QTabWidget):
    def __init__(tabwgt, parent=None, horizontalStretch=1):
        QtGui.QTabWidget.__init__(tabwgt, parent)
        tabwgt.ibswgt = parent
        tabwgt._sizePolicy = guitool.newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
        tabwgt.setSizePolicy(tabwgt._sizePolicy)
        #tabwgt.currentChanged.connect(tabwgt.setCurrentIndex)

    #def setCurrentIndex(tabwgt, index):
    #    tblname = tabwgt.ibswgt.tblname_list[index]
    #    print('Set %r current Index: %r ' % (tblname, index))
    #    #model = tabwgt.ibswgt.models[tblname]
    #    #with ChangeLayoutContext([model]):
    #    #    QtGui.QTabWidget.setCurrentIndex(tabwgt, index)


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
        enc_tabwgt._add_enc_tab(None, constants.ALLIMAGE_ENCTEXT)

    @slot_(int)
    def _on_change(enc_tabwgt, index):
        """ Switch to the current encounter tab """
        if 0 <= index and index < len(enc_tabwgt.eid_list):
            eid = enc_tabwgt.eid_list[index]
            if utool.VERBOSE:
                print('[ENCTAB.ONCHANGE] eid = %r' % (eid,))
            enc_tabwgt.ibswgt._change_enc(eid)

    @slot_(int)
    def _close_tab(enc_tabwgt, index):
        if enc_tabwgt.eid_list[index] is not None:
            enc_tabwgt.eid_list.pop(index)
            enc_tabwgt.removeTab(index)

    def _add_enc_tab(enc_tabwgt, eid, enctext):
        # <HACK>
        if enctext == constants.ALLIMAGE_ENCTEXT:
            eid = None
        # </HACK>
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
    quitSignal = signal_()
    def __init__(mainwin, back=None, ibs=None, parent=None):
        QtGui.QMainWindow.__init__(mainwin, parent)
        # Menus
        mainwin.setUnifiedTitleAndToolBarOnMac(False)
        guimenus.setup_menus(mainwin, back)
        # Central Widget
        mainwin.ibswgt = IBEISGuiWidget(back=back, ibs=ibs, parent=mainwin)
        mainwin.setCentralWidget(mainwin.ibswgt)
        if back is not None:
            mainwin.quitSignal.connect(back.quit)
        else:
            raise AssertionError('need backend')
        #
        mainwin.resize(900, 600)

    @slot_()
    def closeEvent(mainwin, event):
        event.accept()
        mainwin.quitSignal.emit()


class IBEISGuiWidget(IBEIS_WIDGET_BASE):
    #@checks_qt_error
    def __init__(ibswgt, back=None, ibs=None, parent=None):
        IBEIS_WIDGET_BASE.__init__(ibswgt, parent)
        ibswgt.ibs = ibs
        ibswgt.back = back
        # Sturcutres that will hold models and views
        ibswgt.models       = {}
        ibswgt.views        = {}
        #ibswgt.widgets      = {}
        ibswgt.tblname_list = [IMAGE_TABLE, ANNOTATION_TABLE, NAME_TABLE, NAMES_TREE]
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
        # Define the abstract item models and views for the tables
        ibswgt.modelview_defs = [
            (IMAGE_TABLE,     IBEISTableWidget, IBEISTableModel, IBEISTableView),
            (ANNOTATION_TABLE,       IBEISTableWidget, IBEISTableModel, IBEISTableView),
            (NAME_TABLE,      IBEISTableWidget, IBEISTableModel, IBEISTableView),
            (NAMES_TREE,      IBEISTreeWidget, IBEISTableModel, IBEISTreeView),
            (ENCOUNTER_TABLE, EncTableWidget,   EncTableModel,   EncTableView),
        ]
        for tblname, WidgetClass, ModelClass, ViewClass in ibswgt.modelview_defs:
            #widget = WidgetClass(parent=ibswgt)
            #ibswgt.widgets[tblname] = widget
            #ibswgt.models[tblname]  = widget.model
            #ibswgt.views[tblname]   = widget.view
            ibswgt.views[tblname]  = ViewClass(parent=ibswgt)  # Make view first to pass as parent
            # FIXME: It is very bad to give the model a view. Only the view should have a model
            ibswgt.models[tblname] = ModelClass(parent=ibswgt.views[tblname])
        # Connect models and views
        for tblname in ibswgt.super_tblname_list:
            ibswgt.views[tblname].setModel(ibswgt.models[tblname])
        # Add Image, ANNOTATION, and Names as tabs
        for tblname in ibswgt.tblname_list:
            #ibswgt._tab_table_wgt.addTab(ibswgt.widgets[tblname], tblname)
            ibswgt._tab_table_wgt.addTab(ibswgt.views[tblname], tblname)
        # Custom Encounter Tab Wiget
        ibswgt.enc_tabwgt = EncoutnerTabWidget(parent=ibswgt, horizontalStretch=19)
        # Other components
        ibswgt.outputLog   = guitool.newOutputLog(ibswgt, pointSize=8,
                                                  visible=True, verticalStretch=6)
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

        ibswgt.buttonBars = []
        _NEWBUT = functools.partial(guitool.newButton, ibswgt)
        back = ibswgt.back
        #_SEP = lambda: None
        ibswgt.button_list = [
            [
                _NEWBUT('Import Images\n(via files)',
                        back.import_images_from_file,
                        bgcolor=(235, 200, 200),),
                #_NEWBUT('Import Images\n(via dir)',
                #        back.import_images_from_dir,
                #        bgcolor=(235, 200, 200)),
                #_NEWBUT('Import Images\n(via dir + size filter)',
                #        bgcolor=(235, 200, 200)),

                #_SEP(),

                #_NEWBUT('Filter Images (GIST)'),

                #_SEP(),

                _NEWBUT('Group Images into Encounters', ibswgt.back.compute_encounters,
                        bgcolor=(255, 255, 150)),
                #_NEWBUT('Compute {algid} Encounters'),

                #_SEP(),

                _NEWBUT('Detect',
                        ibswgt.back.run_detection_coarse,
                        bgcolor=(150, 255, 150)),

                _NEWBUT('Detect',
                        ibswgt.back.run_detection_coarse,
                        bgcolor=(150, 255, 150)),
            ],
            [
                #_NEWBUT('Review Detections',
                #        ibswgt.back.review_detections,
                #        bgcolor=(170, 250, 170)),

                #_SEP(),

                _NEWBUT('Identify\n(intra-encounter)',
                        ibswgt.back.compute_queries,
                        bgcolor=(150, 150, 255),
                        fgcolor=(0, 0, 0)),

                _NEWBUT('Identify\n(vs exemplar database)',
                        ibswgt.back.compute_queries,
                        bgcolor=(150, 150, 255),
                        fgcolor=(0, 0, 0)),
                #_NEWBUT('Review Recognitions',
                #        ibswgt.back.review_queries,
                #        bgcolor=(170, 170, 250),
                #        fgcolor=(0, 0, 0)),

                # _SEP(),

                _NEWBUT('Delete Encounters', ibswgt.back.delete_all_encounters,
                        bgcolor=(255, 0, 0),
                        fgcolor=(0, 0, 0)),
            ]
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
        # Add buttonbar
        for row in ibswgt.button_list:
            ibswgt.buttonBars.append(QtGui.QHBoxLayout(ibswgt))
            ibswgt.status_wgt.addLayout(ibswgt.buttonBars[-1])
            for button in row:
                ibswgt.buttonBars[-1].addWidget(button)
        ibswgt.status_wgt.addLayout(ibswgt.statusBar)
        # Add statusbar
        for widget in ibswgt.statusLabel_list:
            ibswgt.statusBar.addWidget(widget)

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
            ibs.update_special_encounters()
            # Update the api models to use the new control
            header_dict = gh.make_ibeis_headers_dict(ibswgt.ibs)
            title = ibsfuncs.get_title(ibswgt.ibs)
            ibswgt.setWindowTitle(title)
            if utool.VERBOSE:
                print('[newgui] Calling model _update_headers')
            block_wgt_flag = ibswgt._tab_table_wgt.blockSignals(True)
            for tblname in ibswgt.changing_models_gen(ibswgt.super_tblname_list):
                model = ibswgt.models[tblname]
                view = ibswgt.views[tblname]
                header = header_dict[tblname]
                #widget = ibswgt.widgets[tblname]
                #widget.change_headers(header)
                block_model_flag = model.blockSignals(True)
                model._update_headers(**header)
                view._update_headers(**header)  # should use model headers
                model.blockSignals(block_model_flag)
                #view.infer_delegates_from_model()
            for tblname in ibswgt.super_tblname_list:
                view = ibswgt.views[tblname]
                view.hide_cols()
            ibswgt._tab_table_wgt.blockSignals(block_wgt_flag)
            ibswgt._change_enc(None)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            IBEIS_WIDGET_BASE.setWindowTitle(ibswgt, title)

    def _change_enc(ibswgt, eid):
        for tblname in ibswgt.changing_models_gen(tblnames=ibswgt.tblname_list):
            ibswgt.views[tblname]._change_enc(eid)
            ibswgt.models[tblname]._change_enc(eid)
        try:
            if eid is None:
                # HACK
                enctext = constants.ALLIMAGE_ENCTEXT
            else:
                enctext = ibswgt.ibs.get_encounter_enctext(eid)
            ibswgt.button_list[4].setText('Identify (intra-encounter)\nQUERY(%r vs. %r)' % (enctext, enctext))
            ibswgt.button_list[5].setText('Identify (vs exemplar database)\nQUERY(%r vs. %r)' % (enctext, constants.EXEMPLAR_ENCTEXT))
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
            tblview.rows_updated.connect(ibswgt.on_rows_updated)
            #front.printSignal.connect(back.backend_print)
            #front.raiseExceptionSignal.connect(back.backend_exception)

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
        view = ibswgt.views[tblname]
        index = ibswgt._tab_table_wgt.indexOf(view)
        text = tblname + ' ' + str(nRows)
        #printDBG('Rows updated in index=%r, text=%r' % (index, text))
        ibswgt._tab_table_wgt.setTabText(index, text)

    @slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuClicked(ibswgt, qtindex, pos):
        #printDBG('[newgui] contextmenu')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
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
            elif model.name == ANNOTATION_TABLE:
                aid = id_
                guitool.popup_menu(tblview, pos, [
                    ('delete annotation', lambda: ibswgt.back.delete_annotation(aid)),
                ])

    @slot_(QtCore.QModelIndex)
    def on_click(ibswgt, qtindex):
        #printDBG('on_click')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        if model.name == ENCOUNTER_TABLE:
            pass
            #printDBG('clicked encounter')
        else:
            eid = model.eid
            if model.name == IMAGE_TABLE:
                gid = id_
                ibswgt.back.select_gid(gid, eid, show=False)
            elif model.name == ANNOTATION_TABLE:
                aid = id_
                ibswgt.back.select_aid(aid, eid, show=False)
            elif model.name == NAME_TABLE:
                nid = id_
                ibswgt.back.select_nid(nid, eid, show=False)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick(ibswgt, qtindex):
        #printDBG('on_doubleclick')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        if model.name == ENCOUNTER_TABLE:
            eid = id_
            enctext = ibswgt.ibs.get_encounter_enctext(eid)
            ibswgt.enc_tabwgt._add_enc_tab(eid, enctext)
            ibswgt.back.select_eid(eid)
        else:
            eid = model.eid
            if model.name == IMAGE_TABLE:
                gid = id_
                ibswgt.annotation_interact = interact_annotations2.ANNOTATION_Interaction2(ibswgt.ibs, gid, ibswgt.update_tables)
                ibswgt.back.select_gid(gid, eid, show=False)
            elif model.name == ANNOTATION_TABLE:
                aid = id_
                ibswgt.back.select_aid(aid, eid)
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
