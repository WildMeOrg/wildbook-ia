#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtGui, QtCore
from guitool import slot_, StripeProxyModel
from ibeis.control import IBEISControl
from ibeis.other import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui.guiheaders import THUMB_TABLE
from ibeis.gui.models_and_views import IBEISStripeModel, IBEISTableView
from ibeis.viz.interact import interact_annotations2
import guitool
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


IBEIS_WIDGET_BASE = QtWidgets.QWidget


#############################
###### Window Widgets #######
#############################


class IBEISGuiWidget(IBEIS_WIDGET_BASE):
    #@checks_qt_error
    def __init__(ibswgt, ibs=None, parent=None):
        IBEIS_WIDGET_BASE.__init__(ibswgt, parent)
        ibswgt.ibs = ibs
        ibswgt.tblname_list = [THUMB_TABLE]
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
        ibswgt.vlayout = QtWidgets.QVBoxLayout(ibswgt)
        # Create models and views
        ibswgt.view = IBEISTableView(parent=ibswgt)
        ibswgt.model = IBEISStripeModel(parent=ibswgt.view)
        ibswgt.proxy = StripeProxyModel(numduplicates=3)
        #ibswgt.proxy = QtGui.QIdentityProxyModel()
        ibswgt.proxy.setSourceModel(ibswgt.model)
        ibswgt.view.setModel(ibswgt.model)
        #ibswgt.view.setModel(ibswgt.model)

    def _init_layout(ibswgt):
        """ Lays out the defined components """
        # Add elements to the layout
        ibswgt.vlayout.addWidget(ibswgt.view)

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
            model = ibswgt.model
            view = ibswgt.view
            header = header_dict[THUMB_TABLE]
            model._update_headers(**header)
            view._update_headers(**header)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            IBEIS_WIDGET_BASE.setWindowTitle(ibswgt, title)

    #@checks_qt_error
    def _connect_signals_and_slots(ibswgt):
        tblview = ibswgt.view
        tblview.doubleClicked.connect(ibswgt.on_doubleclick)
        tblview.clicked.connect(ibswgt.on_click)
        tblview.contextMenuClicked.connect(ibswgt.on_contextMenuClicked)
        print('signals and slots connected')

    #------------
    # SLOTS
    #------------

    @slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuClicked(ibswgt, qtindex, pos):
        printDBG('[newgui] contextmenu')
        model = qtindex.model()
        #id_ = model._get_row_id(qtindex)
        if model.name == THUMB_TABLE:
            tblview = ibswgt.view
            # imgsetid = model.imgsetid
            # gid = id_
            guitool.popup_menu(tblview, pos, [
                ('right click action', lambda: None),
            ])

    @slot_(QtCore.QModelIndex)
    def on_click(ibswgt, qtindex):
        print("on single click")
        printDBG('on_click')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        if model.name == THUMB_TABLE:
            # imgsetid = model.imgsetid
            gid = id_
            if gid is None:
                return
            print("SINGLE CLICKED ID: %r" % gid)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick(ibswgt, qtindex):
        print("on double click")
        printDBG('on_doubleclick')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        print('dblclick (%r,%r)' % (qtindex.row(), qtindex.column()))
        if model.name == THUMB_TABLE:
            # imgsetid = model.imgsetid
            gid = id_
            if gid is None:
                return
            ibswgt.roi_interact = interact_annotations2.ANNOTATION_Interaction2(ibswgt.ibs, gid)
            print("DOUBLECLICKED ID: %r" % gid)

if __name__ == '__main__':
    import ibeis
    import sys
    ibeis._preload(mpl=False, par=False)
    guitool.ensure_qtapp()
    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')
    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    ibswgt = IBEISGuiWidget(ibs=ibs)
    ibswgt.resize(900, 600)

    if '--cmd' in sys.argv:
        guitool.qtapp_loop(qwin=ibswgt, ipy=True)
        exec(utool.ipython_execstr())
    else:
        guitool.qtapp_loop(qwin=ibswgt)
