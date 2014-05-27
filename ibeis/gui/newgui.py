#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import guitool
from itertools import izip
from guitool import slot_
from ibeis.gui import newgui_views
from ibeis.gui.newgui_views import IBEISTableView, EncView
from ibeis.gui.newgui_models import IBEISTableModel, EncModel
from guitool.APITableModel import ChangingModelLayout
from PyQt4 import QtGui, QtCore
from ibeis.gui import guiheaders as gh
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


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


class IBEISGuiWidget(QtGui.QWidget):
    def __init__(ibswin, ibs=None, parent=None):
        QtGui.QWidget.__init__(ibswin, parent)
        ibswin.ibs = None
        ibswin._init_layout()
        ibswin._connect_signals_and_slots()
        ibswin.connect_ibeis_control(ibs)

    def _init_layout(ibswin):
        ibswin.vlayout = QtGui.QVBoxLayout(ibswin)
        ibswin.hsplitter = guitool.newHorizontalSplitter(ibswin)
        # Tabes Tab
        ibswin._tab_table_wgt = QtGui.QTabWidget(ibswin)
        # Models
        ibswin._image_model = IBEISTableModel(parent=ibswin)
        ibswin._roi_model   = IBEISTableModel(parent=ibswin)
        ibswin._name_model  = IBEISTableModel(parent=ibswin)
        ibswin._enc_model   = EncModel(parent=ibswin)
        # Views
        ibswin._image_view = IBEISTableView(parent=ibswin)
        ibswin._roi_view   = IBEISTableView(parent=ibswin)
        ibswin._name_view  = IBEISTableView(parent=ibswin)
        ibswin._enc_view   = EncView(parent=ibswin)
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
        ibswin._tab_enc_wgt = newgui_views.EncoutnerTabWidget(parent=ibswin)
        # Add Other elements to the view
        ibswin.vlayout.addWidget(ibswin._tab_enc_wgt)
        ibswin.vlayout.addWidget(ibswin.hsplitter)
        ibswin.hsplitter.addWidget(ibswin._enc_view)
        ibswin.hsplitter.addWidget(ibswin._tab_table_wgt)

    def connect_ibeis_control(ibswin, ibs):
        print('connect control')
        if ibs is not None:
            print('Connecting valid ibs=%r'  % ibs.get_dbname())
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
        ibswin.refresh_state()

    def refresh_state(ibswin):
        print('Refresh State')
        title = 'No Database Opened'
        if ibswin.ibs is not None:
            title = ibsfuncs.get_title(ibswin.ibs)
        ibswin.setWindowTitle(title)

    def _change_enc(ibswin, eid):
        ibswin._image_view._change_enc(eid)
        ibswin._roi_view._change_enc(eid)
        ibswin._name_view._change_enc(eid)

    def _add_enc_tab(ibswin, eid, enctext):
        ibswin._tab_enc_wgt._add_enc_tab(eid, enctext)

    def _update_enc_tab_name(ibswin, eid, enctext):
        ibswin._tab_enc_wgt._update_enc_tab_name(eid, enctext)

    def _connect_signals_and_slots(ibswin):
        ibswin._image_view.doubleClicked.connect(ibswin.on_doubleclick_image)
        ibswin._roi_view.doubleClicked.connect(ibswin.on_doubleclick_roi)
        ibswin._name_view.doubleClicked.connect(ibswin.on_doubleclick_name)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_image(ibswin, qtindex):
        row = qtindex.row()
        model = qtindex.model()
        row_id = model._get_row_id(row)
        print("Image Selected, %r (ENC %r)" % (row_id, model.eid))
        print('img')

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_roi(ibswin, qtindex):
        print('roi')
        row = qtindex.row()
        model = qtindex.model()
        row_id = model._get_row_id(row)
        print("ROI Selected, %r (ENC %r)" % (row_id, model.eid))

    @slot_(QtCore.QModelIndex)
    def on_doubleclick_name(ibswin, qtindex):
        print('name')
        model = qtindex.model()
        row = qtindex.row()
        row_id = model._get_row_id(row)
        print("Name Selected, %r (ENC %r)" % (row_id, model.eid))


if __name__ == '__main__':
    from ibeis.gui import newgui
    import ibeis
    import guitool  # NOQA
    import sys
    ibeis._preload(mpl=False, par=False)
    print('app')

    guitool.ensure_qtapp()

    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')

    dbdir2 = ibeis.sysres.db_to_dbdir('GZ_ALL')

    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    ibs2 = IBEISControl.IBEISController(dbdir=dbdir2)

    ibswin = newgui.IBEISGuiWidget(ibs)

    if '--cmd' in sys.argv:
        guitool.qtapp_loop(qwin=ibswin, ipy=True)
        exec(utool.ipython_execstr())
    else:
        guitool.qtapp_loop(qwin=ibswin)
