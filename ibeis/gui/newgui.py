#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import guitool
from ibeis.gui import newgui_views
from ibeis.gui import newgui_models
from PyQt4 import QtGui
from ibeis.gui import guiheaders as gh
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


#############################
###### Window Widgets #######
#############################


VIEWCLASS_DICT = {
    gh.IMAGE_TABLE     : newgui_views.ImageView,
    gh.ROI_TABLE       : newgui_views.ROIView,
    gh.NAME_TABLE      : newgui_views.NameView,
    gh.ENCOUNTER_TABLE : newgui_views.EncView,
}


def make_modelview(ibswin, tblname):
    ViewClass = VIEWCLASS_DICT[tblname]
    header = ibswin.header_dict[tblname]
    # TODO Unify these models:
    if tblname == gh.ENCOUNTER_TABLE:
        model = newgui_models.EncTableModel(header, parent=ibswin)
    else:
        model = newgui_models.IBEISTableModel(header, parent=ibswin)
    view = ViewClass(parent=ibswin)
    view.setModel(model)
    return model, view


class IBEISGuiWidget(QtGui.QWidget):
    def __init__(ibswin, ibs, parent=None):
        QtGui.QWidget.__init__(ibswin, parent)
        ibswin.connect_ibeis_controller(ibs)
        ibswin._init_layout()

    def connect_ibeis_controller(ibswin, ibs):
        ibswin.ibs = ibs
        ibswin.header_dict = gh.get_ibeis_headers_dict(ibswin.ibs)
        ibswin._refresh()

    def _refresh(ibswin):
        ibswin.setWindowTitle(ibsfuncs.get_title(ibswin.ibs))

    def _init_layout(ibswin):
        ibswin.vlayout = QtGui.QVBoxLayout(ibswin)
        ibswin.hsplitter = guitool.newHorizontalSplitter(ibswin)
        # Tabes Tab
        ibswin._tab_table_wgt = QtGui.QTabWidget(ibswin)
        # Images/ROI/Name Table
        ibswin._image_model, ibswin._image_view = make_modelview(ibswin, gh.IMAGE_TABLE)
        ibswin._roi_model,   ibswin._roi_view   = make_modelview(ibswin, gh.ROI_TABLE)
        ibswin._name_model,  ibswin._name_view  = make_modelview(ibswin, gh.NAME_TABLE)
        # Add Tabes to Tables Tab
        ibswin._tab_table_wgt.addTab(ibswin._image_view, gh.IMAGE_TABLE)
        ibswin._tab_table_wgt.addTab(ibswin._roi_view,   gh.ROI_TABLE)
        ibswin._tab_table_wgt.addTab(ibswin._name_view,  gh.NAME_TABLE)
        # Enc List
        ibswin._enc_model,  ibswin._enc_view  = make_modelview(ibswin, gh.ENCOUNTER_TABLE)
        # Encs Tabs
        ibswin._tab_enc_wgt = newgui_views.EncoutnerTabWidget(parent=ibswin)
        # Add Other elements to the view
        ibswin.vlayout.addWidget(ibswin._tab_enc_wgt)
        ibswin.vlayout.addWidget(ibswin.hsplitter)
        ibswin.hsplitter.addWidget(ibswin._enc_view)
        ibswin.hsplitter.addWidget(ibswin._tab_table_wgt)

    def _update_data(ibswin):
        ibswin._image_view._update_data()
        ibswin._roi_view._update_data()
        ibswin._name_view._update_data()

    def _change_enc(ibswin, eid):
        ibswin._image_view._change_enc(eid)
        ibswin._roi_view._change_enc(eid)
        ibswin._name_view._change_enc(eid)

    def _add_enc_tab(ibswin, eid, enctext):
        ibswin._tab_enc_wgt._add_enc_tab(eid, enctext)

    def _update_enc_tab_name(ibswin, eid, enctext):
        ibswin._tab_enc_wgt._update_enc_tab_name(eid, enctext)


if __name__ == '__main__':
    from ibeis.gui import newgui
    import ibeis
    import guitool  # NOQA
    ibeis._preload(mpl=False, par=False)
    print('app')

    guitool.ensure_qtapp()

    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')

    dbdir2 = ibeis.sysres.db_to_dbdir('GZ_ALL')

    ibs = IBEISControl.IBEISController(dbdir=dbdir)

    ibs2 = IBEISControl.IBEISController(dbdir=dbdir2)

    ibswin = newgui.IBEISGuiWidget(ibs)
    ibswin2 = newgui.IBEISGuiWidget(ibs2)

    guitool.qtapp_loop(qwin=ibswin)
