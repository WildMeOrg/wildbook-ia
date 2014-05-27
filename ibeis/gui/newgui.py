#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
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


def make_model_and_view(wgt, header, ViewClass):
    model = newgui_models.IBEISTableModel(wgt.headers[header], parent=wgt)
    view = ViewClass(parent=wgt)
    view.setModel(model)
    return model, view


class IBEISGuiWidget(QtGui.QWidget):
    def __init__(wgt, ibs, parent=None):
        QtGui.QWidget.__init__(wgt, parent)
        wgt.connect_ibeis_controller(ibs)
        wgt._init_layout()

    def connect_ibeis_controller(wgt, ibs):
        wgt.ibs = ibs
        wgt.headers = gh.ibeis_gui_headers(wgt.ibs)

    def _refresh(wgt):
        wgt.setWindowTitle(ibsfuncs.get_title(wgt.ibs))

    def _init_layout(wgt):
        wgt.vlayout = QtGui.QVBoxLayout(wgt)
        wgt.hsplitter = guitool.newHorizontalSplitter(wgt)
        # Tabes Tab
        wgt._tab_table_wgt = QtGui.QTabWidget(wgt)
        # Images/ROI/Name Table
        wgt._image_model, wgt._image_view = make_model_and_view(wgt, 'images', newgui_views.ImageView)
        wgt._roi_model, wgt._roi_view = make_model_and_view(wgt, 'rois', newgui_views.ROIView)
        wgt._name_model, wgt._name_view = make_model_and_view(wgt, 'names', newgui_views.NameView)
        # Add Tabes to Tables Tab
        wgt._tab_table_wgt.addTab(wgt._image_view, 'Images')
        wgt._tab_table_wgt.addTab(wgt._roi_view, 'ROIs')
        wgt._tab_table_wgt.addTab(wgt._name_view, 'Names')
        # Enc List
        wgt._enc_model = newgui_models.EncModel(wgt.headers['encounters'], parent=wgt)
        wgt._enc_view = newgui_views.EncView(parent=wgt)
        wgt._enc_view.setModel(wgt._enc_model)
        # Encs Tabs
        wgt._tab_enc_wgt = newgui_views.EncoutnerTabWidget(parent=wgt)
        # Add Other elements to the view
        wgt.vlayout.addWidget(wgt._tab_enc_wgt)
        wgt.vlayout.addWidget(wgt.hsplitter)
        wgt.hsplitter.addWidget(wgt._enc_view)
        wgt.hsplitter.addWidget(wgt._tab_table_wgt)

    def _update_data(wgt):
        wgt._image_view._update_data()
        wgt._roi_view._update_data()
        wgt._name_view._update_data()

    def _change_enc(wgt, encounter_id):
        wgt._image_view._change_enc(encounter_id)
        wgt._roi_view._change_enc(encounter_id)
        wgt._name_view._change_enc(encounter_id)

    def _add_enc_tab(wgt, encounter_id, encounter_name):
        wgt._tab_enc_wgt._add_enc_tab(encounter_id, encounter_name)

    def _update_enc_tab_name(wgt, encounter_id, encounter_name):
        wgt._tab_enc_wgt._update_enc_tab_name(encounter_id, encounter_name)


if __name__ == '__main__':
    import sys
    import guitool
    import ibeis
    ibeis._preload(mpl=False, par=False)
    print('app')
    app = QtGui.QApplication(sys.argv)
    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')
    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    wgt = IBEISGuiWidget(ibs)
    wgt.show()
    wgt.timer = guitool.ping_python_interpreter()
    wgt.raise_()
    sys.exit(app.exec_())
