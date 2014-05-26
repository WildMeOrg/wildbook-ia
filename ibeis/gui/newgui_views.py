from __future__ import absolute_import, division, print_function
import sys
from PyQt4 import QtGui
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui_views] ')


#############################
######### Data Views ########
#############################


def default_view_layout(view):
    view.setSortingEnabled(True)
    vh = view.verticalHeader()
    vh.setVisible(False)
    view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
    view.resizeColumnsToContents()


class ImageView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        default_view_layout(view)

    def _change_enc(view, encounter_id):
        view.model()._change_enc(encounter_id)

    def _update_data(view):
        view.model()._update_data()

    def mouseDoubleClickEvent(view, event):
        qtindex = view.selectedIndexes()[0]
        row, col = view.model()._row_col(qtindex)
        row_id = view.model()._get_row_id(row)
        print("Image Selected, %r (ENC %r)" % (row_id, view.model().encounter_id))


class ROIView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        default_view_layout(view)

    def _change_enc(view, encounter_id):
        view.model()._change_enc(encounter_id)

    def _update_data(view):
        view.model()._update_data()

    def mouseDoubleClickEvent(view, event):
        qtindex = view.selectedIndexes()[0]
        row, col = view.model()._row_col(qtindex)
        row_id = view.model()._get_row_id(row)
        print("ROI Selected, %r (ENC %r)" % (row_id, view.model().encounter_id))


class NameView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        default_view_layout(view)

    def _change_enc(view, encounter_id):
        view.model()._change_enc(encounter_id)

    def _update_data(view):
        view.model()._update_data()

    def mouseDoubleClickEvent(view, event):
        qtindex = view.selectedIndexes()[0]
        row, col = view.model()._row_col(qtindex)
        row_id = view.model()._get_row_id(row)
        print("Name Selected, %r (ENC %r)" % (row_id, view.model().encounter_id))


class EncView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        default_view_layout(view)
        view.setMaximumSize(600, 9999)
        #hh = view.horizontalHeader()
        #hh.setVisible(False)

    def mouseDoubleClickEvent(view, event):
        qtindex = view.selectedIndexes()[0]
        encounter_id, encounter_name = view.model()._get_enc_id_name(qtindex)
        view.window._add_enc_tab(encounter_id, encounter_name)


#############################
###### Tab Widgets #######
#############################


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(wgt, parent=None):
        QtGui.QTabWidget.__init__(wgt, parent)
        wgt.window = parent
        wgt.setTabsClosable(True)
        if sys.platform.startswith('darwin'):
            tab_height = 21
        else:
            tab_height = 30
        wgt.setMaximumSize(9999, tab_height)
        wgt._tb = wgt.tabBar()
        wgt._tb.setMovable(True)
        wgt.setStyleSheet('border: none;')
        wgt._tb.setStyleSheet('border: none;')

        wgt.tabCloseRequested.connect(wgt._close_tab)
        wgt.currentChanged.connect(wgt._on_change)

        wgt.encounter_id_list = []
        wgt._add_enc_tab(None, 'Recognition Database')

    def _on_change(wgt, index):
        if 0 <= index and index < len(wgt.encounter_id_list):
            wgt.window._change_enc(wgt.encounter_id_list[index])

    def _close_tab(wgt, index):
        if wgt.encounter_id_list[index] is not None:
            wgt.encounter_id_list.pop(index)
            wgt.removeTab(index)

    def _add_enc_tab(wgt, encounter_id, encounter_name):
        if encounter_id not in wgt.encounter_id_list:
            # tab_name = str(encounter_id) + ' - ' + str(encounter_name)
            tab_name = str(encounter_name)
            wgt.addTab(QtGui.QWidget(), tab_name)

            wgt.encounter_id_list.append(encounter_id)
            index = len(wgt.encounter_id_list) - 1
        else:
            index = wgt.encounter_id_list.index(encounter_id)

        wgt.setCurrentIndex(index)
        wgt._on_change(index)

    def _update_enc_tab_name(wgt, encounter_id, encounter_name):
        for index, _id in enumerate(wgt.encounter_id_list):
            if encounter_id == _id:
                wgt.setTabText(index, encounter_name)
