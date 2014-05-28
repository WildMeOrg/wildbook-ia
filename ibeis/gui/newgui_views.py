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


class IBEISTableView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.ibswin = parent
        default_view_layout(view)

    def _change_enc(view, eid):
        view.model()._change_enc(eid)


class EncView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.ibswin = parent
        default_view_layout(view)
        view.setMaximumSize(500, 9999)


#############################
###### Tab Widgets #######
#############################


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(enc_tabwgt, parent=None):
        QtGui.QTabWidget.__init__(enc_tabwgt, parent)
        enc_tabwgt.ibswin = parent
        enc_tabwgt.setTabsClosable(True)
        if sys.platform.startswith('darwin'):
            tab_height = 21
        else:
            tab_height = 30
        enc_tabwgt.setMaximumSize(9999, tab_height)
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
        enc_tabwgt.ibswin.refresh_state()
        #enc_tabwgt.setTabText(index,  '?')

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
