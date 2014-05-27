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

    def _change_enc(view, eid):
        view.model()._change_enc(eid)

    def mouseDoubleClickEvent(view, event):
        qtindex_list = view.selectedIndexes()[0]
        print('[imgview] selected %d ' % len(qtindex_list))
        print('[imgview] event= %r ' % event)
        if len(qtindex_list) > 0:
            qtindex = qtindex_list[0]
            row = qtindex.row()
            row_id = view.model()._get_row_id(row)
            print("Image Selected, %r (ENC %r)" % (row_id, view.model().eid))


class ROIView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        default_view_layout(view)

    def _change_enc(view, eid):
        view.model()._change_enc(eid)

    def mouseDoubleClickEvent(view, event):
        qtindex = view.selectedIndexes()[0]
        row = qtindex.row()
        row_id = view.model()._get_row_id(row)
        print("ROI Selected, %r (ENC %r)" % (row_id, view.model().eid))


class NameView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        default_view_layout(view)

    def _change_enc(view, eid):
        view.model()._change_enc(eid)

    def mouseDoubleClickEvent(view, event):
        qtindex = view.selectedIndexes()[0]
        row = qtindex.row()
        row_id = view.model()._get_row_id(row)
        print("Name Selected, %r (ENC %r)" % (row_id, view.model().eid))


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
        row = qtindex.row()
        model = view.model()
        eid = model._get_row_id(row)
        enctext = view.window.ibs.get_encounter_enctext(eid)
        view.window._add_enc_tab(eid, enctext)


#############################
###### Tab Widgets #######
#############################


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(tabwgt, parent=None):
        QtGui.QTabWidget.__init__(tabwgt, parent)
        tabwgt.window = parent
        tabwgt.setTabsClosable(True)
        if sys.platform.startswith('darwin'):
            tab_height = 21
        else:
            tab_height = 30
        tabwgt.setMaximumSize(9999, tab_height)
        tabwgt.tabbar = tabwgt.tabBar()
        tabwgt.tabbar.setMovable(True)
        tabwgt.setStyleSheet('border: none;')
        tabwgt.tabbar.setStyleSheet('border: none;')

        tabwgt.tabCloseRequested.connect(tabwgt._close_tab)
        tabwgt.currentChanged.connect(tabwgt._on_change)

        tabwgt.encounter_id_list = []
        tabwgt._add_enc_tab(None, 'Recognition Database')

    def _on_change(tabwgt, index):
        if 0 <= index and index < len(tabwgt.encounter_id_list):
            tabwgt.window._change_enc(tabwgt.encounter_id_list[index])

    def _close_tab(tabwgt, index):
        if tabwgt.encounter_id_list[index] is not None:
            tabwgt.encounter_id_list.pop(index)
            tabwgt.removeTab(index)

    def _add_enc_tab(tabwgt, eid, enctext):
        if eid not in tabwgt.encounter_id_list:
            # tab_name = str(eid) + ' - ' + str(enctext)
            tab_name = str(enctext)
            tabwgt.addTab(QtGui.QWidget(), tab_name)

            tabwgt.encounter_id_list.append(eid)
            index = len(tabwgt.encounter_id_list) - 1
        else:
            index = tabwgt.encounter_id_list.index(eid)

        tabwgt.setCurrentIndex(index)
        tabwgt._on_change(index)

    def _update_enc_tab_name(tabwgt, eid, enctext):
        for index, _id in enumerate(tabwgt.encounter_id_list):
            if eid == _id:
                tabwgt.setTabText(index, enctext)
