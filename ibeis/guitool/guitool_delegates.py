from __future__ import absolute_import, division, print_function
from guitool_ibeis.__PYQT__ import QtCore, QtGui
from guitool_ibeis.__PYQT__ import QtWidgets
from guitool_ibeis.__PYQT__ import GUITOOL_PYQT_VERSION
import utool as ut
ut.noinject(__name__, '[guitool_ibeis.delegates]', DEBUG=False)


class APIDelegate(QtWidgets.QItemDelegate):
    is_persistant_editable = True
    def __init__(self, parent):
        QtWidgets.QItemDelegate.__init__(self, parent)

    def sizeHint(option, qindex):
        # QStyleOptionViewItem option
        return QtCore.QSize(50, 50)


class ImageDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent):
        print(dir(self))
        QtWidgets.QStyledItemDelegate.__init__(self, parent)

    def paint(self, painter, option, index):

        painter.fillRect(option.rect, QtGui.QColor(191, 222, 185))

        # path = "path\to\my\image.jpg"
        self.path = "image.bmp"

        image = QtGui.QImage(str(self.path))
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap.scaled(50, 40, QtCore.Qt.KeepAspectRatio)
        painter.drawPixmap(option.rect, pixmap)


class ComboDelegate(APIDelegate):
    """
    A delegate that places a fully functioning QComboBox in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        APIDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        combo = QtWidgets.QComboBox(parent)
        combo.addItems(['option1', 'option2', 'option3'])
        # FIXME: Change to newstyle signal slot
        if GUITOOL_PYQT_VERSION == 5:
            self.connect(combo.currentIndexChanged, self.currentIndexChanged)
        else:
            # I believe this particular option is broken in pyqt4
            self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
                         self, QtCore.SLOT("currentIndexChanged()"))
        return combo

    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        editor.setCurrentIndex(int(index.model().data(index).toString()))
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentIndex())

    @QtCore.pyqtSlot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())


class ButtonDelegate(APIDelegate):
    """
    A delegate that places a fully functioning QPushButton in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        # The parent is not an optional argument for the delegate as
        # we need to reference it in the paint method (see below)
        APIDelegate.__init__(self, parent)

    def paint(self, painter, option, index):
        # This method will be called every time a particular cell is
        # in view and that view is changed in some way. We ask the
        # delegates parent (in this case a table view) if the index
        # in question (the table cell) already has a widget associated
        # with it. If not, create one with the text for this index and
        # connect its clicked signal to a slot in the parent view so
        # we are notified when its used and can do something.
        if not self.parent().indexWidget(index):
            self.parent().setIndexWidget(
                index,
                QtWidgets.QPushButton(
                    index.data().toString(),
                    self.parent(),
                    clicked=self.parent().cellButtonClicked
                )
            )


# DELEGATE_MAP = {
#     'BUTTON': ButtonDelegate,
#     'COMBO': ComboDelegate,
# }
