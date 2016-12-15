# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import guitool as gt
import plottool as pt
from guitool.__PYQT__.QtCore import Qt
from guitool.__PYQT__ import QtCore, QtWidgets, QtGui  # NOQA
from plottool import interact_helpers as ih
from matplotlib.backend_bases import MouseEvent, KeyEvent, PickEvent

from guitool import __PYQT__
if __PYQT__._internal.GUITOOL_PYQT_VERSION == 4:
    import matplotlib.backends.backend_qt4agg as backend_qt
else:
    import matplotlib.backends.backend_qt5agg as backend_qt
FigureCanvas = backend_qt.FigureCanvasQTAgg


class MatplotlibWidget(gt.GuitoolWidget):
    click_inside_signal = QtCore.pyqtSignal(MouseEvent, object)
    key_press_signal = QtCore.pyqtSignal(KeyEvent)
    pick_event_signal = QtCore.pyqtSignal(PickEvent)

    def initialize(self):
        from plottool.interactions import zoom_factory, pan_factory
        self.fig = pt.plt.figure()
        self.fig._no_raise_plottool = True
        # Add a figure canvas widget to this widget
        self.canvas = FigureCanvas(self.fig)
        # Workaround key_press bug
        # References: https://github.com/matplotlib/matplotlib/issues/707
        self.canvas.setFocusPolicy(Qt.ClickFocus)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.addWidget(self.canvas)

        self.pan_events = pan_factory(self.ax)
        self.zoon_events = zoom_factory(self.ax)
        self.fig.canvas.mpl_connect('button_press_event', self._emit_button_press)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_signal.emit)
        self.fig.canvas.mpl_connect('pick_event', self.pick_event_signal.emit)

        # self.MOUSE_BUTTONS = abstract_interaction.AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

        self.installEventFilter(self.parent())

    def _emit_button_press(self, event):
        if ih.clicked_inside_axis(event):
            self.click_inside_signal.emit(event, event.inaxes)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m guitool.mpl_widget
        python -m guitool.mpl_widget --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
