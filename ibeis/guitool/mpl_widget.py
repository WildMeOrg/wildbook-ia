# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import guitool_ibeis as gt
import matplotlib as mpl
from guitool_ibeis.__PYQT__.QtCore import Qt
from guitool_ibeis.__PYQT__ import QtCore, QtWidgets, QtGui  # NOQA
from matplotlib.backend_bases import MouseEvent, KeyEvent, PickEvent

from guitool_ibeis import __PYQT__
if __PYQT__._internal.GUITOOL_PYQT_VERSION == 4:
    import matplotlib.backends.backend_qt4agg as backend_qt
else:
    import matplotlib.backends.backend_qt5agg as backend_qt
FigureCanvas = backend_qt.FigureCanvasQTAgg


class MatplotlibWidget(gt.GuitoolWidget):
    """
    A qt widget that contains a matplotlib figure

    References:
        http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
    """
    click_inside_signal = QtCore.pyqtSignal(MouseEvent, object)
    key_press_signal = QtCore.pyqtSignal(KeyEvent)
    pick_event_signal = QtCore.pyqtSignal(PickEvent)

    def initialize(self, pan_and_zoom=False):
        from plottool_ibeis.interactions import zoom_factory, pan_factory
        from plottool_ibeis import abstract_interaction

        # Create unmanaged figure and a canvas
        self.fig = mpl.figure.Figure()
        self.fig._no_raise_plottool_ibeis = True
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.addWidget(self.canvas)

        # Workaround key_press bug
        # References: https://github.com/matplotlib/matplotlib/issues/707
        self.canvas.setFocusPolicy(Qt.ClickFocus)

        self.reset_ax()

        # self.ax = self.fig.add_subplot(1, 1, 1)
        # pt.adjust_subplots(left=0, right=1, top=1, bottom=0, fig=self.fig)

        if pan_and_zoom or True:
            self.pan_events = pan_factory(self.ax)
            self.zoon_events = zoom_factory(self.ax)

        self.fig.canvas.mpl_connect('button_press_event', self._emit_button_press)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_signal.emit)
        self.fig.canvas.mpl_connect('pick_event', self.pick_event_signal.emit)

        self.MOUSE_BUTTONS = abstract_interaction.AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

        self.installEventFilter(self.parent())

    def clf(self):
        self.fig.clf()
        self.reset_ax()

    def reset_ax(self):
        # from plottool_ibeis.interactions import zoom_factory, pan_factory
        import plottool_ibeis as pt
        self.ax = self.fig.add_subplot(1, 1, 1)
        pt.adjust_subplots(left=0, right=1, top=1, bottom=0, fig=self.fig)
        # self.pan_events = pan_factory(self.ax)
        # self.zoon_events = zoom_factory(self.ax)
        return self.ax

    def _emit_button_press(self, event):
        from plottool_ibeis import interact_helpers as ih
        if ih.clicked_inside_axis(event):
            self.click_inside_signal.emit(event, event.inaxes)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m guitool_ibeis.mpl_widget
        python -m guitool_ibeis.mpl_widget --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
