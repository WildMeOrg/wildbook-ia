# -*- coding: utf-8 -*-
"""
http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# import utool as ut
# import sys
# import os
# import random
import time
import utool as ut
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__ import QtGui as QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# from matplotlib.backends import qt_compat


BASE = FigureCanvas


class QtAbstractMplInteraction(BASE):
    """
    Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).

    Args:
        self (?):
        parent (None): (default = None)
        width (int): (default = 5)
        height (int): (default = 4)
        dpi (int): (default = 100)

    CommandLine:
        python -m wbia.guitool.mpl_embed --exec-QtAbstractMplInteraction --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.guitool.mpl_embed import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import wbia.guitool
        >>> guitool.ensure_qapp()  # must be ensured before any embeding
        >>> self = QtAbstractMplInteraction()
        >>> parent = None
        >>> width = 5
        >>> height = 4
        >>> dpi = 100
        >>> self = QtAbstractMplInteraction(parent)
        >>> self.show()
        >>> print('Blocking')
        >>> self.start_blocking()
        >>> print('Done')
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
        >>> guitool.qtapp_loop(self, frequency=100, init_signals=True)

    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self._running = None
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)
        self.compute_initial_figure()
        #
        BASE.__init__(self, fig)
        self.setParent(parent)

        BASE.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        BASE.updateGeometry(self)
        self.fig = fig

    def compute_initial_figure(self):
        pass

    def closeEvent(self, event):
        self.stop_blocking()
        event.accept()
        # BASE.closeEvent(self, event)

    @QtCore.pyqtSlot()
    def start_blocking(self):
        # self.buttonStart.setDisabled(True)
        self._running = True
        while self._running:
            QtWidgets.qApp.processEvents()
            time.sleep(0.05)
        # self.buttonStart.setDisabled(False)

    @QtCore.pyqtSlot()
    def stop_blocking(self):
        self._running = False


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.guitool.mpl_embed
        python -m wbia.guitool.mpl_embed --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
