from __future__ import absolute_import, division, print_function
import functools
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt      # NOQA
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[guitool.decorators]', DEBUG=False)

DEBUG = False


signal_ = QtCore.pyqtSignal


# SLOT DECORATOR
def slot_(*types):  # This is called at wrap time to get args
    """
    wrapper around pyqtslot decorator
    *args = types
    """
    def pyqtSlotWrapper(func):
        printDBG('[GUITOOL._SLOT] Wrapping: %r' % func.func_name)
        @QtCore.pyqtSlot(*types, name=func.func_name)
        @utool.ignores_exc_tb
        def slot_wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result
        slot_wrapper = functools.update_wrapper(slot_wrapper, func)
        return slot_wrapper
    return pyqtSlotWrapper


# DRAWING DECORATOR
def drawing(func):
    """
    Wraps a class function and draws windows on completion
    """
    @utool.ignores_exc_tb
    @functools.wraps(func)
    def drawing_wrapper(self, *args, **kwargs):
        #print('[DRAWING]: ' + utool.func_str(func, args, kwargs))
        result = func(self, *args, **kwargs)
        if kwargs.get('dodraw', True):
            pass
        return result
    return drawing_wrapper
