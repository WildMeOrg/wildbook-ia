from __future__ import division, print_function
import functools
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt      # NOQA

DEBUG = False


signal_ = QtCore.pyqtSignal


# SLOT DECORATOR
def slot_(*types):  # This is called at wrap time to get args
    '''
    wrapper around pyqtslot decorator
    *args = types
    '''
    def pyqtSlotWrapper(func):
        @QtCore.pyqtSlot(*types, name=func.func_name)
        def slot_wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result
        slot_wrapper = functools.update_wrapper(slot_wrapper, func)
        return slot_wrapper
    return pyqtSlotWrapper


# DRAWING DECORATOR
def drawing(func):
    'Wraps a class function and draws windows on completion'
    @functools.wraps(func)
    def drawing_wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if kwargs.get('dodraw', True):
            pass
        return result
    return drawing_wrapper
