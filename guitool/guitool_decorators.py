from __future__ import division, print_function
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt      # NOQA

DEBUG = False


# SLOT DECORATOR
def slot_(*types):  # This is called at wrap time to get args
    '''
    wrapper around pyqtslot decorator
    *args = types
    '''
    def pyqtSlotWrapper(func):
        func_name = func.func_name

        @QtCore.pyqtSlot(*types, name=func.func_name)
        def slot_wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result

        slot_wrapper.func_name = func_name
        return slot_wrapper
    return pyqtSlotWrapper


signal_ = QtCore.pyqtSignal


# DRAWING DECORATOR
def drawing(func):
    'Wraps a class function and draws windows on completion'
    def drawing_wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if kwargs.get('dodraw', True):
            pass
        return result
    drawing_wrapper.func_name = func.func_name
    return drawing_wrapper
