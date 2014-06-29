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
        #printDBG('[GUITOOL._SLOT] Wrapping: %r' % func.func_name)
        @QtCore.pyqtSlot(*types, name=func.func_name)
        @utool.ignores_exc_tb
        def slot_wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result
        slot_wrapper = functools.update_wrapper(slot_wrapper, func)
        return slot_wrapper
    return pyqtSlotWrapper


def checks_qt_error(func):
    """
    Decorator which reports qt errors which would otherwise be silent Useful if
    we haven't overriden sys.excepthook but we have, so this isnt useful.
    """
    @functools.wraps(func)
    def checkqterr_wrapper(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
        except Exception as ex:
            utool.printex(ex, 'caught exception in %r' % func.func_name,
                          tb=True, separate=True)
            raise
        return result
    return checkqterr_wrapper
