# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import functools
from wbia.guitool.__PYQT__ import QtCore, QtGui  # NOQA
from wbia.guitool.__PYQT__.QtCore import Qt  # NOQA
import utool as ut
from utool._internal import meta_util_six

ut.noinject(__name__, '[guitool.decorators]', DEBUG=False)

DEBUG = False


signal_ = QtCore.pyqtSignal


# SLOT DECORATOR
def slot_(*types):  # This is called at wrap time to get args
    """
    wrapper around pyqtslot decorator keep original function info
    """

    def pyqtSlotWrapper(func):
        # printDBG('[GUITOOL._SLOT] Wrapping: %r' % func.__name__)
        funcname = meta_util_six.get_funcname(func)

        @QtCore.pyqtSlot(*types, name=funcname)
        @ut.ignores_exc_tb
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
            funcname = meta_util_six.get_funcname(func)
            msg = 'caught exception in %r' % (funcname,)
            ut.printex(ex, msg, tb=True, pad_stdout=True)
            raise
        return result

    return checkqterr_wrapper
