from __future__ import division, print_function
# Python
import sys
# Qt
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt      # NOQA
# HotSpotter

IS_ROOT = False
QAPP = None
DEBUG = False

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


#---------------
# SLOT DECORATORS


def slot_(*types, **kwargs_):  # This is called at wrap time to get args
    '''
    wrapper around pyqtslot decorator
    *args = types
    kwargs_['initdbg']
    kwargs_['rundbg']
    '''
    initdbg = kwargs_.get('initdbg', DEBUG)
    rundbg  = kwargs_.get('rundbg', DEBUG)

    # Wrap with debug statments
    def pyqtSlotWrapper(func):
        func_name = func.func_name
        if initdbg:
            print('[@guitool] Wrapping %r with slot_' % func.func_name)

        if rundbg:
            @QtCore.pyqtSlot(*types, name=func.func_name)
            def slot_wrapper(self, *args, **kwargs):
                argstr_list = map(str, args)
                kwastr_list = ['%s=%s' % item for item in kwargs.iteritems()]
                argstr = ', '.join(argstr_list + kwastr_list)
                print('[**slot_.Begining] %s(%s)' % (func_name, argstr))
                #with util.Indenter():
                result = func(self, *args, **kwargs)
                print('[**slot_.Finished] %s(%s)' % (func_name, argstr))
                return result
        else:
            @QtCore.pyqtSlot(*types, name=func.func_name)
            def slot_wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                return result

        slot_wrapper.func_name = func_name
        return slot_wrapper
    return pyqtSlotWrapper


#/SLOT DECORATOR
#---------------


# BLOCKING DECORATOR
# TODO: This decorator has to be specific to either front or back. Is there a
# way to make it more general?
def backblocking(func):
    #printDBG('[@guitool] Wrapping %r with backblocking' % func.func_name)

    def block_wrapper(back, *args, **kwargs):
        #print('[guitool] BLOCKING')
        wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception as ex:
            back.front.blockSignals(wasBlocked_)
            print('Block wrapper caugt exception in %r' % func.func_name)
            print('back = %r' % back)
            VERBOSE = False
            if VERBOSE:
                print('*args = %r' % (args,))
                print('**kwargs = %r' % (kwargs,))
            #print('ex = %r' % ex)
            import traceback
            print(traceback.format_exc())
            #back.user_info('Error in blocking ex=%r' % ex)
            back.user_info('Error while blocking gui:\nex=%r' % ex)
            raise
        back.front.blockSignals(wasBlocked_)
        #print('[guitool] UNBLOCKING')
        return result
    block_wrapper.func_name = func.func_name
    return block_wrapper


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


def select_orientation():
    print('[guitool] select_orientation()')
    pass


def select_roi():
    print('[guitool] select_roi()')


def init_qtapp():
    global IS_ROOT
    global QAPP
    if QAPP is not None:
        return QAPP, IS_ROOT
    app = QtCore.QCoreApplication.instance()
    is_root = app is None
    if is_root:  # if not in qtconsole
        print('[guitool] Initializing QApplication')
        app = QtGui.QApplication(sys.argv)
        QAPP = app
    try:
        # You are not root if you are in IPYTHON
        __IPYTHON__
        is_root = False
    except NameError:
        pass
    return app, is_root


def qtapp_loop(back=None, **kwargs):
    if back is not None:
        print('[guitool] setting active window')
        QAPP.setActiveWindow(back.front)
        back.timer = ping_python_interpreter(**kwargs)
    if IS_ROOT:
        print('[guitool] running core application loop.')
        QAPP.exec_()
    else:
        print('[guitool] using roots main loop')


def ping_python_interpreter(frequency=4200):  # 4200):
    'Create a QTimer which lets the python catch ctrl+c'
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(frequency)
    return timer
