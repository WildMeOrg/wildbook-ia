from __future__ import absolute_import, division, print_function
import sys
import os

__IS_INITIALIZED__ = False


def init_matplotlib():
    global __IS_INITIALIZED__
    import matplotlib as mpl
    import utool
    try:
        from guitool import __PYQT__  # NOQA
    except ImportError:
        print('[!plotttool] WARNING guitool does not have __PYQT__')
        pass
    backend = mpl.get_backend()
    if (not sys.platform.startswith('win32') and
        not sys.platform.startswith('darwin') and
         os.environ.get('DISPLAY', None) is None):
        # Write to files if we cannot display
        TARGET_BACKEND = 'PDF'
    else:
        TARGET_BACKEND = 'Qt4Agg'
    if utool.in_main_process():
        if __IS_INITIALIZED__:
            print('[!plottool] matplotlib has already been initialized')
            return False
        __IS_INITIALIZED__ = True
        if not utool.QUIET and utool.VERBOSE:
            print('--- INIT MPL---')
            print('[pt] current backend is: %r' % backend)
            print('[pt] mpl.use(%r)' % TARGET_BACKEND)
        if backend != TARGET_BACKEND:
            if utool.get_flag('--leave-mpl-backend-alone'):
                print('[pt] LEAVE THE BACKEND ALONE !!! was specified')
                print('[pt] not changing mpl backend')
            else:
                mpl.use(TARGET_BACKEND, warn=True, force=True)
                backend = mpl.get_backend()
            if not utool.QUIET and utool.VERBOSE:
                print('[pt] current backend is: %r' % backend)
        if utool.get_flag('--notoolbar'):
            toolbar = 'None'
        else:
            toolbar = 'toolbar2'
        mpl.rcParams['toolbar'] = toolbar
        mpl.rc('text', usetex=False)
        mpl_keypress_shortcuts = [key for key in mpl.rcParams.keys() if key.find('keymap') == 0]
        for key in mpl_keypress_shortcuts:
            mpl.rcParams[key] = ''
        #mpl.rcParams['text'].usetex = False
        #for key in mpl_keypress_shortcuts:
            #print('%s = %s' % (key, mpl.rcParams[key]))
        # Disable mpl shortcuts
            #mpl.rcParams['toolbar'] = 'None'
            #mpl.rcParams['interactive'] = True
