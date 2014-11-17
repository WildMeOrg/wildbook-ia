from __future__ import absolute_import, division, print_function
import sys
import os

__IS_INITIALIZED__ = False
__WHO_INITIALIZED__ = None


VERBOSE_MPLINIT = '--verb-mpl' in sys.argv or '--verbose' in sys.argv


def init_matplotlib(verbose=VERBOSE_MPLINIT):
    global __IS_INITIALIZED__
    global __WHO_INITIALIZED__
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
            if verbose:
                print('[!plottool] matplotlib has already been initialized.  backend=%r' % backend)
                print('[!plottool] Initially initialized by %r' % __WHO_INITIALIZED__)
                print('[!plottool] Trying to be init by %r' % (utool.get_caller_name(N=range(0, 5))))
            return False
        else:
            __WHO_INITIALIZED__ = utool.get_caller_name(N=range(0, 5))
            if verbose:
                print('[plottool] matplotlib initialized by %r' % __WHO_INITIALIZED__)
            #__WHO_INITIALIZED__ = utool.get_caller_name()
            __IS_INITIALIZED__ = True
        if not utool.QUIET and utool.VERBOSE:
            if verbose:
                print('--- INIT MPL---')
                print('[pt] current backend is: %r' % backend)
                print('[pt] mpl.use(%r)' % TARGET_BACKEND)
        if backend != TARGET_BACKEND:
            if utool.get_argflag('--leave-mpl-backend-alone'):
                print('[pt] LEAVE THE BACKEND ALONE !!! was specified')
                print('[pt] not changing mpl backend')
            else:
                mpl.use(TARGET_BACKEND, warn=True, force=True)
                backend = mpl.get_backend()
            if not utool.QUIET and utool.VERBOSE:
                print('[pt] current backend is: %r' % backend)
        if utool.get_argflag('--notoolbar'):
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
