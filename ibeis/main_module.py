from __future__ import absolute_import, division, print_function
from six.moves import builtins
import sys
import multiprocessing

PREINIT_MULTIPROCESSING_POOLS = '--preinit' in sys.argv
QUIET = '--quiet' in sys.argv
USE_GUI = '--gui' in sys.argv or '--nogui' not in sys.argv

try:
    profile = getattr(builtins, 'profile')
except AttributeError:
    def profile(func):
        return func


def _on_ctrl_c(signal, frame):
    print('[ibeis.main_module] Caught ctrl+c')
    _close_parallel()
    sys.exit(0)

#-----------------------
# private init functions


def _init_signals():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)


def _reset_signals():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior


def _parse_args():
    from ibeis import params
    params.parse_args()


#@profile
def _init_matplotlib():
    from plottool import __MPL_INIT__
    __MPL_INIT__.init_matplotlib()
    #try:
    #    from guitool import __PYQT__  # NOQA
    #except ImportError:
    #    print('[!main] WARNING guitool does not have __PYQT__')
    #    pass
    #import matplotlib as mpl
    #import utool
    #backend = mpl.get_backend()
    #if not sys.platform.startswith('win32') and not sys.platform.startswith('darwin') and os.environ.get('DISPLAY', None) is None:
    #    # Write to files if we cannot display
    #    TARGET_BACKEND = 'PDF'
    #else:
    #    TARGET_BACKEND = 'Qt4Agg'
    #if utool.in_main_process():
    #    if not utool.QUIET and utool.VERBOSE:
    #        print('--- INIT MPL---')
    #        print('[main] current backend is: %r' % backend)
    #        print('[main] mpl.use(%r)' % TARGET_BACKEND)
    #    if backend != TARGET_BACKEND:
    #        mpl.use(TARGET_BACKEND, warn=True, force=True)
    #        backend = mpl.get_backend()
    #        if not utool.QUIET and utool.VERBOSE:
    #            print('[main] current backend is: %r' % backend)
    #    if utool.get_argflag('--notoolbar'):
    #        toolbar = 'None'
    #    else:
    #        toolbar = 'toolbar2'
    #    mpl.rcParams['toolbar'] = toolbar
    #    mpl.rc('text', usetex=False)
    #    mpl_keypress_shortcuts = [key for key in mpl.rcParams.keys() if key.find('keymap') == 0]
    #    for key in mpl_keypress_shortcuts:
    #        mpl.rcParams[key] = ''
    #    #mpl.rcParams['text'].usetex = False
    #    #for key in mpl_keypress_shortcuts:
    #        #print('%s = %s' % (key, mpl.rcParams[key]))
    #    # Disable mpl shortcuts
    #        #mpl.rcParams['toolbar'] = 'None'
    #        #mpl.rcParams['interactive'] = True


#@profile
def _init_gui():
    import guitool
    import utool
    if not utool.QUIET:
        print('[main] _init_gui()')
    guitool.ensure_qtapp()
    #USE_OLD_BACKEND = '--old-backend' in sys.argv
    #if USE_OLD_BACKEND:
    from ibeis.gui import guiback
    back = guiback.MainWindowBackend()
    #else:
    #    from ibeis.gui import newgui
    #    back = newgui.IBEISGuiWidget()
    guitool.activate_qwindow(back.mainwin)
    return back


#@profile
def _init_ibeis(dbdir=None, verbose=True):
    import utool
    from ibeis.control import IBEISControl
    if verbose and not utool.QUIET:
        print('[main] _init_ibeis()')
    # Use command line dbdir unless user specifies it
    if dbdir is None:
        ibs = None
        utool.printWARN('[main!] WARNING args.dbdir is None')
    else:
        ibs = IBEISControl.IBEISController(dbdir=dbdir)
    return ibs


def __import_parallel_modules():
    # Import any modules which parallel process will use here
    # so they are accessable when the program forks
    #from utool import util_sysreq
    #util_sysreq.ensure_in_pythonpath('hesaff')
    #util_sysreq.ensure_in_pythonpath('pyrf')
    #util_sysreq.ensure_in_pythonpath('code')
    #import pyhesaff  # NOQA
    #import pyrf  # NOQA
    from ibeis.model.preproc import preproc_chip  # NOQA


def _init_parallel():
    from utool import util_parallel
    from ibeis import params
    __import_parallel_modules()
    util_parallel.set_num_procs(params.args.num_procs)
    if PREINIT_MULTIPROCESSING_POOLS:
        util_parallel.init_pool(params.args.num_procs)


def _close_parallel():
    from utool import util_parallel
    util_parallel.close_pool(terminate=True)


def _init_numpy():
    import numpy as np
    floating_error_options = ['ignore', 'warn', 'raise', 'call', 'print', 'log']
    on_float_err = floating_error_options[0]
    numpy_err = {
        'divide':  on_float_err,
        'over':    on_float_err,
        'under':   on_float_err,
        'invalid': on_float_err,
    }
    numpy_print = {
        'precision': 8,
        'threshold': 500,
        'edgeitems': 3,
        'linewidth': 200,  # default 75
        'suppress': False,
        'nanstr': 'nan',
        'formatter': None,
    }
    np.seterr(**numpy_err)
    np.set_printoptions(**numpy_print)


#-----------------------
# private loop functions


def _guitool_loop(main_locals, ipy=False):
    import guitool
    import utool
    from ibeis import params
    print('[main] guitool loop')
    back = main_locals.get('back', None)
    if back is not None:
        loop_freq = params.args.loop_freq
        guitool.qtapp_loop(qwin=back.mainwin, ipy=ipy or params.args.cmd, frequency=loop_freq)
        back.refresh_state()
    else:
        if not utool.QUIET and utool.VERBOSE:
            print('WARNING: back was not expected to be None')


#@profile
def main(gui=True, dbdir=None, defaultdb='cache',
         allow_newdir=False, db=None,
         **kwargs):
    """
    Program entry point
    Inits the system environment, an IBEISControl, and a GUI if requested

    If gui is False a gui instance will not be created
    """
    from ibeis.dev import main_commands
    from ibeis.dev import sysres
    # Display a visible intro message
    msg1 = '''
    _____ ....... _______ _____ _______
      |   |_____| |______   |   |______
    ..|.. |.....| |______s__|__ ______|
    '''
    msg2 = '''
    _____ ______  _______ _____ _______
      |   |_____] |______   |   |______
    __|__ |_____] |______ __|__ ______|
    '''
    print(msg2 if '--myway' not in sys.argv else msg1)
    # Init the only two main system api handles
    ibs = None
    back = None
    if not QUIET:
        print('[main] ibeis.main_module.main()')
    _preload()
    # Parse directory to be loaded from command line args
    # and explicit kwargs
    dbdir = sysres.get_args_dbdir(defaultdb, allow_newdir, db, dbdir, cache_priority=False)
    # Execute preload commands
    main_commands.preload_commands(dbdir, **kwargs)  # PRELOAD CMDS
    try:
        # Build IBEIS Control object
        ibs = _init_ibeis(dbdir)
        if gui and USE_GUI:
            back = _init_gui()
            back.connect_ibeis_control(ibs)
    except Exception as ex:
        print('[main()] IBEIS LOAD encountered exception: %s %s' % (type(ex), ex))
        raise
    main_commands.postload_commands(ibs, back)  # POSTLOAD CMDS
    main_locals = {'ibs': ibs, 'back': back}
    return main_locals


def opendb(db=None, dbdir=None, defaultdb='cache', allow_newdir=False,
           delete_ibsdir=False, verbose=False):
    """ main without the preload (except for option to delete database before
    opening) """
    from ibeis.dev import sysres
    from ibeis import ibsfuncs
    dbdir = sysres.get_args_dbdir(defaultdb, allow_newdir, db, dbdir, cache_priority=False)
    if delete_ibsdir is True:
        assert allow_newdir, 'must be making new directory if you are deleting everything!'
        ibsfuncs.delete_ibeis_database(dbdir)
    ibs = _init_ibeis(dbdir, verbose=verbose)
    return ibs


def start(*args, **kwargs):
    """ alias for main() """  # + main.__doc__
    return main(*args, **kwargs)


def test_main(gui=True, dbdir=None, defaultdb='cache', allow_newdir=False,
              db=None):
    """ alias for main() """  # + main.__doc__
    from ibeis.dev import sysres
    _preload()
    dbdir = sysres.get_args_dbdir(defaultdb, allow_newdir, db, dbdir, cache_priority=False)
    ibs = _init_ibeis(dbdir)
    return ibs


#@profile
def _preload(mpl=True, par=True, logging=True):
    """ Sets up python environment """
    import utool
    #from ibeis.dev import main_helpers
    from ibeis import params
    if  multiprocessing.current_process().name != 'MainProcess':
        return
    #_parse_args()
    # mpl backends
    if logging and not params.args.nologging:
        utool.start_logging(appname='ibeis')
    if mpl:
        _init_matplotlib()
    # numpy print settings
    _init_numpy()
    # parallel servent processes
    if par:
        _init_parallel()
    # ctrl+c
    _init_signals()
    # inject colored exceptions
    utool.util_inject.inject_colored_exceptions()
    # register type aliases for debugging
    #main_helpers.register_utool_aliases()
    #return params.args


#@profile
def main_loop(main_locals, rungui=True, ipy=False, persist=True):
    """
    Runs the qt loop if the GUI was initialized and returns an executable string
    for embedding an IPython terminal if requested.

    If rungui is False the gui will not loop even if back has been created
    """
    print('[main] ibeis.main_module.main_loop()')
    from ibeis import params
    import utool
    #print('current process = %r' % (multiprocessing.current_process().name,))
    #== 'MainProcess':
    if rungui and not params.args.nogui:
        try:
            _guitool_loop(main_locals, ipy=ipy)
        except Exception as ex:
            utool.printex(ex, 'error in main_loop')
            raise
    #if not persist or params.args.cmd:
    #    main_close()
    # Put locals in the exec namespace
    ipycmd_execstr = utool.ipython_execstr()
    locals_execstr = utool.execstr_dict(main_locals, 'main_locals')
    execstr = locals_execstr + '\n' + ipycmd_execstr
    return execstr


def main_close(main_locals=None):
    _close_parallel()
    _reset_signals()


#if __name__ == '__main__':
#    multiprocessing.freeze_support()
