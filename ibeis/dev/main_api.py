from __future__ import absolute_import, division, print_function
import sys

sys.argv.append('--strict')  # do not supress any errors


def _on_ctrl_c(signal, frame):
    print('Caught ctrl+c')
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


def _parse_args(**kwargs):
    from ibeis.dev import params
    params.parse_args(**kwargs)


def _init_matplotlib():
    import matplotlib
    import multiprocessing
    import utool
    backend = matplotlib.get_backend()
    if  multiprocessing.current_process().name == 'MainProcess':
        if not utool.QUIET:
            print('--- INIT MPL---')
            print('[main]  current backend is: %r' % backend)
            print('[main]  matplotlib.use(Qt4Agg)')
        if backend != 'Qt4Agg':
            matplotlib.use('Qt4Agg', warn=True, force=True)
            backend = matplotlib.get_backend()
            if not utool.QUIET:
                print('[main] current backend is: %r' % backend)
        if utool.get_flag('--notoolbar') or utool.get_flag('--devmode'):
            toolbar = 'None'
        else:
            toolbar = 'toolbar2'
        matplotlib.rcParams['toolbar'] = toolbar
        matplotlib.rc('text', usetex=False)
        mpl_keypress_shortcuts = [key for key in matplotlib.rcParams.keys() if key.find('keymap') == 0]
        for key in mpl_keypress_shortcuts:
            matplotlib.rcParams[key] = ''
        #matplotlib.rcParams['text'].usetex = False
        #for key in mpl_keypress_shortcuts:
            #print('%s = %s' % (key, matplotlib.rcParams[key]))
        # Disable mpl shortcuts
            #matplotlib.rcParams['toolbar'] = 'None'
            #matplotlib.rcParams['interactive'] = True


def _init_gui():
    import guitool
    from ibeis.view import guiback
    import utool
    if not utool.QUIET:
        print('[main] _init_gui()')
    guitool.ensure_qtapp()
    back = guiback.MainWindowBackend()
    guitool.activate_qwindow(back)
    return back


def _init_ibeis():
    import utool
    from ibeis.control import IBEISControl
    from . import params
    if not utool.QUIET:
        print('[main] _init_ibeis()')
    dbdir = params.args.dbdir
    if dbdir is None:
        print('[main!] WARNING args.dbdir is None')
        ibs = None
    else:
        ibs = IBEISControl.IBEISControl(dbdir=dbdir)
    return ibs


def __import_parallel_modules():
    # Import any modules which parallel process will use here
    # so they are accessable when the program forks
    from utool import util_sysreq
    util_sysreq.ensure_in_pythonpath('hesaff')
    import pyhesaff  # NOQA


def _init_parallel():
    from utool import util_parallel
    from ibeis.dev import params
    __import_parallel_modules()
    util_parallel.init_pool(params.args.num_procs)


def _close_parallel():
    from utool import util_parallel
    util_parallel.close_pool()


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
        'threshold': 1000,
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


def _guitool_loop(main_locals):
    import guitool
    from ibeis.dev import params
    back = main_locals['back']
    loop_freq = params.args.loop_freq
    guitool.qtapp_loop(back=back, frequency=loop_freq)
    return True


def _ipython_loop(main_locals):
    import utool
    embedded = utool.util_dbg.inIPython()
    if not embedded:
        exec_lines = []
        if 'back' in main_locals:
            import guitool  # NOQA
            back = main_locals['back']  # NOQA
            #exec_lines.append('guitool.qtapp_loop_nonblocking(back)')
        def fixgui():
            from IPython.lib.inputhook import enable_qt4
            from IPython.lib.guisupport import start_event_loop_qt4
            print('[guitool] Starting ipython qt4 hook')
            enable_qt4()
            start_event_loop_qt4(guitool.get_qtapp())
        print('Embedding run fix_gui to actually see whats going on')
        main_locals.update(locals())
        utool.util_dbg.embed(parent_locals=main_locals,
                             parent_globals=globals(),
                             exec_lines=exec_lines)
        return True
    return False


def main(**kwargs):
    import utool
    from ibeis.dev import main_commands
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
    print(msg2 if not utool.getflag('--myway') in sys.argv else msg1)
    if not utool.QUIET:
        print('[main] ibeis.main_api.main()')
    try:

        from ibeis.dev import params
        _init_numpy()
        _parse_args(**kwargs)
        _init_parallel()
        utool.util_inject._inject_colored_exception_hook()
        _init_signals()
        _init_matplotlib()
        main_commands.preload_commands()  # PRELOAD CMDS
        if not params.args.nogui:
            back = _init_gui()
        ibs = _init_ibeis()
        if 'back' in vars() and ibs is not None:
            back.connect_ibeis_control(ibs)
        main_commands.postload_commands(ibs)  # POSTLOAD CMDS
    except Exception as ex:
        print('[main()] IBEIS Caught: %s %s' % (type(ex), ex))
        print(ex)
        if '--strict' in sys.argv:
            raise
    main_locals = locals()
    return main_locals


def main_loop(main_locals, loop=True, rungui=True):
    print('[main] ibeis.main_api.main_loop()')
    from ibeis.dev import params
    try:
        ipython_ran = False
        guiloop_ran = not rungui
        if loop:
            # Choose a main loop depending on params.args
            if params.args.cmd:
                ipython_ran = _ipython_loop(main_locals)
                rungui = rungui and ipython_ran
            if rungui and not params.args.nogui:
                guiloop_ran = _guitool_loop(main_locals)
    except Exception as ex:
        print('[main_loop] IBEIS Caught: %s %s' % (type(ex), ex))
        if '--strict' in sys.argv:
            raise
    _close_parallel()
    _reset_signals()
    if guiloop_ran or ipython_ran:
        # Exit cleanly if a main loop ran
        print('[main] ibeis clean EXIT')
        #sys.exit(0)
    else:
        # Something else happened
        print('[main] ibeis unclean EXIT')
