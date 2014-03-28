from __future__ import division, print_function
import sys

sys.argv.append('--strict')


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


def init_matplotlib(module_prefix='[???]'):
    import matplotlib
    import multiprocessing
    backend = matplotlib.get_backend()
    if  multiprocessing.current_process().name == 'MainProcess':
        if not '--quiet' in sys.argv:
            print('--- INIT MPL---')
            print('[main]  current backend is: %r' % backend)
            print('[main]  matplotlib.use(Qt4Agg)')
        if backend != 'Qt4Agg':
            matplotlib.use('Qt4Agg', warn=True, force=True)
            backend = matplotlib.get_backend()
            print(module_prefix + ' current backend is: %r' % backend)
        if '--notoolbar' in sys.argv or '--devmode' in sys.argv:
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
    print('[main] _init_gui()')
    guitool.ensure_qtapp()
    back = guiback.MainWindowBackend()
    guitool.activate_qwindow(back)
    return back


def _init_ibeis():
    print('[main] _init_ibeis()')
    from ibeis.control import IBEISControl
    import params
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
    print('[main] ibeis.main_api.main()')
    try:

        from ibeis.dev import params
        _parse_args(**kwargs)
        _init_parallel()
        utool.util_inject._inject_colored_exception_hook()
        _init_signals()
        init_matplotlib()
        if not params.args.nogui:
            back = _init_gui()
        ibs = _init_ibeis()
        if 'back' in vars() and ibs is not None:
            print('[main] Attatch ibeis control')
            back.connect_ibeis_control(ibs)
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
        _reset_signals()
        _close_parallel()
        if guiloop_ran or ipython_ran:
            # Exit cleanly if a main loop ran
            print('[main] ibeis clean exit')
            #sys.exit(0)
        else:
            # Something else happened
            print('[main] ibeis unclean exit')
    except Exception as ex:
        print('[main_loop] IBEIS Caught: %s %s' % (type(ex), ex))
        if '--strict' in sys.argv:
            raise
