from __future__ import division, print_function


def _on_ctrl_c(signal, frame):
    import sys
    print('Caught ctrl+c')
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


def _init_gui():
    from ibeis.view import guitool, guiback
    print('[main] _init_gui()')
    guitool.init_qtapp()
    back = guiback.MainWindowBackend()
    back.show()
    return back


def _init_ibeis():
    print('[main] _init_ibeis()')
    from ibeis.control import IBEISControl
    ibs = IBEISControl.IBEISControl()
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
    from ibeis.view import guitool
    from ibeis.dev import params
    back = main_locals['back']
    loop_freq = params.args.loop_freq
    guitool.qtapp_loop(back=back, frequency=loop_freq)
    return True


def _ipython_loop(main_locals):
    from utool import util_dbg
    embedded = util_dbg.inIPython()
    if not embedded:
        util_dbg.embed(parent_locals=main_locals)
        return True
    return False


def main(**kwargs):
    print('[main] ibeis.main_api.main()')
    from utool.util_inject import _inject_colored_exception_hook
    from ibeis.dev import params
    _parse_args(**kwargs)
    _init_parallel()
    _inject_colored_exception_hook()
    _init_signals()
    if not params.args.nogui:
        back = _init_gui()
    ibs = _init_ibeis()
    main_locals = locals()
    return main_locals


def main_loop(main_locals, loop=True):
    print('[main] ibeis.main_api.main_loop()')
    import sys
    from ibeis.dev import params
    exit_bit = True
    if loop:
        # Choose a main loop depending on params.args
        if exit_bit and params.args.cmd:
            exit_bit = _ipython_loop(main_locals)
        if exit_bit and not params.args.nogui:
            exit_bit = _guitool_loop(main_locals)
    _reset_signals()
    _close_parallel()
    if exit_bit:
        # Exit cleanly if a main loop ran
        print('[main] ibeis clean exit')
        sys.exit(0)
    else:
        # Something else happened
        print('[main] ibeis unclean exit')
