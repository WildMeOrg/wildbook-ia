# -*- coding: utf-8 -*-
"""
This module defines the entry point into the IBEIS system
wbia.opendb and wbia.main are the main entry points
"""
import logging
import sys
import multiprocessing
from contextlib import contextmanager

import utool as ut

from wbia import params


QUIET = '--quiet' in sys.argv
NOT_QUIET = not QUIET
USE_GUI = '--gui' in sys.argv or '--nogui' not in sys.argv


(print, _, __) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


def _on_ctrl_c(signal, frame):
    proc_name = multiprocessing.current_process().name
    logger.info('[wbia.entry_points] Caught ctrl+c in %s' % (proc_name,))
    sys.exit(0)


# -----------------------
# private init functions


def _init_signals():
    import signal

    signal.signal(signal.SIGINT, _on_ctrl_c)


def _reset_signals():
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior


def _init_matplotlib():
    from wbia.plottool import __MPL_INIT__

    __MPL_INIT__.init_matplotlib()


def _init_gui(activate=True):
    from wbia import guitool

    if NOT_QUIET:
        logger.info('[main] _init_gui()')
    guitool.ensure_qtapp()
    from wbia.gui import guiback

    back = guiback.MainWindowBackend()
    if activate:
        guitool.activate_qwindow(back.mainwin)
    return back


def _init_wbia(dbdir=None, verbose=None, use_cache=True, web=None, **kwargs):
    """
    Private function that calls code to create an wbia controller
    """
    import utool as ut

    params.parse_args()
    from wbia.control import IBEISControl

    # Set up logging
    # TODO (30-Nov-12020) This is intended to be a temporary fix to logging.
    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(logging.StreamHandler())

    if verbose is None:
        verbose = ut.VERBOSE
    if verbose and NOT_QUIET:
        logger.info('[main] _init_wbia()')
    # Use command line dbdir unless user specifies it
    if dbdir is None:
        ibs = None
        logger.info('[main!] WARNING: args.dbdir is None')
    else:
        kwargs = kwargs.copy()
        request_dbversion = kwargs.pop('request_dbversion', None)
        force_serial = kwargs.get('force_serial', None)
        ibs = IBEISControl.request_IBEISController(
            dbdir=dbdir,
            use_cache=use_cache,
            request_dbversion=request_dbversion,
            force_serial=force_serial,
        )
        # BBB (12-Jan-12021) daily database backup for the sqlite database
        if not ibs.is_using_postgres_db:
            ibs.daily_backup_database()

        if web is None:
            web = ut.get_argflag(
                ('--webapp', '--webapi', '--web', '--browser'),
                help_='automatically launch the web app / web api',
            )
            # web = params.args.webapp
        if web:
            from wbia.web import app

            port = params.args.webport
            app.start_from_wbia(ibs, port=port, **kwargs)
    return ibs


def _init_parallel():
    import utool as ut

    if ut.VERBOSE:
        logger.info('_init_parallel')
    from utool import util_parallel

    params.parse_args()

    # Import any modules which parallel process will use here
    # so they are accessable when the program forks
    # from utool import util_sysreq
    # util_sysreq.ensure_in_pythonpath('hesaff')
    # util_sysreq.ensure_in_pythonpath('pyrf')
    # util_sysreq.ensure_in_pythonpath('code')
    # import pyhesaff  # NOQA
    # import pyrf  # NOQA
    from wbia import core_annots  # NOQA

    # .algo.preproc import preproc_chip  # NOQA
    util_parallel.set_num_procs(params.args.num_procs)
    # if PREINIT_MULTIPROCESSING_POOLS:
    #    util_parallel.init_pool(params.args.num_procs)


def _init_numpy():
    import utool as ut
    import numpy as np

    if ut.VERBOSE:
        logger.info('_init_numpy')
    error_options = ['ignore', 'warn', 'raise', 'call', 'print', 'log']
    on_err = error_options[0]
    # np.seterr(divide='ignore', invalid='ignore')
    numpy_err = {
        'divide': on_err,
        'over': on_err,
        'under': on_err,
        'invalid': on_err,
    }
    # numpy_print = {
    #    'precision': 8,
    #    'threshold': 500,
    #    'edgeitems': 3,
    #    'linewidth': 200,  # default 75
    #    'suppress': False,
    #    'nanstr': 'nan',
    #    'formatter': None,
    # }
    np.seterr(**numpy_err)
    # np.set_printoptions(**numpy_print)


# -----------------------
# private loop functions


def _guitool_loop(main_locals, ipy=False):
    from wbia import guitool, params

    logger.info('[main] guitool loop')
    back = main_locals.get('back', None)
    if back is not None:
        loop_freq = params.args.loop_freq
        ipy = ipy or params.args.cmd
        guitool.qtapp_loop(
            qwin=back.mainwin, ipy=ipy, frequency=loop_freq, init_signals=False
        )
        if ipy:  # If we're in IPython, the qtapp loop won't block, so we need to refresh
            back.refresh_state()
    else:
        if NOT_QUIET:
            logger.info('WARNING: back was not expected to be None')


def set_newfile_permissions():
    r"""
    sets this processes default permission bits when creating new files

    CommandLine:
        python -m wbia.entry_points --test-set_newfile_permissions

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.entry_points import *  # NOQA
        >>> import os
        >>> import utool as ut
        >>> # write before umask
        >>> ut.delete('tempfile1.txt')
        >>> ut.write_to('tempfile1.txt', 'foo')
        >>> stat_result1 = os.stat('tempfile1.txt')
        >>> # apply umask
        >>> set_newfile_permissions()
        >>> ut.delete('tempfile2.txt')
        >>> ut.write_to('tempfile2.txt', 'foo')
        >>> stat_result2 = os.stat('tempfile2.txt')
        >>> # verify results
        >>> print('old masked all bits = %o' % (stat_result1.st_mode))
        >>> print('new masked all bits = %o' % (stat_result2.st_mode))
    """
    import os

    # import stat
    # Set umask so all files written will be group read and writable
    # To get the permissions we want subtract what you want from 0o0666 because
    # umask subtracts the mask you give it.
    # mask = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH
    # mask = 0o000  # most permissive umask
    mask = 0o000  # most permissive umask
    prev_mask = os.umask(mask)
    return prev_mask
    # logger.info('prev_mask = %o' % (prev_mask,))
    # logger.info('new_mask  = %o' % (mask,))


def main(
    gui=True,
    dbdir=None,
    defaultdb='cache',
    allow_newdir=False,
    db=None,
    delete_ibsdir=False,
    **kwargs,
):
    """
    Program entry point
    Inits the system environment, an IBEISControl, and a GUI if requested

    Args:
        gui (bool): (default=True) If gui is False a gui instance will not be created
        dbdir (None): full directory of a database to load
        db (None): name of database to load relative to the workdir
        allow_newdir (bool): (default=False) if False an error is raised if a
            a new database is created
        defaultdb (str): codename of database to load if db and dbdir is None. a value
            of 'cache' will open the last database opened with the GUI.

    Returns:
        dict: main_locals
    """
    _preload()
    set_newfile_permissions()
    from wbia.init import main_commands
    from wbia.init import sysres

    # Display a visible intro message
    msg = """
    _____ ______  _______ _____ _______
      |   |_____] |______   |   |______
    __|__ |_____] |______ __|__ ______|
    """
    if NOT_QUIET:
        logger.info(msg)
    # Init the only two main system api handles
    ibs = None
    back = None
    if NOT_QUIET:
        logger.info('[main] wbia.entry_points.main()')
    DIAGNOSTICS = NOT_QUIET
    if DIAGNOSTICS:
        import os
        import utool as ut
        import wbia

        logger.info('[main] MAIN DIAGNOSTICS')
        logger.info('[main]  * username = %r' % (ut.get_user_name()))
        logger.info('[main]  * wbia.__version__ = %r' % (wbia.__version__,))
        logger.info('[main]  * computername = %r' % (ut.get_computer_name()))
        logger.info('[main]  * cwd = %r' % (os.getcwd(),))
        logger.info('[main]  * sys.argv = %r' % (sys.argv,))
    # Parse directory to be loaded from command line args
    # and explicit kwargs
    dbdir = sysres.get_args_dbdir(
        defaultdb=defaultdb, allow_newdir=allow_newdir, db=db, dbdir=dbdir
    )
    if delete_ibsdir is True:
        from wbia.other import ibsfuncs

        assert (
            allow_newdir
        ), 'must be making new directory if you are deleting everything!'
        ibsfuncs.delete_wbia_database(dbdir)

    # limit = sys.getrecursionlimit()
    # if limit == 1000:
    #    logger.info('Setting Recursion Limit to 3000')
    #    sys.setrecursionlimit(3000)
    # Execute preload commands
    main_commands.preload_commands(dbdir, **kwargs)  # PRELOAD CMDS
    try:
        # Build IBEIS Control object
        ibs = _init_wbia(dbdir)
        if gui and USE_GUI:
            back = _init_gui(activate=kwargs.get('activate', True))
            back.connect_wbia_control(ibs)
    except Exception as ex:
        logger.info('[main()] IBEIS LOAD encountered exception: %s %s' % (type(ex), ex))
        raise
    main_commands.postload_commands(ibs, back)  # POSTLOAD CMDS
    main_locals = {'ibs': ibs, 'back': back}
    return main_locals


def opendb_in_background(*args, **kwargs):
    """
    Starts a web server in the background
    """
    import utool as ut
    import time

    sec = kwargs.pop('wait', 0)
    if sec != 0:
        raise AssertionError('wait is depricated')
        logger.info('waiting %s seconds for startup' % (sec,))
    proc = ut.spawn_background_process(opendb, *args, **kwargs)
    if sec != 0:
        raise AssertionError('wait is depricated')
        time.sleep(sec)  # wait for process to initialize
    return proc


@contextmanager
def opendb_with_web(*args, with_job_engine=False, **kwargs):
    """Opens the database and starts the web server.

    Returns:
        ibs, client - IBEISController and Werkzeug Client

    Example:
        >>> from wbia.entry_points import opendb_with_web
        >>> expected_response_data = {'status': {'success': True, 'code': 200, 'message': '', 'cache': -1}, 'response': True}
        >>> with opendb_with_web('testdb1') as (ibs, client):
        ...     response = client.get('/api/test/heartbeat/')
        ...     assert expected_response_data == response.json

    """
    from wbia.control.controller_inject import get_flask_app

    # Create the controller instance
    ibs = opendb(*args, **kwargs)
    if with_job_engine:
        # TODO start jobs engine
        pass

    # Create the web application
    app = get_flask_app()
    # ??? Gotta attach the controller to the application?
    setattr(app, 'ibs', ibs)

    # Return the controller and client instances to the caller
    with app.test_client() as client:
        yield ibs, client


def opendb_fg_web(*args, **kwargs):
    """
    Ignore:
        >>> from wbia.entry_points import *  # NOQA
        >>> kwargs = {'db': 'testdb1'}
        >>> args = tuple()

        >>> import wbia
        >>> ibs = wbia.opendb_fg_web()

    """
    # Gives you context inside the web app for testing
    kwargs['start_web_loop'] = False
    kwargs['web'] = True
    kwargs['browser'] = False
    ibs = opendb(*args, **kwargs)
    from wbia.control import controller_inject

    app = controller_inject.get_flask_app()
    ibs.app = app
    return ibs


def opendb_bg_web(*args, managed=False, **kwargs):
    """
    Wrapper around opendb_in_background, returns a nice web_ibs
    object to execute web calls using normal python-like syntax

    Args:
        *args: passed to opendb_in_background
        **kwargs:
            port (int):
            domain (str): if specified assumes server is already running
                somewhere otherwise kwargs is passed to opendb_in_background
            start_job_queue (bool)
            managed (bool): if True, return a context manager that terminates the server upon completion of the block

    Returns:
        web_ibs - this is a KillableProcess object with special functions

    CommandLine:
        python -m wbia.entry_points opendb_bg_web

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.entry_points import *  # NOQA
        >>> args = tuple()
        >>> kwargs = {}
        >>> print('Opening a web_ibs')
        >>> web_ibs = opendb_bg_web()
        >>> print('SUCESS Opened a web_ibs!')
        >>> print(web_ibs)
        >>> print('Now kill the web_ibs')
        >>> web_ibs.terminate2()
    """
    import utool as ut
    from wbia.web import appfuncs

    domain = kwargs.pop('domain', ut.get_argval('--domain', type_=str, default=None))
    port = kwargs.pop('port', appfuncs.DEFAULT_WEB_API_PORT)

    if 'wait' in kwargs:
        logger.info(
            'NOTE: No need to specify wait param anymore. '
            'This is automatically taken care of.'
        )

    if domain is None:
        # Requesting a local test server
        _kw = dict(web=True, browser=False)
        _kw.update(kwargs)
        web_ibs = opendb_in_background(*args, **_kw)
    else:
        # Using a remote controller, no need to spin up anything
        web_ibs = ut.DynStruct()
        web_ibs.terminate2 = lambda: None
    # Augment web instance with usefull test functions
    if domain is None:
        domain = 'http://127.0.1.1'
    if not domain.startswith('http://'):
        domain = 'http://' + domain
    baseurl = domain + ':' + str(port)

    web_ibs.domain = domain
    web_ibs.port = port
    web_ibs.baseurl = baseurl

    def get(suffix, **kwargs):
        import requests

        return requests.get(baseurl + suffix)

    def post(suffix, **kwargs):
        import requests

        return requests.post(baseurl + suffix)

    def send_wbia_request(suffix, type_='post', json=True, **kwargs):
        """
        Posts a request to a url suffix
        """
        import requests
        import utool as ut

        if not suffix.endswith('/'):
            raise Exception('YOU PROBABLY WANT A / AT THE END OF YOUR URL')
        payload = ut.map_dict_vals(ut.to_json, kwargs)
        if type_ == 'post':
            resp = requests.post(baseurl + suffix, data=payload)
            content = resp._content
        elif type_ == 'get':
            resp = requests.get(baseurl + suffix, data=payload)
            content = resp.content
        if json:
            try:
                content = ut.from_json(content)
            except ValueError:
                raise Exception('Expected JSON string but got content=%r' % (content,))
            else:
                # logger.info('content = %r' % (content,))
                if content['status']['code'] != 200:
                    logger.info(content['status']['message'])
                    raise Exception(content['status']['message'])
            content = content['response']
        return content

    def wait_for_results(jobid, timeout=None, delays=[1, 3, 10]):
        """
        Waits for results from an engine
        """
        for _ in ut.delayed_retry_gen(delays):
            logger.info('Waiting for jobid = %s' % (jobid,))
            status_response = web_ibs.send_wbia_request(
                '/api/engine/job/status/', jobid=jobid
            )
            if status_response['jobstatus'] in ('completed', 'exception'):
                break
        return status_response

    def read_engine_results(jobid):
        result_response = web_ibs.send_wbia_request(
            '/api/engine/job/result/', jobid=jobid
        )
        return result_response

    def send_request_and_wait(suffix, type_='post', timeout=None, **kwargs):
        jobid = web_ibs.send_wbia_request(suffix, type_=type_, **kwargs)
        status_response = web_ibs.wait_for_results(jobid, timeout)  # NOQA
        result_response = web_ibs.read_engine_results(jobid)
        # >>> cmdict = ut.from_json(result_response['json_result'])[0]
        return result_response

    web_ibs.send_wbia_request = send_wbia_request
    web_ibs.wait_for_results = wait_for_results
    web_ibs.read_engine_results = read_engine_results
    web_ibs.send_request_and_wait = send_request_and_wait
    web_ibs.get = get
    web_ibs.post = post

    def wait_until_started():
        """ waits until the web server responds to a request """
        import requests

        for count in ut.delayed_retry_gen([1], timeout=15):
            if True or ut.VERBOSE:
                logger.info('Waiting for server to be up. count=%r' % (count,))
            try:
                web_ibs.send_wbia_request('/api/test/heartbeat/', type_='get')
                break
            except requests.ConnectionError:
                pass

    wait_until_started()

    @contextmanager
    def managed_server():
        try:
            yield web_ibs
        finally:
            web_ibs.terminate2()

    if managed:
        return managed_server()
    return web_ibs


def opendb(
    db=None,
    dbdir=None,
    defaultdb='cache',
    allow_newdir=False,
    delete_ibsdir=False,
    verbose=False,
    use_cache=True,
    web=None,
    **kwargs,
):
    """
    main without the preload (except for option to delete database before
    opening)

    Args:
        db (str):  database name in your workdir used only if dbdir is None
        dbdir (None): full database path
        defaultdb (str): dbdir search stratagy when db is None and dbdir is
            None
        allow_newdir (bool): (default=True) if True errors when opening a
            nonexisting database
        delete_ibsdir (bool): BE CAREFUL! (default=False) if True deletes the
            entire
        verbose (bool): verbosity flag
        web (bool): starts webserver if True (default=param specification)
        use_cache (bool): if True will try to return a previously loaded
            controller

    Returns:
        wbia.IBEISController: ibs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.entry_points import *  # NOQA
        >>> db = None
        >>> dbdir = None
        >>> defaultdb = 'cache'
        >>> allow_newdir = False
        >>> delete_ibsdir = False
        >>> verbose = False
        >>> use_cache = True
        >>> ibs = opendb(db, dbdir, defaultdb, allow_newdir, delete_ibsdir,
        >>>              verbose, use_cache)
        >>> result = str(ibs)
        >>> print(result)
    """
    from wbia.init import sysres
    from wbia.other import ibsfuncs

    dbdir = sysres.get_args_dbdir(
        defaultdb=defaultdb, allow_newdir=allow_newdir, db=db, dbdir=dbdir
    )
    if delete_ibsdir is True:
        assert (
            allow_newdir
        ), 'must be making new directory if you are deleting everything!'
        ibsfuncs.delete_wbia_database(dbdir)
    ibs = _init_wbia(dbdir, verbose=verbose, use_cache=use_cache, web=web, **kwargs)
    return ibs


def start(*args, **kwargs):
    """ alias for main() """  # + main.__doc__
    return main(*args, **kwargs)


def opendb_test(gui=True, dbdir=None, defaultdb='cache', allow_newdir=False, db=None):
    """ alias for main() """  # + main.__doc__
    from wbia.init import sysres

    _preload()
    dbdir = sysres.get_args_dbdir(
        defaultdb=defaultdb, allow_newdir=allow_newdir, db=db, dbdir=dbdir
    )
    ibs = _init_wbia(dbdir)
    return ibs


def _preload(mpl=True, par=True, logging=True):
    """ Sets up python environment """
    import utool as ut

    # from wbia.init import main_helpers
    # params.parse_args()
    # from wbia.init import sysres
    if multiprocessing.current_process().name != 'MainProcess':
        return
    if ut.VERBOSE:
        logger.info('[wbia] _preload')
    params.parse_args()
    # mpl backends
    # if logging and not params.args.nologging:
    #     if params.args.logdir is not None:
    #         sysres.set_logdir(params.args.logdir)
    #     else:
    #         # Log in the configured wbia log dir (which is maintained by utool)
    #         # fix this to be easier to figure out where the logs actually are
    #         ut.start_logging(appname='wbia')
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
    ut.util_inject.inject_colored_exceptions()
    # register type aliases for debugging
    # main_helpers.register_utool_aliases()
    # return params.args


def main_loop(main_locals, rungui=True, ipy=False, persist=True):
    """
    Runs the qt loop if the GUI was initialized and returns an executable string
    for embedding an IPython terminal if requested.

    If rungui is False the gui will not loop even if back has been created

    the main locals dict must be callsed main_locals in the scope you call this
    function in.

    Args:
        main_locals (dict_):
        rungui      (bool):
        ipy         (bool):
        persist     (bool):

    Returns:
        str: execstr
    """
    logger.info('[main] wbia.entry_points.main_loop()')
    params.parse_args()
    import utool as ut

    # logger.info('current process = %r' % (multiprocessing.current_process().name,))
    # == 'MainProcess':
    if rungui and not params.args.nogui:
        try:
            _guitool_loop(main_locals, ipy=ipy)
        except Exception as ex:
            ut.printex(ex, 'error in main_loop')
            raise
    # if not persist or params.args.cmd:
    #    main_close()
    # Put locals in the exec namespace
    ipycmd_execstr = ut.ipython_execstr()
    locals_execstr = ut.execstr_dict(main_locals, 'main_locals')
    execstr = locals_execstr + '\n' + ipycmd_execstr
    return execstr


def main_close(main_locals=None):
    _reset_signals()
