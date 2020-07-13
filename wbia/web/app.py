# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
import tornado.wsgi
import tornado.httpserver
import logging
import socket
from wbia.control import controller_inject
from wbia.web import apis_engine
from wbia.web import job_engine
from wbia.web import appfuncs as appf
from tornado.log import access_log
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)

try:
    try:
        from werkzeug.wsgi import DispatcherMiddleware
    except Exception:
        from werkzeug.middleware.dispatcher import DispatcherMiddleware
    import prometheus_client
    from wbia.web import prometheus  # NOQA

    PROMETHEUS = True
except ImportError:
    PROMETHEUS = False


def tst_html_error():
    r"""
    This test will show what our current errors look like

    CommandLine:
        python -m wbia.web.app --exec-tst_html_error

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> web_ibs = wbia.opendb_bg_web(browser=True, start_job_queue=False, url_suffix='/api/image/imagesettext/?__format__=True')
    """
    pass


class TimedWSGIContainer(tornado.wsgi.WSGIContainer):
    def _log(self, status_code, request):
        if status_code < 400:
            log_method = access_log.info
        elif status_code < 500:
            log_method = access_log.warning
        else:
            log_method = access_log.error

        timestamp = ut.timestamp()
        request_time = 1000.0 * request.request_time()

        quiet_list = [
            '/api/test/heartbeat',
            '/api/test/heartbeat/',
            '/metrics',
            '/metrics/',
        ]
        if status_code == 200 and request.uri in quiet_list:
            return

        log_method(
            'WALL=%s STATUS=%s METHOD=%s URL=%s IP=%s TIME=%.2fms',
            timestamp,
            status_code,
            request.method,
            request.uri,
            request.remote_ip,
            request_time,
        )


def start_tornado(
    ibs, port=None, browser=None, url_suffix=None, start_web_loop=True, fallback=True
):
    """Initialize the web server"""
    if browser is None:
        browser = ut.get_argflag('--browser')
    if url_suffix is None:
        url_suffix = ut.get_argval('--url', default='')

    # from wbia import constants as const
    # ibs.https = const.HTTPS

    def _start_tornado(ibs_, port_):
        # Get Flask app
        app = controller_inject.get_flask_app()

        app.ibs = ibs_
        # Try to ascertain the socket's domain name
        socket.setdefaulttimeout(0.1)
        try:
            app.server_domain = socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            app.server_domain = '127.0.0.1'
        socket.setdefaulttimeout(None)
        app.server_port = port_
        # URL for the web instance
        app.server_url = 'http://%s:%s' % (app.server_domain, app.server_port)
        print('[web] Tornado server starting at %s' % (app.server_url,))
        # Launch the web browser to view the web interface and API
        if browser:
            url = app.server_url + url_suffix
            import webbrowser

            print('[web] opening browser with url = %r' % (url,))
            webbrowser.open(url)

        if PROMETHEUS:
            # Add prometheus wsgi middleware to route /metrics requests
            print('LOADING PROMETHEUS')
            app_ = DispatcherMiddleware(
                app, {'/metrics': prometheus_client.make_wsgi_app()}
            )
            # Migrate the most essential settings
            app_.server_port = app.server_port
            app_.server_url = app.server_url
            app_.ibs = app.ibs
            app = app_
        else:
            print('SKIPPING PROMETHEUS')

        # Start the tornado web handler
        # WSGI = Web Server Gateway Interface
        # WSGI is Python standard described in detail in PEP 3333
        wsgi_container = TimedWSGIContainer(app)

        # # Try wrapping with newrelic performance monitoring
        # try:
        #     import newrelic
        #     wsgi_container = newrelic.agent.WSGIApplicationWrapper(wsgi_container)
        # except (ImportError, AttributeError):
        #     pass

        http_server = tornado.httpserver.HTTPServer(wsgi_container)

        try:
            http_server.listen(app.server_port)
        except socket.error:
            fallback_port = ut.find_open_port(app.server_port)
            if fallback:
                print(
                    'Port %s is unavailable, using fallback_port = %r'
                    % (port, fallback_port,)
                )
                start_tornado(
                    ibs,
                    port=fallback_port,
                    browser=browser,
                    url_suffix=url_suffix,
                    start_web_loop=start_web_loop,
                    fallback=False,
                )
            else:
                raise RuntimeError(
                    (
                        ('The specified IBEIS web port %d is not available, ' 'but %d is')
                        % (app.server_port, fallback_port)
                    )
                )

        # Add more verbose logging
        utool_logfile_handler = ut.util_logging.__CURRENT_LOGFILE_HANDLER__
        if utool_logfile_handler is not None:
            logger_list = []
            try:
                logger_list += [
                    app.logger,
                ]
            except AttributeError:
                pass
            try:
                logger_list += [
                    app.app.logger,
                ]
            except AttributeError:
                pass
            logger_list += [
                logging.getLogger('concurrent'),
                logging.getLogger('concurrent.futures'),
                logging.getLogger('flask_cors.core'),
                logging.getLogger('flask_cors'),
                logging.getLogger('flask_cors.decorator'),
                logging.getLogger('flask_cors.extension'),
                logging.getLogger('urllib3'),
                logging.getLogger('requests'),
                logging.getLogger('tornado'),
                logging.getLogger('tornado.access'),
                logging.getLogger('tornado.application'),
                logging.getLogger('tornado.general'),
                logging.getLogger('websocket'),
            ]
            for logger in logger_list:
                logger.setLevel(logging.INFO)
                logger.addHandler(utool_logfile_handler)

        if start_web_loop:
            tornado.ioloop.IOLoop.instance().start()

    # Get the port if unspecified
    if port is None:
        port = appf.DEFAULT_WEB_API_PORT
    # Launch the web handler
    _start_tornado(ibs, port)


def start_from_wbia(
    ibs,
    port=None,
    browser=None,
    precache=None,
    url_suffix=None,
    start_job_queue=None,
    start_web_loop=True,
):
    """
    Parse command line options and start the server.

    CommandLine:
        python -m wbia --db PZ_MTEST --web
        python -m wbia --db PZ_MTEST --web --browser
    """
    print('[web] start_from_wbia()')

    if start_job_queue is None:
        if ut.get_argflag('--noengine'):
            start_job_queue = False
        else:
            start_job_queue = True

    if precache is None:
        precache = ut.get_argflag('--precache')

    if precache:
        gid_list = ibs.get_valid_gids()
        print('[web] Pre-computing all image thumbnails (with annots)...')
        ibs.get_image_thumbpath(gid_list, draw_annots=True)
        print('[web] Pre-computing all image thumbnails (without annots)...')
        ibs.get_image_thumbpath(gid_list, draw_annots=False)
        print('[web] Pre-computing all annotation chips...')
        ibs.check_chip_existence()
        ibs.compute_all_chips()

    if start_job_queue:
        print('[web] opening job manager')
        ibs.load_plugin_module(job_engine)
        ibs.load_plugin_module(apis_engine)
        # import time
        # time.sleep(1)
        # No need to sleep, this call should block until engine is live.
        ibs.initialize_job_manager()
        # time.sleep(10)

    print('[web] starting tornado')
    try:
        start_tornado(ibs, port, browser, url_suffix, start_web_loop)
    except KeyboardInterrupt:
        print('Caught ctrl+c in webserver. Gracefully exiting')
    if start_web_loop:
        print('[web] closing job manager')
        ibs.close_job_manager()


def start_web_annot_groupreview(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):  list of annotation rowids

    CommandLine:
        python -m wbia.tag_funcs --exec-start_web_annot_groupreview --db PZ_Master1
        python -m wbia.tag_funcs --exec-start_web_annot_groupreview --db GZ_Master1
        python -m wbia.tag_funcs --exec-start_web_annot_groupreview --db GIRM_Master1

    Example:
        >>> # SCRIPT
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> #ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> ibs = wbia.opendb(defaultdb='GZ_Master1')
        >>> #aid_list = ibs.get_valid_aids()
        >>> # -----
        >>> any_tags = ut.get_argval('--tags', type_=list, default=['Viewpoint'])
        >>> min_num = ut.get_argval('--min_num', type_=int, default=1)
        >>> prop = any_tags[0]
        >>> filtered_annotmatch_rowids = filter_annotmatch_by_tags(ibs, None, any_tags=any_tags, min_num=min_num)
        >>> aid1_list = (ibs.get_annotmatch_aid1(filtered_annotmatch_rowids))
        >>> aid2_list = (ibs.get_annotmatch_aid2(filtered_annotmatch_rowids))
        >>> aid_list = list(set(ut.flatten([aid2_list, aid1_list])))
        >>> result = start_web_annot_groupreview(ibs, aid_list)
        >>> print(result)
    """
    import wbia.web

    aid_strs = ','.join(list(map(str, aid_list)))
    url_suffix = '/group_review/?aid_list=%s' % (aid_strs)
    wbia.web.app.start_from_wbia(ibs, url_suffix=url_suffix, browser=True)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.app
        python -m wbia.web.app --allexamples
        python -m wbia.web.app --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
