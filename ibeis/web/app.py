# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
import tornado.wsgi
import tornado.httpserver
import logging
import socket
from ibeis.control import controller_inject
from ibeis.web import apis_engine
from ibeis.web import appfuncs as appf
import utool as ut


def test_html_error():
    r"""
    This test will show what our current errors look like

    CommandLine:
        python -m ibeis.web.app --exec-test_html_error

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.web.app import *  # NOQA
        >>> import ibeis
        >>> web_ibs = ibeis.opendb_bg_web(browser=True, start_job_queue=False, url_suffix='/api/image/imagesettext/?__format__=True')
    """
    pass


def start_tornado(ibs, port=None, browser=None, url_suffix=None):
    """
        Initialize the web server
    """
    if browser is None:
        browser = ut.get_argflag('--browser')
    if url_suffix is None:
        url_suffix = ''
    def _start_tornado(ibs_, port_):
        # Get Flask app
        app = controller_inject.get_flask_app()
        app.ibs = ibs_
        # Try to ascertain the socket's domain name
        try:
            app.server_domain = socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            app.server_domain = '127.0.0.1'
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
        # Start the tornado web handler
        # WSGI = Web Server Gateway Interface
        # WSGI is Python standard described in detail in PEP 3333
        http_server = tornado.httpserver.HTTPServer(
            tornado.wsgi.WSGIContainer(app))
        http_server.listen(app.server_port)
        tornado.ioloop.IOLoop.instance().start()

    # Set logging level
    logging.getLogger().setLevel(logging.INFO)
    # Get the port if unspecified
    if port is None:
        port = appf.DEFAULT_WEB_API_PORT
    # Launch the web handler
    _start_tornado(ibs, port)


def start_from_ibeis(ibs, port=None, browser=None, precache=None,
                     url_suffix=None, start_job_queue=True):
    """
    Parse command line options and start the server.

    CommandLine:
        python -m ibeis --db PZ_MTEST --web
        python -m ibeis --db PZ_MTEST --web --browser
    """
    print('[web] start_from_ibeis()')
    if precache is None:
        precache = ut.get_argflag('--precache')

    if precache:
        print('[web] Pre-computing all image thumbnails (with annots)...')
        ibs.preprocess_image_thumbs()
        print('[web] Pre-computing all image thumbnails (without annots)...')
        ibs.preprocess_image_thumbs(draw_annots=False)
        print('[web] Pre-computing all annotation chips...')
        ibs.check_chip_existence()
        ibs.compute_all_chips()

    if start_job_queue:
        print('[web] opening job manager')
        ibs.load_plugin_module(apis_engine)
        #import time
        #time.sleep(1)
        ibs.initialize_job_manager()
        #time.sleep(10)

    print('[web] starting tornado')
    start_tornado(ibs, port, browser, url_suffix)
    print('[web] closing job manager')
    ibs.close_job_manager()


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
