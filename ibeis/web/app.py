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
from ibeis.web import job_engine
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
        # Start the tornado web handler
        # WSGI = Web Server Gateway Interface
        # WSGI is Python standard described in detail in PEP 3333
        http_server = tornado.httpserver.HTTPServer(
            tornado.wsgi.WSGIContainer(app))
        try:
            http_server.listen(app.server_port)
        except socket.error:
            while not ut.is_local_port_open(app.server_port):
                app.server_port += 1
            args = (app.server_port, )
            message = 'The port specified for the IBEIS web interface is ' + \
                      'not available. (Hint: port %d is available)' % args
            raise RuntimeError(message)
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
        #import time
        #time.sleep(1)
        # No need to sleep, this call should block until engine is live.
        ibs.initialize_job_manager()
        #time.sleep(10)

    print('[web] starting tornado')
    try:
        start_tornado(ibs, port, browser, url_suffix)
    except KeyboardInterrupt:
        print('Caught ctrl+c in webserver. Gracefully exiting')
    print('[web] closing job manager')
    ibs.close_job_manager()


def start_web_annot_groupreview(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids

    CommandLine:
        python -m ibeis.tag_funcs --exec-start_web_annot_groupreview --db PZ_Master1
        python -m ibeis.tag_funcs --exec-start_web_annot_groupreview --db GZ_Master1
        python -m ibeis.tag_funcs --exec-start_web_annot_groupreview --db GIRM_Master1

    Example:
        >>> # SCRIPT
        >>> from ibeis.tag_funcs import *  # NOQA
        >>> import ibeis
        >>> #ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> ibs = ibeis.opendb(defaultdb='GZ_Master1')
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
    import ibeis.web
    aid_strs = ','.join(list(map(str, aid_list)))
    url_suffix = '/group_review/?aid_list=%s' % (aid_strs)
    ibeis.web.app.start_from_ibeis(ibs, url_suffix=url_suffix, browser=True)


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
