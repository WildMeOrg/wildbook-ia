# Dependencies: flask, tornado
from __future__ import absolute_import, division, print_function
# HTTP / HTML
import tornado.wsgi
import tornado.httpserver
import flask
from flask import request, redirect, url_for
import optparse
import logging
import socket
import simplejson as json
# IBEIS
import ibeis
from ibeis.control.SQLDatabaseControl import (SQLDatabaseController,  # NOQA
                                              SQLAtomicContext)
from ibeis.control import _sql_helpers
import utool
# Web Internal
from ibeis.web import appfuncs, navbar, DBWEB_SCHEMA
# Others
from datetime import date
from os.path import join
import random


DEFAULT_PORT = 5000
app = flask.Flask(__name__)
global_args = {
    'NAVBAR': navbar.NavbarClass(),
    'YEAR':   date.today().year,
}


@app.route('/')
@app.route('/<filename>.html')
def root(filename=''):
    return template('', filename)


@app.route('/turk')
@app.route('/turk/<filename>.html')
def turk(filename=''):
    aid_list = app.ibeis.get_valid_aids()
    gpath_list = app.ibeis.get_annot_gpaths(aid_list)
    index = random.randint(0, len(aid_list) - 1)
    aid = aid_list[index]
    gpath = gpath_list[index]
    image = appfuncs.open_oriented_image(gpath)
    return template('turk', filename,
                    aid=aid,
                    image_path=gpath,
                    image_src=appfuncs.embed_image_html(image))


@app.route('/submit/turk.html', methods=['POST'])
def submit():
    print(request)
    print(request.form)
    if request.form['viewpoint-submit'].lower() == "accept":
        aid = int(request.form['viewpoint-aid'])
        value = int(request.form['viewpoint-value'])
        print("aid: %r, viewpoint: %r" % (aid, value, ))
        app.ibeis.set_annot_viewpoint([aid], [value], convert_radians=True)
    return redirect(url_for('turk', filename='viewpoint'))


@app.route('/api')
@app.route('/api/<function>.json', methods=['GET', 'POST'])
def api(function=None):
    template = {
        'status': {
            'success': False,
            'code': '',
        },
    }
    print("Function:", function)
    print("POST:", dict(request.form))
    print("GET:",  dict(request.args))
    if function is None:
        template['status']['success'] = True
        template['status']['code'] = 'USAGE: /api/[ibeis_function_name].json'
    else:
        function = function.lower()
        if appfuncs.check_valid_function_name(function):
            function = 'app.ibeis.%s' % function
            exists = True
            try:
                func = eval(function)
                ret = func()
            except AttributeError:
                exists = False
            if exists:
                template['status']['success'] = True
                template['function'] = function
                template['return'] = ret
            else:
                template['status']['success'] = False
                template['status']['code'] = 'ERROR: Specified IBEIS function not visible or implemented'
        else:
            template['status']['success'] = False
            template['status']['code'] = 'ERROR: Specified IBEIS function not valid Python function'
    return json.dumps(template)


################################################################################


def template(template_directory='', template_filename='', **kwargs):
    if len(template_filename) == 0:
        template_filename = 'index'
    template_ = join(template_directory, template_filename + '.html')
    # Update global args with the template's args
    _global_args = dict(global_args)
    _global_args.update(kwargs)
    print(template_)
    return flask.render_template(template_, **_global_args)


################################################################################


def start_tornado(app, port=5000, browser=False, blocking=False, reset_db=True, database_init=None):
    def _start_tornado():
        http_server = tornado.httpserver.HTTPServer(
            tornado.wsgi.WSGIContainer(app))
        http_server.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    # Open the web internal database
    database_dir = utool.get_app_resource_dir('ibeis', 'web')
    database_filename = 'app.sqlite3'
    database_filepath = join(database_dir, database_filename)
    utool.ensuredir(database_dir)
    print(database_filepath)
    if reset_db:
        utool.remove_file(database_filepath)
    app.dbweb_version_expected = "1.0.0"
    app.db = SQLDatabaseController(database_dir, database_filename)
    _sql_helpers.ensure_correct_version(
        app.ibeis,
        app.db,
        app.dbweb_version_expected,
        DBWEB_SCHEMA.VALID_VERSIONS
    )
    # Initialize the web server
    logging.getLogger().setLevel(logging.INFO)
    server_ip_address = socket.gethostbyname(socket.gethostname())
    url = 'http://%s:%s' % (server_ip_address, port)
    print("Tornado server starting at %s" % (url,))
    if browser:
        import webbrowser
        webbrowser.open(url)
    if database_init is not None:
        database_init(app)
    # Blocking
    _start_tornado()
    # if blocking:
    #     _start_tornado()
    # else:
    #     import threading
    #     threading.Thread(target=_start_tornado).start()


def start_from_terminal():
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=DEFAULT_PORT)
    parser.add_option(
        '--db',
        help="specify an IBEIS database",
        type='str', default='testdb0')

    opts, args = parser.parse_args()
    app.ibeis = ibeis.opendb(db=opts.db)
    start_tornado(app, opts.port, database_init=appfuncs.database_init)


def start_from_ibeis(ibeis, port=DEFAULT_PORT):
    """
    Parse command line options and start the server.
    """
    app.ibeis = ibeis
    start_tornado(app, port, database_init=appfuncs.database_init)


if __name__ == '__main__':
    start_from_terminal()
