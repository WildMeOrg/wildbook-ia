# -*- coding: utf-8 -*-
"""
A example for creating a Table that is sortable by its header
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import flask_table
import six

# from flask_table import Table, Col, LinkCol
import tornado.wsgi
import tornado.httpserver
import socket
import flask
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


task_data = {
    0: {
        'index': 0,
        'dbname': 'testdb1',
        'task': 'match_state',
        'mcc': 0.99,
        'auc_ovr': [1, 1, 1],
    },
    1: {
        'index': 1,
        'dbname': 'testdb1',
        'task': 'photobomb_state',
        'mcc': 0.12,
        'auc_ovr': [1, 1, 3],
    },
    2: {
        'index': 2,
        'dbname': 'testdb2',
        'task': 'photobomb_state',
        'mcc': -3.0,
        'auc_ovr': [2, 2, 2],
    },
}

app = flask.Flask(__name__)
app.task_data = task_data


def ensure_task_table():
    if not hasattr(app, 'DBTaskTable'):

        class DBTaskTable(flask_table.Table):
            allow_sort = True

            def sort_url(self, col_key, reverse=False):
                if reverse:
                    direction = 'desc'
                else:
                    direction = 'asc'
                return flask.url_for(
                    ut.get_funcname(index), sort=col_key, direction=direction
                )

        col_nice_lookup = {}
        columns = [
            'index',
            'dbname',
            ('task', task_link),
            # ('link', task_link)
        ]
        for tup in columns:
            if isinstance(tup, tuple):
                colname, link = tup
                colnice = col_nice_lookup.get(colname, colname)
                url_kwargs = {a: a for a in ut.get_func_argspec(link).args}
                endpoint = ut.get_funcname(link)
                link_kw = dict(
                    name=colnice,
                    attr=colname,
                    endpoint=endpoint,
                    url_kwargs=url_kwargs,
                    allow_sort=True,
                    show=True,
                )
                new_col = flask_table.LinkCol(**link_kw)
            elif isinstance(tup, six.string_types):
                colname = tup
                colnice = col_nice_lookup.get(colname, colname)
                new_col = flask_table.Col(
                    name=colnice, attr=colname, allow_sort=True, show=True
                )
            else:
                assert False, 'unkonown tup'
            DBTaskTable.add_column(colname, new_col)
        app.DBTaskTable = DBTaskTable
    return app.DBTaskTable


@app.route('/')
def index():
    DBTaskTable = ensure_task_table()

    # DBTaskTable._cols['index'].show = not DBTaskTable._cols['index'].show

    sort = flask.request.args.get('sort', 'index')
    reverse = flask.request.args.get('direction', 'asc') == 'desc'

    # print('task_data =\n%s' % (ut.repr4(task_data),))
    sorted_data = sorted(task_data.values(), key=lambda x: x[sort], reverse=reverse)
    # print('sorted_data =\n%s' % (ut.repr4(sorted_data),))
    table = DBTaskTable(sorted_data, sort_by=sort, sort_reverse=reverse)
    html = table.__html__()
    return html


@app.route('/item/<int:index>')
def task_link(index):
    item = task_data[index]
    return ut.codeblock(
        """
        <h1>Task</h1>
        <p>dbname={dbname}</p>
        <p>task={task}</p>
        <p>mcc={mcc}</p>
        <p>auc_ovr={auc_ovr}</p>
        <hr><small>index: {index}</small>
        """
    ).format(**item)


def run_clf_server():
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/wbia/wbia/scripts/clfweb
        python -m clfserve run_clf_server

    Example:
        >>> # DISABLE_DOCTEST
        >>> from clfserve import *  # NOQA
        >>> run_clf_server()
    """
    socket.setdefaulttimeout(0.1)
    try:
        app.server_domain = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        app.server_domain = '127.0.0.1'
    port = 5555
    app.server_port = port
    app.server_url = 'http://%s:%s' % (app.server_domain, app.server_port)
    browser = True
    if browser:
        import webbrowser

        webbrowser.open(app.server_url)
    print('Tornado server starting at %s' % (app.server_url,))
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    try:
        http_server.listen(app.server_port)
    except socket.error:
        fallback_port = ut.find_open_port(app.server_port)
        raise RuntimeError(
            (
                'The specified port %d is not available, but %d is'
                % (app.server_port, fallback_port)
            )
        )
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/wbia/wbia/scripts/clfweb
        python ~/code/wbia/wbia/scripts/clfweb/clfserve.py
        python ~/code/wbia/wbia/scripts/clfweb/clfserve.py --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
