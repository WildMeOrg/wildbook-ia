# flask, tornado
import logging
import flask
import optparse
import tornado.wsgi
import tornado.httpserver
from datetime import date
from navbar import NavbarClass
import ibeis
from os.path import join
import random
import appfuncs

# Obtain the flask app object
app = flask.Flask(__name__)
global_args = {
    'NAVBAR': NavbarClass(),
    'YEAR':   date.today().year,
}


@app.route('/')
@app.route('/<filename>.html')
def root(filename=''):
    return template('', filename)


@app.route('/turk')
@app.route('/turk/<filename>.html')
def turk(filename=''):
    print app.ibeis
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


def template(template_directory='', template_filename='', **kwargs):
    if len(template_filename) == 0:
        template_filename = 'index'
    template_ = join(template_directory, template_filename + '.html')
    # Update global args with the template's args
    _global_args = dict(global_args)
    _global_args.update(kwargs)
    print template_
    return flask.render_template(template_, **_global_args)


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '--db',
        help="specify an IBEIS database",
        type='str', default='testdb0')

    opts, args = parser.parse_args()
    app.ibeis  = ibeis.opendb(db=opts.db)
    start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
