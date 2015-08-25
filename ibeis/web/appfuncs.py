# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from PIL import Image
import numpy as np
import cStringIO as StringIO
import flask
from os.path import join, dirname, abspath  # NOQA
from datetime import date
import base64
import jinja2
import utool as ut


TARGET_WIDTH = 1200.0


class NavbarClass(object):
    def __init__(nav):
        nav.item_list = [
            ('root', 'Home'),
            ('view', 'View'),
            ('turk', 'Turk'),
            ('api',  'API'),
            ('group_review',  'Group Review'),
        ]

    def __iter__(nav):
        _link = flask.request.path.strip('/').split('/')
        for link, nice in nav.item_list:
            yield link == _link[0], link, nice


ORIENTATIONS = {   # used in apply_orientation
    2: (Image.FLIP_LEFT_RIGHT,),
    3: (Image.ROTATE_180,),
    4: (Image.FLIP_TOP_BOTTOM,),
    5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
    6: (Image.ROTATE_270,),
    7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
    8: (Image.ROTATE_90,)
}


def open_oriented_image(im_path):
    im = Image.open(im_path)
    if hasattr(im, '_getexif'):
        exif = im._getexif()
        if exif is not None and 274 in exif:
            orientation = exif[274]
            im = apply_orientation(im, orientation)
    img = np.asarray(im).astype(np.float32) / 255.
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def apply_orientation(im, orientation):
    """
    This script handles the skimage exif problem.
    """
    if orientation in ORIENTATIONS:
        for method in ORIENTATIONS[orientation]:
            im = im.transpose(method)
    return im


def embed_image_html(image, filter_width=True):
    """ Creates an image embedded in HTML base64 format. """
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    width, height = image_pil.size
    if filter_width:
        _height = int(TARGET_WIDTH / 2)
        _width = int((float(_height) / height) * width)
    else:
        _width = int(TARGET_WIDTH)
        _height = int((float(_width) / width) * height)
    image_pil = image_pil.resize((_width, _height))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='jpeg')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/jpeg;base64,' + data


def return_src(gpath):
    image = open_oriented_image(gpath)
    image_src = embed_image_html(image, filter_width=True)
    return image_src


def check_valid_function_name(string):
    return all([ char.isalpha() or char == '_' or char.isalnum() for char in string])


def encode_refer_url(url):
    return base64.urlsafe_b64encode(str(url))


def decode_refer_url(encode):
    return base64.urlsafe_b64decode(str(encode))


def template(template_directory=None, template_filename=None, **kwargs):
    global_args = {
        'NAVBAR': NavbarClass(),
        'YEAR':   date.today().year,
        'URL':    flask.request.url,
        'REFER_SRC_STR':  flask.request.url.replace(flask.request.url_root, ''),
    }
    global_args['REFER_SRC_ENCODED'] = encode_refer_url(global_args['REFER_SRC_STR'])
    if 'refer' in flask.request.args.keys():
        refer = flask.request.args['refer']
        print('[web] REFER: %r' % (refer, ))
        global_args['REFER_DST_ENCODED'] = refer
        global_args['REFER_DST_STR'] = decode_refer_url(refer)
    if template_directory is None:
        template_directory = ''
        #template_directory = abspath(join(dirname(__file__), 'templates'))
        #template_directory = join(dirname(dirname(__file__)))
    if template_filename is None:
        template_filename = 'index'
    template_ = join(template_directory, template_filename + '.html')
    # Update global args with the template's args
    _global_args = dict(global_args)
    _global_args.update(kwargs)
    print('[appfuncs] template()')
    from ibeis.control import controller_inject
    app = controller_inject.get_flask_app()
    # flask hates windows apparently
    template_ = template_.replace('\\', '/')
    print('[appfuncs.template] * app.template_folder = %r' % (app.template_folder,))
    print('[appfuncs.template] * template_directory = %r' % (template_directory,))
    print('[appfuncs.template] * template_filename = %r' % (template_filename,))
    print('[appfuncs.template] * template_ = %r' % (template_,))
    try:
        ret = flask.render_template(template_, **_global_args)
        #ret = flask.render_template(full_template_fpath, **_global_args)
    except jinja2.exceptions.TemplateNotFound as ex:
        print('Error template not found')
        full_template_fpath = join(app.template_folder, template_)
        print('[appfuncs.template] * full_template_fpath = %r' % (full_template_fpath,))
        ut.checkpath(full_template_fpath, verbose=True)
        ut.printex(ex, 'Template error in appfuncs', tb=True)
        raise
    except Exception as ex:
        ut.printex(ex, 'Error in appfuncs', tb=True)
        raise
    return ret


def send_file(string, filename):
    response = flask.make_response(str(string))
    response.headers['Content-Description'] = 'File Transfer'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=%s' % filename
    response.headers['Content-Length'] = len(string)
    return response
