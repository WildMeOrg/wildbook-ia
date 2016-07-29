# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from flask import request, make_response, current_app
from ibeis.control import controller_inject
from ibeis.web import appfuncs as appf
import utool as ut
import vtool as vt

register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/ajax/cookie/', methods=['GET'])
def set_cookie():
    response = make_response('true')
    response.set_cookie(request.args['name'], request.args['value'])
    print('[web] Set Cookie: %r -> %r' % (request.args['name'], request.args['value'], ))
    return response


@register_route('/ajax/image/src/<gid>/', methods=['GET'])
def image_src(gid=None, thumbnail=False, fresh=False, **kwargs):
    thumbnail = thumbnail or 'thumbnail' in request.args or 'thumbnail' in request.form
    ibs = current_app.ibs
    gid = int(gid)
    if thumbnail:
        gpath = ibs.get_image_thumbpath(gid, ensure_paths=True)
        fresh = fresh or 'fresh' in request.args or 'fresh' in request.form
        if fresh:
            import os
            os.remove(gpath)
            gpath = ibs.get_image_thumbpath(gid, ensure_paths=True)
    else:
        gpath = ibs.get_image_paths(gid)

    # Load image
    image = vt.imread(gpath, orient='auto')
    image = appf.resize_via_web_parameters(image)
    return appf.embed_image_html(image, target_width=None)


@register_route('/ajax/annot/src/<aid>/', methods=['GET'])
def annotation_src(aid=None):
    ibs = current_app.ibs
    gpath = ibs.get_annot_chip_fpath(aid)
    image = vt.imread(gpath)
    image_src = appf.embed_image_html(image, target_width=None, target_height=300)
    return image_src


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
