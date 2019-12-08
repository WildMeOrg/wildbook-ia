# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from flask import request, make_response, current_app
from ibeis.control import controller_inject
from ibeis.web import appfuncs as appf
import utool as ut

register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/ajax/cookie/', methods=['GET'])
def set_cookie(**kwargs):
    response = make_response('true')
    response.set_cookie(request.args['name'], request.args['value'])
    print('[web] Set Cookie: %r -> %r' % (request.args['name'], request.args['value'], ))
    return response


def _resize_src(image, resize=False, **kwargs):
    # Load image
    if resize is None:
        image_src = appf.embed_image_html(image, target_width=None, target_height=None)
    elif resize:
        image = appf.resize_via_web_parameters(image)
        image_src = appf.embed_image_html(image, target_width=None, target_height=None)
    else:
        image_src = appf.embed_image_html(image)

    return image_src


@register_route('/ajax/image/src/<gid>/', methods=['GET'])
def image_src(gid=None, thumbnail=False, ibs=None, **kwargs):
    if ibs is None:
        ibs = current_app.ibs

    gid = int(gid)
    image = None

    if 'thumbsize' not in kwargs:
        kwargs['thumbsize'] = max(
            int(appf.TARGET_WIDTH),
            int(appf.TARGET_HEIGHT),
        )

    if 'draw_annots' not in kwargs:
        kwargs['draw_annots'] = False

    if thumbnail:
        try:
            image = ibs.get_image_thumbnail(gid, **kwargs)
            h, w = image.shape[:2]
            assert h > 0, 'Invalid image thumbnail'
            assert w > 0, 'Invalid image thumbnail'
        except AssertionError:
            image = None

    if image is None:
        image = ibs.get_images(gid)

    image_src = _resize_src(image, **kwargs)
    return image_src


@register_route('/ajax/annot/src/<aid>/', methods=['GET'])
def annotation_src(aid=None, ibs=None, **kwargs):
    if ibs is None:
        ibs = current_app.ibs

    if 'dim_size' not in kwargs:
        kwargs['dim_size'] = max(
            int(appf.TARGET_WIDTH),
            int(appf.TARGET_HEIGHT),
        )
    image = ibs.get_annot_chips(aid, config2_=kwargs)

    # image_src = _resize_src(image, **kwargs)
    image_src = appf.embed_image_html(image, target_height=300)
    return image_src


@register_route('/ajax/part/src/<part_rowid>/', methods=['GET'])
def part_src(part_rowid, **kwargs):
    ibs = current_app.ibs
    if 'dim_size' not in kwargs:
        kwargs['dim_size'] = max(
            int(appf.TARGET_WIDTH),
            int(appf.TARGET_HEIGHT),
        )
    image = ibs.get_part_chips(part_rowid, config2_=kwargs)
    image_src = appf.embed_image_html(image, target_height=300)
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
