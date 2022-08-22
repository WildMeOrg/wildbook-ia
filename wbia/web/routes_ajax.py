# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
import logging

import utool as ut
import vtool as vt
from flask import current_app, make_response, request

from wbia.control import controller_inject
from wbia.web import appfuncs as appf

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

register_route = controller_inject.get_wbia_flask_route(__name__)


@register_route('/ajax/cookie/', methods=['GET'])
def set_cookie(**kwargs):
    response = make_response('true')
    response.set_cookie(request.args['name'], request.args['value'])
    logger.info(
        '[web] Set Cookie: {!r} -> {!r}'.format(
            request.args['name'], request.args['value']
        )
    )
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


@register_route(
    '/ajax/image/src/<gid>.jpg', methods=['GET'], __route_postfix_check__=False
)
def image_src_ext(*args, **kwargs):
    return image_src(*args, **kwargs)


@register_route('/ajax/image/src/<gid>/', methods=['GET'])
def image_src(gid=None, thumbnail=False, ibs=None, **kwargs):
    if ibs is None:
        ibs = current_app.ibs

    gid = int(gid)
    gpath = None

    if 'thumbsize' not in kwargs:
        kwargs['thumbsize'] = max(int(appf.TARGET_WIDTH), int(appf.TARGET_HEIGHT))

    if 'draw_annots' not in kwargs:
        kwargs['draw_annots'] = False

    if thumbnail:
        try:
            gpath = ibs.get_image_thumbpath(gid, ensure_paths=True, **kwargs)
            orient = ibs.get_image_orientation(gid)
            image = vt.imread(gpath, orient=orient)
            h, w = image.shape[:2]
            assert h > 0, 'Invalid image thumbnail'
            assert w > 0, 'Invalid image thumbnail'
        except AssertionError:
            gpath = None

    if gpath is None:
        gpath = ibs.get_image_paths(gid)

    image_src = image_src_path(gpath, **kwargs)
    return image_src


def image_src_path(gpath, orient='auto', **kwargs):
    image = vt.imread(gpath, orient=orient)
    image_src = _resize_src(image, **kwargs)
    return image_src


@register_route('/ajax/annot/src/<aid>/', methods=['GET'])
def annotation_src(aid=None, ibs=None, **kwargs):
    if ibs is None:
        ibs = current_app.ibs

    if 'dim_size' not in kwargs:
        kwargs['dim_size'] = max(int(appf.TARGET_WIDTH), int(appf.TARGET_HEIGHT))
    image = ibs.get_annot_chips(aid, config2_=kwargs)

    # image_src = _resize_src(image, **kwargs)
    image_src = appf.embed_image_html(image, target_height=300)
    return image_src


@register_route('/ajax/background/src/<aid>/', methods=['GET'])
def probchip_src(aid=None, ibs=None, **kwargs):
    import cv2

    if ibs is None:
        ibs = current_app.ibs

    if 'dim_size' not in kwargs:
        kwargs['dim_size'] = max(int(appf.TARGET_WIDTH), int(appf.TARGET_HEIGHT))

    image_filepath = ibs.get_annot_probchip_fpath(aid, config2_=kwargs)
    image = cv2.imread(image_filepath)
    x, y, w, h = ibs.get_annot_bboxes(aid)

    image = cv2.resize(image, (w, h))
    # image_src = _resize_src(image, **kwargs)
    image_src = appf.embed_image_html(image, target_height=300)
    return image_src


@register_route('/ajax/part/src/<part_rowid>/', methods=['GET'])
def part_src(part_rowid, **kwargs):
    ibs = current_app.ibs
    if 'dim_size' not in kwargs:
        kwargs['dim_size'] = max(int(appf.TARGET_WIDTH), int(appf.TARGET_HEIGHT))
    image = ibs.get_part_chips(part_rowid, config2_=kwargs)
    image_src = appf.embed_image_html(image, target_height=300)
    return image_src
