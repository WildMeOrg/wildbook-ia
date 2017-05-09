# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
import math
from flask import request, current_app
from ibeis.control import controller_inject
from ibeis.web import appfuncs as appf
import utool as ut


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/experiments/', methods=['GET'])
def view_experiments(**kwargs):
    return appf.template('experiments')


@register_route('/experiments/interest/', methods=['GET'])
def experiments_interest(**kwargs):
    return view_images(**kwargs)


def view_images(**kwargs):
    ibs = current_app.ibs
    filtered = True
    imgsetid_list = []
    gid = request.args.get('gid', '')
    imgsetid = request.args.get('imgsetid', '')
    page = max(0, int(request.args.get('page', 1)))
    if len(gid) > 0:
        gid_list = gid.strip().split(',')
        gid_list = [ None if gid_ == 'None' or gid_ == '' else int(gid_) for gid_ in gid_list ]
    elif len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [ None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_) for imgsetid_ in imgsetid_list ]
        gid_list = ut.flatten([ ibs.get_valid_gids(imgsetid=imgsetid) for imgsetid_ in imgsetid_list ])
    else:
        gid_list = ibs.get_valid_gids()
        filtered = False
    # Page
    page_start = min(len(gid_list), (page - 1) * appf.PAGE_SIZE)
    page_end   = min(len(gid_list), page * appf.PAGE_SIZE)
    page_total = int(math.ceil(len(gid_list) / appf.PAGE_SIZE))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(gid_list) else page + 1
    gid_list = gid_list[page_start:page_end]
    print('[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s' % (page_start, page_end, len(gid_list), page_previous, page_next, ))
    image_unixtime_list = ibs.get_image_unixtime(gid_list)
    datetime_list = [
        ut.unixtime_to_datetimestr(image_unixtime)
        if image_unixtime is not None
        else
        'Unknown'
        for image_unixtime in image_unixtime_list
    ]
    image_list = zip(
        gid_list,
        [ ','.join(map(str, imgsetid_list_)) for imgsetid_list_ in ibs.get_image_imgsetids(gid_list) ],
        ibs.get_image_gnames(gid_list),
        image_unixtime_list,
        datetime_list,
        ibs.get_image_gps(gid_list),
        ibs.get_image_party_tag(gid_list),
        ibs.get_image_contributor_tag(gid_list),
        ibs.get_image_notes(gid_list),
        appf.imageset_image_processed(ibs, gid_list),
    )
    image_list.sort(key=lambda t: t[3])
    return appf.template('experiments', 'interest',
                         filtered=filtered,
                         imgsetid_list=imgsetid_list,
                         imgsetid_list_str=','.join(map(str, imgsetid_list)),
                         num_imgsetids=len(imgsetid_list),
                         gid_list=gid_list,
                         gid_list_str=','.join(map(str, gid_list)),
                         num_gids=len(gid_list),
                         image_list=image_list,
                         num_images=len(image_list),
                         page=page,
                         page_start=page_start,
                         page_end=page_end,
                         page_total=page_total,
                         page_previous=page_previous,
                         page_next=page_next)


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
