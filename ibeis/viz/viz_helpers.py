from __future__ import absolute_import, division, print_function
import numpy as np
from itertools import izip
import plottool.draw_func2 as df2
from plottool import plot_helpers as ph
import utool
import vtool.keypoint as ktool
from ibeis.dev import ibsfuncs
from ibeis.control.accessor_decors import getter, getter_vector_output
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_helpers]', DEBUG=False)


NO_LABEL_OVERRIDE = utool.get_arg('--no-label-override', type_=bool, default=None)


FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5, name=6)

IN_IMAGE_OVERRIDE = utool.get_arg('--in-image-override', type_=bool, default=None)
SHOW_QUERY_OVERRIDE = utool.get_arg('--show-query-override', type_=bool, default=None)
NO_LABEL_OVERRIDE = utool.get_arg('--no-label-override', type_=bool, default=None)

SIFT_OR_VECFIELD  = ph.SIFT_OR_VECFIELD


def register_FNUMS(FNUMS_):
    # DEPREICATE
    global FNUMS
    FNUMS = FNUMS_

draw = ph.draw
get_square_row_cols = ph.get_square_row_cols
get_ibsdat = ph.get_plotdat
set_ibsdat = ph.set_plotdat


@getter_vector_output
def get_annotion_kpts_in_imgspace(ibs, aid_list, **kwargs):
    """ Transforms keypoints so they are plotable in imagespace """
    ensure = kwargs.get('ensure', True)
    bbox_list   = ibs.get_annotion_bboxes(aid_list)
    theta_list  = ibs.get_annotion_thetas(aid_list)
    try:
        chipsz_list = ibs.get_annotion_chipsizes(aid_list, ensure=ensure)
    except AssertionError as ex:
        utool.printex(ex, '[!ibs.get_annotion_kpts_in_imgspace]')
        print('[!ibs.get_annotion_kpts_in_imgspace] aid_list = %r' % (aid_list,))
        raise
    kpts_list    = ibs.get_annotion_kpts(aid_list, ensure=ensure)
    imgkpts_list = [ktool.transform_kpts_to_imgspace(kpts, bbox, theta, chipsz)
                    for kpts, bbox, theta, chipsz
                    in izip(kpts_list, bbox_list, theta_list, chipsz_list)]
    return imgkpts_list


@getter_vector_output
def get_chips(ibs, aid_list, in_image=False, **kwargs):
    #if 'chip' in kwargs:
        #return kwargs['chip']
    if in_image:
        return ibs.get_annotion_images(aid_list)
    else:
        return ibs.get_annotion_chips(aid_list)


@getter_vector_output
def get_kpts(ibs, aid_list, in_image=False, **kwargs):
    #if 'kpts' in kwargs:
        #return kwargs['kpts']
    kpts_subset = kwargs.get('kpts_subset', None)
    ensure = kwargs.get('ensure', True)
    if in_image:
        kpts_list = get_annotion_kpts_in_imgspace(ibs, aid_list, **kwargs)
    else:
        kpts_list = ibs.get_annotion_kpts(aid_list, ensure=ensure)
    if kpts_subset is not None:
        kpts_list = [utool.spaced_items(kpts, kpts_subset, trunc=True) for kpts in kpts_list]
    return kpts_list


@getter_vector_output
def get_bboxes(ibs, aid_list, offset_list=None):
    bbox_list = ibs.get_annotion_bboxes(aid_list)
    if offset_list is not None:
        assert len(offset_list) == len(bbox_list)
        # convert (ofx, ofy) offsets to (ofx, ofy, 0, 0) numpy arrays
        np_offsts = [np.array(list(offst) + [0, 0]) for offst in offset_list]
        # add offsets to (x, y, w, h) bounding boxes
        bbox_list = [bbox + offst for bbox, offst in izip(bbox_list, np_offsts)]
    return bbox_list


def get_aidstrs(aid_list, **kwargs):
    if utool.isiterable(aid_list):
        return [ibsfuncs.aidstr(aid, **kwargs) for aid in aid_list]
    else:
        return ibsfuncs.aidstr(aid_list, **kwargs)


def get_vsstr(qaid, aid):
    return ibsfuncs.vsstr(qaid, aid)


def get_bbox_centers(bbox_list):
    center_pts = [((x + w / 2), (y + h / 2))
                  for (x, y, w, h) in bbox_list]
    center_pts = np.array(center_pts)
    return center_pts


def is_unknown(ibs, nid_list):
    return [nid == ibs.UNKNOWN_NID or nid < 0 for nid in nid_list]


def get_truth_label(ibs, truth):
    truth_labels = [
        'FALSE',
        'TRUE',
        '???'
    ]
    return truth_labels[truth]


def get_truth_color(truth, base255=False, lighten_amount=None):
    truth_colors = [
        df2.FALSE_RED,
        df2.TRUE_GREEN,
        df2.UNKNOWN_PURP,
    ]
    color = truth_colors[truth]
    if lighten_amount is not None:
        #print('color = %r, lighten_amount=%r' % (color, lighten_amount))
        color = df2.lighten_rgb(color, lighten_amount)
        #print('color = %r' % (color))
    if base255:
        color = df2.to_base255(color)
    return color


def get_timedelta_str(ibs, aid1, aid2):
    gid1, gid2 = ibs.get_annotion_gids([aid1, aid2])
    unixtime1, unixtime2 = ibs.get_image_unixtime([gid1, gid2])
    if -1 in [unixtime1, unixtime2]:
        timedelta_str_ = 'NA'
    else:
        unixtime_diff = unixtime2 - unixtime1
        timedelta_str_ = utool.get_unix_timedelta_str(unixtime_diff)
    timedelta_str = 'timedelta(%s)' % (timedelta_str_)
    return timedelta_str


def get_annotion_texts(ibs, aid_list, **kwargs):
    """ Add each type of label_list to the strings list """
    try:
        ibsfuncs.assert_valid_aids(ibs, aid_list)
        assert utool.isiterable(aid_list), 'input must be iterable'
        assert all([isinstance(aid, int) for aid in aid_list]), 'invalid input'
    except AssertionError as ex:
        utool.printex(ex, 'invalid input', 'viz', key_list=['aid_list'])
        raise
    texts_list = []  # list of lists of texts
    if kwargs.get('show_aidstr', True):
        aidstr_list = get_aidstrs(aid_list)
        texts_list.append(aidstr_list)
    if kwargs.get('show_gname', False):
        gname_list = ibs.get_annotion_gnames(aid_list)
        texts_list.append(['gname=%s' % gname for gname in gname_list])
    if kwargs.get('show_name', True):
        name_list = ibs.get_annotion_names(aid_list)
        texts_list.append(['name=%s' % name for name in name_list])
    if kwargs.get('show_exemplar', True):
        flag_list = ibs.get_annotion_exemplar_flag(aid_list)
        texts_list.append(['EX' if flag else '' for flag in flag_list])
    # zip them up to get a tuple for each chip and join the fields
    if len(texts_list) > 0:
        annotion_text_list = [', '.join(tup) for tup in izip(*texts_list)]
    else:
        # no labels were specified return empty string for each input
        annotion_text_list = [''] * len(aid_list)
    return annotion_text_list


@getter
def get_image_titles(ibs, gid_list):
    gname_list = ibs.get_image_gnames(gid_list)
    title_list = [
        'gid=%r gname=%r' % (gid, str(gname))
        for gid, gname in izip(gid_list, gname_list)
    ]
    return title_list


def get_annotion_labels(ibs, aid_list, draw_lbls):
    if draw_lbls:
        label_list = ibs.get_annotion_names(aid_list)
        #label = aid if label == '____' else label
    else:
        label_list = utool.alloc_nones(len(aid_list))
    return label_list


def get_query_label(ibs, qres, aid2, truth, **kwargs):
    """ returns title based on the query chip and result """
    label_list = []
    if kwargs.get('show_truth', False):
        truth_str = '*%s*' % get_truth_label(ibs, truth)
        label_list.append(truth_str)
    if kwargs.get('show_rank', True):
        rank_str = 'rank=%s' % str(qres.get_aid_ranks([aid2])[0] + 1)
        label_list.append(rank_str)
    if kwargs.get('show_score', True):
        score = qres.aid2_score[aid2]
        score_str = ('score=' + utool.num_fmt(score))
        if len(label_list) > 0:
            score_str = '\n' + score_str
        label_list.append(score_str)
    if kwargs.get('show_timedelta', False):
        timedelta_str = ('\n' + get_timedelta_str(ibs, qres.qaid, aid2))
        label_list.append(timedelta_str)
    query_label = ', '.join(label_list)
    return query_label


#==========================#
#  --- TESTING FUNCS ---   #
#==========================#


def show_keypoint_gradient_orientations(ibs, aid, fx, fnum=None, pnum=None):
    # Draw the gradient vectors of a patch overlaying the keypoint
    if fnum is None:
        fnum = df2.next_fnum()
    rchip = ibs.get_annotion_chips(aid)
    kp    = ibs.get_annotion_kpts(aid)[fx]
    sift  = ibs.get_annotion_desc(aid)[fx]
    df2.draw_keypoint_gradient_orientations(rchip, kp, sift=sift,
                                            mode='vec', fnum=fnum, pnum=pnum)
    df2.set_title('Gradient orientation\n %s, fx=%d' % (get_aidstrs(aid), fx))


def kp_info(kp):
    kpts = np.array([kp])
    xy_str    = ktool.get_xy_strs(kpts)[0]
    shape_str = ktool.get_shape_strs(kpts)[0]
    ori_ = ktool.get_oris(kpts)[0]
    ori_str = 'ori=%.2f' % ori_
    scale = ktool.get_scales(kpts)[0]
    return xy_str, shape_str, scale, ori_str
#----
