from __future__ import absolute_import, division, print_function
import numpy as np
from itertools import izip
import plottool.draw_func2 as df2
import utool
import vtool.keypoint as ktool
from ibeis.control.accessor_decors import getter, getter_vector_output, getter_numpy_vector_output
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_helpers]', DEBUG=False)


NO_LABEL_OVERRIDE = utool.get_arg('--no-label-override', type_=bool, default=None)


FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5, name=6)

IN_IMAGE_OVERRIDE = utool.get_arg('--in-image-override', type_=bool, default=None)
SHOW_QUERY_OVERRIDE = utool.get_arg('--show-query-override', type_=bool, default=None)
NO_LABEL_OVERRIDE = utool.get_arg('--no-label-override', type_=bool, default=None)

SIFT_OR_VECFIELD = utool.get_arg('--vecfield', type_=bool)


def register_FNUMS(FNUMS_):
    # DEPREICATE
    global FNUMS
    FNUMS = FNUMS_


def draw():
    df2.adjust_subplots_safe()
    df2.draw()


def get_square_row_cols(nSubplots, max_cols=5):
    nCols = int(min(nSubplots, max_cols))
    #nCols = int(min(np.ceil(np.sqrt(nrids)), 5))
    nRows = int(np.ceil(nSubplots / nCols))
    return nRows, nCols


def get_ibsdat(ax, key, default=None):
    """ returns internal IBEIS property from a matplotlib axis """
    _ibsdat = ax.__dict__.get('_ibsdat', None)
    if _ibsdat is None:
        return default
    val = _ibsdat.get(key, default)
    return val


def set_ibsdat(ax, key, val):
    """ sets internal IBEIS property to a matplotlib axis """
    if not '_ibsdat' in ax.__dict__:
        ax.__dict__['_ibsdat'] = {}
    _ibsdat = ax.__dict__['_ibsdat']
    _ibsdat[key] = val


@getter_vector_output
def get_roi_kpts_in_imgspace(ibs, rid_list, **kwargs):
    """ Transforms keypoints so they are plotable in imagespace """
    ensure = kwargs.get('ensure', True)
    bbox_list   = ibs.get_roi_bboxes(rid_list)
    theta_list  = ibs.get_roi_thetas(rid_list)
    try:
        chipsz_list = ibs.get_roi_chipsizes(rid_list, ensure=ensure)
    except AssertionError as ex:
        utool.print_exception(ex, '[!ibs.get_roi_kpts_in_imgspace]')
        print('[!ibs.get_roi_kpts_in_imgspace] rid_list = %r' % (rid_list,))
        raise
    kpts_list    = ibs.get_roi_kpts(rid_list, ensure=ensure)
    imgkpts_list = [ktool.transform_kpts_to_imgspace(kpts, bbox, theta, chipsz)
                    for kpts, bbox, theta, chipsz
                    in izip(kpts_list, bbox_list, theta_list, chipsz_list)]
    return imgkpts_list


@getter_numpy_vector_output
def get_chips(ibs, rid_list, in_image=False, **kwargs):
    if 'chip' in kwargs:
        return kwargs['chip']
    if in_image:
        return ibs.get_roi_images(rid_list)
    else:
        return ibs.get_roi_chips(rid_list)


@getter_numpy_vector_output
def get_kpts(ibs, rid_list, in_image=False, kpts_subset=None, **kwargs):
    if 'kpts' in kwargs:
        return kwargs['kpts']
    ensure = kwargs.get('ensure', True)
    if in_image:
        kpts_list = get_roi_kpts_in_imgspace(ibs, rid_list, **kwargs)
    else:
        kpts_list = ibs.get_roi_kpts(rid_list, ensure=ensure)
    if kpts_subset:
        kpts_list = [utool.spaced_items(kpts, kpts_subset, trunc=True) for kpts in kpts_list]
    return kpts_list


@getter_numpy_vector_output
def get_bboxes(ibs, rid_list, offset_list=None):
    bbox_list = ibs.get_roi_bboxes(rid_list)
    if offset_list is not None:
        assert len(offset_list) == len(bbox_list)
        # convert (ofx, ofy) offsets to (ofx, ofy, 0, 0) numpy arrays
        np_offsts = [np.array(list(offst) + [0, 0]) for offst in offset_list]
        # add offsets to (x, y, w, h) bounding boxes
        bbox_list = [bbox + offst for bbox, offst in izip(bbox_list, np_offsts)]
    return bbox_list


@getter
def get_thetas(ibs, rid_list):
    theta_list = ibs.get_roi_thetas(rid_list)
    return theta_list


@getter
def get_groundtruth(ibs, rid_list):
    gt_list = ibs.get_roi_groundtruth(rid_list)
    return gt_list


@getter
def get_names(ibs, rid_list):
    name_list = ibs.get_roi_names(rid_list)
    return name_list


@getter
def get_gnames(ibs, rid_list):
    return ibs.get_roi_gnames(rid_list)


def get_ridstrs(rid_list):
    fmtstr = 'rid=%r'
    if utool.isiterable(rid_list):
        return [fmtstr % rid for rid in rid_list]
    else:
        return fmtstr % rid_list


def get_bbox_centers(bbox_list):
    bbox_centers = np.array([np.array([x + (w / 2), y + (h / 2)])
                             for (x, y, w, h) in bbox_list])
    return bbox_centers


def get_match_truth(ibs, rid1, rid2):
    nid1, nid2 = ibs.get_roi_nids((rid1, rid2))
    if nid1 != nid2 and nid1 > 1 and nid2 > 1:
        truth = 0
    elif nid1 > 1 and nid2 > 1:
        truth = 1
    else:
        truth = 2
    return truth


def get_truth_label(ibs, truth):
    truth_labels = [
        'FALSE',
        'TRUE',
        '???'
    ]
    return truth_labels[truth]


def get_truth_color(ibs, truth):
    truth_colors = [
        df2.FALSE_RED,
        df2.TRUE_GREEN,
        df2.UNKNOWN_PURP,
    ]
    return truth_colors[truth]


def get_timedelta_str(ibs, rid1, rid2):
    gid1, gid2 = ibs.get_roi_gids([rid1, rid2])
    unixtime1, unixtime2 = ibs.get_image_unixtime([gid1, gid2])
    if -1 in [unixtime1, unixtime2]:
        timedelta_str_ = 'NA'
    else:
        unixtime_diff = unixtime2 - unixtime1
        timedelta_str_ = utool.get_unix_timedelta_str(unixtime_diff)
    timedelta_str = 'timedelta(%s)' % (timedelta_str_)
    return timedelta_str


@getter
def get_chip_labels(ibs, rid_list, **kwargs):
    # Add each type of label_list to the strings list
    label_strs = []
    if kwargs.get('show_ridstr', True):
        ridstr_list = get_ridstrs(rid_list)
        label_strs.append(ridstr_list)
    if kwargs.get('show_gname', True):
        gname_list = get_gnames(ibs, rid_list)
        label_strs.append(['gname=%s' % gname for gname in gname_list])
    if kwargs.get('show_name', True):
        name_list = get_names(ibs, rid_list)
        label_strs.append(['name=%s' % name for name in name_list])
    # zip them up to get a tuple for each chip and join the fields
    title_list = [', '.join(tup) for tup in izip(*label_strs)]
    return title_list


@getter
def get_image_titles(ibs, gid_list):
    gname_list = ibs.get_image_gnames(gid_list)
    title_list = [
        'gid=%r gname=%r' % (gid, gname)
        for gid, gname in izip(gid_list, gname_list)
    ]
    return title_list


def get_roi_labels(ibs, rid_list, draw_lbls):
    if draw_lbls:
        label_list = ibs.get_roi_names(rid_list)
        #label = rid if label == '____' else label
    else:
        label_list = utool.alloc_nones(len(rid_list))
    return label_list


def get_query_label(ibs, qres, rid2, truth, **kwargs):
    """ returns title based on the query chip and result """
    label_list = []
    if kwargs.get('show_truth', True):
        truth_str = '*%s*' % get_truth_label(ibs, truth)
        label_list.append(truth_str)
    if kwargs.get('show_rank', True):
        rank_str = ' rank=%s' % str(qres.get_rid_ranks([rid2])[0] + 1)
        label_list.append(rank_str)
    if kwargs.get('show_score', True):
        score = qres.rid2_score[rid2]
        score_str = (' score=' + utool.num_fmt(score))
        label_list.append(score_str)
    if kwargs.get('show_timedelta', False):
        timedelta_str = ('\n' + get_timedelta_str(ibs, qres.qrid, rid2))
        label_list.append(timedelta_str)
    query_label = ', '.join(label_list)
    return query_label


#==========================#
#  --- TESTING FUNCS ---   #
#==========================#


def show_keypoint_gradient_orientations(ibs, rid, fx, fnum=None, pnum=None):
    # Draw the gradient vectors of a patch overlaying the keypoint
    if fnum is None:
        fnum = df2.next_fnum()
    rchip = ibs.get_roi_chips(rid)
    kp    = ibs.get_roi_kpts(rid)[fx]
    sift  = ibs.get_roi_desc(rid)[fx]
    df2.draw_keypoint_gradient_orientations(rchip, kp, sift=sift,
                                            mode='vec', fnum=fnum, pnum=pnum)
    df2.set_title('Gradient orientation\n %s, fx=%d' % (get_ridstrs(rid), fx))


def kp_info(kp):
    kpts = np.array([kp])
    xy_str    = ktool.get_xy_strs(kpts)[0]
    shape_str = ktool.get_shape_strs(kpts)[0]
    ori_ = ktool.get_oris(kpts)[0]
    ori_str = 'ori=%.2f' % ori_
    scale = ktool.get_scales(kpts)[0]
    return xy_str, shape_str, scale, ori_str
#----
