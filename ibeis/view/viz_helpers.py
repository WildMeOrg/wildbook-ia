from __future__ import division, print_function
import numpy as np
from itertools import izip
import drawtool.draw_func2 as df2
import utool
import vtool.keypoint as ktool
from ibeis.control.accessor_decors import getter, getter_vector_output
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_helpers]', DEBUG=False)


NO_LABEL_OVERRIDE = utool.get_arg('--no-label-override', type_=bool, default=None)


def draw():
    df2.adjust_subplots_safe()
    df2.draw()


def get_square_row_cols(nSubplots, max_cols=5):
    nCols = int(min(nSubplots, max_cols))
    #nCols = int(min(np.ceil(np.sqrt(ncids)), 5))
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
def get_roi_kpts_in_imgspace(ibs, rid_list):
    """ Transforms keypoints so they are plotable in imagespace """
    bbox_list   = ibs.get_roi_bboxes(rid_list)
    theta_list  = ibs.get_roi_thetas(rid_list)
    try:
        chipsz_list = ibs.get_roi_chipsizes(rid_list)
    except AssertionError as ex:
        utool.print_exception(ex, '[!ibs.get_roi_kpts_in_imgspace]')
        print('[!ibs.get_roi_kpts_in_imgspace] rid_list = %r' % (rid_list,))
        raise
    kpts_list    = ibs.get_roi_kpts(rid_list)
    imgkpts_list = [ktool.transform_kpts_to_imgspace(kpts, bbox, theta, chipsz)
                    for bbox, theta, chipsz, kpts
                    in izip(bbox_list, theta_list, chipsz_list, kpts_list)]
    return imgkpts_list


@getter
def get_chips(ibs, cid_list, in_image=False, **kwargs):
    if 'chip' in kwargs:
        return kwargs['chip']
    if in_image:
        rid_list = ibs.get_chip_rids(cid_list)
        return ibs.get_roi_images(rid_list)
    else:
        return ibs.get_chips(cid_list)


@getter
def get_kpts(ibs, cid_list, in_image=False, **kwargs):
    if 'kpts' in kwargs:
        return kwargs['kpts']
    if in_image:
        rid_list = ibs.get_chip_rids(cid_list)
        kpts_list = get_roi_kpts_in_imgspace(ibs, rid_list)
    else:
        kpts_list = ibs.get_chip_kpts(cid_list)
    return kpts_list


@getter
def get_bboxes(ibs, cid_list, offset_list=None):
    rid_list = ibs.get_chip_rids(cid_list)
    bbox_list = ibs.get_roi_bboxes(rid_list)
    if offset_list is not None:
        assert len(offset_list) == len(bbox_list)
        # convert (ofx, ofy) offsets to (ofx, ofy, 0, 0) numpy arrays
        np_offsts = (np.array(list(offst) + [0, 0]) for offst in offset_list)
        # add offsets to (x, y, w, h) bounding boxes
        bbox_list = [bbox + offst for bbox, offst in izip(bbox_list, np_offsts)]
    return bbox_list


@getter
def get_thetas(ibs, cid_list):
    rid_list = ibs.get_chip_rids(cid_list)
    theta_list = ibs.get_roi_thetas(rid_list)
    return theta_list


@getter
def get_groundtruth(ibs, cid_list):
    rid_list = ibs.get_chip_rids(cid_list)
    gt_list = ibs.get_roi_groundtruth(rid_list)
    return gt_list


@getter
def get_names(ibs, cid_list):
    name_list = ibs.get_chip_names(cid_list)
    return name_list


@getter
def get_gnames(ibs, cid_list):
    rid_list = ibs.get_chip_rids(cid_list)
    return ibs.get_roi_gnames(rid_list)


def get_cidstrs(cid_list):
    fmtstr = 'cid=%r'
    if utool.isiterable(cid_list):
        return [fmtstr % cid for cid in cid_list]
    else:
        cid = cid_list
        return fmtstr % cid_list


def get_bbox_centers(bbox_list):
    bbox_centers = np.array([np.array([x + (w / 2), y + (h / 2)])]
                            for (x, y, w, h) in bbox_list)
    return bbox_centers


def get_match_truth(ibs, cid1, cid2):
    nid1, nid2 = ibs.get_chip_nids((cid1, cid2))
    if nid1 != nid2:
        truth = 0
    elif nid1 > 0 and nid2 > 0:
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


def get_timedelta_str(ibs, cid1, cid2):
    gid1, gid2 = ibs.get_chip_gids([cid1, cid2])
    unixtime1, unixtime2 = ibs.get_image_unixtime([gid1, gid2])
    if -1 in [unixtime1, unixtime2]:
        timedelta_str_ = 'NA'
    else:
        unixtime_diff = unixtime2 - unixtime1
        timedelta_str_ = utool.get_unix_timedelta_str(unixtime_diff)
    timedelta_str = 'timedelta(%s)' % (timedelta_str_)
    return timedelta_str


@getter
def get_chip_labels(ibs, cid_list, **kwargs):
    # Add each type of label_list to the strings list
    label_strs = []
    if kwargs.get('show_cidstr', True):
        cidstr_list = get_cidstrs(cid_list)
        label_strs.append(cidstr_list)
    if kwargs.get('show_gname', True):
        gname_list = get_gnames(ibs, cid_list)
        label_strs.append(['gname=%s' % gname for gname in gname_list])
    if kwargs.get('show_name', True):
        name_list = get_names(ibs, cid_list)
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


def get_query_label(ibs, qres, cid2, truth, **kwargs):
    """ returns title based on the query chip and result """
    label_list = []
    if kwargs.get('show_truth', True):
        truth_str = '*%s*' % get_truth_label(ibs, truth)
        label_list.append(truth_str)
    if kwargs.get('show_rank', True):
        rank_str = ' rank=%s' % str(qres.get_cid_ranks([cid2])[0] + 1)
        label_list.append(rank_str)
    if kwargs.get('show_score', True):
        score = qres.cid2_score[cid2]
        score_str = (' score=' + utool.num_fmt(score))
        label_list.append(score_str)
    if kwargs.get('show_timedelta', False):
        timedelta_str = ('\n' + get_timedelta_str(ibs, qres.qcid, cid2))
        label_list.append(timedelta_str)
    query_label = ', '.join(label_list)
    return query_label
