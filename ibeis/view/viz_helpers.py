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


def cidstr(cid):
    return 'cid=%r' % cid


@getter_vector_output
def get_roi_kpts_in_imgspace(ibs, rid_list):
    """ Transforms keypoints so they are plotable in imagespace """
    bbox_list   = ibs.get_roi_bboxes(rid_list)
    theta_list  = ibs.get_roi_thetas(rid_list)
    chipsz_list = ibs.get_roi_sizes(rid_list)
    kpts_list   = ibs.get_roi_kpts(rid_list)
    imgkpts_list = [ktool.transform_kpts_to_imgspace(kpts, bbox, bbox_theta, chipsz)
                    for bbox, bbox_theta, chipsz, kpts
                    in izip(bbox_list, theta_list, chipsz_list, kpts_list)]
    return imgkpts_list


@getter
def get_chips(ibs, cid_list, in_image=False, **kwargs):
    if 'chip' in kwargs:
        return kwargs['chip']
    if in_image:
        rid_list = ibs.get_chip_rids(cid_list)
        return get_roi_kpts_in_imgspace(ibs, rid_list)
    else:
        return ibs.get_chips(cid_list)


@getter
def get_kpts(ibs, cid_list, in_image=False, **kwargs):
    if 'kpts' in kwargs:
        return kwargs['kpts']
    if in_image:
        kpts_list = get_roi_kpts_in_imgspace(ibs, cid_list)
    else:
        kpts_list = ibs.get_chip_kpts(cid_list)
    return kpts_list


@getter
def get_names(ibs, cid_list):
    rid_list = ibs.get_chip_rids(cid_list)
    return ibs.get_roi_names(rid_list)


@getter
def get_gnames(ibs, cid_list):
    rid_list = ibs.get_chip_rids(cid_list)
    return ibs.get_roi_gnames(rid_list)


@getter
def get_chip_titles(ibs, cid_list):
    title_list = ', '.join([
        cidstr(cid),
        'gname=%r' % get_gnames(ibs, cid),
        'name=%r'  % get_names(ibs, cid),
    ] for cid in cid_list)
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


def get_bbox_centers(bbox_list):
    bbox_centers = np.array([np.array([x + (w / 2), y + (h / 2)])]
                            for (x, y, w, h) in bbox_list)
    return bbox_centers
