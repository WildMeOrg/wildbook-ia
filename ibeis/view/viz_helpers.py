from __future__ import division, print_function
import numpy as np
import drawtool.draw_func2 as df2
import utool
import vtool.chip as ctool
import vtool.keypoint as ktool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_helpers]', DEBUG=False)


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


def get_imgspace_chip_kpts(ibs, rid_list):
    # HACK! less of a hack than the hotspotter version
    bbox_list   = ibs.get_roi_bbox(rid_list)
    theta_list  = ibs.get_roi_theta(rid_list)
    chipsz_list = ibs.get_chip_size(rid_list)
    kpts_list   = ibs.get_chip_kpts(rid_list)

    imgkpts_list = []
    flatten_xs = np.array([[0, 2], [1, 2], [0, 0], [1, 0], [1, 1]])
    for bbox, theta, chipsz, kpts in zip(bbox_list, theta_list, chipsz_list, kpts_list):
        # HOLY SHIT THIS IS JANKY
        M = ctool._get_image_to_chip_transform(bbox, chipsz, theta)
        invV_list = ktool.get_invV_mats(kpts, homog=True)
        invM = np.linalg.inv(M)
        invMinvV_list = [invM.dot(invV) for invV in invV_list]
        flatten_xs = np.array([[0, 2], [1, 2], [0, 0], [1, 0], [1, 1]])
        imgkpts = [[invMinvV[index[0], index[1]] for index in flatten_xs] for invMinvV in invMinvV_list]
        imgkpts_list.append(np.array(imgkpts))
    return imgkpts_list
