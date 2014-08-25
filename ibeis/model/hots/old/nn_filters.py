from __future__ import absolute_import, division, print_function
import utool
import six
print, print_,  printDBG, rrr, profile =\
    utool.inject(__name__, '[nnfilt]', DEBUG=False)
import numpy as np
import vtool.linalg as ltool
from numpy import array
from six.moves import zip


eps = 1E-8


def LNRAT_fn(vdist, ndist):
    return np.log(np.divide(ndist, vdist + eps) + 1)


def RATIO_fn(vdist, ndist):
    return np.divide(ndist, vdist + eps)


def LNBNN_fn(vdist, ndist):
    return (ndist - vdist)  # / 1000.0


# normweight_fn = LNBNN_fn
"""
ndist = array([[0, 1, 2], [3, 4, 5], [3, 4, 5], [3, 4, 5],  [9, 7, 6] ])
vdist = array([[3, 2, 1, 5], [3, 2, 5, 6], [3, 4, 5, 3], [3, 4, 5, 8],  [9, 7, 6, 3] ])
vdist1 = vdist[:,0:1]
vdist2 = vdist[:,0:2]
vdist3 = vdist[:,0:3]
vdist4 = vdist[:,0:4]
print(LNBNN_fn(vdist1, ndist)) * 1000
print(LNBNN_fn(vdist2, ndist)) * 1000
print(LNBNN_fn(vdist3, ndist)) * 1000
print(LNBNN_fn(vdist4, ndist)) * 1000
"""


def mark_name_valid_normalizers(qfx2_normnid, qfx2_topnid, qnid=None):
    #columns = qfx2_topnid
    #matrix = qfx2_normnid
    Kn = qfx2_normnid.shape[1]
    qfx2_valid = True - ltool.compare_matrix_columns(qfx2_normnid, qfx2_topnid)
    if qnid is not None:
        qfx2_valid = np.logical_and(qfx2_normnid != qnid, qfx2_valid)
    qfx2_validlist = [np.where(normrow)[0] for normrow in qfx2_valid]
    qfx2_selnorm = array([poslist[0] - Kn if len(poslist) != 0 else -1 for
                          poslist in qfx2_validlist], np.int32)
    return qfx2_selnorm


def _nn_normalized_weight(normweight_fn, ibs, qaid2_nns, qreq):
    #import utool
    #utool.stash_testdata('qaid2_nns')
    # Only valid for vsone
    K = qreq.cfg.nn_cfg.K
    Knorm = qreq.cfg.nn_cfg.Knorm
    rule  = qreq.cfg.nn_cfg.normalizer_rule
    qaid2_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    qaid2_selnorms = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    # Database feature index to chip index
    dx2_aid = qreq.data_index.dx2_aid
    dx2_fx = qreq.data_index.dx2_fx
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_dx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_nndist = qfx2_dist[:, 0:K]
        if rule == 'last':
            # Use the last normalizer
            qfx2_normk = np.zeros(len(qfx2_dist), np.int32) + (K + Knorm - 1)
        elif rule == 'name':
            # Get the top names you do not want your normalizer to be from
            qnid = ibs.get_annot_nids(qaid)
            nTop = max(1, K)
            qfx2_topdx = qfx2_dx.T[0:nTop, :].T
            qfx2_normdx = qfx2_dx.T[-Knorm:].T
            # Apply temporary uniquish name
            qfx2_topnid = ibs.get_annot_nids(dx2_aid[qfx2_topdx])
            qfx2_normnid = ibs.get_annot_nids(dx2_aid[qfx2_normdx])
            # Inspect the potential normalizers
            qfx2_normk = mark_name_valid_normalizers(qfx2_normnid, qfx2_topnid, qnid)
            qfx2_normk += (K + Knorm)  # convert form negative to pos indexes
        else:
            raise NotImplementedError('[nn_filters] no rule=%r' % rule)
        qfx2_normdist = [dists[normk]
                         for (dists, normk) in zip(qfx2_dist, qfx2_normk)]
        qfx2_normdx   = [dxs[normk]
                         for (dxs, normk)   in zip(qfx2_dx, qfx2_normk)]
        qfx2_normmeta = [(dx2_aid[dx], dx2_fx[dx], normk)
                         for (normk, dx) in zip(qfx2_normk, qfx2_normdx)]
        qfx2_normdist = array(qfx2_normdist)
        qfx2_normdx   = array(qfx2_normdx)
        qfx2_normmeta = array(qfx2_normmeta)
        # Ensure shapes are valid
        qfx2_normdist.shape = (len(qfx2_dx), 1)
        qfx2_normweight = normweight_fn(qfx2_nndist, qfx2_normdist)
        # Output
        qaid2_weight[qaid]   = qfx2_normweight
        qaid2_selnorms[qaid] = qfx2_normmeta
    return qaid2_weight, qaid2_selnorms


def nn_ratio_weight(*args):
    return _nn_normalized_weight(RATIO_fn, *args)


def nn_lnbnn_weight(*args):
    return _nn_normalized_weight(LNBNN_fn, *args)


def nn_lnrat_weight(*args):
    return _nn_normalized_weight(LNRAT_fn, *args)


def nn_bursty_weight(ibs, qaid2_nns, qreq):
    'Filters matches to a feature which is matched > burst_thresh #times'
    # Half-generalized to vsmany
    # Assume the first nRows-1 rows are the matches (last row is normalizer)
    K = qreq.cfg.nn_cfg.K
    qaid2_bursty_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    qaid2_metaweight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_dx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_nn = qfx2_dx[:, 0:K]
        dx2_frequency  = np.bincount(qfx2_nn.flatten())
        qfx2_bursty = dx2_frequency[qfx2_nn]
        qaid2_bursty_weight[qaid] = qfx2_bursty
    return qaid2_bursty_weight, qaid2_metaweight


def nn_recip_weight(ibs, qaid2_nns, qreq):
    'Filters a nearest neighbor to only reciprocals'
    data_index = qreq.data_index
    K = qreq.cfg.nn_cfg.K
    Krecip = qreq.cfg.filt_cfg.Krecip
    checks = qreq.cfg.nn_cfg.checks
    dx2_data = data_index.dx2_data
    data_flann = data_index.flann
    qaid2_recip_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    qaid2_metaweight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_dx, qfx2_dist) = qaid2_nns[qaid]
        nQuery = len(qfx2_dx)
        dim = dx2_data.shape[1]
        # Get the original K nearest features
        qx2_nndx = dx2_data[qfx2_dx[:, 0:K]]
        qx2_nndist = qfx2_dist[:, 0:K]
        qx2_nndx.shape = (nQuery * K, dim)
        # TODO: Have the option for this to be both indexes.
        (_nn2_rdx, _nn2_rdists) = data_flann.nn_index(qx2_nndx, Krecip, checks=checks)
        # Get the maximum distance of the Krecip reciprocal neighbors
        _nn2_rdists.shape = (nQuery, K, Krecip)
        qfx2_recipmaxdist = _nn2_rdists.max(2)
        # Test if nearest neighbor distance is less than reciprocal distance
        qfx2_reciprocalness = qfx2_recipmaxdist - qx2_nndist
        qaid2_recip_weight[qaid] = qfx2_reciprocalness
    return qaid2_recip_weight, qaid2_metaweight


def nn_bboxdist_weight(ibs, qaid2_nns, qreq):
    'Filters a matches to those within roughly the same spatial arangement'
    data_index = qreq.data_index
    K = qreq.cfg.nn_cfg.K
    dx2_aid = data_index.dx2_aid
    dx2_fx = data_index.dx2_fx
    aid2_bboxdist_weight = {}
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_dx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_nn = qfx2_dx[:, 0:K]
        # Get matched chip sizes #.0300s
        qfx2_kpts = ibs.get_annot_kpts(qaid)
        nQuery = len(qfx2_dx)
        qfx2_aid = dx2_aid[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_chipsize2 = array([ibs.get_annot_chipsizes(aid) for aid in qfx2_aid.flat])
        qfx2_chipsize2.shape = (nQuery, K, 2)
        qfx2_chipdiag2 = np.sqrt((qfx2_chipsize2 ** 2).sum(2))
        # Get query relative xy keypoints #.0160s / #.0180s (+cast)
        qdiag = np.sqrt((array(ibs.get_annot_chipsizes(qaid)) ** 2).sum())
        qfx2_xy1 = array(qfx2_kpts[:, 0:2], np.float)
        qfx2_xy1[:, 0] /= qdiag
        qfx2_xy1[:, 1] /= qdiag
        # Get database relative xy keypoints
        qfx2_xy2 = array([ibs.get_annot_kpts(aid)[fx, 0:2] for (aid, fx) in
                          zip(qfx2_aid.flat, qfx2_fx.flat)], np.float)
        qfx2_xy2.shape = (nQuery, K, 2)
        qfx2_xy2[:, :, 0] /= qfx2_chipdiag2
        qfx2_xy2[:, :, 1] /= qfx2_chipdiag2
        # Get the relative distance # .0010s
        qfx2_K_xy1 = np.rollaxis(np.tile(qfx2_xy1, (K, 1, 1)), 1)
        qfx2_xydist = ((qfx2_K_xy1 - qfx2_xy2) ** 2).sum(2)
        aid2_bboxdist_weight[qaid] = qfx2_xydist
    return aid2_bboxdist_weight


def nn_scale_weight(ibs, qaid2_nns, qreq):
    # Filter by scale for funzies
    K = qreq.cfg.nn_cfg.K
    aid2_scale_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    qaid2_metaweight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    data_index = qreq.data_index
    K = qreq.cfg.nn_cfg.K
    dx2_aid = data_index.dx2_aid
    dx2_fx = data_index.dx2_fx
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_dx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_kpts = ibs.get_annot_kpts(qaid)
        qfx2_nn = qfx2_dx[:, 0:K]
        nQuery = len(qfx2_dx)
        qfx2_aid = dx2_aid[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_det1 = array(qfx2_kpts[:, [2, 4]], np.float).prod(1)
        qfx2_det1 = np.sqrt(1.0 / qfx2_det1)
        qfx2_K_det1 = np.rollaxis(np.tile(qfx2_det1, (K, 1)), 1)
        qfx2_det2 = array([ibs.get_annot_kpts(aid)[fx, [2, 4]] for (aid, fx) in
                           zip(qfx2_aid.flat, qfx2_fx.flat)], np.float).prod(1)
        qfx2_det2.shape = (nQuery, K)
        qfx2_det2 = np.sqrt(1.0 / qfx2_det2)
        qfx2_scaledist = qfx2_det2 / qfx2_K_det1
        aid2_scale_weight[qaid] = qfx2_scaledist
    return aid2_scale_weight, qaid2_metaweight


# TODO: Make a more elegant way of mapping weighting parameters to weighting
# function. A dict is better than eval, but there may be a better way.
NN_FILTER_FUNC_DICT = {
    'scale':   nn_scale_weight,
    'bboxdist': nn_bboxdist_weight,
    'recip':   nn_recip_weight,
    'bursty':  nn_bursty_weight,
    'lnrat':   nn_lnrat_weight,
    'lnbnn':   nn_lnbnn_weight,
    'ratio':   nn_ratio_weight,
}
