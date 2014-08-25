from __future__ import absolute_import, division, print_function
import utool
import six
import numpy as np
import vtool.linalg as ltool
from numpy import array
from six.moves import zip
print, print_,  printDBG, rrr, profile = utool.inject(__name__, '[nnweight]')


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


def _nn_normalized_weight(normweight_fn, qaid2_nns, qreq_):
    #import utool
    #utool.stash_testdata('qaid2_nns')
    # Only valid for vsone
    K = qreq_.qparams.K
    Knorm = qreq_.qparams.Knorm
    rule  = qreq_.qparams.normalizer_rule
    qaid2_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    qaid2_selnorms = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    # Database feature index to chip index
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_nndist = qfx2_dist[:, 0:K]
        if rule == 'last':
            # Use the last normalizer
            qfx2_normk = np.zeros(len(qfx2_dist), np.int32) + (K + Knorm - 1)
        elif rule == 'name':
            # Get the top names you do not want your normalizer to be from
            qnid = qreq_.get_annot_nids(qaid)
            nTop = max(1, K)
            qfx2_topidx = qfx2_idx.T[0:nTop, :].T
            qfx2_normidx = qfx2_idx.T[-Knorm:].T
            # Apply temporary uniquish name
            qfx2_topaid  = qreq_.indexer.get_nn_aids(qfx2_topidx)
            qfx2_normaid = qreq_.indexer.get_nn_aids(qfx2_normidx)
            qfx2_topnid  = qreq_.get_annot_nids(qfx2_topaid)
            qfx2_normnid = qreq_.get_annot_nids(qfx2_normaid)
            # Inspect the potential normalizers
            qfx2_normk = mark_name_valid_normalizers(qfx2_normnid, qfx2_topnid, qnid)
            qfx2_normk += (K + Knorm)  # convert form negative to pos indexes
        else:
            raise NotImplementedError('[nn_weights] no rule=%r' % rule)
        qfx2_normdist = [dists[normk] for (dists, normk) in zip(qfx2_dist, qfx2_normk)]
        qfx2_normidx   = [idxs[normk]   for (idxs, normk)   in zip(qfx2_idx,   qfx2_normk)]
        # build meta
        qfx2_normmeta = [(qreq_.get_nn_aids(idx), qreq_.get_nn_featxs(idx), normk)
                                      for (normk, idx)    in zip(qfx2_normk,
                                                                 qfx2_normidx)]
        qfx2_normdist = array(qfx2_normdist)
        qfx2_normidx  = array(qfx2_normidx)
        qfx2_normmeta = array(qfx2_normmeta)
        # Ensure shapes are valid
        qfx2_normdist.shape = (len(qfx2_idx), 1)
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


# TODO: Make a more elegant way of mapping weighting parameters to weighting
# function. A dict is better than eval, but there may be a better way.
NN_WEIGHT_FUNC_DICT = {
    'lnrat':   nn_lnrat_weight,
    'lnbnn':   nn_lnbnn_weight,
    'ratio':   nn_ratio_weight,
}
