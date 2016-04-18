#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.spatial.distance import cdist
import numpy as np
from config import PRIOR_MATCHING, PRIORS_


def _assignment_vector(net_output, index):
        x = np.zeros((net_output, 1), dtype=np.uint8)
        x[index, 0] = 1
        return x


def _invert_x(x):
    x = x[:, ::-1]
    indices = np.argsort(x[:, 0], axis=0)
    x = x[indices]
    return x


def F_loc(x, l, g):
    pairwise = cdist(l, g, metric='euclidean')
    pairwise = pairwise ** 2
    pairwise = pairwise * x
    result = 0.5 * np.sum(pairwise)
    return result


def F_conf(x, c):
    value1 = np.log(c)
    value1 = x * value1[:, None]
    value1 = np.sum(value1)
    value2 = np.log(1.0 - c)
    value_ = np.sum(x, axis=1)
    value2 = (1.0 - value_) * value2
    value2 = np.sum(value2)
    result = -1.0 * value1 + -1.0 * value2
    return result


def F(x, c, l, g, alpha, verbose=False, **kwargs):
    f_conf = F_conf(x, c)
    f_loc = F_loc(x, l, g)

    # f_final = f_conf + alpha * f_loc
    f_final = f_loc

    if verbose:
        args = (f_conf, f_loc, f_loc * alpha, alpha, )
        print('f_conf: %r, f_loc: %r (%r) [%r]' % args)
        print('f_final: %r' % (f_final, ))

    return f_final


def assignment_hungarian(cand_bbox_list, cand_prob_list, bbox_list, **kwargs):
    net_output = cand_bbox_list.shape[0]
    num, _ = bbox_list.shape
    cost_matrix = np.zeros((net_output, num))
    index_list = np.array([
        (i, j)
        for i in range(net_output)
        for j in range(num)
    ])
    cost_list = np.array([
        F(
            _assignment_vector(net_output, i),
            cand_prob_list,
            cand_bbox_list,
            bbox_list[j][None, :],
            **kwargs
        )
        for (i, j) in index_list
    ])
    cost_matrix[index_list[:, 0], index_list[:, 1]] = cost_list
    if np.isinf(np.max(cost_matrix)):
        return None
    x = linear_assignment(cost_matrix)
    x = _invert_x(x)
    return x


def assignment_partitioning(cand_bbox_list, cand_prob_list, bbox_list,
                            **kwargs):
    net_output = cand_bbox_list.shape[0]
    distance_list = cdist(PRIORS_, bbox_list, metric='euclidean')
    selection_list = np.argmin(distance_list, axis=1)

    indices = range(len(bbox_list))
    assignments = []
    for i in indices:
        selection = np.where(selection_list == i)[0]
        if len(selection) == 0:
            return None
        best_energy = np.inf
        best_selection = np.nan
        energy_list = []
        for j in selection:
            energy = F(
                _assignment_vector(net_output, j),
                cand_prob_list,
                cand_bbox_list,
                bbox_list[i][None, :],
                **kwargs
            )
            energy_list.append(energy)
            if energy <= best_energy:
                best_energy = energy
                best_selection = j
        assignments.append(best_selection)
    x = np.array(zip(assignments, indices))
    x = _invert_x(x)
    return x


def assignment_solution(*args, **kwargs):
    if PRIOR_MATCHING:
        x = assignment_partitioning(*args, **kwargs)
    else:
        x = assignment_hungarian(*args, **kwargs)
    if x is None:
        return x
    assert not np.isnan(np.min(x))
    assert not np.isinf(np.max(x))
    x = x.astype(np.int_)
    return x
