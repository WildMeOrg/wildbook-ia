# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import vtool as vt
print, rrr, profile = ut.inject2(__name__)


def estimate_threshold(curve, method=None):
    """
        import plottool as pt
        idx3 = vt.find_elbow_point(curve[idx1:idx2 + 1]) + idx1
        pt.plot(curve)
        pt.plot(idx1, curve[idx1], 'bo')
        pt.plot(idx2, curve[idx2], 'ro')
        pt.plot(idx3, curve[idx3], 'go')
    """
    if len(curve) == 0:
        return None
    if method is None:
        method = 'mean'
    if method == 'mean':
        thresh = np.mean(curve)
    elif method == 'elbow':
        idx1 = vt.find_elbow_point(curve)
        idx2 = vt.find_elbow_point(curve[idx1:]) + idx1
        thresh = curve[idx2]
    else:
        raise ValueError('method = %r' % (method,))
    return thresh


@six.add_metaclass(ut.ReloadingMetaclass)
class InfrModel(ut.NiceRepr):
    """
    Wrapper around graphcut algorithms
    """

    def __init__(model, graph):
        model.graph = graph
        model._update_state()

    def _update_state(model, weight_key='cut_weight',
                      name_label_key='name_label'):
        import networkx as nx
        # Get nx graph properties
        external_nodes = sorted(list(model.graph.nodes()))
        external_edges = list(model.graph.edges())
        edge2_weights = nx.get_edge_attrs(model.graph, weight_key)
        node2_labeling = nx.get_node_attrs(model.graph, name_label_key)
        edge_weights = ut.dict_take(edge2_weights, external_edges, 0)
        external_labeling = ut.take(node2_labeling, external_nodes)
        # Map to internal ids for pygco
        internal_nodes = ut.rebase_labels(external_nodes)
        extern2_intern = dict(zip(external_nodes, internal_nodes))
        internal_edges = ut.unflat_take(extern2_intern, external_edges)
        internal_labeling = ut.rebase_labels(external_labeling)
        n_nodes = len(internal_nodes)
        # Model state
        model.n_nodes = n_nodes
        model.extern2_intern = extern2_intern
        model.intern2_extern = ut.invert_dict(extern2_intern)
        model.edges = internal_edges
        model.edge_weights = edge_weights
        # Model parameters
        model.labeling = np.zeros(model.n_nodes, dtype=np.int32)
        model._update_labels(labeling=internal_labeling)
        model._update_weights()

    def __nice__(self):
        return 'n_nodes=%r, n_labels=%r' % (self.n_nodes, self.n_labels)
        #return '(n_nodes=%r, n_labels=%r, nrg=%r)' % (self.n_nodes,
        #self.n_labels, self.total_energy)

    def _update_labels(model, n_labels=None, unaries=None, labeling=None):
        if labeling is not None:
            n_labels_ = max(labeling) + 1
            assert n_labels is None or n_labels == n_labels_
            n_labels = n_labels_
        if n_labels is None:
            n_labels = 2
        if unaries is None:
            unaries = np.zeros((model.n_nodes, n_labels), dtype=np.int32)
        # Update internals
        model.pairwise_potts = -1 * np.eye(n_labels, dtype=np.int32)
        model.n_labels = n_labels
        model.unaries = unaries
        if model.labeling.max() >= n_labels:
            model.labeling = np.zeros(model.n_nodes, dtype=np.int32)

    def _update_weights(model, thresh=None):
        int_factor = 1E2
        edge_weights = np.array(model.edge_weights)
        if thresh is None:
            thresh = model._estimate_threshold()
        else:
            if isinstance(thresh, six.string_types):
                thresh = model._estimate_threshold(method=thresh)
            #np.mean(edge_weights)
        if True:
            # Center and scale weights between -1 and 1
            centered = (edge_weights - thresh)
            centered[centered < 0] = (centered[centered < 0] / thresh)
            centered[centered > 0] = (centered[centered > 0] / (1 - thresh))
            newprob = (centered + 1) / 2
            newprob[np.isnan(newprob)] = .5
            # Apply logit rule
            # prevent infinity
            #pad = 1 / (int_factor * 2)
            pad = 1E6
            perbprob = (newprob * (1.0 - pad * 2)) + pad
            weights = vt.logit(perbprob)
        else:
            weights = (edge_weights - thresh)
            # Conv
            weights[np.isnan(edge_weights)] = 0

        weights = (weights * int_factor).astype(np.int32)
        edges_ = np.round(model.edges).astype(np.int32)
        edges_ = vt.atleast_nd(edges_, 2)
        edges_.shape = (edges_.shape[0], 2)
        weighted_edges = np.vstack((edges_.T, weights)).T
        weighted_edges = np.ascontiguousarray(weighted_edges)
        weighted_edges = np.nan_to_num(weighted_edges)
        # Remove edges with 0 weight as they have no influence
        weighted_edges = weighted_edges.compress(weighted_edges.T[2] != 0, axis=0)
        # Update internals
        model.thresh = thresh
        model.weighted_edges = weighted_edges
        model.weights = weights

    @property
    def total_energy(model):
        pairwise_potts = model.pairwise_potts
        wedges = model.weighted_edges
        unary_idxs = (model.labeling,)
        pairwise_idxs = (model.labeling[wedges.T[0]],
                         model.labeling[wedges.T[1]])
        _unary_energies = model.unaries[unary_idxs]
        _potts_energies = pairwise_potts[pairwise_idxs]
        unary_energy = _unary_energies.sum()
        pairwise_energy = (wedges.T[2] * _potts_energies).sum()
        total_energy = unary_energy + pairwise_energy
        return total_energy

    @property
    def node_to_label(model):
        # External nodes to label
        nodes = ut.take(model.intern2_extern, range(model.n_nodes))
        extern_node2_new_label = dict(zip(nodes, model.labeling))
        return extern_node2_new_label

    def _estimate_threshold(model, method=None, curve=None):
        """
            import plottool as pt
            idx3 = vt.find_elbow_point(curve[idx1:idx2 + 1]) + idx1
            pt.plot(curve)
            pt.plot(idx1, curve[idx1], 'bo')
            pt.plot(idx2, curve[idx2], 'ro')
            pt.plot(idx3, curve[idx3], 'go')
        """
        if curve is None:
            isvalid = ~np.isnan(model.edge_weights)
            curve = sorted(ut.compress(model.edge_weights, isvalid))
        thresh = estimate_threshold(curve, method)
        #if len(curve) == 0:
        #    return 0
        #if method is None:
        #    method = 'mean'
        #if method == 'mean':
        #    thresh = np.mean(curve)
        #elif method == 'elbow':
        #    idx1 = vt.find_elbow_point(curve)
        #    idx2 = vt.find_elbow_point(curve[idx1:]) + idx1
        #    thresh = curve[idx2]
        #else:
        #    raise ValueError('method = %r' % (method,))
        return thresh

    def run_inference(model, thresh=None, n_labels=None, n_iter=5,
                      algorithm='expansion'):
        import pygco
        if n_labels is not None:
            model._update_labels(n_labels)
        if thresh is not None:
            model._update_weights(thresh=thresh)
        if model.n_labels <= 0:
            raise ValueError('cannot run inference with zero labels')
        if model.n_labels == 1:
            labeling = np.zeros(model.n_nodes, dtype=np.int32)
        else:
            cutkw = dict(n_iter=n_iter, algorithm=algorithm)
            if 0:
                print(ut.code_repr(model.unaries, 'unaries'))
                print(ut.code_repr(model.weighted_edges, 'weighted_edges'))
                print(ut.code_repr(model.pairwise_potts, 'pairwise_potts'))
                print(ut.code_repr(cutkw, 'cutkw'))
            labeling = pygco.cut_from_graph(model.weighted_edges, model.unaries,
                                            model.pairwise_potts, **cutkw)
            model.labeling = labeling
        #print('model.total_energy = %r' % (model.total_energy,))
        return labeling

    def run_inference2(model, min_labels=1, max_labels=10):
        cut_params = ut.all_dict_combinations({
            #'n_labels': list(range(min_labels, max_labels + 1)),
            #'n_labels': list(range(min_labels, max_labels + 1)),
            'n_labels': list(range(max_labels, max_labels + 1)),
        })
        cut_energies = []
        cut_labeling = []
        for params in cut_params:
            model.run_inference(**params)
            nrg = model.total_energy
            #complexity = .1 * model.n_nodes * model.thresh * params['n_labels']
            complexity = 0
            nrg2 = nrg + complexity
            print('used %d labels' % (len(set(model.labeling))),)
            print('complexity = %r' % (complexity,))
            print('nrg = %r' % (nrg,))
            print('nrg + complexity = %r' % (nrg2,))
            cut_energies.append(nrg2)
            cut_labeling.append(model.labeling)

        best_paramx = np.argmin(cut_energies)
        print('best_paramx = %r' % (best_paramx,))
        params = cut_params[best_paramx]
        print('params = %r' % (params,))
        labeling = cut_labeling[best_paramx]
        model.labeling = labeling
        #labeling = model.run_inference(**params)
        return labeling, params

    @staticmethod
    def weights_as_matrix(weighted_edges):
        n_labels = weighted_edges.T[0:2].max() + 1
        mat = np.zeros((n_labels, n_labels))
        flat_idxs = np.ravel_multi_index(weighted_edges.T[0:2], dims=(n_labels, n_labels))
        assert ut.isunique(flat_idxs)
        mat.ravel()[flat_idxs] = weighted_edges.T[2]
        #mat[tuple(weighted_edges.T[0:2])] = weighted_edges.T[2]

    def get_cut_edges(model):
        extern_uv_list = np.array(list(model.graph.edges()))
        intern_uv_list = ut.unflat_take(model.extern2_intern, extern_uv_list)
        intern_uv_list = np.array(intern_uv_list)
        u_labels = model.labeling[intern_uv_list.T[0]]
        v_labels = model.labeling[intern_uv_list.T[1]]
        # Remove edges between all annotations with different labels
        cut_edges = extern_uv_list[u_labels != v_labels]
        cut_edges = [tuple(uv.tolist()) for uv in cut_edges]
        return cut_edges


# def piecewise_weighting(infr, normscores, edges):
#     # Old code
#     edge_scores = normscores
#     # Try to put scores in a 0 to 1 range
#     control_points = [
#         (0.0, .001),
#         (3.0, .05),
#         (15.0, .95),
#         (None, .99),
#     ]
#     edge_weights = edge_scores.copy()
#     for (pt1, prob1), (pt2, prob2) in ut.itertwo(control_points):
#         if pt1 is None:
#             pt1 = np.nanmin(edge_scores)
#         if pt2 is None:
#             pt2 = np.nanmax(edge_scores) + .0001
#         pt_len = pt2 - pt1
#         prob_len = prob2 - prob1
#         flag = np.logical_and(edge_scores >= pt1, edge_scores < pt2)
#         edge_weights[flag] = (((edge_scores[flag] - pt1) / pt_len) * prob_len) + prob1

#     nx.set_edge_attrs(infr.graph, CUT_WEIGHT_KEY, _dz(edges, edge_weights))

#     p_same, unique_pairs = infr._get_feedback_probs()
#     unique_pairs = [tuple(x.tolist()) for x in unique_pairs]
#     for aid1, aid2 in unique_pairs:
#         if not infr.graph.has_edge(aid1, aid2):
#             infr.graph.add_edge(aid1, aid2)
#     nx.set_edge_attrs(infr.graph, CUT_WEIGHT_KEY, _dz(unique_pairs, p_same))
#     #nx.set_edge_attrs(infr.graph, 'lw', _dz(unique_pairs, [6.0]))
#     """
#     pt.plot(sorted(edge_weights))
#     pt.plot(sorted(vt.norm01(edge_scores)))
#     """
#     #import scipy.special
#     #a = 1.5
#     #b = 2
#     #p_same = scipy.special.expit(b * edge_scores - a)
#     #confidence = (2 * np.abs(0.5 - p_same)) ** 2
