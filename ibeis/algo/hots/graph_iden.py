# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt  # NOQA
import six
import networkx as nx
nx.set_edge_attrs = nx.set_edge_attributes
nx.get_edge_attrs = nx.get_edge_attributes
nx.set_node_attrs = nx.set_node_attributes
nx.get_node_attrs = nx.get_node_attributes
print, rrr, profile = ut.inject2(__name__, '[graph_inference]')


@six.add_metaclass(ut.ReloadingMetaclass)
class InfrModel(ut.NiceRepr):

    def __init__(model, graph):
        #def __init__(model, n_nodes, edges, edge_weights=None, n_labels=None,
        model.graph = graph
        model._update_state()

    def _update_state(model):
        import networkx as nx
        name_label_key = 'name_label'
        weight_key = 'weight'
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
        return '(n_nodes=%r, n_labels=%r)' % (self.n_nodes, self.n_labels)
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
        int_factor = 100
        edge_weights = np.array(model.edge_weights)
        if thresh is None:
            thresh = model.estimate_threshold()
        else:
            if isinstance(thresh, six.string_types):
                thresh = model.estimate_threshold(method=thresh)
            #np.mean(edge_weights)
        weights = (edge_weights - thresh) * int_factor
        weights = weights.astype(np.int32)
        edges_ = np.array(model.edges).astype(np.int32)
        edges_ = vt.atleast_nd(edges_, 2)
        edges_.shape = (edges_.shape[0], 2)
        weighted_edges = np.vstack((edges_.T, weights)).T
        weighted_edges = np.ascontiguousarray(weighted_edges)
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

    def estimate_threshold(model, method=None):
        """
            import plottool as pt
            idx3 = vt.find_elbow_point(curve[idx1:idx2 + 1]) + idx1
            pt.plot(curve)
            pt.plot(idx1, curve[idx1], 'bo')
            pt.plot(idx2, curve[idx2], 'ro')
            pt.plot(idx3, curve[idx3], 'go')
        """
        if method is None:
            method = 'mean'
        curve = sorted(model.edge_weights)
        if method == 'mean':
            thresh = np.mean(curve)
        elif method == 'elbow':
            idx1 = vt.find_elbow_point(curve)
            idx2 = vt.find_elbow_point(curve[idx1:]) + idx1
            thresh = curve[idx2]
        else:
            raise ValueError('method = %r' % (method,))
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
            labeling = pygco.cut_from_graph(model.weighted_edges, model.unaries,
                                            model.pairwise_potts, **cutkw)
            model.labeling = labeling
        #print('model.total_energy = %r' % (model.total_energy,))
        return labeling

    def run_inference2(model, min_labels=1, max_labels=5):
        cut_params = ut.all_dict_combinations({
            'n_labels': list(range(min_labels, max_labels + 1)),
        })
        cut_energies = []
        cut_labeling = []
        for params in cut_params:
            model.run_inference(**params)
            nrg = model.total_energy
            complexity = .1 * model.n_nodes * model.thresh * params['n_labels']
            nrg2 = nrg + complexity
            print('complexity = %r' % (complexity,))
            print('nrg = %r' % (nrg,))
            print('nrg2 = %r' % (nrg2,))
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
        #for (u, v) in cut_edges:
        #    graph_.remove_edge(u, v)
        ##list(nx.connected_components(graph_.to_undirected()))
        #return graph_


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInference2(object):

    def __init__(infr, ibs, aids, nids, current_nids=None, user_feedback=None):
        infr.ibs = ibs
        infr.aids = aids
        infr.orig_name_labels = nids
        if current_nids is None:
            current_nids = nids
        assert nids is not None, 'cant be none'
        assert len(aids) == len(nids)
        assert len(aids) == len(current_nids)
        infr.current_name_labels = current_nids
        infr.graph = None
        if user_feedback is None:
            user_feedback = {}
        infr.user_feedback = {
            lbl: x.tolist() if isinstance(x, np.ndarray) else x
            for lbl, x in user_feedback.items()
        }

    def add_feedback(infr, aid1, aid2, state):
        infr.user_feedback['aid1'].append(aid1)
        infr.user_feedback['aid2'].append(aid2)
        if state == 'match':
            infr.user_feedback['p_match'].append(1.0)
            infr.user_feedback['p_nomatch'].append(0.0)
            infr.user_feedback['p_notcomp'].append(0.0)
        elif state == 'nonmatch':
            infr.user_feedback['p_match'].append(0.0)
            infr.user_feedback['p_nomatch'].append(1.0)
            infr.user_feedback['p_notcomp'].append(0.0)
        #infr.apply_feedback()

    def get_feedback_probs(infr):
        user_feedback = ut.map_dict_vals(np.array, infr.user_feedback)
        print('user_feedback = %r' % (user_feedback,))

        aid_pairs = np.vstack([user_feedback['aid1'],
                               user_feedback['aid2']]).T
        aid_pairs = vt.atleast_nd(aid_pairs, 2, tofront=True)
        edge_ids = vt.get_undirected_edge_ids(aid_pairs)
        unique_ids, groupxs = ut.group_indices(edge_ids)

        # Resolve duplicate reviews
        pair_groups = vt.apply_grouping(aid_pairs, groupxs)
        unique_pairs = ut.take_column(pair_groups, 0)

        p_match = np.array([g[-1] for g in vt.apply_grouping(user_feedback['p_match'], groupxs)])
        #p_nomatch = [g[-1] for g in vt.apply_grouping(user_feedback['p_nomatch'], groupxs)]
        p_notcomp = np.array([g[-1] for g in vt.apply_grouping(user_feedback['p_notcomp'], groupxs)])

        p_bg = 0.5  # Needs to be thresh value
        part1 = p_match * (1 - p_notcomp)
        part2 = p_bg * p_notcomp
        p_same = part1 + part2
        return p_same, unique_pairs

    def apply_feedback(infr):
        graph = infr.graph
        user_feedback = ut.map_dict_vals(np.array, infr.user_feedback)
        print('user_feedback = %r' % (user_feedback,))

        aid_pairs = np.vstack([user_feedback['aid1'],
                               user_feedback['aid2']]).T
        aid_pairs = vt.atleast_nd(aid_pairs, 2, tofront=True)
        edge_ids = vt.get_undirected_edge_ids(aid_pairs)
        unique_ids, groupxs = ut.group_indices(edge_ids)

        # Resolve duplicate reviews
        pair_groups = vt.apply_grouping(aid_pairs, groupxs)
        unique_pairs = ut.take_column(pair_groups, 0)

        p_match = np.array([g[-1] for g in vt.apply_grouping(user_feedback['p_match'], groupxs)])
        #p_nomatch = [g[-1] for g in vt.apply_grouping(user_feedback['p_nomatch'], groupxs)]
        p_notcomp = np.array([g[-1] for g in vt.apply_grouping(user_feedback['p_notcomp'], groupxs)])

        p_bg = 0.5  # Needs to be thresh value
        part1 = p_match * (1 - p_notcomp)
        part2 = p_bg * p_notcomp
        p_same = part1 + part2

        unique_pairs = [(u, v) if graph.has_edge(u, v) else (v, u) for u, v in unique_pairs]
        #flags = [not graph.has_edge(u, v) for u, v in unique_pairs]
        #for aid1, aid2 in ut.compress(unique_pairs, flags):
        #    graph.add_edge(aid1, aid2)
        #nx.set_edge_attrs(graph, 'style',
        #                  {e: 'invis' if s <= .5 else None for s, e in zip(p_same, unique_pairs)})
        #nx.set_edge_attrs(graph, 'constraint',
        #                  {e: s > .5 for s, e in zip(p_same, unique_pairs)})

        for (aid1, aid2), groupx in zip(unique_pairs, groupxs):
            p_match = user_feedback['p_match'].take(groupx).mean()
            p_notcomp = user_feedback['p_notcomp'].take(groupx).mean()
            #p_nomatch = user_feedback['p_nomatch'].take(groupx).mean()

            part1 = p_match * (1 - p_notcomp)
            part2 = p_bg * p_notcomp
            p_same = part1 + part2

            if p_same > .5:
                if not graph.has_edge(aid1, aid2):
                    if not graph.has_edge(aid2, aid1):
                        graph.add_edge(aid1, aid2)
            else:
                if graph.has_edge(aid1, aid2) or graph.has_edge(aid1, aid2):
                    graph.remove_edge(aid1, aid2)

    def initialize_graph(infr):
        infr.graph = graph = nx.DiGraph()
        graph.add_nodes_from(infr.aids)

        node2_aid = {aid: aid for aid in infr.aids}
        node2_nid = {aid: nid for aid, nid in
                     zip(infr.aids, infr.orig_name_labels)}
        assert len(node2_nid) == len(node2_aid), '%r - %r' % (
            len(node2_nid), len(node2_aid))
        nx.set_node_attrs(graph, 'aid', node2_aid)
        nx.set_node_attrs(graph, 'name_label', node2_nid)

        infr.ensure_name_labels_are_connected_mst(infr.graph)

        infr.initialize_visual_node_attrs()

    def initialize_visual_node_attrs(infr):
        import networkx as nx
        #import plottool as pt
        graph = infr.graph
        node_to_aid = nx.get_node_attrs(graph, 'aid')
        nodes = list(graph.nodes())
        aid_list = [node_to_aid.get(node, node) for node in nodes]
        #aid_list = sorted(list(graph.nodes()))
        imgpath_list = infr.ibs.depc_annot.get('chips', aid_list, 'img',
                                               config=dict(dim_size=200),
                                               read_extern=False)
        nx.set_node_attrs(graph, 'framewidth', 3.0)
        #nx.set_node_attrs(graph, 'framecolor', pt.DARK_BLUE)
        nx.set_node_attrs(graph, 'shape', 'rect')
        nx.set_node_attrs(graph, 'image', dict(zip(nodes, imgpath_list)))
        infr.update_graph_visual_attributes()

    def update_graph_visual_attributes(infr):
        import plottool as pt
        edge2_weight = nx.get_edge_attrs(infr.graph, 'weight')
        #edge2_weight = nx.get_edge_attrs(infr.graph, 'score')
        edges = list(edge2_weight.keys())
        edge_weights = np.array(list(edge2_weight.values()))
        edge_colors = pt.scores_to_color(edge_weights, cmap_='viridis')
        #import utool
        #utool.embed()
        #edge_colors = [pt.color_funcs.ensure_base255(color) for color in edge_colors]
        #print('edge_colors = %r' % (edge_colors,))
        nx.set_edge_attrs(infr.graph, 'color', dict(zip(edges, edge_colors)))
        ut.color_nodes(infr.graph, labelattr='name_label')

    def ensure_name_labels_are_connected_mst(infr, graph):
        """
        Use minimum spannning tree to ensure all names are connected
        """
        import networkx as nx
        node2_label = nx.get_node_attrs(graph, 'name_label')
        label2_nodes = ut.group_items(node2_label.keys(), node2_label.values())

        # TODO: allow mst edges to change
        mst_edges = nx.get_edge_attrs(graph, '_mst_edge')
        graph.remove_edges_from(mst_edges.keys())

        aug_graph = graph.copy().to_undirected()

        # Find that don't exist in the original graph
        nodes_list = list(label2_nodes.values())
        unflat_edges = [list(ut.product(nodes, nodes)) for nodes in nodes_list]
        node_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
        orig_edges = ut.setdiff_ordered(aug_graph.edges(), mst_edges.keys())
        new_edges = ut.setdiff_ordered(node_pairs, orig_edges)

        preweighted_edges = nx.get_edge_attrs(aug_graph, 'weight')
        orig_edges = ut.setdiff(aug_graph.edges(), list(preweighted_edges.keys()))

        # randomness gets rid of all notdes connecting to one
        # visually looks better
        rng = np.random.RandomState(42)

        aug_graph.add_edges_from(new_edges)
        # Ensure the largest possible set of original edges is in the MST
        nx.set_edge_attributes(aug_graph, 'weight',
                               dict([(edge, 1.0 + rng.randint(1, 100))
                                     for edge in new_edges]))
        nx.set_edge_attributes(aug_graph, 'weight',
                               dict([(edge, 0.1) for edge in orig_edges]))
        for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
            mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
            for edge in mst_sub_graph.edges():
                redge = edge[::-1]
                if not (graph.has_edge(*edge) or graph.has_edge(*redge)):
                    attr_dict = {'p_comp': 0.0, '_mst_edge': True}
                    graph.add_edge(*redge, attr_dict=attr_dict)
        return graph

    def exec_split_check(infr):
        #from ibeis.algo.hots import graph_iden
        ibs = infr.ibs
        aid_list = infr.aids
        cfgdict = {
            'can_match_samename': True,
            'K': 3,
            'Knorm': 3,
            'prescore_method': 'csum',
            'score_method': 'csum'
        }
        # TODO: use current nids
        qreq_ = ibs.new_query_request(aid_list, aid_list, cfgdict=cfgdict)
        cm_list = qreq_.execute()
        infr.cm_list = cm_list
        infr.qreq_ = qreq_
        #infr = graph_iden.AnnotInference(qreq_, cm_list)
        #infr.initialize_graph_and_model()
        #print("BUILT SPLIT GRAPH")
        #return infr

    def score_edges(infr):
        print('score edges')
        infr.exec_split_check()
        #import networkx as nx
        #import itertools
        cm_list = infr.cm_list
        qaid_list = [cm.qaid for cm in cm_list]

        # Construct K-broken graph
        edges = []
        edge_scores = []
        #top = (infr.qreq_.qparams.K + 1) * 2
        #top = (infr.qreq_.qparams.K) * 2
        top = (infr.qreq_.qparams.K + 2)
        for count, cm in enumerate(cm_list):
            score_list = cm.annot_score_list
            sortx = ut.argsort(score_list)[::-1]
            score_list = ut.take(score_list, sortx)[:top]
            daid_list = ut.take(cm.daid_list, sortx)[:top]
            for score, daid in zip(score_list, daid_list):
                if daid not in qaid_list:
                    continue
                edge_scores.append(score)
                edges.append((cm.qaid, daid))

        # Maybe just K-break graph here?
        # Condense graph?

        # make symmetric
        directed_edges = dict(zip(edges, edge_scores))
        # Find edges that point in both directions
        undirected_edges = {}
        for (u, v), w in directed_edges.items():
            if (v, u) in undirected_edges:
                undirected_edges[(v, u)] += w
                undirected_edges[(v, u)] /= 2
            else:
                undirected_edges[(u, v)] = w

        edges = list(undirected_edges.keys())
        edge_scores = np.array(list(undirected_edges.values()))

        # Create match-based graph structure
        infr.graph.add_edges_from(edges)
        nx.set_edge_attrs(infr.graph, 'score', dict(zip(edges, edge_scores)))
        #nx.set_edge_attrs(infr.graph, 'weight', dict(zip(edges, edge_scores)))

        #import scipy.special
        #a = 1.5
        #b = 2
        #p_same = scipy.special.expit(b * edge_scores - a)
        #confidence = (2 * np.abs(0.5 - p_same)) ** 2

    def weight_edges(infr):
        #graph = infr.graph
        ut.nx_delete_edge_attr(infr.graph, 'weight')
        ut.nx_delete_edge_attr(infr.graph, 'lw')
        edge2_score = nx.get_edge_attrs(infr.graph, 'score')
        edges = list(infr.graph.edges())
        edge_scores = np.array(ut.dict_take(edge2_score, edges))

        control_points = [
            (0.0, .001),
            (3.0, .05),
            (15.0, .95),
            (None, .99),
        ]
        edge_weights = edge_scores.copy()
        for (pt1, prob1), (pt2, prob2) in ut.itertwo(control_points):
            if pt1 is None:
                pt1 = edge_scores.min()
            if pt2 is None:
                pt2 = edge_scores.max() + .0001
            pt_len = pt2 - pt1
            prob_len = prob2 - prob1
            flag = np.logical_and(edge_scores >= pt1, edge_scores < pt2)
            edge_weights[flag] = (((edge_scores[flag] - pt1) / pt_len) * prob_len) + prob1

        nx.set_edge_attrs(infr.graph, 'weight', dict(zip(edges, edge_weights)))
        p_same, unique_pairs = infr.get_feedback_probs()
        unique_pairs = [tuple(x.tolist()) for x in unique_pairs]
        for aid1, aid2 in unique_pairs:
            if not infr.graph.has_edge(aid1, aid2):
                infr.graph.add_edge(aid1, aid2)
        nx.set_edge_attrs(infr.graph, 'weight', dict(zip(unique_pairs, p_same)))
        nx.set_edge_attrs(infr.graph, 'lw', dict(zip(unique_pairs, [4.0] * len(unique_pairs))))

        """
        pt.plot(sorted(edge_weights))
        pt.plot(sorted(vt.norm01(edge_scores)))
        """
        edge_weights
        pass

    def infer_cut(infr, min_labels=1, max_labels=5):
        from ibeis.algo.hots import graph_iden
        infr.score_edges()
        model = graph_iden.InfrModel(infr.graph)
        infr.model = model
        model = infr.model
        labeling, params = model.run_inference2(min_labels=min_labels,
                                                max_labels=max_labels)

        graph = infr.graph.copy()
        nx.set_node_attrs(graph, 'name_label', model.node_to_label)
        cut_edges = infr.model.get_cut_edges()

        print('cut_edges = %r' % (cut_edges,))
        nx.set_edge_attrs(graph, 'implicit', {edge: True for edge in cut_edges})
        nx.set_edge_attrs(graph, 'color', {edge: 'red' for edge in cut_edges})
        #for (u, v) in cut_edges:
        #    graph.remove_edge(u, v)
        infr.ensure_name_labels_are_connected_mst(graph)
        infr.graph = graph
        infr.update_graph_visual_attributes()
        return graph

    def reset_cut(infr):
        infr.initialize_graph()


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInference(object):
    """
    Make name inferences about a series of AnnotMatches

    CommandLine:
        python -m ibeis.algo.hots.graph_iden AnnotInference --show --no-cnn

    Example:
        >>> from ibeis.algo.hots.graph_iden import *  # NOQA
        >>> import ibeis
        >>> #qreq_ = ibeis.testdata_qreq_(default_qaids=[1, 2, 3, 4], default_daids=[2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> # a='default:dsize=20,excluderef=True,qnum_names=5,min_pername=3,qsample_per_name=1,dsample_per_name=2',
        >>> a='default:dsize=20,excluderef=True,qnum_names=5,qsize=1,min_pername=3,qsample_per_name=1,dsample_per_name=2'
        >>> qreq_ = ibeis.testdata_qreq_(defaultdb='PZ_MTEST', a=a, verbose=0, use_cache=False)
        >>> # a='default:dsize=2,qsize=1,excluderef=True,qnum_names=5,min_pername=3,qsample_per_name=1,dsample_per_name=2',
        >>> ibs = qreq_.ibs
        >>> cm_list = qreq_.execute()
        >>> self1 = AnnotInference(qreq_, cm_list)
        >>> inf_dict1 = self1.make_annot_inference_dict(True)
        >>> user_feedback =  self1.simulate_user_feedback()
        >>> self2 = AnnotInference(qreq_, cm_list, user_feedback)
        >>> inf_dict2 = self2.make_annot_inference_dict(True)
        >>> print('inference_dict = ' + ut.repr3(inf_dict1, nl=3))
        >>> print('inference_dict2 = ' + ut.repr3(inf_dict2, nl=3))
        >>> ut.quit_if_noshow()
        >>> graph1 = self1.make_graph(show=True)
        >>> graph2 = self2.make_graph(show=True)
        >>> ut.show_if_requested()
    """

    def __init__(infr, qreq_, cm_list, user_feedback=None):
        infr.qreq_ = qreq_
        infr.cm_list = cm_list
        infr.needs_review_list = []
        infr.cluster_tuples = []
        infr.user_feedback = user_feedback
        infr.make_inference()

    def simulate_user_feedback(infr):
        qreq_ = infr.qreq_
        aid_pairs = np.array(ut.take_column(infr.needs_review_list, [0, 1]))
        nid_pairs = qreq_.ibs.get_annot_nids(aid_pairs)
        truth = nid_pairs.T[0] == nid_pairs.T[1]
        user_feedback = ut.odict([
            ('aid1', aid_pairs.T[0]),
            ('aid2', aid_pairs.T[1]),
            ('p_match', truth.astype(np.float)),
            ('p_nomatch', 1.0 - truth),
            ('p_notcomp', np.array([0.0] * len(aid_pairs))),
        ])
        return user_feedback

    def make_prob_annots(infr):
        cm_list = infr.cm_list
        unique_aids = sorted(ut.list_union(*[cm.daid_list for cm in cm_list] +
                                           [[cm.qaid for cm in cm_list]]))
        aid2_aidx = ut.make_index_lookup(unique_aids)
        prob_annots = np.zeros((len(unique_aids), len(unique_aids)))
        for count, cm in enumerate(cm_list):
            idx = aid2_aidx[cm.qaid]
            annot_scores = ut.dict_take(cm.aid2_annot_score, unique_aids, 0)
            prob_annots[idx][:] = annot_scores
        prob_annots[np.diag_indices(len(prob_annots))] = np.inf
        prob_annots += 1E-9
        #print(ut.hz_str('prob_names = ', ut.array2string2(prob_names,
        #precision=2, max_line_width=140, suppress_small=True)))
        return unique_aids, prob_annots

    @ut.memoize
    def make_prob_names(infr):
        cm_list = infr.cm_list
        # Consolodate information from a series of chip matches
        unique_nids = sorted(ut.list_union(*[cm.unique_nids for cm in cm_list]))
        #nid2_nidx = ut.make_index_lookup(unique_nids)
        # Populate matrix of raw name scores
        prob_names = np.zeros((len(cm_list), len(unique_nids)))
        for count, cm in enumerate(cm_list):
            try:
                name_scores = ut.dict_take(cm.nid2_name_score, unique_nids, 0)
            except AttributeError:
                unique_nidxs = ut.take(cm.nid2_nidx, unique_nids)
                name_scores = ut.take(cm.name_score_list, unique_nidxs)
            prob_names[count][:] = name_scores

        # Normalize to row stochastic matrix
        prob_names /= prob_names.sum(axis=1)[:, None]
        #print(ut.hz_str('prob_names = ', ut.array2string2(prob_names,
        #precision=2, max_line_width=140, suppress_small=True)))
        return unique_nids, prob_names

    def choose_thresh(infr):
        #prob_annots /= prob_annots.sum(axis=1)[:, None]
        # Find connected components
        #thresh = .25
        #thresh = 1 / (1.2 * np.sqrt(prob_names.shape[1]))
        unique_nids, prob_names = infr.make_prob_names()

        if len(unique_nids) <= 2:
            return .5

        nscores = np.sort(prob_names.flatten())
        # x = np.gradient(nscores).argmax()
        # x = (np.gradient(np.gradient(nscores)) ** 2).argmax()
        # thresh = nscores[x]

        curve = nscores
        idx1 = vt.find_elbow_point(curve)
        idx2 = vt.find_elbow_point(curve[idx1:]) + idx1
        if False:
            import plottool as pt
            idx3 = vt.find_elbow_point(curve[idx1:idx2 + 1]) + idx1
            pt.plot(curve)
            pt.plot(idx1, curve[idx1], 'bo')
            pt.plot(idx2, curve[idx2], 'ro')
            pt.plot(idx3, curve[idx3], 'go')
        thresh = nscores[idx2]
        #print('thresh = %r' % (thresh,))
        #thresh = .999
        #thresh = .1
        return thresh

    def make_graph(infr, show=False):
        import networkx as nx
        import itertools
        cm_list = infr.cm_list
        unique_nids, prob_names = infr.make_prob_names()
        thresh = infr.choose_thresh()

        # Simply cut any edge with a weight less than a threshold
        qaid_list = [cm.qaid for cm in cm_list]
        postcut = prob_names > thresh
        qxs, nxs = np.where(postcut)
        if False:
            kw = dict(precision=2, max_line_width=140, suppress_small=True)
            print(ut.hz_str('prob_names = ', ut.array2string2((prob_names), **kw)))
            print(ut.hz_str('postcut = ', ut.array2string2((postcut).astype(np.int), **kw)))
        matching_qaids = ut.take(qaid_list, qxs)
        matched_nids = ut.take(unique_nids, nxs)

        qreq_ = infr.qreq_

        nodes = ut.unique(qreq_.qaids.tolist() + qreq_.daids.tolist())
        if not hasattr(qreq_, 'dnids'):
            qreq_.dnids = qreq_.ibs.get_annot_nids(qreq_.daids)
            qreq_.qnids = qreq_.ibs.get_annot_nids(qreq_.qaids)
        dnid2_daids = ut.group_items(qreq_.daids, qreq_.dnids)
        grouped_aids = dnid2_daids.values()
        matched_daids = ut.take(dnid2_daids, matched_nids)
        name_cliques = [list(itertools.combinations(aids, 2)) for aids in grouped_aids]
        aid_matches = [list(ut.product([qaid], daids)) for qaid, daids in
                       zip(matching_qaids, matched_daids)]

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(ut.flatten(name_cliques))
        graph.add_edges_from(ut.flatten(aid_matches))

        #matchless_quries = ut.take(qaid_list, ut.index_complement(qxs, len(qaid_list)))
        name_nodes = [('nid', l) for l in qreq_.dnids]
        db_aid_nid_edges = list(zip(qreq_.daids, name_nodes))
        #query_aid_nid_edges = list(zip(matching_qaids, [('nid', l) for l in matched_nids]))
        #G = nx.Graph()
        #G.add_nodes_from(matchless_quries)
        #G.add_edges_from(db_aid_nid_edges)
        #G.add_edges_from(query_aid_nid_edges)

        graph.add_edges_from(db_aid_nid_edges)

        if infr.user_feedback is not None:
            user_feedback = ut.map_dict_vals(np.array, infr.user_feedback)
            p_bg = 0.0
            part1 = user_feedback['p_match'] * (1 - user_feedback['p_notcomp'])
            part2 = p_bg * user_feedback['p_notcomp']
            p_same_list = part1 + part2
            for aid1, aid2, p_same in zip(user_feedback['aid1'],
                                          user_feedback['aid2'], p_same_list):
                if p_same > .5:
                    if not graph.has_edge(aid1, aid2):
                        graph.add_edge(aid1, aid2)
                else:
                    if graph.has_edge(aid1, aid2):
                        graph.remove_edge(aid1, aid2)
        if show:
            import plottool as pt
            nx.set_node_attributes(graph, 'color', {aid: pt.LIGHT_PINK
                                                    for aid in qreq_.daids})
            nx.set_node_attributes(graph, 'color', {aid: pt.TRUE_BLUE
                                                    for aid in qreq_.qaids})
            nx.set_node_attributes(graph, 'color', {
                aid: pt.LIGHT_PURPLE
                for aid in np.intersect1d(qreq_.qaids, qreq_.daids)})
            nx.set_node_attributes(graph, 'label', {node: 'n%r' % (node[1],)
                                                    for node in name_nodes})
            nx.set_node_attributes(graph, 'color', {node: pt.LIGHT_GREEN
                                                    for node in name_nodes})
        if show:
            import plottool as pt
            pt.show_nx(graph, layoutkw={'prog': 'neato'}, verbose=False)
        return graph

    def make_clusters(infr):
        import itertools
        import networkx as nx
        cm_list = infr.cm_list

        graph = infr.make_graph()

        # hack for orig aids
        orig_aid2_nid = {}
        for cm in cm_list:
            orig_aid2_nid[cm.qaid] = cm.qnid
            for daid, dnid in zip(cm.daid_list, cm.dnid_list):
                orig_aid2_nid[daid] = dnid

        cluster_aids = []
        cluster_nids = []
        connected = list(nx.connected_components(graph))
        for comp in connected:
            cluster_nids.append([])
            cluster_aids.append([])
            for x in comp:
                if isinstance(x, tuple):
                    cluster_nids[-1].append(x[1])
                else:
                    cluster_aids[-1].append(x)

        # Make first part of inference output
        qaid_list = [cm.qaid for cm in cm_list]
        qaid_set = set(qaid_list)
        #start_nid = 9001
        # Find an nid that doesn't exist in the database
        start_nid = len(infr.qreq_.ibs._get_all_known_name_rowids()) + 1
        next_new_nid = itertools.count(start_nid)
        cluster_tuples = []
        for aids, nids in zip(cluster_aids, cluster_nids):
            other_nid_clusters = cluster_nids[:]
            other_nid_clusters.remove(nids)
            other_nids = ut.flatten(other_nid_clusters)
            split_case = len(ut.list_intersection(other_nids, nids)) > 0
            merge_case = len(nids) > 1
            new_name = len(nids) == 0

            #print('[chip_match > AnnotInference > make_inference] WARNING:
            #      EXEMPLAR FLAG SET TO TRUE, NEEDS TO BE IMPLEMENTED')
            error_flag = (split_case << 1) + (merge_case << 2) + (new_name << 3)
            strflags = ['split', 'merge', 'new']
            error_flag = ut.compress(strflags, [split_case, merge_case, new_name])
            #error_flag = split_case or merge_case

            # <HACK>
            # SET EXEMPLARS
            ibs = infr.qreq_.ibs
            viewpoint_texts = ibs.get_annot_yaw_texts(aids)
            view_to_aids = ut.group_items(aids, viewpoint_texts)
            num_wanted_exemplars_per_view = 4
            hack_set_these_qaids_as_exemplars = set([])
            for view, aids_ in view_to_aids.items():
                heuristic_exemplar_aids = set(aids) - qaid_set
                heuristic_non_exemplar_aids = set(aids).intersection(qaid_set)
                num_needed_exemplars = (num_wanted_exemplars_per_view -
                                        len(heuristic_exemplar_aids))
                # Choose the best query annots to fill out exemplars
                if len(heuristic_non_exemplar_aids) == 0:
                    continue
                quality_ints = ibs.get_annot_qualities(heuristic_non_exemplar_aids)
                okish = ibs.const.QUALITY_TEXT_TO_INT[ibs.const.QUAL_OK] - .1
                quality_ints = [x if x is None else okish for x in quality_ints]
                aids_ = ut.sortedby(heuristic_non_exemplar_aids, quality_ints)[::-1]
                chosen = aids_[:num_needed_exemplars]
                for qaid_ in chosen:
                    hack_set_these_qaids_as_exemplars.add(qaid_)
            # </HACK>
            if not error_flag and not new_name:
                new_nid = nids[0]
            else:
                new_nid = six.next(next_new_nid)
            for aid in aids:
                if aid not in qaid_set:
                    if len(error_flag) == 0:
                        continue
                orig_nid = orig_aid2_nid[aid]
                exemplar_flag = aid in hack_set_these_qaids_as_exemplars
                #clusters is list 4 tuple: (aid, orig_name_uuid, new_name_uuid, error_flag)
                tup = (aid, orig_nid, new_nid, exemplar_flag, error_flag)
                cluster_tuples.append(tup)
        return cluster_tuples

    def make_inference(infr):
        cm_list = infr.cm_list
        unique_nids, prob_names = infr.make_prob_names()
        cluster_tuples = infr.make_clusters()

        # Make pair list for output
        if infr.user_feedback is not None:
            keys = list(zip(infr.user_feedback['aid1'], infr.user_feedback['aid2']))
            feedback_lookup = ut.make_index_lookup(keys)
            user_feedback = infr.user_feedback
            p_bg = 0
            user_feedback = ut.map_dict_vals(np.array, infr.user_feedback)
            part1 = user_feedback['p_match'] * (1 - user_feedback['p_notcomp'])
            part2 = p_bg * user_feedback['p_notcomp']
            p_same_list = part1 + part2
        else:
            feedback_lookup = {}
        infr.user_feedback
        needs_review_list = []
        num_top = 4
        for cm, row in zip(cm_list, prob_names):
            # Find top scoring names for this chip match in the posterior distribution
            idxs = row.argsort()[::-1]
            top_idxs = idxs[:num_top]
            nids = ut.take(unique_nids, top_idxs)
            # Find the matched annotations in the pairwise prior distributions
            nidxs = ut.dict_take(cm.nid2_nidx, nids, None)
            name_groupxs = ut.take(cm.name_groupxs, ut.filter_Nones(nidxs))
            daids_list = ut.take(cm.daid_list, name_groupxs)
            for daids in daids_list:
                ut.take(cm.score_list, ut.take(cm.daid2_idx, daids))
                scores_all = cm.annot_score_list / cm.annot_score_list.sum()
                idxs = ut.take(cm.daid2_idx, daids)
                scores = scores_all.take(idxs)
                raw_scores = cm.score_list.take(idxs)
                scorex = scores.argmax()
                raw_score = raw_scores[scorex]
                daid = daids[scorex]
                import scipy.special
                # SUPER HACK: these are not probabilities
                # TODO: set a and b based on dbsize and param configuration
                # python -m plottool.draw_func2 --exec-plot_func --show --range=0,3 --func="lambda x: scipy.special.expit(2 * x - 2)"
                #a = 2.0
                a = 1.5
                b = 2
                p_same = scipy.special.expit(b * raw_score - a)
                #confidence = scores[scorex]
                #p_diff = 1 - p_same
                #decision = 'same' if confidence > thresh else 'diff'
                #confidence = p_same if confidence > thresh else p_diff
                #tup = (cm.qaid, daid, decision, confidence, raw_score)
                confidence = (2 * np.abs(0.5 - p_same)) ** 2
                #if infr.user_feedback is not None:
                #    import utool
                #    utool.embed(
                key = (cm.qaid, daid)
                fb_idx = feedback_lookup.get(key)
                if fb_idx is not None:
                    confidence = p_same_list[fb_idx]
                tup = (cm.qaid, daid, p_same, confidence, raw_score)
                needs_review_list.append(tup)

        # Sort resulting list by confidence
        sortx = ut.argsort(ut.take_column(needs_review_list, 3))
        needs_review_list = ut.take(needs_review_list, sortx)

        infr.needs_review_list = needs_review_list
        infr.cluster_tuples = cluster_tuples

        #print('needs_review_list = %s' % (ut.repr3(needs_review_list, nl=1),))
        #print('cluster_tuples = %s' % (ut.repr3(cluster_tuples, nl=1),))

        #prob_annots = None
        #print(ut.array2string2prob_names precision=2, max_line_width=100,
        #      suppress_small=True))

    def make_annot_inference_dict(infr, internal=False):
        #import uuid

        def convert_to_name_uuid(nid):
            #try:
            text = ibs.get_name_texts(nid, apply_fix=False)
            if text is None:
                text = 'NEWNAME_%s' % (str(nid),)
            #uuid_ = uuid.UUID(text)
            #except ValueError:
            #    text = 'NEWNAME_%s' % (str(nid),)
            #    #uuid_ = nid
            return text
        ibs = infr.qreq_.ibs

        if internal:
            get_annot_uuids = ut.identity
        else:
            get_annot_uuids = ibs.get_annot_uuids
            #return uuid_

        # Compile the cluster_dict
        col_list = ['aid_list', 'orig_nid_list', 'new_nid_list',
                    'exemplar_flag_list', 'error_flag_list']
        cluster_dict = dict(zip(col_list, ut.listT(infr.cluster_tuples)))
        cluster_dict['annot_uuid_list'] = get_annot_uuids(cluster_dict['aid_list'])
        # We store the name's UUID as the name's text
        #cluster_dict['orig_name_uuid_list'] = [convert_to_name_uuid(nid)
        #                                       for nid in cluster_dict['orig_nid_list']]
        #cluster_dict['new_name_uuid_list'] = [convert_to_name_uuid(nid)
        # for nid in cluster_dict['new_nid_list']]
        cluster_dict['orig_name_list'] = [convert_to_name_uuid(nid)
                                          for nid in cluster_dict['orig_nid_list']]
        cluster_dict['new_name_list'] = [convert_to_name_uuid(nid)
                                         for nid in cluster_dict['new_nid_list']]
        # Filter out only the keys we want to send back in the dictionary
        #key_list = ['annot_uuid_list', 'orig_name_uuid_list',
        #            'new_name_uuid_list', 'exemplar_flag_list',
        #            'error_flag_list']
        key_list = ['annot_uuid_list', 'orig_name_list', 'new_name_list',
                    'exemplar_flag_list', 'error_flag_list']
        cluster_dict = ut.dict_subset(cluster_dict, key_list)

        # Compile the annot_pair_dict
        col_list = ['aid_1_list', 'aid_2_list', 'p_same_list',
                    'confidence_list', 'raw_score_list']
        annot_pair_dict = dict(zip(col_list, ut.listT(infr.needs_review_list)))
        annot_pair_dict['annot_uuid_1_list'] = get_annot_uuids(annot_pair_dict['aid_1_list'])
        annot_pair_dict['annot_uuid_2_list'] = get_annot_uuids(annot_pair_dict['aid_2_list'])
        zipped = zip(annot_pair_dict['annot_uuid_1_list'],
                     annot_pair_dict['annot_uuid_2_list'],
                     annot_pair_dict['p_same_list'])
        annot_pair_dict['review_pair_list'] = [
            {
                'annot_uuid_key'       : annot_uuid_1,
                'annot_uuid_1'         : annot_uuid_1,
                'annot_uuid_2'         : annot_uuid_2,
                'prior_matching_state' : {
                    'p_match'   : p_same,
                    'p_nomatch' : 1.0 - p_same,
                    'p_notcomp' : 0.0,
                }
            }
            for (annot_uuid_1, annot_uuid_2, p_same) in zipped
        ]
        # Filter out only the keys we want to send back in the dictionary
        key_list = ['review_pair_list', 'confidence_list']
        annot_pair_dict = ut.dict_subset(annot_pair_dict, key_list)

        # Compile the inference dict
        inference_dict = ut.odict([
            ('cluster_dict', cluster_dict),
            ('annot_pair_dict', annot_pair_dict),
            ('_internal_state', None),
        ])
        return inference_dict


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.graph_iden
        python -m ibeis.algo.hots.graph_iden --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
