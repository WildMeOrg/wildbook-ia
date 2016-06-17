# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt  # NOQA
import six
print, rrr, profile = ut.inject2(__name__, '[graph_inference]')


@six.add_metaclass(ut.ReloadingMetaclass)
class InfrModel(ut.NiceRepr):

    def __init__(model, graph):
        #def __init__(model, n_nodes, edges, edge_weights=None, n_labels=None,
        model.graph = graph
        model._update_state()

    def _update_state(model):
        import networkx as nx
        nodes = sorted(list(model.graph.nodes()))
        edges = list(model.graph.edges())
        edge2_weights = nx.get_edge_attributes(model.graph, 'weight')
        node2_labeling = nx.get_node_attributes(model.graph, 'name_label')
        edge_weights = ut.take(edge2_weights, edges)
        labeling = ut.take(node2_labeling, nodes)
        n_nodes = len(nodes)
        # Model state
        model.n_nodes = n_nodes
        model.edges = edges
        model.edge_weights = edge_weights
        # Model parameters
        model.labeling = np.zeros(model.n_nodes, dtype=np.int32)
        model._update_labels(labeling=labeling)
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

    def run_inference(model, thresh=None, n_labels=None, n_iter=5, algorithm='expansion'):
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

    def run_inference2(model, max_labels=5):
        cut_params = ut.all_dict_combinations({
            'n_labels': list(range(1, max_labels)),
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
        labeling = model.run_inference(**params)
        return labeling, params

    def update_graph(model):
        uv_list = np.array(list(model.graph.edges()))
        u_labels = model.labeling[uv_list.T[0]]
        v_labels = model.labeling[uv_list.T[1]]
        graph_ = model.graph.copy()
        # Remove edges between all annotations with different labels
        cut_edges = uv_list[u_labels != v_labels]
        for (u, v) in cut_edges:
            graph_.remove_edge(u, v)
        #list(nx.connected_components(graph_.to_undirected()))
        return graph_


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

    def initialize_graph_and_model(infr):
        """ Unused in internal split stuff

        pt.qt4ensure()
        layout_info = pt.show_nx(graph, as_directed=False, fnum=1,
                                 layoutkw=dict(prog='neato'), use_image=True,
                                 verbose=0)
        ax = pt.gca()
        pt.zoom_factory()
        pt.interactions.PanEvents()
        """
        #import networkx as nx
        #import itertools
        cm_list = infr.cm_list
        hack = True
        hack = False
        if hack:
            cm_list = cm_list[:10]
        qaid_list = [cm.qaid for cm in cm_list]
        daids_list = [cm.daid_list for cm in cm_list]
        unique_aids = sorted(ut.list_union(*daids_list + [qaid_list]))
        if hack:
            unique_aids = sorted(ut.isect(unique_aids, qaid_list))
        aid2_aidx = ut.make_index_lookup(unique_aids)

        # Construct K-broken graph
        edges = []
        edge_weights = []
        #top = (infr.qreq_.qparams.K + 1) * 2
        #top = (infr.qreq_.qparams.K) * 2
        top = (infr.qreq_.qparams.K + 2)
        for count, cm in enumerate(cm_list):
            qidx = aid2_aidx[cm.qaid]
            score_list = cm.annot_score_list
            sortx = ut.argsort(score_list)[::-1]
            score_list = ut.take(score_list, sortx)[:top]
            daid_list = ut.take(cm.daid_list, sortx)[:top]
            for score, daid in zip(score_list, daid_list):
                if daid not in qaid_list:
                    continue
                didx = aid2_aidx[daid]
                edge_weights.append(score)
                edges.append((qidx, didx))

        # make symmetric
        directed_edges = dict(zip(edges, edge_weights))
        # Find edges that point in both directions
        undirected_edges = {}
        for (u, v), w in directed_edges.items():
            if (v, u) in undirected_edges:
                undirected_edges[(v, u)] += w
                undirected_edges[(v, u)] /= 2
            else:
                undirected_edges[(u, v)] = w

        edges = list(undirected_edges.keys())
        edge_weights = list(undirected_edges.values())
        nodes = list(range(len(unique_aids)))

        nid_labeling = infr.qreq_.ibs.get_annot_nids(unique_aids)
        labeling = ut.rebase_labels(nid_labeling)

        import networkx as nx
        from ibeis.viz import viz_graph
        set_node_attrs = nx.set_node_attributes
        set_edge_attrs = nx.set_edge_attributes

        # Create match-based graph structure
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        # Important properties
        nid_list = infr.qreq_.ibs.get_annot_nids(unique_aids)
        labeling = ut.rebase_labels(nid_list)

        set_node_attrs(graph, 'name_label', dict(zip(nodes, labeling)))
        set_edge_attrs(graph, 'weight', dict(zip(edges, edge_weights)))

        # Visualization properties
        import plottool as pt
        ax2_aid = ut.invert_dict(aid2_aidx)
        set_node_attrs(graph, 'aid', ax2_aid)
        viz_graph.ensure_node_images(infr.qreq_.ibs, graph)
        set_node_attrs(graph, 'framewidth', dict(zip(nodes, [3.0] * len(nodes))))
        set_node_attrs(graph, 'framecolor', dict(zip(nodes, [pt.DARK_BLUE] * len(nodes))))
        ut.color_nodes(graph, labelattr='name_label')

        edge_colors = pt.scores_to_color(np.array(edge_weights), cmap_='viridis')
        #import utool
        #utool.embed()
        #edge_colors = [pt.color_funcs.ensure_base255(color) for color in edge_colors]
        #print('edge_colors = %r' % (edge_colors,))
        set_edge_attrs(graph, 'color', dict(zip(edges, edge_colors)))

        # Build inference model
        from ibeis.algo.hots import graph_iden
        #graph_iden.rrr()
        model = graph_iden.InfrModel(graph)
        #model = graph_iden.InfrModel(len(nodes), edges, edge_weights, labeling=labeling)
        infr.model = model

    def infer_cut(infr):
        model = infr.model
        labeling, params = model.run_inference2(max_labels=5)
        #import networkx as nx
        #from ibeis.viz import viz_graph
        graph_ = infr.model.update_graph()
        return graph_

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
