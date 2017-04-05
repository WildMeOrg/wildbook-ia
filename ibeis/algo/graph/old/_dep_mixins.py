# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import itertools as it
import collections
import vtool as vt
import utool as ut
import six
from ibeis.algo.hots import infr_model
from ibeis.algo.graph.nx_utils import e_, _dz
from ibeis.algo.graph.nx_utils import edges_inside, edges_cross
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


def filter_between_ccs_neg(aids1, aids2, aid_to_nid, nid_to_aids, isneg_flags):
    """
    If two cc's have at least 1 negative review between them, then
    remove all other potential reviews between those cc's

    CommandLine:
        python -m ibeis.algo.graph.core filter_between_ccs_neg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.graph.core import *  # NOQA
        >>> edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 2)]
        >>> aids1 = ut.take_column(edges, 0)
        >>> aids2 = ut.take_column(edges, 1)
        >>> isneg_flags = [0, 0, 1, 0, 0]
        >>> aid_to_nid = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        >>> nid_to_aids = {0: [0, 1, 2], 1: [3, 4]}
        >>> valid_flags = filter_between_ccs_neg(aids1, aids2, aid_to_nid,
        >>>                                      nid_to_aids, isneg_flags)
        >>> result = ('valid_flags = %s' % (ut.repr2(valid_flags),))
        >>> print(result)
        valid_flags = [True, True, False, True, False]
    """
    neg_aids1 = ut.compress(aids1, isneg_flags)
    neg_aids2 = ut.compress(aids2, isneg_flags)
    neg_nids1 = ut.take(aid_to_nid, neg_aids1)
    neg_nids2 = ut.take(aid_to_nid, neg_aids2)

    # Ignore inconsistent names
    # Determine which CCs photobomb each other
    invalid_nid_map = ut.ddict(set)
    for nid1, nid2 in zip(neg_nids1, neg_nids2):
        if nid1 != nid2:
            invalid_nid_map[nid1].add(nid2)
            invalid_nid_map[nid2].add(nid1)

    impossible_aid_map = ut.ddict(set)
    for nid1, other_nids in invalid_nid_map.items():
        for aid1 in nid_to_aids[nid1]:
            for nid2 in other_nids:
                for aid2 in nid_to_aids[nid2]:
                    impossible_aid_map[aid1].add(aid2)
                    impossible_aid_map[aid2].add(aid1)

    valid_flags = [aid2 not in impossible_aid_map[aid1]
                   for aid1, aid2 in zip(aids1, aids2)]
    return valid_flags


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrDepMixin(object):

    def _get_strong_positives(infr, graph, reviewed_positives, pos_diameter):
        # Reconsider edges within connected components that are
        # separated by a large distance over reviewed edges.
        strong_positives = []
        weak_positives = []
        for nid, edges in reviewed_positives.items():
            strong_edges = []
            weak_edges = []
            reviewed_subgraph = nx.Graph(edges)
            aspl = nx.all_pairs_shortest_path_length(reviewed_subgraph)
            for u, dist_dict in aspl:
                for v, dist in dist_dict.items():
                    if u <= v and graph.has_edge(u, v):
                        if dist <= pos_diameter:
                            strong_edges.append((u, v))
                        else:
                            weak_edges.append((u, v))
            weak_positives.extend(weak_edges)
            strong_positives.extend(strong_edges)
        return strong_positives

    def _get_strong_negatives(infr, graph, reviewed_positives, reviewed_negatives,
                              negative, node_to_label, nid_to_cc,
                              neg_diameter):
        # FIXME: Change the forumlation of this problem to:
        # Given two connected components, a set of potential edges,
        # and a number K Find the minimum cost set of potential
        # edges such that the maximum distance between two nodes in
        # different components is less than K.

        # distance_matrix = dict(nx.shortest_path_length(reviewed_subgraph))
        # cc1 = nid_to_cc[nid1]
        # cc2 = nid_to_cc[nid2]
        # for u in cc1:
        #     is_violated = np.array(list(ut.dict_subset(
        #         distance_matrix[u], cc2).values())) > neg_diameter
        strong_negatives = []
        weak_negatives = []

        # Reconsider edges between connected components that are
        # separated by a large distance over reviewed edges.
        for nid_edge, neg_edges in reviewed_negatives.items():
            nid1, nid2 = nid_edge
            pos_edges1 = reviewed_positives[nid1]
            pos_edges2 = reviewed_positives[nid2]
            edges = pos_edges2 + pos_edges1 + neg_edges
            reviewed_subgraph = nx.Graph(edges)
            strong_edges = []
            weak_edges = []
            unreviewed_neg_edges = negative[nid_edge]

            for u, v in unreviewed_neg_edges:
                # Ensure u corresponds to nid1 and v corresponds to nid2
                if node_to_label[u] == nid2:
                    u, v = v, u
                # Is the distance from u to any node in cc[nid2] large?
                splu = nx.shortest_path_length(reviewed_subgraph, source=u)
                for v_, dist in splu:
                    if v_ in nid_to_cc[nid2] and graph.has_edge(u, v_):
                        if dist > neg_diameter:
                            weak_edges.append(e_(u, v_))
                        else:
                            strong_edges.append(e_(u, v_))
                # Is the distance from v to any node in cc[nid1] large?
                splv = nx.shortest_path_length(reviewed_subgraph, source=v)
                for u_, dist in splv:
                    if u_ in nid_to_cc[nid1] and graph.has_edge(u_, v):
                        if dist > neg_diameter:
                            weak_edges.append(e_(u_, v))
                        else:
                            strong_edges.append(e_(u_, v))
            strong_negatives.extend(strong_edges)
            weak_negatives.extend(weak_edges)
        return strong_negatives
        # print('strong_edges.append = %r' % (strong_edges,))
        # print('weak_edges.append = %r' % (weak_edges,))

    def _update_priority_queue_older(infr, graph, positive, negative,
                               reviewed_positives, reviewed_negatives,
                               node_to_label, nid_to_cc, suggested_fix_edges,
                               other_error_edges, unreviewed_edges):
        r"""
        TODO refactor

        TODO Reformulate this as a "Graph Diameter Augmentation" problem.
        It turns out this problem is NP-hard.
        Bounded
        (BCMB Bounded Cost Minimum Diameter Edge Addition)
        https://www.cse.unsw.edu.au/~sergeg/papers/FratiGGM13isaac.pdf
        http://www.cis.upenn.edu/~sanjeev/papers/diameter.pdf

        Example:
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> from ibeis.algo.graph.core import _dz
            >>> from ibeis.algo.hots import demo
            >>> infr = demo.synthetic_infr(
            >>>     ccs=[[1, 2, 3, 4, 5],
            >>>            [6, 7, 8, 9, 10]],
            >>>     edges=[
            >>>         #(1, 6, {'decision': 'nomatch'}),
            >>>         (1, 6, {}),
            >>>         (4, 9, {}),
            >>>     ]
            >>> )
            >>> infr._init_priority_queue()
            >>> assert len(infr.queue) == 2
            >>> infr.queue_params['neg_diameter'] = None
            >>> infr.add_feedback_old(1, 6, 'nomatch', apply=True)
            >>> assert len(infr.queue) == 0
            >>> graph = infr.graph
            >>> ut.exec_func_src(infr.apply_review_inference,
            >>>                  sentinal='if neg_diameter', stop=-1, verbose=True)
            >>> infr.queue_params['neg_diameter'] = 1
            >>> infr.apply_review_inference()
        """
        # update the priority queue on the fly
        queue = infr.queue
        pos_diameter = infr.queue_params['pos_diameter']
        neg_diameter = infr.queue_params['neg_diameter']

        if pos_diameter is not None:
            strong_positives = infr._get_strong_positives(
                graph, reviewed_positives, pos_diameter)
            queue.delete_items(strong_positives)
        else:
            for edges in positive.values():
                queue.delete_items(edges)

        if neg_diameter is not None:
            strong_negatives = infr._get_strong_negatives(
                graph, reviewed_positives, reviewed_negatives, negative,
                node_to_label, nid_to_cc, neg_diameter)
            queue.delete_items(strong_negatives)
        else:
            for edges in negative.values():
                queue.delete_items(edges)

        # Add error edges back in with higher priority
        queue.update(zip(suggested_fix_edges,
                         -infr._get_priorites(suggested_fix_edges)))

        queue.delete_items(other_error_edges)

        needs_priority = [e for e in unreviewed_edges if e not in queue]
        # assert not needs_priority, (
        #     'shouldnt need this needs_priority=%r ' % (needs_priority,))
        queue.update(zip(needs_priority, -infr._get_priorites(needs_priority)))


    def break_graph(infr, num):
        """
        This is the b-matching problem and is P-time solvable.
        This problem is equivalent to bidirectional flow.

        References:
            http://www.ams.sunysb.edu/~jsbm/papers/b-matching.pdf

        # given (graph, K):
        # Let x[e] be 1 if we keep an edge e and 0 if we cut it

        # Keep the best set of edges for each node
        maximize
            sum(d['weight'] * x[(u, v)]
                for u in graph.nodes()
                for v, d in graph.node[u].items())

        # The degree of each node must be less than K
        subject to
            all(
                sum(x[(u, v)] for v in graph.node[u]) <= K
                for u in graph.nodes()
            )

            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching()
            >>> infr.apply_match_edges()
            >>> infr.apply_match_scores()
            >>> infr.ensure_full()

        The linear program is based on the blossom algorithm

        implicit summation: b(W) = sum(b_v for v in W)

        Let omega = [S \subset V where len(S) >= 3 and abs(sum(v_b for v in S)) % 2 == 1]

        Let q_S = .5 * sum(v_b for v in S) - 1 for S in omega

        For each W \subset V
        Let delta(W) be the set of edges that meet exactly one node in W
        Let gamma(W) be the set of edges with both endpoints in W

        maximize c.dot(x)
        subject to x(delta(v)) = b_v forall v in V
        x_e >= 0 forall e in E
        x(gamma(S)) <= q_S foall S in omega

        """
        # prev_degrees = np.array([infr.graph.degree(n) for n in infr.graph.nodes()])

        weight = 'normscore'
        if len(infr.graph) < 100:
            # Ineffcient but exact integer programming solution
            K = num
            graph = infr.graph
            import pulp
            # Formulate integer program
            prob = pulp.LpProblem("B-Matching", pulp.LpMaximize)
            # Solution variables
            indexs = [e_(*e) for e in graph.edges()]
            # cat = pulp.LpContinuous
            cat = pulp.LpInteger
            x = pulp.LpVariable.dicts(name='x', indexs=indexs,
                                      lowBound=0, upBound=1, cat=cat)
            # maximize objective function
            prob.objective = sum(d.get(weight, 0) * x[e_(u, v)]
                                 for u in graph.nodes()
                                 for v, d in graph.edge[u].items())
            # subject to
            for u in graph.nodes():
                prob.add(sum(x[e_(u, v)] for v in graph.edge[u]) <= K)
            # Solve using with solver like CPLEX, GLPK, or SCIP.
            #pulp.CPLEX().solve(prob)
            pulp.PULP_CBC_CMD().solve(prob)
            # Read solution
            xvalues = [x[e].varValue for e in indexs]
            to_remove = [e for e, xval in zip(indexs, xvalues)
                         if not xval == 1.0]
            graph.remove_edges_from(to_remove)
        else:
            # Hacky solution. TODO: implement b-matching using blossom with
            # networkx
            to_remove = set([])
            # nodes = infr.graph.nodes()
            # degrees = np.array([infr.graph.degree(n) for n in infr.graph.nodes()])
            for u in infr.graph.nodes():
                if len(infr.graph[u]) > num:
                    edges = []
                    scores = []
                    for v, d in infr.graph[u].items():
                        e = e_(u, v)
                        if e not in to_remove:
                            # hack because I think this may be a hard problem
                            edges.append(e)
                            scores.append(d.get(weight, -1))
                    bottomx = ut.argsort(scores)[::-1][num:]
                    to_remove.update(set(ut.take(edges, bottomx)))
            infr.graph.remove_edges_from(to_remove)
        degrees = np.array([infr.graph.degree(n) for n in infr.graph.nodes()])
        assert np.all(degrees <= num)

    def relabel_using_inference(infr, **kwargs):
        """
        Applies name labels based on graph inference and then cuts edges
        """
        if infr.verbose > 1:
            print('[infr] relabel_using_inference')
        raise NotImplementedError('probably doesnt work anymore')

        infr.remove_dummy_edges()
        infr.model = infr_model.InfrModel(infr.graph, infr.CUT_WEIGHT_KEY)
        model = infr.model
        thresh = infr.get_threshold()
        model._update_weights(thresh=thresh)
        labeling, params = model.run_inference2(max_labels=len(infr.aids))

        infr.set_node_attrs('name_label', model.node_to_label)

    def get_threshold(infr):
        # Only use the normalized scores to estimate a threshold
        normscores = np.array(infr.get_edge_attrs('normscore').values())
        if infr.verbose >= 1:
            print('len(normscores) = %r' % (len(normscores),))
        isvalid = ~np.isnan(normscores)
        curve = np.sort(normscores[isvalid])
        thresh = infr_model.estimate_threshold(curve, method=None)
        if infr.verbose >= 1:
            print('[estimate] thresh = %r' % (thresh,))
        if thresh is None:
            thresh = .5
        infr.thresh = thresh
        return thresh

    @profile
    def get_filtered_edges(infr, review_cfg):
        """
        DEPRICATE OR MOVE

        Returns a list of edges (typically for user review) based on a specific
        filter configuration.

        CommandLine:
            python -m ibeis get_filtered_edges --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.exec_matching()
            >>> infr.apply_match_edges()
            >>> infr.apply_match_scores()
            >>> infr.apply_feedback_edges()
            >>> review_cfg = {'max_num': 3}
            >>> aids1, aids2 = infr.get_filtered_edges(review_cfg)
            >>> assert len(aids1) == 3
        """
        review_cfg_defaults = {
            'ranks_top': 3,
            'ranks_bot': 2,

            'score_thresh': None,
            'max_num': None,

            'filter_reviewed': True,
            'filter_photobombs': False,

            'filter_true_matches': True,
            'filter_false_matches': False,

            'filter_nonmatch_between_ccs': True,
            'filter_dup_namepairs': True,
        }

        review_cfg = ut.update_existing(
            review_cfg_defaults, review_cfg,
            assert_exists=False, iswarning=True
            # assert_exists=True, iswarning=True
        )

        ibs = infr.ibs
        graph = infr.graph
        nodes = list(graph.nodes())
        uv_list = list(graph.edges())

        node_to_aids = infr.get_node_attrs('aid')
        node_to_nids = infr.get_node_attrs('name_label')
        aids = ut.take(node_to_aids, nodes)
        nids = ut.take(node_to_nids, nodes)
        aid_to_nid = dict(zip(aids, nids))
        nid_to_aids = ut.group_items(aids, nids)

        # Initial set of edges
        aids1 = ut.take_column(uv_list, 0)
        aids2 = ut.take_column(uv_list, 1)

        num_filtered = 0

        if review_cfg['filter_nonmatch_between_ccs']:
            review_states = [
                graph.get_edge_data(*edge).get('decision', 'unreviewed')
                for edge in zip(aids1, aids2)]
            is_nonmatched = [state == 'nomatch' for state in review_states]
            #isneg_flags = is_nonmatched
            valid_flags = filter_between_ccs_neg(aids1, aids2, aid_to_nid,
                                                 nid_to_aids, is_nonmatched)
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_photobombs']:
            # TODO: store photobomb status internally
            am_list = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
            ispb_flags = ibs.get_annotmatch_prop('Photobomb', am_list)
            valid_flags = filter_between_ccs_neg(aids1, aids2, aid_to_nid,
                                                 nid_to_aids, ispb_flags)
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_true_matches']:
            nids1 = ut.take(aid_to_nid, aids1)
            nids2 = ut.take(aid_to_nid, aids2)
            valid_flags = [nid1 != nid2 for nid1, nid2 in zip(nids1, nids2)]
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_false_matches']:
            nids1 = ut.take(aid_to_nid, aids1)
            nids2 = ut.take(aid_to_nid, aids2)
            valid_flags = [nid1 == nid2 for nid1, nid2 in zip(nids1, nids2)]
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_reviewed']:
            valid_flags = [
                graph.get_edge_data(*edge).get(
                    'decision', 'unreviewed') == 'unreviewed'
                for edge in zip(aids1, aids2)]
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_dup_namepairs']:
            # Only look at a maximum of one review between the current set of
            # connected components
            nids1 = ut.take(aid_to_nid, aids1)
            nids2 = ut.take(aid_to_nid, aids2)
            scores = np.array([
                # hack
                max(graph.get_edge_data(*edge).get('score', -1), -1)
                for edge in zip(aids1, aids2)])
            review_states = [
                graph.get_edge_data(*edge).get('decision', 'unreviewed')
                for edge in zip(aids1, aids2)]
            is_notcomp = np.array([state == 'notcomp'
                                   for state in review_states], dtype=np.bool)
            # Notcomps should not be considered in this filtering
            scores[is_notcomp] = -2
            namepair_id_list = vt.compute_unique_data_ids_(ut.lzip(nids1, nids2))
            namepair_id_list = np.array(namepair_id_list, dtype=np.int)
            unique_np_ids, np_groupxs = vt.group_indices(namepair_id_list)
            score_np_groups = vt.apply_grouping(scores, np_groupxs)
            unique_rowx2 = sorted([
                groupx[score_group.argmax()]
                for groupx, score_group in zip(np_groupxs, score_np_groups)
            ])
            aids1 = ut.take(aids1, unique_rowx2)
            aids2 = ut.take(aids2, unique_rowx2)

        # Hack, sort by scores
        scores = np.array([
            max(graph.get_edge_data(*edge).get('score', -1), -1)
            for edge in zip(aids1, aids2)])
        sortx = scores.argsort()[::-1]
        aids1 = ut.take(aids1, sortx)
        aids2 = ut.take(aids2, sortx)

        if review_cfg['max_num'] is not None:
            scores = np.array([
                # hack
                max(graph.get_edge_data(*edge).get('score', -1), -1)
                for edge in zip(aids1, aids2)])
            sortx = scores.argsort()[::-1]
            top_idx = sortx[:review_cfg['max_num']]
            aids1 = ut.take(aids1, top_idx)
            aids2 = ut.take(aids2, top_idx)

        # print('[infr] num_filtered = %r' % (num_filtered,))
        return aids1, aids2

    def add_feedback_df(infr, decision_df, user_id=None, apply=False,
                        verbose=None):
        """
        Adds multiple feedback at once (usually for auto reviewer)

        maybe move back later
        """
        import pandas as pd
        if verbose is None:
            verbose = infr.verbose
        if verbose >= 1:
            print('[infr] add_feedback_df()')
        tags_list = [None] * len(decision_df)
        if isinstance(decision_df, pd.Series):
            decisions = decision_df
        elif isinstance(decision_df, pd.DataFrame):
            if 'decision' in decision_df:
                assert 'match_state' not in decision_df
                decisions = decision_df['decision']
            else:
                decisions = decision_df['match_state']
            if 'tags' in decision_df:
                tags_list = decision_df['tags']
        else:
            raise ValueError(type(decision_df))
        index = decisions.index
        assert index.get_level_values(0).isin(infr.aids_set).all()
        assert index.get_level_values(1).isin(infr.aids_set).all()
        timestamp = ut.get_timestamp('int', isutc=True)
        confidence = None
        uv_iter = it.starmap(e_, index.tolist())
        _iter = zip(uv_iter, decisions, tags_list)
        prog = ut.ProgIter(_iter, nTotal=len(tags_list), enabled=verbose,
                           label='adding feedback')
        for edge, decision, tags in prog:
            if tags is None:
                tags = []
            infr.internal_feedback[edge].append({
                'decision': decision,
                'tags': tags,
                'timestamp': timestamp,
                'confidence': confidence,
                'user_id': user_id,
            })
            if infr.refresh:
                raise NotImplementedError('TODO')
        if apply:
            infr.apply_feedback_edges()

    def _compute_p_same(infr, p_match, p_notcomp):
        p_bg = 0.5  # Needs to be thresh value
        part1 = p_match * (1 - p_notcomp)
        part2 = p_bg * p_notcomp
        p_same = part1 + part2
        return p_same

    @profile
    def categorize_edges_old(infr, graph=None):
        r"""
        Categorizies edges into disjoint types. The types are: positive,
        negative, unreviewed, incomparable, inconsistent internal, and
        inconsistent external.
        """
        if graph is None:
            graph = infr.graph
        # Get node -> name label (PCC)
        node_to_label = nx.get_node_attributes(graph, 'name_label')

        # Group nodes by PCC
        nid_to_cc = collections.defaultdict(set)
        for item, groupid in node_to_label.items():
            nid_to_cc[groupid].add(item)

        # Get edge -> decision
        all_edges = [e_(u, v) for u, v in graph.edges()]
        # edge_to_reviewstate = {}
        # for u, v in infr.pos_graph.edges():
        #     edge_to_reviewstate[(u, v)] = 'match'
        # for u, v in infr.neg_graph.edges():
        #     edge_to_reviewstate[(u, v)] = 'nomatch'
        # for u, v in infr.incomp_graph.edges():
        #     edge_to_reviewstate[(u, v)] = 'notcomp'

        edge_to_reviewstate = {
            (u, v): infr.graph.edge[u][v].get('decision', 'unreviewed')
            for u, v in all_edges
        }

        # Group edges by review type
        grouped_review_edges = ut.group_pairs(edge_to_reviewstate.items())
        neg_review_edges = grouped_review_edges.pop('nomatch', [])
        pos_review_edges = grouped_review_edges.pop('match', [])
        notcomp_review_edges = grouped_review_edges.pop('notcomp', [])
        unreviewed_edges = grouped_review_edges.pop('unreviewed', [])

        if grouped_review_edges:
            raise AssertionError('Reviewed state has unknown values: %r' % (
                list(grouped_review_edges.keys())),)

        def group_name_edges(edges):
            nid_to_cc = collections.defaultdict(set)
            name_edges = (e_(node_to_label[u], node_to_label[v])
                          for u, v in edges)
            for edge, name_edge in zip(edges, name_edges):
                nid_to_cc[name_edge].add(edge)
            return nid_to_cc

        # Group edges by the PCCs they are between
        pos_review_ne_groups = group_name_edges(pos_review_edges)
        neg_review_ne_groups = group_name_edges(neg_review_edges)
        notcomp_review_ne_groups = group_name_edges(notcomp_review_edges)
        unreviewed_ne_groups = group_name_edges(unreviewed_edges)

        # Infer status of PCCs
        pos_ne = set(pos_review_ne_groups.keys())
        neg_ne = set(neg_review_ne_groups.keys())
        notcomp_ne = set(notcomp_review_ne_groups.keys())
        unreviewed_ne = set(unreviewed_ne_groups.keys())

        assert all(n1 == n2 for n1, n2 in pos_ne), 'pos should have same lbl'
        incon_internal_ne = neg_ne.intersection(pos_ne)

        # positive overrides notcomp and unreviewed
        notcomp_ne.difference_update(pos_ne)
        unreviewed_ne.difference_update(pos_ne)

        # negative overrides notcomp and unreviewed
        notcomp_ne.difference_update(neg_ne)
        unreviewed_ne.difference_update(neg_ne)

        # internal inconsistencies override all other categories
        pos_ne.difference_update(incon_internal_ne)
        neg_ne.difference_update(incon_internal_ne)
        notcomp_ne.difference_update(incon_internal_ne)
        unreviewed_ne.difference_update(incon_internal_ne)

        # External inconsistentices are edges leaving inconsistent components
        # internal_incon_ne = set([(n1, n2) for (n1, n2) in neg_ne if n1 == n2])
        assert all(n1 == n2 for n1, n2 in incon_internal_ne), (
            'incon_internal should have same lbl')
        incon_internal_nids = set([n1 for n1, n2 in incon_internal_ne])
        incon_external_ne = set([])
        for _ne in [neg_ne, notcomp_ne, unreviewed_ne]:
            incon_external_ne.update({
                (nid1, nid2) for nid1, nid2 in _ne
                if nid1 in incon_internal_nids or nid2 in incon_internal_nids
            })
        neg_ne.difference_update(incon_external_ne)
        notcomp_ne.difference_update(incon_external_ne)
        unreviewed_ne.difference_update(incon_external_ne)

        # Inference is made on a name(cc) based level now expand this inference
        # back onto the edges.
        positive = {
            nid1: (edges_inside(graph, nid_to_cc[nid1]))
            for nid1, nid2 in pos_ne
        }
        negative = {
            (nid1, nid2): (edges_cross(graph, nid_to_cc[nid1], nid_to_cc[nid2]))
            for nid1, nid2 in neg_ne
        }
        inconsistent_internal = {
            nid1: (edges_inside(graph, nid_to_cc[nid1]))
            for nid1, nid2 in incon_internal_ne
        }
        inconsistent_external = {
            (nid1, nid2): (edges_cross(graph, nid_to_cc[nid1], nid_to_cc[nid2]))
            for nid1, nid2 in incon_external_ne
        }
        # No bridges are formed for notcomparable edges. Just take
        # the set of reviews
        notcomparable = {
            (nid1, nid2): notcomp_review_ne_groups[(nid1, nid2)]
            for (nid1, nid2) in notcomp_ne
        }
        unreviewed = {
            (nid1, nid2):
                edges_cross(graph, nid_to_cc[nid1], nid_to_cc[nid2])
            for (nid1, nid2) in unreviewed_ne
        }
        # Removed not-comparable edges from unreviewed
        for name_edge in unreviewed_ne.intersection(notcomp_ne):
            unreviewed[name_edge].difference_update(notcomparable[name_edge])

        reviewed_positives = {n1: pos_review_ne_groups[(n1, n2)] for
                              n1, n2 in pos_ne}
        reviewed_negatives = {ne: neg_review_ne_groups[ne] for ne in neg_ne}

        ne_categories = {
            'positive': positive,
            'negative': negative,
            'unreviewed': unreviewed,
            'notcomp': notcomparable,
            'inconsistent_internal': inconsistent_internal,
            'inconsistent_external': inconsistent_external,
        }
        categories = {}
        categories['ne_categories'] = ne_categories
        categories['reviewed_positives'] = reviewed_positives
        categories['reviewed_negatives'] = reviewed_negatives
        categories['nid_to_cc'] = nid_to_cc
        categories['node_to_label'] = node_to_label
        categories['edge_to_reviewstate'] = edge_to_reviewstate
        categories['all_edges'] = all_edges
        return categories

    def _debug_edge_categories(infr, graph, ne_categories, edge_categories,
                               node_to_label):
        """
        Checks if edges are categorized and if categoriziations are disjoint.
        """
        name_edge_to_category = {
            name_edge: cat
            for cat, edges in ne_categories.items()
            for name_edge in edges.keys()}

        num_edges = sum(map(len, edge_categories.values()))
        num_edges_real = graph.number_of_edges()

        if num_edges != num_edges_real:
            print('num_edges = %r' % (num_edges,))
            print('num_edges_real = %r' % (num_edges_real,))
            all_edges = ut.flatten(edge_categories.values())
            dup_edges = ut.find_duplicate_items(all_edges)
            if len(dup_edges) > 0:
                # Check where the duplicates are if any
                for k1, k2 in it.combinations(edge_categories.keys(), 2):
                    v1 = edge_categories[k1]
                    v2 = edge_categories[k2]
                    overlaps = ut.set_overlaps(v1, v2)
                    if overlaps['isect'] != 0:
                        print('%r-%r: %s' % (k1, k2, ut.repr4(overlaps)))
                for k1 in edge_categories.keys():
                    v1 = edge_categories[k1]
                    dups = ut.find_duplicate_items(v1)
                    if dups:
                        print('%r, has %s dups' % (k1, len(dups)))
            assert len(dup_edges) == 0, 'edge not same and duplicates'

            edges = ut.lstarmap(e_, graph.edges())
            missing12 = ut.setdiff(edges, all_edges)
            missing21 = ut.setdiff(all_edges, edges)
            print('missing12 = %r' % (missing12,))
            print('missing21 = %r' % (missing21,))
            print(ut.repr4(ut.set_overlaps(graph.edges(), all_edges)))

            for u, v in missing12:
                edge = graph.edge[u][v]
                print('missing edge = %r' % ((u, v),))
                print('state = %r' % (edge.get('decision',
                                               'unreviewed')))
                nid1 = node_to_label[u]
                nid2 = node_to_label[v]
                name_edge = e_(nid1, nid2)
                print('name_edge = %r' % (name_edge,))
                cat = name_edge_to_category.get(name_edge, None)
                print('cat = %r' % (cat,))

            print('ERROR: Not all edges accounted for. '
                  'Is name labeling computed using connected components?')
            import utool
            utool.embed()
            raise AssertionError('edges not the same')


    @profile
    def apply_review_inference(infr, graph=None):
        """
        Updates the inferred state of each edge based on reviews and current
        labeling. State of the graph is only changed at the very end of the
        function.

        TODO: split into simpler functions

        CommandLine:
            python -m ibeis.algo.graph.core apply_review_inference

        Example:
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> aids = list(range(1, 10))
            >>> infr = AnnotInference(None, aids, autoinit=True, verbose=1)
            >>> infr.ensure_full()
            >>> infr._init_priority_queue()
            >>> infr.add_feedback((1, 2), 'match')
            >>> infr.add_feedback((2, 3), 'match')
            >>> infr.add_feedback((2, 3), 'match')
            >>> infr.add_feedback((3, 4), 'match')
            >>> infr.add_feedback((4, 5), 'nomatch')
            >>> infr.add_feedback((6, 7), 'match')
            >>> infr.add_feedback((7, 8), 'match')
            >>> infr.add_feedback((6, 8), 'nomatch')
            >>> infr.add_feedback((6, 1), 'notcomp')
            >>> infr.add_feedback((1, 9), 'notcomp')
            >>> infr.add_feedback((8, 9), 'notcomp')
            >>> #infr.show_graph(hide_cuts=False)
            >>> graph = infr.graph
            >>> infr.apply_review_inference(graph)
        """
        if graph is None:
            graph = infr.graph

        if infr.verbose >= 2:
            print('[infr] apply_review_inference on %d nodes' % (len(graph)))

        categories = infr.categorize_edges_old(graph)
        ne_categories = categories['ne_categories']
        reviewed_positives = categories['reviewed_positives']
        reviewed_negatives = categories['reviewed_negatives']
        edge_to_reviewstate = categories['edge_to_reviewstate']
        nid_to_cc = categories['nid_to_cc']
        node_to_label = categories['node_to_label']
        all_edges = categories['all_edges']

        # Find possible fixes for inconsistent components
        if infr.verbose >= 2:
            if ne_categories['inconsistent_internal']:
                print('[infr] found %d inconsistencies searching for fixes' %
                      (len(ne_categories['inconsistent_internal']),))
            else:
                print('no inconsistencies')

        suggested_fix_edges = []
        other_error_edges = []
        if infr.method == 'graph':
            # dont do this in ranking mode
            # Check for inconsistencies
            for nid, cc_incon_edges in ne_categories['inconsistent_internal'].items():
                # Find possible edges to fix in the reviewed subgarph
                reviewed_inconsistent = [
                    (u, v, infr.graph.edge[u][v].copy()) for (u, v) in cc_incon_edges
                    if edge_to_reviewstate[(u, v)] != 'unreviewed'
                ]
                subgraph = nx.Graph(reviewed_inconsistent)
                # TODO: only need to use one fix edge here.
                cc_error_edges = infr._find_possible_error_edges_old(subgraph)
                import utool
                with utool.embed_on_exception_context:
                    assert len(cc_error_edges) > 0, 'no fixes found'
                cc_other_edges = ut.setdiff(cc_incon_edges, cc_error_edges)
                suggested_fix_edges.extend(cc_error_edges)
                other_error_edges.extend(cc_other_edges)
                # just add one for now
                # break

        if infr.verbose >= 2 and ne_categories['inconsistent_internal']:
            print('[infr] found %d possible fixes' % len(suggested_fix_edges))

        edge_categories = {k: ut.flatten(v.values())
                           for k, v in ne_categories.items()}

        # if __debug__:
        #     infr._debug_edge_categories(graph, ne_categories, edge_categories,
        #                                 node_to_label)

        # Update the attributes of all edges in the subgraph

        # Update the infered state
        infr.set_edge_attrs('inferred_state', _dz(
            edge_categories['inconsistent_external'], ['inconsistent_external']))
        infr.set_edge_attrs('inferred_state', _dz(
            edge_categories['inconsistent_internal'], ['inconsistent_internal']))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['unreviewed'], [None]))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['notcomp'], ['notcomp']))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['positive'], ['same']))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['negative'], ['diff']))

        # Suggest possible fixes
        infr.set_edge_attrs('maybe_error', ut.dzip(all_edges, [None]))
        infr.set_edge_attrs('maybe_error', _dz(suggested_fix_edges, [True]))

        # Update the cut state
        # TODO: DEPRICATE the cut state is not relevant anymore
        infr.set_edge_attrs('is_cut', _dz(edge_categories['inconsistent_external'],
                                          [True]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['inconsistent_internal'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['unreviewed'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['notcomp'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['positive'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['negative'], [True]))

        # Update basic priorites
        # FIXME: this must agree with queue
        # infr.set_edge_attrs('priority', infr.get_edge_attrs(
        #         infr.PRIORITY_METRIC, inconsistent_external_edges,
        #         default=.01))
        # infr.set_edge_attrs('priority', infr.get_edge_attrs(
        #         infr.PRIORITY_METRIC, incon_intern_edges, default=.01))
        # infr.set_edge_attrs('priority', infr.get_edge_attrs(
        #         infr.PRIORITY_METRIC, unreviewed_edges, default=.01))
        # infr.set_edge_attrs('priority', _dz(notcomparable_edges, [0]))
        # infr.set_edge_attrs('priority', _dz(positive_edges, [0]))
        # infr.set_edge_attrs('priority', _dz(negative_edges, [0]))
        # infr.set_edge_attrs('priority', _dz(suggested_fix_edges, [2]))

        # print('suggested_fix_edges = %r' % (sorted(suggested_fix_edges),))
        # print('other_error_edges = %r' % (sorted(other_error_edges),))

        if infr.queue is not None:
            # hack this off if method is ranking, we dont want to update any
            # priority
            if infr.method != 'ranking':
                if infr.verbose >= 2:
                    print('[infr] updating priority queue')
                infr._update_priority_queue_old(graph,
                                                ne_categories['positive'],
                                                ne_categories['negative'],
                                                reviewed_positives,
                                                reviewed_negatives,
                                                node_to_label, nid_to_cc,
                                                suggested_fix_edges,
                                                other_error_edges,
                                                edge_categories['unreviewed'])
        else:
            if infr.verbose >= 2:
                print('[infr] no priority queue to update')
        if infr.verbose >= 3:
            print('[infr] finished review inference')

    @profile
    def _find_possible_error_edges_old(infr, subgraph):
        """
        Args:
            subgraph (nx.Graph): a subgraph of a positive compomenent
                with only reviewed edges.
        """
        inconsistent_edges = [
            edge for edge, state in
            nx.get_edge_attributes(subgraph, 'decision').items()
            if state == 'nomatch'
        ]
        maybe_error_edges = set([])
        # subgraph_ = infr.simplify_graph(subgraph, copy=copy)
        subgraph_ = subgraph.copy()
        subgraph_.remove_edges_from(inconsistent_edges)

        # This is essentially solving a multicut problem for multiple pairs of
        # terminal nodes. The multiple min-cut runs produces a feasible
        # solution. Could use a multicut approximation.

        ut.nx_set_default_edge_attributes(subgraph_, 'num_reviews', 1)
        for s, t in inconsistent_edges:
            cut_edgeset = ut.nx_mincut_edges_weighted(subgraph_, s, t,
                                                      capacity='num_reviews')
            cut_edgeset = set([e_(*edge) for edge in cut_edgeset])
            join_edgeset = {(s, t)}
            cut_edgeset_weight = sum([
                subgraph_.get_edge_data(u, v).get('num_reviews', 1)
                for u, v in cut_edgeset])
            join_edgeset_weight = sum([
                subgraph.get_edge_data(u, v).get('num_reviews', 1)
                for u, v in join_edgeset])
            # Determine if this is more likely a split or a join
            # if len(cut_edgeset) == 0:
            #     maybe_error_edges.update(join_edgeset)
            if join_edgeset_weight < cut_edgeset_weight:
                maybe_error_edges.update(join_edgeset)
            else:
                maybe_error_edges.update(cut_edgeset)

        maybe_error_edges_ = ut.lstarmap(e_, maybe_error_edges)
        return maybe_error_edges_

    @profile
    def _update_priority_queue_old(infr, graph, positive, negative,
                                   reviewed_positives, reviewed_negatives,
                                   node_to_label, nid_to_cc,
                                   suggested_fix_edges, other_error_edges,
                                   unreviewed_edges):
        r"""
        TODO refactor

        Example:
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> from ibeis.algo.graph.core import _dz
            >>> from ibeis.algo.hots import demo
            >>> infr = demo.make_demo_infr(
            >>>     ccs=[[1, 2, 3, 4, 5],
            >>>            [6, 7, 8, 9, 10]],
            >>>     edges=[
            >>>         #(1, 6, {'decision': 'nomatch'}),
            >>>         (1, 6, {}),
            >>>         (4, 9, {}),
            >>>     ]
            >>> )
            >>> infr._init_priority_queue()
            >>> assert len(infr.queue) == 2
            >>> infr.queue_params['neg_redundancy'] = None
            >>> infr.add_feedback((1, 6), 'nomatch')
            >>> assert len(infr.queue) == 0
            >>> graph = infr.graph
            >>> ut.exec_func_src(infr.apply_review_inference,
            >>>                  sentinal='if infr.queue is not None', stop=-1,
            >>>                  verbose=True)
            >>> infr.queue_params['neg_redundancy'] = 1
            >>> infr.apply_review_inference()
        """
        # update the priority queue on the fly
        queue = infr.queue
        pos_redundancy = infr.queue_params['pos_redundancy']
        neg_redundancy = infr.queue_params['neg_redundancy']

        if neg_redundancy and neg_redundancy < np.inf:
            # Remove priority of PCC-pairs with k-negative edges between them
            for (nid1, nid2), neg_edges in reviewed_negatives.items():
                if len(neg_edges) >= neg_redundancy:
                    other_edges = negative[(nid1, nid2)]
                    queue.delete_items(other_edges)

        if pos_redundancy and pos_redundancy < np.inf:
            # Remove priority internal edges of k-consistent PCCs.
            for nid, pos_edges in reviewed_positives.items():
                if pos_redundancy == 1:
                    # trivially computed
                    pos_conn = 1
                else:
                    pos_conn = nx.edge_connectivity(nx.Graph(list(pos_edges)))
                if pos_conn >= pos_redundancy:
                    other_edges = positive[nid]
                    queue.delete_items(other_edges)

        if suggested_fix_edges:
            # Add error edges back in with higher priority
            queue.update(zip(suggested_fix_edges,
                             -infr._get_priorites(suggested_fix_edges)))

            queue.delete_items(other_error_edges)

        needs_priority = [e for e in unreviewed_edges if e not in queue]
        queue.update(zip(needs_priority, -infr._get_priorites(needs_priority)))

    @profile
    def add_feedback_old(infr, aid1=None, aid2=None, decision=None, tags=[],
                     apply=False, user_id=None, confidence=None,
                     edge=None, verbose=None, rectify=True):
        """
        Public interface to add feedback for a single edge to the buffer.
        Feedback is not applied to the graph unless `apply=True`.

        Args:
            aid1 (int):  annotation id
            aid2 (int):  annotation id
            decision (str): decision from `ibs.const.REVIEW.CODE_TO_INT`
            tags (list of str): specify Photobomb / Scenery / etc
            user_id (str): id of agent who did the review
            confidence (str): See ibs.const.CONFIDENCE
            apply (bool): if True feedback is dynamically applied

        CommandLine:
            python -m ibeis.algo.graph.core add_feedback_old

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback((5, 6), 'match')
            >>> infr.add_feedback((5, 6), 'nomatch', ['Photobomb'])
            >>> infr.add_feedback((1, 2), 'notcomp')
            >>> print(ut.repr2(infr.internal_feedback, nl=2))
            >>> assert len(infr.external_feedback) == 0
            >>> assert len(infr.internal_feedback) == 2
            >>> assert len(infr.internal_feedback[(5, 6)]) == 2
            >>> assert len(infr.internal_feedback[(1, 2)]) == 1
        """
        if verbose is None:
            verbose = infr.verbose
        if edge:
            aid1, aid2 = edge
        if verbose >= 1:
            print(('[infr] add_feedback_old(%r, %r, decision=%r, tags=%r, '
                                        'user_id=%r, confidence=%r)') % (
                aid1, aid2, decision, tags, user_id, confidence))

        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))
        assert isinstance(decision, six.string_types)
        edge = e_(aid1, aid2)

        if decision == 'unreviewed':
            feedback_item = None
            if edge in infr.external_feedback:
                raise ValueError(
                    "Can't unreview an edge that has been committed")
            if edge in infr.internal_feedback:
                del infr.internal_feedback[edge]
            for G in infr.review_graphs.values():
                if G.has_edge(*edge):
                    G.remove_edge(*edge)
        else:
            feedback_item = {
                'decision': decision,
                'tags': tags,
                'timestamp': ut.get_timestamp('int', isutc=True),
                'confidence': confidence,
                'user_id': user_id,
            }
            infr.internal_feedback[edge].append(feedback_item)
            # Add to appropriate review graph and change review if it existed
            # previously
            infr.review_graphs[decision].add_edge(*edge)
            for k, G in infr.review_graphs.items():
                if k != decision:
                    if G.has_edge(*edge):
                        G.remove_edge(*edge)

            if infr.refresh:
                infr.refresh.add(decision, user_id)

            if infr.test_mode:
                if user_id.startswith('auto'):
                    infr.test_state['n_auto'] += 1
                elif user_id == 'oracle':
                    infr.test_state['n_manual'] += 1
                else:
                    raise AssertionError('unknown user_id=%r' % (user_id,))

        if apply:
            # Apply new results on the fly
            infr._dynamically_apply_feedback_old(edge, feedback_item, rectify)

            if infr.test_mode:
                metrics = infr.measure_metrics_old()
                infr.metrics_list.append(metrics)
        else:
            assert not infr.test_mode, 'breaks tests'

    @profile
    def _dynamically_apply_feedback_old(infr, edge, feedback_item, rectify):
        """
        Dynamically updates all states based on a single dynamic change

        CommandLine:
            python -m ibeis.algo.graph.core _dynamically_apply_feedback_old:0
            python -m ibeis.algo.graph.core _dynamically_apply_feedback_old:1 --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.relabel_using_reviews()
            >>> infr.add_feedback((1, 2), 'match')
            >>> infr.add_feedback((2, 3), 'match')
            >>> infr.add_feedback((2, 3), 'match')
            >>> assert infr.graph.edge[1][2]['num_reviews'] == 1
            >>> assert infr.graph.edge[2][3]['num_reviews'] == 2
            >>> infr._del_feedback_edges()
            >>> infr.apply_feedback_edges()
            >>> assert infr.graph.edge[1][2]['num_reviews'] == 1
            >>> assert infr.graph.edge[2][3]['num_reviews'] == 2

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.relabel_using_reviews()
            >>> infr.verbose = 2
            >>> ut.qtensure()
            >>> infr.ensure_full()
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback((6, 2), 'match')
            >>> infr.add_feedback((2, 3), 'match')
            >>> infr.add_feedback((3, 4), 'match')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback((2, 3), 'nomatch')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback((6, 4), 'match')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback((1, 5), 'nomatch')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback((1, 3), 'nomatch')
            >>> infr.show_graph(show_cuts=True)
            >>> import plottool as pt
            >>> pt.present()
            >>> ut.show_if_requested()
        """
        if feedback_item is None:
            if infr.verbose >= 2:
                print('[infr] _dynamically_apply_feedback_old (removing edge=%r)'
                      % (edge,))
            state = 'unreviewed'
            infr._del_feedback_edges([edge])
            infr.set_edge_attrs(
                infr.CUT_WEIGHT_KEY,
                infr.get_edge_attrs('normscore', [edge], np.nan))
        else:
            # Apply the review to the specified edge
            state = feedback_item['decision']
            tags = feedback_item['tags']
            confidence = feedback_item['confidence']
            user_id = feedback_item['user_id']
            if infr.verbose >= 2:
                print('[infr] _dynamically_apply_feedback_old edge=%r, state=%r'
                      % (edge, state,))
            p_same_lookup = {
                'match': infr._compute_p_same(1.0, 0.0),
                'nomatch': infr._compute_p_same(0.0, 0.0),
                'notcomp': infr._compute_p_same(0.0, 1.0),
            }
            p_same = p_same_lookup[state]
            num_reviews = infr.get_edge_attrs('num_reviews', [edge],
                                              default=0).get(edge, 0)
            infr._set_feedback_edges([edge], [state], [p_same], [tags],
                                     [confidence],
                                     [user_id], [num_reviews + 1])
            # TODO: change num_reviews to num_consistent_reviews
            if state != 'notcomp':
                ut.nx_delete_edge_attr(infr.graph, 'inferred_state', [edge])

        # Dynamically update names and inferred attributes of relevant nodes
        # subgraph, subgraph_cuts = infr._get_influenced_subgraph(edge)
        n1, n2 = edge

        import utool
        with utool.embed_on_exception_context:
            cc1 = infr.pos_graph.connected_to(n1)
            cc2 = infr.pos_graph.connected_to(n2)
        relevant_nodes = cc1.union(cc2)
        if DEBUG_CC:
            cc1_ = infr.get_annot_cc(n1)
            cc2_ = infr.get_annot_cc(n2)
            relevant_nodes_ = cc1_.union(cc2_)
            assert relevant_nodes_ == relevant_nodes
            # print('seems good')

        subgraph = infr.graph.subgraph(relevant_nodes)

        # Change names of nodes
        infr.relabel_using_reviews(graph=subgraph, rectify=rectify)

        # Include other components where there are external consequences
        # This is only the case if two annotations are merged or a single
        # annotation is split.
        nomatch_ccs = infr.get_nomatch_ccs(relevant_nodes)
        extended_nodes = ut.flatten(nomatch_ccs)
        extended_nodes.extend(relevant_nodes)
        extended_subgraph = infr.graph.subgraph(extended_nodes)

        # This re-infers all attributes of the influenced sub-graph only
        infr.apply_review_inference(graph=extended_subgraph )


    @profile
    def get_nomatch_ccs(infr, cc):
        """
        Returns a set of PCCs that are known to have at least one negative
        match to any node in the input nodes.

        Search every neighbor in this cc for a nomatch connection. Then add the
        cc belonging to that connected node.  In the case of an inconsistent
        cc, nodes within the cc will not be returned.
        """
        if DEBUG_CC:
            visited = set(cc)
            # visited_nodes = set([])
            nomatch_ccs = []
            for n1 in cc:
                for n2 in infr.graph.neighbors(n1):
                    if n2 not in visited:
                        # data = infr.graph.get_edge_data(n1, n2)
                        # _state = data.get('decision', 'unreviewed')
                        _state = infr.graph.edge[n1][n2].get('decision',
                                                             'unreviewed')
                        if _state == 'nomatch':
                            cc2 = infr.get_annot_cc(n2)
                            nomatch_ccs.append(cc2)
                            visited.update(cc2)
            nomatch_ccs_old = nomatch_ccs
        else:
            neg_graph = infr.neg_graph
            pos_graph = infr.pos_graph
            cc_labels = {
                pos_graph.node_label(n2)
                for n1 in cc
                for n2 in neg_graph.neighbors(n1)
            }
            nomatch_ccs = [pos_graph.connected_to(node)
                           for node in cc_labels]
        if DEBUG_CC:
            assert nomatch_ccs_old == nomatch_ccs
        return nomatch_ccs

    @profile
    def get_annot_cc(infr, source, visited_nodes=None):
        """
        Get the name_label cc connected to `source`

        TODO:
            Currently instead of using BFS to find the connected compoments
            each time dynamically maintain connected compoments as new
            information is added.

            The problem is "Dynamic Connectivity"

            Union-find can be used as long as no edges are deleted

            Refactor to a union-split-find data structure
                https://courses.csail.mit.edu/6.851/spring14/lectures/L20.html
                http://cs.stackexchange.com/questions/33595/maintaining-connect
                http://cs.stackexchange.com/questions/32077/
                https://networkx.github.io/documentation/development/_modules/
                    networkx/utils/union_find.html
        """
        # Speed hack for BFS conditional
        G = infr.graph
        cc = set([source])
        queue = collections.deque([])
        # visited_nodes = set([source])
        if visited_nodes is None:
            visited_nodes = set([])
        if source not in visited_nodes:
            visited_nodes.add(source)
            new_edges = iter([(source, n) for n in G.adj[source]])
            queue.append((source, new_edges))
        while queue:
            parent, edges = queue[0]
            parent_attr = G.node[parent]['name_label']
            for edge in edges:
                child = edge[1]
                # only move forward if the child shares name_label
                if child not in visited_nodes:
                    visited_nodes.add(child)
                    if parent_attr == G.node[child]['name_label']:
                        cc.add(child)
                        new_edges = iter([(child, n) for n in G.adj[child]])
                        queue.append((child, new_edges))
            queue.popleft()
        # def condition(G, child, edge):
        #     u, v = edge
        #     nid1 = G.node[u]['name_label']
        #     nid2 = G.node[v]['name_label']
        #     return nid1 == nid2
        # cc = set(ut.util_graph.bfs_same_attr_nodes(infr.graph, node,
        #                                            key='name_label'))
        # cc = set(ut.util_graph.bfs_conditional(
        #     infr.graph, node, yield_condition=condition,
        #     continue_condition=condition))
        # cc.add(node)
        return cc

    def measure_metrics_old(infr):
        real_pos_edges = []
        n_error_edges = 0
        pred_n_pcc_mst_edges = 0
        n_fn = 0
        n_fp = 0

        for edge, data in infr.edges(data=True):
            true_state = infr.edge_truth[edge]
            decision = data.get('decision', 'unreviewed')
            if true_state == decision and true_state == 'match':
                real_pos_edges.append(edge)
            elif decision != 'unreviewed':
                if true_state != decision:
                    n_error_edges += 1
                    if true_state == 'match':
                        n_fn += 1
                    elif true_state == 'nomatch':
                        n_fp += 1

        import networkx as nx
        for cc in nx.connected_components(nx.Graph(real_pos_edges)):
            pred_n_pcc_mst_edges += len(cc) - 1

        pos_acc = pred_n_pcc_mst_edges / infr.real_n_pcc_mst_edges
        metrics = {
            'n_manual': infr.test_state['n_manual'],
            'n_auto': infr.test_state['n_auto'],
            'pos_acc': pos_acc,
            'n_merge_remain': infr.real_n_pcc_mst_edges - pred_n_pcc_mst_edges,
            'merge_remain': 1 - pos_acc,
            'n_errors': n_error_edges,
            'n_fn': n_fn,
            'n_fp': n_fp,
        }
        return metrics

    def edge_confusion(infr):
        confusion = {
            'correct': {
                'pred_pos': [],
                'pred_neg': [],
            },
            'incorrect': {
                'pred_pos': [],
                'pred_neg': [],
            },
        }
        for edge, data in infr.edges(data=True):
            # nid1 = infr.pos_graph.node_label(edge[0])
            # nid2 = infr.pos_graph.node_label(edge[1])
            true_state = infr.edge_truth[edge]
            decision = data.get('decision', 'unreviewed')
            if decision == 'unreviewed':
                pass
            elif true_state == decision:
                if true_state == 'match':
                    confusion['correct']['pred_pos'].append(edge)
            elif true_state != decision:
                if decision == 'match':
                    confusion['incorrect']['pred_pos'].append(edge)
                elif decision == 'nomatch':
                    confusion['incorrect']['pred_neg'].append(edge)

    @profile
    def apply_weights(infr):
        """
        Combines normalized scores and user feedback into edge weights used in
        the graph cut inference.
        """
        infr.print('apply_weights', 1)
        ut.nx_delete_edge_attr(infr.graph, infr.CUT_WEIGHT_KEY)
        # mst not needed. No edges are removed

        edges = list(infr.graph.edges())
        edge_to_normscore = infr.get_edge_attrs('normscore')
        normscores = np.array(ut.dict_take(edge_to_normscore, edges, np.nan))

        edge_to_reviewed_weight = infr.get_edge_attrs('reviewed_weight')
        reviewed_weights = np.array(ut.dict_take(edge_to_reviewed_weight,
                                                 edges, np.nan))
        # Combine into weights
        weights = normscores.copy()
        has_review = ~np.isnan(reviewed_weights)
        weights[has_review] = reviewed_weights[has_review]
        # remove nans
        is_valid = ~np.isnan(weights)
        weights = weights.compress(is_valid, axis=0)
        edges = ut.compress(edges, is_valid)
        infr.set_edge_attrs(infr.CUT_WEIGHT_KEY, _dz(edges, weights))

        # infr.set_edge_attrs(infr.CUT_WEIGHT_KEY, _dz(edges, p_same_list))
        # p_same_lookup = {
        #     'match': infr._compute_p_same(1.0, 0.0),
        #     'nomatch': infr._compute_p_same(0.0, 0.0),
        #     'notcomp': infr._compute_p_same(0.0, 1.0),
        # }
        # p_same_list = ut.take(p_same_lookup, decision_list)
        # infr.set_edge_attrs('reviewed_weight', _dz(edges, p_same_list))

    # Scores are ordered in priority order:
    # CUT_WEIGHT - final weight used for inference (overridden by user)
    # NORMSCORE - normalized score computed by an automatic process
    # SCORE - raw score computed by automatic process

    CUT_WEIGHT_KEY = 'cut_weight'
