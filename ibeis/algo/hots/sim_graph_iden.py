# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import utool as ut
import numpy as np
print, rrr, profile = ut.inject2(__name__)


def compare_groups(true_groups, pred_groups):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.sim_graph_iden import *  # NOQA
        >>> true_groups = [
        >>>   [20, 21], [22, 23], [1, 2], [12, 13, 14], [4], [5, 6, 3], [7, 8],
        >>>   [9, 10, 11], [31, 32, 33, 34, 35],   [41, 42, 43, 44], [45], [50]
        >>> ]
        >>> pred_groups = [
        >>>     [20, 21, 22, 23], [1, 2], [12], [13, 14], [3, 4], [5, 6,11],
        >>>     [7], [8, 9], [10], [31, 32], [33, 34, 35], [41, 42, 43, 44, 45]
        >>> ]
        >>> result = compare_groups(true_groups, pred_groups)
        >>> print(result)
        >>> print(ut.repr4(result))
    """
    true = {frozenset(_group) for _group in true_groups}
    pred = {frozenset(_group) for _group in pred_groups}

    # Find the groups that are exactly the same
    common = true.intersection(pred)

    true_sets = true.difference(common)
    pred_sets = pred.difference(common)

    # connected compoment lookups
    pred_conn = {p: frozenset(ps) for ps in pred for p in ps}
    true_conn = {t: frozenset(ts) for ts in true for t in ts}

    # How many predictions can be merged into perfect pieces?
    # For each true sets, find if it can be made via merging pred sets
    predict_merges = []
    true_merges = []
    for ts in true_sets:
        ccs = set([pred_conn.get(t, frozenset()) for t in ts])
        if frozenset.union(*ccs) == ts:
            # This is a pure merge
            predict_merges.append(ccs)
            true_merges.append(ts)

    # How many predictions can be split into perfect pieces?
    true_splits = []
    predict_splits = []
    for ps in pred_sets:
        ccs = set([true_conn.get(p, frozenset()) for p in ps])
        if frozenset.union(*ccs) == ps:
            # This is a pure merge
            true_splits.append(ccs)
            predict_splits.append(ps)

    pred_merges_flat = ut.flatten(predict_merges)
    true_splits_flat = ut.flatten(true_splits)

    pred_hybrid = frozenset(map(frozenset, pred_sets)).difference(
        set(predict_splits + pred_merges_flat))

    true_hybrid = frozenset(map(frozenset, true_sets)).difference(
        set(true_merges + true_splits_flat))

    result = {
        'common': common,
        'true_splits_flat': true_splits_flat,
        'true_merges': true_merges,
        'true_hybrid': true_hybrid,
        'pred_splits': predict_splits,
        'pred_merges_flat': pred_merges_flat,
        'pred_hybrid': pred_hybrid,
    }
    return result


@ut.reloadable_class
class InfrSimulation(object):
    """
    Methods for simulating an inference with automatic decisions and a user in
    the loop
    """

    def __init__(sim, infr, primary_truth, primary_probs, auto_decisions):
        sim.infr = infr
        sim.primary_probs = primary_probs
        sim.primary_truth = primary_truth
        sim.auto_decisions = auto_decisions

        sim.results = {}

    @profile
    def initialize(sim):
        """ reset state of infr back to only auto decisions """
        sim.results['n_incon_reviews'] = 0
        sim.results['n_incon_fixes'] = 0

        infr = sim.infr
        auto_decisions = sim.auto_decisions
        primary_probs = sim.primary_probs
        primary_truth = sim.primary_truth

        infr.verbose = 1

        infr.remove_feedback()

        # Apply probabilities to edges in infr
        infr.set_edge_attrs(infr.PRIORITY_METRIC,
                            primary_probs['match'].to_dict())

        # Add automatic feedback
        infr.add_feedback_df(auto_decisions, user_id='clf')
        # (Apply feedback edges is a bottleneck of the function)
        infr.apply_feedback_edges()

        # Cluster based on automatic feedback
        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)

        # infr.print_graph_info()

        # Infer what must be done next
        infr.apply_review_inference()

        sim.results['n_auto_clusters'] = n_clusters
        sim.results['n_auto_inconsistent'] = n_inconsistent

        # Determine how the auto-decisions and groundtruth differ
        auto_truth = primary_truth.loc[auto_decisions.index].idxmax(axis=1)
        is_mistake = auto_decisions != auto_truth
        sim.results['n_auto_mistakes'] = sum(is_mistake)

    @profile
    def check_baseline_results(sim):
        import networkx as nx
        infr = sim.infr
        n_clusters_possible = 0
        gt_clusters = ut.group_pairs(infr.gen_node_attrs('orig_name_label'))
        for nid, nodes in gt_clusters.items():
            if len(nodes) == 1:
                n_clusters_possible += 1
                continue
            cc_cand_edges = list(ut.nx_edges_between(infr.graph, nodes))
            cc = ut.nx_from_node_edge(nodes, cc_cand_edges)
            mst = nx.minimum_spanning_tree(cc)
            n_clusters_possible += (
                len(list(nx.connected_component_subgraphs(mst)))
            )
        real_nids = ut.unique(sim.infr.orig_name_labels)

        sim.results['n_clusters_possible'] = n_clusters_possible
        sim.results['n_clusters_real'] = len(real_nids)

    @profile
    def review_inconsistencies(sim):
        """
        Within each inconsistent component simulate the reviews that would be
        done to fix the issue.

            >>> sim.initialize()
        """
        infr = sim.infr
        primary_truth = sim.primary_truth

        prev = infr.verbose
        infr.verbose = 0

        # In the worst case all edges in flagged ccs would need to be reviewed
        incon_edges = []
        n_worst_case = 0
        for cc in infr.inconsistent_components():
            edges = ut.lstarmap(infr.e_, list(cc.edges()))
            reviewed_edges = list(infr.get_edges_where_ne(
                'reviewed_state', 'unreviewed', edges=edges,
                default='unreviewed'))
            incon_edges.extend(reviewed_edges)
            n_worst_case += len(reviewed_edges)

        incon_truth = primary_truth.loc[incon_edges].idxmax(axis=1)
        # incon_pred1 = infr.edge_attr_df('reviewed_state', incon_edges)

        # We can do better, but there might still be some superflous reviews
        n_superflouous = 0
        review_edges = infr.generate_reviews()

        if False:
            review_edges = infr.generate_reviews()
            aid1, aid2 = next(review_edges)
            d = infr.graph.edge[aid1][aid2]
            print('d = %r' % (d,))

        for count, (aid1, aid2) in enumerate(ut.ProgIter(review_edges)):
            d = infr.graph.edge[aid1][aid2]
            if not d.get('maybe_error', False):
                # print('Found edge (%r, %r) without error: %r' % (
                #     aid1, aid2, d,))
                if d.get('inferred_state') == 'inconsistent_internal':
                    # import utool
                    # utool.embed()
                    print('ERROR')
                    import sys
                    sys.exit(1)
                # Stop once inconsistent compmoents stop coming up
                break
            else:
                pass
                # print('Fixing edge: %r' % (d,))
            prev_state = d['reviewed_state']
            state = primary_truth.loc[(aid1, aid2)].idxmax()
            if state == prev_state:
                n_superflouous += 1
            tags = []
            infr.add_feedback(aid1, aid2, state, tags, apply=True,
                              rectify=False, user_id='oracle',
                              user_confidence='absolutely_sure')
        n_reviews = count
        n_fixes = n_reviews - n_superflouous
        print('n_worst_case = %r' % (n_worst_case,))
        print('n_superflouous = %r' % (n_superflouous,))
        print('n_fixes = %r' % (n_fixes,))
        print('count = %r' % (count,))
        sim.results['n_incon_reviews'] = count
        sim.results['n_incon_fixes'] = n_fixes

        # Should have fixed everything
        infr.apply_review_inference()
        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)
        import utool
        with utool.embed_on_exception_context:
            print('n_inconsistent = %r' % (n_inconsistent,))
            assert n_inconsistent == 0, 'should have fixed everything'

        if False:
            from ibeis.scripts import clf_helpers
            incon_pred2 = infr.edge_attr_df('reviewed_state', incon_edges)

            print('--------')
            # clf_helpers.classification_report2(incon_truth, incon_pred1)
            print('--------')
            clf_helpers.classification_report2(incon_truth, incon_pred2)

        infr.verbose = prev

        # for cc in infr.inconsistent_components():
        #     cc_error_edges = infr._find_possible_error_edges(cc)
        #     pass

    @profile
    def rank_priority_edges(sim):
        """
        Find the order of reviews that would be selected by the priority queue
        for an oracle reviewer

        Differences between this and iterative review
            * Here we know what the minimum set of negative edges is in advance
              the iterative review may have to go through several more
              negatives
            * This positive reviews done here are minimal if the current
              decisions contain no split errors.
        """
        import networkx as nx
        import pandas as pd
        # auto_decisions = sim.auto_decisions
        primary_probs = sim.primary_probs
        primary_truth = sim.primary_truth
        infr = sim.infr

        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)
        import utool
        with utool.embed_on_exception_context:
            assert n_inconsistent == 0, 'must be zero here'

        curr_decisions = infr.edge_attr_df('reviewed_state')

        # Choose weights proportional to the liklihood an edge will be reviewed
        # Give a negative weight to edges that are already reviewed.
        mwc_weights = primary_probs['match'].copy()
        # mwc_weights.loc[curr_decisions.index] = 2
        for k, sub in curr_decisions.groupby(curr_decisions):
            if k == 'match':
                mwc_weights[sub.index] += 1
            if k == 'notcomp':
                mwc_weights[sub.index] += 1
            if k == 'nomatch':
                mwc_weights[sub.index] += 2
        mwc_weights = 2 - mwc_weights

        undiscovered_errors = []
        gt_forests = []
        gt_clusters = ut.group_pairs(infr.gen_node_attrs('orig_name_label'))
        for nid, nodes in gt_clusters.items():
            if len(nodes) == 1:
                continue
            cc_cand_edges = list(ut.nx_edges_between(infr.graph, nodes))
            cc = ut.nx_from_node_edge(nodes, cc_cand_edges)
            cc_weights = mwc_weights.loc[cc_cand_edges]
            nx.set_edge_attributes(cc, 'weight', cc_weights.to_dict())
            # Minimum Spanning Compoment will contain the minimum possible
            # positive reviews and previously reviewed edges
            mwc = ut.nx_minimum_weight_component(cc)
            # Remove all no-matching edges to disconnect impossible matches
            mwc_decision = curr_decisions.loc[ut.lstarmap(infr.e_, mwc.edges())]
            remove_edges = list(mwc_decision[mwc_decision == 'nomatch'].index)
            if len(remove_edges) > 0:
                undiscovered_errors.append(remove_edges)
                mwc.remove_edges_from(remove_edges)
            ccs = list(nx.connected_component_subgraphs(mwc))
            if len(remove_edges) > 0:
                # print(len(remove_edges))
                if len(ccs) <= len(remove_edges):
                    print('negatives not breaking things in two')
                    # import utool
                    # utool.embed()
                # print(len(ccs))
            gt_forests.append(ccs)
        # ut.dict_hist(ut.lmap(len, gt_forests))

        pos_edges_ = [[ut.lstarmap(infr.e_, t.edges()) for t in f]
                      for f in gt_forests]
        minimal_positive_edges = ut.flatten(ut.flatten(pos_edges_))
        minimal_positive_edges = list(set(minimal_positive_edges).difference(
            curr_decisions.index))

        priority_edges = minimal_positive_edges

        # need to also get all negative edges I think to know what the set of
        # negative edges is you need sequential information because it depends
        # on the current set of known positive edges. For this reason we
        # primarilly measure how many positive reviews we do.
        # WE hack negative edges and assume they are mostly skipped
        HACK_MIN_NEG_EDGES = False
        if HACK_MIN_NEG_EDGES:
            import vtool as vt
            node_to_nid = dict(infr.graph.nodes(data='orig_name_label'))
            is_nomatch = sim.primary_truth['nomatch']
            negative_edges = is_nomatch[is_nomatch].index.tolist()

            neg_nid_edges = ut.lstarmap(infr.e_, ut.unflat_take(node_to_nid,
                                                                negative_edges))
            # Group all negative edges between components
            groupxs = ut.group_indices(neg_nid_edges)[1]
            grouped_scores = vt.apply_grouping(
                primary_probs.loc[negative_edges]['match'].values,
                groupxs)
            max_xs = [xs[s.argmax()] for xs, s in zip(groupxs, grouped_scores)]

            # Take only negative edges that don't produce inconsistencies
            # given our current decisions
            split_flags = curr_decisions.loc[negative_edges] == 'match'

            split_flags = np.array([
                np.any(flags) for flags in
                vt.apply_grouping(split_flags.values, groupxs)
            ])
            max_xs2 = ut.compress(max_xs, ~split_flags)
            minimal_negative_edges = ut.take(negative_edges, max_xs2)
            priority_edges = priority_edges + minimal_negative_edges

        # priority_edges = minimal_positive_edges + minimal_negative_edges
        priorites = infr._get_priorites(priority_edges)
        priority_edges = ut.sortedby(priority_edges, priorites)[::-1]
        priorites = priorites[priorites.argsort()[::-1]]

        lowest_pos = infr._get_priorites(minimal_positive_edges).min()
        (priorites >= lowest_pos).sum()

        sim.results['n_pos_want'] = len(minimal_positive_edges)

        # primary_truth.loc[minimal_positive_edges]['match']
        # import numpy as np
        # true_ranks = np.where(primary_truth.loc[priority_edges]['match'])[0]

        # User will only review so many pairs for us
        max_user_reviews = 100
        user_edges = priority_edges[:max_user_reviews]

        infr.add_feedback_df(
            primary_truth.loc[user_edges].idxmax(axis=1),
            user_id='oracle', apply=True)

        # infr.relab()
        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)
        import utool
        with utool.embed_on_exception_context:
            print('n_inconsistent = %r' % (n_inconsistent,))
            assert n_inconsistent == 0, 'should not create any inconsistencies'

        if False:
            undiscovered_errors
            len(primary_truth.loc[user_edges].idxmax(axis=1))

        sim.results['n_user_clusters'] = n_clusters
        infr.apply_review_inference()

        curr_decisions = infr.edge_attr_df('reviewed_state')
        curr_truth = primary_truth.loc[curr_decisions.index].idxmax(axis=1)
        n_user_mistakes = curr_decisions != curr_truth
        sim.results['n_user_mistakes'] = sum(n_user_mistakes)

        if True:
            print("Post-User classification")
            from ibeis.scripts import clf_helpers
            user_pred = infr.infr_pred_df(primary_truth.index)
            # If we don't predict on a pair it is implicitly different
            user_pred[pd.isnull(user_pred)] = 'diff'
            # = user_pred.replace(None, 'diff')
            real_truth = pd.Series(infr.is_same(list(primary_truth.index)),
                                   index=primary_truth.index)
            real_truth = real_truth.replace(True, 'same').replace(False, 'diff')

            clf_helpers.classification_report2(real_truth, user_pred)
        return user_edges
        # pass

    @profile
    def oracle_review(sim):
        queue_params = {
            'pos_diameter': None,
            'neg_diameter': None,
        }
        infr = sim.infr
        prev = infr.verbose
        infr.verbose = 0
        # rng = np.random.RandomState(0)
        infr = sim.infr
        primary_truth = sim.primary_truth
        review_edges = infr.generate_reviews(**queue_params)
        max_reviews = 1000
        for count, (aid1, aid2) in enumerate(ut.ProgIter(review_edges)):
            state = primary_truth.loc[(aid1, aid2)].idxmax()
            tags = []
            infr.add_feedback(aid1, aid2, state, tags, apply=True,
                              rectify=False, user_id='oracle',
                              user_confidence='absolutely_sure')
            if count > max_reviews:
                break
        infr.verbose = prev

        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)
        import utool
        with utool.embed_on_exception_context:
            assert n_inconsistent == 0, 'should not create any inconsistencies'

        sim.results['n_user_clusters'] = n_clusters
        # infr.apply_review_inference()

        curr_decisions = infr.edge_attr_df('reviewed_state')
        curr_truth = primary_truth.loc[curr_decisions.index].idxmax(axis=1)
        n_user_mistakes = curr_decisions != curr_truth
        sim.results['n_user_mistakes'] = sum(n_user_mistakes)

        gt_clusters = ut.group_pairs(infr.gen_node_attrs('orig_name_label'))
        curr_clusters = ut.group_pairs(infr.gen_node_attrs('name_label'))

        compare_results = compare_groups(
            list(gt_clusters.values()),
            list(curr_clusters.values())
        )
        sim.results.update(ut.map_vals(len, compare_results))

        common_per_num = ut.group_items(compare_results['common'],
                                        map(len, compare_results['common']))
        greater = []
        for i in common_per_num.keys():
            if i > 4:
                greater.append(i)
        common_per_num['>4'] = ut.flatten(ut.take(common_per_num, greater))
        ut.delete_keys(common_per_num, greater)
        import six

        for k, v in common_per_num.items():
            if isinstance(k, six.string_types):
                sim.results['common@' + k] = len(v)
            else:
                sim.results['common@' + str(k)] = len(v)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.sim_graph_iden
        python -m ibeis.algo.hots.sim_graph_iden --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
