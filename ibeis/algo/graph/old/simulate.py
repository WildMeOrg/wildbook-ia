# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import utool as ut
print, rrr, profile = ut.inject2(__name__)


def compare_groups(true_groups, pred_groups):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.simulate import *  # NOQA
        >>> true_groups = [
        >>>   [20, 21], [22, 23], [1, 2], [12, 13, 14], [4], [5, 6, 3], [7, 8],
        >>>   [9, 10, 11], [31, 32, 33, 34, 35],   [41, 42, 43, 44], [45], [50]
        >>> ]
        >>> pred_groups = [
        >>>     [20, 21, 22, 23], [1, 2], [12], [13, 14], [3, 4], [5, 6,11],
        >>>     [7], [8, 9], [10], [31, 32], [33, 34, 35], [41, 42, 43, 44, 45]
        >>> ]
        >>> comparisons = compare_groups(true_groups, pred_groups)
        >>> print(comparisons)
        >>> print(ut.repr4(comparisons))
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
    pred_merges = []
    true_merges = []
    for ts in true_sets:
        ccs = set([pred_conn.get(t, frozenset()) for t in ts])
        if frozenset.union(*ccs) == ts:
            # This is a pure merge
            pred_merges.append(ccs)
            true_merges.append(ts)

    # How many predictions can be split into perfect pieces?
    true_splits = []
    pred_splits = []
    for ps in pred_sets:
        ccs = set([true_conn.get(p, frozenset()) for p in ps])
        if frozenset.union(*ccs) == ps:
            # This is a pure merge
            true_splits.append(ccs)
            pred_splits.append(ps)

    pred_merges_flat = ut.flatten(pred_merges)
    true_splits_flat = ut.flatten(true_splits)

    pred_hybrid = frozenset(map(frozenset, pred_sets)).difference(
        set(pred_splits + pred_merges_flat))

    true_hybrid = frozenset(map(frozenset, true_sets)).difference(
        set(true_merges + true_splits_flat))

    comparisons = {
        'common': common,
        # 'true_splits_flat': true_splits_flat,
        'true_splits': true_splits,
        'true_merges': true_merges,
        'true_hybrid': true_hybrid,
        'pred_splits': pred_splits,
        'pred_merges': pred_merges,
        # 'pred_merges_flat': pred_merges_flat,
        'pred_hybrid': pred_hybrid,
    }
    return comparisons


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
        n_names_possible = 0
        real_groups = ut.group_pairs(infr.gen_node_attrs('orig_name_label'))
        possible_clusters = []
        for nid, nodes in real_groups.items():
            if len(nodes) == 1:
                possible_clusters.append(nodes)
                n_names_possible += 1
                continue
            cc_cand_edges = list(ut.nx_edges_between(infr.graph, nodes))
            cc = ut.nx_from_node_edge(nodes, cc_cand_edges)
            mst = nx.minimum_spanning_tree(cc)
            ccs = list(nx.connected_components(mst))
            possible_clusters.extend(ccs)
            n_names_possible += (len(ccs))

        sumafter = 3

        best_possible_compare_results = compare_groups(
            list(real_groups.values()),
            list(possible_clusters)
        )
        possible_per_num = ut.map_vals(
            len, ut.group_items(best_possible_compare_results['common'],
                                map(len, best_possible_compare_results['common'])))
        greater = [i for i in possible_per_num.keys() if i > sumafter]
        possible_per_num['>%s' % sumafter] = sum(ut.take(possible_per_num, greater))
        ut.delete_keys(possible_per_num, greater)
        for k, v in possible_per_num.items():
            sim.results['possible@' + str(k)] = v
        sim.results['possible'] = len(best_possible_compare_results['common'])

        # Measure the number of real names in the test (per number of annots)
        real_per_num = ut.dict_hist(map(len, real_groups.values()))
        greater = [i for i in real_per_num.keys() if i > sumafter]
        real_per_num['>%s' % sumafter] = sum(ut.take(real_per_num, greater))
        ut.delete_keys(real_per_num, greater)
        for k, v in real_per_num.items():
            sim.results['real@' + str(k)] = v

        sim.results['n_names_possible'] = n_names_possible
        sim.results['n_names_real'] = len(real_groups)
        sim.results['real'] = len(real_groups)

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
                'decision', 'unreviewed', edges=edges,
                default='unreviewed'))
            incon_edges.extend(reviewed_edges)
            n_worst_case += len(reviewed_edges)

        incon_truth = primary_truth.loc[incon_edges].idxmax(axis=1)
        # incon_pred1 = infr.edge_attr_df('decision', incon_edges)

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
                    print('ERROR')
                    import sys
                    sys.exit(1)
                # Stop once inconsistent compmoents stop coming up
                break
            else:
                pass
                # print('Fixing edge: %r' % (d,))
            prev_state = d['decision']
            state = primary_truth.loc[(aid1, aid2)].idxmax()
            if state == prev_state:
                n_superflouous += 1
            tags = []
            infr.add_feedback((aid1, aid2), decision=state, tags=tags,
                               rectify=False, user_id='oracle',
                               confidence='absolutely_sure')
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
        print('n_inconsistent = %r' % (n_inconsistent,))
        assert n_inconsistent == 0, 'should have fixed everything'

        if False:
            from ibeis.scripts import clf_helpers
            incon_pred2 = infr.edge_attr_df('decision', incon_edges)

            print('--------')
            # clf_helpers.classification_report2(incon_truth, incon_pred1)
            print('--------')
            clf_helpers.classification_report2(incon_truth, incon_pred2)

        infr.verbose = prev

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
                              confidence='absolutely_sure')
            if count > max_reviews:
                break
        infr.verbose = prev

        sim.results['max_reviews'] = max_reviews

        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)
        assert n_inconsistent == 0, 'should not create any inconsistencies'

        sim.results['n_user_clusters'] = n_clusters
        # infr.apply_review_inference()

        curr_decisions = infr.edge_attr_df('decision')
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
        sumafter = 3
        greater = [i for i in common_per_num.keys() if i > sumafter]
        common_per_num['>%s' % sumafter] = ut.flatten(ut.take(common_per_num,
                                                              greater))
        ut.delete_keys(common_per_num, greater)
        for k, v in common_per_num.items():
            sim.results['common@' + str(k)] = len(v)

        sim.results['n_names_common'] = len(compare_results['common'])


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.graph.simulate
        python -m ibeis.algo.graph.simulate --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
