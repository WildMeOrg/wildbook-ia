# -*- coding: utf-8 -*-
r"""
Review Interactions


Key:
    A --- B : A and B are potentially connected. No review.
    A -X- B : A and B have been reviewed as non matching.
    A -?- B : A and B have been reviewed as not comparable.
    A -O- B : A and B have been reviewed as matching.


The Total Review Clique Compoment

    A -O- B -|
    |\    |  |
    O  O  O  |
    |    \|  |
    C -O- D  |
    |________O


A Minimal Review Compoment

    A -O- B -|     A -O- B
    |\    |  |     |     |
    O  \  O  | ==  O     O
    |    \|  |     |     |
    C --- D  |     C     D
    |________|


Inconsistent Compoment

    A -O- B
    |    /
    O  X
    |/
    C


Consistent Compoment (with not-comparable)

    A -O- B
    |    /
    O  ?
    |/
    C


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
print, rrr, profile = ut.inject2(__name__)


@profile
def demo_graph_iden():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden demo_graph_iden --show
    """
    from ibeis.algo.hots import graph_iden
    import ibeis
    ibs = ibeis.opendb('PZ_MTEST')
    # Initially the entire population is unnamed
    graph_freq = 1
    n_reviews = 5
    n_queries = 1
    # n_reviews = 4
    # n_queries = 1
    # n_reviews = 3
    aids = ibs.get_valid_aids()[1:9]
    # nids = [-aid for aid in aids]
    # infr = graph_iden.AnnotInference(ibs, aids, nids=nids, autoinit=True, verbose=1)
    infr = graph_iden.AnnotInference(ibs, aids, autoinit=True, verbose=1)
    # Pin nodes in groundtruth positions
    infr.ensure_cliques()
    # infr.initialize_visual_node_attrs()
    import plottool as pt
    showkw = dict(fontsize=8, show_cuts=True, with_colorbar=True)
    infr.show_graph(**ut.update_existing(showkw.copy(), dict(with_colorbar=True)))
    # pt.dark_background(force=True)
    pt.set_title('target-gt')
    infr.set_node_attrs('pin', 'true')
    infr.remove_name_labels()
    infr.remove_dummy_edges()

    total = 0
    for query_num in range(n_queries):

        # Build hypothesis links
        infr.exec_matching()
        infr.apply_match_edges(dict(ranks_top=3, ranks_bot=1))
        infr.apply_match_scores()
        infr.apply_feedback_edges()
        infr.apply_weights()

        # infr.relabel_using_reviews()
        # infr.apply_cuts()
        if query_num == 0:
            infr.show_graph(**ut.update_existing(showkw.copy(), dict(with_colorbar=True)))
            pt.set_title('pre-review-%r' % (query_num))
            # pt.dark_background(force=True)

        # Now either a manual or automatic reviewer must
        # determine which matches are correct
        oracle_mode = True
        def oracle_decision(aid1, aid2):
            # Assume perfect reviewer
            nid1, nid2 = ibs.get_annot_nids([aid1, aid2])
            truth = nid1 == nid2
            status = infr.truth_texts[truth]
            tags = []
            # TODO:
            # if view1 != view1: infr.add_feedback(aid1, aid2, 'notcomp', apply=True)
            return status, tags

        # for count in ut.ProgIter(range(1, n_reviews + 1), 'review'):
        for count, (aid1, aid2) in enumerate(infr.generate_reviews()):
            if oracle_mode:
                status, tags = oracle_decision(aid1, aid2)
                # if total == 6:
                #     infr.add_feedback(8, 7, 'nomatch', apply=True)
                # else:
                infr.add_feedback(aid1, aid2, status, tags, apply=True)
                # infr.apply_feedback_edges()
                # infr.apply_weights()
                # infr.relabel_using_reviews()
                # infr.apply_cuts()
            else:
                raise NotImplementedError('review based on thresholded graph cuts')

            if (total) % graph_freq == 0:
                infr.show_graph(**showkw)
                # pt.dark_background(force=True)
                pt.set_title('review #%d-%d' % (total, query_num))
                # print(ut.repr3(ut.graph_info(infr.graph)))
                if 0:
                    _info = ut.graph_info(infr.graph, stats=True,
                                          ignore=(infr.visual_edge_attrs +
                                                  infr.visual_node_attrs))
                    _info = ut.graph_info(infr.graph, stats=False,
                                          ignore=(infr.visual_edge_attrs +
                                                  infr.visual_node_attrs))
                    print(ut.repr3(_info, precision=2))
            if count >= n_reviews:
                break
            total += 1

    if (total) % graph_freq != 0:
        infr.show_graph(**showkw)
        # pt.dark_background(force=True)
        pt.set_title('review #%d-%d' % (total, query_num))
        # print(ut.repr3(ut.graph_info(infr.graph)))
        if 0:
            _info = ut.graph_info(infr.graph, stats=True,
                                  ignore=(infr.visual_edge_attrs +
                                          infr.visual_node_attrs))
            print(ut.repr3(_info, precision=2))

    # print(ut.repr3(ut.graph_info(infr.graph)))
    # infr.show_graph()
    # pt.set_title('post-review')

    if ut.get_computer_name() in ['hyrule']:
        pt.all_figures_tile(monitor_num=0, percent_w=.5)
    elif ut.get_computer_name() in ['ooo']:
        pt.all_figures_tile(monitor_num=1, percent_w=.5)
    else:
        pt.all_figures_tile()
    ut.show_if_requested()


@profile
def demo_graph_iden2():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden demo_graph_iden2 --show
    """
    from ibeis.algo.hots import graph_iden
    import plottool as pt
    # Create dummy data
    nids = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
    aids = range(len(nids))
    infr = graph_iden.AnnotInference(None, aids, nids=nids, autoinit=True, verbose=1)
    infr.ensure_cliques()

    showkw = dict(fontsize=8, show_cuts=False, with_colorbar=True)

    # Pin Nodes into groundtruth position
    infr.show_graph(**ut.update_existing(showkw.copy(), dict(with_colorbar=True)))
    pt.set_title('target-gt')
    infr.set_node_attrs('pin', 'true')
    infr.remove_name_labels()
    infr.remove_dummy_edges()

    def get_truth(n1, n2):
        return G.node[n1]['orig_name_label'] == G.node[n2]['orig_name_label']

    def oracle_decision(n1, n2):
        # Assume perfect reviewer
        truth = get_truth(n1, n2)
        status = infr.truth_texts[truth]
        tags = []
        return status, tags

    # Dummy scoring
    infr.ensure_full()
    G = infr.graph
    edge_to_truth = {
        (n1, n2): get_truth(n1, n2)
        for n1, n2 in infr.graph.edges()
    }
    import numpy as np
    rng = np.random.RandomState(0)
    dummy_params = {
        True: {'mu': .8, 'sigma': .2},
        False: {'mu': .2, 'sigma': .2},
    }
    def randpn(mu, sigma):
        return np.clip((rng.randn() * sigma) + mu, 0, 1)
    edge_to_normscore = {
        (n1, n2): randpn(**dummy_params[get_truth(n1, n2)])
        for (n1, n2), truth in edge_to_truth.items()
    }
    print('edge_to_normscore = %r' % (edge_to_normscore,))
    infr.set_edge_attrs('normscore', edge_to_normscore)
    ut.nx_delete_edge_attr(infr.graph, '_dummy_edge')
    infr.apply_weights()

    infr.show_graph(**showkw)
    pt.set_title('target-gt')

    for count, (aid1, aid2) in enumerate(infr.generate_reviews()):
        status, tags = oracle_decision(aid1, aid2)
        infr.add_feedback(aid1, aid2, status, tags, apply=True)

        infr.show_graph(**showkw)
        pt.set_title('review #%d' % (count))

        if count > 15:
            break

    if ut.get_computer_name().lower() in ['hyrule', 'ooo']:
        pt.all_figures_tile(monitor_num=0, percent_w=.5)
    else:
        pt.all_figures_tile()
    ut.show_if_requested()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden
        python -m ibeis.algo.hots.demo_graph_iden --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
