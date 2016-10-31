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
import numpy as np
import utool as ut
print, rrr, profile = ut.inject2(__name__)


def randn_clip(rng, mu, sigma, a_max, a_min):
    a = rng.randn() * sigma + mu
    a = np.clip(a, a_max, a_min)
    return a


def get_edge_truth(infr, n1, n2):
    nid1 = infr.graph.node[n1]['orig_name_label']
    nid2 = infr.graph.node[n2]['orig_name_label']
    return nid1 == nid2


def apply_dummy_scores(infr):
    rng = np.random.RandomState(0)
    dummy_params = {True: {'mu': .8, 'sigma': .2},
                    False: {'mu': .2, 'sigma': .2}}
    edges = infr.graph.edges()
    truths = [get_edge_truth(infr, n1, n2) for n1, n2 in edges]
    normscores = [randn_clip(rng, a_max=0, a_min=1, **dummy_params[truth])
                  for truth in truths]
    infr.set_edge_attrs('normscore', ut.dzip(edges, normscores))
    ut.nx_delete_edge_attr(infr.graph, '_dummy_edge')


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
    infr.set_node_attrs('shape', 'circle')

    dpath = ut.ensuredir(ut.truepath('~/Desktop/demo_graph_iden'))
    ut.remove_files_in_dir(dpath)

    def show_graph(infr, title):
        # showkw = dict(fontsize=6, show_cuts=False, with_colorbar=True)
        showkw = dict(fontsize=6, show_cuts=True, with_colorbar=True)
        infr.show_graph(**ut.update_existing(showkw.copy(), dict(with_colorbar=True)))
        pt.set_title(title)
        dpath = ut.ensuredir(ut.truepath('~/Desktop/demo_graph_iden'))
        pt.gca().set_aspect('equal')
        pt.save_figure(dpath=dpath)

    # Pin Nodes into the target groundtruth position
    infr.ensure_cliques()
    show_graph(infr, 'target-gt')
    infr.set_node_attrs('pin', 'true')

    def oracle_decision(infr, n1, n2):
        """ The perfect reviewer """
        truth = get_edge_truth(infr, n1, n2)
        state = infr.truth_texts[truth]
        tags = []
        return state, tags

    # Dummy scoring
    infr.ensure_full()
    apply_dummy_scores(infr)
    infr.remove_name_labels()
    infr.apply_weights()

    TARGET_REVIEW = None
    # TARGET_REVIEW = 9
    # TARGET_REVIEW = 7
    PRESHOW = True

    if PRESHOW or TARGET_REVIEW is None or TARGET_REVIEW == 0:
        show_graph(infr, 'pre-reveiw')

    for count, (aid1, aid2) in enumerate(infr.generate_reviews()):

        # Make the next review decision
        state, tags = oracle_decision(infr, aid1, aid2)

        if count == TARGET_REVIEW:
            infr.EMBEDME = True

        infr.add_feedback(aid1, aid2, state, tags, apply=True)

        # Show the result
        if PRESHOW or TARGET_REVIEW is None or count >= TARGET_REVIEW - 1:
            show_graph(infr, 'review #%d' % (count))

        if count == TARGET_REVIEW:
            break

    if not getattr(infr, 'EMBEDME', False):
        if ut.get_computer_name().lower() in ['hyrule', 'ooo']:
            pt.all_figures_tile(monitor_num=0, percent_w=.5)
        else:
            pt.all_figures_tile()
        ut.show_if_requested()


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

        # Now either a manual or automatic reviewer must
        # determine which matches are correct
        oracle_mode = True
        def oracle_decision(aid1, aid2):
            # Assume perfect reviewer
            nid1, nid2 = ibs.get_annot_nids([aid1, aid2])
            truth = nid1 == nid2
            state = infr.truth_texts[truth]
            tags = []
            # TODO:
            # if view1 != view1: infr.add_feedback(aid1, aid2, 'notcomp', apply=True)
            return state, tags

        # for count in ut.ProgIter(range(1, n_reviews + 1), 'review'):
        for count, (aid1, aid2) in enumerate(infr.generate_reviews()):
            if oracle_mode:
                state, tags = oracle_decision(aid1, aid2)
                # if total == 6:
                #     infr.add_feedback(8, 7, 'nomatch', apply=True)
                # else:
                infr.add_feedback(aid1, aid2, state, tags, apply=True)
                # infr.apply_feedback_edges()
                # infr.apply_weights()
                # infr.relabel_using_reviews()
                # infr.apply_cuts()
            else:
                raise NotImplementedError('review based on thresholded graph cuts')

            if (total) % graph_freq == 0:
                infr.show_graph(**showkw)
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


def do_infr_test(ccs, edges, new_edges):
    from ibeis.algo.hots import graph_iden
    import networkx as nx
    import plottool as pt
    pt.qt4ensure()
    G = nx.Graph()
    import numpy as np
    rng = np.random.RandomState(42)
    for cc in ccs:
        if len(cc) == 1:
            G.add_nodes_from(cc)
        G.add_path(cc, reviewed_state='match', reviewed_weight=1.0)
    for u, v, d in edges:
        reviewed_state = d.get('reviewed_state')
        if reviewed_state:
            if reviewed_state == 'match':
                d['reviewed_weight'] = 1.0
            if reviewed_state == 'nomatch':
                d['reviewed_weight'] = 0.0
            if reviewed_state == 'notcomp':
                d['reviewed_weight'] = 0.5
        else:
            d['normscore'] = rng.rand()

    G.add_edges_from(edges)
    infr = graph_iden.AnnotInference.from_netx(G)
    infr.relabel_using_reviews()
    infr.apply_cuts()
    infr.apply_review_inference()
    infr.apply_weights()
    infr.graph.graph['dark_background'] = True

    def repr_edge_data(infr, all_edge_data):
        visual_edge_data = {k: v for k, v in all_edge_data.items()
                            if k in infr.visual_edge_attrs}
        edge_data = ut.delete_dict_keys(all_edge_data, infr.visual_edge_attrs)
        lines = [
            ('visual_edge_data: ' + ut.repr2(visual_edge_data, nl=1)),
            ('edge_data: ' + ut.repr2(edge_data, nl=1)),
        ]
        return '\n'.join(lines)

    def on_pick(event, infr=None):
        print('ON PICK: %r' % (event,))
        artist = event.artist
        plotdat = pt.get_plotdat_dict(artist)
        if plotdat:
            if 'node' in plotdat:
                all_node_data = ut.sort_dict(plotdat['node_data'].copy())
                visual_node_data = ut.dict_subset(all_node_data, infr.visual_node_attrs, None)
                node_data = ut.delete_dict_keys(all_node_data, infr.visual_node_attrs)
                print('visual_node_data: ' + ut.repr2(visual_node_data, nl=1))
                print('node_data: ' + ut.repr2(node_data, nl=1))
                ut.cprint('node: ' + ut.repr2(plotdat['node']), 'blue')
                print('artist = %r' % (artist,))
            elif 'edge' in plotdat:
                all_edge_data = ut.sort_dict(plotdat['edge_data'].copy())
                print(repr_edge_data(infr, all_edge_data))
                ut.cprint('edge: ' + ut.repr2(plotdat['edge']), 'blue')
                print('artist = %r' % (artist,))
            else:
                print('???: ' + ut.repr2(plotdat))
        print(ut.get_timestamp())

    # Preshow
    fnum = 1
    if ut.show_was_requested():
        infr.set_node_attrs('shape', 'circle')
        infr.show(show_cuts=True, pnum=(2, 1, 1), fnum=fnum)
        pt.set_title('pre-review')
        pt.gca().set_aspect('equal')
        infr.set_node_attrs('pin', 'true')
        fig1 = pt.gcf()
        fig1.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr))

    infr1 = infr
    infr2 = infr.copy()
    for new_edge in new_edges:
        aid1, aid2, data = new_edge
        state = data['reviewed_state']
        infr2.add_feedback(aid1, aid2, state, apply=True)

    # Postshow
    if ut.show_was_requested():
        infr2.show(show_cuts=True, pnum=(2, 1, 2), fnum=fnum)
        pt.gca().set_aspect('equal')
        pt.set_title('post-review')
        fig2 = pt.gcf()
        if fig2 is not fig1:
            fig2.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr2))

    _errors = []
    def check(infr, u, v, key, val, msg):
        data = infr.get_edge_data(u, v)
        if data.get(key) != val:
            _errors.append(msg + '\nedge=' + ut.repr2((u, v)) + '\n' +
                           repr_edge_data(infr, data))

    def after(errors=[]):
        errors = errors + _errors
        if errors:
            ut.cprint('PRINTING %d FAILURE' % (len(errors)), 'red')
            for msg in errors:
                print(msg)
            ut.cprint('HAD %d FAILURE' % (len(errors)), 'red')
        if ut.show_was_requested():
            pt.all_figures_tile(percent_w=.5)
            ut.show_if_requested()
        if errors:
            raise AssertionError('There were errors')
    return infr1, infr2, after, check


def case_nomatch_infr():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_nomatch_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_nomatch_infr()
    """
    # Initial positive reviews
    ccs = [[1, 2, 3], [9]]
    # Add in initial reviews
    edges = [
        (9, 7, {'reviewed_state': 'nomatch', 'is_cut': True}),
        (1, 7, {'infered_state': None}),
        (1, 9, {'infered_state': None}),
    ]
    # Add in scored, but unreviewed edges
    new_edges = [(3, 9, {'reviewed_state': 'nomatch'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)

    check(infr2, 1, 7, 'infered_state', None,
          'negative review of an edge should not jump more than one compoment')

    check(infr2, 1, 9, 'infered_state', 'diff',
          'negative review of an edge should cut within one jump')

    after()


def case_match_infr():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_match_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_match_infr()
    """
    # Initial positive reviews
    ccs = [[2], [7], [9, 10]]
    # Add in initial reviews
    edges = [
        (9, 8, {'reviewed_state': 'nomatch', 'is_cut': True}),
        (7, 2, {}),
    ]
    # Add in scored, but unreviewed edges
    edges += [
        (2, 8, {'infered_state': None}),
        (2, 9, {'infered_state': None}),
    ]
    new_edges = [(2, 10, {'reviewed_state': 'match'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)

    check(infr2, 2, 9, 'infered_state', 'same', 'should infer a match')
    check(infr2, 2, 8, 'infered_state', 'diff', 'should infer a nomatch')
    check(infr1, 2, 7, 'infered_state', None, 'discon should have inference')

    check(infr2, 2, 7, 'infered_state', None, 'discon should have inference')
    after()


def case_inconsistent():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_inconsistent --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_inconsistent()
    """
    ccs = [[1, 2], [3, 4, 5]]  # [6, 7]]
    edges = [
        (2, 3, {'reviewed_state': 'nomatch', 'is_cut': True}),
    ]
    edges += [
        (4, 1, {'infered_state': None}),
        # (2, 7, {'infered_state': None}),
    ]
    new_edges = [(1, 5, {'reviewed_state': 'match'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)
    # Make sure the previously infered edge is no longer infered
    check(infr1, 4, 1, 'infered_state', 'diff', 'should initially be an infered diff')
    check(infr2, 4, 1, 'infered_state', None, 'should not be infered after incon')
    check(infr2, 4, 3, 'maybe_split', True, 'need to have a maybe split')
    after()


def case_redo_incon():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_redo_incon --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_redo_incon()
    """
    ccs = [[1, 2], [3, 4]]  # [6, 7]]
    edges = [
        (2, 3, {'reviewed_state': 'nomatch'}),
        (1, 4, {'reviewed_state': 'nomatch'}),
    ]
    edges += []
    new_edges = [(2, 3, {'reviewed_state': 'match'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)

    maybe_splits = infr2.get_edge_attrs('maybe_split')
    print('maybe_splits = %r' % (maybe_splits,))
    if not any(maybe_splits.values()):
        ut.cprint('FAILURE', 'red')
        print('At least one edge should be marked as a split')

    after()


def case_override_inference():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_override_inference --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_override_inference()
    """
    ccs = [[1, 2, 3, 4, 5]]
    edges = [
        (1, 3, {'infered_state': 'same'}),
        (1, 4, {'infered_state': 'same'}),
        # (1, 5, {'infered_state': 'same'}),
        (2, 4, {'infered_state': 'same'}),
        (2, 5, {'infered_state': 'same'}),
    ]
    edges += []
    new_edges = [
        (1, 5, {'reviewed_state': 'nomatch'}),
        (5, 2, {'reviewed_state': 'nomatch'}),
    ]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)
    # Make sure that the infered edges are no longer infered when an
    # inconsistent case is introduced
    check(infr2, 1, 4, 'maybe_split', False, 'should not split infered edge')
    check(infr2, 5, 2, 'infered_state', None, 'inference should be overriden')
    after()


def case_undo_match():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_undo_match --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_undo_match()
    """
    ccs = [[1, 2]]
    edges = []
    new_edges = [(1, 2, {'reviewed_state': 'nomatch'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)

    check(infr2, 1, 2, 'is_cut', True, 'should have cut edge')
    after()


def case_undo_nomatch():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_undo_nomatch --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_undo_nomatch()
    """
    ccs = [[1], [2]]
    edges = [
        (1, 2, {'reviewed_state': 'nomatch'}),
    ]
    new_edges = [(1, 2, {'reviewed_state': 'match'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 2, 'is_cut', False, 'should have matched edge')
    after()


def case_incon_removes_inference():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_incon_removes_inference --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_incon_removes_inference()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (3, 4, {'reviewed_state': 'nomatch'}),
        (1, 5, {'reviewed_state': 'nomatch'}),
        (2, 5, {}),
        (1, 6, {}),
    ]
    new_edges = [(3, 4, {'reviewed_state': 'match'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)

    check(infr1, 2, 5, 'infered_state', 'diff', 'should be preinfered')
    check(infr2, 2, 5, 'infered_state', None, 'should be uninfered on incon')
    after()


def case_inferable_notcomp1():
    """
    make sure notcomparable edges can be infered

    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_inferable_notcomp1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_inferable_notcomp1()
    """
    ccs = [[1, 2], [3, 4]]
    edges = [
        (2, 3, {'reviewed_state': 'nomatch'}),
    ]
    new_edges = [(1, 4, {'reviewed_state': 'notcomp'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 4, 'infered_state', 'diff', 'should be infered')
    after()


def case_inferable_update_notcomp():
    """
    make sure inference updates for nocomparable edges

    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_inferable_update_notcomp --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_inferable_update_notcomp()
    """
    ccs = [[1, 2], [3, 4]]
    edges = [
        (2, 3, {'reviewed_state': 'nomatch'}),
        (1, 4, {'reviewed_state': 'notcomp'}),
    ]
    new_edges = [(2, 3, {'reviewed_state': 'match'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 1, 4, 'infered_state', 'diff', 'should be infered diff')
    check(infr2, 1, 4, 'infered_state', 'same', 'should be infered same')
    after()


def case_notcomp_remove_infr():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_notcomp_remove_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_notcomp_remove_infr()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (1, 4, {'reviewed_state': 'match'}),
        # (1, 4, {'reviewed_state': 'notcomp'}),
        (2, 5, {'reviewed_state': 'notcomp'}),
        (3, 6, {'reviewed_state': 'notcomp'}),
    ]
    new_edges = [(1, 4, {'reviewed_state': 'notcomp'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 4, 'infered_state', None, 'can not infer match here!')
    check(infr2, 2, 5, 'infered_state', None, 'can not infer match here!')
    check(infr2, 3, 6, 'infered_state', None, 'can not infer match here!')
    after()


def case_notcomp_remove_cuts():
    """
    CommandLine:
        python -m ibeis.algo.hots.demo_graph_iden case_notcomp_remove_cuts --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.demo_graph_iden import *  # NOQA
        >>> case_notcomp_remove_cuts()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (1, 4, {'reviewed_state': 'nomatch'}),
        # (1, 4, {'reviewed_state': 'notcomp'}),
        (2, 5, {'reviewed_state': 'notcomp'}),
        (3, 6, {'reviewed_state': 'notcomp'}),
    ]
    new_edges = [(1, 4, {'reviewed_state': 'notcomp'})]
    infr1, infr2, after, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 4, 'is_cut', False, 'can not infer cut here!')
    check(infr2, 2, 5, 'is_cut', False, 'can not infer cut here!')
    check(infr2, 3, 6, 'is_cut', False, 'can not infer cut here!')
    after()

if __name__ == '__main__':
    r"""
    CommandLine:
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7 --graph
        python -m ibeis.algo.hots.demo_graph_iden
        python -m ibeis.algo.hots.demo_graph_iden --allexamples
        python -m ibeis.algo.hots.demo_graph_iden --allexamples --show
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
