# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools as it
import numpy as np
import utool as ut
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV
print, rrr, profile = ut.inject2(__name__)


def make_dummy_infr(annots_per_name):
    import ibeis
    nids = [val for val, num in enumerate(annots_per_name, start=1)
            for _ in range(num)]
    aids = range(len(nids))
    infr = ibeis.AnnotInference(None, aids, nids=nids, autoinit=True,
                                verbose=1)
    return infr


def simple_simulation():
    # ---- Synthetic data params
    queue_params = {
        'pos_redun': 2,
        'neg_redun': 2,
    }
    oracle_accuracy = 1.0
    # .99

    infr = demodata_infr(num_pccs=200, size=4, size_std=1, ignore_pair=True)
    infr.queue_params.update(**queue_params)
    infr.review_dummy_edges(method='clique')

    # FIXME: make an implicit negative edge truth state

    infr.ensure_full()
    # infr.apply_edge_truth()
    # Dummy scoring
    rng = np.random.RandomState(0)
    apply_dummy_scores(infr, rng)
    infr.init_simulation(oracle_accuracy=oracle_accuracy, name='simple_sim')
    infr_gt = infr.copy()

    def dummy_ranker(u):
        u_edges = list(infr_gt.graph.neighbors(u))
        u_probs = []
        for v in u_edges:
            prob = infr_gt.graph.edge[u][v]['prob_match']
            u_probs.append(prob)
        k = 5
        sortx = np.argsort(u_probs)[::-1][0:k]
        ranked_edges = [(u, v) if u < v else (v, u)
                        for v in ut.take(u_edges, sortx)]
        assert len(ranked_edges) == k
        return ranked_edges

    def find_dummy_candidate_edges():
        new_edges = []
        for u in infr.graph.nodes():
            new_edges.extend(dummy_ranker(u))
        new_edges = set(new_edges)
        return new_edges

    infr.clear_feedback()
    infr.clear_name_labels()
    infr.clear_edges()

    infr.prioritize('prob_match')
    new_edges = find_dummy_candidate_edges()
    infr.add_new_candidate_edges(new_edges)
    infr.init_refresh()

    # remains = []
    # mus = []
    # window = 500

    for edge, priority in infr._generate_reviews(data=True):
        if len(infr.queue) <= 10:
            break
        feedback = infr.request_oracle_review(edge)
        infr.add_feedback(edge, **feedback)

        # num_pccs = infr.pos_graph.number_of_components()
        # num_edges_remain = (num_pccs * num_pccs - 1) / 2
        # num_neg_redun = infr.neg_redun_nids.number_of_edges()
        # num_edges_need = num_edges_remain - num_neg_redun

        # mu = np.mean(infr.refresh.manual_decisions[-window:])
        # remains.append(num_edges_need)
        # mus.append(mu)

    # remains = np.array(remains)
    # mus = np.array(mus)

    # upper_bound_left = remains[window:] * mus[window:]
    # print(upper_bound_left.min())

    # pt.plot(upper_bound_left, label=window)
    # pt.set_xlabel('number of manual reviews')
    # pt.set_ylabel('upper bound on expected number of remaining merges')
    # pt.set_title('Poisson Convergence')
    return infr


@profile
def demo2():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo demo2 --viz
    """
    import plottool as pt

    # ---- Synthetic data params
    queue_params = {
        'pos_redun': 2,
        'neg_redun': 2,
    }
    # oracle_accuracy = .98
    # oracle_accuracy = .90
    oracle_accuracy = .8
    # oracle_accuracy = 1.0

    # --- draw params

    VISUALIZE = ut.get_argflag('--viz')
    # QUIT_OR_EMEBED = 'embed'
    QUIT_OR_EMEBED = 'quit'
    TARGET_REVIEW = ut.get_argval('--target', type_=int, default=None)
    PRESHOW = VISUALIZE

    # ------------------

    # rng = np.random.RandomState(42)

    # infr = demodata_infr(num_pccs=4, size=3, size_std=1, p_incon=0)
    infr = demodata_infr(num_pccs=6, size=7, size_std=1, p_incon=0)
    # apply_dummy_viewpoints(infr)
    # infr.ensure_cliques()
    infr.review_dummy_edges(method='clique')
    infr.ensure_full()
    # infr.apply_edge_truth()
    # Dummy scoring
    # apply_dummy_scores(infr, rng)

    infr.init_simulation(oracle_accuracy=oracle_accuracy, name='demo2')

    infr_gt = infr.copy()

    dpath = ut.ensuredir(ut.truepath('~/Desktop/demo'))
    ut.remove_files_in_dir(dpath)

    def show_graph(infr, title, final=False):
        if not VISUALIZE:
            return
        latest = '\n'.join(infr.latest_logs())
        showkw = dict(
            # fontsize=infr.graph.graph['fontsize'],
            # fontname=infr.graph.graph['fontname'],
            show_unreviewed_edges=True,
            show_inferred_same=False,
            show_inferred_diff=False,
            # show_inferred_same=True,
            # show_inferred_diff=True,
            show_labels=True,
            show_recent_review=not final,
            # splines=infr.graph.graph['splines'],
            reposition=False,
            # with_colorbar=True
        )
        verbose = infr.verbose
        infr.verbose = 0
        infr_ = infr.copy()
        infr_ = infr
        infr_.verbose = verbose
        infr_.show(pickable=True, **showkw)
        infr.verbose = verbose
        print('status ' + ut.repr4(infr_.status()))
        # infr.show(**showkw)
        pt.set_title(title)
        ax = pt.gca()
        fig = pt.gcf()
        pt.adjust_subplots(top=.95, left=0, right=1, bottom=.2, fig=fig)
        ax.set_aspect('equal')
        ax.set_xlabel(latest)
        dpath = ut.ensuredir(ut.truepath('~/Desktop/demo'))
        pt.save_figure(dpath=dpath, dpi=128, figsize=(9, 10))
        infr.latest_logs()

    if VISUALIZE:
        infr.update_visual_attrs(groupby='name_label')
        infr.set_node_attrs('pin', 'true')
        print(ut.repr4(infr.graph.node[1]))

    if VISUALIZE:
        infr.latest_logs()
        # Pin Nodes into the target groundtruth position
        show_graph(infr, 'target-gt')

    def dummy_ranker(u):
        u_edges = list(infr_gt.graph.neighbors(u))
        u_probs = []
        for v in u_edges:
            prob = infr_gt.graph.edge[u][v]['prob_match']
            u_probs.append(prob)
        k = 10
        sortx = np.argsort(u_probs)[::-1][0:k]
        ranked_edges = [(u, v) if u < v else (v, u)
                        for v in ut.take(u_edges, sortx)]
        assert len(ranked_edges) == k
        return ranked_edges

    def find_dummy_candidate_edges():
        new_edges = []
        for u in infr.graph.nodes():
            new_edges.extend(dummy_ranker(u))
        new_edges = set(new_edges)
        return new_edges

    print(ut.repr4(infr.status()))
    infr.clear_feedback()
    infr.clear_name_labels()
    infr.clear_edges()
    print(ut.repr4(infr.status()))
    infr.latest_logs()

    if VISUALIZE:
        infr.update_visual_attrs()

    infr.prioritize('prob_match')
    if PRESHOW or TARGET_REVIEW is None or TARGET_REVIEW == 0:
        show_graph(infr, 'pre-reveiw')

    def on_new_candidate_edges(infr, edges):
        # hack updateing visual attrs as a callback
        infr.update_visual_attrs()

    infr.on_new_candidate_edges = on_new_candidate_edges

    infr.queue_params.update(**queue_params)
    infr.print('Searching for candidates')
    new_edges = find_dummy_candidate_edges()
    infr.add_new_candidate_edges(new_edges)

    if PRESHOW or TARGET_REVIEW is None or TARGET_REVIEW == 0:
        show_graph(infr, 'find-candidates')

    # _iter2 = enumerate(infr.generate_reviews(**queue_params))
    # _iter2 = list(_iter2)
    # assert len(_iter2) > 0

    # prog = ut.ProgIter(_iter2, label='demo2', bs=False, adjust=False,
    #                    enabled=False)
    count = 0
    for edge, priority in infr._generate_reviews(data=True):
        msg = 'review #%d, priority=%.2f' % (count, priority)
        print('\n----------')
        infr.print(msg)
        # print('remaining_reviews = %r' % (infr.remaining_reviews()),)
        # Make the next review decision
        feedback = infr.request_oracle_review(edge)
        if count == TARGET_REVIEW:
            infr.EMBEDME = QUIT_OR_EMEBED == 'embed'
        infr.add_feedback(edge, **feedback)
        infr.print('len(queue) = %r' % (len(infr.queue)))
        # infr.apply_nondynamic_update()
        # Show the result
        if PRESHOW or TARGET_REVIEW is None or count >= TARGET_REVIEW - 1:
            show_graph(infr, msg)
            # import sys
            # sys.exit(1)
        if count == TARGET_REVIEW:
            break
        count += 1

    show_graph(infr, 'post-review', final=True)

    # ROUND 2 FIGHT
    # if TARGET_REVIEW is None and round2_params is not None:
    #     # HACK TO GET NEW THINGS IN QUEUE
    #     infr.queue_params = round2_params

    #     _iter2 = enumerate(infr.generate_reviews(**queue_params))
    #     prog = ut.ProgIter(_iter2, label='round2', bs=False, adjust=False,
    #                        enabled=False)
    #     for count, (aid1, aid2) in prog:
    #         msg = 'reviewII #%d' % (count)
    #         print('\n----------')
    #         print(msg)
    #         print('remaining_reviews = %r' % (infr.remaining_reviews()),)
    #         # Make the next review decision
    #         feedback = infr.request_oracle_review(edge)
    #         if count == TARGET_REVIEW:
    #             infr.EMBEDME = QUIT_OR_EMEBED == 'embed'
    #         infr.add_feedback(edge, **feedback)
    #         # Show the result
    #         if PRESHOW or TARGET_REVIEW is None or count >= TARGET_REVIEW - 1:
    #             show_graph(infr, msg)
    #         if count == TARGET_REVIEW:
    #             break

    #     show_graph(infr, 'post-re-review', final=True)

    if not getattr(infr, 'EMBEDME', False):
        if ut.get_computer_name().lower() in ['hyrule', 'ooo']:
            pt.all_figures_tile(monitor_num=0, percent_w=.5)
        else:
            pt.all_figures_tile()
        ut.show_if_requested()


valid_views = ['L', 'F', 'R', 'B']
adjacent_views = {
    v: [valid_views[(count + i) % len(valid_views)] for i in [-1, 0, 1]]
    for count, v in enumerate(valid_views)
}


def get_edge_truth(infr, n1, n2):
    nid1 = infr.graph.node[n1]['orig_name_label']
    nid2 = infr.graph.node[n2]['orig_name_label']
    try:
        view1 = infr.graph.node[n1]['viewpoint']
        view2 = infr.graph.node[n2]['viewpoint']
        comparable = view1 in adjacent_views[view2]
    except KeyError:
        comparable = True
        # raise
    same = nid1 == nid2

    if not comparable:
        return 2
    else:
        return int(same)


def apply_dummy_scores(infr, rng=None):
    """
    CommandLine:
        python -m ibeis.algo.graph.demo apply_dummy_scores --show

    Ignore:
        import sympy
        x, y = sympy.symbols('x, y', positive=True, real=True)

        num_pos_edges = (x * (x - 1)) / 2 * y
        num_neg_edges = x * x * (y - 1)
        subs = {x: 3, y: 3}
        print(num_pos_edges.evalf(subs=subs))
        print(num_neg_edges.evalf(subs=subs))
        sympy.solve(num_pos_edges - num_neg_edges)

        2*x/(x + 1) - y

    Example:
        >>> # DISABLE_DOCTEST
        >>> import vtool as vt
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> infr = make_dummy_infr([100] * 2)
        >>> infr.ensure_full()
        >>> rng = None
        >>> #infr.apply_edge_truth()
        >>> apply_dummy_scores(infr, rng)
        >>> edges = list(infr.graph.edges())
        >>> truths = np.array(ut.take(infr.edge_truth, edges))
        >>> edge_to_normscore = infr.get_edge_attrs('prob_match')
        >>> scores = np.array(list(ut.take(edge_to_normscore, edges)))
        >>> scorenorm = vt.ScoreNormalizer()
        >>> import ibeis.constants as const
        >>> truth_enc = np.array(ut.take(const.REVIEW.CODE_TO_INT, truths))
        >>> scorenorm.fit(scores, truth_enc)
        >>> ut.quit_if_noshow()
        >>> scorenorm.visualize()
        >>> ut.show_if_requested()
    """
    print('[demo] apply dummy scores')
    rng = ut.ensure_rng(rng)
    dummy_params = {
        NEGTV: {'mean': .2, 'std': .25},
        POSTV: {'mean': .8, 'std': .2},
        INCMP: {'mean': .2, 'std': .4},
    }
    # edges = list(infr.graph.edges())
    grouped_edges = ut.group_pairs(infr.edge_truth.items())
    for key, group in grouped_edges.items():
        probs = randn(shape=[len(group)], rng=rng, a_max=1, a_min=0,
                      **dummy_params[key])
        infr.set_edge_attrs('prob_match', ut.dzip(group, probs))


def apply_dummy_viewpoints(infr):
    transition_rate = .5
    transition_rate = 0
    valid_views = ['L', 'F', 'R', 'B']
    rng = np.random.RandomState(42)
    class MarkovView(object):
        def __init__(self):
            self.dir_ = +1
            self.state = 0

        def __call__(self):
            return self.next_state()

        def next_state(self):
            if self.dir_ == -1 and self.state <= 0:
                self.dir_ = +1
            if self.dir_ == +1 and self.state >= len(valid_views) - 1:
                self.dir_ = -1
            if rng.rand() < transition_rate:
                self.state += self.dir_
            return valid_views[self.state]
    mkv = MarkovView()
    nid_to_aids = ut.group_pairs([
        (n, d['name_label']) for n, d in infr.graph.nodes(data=True)])
    grouped_nodes = list(nid_to_aids.values())
    node_to_view = {node: mkv() for nodes in grouped_nodes for node in nodes}
    infr.set_node_attrs('viewpoint', node_to_view)


def make_demo_infr(ccs, edges=[], nodes=[], infer=True):
    import ibeis
    import networkx as nx

    if nx.__version__.startswith('1'):
        nx.add_path = nx.Graph.add_path

    G = ibeis.AnnotInference._graph_cls()
    G.add_nodes_from(nodes)
    import numpy as np
    rng = np.random.RandomState(42)
    for cc in ccs:
        if len(cc) == 1:
            G.add_nodes_from(cc)
        nx.add_path(G, cc, decision=POSTV, reviewed_weight=1.0)
    for edge in edges:
        u, v, d = edge if len(edge) == 3 else tuple(edge) + ({},)
        decision = d.get('decision')
        if decision:
            if decision == POSTV:
                d['reviewed_weight'] = 1.0
            if decision == NEGTV:
                d['reviewed_weight'] = 0.0
            if decision == INCMP:
                d['reviewed_weight'] = 0.5
        else:
            d['prob_match'] = rng.rand()

    G.add_edges_from(edges)
    infr = ibeis.AnnotInference.from_netx(G, infer=infer)
    infr.verbose = 3

    infr.relabel_using_reviews(rectify=False)
    # infr.apply_nondynamic_update()

    infr.graph.graph['dark_background'] = False
    infr.graph.graph['ignore_labels'] = True
    infr.set_node_attrs('width', 40)
    infr.set_node_attrs('height', 40)
    # infr.set_node_attrs('fontsize', fontsize)
    # infr.set_node_attrs('fontname', fontname)
    infr.set_node_attrs('fixed_size', True)
    return infr


def do_infr_test(ccs, edges, new_edges):
    """
    Creates a graph with `ccs` + `edges` and then adds `new_edges`
    """
    # import networkx as nx
    import plottool as pt

    infr = make_demo_infr(ccs, edges)

    pt.qtensure()

    # Preshow
    fnum = 1
    if ut.show_was_requested():
        infr.set_node_attrs('shape', 'circle')
        infr.show(pnum=(2, 1, 1), fnum=fnum, show_unreviewed_edges=True,
                  show_reviewed_cuts=True,
                  splines='spline',
                  show_inferred_diff=True, groupby='name_label',
                  show_labels=True, pickable=True)
        pt.set_title('pre-review')
        pt.gca().set_aspect('equal')
        infr.set_node_attrs('pin', 'true')
        # fig1 = pt.gcf()
        # fig1.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr))

    infr1 = infr
    infr2 = infr.copy()
    for new_edge in new_edges:
        aid1, aid2, data = new_edge
        state = data['decision']
        infr2.add_feedback((aid1, aid2), state)
    infr2.relabel_using_reviews(rectify=False)
    infr2.apply_nondynamic_update()

    # Postshow
    if ut.show_was_requested():
        infr2.show(pnum=(2, 1, 2), fnum=fnum, show_unreviewed_edges=True,
                   show_inferred_diff=True, show_labels=True)
        pt.gca().set_aspect('equal')
        pt.set_title('post-review')
        # fig2 = pt.gcf()
        # if fig2 is not fig1:
        #     fig2.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr2))

    class Checker(object):
        """
        Asserts pre and post test properties of the graph
        """
        def __init__(self, infr1, infr2):
            self._errors = []
            self.infr1 = infr1
            self.infr2 = infr2

        def __call__(self, infr, u, v, key, val, msg):
            data = infr.get_nonvisual_edge_data((u, v))
            if data is None:
                assert infr.graph.has_edge(u, v), (
                    'uv=%r, %r does not exist'  % (u, v))
            got = data.get(key)
            if got != val:
                msg1 = 'key=%s %r!=%r, ' % (key, got, val)
                errmsg = ''.join([msg1, msg, '\nedge=', ut.repr2((u, v)), '\n',
                                 infr.repr_edge_data(data)])
                self._errors.append(errmsg)

        def custom_precheck(self, func):
            try:
                func(self.infr1)
            except AssertionError as ex:
                self._errors.append(str(ex))

        def after(self, errors=[]):
            """
            Delays error reporting until after visualization

            prints errors, then shows you the graph, then
            finally if any errors were discovered they are raised
            """

            errors = errors + self._errors
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

    check = Checker(infr1, infr2)
    return infr1, infr2, check


def case_negative_infr():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_negative_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_negative_infr()
    """
    # Initial positive reviews
    ccs = [[1, 2, 3], [9]]
    # Add in initial reviews
    edges = [
        (9, 7, {'decision': NEGTV, 'is_cut': True}),
        (1, 7, {'inferred_state': None}),
        (1, 9, {'inferred_state': None}),
    ]
    # Add in scored, but unreviewed edges
    new_edges = [(3, 9, {'decision': NEGTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    check(infr2, 1, 7, 'inferred_state', None,
          'negative review of an edge should not jump more than one component')

    check(infr2, 1, 9, 'inferred_state', 'diff',
          'negative review of an edge should cut within one jump')

    check.after()


def case_match_infr():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_match_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_match_infr()
    """
    # Initial positive reviews
    ccs = [[2], [7], [9, 10]]
    # Add in initial reviews
    edges = [
        (9, 8, {'decision': NEGTV, 'is_cut': True}),
        (7, 2, {}),
    ]
    # Add in scored, but unreviewed edges
    edges += [
        (2, 8, {'inferred_state': None}),
        (2, 9, {'inferred_state': None}),
    ]
    new_edges = [(2, 10, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    # Checks out of cc inferences
    check(infr2, 2, 9, 'inferred_state', 'same', 'should infer a match')
    check(infr2, 2, 8, 'inferred_state', 'diff', 'should infer a negative')
    check(infr1, 2, 7, 'inferred_state', None, 'discon should have inference')

    check(infr2, 2, 7, 'inferred_state', None, 'discon should have inference')
    check.after()


def case_inconsistent():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_inconsistent --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_inconsistent()
    """
    ccs = [[1, 2], [3, 4, 5]]  # [6, 7]]
    edges = [
        (2, 3, {'decision': NEGTV, 'is_cut': True}),
    ]
    edges += [
        (4, 1, {'inferred_state': None}),
        # (2, 7, {'inferred_state': None}),
    ]
    new_edges = [(1, 5, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    # Make sure the previously inferred edge is no longer inferred
    check(infr1, 4, 1, 'inferred_state', 'diff', 'should initially be an inferred diff')
    check(infr2, 4, 1, 'inferred_state', 'inconsistent_internal', 'should not be inferred after incon')
    check(infr2, 4, 3, 'maybe_error', True, 'need to have a maybe split')
    check.after()


def case_redo_incon():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_redo_incon --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_redo_incon()
    """
    ccs = [[1, 2], [3, 4]]  # [6, 7]]
    edges = [
        (2, 3, {'decision': NEGTV}),
        (1, 4, {'decision': NEGTV}),
    ]
    edges += []
    new_edges = [(2, 3, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    maybe_splits = infr2.get_edge_attrs('maybe_error')
    print('maybe_splits = %r' % (maybe_splits,))
    if not any(maybe_splits.values()):
        ut.cprint('FAILURE', 'red')
        print('At least one edge should be marked as a split')

    check.after()


def case_override_inference():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_override_inference --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_override_inference()
    """
    ccs = [[1, 2, 3, 4, 5]]
    edges = [
        (1, 3, {'inferred_state': 'same'}),
        (1, 4, {'inferred_state': 'same'}),
        # (1, 5, {'inferred_state': 'same'}),
        (1, 2, {'inferred_state': 'same', 'num_reviews': 2}),
        (2, 3, {'inferred_state': 'same', 'num_reviews': 2}),
        (2, 4, {'inferred_state': 'same'}),
        (2, 5, {'inferred_state': 'same'}),
        (3, 4, {'inferred_state': 'same', 'num_reviews': 100}),
        (4, 5, {'inferred_state': 'same', 'num_reviews': .01}),
    ]
    edges += []
    new_edges = [
        (1, 5, {'decision': NEGTV}),
        (5, 2, {'decision': NEGTV}),
    ]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    # Make sure that the inferred edges are no longer inferred when an
    # inconsistent case is introduced
    check(infr2, 1, 4, 'maybe_error', None, 'should not split inferred edge')
    check(infr2, 4, 5, 'maybe_error', True, 'split me')
    check(infr2, 5, 2, 'inferred_state', 'inconsistent_internal', 'inference should be overriden')
    check.after()


def case_undo_match():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_undo_match --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_undo_match()
    """
    ccs = [[1, 2]]
    edges = []
    new_edges = [(1, 2, {'decision': NEGTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    check(infr2, 1, 2, 'inferred_state', 'diff', 'should have cut edge')
    check.after()


def case_undo_negative():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_undo_negative --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_undo_negative()
    """
    ccs = [[1], [2]]
    edges = [
        (1, 2, {'decision': NEGTV}),
    ]
    new_edges = [(1, 2, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 2, 'inferred_state', 'same', 'should have matched edge')
    check.after()


def case_incon_removes_inference():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_incon_removes_inference --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_incon_removes_inference()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (3, 4, {'decision': NEGTV}),
        (1, 5, {'decision': NEGTV}),
        (2, 5, {}),
        (1, 6, {}),
    ]
    new_edges = [(3, 4, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    check(infr1, 2, 5, 'inferred_state', 'diff', 'should be preinferred')
    check(infr2, 2, 5, 'inferred_state', 'inconsistent_internal', 'should be uninferred on incon')
    check.after()


def case_inferable_notcomp1():
    """
    make sure notcomparable edges can be inferred

    CommandLine:
        python -m ibeis.algo.graph.demo case_inferable_notcomp1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_inferable_notcomp1()
    """
    ccs = [[1, 2], [3, 4]]
    edges = [
        (2, 3, {'decision': NEGTV}),
    ]
    new_edges = [(1, 4, {'decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 4, 'inferred_state', 'diff', 'should be inferred')
    check.after()


def case_inferable_update_notcomp():
    """
    make sure inference updates for nocomparable edges

    CommandLine:
        python -m ibeis.algo.graph.demo case_inferable_update_notcomp --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_inferable_update_notcomp()
    """
    ccs = [[1, 2], [3, 4]]
    edges = [
        (2, 3, {'decision': NEGTV}),
        (1, 4, {'decision': INCMP}),
    ]
    new_edges = [(2, 3, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 1, 4, 'inferred_state', 'diff', 'should be inferred diff')
    check(infr2, 1, 4, 'inferred_state', 'same', 'should be inferred same')
    check.after()


def case_notcomp_remove_infr():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_notcomp_remove_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_notcomp_remove_infr()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (1, 4, {'decision': POSTV}),
        # (1, 4, {'decision': INCMP}),
        (2, 5, {'decision': INCMP}),
        (3, 6, {'decision': INCMP}),
    ]
    new_edges = [(1, 4, {'decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 4, 'inferred_state', INCMP, 'can not infer match here!')
    check(infr2, 2, 5, 'inferred_state', INCMP, 'can not infer match here!')
    check(infr2, 3, 6, 'inferred_state', INCMP, 'can not infer match here!')
    check.after()


def case_notcomp_remove_cuts():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_notcomp_remove_cuts --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_notcomp_remove_cuts()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (1, 4, {'decision': NEGTV}),
        # (1, 4, {'decision': INCMP}),
        (2, 5, {'decision': INCMP}),
        (3, 6, {'decision': INCMP}),
    ]
    new_edges = [(1, 4, {'decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 1, 4, 'inferred_state', 'diff', 'should infer diff!')
    check(infr1, 2, 5, 'inferred_state', 'diff', 'should infer diff!')
    check(infr1, 3, 6, 'inferred_state', 'diff', 'should infer diff!')
    check(infr2, 1, 4, 'decision', INCMP, 'can not infer cut here!')
    check(infr2, 2, 5, 'inferred_state', INCMP, 'can not infer cut here!')
    check(infr2, 3, 6, 'inferred_state', INCMP, 'can not infer cut here!')
    check.after()


def case_keep_in_cc_infr_post_negative():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_keep_in_cc_infr_post_negative --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_keep_in_cc_infr_post_negative()
    """
    ccs = [[1, 2, 3], [4]]
    edges = [(1, 3), (1, 4), (2, 4), (3, 4)]
    new_edges = [(4, 2, {'decision': NEGTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 3, 4, 'inferred_state', None, 'should be no inference')
    check(infr1, 1, 3, 'inferred_state', 'same', 'should be inferred')
    check(infr2, 1, 3, 'inferred_state', 'same', 'should remain inferred')
    check(infr2, 3, 4, 'inferred_state', 'diff', 'should become inferred')
    check.after()


def case_keep_in_cc_infr_post_notcomp():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_keep_in_cc_infr_post_notcomp --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_keep_in_cc_infr_post_notcomp()
    """
    ccs = [[1, 2, 3], [4]]
    edges = [(1, 3), (1, 4), (2, 4), (3, 4)]
    new_edges = [(4, 2, {'decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 3, 4, 'inferred_state', None, 'should not be inferred')
    check(infr1, 1, 3, 'inferred_state', 'same', 'should be inferred')
    check(infr2, 1, 3, 'inferred_state', 'same', 'should remain inferred')
    check(infr2, 3, 4, 'inferred_state', None, 'should not become inferred')
    check.after()


def case_out_of_subgraph_modification():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_out_of_subgraph_modification --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_out_of_subgraph_modification()
    """
    # A case where a review between two ccs modifies state outside of
    # the subgraph of ccs
    ccs = [[1, 2], [3, 4], [5, 6]]
    edges = [
        (2, 6), (4, 5, {'decision': NEGTV})
    ]
    new_edges = [(2, 3, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 2, 6, 'inferred_state', None, 'should not be inferred')
    check(infr2, 2, 6, 'inferred_state', 'diff', 'should be inferred')
    check.after()


def case_flag_merge():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_flag_merge --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_flag_merge()
    """
    # A case where a review between two ccs modifies state outside of
    # the subgraph of ccs
    ccs = []
    edges = [
        (1, 2, {'decision': POSTV, 'num_reviews': 2}),
        (4, 1, {'decision': POSTV, 'num_reviews': 1}),
        (2, 4, {'decision': NEGTV, 'num_reviews': 1}),
    ]
    # Ensure that the negative edge comes back as potentially in error
    new_edges = [(1, 4, {'decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    @check.custom_precheck
    def check_pre_state(infr):
        assert infr.nid_to_errors[1] == {(1, 4)}

    check(infr1, 2, 4, 'maybe_error', None, 'match edge should flag first None')
    check(infr1, 1, 4, 'maybe_error', True, 'match edge should flag first True')
    check(infr2, 2, 4, 'maybe_error', True, 'negative edge should flag second True')
    check(infr2, 1, 4, 'maybe_error', None, 'negative edge should flag second None')
    check.after()


def case_all_types():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo case_all_types --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> case_all_types()
    """
    # A case where a review between two ccs modifies state outside of
    # the subgraph of ccs
    ccs = []
    # Define edges within components
    edges = [
        # Inconsistent component
        (11, 12, {'decision': POSTV}),
        (12, 13, {'decision': POSTV}),
        (11, 13, {'decision': NEGTV}),
        (11, 14, {'decision': POSTV}),
        (12, 14, {'decision': POSTV}),
        (13, 14, {}),
        (11, 15, {'decision': POSTV}),
        (12, 15, {'decision': INCMP}),

        # Positive component (with notcomp)
        (21, 22, {'decision': POSTV}),
        (22, 23, {'decision': POSTV}),
        (21, 23, {'decision': INCMP}),
        (21, 24, {'decision': POSTV}),
        (22, 24, {'decision': POSTV}),
        (23, 24, {}),

        # Positive component (with unreview)
        (31, 32, {'decision': POSTV}),
        (32, 33, {'decision': POSTV}),
        (31, 33, {'decision': POSTV}),
        (31, 34, {'decision': POSTV}),
        (32, 34, {'decision': POSTV}),
        (33, 34, {}),

        # Positive component
        (41, 42, {'decision': POSTV}),
        (42, 43, {'decision': POSTV}),
        (41, 43, {'decision': POSTV}),

        # Positive component (extra)
        (51, 52, {'decision': POSTV}),
        (52, 53, {'decision': POSTV}),
        (51, 53, {'decision': POSTV}),

        # Positive component (isolated)
        (61, 62, {'decision': POSTV}),
    ]
    # Define edges between components
    edges += [
        # 1 - 2
        (11, 21, {}),
        (12, 22, {}),
        # 1 - 3
        (11, 31, {}),
        (12, 32, {'decision': NEGTV}),
        (13, 33, {}),
        # 1 - 4
        (11, 41, {}),
        (12, 42, {'decision': INCMP}),
        (13, 43, {}),
        # 1 - 5
        (11, 51, {'decision': INCMP}),
        (12, 52, {'decision': NEGTV}),
        (13, 53, {}),

        # 2 - 3
        (21, 31, {'decision': INCMP}),
        (22, 32, {}),
        # 2 - 4
        (21, 41, {}),
        (22, 42, {}),
        # 2 - 5
        (21, 51, {'decision': INCMP}),
        (22, 52, {'decision': NEGTV}),

        # 3 - 4
        (31, 41, {'decision': NEGTV}),
        (32, 42, {}),
    ]
    # Ensure that the negative edge comes back as potentially in error
    # new_edges = [(2, 5, {'decision': POSTV})]
    new_edges = []
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    errors = []
    for u, v, d in infr2.graph.edges(data=True):
        state = d.get('inferred_state', '')
        if u < 20 or v < 20:
            if state is not None and 'inconsistent' not in state:
                print('u, v, state = %r, %r, %r' % (u, v, state))
                err = AssertionError('all of cc0 should be incon')
                print(err)
                errors.append(err)
        else:
            if state is not None and 'inconsistent' in state:
                print('u, v, state = %r, %r, %r' % (u, v, state))
                err = AssertionError('outside of cc0 should not be incon')
                print(err)
                errors.append(err)
    check(infr1, 13, 14, 'inferred_state', 'inconsistent_internal',
          'notcomp edge should be incon')
    check(infr1, 21, 31, 'inferred_state', INCMP, 'notcomp edge should remain notcomp')
    check(infr1, 22, 32, 'inferred_state', None, 'notcomp edge should transfer knowledge')
    check(infr1, 12, 42, 'inferred_state', 'inconsistent_external',
          'inconsistency should override notcomp')
    # check(infr1, 1, 4, 'maybe_error', True, 'match edge should flag first')
    # check(infr2, 2, 4, 'maybe_error', True, 'negative edge should flag second')
    # check(infr2, 1, 4, 'maybe_error', False, 'negative edge should flag second')
    check.after(errors)


def demodata_infr(**kwargs):
    """
    kwargs = {}

    CommandLine:
        python -m ibeis.algo.graph.demo demodata_infr --show

    Example:
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> from ibeis.algo.graph import demo
        >>> import networkx as nx
        >>> kwargs = dict(num_pccs=6, p_incon=.5, size_std=2)
        >>> infr = demo.demodata_infr(**kwargs)
        >>> pccs = list(infr.positive_components())
        >>> assert len(pccs) == kwargs['num_pccs']
        >>> nonfull_pccs = [cc for cc in pccs if len(cc) > 1 and nx.is_empty(nx.complement(infr.pos_graph.subgraph(cc)))]
        >>> expected_n_incon = len(nonfull_pccs) * kwargs['p_incon']
        >>> n_incon = len(list(infr.inconsistent_components()))
        >>> # TODO can test that we our sample num incon agrees with pop mean
        >>> #sample_mean = n_incon / len(nonfull_pccs)
        >>> #pop_mean = kwargs['p_incon']
        >>> print(infr.status)
        >>> infr.show(pickable=True, groupby='name_label')
        >>> ut.show_if_requested()
    """

    def kwalias(*args):
        params = args[0:-1]
        default = args[-1]
        for key in params:
            if key in kwargs:
                return kwargs[key]
        return default

    rng = np.random.RandomState(0)
    counter = 1
    new_ccs = []
    num_pccs = kwalias('num_pccs', 16)
    size_mean = kwalias('pcc_size_mean', 'pcc_size', 'size', 5)
    size_std = kwalias('pcc_size_std', 'size_std', 0)
    # p_pcc_incon = kwargs.get('p_incon', .1)
    p_pcc_incon = kwargs.get('p_incon', 0)
    p_pcc_incomp = kwargs.get('p_incomp', 0)

    # number of maximum inconsistent edges per pcc
    max_n_incon = kwargs.get('n_incon', 3)

    pcc_iter = list(range(num_pccs))
    pcc_iter = ut.ProgIter(pcc_iter, enabled=num_pccs > 20,
                           label='make pos-demo')
    for i in pcc_iter:
        size = int(randn(size_mean, size_std, rng=rng, a_min=1))
        p = .1
        import networkx as nx
        want_connectivity = rng.choice([1, 2, 3])
        want_connectivity = min(size - 1, want_connectivity)
        # print('want_connectivity = %r' % (want_connectivity,))
        while True:
            import sys
            g = nx.fast_gnp_random_graph(size, p, seed=rng.randint(sys.maxsize))
            conn = nx.edge_connectivity(g)
            if conn == want_connectivity:
                break
            elif conn < want_connectivity:
                p = 2 * p - p ** 2
            elif conn > want_connectivity:
                p = p / 2
        new_nodes = np.array(list(g.nodes()))
        new_edges_ = np.array(list(g.edges()))
        new_edges_ += counter
        new_nodes += counter
        counter = new_nodes.max() + 1
        new_edges = [
            (int(min(u, v)), int(max(u, v)), {'decision': POSTV, 'truth': POSTV})
            for u, v in new_edges_
        ]
        new_g = nx.Graph(new_edges)
        new_g.add_nodes_from(new_nodes)
        assert nx.is_connected(new_g)

        # The probability any edge is inconsistent is `p_incon`
        # This is 1 - P(all edges consistent)
        # which means p(edge is consistent) = (1 - p_incon) / N
        complement_edges = list(nx.complement(new_g).edges())
        if len(complement_edges) > 0:
            # compute probability that any particular edge is inconsistent
            # to achieve probability the PCC is inconsistent
            p_edge_inconn = 1 - (1 - p_pcc_incon) ** (1 / len(complement_edges))
            p_edge_unrev = .1
            p_edge_notcomp = 1 - (1 - p_pcc_incomp) ** (1 / len(complement_edges))
            probs = np.array([p_edge_inconn, p_edge_unrev, p_edge_notcomp])
            # if the total probability is greater than 1 the parameters
            # are invalid, so we renormalize to "fix" it.
            # if probs.sum() > 1:
            #     warnings.warn('probabilities sum to more than 1')
            #     probs = probs / probs.sum()
            pcumsum = probs.cumsum()
            # Determine which mutually exclusive state each complement edge is in
            # print('pcumsum = %r' % (pcumsum,))
            states = np.searchsorted(pcumsum, rng.rand(len(complement_edges)))

            incon_idxs = np.where(states == 0)[0]
            if len(incon_idxs) > max_n_incon:
                print('max_n_incon = %r' % (max_n_incon,))
                chosen = rng.choice(incon_idxs, max_n_incon, replace=False)
                states[np.setdiff1d(incon_idxs, chosen)] = len(probs)

            # print('states = %r' % (states,))
            for (u, v), state in zip(complement_edges, states):
                u, v = (u, v) if u < v else (v, u)
                # Add in candidate edges
                truth = POSTV
                if state == 0:
                    # Add in inconsistent edges
                    decision = NEGTV
                    # TODO: truth could be INCMP or POSTV
                    # new_edges.append((u, v, {'decision': NEGTV}))
                elif state == 1:
                    decision = UNREV
                    # TODO: truth could be INCMP or POSTV
                    # new_edges.append((u, v, {'decision': UNREV}))
                elif state == 2:
                    decision = INCMP
                    truth = INCMP
                    # new_edges.append((u, v, {'decision': INCMP}))
                else:
                    continue
                new_edges.append((u, v, {'decision': decision, 'truth':
                                         truth}))
        new_ccs.append((new_nodes, new_edges))

    import networkx as nx
    pos_g = nx.Graph(ut.flatten(ut.take_column(new_ccs, 1)))
    pos_g.add_nodes_from(ut.flatten(ut.take_column(new_ccs, 0)))
    assert num_pccs == len(list(nx.connected_components(pos_g)))

    neg_edges = []

    print('makingxx pairs')
    if not kwalias('ignore_pair', False):
        print('making pairs')
        p_pair_neg = kwalias('p_pair_neg', .4)
        p_pair_incmp = kwalias('p_pair_incmp', .2)
        p_pair_unrev = kwalias('p_pair_unrev', .2)
        print('p_pair_neg = %r' % (p_pair_neg,))
        # p_pair_neg = 1
        for cc1, cc2 in ut.ProgIter(list(it.combinations(new_ccs, 2)),
                                    label='make neg-demo'):
            nodes1 = cc1[0]
            nodes2 = cc2[0]

            if len(nodes1) == 0 or len(nodes2) == 0:
                continue

            possible_edges = list(it.product(nodes1, nodes2))
            # probability that any edge between these PCCs is negative
            p_edge_neg = 1 - (1 - p_pair_neg) ** (1 / len(possible_edges))
            p_edge_incmp = 1 - (1 - p_pair_incmp) ** (1 / len(possible_edges))
            p_edge_unrev = 1 - (1 - p_pair_unrev) ** (1 / len(possible_edges))
            pcumsum = np.cumsum([p_edge_neg, p_edge_incmp, p_edge_unrev])
            states = np.searchsorted(pcumsum, rng.rand(len(possible_edges)))

            # print('checking pair')
            for (u, v), state in zip(possible_edges, states):
                u, v = (u, v) if u < v else (v, u)
                # Add in candidate edges
                if state == 0:
                    decision == NEGTV
                    truth = NEGTV
                elif state == 1:
                    decision = INCMP
                    # TODO: could be incomp or neg
                    truth = INCMP
                elif state == 2:
                    decision = UNREV
                    # TODO: could be incomp or neg
                    truth = NEGTV
                    # neg_edges.append((u, v, {'decision': UNREV}))
                else:
                    continue
                neg_edges.append((u, v, {'decision': decision, 'truth': truth}))

    edges = ut.flatten(ut.take_column(new_ccs, 1)) + neg_edges
    edges = [(int(u), int(v), d) for u, v, d in edges]
    nodes = ut.flatten(ut.take_column(new_ccs, 0))
    infr = make_demo_infr([], edges, nodes=nodes,
                          infer=kwargs.get('infer', True))

    # fontname = 'Ubuntu'
    fontsize = 12
    fontname = 'sans'
    splines = 'spline'
    # splines = 'ortho'
    # splines = 'line'
    infr.set_node_attrs('shape', 'circle')
    infr.graph.graph['ignore_labels'] = True
    infr.graph.graph['dark_background'] = False
    infr.graph.graph['fontname'] = fontname
    infr.graph.graph['fontsize'] = fontsize
    infr.graph.graph['splines'] = splines
    infr.set_node_attrs('width', 29)
    infr.set_node_attrs('height', 29)
    infr.set_node_attrs('fontsize', fontsize)
    infr.set_node_attrs('fontname', fontname)
    infr.set_node_attrs('fixed_size', True)

    # Set synthetic ground-truth attributes for testing
    # infr.apply_edge_truth()
    infr.edge_truth = infr.get_edge_attrs('truth')
    # Make synthetic matcher
    infr.dummy_matcher = DummyMatcher(infr)

    return infr


def randn(mean=0, std=1, shape=[], a_max=None, a_min=None, rng=None):
    a = (rng.randn(*shape) * std) + mean
    if a_max is not None or a_min is not None:
        a = np.clip(a, a_min, a_max)
    return a


class DummyMatcher(object):
    """
    generates dummy scores between edges (not necesarilly in the graph)
    """
    def __init__(matcher, infr):
        matcher.infr = infr
        matcher.cache = {}
        matcher.rng = np.random.RandomState(0)
        matcher.dummy_params = {
            NEGTV: {'mean': .2, 'std': .25},
            POSTV: {'mean': .8, 'std': .2},
            INCMP: {'mean': .2, 'std': .4},
        }

    def predict(matcher, edges):
        edges = list(edges)
        infr = matcher.infr
        is_miss = np.array([e not in matcher.cache for e in edges])
        # is_hit = ~is_miss
        if np.any(is_miss):
            miss_edges = ut.compress(edges, is_miss)
            def guess_truth(edge):
                nid1 = infr.graph.node[edge[0]]['orig_name_label']
                nid2 = infr.graph.node[edge[1]]['orig_name_label']
                return POSTV if nid1 == nid2 else NEGTV
            miss_truths = [
                infr.edge_truth[edge] if edge in infr.edge_truth else
                guess_truth(edge)
                for edge in miss_edges
            ]
            grouped_edges = ut.group_items(miss_edges, miss_truths)
            for key, group in grouped_edges.items():
                probs = randn(shape=[len(group)], rng=matcher.rng, a_max=1, a_min=0,
                              **matcher.dummy_params[key])
                for edge, prob in zip(group, probs):
                    matcher.cache[edge] = prob

        return ut.take(matcher.cache, edges)

    # print('[demo] apply dummy scores')
    # rng = ut.ensure_rng(rng)
    # # edges = list(infr.graph.edges())
    # grouped_edges = ut.group_pairs(infr.edge_truth.items())
    # for key, group in grouped_edges.items():
    #     probs = randn(shape=[len(group)], rng=rng, a_max=1, a_min=0,
    #                   **dummy_params[key])
    #     infr.set_edge_attrs('prob_match', ut.dzip(group, probs))


# TODO: inconsistent out of subgraph modification
# should an inconsistent component (a component all of the same name
# but with at least one non-match edge) still be allowed to have infered
# reviews outside the component? ...
# I think yes because in the case the component is split the inferred
# reviews should go away, and in the case of the component is merged
# then they are fine.

if __name__ == '__main__':
    r"""
    CommandLine:
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7 --graph
        python -m ibeis.algo.graph.demo demo2
        python -m ibeis.algo.graph.demo
        python -m ibeis.algo.graph.demo --allexamples
        python -m ibeis.algo.graph.demo --allexamples --show
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
