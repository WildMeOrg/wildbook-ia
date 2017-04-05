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


@profile
def demo2():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo demo2
    """
    import plottool as pt

    # ---- Synthetic data params

    # Create dummy data
    # nids = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
    # annots_per_name = [4, 3, 2, 1]
    # annots_per_name = [9, 9, 9, 9, 9, 9, 9, 9, 9]
    # annots_per_name = [8, 8, 8, 8, 8, 8, 8, 8]
    # annots_per_name = [8, 8, 8, 8, 8]
    annots_per_name = [8, 8, 8, 8]
    # 8, 8]
    # annots_per_name = [1, 2, 3, 4, 4, 2, 5]
    # annots_per_name = (np.random.rand(9) * 10).astype(np.int32) + 1

    queue_params = {
        'pos_redundancy': None,
        'neg_redundancy': None,
        # 'pos_redundancy': 1,
        # 'neg_redundancy': 2,
    }
    # oracle_accuracy = 1.0
    oracle_accuracy = .8
    # oracle_accuracy = .9
    # oracle_accuracy = 1.0

    b = 10

    round2_params = {
        # 'pos_redundancy': None,
        # 'neg_redundancy': None,
        # 'pos_redundancy': 6,
        # 'neg_redundancy': 7,
        'pos_redundancy': 2,
        'neg_redundancy': 3,
    }
    # round2_params = None

    # --- draw params

    VISUALIZE = False
    VISUALIZE = True
    SHOW_NEG = False
    SHOW_NEG = True

    SHOW_GT = True
    # QUIT_OR_EMEBED = 'embed'
    QUIT_OR_EMEBED = 'quit'
    TARGET_REVIEW = ut.get_argval('--target', type_=int, default=None)
    PRESHOW = True
    # PRESHOW = False
    # SHOW_GT = False
    # QUIT_OR_EMEBED = 'quit'
    # TARGET_REVIEW = 14

    fontsize = 12
    # fontname = 'Ubuntu'
    fontname = 'sans'
    splines = 'spline'
    # splines = 'ortho'
    # splines = 'line'

    # ------------------

    rng = np.random.RandomState(42)

    infr = make_dummy_infr(annots_per_name)
    infr.set_node_attrs('shape', 'circle')
    infr.graph.graph['ignore_labels'] = True
    infr.graph.graph['dark_background'] = True
    infr.set_node_attrs('width', 29)
    infr.set_node_attrs('height', 29)
    infr.set_node_attrs('fontsize', fontsize)
    infr.set_node_attrs('fontname', fontname)
    infr.set_node_attrs('fixed_size', True)

    # Assign random viewpoints
    apply_dummy_viewpoints(infr)
    infr.ensure_cliques()

    dpath = ut.ensuredir(ut.truepath('~/Desktop/demo'))
    ut.remove_files_in_dir(dpath)

    def show_graph(infr, title, final=False):
        if not VISUALIZE:
            return
        if len(infr.graph) < 35:
            pass

        showkw = dict(fontsize=fontsize, fontname=fontname,
                      show_reviewed_cuts=SHOW_NEG,
                      show_inferred_same=False,
                      show_inferred_diff=False,
                      show_labels=False,
                      show_recent_review=not final,
                      splines=splines,
                      reposition=False,
                      with_colorbar=True)
        # showkw = dict(fontsize=6, show_cuts=True, with_colorbar=True)
        infr_ = infr
        verbose = infr_.verbose
        infr_.verbose = 0
        infr = infr_.copy()
        infr_.verbose = verbose
        infr.show_graph(**showkw)
        pt.set_title(title)
        pt.adjust_subplots(top=.95, left=0, right=1, bottom=.01)
        pt.gca().set_aspect('equal')
        pt.gcf().canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr))
        dpath = ut.ensuredir(ut.truepath('~/Desktop/demo'))
        pt.save_figure(dpath=dpath, dpi=128)

    if VISUALIZE:
        infr.update_visual_attrs(splines=splines, groupby='name_label')
        infr.set_node_attrs('pin', 'true')
        print(ut.repr4(infr.graph.node[1]))

    if SHOW_GT:
        # Pin Nodes into the target groundtruth position
        show_graph(infr, 'target-gt')

    # Dummy scoring
    apply_random_negative_edges(infr, rng)
    # infr.ensure_full()
    apply_dummy_scores(infr, rng)
    infr.break_graph(b)
    infr.remove_name_labels()

    if VISUALIZE:
        infr.update_visual_attrs(splines=splines)

    if PRESHOW or TARGET_REVIEW is None or TARGET_REVIEW == 0:
        show_graph(infr, 'pre-reveiw')

    def oracle_decision(infr, n1, n2):
        """ The perfect reviewer """
        truth = get_edge_truth(infr, n1, n2)

        # Baseline accuracy
        oracle_accuracy_ = oracle_accuracy

        # Increase oracle accuracy for edges that it has seen before
        num_reviews = infr.graph.edge[n1][n2].get('num_reviews', 1)
        oracle_accuracy_ = 1 - ((1 - oracle_accuracy_) ** num_reviews)

        # Decrease oracle accuracy for harder cases
        # if truth == 1:
        #     easiness = infr.graph.edge[n1][n2]['normscore']
        # elif truth == 1:
        #     easiness = infr.graph.edge[n1][n2]['normscore']
        # else:
        #     easiness = oracle_accuracy_
        # oracle_accuracy_ = (oracle_accuracy_) * 1.0 + (1 - oracle_accuracy_) * easiness

        if rng.rand() > oracle_accuracy_:
            print('oops')
            # truth = rng.choice(list({0, 1, 2} - {truth}))
            observed = rng.choice(list({0, 1} - {truth}))
        else:
            observed = truth
        from ibeis import constants as const
        state = const.REVIEW.INT_TO_CODE[observed]
        tags = []
        return state, tags

    _iter = infr.generate_reviews(randomness=.1, rng=rng, **queue_params)
    _iter2 = enumerate(_iter)
    prog = ut.ProgIter(_iter2, label='demo2', bs=False, adjust=False)
    for count, (aid1, aid2) in prog:
        msg = 'review #%d' % (count)
        print('\n----------')
        print(msg)
        print('remaining_reviews = %r' % (infr.remaining_reviews()),)

        # Make the next review decision
        state, tags = oracle_decision(infr, aid1, aid2)

        if count == TARGET_REVIEW:
            infr.EMBEDME = QUIT_OR_EMEBED == 'embed'

        infr.add_feedback((aid1, aid2), decision=state, tags=tags)

        # Show the result
        if PRESHOW or TARGET_REVIEW is None or count >= TARGET_REVIEW - 1:
            show_graph(infr, msg)

        if count == TARGET_REVIEW:
            break

    show_graph(infr, 'post-review', final=True)

    # ROUND 2 FIGHT
    if TARGET_REVIEW is None and round2_params is not None:
        # HACK TO GET NEW THINGS IN QUEUE
        infr.queue_params = round2_params
        infr.apply_review_inference()

        _iter = infr.generate_reviews(randomness=.1, rng=rng, **infr.queue_params)
        _iter2 = enumerate(_iter)
        prog = ut.ProgIter(_iter2, label='round2', bs=False, adjust=False)
        for count, (aid1, aid2) in prog:
            msg = 'reviewII #%d' % (count)
            print('\n----------')
            print(msg)
            print('remaining_reviews = %r' % (infr.remaining_reviews()),)

            # Make the next review decision
            state, tags = oracle_decision(infr, aid1, aid2)

            if count == TARGET_REVIEW:
                infr.EMBEDME = QUIT_OR_EMEBED == 'embed'

            infr.add_feedback((aid1, aid2), decision=state, tags=tags)

            # Show the result
            if PRESHOW or TARGET_REVIEW is None or count >= TARGET_REVIEW - 1:
                show_graph(infr, msg)

            if count == TARGET_REVIEW:
                break

        show_graph(infr, 'post-re-review', final=True)

    if not getattr(infr, 'EMBEDME', False):
        if ut.get_computer_name().lower() in ['hyrule', 'ooo']:
            pt.all_figures_tile(monitor_num=0, percent_w=.5)
        else:
            pt.all_figures_tile()
        ut.show_if_requested()


def randn_clip(rng, mu, sigma, a_max, a_min):
    a = rng.randn() * sigma + mu
    a = np.clip(a, a_max, a_min)
    return a


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
        >>> apply_dummy_scores(infr, rng)
        >>> edges = list(infr.graph.edges())
        >>> truths = np.array([get_edge_truth(infr, n1, n2) for n1, n2 in edges])
        >>> edge_to_normscore = infr.get_edge_attrs('normscore')
        >>> scores = np.array(list(ut.take(edge_to_normscore, edges)))
        >>> scorenorm = vt.ScoreNormalizer()
        >>> scorenorm.fit(scores, truths)
        >>> ut.quit_if_noshow()
        >>> scorenorm.visualize()
        >>> ut.show_if_requested()
    """
    print('[demo] apply dummy scores')
    rng = ut.ensure_rng(rng)
    dummy_params = {
        0: {'mu': .2, 'sigma': .5},
        1: {'mu': .8, 'sigma': .4},
        2: {'mu': .2, 'sigma': .8},

        # 0: {'mu': .2, 'sigma': .2},
        # 1: {'mu': .8, 'sigma': .2},
        # 2: {'mu': .2, 'sigma': .4},

        # 0: {'mu': .2, 'sigma': .02},
        # 1: {'mu': .8, 'sigma': .02},
        # 2: {'mu': .2, 'sigma': .04},
    }
    edges = list(infr.graph.edges())
    truths = [get_edge_truth(infr, n1, n2) for n1, n2 in ut.ProgIter(edges)]
    normscores = [randn_clip(rng, a_max=0, a_min=1, **dummy_params[truth])
                  for truth in ut.ProgIter(truths, label='clipping')]
    infr.set_edge_attrs('normscore', ut.dzip(edges, normscores))
    ut.nx_delete_edge_attr(infr.graph, '_dummy_edge')


def apply_random_negative_edges(infr, rng=None):
    thresh = .1
    rng = ut.ensure_rng(rng)
    nid_to_aids = ut.group_pairs([
        (n, d['name_label']) for n, d in infr.graph.nodes(data=True)])
    nid_pairs = list(ut.combinations(nid_to_aids.keys(), 2))
    random_edges = []
    total = 0
    for nid1, nid2 in nid_pairs:
        aids1 = nid_to_aids[nid1]
        aids2 = nid_to_aids[nid2]
        aid_pairs = list(ut.product(aids1, aids2))
        flags = rng.rand(len(aid_pairs)) < thresh
        total += len(aid_pairs)
        chosen = ut.compress(aid_pairs, flags)
        random_edges.extend(chosen)
    infr.graph.add_edges_from(random_edges)
    infr.set_edge_attrs('_dummy_edge', ut.dzip(random_edges, [True]))


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


@profile
def demo_ibeis_graph_iden():
    """
    CommandLine:
        python -m ibeis.algo.graph.demo demo_ibeis_graph_iden --show
    """
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
    infr = ibeis.AnnotInference(ibs, aids, autoinit=True, verbose=1)
    # Pin nodes in groundtruth positions
    infr.ensure_cliques()
    # infr.initialize_visual_node_attrs()
    import plottool as pt
    showkw = dict(fontsize=8, with_colorbar=True)
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
        infr.apply_feedback_edges()

        # infr.relabel_using_reviews()
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
            state = ibs.const.REVIEW.INT_TO_CODE[truth]
            tags = []
            # TODO:
            # if view1 != view1: infr.add_feedback((aid1, aid2), INCMP)
            return state, tags

        # for count in ut.ProgIter(range(1, n_reviews + 1), 'review'):
        for count, (aid1, aid2) in enumerate(infr.generate_reviews()):
            if oracle_mode:
                state, tags = oracle_decision(aid1, aid2)
                infr.add_feedback((aid1, aid2), decision=state, tags=tags)
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


def repr_edge_data(infr, all_edge_data):
    visual_edge_data = {k: v for k, v in all_edge_data.items()
                        if k in infr.visual_edge_attrs}
    edge_data = ut.delete_dict_keys(all_edge_data.copy(), infr.visual_edge_attrs)
    lines = [
        ('visual_edge_data: ' + ut.repr2(visual_edge_data, nl=1)),
        ('edge_data: ' + ut.repr2(edge_data, nl=1)),
    ]
    return '\n'.join(lines)


def on_pick(event, infr=None):
    import plottool as pt
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
            node = plotdat['node']
            ut.cprint('node: ' + ut.repr2(plotdat['node']), 'blue')
            node_label = infr.pos_graph.node_label(node)
            print('(pcc) node_label = %r' % (node_label,))
            print('artist = %r' % (artist,))
        elif 'edge' in plotdat:
            all_edge_data = ut.sort_dict(plotdat['edge_data'].copy())
            print(repr_edge_data(infr, all_edge_data))
            ut.cprint('edge: ' + ut.repr2(plotdat['edge']), 'blue')
            print('artist = %r' % (artist,))
        else:
            print('???: ' + ut.repr2(plotdat))
    print(ut.get_timestamp())


def make_demo_infr(ccs, edges, nodes=[], infer=True):
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
            d['normscore'] = rng.rand()

    G.add_edges_from(edges)
    infr = ibeis.AnnotInference.from_netx(G, infer=infer)
    infr.verbose = 3

    # infr.relabel_using_reviews()
    # infr.apply_review_inference()
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
                  show_labels=True)
        pt.set_title('pre-review')
        pt.gca().set_aspect('equal')
        infr.set_node_attrs('pin', 'true')
        fig1 = pt.gcf()
        fig1.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr))

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
        fig2 = pt.gcf()
        if fig2 is not fig1:
            fig2.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr2))

    class Checker(object):
        """
        Asserts pre and post test properties of the graph
        """
        def __init__(self, infr1, infr2):
            self._errors = []
            self.infr1 = infr1
            self.infr2 = infr2

        def __call__(self, infr, u, v, key, val, msg):
            data = infr.get_edge_data(u, v)
            if data is None:
                assert infr.graph.has_edge(u, v), (
                    'uv=%r, %r does not exist'  % (u, v))
            got = data.get(key)
            if got != val:
                msg1 = 'key=%s %r!=%r, ' % (key, got, val)
                errmsg = ''.join([msg1, msg, '\nedge=', ut.repr2((u, v)), '\n',
                                 repr_edge_data(infr, data)])
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
        assert infr.error_edges == {(1, 4)}

    check(infr1, 2, 4, 'maybe_error', None, 'match edge should flag first None')
    check(infr1, 1, 4, 'maybe_error', True, 'match edge should flag first True')
    check(infr2, 2, 4, 'maybe_error', True, 'negative edge should flag second True')
    check(infr2, 1, 4, 'maybe_error', None, 'negative edge should flag second None')
    check.after()


def demodata_infr(**kwargs):
    """
    kwargs = {}

    Example:
        >>> from ibeis.algo.graph.demo import *  # NOQA
        >>> from ibeis.algo.graph import demo
        >>> import networkx as nx
        >>> kwargs = dict(num_pccs=50, p_incon=.1, size_std=4)
        >>> infr = demo.demodata_infr(**kwargs)
        >>> pccs = list(infr.positive_components())
        >>> assert len(pccs) == kwargs['num_pccs']
        >>> nonfull_pccs = [cc for cc in pccs if len(cc) > 1 and nx.is_empty(nx.complement(infr.pos_graph.subgraph(cc)))]
        >>> expected_n_incon = len(nonfull_pccs) * kwargs['p_incon']
        >>> n_incon = len(list(infr.inconsistent_components()))
        >>> # TODO can test that we our sample num incon agrees with pop mean
        >>> sample_mean = n_incon / len(nonfull_pccs)
        >>> pop_mean = kwargs['p_incon']
    """

    def rand_normal(mean=0, std=1, shape=[]):
        return (rng.randn(*shape) * std) + mean
    def kwalias(*args):
        params = args[0:-1]
        default = args[-1]
        for key in params:
            if key in kwargs:
                return kwargs[key]
        return default

    import itertools as it
    rng = np.random.RandomState(0)
    counter = 0
    new_ccs = []
    num_pccs = kwalias('num_pccs', 16)
    size_mean = kwalias('pcc_size_mean', 'pcc_size', 'size', 5)
    size_std = kwalias('pcc_size_std', 'size_std', 0)

    for i in ut.ProgIter(list(range(num_pccs)), label='make pos-demo'):
        size = max(1, int(rand_normal(size_mean, size_std)))

        p = .1
        import networkx as nx
        want_connectivity = rng.choice([1, 2, 3])
        want_connectivity = min(size - 1, want_connectivity)
        # print('want_connectivity = %r' % (want_connectivity,))
        while True:
            g = nx.fast_gnp_random_graph(size, p)
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
            (int(min(u, v)), int(max(u, v)), {'decision': POSTV})
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
            p_pcc_incon = kwargs.get('p_incon', .1)
            # compute probability that any particular edge is inconsistent
            # to achieve probability the PCC is inconsistent
            p_edge_inconn = 1 - (1 - p_pcc_incon) ** (1 / len(complement_edges))
            p_edge_unrev = .1
            p_edge_notcomp = .05
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
            # print('states = %r' % (states,))
            for (u, v), state in zip(complement_edges, states):
                u, v = (u, v) if u < v else (v, u)
                # Add in candidate edges
                if state == 0:
                    # Add in inconsistent edges
                    # print('made inconsistent cc')
                    new_edges.append((u, v, {'decision': NEGTV}))
                elif state == 1:
                    new_edges.append((u, v, {'decision': UNREV}))
                elif state == 2:
                    # Add in noncomparable edges
                    new_edges.append((u, v, {'decision': INCMP}))
        new_ccs.append((new_nodes, new_edges))

    import networkx as nx
    pos_g = nx.Graph(ut.flatten(ut.take_column(new_ccs, 1)))
    pos_g.add_nodes_from(ut.flatten(ut.take_column(new_ccs, 0)))
    assert num_pccs == len(list(nx.connected_components(pos_g)))

    neg_edges = []

    for cc1, cc2 in ut.ProgIter(list(it.combinations(new_ccs, 2)),
                                label='make neg-demo'):
        nodes1 = cc1[0]
        nodes2 = cc2[0]
        # print('checking pair')
        for u, v in it.product(nodes1, nodes2):
            u, v = (u, v) if u < v else (v, u)
            p = .01
            # Add in candidate edges
            if rng.rand() < p:
                neg_edges.append((u, v, {'decision': UNREV}))
            # Add in noncomparable edges
            if rng.rand() < p / 2:
                neg_edges.append((u, v, {'decision': INCMP}))
            # Add in negative edges
            if rng.rand() < p / 4:
                # print('made negative cc pair')
                neg_edges.append((u, v, {'decision': NEGTV}))

    edges = ut.flatten(ut.take_column(new_ccs, 1)) + neg_edges
    edges = [(int(u), int(v), d) for u, v, d in edges]
    nodes = ut.flatten(ut.take_column(new_ccs, 0))
    infr = make_demo_infr([], edges, nodes=nodes, infer=kwargs.get('infer', True))
    return infr


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
