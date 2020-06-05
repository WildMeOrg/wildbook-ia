# -*- coding: utf-8 -*-
"""
TODO: move to wbia/scripts
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def draw_em_graph(P, Pn, PL, gam, num_labels):
    """
    python -m wbia.algo.hots.testem test_em --show --no-cnn
    """
    num_labels = PL.shape[1]
    name_nodes = ['N%d' % x for x in list(range(1, num_labels + 1))]
    annot_nodes = ['X%d' % x for x in list(range(1, len(Pn) + 1))]

    nodes = name_nodes + annot_nodes

    PL2 = gam[:, num_labels:].T
    PL2 += 0.01
    PL2 = PL2 / PL2.sum(axis=1)[:, None]
    # PL2 = PL2 / np.linalg.norm(PL2, axis=0)
    zero_part = np.zeros((num_labels, len(Pn) + num_labels))
    prob_part = np.hstack([PL2, Pn])
    print(ut.hz_str(' PL2 = ', ut.repr2(PL2, precision=2)))
    # Redo p with posteriors
    if ut.get_argflag('--postem'):
        P = np.vstack([zero_part, prob_part])

    weight_matrix = P  # NOQA
    graph = ut.nx_from_matrix(P, nodes=nodes)
    graph = graph.to_directed()
    # delete graph
    dup_edges = []
    seen_ = set([])
    for u, v in graph.edges():
        if u < v:
            u, v = v, u
        if (u, v) not in seen_:
            seen_.add((u, v))
        else:
            dup_edges.append((u, v))
    graph.remove_edges_from(dup_edges)
    import wbia.plottool as pt
    import networkx as nx

    if len(name_nodes) == 3 and len(annot_nodes) == 4:
        graph.nodes[annot_nodes[0]]['pos'] = (20.0, 200.0)
        graph.nodes[annot_nodes[1]]['pos'] = (220.0, 200.0)
        graph.nodes[annot_nodes[2]]['pos'] = (20.0, 100.0)
        graph.nodes[annot_nodes[3]]['pos'] = (220.0, 100.0)
        graph.nodes[name_nodes[0]]['pos'] = (10.0, 300.0)
        graph.nodes[name_nodes[1]]['pos'] = (120.0, 300.0)
        graph.nodes[name_nodes[2]]['pos'] = (230.0, 300.0)
        nx.set_node_attributes(graph, name='pin', values='true')

        print('annot_nodes = %r' % (annot_nodes,))
        print('name_nodes = %r' % (name_nodes,))

        for u in annot_nodes:
            for v in name_nodes:
                if graph.has_edge(u, v):
                    print('1) u, v = %r' % ((u, v),))
                    graph.edge[u][v]['taillabel'] = graph.edge[u][v]['label']
                    graph.edge[u][v]['color'] = pt.ORANGE
                    graph.edge[u][v]['labelcolor'] = pt.BLUE
                    del graph.edge[u][v]['label']
                elif graph.has_edge(v, u):
                    print('2) u, v = %r' % ((u, v),))
                    graph.edge[v][u]['headlabel'] = graph.edge[v][u]['label']
                    graph.edge[v][u]['color'] = pt.ORANGE
                    graph.edge[v][u]['labelcolor'] = pt.BLUE
                    del graph.edge[v][u]['label']
                else:
                    print((u, v))
                    print('!!')

    # import itertools
    # name_const_edges = [(u, v, {'style': 'invis'}) for u, v in itertools.combinations(name_nodes, 2)]
    # graph.add_edges_from(name_const_edges)
    # nx.set_edge_attributes(graph, name='constraint', values={edge: False for edge in graph.edges() if edge[0] == 'b' or edge[1] == 'b'})
    # nx.set_edge_attributes(graph, name='constraint', values={edge: False for edge in graph.edges() if edge[0] in annot_nodes and edge[1] in annot_nodes})
    # nx.set_edge_attributes(graph, name='constraint', values={edge: True for edge in graph.edges() if edge[0] in name_nodes or edge[1] in name_nodes})
    # nx.set_edge_attributes(graph, name='constraint', values={edge: True for edge in graph.edges() if (edge[0] in ['a', 'b'] and edge[1] in ['a', 'b']) and edge[0] in annot_nodes and edge[1] in annot_nodes})
    # nx.set_edge_attributes(graph, name='constraint', values={edge: True for edge in graph.edges() if (edge[0] in ['c'] or edge[1] in ['c']) and edge[0] in annot_nodes and edge[1] in annot_nodes})
    # nx.set_edge_attributes(graph, name='constraint', values={edge: True for edge in graph.edges() if (edge[0] in ['a'] or edge[1] in ['a']) and edge[0] in annot_nodes and edge[1] in annot_nodes})
    # nx.set_edge_attributes(graph, name='constraint', values={edge: True for edge in graph.edges() if (edge[0] in ['b'] or edge[1] in ['b']) and edge[0] in annot_nodes and edge[1] in annot_nodes})
    # graph.add_edges_from([('root', n) for n in nodes])
    # {node: 'names' for node in name_nodes})
    nx.set_node_attributes(
        graph, name='color', values={node: pt.RED for node in name_nodes}
    )
    # nx.set_node_attributes(graph, name='width', values={node: 20 for node in nodes})
    # nx.set_node_attributes(graph, name='height', values={node: 20 for node in nodes})
    # nx.set_node_attributes(graph, name='group', values={node: 'names' for node in name_nodes})
    # nx.set_node_attributes(graph, name='group', values={node: 'annots' for node in annot_nodes})
    nx.set_node_attributes(
        graph, name='groupid', values={node: 'names' for node in name_nodes}
    )
    nx.set_node_attributes(
        graph, name='groupid', values={node: 'annots' for node in annot_nodes}
    )
    graph.graph['clusterrank'] = 'local'
    # graph.graph['groupattrs'] = {
    #     'names': {'rankdir': 'LR', 'rank': 'source'},
    #     'annots': {'rankdir': 'TB', 'rank': 'source'},
    # }
    ut.nx_delete_edge_attr(graph, 'weight')
    # pt.show_nx(graph, fontsize=10, layoutkw={'splines': 'spline', 'prog': 'dot', 'sep': 2.0}, verbose=1)
    layoutkw = {
        # 'rankdir': 'LR',
        'splines': 'spline',
        # 'splines': 'ortho',
        # 'splines': 'curved',
        # 'compound': 'True',
        # 'prog': 'dot',
        'prog': 'neato',
        # 'packMode': 'clust',
        # 'sep': 4,
        # 'nodesep': 1,
        # 'ranksep': 1,
    }
    # pt.show_nx(graph, fontsize=12, layoutkw=layoutkw, verbose=0, as_directed=False)
    pt.show_nx(
        graph,
        fontsize=6,
        fontname='Ubuntu',
        layoutkw=layoutkw,
        verbose=0,
        as_directed=False,
    )
    pt.interactions.zoom_factory()


def random_test_annot(num_names=5, rng=np.random):
    """
    Create a single test annotation with random properties

    Args:
        num_names (int): (default = 5)
        rng (module):  random number generator (default = numpy.random)

    CommandLine:
        python -m wbia.algo.hots.testem random_test_annot --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.testem import *  # NOQA
        >>> num_names = 5
        >>> rng = np.random.RandomState(0)
        >>> result = random_test_annot(num_names, rng)
        >>> print(result)
        {u'qual': 1, u'yaw': 0.0, u'nfeats': 1529, u'name': 0, u'view': u'R'}
    """
    from wbia import constants as const

    # num_names = 10
    valid_names = list(range(num_names))
    valid_views = list(const.YAWALIAS.values())
    # valid_views.remove('
    valid_quals = list(const.QUALITY_INT_TO_TEXT.keys())
    ut.delete_list_items(valid_quals, [-1, 0, None])

    def sampleone(list_):
        return ut.random_sample(list_, 1, rng=rng)[0]

    view_to_ori = ut.map_dict_keys(
        lambda x: const.YAWALIAS[x], const.VIEWTEXT_TO_YAW_RADIANS
    )
    case = {
        'nfeats': np.clip(rng.normal(1000, 300, size=1)[0], 0, np.inf).astype(np.int),
        'name': sampleone(valid_names),
        'view': sampleone(valid_views),
        'qual': sampleone(valid_quals),
    }
    case['yaw'] = view_to_ori[case['view']]
    return case


def random_case_set():
    r"""
    Returns:
        tuple: (labels, pairwise_feats)

    CommandLine:
        python -m wbia.algo.hots.testem random_case_set --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.testem import *  # NOQA
        >>> (labels, pairwise_feats) = random_case_set()
        >>> result = ('(labels, pairwise_feats) = %s' % (ut.repr2((labels, pairwise_feats)),))
        >>> print(result)
    """
    rng = np.random.RandomState(0)
    case_params = dict(num_names=5, rng=rng)
    num_annots = 600
    test_cases = [
        random_test_annot(**case_params) for _ in ut.ProgIter(range(num_annots), bs=1)
    ]
    pairxs = list(ut.product_nonsame(range(num_annots), range(num_annots)))
    import utool

    utool.embed()

    test_pairs = list(ut.unflat_take(test_cases, pairxs))
    cases1 = ut.instancelist(ut.take_column(test_pairs, 0), check=False)
    cases2 = ut.instancelist(ut.take_column(test_pairs, 1), check=False)
    # FIXME
    labels = labels1 = make_test_pairwise_labels2(cases1, cases2)  # NOQA

    # labels = np.array([make_test_pairwise_labels(case1, case2)
    #                   for case1, case2 in ut.ProgIter(test_pairs, bs=1)])
    pairwise_feats_ = [
        make_test_pairwise_fetaures(case1, case2, label, rng)
        for label, (case1, case2) in ut.ProgIter(list(zip(labels, test_pairs)), bs=1)
    ]
    pairwise_feats = np.vstack(pairwise_feats_)
    print(ut.dict_hist(labels))
    return labels, pairwise_feats


def try_rf_classifier():
    # TODO: Evaluate TPOT
    # http://www.randalolson.com/2016/05/08/tpot-a-python-tool-for-automating-data-science/
    # https://www.reddit.com/r/MachineLearning/comments/4ij8dw/tpot_a_python_tool_for_automating_machine_learning/
    # http://keras.io/ --- unifies tensorflow / theano
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss

    # http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html
    pairwise_feats, labels = random_case_set()
    X = pairwise_feats
    y = labels
    X_train, y_train = X[:600], y[:600]
    X_valid, y_valid = X[600:800], y[600:800]
    X_train_valid, y_train_valid = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    # Train uncalibrated random forest classifier on whole train and validation
    # data and evaluate on test data
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X_train_valid, y_train_valid)
    clf_probs = clf.predict_proba(X_test)
    score = log_loss(y_test, clf_probs)
    print('score = %r' % (score,))

    # Train random forest classifier, calibrate on validation data and evaluate
    # on test data
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X_train, y_train)
    clf_probs = clf.predict_proba(X_test)
    sig_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    sig_clf.fit(X_valid, y_valid)
    sig_clf_probs = sig_clf.predict_proba(X_test)
    sig_score = log_loss(y_test, sig_clf_probs)
    print('sig_score = %r' % (sig_score,))


def make_test_pairwise_fetaures(case1, case2, label, rng):
    import vtool as vt

    mu_fm = 50 if label == 1 else 10
    sigma_fm = 10 if label == 1 else 20
    mu_fs = 0.2 if label == 1 else 0.4
    sigma_fs = 0.1 if label == 1 else 0.1
    num_top = 4
    max_feats = min(case1['nfeats'], case2['nfeats'])
    num_matches = (
        np.clip(rng.normal(mu_fm, sigma_fm, size=1)[0], num_top + 1, max_feats).astype(
            np.int
        ),
    )
    perb = np.abs(rng.normal(0.001, 0.001, size=num_matches))
    sift_dists = np.clip(rng.normal(mu_fs, sigma_fs, size=num_matches), 0, 1) + perb
    sortx = np.argsort(sift_dists)
    local_feat_simvecs = np.vstack([sift_dists]).T[sortx]
    local_simvec = local_feat_simvecs[0:num_top]
    yaw1 = case1['yaw']
    yaw2 = case2['yaw']
    global_simvec = np.array(
        [
            case1['qual'],
            case2['qual'],
            yaw1 / vt.TAU,
            yaw2 / vt.TAU,
            vt.ori_distance(yaw1, yaw2) / np.pi,
            np.abs(case1['qual'] - case1['qual']),
        ]
    )
    simvec = np.hstack([global_simvec, local_simvec.ravel()])
    return simvec


def make_test_pairwise_labels(case1, case2):
    import vtool as vt

    is_same = case1['name'] == case2['name']
    yaw1 = case1['yaw']
    yaw2 = case2['yaw']
    yaw_dist = vt.ori_distance(yaw1, yaw2) / np.pi
    if case1['qual'] < 2 or case2['qual'] < 2:
        # Bad quality means not comparable
        is_comp = False
    else:
        if case1['qual'] > 3 or case2['qual'] > 3:
            # Better quality, better chance of being comparable
            is_comp = yaw_dist <= (1 / 4)
        else:
            is_comp = yaw_dist <= (1 / 8)
    if is_comp:
        label = int(is_same)
    else:
        label = 2
    return label


def make_test_pairwise_labels2(cases1, cases2):
    import vtool as vt

    is_same = np.array(cases1['name']) == np.array(cases2['name'])
    yaw1 = np.array(cases1['yaw'])
    yaw2 = np.array(cases2['yaw'])
    yaw_dist = vt.ori_distance(yaw1, yaw2) / np.pi

    qual1 = np.array(cases1['qual'])
    qual2 = np.array(cases2['qual'])

    _and = np.logical_and
    _or = np.logical_or
    _not = np.logical_not

    poorqual = _or(qual1 < 2, qual2 < 2)
    bestqual = _and(_or(qual1 > 3, qual2 > 3), _not(poorqual))
    goodqual = _and(_not(bestqual), _not(poorqual))

    is_comp = _or.reduce(
        [
            _not(poorqual),
            _and(bestqual, yaw_dist <= (1 / 4)),
            _and(goodqual, yaw_dist <= (1 / 8)),
        ]
    )

    label = is_same.astype(np.int32)
    label[_not(is_comp)] = 2
    return label


def try_em():
    """
    CommandLine:
        python -m wbia.algo.hots.testem test_em --show
        python -m wbia.algo.hots.testem test_em --show --no-cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.testem import *  # NOQA
        >>> P, Pn, PL, gam, num_labels = test_em()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.qt4ensure()
        >>> draw_em_graph(P, Pn, PL, gam, num_labels)
        >>> ut.show_if_requested()
    """
    print('EM')

    # Matrix if unary probabilites, The probability that each node takes on a
    # given label, independent of its edges.

    test_case = [
        {'name': 1, 'view': 'L'},
        {'name': 1, 'view': 'L'},
        {'name': 2, 'view': 'L'},
        {'name': 2, 'view': 'R'},
        {'name': 2, 'view': 'B'},
        {'name': 3, 'view': 'L'},
        # {'name': 3, 'view': 'L'},
        # {'name': 4, 'view': 'L'},
    ]

    def make_test_similarity(test_case):
        # toy_params = {
        #    True:  {'mu': 0.9, 'sigma': .1},
        #    False: {'mu': 0.1, 'sigma': .4}
        # }
        # tau = np.pi * 2
        from wbia import constants as const

        # view_to_ori = const.VIEWTEXT_TO_YAW_RADIANS
        view_to_ori = ut.map_dict_keys(
            lambda x: const.YAWALIAS[x], const.VIEWTEXT_TO_YAW_RADIANS
        )
        # view_to_ori = {
        #     'F': -1 * tau / 4,
        #     'L':  0 * tau / 4,
        #     'B':  1 * tau / 4,
        #     'R':  2 * tau / 4,
        # }
        import vtool as vt

        nid_list = np.array(ut.dict_take_column(test_case, 'name'))
        yaw_list = np.array(
            ut.dict_take(view_to_ori, ut.dict_take_column(test_case, 'view'))
        )

        rng = np.random.RandomState(0)
        pmat = []
        for idx in range(len(test_case)):
            nid = nid_list[idx]
            yaw = yaw_list[idx]
            p_same = nid == nid_list
            p_comp = 1 - vt.ori_distance(yaw_list, yaw) / np.pi
            # estimate noisy measurements
            p_same_m = np.clip(p_same + rng.normal(0, 0.5, size=len(p_same)), 0, 0.9)
            p_comp_m = np.clip(p_comp + rng.normal(0, 0.5, size=len(p_comp)), 0, 0.9)
            #
            p_same_and_comp = p_same_m * p_comp_m
            pmat.append(p_same_and_comp)
        #
        P = np.array(pmat)
        P[np.diag_indices(len(P))] = 0
        P = P + P.T / 2
        P = np.clip(P, 0.01, 0.99)
        print(ut.hz_str(' P = ', ut.repr2(P, precision=2, max_line_width=140)))
        return P

    Pn = make_test_similarity(test_case)

    if False:
        Pn = np.array(
            np.matrix(
                b"""
            .0 .7 .3 .2 .4 .5;
            .7 .0 .4 .4 .3 .5;
            .3 .4 .0 .6 .1 .5;
            .2 .4 .6 .0 .2 .3;
            .4 .3 .1 .2 .0 .8;
            .5 .5 .5 .3 .8 .0
            """
            )
        )

        PL = np.array(
            np.matrix(
                b"""
            .7 .5 .5;
            .8 .4 .3;
            .5 .7 .3;
            .5 .8 .4;
            .3 .2 .8;
            .5 .5 .8
            """
            )
        )

    if True:
        Pn = np.array(
            np.matrix(
                b"""
            1.0  0.7  0.4  0.2;
            0.7  1.0  0.4  0.4;
            0.4  0.4  1.0  0.6;
            0.2  0.4  0.6  1.0
            """
            )
        )

        PL = np.array(
            np.matrix(
                b"""
            0.7  0.5  0.5;
            0.8  0.4  0.3;
            0.5  0.7  0.3;
            0.5  0.8  0.4
            """
            )
        )
    num_nodes = Pn.shape[0]

    for num_labels in range(1, 2):
        # Pn = np.array(np.matrix(
        #    b"""
        #    .0 .7 .3 .2 .4 .5;
        #    .7 .0 .4 .4 .3 .5;
        #    .3 .4 .0 .6 .1 .5;
        #    .2 .4 .6 .0 .2 .3;
        #    .4 .3 .1 .2 .0 .8;
        #    .5 .5 .5 .3 .8 .0
        #    """))

        # Uniform distribution over labels
        if 0:
            PL = np.ones((num_nodes, num_labels)) / num_labels
            # Give nodes preferences
            PL[np.diag_indices(num_labels)] *= 1.01
            PL /= np.linalg.norm(PL, axis=0)
            # PL[0, :] = .01 / (num_labels - 1)
            # PL[0, 0] = .99
        else:
            PL /= np.linalg.norm(PL, axis=0)

        # Number of nodes
        num_nodes = Pn.shape[0]
        # Number of classes
        num_labels = PL.shape[1]
        # num_labels = num_nodes
        # if 0 or num_labels != 3:
        #    PL = np.ones((num_nodes, num_labels)) / num_labels
        #    # PL[0, :] = .01 / (num_labels - 1)
        #    # PL[0, 0] = .99
        d = num_labels + num_nodes

        # Stack everything into a single matrix
        zero_part = np.zeros((num_labels, num_nodes + num_labels))
        prob_part = np.hstack([PL, Pn])
        # print(ut.hz_str(' prob_part = ', ut.repr2(prob_part[:, :], precision=2)))
        P = np.vstack([zero_part, prob_part])

        # Gamma will hold a probability distribution over the nodes
        # The labeled nodes must match themselves.
        # The unlabeld nodes are initialized with a uniform distribution.
        gam = np.hstack(
            [np.eye(num_labels), np.ones((num_labels, num_nodes)) / num_labels]
        )

        print('Initialize')
        print('num_labels = %r' % (num_labels,))
        # print(ut.hz_str(' gamma = ', ut.repr2(gam[:, num_labels:], max_line_width=140, precision=2)))
        print(ut.hz_str(' gamma = ', ut.repr2(gam, max_line_width=140, precision=2)))

        delta_i = np.zeros(num_labels)

        def dErr(i, gam, P, delta_i=delta_i):
            # exepcted liklihood is cross entropy error
            delta_i[:] = 0
            # Compute the gradient of the cross entropy error
            # This is over both names and annotations
            for j in range(d):
                if i != j:
                    delta_i += gam[:, j] * np.log(P[i, j] / (1 - P[i, j]))
            # compute the projected gradient
            delta_i_hat = delta_i - delta_i.sum() / num_labels
            return delta_i_hat

        # Maximies the expected liklihood of gamma
        learn_rate = 0.05
        num_iters = 1000
        dGam = np.zeros(gam.shape)
        # for count in range(num_iters):
        for count in ut.ProgIter(range(num_iters), label='EM', bs=True):
            # Compute error gradient
            for i in range(num_labels, d):
                dGam[:, i] = dErr(i, gam, P)
            # Make a step in the gradient direction
            # print(ut.hz_str(' dGam = ', ut.repr2(dGam, max_line_width=140, precision=2)))
            gam = gam + learn_rate * dGam
            # Normalize
            gam = np.clip(gam, 0, 1)
            for i in range(num_labels, d):
                gam[:, i] = gam[:, i] / np.sum(gam[:, i])
        # print(ut.hz_str(' gamma = ', ut.repr2(gam, max_line_width=140, precision=2)))
        # print(ut.hz_str(' gamma = ', ut.repr2(gam[:, num_labels:], max_line_width=140, precision=2)))
        print('Finished')
    return P, Pn, PL, gam, num_labels


def try_em2(prob_names, prob_annots=None):
    """
    assert prob_names.shape == (nAnnots, nNames)
    """
    learn_rate = 0.05
    num_iters = 1

    # Matrix if unary probabilites, The probability that each node takes on a
    # given label, independent of its edges.
    num_annots, num_names = prob_names.shape

    # prevent zero probabilities
    prob_names_ = prob_names + 1e-9
    prob_names_ /= prob_names_.sum(axis=1)[:, None]

    if prob_annots is None:
        prob_annots_ = np.full((num_annots, num_annots), 1 / num_annots)
        prob_annots_[np.diag_indices(num_annots)] *= 1.01
        # perterb
        rng = np.random.RandomState(0)
        prob_annots_ += (rng.randn(*prob_annots_.shape)) / 100
        prob_annots_ /= prob_annots_.sum(axis=1)[:, None]
        prob_annots_ = (prob_annots_.T + prob_annots_) / 2
    else:
        prob_annots_ = prob_annots + 1e-9
        prob_annots_ /= prob_annots_.sum(axis=1)[:, None]

    # Stack everything into a single matrix
    prob_part = np.hstack([prob_names_, prob_annots_])
    zero_part = np.zeros((num_names, num_annots + num_names))
    prior = np.vstack([zero_part, prob_part])

    # Gamma will hold a probability distribution over the nodes
    # The labeled nodes must match themselves.
    # The unlabeld nodes are initialized with a uniform distribution.
    gam = np.hstack([np.eye(num_names), np.ones((num_names, num_annots)) / num_names])

    verbose = 1
    if verbose:
        print('Initialize')
        print('num_names = %r' % (num_names,))
        print(
            ut.hz_str(
                'prior = ',
                ut.repr2(
                    prob_part[:, :], precision=2, max_line_width=140, suppress_small=True
                ),
            )
        )
        print(
            ut.hz_str(
                'gamma = ',
                ut.repr2(gam[:, :], max_line_width=140, precision=2, suppress_small=True),
            )
        )
    # print(ut.hz_str(' gamma = ', ut.repr2(gam, max_line_width=140, precision=2)))

    delta_i = np.zeros(num_names)

    def dErr(i, gam, prior, delta_i=delta_i, num_names=num_names):
        # exepcted liklihood is cross entropy error
        delta_i[:] = 0
        # Compute the gradient of the cross entropy error
        # This is over both names and annotations
        jdxs = [j for j in range(prior.shape[0]) if j != i]
        prior_ij = prior[i, jdxs]
        np.log(prior_ij / (1 - prior_ij))
        gam[:, jdxs]

        for j in range(prior.shape[0]):
            if i != j:
                delta_i += gam[:, j] * np.log(prior[i, j] / (1 - prior[i, j]))
        # compute the projected gradient
        delta_i_hat = delta_i - delta_i.sum() / num_names
        return delta_i_hat

    # Build node for each annot and each name
    num_nodes = num_annots + num_names

    # Maximies the expected liklihood of gamma
    dGam = np.zeros(gam.shape)
    # for count in range(num_iters):
    for count in ut.ProgIter(range(num_iters), label='EM', bs=True):
        # Compute error gradient
        for i in range(num_names, num_nodes):
            dGam[:, i] = dErr(i, gam, prior)
        # Make a step in the gradient direction
        # print(ut.hz_str(' dGam = ', ut.repr2(dGam, max_line_width=140, precision=2)))
        gam = gam + learn_rate * dGam
        # Normalize
        gam = np.clip(gam, 0, 1)
        for i in range(num_names, num_nodes):
            gam[:, i] = gam[:, i] / np.sum(gam[:, i])
    # print(ut.hz_str(' gamma = ', ut.repr2(gam, max_line_width=140, precision=2)))
    if verbose:
        print(
            ut.hz_str(
                ' gamma = ',
                ut.repr2(
                    gam[:, num_names:],
                    max_line_width=140,
                    precision=2,
                    suppress_small=True,
                ),
            )
        )
        print('Finished')
    return gam


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.hots.testem
        python -m wbia.algo.hots.testem --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()

r"""

I've gone over both the pdf and the code and I would like to clarify a few things.

In the pdf the measure of uncertainty u_{ij} seems to not be used anywhere
else, is this true or am I missing it?

In the code there is a matrix PL and Pn.
I believe Pn is represented by the edge between the numbered nodes and the
lettered nodes, while PL (corresponds to gamma and) represents the connections
between the lettered nodes.  Is it correct that PL is proportional to the
probability that node i takes label j, and Pn is the probability that node i
matches node j?

In the animal ID problem there is not always a notion of groundtruth likelihood.
Nor, is the groundtruth fixed.  We need to find and correct errors in the
groundtruth.

For instance node 1 and node 2 might actually need to have the same label (node
1 may be a right side zebra and node 2 may be a left side zebra. Identity
equivalence is a transitive property. We need to be able to infer labeling
through chains of high likelihood connections even though the end points might
have a low probability edge between them.)

Using the language of image segmentation, I believe that without the "unary"
terms on each node (the probability of them taking on a given label) the
solution converges to a uniform distribution regardless of the "pairwise" terms
between the lettered nodes.


In summary:
    * How does this framework address uncertainty in the pairwise probabilities?

    * Does this framework work in the scenario where there is no groundtruth
      and the number of labels is unknown?


I'm having trouble understanding how this EM algorithm addresses uncertainty
and the case where there is no label information.

First, I do not see how uncertaint plays a role in the formulation. In the
attached pdf it is defined, but never used.

Second, I don't see how this works in the case where there are no "groundtruth"
exemplars and the number of classes is unknown.  Consider the case where we have
6 unlabeled annotations and we compute the pairwise similarity between each of
them.  We do not know how many individuals (classes) there are in this set.
There is a minimum of 1 and a maximum of 6.  We cannot compute the (unary)
probability that a pariticular annotation takes on a particular class because
the class is just an abstract notion used to define a partition of this graph.








If we have a unlabeled set of annotations and we compute the pairwise similarity between each of them,
how does this

I don't see how the uncertainty measure plays a role in the

I don't see how to get away from the "unary"
terms in this formulation

In fact in the example you gave, if you simply look at the probabilities on
each edge from the labeled nodes to the unlabeled nodes, you can simply take
the edge with the highest weight to a labeled node and arive at the same solution.
"""
