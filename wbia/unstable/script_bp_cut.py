# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import pandas as pd
import vtool as vt  # NOQA
import networkx as nx
import opengm
import wbia.plottool as pt  # NOQA
import utool as ut

print, rrr, profile = ut.inject2(__name__)

# pd.set_option('display.float_format', lambda x: '%.4f' % x)
# pd.options.display.precision = 4


# a---b
# c---d


tests = ut.odict()


def register_test(func):
    tests[ut.get_funcname(func)] = func
    return func


@register_test
def chain1():
    uvw_list = [
        ('a', 'b', 0.8),
        ('b', 'c', 0.8),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def chain2():
    uvw_list = [
        ('a', 'b', 0.8),
        ('b', 'c', 0.2),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def chain3():
    uvw_list = [
        ('a', 'b', 0.2),
        ('b', 'c', 0.2),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def triangle1():
    uvw_list = [
        ('a', 'b', 0.8),
        ('b', 'c', 0.8),
        ('c', 'a', 0.8),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def triangle2():
    uvw_list = [
        ('a', 'b', 0.8),
        ('b', 'c', 0.2),
        ('c', 'a', 0.2),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def triangle3():
    uvw_list = [
        ('a', 'b', 0.8),
        ('b', 'c', 0.8),
        ('c', 'a', 0.2),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def missing_lots():
    uvw_list = [
        ('a', 'b', 0.8),
        ('c', 'd', 0.2),
        ('e', 'f', 0.2),
        ('g', 'h', 0.2),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def consistent_info():
    r"""
    Test Consistent Info
    ----------------------
    A -- B
    | \/ |
    | /\ |
    C----D
    In this test the most likely partitioning is
    A -- B
    C -- D
    Edges between these partitions have low probabilites.
    This makes all information in this graph consistent.

    Correct labeling should be
    [0, 0, 1, 1]
    """
    uvw_list = [
        ('a', 'b', 0.8),
        ('c', 'd', 0.8),
        ('b', 'c', 0.2),
        ('a', 'c', 0.2),
        ('a', 'd', 0.2),
        ('b', 'd', 0.2),
    ]
    pass_values = [[0, 0, 1, 1]]
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def inconsistent_info():
    r"""
    Test Inconsistent Info
    ----------------------
    This test adds onto the first and makes it almost impossible that b and d are
    not the same. a, b and c, d are still likely to match, but there is also lots
    of negative evidence that a!=c, a!=d, c!=b.
    A -- B
    | \/ ‖
    | /\ ‖
    C -- D
    The network must rectify this contradictory information.

    Correct labeling should be
    [0, 1, 1, 1]?
    OR
    [1, 1, 0, 1]?
    OR
    [1, 1, 1, 1]?
    OR
    [0, 1, 2, 1]?
    BUT IT SHOULD NOT BE [0, 0, 1, 1]
    """
    uvw_list = [
        ('a', 'b', 0.8),
        ('c', 'd', 0.8),
        ('b', 'c', 0.2),
        ('a', 'c', 0.2),
        ('a', 'd', 0.2),
        ('b', 'd', 0.99999),
    ]
    pass_values = [
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
    ]
    fail_values = [[0, 0, 1, 1]]
    return uvw_list, pass_values, fail_values


@register_test
def inconsistent_info2():
    """
    ALso ensures that b and c are very likely to be different
    """
    uvw_list = [
        ('a', 'b', 0.8),
        ('c', 'd', 0.8),
        ('b', 'c', 0.001),
        ('a', 'c', 0.2),
        ('a', 'd', 0.2),
        ('b', 'd', 0.99999),
    ]
    pass_values = [
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 2, 1],
    ]
    fail_values = [[0, 0, 1, 1]]
    return uvw_list, pass_values, fail_values


@register_test
def pos_incomplete():
    r"""
    Test Postive Incomplete Info
    ----------------------
    This test adds an exta node, e, with a tiny preference towards matching c, d
     +---A --- B---+
     |   | \ / |   |
     |   | / \ |   |
     |   C --- D   |
     |    \   /    |
     |_____ E _____|
    Correct labeling should be
    [0, 0, 1, 1, 1]
    """
    uvw_list = [
        ('a', 'b', 0.8),
        ('c', 'd', 0.8),
        ('b', 'c', 0.2),
        ('a', 'c', 0.2),
        ('a', 'd', 0.2),
        ('b', 'd', 0.2),
        # ('e', 'a', .00001),
        # ('e', 'b', .00001),
        ('e', 'c', 0.51),
        ('e', 'd', 0.51),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def neg_incomplete():
    r"""
    Test Negative Incomplete Info
    ----------------------
    This test adds an exta node, e, with a tiny preference of not matching c, d
     +---A --- B---+
     |   | \ / |   |
     |   | / \ |   |
     |   C --- D   |
     |    \   /    |
     |_____ E _____|
    Correct labeling should be
    [0, 0, 1, 1, 1]
    """
    uvw_list = [
        ('a', 'b', 0.8),
        ('c', 'd', 0.8),
        ('b', 'c', 0.2),
        ('a', 'c', 0.2),
        ('a', 'd', 0.2),
        ('b', 'd', 0.2),
        # ('e', 'a', .00001),
        # ('e', 'b', .00001),
        ('e', 'c', 0.49),
        ('e', 'd', 0.49),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


index_type = opengm.index_type


def rectify_labels(G, labels):
    # Ensure labels are rebased and
    # are different between different connected compoments
    graph = G.copy()
    node_to_annot_idx = nx.get_node_attributes(graph, 'annot_idx')
    cut_edges = []
    for u, v in graph.edges():
        idx1 = node_to_annot_idx[u]
        idx2 = node_to_annot_idx[v]
        if labels[idx1] != labels[idx2]:
            cut_edges.append((u, v))
    graph.remove_edges_from(cut_edges)
    ccs_nodes = list(nx.connected_components(graph))
    ccs_idxs = ut.unflat_take(node_to_annot_idx, ccs_nodes)
    # Make consistent sorting
    ccs_idxs = [sorted(idxs) for idxs in ccs_idxs]
    ccs_idxs = ut.sortedby(ccs_idxs, ut.take_column(ccs_idxs, 0))
    labels = ut.ungroup([[c] * len(x) for c, x in enumerate(ccs_idxs)], ccs_idxs)
    labels = np.array(labels)
    return labels


def get_edge_id_probs(G, aid1, aid2, n_names):
    p_match = G.get_edge_data(aid1, aid2)['weight']
    p_noncomp = 0
    p_diff = 1 - (p_match + p_noncomp)
    p_bg = 1 / n_names
    p_same = p_match + p_noncomp * p_bg
    p_diff = 1 - p_same
    return p_same, p_diff


def build_factor_graph(
    G,
    nodes,
    edges,
    n_annots,
    n_names,
    lookup_annot_idx,
    use_unaries=True,
    edge_probs=None,
    operator='multiplier',
):

    node_state_card = np.ones(n_annots, dtype=index_type) * n_names
    numberOfStates = node_state_card
    # n_edges = len(edges)
    # n_edge_states = 2
    # edge_state_card = np.ones(n_edges, dtype=index_type) * n_edge_states
    # numberOfStates = np.hstack([node_state_card, edge_state_card])
    # gm = opengm.graphicalModel(numberOfStates, operator='adder')
    gm = opengm.graphicalModel(numberOfStates, operator=operator)

    annot_idxs = list(range(n_annots))
    # edge_idxs = list(range(n_annots, n_annots + n_edges))
    import scipy.special

    if use_unaries:
        unaries = np.ones((n_annots, n_names)) / n_names
        # unaries[0][0] = 1
        # unaries[0][1:] = 0
        for annot_idx in annot_idxs:
            fid = gm.addFunction(unaries[annot_idx])
            gm.addFactor(fid, annot_idx)

    # Add Potts function for each edge
    pairwise_factor_idxs = []
    for count, (aid1, aid2) in enumerate(edges, start=len(list(gm.factors()))):
        varx1, varx2 = ut.take(lookup_annot_idx, [aid1, aid2])
        var_indicies = np.array([varx1, varx2])

        if edge_probs is None:
            p_same, p_diff = get_edge_id_probs(G, aid1, aid2, n_names)
        else:
            p_same, p_diff = edge_probs[count]

        use_logit = operator == 'adder'
        if use_logit:
            eps = 1e-9
            p_same = np.clip(p_same, eps, 1.0 - eps)
            same_weight = scipy.special.logit(p_same)
            # valueEqual = -same_weight
            valueEqual = 0
            valueNotEqual = same_weight
            if not np.isfinite(valueNotEqual):
                """
                python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=-1,1 --func=scipy.special.logit
                """
                print('valueNotEqual = %r' % (valueNotEqual,))
                print('p_same = %r' % (p_same,))
                raise ValueError('valueNotEqual')
        else:
            valueEqual = p_same
            valueNotEqual = p_diff

        p_same, p_diff = get_edge_id_probs(G, aid1, aid2, n_names)
        pairwise_factor_idxs.append(count)

        potts_func = opengm.PottsFunction(
            (n_names, n_names), valueEqual=valueEqual, valueNotEqual=valueNotEqual
        )
        potts_func_id = gm.addFunction(potts_func)
        gm.addFactor(potts_func_id, var_indicies)

    gm.pairwise_factor_idxs = pairwise_factor_idxs
    gm.G = G
    return gm


def cut_step(
    G,
    nodes,
    edges,
    n_annots,
    n_names,
    lookup_annot_idx,
    edge_probs,
    pass_values,
    fail_values,
):
    # Create nodes in the graphical model.  In this case there are <num_vars>
    # nodes and each node can be assigned to one of <num_vars> possible labels
    space = np.full((n_annots,), fill_value=n_names, dtype=opengm.index_type)
    gm = opengm.gm(space, operator='adder')

    # Use one potts function for each edge
    gm = build_factor_graph(
        G,
        nodes,
        edges,
        n_annots,
        n_names,
        lookup_annot_idx,
        use_unaries=False,
        edge_probs=edge_probs,
        operator='adder',
    )

    with ut.Indenter('[CUTS]'):
        ut.cprint('Brute Force Labels: (energy minimization)', 'blue')
        infr = opengm.inference.Bruteforce(gm, accumulator='minimizer')
        infr.infer()
        labels = rectify_labels(G, infr.arg())
        print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)
        print('value = %r' % (infr.value(),))

        mc_params = opengm.InfParam(
            maximalNumberOfConstraintsPerRound=1000000,
            initializeWith3Cycles=True,
            edgeRoundingValue=1e-08,
            timeOut=36000000.0,
            cutUp=1e75,
            reductionMode=3,
            numThreads=0,
            # allowCutsWithin=?
            # workflow=workflow
            verbose=False,
            verboseCPLEX=False,
        )
        infr = opengm.inference.Multicut(gm, parameter=mc_params, accumulator='minimizer')

        infr.infer()
        labels = infr.arg()
        labels = rectify_labels(G, infr.arg())

        ut.cprint('Multicut Labels: (energy minimization)', 'blue')
        print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)
        print('value = %r' % (infr.value(),))

        if pass_values is not None:
            gotany = False
            for pval in pass_values:
                if all(labels == pval):
                    gotany = True
                    break
            if not gotany:
                ut.cprint('INCORRECT DID NOT GET PASS VALUES', 'red')
                print('pass_values = %r' % (pass_values,))

        if fail_values is not None:
            for fail in fail_values:
                if all(labels == fail):
                    ut.cprint('INCORRECT', 'red')

    # ae_params = opengm.InfParam(steps=1000)  # scale=1.0, minStCut='boost-kolmogorov')
    # infr = opengm.inference.AlphaExpansion(gm, parameter=ae_params, accumulator='minimizer')
    # infr.infer()
    # labels = infr.arg()
    # print('AlphaExpansion labels:')
    # print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)


def bp_step(G, nodes, edges, n_annots, n_names, lookup_annot_idx):
    gm = build_factor_graph(
        G,
        nodes,
        edges,
        n_annots,
        n_names,
        lookup_annot_idx,
        use_unaries=False,
        edge_probs=None,
        operator='multiplier',
    )

    with ut.Indenter('[BELIEF]'):
        ut.cprint('Brute Force Labels: (probability maximization)', 'blue')
        infr = opengm.inference.Bruteforce(gm, accumulator='maximizer')
        infr.infer()
        labels = rectify_labels(G, infr.arg())
        print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)
        print('value = %r' % (infr.value(),))

        lpb_parmas = opengm.InfParam(
            damping=0.00,
            steps=10000,
            # convergenceBound=0,
            isAcyclic=False,
        )
        # http://www.andres.sc/publications/opengm-2.0.2-beta-manual.pdf
        # I believe multiplier + integrator = marginalization
        # Manual says multiplier + adder = marginalization
        # Manual says multiplier + maximizer = probability maximization
        # infr = opengm.inference.TreeReweightedBp(
        LBP_algorithm = opengm.inference.BeliefPropagation
        # LBP_algorithm = opengm.inference.TreeReweightedBp

        ut.cprint('Belief Propogation (maximization)', 'blue')
        infr = LBP_algorithm(gm, parameter=lpb_parmas, accumulator='maximizer')
        infr.infer()
        labels = rectify_labels(G, infr.arg())
        pairwise_factor_idxs = gm.pairwise_factor_idxs
        factor_marginals = infr.factorMarginals(pairwise_factor_idxs)
        # print('factor_marginals =\n%r' % (factor_marginals,))
        edge_marginals_same_diff_ = [
            (np.diag(f).sum(), f[~np.eye(f.shape[0], dtype=bool)].sum())
            for f in factor_marginals
        ]
        edge_marginals_same_diff_ = np.array(edge_marginals_same_diff_)
        edge_marginals_same_diff = edge_marginals_same_diff_.copy()
        edge_marginals_same_diff /= edge_marginals_same_diff.sum(axis=1, keepdims=True)
        print('Unnormalized Edge Marginals:')
        print(
            pd.DataFrame(
                edge_marginals_same_diff,
                columns=['same', 'diff'],
                index=pd.Series(edges),
            )
        )
        # print('Edge marginals after Belief Propogation')
        # print(pd.DataFrame(edge_marginals_same_diff, columns=['same', 'diff'], index=pd.Series(edges)))
        print('Labels:')
        print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)
        print('value = %r' % (infr.value(),))

        ut.cprint('Belief Propogation (marginalization)', 'blue')
        infr = LBP_algorithm(gm, parameter=lpb_parmas, accumulator='integrator')
        infr.infer()
        labels = rectify_labels(G, infr.arg())
        pairwise_factor_idxs = gm.pairwise_factor_idxs
        factor_marginals = infr.factorMarginals(pairwise_factor_idxs)
        # print('factor_marginals =\n%r' % (factor_marginals,))
        edge_marginals_same_diff_ = [
            (np.diag(f).sum(), f[~np.eye(f.shape[0], dtype=bool)].sum())
            for f in factor_marginals
        ]
        edge_marginals_same_diff_ = np.array(edge_marginals_same_diff_)
        edge_marginals_same_diff = edge_marginals_same_diff_.copy()
        edge_marginals_same_diff /= edge_marginals_same_diff.sum(axis=1, keepdims=True)
        print('Unnormalized Edge Marginals:')
        print(
            pd.DataFrame(
                edge_marginals_same_diff,
                columns=['same', 'diff'],
                index=pd.Series(edges),
            )
        )
        # print('Edge marginals after Belief Propogation')
        # print(pd.DataFrame(edge_marginals_same_diff, columns=['same', 'diff'], index=pd.Series(edges)))
        print('Labels:')
        print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)
        print('value = %r' % (infr.value(),))

    # import wbia.plottool as pt
    # viz_factor_graph(gm)
    # # _ = pt.show_nx(G)
    # print("SHOW")
    # pt.plt.show()

    # marginals = infr.marginals(annot_idxs)
    # print('node marginals are')
    # print(pd.DataFrame(marginals, index=pd.Series(nodes)))
    return edge_marginals_same_diff


def viz_factor_graph(gm):
    """
    ut.qtensure()
    gm = build_factor_graph(G, nodes, edges , n_annots, n_names, lookup_annot_idx,
                            use_unaries=True, edge_probs=None, operator='multiplier')
    """
    ut.qtensure()
    import networkx
    from networkx.drawing.nx_agraph import graphviz_layout

    networkx.graphviz_layout = graphviz_layout
    opengm.visualizeGm(
        gm,
        show=False,
        layout='neato',
        plotUnaries=True,
        iterations=1000,
        plotFunctions=True,
        plotNonShared=False,
        relNodeSize=1.0,
    )

    _ = pt.show_nx(gm.G)  # NOQA

    # import utool
    # utool.embed()
    # infr = opengm.inference.Bruteforce
    # infr = opengm.inference.Bruteforce(gm, accumulator='maximizer')
    # # infr = opengm.inference.Bruteforce(gm, accumulator='maximizer')
    # # infr = opengm.inference.Bruteforce(gm, accumulator='integrator')
    # infr.infer()
    # print(infr.arg())
    # print(infr.value())


def build_graph(uvw_list):
    _edges = [(u, v, {'weight': w}) for (u, v, w) in uvw_list]
    G = nx.Graph()
    G.add_edges_from(_edges)
    node_to_label = {
        e: '%s,%s\n%s' % (e + (d,))
        for e, d in nx.get_edge_attributes(G, 'weight').items()
    }
    nx.set_edge_attributes(G, name='label', values=node_to_label)
    return G


def main():
    tests_ = tests
    subset = ['consistent_info', 'inconsistent_info']
    subset = ['chain1', 'chain2', 'chain3']
    subset += ['triangle1', 'triangle2', 'triangle3']
    # subset = ['inconsistent_info']
    tests_ = ut.dict_subset(tests, subset)

    for name, func in tests_.items():
        print('\n==============')
        ut.cprint('name = %r' % (name,), 'yellow')
        uvw_list, pass_values, fail_values = func()
        G = build_graph(uvw_list)

        nodes = sorted(G.nodes())
        edges = [tuple(sorted(e)) for e in G.edges()]
        edges = ut.sortedby2(edges, edges)

        n_annots = len(nodes)
        n_names = n_annots

        annot_idxs = list(range(n_annots))
        lookup_annot_idx = ut.dzip(nodes, annot_idxs)
        nx.set_node_attributes(G, name='annot_idx', values=lookup_annot_idx)

        edge_probs = np.array(
            [get_edge_id_probs(G, aid1, aid2, n_names) for aid1, aid2 in edges]
        )

        print('nodes = %r' % (nodes,))
        # print('edges = %r' % (edges,))
        print('Noisy Observations')
        print(pd.DataFrame(edge_probs, columns=['same', 'diff'], index=pd.Series(edges)))
        edge_probs = None

        cut_step(
            G,
            nodes,
            edges,
            n_annots,
            n_names,
            lookup_annot_idx,
            edge_probs,
            pass_values,
            fail_values,
        )

        edge_probs = bp_step(G, nodes, edges, n_annots, n_names, lookup_annot_idx)

        # cut_step(G, nodes, edges , n_annots, n_names, lookup_annot_idx, edge_probs, None, fail_values)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.hots.script_bp_cut
        python -m wbia.algo.hots.script_bp_cut --allexamples
    """
    main()
    # ut.quit_if_noshow()
    ut.show_if_requested()
