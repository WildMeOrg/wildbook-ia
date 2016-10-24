# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import pandas as pd
import vtool as vt  # NOQA
import networkx as nx
import opengm
import plottool as pt  # NOQA
import utool as ut

# pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.options.display.precision = 4


# a---b
# c---d


tests = []


def register_test(func):
    tests.append(func)
    return func


@register_test
def missing_lots():
    uvw_list = [
        ('a', 'b', .8),
        ('c', 'd', .2),
        ('e', 'f', .2),
        ('g', 'h', .2),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def consistent_info():
    """
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
        ('a', 'b', .8),
        ('c', 'd', .8),
        ('b', 'c', .2),
        ('a', 'c', .2),
        ('a', 'd', .2),
        ('b', 'd', .2),
    ]
    pass_values = [
        [0, 0, 1, 1]
    ]
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def inconsistent_info():
    """
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
        ('a', 'b', .8),
        ('c', 'd', .8),
        ('b', 'c', .2),
        ('a', 'c', .2),
        ('a', 'd', .2),
        ('b', 'd', .99999),
    ]
    pass_values = [
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
    ]
    fail_values = [
        [0, 0, 1, 1]
    ]
    return uvw_list, pass_values, fail_values


@register_test
def inconsistent_info2():
    """
    ALso ensures that b and c are very likely to be different
    """
    uvw_list = [
        ('a', 'b', .8),
        ('c', 'd', .8),
        ('b', 'c', .001),
        ('a', 'c', .2),
        ('a', 'd', .2),
        ('b', 'd', .99999),
    ]
    pass_values = [
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 2, 1],
    ]
    fail_values = [
        [0, 0, 1, 1]
    ]
    return uvw_list, pass_values, fail_values


@register_test
def pos_incomplete():
    """
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
        ('a', 'b', .8),
        ('c', 'd', .8),
        ('b', 'c', .2),
        ('a', 'c', .2),
        ('a', 'd', .2),
        ('b', 'd', .2),
        # ('e', 'a', .00001),
        # ('e', 'b', .00001),
        ('e', 'c', .51),
        ('e', 'd', .51),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


@register_test
def neg_incomplete():
    """
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
        ('a', 'b', .8),
        ('c', 'd', .8),
        ('b', 'c', .2),
        ('a', 'c', .2),
        ('a', 'd', .2),
        ('b', 'd', .2),
        # ('e', 'a', .00001),
        # ('e', 'b', .00001),
        ('e', 'c', .49),
        ('e', 'd', .49),
    ]
    pass_values = None
    fail_values = None
    return uvw_list, pass_values, fail_values


index_type = opengm.index_type


def get_edge_id_probs(G, aid1, aid2, n_names):
    p_match = G.get_edge_data(aid1, aid2)['weight']
    p_noncomp = 0
    p_diff = 1 - (p_match + p_noncomp)
    p_bg = 1 / n_names
    p_same = p_match + p_noncomp * p_bg
    p_diff = 1 - p_same
    return p_same, p_diff


def cut_step(G, nodes, edges, n_annots, n_names, lookup_annot_idx, edge_probs, pass_values, fail_values):
    # Create nodes in the graphical model.  In this case there are <num_vars>
    # nodes and each node can be assigned to one of <num_vars> possible labels
    space = np.full((n_annots,), fill_value=n_names, dtype=opengm.index_type)
    gm = opengm.gm(space, operator='adder')

    # Use one potts function for each edge
    # Add Potts function for each edge
    for count, (aid1, aid2) in enumerate(edges, start=len(list(gm.factors()))):
        varx1, varx2 = ut.take(lookup_annot_idx, [aid1, aid2])
        var_indicies = np.array([varx1, varx2])

        if edge_probs is None:
            p_same, p_diff = get_edge_id_probs(G, aid1, aid2, n_names)
        else:
            p_same, p_diff = edge_probs[count]

        import scipy.special
        mode = 'logit1'
        if mode == 0:
            valueEqual = p_diff
            valueNotEqual = p_same
        elif mode == 'logit1':
            eps = 1E-9
            p_same = np.clip(p_same, eps, 1.0 - eps)
            same_weight = scipy.special.logit(p_same)
            # valueEqual = -same_weight
            valueEqual = 0
            valueNotEqual = same_weight
            if not np.isfinite(valueNotEqual):
                """
                python -m plottool.draw_func2 --exec-plot_func --show --range=-1,1 --func=scipy.special.logit
                """
                print('valueNotEqual = %r' % (valueNotEqual,))
                print('p_same = %r' % (p_same,))
                raise ValueError('valueNotEqual')

        potts_func = opengm.PottsFunction((n_names, n_names), valueEqual=valueEqual, valueNotEqual=valueNotEqual)
        potts_func_id = gm.addFunction(potts_func)
        gm.addFactor(potts_func_id, var_indicies)

    mc_params = opengm.InfParam(maximalNumberOfConstraintsPerRound=1000000,
                                initializeWith3Cycles=True,
                                edgeRoundingValue=1e-08, timeOut=36000000.0,
                                cutUp=1e+75, reductionMode=3, numThreads=0,
                                # allowCutsWithin=?
                                # workflow=workflow
                                verbose=False, verboseCPLEX=False)
    infr = opengm.inference.Multicut(gm, parameter=mc_params,
                                     accumulator='minimizer')

    infr.infer()
    labels = infr.arg()
    print('Multicut labels:')
    print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)
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

    ae_params = opengm.InfParam(steps=1000)  # scale=1.0, minStCut='boost-kolmogorov')
    infr = opengm.inference.AlphaExpansion(gm, parameter=ae_params, accumulator='minimizer')
    infr.infer()
    labels = infr.arg()
    print('AlphaExpansion labels:')
    print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)


def bp_step(G, nodes, edges , n_annots, n_names, lookup_annot_idx):
    node_state_card = np.ones(n_annots, dtype=index_type) * n_names
    numberOfStates = node_state_card
    # n_edges = len(edges)
    # n_edge_states = 2
    # edge_state_card = np.ones(n_edges, dtype=index_type) * n_edge_states
    # numberOfStates = np.hstack([node_state_card, edge_state_card])
    # gm = opengm.graphicalModel(numberOfStates, operator='adder')
    gm = opengm.graphicalModel(numberOfStates, operator='multiplier')

    annot_idxs = list(range(n_annots))
    # edge_idxs = list(range(n_annots, n_annots + n_edges))

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

        p_same, p_diff = get_edge_id_probs(G, aid1, aid2, n_names)
        pairwise_factor_idxs.append(count)

        potts_func = opengm.PottsFunction((n_names, n_names), valueEqual=p_same, valueNotEqual=p_diff)
        potts_func_id = gm.addFunction(potts_func)
        gm.addFactor(potts_func_id, var_indicies)

    lpb_parmas = opengm.InfParam(damping=0.01, steps=100,
                                 # convergenceBound=0,
                                 isAcyclic=False)
    # http://www.andres.sc/publications/opengm-2.0.2-beta-manual.pdf
    # I believe multiplier + integrator = marginalization
    # Manual says multiplier + adder = marginalization
    # Manual says multiplier + maximizer = probability maximization
    # infr = opengm.inference.TreeReweightedBp(
    infr = opengm.inference.BeliefPropagation(
        gm, parameter=lpb_parmas,
        # accumulator='integrator'
        accumulator='maximizer'
    )
    print('BP labels')
    labels = infr.infer()
    print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)).T)

    factor_marginals = infr.factorMarginals(pairwise_factor_idxs)
    edge_marginals_same_diff = [(np.diag(f).sum(), f[~np.eye(f.shape[0], dtype=bool)].sum()) for f in factor_marginals]
    edge_marginals_same_diff = np.array(edge_marginals_same_diff)
    edge_marginals_same_diff /= edge_marginals_same_diff.sum(axis=1, keepdims=True)
    print('Edge marginals after Belief Propogation')
    print(pd.DataFrame(edge_marginals_same_diff, columns=['same', 'diff'], index=pd.Series(edges)))

    # marginals = infr.marginals(annot_idxs)
    # print('node marginals are')
    # print(pd.DataFrame(marginals, index=pd.Series(nodes)))
    return edge_marginals_same_diff


def build_graph(uvw_list):
    _edges = [(u, v, {'weight': w}) for (u, v, w) in uvw_list]
    G = nx.Graph()
    G.add_edges_from(_edges)
    node_to_label = {e: '%s,%s\n%s' % (e + (d,)) for e, d in nx.get_edge_attributes(G, 'weight').items()}
    nx.set_edge_attributes(G, 'label', node_to_label)
    return G


def main():
    for func in tests:
        print('\n==============')
        print(ut.get_funcname(func))
        uvw_list, pass_values, fail_values = func()
        G = build_graph(uvw_list)

        nodes = sorted(G.nodes())
        edges = [tuple(sorted(e)) for e in G.edges()]
        edges = ut.sortedby2(edges, edges)

        n_annots = len(nodes)
        n_names = n_annots

        annot_idxs = list(range(n_annots))
        lookup_annot_idx = ut.dzip(nodes, annot_idxs)

        edge_probs = np.array([get_edge_id_probs(G, aid1, aid2, n_names) for aid1, aid2 in edges])

        print('nodes = %r' % (nodes,))
        # print('edges = %r' % (edges,))
        print('Noisy Observations')
        print(pd.DataFrame(edge_probs, columns=['same', 'diff'], index=pd.Series(edges)))
        edge_probs = None

        cut_step(G, nodes, edges , n_annots, n_names, lookup_annot_idx, edge_probs, pass_values, fail_values)

        # edge_probs = bp_step(G, nodes, edges , n_annots, n_names, lookup_annot_idx)

        # cut_step(G, nodes, edges , n_annots, n_names, lookup_annot_idx, edge_probs)

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.script_bp_cut
        python -m ibeis.algo.hots.script_bp_cut --allexamples
    """
    main()
    # ut.quit_if_noshow()
    ut.show_if_requested()
