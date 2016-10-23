import numpy as np
import pandas as pd
import vtool as vt  # NOQA
import networkx as nx
import opengm
# import plottool as pt
import utool as ut

# a---b
# c---d

_edges = [(u, v, {'weight': w}) for (u, v, w) in [
    ('a', 'b', .8),
    ('c', 'd', .8),
    ('b', 'c', .2),
    ('a', 'c', .2),
    ('a', 'd', .2),
    # ('b', 'd', .2),
    ('b', 'd', .99999999999999999),
    # ('e', 'd', .00001),
    # ('e', 'c', .00001),
    ('e', 'a', .51),
    # ('e', 'b', .51),
]]
G = nx.Graph()
G.add_edges_from(_edges)
node_to_label = {e: '%s,%s\n%s' % (e + (d,)) for e, d in nx.get_edge_attributes(G, 'weight').items()}
nx.set_edge_attributes(G, 'label', node_to_label)
# G.remove_edge('a', 'c')

index_type = opengm.index_type
nodes = sorted(G.nodes())
edges = [tuple(sorted(e)) for e in G.edges()]
edges = ut.sortedby2(edges, edges)

n_edges = len(edges)
n_annots = len(nodes)
n_names = n_annots
n_edge_states = 2

annot_idxs = list(range(n_annots))
lookup_annot_idx = ut.dzip(nodes, annot_idxs)

print('nodes = %r' % (nodes,))
print('edges = %r' % (edges,))
print('lookup_annot_idx = %r' % (lookup_annot_idx,))


def get_edge_match_probs(aid1, aid2):
    p_match = G.get_edge_data(aid1, aid2)['weight']
    p_noncomp = 0
    p_diff = 1 - (p_match + p_noncomp)
    return dict(p_match=p_match, p_noncomp=p_noncomp, p_diff=p_diff)


def get_edge_id_probs(aid1, aid2):
    probs = get_edge_match_probs(aid1, aid2)
    B = 1 / n_names
    p_same = probs['p_match'] + probs['p_match'] * B
    p_diff = 1 - p_same
    return p_same, p_diff


# CUT STEP

def cut_step(edge_probs):

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
            p_same, p_diff = get_edge_id_probs(aid1, aid2)
        else:
            p_same, p_diff = edge_probs[count]

        # valueEqual = p_diff
        # valueNotEqual = p_same
        valueEqual = 0
        valueNotEqual = p_same - .5
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
    print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)))

    ae_params = opengm.InfParam(steps=1000)  # scale=1.0, minStCut='boost-kolmogorov')
    infr = opengm.inference.AlphaExpansion(gm, parameter=ae_params, accumulator='minimizer')
    infr.infer()
    labels = infr.arg()
    print('AlphaExpansion labels:')
    print(pd.DataFrame(labels, columns=['nid'], index=pd.Series(nodes)))


# BP STEP
def bp_step():
    node_state_card = np.ones(n_annots, dtype=index_type) * n_names
    numberOfStates = node_state_card
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

        p_same, p_diff = get_edge_id_probs(aid1, aid2)
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
    # infr = opengm.inference.BeliefPropagation(
    #     gm, parameter=lpb_parmas,
    #     accumulator='integrator'
    #     # accumulator='maximizer'
    # )
    infr.infer()

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


# pd.options.display.precision = 3
pd.set_option('display.float_format', lambda x: '%.3f' % x)
edge_probs = np.array([get_edge_id_probs(aid1, aid2) for aid1, aid2 in edges])
print('Noisy Observations')
print(pd.DataFrame(edge_probs, columns=['same', 'diff'], index=pd.Series(edges)))
edge_probs = None
cut_step(edge_probs)
edge_probs = bp_step()
cut_step(edge_probs)
