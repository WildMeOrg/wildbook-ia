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


Probability calculations:
    # The probability A matches B is maximum of the product of the
    # probabilities u matches v along every path between A and B.

    P(A matches B) = max(
        prod([data['p_match'] for u, v, data in path])
        for path in all_simple_paths(A, B)
        )
    # Given a linear chain of independant probabilities the joint probability
    # is easy to compute. This minimum of these probabilities over all paths
    # is the most probable way to argue that A matches B.
    # Note not-comparable edges severly undercut the probability of the most
    # probable path.

    Well... what if

    A -O- B -O- C
    |           |
    |_____X_____|

    The most probable path is A-B-C, but A-X-C directly contradicts it.

    Maybe if we find the most probable path and then consider all paths of
    smaller length? And take the min over that?

    p(A is B | AB) = .8
    p(B is C | BC) = .8
    p(A is C | AC) = .01

    I think we need to use belief propogation. First need to transform edges
    into nodes. This is done by transforming the original graph G with
    probabilities on edges into a "Line Graph" L(G)
    (https://en.wikipedia.org/wiki/Line_graph) with probabilities on nodes.

    Given a graph G, its line graph L(G) is a graph such that
    (1) each vertex of L(G) represents an edge of G; and
    (2) two vertices of L(G) are adjacent if and only if their corresponding
    edges share a common endpoint ("are incident") in G.

    --------------------
    Eg1

    (A) -.8- (B) -.8- (C)
    |                  |
    |_____.01__________|

       (.8)-------(.8)
         \        /
          \      /
          (.01)_/
    --------------------
    Eg2

    (A) -.8- (B) -|
    |  \      |   |
   .8   .01  .8   |
    |       \ |   |
    (C) -.8- (D)  |
    |             |
    |______.8_____|

  ______________
 |              |
 |       ______(AB,.8)_____
 |      /        |         \
 |  (AC,.8) -- (AD,.01) -- (BD,.8)
 |     |  \     /         / |
 |     |   (CD,.8)_______/  |
 |     |  /                 |
 +-----(CB,.8)--------------+


 >>> import networkx as nx
 >>> import plottool as pt
 >>> edges = [(u, v, {'weight': w}) for (u, v, w) in [
 >>>     ('a', 'b', .8),
 >>>     ('a', 'c', .8),
 >>>     ('a', 'd', .01),
 >>>     ('b', 'c', .8),
 >>>     ('b', 'd', .8),
 >>>     ('c', 'd', .8),
 >>> ]]
 >>> G = nx.Graph()
 >>> G.add_edges_from(edges)
 >>> node_to_label = {e: '%s,%s\n%s' % (e + (d,)) for e, d in nx.get_edge_attributes(G, 'weight').items()}
 >>> nx.set_edge_attributes(G, 'label', node_to_label)
 >>> nx.set_node_attributes(G, 'size', (10, 10))
 >>> pt.nx_agraph_layout(G, inplace=True, prog='neato')
 >>> # Create Line Graph of G (propogating information)
 >>> L = nx.line_graph(G)
 >>> for _key in [('weight',), ('label',)]:
 >>>     edge_to_attr = {tuple(sorted(e)): v for e, v in nx.get_edge_attributes(G, _key[0]).items()}
 >>>     nx.set_node_attributes(L, _key[-1], edge_to_attr)
 >>> for _key in [('lp', 'pos')]:
 >>>     edge_to_attr = {tuple(sorted(e)): v * 2 for e, v in nx.get_edge_attributes(G, _key[0]).items()}
 >>>     nx.set_node_attributes(L, _key[-1], edge_to_attr)
 >>> nx.set_node_attributes(L, 'pin', 'true')
 >>> print(G.edge['a']['c']['lp'])
 >>> print(L.node[('a', 'c')]['pos'])
 >>> pt.nx_agraph_layout(L, inplace=True, prog='neato')
 >>> pt.qt4ensure()
 >>> fnum = 1
 >>> _ = pt.show_nx(G, pnum=(1, 2, 1), fnum=fnum, layout='custom')
 >>> _ = pt.show_nx(L, pnum=(1, 2, 2), fnum=fnum, layout='custom')
 >>> print(L.node[('a', 'c')]['pos'])
 >>> # NOW WE RUN LOOPY BELIEF PROP WITH OPENGM
 >>> # FIRST IMPORT INTO OPENGM THEN MAKE FACTOR GRAPH THEN EXECUTE
 >>> # http://trgao10.github.io/bglbp.html

import opengm
index_type = opengm.index_type
numVar = len(L.nodes())
numLabels = 2
beta = 0.75
numberOfStates = np.ones(numVar, dtype=index_type) * numLabels

nodex_lookup = ut.make_index_lookup(L.nodes())
nodes = np.array(ut.take(nodex_lookup, L.nodes()))
edges = np.array([ut.take(nodex_lookup, e) for e in L.edges()])
weights = np.array(ut.take(nx.get_node_attributes(L, 'weight'), L.nodes()))
# For each node write the probability of it takeing a certain state (same/diff)
unaries = np.vstack([weights, 1 - weights]).T

gm = opengm.graphicalModel(numberOfStates, operator="multiplier")
for r in range(unaries.shape[0]):
    fid = gm.addFunction(unaries[r])
    gm.addFactor(fid, r)

for c in range(edges.shape[0]):
    fid = gm.addFunction(np.array([[beta, 1 - beta], [1 - beta, beta]]))
    variableIndices = np.array(edges[c], dtype=index_type)
    gm.addFactor(fid, variableIndices)

opengm.visualizeGm(gm, show=False, layout="neato", plotUnaries=True,
                    iterations=1000, plotFunctions=False,
                    plotNonShared= False, relNodeSize=1.0)

lpb_parmas = opengm.InfParam(damping=0.01,steps=1000)
infr = opengm.inference.BeliefPropagation(gm, parameter=lpb_parmas,
                                                    accumulator="integrator")
infr.infer()

marginals = infr.marginals(nodes)
import pandas as pd
print('new marginals are')
print(pd.DataFrame(marginals, columns=['same', 'diff'], index=pd.Series(nodes)))


# -------------

import networkx as nx
import plottool as pt
edges = [(u, v, {'weight': w}) for (u, v, w) in [
    ('a', 'b', .8),
    ('a', 'c', .8),
    ('a', 'd', .000001),
    ('b', 'c', .8),
    ('b', 'd', .4),
    ('c', 'd', .8),
]]
G = nx.Graph()
G.add_edges_from(edges)
node_to_label = {e: '%s,%s\n%s' % (e + (d,)) for e, d in nx.get_edge_attributes(G, 'weight').items()}
nx.set_edge_attributes(G, 'label', node_to_label)
# G.remove_edge('a', 'c')

import opengm
index_type = opengm.index_type
nodes = list(G.nodes())
edges = list(G.edges())
n_annots = len(nodes)
n_edges = len(edges)
n_names = n_annots
n_edge_states = 2

node_state_card = np.ones(n_annots, dtype=index_type) * n_names
edge_state_card = np.ones(n_edges, dtype=index_type) * n_edge_states
numberOfStates = node_state_card
# numberOfStates = np.hstack([node_state_card, edge_state_card])
gm = opengm.graphicalModel(numberOfStates, operator="multiplier")

annot_idxs = list(range(n_annots))
edge_idxs = list(range(n_annots, n_annots + n_edges))
lookup_annot_idx = ut.dzip(nodes, annot_idxs)
lookup_edge_idx = ut.dzip(edges, edge_idxs)

unaries = np.ones((n_annots, n_names)) / n_names
# unaries[0][0] = 1
# unaries[0][1:] = 0
for annot_idx in annot_idxs:
    fid = gm.addFunction(unaries[annot_idx])
    gm.addFactor(fid, annot_idx)

# Add Pots function for each edge
pairwise_factor_idxs = []
count = len(list(gm.factors()))
for aid1, aid2 in edges:
    varx1 = lookup_annot_idx[aid1]
    varx2 = lookup_annot_idx[aid2]
    pairwise_factor_idxs.append(count)
    count += 1
    p_match = G.get_edge_data(aid1, aid2)['weight']
    p_noncomp = 0
    B = 1 / n_names
    prob_same = p_match + p_noncomp * B
    prob_diff = 1 - prob_same

    potts_func = opengm.PottsFunction((n_names, n_names), valueEqual=prob_same, valueNotEqual=prob_diff)
    potts_func_id = gm.addFunction(potts_func)
    var_indicies = np.array([varx1, varx2])
    gm.addFactor(potts_func_id, var_indicies)

lpb_parmas = opengm.InfParam(damping=0.01,steps=1000)
infr = opengm.inference.BeliefPropagation(gm, parameter=lpb_parmas, accumulator="integrator")
infr.infer()

import pandas as pd

factors = list(gm.factors())
factor_marginals = infr.factorMarginals(pairwise_factor_idxs)
edge_marginals_same_diff = [(np.diag(f).sum(), f[~np.eye(f.shape[0],dtype=bool)].sum()) for f in factor_marginals]
print(pd.DataFrame(edge_marginals_same_diff, columns=['same', 'diff'], index=pd.Series(edges)))

marginals = infr.marginals(annot_idxs)
print('node marginals are')
print(pd.DataFrame(marginals, index=pd.Series(nodes)))


import opengm
index_type = opengm.index_type
nodes = list(G.nodes())
n_annots = len(nodes)
n_names = n_annots
lookup_annot_idx = ut.dzip(nodes, annot_idxs)

# Create nodes in the graphical model.  In this case there are <num_vars>
# nodes and each node can be assigned to one of <num_vars> possible labels
space = np.full((n_annots,), fill_value=n_names, dtype=opengm.index_type)
gm = opengm.gm(space, operator='adder')

# Use one potts function for each edge
# Add Pots function for each edge
pairwise_factor_idxs = []
count = len(list(gm.factors()))
for (p_same, p_diff), (aid1, aid2) in zip(edge_marginals_same_diff, G.edges()):
    varx1 = lookup_annot_idx[aid1]
    varx2 = lookup_annot_idx[aid2]
    valueEqual = 0
    valueNotEqual = vt.logit(p_same)
    potts_func = opengm.PottsFunction((n_names, n_names), valueEqual=valueEqual, valueNotEqual=valueNotEqual)
    potts_func_id = gm.addFunction(potts_func)
    var_indicies = np.array([varx1, varx2])
    gm.addFactor(potts_func_id, var_indicies)

parameter = opengm.InfParam()
infr = opengm.inference.Multicut(gm, parameter=parameter)
infr.infer()
labels = infr.arg()
print('labels = %r' % (labels,))


#----

import networkx
from networkx.drawing.nx_agraph import graphviz_layout
networkx.graphviz_layout = graphviz_layout
opengm.visualizeGm(gm, show=False, layout="neato", plotUnaries=True,
                    iterations=1000, plotFunctions=False,
                    plotNonShared= False, relNodeSize=1.0)
_ = pt.show_nx(G)

# marginals = infr.marginals(edge_idxs)
# print('edge marginals are')
# print(marginals)
# print(pd.DataFrame(marginals, columns=['same', 'diff'], index=pd.Series(edges)))

# for edge_idx in edge_idxs:
#     p_match = G.get_edge_data(aid1, aid2)['weight']
#     fid = gm.addFunction(np.array([.5, .5]))
#     gm.addFactor(fid, edge_idx)

# edge_idx = lookup_edge_idx[(aid1, aid2)]
# var_indicies = np.array([varx1, varx2, edge_idx])
# expl_shape = (n_names, n_names, n_edge_states)
# values = np.zeros(expl_shape)
# for s in range(n_edge_states):
#     for i in range(n_names):
#         for j in range(n_names):
#             if s == 0 and i == j:
#                 values[i, j, s] = prob_same
#             if s == 1 and i == j:
#                 values[i, j, s] = 0
#             if s == 0 and i != j:
#                 values[i, j, s] = 0
#             if s == 1 and i != j:
#                 values[i, j, s] = prob_diff
# fid = gm.addFunction(values)
# gm.addFactor(fid, var_indicies)


 >>> import networkx as nx
 >>> main_edges = [
 >>>     ('a', 'b', .8),
 >>>     ('a', 'c', .8),
 >>>     ('a', 'd', .01),
 >>>     ('b', 'c', .8),
 >>>     ('b', 'd', .8),
 >>>     ('c', 'd', .8),
 >>> ]
 >>> edges = ut.flatten([[(u, (u, v)), ((u, v), v)] for u, v, d in main_edges])
 >>> G = nx.Graph()
 >>> G.add_edges_from(edges)
 >>> node_to_label = {n: ut.repr2(n) for n in G.nodes()}
 >>> nx.set_node_attributes(G, 'label', node_to_label)
 >>> pt.show_nx(G, prog='neato')
 # Ok, this looks better. This is a bipartite graph.
 # Each (annot node) will take a state from 0-N where N is the number of annot nodes.
 # Each (match node) will take a state that is 0 or 1.
 # There is no penalty for a name node to take a name
 # A penalty exists for a match node.
 # if the probs are {'match': .1, 'nomatch', .3, 'noncomp' .6}
 # The penalty is nomatch if the labels on each end are different and match if
 # they are the same note that if noncomp has full probability there is no
 # penalty for the labels being the same or different.
 # phi_{i,j}(s) = p['nomatch'] if a[i] == a[j] else p['match']
 # This seems to reduce to multicut pretty nicely




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
