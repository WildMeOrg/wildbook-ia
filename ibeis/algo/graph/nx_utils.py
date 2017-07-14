# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import networkx as nx
import itertools as it
import vtool as vt  # NOQA
from collections import defaultdict
print, rrr, profile = ut.inject2(__name__)


def _dz(a, b):
    a = a.tolist() if isinstance(a, np.ndarray) else list(a)
    b = b.tolist() if isinstance(b, np.ndarray) else list(b)
    return ut.dzip(a, b)


def diag_product(s1, s2):
    """ Does product, but iterates over the diagonal first """
    s1 = list(s1)
    s2 = list(s2)
    if len(s1) > len(s2):
        for _ in range(len(s1)):
            for a, b in zip(s1, s2):
                yield (a, b)
            s1 = ut.list_roll(s1, 1)
    else:
        for _ in range(len(s2)):
            for a, b in zip(s1, s2):
                yield (a, b)
            s2 = ut.list_roll(s2, 1)


def e_(u, v):
    return (u, v) if u < v else (v, u)


def edges_inside(graph, nodes):
    """
    Finds edges within a set of nodes
    Running time is O(len(nodes) ** 2)

    Args:
        graph (nx.Graph): an undirected graph
        nodes1 (set): a set of nodes
    """
    result = set([])
    upper = nodes.copy()
    graph_adj = graph.adj
    for u in nodes:
        for v in upper.intersection(graph_adj[u]):
            result.add(e_(u, v))
        upper.remove(u)
    return result


def edges_outgoing(graph, nodes):
    """
    Finds edges leaving a set of nodes.
    Average running time is O(len(nodes) * ave_degree(nodes))
    Worst case running time is O(G.number_of_edges()).

    Args:
        graph (nx.Graph): a graph
        nodes (set): set of nodes

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> import utool as ut
        >>> G = demodata_bridge()
        >>> bridges = find_bridges(G)
        >>> nodes = {1, 2, 3, 4}
        >>> outgoing = edges_outgoing(G, nodes)
        >>> assert outgoing == {(3, 5), (4, 8)}
    """
    if not isinstance(nodes, set):
        nodes = set(nodes)
    return {e_(u, v) for u in nodes for v in graph.adj[u] if v not in nodes}


def edges_cross(graph, nodes1, nodes2):
    """
    Finds edges between two sets of disjoint nodes.
    Running time is O(len(nodes1) * len(nodes2))

    Args:
        graph (nx.Graph): an undirected graph
        nodes1 (set): set of nodes disjoint from `nodes2`
        nodes2 (set): set of nodes disjoint from `nodes1`.
    """
    return {e_(u, v) for u in nodes1
            for v in nodes2.intersection(graph.adj[u])}


def edges_between(graph, nodes1, nodes2=None, assume_disjoint=False,
                  assume_dense=True):
    r"""
    Get edges between two components or within a single component

    Args:
        graph (nx.Graph): the graph
        nodes1 (set): list of nodes
        nodes2 (set): if None it is equivlanet to nodes2=nodes1 (default=None)
        assume_disjoint (bool): skips expensive check to ensure edges arnt
            returned twice (default=False)

    CommandLine:
        python -m ibeis.algo.graph.nx_utils --test-edges_between

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> import utool as ut
        >>> edges = [
        >>>     (1, 2), (2, 3), (3, 4), (4, 1), (4, 3),  # cc 1234
        >>>     (1, 5), (7, 2), (5, 1),  # cc 567 / 5678
        >>>     (7, 5), (5, 6), (8, 7),
        >>> ]
        >>> digraph = nx.DiGraph(edges)
        >>> graph = nx.Graph(edges)
        >>> nodes1 = [1, 2, 3, 4]
        >>> nodes2 = [5, 6, 7]
        >>> n2 = sorted(edges_between(graph, nodes1, nodes2))
        >>> n4 = sorted(edges_between(graph, nodes1))
        >>> n5 = sorted(edges_between(graph, nodes1, nodes1))
        >>> n1 = sorted(edges_between(digraph, nodes1, nodes2))
        >>> n3 = sorted(edges_between(digraph, nodes1))
        >>> print('n2 == %r' % (n2,))
        >>> print('n4 == %r' % (n4,))
        >>> print('n5 == %r' % (n5,))
        >>> print('n1 == %r' % (n1,))
        >>> print('n3 == %r' % (n3,))
        >>> assert n2 == ([(1, 5), (2, 7)]), '2'
        >>> assert n4 == ([(1, 2), (1, 4), (2, 3), (3, 4)]), '4'
        >>> assert n5 == ([(1, 2), (1, 4), (2, 3), (3, 4)]), '5'
        >>> assert n1 == ([(1, 5), (5, 1), (7, 2)]), '1'
        >>> assert n3 == ([(1, 2), (2, 3), (3, 4), (4, 1), (4, 3)]), '3'
        >>> n6 = sorted(edges_between(digraph, nodes1 + [6], nodes2 + [1, 2], assume_dense=False))
        >>> print('n6 = %r' % (n6,))
        >>> n6 = sorted(edges_between(digraph, nodes1 + [6], nodes2 + [1, 2], assume_dense=True))
        >>> print('n6 = %r' % (n6,))
        >>> assert n6 == ([(1, 2), (1, 5), (2, 3), (4, 1), (5, 1), (5, 6), (7, 2)]), '6'
    """
    if assume_dense:
        edges = _edges_between_dense(graph, nodes1, nodes2, assume_disjoint)
    else:
        edges = _edges_between_sparse(graph, nodes1, nodes2, assume_disjoint)
    if graph.is_directed():
        for u, v in edges:
            yield u, v
    else:
        for u, v in edges:
            yield e_(u, v)


def _edges_between_dense(graph, nodes1, nodes2=None, assume_disjoint=False):
    """
    The dense method is where we enumerate all possible edges and just take the
    ones that exist (faster for very dense graphs)
    """
    if nodes2 is None or nodes2 is nodes1:
        # Case where we are looking at internal nodes only
        edge_iter = it.combinations(nodes1, 2)
    elif assume_disjoint:
        # We assume len(isect(nodes1, nodes2)) == 0
        edge_iter = it.product(nodes1, nodes2)
    else:
        # make sure a single edge is not returned twice
        # in the case where len(isect(nodes1, nodes2)) > 0
        if not isinstance(nodes1, set):
            nodes1 = set(nodes1)
        if not isinstance(nodes2, set):
            nodes2 = set(nodes2)
        nodes_isect = nodes1.intersection(nodes2)
        nodes_only1 = nodes1 - nodes_isect
        nodes_only2 = nodes2 - nodes_isect
        edge_sets = [it.product(nodes_only1, nodes_only2),
                     it.product(nodes_only1, nodes_isect),
                     it.product(nodes_only2, nodes_isect),
                     it.combinations(nodes_isect, 2)]
        edge_iter = it.chain.from_iterable(edge_sets)

    if graph.is_directed():
        for n1, n2 in edge_iter:
            if graph.has_edge(n1, n2):
                yield n1, n2
            if graph.has_edge(n2, n1):
                yield n2, n1
    else:
        for n1, n2 in edge_iter:
            if graph.has_edge(n1, n2):
                yield n1, n2


def _edges_inside_lower(graph, both_adj):
    """ finds lower triangular edges inside the nodes """
    both_lower = set([])
    for u, neighbs in both_adj.items():
        neighbsBB_lower = neighbs.intersection(both_lower)
        for v in neighbsBB_lower:
            yield (u, v)
        both_lower.add(u)


def _edges_inside_upper(graph, both_adj):
    """ finds upper triangular edges inside the nodes """
    both_upper = set(both_adj.keys())
    for u, neighbs in both_adj.items():
        neighbsBB_upper = neighbs.intersection(both_upper)
        for v in neighbsBB_upper:
            yield (u, v)
        both_upper.remove(u)


def _edges_between_disjoint(graph, only1_adj, only2):
    """ finds edges between disjoint nodes """
    for u, neighbs in only1_adj.items():
        # Find the neighbors of u in only1 that are also in only2
        neighbs12 = neighbs.intersection(only2)
        for v in neighbs12:
            yield (u, v)


def _edges_between_sparse(graph, nodes1, nodes2=None, assume_disjoint=False):
    """
    In this version we check the intersection of existing edges and the edges
    in the second set (faster for sparse graphs)
    """
    # Notes:
    # 1 = edges only in `nodes1`
    # 2 = edges only in `nodes2`
    # B = edges only in both `nodes1` and `nodes2`

    # Test for special cases
    if nodes2 is None or nodes2 is nodes1:
        # Case where we just are finding internal edges
        both = set(nodes1)
        both_adj  = {u: set(graph.adj[u]) for u in both}
        if graph.is_directed():
            edge_sets = (
                _edges_inside_upper(graph, both_adj),  # B-to-B (u)
                _edges_inside_lower(graph, both_adj),  # B-to-B (l)
            )
        else:
            edge_sets = (
                _edges_inside_upper(graph, both_adj),  # B-to-B (u)
            )
    elif assume_disjoint:
        # Case where we find edges between disjoint sets
        if not isinstance(nodes1, set):
            nodes1 = set(nodes1)
        if not isinstance(nodes2, set):
            nodes2 = set(nodes2)
        only1 = nodes1
        only2 = nodes2
        if graph.is_directed():
            only1_adj = {u: set(graph.adj[u]) for u in only1}
            only2_adj = {u: set(graph.adj[u]) for u in only2}
            edge_sets = (
                _edges_between_disjoint(graph, only1, only2),  # 1-to-2
                _edges_between_disjoint(graph, only2, only1),  # 2-to-1
            )
        else:
            only1_adj = {u: set(graph.adj[u]) for u in only1}
            edge_sets = (
                _edges_between_disjoint(graph, only1, only2),  # 1-to-2
            )
    else:
        # Full general case
        if not isinstance(nodes1, set):
            nodes1 = set(nodes1)
        if nodes2 is None:
            nodes2 = nodes1
        elif not isinstance(nodes2, set):
            nodes2 = set(nodes2)
        both = nodes1.intersection(nodes2)
        only1 = nodes1 - both
        only2 = nodes2 - both

        # Precompute all calls to set(graph.adj[u]) to avoid duplicate calls
        only1_adj = {u: set(graph.adj[u]) for u in only1}
        only2_adj = {u: set(graph.adj[u]) for u in only2}
        both_adj  = {u: set(graph.adj[u]) for u in both}
        if graph.is_directed():
            edge_sets = (
                _edges_between_disjoint(graph, only1_adj, only2),  # 1-to-2
                _edges_between_disjoint(graph, only1_adj, both),   # 1-to-B
                _edges_inside_upper(graph, both_adj),              # B-to-B (u)
                _edges_inside_lower(graph, both_adj),              # B-to-B (l)
                _edges_between_disjoint(graph, both_adj, only1),   # B-to-1
                _edges_between_disjoint(graph, both_adj, only2),   # B-to-2
                _edges_between_disjoint(graph, only2_adj, both),   # 2-to-B
                _edges_between_disjoint(graph, only2_adj, only1),  # 2-to-1
            )
        else:
            edge_sets = (
                _edges_between_disjoint(graph, only1_adj, only2),  # 1-to-2
                _edges_between_disjoint(graph, only1_adj, both),   # 1-to-B
                _edges_inside_upper(graph, both_adj),              # B-to-B (u)
                _edges_between_disjoint(graph, only2_adj, both),   # 2-to-B
            )

    for u, v in it.chain.from_iterable(edge_sets):
        yield u, v


def group_name_edges(g, node_to_label):
    ne_to_edges = defaultdict(set)
    for u, v in g.edges():
        name_edge = e_(node_to_label[u], node_to_label[v])
        ne_to_edges[name_edge].add(e_(u, v))
    return ne_to_edges


def ensure_multi_index(index, names):
    import pandas as pd
    if not isinstance(index, (pd.MultiIndex, pd.Index)):
        names = ('aid1', 'aid2')
        if len(index) == 0:
            index = pd.MultiIndex([[], []], [[], []], names=names)
        else:
            index = pd.MultiIndex.from_tuples(index, names=names)
    return index


def demodata_bridge():
    # define 2-connected compoments and bridges
    cc2 = [(1, 2, 4, 3, 1, 4), (8, 9, 10, 8), (11, 12, 13, 11)]
    bridges = [(4, 8), (3, 5), (20, 21), (22, 23, 24)]
    G = nx.Graph(ut.flatten(ut.itertwo(path) for path in cc2 + bridges))
    return G


def demodata_tarjan_bridge():
    """
    CommandLine:
        python -m ibeis.algo.graph.nx_utils demodata_tarjan_bridge --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = demodata_tarjan_bridge()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.show_nx(G)
        >>> ut.show_if_requested()
    """
    # define 2-connected compoments and bridges
    cc2 = [(1, 2, 4, 3, 1, 4), (5, 6, 7, 5), (8, 9, 10, 8),
             (17, 18, 16, 15, 17), (11, 12, 14, 13, 11, 14)]
    bridges = [(4, 8), (3, 5), (3, 17)]
    G = nx.Graph(ut.flatten(ut.itertwo(path) for path in cc2 + bridges))
    return G


# def is_tri_edge_connected(G):
#     """
#     Yet another Simple Algorithm for Triconnectivity
#     http://www.sciencedirect.com/science/article/pii/S1570866708000415
#     """
#     pass


@profile
def is_edge_connected(G, k):
    """
    Determines if G is k-edge-connected

    References:
        https://arxiv.org/pdf/1211.6553.pdf
        https://github.com/adrianN/edge-connectivity

        Reducing edge connectivity to vertex connectivity
        Yet another optimal algorithm for 3-edge-connectivity
        A simple 3-edge-connected component algorithm.
        A linear time algorithm for computing all 3-edgeconnected components of a multigraph.
        Finding triconnected components of graphs

        A Simple Algorithm for Triconnectivity of a Multigraph
           * reduces 3-edge-connectivity to 3-vertex-connectivity

    """
    if k < 0:
        raise ValueError('k must be an integer greater than 0, not=%r' % (k,))
    elif k == 0:
        return True
    elif k == 1:
        return nx.is_connected(G)
    elif k == 2:
        return is_bridge_connected(G)
    else:
        if any(d < k for n, d in G.degree()):
            # quick short circuit for false cases
            return False
        return nx.edge_connectivity(G) >= k


def is_bridge_connected(G):
    """
    Example:
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> from ibeis.algo.graph import demo
        >>> G1 = nx.Graph()
        >>> nx.add_path(G1, [1, 2, 3, 4, 5])
        >>> G2 = G1.copy()
        >>> G2.add_edge(1, 5)
        >>> assert not is_bridge_connected(G1)
        >>> assert is_bridge_connected(G2)
    """
    # Check that G has edges and none of them are bridges
    return not any(find_bridges(G)) and any(G.edges())


@profile
def find_bridges(G):
    """
    Returns all bridge edges. A bridge edge is any edge that, if removed, would
    diconnect a compoment in G.

    Notes:
        Bridges can be found using chain decomposition.  An edge e in G is a
        bridge if and only if e is not contained in any chain.

    References:
        https://en.wikipedia.org/wiki/Bridge_(graph_theory)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = demodata_bridge()
        >>> bridges = set(find_bridges(G))
        >>> assert bridges == {(3, 5), (4, 8), (20, 21), (22, 23), (23, 24)}
        >>> import plottool as pt
        >>> ut.quit_if_noshow()
        >>> pt.qtensure()
        >>> pt.show_nx(G)
        >>> ut.show_if_requested()
    """
    # It is just faster to do this
    chains = nx.chain_decomposition(G)
    chain_edges = set(it.starmap(e_, it.chain.from_iterable(chains)))
    bridges = set(it.starmap(e_, G.edges())) - chain_edges
    # Taken partially from nx.chain_decomposition
    # We are concerned with the edges not part of any cycle
    # def _dfs_cycle_forest(G, root):
    #     H = nx.DiGraph()
    #     nodes = []
    #     for u, v, d in nx.dfs_labeled_edges(G, source=root):
    #         if d == 'forward':
    #             nodes.append(v)
    #             if u == v:
    #                 H.add_node(v, parent=None)
    #             else:
    #                 H.add_node(v, parent=u)
    #                 H.add_edge(v, u, nontree=False)
    #         elif d == 'nontree' and v not in H[u]:
    #             H.add_edge(v, u, nontree=True)
    #     return H, nodes

    # def _remove_chains(H, u, v, visited):
    #     while v not in visited:
    #         H.remove_edge(u, v)
    #         visited.add(v)
    #         u, v = v, H.node[v]['parent']
    #     H.remove_edge(u, v)

    # H, nodes = _dfs_cycle_forest(G, None)

    # # Retrace the DFS tree and remove the edges in each cycle
    # visited = set()
    # for u in nodes:
    #     visited.add(u)
    #     edges = [(u, v) for u, v, d in H.out_edges(u, data='nontree') if d]
    #     for u, v in edges:
    #         _remove_chains(H, u, v, visited)

    # if G.is_directed():
    #     bridges = H.edges()
    # else:
    #     bridges = it.starmap(e_, H.edges())
    #     if True:
    return bridges


def bridge_connected_compoments(G):
    """
    Also referred to as blocks

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = demodata_bridge()
        >>> bridge_ccs = bridge_connected_compoments(G)
        >>> assert bridge_ccs == [
        >>>     {1, 2, 3, 4}, {5}, {8, 9, 10}, {11, 12, 13}, {20},
        >>>     {21}, {22}, {23}, {24}
        >>> ]
    """
    bridges = find_bridges(G)
    H = G.copy()
    H.remove_edges_from(bridges)
    return list(nx.connected_components(H))


@profile
def one_connected_augmentation(G, avail=None, weight='weight'):
    """
    Finds minimum weight set of edges to connect G.

    Args:
        G (nx.Graph): graph to make connected
        avail (list): available edges, if None nx.complement(G) is assumed.
            if each item is a (u, v), the problem is unweighted.
            if each item is a (u, v, d), the problem is weighted.  with
            d[weight] corresponding to the weight.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = nx.Graph()
        >>> G.add_nodes_from([
        >>>     1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> G.add_edges_from([(3, 8), (1, 2), (2, 3)])
        >>> impossible = {(6, 3), (3, 9)}
        >>> rng = np.random.RandomState(0)
        >>> avail = list(set(complement_edges(G)) - impossible)
        >>> avail_uvd = [(u, v, {'weight': rng.rand()}) for u, v in avail]
        >>> aug_edges1 = list(one_connected_augmentation(G))
        >>> aug_edges2 = list(one_connected_augmentation(G, avail))
        >>> aug_edges3 = list(one_connected_augmentation(G, avail_uvd))
    """
    ccs1 = list(nx.connected_components(G))
    C = collapse(G, ccs1)
    mapping = C.graph['mapping']

    if avail is not None:
        avail_uv = [tup[0:2] for tup in avail]
        # Each edge gets a weight that can be specified, or it defaults to 1
        avail_w = [1 if len(tup) == 2 else tup[-1][weight] for tup in avail]
        meta_avail_uv = [(mapping[u], mapping[v]) for u, v in avail_uv]

        # only need exactly 1 edge at most between each CC, so choose lightest
        avail_ew = zip(avail_uv, avail_w)
        grouped_we = ut.group_items(avail_ew, meta_avail_uv)
        candidates = []
        for meta_edge, choices in grouped_we.items():
            edge, w = min(choices, key=lambda t: t[1])
            candidates.append((meta_edge, edge, w))
        candidates = sorted(candidates, key=lambda t: t[2])

        # kruskals algorithm on metagraph to find the best connecting edges
        subtrees = nx.utils.UnionFind()
        for (mu, mv), (u, v), w in candidates:
            if subtrees[mu] != subtrees[mv]:
                yield (u, v)
            subtrees.union(mu, mv)
    else:
        # When we are not constrained, we can just make a meta graph tree.
        meta_nodes = list(C.nodes())
        # build a path in the metagraph
        meta_aug = list(zip(meta_nodes, meta_nodes[1:]))
        # map that path to the original graph
        inverse = ut.group_pairs(C.graph['mapping'].items())
        for mu, mv in meta_aug:
            yield (inverse[mu][0], inverse[mv][0])


def complement_edges(G):
    return ((n, n2) for n, nbrs in G.adjacency()
            for n2 in G if n2 not in nbrs if n != n2)


def greedy_edge_augmentation(G, k, avail=None, return_anyway=False):
    """
    Randomly add edges that are not locally k-edge-connected until we satisfy
    connectivity in general. Then try and remove edges to reduce the size.

    Ignore:
        >>> ut.qtensure()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> aug_edges1 = greedy_edge_augmentation(G, k=1)
        >>> aug_edges2 = greedy_edge_augmentation(G, k=2)
        >>> aug_edges3 = greedy_edge_augmentation(G, k=3)
        >>> aug_edges4 = greedy_edge_augmentation(G, k=4)
        >>> ut.quit_if_noshow()
        >>> len(aug_edges_opt)
        >>> len(aug_edges_greedy)
        >>> import plottool as pt
        >>> def aug_graph(G, edges):
        >>>     H = G.copy()
        >>>     H.add_edges_from(edges)
        >>>     return H
        >>> pnum_ = pt.make_pnum_nextgen(nRows=2, nCols=2)
        >>> fnum = 1
        >>> pt.show_nx(aug_graph(G, aug_edges1), fnum=fnum, pnum=pnum_())
        >>> pt.show_nx(aug_graph(G, aug_edges2), fnum=fnum, pnum=pnum_())
        >>> pt.show_nx(aug_graph(G, aug_edges3), fnum=fnum, pnum=pnum_())
        >>> pt.show_nx(aug_graph(G, aug_edges4), fnum=fnum, pnum=pnum_())

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3])
        >>> aug_edges1 = greedy_edge_augmentation(G, k=1, return_anyway=True)
        >>> aug_edges2 = greedy_edge_augmentation(G, k=2, return_anyway=True)
        >>> aug_edges3 = greedy_edge_augmentation(G, k=3, return_anyway=True)
        >>> aug_edges4 = greedy_edge_augmentation(G, k=4, return_anyway=True)
        >>> ut.quit_if_noshow()
        >>> len(aug_edges_opt)
        >>> len(aug_edges_greedy)
        >>> import plottool as pt
        >>> def aug_graph(G, edges):
        >>>     H = G.copy()
        >>>     H.add_edges_from(edges)
        >>>     return H
        >>> pnum_ = pt.make_pnum_nextgen(nRows=2, nCols=2)
        >>> fnum = 1
        >>> pt.show_nx(aug_graph(G, aug_edges1), fnum=fnum, pnum=pnum_())
        >>> pt.show_nx(aug_graph(G, aug_edges2), fnum=fnum, pnum=pnum_())
        >>> pt.show_nx(aug_graph(G, aug_edges3), fnum=fnum, pnum=pnum_())
        >>> pt.show_nx(aug_graph(G, aug_edges4), fnum=fnum, pnum=pnum_())
    """
    # Because I have not implemented a better algorithm yet:
    # randomly add edges until we satisfy the criteria
    import random
    # very hacky and not minimal
    done = is_edge_connected(G, k)
    if done:
        return []
    if avail is None:
        avail = list(complement_edges(G))
    else:
        avail = list(avail)
    aug_edges = []
    rng = random.Random(0)
    avail = list({e_(u, v) for u, v in avail})
    avail = ut.shuffle(avail, rng=rng)
    H = G.copy()
    # Randomly throw edges in until we are k-connected
    for edge in avail:
        local_k = nx.connectivity.local_edge_connectivity(H, *edge)
        if local_k < k:
            aug_edges.append(edge)
            H.add_edge(*edge)
        done = is_edge_connected(H, k)
        if done:
            break
    if not done:
        if not return_anyway:
            raise ValueError('not able to k-connect with available nodes')
        else:
            return avail
    aug_edges = ut.shuffle(aug_edges, rng=rng)
    # Greedy attempt to reduce the size
    for edge in list(aug_edges):
        if min(H.degree(edge), key=lambda t: t[1])[1] <= k:
            continue
        H.remove_edge(*edge)
        aug_edges.remove(edge)
        conn = nx.edge_connectivity(H)
        if conn < k:
            # If no longer feasible undo
            H.add_edge(*edge)
            aug_edges.append(edge)
    return aug_edges


@profile
def edge_connected_augmentation(G, k, avail=None, hack=False,
                                return_anyway=False):
    r"""
    Finds set of edges to k-edge-connect G. In the case of k=1
    this is a minimum weight set. For k=2 it becomes exact only if avail is
    None. If k > 2, a greedy algorithm is used.

    Args:
        G (nx.Graph): graph to augment
        k (int): desired edge connectivity
        avail (set): set of edges that can be used for the augmentation
           each item is either a 2 tuple of vertices or a 3 tuple
           of vertices and a dictionary containing a weight key.

    CommandLine:
        python -m ibeis.algo.graph.nx_utils edge_connected_augmentation

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3, 4, 5, 6])
        >>> k = 4
        >>> aug_edges = edge_connected_augmentation(G, k)
        >>> G.add_edges_from(aug_edges)
        >>> print(nx.edge_connectivity(G))
        >>> import plottool as pt
        >>> ut.quit_if_noshow()
        >>> pt.qtensure()
        >>> pt.show_nx(G)
        >>> ut.show_if_requested()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> G = nx.Graph()
        >>> G.add_nodes_from([
        >>>     1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> G.add_edges_from([(3, 8)])
        >>> impossible = {(6, 3), (3, 9)}
        >>> avail = list(set(complement_edges(G)) - impossible)
        >>> aug_edges = edge_connected_augmentation(G, k=1)
        >>> aug_edges = edge_connected_augmentation(G, 1, avail)
    """
    if avail is not None and len(avail) == 0:
        return []
    if G.number_of_nodes() < k + 1:
        if return_anyway:
            if avail is None:
                avail = list(complement_edges(G))
            else:
                avail = list(avail)
            return avail
        raise ValueError(
            ('impossible to {} connect in graph with less than {} '
             'verticies').format(k, k + 1))
    # if is_edge_connected(G, k):
    #     aug_edges = []
    elif k == 1 and not hack:
        aug_edges = one_connected_augmentation(G, avail)
    elif k == 2 and avail is None and not hack:
        aug_edges = list(bridge_connected_augmentation(G))
    elif k == 2 and avail is not None:
        aug_edges = weighted_bridge_connected_augmentation(G, avail,
                                                           return_anyway)
    else:
        # Fallback on greedy algorithm when we have no better option
        aug_edges = greedy_edge_augmentation(G, k, avail, return_anyway)
    aug_edges = list(it.starmap(e_, aug_edges))
    return aug_edges


def weighted_one_edge_connected_augmentation(G, avail):
    """ this is the MST problem """
    G2 = G.copy()
    nx.set_edge_attributes(G2, 'weight', 0)
    G2.add_edges_from(avail, weight=1)
    mst = nx.minimum_spanning_tree(G2)
    aug_edges = [(u, v) for u, v in avail if mst.has_edge(u, v)]
    return aug_edges


def greedy_local_bridge_augment(G, avail):
    # If it is not possible, be greedy and increase local connectivity
    # Can be made better by condensing the graph
    local_edge_connectivity = nx.connectivity.local_edge_connectivity
    local_greedy_edges = []
    H = G.copy()
    for u, v in avail:
        if local_edge_connectivity(H, u, v, cutoff=2) < 2:
            local_greedy_edges.append((u, v))
            H.add_edge(u, v)
    edges = local_greedy_edges
    return edges


def weighted_bridge_connected_augmentation(G, avail, return_anyway=False):
    """
    Chooses a set of edges from avail to add to G that renders it
    2-edge-connected if such a subset exists.

    Because we are constrained by edges in avail this problem is NP-hard, and
    this function is a 2-approximation if the input graph is connected, and a
    3-approximation if it is not. Runs in O(m + nlog(n)) time

    Args:
        G (nx.Graph): input graph
        avail (set): candidate edges to choose from

    Returns:
        aug_edges (set): subset of avail chosen to augment G

    References:
        Approximation algorithms for graph augmentation
        S Khuller, R Thurimella - Journal of Algorithms, 1993
        http://www.sciencedirect.com/science/article/pii/S0196677483710102
        https://www.cs.umd.edu/class/spring2011/cmsc651/lec07.pdf

    Notes:
        G0 - currrent network.
        branching - of directed graph G rooted at r.
            Every vertex except r has indegree=1 and r has indegree=0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *
        >>> # test1
        >>> G = demodata_tarjan_bridge()
        >>> avail = [(9, 7), (8, 5), (2, 10), (6, 13), (11, 18), (1, 17), (2, 3),
        >>>          (16, 17), (18, 14), (15, 14)]
        >>> bridge_augment = weighted_bridge_connected_augmentation
        >>> ut.assert_raises(ValueError, bridge_augment, G, [])
        >>> bridge_augment(G, [(9, 7)], return_anyway=True)
        >>> aug_edges = set(bridge_augment(G, avail))
        >>> assert aug_edges == {(6, 13), (2, 10), (1, 17)}
        >>> # test2
        >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (4, 2), (2, 5)])
        >>> avail = [(4, 5)]
        >>> ut.assert_raises(ValueError, bridge_augment, G, avail)
        >>> aug_edges = bridge_augment(G, avail, True)
        >>> avail = [(3, 5), (4, 5)]
        >>> aug_edges = bridge_augment(G, avail)
    """
    # def _simple_paths(D, r, u):
    #     if r == u:
    #         paths = [[r]]
    #     else:
    #         paths = list(nx.all_simple_paths(D, r, u))
    #     return paths

    # def _least_common_ancestor(D, u, v, root=None):
    #     # Find least common ancestor in a rooted tree
    #     paths1 = _simple_paths(D, root, u)
    #     paths2 = _simple_paths(D, root, v)
    #     assert len(paths1) == 1
    #     assert len(paths2) == 1
    #     path1 = paths1[0]
    #     path2 = paths2[0]
    #     lca = None
    #     for p1, p2 in zip(path1, path2):
    #         if p1 != p2:
    #             break
    #         lca = p1
    #     return lca

    def _most_recent_descendant(D, u, v):
        # Find a closest common descendant
        assert nx.is_directed_acyclic_graph(D), 'Must be DAG'
        v_branch = nx.descendants(D, v).union({v})
        u_branch = nx.descendants(D, u).union({u})
        common = v_branch & u_branch
        node_depth = (
            ((c, (nx.shortest_path_length(D, u, c) +
                  nx.shortest_path_length(D, v, c)))
             for c in common))
        mrd = min(node_depth, key=lambda t: t[1])[0]
        return mrd

    def _lowest_common_anscestor(D, u, v):
        # Find a common ancestor furthest away
        assert nx.is_directed_acyclic_graph(D), 'Must be DAG'
        v_branch = nx.anscestors(D, v).union({v})
        u_branch = nx.anscestors(D, u).union({u})
        common = v_branch & u_branch
        node_depth = (
            ((c, (nx.shortest_path_length(D, c, u) +
                  nx.shortest_path_length(D, c, v)))
             for c in common))
        mrd = max(node_depth, key=lambda t: t[1])[0]
        return mrd

    # If input G is not connected the approximation factor increases to 3
    aug_edges = []
    if not nx.is_connected(G):
        H = G.copy()
        connectors = weighted_one_edge_connected_augmentation(H, avail)
        H.add_edges_from(connectors)
        aug_edges.extend(connectors)
    else:
        H = G
    if not nx.is_connected(H):
        if return_anyway:
            return greedy_local_bridge_augment(G, avail)
        raise ValueError('no augmentation possible')

    uv_avail = [tup[0:2] for tup in avail]
    uv_avail = [
        (u, v) for u, v in uv_avail if (
            H.has_node(u) and H.has_node(v) and not H.has_edge(u, v))
    ]

    # Collapse input into a metagraph. Meta nodes are bridge-ccs
    bridge_ccs = bridge_connected_compoments(H)
    C = collapse(H, bridge_ccs)

    # Use the meta graph to filter out a small feasible subset of avail
    # Choose the minimum weight edge from each group. TODO WEIGHTS
    mapping = C.graph['mapping']
    mapped_avail = [(mapping[u], mapping[v]) for u, v in uv_avail]
    grouped_avail = ut.group_items(uv_avail, mapped_avail)
    feasible_uv = [
        group[0] for key, group in grouped_avail.items()
        if key[0] != key[1]]
    feasible_mapped_uv = {
        e_(mapping[u], mapping[v]): e_(u, v) for u, v in feasible_uv
    }
    # feasible_mapped_uv = {
    #     muv: uv for muv, uv in _feasible_mapped_uv
    #     # if not C.has_edge(*muv)
    # }

    if len(feasible_mapped_uv) > 0:
        """
        Mapping of terms from (Khuller and Thurimella):
            C         : G^0 = (V, E^0)
            mapped_uv : E - E^0  # they group both avail and given edges in E
            T         : \Gamma
            D         : G^D = (V, E_D)

            The paper uses ancestor because children point to parents,
            in the networkx context this would be descendant.
            So, lowest_common_ancestor = most_recent_descendant

        """
        # Pick an arbitrary leaf from C as the root
        root = next(n for n in C.nodes() if C.degree(n) == 1)
        # Root C into a tree T by directing all edges towards the root
        T = nx.reverse(nx.dfs_tree(C, root))
        # Add to D the directed edges of T and set their weight to zero
        # This indicates that it costs nothing to use edges that were given.
        D = T.copy()
        nx.set_edge_attributes(D, 'weight', 0)
        # Add in feasible edges with respective weights
        for u, v in feasible_mapped_uv.keys():
            mrd = _most_recent_descendant(T, u, v)
            # print('(u, v)=({}, {})  mrd={}'.format(u, v, mrd))
            if mrd == u:
                # If u is descendant of v, then add edge u->v
                D.add_edge(mrd, v, weight=1, implicit=True)
            elif mrd == v:
                # If v is descendant of u, then add edge v->u
                D.add_edge(mrd, u, weight=1, implicit=True)
            else:
                # If neither u nor v is a descendant of the other
                # let t = mrd(u, v) and add edges t->u and t->v
                D.add_edge(mrd, u, weight=1, implicit=True)
                D.add_edge(mrd, v, weight=1, implicit=True)

        # root the graph by removing all predecessors to `root`.
        D_ = D.copy()
        D_.remove_edges_from([(u, root) for u in D.predecessors(root)])

        # Then compute a minimum rooted branching
        try:
            A = nx.minimum_spanning_arborescence(D_)
        except nx.NetworkXException:
            # If there is no arborescence then augmentation is not possible
            if not return_anyway:
                raise ValueError('There is no 2-edge-augmentation possible')
            # If it is not possible, be greedy and increase local connectivity
            local_edge_connectivity = nx.connectivity.local_edge_connectivity
            local_greedy_edges = []
            M = nx.MultiGraph(C.edges())
            for u, v in feasible_mapped_uv.keys():
                if local_edge_connectivity(M, u, v, cutoff=2) < 2:
                    local_greedy_edges.append((u, v, {}))
                    M.add_edge(u, v)
            edges = local_greedy_edges
        else:
            edges = list(A.edges(data=True))
            # edges = list(nx.minimum_branching(nx.reverse(D)).edges())

        chosen_mapped = []
        for u, v, d in edges:
            edge = e_(u, v)
            if edge in feasible_mapped_uv:
                chosen_mapped.append(edge)

        for edge in chosen_mapped:
            orig_edge = feasible_mapped_uv[edge]
            aug_edges.append(orig_edge)

    if False:
        import plottool as pt
        # C2 = C.copy()
        # C2.add_edges_from(chosen_mapped, implicit=True)
        G2 = G.copy()
        G2.add_edges_from(aug_edges, implicit=True)
        C_labels = {k: '{}:\n{}'.format(k, v)
                    for k, v in nx.get_node_attributes(C, 'members').items()}
        nx.set_node_attributes(C, 'label', C_labels)
        print('is_strongly_connected(D) = %r' % nx.is_strongly_connected(D))
        pnum_ = pt.make_pnum_nextgen(nSubplots=6)
        _ = pt.show_nx(C, arrow_width=2, fnum=1, pnum=pnum_(), title='C')
        _ = pt.show_nx(T, arrow_width=2, fnum=1, pnum=pnum_(), title='T')
        _ = pt.show_nx(D, arrow_width=2, fnum=1, pnum=pnum_(), title='D')
        _ = pt.show_nx(D_, arrow_width=2, fnum=1, pnum=pnum_(), title='D_')
        _ = pt.show_nx(A, arrow_width=2, fnum=1, pnum=pnum_(), title='A')
        # _ = pt.show_nx(G, arrow_width=2, fnum=1, pnum=pnum_(), title='G')
        _ = pt.show_nx(G2, arrow_width=2, fnum=1, pnum=pnum_(), title='G2')
        _ = _  # NOQA

    return aug_edges


def bridge_connected_augmentation(G):
    """
    References:
        http://www.openu.ac.il/home/nutov/Gilad-Thesis.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.30.2256&rep=rep1&type=pdf
        http://search.proquest.com/docview/918506142?accountid=28525
        http://epubs.siam.org/doi/abs/10.1137/0205044
        https://en.wikipedia.org/wiki/Bridge_(graph_theory)#Bridge-Finding_with_Chain_Decompositions

        https://books.google.com/books?id=yPtpCQAAQBAJ&pg=PA91&lpg=PA91&dq=bridge+augmentation+with+0+1+weights&source=bl&ots=9NwP3oRoKP&sig=UQqH8ae3yy0bSbPW5vmRLc-9xDY&hl=en&sa=X&ved=0ahUKEwjxtPP_06HTAhVMxWMKHa1HCigQ6AEILDAB#v=onepage&q=bridge%20augmentation%20with%200%201%20weights&f=false

        Approximation algorithms for graph augmentation
        S Khuller, R Thurimella - Journal of Algorithms, 1993

    Notes:
        The weighted versoin of 2-ECST (or equivalently TAP) is called
        Bridg-Connectivity Augmentation (BRA) in [7].

        2-edge connected subgraph problem containing a spanning tree.
        Tree augmentation problem.


    Notes:
        bridge-connected:
            G is bridge-connected if it is connected and contains no bridges.

        arborescence:
            An arborescence is a DAG with only one source vertex.
            IE.  The root (source) has no entering edge, and all other
            verticies have at least one entering edge.
            An arborescence is thus the directed-graph form of a rooted tree.
            Is a directed graph in which, for a vertex u called the root and
            any other vertex v, there is exactly one directed path from u to v.

        pendant / leaf:
            A vertex of a graph is said to be (pendant / leaf) if its
            neighborhood contains exactly one vertex.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *
        >>> import networkx as nx
        >>> G = demodata_tarjan_bridge()
        >>> bridge_edges = list(bridge_connected_augmentation(G))
        >>> import plottool as pt
        >>> ut.quit_if_noshow()
        >>> pt.qtensure()
        >>> pt.nx_agraph_layout(G, inplace=True, prog='neato')
        >>> nx.set_node_attributes(G, 'pin', 'true')
        >>> G2 = G.copy()
        >>> G2.add_edges_from(bridge_edges)
        >>> pt.nx_agraph_layout(G2, inplace=True, prog='neato')
        >>> pt.show_nx(G, fnum=1, pnum=(1, 2, 1), layout='custom')
        >>> pt.show_nx(G2, fnum=1, pnum=(1, 2, 2), layout='custom')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3, 4])
        >>> bridge_edges = list(bridge_connected_augmentation(G))
        >>> import plottool as pt
        >>> ut.quit_if_noshow()
        >>> pt.qtensure()
        >>> pt.nx_agraph_layout(G, inplace=True, prog='neato')
        >>> nx.set_node_attributes(G, 'pin', 'true')
        >>> G2 = G.copy()
        >>> G2.add_edges_from(bridge_edges)
        >>> pt.nx_agraph_layout(G2, inplace=True, prog='neato')
        >>> pt.show_nx(G, fnum=1, pnum=(1, 2, 1), layout='custom')
        >>> pt.show_nx(G2, fnum=1, pnum=(1, 2, 2), layout='custom')

    Example:
        G = nx.Graph([(2393, 2257), (2393, 2685), (2685, 2257), (1758, 2257)])
        bridge_edges = list(bridge_connected_augmentation(G))
        assert not any([G.has_edge(*e) for e in bridge_edges])
    """
    if G.number_of_nodes() < 3:
        raise ValueError('impossible to bridge connect less than 3 verticies')
    # find the bridge-connected components of G
    bridge_ccs = bridge_connected_compoments(G)
    # condense G into an forest C
    C = collapse(G, bridge_ccs)
    # Connect each tree in the forest to construct an arborescence
    # (I think) these must use nodes with minimum degree
    roots = [min(cc, key=C.degree) for cc in nx.connected_components(C)]
    forest_bridges = list(zip(roots, roots[1:]))
    C.add_edges_from(forest_bridges)
    # order the leaves of C by preorder
    leafs = [n for n in nx.dfs_preorder_nodes(C) if C.degree(n) == 1]
    # construct edges to bridge connect the tree
    tree_bridges = list(zip(leafs, leafs[1:]))
    # collect the edges used to augment the original forest
    aug_tree_edges = tree_bridges + forest_bridges
    # map these edges back to edges in the original graph
    # ensure that you choose a pair that does not yet exist
    inverse = ut.ddict(list)
    for k, v in C.graph['mapping'].items():
        inverse[v].append(k)
    # sort so we choose minimum degree nodes first
    inverse = {augu: sorted(mapped, key=lambda u: (G.degree(u), u))
               for augu, mapped in inverse.items()}
    for augu, augv in aug_tree_edges:
        # Find the first available edge that doesn't exist and return it
        for u, v in it.product(inverse[augu], inverse[augv]):
            if not G.has_edge(u, v):
                yield u, v
                break


@profile
def collapse(G, grouped_nodes):
    r"""
    Collapses each group of nodes into a single node.

    TODO: submit as PR

    This is similar to condensation, but works on undirected graphs.

    Parameters
    ----------
    G : NetworkX Graph
       A directed graph.

    grouped_nodes:  list or generator
       Grouping of nodes to collapse. The grouping must be disjoint.
       If grouped_nodes are strongly_connected_components then this is
       equivalent to condensation.

    Returns
    -------
    C : NetworkX Graph
       The collapsed graph C of G with respect to the node grouping.  The node
       labels are integers corresponding to the index of the component in the
       list of strongly connected components of G.  C has a graph attribute
       named 'mapping' with a dictionary mapping the original nodes to the
       nodes in C to which they belong.  Each node in C also has a node
       attribute 'members' with the set of original nodes in G that form the
       group that the node in C represents.

    --------
    Collapses a graph using disjoint groups, but not necesarilly connected
    >>> G = nx.Graph([(1, 0), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (5, 7)])
    >>> G.add_node('A')
    >>> grouped_nodes = [{0, 1, 2, 3}, {5, 6, 7}]
    >>> C = collapse(G, grouped_nodes)
    >>> assert nx.get_node_attributes(C, 'members') == {
    >>>     0: {0, 1, 2, 3}, 1: {5, 6, 7}, 2: {4}, 3: {'A'}
    >>> }
    """
    mapping = {}
    members = {}
    C = G.__class__()
    i = 0  # required if G is empty
    remaining = set(G.nodes())
    for i, group in enumerate(grouped_nodes):
        group = set(group)
        assert remaining.issuperset(group), (
            'grouped nodes must exist in G and be disjoint')
        remaining.difference_update(group)
        members[i] = group
        mapping.update((n, i) for n in group)
    # remaining nodes are in their own group
    for i, node in enumerate(remaining, start=i + 1):
        group = set([node])
        members[i] = group
        mapping.update((n, i) for n in group)
    number_of_groups = i + 1
    C.add_nodes_from(range(number_of_groups))
    C.add_edges_from((mapping[u], mapping[v]) for u, v in G.edges()
                     if mapping[u] != mapping[v])
    # Add a list of members (ie original nodes) to each node (ie scc) in C.
    nx.set_node_attributes(C, 'members', members)
    # Add mapping dict as graph attribute
    C.graph['mapping'] = mapping
    return C


def edge_connected_components(G, k):
    """
    We can find all k-edge-connected-components

    For k in {1, 2} the algorithm runs in O(n)
    For other k the algorithm runs in O(n^5)

    References:
        wang_simple_2015
        http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136264

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *
        >>> import networkx as nx
        >>> G = demodata_tarjan_bridge()
        >>> print(list(edge_connected_components(G, k=1)))
        >>> print(list(edge_connected_components(G, k=2)))
        >>> print(list(edge_connected_components(G, k=3)))
        >>> print(list(edge_connected_components(G, k=4)))
    """
    if k == 1:
        return nx.connected_components(G)
    elif k == 2:
        return bridge_connected_compoments(G)
    else:
        # FIXME: there is an efficient algorithm for k == 3
        G = G.copy()
        nx.set_edge_attributes(G, 'capacity', ut.dzip(G.edges(), [1]))
        A = aux_graph(G)
        return query_aux_graph(A, k)


def aux_graph(G, source=None, avail=None, A=None):
    """
    Max-flow O(F) = O(n^3)
    Auxiliary Graph = O(Fn) = O(n^4)

    on receiving a graph G = (V, E), a vertex s (the source) and a set of
    available vertices N (vertices that can be chosen as the sink), the
    algorithm randomly picks a vertex t 2 N − {s}, and runs the max-flow
    algorithm to determine the max-flow from s to t.

    We also set (S, T) to the corresponding min-cut (
        for the case where G is undirected, (S, T) is already the desired
        min-cut).

    Then, an edge (s, t) with weight x is added to the auxiliary graph A.

    The procedure then calls itself recursively, first with S as the set of
    available vertices and s as the source, and then with T as the set of
    available vertices and t as the source.

    The recursive calls terminate when S or T is reduced to a single vertex.

    CommandLine:
        python -m ibeis.algo.graph.nx_utils aux_graph --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.nx_utils import *
        >>> a, b, c, d, e, f, g = ut.chr_range(7)
        >>> di_paths = [
        >>>     (a, d, b, f, c),
        >>>     (a, e, b),
        >>>     (a, e, b, c, g, b, a),
        >>>     (c, b),
        >>>     (f, g, f),
        >>> ]
        >>> G = nx.DiGraph(ut.flatten(ut.itertwo(path) for path in di_paths))
        >>> nx.set_edge_attributes(G, 'capacity', ut.dzip(G.edges(), [1]))
        >>> A = aux_graph(G, source=a)
        >>> import plottool as pt
        >>> attrs = pt.nx_agraph_layout(G, inplace=True, prog='neato')[1]
        >>> nx.set_node_attributes(G, 'pin', 'true')
        >>> nx.set_edge_attributes(A, 'label', nx.get_edge_attributes(A, 'capacity'))
        >>> for key in list(attrs['node'].keys()) + ['pin']:
        >>>     nx.set_node_attributes(A, key, nx.get_node_attributes(G, key))
        >>> pt.nx_agraph_layout(A, inplace=True, prog='neato')
        >>> pt.show_nx(G, fnum=1, pnum=(1, 2, 1), layout='custom', arrow_width=1)
        >>> pt.show_nx(A, fnum=1, pnum=(1, 2, 2), layout='custom', arrow_width=1)
        >>> ut.show_if_requested()

    G = G.copy()
    nx.set_edge_attributes(G, 'capacity', ut.dzip(G.edges(), [1]))
    A = aux_graph(G)
    G.node
    A.node
    A.edge

    """
    if source is None:
        source = next(G.nodes())
    if avail is None:
        avail = set(G.nodes())
    if A is None:
        A = G.__class__()
        # A.add_node(source)
    if {source} == avail:
        return A
    # pick an arbitrary vertex as the sink
    sink = next(iter(avail - {source}))

    x, (S, T) = nx.minimum_cut(G, source, sink)
    if G.is_directed():
        x_, (T_, S_) = nx.minimum_cut(G, source, sink)
        if x_ < x:
            x, S, T = x_, T_, S_

    # add edge with weight of cut to the aug graph
    A.add_edge(source, sink, capacity=x)

    # if len(S) == 1 or len(T) == 1:
    #     return A

    aux_graph(G, source, avail.intersection(S), A=A)
    aux_graph(G, sink, avail.intersection(T), A=A)
    return A


def query_aux_graph(A, k):
    """ Query of the aux graph can be done via DFS in O(n) """
    # After the auxiliary graph A is constructed, for each query k, the
    # k-edge-connected components can be easily determined as follows:
    # traverse A and delete all edges with weights less than k.
    # Then, each connected component in the resulting graph represents a
    # k-edge-connected component in G.
    weights = nx.get_edge_attributes(A, 'capacity')
    relevant_edges = {e for e, w in weights.items() if w >= k}
    relevant_graph = nx.Graph(list(relevant_edges))
    relevant_graph.add_nodes_from(A.nodes())
    return nx.connected_components(relevant_graph)


def is_complete(G, self_loops=False):
    assert not G.is_multigraph()
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    if G.is_directed():
        n_need = (n_nodes * (n_nodes - 1))
    else:
        n_need = (n_nodes * (n_nodes - 1)) // 2
    if self_loops:
        n_need += n_nodes
    return n_edges == n_need


def random_k_edge_connected_graph(size, k, p=.1, rng=None):
    """
    Super hacky way of getting a random k-connected graph

    Example:
        >>> # ENABLE_DOCTEST
        >>> import plottool as pt
        >>> from ibeis.algo.graph.nx_utils import *  # NOQA
        >>> size, k, p = 25, 3, .1
        >>> rng = ut.ensure_rng(0)
        >>> gs = []
        >>> for x in range(4):
        >>>     G = random_k_edge_connected_graph(size, k, p, rng)
        >>>     gs.append(G)
        >>> ut.quit_if_noshow()
        >>> pnum_ = pt.make_pnum_nextgen(nRows=2, nSubplots=len(gs))
        >>> fnum = 1
        >>> for g in gs:
        >>>     pt.show_nx(g, fnum=fnum, pnum=pnum_())
    """
    import sys
    for count in it.count(0):
        seed = None if rng is None else rng.randint(sys.maxsize)
        # Randomly generate a graph
        g = nx.fast_gnp_random_graph(size, p, seed=seed)
        conn = nx.edge_connectivity(g)
        # If it has exactly the desired connectivity we are one
        if conn == k:
            break
        # If it has more, then we regenerate the graph with fewer edges
        elif conn > k:
            p = p / 2
        # If it has less then we add a small set of edges to get there
        elif conn < k:
            # p = 2 * p - p ** 2
            # if count == 2:
            aug_edges = edge_connected_augmentation(g, k)
            g.add_edges_from(aug_edges)
            break
    return g


def edge_df(graph, edges, ignore=None):
    import pandas as pd
    edge_dict = {e: graph.get_edge_data(*e) for e in edges}
    df = pd.DataFrame.from_dict(edge_dict, orient='index')

    if len(df):
        if ignore:
            ignore = df.columns.intersection(ignore)
            df = df.drop(ignore, axis=1)
        try:
            df.index.names = ('u', 'v')
        except Exception:
            pass
    return df


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.graph.nx_utils
        python -m ibeis.algo.graph.nx_utils --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
