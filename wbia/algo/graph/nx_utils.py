# -*- coding: utf-8 -*-
"""
TODO: the k-components will soon be implemented in networkx 2.0 use those instead
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import networkx as nx
import itertools as it
import vtool as vt  # NOQA

# import wbia.algo.graph.nx_edge_kconnectivity as nx_ec
from wbia.algo.graph import nx_edge_augmentation as nx_aug
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
        >>> from wbia.algo.graph.nx_utils import *  # NOQA
        >>> import utool as ut
        >>> G = demodata_bridge()
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
    return {e_(u, v) for u in nodes1 for v in nodes2.intersection(graph.adj[u])}


def edges_between(graph, nodes1, nodes2=None, assume_disjoint=False, assume_dense=True):
    r"""
    Get edges between two components or within a single component

    Args:
        graph (nx.Graph): the graph
        nodes1 (set): list of nodes
        nodes2 (set): if None it is equivlanet to nodes2=nodes1 (default=None)
        assume_disjoint (bool): skips expensive check to ensure edges arnt
            returned twice (default=False)

    CommandLine:
        python -m wbia.algo.graph.nx_utils --test-edges_between

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.nx_utils import *  # NOQA
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
        edge_sets = [
            it.product(nodes_only1, nodes_only2),
            it.product(nodes_only1, nodes_isect),
            it.product(nodes_only2, nodes_isect),
            it.combinations(nodes_isect, 2),
        ]
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
        both_adj = {u: set(graph.adj[u]) for u in both}
        if graph.is_directed():
            edge_sets = (
                _edges_inside_upper(graph, both_adj),  # B-to-B (u)
                _edges_inside_lower(graph, both_adj),  # B-to-B (l)
            )
        else:
            edge_sets = (_edges_inside_upper(graph, both_adj),)  # B-to-B (u)
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
            edge_sets = (_edges_between_disjoint(graph, only1, only2),)  # 1-to-2
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
        both_adj = {u: set(graph.adj[u]) for u in both}
        if graph.is_directed():
            edge_sets = (
                _edges_between_disjoint(graph, only1_adj, only2),  # 1-to-2
                _edges_between_disjoint(graph, only1_adj, both),  # 1-to-B
                _edges_inside_upper(graph, both_adj),  # B-to-B (u)
                _edges_inside_lower(graph, both_adj),  # B-to-B (l)
                _edges_between_disjoint(graph, both_adj, only1),  # B-to-1
                _edges_between_disjoint(graph, both_adj, only2),  # B-to-2
                _edges_between_disjoint(graph, only2_adj, both),  # 2-to-B
                _edges_between_disjoint(graph, only2_adj, only1),  # 2-to-1
            )
        else:
            edge_sets = (
                _edges_between_disjoint(graph, only1_adj, only2),  # 1-to-2
                _edges_between_disjoint(graph, only1_adj, both),  # 1-to-B
                _edges_inside_upper(graph, both_adj),  # B-to-B (u)
                _edges_between_disjoint(graph, only2_adj, both),  # 2-to-B
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
        python -m wbia.algo.graph.nx_utils demodata_tarjan_bridge --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.nx_utils import *  # NOQA
        >>> G = demodata_tarjan_bridge()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.show_nx(G)
        >>> ut.show_if_requested()
    """
    # define 2-connected compoments and bridges
    cc2 = [
        (1, 2, 4, 3, 1, 4),
        (5, 6, 7, 5),
        (8, 9, 10, 8),
        (17, 18, 16, 15, 17),
        (11, 12, 14, 13, 11, 14),
    ]
    bridges = [(4, 8), (3, 5), (3, 17)]
    G = nx.Graph(ut.flatten(ut.itertwo(path) for path in cc2 + bridges))
    return G


# def is_tri_edge_connected(G):
#     """
#     Yet another Simple Algorithm for Triconnectivity
#     http://www.sciencedirect.com/science/article/pii/S1570866708000415
#     """
#     pass


def is_k_edge_connected(G, k):
    return nx_aug.is_k_edge_connected(G, k)


def complement_edges(G):
    return it.starmap(e_, nx_aug.complement_edges(G))


def k_edge_augmentation(G, k, avail=None, partial=False):
    return it.starmap(e_, nx_aug.k_edge_augmentation(G, k, avail=avail, partial=partial))


def is_complete(G, self_loops=False):
    assert not G.is_multigraph()
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    if G.is_directed():
        n_need = n_nodes * (n_nodes - 1)
    else:
        n_need = (n_nodes * (n_nodes - 1)) // 2
    if self_loops:
        n_need += n_nodes
    return n_edges == n_need


def random_k_edge_connected_graph(size, k, p=0.1, rng=None):
    """
    Super hacky way of getting a random k-connected graph

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia.plottool as pt
        >>> from wbia.algo.graph.nx_utils import *  # NOQA
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
            aug_edges = list(k_edge_augmentation(g, k))
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
        python -m wbia.algo.graph.nx_utils
        python -m wbia.algo.graph.nx_utils --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
