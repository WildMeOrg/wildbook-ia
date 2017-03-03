# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt  # NOQA
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


class nx_UnionFind(nx.utils.UnionFind):
    def add_element(self, x):
        if x not in self.weights:
            self.weights[x] = 1
            self.parents[x] = x

    def add_elements(self, elements):
        for x in elements:
            if x not in self.weights:
                self.weights[x] = 1
                self.parents[x] = x


class DynConnGraph(nx.Graph):
    """
    Dynamically connected graph.

    Maintains a data structure parallel to a normal networkx graph that
    maintains dynamic connectivity for fast connected compoment queries.

    Underlying Data Structures and limitations are

    Data Structure            | Insertion | Deletion | CC Find |
    -----------------------------------------------------
    * UnionFind               | Yes       |   No     |  lg(n)
    * SpanningEulerTourForest | lg^2(n)   | lg^2(n)  |  lg(n) / lglg(n) - - Ammortized

    References:
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.pdf
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.html
        http://cs.stackexchange.com/questions/33595/what-is-the-most-efficient-algorithm-and-data-structure-for-maintaining-connecte

    # todo: check if nodes exist when adding
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('unfinished')
        super(DynConnGraph, self).__init__(*args, **kwargs)
        self._reset_union_find()

    def _reset_union_find(self):
        self._union_find = nx_UnionFind(self.nodes())
        for e in self.edges():
            self._union_find.union(*e)

    # -----

    def add_edge(self, u, v, **attr):
        self._union_find.union(u, v)
        super(DynConnGraph, self).add_edge(u, v, **attr)

    def add_edges_from(self, ebunch, **attr):
        for e in ebunch:
            self._union_find.union(*e)
        super(DynConnGraph, self).add_edges_from(self, ebunch, **attr)

    # ----

    def add_node(self, n, **attr):
        self._union_find.add_element(n)
        super(DynConnGraph, self).add_node(self, n, **attr)

    def add_nodes_from(self, nodes, **attr):
        self._union_find.add_elements(nodes)
        super(DynConnGraph, self).add_nodes_from(self, nodes, **attr)

    # ----

    def remove_edge(self, u, v):
        self._reset_union_find()
        super(DynConnGraph, self).remove_edge(self, u, v)

    def remove_edges_from(self, ebunch):
        self._reset_union_find()
        super(DynConnGraph, self).remove_edges_from(self, ebunch)

    # -----

    def remove_nodes_from(self, nodes):
        # remove edges as well
        for n in nodes:
            nbrs = list(self.adj[n].keys())
            self.remove_edges_from((n, v) for v in nbrs)
        super(DynConnGraph, self).remove_nodes_from(self, nodes)

    def remove_node(self, n):
        # remove edges as well
        nbrs = list(self.adj[n].keys())
        self.remove_edges_from((n, v) for v in nbrs)
        super(DynConnGraph, self).remove_node(n)


def _dz(a, b):
    a = a.tolist() if isinstance(a, np.ndarray) else list(a)
    b = b.tolist() if isinstance(b, np.ndarray) else list(b)
    return ut.dzip(a, b)


@profile
def e_(u, v):
    return (u, v) if u < v else (v, u)


@profile
def bridges_inside(graph, nodes):
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


@profile
def bridges_cross(graph, nodes1, nodes2):
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


# @profile
# def bridges(graph, cc1, cc2=None):
#     if cc2 is None or cc2 is cc1:
#         yielder = []
#         both = set(cc1)
#         both_upper = both.copy()
#         for u in both:
#             neighbs = set(graph.adj[u])
#             neighbsBB_upper = neighbs.intersection(both_upper)
#             for v in neighbsBB_upper:
#                 yielder.append(e_(u, v))
#                 # yield e_(u, v)
#             both_upper.remove(u)
#         return yielder
#     else:
#         yielder = []
#         # assume cc1 and cc2 are disjoint
#         only1 = set(cc1)
#         only2 = set(cc2)
#         for u in only1:
#             neighbs = set(graph.adj[u])
#             neighbs12 = neighbs.intersection(only2)
#             for v in neighbs12:
#                 yielder.append(e_(u, v))
#                 # yield e_(u, v)
#         return yielder
    # test2 =  [e_(u, v) for u, v in ut.nx_edges_between(graph, cc1, cc2,
    #                                                    assume_sparse=True,
    #                                                    assume_disjoint=True)]
    # test3 =  [e_(u, v) for u, v in ut.nx_edges_between(graph, cc1, cc2,
    #                                                    assume_sparse=False,
    #                                                    assume_disjoint=True)]
    # assert sorted(test1) == sorted(test2)
    # assert sorted(test1) == sorted(test3)
    # return test1
