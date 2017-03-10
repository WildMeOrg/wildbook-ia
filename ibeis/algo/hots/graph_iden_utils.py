# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt  # NOQA
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


class nx_UnionFind(object):
    """
    Based of nx code
    """

    def __init__(self, elements=None):
        if elements is None:
            elements = ()
        self.parents = {}
        self.weights = {}
        self.add_elements(elements)

    def __getitem__(self, element):
        # check for previously unknown element
        if self.add_element(element):
            return element
        # find path of objects leading to the root
        path = [element]
        root = self.parents[element]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def to_sets(self):
        import itertools as it
        for block in it.groups(self.parents).values():
            yield block

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        # Find the heaviest root according to its weight.
        heaviest = max(roots, key=lambda r: self.weights[r])
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest

    def remove_entire_cc(self, elements):
        # NOTE: this will not work in general. This only
        # works if all elements are a unique component.
        for x in elements:
            del self.weights[x]
            del self.parents[x]

    def add_element(self, x):
        if x not in self.parents:
            self.weights[x] = 1
            self.parents[x] = x
            return True
        return False

    def add_elements(self, elements):
        for x in elements:
            if x not in self.parents:
                self.weights[x] = 1
                self.parents[x] = x


class DynConnGraph(nx.Graph):
    """
    Dynamically connected graph.

    Maintains a data structure parallel to a normal networkx graph that
    maintains dynamic connectivity for fast connected compoment queries.

    Underlying Data Structures and limitations are

    Data Structure     | Insertion | Deletion | CC Find |
    -----------------------------------------------------
    * UnionFind        | Yes       |   No     |  lg(n)
    * EulerTourForest  | lg^2(n)   | lg^2(n)  |  lg(n) / lglg(n) - - Ammortized

    References:
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.pdf
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.html
        http://cs.stackexchange.com/questions/33595/maintaining-connecte

    Args:
        self (?):
        *args:
        **kwargs:

    CommandLine:
        python -m ibeis.algo.hots.graph_iden_utils DynConnGraph

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.graph_iden_utils import *  # NOQA
        >>> self = DynConnGraph()
        >>> self.add_edges_from([(1, 2), (2, 3), (4, 5), (6, 7), (7, 4)])
        >>> self.add_edges_from([(10, 20), (20, 30), (40, 50), (60, 70), (70, 40)])
        >>> self._ccs
        >>> u, v = 20, 1
        >>> assert self.node_label(u) != self.node_label(v)
        >>> assert self.connected_to(u) != self.connected_to(v)
        >>> self.add_edge(u, v)
        >>> assert self.node_label(u) == self.node_label(v)
        >>> assert self.connected_to(u) == self.connected_to(v)
        >>> self.remove_edge(u, v)
        >>> assert self.node_label(u) != self.node_label(v)
        >>> assert self.connected_to(u) != self.connected_to(v)
        >>> import plottool as pt
        >>> ccs = list(self.connected_components())
        >>> pt.qtensure()
        >>> pt.show_nx(self)

    # todo: check if nodes exist when adding
    """
    def __init__(self, *args, **kwargs):
        # raise NotImplementedError('unfinished')
        self._ccs = {}
        self._union_find = nx_UnionFind()
        super(DynConnGraph, self).__init__(*args, **kwargs)

    def connected_to(self, node):
        return self._ccs[self._union_find[node]]

    def node_label(self, node):
        return self._union_find[node]

    def connected_components(self):
        for cc in self._ccs.values():
            yield cc

    # -----

    @profile
    def _cut(self, u, v):
        """ Decremental connectivity (slow) """
        old_nid1 = self._union_find[u]
        old_nid2 = self._union_find[v]
        if old_nid1 != old_nid2:
            return
        # Need to break appart entire component and then reconstruct it
        old_cc = self._ccs[old_nid1]
        del self._ccs[old_nid1]
        self._union_find.remove_entire_cc(old_cc)
        # Might be faster to just do DFS to find the CC
        internal_edges = bridges_inside(self, old_cc)
        for edge in internal_edges:
            self._union(*edge)

    @profile
    def _union(self, u, v):
        """ Incremental connectivity (fast) """
        # print('Union ({})'.format((u, v)))
        self._add_node(u)
        self._add_node(v)
        old_nid1 = self._union_find[u]
        old_nid2 = self._union_find[v]
        self._union_find.union(u, v)
        new_nid = self._union_find[u]
        for old_nid in [old_nid1, old_nid2]:
            if new_nid != old_nid:
                parts = self._ccs.pop(old_nid)
                self._ccs[new_nid].update(parts)

    def _add_node(self, n):
        if self._union_find.add_element(n):
            # print('Add ({})'.format((n)))
            self._ccs[n] = {n}

    def add_edge(self, u, v, **attr):
        self._union(u, v)
        super(DynConnGraph, self).add_edge(u, v, **attr)

    def add_edges_from(self, ebunch, **attr):
        ebunch = list(ebunch)
        # print('add_edges_from %r' % (ebunch,))
        for e in ebunch:
            self._union(*e)
        super(DynConnGraph, self).add_edges_from(ebunch, **attr)

    # ----

    def add_node(self, n, **attr):
        self._add_node(n)
        super(DynConnGraph, self).add_node(n, **attr)

    def add_nodes_from(self, nodes, **attr):
        nodes = list(nodes)
        for n in nodes:
            self._add_node(n)
        super(DynConnGraph, self).add_nodes_from(nodes, **attr)

    # ----

    def remove_edge(self, u, v):
        super(DynConnGraph, self).remove_edge(u, v)
        self._cut(u, v)

    def remove_edges_from(self, ebunch):
        ebunch = list(ebunch)
        super(DynConnGraph, self).remove_edges_from(ebunch)
        for e in ebunch:
            self._cut(*e)

    # -----

    def remove_nodes_from(self, nodes):
        # remove edges as well
        for n in nodes:
            nbrs = list(self.adj[n].keys())
            self.remove_edges_from((n, v) for v in nbrs)
        super(DynConnGraph, self).remove_nodes_from(nodes)

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
