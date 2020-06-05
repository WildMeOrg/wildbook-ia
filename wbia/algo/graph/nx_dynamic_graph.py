# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import networkx as nx
import itertools as it
from wbia.algo.graph.nx_utils import edges_inside, e_

print, rrr, profile = ut.inject2(__name__)


class GraphHelperMixin(ut.NiceRepr):
    def __nice__(self):
        return 'nNodes={}, nEdges={}'.format(
            self.number_of_nodes(), self.number_of_edges(),
        )

    def has_nodes(self, nodes):
        return (self.has_node(node) for node in nodes)

    def has_edges(self, edges):
        return (self.has_edge(*edge) for edge in edges)

    def edges(self, nbunch=None, data=False, default=None):
        # Force edges to always be returned in upper triangular form
        edges = super(GraphHelperMixin, self).edges(nbunch, data, default)
        if data:
            return (e_(u, v) + (d,) for u, v, d in edges)
        else:
            return (e_(u, v) for u, v in edges)


class NiceGraph(nx.Graph, GraphHelperMixin):
    pass


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

    def clear(self):
        self.parents = {}
        self.weights = {}

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

    def rebalance(self, elements=None):
        if elements is None:
            elements = list(self.parents.keys())
        # Make sure only one operation is needed to lookup any node
        for x in elements:
            parent = self[x]
            self.parents[x] = parent
            self.weights[x] = 1

    def to_sets(self):
        for block in it.groups(self.parents).values():
            yield block

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        # HACK: use the lowest node number to preserve
        # node labels through cuts. (has some runtime penalty)
        if False:
            # Find the best root with the maximum weight
            best = max(roots, key=lambda r: self.weights[r])
        else:
            best = min(roots)
        for r in roots:
            if r != best:
                self.weights[best] += self.weights[r]
                self.parents[r] = best

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


class DynConnGraph(nx.Graph, GraphHelperMixin):
    """
    Dynamically connected graph.

    Maintains a data structure parallel to a normal networkx graph that
    maintains dynamic connectivity for fast connected compoment queries.

    Underlying Data Structures and limitations are

    Data Structure     | Insertion | Deletion | CC Find |
    -----------------------------------------------------
    * UnionFind        | lg(n)     |    n     |  No
    * UnionFind2       |    n*     |    n     |  1
    * EulerTourForest  | lg^2(n)   | lg^2(n)  |  lg(n) / lglg(n) - - Ammortized

    * it seems to be very quick

    References:
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.pdf
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.html
        http://cs.stackexchange.com/questions/33595/maintaining-connecte
        https://en.wikipedia.org/wiki/Dynamic_connectivity#Fully_dynamic_connectivity

    CommandLine:
        python -m wbia.algo.graph.nx_dynamic_graph DynConnGraph

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.nx_dynamic_graph import *  # NOQA
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
        >>> ccs = list(self.connected_components())
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.qtensure()
        >>> pt.show_nx(self)

    # todo: check if nodes exist when adding
    """

    def __init__(self, *args, **kwargs):
        # raise NotImplementedError('unfinished')
        self._ccs = {}
        self._union_find = nx_UnionFind()
        super(DynConnGraph, self).__init__(*args, **kwargs)

    def clear(self):
        super(DynConnGraph, self).clear()
        self._ccs = {}
        self._union_find.clear()

    def __nice__(self):
        return 'nNodes={}, nEdges={}, nCCs={}'.format(
            self.number_of_nodes(), self.number_of_edges(), self.number_of_components(),
        )

    def number_of_components(self):
        return len(self._ccs)

    def component(self, label):
        return self._ccs[label]

    component_nodes = component

    def connected_to(self, node):
        return self._ccs[self._union_find[node]]

    def node_label(self, node):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.nx_dynamic_graph import *  # NOQA
            >>> self = DynConnGraph()
            >>> self.add_edges_from([(1, 2), (2, 3), (4, 5), (6, 7)])
            >>> assert self.node_label(2) == self.node_label(1)
            >>> assert self.node_label(2) != self.node_label(4)
        """
        return self._union_find[node]

    def node_labels(self, *nodes):
        return [self._union_find[node] for node in nodes]

    def are_nodes_connected(self, u, v):
        return ut.allsame(self.node_labels(u, v))

    def connected_components(self):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.nx_dynamic_graph import *  # NOQA
            >>> self = DynConnGraph()
            >>> self.add_edges_from([(1, 2), (2, 3), (4, 5), (6, 7)])
            >>> ccs = list(self.connected_components())
            >>> result = 'ccs = {}'.format(ut.repr2(ccs, nl=0))
            >>> print(result)
            ccs = [{1, 2, 3}, {4, 5}, {6, 7}]
        """
        for cc in self._ccs.values():
            yield cc

    def component_labels(self):
        for label in self._ccs.keys():
            yield label

    # -----

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
        internal_edges = edges_inside(self, old_cc)
        # Add nodes in case there are no edges to it
        for n in old_cc:
            self._add_node(n)
        for edge in internal_edges:
            self._union(*edge)

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
                # FIXME: this step can be quite bad for time complexity.
                # An Euler Tour Tree might solve the issue
                self._ccs[new_nid].update(parts)

    def _add_node(self, n):
        if self._union_find.add_element(n):
            # print('Add ({})'.format((n)))
            self._ccs[n] = {n}

    def _remove_node(self, n):
        if n in self._union_find.parents:
            del self._union_find.weights[n]
            del self._union_find.parents[n]
            del self._ccs[n]

    def add_edge(self, u, v, **attr):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.nx_dynamic_graph import *  # NOQA
            >>> self = DynConnGraph()
            >>> self.add_edges_from([(1, 2), (2, 3), (4, 5), (6, 7), (7, 4)])
            >>> assert self._ccs == {1: {1, 2, 3}, 4: {4, 5, 6, 7}}
            >>> self.add_edge(1, 5)
            >>> assert self._ccs == {1: {1, 2, 3, 4, 5, 6, 7}}
        """
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
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.nx_dynamic_graph import *  # NOQA
            >>> self = DynConnGraph()
            >>> self.add_edges_from([(1, 2), (2, 3), (4, 5), (6, 7), (7, 4)])
            >>> assert self._ccs == {1: {1, 2, 3}, 4: {4, 5, 6, 7}}
            >>> self.add_edge(1, 5)
            >>> assert self._ccs == {1: {1, 2, 3, 4, 5, 6, 7}}
            >>> self.remove_edge(1, 5)
            >>> assert self._ccs == {1: {1, 2, 3}, 4: {4, 5, 6, 7}}
        """
        super(DynConnGraph, self).remove_edge(u, v)
        self._cut(u, v)

    def remove_edges_from(self, ebunch):
        ebunch = list(ebunch)
        super(DynConnGraph, self).remove_edges_from(ebunch)
        # Can do this more efficiently for bulk edges
        for e in ebunch:
            self._cut(*e)

    # -----

    def remove_nodes_from(self, nodes):
        # remove edges as well
        nodes = list(nodes)
        for n in nodes:
            nbrs = list(self.adj[n].keys())
            self.remove_edges_from((n, v) for v in nbrs)
        for n in nodes:
            self._remove_node(n)
        super(DynConnGraph, self).remove_nodes_from(nodes)

    def remove_node(self, n):
        r"""
        CommandLine:
            python -m wbia.algo.graph.nx_dynamic_graph remove_node

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.nx_dynamic_graph import *  # NOQA
            >>> self = DynConnGraph()
            >>> self.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)])
            >>> assert self._ccs == {1: {1, 2, 3}, 4: {4, 5, 6, 7, 8, 9}}
            >>> self.remove_node(2)
            >>> assert self._ccs == {1: {1}, 3: {3}, 4: {4, 5, 6, 7, 8, 9}}
            >>> self.remove_node(7)
            >>> assert self._ccs == {1: {1}, 3: {3}, 4: {4, 5, 6}, 8: {8, 9}}
        """
        # remove edges as well
        nbrs = list(self.adj[n].keys())
        self.remove_edges_from((n, v) for v in nbrs)
        self._remove_node(n)
        super(DynConnGraph, self).remove_node(n)

    @profile
    def subgraph(self, nbunch, dynamic=False):
        if dynamic is False:
            H = nx.Graph()
            nbunch = set(nbunch)
            H.add_nodes_from(nbunch)
            H.add_edges_from(edges_inside(self, nbunch))
        else:
            H = super(DynConnGraph, self).subgraph(nbunch)
            for n in nbunch:
                # need to add individual nodes
                H._add_node(n)
            # Recreate the connected compoment structure
            for u, v in H.edges():
                H._union(u, v)
        return H


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph all
        python -m wbia.algo.graph.nx_dynamic_graph all
        python -m wbia.algo.graph.nx_dynamic_graph --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
