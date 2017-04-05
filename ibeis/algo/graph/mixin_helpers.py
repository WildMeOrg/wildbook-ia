# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools as it
import networkx as nx
import operator
import numpy as np
import utool as ut
import vtool as vt
from ibeis.algo.graph.nx_utils import e_
import six
print, rrr, profile = ut.inject2(__name__)


DEBUG_INCON = True


class AttrAccess(object):
    """ Contains non-core helper functions """

    def gen_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        return ut.util_graph.nx_gen_node_attrs(
                infr.graph, key, nodes=nodes, default=default)

    def gen_edge_attrs(infr, key, edges=None, default=ut.NoParam,
                       check_exist=True):
        """ maybe change to gen edge items """
        return ut.util_graph.nx_gen_edge_attrs(
                infr.graph, key, edges=edges, default=default,
                check_exist=check_exist)

    def gen_edge_values(infr, key, edges=None, default=ut.NoParam,
                        check_exist=True):
        return (t[1] for t in ut.util_graph.nx_gen_edge_attrs(
                infr.graph, key, edges=edges, default=default,
                check_exist=check_exist))

    def get_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        """ Networkx node getter helper """
        return dict(infr.gen_node_attrs(key, nodes=nodes, default=default))

    def get_edge_attrs(infr, key, edges=None, default=ut.NoParam):
        """ Networkx edge getter helper """
        return dict(infr.gen_edge_attrs(key, edges=edges, default=default))

    def _get_edges_where(infr, key, op, val, edges=None, default=ut.NoParam):
        edge_to_attr = infr.gen_edge_attrs(key, edges=edges, default=default)
        return (e for e, v in edge_to_attr if op(v, val))

    def get_edges_where_eq(infr, key, val, edges=None, default=ut.NoParam):
        return infr._get_edges_where(key, operator.eq, val, edges=edges,
                                     default=default)

    def get_edges_where_ne(infr, key, val, edges=None, default=ut.NoParam):
        return infr._get_edges_where(key, operator.ne, val, edges=edges,
                                     default=default)

    def set_node_attrs(infr, key, node_to_prop):
        """ Networkx node setter helper """
        return nx.set_node_attributes(infr.graph, key, node_to_prop)

    def set_edge_attrs(infr, key, edge_to_prop):
        """ Networkx edge setter helper """
        return nx.set_edge_attributes(infr.graph, key, edge_to_prop)

    def get_edge_attr(infr, edge, key, default=ut.NoParam):
        """ single edge getter helper """
        return infr.get_edge_attrs(key, [edge], default=default)[edge]

    def set_edge_attr(infr, edge, attr):
        """ single edge setter helper """
        for key, value in attr.items():
            infr.set_edge_attrs(key, {edge: value})

    def get_annot_attrs(infr, key, aids):
        """ Wrapper around get_node_attrs specific to annotation nodes """
        nodes = ut.take(infr.aid_to_node, aids)
        attr_list = list(infr.get_node_attrs(key, nodes).values())
        return attr_list

    def edges(infr, data=False):
        if data:
            return ((e_(u, v), d) for u, v, d in infr.graph.edges(data=True))
        else:
            return (e_(u, v) for u, v in infr.graph.edges())

    def has_edge(infr, edge):
        return infr.graph.has_edge(*edge)
        # redge = edge[::-1]
        # flag = infr.graph.has_edge(*edge) or infr.graph.has_edge(*redge)
        # return flag

    def get_edge_data(infr, u, v):
        data = infr.graph.get_edge_data(u, v)
        if data is not None:
            data = ut.delete_dict_keys(data.copy(), infr.visual_edge_attrs)
        return data

    @property
    def pos_graph(infr):
        return infr.review_graphs['match']

    @property
    def neg_graph(infr):
        return infr.review_graphs['nomatch']

    @property
    def incomp_graph(infr):
        return infr.review_graphs['notcomp']

    @property
    def unreviewed_graph(infr):
        return infr.review_graphs['unreviewed']


class Convenience(object):
    @staticmethod
    def e_(u, v):
        return e_(u, v)

    def print_graph_info(infr):
        print(ut.repr3(ut.graph_info(infr.simplify_graph())))


@six.add_metaclass(ut.ReloadingMetaclass)
class DummyEdges(object):
    def remove_dummy_edges(infr):
        infr.print('remove_dummy_edges', 2)
        edge_to_isdummy = infr.get_edge_attrs('_dummy_edge')
        dummy_edges = [edge for edge, flag in edge_to_isdummy.items() if flag]
        infr.graph.remove_edges_from(dummy_edges)

    def apply_mst(infr):
        """
        MST edges connect nodes labeled with the same name.
        This is done in case an explicit feedback or score edge does not exist.
        """
        infr.print('apply_mst', 2)
        # Remove old MST edges
        infr.remove_dummy_edges()
        infr.ensure_mst()

    def ensure_full(infr):
        infr.print('ensure_full with %d nodes' % (len(infr.graph)), 2)
        new_edges = nx.complement(infr.graph).edges()
        # if infr.verbose:
        #     infr.print('adding %d complement edges' % (len(new_edges)))
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', ut.dzip(new_edges, [True]))

    def ensure_cliques(infr, label='name_label'):
        """
        Force each name label to be a clique
        """
        infr.print('ensure_cliques', 1)
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())
        new_edges = []
        for label, nodes in label_to_nodes.items():
            for edge in it.combinations(nodes, 2):
                if not infr.has_edge(edge):
                    new_edges.append(edge)
        infr.print('adding %d clique edges' % (len(new_edges)), 2)
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', ut.dzip(new_edges, [True]))

    def find_mst_edges2(infr):
        """
        Returns edges to augment existing PCCs (by label) in order to ensure
        they are connected

        nid = 5977
        """
        import networkx as nx
        # Find clusters by labels
        name_attr = 'name_label'
        # name_attr = 'orig_name_label'
        node_to_label = infr.get_node_attrs(name_attr)
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())

        aug_graph = infr.simplify_graph()

        new_edges = []
        prog = ut.ProgIter(list(label_to_nodes.keys()),
                           label='finding mst edges',
                           enabled=infr.verbose > 0)
        for nid in prog:
            nodes = label_to_nodes[nid]
            # We want to make this CC connected
            target_cc = aug_graph.subgraph(nodes)
            if len(nodes) > 10 and len(list(target_cc.edges())):
                break
            positive_edges = [
                e_(*e) for e, v in
                nx.get_edge_attributes(target_cc, 'decision').items()
                if v == 'match'
            ]
            tmp = nx.Graph()
            tmp.add_nodes_from(nodes)
            tmp.add_edges_from(positive_edges)
            # Need to find a way to connect these components
            sub_ccs = list(nx.connected_components(tmp))

            connecting_edges = []

            if False:
                for c1, c2 in it.combinations(sub_ccs, 2):
                    for u, v in it.product(c1, c2):
                        if not target_cc.has_edge(u, v):
                            # Once we find one edge we've completed the connection
                            connecting_edges.append((u, v))
                            break
            else:
                # TODO: prioritize based on comparability
                for c1, c2 in it.combinations(sub_ccs, 2):
                    found = False
                    for u, v in it.product(c1, c2):
                        if not target_cc.has_edge(u, v):
                            if infr.is_comparable([(u, v)])[0]:
                                # Once we find one edge we've completed the
                                # connection
                                connecting_edges.append((u, v))
                                found = True
                                break
                    if not found:
                        connecting_edges.append((u, v))
                        # no comparable edges, so add them all
                        # connecting_edges.extend(list(it.product(c1, c2)))

            # Find the MST of the candidates to eliminiate complexity
            # (mostly handles singletons, when existing CCs are big this wont
            #  matter)
            candidate_graph = nx.Graph(connecting_edges)
            mst_edges = list(nx.minimum_spanning_tree(candidate_graph).edges())
            new_edges.extend(mst_edges)

            target_cc.add_edges_from(mst_edges)
            assert nx.is_connected(target_cc)
        prog.ensure_newline()

        for edge in new_edges:
            assert not infr.graph.has_edge(*edge)

        return new_edges

        # aug_graph = infr.graph.copy()

    def find_mst_edges(infr):
        """
        Find a set of edges that need to be inserted in order to complete the
        given labeling. Respects the current edges that exist.
        """
        import networkx as nx
        # Find clusters by labels
        node_to_label = infr.get_node_attrs('name_label')
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())

        aug_graph = infr.graph.copy().to_undirected()

        # remove cut edges from augmented graph
        edge_to_iscut = nx.get_edge_attributes(aug_graph, 'is_cut')
        cut_edges = [
            (u, v)
            for (u, v, d) in aug_graph.edges(data=True)
            if not (
                d.get('is_cut') or
                d.get('decision', 'unreviewed') in ['nomatch']
            )
        ]
        cut_edges = [edge for edge, flag in edge_to_iscut.items() if flag]
        aug_graph.remove_edges_from(cut_edges)

        # Enumerate cliques inside labels
        unflat_edges = [list(ut.itertwo(nodes))
                        for nodes in label_to_nodes.values()]
        node_pairs = [tup for tup in ut.iflatten(unflat_edges)
                      if tup[0] != tup[1]]

        # Remove candidate MST edges that exist in the original graph
        orig_edges = list(aug_graph.edges())
        candidate_mst_edges = [edge for edge in node_pairs
                               if not aug_graph.has_edge(*edge)]
        # randomness prevents chains and visually looks better
        rng = np.random.RandomState(42)

        def _randint():
            return 0
            return rng.randint(0, 100)
        aug_graph.add_edges_from(candidate_mst_edges)
        # Weight edges in aug_graph such that existing edges are chosen
        # to be part of the MST first before suplementary edges.
        nx.set_edge_attributes(aug_graph, 'weight',
                               {edge: 0.1 for edge in orig_edges})

        try:
            # Try linking by time for lynx data
            nodes = list(set(ut.iflatten(candidate_mst_edges)))
            aids = ut.take(infr.node_to_aid, nodes)
            times = infr.ibs.annots(aids).time
            node_to_time = ut.dzip(nodes, times)
            time_deltas = np.array([
                abs(node_to_time[u] - node_to_time[v])
                for u, v in candidate_mst_edges
            ])
            # print('time_deltas = %r' % (time_deltas,))
            maxweight = vt.safe_max(time_deltas, nans=False, fill=0) + 1
            time_deltas[np.isnan(time_deltas)] = maxweight
            time_delta_weight = 10 * time_deltas / (time_deltas.max() + 1)
            is_comp = infr.guess_if_comparable(candidate_mst_edges)
            comp_weight = 10 * (1 - is_comp)
            extra_weight = comp_weight + time_delta_weight

            # print('time_deltas = %r' % (time_deltas,))
            nx.set_edge_attributes(
                aug_graph, 'weight', {
                    edge: 10.0 + extra for edge, extra in
                    zip(candidate_mst_edges, extra_weight)})
        except Exception:
            infr.print('FAILED WEIGHTING USING TIME')
            nx.set_edge_attributes(aug_graph, 'weight',
                                   {edge: 10.0 + _randint()
                                    for edge in candidate_mst_edges})
        new_edges = []
        for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
            mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
            # Only add edges not in the original graph
            for edge in mst_sub_graph.edges():
                if not infr.has_edge(edge):
                    new_edges.append(e_(*edge))
        return new_edges

    def ensure_mst(infr):
        """
        Use minimum spannning tree to ensure all names are connected
        Needs to be applied after any operation that adds/removes edges if we
        want to maintain that name labels must be connected in some way.
        """
        infr.print('ensure_mst', 1)
        new_edges = infr.find_mst_edges()
        # Add new MST edges to original graph
        infr.print('adding %d MST edges' % (len(new_edges)), 2)
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', ut.dzip(new_edges, [True]))

    def review_dummy_edges(infr, method=1):
        """
        Creates just enough dummy reviews to maintain a consistent labeling if
        relabel_using_reviews is called. (if the existing edges are consistent).
        """
        infr.print('review_dummy_edges', 2)
        if method == 2:
            new_edges = infr.find_mst_edges2()
        else:
            new_edges = infr.find_mst_edges()
        infr.print('reviewing %s dummy edges' % (len(new_edges),), 1)
        # TODO apply set of new edges in bulk
        for u, v in new_edges:
            infr.add_feedback((u, v), decision='match', confidence='guessing',
                               user_id='mst', verbose=False)


class AssertInvariants(object):
    def assert_invariants(infr, msg=''):
        infr.assert_disjoint_invariant(msg)
        infr.assert_consistency_invariant(msg)
        infr.assert_recovery_invariant(msg)

    def assert_disjoint_invariant(infr, msg=''):
        edge_sets = [
            set(it.starmap(e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        ]
        for es1, es2 in it.combinations(edge_sets, 2):
            assert es1.isdisjoint(es2), 'edge sets must be disjoint'
        all_edges = set(it.starmap(e_, infr.graph.edges()))
        edge_union = set.union(*edge_sets)
        assert edge_union == all_edges, 'edge sets must have full union'

    def assert_consistency_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        if infr.enable_inference:
            incon_ccs = list(infr.inconsistent_components())
            with ut.embed_on_exception_context:
                if len(incon_ccs) > 0:
                    raise AssertionError('The graph is not consistent. ' +
                                         msg)

    def assert_recovery_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        inconsistent_ccs = list(infr.inconsistent_components())
        incon_cc = set(ut.flatten(inconsistent_ccs))  # NOQA
        # import utool
        # with utool.embed_on_exception_context:
        #     assert infr.recovery_cc.issuperset(incon_cc), 'diff incon'
        #     if False:
        #         # nid_to_cc2 = ut.group_items(
        #         #     incon_cc,
        #         #     map(pos_graph.node_label, incon_cc))
        #         infr.print('infr.recovery_cc = %r' % (infr.recovery_cc,))
        #         infr.print('incon_cc = %r' % (incon_cc,))

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.graph.mixin_helpers
        python -m ibeis.algo.graph.mixin_helpers --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
