# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools as it
import networkx as nx
import operator
import numpy as np
import utool as ut
import vtool as vt
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN
from ibeis.algo.graph.nx_utils import e_, edges_inside
from ibeis.algo.graph import nx_utils
import six
print, rrr, profile = ut.inject2(__name__)


DEBUG_INCON = True


class AttrAccess(object):
    """ Contains non-core helper functions """

    def gen_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        return ut.util_graph.nx_gen_node_attrs(
                infr.graph, key, nodes=nodes, default=default)

    def gen_edge_attrs(infr, key, edges=None, default=ut.NoParam,
                       check_exist=False):
        """ maybe change to gen edge items """
        return ut.util_graph.nx_gen_edge_attrs(
                infr.graph, key, edges=edges, default=default,
                check_exist=check_exist)

    def gen_node_values(infr, key, nodes, default=ut.NoParam):
        return ut.util_graph.nx_gen_node_values(
            infr.graph, key, nodes, default=default)

    def gen_edge_values(infr, key, edges, default=ut.NoParam):
        return ut.util_graph.nx_gen_edge_values(
            infr.graph, key, edges, default=default)

    def get_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        """ Networkx node getter helper """
        return dict(infr.gen_node_attrs(key, nodes=nodes, default=default))

    def get_edge_attrs(infr, key, edges=None, default=ut.NoParam,
                       check_exist=False):
        """ Networkx edge getter helper """
        return dict(infr.gen_edge_attrs(key, edges=edges, default=default,
                                        check_exist=check_exist))

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
        attr_list = list(infr.get_node_attrs(key, aids).values())
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

    def get_edge_data(infr, edge):
        return infr.graph.get_edge_data(*edge)

    def get_nonvisual_edge_data(infr, edge):
        data = infr.get_edge_data(edge)
        if data is not None:
            data = ut.delete_dict_keys(data.copy(), infr.visual_edge_attrs)
        return data


class Convenience(object):
    @staticmethod
    def e_(u, v):
        return e_(u, v)

    @property
    def pos_graph(infr):
        return infr.review_graphs[POSTV]

    @property
    def neg_graph(infr):
        return infr.review_graphs[NEGTV]

    @property
    def incomp_graph(infr):
        return infr.review_graphs[INCMP]

    @property
    def unreviewed_graph(infr):
        return infr.review_graphs[UNREV]

    @property
    def unknown_graph(infr):
        return infr.review_graphs[UNKWN]

    def print_graph_info(infr):
        print(ut.repr3(ut.graph_info(infr.simplify_graph())))


@six.add_metaclass(ut.ReloadingMetaclass)
class DummyEdges(object):
    def apply_mst(infr):
        """
        MST edges connect nodes labeled with the same name.
        This is done in case an explicit feedback or score edge does not exist.
        """
        infr.print('apply_mst', 2)
        infr.ensure_mst()

    def find_clique_edges(infr, label='name_label'):
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())
        new_edges = []
        for label, nodes in label_to_nodes.items():
            for edge in it.combinations(nodes, 2):
                if not infr.has_edge(edge):
                    new_edges.append(edge)
        return new_edges

    def review_dummy_edges(infr, method=1):
        """
        Creates just enough dummy reviews to maintain a consistent labeling if
        relabel_using_reviews is called. (if the existing edges are consistent).
        """
        infr.print('review_dummy_edges', 2)
        if method == 'clique':
            new_edges = infr.find_clique_edges()
        elif method == 1:
            new_edges = infr.find_mst_edges()
        elif method == 3:
            new_edges = infr.find_connecting_edges()
        else:
            raise ValueError('unknown method')
        infr.print('reviewing %s dummy edges' % (len(new_edges),), 1)
        # TODO apply set of new edges in bulk
        for u, v in new_edges:
            infr.add_feedback((u, v), decision=POSTV, confidence='guessing',
                               user_id='dummy', verbose=False)

    def ensure_cliques(infr, label='name_label', decision=UNREV):
        """
        Force each name label to be a clique
        """
        infr.print('ensure_cliques', 1)
        new_edges = infr.find_clique_edges(label)
        infr.print('adding %d clique edges' % (len(new_edges)), 2)
        infr.graph.add_edges_from(new_edges, decision=decision, _dummy_edge=True)
        infr.review_graphs[decision].add_edges_from(new_edges)
        # infr.assert_disjoint_invariant()

    def ensure_full(infr):
        infr.print('ensure_full with %d nodes' % (len(infr.graph)), 2)
        new_edges = list(nx.complement(infr.graph).edges())
        # if infr.verbose:
        #     infr.print('adding %d complement edges' % (len(new_edges)))
        infr.graph.add_edges_from(new_edges, decision=UNREV, _dummy_edge=True)
        infr.review_graphs[UNREV].add_edges_from(new_edges)

    def ensure_mst(infr, decision=POSTV):
        """
        Ensures that all names are names are connected
        """
        infr.print('ensure_mst', 1)
        new_edges = infr.find_mst_edges()
        # Add new MST edges to original graph
        infr.print('adding %d MST edges' % (len(new_edges)), 2)
        for u, v in new_edges:
            infr.add_feedback((u, v), decision=decision, confidence='guessing',
                              user_id='mst', verbose=False)

    def find_connecting_edges(infr):
        """
        Searches for a small set of edges, which if reviewed as positive would
        ensure that each PCC is k-connected.  Note that in somes cases this is
        not possible
        """
        name_attr = 'name_label'
        node_to_label = infr.get_node_attrs(name_attr)
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())

        # k = infr.queue_params['pos_redun']
        k = 1
        new_edges = []
        prog = ut.ProgIter(list(label_to_nodes.keys()),
                           label='finding connecting edges',
                           enabled=infr.verbose > 0)
        for nid in prog:
            nodes = set(label_to_nodes[nid])
            G = infr.pos_graph.subgraph(nodes, dynamic=False)
            impossible = edges_inside(infr.neg_graph, nodes)
            impossible |= edges_inside(infr.incomp_graph, nodes)

            candidates = set(nx.complement(G).edges())
            candidates.difference_update(impossible)

            aug_edges = nx_utils.edge_connected_augmentation(
                G, k=k, candidates=candidates, hack=False)
            new_edges += aug_edges
        prog.ensure_newline()
        return new_edges

    @profile
    def find_mst_edges(infr, name_attr='name_label'):
        """
        Returns edges to augment existing PCCs (by label) in order to ensure
        they are connected with positive edges.

        CommandLine:
            python -m ibeis.algo.graph.mixin_helpers find_mst_edges --profile

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_helpers import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
            >>> infr = ibeis.AnnotInference(ibs, 'all', autoinit=True)
            >>> name_label = 'orig_name_label'
            >>> name_label = 'name_label'
            >>> infr.find_mst_edges()
            >>> infr.ensure_mst()
        """
        from ibeis.algo.graph import nx_utils
        # import networkx as nx
        # Find clusters by labels
        # name_attr = 'orig_name_label'
        # name_attr = 'name_label'
        node_to_label = infr.get_node_attrs(name_attr)
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())

        special_weighting = False
        if special_weighting:
            annots = infr.ibs.annots(infr.aids)
            node_to_time = ut.dzip(annots, annots.time)

        new_edges = []
        prog = ut.ProgIter(list(label_to_nodes.keys()),
                           label='finding mst edges',
                           enabled=infr.verbose > 0)
        for nid in prog:
            nodes = set(label_to_nodes[nid])
            if len(nodes) == 1:
                continue
            # We want to make this CC connected
            pos_sub = infr.pos_graph.subgraph(nodes, dynamic=False)
            impossible = set(it.starmap(e_, it.chain(
                edges_inside(infr.neg_graph, nodes),
                edges_inside(infr.incomp_graph, nodes),
                # edges_inside(infr.unknown_graph, nodes),
            )))
            if impossible or special_weighting:
                complement = it.starmap(e_, nx_utils.complement_edges(pos_sub))
                avail_uv = [
                    (u, v) for u, v in complement if (u, v) not in impossible
                ]
                # TODO: can do special weighting to improve the MST
                if special_weighting:
                    avail_uv = np.array(avail_uv)
                    times = ut.take(node_to_time, nodes)
                    maxtime = vt.safe_max(times, fill=1, nans=False)
                    mintime = vt.safe_min(times, fill=0, nans=False)
                    time_denom = maxtime - mintime

                    # Try linking by time for lynx data
                    comp_weight = 1 - infr.is_comparable(avail_uv)
                    time_delta = np.array([
                        abs(node_to_time[u] - node_to_time[v])
                        for u, v in avail_uv
                    ])
                    time_weight = time_delta / time_denom
                    weights = (10 * comp_weight) + time_weight
                    avail = [(u, v, {'weight': w})
                             for (u, v), w in zip(avail_uv, weights)]
                else:
                    avail = avail_uv
                aug_edges = list(nx_utils.edge_connected_augmentation(
                    pos_sub, k=1, avail=avail))
            else:
                aug_edges = list(nx_utils.edge_connected_augmentation(
                    pos_sub, k=1))
            new_edges.extend(aug_edges)
        prog.ensure_newline()

        for edge in new_edges:
            assert not infr.graph.has_edge(*edge)
        return new_edges


class AssertInvariants(object):
    def assert_invariants(infr, msg=''):
        infr.assert_disjoint_invariant(msg)
        infr.assert_union_invariant(msg)
        infr.assert_consistency_invariant(msg)
        infr.assert_recovery_invariant(msg)

    def assert_union_invariant(infr, msg=''):
        edge_sets = {
            key: set(it.starmap(infr.e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        }
        edge_union = set.union(*edge_sets.values())
        all_edges = set(it.starmap(infr.e_, infr.graph.edges()))
        if edge_union != all_edges:
            print('ERROR STATUS DUMP:')
            print(ut.repr4(infr.status()))
            raise AssertionError(
                'edge sets must have full union. Found union=%d vs all=%d' % (
                    len(edge_union), len(all_edges)
                ))

    def assert_disjoint_invariant(infr, msg=''):
        # infr.print('assert_disjoint_invariant', 200)
        edge_sets = {
            key: set(it.starmap(e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        }
        for es1, es2 in it.combinations(edge_sets.values(), 2):
            assert es1.isdisjoint(es2), 'edge sets must be disjoint'

    def assert_consistency_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        # infr.print('assert_consistency_invariant', 200)
        if infr.enable_inference:
            incon_ccs = list(infr.inconsistent_components())
            with ut.embed_on_exception_context:
                if len(incon_ccs) > 0:
                    raise AssertionError('The graph is not consistent. ' +
                                         msg)

    def assert_recovery_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        # infr.print('assert_recovery_invariant', 200)
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
