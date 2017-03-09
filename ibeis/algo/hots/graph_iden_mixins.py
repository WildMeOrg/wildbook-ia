# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import operator
import utool as ut
import networkx as nx
from ibeis.algo.hots.graph_iden_utils import e_
import six
print, rrr, profile = ut.inject2(__name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrHelpers(object):
    """ Contains non-core helper functions """

    @staticmethod
    def e_(u, v):
        return e_(u, v)

    def gen_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        return ut.util_graph.nx_gen_node_attrs(
                infr.graph, key, nodes=nodes, default=default)

    def gen_edge_attrs(infr, key, edges=None, default=ut.NoParam,
                       check_exist=True):
        return ut.util_graph.nx_gen_edge_attrs(
                infr.graph, key, edges=edges, default=default,
                check_exist=check_exist)

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
        redge = edge[::-1]
        flag = infr.graph.has_edge(*edge) or infr.graph.has_edge(*redge)
        return flag

    def get_edge_data(infr, u, v):
        data = infr.graph.get_edge_data(u, v)
        if data is not None:
            data = ut.delete_dict_keys(data.copy(), infr.visual_edge_attrs)
        return data

    def print_graph_info(infr):
        print(ut.repr3(ut.graph_info(infr.simplify_graph())))
