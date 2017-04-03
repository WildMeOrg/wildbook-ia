# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import operator
import utool as ut
import networkx as nx
from ibeis.algo.hots.graph_iden_utils import e_
from ibeis.algo.hots.graph_iden_utils import ensure_multi_index
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

    def print_graph_info(infr):
        print(ut.repr3(ut.graph_info(infr.simplify_graph())))


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrGroundtruth(object):
    """
    Methods for generating training labels for classifiers
    """
    def guess_if_comparable(infr, aid_pairs):
        """
        Takes a guess as to which annots are not comparable based on scores and
        viewpoints. If either viewpoints is null assume they are comparable.
        """
        # simple_scores = labels.simple_scores
        # key = 'sum(weighted_ratio)'
        # if key not in simple_scores:
        #     key = 'sum(ratio)'
        # scores = simple_scores[key].values
        # yaws1 = labels.annots1.yaws_asfloat
        # yaws2 = labels.annots2.yaws_asfloat
        aid_pairs = np.asarray(aid_pairs)
        ibs = infr.ibs
        yaws1 = ibs.get_annot_yaws_asfloat(aid_pairs.T[0])
        yaws2 = ibs.get_annot_yaws_asfloat(aid_pairs.T[1])
        dists = vt.ori_distance(yaws1, yaws2)
        tau = np.pi * 2
        # scores = np.full(len(aid_pairs), np.nan)
        comp_by_viewpoint = (dists < tau / 8.1) | np.isnan(dists)
        # comp_by_score = (scores > .1)
        # is_comp = comp_by_score | comp_by_viewpoint
        is_comp_guess = comp_by_viewpoint
        return is_comp_guess

    def is_comparable(infr, aid_pairs, allow_guess=True):
        """
        Guesses by default when real comparable information is not available.
        """
        ibs = infr.ibs
        if allow_guess:
            # Guess if comparability information is unavailable
            is_comp_guess = infr.guess_if_comparable(aid_pairs)
            is_comp = is_comp_guess.copy()
        else:
            is_comp = np.full(len(aid_pairs), np.nan)
        # But use information that we have
        am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        truths = ut.replace_nones(ibs.get_annotmatch_truth(am_rowids), np.nan)
        truths = np.asarray(truths)
        is_notcomp_have = truths == ibs.const.REVIEW.NOT_COMPARABLE
        is_comp_have = ((truths == ibs.const.REVIEW.MATCH) |
                        (truths == ibs.const.REVIEW.NON_MATCH))
        is_comp[is_notcomp_have] = False
        is_comp[is_comp_have] = True
        return is_comp

    def is_photobomb(infr, aid_pairs):
        ibs = infr.ibs
        am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        am_tags = ibs.get_annotmatch_case_tags(am_rowids)
        is_pb = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
        return is_pb

    def is_same(infr, aid_pairs):
        aids1, aids2 = np.asarray(aid_pairs).T
        nids1 = infr.ibs.get_annot_nids(aids1)
        nids2 = infr.ibs.get_annot_nids(aids2)
        is_same = (nids1 == nids2)
        return is_same

    def match_state_df(infr, index):
        """ Returns groundtruth state based on ibeis controller """
        import pandas as pd
        index = ensure_multi_index(index, ('aid1', 'aid2'))
        aid_pairs = np.asarray(index.tolist())
        aid_pairs = vt.ensure_shape(aid_pairs, (None, 2))
        is_same = infr.is_same(aid_pairs)
        is_comp = infr.is_comparable(aid_pairs)
        match_state_df = pd.DataFrame.from_items([
            ('nomatch', ~is_same & is_comp),
            ('match',    is_same & is_comp),
            ('notcomp', ~is_comp),
        ])
        match_state_df.index = index
        return match_state_df

    def match_state_gt(infr, edge):
        import pandas as pd
        aid_pairs = np.asarray([edge])
        is_same = infr.is_same(aid_pairs)[0]
        is_comp = infr.is_comparable(aid_pairs)[0]
        match_state = pd.Series(dict([
            ('nomatch', ~is_same & is_comp),
            ('match',    is_same & is_comp),
            ('notcomp', ~is_comp),
        ]))
        return match_state

    def edge_attr_df(infr, key, edges=None, default=ut.NoParam):
        """ constructs DataFrame using current predictions """
        import pandas as pd
        edge_states = infr.gen_edge_attrs(key, edges=edges, default=default)
        edge_states = list(edge_states)
        if isinstance(edges, pd.MultiIndex):
            index = edges
        else:
            if edges is None:
                edges_ = ut.take_column(edge_states, 0)
            else:
                edges_ = ut.lmap(tuple, ut.aslist(edges))
            index = pd.MultiIndex.from_tuples(edges_, names=('aid1', 'aid2'))
        records = ut.itake_column(edge_states, 1)
        edge_df = pd.Series.from_array(records)
        edge_df.name = key
        edge_df.index = index
        return edge_df

    def infr_pred_df(infr, edges=None):
        """ technically not groundtruth but current infererence predictions """
        return infr.edge_attr_df('inferred_state', edges, default=np.nan)
