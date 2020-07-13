# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import vtool as vt
import pandas as pd
from wbia.algo.graph.nx_utils import ensure_multi_index
from wbia.algo.graph.state import POSTV, NEGTV, INCMP

print, rrr, profile = ut.inject2(__name__)


class Groundtruth(object):
    def is_comparable(infr, aid_pairs, allow_guess=True):
        """
        Guesses by default when real comparable information is not available.
        """
        if infr.ibs is not None:
            return infr.wbia_is_comparable(aid_pairs, allow_guess)
        is_comp = list(
            infr.gen_edge_values(
                'gt_comparable', edges=aid_pairs, default=True, on_missing='default'
            )
        )
        return np.array(is_comp)

    def is_photobomb(infr, aid_pairs):
        if infr.ibs is not None:
            return infr.wbia_is_photobomb(aid_pairs)
        return np.array([False] * len(aid_pairs))

    def is_same(infr, aid_pairs):
        if infr.ibs is not None:
            return infr.wbia_is_same(aid_pairs)
        node_dict = ut.nx_node_dict(infr.graph)
        nid1 = [node_dict[n1]['orig_name_label'] for n1, n2 in aid_pairs]
        nid2 = [node_dict[n2]['orig_name_label'] for n1, n2 in aid_pairs]
        return np.equal(nid1, nid2)

    def apply_edge_truth(infr, edges=None):
        if edges is None:
            edges = list(infr.edges())
        edge_truth_df = infr.match_state_df(edges)
        edge_truth = edge_truth_df.idxmax(axis=1).to_dict()
        infr.set_edge_attrs('truth', edge_truth)
        infr.edge_truth.update(edge_truth)

    def match_state_df(infr, index):
        """ Returns groundtruth state based on wbia controller """
        index = ensure_multi_index(index, ('aid1', 'aid2'))
        aid_pairs = np.asarray(index.tolist())
        aid_pairs = vt.ensure_shape(aid_pairs, (None, 2))
        is_same = infr.is_same(aid_pairs)
        is_comp = infr.is_comparable(aid_pairs)
        match_state_df = pd.DataFrame.from_items(
            [(NEGTV, ~is_same & is_comp), (POSTV, is_same & is_comp), (INCMP, ~is_comp)]
        )
        match_state_df.index = index
        return match_state_df

    def match_state_gt(infr, edge):
        if edge in infr.edge_truth:
            truth = infr.edge_truth[edge]
        elif hasattr(infr, 'dummy_verif'):
            truth = infr.dummy_verif._get_truth(edge)
        else:
            aid_pairs = np.asarray([edge])
            is_same = infr.is_same(aid_pairs)[0]
            is_comp = infr.is_comparable(aid_pairs)[0]
            match_state = pd.Series(
                dict(
                    [
                        (NEGTV, ~is_same & is_comp),
                        (POSTV, is_same & is_comp),
                        (INCMP, ~is_comp),
                    ]
                )
            )
            truth = match_state.idxmax()
        return truth

    def edge_attr_df(infr, key, edges=None, default=ut.NoParam):
        """ constructs DataFrame using current predictions """
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
