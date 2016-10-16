# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt  # NOQA
import six
from ibeis.algo.hots import viz_graph_iden
from ibeis.algo.hots import infr_model
import networkx as nx
print, rrr, profile = ut.inject2(__name__)

# Monkey patch networkx
nx.set_edge_attrs = nx.set_edge_attributes
nx.get_edge_attrs = nx.get_edge_attributes
nx.set_node_attrs = nx.set_node_attributes
nx.get_node_attrs = nx.get_node_attributes


def _dz(a, b):
    a = a.tolist() if isinstance(a, np.ndarray) else list(a)
    b = b.tolist() if isinstance(b, np.ndarray) else list(b)
    if len(a) == 0 and len(b) == 1:
        # This introduces a corner case
        b = []
    elif len(b) == 1 and len(a) > 1:
        b = b * len(a)
    assert len(a) == len(b), 'out of alignment a=%r, b=%r' % (a, b)
    return dict(zip(a, b))


def get_cm_breaking(qreq_, cm_list, ranks_top=None, ranks_bot=None):
    """
        >>> from ibeis.algo.hots.graph_iden import *  # NOQA
    """
    # Construct K-broken graph
    edges = []

    if ranks_bot is None:
        ranks_bot = 0

    for count, cm in enumerate(cm_list):
        score_list = cm.annot_score_list
        rank_list = ut.argsort(score_list)[::-1]
        sortx = ut.argsort(rank_list)

        top_sortx = sortx[:ranks_top]
        bot_sortx = sortx[-ranks_bot:]
        short_sortx = ut.unique(top_sortx + bot_sortx)

        daid_list = ut.take(cm.daid_list, short_sortx)
        for daid in daid_list:
            u, v = (cm.qaid, daid)
            if v < u:
                u, v = v, u
            edges.append((u, v))
    return edges


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInference(ut.NiceRepr, viz_graph_iden.AnnotInferenceVisualization):
    """
    Sandbox class for maintaining state of an identification

    TODO:
        * Accept external query results
        * Accept external feedback
        * Return filtered edges

    Notes:
        General workflow goes
        * Initialize Step
            * Add annots/names/configs/matches to AnnotInference Object
            * Apply Edges (mst/matches/feedback)
            * Apply Scores
            * Apply Weights
            * Apply Inference
        * Review Step
            * TODO: Get shortlist of results
            * Present results to user
            * Apply user feedback
            * Apply Inference
            * Record results
            * Repeat

    Terminology and Concepts:

        Each node contains:
            * annotation id
            * original name label
            * current name label
            * feature vector(s)

        Each edge contains:
            * raw matching scores
            * pairwise features for learning
            * probability match/notmatch/notcomp
            * probability same/diff | features
            * User feedback:
                match - confidence / trust
                notmatch - confidence / trust
                notcomp - confidence / trust
            * probability same/diff | feedback

        * MST Edge - connects two nodes that with the same name label
                     that would otherwise have been separated in the graph.
                     This essentially corresponds to a low-trust feedback edge
                     because somebody marked these two as the same at one
                     point.

    CommandLine:
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7
        ibeis AnnotInference:0 --show
        ibeis AnnotInference:1 --show
        ibeis AnnotInference:2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.graph_iden import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aids = [1, 2, 3, 4, 5, 6]
        >>> infr = AnnotInference(ibs, aids, autoinit=True)
        >>> result = ('infr = %s' % (infr,))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> use_image = True
        >>> infr.initialize_visual_node_attrs()
        >>> # Note that there are initially no edges
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()
        infr = <AnnotInference(nAids=6, nEdges=0)>

    Example:
        >>> # SCRIPT
        >>> from ibeis.algo.hots.graph_iden import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aids = [1, 2, 3, 4, 5, 6, 7, 9]
        >>> infr = AnnotInference(ibs, aids, autoinit=True)
        >>> result = ('infr = %s' % (infr,))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> use_image = False
        >>> infr.initialize_visual_node_attrs()
        >>> # Note that there are initially no edges
        >>> infr.show_graph(use_image=use_image)
        >>> # But we can add nodes between the same names
        >>> infr.apply_mst()
        >>> infr.show_graph(use_image=use_image)
        >>> # Add some feedback
        >>> infr.add_feedback(1, 4, 'nomatch')
        >>> infr.apply_feedback_edges()
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()
        infr = <AnnotInference(nAids=6, nEdges=0)>

    Example:
        >>> # SCRIPT
        >>> from ibeis.algo.hots.graph_iden import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aids = [1, 2, 3, 4, 5, 6, 7, 9]
        >>> infr = AnnotInference(ibs, aids, autoinit=True)
        >>> result = ('infr = %s' % (infr,))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> use_image = False
        >>> infr.initialize_visual_node_attrs()
        >>> infr.apply_mst()
        >>> # Add some feedback
        >>> infr.add_feedback(1, 4, 'nomatch')
        >>> try:
        >>>     infr.add_feedback(1, 10, 'nomatch')
        >>> except ValueError:
        >>>     pass
        >>> try:
        >>>     infr.add_feedback(11, 12, 'nomatch')
        >>> except ValueError:
        >>>     pass
        >>> infr.apply_feedback_edges()
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()
        infr = <AnnotInference(nAids=6, nEdges=0)>

    """

    CUT_WEIGHT_KEY = 'cut_weight'

    truth_texts = {
        0: 'nomatch',
        1: 'match',
        2: 'notcomp',
        3: 'unreviewed',
    }

    def __init__(infr, ibs, aids, nids=None, autoinit=False, verbose=False):
        infr.verbose = verbose
        if infr.verbose:
            print('[infr] __init__')
        infr.ibs = ibs
        infr.aids = aids
        infr.aids_set = set(infr.aids)
        if nids is None:
            nids = ibs.get_annot_nids(aids)
        if ut.isscalar(nids):
            nids = [nids] * len(aids)
        infr.orig_name_labels = nids
        #if current_nids is None:
        #    current_nids = nids
        assert len(aids) == len(nids), 'must correspond'
        #assert len(aids) == len(current_nids)
        infr.graph = None
        infr.user_feedback = ut.ddict(list)
        infr.thresh = .5
        infr.cm_list = None
        infr.qreq_ = None
        if autoinit:
            infr.initialize_graph()

    @classmethod
    def from_qreq_(cls, qreq_, cm_list):
        """
        Create a AnnotInference object using a precomputed query / results
        """
        # raise NotImplementedError('do not use')
        aids = ut.unique(ut.flatten([qreq_.qaids, qreq_.daids]))
        nids = qreq_.get_qreq_annot_nids(aids)
        ibs = qreq_.ibs
        infr = cls(ibs, aids, nids, verbose=False)
        infr.cm_list = cm_list
        infr.qreq_ = qreq_
        return infr

    def __nice__(infr):
        if infr.graph is None:
            return 'nAids=%r, G=None' % (len(infr.aids))
        else:
            return 'nAids=%r, nEdges=%r' % (len(infr.aids),
                                              infr.graph.number_of_edges())

    def initialize_graph(infr):
        if infr.verbose:
            print('[infr] initialize_graph')
        #infr.graph = graph = nx.DiGraph()
        infr.graph = graph = nx.Graph()
        graph.add_nodes_from(infr.aids)

        node_to_aid = {aid: aid for aid in infr.aids}
        infr.node_to_aid = node_to_aid
        node_to_nid = {aid: nid for aid, nid in
                       zip(infr.aids, infr.orig_name_labels)}
        assert len(node_to_nid) == len(node_to_aid), '%r - %r' % (
            len(node_to_nid), len(node_to_aid))
        nx.set_node_attrs(graph, 'aid', node_to_aid)
        nx.set_node_attrs(graph, 'name_label', node_to_nid)
        nx.set_node_attrs(graph, 'orig_name_label', node_to_nid)
        infr.aid_to_node = ut.invert_dict(infr.node_to_aid)

    def connected_component_reviewed_subgraphs(infr):
        """
        Two kinds of edges are considered in connected component analysis: user
        reviewed edges, and algorithmally inferred edges.  If an inference
        algorithm is not run, then user review is all that matters.
        """
        graph = infr.graph
        # Make a graph where connections do indicate same names
        graph2 = graph.copy()
        reviewed_states = nx.get_edge_attrs(graph, 'reviewed_state')
        keep_edges = [key for key, val in reviewed_states.items()
                      if val == 'match']
        graph2.remove_edges_from(list(graph2.edges()))
        graph2.add_edges_from(keep_edges)
        ccs = list(nx.connected_components(graph2))
        cc_subgraphs = [graph.subgraph(cc) for cc in ccs]
        return cc_subgraphs

    def connected_component_status(infr):
        r"""
        Returns:
            tuple: (num_names, num_inconsistent)

        CommandLine:
            python -m ibeis connected_component_status --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback(2, 3, 'nomatch')
            >>> infr.add_feedback(5, 6, 'nomatch')
            >>> infr.add_feedback(1, 2, 'match')
            >>> infr.apply_feedback_edges()
            >>> status = infr.connected_component_status()
            >>> print(ut.repr3(status))
        """
        cc_subgraphs = infr.connected_component_reviewed_subgraphs()
        num_names_max = len(cc_subgraphs)

        ccx_to_aids = {
            ccx: list(nx.get_node_attributes(cc, 'aid').values())
            for ccx, cc in enumerate(cc_subgraphs)
        }
        aid_to_ccx = {
            aid: ccx for ccx, aids in ccx_to_aids.items() for aid in aids
        }

        all_reviewed_states = nx.get_edge_attrs(infr.graph, 'reviewed_state')
        separated_ccxs = set([])
        inconsistent_ccxs = set([])
        for edge, state in all_reviewed_states.items():
            if state == 'nomatch':
                ccx1 = aid_to_ccx[edge[0]]
                ccx2 = aid_to_ccx[edge[1]]
                # Determine number of negative matches within a component
                if ccx1 == ccx2:
                    inconsistent_ccxs.add(ccx1)
                # Determine the number of components that should not be joined
                if ccx1 > ccx2:
                    ccx1, ccx2 = ccx2, ccx1
                separated_ccxs.add((ccx1, ccx2))

        num_names_min = ut.approx_min_num_components(infr.aids, separated_ccxs)

        status = dict(
            num_names_max=num_names_max,
            num_inconsistent=len(inconsistent_ccxs),
            num_names_min=num_names_min,
        )

        return status

    def connected_component_reviewed_relabel(infr):
        if infr.verbose:
            print('[infr] connected_component_reviewed_relabel')
        cc_subgraphs = infr.connected_component_reviewed_subgraphs()
        num_inconsistent = 0
        num_names = len(cc_subgraphs)

        for count, subgraph in enumerate(cc_subgraphs):
            reviewed_states = nx.get_edge_attrs(subgraph, 'reviewed_state')
            inconsistent_edges = [edge for edge, val in reviewed_states.items()
                                  if val == 'nomatch']
            if len(inconsistent_edges) > 0:
                #print('Inconsistent')
                num_inconsistent += 1

            nx.set_node_attrs(infr.graph, 'name_label',
                              _dz(list(subgraph.nodes()), [count]))
            # Check for consistency
        return num_names, num_inconsistent

    def read_user_feedback(infr):
        """
        Loads feedback from annotmatch table

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> user_feedback = infr.read_user_feedback()
            >>> result =('user_feedback = %s' % (ut.repr2(user_feedback, nl=1),))
            >>> print(result)
            user_feedback = {
                (2, 3): [{'p_match': 0.0, 'p_nomatch': 1.0, 'p_notcomp': 0.0}],
                (5, 6): [{'p_match': 0.0, 'p_nomatch': 1.0, 'p_notcomp': 0.0}],
            }
        """
        if infr.verbose:
            print('[infr] read_user_feedback')
        ibs = infr.ibs
        annots = ibs.annots(infr.aids)
        am_rowids, aid_pairs = annots.get_am_rowids_and_pairs()
        aids1 = ut.take_column(aid_pairs, 0)
        aids2 = ut.take_column(aid_pairs, 1)

        # Use tags to infer truth
        props = ['SplitCase', 'JoinCase', 'Photobomb']
        flags_list = ibs.get_annotmatch_prop(props, am_rowids)
        is_split, is_merge, is_pb = flags_list
        is_split = np.array(is_split).astype(np.bool)
        is_merge = np.array(is_merge).astype(np.bool)
        is_pb = np.array(is_pb).astype(np.bool)

        # Use explicit truth state to mark truth
        truth = np.array(ibs.get_annotmatch_truth(am_rowids))
        # Hack, if we didnt set it, it probably means it matched
        need_truth = np.array(ut.flag_None_items(truth)).astype(np.bool)
        need_aids1 = ut.compress(aids1, need_truth)
        need_aids2 = ut.compress(aids2, need_truth)
        needed_truth = ibs.get_aidpair_truths(need_aids1, need_aids2)
        truth[need_truth] = needed_truth

        # Add information from relevant tags
        truth = np.array(truth, dtype=np.int)
        truth[is_split] = ibs.const.TRUTH_NOT_MATCH
        truth[is_pb] = ibs.const.TRUTH_NOT_MATCH
        truth[is_merge] = ibs.const.TRUTH_MATCH

        p_match = (truth == ibs.const.TRUTH_MATCH).astype(np.float)
        p_nomatch = (truth == ibs.const.TRUTH_NOT_MATCH).astype(np.float)
        p_notcomp = (truth == ibs.const.TRUTH_UNKNOWN).astype(np.float)

        # CHANGE OF FORMAT
        user_feedback = ut.ddict(list)
        for count, (aid1, aid2) in enumerate(zip(aids1, aids2)):
            edge = tuple(sorted([aid1, aid2]))
            review = {
                'p_match': p_match[count],
                'p_nomatch': p_nomatch[count],
                'p_notcomp': p_notcomp[count],
            }
            user_feedback[edge].append(review)
        return user_feedback

    #@staticmethod
    def _pandas_feedback_format(infr, user_feedback):
        import pandas as pd
        aid_pairs = list(user_feedback.keys())
        aids1 = ut.take_column(aid_pairs, 0)
        aids2 = ut.take_column(aid_pairs, 1)
        ibs = infr.ibs

        am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
        #am_rowids = np.array(ut.replace_nones(am_rowids, np.nan))
        probs_ = list(user_feedback.values())
        probs = ut.take_column(probs_, -1)
        df = pd.DataFrame.from_dict(probs)
        df['aid1'] = aids1
        df['aid2'] = aids2
        df['am_rowid'] = am_rowids
        df.set_index('am_rowid')
        df.index = pd.Index(am_rowids, name='am_rowid')
        #df.index = pd.Index(aid_pairs, name=('aid1', 'aid2'))
        return df

    def match_state_delta(infr):
        """ Returns information about state change of annotmatches """
        old_feedback = infr._pandas_feedback_format(infr.read_user_feedback())
        new_feedback = infr._pandas_feedback_format(infr.user_feedback)
        new_df, old_df = infr._make_state_delta(old_feedback, new_feedback)
        return new_df, old_df

    @staticmethod
    def _make_state_delta(old_feedback, new_feedback):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> import pandas as pd
            >>> old_data = [
            >>>     [1, 0, 0, 100, 101, 1000],
            >>>     [0, 1, 0, 101, 102, 1001],
            >>>     [0, 1, 0, 103, 104, 1003],
            >>>     [1, 0, 0, 101, 104, 1004],
            >>> ]
            >>> new_data = [
            >>>     [1, 0, 0, 101, 102, 1001],
            >>>     [0, 1, 0, 103, 104, 1002],
            >>>     [0, 1, 0, 101, 104, 1003],
            >>>     [1, 0, 0, 102, 103, None],
            >>>     [1, 0, 0, 100, 103, None],
            >>>     [0, 0, 1, 107, 109, None],
            >>> ]
            >>> columns = ['p_match', 'p_nomatch', 'p_noncomp', 'aid1', 'aid2', 'am_rowid']
            >>> old_feedback = pd.DataFrame(old_data, columns=columns)
            >>> new_feedback = pd.DataFrame(new_data, columns=columns)
            >>> old_feedback.set_index('am_rowid', inplace=True, drop=False)
            >>> new_feedback.set_index('am_rowid', inplace=True, drop=False)
            >>> new_df, old_df = AnnotInference._make_state_delta(old_feedback, new_feedback)
            >>> # post
            >>> is_add = np.isnan(new_df['am_rowid'].values)
            >>> add_df = new_df.loc[is_add]
            >>> add_ams = [2000, 2001, 2002]
            >>> new_df.loc[is_add, 'am_rowid'] = add_ams
            >>> new_df.set_index('am_rowid', drop=False, inplace=True)
        """
        import pandas as pd
        existing_ams = new_feedback['am_rowid'][~np.isnan(new_feedback['am_rowid'])]
        both_ams = np.intersect1d(old_feedback['am_rowid'], existing_ams).astype(np.int)

        all_new_df = new_feedback.loc[both_ams]
        all_old_df = old_feedback.loc[both_ams]
        is_changed = ~np.all(all_new_df.values == all_old_df.values, axis=1)

        new_df_ = all_new_df[is_changed]
        add_df = new_feedback.loc[np.isnan(new_feedback['am_rowid'])].copy()

        old_df = all_old_df[is_changed]
        new_df = pd.concat([new_df_, add_df])
        return new_df, old_df

    def lookup_cm(infr, aid1, aid2):
        if infr.cm_list is None:
            return None, aid1, aid2
        aid2_idx = ut.make_index_lookup(
            [cm.qaid for cm in infr.cm_list])
        try:
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
        except KeyError:
            # switch order
            aid1, aid2 = aid2, aid1
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
        return cm, aid1, aid2

    def get_feedback_probs(infr):
        """ Helper """
        unique_pairs = list(infr.user_feedback.keys())
        # Take most recent review
        review_list = [infr.user_feedback[edge][-1] for edge in unique_pairs]
        p_nomatch = np.array(ut.dict_take_column(review_list, 'p_nomatch'))
        p_match = np.array(ut.dict_take_column(review_list, 'p_match'))
        p_notcomp = np.array(ut.dict_take_column(review_list, 'p_notcomp'))
        state_probs = np.vstack([p_nomatch, p_match, p_notcomp])
        review_stateid = state_probs.argmax(axis=0)
        review_state = ut.take(infr.truth_texts, review_stateid)
        p_bg = 0.5  # Needs to be thresh value
        part1 = p_match * (1 - p_notcomp)
        part2 = p_bg * p_notcomp
        p_same_list = part1 + part2
        return p_same_list, unique_pairs, review_state

    def get_edge_attr(infr, key):
        return nx.get_edge_attributes(infr.graph, key)

    def get_node_attr(infr, key):
        return nx.get_node_attributes(infr.graph, key)

    def reset_name_labels(infr):
        """ Changes annotation names labels back to their initial values """
        if infr.verbose:
            print('[infr] reset_name_labels')
        graph = infr.graph
        orig_names = infr.get_node_attrs('orig_name_label')
        nx.set_node_attrs(graph, 'name_label', orig_names)

    def reset_feedback(infr):
        """ Resets feedback edges to state of the SQL annotmatch table """
        if infr.verbose:
            print('[infr] reset_feedback')
        infr.user_feedback = infr.read_user_feedback()

    def remove_feedback(infr):
        """ Deletes all feedback """
        if infr.verbose:
            print('[infr] remove_feedback')
        infr.user_feedback = ut.ddict(list)

    def remove_name_labels(infr):
        if infr.verbose:
            print('[infr] remove_name_labels()')
        graph = infr.graph
        # make distinct names for all nodes
        distinct_names = {node: -graph.node[node]['aid']
                          for node in graph.nodes()}
        nx.set_node_attrs(graph, 'name_label', distinct_names)

    def remove_mst_edges(infr):
        if infr.verbose:
            print('[infr] remove_mst_edges')
        graph = infr.graph
        edge_to_ismst = nx.get_edge_attrs(graph, '_mst_edge')
        mst_edges = [edge for edge, flag in edge_to_ismst.items() if flag]
        graph.remove_edges_from(mst_edges)

    def add_feedback(infr, aid1, aid2, state):
        """ External helper """
        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))
        if infr.verbose:
            print('[infr] add_feedback(%r, %r, %r)' % (aid1, aid2, state))
        edge = tuple(sorted([aid1, aid2]))
        if isinstance(state, dict):
            assert 'p_match' in state
            assert 'p_nomatch' in state
            assert 'p_notcomp' in state
            review = state
            infr.user_feedback[edge].append(review)
        elif state == 'unreviewed':
            if edge in infr.user_feedback:
                del infr.user_feedback[edge]
        else:
            review = {
                'p_match': 0.0,
                'p_nomatch': 0.0,
                'p_notcomp': 0.0,
            }
            if state == 'match':
                review['p_match'] = 1.0
            elif state == 'nomatch':
                review['p_nomatch'] = 1.0
            elif state == 'notcomp':
                review['p_notcomp'] = 1.0
            else:
                msg = 'state=%r is unknown' % (state,)
                print(msg)
                assert state in infr.truth_texts.values(), msg
            infr.user_feedback[edge].append(review)

    def apply_mst(infr):
        """
        MST edges connect nodes labeled with the same name.
        This is done in case an explicit feedback or score edge does not exist.
        """
        if infr.verbose:
            print('[infr] apply_mst')
        # Remove old MST edges
        infr.remove_mst_edges()
        infr.ensure_mst()

    def ensure_mst(infr):
        """
        Use minimum spannning tree to ensure all names are connected

        Needs to be applied after any operation that adds/removes edges
        """
        if infr.verbose:
            print('[infr] ensure_mst')
        import networkx as nx
        # Find clusters by labels
        node_to_label = infr.get_node_attrs('name_label')
        label2_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())

        aug_graph = infr.graph.copy().to_undirected()

        # remove cut edges
        edge_to_iscut = nx.get_edge_attrs(aug_graph, 'is_cut')
        cut_edges = [edge for edge, flag in edge_to_iscut.items() if flag]
        aug_graph.remove_edges_from(cut_edges)

        # Enumerate cliques inside labels
        unflat_edges = [list(ut.itertwo(nodes))
                        for nodes in label2_nodes.values()]
        node_pairs = [tup for tup in ut.iflatten(unflat_edges)
                      if tup[0] != tup[1]]

        # Remove candidate MST edges that exist in the original graph
        orig_edges = list(aug_graph.edges())
        candidate_mst_edges = [edge for edge in node_pairs
                               if not aug_graph.has_edge(*edge)]
        # randomness prevents chains and visually looks better
        rng = np.random.RandomState(42)
        aug_graph.add_edges_from(candidate_mst_edges)
        # Weight edges in aug_graph such that existing edges are chosen
        # to be part of the MST first before suplementary edges.
        nx.set_edge_attributes(aug_graph, 'weight',
                               {edge: 0.1 for edge in orig_edges})
        nx.set_edge_attributes(aug_graph, 'weight',
                               {edge: 10.0 + rng.randint(1, 100)
                                for edge in candidate_mst_edges})
        new_mst_edges = []
        if infr.verbose:
            print('[infr] adding %d MST edges' % (len(new_mst_edges)))
        for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
            mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
            for edge in mst_sub_graph.edges():
                redge = edge[::-1]
                # Only add if this edge is not in the original graph
                if not (infr.graph.has_edge(*edge) and infr.graph.has_edge(*redge)):
                    new_mst_edges.append(redge)

        # Add new MST edges to original graph
        infr.graph.add_edges_from(new_mst_edges)
        nx.set_edge_attrs(infr.graph, '_mst_edge', _dz(new_mst_edges, [True]))

    def apply_match_scores(infr):
        """
        Applies precomputed matching scores to edges that already exist in the
        graph. Typically you should run infr.apply_match_edges() before running
        this.

        CommandLine:
            python -m ibeis.algo.hots.graph_iden apply_match_scores --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching()
            >>> infr.apply_match_edges()
            >>> infr.apply_match_scores()
            >>> infr.get_edge_attr('score')
        """
        if infr.verbose:
            print('[infr] apply_match_scores')

        if infr.cm_list is None:
            print('[infr] no scores to apply!')
            return

        symmetric = True

        # Find scores for the edges that exist in the graph
        edge_to_data = ut.ddict(dict)
        edges = list(infr.graph.edges())
        node_to_cm = {infr.aid_to_node[cm.qaid]: cm for cm in infr.cm_list}
        for u, v in edges:
            if symmetric and u > v:
                u, v = v, u
            cm1 = node_to_cm.get(u, None)
            cm2 = node_to_cm.get(v, None)
            scores = []
            ranks = []
            for cm in ut.filter_Nones([cm1, cm2]):
                for node in [u, v]:
                    aid = infr.node_to_aid[node]
                    idx = cm.daid2_idx.get(aid, None)
                    if idx is None:
                        continue
                    score = cm.annot_score_list[idx]
                    rank = cm.get_annot_ranks([aid])[0]
                    scores.append(score)
                    ranks.append(rank)
            if len(scores) == 0:
                score = None
                rank = None
            else:
                rank = vt.safe_min(ranks)
                score = np.nanmean(scores)
            edge_to_data[(u, v)]['score'] = score
            edge_to_data[(u, v)]['rank'] = rank

        # Remove existing attrs
        ut.nx_delete_edge_attr(infr.graph, 'score')
        ut.nx_delete_edge_attr(infr.graph, 'rank')
        ut.nx_delete_edge_attr(infr.graph, 'normscore')

        edges = list(edge_to_data.keys())
        edge_scores = list(ut.take_column(edge_to_data.values(), 'score'))
        edge_scores = ut.replace_nones(edge_scores, np.nan)
        edge_scores = np.array(edge_scores)
        edge_ranks = np.array(ut.take_column(edge_to_data.values(), 'rank'))
        # take the inf-norm
        normscores = edge_scores / vt.safe_max(edge_scores, nans=False)

        # Add new attrs
        nx.set_edge_attrs(infr.graph, 'score', dict(zip(edges, edge_scores)))
        nx.set_edge_attrs(infr.graph, 'rank', dict(zip(edges, edge_ranks)))
        nx.set_edge_attrs(infr.graph, 'normscore', dict(zip(edges, normscores)))

    def apply_match_edges(infr, review_cfg={}):
        if infr.verbose:
            print('[infr] apply_match_edges')

        if infr.cm_list is None:
            print('[infr] matching has not been run!')
            return

        qreq_ = infr.qreq_
        cm_list = infr.cm_list
        ranks_top = review_cfg.get('ranks_top', None)
        ranks_bot = review_cfg.get('ranks_bot', None)
        edges = get_cm_breaking(qreq_, cm_list,
                                ranks_top=ranks_top,
                                ranks_bot=ranks_bot)
        # Create match-based graph structure
        infr.remove_mst_edges()
        infr.graph.add_edges_from(edges)
        infr.ensure_mst()

    def apply_feedback_edges(infr):
        """
        Updates nx graph edge attributes for feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.reset_feedback()
            >>> infr.apply_feedback_edges()
            >>> result = str(infr)
            >>> print(result)
            <AnnotInference(nAids=6, nEdges=2)>
        """
        if infr.verbose:
            print('[infr] apply_feedback_edges')
        infr.remove_mst_edges()

        ut.nx_delete_edge_attr(infr.graph, 'reviewed_weight')
        ut.nx_delete_edge_attr(infr.graph, 'reviewed_state')
        p_same_list, unique_pairs_, review_state = infr.get_feedback_probs()
        # Put pair orders in context of the graph
        unique_pairs = [(aid2, aid1) if infr.graph.has_edge(aid2, aid1) else
                        (aid1, aid2) for (aid1, aid2) in unique_pairs_]
        # Ensure edges exist
        for edge in unique_pairs:
            if not infr.graph.has_edge(*edge):
                #print('add review edge = %r' % (edge,))
                infr.graph.add_edge(*edge)
            #else:
            #    #print('have edge edge = %r' % (edge,))
        nx.set_edge_attrs(infr.graph, 'reviewed_state',
                          _dz(unique_pairs, review_state))
        nx.set_edge_attrs(infr.graph, 'reviewed_weight',
                          _dz(unique_pairs, p_same_list))

        infr.ensure_mst()

    def apply_weights(infr):
        """
        Combines scores and user feedback into edge weights used in inference.
        """
        if infr.verbose:
            print('[infr] apply_weights')
        ut.nx_delete_edge_attr(infr.graph, 'cut_weight')
        # mst not needed. No edges are removed

        edges = list(infr.graph.edges())
        edge2_normscore = nx.get_edge_attrs(infr.graph, 'normscore')
        normscores = np.array(ut.dict_take(edge2_normscore, edges, np.nan))

        edge2_reviewed_weight = nx.get_edge_attrs(infr.graph, 'reviewed_weight')
        reviewed_weights = np.array(ut.dict_take(edge2_reviewed_weight,
                                                 edges, np.nan))
        # Combine into weights
        weights = normscores.copy()
        has_review = ~np.isnan(reviewed_weights)
        weights[has_review] = reviewed_weights[has_review]
        # remove nans
        is_valid = ~np.isnan(weights)
        weights = weights.compress(is_valid, axis=0)
        edges = ut.compress(edges, is_valid)
        nx.set_edge_attrs(infr.graph, 'cut_weight', _dz(edges, weights))

    def get_node_attrs(infr, key, nodes=None):
        node_to_label = nx.get_node_attributes(infr.graph, key)
        if nodes is not None:
            node_to_label = ut.dict_subset(node_to_label, nodes)
        return node_to_label

    def get_annot_attrs(infr, key, aids):
        nodes = ut.take(infr.aid_to_node, aids)
        attr_list = list(infr.get_node_attrs(key, nodes).values())
        return attr_list

    def apply_cuts(infr):
        """
        Cuts edges with different names and uncuts edges with the same name.
        """
        if infr.verbose:
            print('[infr] apply_cuts')
        graph = infr.graph
        infr.ensure_mst()
        ut.nx_delete_edge_attr(graph, 'is_cut')
        node_to_label = infr.get_node_attrs('name_label')
        edge_to_cut = {(u, v): node_to_label[u] != node_to_label[v]
                       for (u, v) in graph.edges()}
        nx.set_edge_attrs(graph, 'is_cut', edge_to_cut)

    def get_threshold(infr):
        # Only use the normalized scores to estimate a threshold
        normscores = np.array(nx.get_edge_attrs(infr.graph, 'normscore').values())
        if infr.verbose:
            print('len(normscores) = %r' % (len(normscores),))
        isvalid = ~np.isnan(normscores)
        curve = np.sort(normscores[isvalid])
        thresh = infr_model.estimate_threshold(curve, method=None)
        if infr.verbose:
            print('[estimate] thresh = %r' % (thresh,))
        if thresh is None:
            thresh = .5
        infr.thresh = thresh
        return thresh

    def infer_cut(infr, **kwargs):
        """
        Applies name labels based on graph inference and then cuts edges
        """
        from ibeis.algo.hots import graph_iden
        if infr.verbose:
            print('[infr] infer_cut')

        infr.remove_mst_edges()
        infr.model = graph_iden.InfrModel(infr.graph, infr.CUT_WEIGHT_KEY)
        model = infr.model
        thresh = infr.get_threshold()
        model._update_weights(thresh=thresh)
        labeling, params = model.run_inference2(max_labels=len(infr.aids))

        nx.set_node_attrs(infr.graph, 'name_label', model.node_to_label)
        infr.apply_cuts()
        infr.ensure_mst()

    def apply_all(infr):
        if infr.verbose:
            print('[infr] apply_all')
        infr.exec_matching()
        infr.apply_mst()
        infr.apply_match_edges()
        infr.apply_match_scores()
        infr.apply_feedback_edges()
        infr.apply_weights()
        infr.infer_cut()

    def find_possible_binary_splits(infr):
        flagged_edges = []

        for subgraph in infr.connected_component_reviewed_subgraphs():
            inconsistent_edges = [
                edge
                for edge, state in nx.get_edge_attrs(subgraph, 'reviewed_state').items()
                if state == 'nomatch']
            subgraph.remove_edges_from(inconsistent_edges)
            subgraph = infr.simplify_graph(subgraph)
            for s, t in inconsistent_edges:
                edgeset = nx.minimum_edge_cut(subgraph, s, t)
                edgeset = set([tuple(sorted(edge)) for edge in edgeset])
                flagged_edges.append(edgeset)
        edges = ut.flatten(flagged_edges)
        return edges

    def get_filtered_edges(infr, review_cfg):
        """
        Returns a list of edges (typically for user review) based on a specific
        filter configuration.

        CommandLine:
            python -m ibeis get_filtered_edges --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.exec_matching()
            >>> infr.apply_match_edges()
            >>> infr.apply_match_scores()
            >>> infr.apply_feedback_edges()
            >>> review_cfg = {'max_num': 3}
            >>> aids1, aids2 = infr.get_filtered_edges(review_cfg)
            >>> assert len(aids1) == 3
        """
        review_cfg_defaults = {
            'ranks_top': 3,
            'ranks_bot': 2,

            'score_thresh': None,
            'max_num': None,

            'filter_reviewed': True,
            'filter_photobombs': False,

            'filter_true_matches': True,
            'filter_false_matches': False,

            'filter_nonmatch_between_ccs': True,
            'filter_dup_namepairs': True,
        }

        review_cfg = ut.update_existing(
            review_cfg_defaults, review_cfg, assert_exists=True,
            iswarning=True)

        ibs = infr.ibs
        graph = infr.graph
        nodes = list(graph.nodes())
        uv_list = list(graph.edges())

        node_to_aids = infr.get_node_attrs('aid')
        node_to_nids = infr.get_node_attrs('name_label')
        aids = ut.take(node_to_aids, nodes)
        nids = ut.take(node_to_nids, nodes)
        aid_to_nid = dict(zip(aids, nids))
        nid2_aids = ut.group_items(aids, nids)

        # Initial set of edges
        aids1 = ut.take_column(uv_list, 0)
        aids2 = ut.take_column(uv_list, 1)

        num_filtered = 0

        def filter_between_ccs_neg(aids1, aids2, isneg_flags):
            """
            If two cc's have at least X=1 negative reviews then remove all
            other reviews between those cc's
            """
            neg_aids1 = ut.compress(aids1, isneg_flags)
            neg_aids2 = ut.compress(aids2, isneg_flags)
            neg_nids1 = ut.take(aid_to_nid, neg_aids1)
            neg_nids2 = ut.take(aid_to_nid, neg_aids2)

            # Ignore inconsistent names
            # Determine which CCs photobomb each other
            invalid_nid_map = ut.ddict(set)
            for nid1, nid2 in zip(neg_nids1, neg_nids2):
                if nid1 != nid2:
                    invalid_nid_map[nid1].add(nid2)
                    invalid_nid_map[nid2].add(nid1)

            impossible_aid_map = ut.ddict(set)
            for nid1, other_nids in invalid_nid_map.items():
                for aid1 in nid2_aids[nid1]:
                    for nid2 in other_nids:
                        for aid2 in nid2_aids[nid2]:
                            impossible_aid_map[aid1].add(aid2)
                            impossible_aid_map[aid2].add(aid1)

            valid_flags = [aid2 not in impossible_aid_map[aid1]
                           for aid1, aid2 in zip(aids1, aids2)]
            return valid_flags

        if review_cfg['filter_nonmatch_between_ccs']:
            review_states = [
                graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')
                for edge in zip(aids1, aids2)]
            is_nonmatched = [state == 'nomatch' for state in review_states]
            #isneg_flags = is_nonmatched
            valid_flags = filter_between_ccs_neg(aids1, aids2, is_nonmatched)
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_photobombs']:
            # TODO: store photobomb status internally
            am_list = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
            ispb_flags = ibs.get_annotmatch_prop('Photobomb', am_list)
            valid_flags = filter_between_ccs_neg(aids1, aids2, ispb_flags)
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_true_matches']:
            nids1 = ut.take(aid_to_nid, aids1)
            nids2 = ut.take(aid_to_nid, aids2)
            valid_flags = [nid1 != nid2 for nid1, nid2 in zip(nids1, nids2)]
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_false_matches']:
            nids1 = ut.take(aid_to_nid, aids1)
            nids2 = ut.take(aid_to_nid, aids2)
            valid_flags = [nid1 == nid2 for nid1, nid2 in zip(nids1, nids2)]
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_reviewed']:
            valid_flags = [
                graph.get_edge_data(*edge).get(
                    'reviewed_state', 'unreviewed') == 'unreviewed'
                for edge in zip(aids1, aids2)]
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_dup_namepairs']:
            # Only look at a maximum of one review between the current set of
            # connected components
            nids1 = ut.take(aid_to_nid, aids1)
            nids2 = ut.take(aid_to_nid, aids2)
            scores = np.array([
                # hack
                max(graph.get_edge_data(*edge).get('score', -1), -1)
                for edge in zip(aids1, aids2)])
            review_states = [
                graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')
                for edge in zip(aids1, aids2)]
            is_notcomp = np.array([state == 'notcomp'
                                   for state in review_states], dtype=np.bool)
            # Notcomps should not be considered in this filtering
            scores[is_notcomp] = -2
            namepair_id_list = vt.compute_unique_data_ids_(ut.lzip(nids1, nids2))
            namepair_id_list = np.array(namepair_id_list, dtype=np.int)
            unique_np_ids, np_groupxs = vt.group_indices(namepair_id_list)
            score_np_groups = vt.apply_grouping(scores, np_groupxs)
            unique_rowx2 = sorted([
                groupx[score_group.argmax()]
                for groupx, score_group in zip(np_groupxs, score_np_groups)
            ])
            aids1 = ut.take(aids1, unique_rowx2)
            aids2 = ut.take(aids2, unique_rowx2)

        if review_cfg['max_num'] is not None:
            scores = np.array([
                # hack
                max(graph.get_edge_data(*edge).get('score', -1), -1)
                for edge in zip(aids1, aids2)])
            sortx = scores.argsort()[::-1]
            top_idx = sortx[:review_cfg['max_num']]
            aids1 = ut.take(aids1, top_idx)
            aids2 = ut.take(aids2, top_idx)

        print('[infr] num_filtered = %r' % (num_filtered,))
        return aids1, aids2

    def exec_matching(infr, prog_hook=None, cfgdict=None):
        """ Loads chip matches into the inference structure """
        if infr.verbose:
            print('[infr] exec_matching')
        #from ibeis.algo.hots import graph_iden
        ibs = infr.ibs
        aids = infr.aids
        if cfgdict is None:
            cfgdict = {
                'can_match_samename': True,
                'K': 3,
                'Knorm': 3,
                'prescore_method': 'csum',
                'score_method': 'csum'
            }
        custom_nid_lookup = dict(zip(aids, infr.get_annot_attrs('name_label', aids)))
        # TODO: use current nids
        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup)
        cm_list = qreq_.execute(prog_hook=prog_hook)
        infr.cm_list = cm_list
        infr.qreq_ = qreq_

    def exec_vsone(infr, prog_hook=None):
        # Post process ranks_top and bottom vsmany queries with vsone
        # Execute vsone queries on the best vsmany results
        parent_rowids = list(infr.graph.edges())
        # Hack to get around default product of qaids
        qreq_ = infr.ibs.depc.new_request('vsone', [], [], cfgdict={})
        cm_list = qreq_.execute(parent_rowids=parent_rowids,
                                prog_hook=prog_hook)
        infr.vsone_qreq_ = qreq_
        infr.vsone_cm_list_ = cm_list

    def get_pairwise_features():
        # Extract features from the one-vs-one results
        pass


def demo_graph_iden():
    """
    """
    from ibeis.algo.hots import graph_iden
    import ibeis
    ibs = ibeis.opendb('PZ_MTEST')
    # Initially the entire population is unnamed
    aids = ibs.get_valid_aids()[:20]
    nids = [-aid for aid in aids]
    infr = graph_iden.AnnotInference(ibs, aids, nids=nids, autoinit=True)

    # Build hypothesis links
    infr.exec_matching()
    # TODO: specify ranks top/bot here
    infr.apply_match_edges()
    infr.apply_match_scores()
    infr.apply_feedback_edges()
    infr.apply_weights()

    import plottool as pt
    infr.show_graph()
    pt.set_title('pre-review')

    oracle_mode = True

    # Now either a manual or automatic reviewer must
    # determine which matches are correct
    aids1, aids2 = infr.get_filtered_edges({})
    count = 0
    for aid1, aid2 in ut.ProgIter(list(zip(aids1, aids2)), 'review'):
        if oracle_mode:
            # Assume perfect reviewer
            nid1, nid2 = ibs.get_annot_nids([aid1, aid2])
            truth = nid1 == nid2
            if truth:
                infr.add_feedback(aid1, aid2, 'match')
            else:
                infr.add_feedback(aid1, aid2, 'nomatch')
        count += 1
        if count > 100:
            break

    infr.apply_feedback_edges()
    infr.apply_weights()
    infr.show_graph()
    pt.set_title('post-review')

    infr.connected_component_reviewed_relabel()
    infr.show_graph()
    pt.set_title('post-inference')


def testdata_infr(defaultdb='PZ_MTEST'):
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaultdb)
    aids = [1, 2, 3, 4, 5, 6]
    infr = AnnotInference(ibs, aids, autoinit=True)
    return infr


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.graph_iden
        python -m ibeis.algo.hots.graph_iden --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
