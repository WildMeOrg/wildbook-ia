# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt  # NOQA
import itertools as it
import six
from ibeis.algo.hots import viz_graph_iden
from ibeis.algo.hots import infr_model
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


def _dz(a, b):
    a = a.tolist() if isinstance(a, np.ndarray) else list(a)
    b = b.tolist() if isinstance(b, np.ndarray) else list(b)
    return ut.dzip(a, b)


def e_(u, v):
    return (u, v) if u < v else (v, u)


def filter_between_ccs_neg(aids1, aids2, aid_to_nid, nid_to_aids, isneg_flags):
    """
    If two cc's have at least 1 negative review between them, then
    remove all other potential reviews between those cc's

    CommandLine:
        python -m ibeis.algo.hots.graph_iden filter_between_ccs_neg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.graph_iden import *  # NOQA
        >>> edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 2)]
        >>> aids1 = ut.take_column(edges, 0)
        >>> aids2 = ut.take_column(edges, 1)
        >>> isneg_flags = [0, 0, 1, 0, 0]
        >>> aid_to_nid = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        >>> nid_to_aids = {0: [0, 1, 2], 1: [3, 4]}
        >>> valid_flags = filter_between_ccs_neg(aids1, aids2, aid_to_nid,
        >>>                                      nid_to_aids, isneg_flags)
        >>> result = ('valid_flags = %s' % (ut.repr2(valid_flags),))
        >>> print(result)
        valid_flags = [True, True, False, True, False]
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
        for aid1 in nid_to_aids[nid1]:
            for nid2 in other_nids:
                for aid2 in nid_to_aids[nid2]:
                    impossible_aid_map[aid1].add(aid2)
                    impossible_aid_map[aid2].add(aid1)

    valid_flags = [aid2 not in impossible_aid_map[aid1]
                   for aid1, aid2 in zip(aids1, aids2)]
    return valid_flags


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrHelpers(object):
    """ Contains non-core helper functions """

    def get_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        """ Networkx node getter helper """
        node_to_attr = nx.get_node_attributes(infr.graph, key)
        if nodes is not None:
            node_to_attr = ut.dict_subset(node_to_attr, keys=nodes,
                                          default=default)
        return node_to_attr

    def get_edge_attrs(infr, key, edges=None, default=ut.NoParam):
        """ Networkx edge getter helper """
        if edges is not None:
            if default is ut.NoParam:
                edge_to_attr = {(u, v): infr.graph.edge[u][v][key] for u, v in edges}
            else:
                edge_to_attr = {(u, v): infr.graph.edge[u][v].get(key, default) for u, v in edges}
        else:
            edge_to_attr = nx.get_edge_attributes(infr.graph, key)
            if edges is not None:
                edge_to_attr = ut.dict_subset(edge_to_attr, keys=edges,
                                              default=default)
        return edge_to_attr

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

    def reset_name_labels(infr):
        """ Resets all annotation node name labels to their initial values """
        if infr.verbose >= 1:
            print('[infr] reset_name_labels')
        orig_names = infr.get_node_attrs('orig_name_label')
        infr.set_node_attrs('name_label', orig_names)

    def remove_name_labels(infr):
        """ Sets all annotation node name labels to be unknown """
        if infr.verbose >= 1:
            print('[infr] remove_name_labels()')
        # make distinct names for all nodes
        distinct_names = {
            node: -aid for node, aid in infr.get_node_attrs('aid').items()
        }
        infr.set_node_attrs('name_label', distinct_names)

    def has_edge(infr, edge):
        redge = edge[::-1]
        flag = infr.graph.has_edge(*edge) or infr.graph.has_edge(*redge)
        return flag

    def get_edge_data(infr, u, v):
        data = infr.graph.get_edge_data(u, v)
        if data is not None:
            data = ut.delete_dict_keys(data.copy(), infr.visual_edge_attrs)
        return data


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrDummy(object):
    def remove_dummy_edges(infr):
        if infr.verbose >= 2:
            print('[infr] remove_dummy_edges')
        edge_to_isdummy = infr.get_edge_attrs('_dummy_edge')
        dummy_edges = [edge for edge, flag in edge_to_isdummy.items() if flag]
        infr.graph.remove_edges_from(dummy_edges)

    def apply_mst(infr):
        """
        MST edges connect nodes labeled with the same name.
        This is done in case an explicit feedback or score edge does not exist.
        """
        if infr.verbose >= 2:
            print('[infr] apply_mst')
        # Remove old MST edges
        infr.remove_dummy_edges()
        infr.ensure_mst()

    def ensure_full(infr):
        if infr.verbose:
            print('[infr] ensure_full with %d nodes' % (len(infr.graph)))
        new_edges = nx.complement(infr.graph).edges()
        # if infr.verbose:
        #     print('[infr] adding %d complement edges' % (len(new_edges)))
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', _dz(new_edges, [True]))

    def ensure_cliques(infr):
        """
        Force each name label to be a clique
        """
        if infr.verbose:
            print('[infr] ensure_cliques')
        node_to_label = infr.get_node_attrs('name_label')
        label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())
        new_edges = []
        for label, nodes in label_to_nodes.items():
            for edge in ut.combinations(nodes, 2):
                if not infr.has_edge(edge):
                    new_edges.append(edge)
        if infr.verbose >= 2:
            print('[infr] adding %d clique edges' % (len(new_edges)))
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', _dz(new_edges, [True]))

    def ensure_mst(infr):
        """
        Use minimum spannning tree to ensure all names are connected
        Needs to be applied after any operation that adds/removes edges if we
        want to maintain that name labels must be connected in some way.
        """
        if infr.verbose >= 1:
            print('[infr] ensure_mst')
        import networkx as nx
        # Find clusters by labels
        node_to_label = infr.get_node_attrs('name_label')
        label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())

        aug_graph = infr.graph.copy().to_undirected()

        # remove cut edges from augmented graph
        edge_to_iscut = nx.get_edge_attributes(aug_graph, 'is_cut')
        cut_edges = [
            (u, v)
            for (u, v, d) in aug_graph.edges(data=True)
            if not (
                d.get('is_cut') or
                d.get('reviewed_state', 'unreviewed') in ['nomatch']
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
            # print('time_deltas = %r' % (time_deltas,))
            nx.set_edge_attributes(aug_graph, 'weight',
                                   {edge: 10.0 + delta
                                    for edge, delta in zip(candidate_mst_edges, time_deltas)})
        except Exception:
            print('FAILED WEIGHTING USING TIME')
            nx.set_edge_attributes(aug_graph, 'weight',
                                   {edge: 10.0 + _randint()
                                    for edge in candidate_mst_edges})
        new_edges = []
        for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
            mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
            # Only add edges not in the original graph
            for edge in mst_sub_graph.edges():
                if not infr.has_edge(edge):
                    new_edges.append(edge)
        # Add new MST edges to original graph
        if infr.verbose >= 2:
            print('[infr] adding %d MST edges' % (len(new_edges)))
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', _dz(new_edges, [True]))

    def review_dummy_edges(infr):
        if infr.verbose >= 1:
            print('[infr] review_dummy_edges')
        # print("REMOVE DUMMY EDGES")
        infr.ensure_mst()
        edge_to_dummy = infr.get_edge_attrs('_dummy_edge')
        edges = [edge for edge, dummy in edge_to_dummy.items() if dummy]
        ut.nx_delete_edge_attr(infr.graph, '_dummy_edge')
        if infr.verbose >= 1:
            print('[infr] reviewing %s dummy edges' % (len(edges),))
        # print("HACK REVIEWED STATE")
        for u, v in edges:
            infr.add_feedback(u, v, 'match', verbose=False)
        # if len(edges):
        #     nx.set_edge_attributes(infr.graph, 'reviewed_state', _dz(edges, ['match']))
        # print(ut.repr3(ut.graph_info(infr.simplify_graph())))


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrIBEIS(object):
    """
    Direct interface into ibeis tables
    (most of these should not be used or be reworked)
    """

    def read_ibeis_annotmatch_feedback(infr):
        """
        Reads feedback from annotmatch table.
        TODO: DEPRICATE (make an external helper function?)

        CommandLine:
            python -m ibeis.algo.hots.graph_iden read_ibeis_annotmatch_feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> user_feedback = infr.read_ibeis_annotmatch_feedback()
            >>> result =('user_feedback = %s' % (ut.repr2(user_feedback, nl=1),))
            >>> print(result)
            user_feedback = {
                (2, 3): [({'p_match': 0.0, 'p_nomatch': 1.0, 'p_notcomp': 0.0}, ['photobomb'])],
                (5, 6): [({'p_match': 0.0, 'p_nomatch': 1.0, 'p_notcomp': 0.0}, ['photobomb'])],
            }
        """
        if infr.verbose >= 1:
            print('[infr] read_ibeis_annotmatch_feedback')
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
        tags_list = ibs.get_annotmatch_case_tags(am_rowids)
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
            edge = e_(aid1, aid2)
            review_dict = {
                'p_match': p_match[count],
                'p_nomatch': p_nomatch[count],
                'p_notcomp': p_notcomp[count],
            }
            tags = tags_list[count]
            user_feedback[edge].append((review_dict, tags))
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
        probtags_ = list(user_feedback.values())
        probtags = ut.take_column(probtags_, -1)
        probs = ut.take_column(probtags, 0)
        df = pd.DataFrame.from_dict(probs)
        df['aid1'] = aids1
        df['aid2'] = aids2
        df['am_rowid'] = am_rowids
        df.set_index('am_rowid')
        df.index = pd.Index(am_rowids, name='am_rowid')
        #df.index = pd.Index(aid_pairs, name=('aid1', 'aid2'))
        return df

    def match_state_delta(infr):
        r"""
        Returns information about state change of annotmatches

        Returns:
            tuple: (new_df, old_df)

        CommandLine:
            python -m ibeis.algo.hots.graph_iden match_state_delta

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback(2, 3, 'nomatch')
            >>> infr.add_feedback(5, 6, 'nomatch')
            >>> (new_df, old_df) = infr.match_state_delta()
            >>> result = ('new_df =\n%s' % (new_df,))
            >>> result += ('\nold_df =\n%s' % (old_df,))
            >>> print(result)
        """
        old_feedback = infr._pandas_feedback_format(infr.read_ibeis_annotmatch_feedback())
        new_feedback = infr._pandas_feedback_format(infr.user_feedback)
        new_df, old_df = infr._make_state_delta(old_feedback, new_feedback)
        return new_df, old_df

    @staticmethod
    def _make_state_delta(old_feedback, new_feedback):
        """
        CommandLine:
            python -m ibeis.algo.hots.graph_iden _make_state_delta

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> import pandas as pd
            >>> columns = ['p_match', 'p_nomatch', 'p_noncomp', 'aid1', 'aid2', 'am_rowid']
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
        existing_ams = new_feedback['am_rowid'][~pd.isnull(new_feedback['am_rowid'])]
        both_ams = np.intersect1d(old_feedback['am_rowid'], existing_ams).astype(np.int)

        all_new_df = new_feedback.loc[both_ams]
        all_old_df = old_feedback.loc[both_ams]
        add_df = new_feedback.loc[pd.isnull(new_feedback['am_rowid'])].copy()

        if len(both_ams) > 0:
            is_changed = ~np.all(all_new_df.values == all_old_df.values, axis=1)
            new_df_ = all_new_df[is_changed]
            old_df = all_old_df[is_changed]
        else:
            new_df_ = all_new_df
            old_df = all_old_df
        new_df = pd.concat([new_df_, add_df])
        return new_df, old_df


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrMatching(object):
    def exec_matching(infr, prog_hook=None, cfgdict=None):
        """ Loads chip matches into the inference structure """
        if infr.verbose >= 1:
            print('[infr] exec_matching')
        #from ibeis.algo.hots import graph_iden
        ibs = infr.ibs
        aids = infr.aids
        if cfgdict is None:
            cfgdict = {
                # 'can_match_samename': False,
                'can_match_samename': True,
                'can_match_sameimg': True,
                'K': 3,
                'Knorm': 3,
                'prescore_method': 'csum',
                'score_method': 'csum'
            }
        # hack for using current nids
        custom_nid_lookup = dict(zip(aids, infr.get_annot_attrs('name_label', aids)))

        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup)

        cm_list = qreq_.execute(prog_hook=prog_hook)

        infr.vsmany_qreq_ = qreq_
        infr.vsmany_cm_list = cm_list
        infr.cm_list = cm_list
        infr.qreq_ = qreq_

    def exec_vsone(infr, prog_hook=None):
        r"""
        Args:
            prog_hook (None): (default = None)

        CommandLine:
            python -m ibeis.algo.hots.graph_iden exec_vsone

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.ensure_full()
            >>> result = infr.exec_vsone()
            >>> print(result)
        """
        # Post process ranks_top and bottom vsmany queries with vsone
        # Execute vsone queries on the best vsmany results
        parent_rowids = list(infr.graph.edges())
        qaids = ut.take_column(parent_rowids, 0)
        daids = ut.take_column(parent_rowids, 1)

        config = {
            # 'sv_on': False,
            'ratio_thresh': .9,
        }

        result_list = infr.ibs.depc.get('vsone', (qaids, daids), config=config)
        # result_list = infr.ibs.depc.get('vsone', parent_rowids)
        # result_list = infr.ibs.depc.get('vsone', [list(zip(qaids)), list(zip(daids))])
        # hack copy the postprocess
        import ibeis
        unique_qaids, groupxs = ut.group_indices(qaids)
        grouped_daids = ut.apply_grouping(daids, groupxs)

        unique_qnids = infr.ibs.get_annot_nids(unique_qaids)
        single_cm_list = ut.take_column(result_list, 1)
        grouped_cms = ut.apply_grouping(single_cm_list, groupxs)

        _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_cms)
        cm_list = []
        for qaid, qnid, daids, cms in _iter:
            # Hacked in version of creating an annot match object
            chip_match = ibeis.ChipMatch.combine_cms(cms)
            # chip_match.score_maxcsum(request)
            cm_list.append(chip_match)

        # cm_list = qreq_.execute(parent_rowids)
        infr.vsone_qreq_ = infr.ibs.depc.new_request('vsone', qaids, daids, cfgdict=config)
        infr.vsone_cm_list_ = cm_list
        infr.qreq_  = infr.vsone_qreq_
        infr.cm_list = cm_list

    def lookup_cm(infr, aid1, aid2):
        """
        Get chipmatch object associated with an edge if one exists.
        """
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

    @profile
    def apply_match_edges(infr, review_cfg={}):
        if infr.cm_list is None:
            print('[infr] apply_match_edges - matching has not been run!')
            return
        if infr.verbose >= 1:
            print('[infr] apply_match_edges')
        edges = infr._cm_breaking(review_cfg)
        # Create match-based graph structure
        infr.remove_dummy_edges()
        if infr.verbose >= 1:
            print('[infr] apply_match_edges adding %d edges' % len(edges))
        infr.graph.add_edges_from(edges)
        # infr.ensure_mst()

    def break_graph(infr, num):
        """
        Is this problem NP-hard? No, it is the b-matching problem.
        This problem is equivalent to bidirectional flow.

        References:
            http://www.ams.sunysb.edu/~jsbm/papers/b-matching.pdf

        # given (graph, K):
        # Let x[e] be 1 if we keep an edge e and 0 if we cut it

        # Keep the best set of edges for each node
        maximize
            sum(d['weight'] * x[(u, v)]
                for u in graph.nodes()
                for v, d in graph.node[u].items())

        # The degree of each node must be less than K
        subject to
            all(
                sum(x[(u, v)] for v in graph.node[u]) <= K
                for u in graph.nodes()
            )

            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching()
            >>> infr.apply_match_edges()
            >>> infr.apply_match_scores()
            >>> infr.ensure_full()

        The linear program is based on the blossom algorithm

        implicit summation: b(W) = sum(b_v for v in W)

        Let omega = [S \subset V where len(S) >= 3 and abs(sum(v_b for v in S)) % 2 == 1]

        Let q_S = .5 * sum(v_b for v in S) - 1 for S in omega

        For each W \subset V
        Let delta(W) be the set of edges that meet exactly one node in W
        Let gamma(W) be the set of edges with both endpoints in W

        maximize c.dot(x)
        subject to x(delta(v)) = b_v forall v in V
        x_e >= 0 forall e in E
        x(gamma(S)) <= q_S foall S in omega

        """
        # prev_degrees = np.array([infr.graph.degree(n) for n in infr.graph.nodes()])

        weight = 'normscore'
        if len(infr.graph) < 100:
            # Ineffcient but exact integer programming solution
            K = num
            graph = infr.graph
            import pulp
            # Formulate integer program
            prob = pulp.LpProblem("B-Matching", pulp.LpMaximize)
            # Solution variables
            indexs = [e_(*e) for e in graph.edges()]
            # cat = pulp.LpContinuous
            cat = pulp.LpInteger
            x = pulp.LpVariable.dicts(name='x', indexs=indexs,
                                      lowBound=0, upBound=1, cat=cat)
            # maximize objective function
            prob.objective = sum(d.get(weight, 0) * x[e_(u, v)]
                                 for u in graph.nodes()
                                 for v, d in graph.edge[u].items())
            # subject to
            for u in graph.nodes():
                prob.add(sum(x[e_(u, v)] for v in graph.edge[u]) <= K)
            # Solve using with solver like CPLEX, GLPK, or SCIP.
            #pulp.CPLEX().solve(prob)
            pulp.PULP_CBC_CMD().solve(prob)
            # Read solution
            xvalues = [x[e].varValue for e in indexs]
            to_remove = [e for e, xval in zip(indexs, xvalues)
                         if not xval == 1.0]
            graph.remove_edges_from(to_remove)
        else:
            # Hacky solution. TODO: implement b-matching using blossom with
            # networkx
            to_remove = set([])
            # nodes = infr.graph.nodes()
            # degrees = np.array([infr.graph.degree(n) for n in infr.graph.nodes()])
            for u in infr.graph.nodes():
                if len(infr.graph[u]) > num:
                    edges = []
                    scores = []
                    for v, d in infr.graph[u].items():
                        e = e_(u, v)
                        if e not in to_remove:
                            # hack because I think this may be a hard problem
                            edges.append(e)
                            scores.append(d.get(weight, -1))
                    bottomx = ut.argsort(scores)[::-1][num:]
                    to_remove.update(set(ut.take(edges, bottomx)))
            infr.graph.remove_edges_from(to_remove)
        degrees = np.array([infr.graph.degree(n) for n in infr.graph.nodes()])
        assert np.all(degrees <= num)

    def _cm_breaking(infr, review_cfg={}):
        """
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> review_cfg = {}
        """
        cm_list = infr.cm_list
        ranks_top = review_cfg.get('ranks_top', None)
        ranks_bot = review_cfg.get('ranks_bot', None)

        # Construct K-broken graph
        edges = []

        if ranks_bot is None:
            ranks_bot = 0

        for count, cm in enumerate(cm_list):
            score_list = cm.annot_score_list
            rank_list = ut.argsort(score_list)[::-1]
            sortx = ut.argsort(rank_list)

            top_sortx = sortx[:ranks_top]
            bot_sortx = sortx[len(sortx) - ranks_bot:]
            short_sortx = ut.unique(top_sortx + bot_sortx)

            daid_list = ut.take(cm.daid_list, short_sortx)
            for daid in daid_list:
                u, v = (cm.qaid, daid)
                if v < u:
                    u, v = v, u
                edges.append((u, v))
        return edges

    def _cm_training_pairs(infr, top_gt=2, mid_gt=2, bot_gt=2, top_gf=2,
                           mid_gf=2, bot_gf=2, rand_gt=2, rand_gf=2, rng=None):
        """
        Constructs training data for a pairwise classifier

        Example:
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching(cfgdict={
            >>>     'can_match_samename': True,
            >>>     'K': 4,
            >>>     'Knorm': 1,
            >>>     'prescore_method': 'csum',
            >>>     'score_method': 'csum'
            >>> })
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> exec(ut.execstr_funckw(infr._cm_training_pairs))
            >>> infr._cm_training_pairs()
        """
        cm_list = infr.cm_list
        qreq_ = infr.qreq_
        ibs = infr.ibs
        aid_pairs = []
        dnids = qreq_.ibs.get_annot_nids(qreq_.daids)
        # dnids = qreq_.get_qreq_annot_nids(qreq_.daids)
        rng = ut.ensure_rng(rng)
        for cm in ut.ProgIter(cm_list, lbl='building pairs'):
            all_gt_aids = cm.get_top_gt_aids(ibs)
            all_gf_aids = cm.get_top_gf_aids(ibs)
            gt_aids = ut.take_percentile_parts(all_gt_aids, top_gt, mid_gt,
                                               bot_gt)
            gf_aids = ut.take_percentile_parts(all_gf_aids, top_gf, mid_gf,
                                               bot_gf)
            # get unscored examples
            unscored_gt_aids = [aid for aid in qreq_.daids[cm.qnid == dnids]
                                if aid not in cm.daid2_idx]
            rand_gt_aids = ut.random_sample(unscored_gt_aids, rand_gt, rng=rng)
            # gf_aids = cm.get_groundfalse_daids()
            _gf_aids = qreq_.daids[cm.qnid != dnids]
            _gf_aids = qreq_.daids.compress(cm.qnid != dnids)
            # gf_aids = ibs.get_annot_groundfalse(cm.qaid, daid_list=qreq_.daids)
            rand_gf_aids = ut.random_sample(_gf_aids, rand_gf, rng=rng).tolist()
            chosen_daids = ut.unique(gt_aids + gf_aids + rand_gf_aids +
                                     rand_gt_aids)
            aid_pairs.extend([(cm.qaid, aid) for aid in chosen_daids])
        return aid_pairs

    def get_pairwise_features():
        # Extract features from the one-vs-one results
        pass

    def _get_cm_edge_data(infr, edges):
        symmetric = True

        # Find scores for the edges that exist in the graph
        edge_to_data = ut.ddict(dict)
        node_to_cm = {infr.aid_to_node[cm.qaid]:
                      cm for cm in infr.cm_list}
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
        return edge_to_data

    @profile
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
            >>> infr.get_edge_attrs('score')
        """
        if infr.cm_list is None:
            print('[infr] apply_match_scores - no scores to apply!')
            return
        if infr.verbose >= 1:
            print('[infr] apply_match_scores')
        edges = list(infr.graph.edges())
        edge_to_data = infr._get_cm_edge_data(edges)

        # Remove existing attrs
        ut.nx_delete_edge_attr(infr.graph, 'score')
        ut.nx_delete_edge_attr(infr.graph, 'rank')
        ut.nx_delete_edge_attr(infr.graph, 'normscore')
        ut.nx_delete_edge_attr(infr.graph, 'match_probs')
        ut.nx_delete_edge_attr(infr.graph, 'entropy')

        edges = list(edge_to_data.keys())
        edge_scores = list(ut.take_column(edge_to_data.values(), 'score'))
        edge_scores = ut.replace_nones(edge_scores, np.nan)
        edge_scores = np.array(edge_scores)
        edge_ranks = np.array(ut.take_column(edge_to_data.values(), 'rank'))
        # take the inf-norm
        normscores = edge_scores / vt.safe_max(edge_scores, nans=False)

        # Add new attrs
        infr.set_edge_attrs('score', ut.dzip(edges, edge_scores))
        infr.set_edge_attrs('rank', ut.dzip(edges, edge_ranks))

        nanflags = np.isnan(normscores)
        p_match = normscores
        p_nomatch = 1 - normscores
        p_notcomp = nanflags * 1 / 3
        p_nomatch[nanflags] = 1 / 3
        p_match[nanflags] = 1 / 3

        # Hack away zero probabilites
        probs = np.vstack([p_nomatch, p_match, p_notcomp]).T + 1e-9
        probs = vt.normalize(probs, axis=1, ord=1, out=probs)
        entropy = -(np.log2(probs) * probs).sum(axis=1)

        match_probs = [ut.dzip(['nomatch', 'match', 'notcomp'], p)
                       for p in probs]
        infr.set_edge_attrs('normscore', dict(zip(edges, normscores)))
        infr.set_edge_attrs('match_probs', dict(zip(edges, match_probs)))
        infr.set_edge_attrs('entropy', dict(zip(edges, entropy)))


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrFeedback(object):
    prob_keys = [
        'p_nomatch',
        'p_match',
        'p_notcomp',
    ]

    truth_texts = {
        0: 'nomatch',
        1: 'match',
        2: 'notcomp',
        3: 'unreviewed',
    }

    @profile
    def add_feedback(infr, aid1, aid2, state, tags=[], apply=False,
                     verbose=None):
        """
        Public interface to add feedback for a single edge

        Args:
            aid1 (int):  annotation id
            aid2 (int):  annotation id
            state (str): state from `infr.truth_texts`
            tags (list of str): specify Photobomb / Scenery / etc

        CommandLine:
            python -m ibeis.algo.hots.graph_iden add_feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback(5, 6, 'match')
            >>> infr.add_feedback(5, 6, 'nomatch', ['Photobomb'])
            >>> infr.add_feedback(1, 2, 'notcomp')
            >>> result = ut.repr2(infr.user_feedback, nl=2)
            >>> print(result)
            {
                (1, 2): [
                    ({'p_match': 0.0, 'p_nomatch': 0.0, 'p_notcomp': 1.0}, []),
                ],
                (5, 6): [
                    ({'p_match': 1.0, 'p_nomatch': 0.0, 'p_notcomp': 0.0}, []),
                    ({'p_match': 0.0, 'p_nomatch': 1.0, 'p_notcomp': 0.0}, ['Photobomb']),
                ],
            }
        """
        if verbose is None:
            verbose = infr.verbose
        if verbose >= 1:
            print('[infr] add_feedback(%r, %r, state=%r, tags=%r)' % (
                aid1, aid2, state, tags))
        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))
        edge = e_(aid1, aid2)
        if state == 'unreviewed':
            review_dict = None
            if edge in infr.user_feedback:
                del infr.user_feedback[edge]
        else:
            if isinstance(state, dict):
                assert sorted(state.keys()) == sorted(infr.prob_keys)
                review_dict = state
            else:
                review_dict = {
                    'p_match': 0.0,
                    'p_nomatch': 0.0,
                    'p_notcomp': 0.0,
                }
                if state == 'match':
                    review_dict['p_match'] = 1.0
                elif state == 'nomatch':
                    review_dict['p_nomatch'] = 1.0
                elif state == 'notcomp':
                    review_dict['p_notcomp'] = 1.0
                else:
                    raise ValueError('state=%r is unknown' % (state,))
                infr.user_feedback[edge].append((review_dict, tags))
        if apply:
            # Apply new results on the fly
            infr._dynamically_apply_feedback(edge, review_dict, tags)

    @profile
    def _dynamically_apply_feedback(infr, edge, review_dict, tags):
        """
        Dynamically updates all states based on a single dynamic change

        CommandLine:
            python -m ibeis.algo.hots.graph_iden _dynamically_apply_feedback:1 --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.relabel_using_reviews()
            >>> infr.add_feedback(1, 2, 'match', apply=True)
            >>> infr.add_feedback(2, 3, 'match', apply=True)
            >>> infr.add_feedback(2, 3, 'match', apply=True)
            >>> assert infr.graph.edge[1][2]['num_reviews'] == 1
            >>> assert infr.graph.edge[2][3]['num_reviews'] == 2
            >>> infr._del_feedback_edges()
            >>> infr.apply_feedback_edges()
            >>> assert infr.graph.edge[1][2]['num_reviews'] == 1
            >>> assert infr.graph.edge[2][3]['num_reviews'] == 2

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.relabel_using_reviews()
            >>> infr.verbose = 2
            >>> ut.qt4ensure()
            >>> infr.ensure_full()
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback(6, 2, 'match', apply=True)
            >>> infr.add_feedback(2, 3, 'match', apply=True)
            >>> infr.add_feedback(3, 4, 'match', apply=True)
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback(2, 3, 'nomatch', apply=True)
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback(6, 4, 'match', apply=True)
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback(1, 5, 'nomatch', apply=True)
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback(1, 3, 'nomatch', apply=True)
            >>> infr.show_graph(show_cuts=True)
            >>> import plottool as pt
            >>> pt.present()
            >>> ut.show_if_requested()
        """
        if review_dict is None:
            if infr.verbose >= 2:
                print('[infr] _dynamically_apply_feedback (removing edge=%r)'
                      % (edge,))
            state = 'unreviewed'
            infr._del_feedback_edges([edge])
            infr.set_edge_attrs(
                'cut_weight', infr.get_edge_attrs('normscore', [edge], np.nan))
        else:
            # Apply the review to the specified edge
            review_stateid = ut.argmax(ut.take(review_dict, infr.prob_keys))
            state = infr.truth_texts[review_stateid]
            if infr.verbose >= 2:
                print('[infr] _dynamically_apply_feedback edge=%r, state=%r'
                      % (edge, state,))
            p_same = infr._compute_p_same(review_dict['p_match'],
                                          review_dict['p_notcomp'])
            infr._set_feedback_edges([edge], [state], [p_same], [tags])
            # TODO: change num_reviews to num_consistent_reviews
            num_reviews = infr.get_edge_attrs('num_reviews', [edge],
                                              default=0)[edge]
            infr.set_edge_attrs('num_reviews', {edge: num_reviews + 1})
            infr.set_edge_attrs('cut_weight', {edge: p_same})
            if state != 'notcomp':
                ut.nx_delete_edge_attr(infr.graph, 'inferred_state', [edge])

        # Dynamically update names and inferred attributes of relevant nodes
        # subgraph, subgraph_cuts = infr._get_influenced_subgraph(edge)
        n1, n2 = edge
        cc1 = infr.get_annot_cc(n1)
        cc2 = infr.get_annot_cc(n2)
        relevant_nodes = cc1.union(cc2)
        # print('relevant_nodes = %r' % (relevant_nodes,))
        subgraph = infr.graph.subgraph(relevant_nodes)

        # Change names of nodes
        infr.relabel_using_reviews(graph=subgraph)

        # Get a list of all known connected compoments
        extended_nodes = ut.flatten(infr.get_nomatch_ccs(relevant_nodes))
        extended_nodes += relevant_nodes
        # print('extended_nodes = %r' % (extended_nodes,))
        extended_subgraph = infr.graph.subgraph(extended_nodes)

        # This re-infers all attributes of the influenced sub-graph only
        infr.apply_review_inference(graph=extended_subgraph)

    @profile
    def apply_review_inference(infr, graph=None):
        """
        CommandLine:
            python -m ibeis.algo.hots.graph_iden apply_review_inference

        Example:
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> aids = list(range(1, 10))
            >>> infr = AnnotInference(None, aids, autoinit=True, verbose=1)
            >>> infr.ensure_full()
            >>> infr._init_priority_queue()
            >>> infr.add_feedback(1, 2, 'match', apply=True)
            >>> infr.add_feedback(2, 3, 'match', apply=True)
            >>> infr.add_feedback(2, 3, 'match', apply=True)
            >>> infr.add_feedback(3, 4, 'match', apply=True)
            >>> infr.add_feedback(4, 5, 'nomatch', apply=True)
            >>> infr.add_feedback(6, 7, 'match', apply=True)
            >>> infr.add_feedback(7, 8, 'match', apply=True)
            >>> infr.add_feedback(6, 8, 'nomatch', apply=True)
            >>> infr.add_feedback(6, 1, 'notcomp', apply=True)
            >>> infr.add_feedback(1, 9, 'notcomp', apply=True)
            >>> infr.add_feedback(8, 9, 'notcomp', apply=True)
            >>> #infr.show_graph(hide_cuts=False)
            >>> graph = infr.graph
            >>> infr.apply_review_inference(graph)
        """
        if graph is None:
            graph = infr.graph

        if infr.verbose >= 1:
            print('[infr] apply_review_inference on %d nodes' % (len(graph)))

        node_to_label = nx.get_node_attributes(graph, 'name_label')
        # TODO: uncomment the first line for efficiency once debugging is done
        # edge_to_review = nx.get_edge_attributes(graph, 'reviewed_state')
        edge_to_review = {e_(u, v): infr.graph.edge[u][v].get('reviewed_state', 'unreviewed')
                          for u, v in graph.edges()}
        # edge_to_review = ut.nx_get_default_edge_attributes(graph, 'reviewed_state', default='unreviewed')
        # edge_to_review = ut.map_keys(lambda args: e_(*args), edge_to_review)

        nid_to_cc = ut.invert_dict(node_to_label, unique_vals=False)

        # Get all reviewed edges
        grouped_edges = ut.group_pairs(edge_to_review.items())
        neg_edges = grouped_edges.pop('nomatch', [])
        pos_edges = grouped_edges.pop('match', [])
        notcomp_edges = grouped_edges.pop('notcomp', [])

        # TODO: comment once debugging is done
        unreviewed_edges = grouped_edges.pop('unreviewed', [])

        if grouped_edges:
            raise AssertionError('Reviewed state has unknown values: %r' % (list(grouped_edges.keys())),)

        seen_name_edges = set([])
        seen_nids = set([])

        # First loop through each negative edge looking for inconsistent
        # compoments
        inconsistent = {}
        for u, v in neg_edges:
            nid1 = node_to_label[u]
            nid2 = node_to_label[v]
            if nid1 == nid2 and nid1 not in inconsistent:
                cc = nid_to_cc[nid1]
                cc_inconsistent_edges = [
                    e_(*e) for e in ut.nx_edges_between(graph, cc)
                ]
                inconsistent[nid1] = cc_inconsistent_edges
                # TODO: should we grab all inconsistent outgoing edges here?
        seen_nids.update(inconsistent.keys())
        # For each negative edge, get the compoments belonging to each
        # endpoint. If the two compoments are the same then we have an
        # inconsistent case Otherwise infer all other edges between the
        # compoments are negative
        negative = {}
        inconsistent_outgoing_negatives = {}
        reviewed_negatives = ut.ddict(list)
        for u, v in neg_edges:
            nid1 = node_to_label[u]
            nid2 = node_to_label[v]
            name_edge = e_(nid1, nid2)
            if nid1 != nid2:
                if name_edge not in negative and name_edge not in inconsistent_outgoing_negatives:
                    cc1 = nid_to_cc[nid1]
                    cc2 = nid_to_cc[nid2]
                    cross_cc_edges = [
                        e_(*e) for e in ut.nx_edges_between(graph, cc1, cc2,
                                                            assume_disjoint=True)
                    ]
                    if nid1 in inconsistent or nid2 in inconsistent:
                        inconsistent_outgoing_negatives[name_edge] = cross_cc_edges
                    else:
                        negative[name_edge] = cross_cc_edges
                        reviewed_negatives[name_edge].append((u, v))
        seen_name_edges.update(negative.keys())
        seen_name_edges.update(inconsistent_outgoing_negatives.keys())
        # Then get each positive compoments and do
        # positive inference only in those ccs
        # also keep a grouping of reviewed positive edges
        reviewed_positives = ut.ddict(list)
        positive = {}
        for u, v in pos_edges:
            nid = node_to_label[u]
            cc = nid_to_cc[nid]
            if nid not in inconsistent:
                reviewed_positives[nid].append((u, v))
                if nid not in positive:
                    within_cc_edges = [
                        e_(*e) for e in ut.nx_edges_between(graph, cc)
                    ]
                    positive[nid] = within_cc_edges
        # Look at each not-comparable edge between two compoments not currently
        # marked as either positive or negative
        notcomparable = {}
        inconsistent_outgoing_notcomparable = {}  # NOQA
        for u, v in notcomp_edges:
            nid1 = node_to_label[u]
            nid2 = node_to_label[v]
            name_edge = e_(nid1, nid2)
            if name_edge not in seen_name_edges and name_edge not in notcomparable:
                if nid1 != nid2 or (
                      nid1 not in positive and nid1 not in inconsistent and
                      nid2 not in positive and nid2 not in inconsistent):

                    # TODO: need to update inconsistent_outgoing_noncomp here as well?
                    cc1 = nid_to_cc[nid1]
                    cc2 = nid_to_cc[nid2]
                    cross_cc_edges = [
                        e_(*e) for e in ut.nx_edges_between(graph, cc1, cc2,
                                                            assume_disjoint=True)
                    ]
                    notcomparable[name_edge] = cross_cc_edges
        seen_name_edges.update(notcomparable.keys())

        # maybe remove once debugging is done? No, should categorize each edge.
        unreviewed = {}
        inconsistent_outgoing_unreviewed = {}  # NOQA
        for u, v in unreviewed_edges:
            nid1 = node_to_label[u]
            nid2 = node_to_label[v]
            name_edge = e_(nid1, nid2)
            if name_edge not in seen_name_edges and name_edge not in unreviewed:
                if nid1 != nid2 or (
                      nid1 not in positive and nid1 not in inconsistent and
                      nid2 not in positive and nid2 not in inconsistent):

                    # TODO: need to update inconsistent_outgoing_unreviewed here as well?
                    cc1 = nid_to_cc[nid1]
                    cc2 = nid_to_cc[nid2]
                    cross_cc_edges = [
                        e_(*e) for e in ut.nx_edges_between(graph, cc1, cc2,
                                                            assume_disjoint=True)
                    ]
                    unreviewed[name_edge] = cross_cc_edges
        seen_name_edges.update(unreviewed.keys())

        # Find possible fixes for inconsistent compoments
        suggested_fix_edges = []
        other_error_edges = []
        for nid, cc_inconsistent_edges in inconsistent.items():
            # Find possible edges to fix in the reviewed subgarph
            reviewed_inconsistent = [
                e + (graph.get_edge_data(*e),) for e in cc_inconsistent_edges
                if edge_to_review.get(e, 'unreviewed') != 'unreviewed'
            ]
            subgraph = nx.Graph(reviewed_inconsistent)
            cc_error_edges = infr._find_possible_error_edges(subgraph)
            suggested_fix_edges.extend(cc_error_edges)
            other_error_edges.extend(ut.setdiff(subgraph.edges(), cc_error_edges))

        inconsistent_edges = ut.flatten(inconsistent.values())
        positive_edges = ut.flatten(positive.values())
        negative_edges = ut.flatten(negative.values())
        notcomparable_edges = ut.flatten(notcomparable.values())
        # The only case where an edge is not listed in the previous lists
        # should be when they are between compoments with absolutely no reviews
        unreviewed_edges = ut.flatten(unreviewed.values())
        inconsistent_outgoing_negative_edges = (ut.flatten(inconsistent_outgoing_negatives.values()))

        if True or __debug__:
            num_edges = (len(positive_edges) + len(negative_edges) +
                         len(inconsistent_edges) + len(notcomparable_edges) +
                         len(unreviewed_edges) + len(inconsistent_outgoing_negative_edges))

            num_edges_real = graph.number_of_edges()

            if num_edges != num_edges_real:
                print('num_edges = %r' % (num_edges,))
                all_edges = (positive_edges + negative_edges +
                             inconsistent_edges + notcomparable_edges +
                             unreviewed_edges + inconsistent_outgoing_negative_edges)
                assert len(ut.find_duplicate_items(all_edges)) == 0, 'edge not same and duplicates'

                edges = ut.lstarmap(e_, graph.edges())
                missing12 = ut.setdiff(edges, all_edges)
                missing21 = ut.setdiff(all_edges, edges)
                print('missing12 = %r' % (missing12,))
                print('missing21 = %r' % (missing21,))
                print(ut.repr4(ut.set_overlaps(graph.edges(), all_edges)))
                # import utool
                # utool.embed()
                # u, v = missing12[0]
                # graph.edge[u][v]
                # nid1 = node_to_label[u]
                # nid2 = node_to_label[v]
                # name_edge = e_(nid1, nid2)
                raise AssertionError('edges not the same')

        # Update the attributes of all edges in the subgraph

        # Update the infered state
        infr.set_edge_attrs('inferred_state', _dz(inconsistent_outgoing_negative_edges, [None]))
        infr.set_edge_attrs('inferred_state', _dz(inconsistent_edges, [None]))
        infr.set_edge_attrs('inferred_state', _dz(unreviewed_edges, [None]))
        infr.set_edge_attrs('inferred_state', _dz(notcomparable_edges, [None]))
        infr.set_edge_attrs('inferred_state', _dz(positive_edges, ['same']))
        infr.set_edge_attrs('inferred_state', _dz(negative_edges, ['diff']))

        # Suggest possible fixes
        infr.set_edge_attrs('maybe_error', ut.dzip(graph.edges(), [False]))
        infr.set_edge_attrs('maybe_error', _dz(suggested_fix_edges, [True]))

        # Update state of trivial edges
        infr.set_edge_attrs('inferred_state', _dz(inconsistent_outgoing_negative_edges, [True]))
        infr.set_edge_attrs('is_cut', _dz(negative_edges, [True]))
        infr.set_edge_attrs('is_cut', _dz(positive_edges, [False]))
        infr.set_edge_attrs('is_cut', _dz(inconsistent_edges, [False]))
        infr.set_edge_attrs('is_cut', _dz(notcomparable_edges, [False]))
        infr.set_edge_attrs('is_cut', _dz(unreviewed_edges, [False]))

        if infr.queue is not None:
            # TODO: Reformulate this as a "Graph Diameter Augmentation" problem.
            # It turns out this problem is NP-hard.
            # Bounded
            # (BCMB Bounded Cost Minimum Diameter Edge Addition)
            # https://www.cse.unsw.edu.au/~sergeg/papers/FratiGGM13isaac.pdf
            # http://www.cis.upenn.edu/~sanjeev/papers/diameter.pdf

            # update the priority queue on the fly
            queue = infr.queue
            pos_jump_thresh = infr.queue_params['pos_jump_thresh']
            neg_jump_thresh = infr.queue_params['neg_jump_thresh']

            if pos_jump_thresh is not None:
                # Reconsider edges within connected compoments that are
                # separated by a large distance over reviewed edges.
                strong_positives = []
                weak_positives = []
                for nid, edges in reviewed_positives.items():
                    strong_edges = []
                    weak_edges = []
                    reviewed_subgraph = nx.Graph(edges)
                    for u, dist_dict in nx.all_pairs_shortest_path_length(reviewed_subgraph):
                        for v, dist in dist_dict.items():
                            if u <= v and graph.has_edge(u, v):
                                if dist <= pos_jump_thresh:
                                    strong_edges.append((u, v))
                                else:
                                    weak_edges.append((u, v))
                    weak_positives.extend(weak_edges)
                    strong_positives.extend(strong_edges)
                queue.delete_items(strong_positives)
            else:
                for edges in positive.values():
                    queue.delete_items(edges)

            """
            Example:
                >>> from ibeis.algo.hots.graph_iden import *  # NOQA
                >>> from ibeis.algo.hots.graph_iden import _dz
                >>> from ibeis.algo.hots import demo_graph_iden
                >>> infr = demo_graph_iden.synthetic_infr(
                >>>     ccs=[[1, 2, 3, 4, 5],
                >>>            [6, 7, 8, 9, 10]],
                >>>     edges=[
                >>>         #(1, 6, {'reviewed_state': 'nomatch'}),
                >>>         (1, 6, {}),
                >>>         (4, 9, {}),
                >>>     ]
                >>> )
                >>> infr._init_priority_queue()
                >>> assert len(infr.queue) == 2
                >>> infr.queue_params['neg_jump_thresh'] = None
                >>> infr.add_feedback(1, 6, 'nomatch', apply=True)
                >>> assert len(infr.queue) == 0
                >>> graph = infr.graph
                >>> ut.exec_func_src(infr.apply_review_inference,
                >>>                  sentinal='if neg_jump_thresh', stop=-1, verbose=True)
                >>> infr.queue_params['neg_jump_thresh'] = 1
                >>> infr.apply_review_inference()
            """

            if neg_jump_thresh is not None:
                strong_negatives = []
                weak_negatives = []

                # Reconsider edges between connected compoments that are
                # separated by a large distance over reviewed edges.
                for nid_edge, neg_edges in reviewed_negatives.items():
                    nid1, nid2 = nid_edge
                    pos_edges1 = reviewed_positives[nid1]
                    pos_edges2 = reviewed_positives[nid2]
                    edges = pos_edges2 + pos_edges1 + neg_edges
                    reviewed_subgraph = nx.Graph(edges)
                    strong_edges = []
                    weak_edges = []
                    unreviewed_neg_edges = negative[nid_edge]
                    # FIXME: Change the forumlation of this problem to:
                    # Given two connected compoments, a set of potential edges,
                    # and a number K Find the minimum cost set of potential
                    # edges such that the maximum distance between two nodes in
                    # different compoments is less than K.

                    # distance_matrix = dict(nx.shortest_path_length(reviewed_subgraph))
                    # cc1 = nid_to_cc[nid1]
                    # cc2 = nid_to_cc[nid2]
                    # for u in cc1:
                    #     is_violated = np.array(list(ut.dict_subset(distance_matrix[u], cc2).values())) > neg_jump_thresh

                    for u, v in unreviewed_neg_edges:
                        # Ensure u corresponds to nid1 and v corresponds to nid2
                        if node_to_label[u] == nid2:
                            u, v = v, u
                        # Is the distance from u to any node in cc[nid2] large?
                        for v_, dist in nx.shortest_path_length(reviewed_subgraph, source=u):
                            if v_ in nid_to_cc[nid2] and graph.has_edge(u, v_):
                                if dist > neg_jump_thresh:
                                    weak_edges.append(e_(u, v_))
                                else:
                                    strong_edges.append(e_(u, v_))
                        # Is the distance from v to any node in cc[nid1] large?
                        for u_, dist in nx.shortest_path_length(reviewed_subgraph, source=v):
                            if u_ in nid_to_cc[nid1] and graph.has_edge(u_, v):
                                if dist > neg_jump_thresh:
                                    weak_edges.append(e_(u_, v))
                                else:
                                    strong_edges.append(e_(u_, v))
                    strong_negatives.extend(strong_edges)
                    weak_negatives.extend(weak_edges)
                # print('strong_edges.append = %r' % (strong_edges,))
                # print('weak_edges.append = %r' % (weak_edges,))
                queue.delete_items(strong_negatives)
            else:
                for edges in negative.values():
                    queue.delete_items(edges)

            # Add error edges back in with high (double) priority
            queue.update(zip(suggested_fix_edges, -2 * infr._get_priorites(suggested_fix_edges)))

            queue.delete_items(other_error_edges)

            needs_priority = [e for e in unreviewed_edges if e not in queue]
            # assert not needs_priority, 'shouldnt need this needs_priority=%r ' % (needs_priority,)
            queue.update(zip(needs_priority, -infr._get_priorites(needs_priority)))

    def remaining_reviews(infr):
        assert infr.queue is not None
        return len(infr.queue)

    def _get_priorites(infr, edges):
        priority_metric = 'normscore'
        new_priorities = np.array([max(infr.graph.get_edge_data(*e).get(priority_metric, -1), -1)
                                   for e in edges])
        return new_priorities

    def _init_priority_queue(infr, randomness=0, rng=None):
        if infr.verbose:
            print('[infr] _init_priority_queue')
        graph = infr.graph

        # Candidate edges are unreviewed
        cand_uvds = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if (d.get('reviewed_state', 'unreviewed') == 'unreviewed' or
                d.get('maybe_error', False))
        ]

        # TODO: stagger edges to review based on cc analysis

        priority_metric = 'normscore'
        # priority_metric = 'entropy'

        # Sort edges to review
        priorities = np.array([max(d.get(priority_metric, -1), -1)
                               for u, v, d in cand_uvds])
        edges = [e_(u, v) for u, v, d in cand_uvds]

        if len(priorities) > 0 and randomness > 0:
            minval = priorities.min()
            spread = priorities.max() - minval
            perb = (spread * rng.rand(len(priorities)) + minval)
            priorities = randomness * perb + (1 - randomness) * priorities

        # All operations on a treap except sorting use O(log(N)) time
        infr.queue = ut.PriorityQueue(zip(edges, -priorities))

    def _find_possible_error_edges(infr, subgraph):
        inconsistent_edges = [
            edge for edge, state in
            nx.get_edge_attributes(subgraph, 'reviewed_state').items()
            if state == 'nomatch'
        ]
        maybe_error_edges = set([])

        subgraph_ = subgraph.copy()
        subgraph_.remove_edges_from(inconsistent_edges)
        subgraph_ = infr.simplify_graph(subgraph_)

        ut.util_graph.nx_set_default_edge_attributes(subgraph_, 'num_reviews', 1)
        for s, t in inconsistent_edges:
            cut_edgeset = ut.nx_mincut_edges_weighted(subgraph_, s, t,
                                                      capacity='num_reviews')
            cut_edgeset = set([e_(*edge) for edge in cut_edgeset])
            join_edgeset = {(s, t)}
            cut_edgeset_weight = sum([
                subgraph_.get_edge_data(u, v).get('num_reviews', 1)
                for u, v in cut_edgeset])
            join_edgeset_weight = sum([
                subgraph.get_edge_data(u, v).get('num_reviews', 1)
                for u, v in join_edgeset])
            # Determine if this is more likely a split or a join
            if join_edgeset_weight < cut_edgeset_weight:
                maybe_error_edges.update(join_edgeset)
            else:
                maybe_error_edges.update(cut_edgeset)
        return list(maybe_error_edges)

    @profile
    def get_nomatch_ccs(infr, cc):
        """
        Search every neighbor in this cc for a nomatch connection. Then add the
        cc belonging to that connected node.
        In the case of an inconsistent cc, nodes within the cc will not be
        returned.
        """
        visited = set(cc)
        nomatch_ccs = []
        for n1 in cc:
            for n2 in infr.graph.neighbors(n1):
                if n2 not in visited:
                    data = infr.graph.get_edge_data(n1, n2)
                    _state = data.get('reviewed_state', 'unreviewed')
                    if _state == 'nomatch':
                        cc2 = infr.get_annot_cc(n2)
                        nomatch_ccs.append(cc2)
                        visited.update(set(cc2))
        return nomatch_ccs

    @profile
    def get_annot_cc(infr, node, graph=None):
        """
        Get the cc belonging to a single node
        """
        if graph is None:
            graph = infr.graph
        def condition(G, child, edge):
            u, v = edge
            nid1 = G.node[u]['name_label']
            nid2 = G.node[v]['name_label']
            return nid1 == nid2
        cc = set(ut.util_graph.bfs_conditional(
            infr.graph, node, yield_condition=condition,
            continue_condition=condition))
        cc.add(node)
        return cc

    def _compute_p_same(infr, p_match, p_notcomp):
        p_bg = 0.5  # Needs to be thresh value
        part1 = p_match * (1 - p_notcomp)
        part2 = p_bg * p_notcomp
        p_same = part1 + part2
        return p_same

    def _del_feedback_edges(infr, edges=None):
        if edges is None:
            edges = list(infr.graph.edges())
        if infr.verbose >= 2:
            print('[infr] _del_feedback_edges len(edges) = %r' % (len(edges)))
        ut.nx_delete_edge_attr(infr.graph, 'reviewed_weight', edges)
        ut.nx_delete_edge_attr(infr.graph, 'reviewed_state', edges)
        ut.nx_delete_edge_attr(infr.graph, 'reviewed_tags', edges)
        ut.nx_delete_edge_attr(infr.graph, 'num_reviews', edges)
        # ut.nx_delete_edge_attr(infr.graph, 'is_reviewed', edges)

    def _set_feedback_edges(infr, edges, review_state, p_same_list, tags_list):
        if infr.verbose >= 3:
            print('[infr] _set_feedback_edges')
        # Ensure edges exist
        for edge in edges:
            if not infr.graph.has_edge(*edge):
                infr.graph.add_edge(*edge)

        infr.set_edge_attrs('reviewed_state', _dz(edges, review_state))
        infr.set_edge_attrs('reviewed_weight', _dz(edges, p_same_list))
        infr.set_edge_attrs('reviewed_tags', _dz(edges, tags_list))

        import time
        timestamp = time.time()
        infr.set_edge_attrs('review_timestamp', _dz(edges, [timestamp]))
        # infr.set_edge_attrs('num_reviews', _dz(edges, tags_list))
        # infr.set_edge_attrs('is_reviewed', _dz(edges, [True]))

    @profile
    def apply_feedback_edges(infr):
        """
        Updates nx graph edge attributes for feedback

        CommandLine:
            python -m ibeis.algo.hots.graph_iden apply_feedback_edges

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.reset_feedback()
            >>> infr.apply_feedback_edges()
            >>> print('edges = ' + ut.repr4(infr.graph.edge))
            >>> # assert infr.graph.edge[6][5] is infr.graph.edge[5][6], 'digraph'
            >>> result = str(infr)
            >>> print(result)
            <AnnotInference(nAids=6, nEdges=2)>
        """
        if infr.verbose >= 1:
            print('[infr] apply_feedback_edges')
        infr.remove_dummy_edges()
        infr._del_feedback_edges()
        # Transforms dictionary feedback into numpy array
        feedback_edges = list(infr.user_feedback.keys())
        # Take most recent review
        num_review_list = [len(infr.user_feedback[edge]) for edge in feedback_edges]
        review_tag_list = [infr.user_feedback[edge][-1] for edge in feedback_edges]
        review_list = ut.take_column(review_tag_list, 0)
        tags_list = ut.take_column(review_tag_list, 1)
        p_nomatch = np.array(ut.dict_take_column(review_list, 'p_nomatch'))
        p_match = np.array(ut.dict_take_column(review_list, 'p_match'))
        p_notcomp = np.array(ut.dict_take_column(review_list, 'p_notcomp'))
        state_probs = np.vstack([p_nomatch, p_match, p_notcomp])
        review_stateid = state_probs.argmax(axis=0)
        review_state = ut.take(infr.truth_texts, review_stateid)
        p_same_list = infr._compute_p_same(p_match, p_notcomp)

        # Put pair orders in context of the graph
        unique_pairs = [(aid2, aid1) if infr.graph.has_edge(aid2, aid1) else
                        (aid1, aid2) for (aid1, aid2) in feedback_edges]
        infr._set_feedback_edges(unique_pairs, review_state, p_same_list, tags_list)
        infr.set_edge_attrs('num_reviews', _dz(unique_pairs, num_review_list))
        # infr.ensure_mst()

    def reset_feedback(infr):
        """ Resets feedback edges to state of the SQL annotmatch table """
        if infr.verbose >= 1:
            print('[infr] reset_feedback')
        infr.user_feedback = infr.read_ibeis_annotmatch_feedback()

    def remove_feedback(infr):
        """ Deletes all feedback """
        if infr.verbose >= 1:
            print('[infr] remove_feedback')
        infr.user_feedback = ut.ddict(list)


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrRelabel(object):

    def _next_nid(infr):
        if getattr(infr, 'nid_counter', None) is None:
            nids = nx.get_node_attributes(infr.graph, 'name_label')
            infr.nid_counter = max(nids)
        infr.nid_counter += 1
        new_nid = infr.nid_counter
        return new_nid

    def connected_component_reviewed_subgraphs(infr, graph=None):
        """
        Two kinds of edges are considered in connected component analysis: user
        reviewed edges, and algorithmally inferred edges.  If an inference
        algorithm is not run, then user review is all that matters.
        """
        if graph is None:
            graph = infr.graph
        # Make a graph where connections do indicate same names
        reviewed_states = nx.get_edge_attributes(graph, 'reviewed_state')
        graph2 = infr.graph_cls()
        keep_edges = [key for key, val in reviewed_states.items()
                      if val == 'match']
        graph2.add_nodes_from(graph.nodes())
        graph2.add_edges_from(keep_edges)
        ccs = list(nx.connected_components(graph2))
        cc_subgraphs = [infr.graph.subgraph(cc) for cc in ccs]
        return cc_subgraphs

    def connected_component_status(infr):
        r"""
        Returns:
            dict: num_inconsistent, num_names_max, num_names_min

        CommandLine:
            python -m ibeis.algo.hots.graph_iden connected_component_status

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
        all_reviewed_states = infr.get_edge_attrs('reviewed_state')
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

    @profile
    def relabel_using_reviews(infr, graph=None):
        if infr.verbose > 1:
            print('[infr] relabel_using_reviews')
        cc_subgraphs = infr.connected_component_reviewed_subgraphs(graph=graph)
        num_inconsistent = 0
        num_names = len(cc_subgraphs)

        # if graph is not None:
        #     available_nids = ut.unique(nx.get_node_attributes(graph, 'name_label'))

        # Determine which names can be reused
        from ibeis.scripts import name_recitifer
        grouped_oldnames = [list(nx.get_node_attributes(subgraph, 'name_label').values())
                            for count, subgraph in enumerate(cc_subgraphs)]
        # Make sure negatives dont get priority
        grouped_oldnames = [
            [n for n in group if len(group) == 1 or n > 0]
            for group in grouped_oldnames]
        new_labels = name_recitifer.find_consistent_labeling(grouped_oldnames)
        new_flags = [not isinstance(n, int) and n.startswith('_extra_name')
                     for n in new_labels]
        for idx in ut.where(new_flags):
            new_labels[idx] = infr._next_nid()

        for idx, label in enumerate(new_labels):
            if label < 0 and len(grouped_oldnames[idx]) > 1:
                # Remove negative ids for grouped items
                new_labels[idx] = infr._next_nid()

        for count, subgraph in enumerate(cc_subgraphs):
            reviewed_states = nx.get_edge_attributes(subgraph, 'reviewed_state')
            inconsistent_edges = [edge for edge, val in reviewed_states.items()
                                  if val == 'nomatch']
            if len(inconsistent_edges) > 0:
                #print('Inconsistent')
                num_inconsistent += 1

            # if graph is None:
            #     new_nid = count
            # else:
            new_nid = new_labels[count]
            # if count >= len(available_nids):
            #     new_nid = available_nids[count]
            # else:
            #     new_nid = infr._next_nid()

            infr.set_node_attrs('name_label',
                                _dz(list(subgraph.nodes()), [new_nid]))
            # Check for consistency
        return num_names, num_inconsistent

    def relabel_using_inference(infr, **kwargs):
        """
        Applies name labels based on graph inference and then cuts edges
        """
        if infr.verbose > 1:
            print('[infr] relabel_using_inference')

        infr.remove_dummy_edges()
        infr.model = infr_model.InfrModel(infr.graph, infr.CUT_WEIGHT_KEY)
        model = infr.model
        thresh = infr.get_threshold()
        model._update_weights(thresh=thresh)
        labeling, params = model.run_inference2(max_labels=len(infr.aids))

        infr.set_node_attrs('name_label', model.node_to_label)
        infr.apply_cuts()
        # infr.ensure_mst()

    def get_threshold(infr):
        # Only use the normalized scores to estimate a threshold
        normscores = np.array(infr.get_edge_attrs('normscore').values())
        if infr.verbose >= 1:
            print('len(normscores) = %r' % (len(normscores),))
        isvalid = ~np.isnan(normscores)
        curve = np.sort(normscores[isvalid])
        thresh = infr_model.estimate_threshold(curve, method=None)
        if infr.verbose >= 1:
            print('[estimate] thresh = %r' % (thresh,))
        if thresh is None:
            thresh = .5
        infr.thresh = thresh
        return thresh


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInference(ut.NiceRepr,
                     _AnnotInfrHelpers, _AnnotInfrIBEIS, _AnnotInfrMatching,
                     _AnnotInfrFeedback, _AnnotInfrRelabel, _AnnotInfrDummy,
                     viz_graph_iden._AnnotInfrViz):
    """
    class for maintaining state of an identification

    Notes:
        General workflow goes
        * Initialize Step
            * Add annots/names/configs/matches to AnnotInference Object
            * Apply Edges (mst/matches/feedback)
            * Apply Scores
            * Apply Weights
            * Apply Inference
        * Review Step
            * Get shortlist of results
            * Present results to user
            * Apply user feedback
            * Apply Inference
            * Record results
            * Repeat

    Terminology and Concepts:

        Node Attributes:
            * annotation id
            * original and current name label

        Each Attributes:
            * raw matching scores
            * pairwise features for learning?
            * measured probability match/notmatch/notcomp
            * inferred probability same/diff | features
            * User feedback

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
    graph_cls = nx.Graph
    # graph_cls = nx.DiGraph

    def __init__(infr, ibs, aids, nids=None, autoinit=False, verbose=False):
        infr.verbose = verbose
        if infr.verbose >= 1:
            print('[infr] __init__')
        infr.ibs = ibs
        infr.aids = aids
        infr.aids_set = set(infr.aids)
        if nids is None:
            if ibs is None:
                nids = [-aid for aid in aids]
            else:
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
        infr.thresh = None
        infr.cm_list = None
        infr.qreq_ = None
        infr.nid_counter = None
        infr.queue = None
        infr.queue_params = {
            'pos_jump_thresh': None,
            'neg_jump_thresh': None,
        }
        if autoinit:
            infr.initialize_graph()

    def copy(infr):
        import copy
        # deep copy everything but ibs
        infr2 = AnnotInference(
            infr.ibs, copy.deepcopy(infr.aids),
            copy.deepcopy(infr.orig_name_labels), autoinit=False,
            verbose=infr.verbose)
        infr2.graph = infr.graph.copy()
        infr2.user_feedback = copy.deepcopy(infr.user_feedback)
        infr2.cm_list = copy.deepcopy(infr.cm_list)
        infr2.qreq_ = copy.deepcopy(infr.qreq_)
        infr2.nid_counter = infr.nid_counter
        infr2.thresh = infr.thresh
        return infr2

    @classmethod
    def from_netx(cls, G):
        aids = list(G.nodes())
        nids = [-a for a in aids]
        infr = cls(None, aids, nids, autoinit=False)
        infr.graph = G
        infr.initialize_graph(G)
        return infr

    @classmethod
    def from_qreq_(cls, qreq_, cm_list, autoinit=False):
        """
        Create a AnnotInference object using a precomputed query / results
        """
        # raise NotImplementedError('do not use')
        aids = ut.unique(ut.flatten([qreq_.qaids, qreq_.daids]))
        nids = qreq_.get_qreq_annot_nids(aids)
        ibs = qreq_.ibs
        infr = cls(ibs, aids, nids, verbose=False, autoinit=autoinit)
        infr.cm_list = cm_list
        infr.qreq_ = qreq_
        return infr

    def __nice__(infr):
        if infr.graph is None:
            return 'nAids=%r, G=None' % (len(infr.aids))
        else:
            return 'nAids=%r, nEdges=%r' % (len(infr.aids),
                                              infr.graph.number_of_edges())

    def initialize_graph(infr, graph=None, update_nids=False):
        if infr.verbose >= 1:
            print('[infr] initialize_graph')
        if update_nids:
            if infr.ibs is None:
                nids = [-aid for aid in infr.aids]
            else:
                nids = infr.ibs.get_annot_nids(infr.aids)
            infr.orig_name_labels = nids
        if graph is None:
            infr.graph = infr.graph_cls()
            infr.graph.add_nodes_from(infr.aids)
        else:
            infr.graph = graph

        node_to_aid = {aid: aid for aid in infr.aids}
        infr.node_to_aid = node_to_aid
        node_to_nid = {aid: nid for aid, nid in
                       zip(infr.aids, infr.orig_name_labels)}
        assert len(node_to_nid) == len(node_to_aid), '%r - %r' % (
            len(node_to_nid), len(node_to_aid))
        infr.set_node_attrs('aid', node_to_aid)
        infr.set_node_attrs('name_label', node_to_nid)
        infr.set_node_attrs('orig_name_label', node_to_nid)
        infr.aid_to_node = ut.invert_dict(infr.node_to_aid)

    @profile
    def apply_weights(infr):
        """
        Combines normalized scores and user feedback into edge weights used in
        the graph cut inference.
        """
        if infr.verbose >= 1:
            print('[infr] apply_weights')
        ut.nx_delete_edge_attr(infr.graph, infr.CUT_WEIGHT_KEY)
        # mst not needed. No edges are removed

        edges = list(infr.graph.edges())
        edge_to_normscore = infr.get_edge_attrs('normscore')
        normscores = np.array(ut.dict_take(edge_to_normscore, edges, np.nan))

        edge_to_reviewed_weight = infr.get_edge_attrs('reviewed_weight')
        reviewed_weights = np.array(ut.dict_take(edge_to_reviewed_weight,
                                                 edges, np.nan))
        # Combine into weights
        weights = normscores.copy()
        has_review = ~np.isnan(reviewed_weights)
        weights[has_review] = reviewed_weights[has_review]
        # remove nans
        is_valid = ~np.isnan(weights)
        weights = weights.compress(is_valid, axis=0)
        edges = ut.compress(edges, is_valid)
        infr.set_edge_attrs(infr.CUT_WEIGHT_KEY, _dz(edges, weights))

    @profile
    def apply_cuts(infr, graph=None):
        """
        Cuts edges with different names and uncuts edges with the same name.
        """
        if infr.verbose >= 1:
            print('[infr] apply_cuts')
        # infr.ensure_mst()
        if graph is None:
            graph = infr.graph
        # ut.nx_delete_edge_attr(graph, 'is_cut')
        node_to_label = nx.get_node_attributes(graph, 'name_label')
        edge_to_cut = {(u, v): node_to_label[u] != node_to_label[v]
                       for (u, v) in graph.edges()}
        print('cut %d edges' % (sum(edge_to_cut.values())),)
        infr.set_edge_attrs('is_cut', edge_to_cut)
        # infr.apply_review_inference()

    @profile
    def get_filtered_edges(infr, review_cfg):
        """
        DEPRICATE OR MOVE

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
            review_cfg_defaults, review_cfg,
            assert_exists=False, iswarning=True
            # assert_exists=True, iswarning=True
        )

        ibs = infr.ibs
        graph = infr.graph
        nodes = list(graph.nodes())
        uv_list = list(graph.edges())

        node_to_aids = infr.get_node_attrs('aid')
        node_to_nids = infr.get_node_attrs('name_label')
        aids = ut.take(node_to_aids, nodes)
        nids = ut.take(node_to_nids, nodes)
        aid_to_nid = dict(zip(aids, nids))
        nid_to_aids = ut.group_items(aids, nids)

        # Initial set of edges
        aids1 = ut.take_column(uv_list, 0)
        aids2 = ut.take_column(uv_list, 1)

        num_filtered = 0

        if review_cfg['filter_nonmatch_between_ccs']:
            review_states = [
                graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')
                for edge in zip(aids1, aids2)]
            is_nonmatched = [state == 'nomatch' for state in review_states]
            #isneg_flags = is_nonmatched
            valid_flags = filter_between_ccs_neg(aids1, aids2, aid_to_nid,
                                                 nid_to_aids, is_nonmatched)
            num_filtered += len(valid_flags) - sum(valid_flags)
            aids1 = ut.compress(aids1, valid_flags)
            aids2 = ut.compress(aids2, valid_flags)

        if review_cfg['filter_photobombs']:
            # TODO: store photobomb status internally
            am_list = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
            ispb_flags = ibs.get_annotmatch_prop('Photobomb', am_list)
            valid_flags = filter_between_ccs_neg(aids1, aids2, aid_to_nid,
                                                 nid_to_aids, ispb_flags)
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

        # Hack, sort by scores
        scores = np.array([
            max(graph.get_edge_data(*edge).get('score', -1), -1)
            for edge in zip(aids1, aids2)])
        sortx = scores.argsort()[::-1]
        aids1 = ut.take(aids1, sortx)
        aids2 = ut.take(aids2, sortx)

        if review_cfg['max_num'] is not None:
            scores = np.array([
                # hack
                max(graph.get_edge_data(*edge).get('score', -1), -1)
                for edge in zip(aids1, aids2)])
            sortx = scores.argsort()[::-1]
            top_idx = sortx[:review_cfg['max_num']]
            aids1 = ut.take(aids1, top_idx)
            aids2 = ut.take(aids2, top_idx)

        # print('[infr] num_filtered = %r' % (num_filtered,))
        return aids1, aids2

    @profile
    def get_edges_for_review(infr, randomness=0, rng=None):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr()
            >>> infr.exec_matching()
            >>> infr.relabel_using_reviews()
            >>> infr.apply_match_edges(dict(ranks_top=3, ranks_bot=1))
            >>> infr.apply_match_scores()
            >>> infr.apply_weights()
            >>> infr.apply_review_inference()
            >>> infr.add_feedback(1, 2, 'match', apply=True)
            >>> infr.add_feedback(2, 3, 'match', apply=True)
            >>> infr.add_feedback(3, 1, 'nomatch', apply=True)
            >>> infr.add_feedback(3, 5, 'notcomp', apply=True)
            >>> infr.add_feedback(5, 6, 'match', apply=True)
            >>> infr.add_feedback(4, 5, 'nomatch', apply=True)
            >>> #infr.relabel_using_reviews()
            >>> #infr.apply_cuts()
            >>> edges = infr.get_edges_for_review()
            >>> filtered_edges = zip(*infr.get_filtered_edges({}))
            >>> print('edges = %r' % (edges,))

        Example2:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr()
            >>> infr.exec_matching()
            >>> infr.relabel_using_reviews()
            >>> infr.apply_match_edges(dict(ranks_top=3, ranks_bot=1))
            >>> infr.apply_match_scores()
            >>> infr.apply_weights()
            >>> edges1 = infr.get_edges_for_review()
            >>> print('edges1 = %s' % (ut.repr4(edges1),))
            >>> uv, why = edges[0]
            >>> infr.add_feedback(uv[0], uv[1], 'match', apply=True)
            >>> infr.relabel_using_reviews()
            >>> edges2 = infr.get_edges_for_review()
            >>> print('edges2 = %s' % (ut.repr4(edges2),))
        """
        graph = infr.graph

        node_to_nid = infr.get_node_attrs('name_label')

        rng = ut.ensure_rng(rng)

        # Candidate edges are unreviewed
        cand_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if ((d.get('reviewed_state', 'unreviewed') == 'unreviewed' and
                 d.get('inferred_state', None) is None) or
                d.get('maybe_error', False))
        ]
        error_flags = [d.get('maybe_error', False) for u, v, d in cand_edges]

        error_edges = ut.compress(cand_edges, error_flags)
        new_edges = ut.compress(cand_edges, ut.not_list(error_flags))
        # We only need one candidate edge between existing ccs
        cc_edge_id = [e_(node_to_nid[u], node_to_nid[v])
                      for u, v, d in new_edges]
        grouped_edges = ut.group_items(new_edges, cc_edge_id)

        priority_metric = 'normscore'
        # priority_metric = 'entropy'

        chosen_edges = []
        for key, group in grouped_edges.items():
            # group_uv = ut.take_column(group, [0, 1])
            group_data = ut.take_column(group, 2)
            group_priority = ut.dict_take_column(group_data, priority_metric, 0)
            idx = ut.argmax(group_priority)
            edge = group[idx]
            why = 'unreviewed'
            chosen_edges.append((edge, why))

        if randomness > 0:
            # Randomly review redundant edges
            redundant_edges = [
                (u, v, d) for u, v, d in graph.edges(data=True)
                if d.get('reviewed_state', 'unreviewed') == 'unreviewed' and
                d.get('inferred_state', None) == 'same'
            ]
            why = 'consistency check'
            flags = rng.rand(len(redundant_edges)) > (1 - randomness ** 2)
            for edge in ut.compress(redundant_edges, flags):
                chosen_edges.append((edge, why))

        for edge in error_edges:
            why = 'maybe_error'
            chosen_edges.append((edge, why))

        # Sort edges to review
        scores = np.array([
            # max(infr.graph.get_edge_data(u, v, d).get('entropy', -1), -1)
            max(infr.graph.get_edge_data(u, v, d).get(priority_metric, -1), -1)
            for u, v, d in ut.take_column(chosen_edges, 0)])

        if len(scores) > 0 and randomness > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            scores = randomness * rng.rand(len(scores)) + (1 - randomness) * scores

        sortx = scores.argsort()[::-1]
        needs_review_edges = ut.take(chosen_edges, sortx)
        return needs_review_edges

    def generate_reviews(infr, randomness=0, rng=None, pos_jump_thresh=None,
                         neg_jump_thresh=None):
        rng = ut.ensure_rng(rng)
        infr.queue_params['pos_jump_thresh'] = pos_jump_thresh
        infr.queue_params['neg_jump_thresh'] = neg_jump_thresh
        infr._init_priority_queue(randomness, rng)

        def get_next(idx=0):
            # edges = infr.get_edges_for_review(randomness, rng)
            try:
                edge, priority = infr.queue.pop()
                aid1, aid2 = edge
            except IndexError:
                print('no more edges to reveiw')
                raise StopIteration('no more to review!')
            # if len(edges) == 0:
            #     print('no more edges to reveiw')
            #     raise StopIteration('no more to review!')
            # chosen = edges[idx]
            # aid1, aid2 = chosen[0][0:2]
            return aid1, aid2

        for index in it.count():
            # if index % 2 == 0:
            yield get_next(idx=0)
            # else:
            #     yield get_next(idx=-1)


def testdata_infr(defaultdb='PZ_MTEST'):
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaultdb)
    aids = [1, 2, 3, 4, 5, 6]
    infr = AnnotInference(ibs, aids, autoinit=True)
    return infr


if __name__ == '__main__':
    r"""
    CommandLine:
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7 --graph-tab
        python -m ibeis.algo.hots.graph_iden
        python -m ibeis.algo.hots.graph_iden --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
