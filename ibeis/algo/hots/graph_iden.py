# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt
import itertools as it
import six
import collections
from ibeis.algo.hots import viz_graph_iden
from ibeis.algo.hots import graph_iden_depmixin
from ibeis.algo.hots import graph_iden_mixins
from ibeis.algo.hots import graph_iden_utils
from ibeis.algo.hots.graph_iden_utils import e_, _dz
from ibeis.algo.hots.graph_iden_new import AnnotInfr2
from ibeis.algo.hots.graph_iden_utils import bridges_inside, bridges_cross
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


DEBUG_CC = False
# DEBUG_CC = True


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
        if not isinstance(index, (pd.MultiIndex, pd.Index)):
            if len(index) == 0:
                index = pd.MultiIndex([[], []], [[], []],
                                      names=('aid1', 'aid2'))
            else:
                index = pd.MultiIndex.from_tuples(index, names=('aid1',
                                                                'aid2'))
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
        # edge_df = pd.DataFrame.from_records( records, columns=[key])
        # edge_df.index = index
        return edge_df

    def infr_pred_df(infr, edges=None):
        """ technically not groundtruth but current infererence predictions """
        return infr.edge_attr_df('inferred_state', edges, default=np.nan)


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

    def ensure_cliques(infr, label='name_label'):
        """
        Force each name label to be a clique
        """
        if infr.verbose:
            print('[infr] ensure_cliques')
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())
        new_edges = []
        for label, nodes in label_to_nodes.items():
            for edge in it.combinations(nodes, 2):
                if not infr.has_edge(edge):
                    new_edges.append(edge)
        if infr.verbose >= 2:
            print('[infr] adding %d clique edges' % (len(new_edges)))
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', _dz(new_edges, [True]))

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
                    new_edges.append(e_(*edge))
        return new_edges

    def ensure_mst(infr):
        """
        Use minimum spannning tree to ensure all names are connected
        Needs to be applied after any operation that adds/removes edges if we
        want to maintain that name labels must be connected in some way.
        """
        if infr.verbose >= 1:
            print('[infr] ensure_mst')
        new_edges = infr.find_mst_edges()
        # Add new MST edges to original graph
        if infr.verbose >= 2:
            print('[infr] adding %d MST edges' % (len(new_edges)))
        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('_dummy_edge', ut.dzip(new_edges, [True]))

    def review_dummy_edges(infr, method=1):
        """
        Creates just enough dummy reviews to maintain a consistent labeling if
        relabel_using_reviews is called. (if the existing edges are consistent).
        """
        if infr.verbose >= 2:
            print('[infr] review_dummy_edges')
        if method == 2:
            new_edges = infr.find_mst_edges2()
        else:
            new_edges = infr.find_mst_edges()

        if infr.verbose >= 1:
            print('[infr] reviewing %s dummy edges' % (len(new_edges),))
        # TODO apply set of new edges in bulk
        for u, v in new_edges:
            infr.add_feedback2((u, v), decision='match', confidence='guessing',
                               user_id='mst', verbose=False)
        infr.apply_feedback_edges()


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrIBEIS(object):
    """
    Direct interface into ibeis tables
    (most of these should not be used or be reworked)
    """

    def _pandas_feedback_format2(infr, feedback):
        import pandas as pd
        #am_rowids = np.array(ut.replace_nones(am_rowids, np.nan))
        dicts = []
        feedback = infr.internal_feedback
        for (u, v), vals in feedback.items():
            for val in vals:
                val = val.copy()
                val['aid1'] = u
                val['aid2'] = v
                val['tags'] = ';'.join(val['tags'])
                dicts.append(val)
        df = pd.DataFrame.from_dict(dicts)
        # df.sort('timestamp')
        return df

    def write_ibeis_staging_feedback(infr):
        """
        Commit all reviews in internal_feedback into the staging table.  The
        edges are removed from interal_feedback and added to external feedback.
        """
        if infr.verbose > 0:
            print('[infr] write_ibeis_staging_feedback %d' %
                  (len(infr.internal_feedback),))
        if len(infr.internal_feedback) == 0:
            return
        aid_1_list = []
        aid_2_list = []
        decision_list = []
        timestamp_list = []
        tags_list = []
        user_confidence_list = []
        identity_list = []
        ibs = infr.ibs
        _iter = (
            (aid1, aid2, feedback_item)
            for (aid1, aid2), feedbacks in infr.internal_feedback.items()
            for feedback_item in feedbacks
        )
        for aid1, aid2, feedback_item in _iter:
            decision_key = feedback_item['decision']
            tags = feedback_item['tags']
            timestamp = feedback_item.get('timestamp', None)
            confidence_key = feedback_item.get('confidence', None)
            user_id = feedback_item.get('user_id', None)
            decision_int = ibs.const.REVIEW.CODE_TO_INT[decision_key]
            confidence_int = infr.ibs.const.CONFIDENCE.CODE_TO_INT.get(
                    confidence_key, None)
            aid_1_list.append(aid1)
            aid_2_list.append(aid2)
            decision_list.append(decision_int)
            tags_list.append(tags)
            user_confidence_list.append(confidence_int)
            timestamp_list.append(timestamp)
            identity_list.append(user_id)
        review_id_list = ibs.add_review(
                aid_1_list, aid_2_list, decision_list,
                tags_list=tags_list,
                identity_list=identity_list,
                user_confidence_list=user_confidence_list,
                timestamp_list=timestamp_list)
        assert len(ut.find_duplicate_items(review_id_list)) == 0
        # Copy internal feedback into external
        for edge, feedbacks in infr.internal_feedback.items():
            infr.external_feedback[edge].extend(feedbacks)
        # Delete internal feedback
        infr.internal_feedback = ut.ddict(list)

    def write_ibeis_annotmatch_feedback(infr, edge_delta_df=None):
        """
        Commits the current state in external and internal into the annotmatch
        table.
        """
        if edge_delta_df is None:
            edge_delta_df = infr.match_state_delta(old='annotmatch', new='all')
        if infr.verbose > 0:
            print('[infr] write_ibeis_annotmatch_feedback %r' % (
                len(edge_delta_df)))
        ibs = infr.ibs
        edge_delta_df_ = edge_delta_df.reset_index()
        # Find the rows not yet in the annotmatch table
        is_add = edge_delta_df_['am_rowid'].isnull().values
        add_df = edge_delta_df_.loc[is_add]
        # Assign then a new annotmatch rowid
        add_ams = ibs.add_annotmatch_undirected(add_df['aid1'].values,
                                                add_df['aid2'].values)
        edge_delta_df_.loc[is_add, 'am_rowid'] = add_ams

        # Set residual matching data
        new_truth = ut.take(ibs.const.REVIEW.CODE_TO_INT,
                            edge_delta_df_['new_decision'])
        new_tags = [';'.join(tags) for tags in edge_delta_df_['new_tags']]
        new_conf = ut.dict_take(ibs.const.CONFIDENCE.CODE_TO_INT,
                                edge_delta_df_['new_user_confidence'], None)
        new_timestamp = edge_delta_df_['timestamp']
        new_reviewer = edge_delta_df_['user_id']
        am_rowids = edge_delta_df_['am_rowid'].values
        ibs.set_annotmatch_truth(am_rowids, new_truth)
        ibs.set_annotmatch_tag_text(am_rowids, new_tags)
        ibs.set_annotmatch_confidence(am_rowids, new_conf)
        ibs.set_annotmatch_reviewer(am_rowids, new_reviewer)
        ibs.set_annotmatch_posixtime_modified(am_rowids, new_timestamp)

    def write_ibeis_name_assignment(infr, name_delta_df=None):
        if name_delta_df is None:
            name_delta_df = infr.get_ibeis_name_delta()
        if infr.verbose > 0:
            print('[infr] write_ibeis_name_assignment %d' % len(name_delta_df))
        aid_list = name_delta_df.index.values
        new_name_list = name_delta_df['new_name'].values
        infr.ibs.set_annot_names(aid_list, new_name_list)

    def get_ibeis_name_delta(infr, ignore_unknown=True):
        """
        Rectifies internal name_labels with the names stored in the name table.
        """
        if infr.verbose >= 3:
            print('[infr] constructing name delta')
        import pandas as pd
        graph = infr.graph
        node_to_new_label = nx.get_node_attributes(graph, 'name_label')
        nodes = list(node_to_new_label.keys())
        aids = ut.take(infr.node_to_aid, nodes)
        old_names = infr.ibs.get_annot_name_texts(aids)
        # Indicate that unknown names should be replaced
        old_names = [None if n == infr.ibs.const.UNKNOWN else n
                     for n in old_names]
        new_labels = ut.take(node_to_new_label, aids)
        # Recycle as many old names as possible
        label_to_name, needs_assign, unknown_labels = infr._rectify_names(
            old_names, new_labels)
        if ignore_unknown:
            label_to_name = ut.delete_dict_keys(label_to_name, unknown_labels)
            needs_assign = ut.setdiff(needs_assign, unknown_labels)
        if infr.verbose >= 3:
            print('[infr] had %d unknown labels' % (len(unknown_labels)))
            print('ignore_unknown = %r' % (ignore_unknown,))
            print('[infr] need to make %d new names' % (len(needs_assign)))
        # Overwrite names of labels with temporary names
        needed_names = infr.ibs.make_next_name(len(needs_assign))
        for unassigned_label, new in zip(needs_assign, needed_names):
            label_to_name[unassigned_label] = new
        # Assign each node to the rectified label
        if ignore_unknown:
            unknown_labels_ = set(unknown_labels)
            node_to_new_label = {
                node: label for node, label in node_to_new_label.items()
                if label not in unknown_labels_
            }
        aid_list = ut.take(infr.node_to_aid, node_to_new_label.keys())
        new_name_list = ut.take(label_to_name, node_to_new_label.values())
        old_name_list = infr.ibs.get_annot_name_texts(aid_list)
        # Put into a dataframe for convinience
        name_delta_df_ = pd.DataFrame(
            {'old_name': old_name_list, 'new_name': new_name_list},
            columns=['old_name', 'new_name'],
            index=pd.Index(aid_list, name='aid')
        )
        changed_flags = name_delta_df_['old_name'] != name_delta_df_['new_name']
        name_delta_df = name_delta_df_[changed_flags]
        if infr.verbose >= 3:
            print('[infr] finished making name delta')
        return name_delta_df

    def read_ibeis_staging_feedback(infr):
        """
        Reads feedback from review staging table.
        """
        if infr.verbose >= 1:
            print('[infr] read_ibeis_staging_feedback')
        ibs = infr.ibs
        # annots = ibs.annots(infr.aids)
        review_ids = ibs.get_review_rowids_between(infr.aids)
        review_ids = sorted(review_ids)
        # aid_pairs = ibs.get_review_aid_tuple(review_ids)
        # flat_review_ids, cumsum = ut.invertible_flatten2(review_ids)

        from ibeis.control.manual_review_funcs import hack_create_aidpair_index
        hack_create_aidpair_index(ibs)

        from ibeis.control.manual_review_funcs import (
            REVIEW_AID1, REVIEW_AID2, REVIEW_COUNT, REVIEW_DECISION,
            REVIEW_USER_IDENTITY, REVIEW_USER_CONFIDENCE, REVIEW_TIMESTAMP,
            REVIEW_TAGS)

        colnames = (REVIEW_AID1, REVIEW_AID2, REVIEW_COUNT, REVIEW_DECISION,
                    REVIEW_USER_IDENTITY, REVIEW_USER_CONFIDENCE,
                    REVIEW_TIMESTAMP, REVIEW_TAGS)
        review_data = ibs.staging.get(ibs.const.REVIEW_TABLE, colnames,
                                      review_ids)

        feedback = ut.ddict(list)
        lookup_truth = ibs.const.REVIEW.INT_TO_CODE
        lookup_conf = ibs.const.CONFIDENCE.INT_TO_CODE

        for data in review_data:
            (aid1, aid2, count, decision_int,
             user_id, conf_int, timestamp, tags) = data
            edge = e_(aid1, aid2)
            feedback_item = {
                'decision': lookup_truth[decision_int],
                'timestamp': timestamp,
                'user_id': user_id,
                'tags': [] if not tags else tags.split(';'),
                'confidence': lookup_conf[conf_int],
            }
            feedback[edge].append(feedback_item)
        return feedback

    def read_ibeis_annotmatch_feedback(infr, only_existing_edges=False):
        """
        Reads feedback from annotmatch table.

        Args:
            only_existing_edges (bool): if True only reads info existing edges

        CommandLine:
            python -m ibeis.algo.hots.graph_iden read_ibeis_annotmatch_feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> feedback = infr.read_ibeis_annotmatch_feedback()
            >>> result =('feedback = %s' % (ut.repr2(feedback, nl=1),))
            >>> print(result)
            feedback = {
                (2, 3): [{'decision': 'match', 'tags': ['photobomb']}],
                (5, 6): [{'decision': 'match', 'tags': ['photobomb']}],
            }
        """
        if infr.verbose >= 1:
            print('[infr] read_ibeis_annotmatch_feedback')
        ibs = infr.ibs
        if only_existing_edges:
            aid_pairs = infr.graph.edges()
            am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        else:
            annots = ibs.annots(infr.aids)
            am_rowids, aid_pairs = annots.get_am_rowids_and_pairs()

        aids1 = ut.take_column(aid_pairs, 0)
        aids2 = ut.take_column(aid_pairs, 1)

        if infr.verbose >= 2:
            print('[infr] read %d annotmatch rowids' % (len(am_rowids)))

        if infr.verbose >= 2:
            print('[infr] * checking truth')

        # Use explicit truth state to mark truth
        truth = np.array(ibs.get_annotmatch_truth(am_rowids))
        if infr.verbose >= 2:
            print('[infr] * checking tags')
        tags_list = ibs.get_annotmatch_case_tags(am_rowids)
        confidence_list = ibs.get_annotmatch_confidence(am_rowids)
        timestamp_list = ibs.get_annotmatch_posixtime_modified(am_rowids)
        userid_list = ibs.get_annotmatch_reviewer(am_rowids)
        # Hack, if we didnt set it, it probably means it matched
        # FIXME: allow for truth to not be set.
        need_truth = np.array(ut.flag_None_items(truth)).astype(np.bool)
        if np.any(need_truth):
            need_aids1 = ut.compress(aids1, need_truth)
            need_aids2 = ut.compress(aids2, need_truth)
            needed_truth = ibs.get_aidpair_truths(need_aids1, need_aids2)
            truth[need_truth] = needed_truth

        truth = np.array(truth, dtype=np.int)

        if False:
            # Add information from relevant tags
            if infr.verbose >= 2:
                print('[infr] * checking split and joins')
            # Use tags to infer truth
            props = ['SplitCase', 'JoinCase']
            flags_list = ibs.get_annotmatch_prop(props, am_rowids)
            is_split, is_merge = flags_list
            is_split = np.array(is_split).astype(np.bool)
            is_merge = np.array(is_merge).astype(np.bool)
            # truth[is_pb] = ibs.const.REVIEW.NON_MATCH
            truth[is_split] = ibs.const.REVIEW.NON_MATCH
            truth[is_merge] = ibs.const.REVIEW.MATCH

        if infr.verbose >= 2:
            print('[infr] * making feedback dict')

        # CHANGE OF FORMAT
        lookup_truth = ibs.const.REVIEW.INT_TO_CODE
        lookup_conf = ibs.const.CONFIDENCE.INT_TO_CODE

        feedback = ut.ddict(list)
        for count, (aid1, aid2) in enumerate(zip(aids1, aids2)):
            edge = e_(aid1, aid2)
            conf = confidence_list[count]
            truth_ = truth[count]
            timestamp = timestamp_list[count]
            user_id = userid_list[count]
            if conf is not None and not isinstance(conf, int):
                import warnings
                warnings.warn('CONF WAS NOT AN INTEGER. conf=%r' % (conf,))
                conf = None
            decision = lookup_truth[truth_]
            conf_ = lookup_conf[conf]
            tag_ = tags_list[count]
            feedback_item = {
                'decision': decision,
                'timestamp': timestamp,
                'tags': tag_,
                'user_id': user_id,
                'confidence': conf_,
            }
            feedback[edge].append(feedback_item)
        if infr.verbose >= 1:
            print('[infr] read %d annotmatch entries' % (len(feedback)))
        return feedback

    def _pandas_feedback_format(infr, feedback):
        import pandas as pd
        aid_pairs = list(feedback.keys())
        aids1 = ut.take_column(aid_pairs, 0)
        aids2 = ut.take_column(aid_pairs, 1)
        ibs = infr.ibs
        am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1,
                                                                      aids2)
        rectified_feedback_ = infr._rectify_feedback(feedback)
        rectified_feedback = ut.take(rectified_feedback_, aid_pairs)
        decision = ut.dict_take_column(rectified_feedback, 'decision')
        tags = ut.dict_take_column(rectified_feedback, 'tags')
        confidence = ut.dict_take_column(rectified_feedback, 'confidence')
        df = pd.DataFrame([])
        df['decision'] = decision
        df['aid1'] = aids1
        df['aid2'] = aids2
        df['tags'] = tags
        df['confidence'] = confidence
        df['am_rowid'] = am_rowids
        df.set_index(['aid1', 'aid2'], inplace=True, drop=True)
        return df

    def match_state_delta(infr, old='annotmatch', new='all'):
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
            >>> infr.reset_feedback()
            >>> infr.add_feedback2((2, 3), 'match')
            >>> infr.add_feedback2((5, 6), 'nomatch')
            >>> infr.add_feedback2((5, 4), 'nomatch')
            >>> (edge_delta_df) = infr.match_state_delta()
            >>> result = ('edge_delta_df =\n%s' % (edge_delta_df,))
            >>> print(result)
        """
        def _lookup_feedback(key):
            if key == 'annotmatch':
                feedback = infr.read_ibeis_annotmatch_feedback()
                df = infr._pandas_feedback_format(feedback)
            elif key == 'staging':
                df = infr._pandas_feedback_format(infr.read_ibeis_staging_feedback())
            elif key == 'all':
                df = infr._pandas_feedback_format(infr.all_feedback())
            elif key == 'internal':
                df = infr._pandas_feedback_format(infr.internal_feedback)
            elif key == 'external':
                df = infr._pandas_feedback_format(infr.external_feedback)
            else:
                raise KeyError('key=%r' % (key,))
            return df

        old_feedback = _lookup_feedback(old)
        new_feedback = _lookup_feedback(new)
        edge_delta_df = infr._make_state_delta(old_feedback, new_feedback)
        return edge_delta_df

    def all_feedback_items(infr):
        for edge, vals in six.iteritems(infr.external_feedback):
            yield edge, vals
        for edge, vals in six.iteritems(infr.internal_feedback):
            yield edge, vals

    def all_feedback(infr):
        all_feedback = ut.ddict(list)
        all_feedback.update(infr.all_feedback_items())
        return all_feedback

    @staticmethod
    def _make_state_delta(old_feedback, new_feedback):
        r"""
        CommandLine:
            python -m ibeis.algo.hots.graph_iden _make_state_delta

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> import pandas as pd
            >>> columns = ['decision', 'aid1', 'aid2', 'am_rowid', 'tags']
            >>> new_feedback = old_feedback = pd.DataFrame([
            >>> ], columns=columns).set_index(['aid1', 'aid2'], drop=True)
            >>> edge_delta_df = AnnotInference._make_state_delta(old_feedback,
            >>>                                                  new_feedback)
            >>> result = ('edge_delta_df =\n%s' % (edge_delta_df,))
            >>> print(result)
            edge_delta_df =
            Empty DataFrame
            Columns: [am_rowid, old_decision, new_decision, old_tags, new_tags]
            Index: []

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> import pandas as pd
            >>> columns = ['decision', 'aid1', 'aid2', 'am_rowid', 'tags']
            >>> old_feedback = pd.DataFrame([
            >>>     ['nomatch', 100, 101, 1000, []],
            >>>     [  'match', 101, 102, 1001, []],
            >>>     [  'match', 103, 104, 1002, []],
            >>>     ['nomatch', 101, 104, 1004, []],
            >>> ], columns=columns).set_index(['aid1', 'aid2'], drop=True)
            >>> new_feedback = pd.DataFrame([
            >>>     [  'match', 101, 102, 1001, []],
            >>>     ['nomatch', 103, 104, 1002, []],
            >>>     [  'match', 101, 104, 1004, []],
            >>>     ['nomatch', 102, 103, None, []],
            >>>     ['nomatch', 100, 103, None, []],
            >>>     ['notcomp', 107, 109, None, []],
            >>> ], columns=columns).set_index(['aid1', 'aid2'], drop=True)
            >>> edge_delta_df = AnnotInference._make_state_delta(old_feedback,
            >>>                                                  new_feedback)
            >>> result = ('edge_delta_df =\n%s' % (edge_delta_df,))
            >>> print(result)
            edge_delta_df =
                       am_rowid old_decision new_decision old_tags new_tags
            aid1 aid2
            101  104     1004.0      nomatch        match       []       []
            103  104     1002.0        match      nomatch       []       []
            100  103        NaN          NaN      nomatch      NaN       []
            102  103        NaN          NaN      nomatch      NaN       []
            107  109        NaN          NaN      notcomp      NaN       []
        """
        import pandas as pd
        from six.moves import reduce
        import operator as op
        # Ensure input is in the expected format
        new_index = new_feedback.index
        old_index = old_feedback.index
        assert new_index.names == ['aid1', 'aid2'], ('not indexed on edges')
        assert old_index.names == ['aid1', 'aid2'], ('not indexed on edges')
        assert all(u < v for u, v in new_index.values), ('bad direction')
        assert all(u < v for u, v in old_index.values), ('bad direction')
        # Determine what edges have changed
        isect_edges = new_index.intersection(old_index)
        isect_new = new_feedback.loc[isect_edges]
        isect_old = old_feedback.loc[isect_edges]

        # If any important column is different we mark the row as changed
        data_columns = ['decision', 'tags', 'confidence']
        important_columns = ['decision', 'tags']
        other_columns = ut.setdiff(data_columns, important_columns)
        if len(isect_edges) > 0:
            changed_gen = (isect_new[c] != isect_old[c]
                           for c in important_columns)
            is_changed = reduce(op.or_, changed_gen)
            # decision_changed = isect_new['decision'] != isect_old['decision']
            # tags_changed = isect_new['tags'] != isect_old['tags']
            # is_changed = tags_changed | decision_changed
            new_df_ = isect_new[is_changed]
            old_df = isect_old[is_changed]
        else:
            new_df_ = isect_new
            old_df = isect_old
        # Determine what edges have been added
        add_edges = new_index.difference(old_index)
        add_df = new_feedback.loc[add_edges]
        # Concat the changed and added edges
        new_df = pd.concat([new_df_, add_df])
        # Prepare the data frames for merging
        old_colmap = {c: 'old_' + c for c in data_columns}
        new_colmap = {c: 'new_' + c for c in data_columns}
        prep_old = old_df.rename(columns=old_colmap).reset_index()
        prep_new = new_df.rename(columns=new_colmap).reset_index()
        # defer to new values for non-important columns
        for col in other_columns:
            oldcol = 'old_' + col
            if oldcol in prep_old:
                del prep_old[oldcol]
        # Combine into a single delta data frame
        merge_keys = ['aid1', 'aid2', 'am_rowid']
        merged_df = prep_old.merge(
            prep_new, how='outer', left_on=merge_keys, right_on=merge_keys)
        # Reorder the columns
        col_order = ['old_decision', 'new_decision', 'old_tags', 'new_tags']
        edge_delta_df = merged_df.reindex(columns=(
            ut.setdiff(merged_df.columns.values, col_order) + col_order))
        edge_delta_df.set_index(['aid1', 'aid2'], inplace=True, drop=True)
        return edge_delta_df


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrFeedback(object):
    @profile
    def add_feedback(infr, aid1=None, aid2=None, decision=None, tags=[],
                     apply=False, user_id=None, confidence=None,
                     edge=None, verbose=None, rectify=True):
        """
        Public interface to add feedback for a single edge to the buffer.
        Feedback is not applied to the graph unless `apply=True`.

        Args:
            aid1 (int):  annotation id
            aid2 (int):  annotation id
            decision (str): decision from `ibs.const.REVIEW.CODE_TO_INT`
            tags (list of str): specify Photobomb / Scenery / etc
            user_id (str): id of agent who did the review
            confidence (str): See ibs.const.CONFIDENCE
            apply (bool): if True feedback is dynamically applied

        CommandLine:
            python -m ibeis.algo.hots.graph_iden add_feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback2((5, 6), 'match')
            >>> infr.add_feedback2((5, 6), 'nomatch', ['Photobomb'])
            >>> infr.add_feedback2((1, 2), 'notcomp')
            >>> print(ut.repr2(infr.internal_feedback, nl=2))
            >>> assert len(infr.external_feedback) == 0
            >>> assert len(infr.internal_feedback) == 2
            >>> assert len(infr.internal_feedback[(5, 6)]) == 2
            >>> assert len(infr.internal_feedback[(1, 2)]) == 1
        """
        if verbose is None:
            verbose = infr.verbose
        if edge:
            aid1, aid2 = edge
        if verbose >= 1:
            print(('[infr] add_feedback(%r, %r, decision=%r, tags=%r, '
                                        'user_id=%r, confidence=%r)') % (
                aid1, aid2, decision, tags, user_id, confidence))

        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))
        assert isinstance(decision, six.string_types)
        edge = e_(aid1, aid2)

        if decision == 'unreviewed':
            feedback_item = None
            if edge in infr.external_feedback:
                raise ValueError(
                    "Can't unreview an edge that has been committed")
            if edge in infr.internal_feedback:
                del infr.internal_feedback[edge]
            for G in infr.review_graphs.values():
                if G.has_edge(*edge):
                    G.remove_edge(*edge)
        else:
            feedback_item = {
                'decision': decision,
                'tags': tags,
                'timestamp': ut.get_timestamp('int', isutc=True),
                'confidence': confidence,
                'user_id': user_id,
            }
            infr.internal_feedback[edge].append(feedback_item)
            # Add to appropriate review graph and change review if it existed
            # previously
            infr.review_graphs[decision].add_edge(*edge)
            for k, G in infr.review_graphs.items():
                if k != decision:
                    if G.has_edge(*edge):
                        G.remove_edge(*edge)

            if infr.refresh:
                infr.refresh.add(decision, user_id)

            if infr.test_mode:
                if user_id.startswith('auto'):
                    infr.test_state['n_auto'] += 1
                elif user_id == 'oracle':
                    infr.test_state['n_manual'] += 1
                else:
                    raise AssertionError('unknown user_id=%r' % (user_id,))

        if apply:
            # Apply new results on the fly
            infr._dynamically_apply_feedback(edge, feedback_item, rectify)

            if infr.test_mode:
                metrics = infr.measure_metrics()
                infr.metrics_list.append(metrics)
        else:
            assert not infr.test_mode, 'breaks tests'

    @profile
    def _dynamically_apply_feedback(infr, edge, feedback_item, rectify):
        """
        Dynamically updates all states based on a single dynamic change

        CommandLine:
            python -m ibeis.algo.hots.graph_iden _dynamically_apply_feedback:0
            python -m ibeis.algo.hots.graph_iden _dynamically_apply_feedback:1 --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.relabel_using_reviews()
            >>> infr.add_feedback2((1, 2), 'match')
            >>> infr.add_feedback2((2, 3), 'match')
            >>> infr.add_feedback2((2, 3), 'match')
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
            >>> ut.qtensure()
            >>> infr.ensure_full()
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback2((6, 2), 'match')
            >>> infr.add_feedback2((2, 3), 'match')
            >>> infr.add_feedback2((3, 4), 'match')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback2((2, 3), 'nomatch')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback2((6, 4), 'match')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback2((1, 5), 'nomatch')
            >>> infr.show_graph(show_cuts=True)
            >>> infr.add_feedback2((1, 3), 'nomatch')
            >>> infr.show_graph(show_cuts=True)
            >>> import plottool as pt
            >>> pt.present()
            >>> ut.show_if_requested()
        """
        if feedback_item is None:
            if infr.verbose >= 2:
                print('[infr] _dynamically_apply_feedback (removing edge=%r)'
                      % (edge,))
            state = 'unreviewed'
            infr._del_feedback_edges([edge])
            infr.set_edge_attrs(
                infr.CUT_WEIGHT_KEY,
                infr.get_edge_attrs('normscore', [edge], np.nan))
        else:
            # Apply the review to the specified edge
            state = feedback_item['decision']
            tags = feedback_item['tags']
            confidence = feedback_item['confidence']
            user_id = feedback_item['user_id']
            if infr.verbose >= 2:
                print('[infr] _dynamically_apply_feedback edge=%r, state=%r'
                      % (edge, state,))
            p_same_lookup = {
                'match': infr._compute_p_same(1.0, 0.0),
                'nomatch': infr._compute_p_same(0.0, 0.0),
                'notcomp': infr._compute_p_same(0.0, 1.0),
            }
            p_same = p_same_lookup[state]
            num_reviews = infr.get_edge_attrs('num_reviews', [edge],
                                              default=0).get(edge, 0)
            infr._set_feedback_edges([edge], [state], [p_same], [tags],
                                     [confidence],
                                     [user_id], [num_reviews + 1])
            # TODO: change num_reviews to num_consistent_reviews
            if state != 'notcomp':
                ut.nx_delete_edge_attr(infr.graph, 'inferred_state', [edge])

        # Dynamically update names and inferred attributes of relevant nodes
        # subgraph, subgraph_cuts = infr._get_influenced_subgraph(edge)
        n1, n2 = edge

        import utool
        with utool.embed_on_exception_context:
            cc1 = infr.pos_graph.connected_to(n1)
            cc2 = infr.pos_graph.connected_to(n2)
        relevant_nodes = cc1.union(cc2)
        if DEBUG_CC:
            cc1_ = infr.get_annot_cc(n1)
            cc2_ = infr.get_annot_cc(n2)
            relevant_nodes_ = cc1_.union(cc2_)
            assert relevant_nodes_ == relevant_nodes
            # print('seems good')

        subgraph = infr.graph.subgraph(relevant_nodes)

        # Change names of nodes
        infr.relabel_using_reviews(graph=subgraph, rectify=rectify)

        # Include other components where there are external consequences
        # This is only the case if two annotations are merged or a single
        # annotation is split.
        nomatch_ccs = infr.get_nomatch_ccs(relevant_nodes)
        extended_nodes = ut.flatten(nomatch_ccs)
        extended_nodes.extend(relevant_nodes)
        extended_subgraph = infr.graph.subgraph(extended_nodes)

        # This re-infers all attributes of the influenced sub-graph only
        infr.apply_review_inference(graph=extended_subgraph )

    def _del_feedback_edges(infr, edges=None):
        """ Delete all edges properties related to feedback """
        if edges is None:
            edges = list(infr.graph.edges())
        if infr.verbose >= 2:
            print('[infr] _del_feedback_edges len(edges) = %r' % (len(edges)))
        keys = ['decision', 'reviewed_tags', 'num_reviews',
                'reviewed_weight']
        ut.nx_delete_edge_attr(infr.graph, keys, edges)

    @profile
    def _set_feedback_edges(infr, edges, review_states, p_same_list, tags_list,
                            confidence_list, userid_list, n_reviews_list):
        if infr.verbose >= 3:
            edges = list(edges)
            print('[infr] _set_feedback_edges(nEdges=%d)' % (len(edges),))
        # Ensure edges exist
        for edge in edges:
            if not infr.graph.has_edge(*edge):
                infr.graph.add_edge(*edge)

        # use UTC timestamps
        timestamp = ut.get_timestamp('int', isutc=True)
        infr.set_edge_attrs('decision', _dz(edges, review_states))
        infr.set_edge_attrs('reviewed_weight', _dz(edges, p_same_list))
        infr.set_edge_attrs('reviewed_tags', _dz(edges, tags_list))
        infr.set_edge_attrs('confidence', _dz(edges, confidence_list))
        infr.set_edge_attrs('user_id', _dz(edges, userid_list))
        infr.set_edge_attrs('num_reviews', _dz(edges, n_reviews_list))
        infr.set_edge_attrs('review_timestamp', _dz(edges, [timestamp]))
        infr.set_edge_attrs(infr.CUT_WEIGHT_KEY, _dz(edges, p_same_list))

    @profile
    def apply_feedback_edges(infr, safe=True):
        r"""
        Transforms the feedback dictionaries into nx graph edge attributes

        CommandLine:
            python -m ibeis.algo.hots.graph_iden apply_feedback_edges

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.reset_feedback()
            >>> infr.add_feedback2((1, 2), 'unknown', tags=[])
            >>> infr.apply_feedback_edges()
            >>> print('edges = ' + ut.repr4(infr.graph.edge))
            >>> result = str(infr)
            >>> print(result)
            <AnnotInference(nAids=6, nEdges=2)>
        """
        if infr.verbose >= 1:
            print('[infr] apply_feedback_edges')
        if safe:
            # You can be unsafe if you know that the current feedback is a
            # strict superset of previous feedback
            infr._del_feedback_edges()
        # Transforms dictionary feedback into numpy array
        feedback_edges = []
        num_review_list = []
        decision_list = []
        confidence_list = []
        userid_list = []
        tags_list = []
        for edge, vals in infr.all_feedback_items():
            # hack for feedback rectification
            feedback_item = infr._rectify_feedback_item(vals)
            decision = feedback_item['decision']
            if decision == 'unknown':
                continue
            feedback_edges.append(edge)
            num_review_list.append(len(vals))
            userid_list.append(feedback_item['user_id'])
            decision_list.append(decision)
            tags_list.append(feedback_item['tags'])
            confidence_list.append(feedback_item['confidence'])

        p_same_lookup = {
            'match': infr._compute_p_same(1.0, 0.0),
            'nomatch': infr._compute_p_same(0.0, 0.0),
            'notcomp': infr._compute_p_same(0.0, 1.0),
        }
        p_same_list = ut.take(p_same_lookup, decision_list)

        # Put pair orders in context of the graph
        infr._set_feedback_edges(feedback_edges, decision_list, p_same_list,
                                 tags_list, confidence_list, userid_list,
                                 num_review_list)

    def reset_feedback(infr, mode='annotmatch'):
        """ Resets feedback edges to state of the SQL annotmatch table """
        if infr.verbose >= 1:
            print('[infr] reset_feedback mode=%r' % (mode,))
        if mode == 'annotmatch':
            infr.external_feedback = infr.read_ibeis_annotmatch_feedback()
        elif mode == 'staging':
            infr.external_feedback = infr.read_ibeis_staging_feedback()
        else:
            raise ValueError('no mode=%r' % (mode,))
        infr.internal_feedback = ut.ddict(list)

    def remove_feedback(infr, apply=False):
        """ Deletes all feedback """
        if infr.verbose >= 1:
            print('[infr] remove_feedback')
        infr.external_feedback = ut.ddict(list)
        infr.internal_feedback = ut.ddict(list)
        if apply:
            infr._del_feedback_edges()

    def _rectify_feedback(infr, feedback):
        return {edge: infr._rectify_feedback_item(vals)
                for edge, vals in feedback.items()}

    def _rectify_feedback_item(infr, vals):
        """ uses most recently use strategy """
        return vals[-1]

    def reset(infr, state='empty'):
        """
        Removes all edges from graph and resets name labels.
        """
        if state == 'empty':
            # Remove all edges, and component names
            infr.graph.remove_edges_from(list(infr.graph.edges()))
            infr.remove_feedback()
            infr.remove_name_labels()
        elif state == 'orig':
            raise NotImplementedError('unused')
            infr.graph.remove_edges_from(list(infr.graph.edges()))
            infr.remove_feedback()
            infr.reset_name_labels()
        else:
            raise ValueError('Unknown state=%r' % (state,))

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


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrMatching(object):
    def exec_matching(infr, prog_hook=None, cfgdict=None):
        """
        Loads chip matches into the inference structure
        Uses graph name labeling and ignores ibeis labeling
        """
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
                # 'augment_queryside_hack': True,
                'K': 3,
                'Knorm': 3,
                'prescore_method': 'csum',
                'score_method': 'csum'
            }
        # hack for ulsing current nids
        custom_nid_lookup = ut.dzip(aids, infr.get_annot_attrs('name_label',
                                                               aids))
        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup,
                                      verbose=infr.verbose >= 2)

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

    def _exec_pairwise_match(infr, edges, config={}, prog_hook=None):
        edges = ut.lmap(tuple, ut.aslist(edges))
        print('[infr] exec_vsone_subset')
        qaids = ut.take_column(edges, 0)
        daids = ut.take_column(edges, 1)
        # TODO: ensure feat/chip configs are resepected
        match_list = infr.ibs.depc.get('pairwise_match', (qaids, daids),
                                       'match', config=config)
        # recompute=True)
        # Hack: Postprocess matches to re-add annotation info in lazy-dict format
        from ibeis import core_annots
        config = ut.hashdict(config)
        configured_lazy_annots = core_annots.make_configured_annots(
            infr.ibs, qaids, daids, config, config, preload=True)
        for qaid, daid, match in zip(qaids, daids, match_list):
            match.annot1 = configured_lazy_annots[config][qaid]
            match.annot2 = configured_lazy_annots[config][daid]
            match.config = config
        return match_list

    def _enrich_matches_lnbnn(infr, matches, inplace=False):
        """
        applies lnbnn scores to pairwise one-vs-one matches
        """
        from ibeis.algo.hots import nn_weights
        qreq_ = infr.qreq_
        qreq_.load_indexer()
        indexer = qreq_.indexer
        if not inplace:
            matches_ = [match.copy() for match in matches]
        else:
            matches_ = matches
        K = qreq_.qparams.K
        Knorm = qreq_.qparams.Knorm
        normalizer_rule  = qreq_.qparams.normalizer_rule

        print('[inf] Stacking vecs for batch lnbnn matching')
        offset_list = np.cumsum([0] + [match_.fm.shape[0] for match_ in matches_])
        stacked_vecs = np.vstack([
            match_.matched_vecs2()
            for match_ in ut.ProgIter(matches_, label='stack matched vecs')
        ])

        vecs = stacked_vecs
        num = (K + Knorm)
        idxs, dists = indexer.batch_knn(vecs, num, chunksize=8192,
                                        label='lnbnn scoring')

        idx_list = [idxs[l:r] for l, r in ut.itertwo(offset_list)]
        dist_list = [dists[l:r] for l, r in ut.itertwo(offset_list)]
        iter_ = zip(matches_, idx_list, dist_list)
        prog = ut.ProgIter(iter_, nTotal=len(matches_), label='lnbnn scoring')
        for match_, neighb_idx, neighb_dist in prog:
            qaid = match_.annot2['aid']
            norm_k = nn_weights.get_normk(qreq_, qaid, neighb_idx, Knorm,
                                          normalizer_rule)
            ndist = vt.take_col_per_row(neighb_dist, norm_k)
            vdist = match_.local_measures['match_dist']
            lnbnn_dist = nn_weights.lnbnn_fn(vdist, ndist)
            lnbnn_clip_dist = np.clip(lnbnn_dist, 0, np.inf)
            match_.local_measures['lnbnn_norm_dist'] = ndist
            match_.local_measures['lnbnn'] = lnbnn_dist
            match_.local_measures['lnbnn_clip'] = lnbnn_clip_dist
            match_.fs = lnbnn_dist
        return matches_

    def _enriched_pairwise_matches(infr, edges, config={}, global_keys=None,
                                   need_lnbnn=True, prog_hook=None):
        if global_keys is None:
            global_keys = ['yaw', 'qual', 'gps', 'time']
        matches = infr._exec_pairwise_match(edges, config=config,
                                            prog_hook=prog_hook)
        print('[infr] enriching matches')
        if need_lnbnn:
            infr._enrich_matches_lnbnn(matches, inplace=True)
        # Ensure matches know about relavent metadata
        for match in matches:
            vt.matching.ensure_metadata_normxy(match.annot1)
            vt.matching.ensure_metadata_normxy(match.annot2)
        for match in ut.ProgIter(matches, label='setup globals'):
            match.add_global_measures(global_keys)
        for match in ut.ProgIter(matches, label='setup locals'):
            match.add_local_measures()
        return matches

    def _make_pairwise_features(infr, edges, config={}, pairfeat_cfg={},
                                global_keys=None, need_lnbnn=True,
                                multi_index=True):
        """
        Construct matches and their pairwise features
        """
        import pandas as pd
        # TODO: ensure feat/chip configs are resepected
        edges = ut.lmap(tuple, ut.aslist(edges))
        matches = infr._enriched_pairwise_matches(edges, config=config,
                                                  global_keys=global_keys,
                                                  need_lnbnn=need_lnbnn)
        # ---------------
        # Try different feature constructions
        print('[infr] building pairwise features')
        X = pd.DataFrame([
            m.make_feature_vector(**pairfeat_cfg)
            for m in ut.ProgIter(matches, label='making pairwise feats')
        ])
        if multi_index:
            # Index features by edges
            if len(edges) == 0:
                uv_index = pd.MultiIndex([[], []], [[], []],
                                         names=('aid1', 'aid2'))
            else:
                uv_index = pd.MultiIndex.from_tuples(edges,
                                                     names=('aid1', 'aid2'))
            X.index = uv_index
        X[pd.isnull(X)] = np.nan
        # Re-order column names to ensure dimensions are consistent
        X = X.reindex_axis(sorted(X.columns), axis=1)
        return matches, X

    def exec_vsone_subset(infr, edges, config={}, prog_hook=None):
        r"""
        Args:
            prog_hook (None): (default = None)

        CommandLine:
            python -m ibeis.algo.hots.graph_iden exec_vsone

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> config = {}
            >>> infr.ensure_full()
            >>> edges = [(1, 2), (2, 3)]
            >>> result = infr.exec_vsone_subset(edges)
            >>> print(result)
        """
        match_list = infr._exec_pairwise_match(edges, config=config,
                                               prog_hook=prog_hook)
        vsone_matches = {e_(u, v): match
                         for (u, v), match in zip(edges, match_list)}
        infr.vsone_matches.update(vsone_matches)
        edge_to_score = {e: match.fs.sum() for e, match in
                         vsone_matches.items()}
        infr.graph.add_edges_from(edge_to_score.keys())
        infr.set_edge_attrs('score', edge_to_score)
        return match_list

    def lookup_cm(infr, aid1, aid2):
        """
        Get chipmatch object associated with an edge if one exists.
        """
        if infr.cm_list is None:
            return None, aid1, aid2
        # TODO: keep chip matches in dictionary by default?
        aid2_idx = ut.make_index_lookup(
            [cm.qaid for cm in infr.cm_list])
        switch_order = False

        if aid1 in aid2_idx:
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
            if aid2 not in cm.daid2_idx:
                switch_order = True
                # raise KeyError('switch order')
        else:
            switch_order = True

        if switch_order:
            # switch order
            aid1, aid2 = aid2, aid1
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
            if aid2 not in cm.daid2_idx:
                raise KeyError('No ChipMatch for edge (%r, %r)' % (aid1, aid2))
        return cm, aid1, aid2

    @profile
    def apply_match_edges(infr, review_cfg={}):
        """
        Adds results from one-vs-many rankings as edges in the graph
        """
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

        CommandLine:
            python -m ibeis.algo.hots.graph_iden _cm_training_pairs

        Example:
            >>> # ENABLE_DOCTEST
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
            >>> rng = np.random.RandomState(42)
            >>> aid_pairs = np.array(infr._cm_training_pairs(rng=rng))
            >>> print(len(aid_pairs))
            >>> assert np.sum(aid_pairs.T[0] == aid_pairs.T[1]) == 0
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
            aid_pairs.extend([(cm.qaid, aid) for aid in chosen_daids if cm.qaid != aid])

        return aid_pairs

    def _get_cm_agg_aid_ranking(infr, cc):
        aid_to_cm = {cm.qaid: cm for cm in infr.cm_list}
        # node_to_cm = {infr.aid_to_node[cm.qaid]:
        #               cm for cm in infr.cm_list}
        all_scores = ut.ddict(list)
        for qaid in cc:
            cm = aid_to_cm[qaid]
            # should we be doing nids?
            for daid, score in zip(cm.get_top_aids(), cm.get_top_scores()):
                all_scores[daid].append(score)

        max_scores = sorted((max(scores), aid)
                            for aid, scores in all_scores.items())[::-1]
        ranked_aids = ut.take_column(max_scores, 1)
        return ranked_aids
        # aid = infr.aid_to_node[node]
        # node_to

    def _get_cm_edge_data(infr, edges):
        symmetric = True

        # Find scores for the edges that exist in the graph
        edge_to_data = ut.ddict(dict)
        node_to_cm = {infr.aid_to_node[cm.qaid]:
                      cm for cm in infr.cm_list}
        for u, v in edges:
            if symmetric:
                u, v = e_(u, v)
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

        # Hack away zero probabilites
        # probs = np.vstack([p_nomatch, p_match, p_notcomp]).T + 1e-9
        # probs = vt.normalize(probs, axis=1, ord=1, out=probs)
        # entropy = -(np.log2(probs) * probs).sum(axis=1)
        infr.set_edge_attrs('normscore', dict(zip(edges, normscores)))


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrPriority(object):
    """ for methods pertaining to the dynamic priority queue """

    PRIORITY_METRIC = 'normscore'

    def remaining_reviews(infr):
        assert infr.queue is not None
        return len(infr.queue)

    def _get_priorites(infr, edges, is_uvd=False):
        """
        returns priorities based on PRIORITY_METRIC and state of 'maybe_error'
        """
        if not is_uvd:
            uvds = ((u, v, infr.graph.get_edge_data(u, v)) for (u, v) in edges)
        else:
            uvds = edges
        new_priorities = np.array([
            d.get(infr.PRIORITY_METRIC, -1) + (2 * d.get('maybe_error', None))
            for u, v, d in uvds
        ])
        flags = np.isnan(new_priorities)
        if np.any(flags):
            # give nan values very small priority
            new_priorities[flags] = 1e-9
        # Need to augment priority of suggested fixes
        return new_priorities

    def _init_priority_queue(infr, randomness=0, rng=None):
        if infr.verbose:
            print('[infr] _init_priority_queue')
        graph = infr.graph

        # Candidate edges are unreviewed
        cand_uvds = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if (d.get('decision', 'unreviewed') == 'unreviewed' or
                d.get('maybe_error', None))
        ]

        # Sort edges to review
        priorities = infr._get_priorites(cand_uvds, is_uvd=True)
        edges = [e_(u, v) for u, v, d in cand_uvds]

        if len(priorities) > 0 and randomness > 0:
            rng = ut.ensure_rng(rng)
            minval = priorities.min()
            spread = priorities.max() - minval
            perb = (spread * rng.rand(len(priorities)) + minval)
            priorities = randomness * perb + (1 - randomness) * priorities

        # All operations on a treap except sorting use O(log(N)) time
        infr.queue = ut.PriorityQueue(zip(edges, -priorities))

    def pop(infr):
        try:
            edge, priority = infr.queue.pop()
        except IndexError:
            raise StopIteration('no more to review!')
            # raise StopIteration('no more to review!') from None
        else:
            assert edge[0] < edge[1]
            return edge, (priority * -1)

    def generate_reviews(infr, randomness=0, rng=None, pos_redundancy=None,
                         neg_redundancy=None, data=False):
        """
        Dynamic generator that yeilds high priority reviews
        """
        infr.queue_params['pos_redundancy'] = pos_redundancy
        infr.queue_params['neg_redundancy'] = neg_redundancy
        infr._init_priority_queue(randomness, rng)

        if data:
            while True:
                edge, priority = infr.pop()
                yield edge, priority
        else:
            while True:
                edge, priority = infr.pop()
                yield edge


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrUpdates(object):

    @profile
    def categorize_edges2(infr, graph=None):
        """
        Infers status of each edge in the graph.

        CommandLine:
            python -m ibeis.algo.hots.graph_iden categorize_edges2 --profile

        Example:
            >>> from ibeis.algo.hots import demo_graph_iden
            >>> infr = demo_graph_iden.demodata_infr(num_pccs=150)
            >>> graph = None
            >>> infr.categorize_edges2()
            >>> infr.categorize_edges()
            >>> infr.categorize_edges2()
            >>> infr.categorize_edges()
        """
        # consistent_ccs = []
        # inconsistent_ccs = []
        # for cc in infr.positive_connected_compoments(graph):
        #     if infr.is_consistent(cc):
        #         consistent_ccs.append(cc)
        #     else:
        #         inconsistent_ccs.append(cc)
        # Q = nx.quotient_graph(graph, consistent_ccs + inconsistent_ccs)

        def group_name_edges(edges):
            ne_to_edges = collections.defaultdict(set)
            name_edges = (
                e_(node_to_label[u], node_to_label[v])
                for u, v in edges
            )
            for edge, name_edge in zip(edges, name_edges):
                ne_to_edges[name_edge].add(edge)
            return ne_to_edges

        def group_name_edges_keys(edges):
            return {
                e_(node_to_label[u], node_to_label[v])
                for u, v in edges
            }
        POSTV = 'match'
        NEGTV = 'nomatch'
        INCMP = 'notcomp'
        UNREV = 'unreviewed'
        states = {POSTV, NEGTV, INCMP, UNREV}

        rev_graph = {s: None for s in states}
        rev_graph[POSTV] = infr.review_graphs[POSTV]
        rev_graph[NEGTV] = infr.review_graphs[NEGTV]
        rev_graph[INCMP] = infr.review_graphs[INCMP]
        rev_graph[UNREV] = infr.review_graphs[UNREV]
        if graph is None or graph is infr.graph:
            graph = infr.graph
            nodes = None
        else:
            # Need to extract relevant subgraphs
            nodes = list(graph.nodes())
            rev_graph[POSTV] = rev_graph[POSTV].subgraph(nodes)
            rev_graph[NEGTV] = rev_graph[NEGTV].subgraph(nodes)
            rev_graph[INCMP] = rev_graph[INCMP].subgraph(nodes)
            rev_graph[UNREV] = rev_graph[UNREV].subgraph(nodes)

        # Rebalance union find to ensure parents is a single lookup
        infr.pos_graph._union_find.rebalance(nodes)
        node_to_label = infr.pos_graph._union_find.parents
        # node_to_label = infr.pos_graph._union_find

        # Get reviewed edges using fast lookup structures
        lsm = ut.lstarmap
        rev_edge = {s: None for s in states}
        rev_edge[POSTV] = lsm(e_, rev_graph[POSTV].edges())
        rev_edge[NEGTV] = lsm(e_, rev_graph[NEGTV].edges())
        rev_edge[INCMP] = lsm(e_, rev_graph[INCMP].edges())
        rev_edge[UNREV] = lsm(e_, rev_graph[UNREV].edges())

        incmp_ne_to_edges = group_name_edges(rev_edge[INCMP])

        # Use reviewed edges to determine status of PCCs (repr by name ids)
        # The next steps will rectify duplicates in these sets
        name_edges = {s: None for s in states}
        name_edges[POSTV] = group_name_edges_keys(rev_edge[POSTV])
        name_edges[NEGTV] = group_name_edges_keys(rev_edge[NEGTV])
        name_edges[INCMP] = set(incmp_ne_to_edges.keys())
        name_edges[UNREV] = group_name_edges_keys(rev_edge[UNREV])

        # Positive and negative decisions override incomparable and unreviewed
        name_edges[INCMP].difference_update(name_edges[POSTV])
        name_edges[INCMP].difference_update(name_edges[NEGTV])
        name_edges[UNREV].difference_update(name_edges[POSTV])
        name_edges[UNREV].difference_update(name_edges[NEGTV])

        assert all(n1 == n2 for n1, n2 in name_edges[POSTV]), (
            'All positive edges should be internal to a PCC')

        # Negative edges within a PCC signals that an inconsistency exists
        # Remove inconsistencies from the name edges
        incon_internal_ne = name_edges[NEGTV].intersection(name_edges[POSTV])
        name_edges[POSTV].difference_update(incon_internal_ne)
        name_edges[NEGTV].difference_update(incon_internal_ne)

        assert len(name_edges[INCMP].intersection(incon_internal_ne)) == 0
        assert len(name_edges[UNREV].intersection(incon_internal_ne)) == 0

        # External inconsistentices are edges leaving inconsistent components
        assert all(n1 == n2 for n1, n2 in incon_internal_ne), (
            'incon_internal edges should be internal to a PCC')
        incon_internal_nids = {n1 for n1, n2 in incon_internal_ne}
        incon_external_ne = set([])
        # Find all edges leaving an inconsistent PCC
        for key in [NEGTV, INCMP, UNREV]:
            _ne = name_edges[key]
            incon_external_ne.update({
                (nid1, nid2) for nid1, nid2 in _ne
                if nid1 in incon_internal_nids or nid2 in incon_internal_nids
            })
        name_edges[NEGTV].difference_update(incon_external_ne)
        name_edges[INCMP].difference_update(incon_external_ne)
        name_edges[UNREV].difference_update(incon_external_ne)

        # Inference between names is now complete.
        # Now we expand this inference and project the labels onto the
        # annotation edges corresponding to each name edge.
        nid_to_cc = infr.pos_graph._ccs

        # Find edges within consistent PCCs
        positive = {
            nid1: (bridges_inside(graph, nid_to_cc[nid1]))
            for nid1, nid2 in name_edges[POSTV]
        }
        # Find edges between 1-negative-redundant consistent PCCs
        negative = {
            (nid1, nid2): (bridges_cross(graph, nid_to_cc[nid1], nid_to_cc[nid2]))
            for nid1, nid2 in name_edges[NEGTV]
        }
        # Find edges internal to inconsistent PCCs
        inconsistent_internal = {
            nid: (bridges_inside(graph, nid_to_cc[nid]))
            for nid in incon_internal_nids
        }
        # Find edges leaving inconsistent PCCs
        inconsistent_external = {
            (nid1, nid2): (bridges_cross(graph, nid_to_cc[nid1], nid_to_cc[nid2]))
            for nid1, nid2 in incon_external_ne
        }
        # No bridges are formed for notcomparable edges. Just take
        # the set of reviews
        notcomparable = {
            (nid1, nid2): incmp_ne_to_edges[(nid1, nid2)]
            for (nid1, nid2) in name_edges[INCMP]
        }
        unreviewed = {
            (nid1, nid2):
                bridges_cross(graph, nid_to_cc[nid1], nid_to_cc[nid2])
            for (nid1, nid2) in name_edges[UNREV]
        }
        # Removed not-comparable edges from unreviewed
        for name_edge in name_edges[UNREV].intersection(name_edges[INCMP]):
            unreviewed[name_edge].difference_update(notcomparable[name_edge])

        ne_categories = {
            'positive': positive,
            'negative': negative,
            'unreviewed': unreviewed,
            'notcomp': notcomparable,
            'inconsistent_internal': inconsistent_internal,
            'inconsistent_external': inconsistent_external,
        }
        return ne_categories

    @profile
    def apply_review_inference(infr, graph=None):
        """
        Updates the inferred state of each edge based on reviews and current
        labeling. State of the graph is only changed at the very end of the
        function.

        TODO: split into simpler functions

        CommandLine:
            python -m ibeis.algo.hots.graph_iden apply_review_inference

        Example:
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> aids = list(range(1, 10))
            >>> infr = AnnotInference(None, aids, autoinit=True, verbose=1)
            >>> infr.ensure_full()
            >>> infr._init_priority_queue()
            >>> infr.add_feedback2((1, 2), 'match')
            >>> infr.add_feedback2((2, 3), 'match')
            >>> infr.add_feedback2((2, 3), 'match')
            >>> infr.add_feedback2((3, 4), 'match')
            >>> infr.add_feedback2((4, 5), 'nomatch')
            >>> infr.add_feedback2((6, 7), 'match')
            >>> infr.add_feedback2((7, 8), 'match')
            >>> infr.add_feedback2((6, 8), 'nomatch')
            >>> infr.add_feedback2((6, 1), 'notcomp')
            >>> infr.add_feedback2((1, 9), 'notcomp')
            >>> infr.add_feedback2((8, 9), 'notcomp')
            >>> #infr.show_graph(hide_cuts=False)
            >>> graph = infr.graph
            >>> infr.apply_review_inference(graph)
        """
        if graph is None:
            graph = infr.graph

        if infr.verbose >= 2:
            print('[infr] apply_review_inference on %d nodes' % (len(graph)))

        categories = infr.categorize_edges(graph)
        ne_categories = categories['ne_categories']
        reviewed_positives = categories['reviewed_positives']
        reviewed_negatives = categories['reviewed_negatives']
        edge_to_reviewstate = categories['edge_to_reviewstate']
        nid_to_cc = categories['nid_to_cc']
        node_to_label = categories['node_to_label']
        all_edges = categories['all_edges']

        # Find possible fixes for inconsistent components
        if infr.verbose >= 2:
            if ne_categories['inconsistent_internal']:
                print('[infr] found %d inconsistencies searching for fixes' %
                      (len(ne_categories['inconsistent_internal']),))
            else:
                print('no inconsistencies')

        suggested_fix_edges = []
        other_error_edges = []
        if infr.method == 'graph':
            # dont do this in ranking mode
            # Check for inconsistencies
            for nid, cc_incon_edges in ne_categories['inconsistent_internal'].items():
                # Find possible edges to fix in the reviewed subgarph
                reviewed_inconsistent = [
                    (u, v, infr.graph.edge[u][v].copy()) for (u, v) in cc_incon_edges
                    if edge_to_reviewstate[(u, v)] != 'unreviewed'
                ]
                subgraph = nx.Graph(reviewed_inconsistent)
                # TODO: only need to use one fix edge here.
                cc_error_edges = infr._find_possible_error_edges(subgraph)
                import utool
                with utool.embed_on_exception_context:
                    assert len(cc_error_edges) > 0, 'no fixes found'
                cc_other_edges = ut.setdiff(cc_incon_edges, cc_error_edges)
                suggested_fix_edges.extend(cc_error_edges)
                other_error_edges.extend(cc_other_edges)
                # just add one for now
                # break

        if infr.verbose >= 2 and ne_categories['inconsistent_internal']:
            print('[infr] found %d possible fixes' % len(suggested_fix_edges))

        edge_categories = {k: ut.flatten(v.values())
                           for k, v in ne_categories.items()}

        # if __debug__:
        #     infr._debug_edge_categories(graph, ne_categories, edge_categories,
        #                                 node_to_label)

        # Update the attributes of all edges in the subgraph

        # Update the infered state
        infr.set_edge_attrs('inferred_state', _dz(
            edge_categories['inconsistent_external'], ['inconsistent_external']))
        infr.set_edge_attrs('inferred_state', _dz(
            edge_categories['inconsistent_internal'], ['inconsistent_internal']))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['unreviewed'], [None]))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['notcomp'], ['notcomp']))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['positive'], ['same']))
        infr.set_edge_attrs('inferred_state', _dz(edge_categories['negative'], ['diff']))

        # Suggest possible fixes
        infr.set_edge_attrs('maybe_error', ut.dzip(all_edges, [None]))
        infr.set_edge_attrs('maybe_error', _dz(suggested_fix_edges, [True]))

        # Update the cut state
        # TODO: DEPRICATE the cut state is not relevant anymore
        infr.set_edge_attrs('is_cut', _dz(edge_categories['inconsistent_external'],
                                          [True]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['inconsistent_internal'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['unreviewed'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['notcomp'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['positive'], [False]))
        infr.set_edge_attrs('is_cut', _dz(edge_categories['negative'], [True]))

        # Update basic priorites
        # FIXME: this must agree with queue
        # infr.set_edge_attrs('priority', infr.get_edge_attrs(
        #         infr.PRIORITY_METRIC, inconsistent_external_edges,
        #         default=.01))
        # infr.set_edge_attrs('priority', infr.get_edge_attrs(
        #         infr.PRIORITY_METRIC, incon_intern_edges, default=.01))
        # infr.set_edge_attrs('priority', infr.get_edge_attrs(
        #         infr.PRIORITY_METRIC, unreviewed_edges, default=.01))
        # infr.set_edge_attrs('priority', _dz(notcomparable_edges, [0]))
        # infr.set_edge_attrs('priority', _dz(positive_edges, [0]))
        # infr.set_edge_attrs('priority', _dz(negative_edges, [0]))
        # infr.set_edge_attrs('priority', _dz(suggested_fix_edges, [2]))

        # print('suggested_fix_edges = %r' % (sorted(suggested_fix_edges),))
        # print('other_error_edges = %r' % (sorted(other_error_edges),))

        if infr.queue is not None:
            # hack this off if method is ranking, we dont want to update any
            # priority
            if infr.method != 'ranking':
                if infr.verbose >= 2:
                    print('[infr] updating priority queue')
                infr._update_priority_queue(graph, ne_categories['positive'],
                                            ne_categories['negative'],
                                            reviewed_positives,
                                            reviewed_negatives, node_to_label,
                                            nid_to_cc, suggested_fix_edges,
                                            other_error_edges,
                                            edge_categories['unreviewed'])
        else:
            if infr.verbose >= 2:
                print('[infr] no priority queue to update')
        if infr.verbose >= 3:
            print('[infr] finished review inference')

    @profile
    def _find_possible_error_edges(infr, subgraph):
        """
        Args:
            subgraph (nx.Graph): a subgraph of a positive compomenent
                with only reviewed edges.
        """
        inconsistent_edges = [
            edge for edge, state in
            nx.get_edge_attributes(subgraph, 'decision').items()
            if state == 'nomatch'
        ]
        maybe_error_edges = set([])
        # subgraph_ = infr.simplify_graph(subgraph, copy=copy)
        subgraph_ = subgraph.copy()
        subgraph_.remove_edges_from(inconsistent_edges)

        # This is essentially solving a multicut problem for multiple pairs of
        # terminal nodes. The multiple min-cut runs produces a feasible
        # solution. Could use a multicut approximation.

        ut.nx_set_default_edge_attributes(subgraph_, 'num_reviews', 1)
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
            # if len(cut_edgeset) == 0:
            #     maybe_error_edges.update(join_edgeset)
            if join_edgeset_weight < cut_edgeset_weight:
                maybe_error_edges.update(join_edgeset)
            else:
                maybe_error_edges.update(cut_edgeset)

        maybe_error_edges_ = ut.lstarmap(e_, maybe_error_edges)
        return maybe_error_edges_

    @profile
    def _update_priority_queue(infr, graph, positive, negative,
                               reviewed_positives, reviewed_negatives,
                               node_to_label, nid_to_cc, suggested_fix_edges,
                               other_error_edges, unreviewed_edges):
        r"""
        TODO refactor

        Example:
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> from ibeis.algo.hots.graph_iden import _dz
            >>> from ibeis.algo.hots import demo_graph_iden
            >>> infr = demo_graph_iden.make_demo_infr(
            >>>     ccs=[[1, 2, 3, 4, 5],
            >>>            [6, 7, 8, 9, 10]],
            >>>     edges=[
            >>>         #(1, 6, {'decision': 'nomatch'}),
            >>>         (1, 6, {}),
            >>>         (4, 9, {}),
            >>>     ]
            >>> )
            >>> infr._init_priority_queue()
            >>> assert len(infr.queue) == 2
            >>> infr.queue_params['neg_redundancy'] = None
            >>> infr.add_feedback2((1, 6), 'nomatch')
            >>> assert len(infr.queue) == 0
            >>> graph = infr.graph
            >>> ut.exec_func_src(infr.apply_review_inference,
            >>>                  sentinal='if infr.queue is not None', stop=-1,
            >>>                  verbose=True)
            >>> infr.queue_params['neg_redundancy'] = 1
            >>> infr.apply_review_inference()
        """
        # update the priority queue on the fly
        queue = infr.queue
        pos_redundancy = infr.queue_params['pos_redundancy']
        neg_redundancy = infr.queue_params['neg_redundancy']

        if neg_redundancy and neg_redundancy < np.inf:
            # Remove priority of PCC-pairs with k-negative edges between them
            for (nid1, nid2), neg_edges in reviewed_negatives.items():
                if len(neg_edges) >= neg_redundancy:
                    other_edges = negative[(nid1, nid2)]
                    queue.delete_items(other_edges)

        if pos_redundancy and pos_redundancy < np.inf:
            # Remove priority internal edges of k-consistent PCCs.
            for nid, pos_edges in reviewed_positives.items():
                if pos_redundancy == 1:
                    # trivially computed
                    pos_conn = 1
                else:
                    pos_conn = nx.edge_connectivity(nx.Graph(list(pos_edges)))
                if pos_conn >= pos_redundancy:
                    other_edges = positive[nid]
                    queue.delete_items(other_edges)

        if suggested_fix_edges:
            # Add error edges back in with higher priority
            queue.update(zip(suggested_fix_edges,
                             -infr._get_priorites(suggested_fix_edges)))

            queue.delete_items(other_error_edges)

        needs_priority = [e for e in unreviewed_edges if e not in queue]
        queue.update(zip(needs_priority, -infr._get_priorites(needs_priority)))

    @profile
    def get_nomatch_ccs(infr, cc):
        """
        Returns a set of PCCs that are known to have at least one negative
        match to any node in the input nodes.

        Search every neighbor in this cc for a nomatch connection. Then add the
        cc belonging to that connected node.  In the case of an inconsistent
        cc, nodes within the cc will not be returned.
        """
        if DEBUG_CC:
            visited = set(cc)
            # visited_nodes = set([])
            nomatch_ccs = []
            for n1 in cc:
                for n2 in infr.graph.neighbors(n1):
                    if n2 not in visited:
                        # data = infr.graph.get_edge_data(n1, n2)
                        # _state = data.get('decision', 'unreviewed')
                        _state = infr.graph.edge[n1][n2].get('decision',
                                                             'unreviewed')
                        if _state == 'nomatch':
                            cc2 = infr.get_annot_cc(n2)
                            nomatch_ccs.append(cc2)
                            visited.update(cc2)
            nomatch_ccs_old = nomatch_ccs
        else:
            neg_graph = infr.neg_graph
            pos_graph = infr.pos_graph
            cc_labels = {
                pos_graph.node_label(n2)
                for n1 in cc
                for n2 in neg_graph.neighbors(n1)
            }
            nomatch_ccs = [pos_graph.connected_to(node)
                           for node in cc_labels]
        if DEBUG_CC:
            assert nomatch_ccs_old == nomatch_ccs
        return nomatch_ccs

    @profile
    def get_annot_cc(infr, source, visited_nodes=None):
        """
        Get the name_label cc connected to `source`

        TODO:
            Currently instead of using BFS to find the connected compoments
            each time dynamically maintain connected compoments as new
            information is added.

            The problem is "Dynamic Connectivity"

            Union-find can be used as long as no edges are deleted

            Refactor to a union-split-find data structure
                https://courses.csail.mit.edu/6.851/spring14/lectures/L20.html
                http://cs.stackexchange.com/questions/33595/maintaining-connect
                http://cs.stackexchange.com/questions/32077/
                https://networkx.github.io/documentation/development/_modules/
                    networkx/utils/union_find.html
        """
        # Speed hack for BFS conditional
        G = infr.graph
        cc = set([source])
        queue = collections.deque([])
        # visited_nodes = set([source])
        if visited_nodes is None:
            visited_nodes = set([])
        if source not in visited_nodes:
            visited_nodes.add(source)
            new_edges = iter([(source, n) for n in G.adj[source]])
            queue.append((source, new_edges))
        while queue:
            parent, edges = queue[0]
            parent_attr = G.node[parent]['name_label']
            for edge in edges:
                child = edge[1]
                # only move forward if the child shares name_label
                if child not in visited_nodes:
                    visited_nodes.add(child)
                    if parent_attr == G.node[child]['name_label']:
                        cc.add(child)
                        new_edges = iter([(child, n) for n in G.adj[child]])
                        queue.append((child, new_edges))
            queue.popleft()
        # def condition(G, child, edge):
        #     u, v = edge
        #     nid1 = G.node[u]['name_label']
        #     nid2 = G.node[v]['name_label']
        #     return nid1 == nid2
        # cc = set(ut.util_graph.bfs_same_attr_nodes(infr.graph, node,
        #                                            key='name_label'))
        # cc = set(ut.util_graph.bfs_conditional(
        #     infr.graph, node, yield_condition=condition,
        #     continue_condition=condition))
        # cc.add(node)
        return cc


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrRelabel(object):
    def _next_nid(infr):
        if getattr(infr, 'nid_counter', None) is None:
            nids = nx.get_node_attributes(infr.graph, 'name_label')
            infr.nid_counter = max(nids)
        infr.nid_counter += 1
        new_nid = infr.nid_counter
        return new_nid

    def is_consistent(infr, cc):
        """ Returns False if cc contains any negative edges """
        return (len(cc) <= 2 or
                infr.neg_graph.subgraph(cc).number_of_edges() == 0)

    def inconsistent_components(infr, graph=None):
        """
        Return components without nomatch edges
        """
        # Find PCCs with negative edges
        for cc in infr.positive_connected_compoments(graph):
            if not infr.is_consistent(cc):
                yield cc
        # cc_subgraphs = infr.positive_connected_compoments(graph)
        # inconsistent_subgraphs = []
        # for subgraph in cc_subgraphs:
        #     edge_to_state = nx.get_edge_attributes(subgraph, 'decision')
        #     if any(state == 'nomatch' for state in edge_to_state.values()):
        #         inconsistent_subgraphs.append(subgraph)
        # return inconsistent_subgraphs

    def consistent_components(infr, graph=None):
        """
        Return components without nomatch edges
        """
        # Find PCCs without any negative edges
        for cc in infr.positive_connected_compoments(graph):
            if infr.is_consistent(cc):
                yield cc
        # cc_subgraphs = infr.positive_connected_compoments(graph)
        # # inconsistent_subgraphs = []
        # consistent_subgraphs = []
        # for subgraph in cc_subgraphs:
        #     edge_to_state = nx.get_edge_attributes(subgraph, 'decision')
        #     if not any(state == 'nomatch' for state in edge_to_state.values()):
        #         consistent_subgraphs.append(subgraph)
        #     # else:
        #     #     inconsistent_subgraphs.append(subgraph)
        # return consistent_subgraphs

    @profile
    def positive_connected_compoments(infr, graph=None):
        """
        Returns the positive connected compoments (PCCs)
        """
        pos_graph = infr.pos_graph
        if graph is None or graph is infr.graph:
            ccs = pos_graph.connected_components()
        else:
            unique_labels = {
                pos_graph.node_label(node) for node in graph.nodes()}
            ccs = (pos_graph.connected_to(node) for node in unique_labels)
        for cc in ccs:
            yield cc
        # cc_subgraphs = [infr.graph.subgraph(cc) for cc in ccs]
        # return cc_subgraphs
        # if graph is None:
        #     graph = infr.graph
        # # Make a graph where connections do indicate same names
        # decisions = nx.get_edge_attributes(graph, 'decision')
        # graph2 = infr._graph_cls()
        # keep_edges = [key for key, val in decisions.items()
        #               if val == 'match']
        # graph2.add_nodes_from(graph.nodes())
        # graph2.add_edges_from(keep_edges)
        # ccs = list(nx.connected_components(graph2))
        # cc_subgraphs = [infr.graph.subgraph(cc) for cc in ccs]
        # return cc_subgraphs

    @profile
    def reset_labels_to_ibeis(infr):
        """ Sets to IBEIS de-facto labels if available """
        nids = infr._rectify_nids(infr.aids, None)
        nodes = ut.take(infr.aid_to_node, infr.aids)
        infr.set_node_attrs('name_label', ut.dzip(nodes, nids))

    def _rectify_names(infr, old_names, new_labels):
        """
        Finds the best assignment of old names based on the new groups each is
        assigned to.

        old_names  = [None, None, None, 1, 2, 3, 3, 4, 4, 4, 5, None]
        new_labels = [   1,    2,    2, 3, 4, 5, 5, 6, 3, 3, 7, 7]
        """
        if infr.verbose >= 3:
            print('rectifying name lists')
        from ibeis.scripts import name_recitifer
        newlabel_to_oldnames = ut.group_items(old_names, new_labels)
        unique_newlabels = list(newlabel_to_oldnames.keys())
        grouped_oldnames_ = ut.take(newlabel_to_oldnames, unique_newlabels)
        # Mark annots that are unknown and still grouped by themselves
        still_unknown = [len(g) == 1 and g[0] is None for g in grouped_oldnames_]
        # Remove nones for name rectifier
        grouped_oldnames = [
            [n for n in oldgroup if n is not None]
            for oldgroup in grouped_oldnames_]
        new_names = name_recitifer.find_consistent_labeling(
            grouped_oldnames, verbose=infr.verbose >= 3, extra_prefix=None)

        unknown_labels = ut.compress(unique_newlabels, still_unknown)

        new_flags = [n is None for n in new_names]
        #     isinstance(n, six.string_types) and n.startswith('_extra_name')
        #     for n in new_names
        # ]
        label_to_name = ut.dzip(unique_newlabels, new_names)
        needs_assign = ut.compress(unique_newlabels, new_flags)
        return label_to_name, needs_assign, unknown_labels

    def _rectified_relabel(infr, cc_subgraphs):
        # Determine which names can be reused
        from ibeis.scripts import name_recitifer
        if infr.verbose >= 3:
            print('grouping names for rectification')
        grouped_oldnames_ = [
            list(nx.get_node_attributes(subgraph, 'name_label').values())
            for count, subgraph in enumerate(cc_subgraphs)
        ]
        # Make sure negatives dont get priority
        grouped_oldnames = [
            [n for n in group if len(group) == 1 or n > 0]
            for group in grouped_oldnames_
        ]
        if infr.verbose >= 2:
            print('begin rectification of %d grouped old names' % (
                len(grouped_oldnames)))
        new_labels = name_recitifer.find_consistent_labeling(
            grouped_oldnames, verbose=infr.verbose >= 3)
        if infr.verbose >= 2:
            print('done rectifying new names')
        new_flags = [
            not isinstance(n, int) and n.startswith('_extra_name')
            for n in new_labels
        ]

        for idx in ut.where(new_flags):
            new_labels[idx] = infr._next_nid()

        for idx, label in enumerate(new_labels):
            if label < 0 and len(grouped_oldnames[idx]) > 1:
                # Remove negative ids for grouped items
                new_labels[idx] = infr._next_nid()
        return new_labels

    @profile
    def relabel_using_reviews(infr, graph=None, rectify=True):
        r"""
        Relabels nodes in graph based on poasitive-review connected components

        Args:
            graph (nx.Graph, optional): only edges in `graph` are relabeled
            rectify (bool, optional): if True names attempt to remain
                consistent otherwise there are no restrictions on name labels
                other than that they are distinct.
        """
        if infr.verbose >= 2:
            print('[infr] relabel_using_reviews')
        if graph is None:
            graph = infr.graph
        cc_subgraphs = [
            infr.graph.subgraph(cc)
            for cc in infr.positive_connected_compoments(graph=graph)
        ]

        # Check consistentcy
        num_inconsistent = 0
        for subgraph in cc_subgraphs:
            edge_to_state = nx.get_edge_attributes(subgraph, 'decision')
            if any(state == 'nomatch' for state in edge_to_state.values()):
                num_inconsistent += 1

        if infr.verbose >= 2:
            print('num_inconsistent = %r' % (num_inconsistent,))
            cc_sizes = list(map(len, cc_subgraphs))
            pcc_size_hist = ut.dict_hist(cc_sizes)
            pcc_size_stats = ut.get_stats(cc_sizes)
            if len(pcc_size_hist) < 8:
                print('PCC size hist = %s' % (ut.repr2(pcc_size_hist),))
            print('PCC size stats = %s' % (ut.repr2(pcc_size_stats),))

        if rectify:
            # Rectified relabeling, preserves grouping and labeling if possible
            new_labels = infr._rectified_relabel(cc_subgraphs)
        else:
            # Arbitrary relabeling, only preserves grouping
            new_labels = {count: infr._next_nid()
                          for count, subgraph in enumerate(cc_subgraphs)}

        for count, subgraph in enumerate(cc_subgraphs):
            new_nid = new_labels[count]
            node_to_newlabel = ut.dzip(subgraph.nodes(), [new_nid])
            infr.set_node_attrs('name_label', node_to_newlabel)

        num_names = len(cc_subgraphs)
        if infr.verbose >= 3:
            print('[infr] done relabeling')
        return num_names, num_inconsistent

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
            >>> infr.add_feedback2((2, 3), 'nomatch')
            >>> infr.add_feedback2((5, 6), 'nomatch')
            >>> infr.add_feedback2((1, 2), 'match')
            >>> infr.apply_feedback_edges()
            >>> status = infr.connected_component_status()
            >>> print(ut.repr3(status))
        """
        if infr.verbose >= 3:
            print('[infr] checking status')
        cc_subgraphs = infr.positive_connected_compoments()
        num_names_max = len(cc_subgraphs)
        ccx_to_aids = {
            ccx: list(nx.get_node_attributes(cc, 'aid').values())
            for ccx, cc in enumerate(cc_subgraphs)
        }
        aid_to_ccx = {
            aid: ccx for ccx, aids in ccx_to_aids.items() for aid in aids
        }
        all_decisions = infr.get_edge_attrs('decision')
        separated_ccxs = set([])
        inconsistent_ccxs = set([])
        for edge, state in all_decisions.items():
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

        if False:
            ccxs = list(ccx_to_aids.keys())
            num_names_min = ut.approx_min_num_components(ccxs, separated_ccxs)
        else:
            num_names_min = '?'

        status = dict(
            num_names_max=num_names_max,
            num_inconsistent=len(inconsistent_ccxs),
            num_names_min=num_names_min,
        )
        if infr.verbose >= 3:
            print('[infr] done checking status')
        return status


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInference(ut.NiceRepr,
                     graph_iden_mixins._AnnotInfrHelpers, _AnnotInfrIBEIS,
                     _AnnotInfrMatching, _AnnotInfrFeedback, _AnnotInfrUpdates,
                     _AnnotInfrPriority, _AnnotInfrRelabel, _AnnotInfrDummy,
                     _AnnotInfrGroundtruth,
                     AnnotInfr2,
                     graph_iden_depmixin._AnnotInfrDepMixin,
                     viz_graph_iden._AnnotInfrViz):
    """
    class for maintaining state of an identification

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
        >>> infr.add_feedback2((1, 4), 'nomatch')
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
        >>> infr.add_feedback2((1, 4), 'nomatch')
        >>> try:
        >>>     infr.add_feedback2((1, 10), 'nomatch')
        >>> except ValueError:
        >>>     pass
        >>> try:
        >>>     infr.add_feedback2((11, 12), 'nomatch')
        >>> except ValueError:
        >>>     pass
        >>> infr.apply_feedback_edges()
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()
        infr = <AnnotInference(nAids=6, nEdges=0)>

    """

    # Scores are ordered in priority order:
    # CUT_WEIGHT - final weight used for inference (overridden by user)
    # NORMSCORE - normalized score computed by an automatic process
    # SCORE - raw score computed by automatic process

    CUT_WEIGHT_KEY = 'cut_weight'

    _graph_cls = nx.Graph
    # _graph_cls = nx.DiGraph

    def init_test_mode(infr):
        print('[infr] init_test_mode')
        infr.test_mode = True
        infr.edge_truth = {}
        infr.metrics_list = []
        infr.test_state = {
            'n_auto': 0,
            'n_manual': 0,
        }
        infr.nid_to_gt_cc = ut.group_items(infr.aids, infr.orig_name_labels)
        infr.real_n_pcc_mst_edges = sum(
            len(cc) - 1 for cc in infr.nid_to_gt_cc.values())
        ut.cprint('real_n_pcc_mst_edges = %r' % (
            infr.real_n_pcc_mst_edges,), 'red')

    def edge_confusion(infr):
        confusion = {
            'correct': {
                'pred_pos': [],
                'pred_neg': [],
            },
            'incorrect': {
                'pred_pos': [],
                'pred_neg': [],
            },
        }
        for edge, data in infr.edges(data=True):
            # nid1 = infr.pos_graph.node_label(edge[0])
            # nid2 = infr.pos_graph.node_label(edge[1])
            true_state = infr.edge_truth[edge]
            decision = data.get('decision', 'unreviewed')
            if decision == 'unreviewed':
                pass
            elif true_state == decision:
                if true_state == 'match':
                    confusion['correct']['pred_pos'].append(edge)
            elif true_state != decision:
                if decision == 'match':
                    confusion['incorrect']['pred_pos'].append(edge)
                elif decision == 'nomatch':
                    confusion['incorrect']['pred_neg'].append(edge)

    def measure_metrics(infr):
        real_pos_edges = []
        n_error_edges = 0
        pred_n_pcc_mst_edges = 0
        n_fn = 0
        n_fp = 0

        for edge, data in infr.edges(data=True):
            true_state = infr.edge_truth[edge]
            decision = data.get('decision', 'unreviewed')
            if true_state == decision and true_state == 'match':
                real_pos_edges.append(edge)
            elif decision != 'unreviewed':
                if true_state != decision:
                    n_error_edges += 1
                    if true_state == 'match':
                        n_fn += 1
                    elif true_state == 'nomatch':
                        n_fp += 1

        import networkx as nx
        for cc in nx.connected_components(nx.Graph(real_pos_edges)):
            pred_n_pcc_mst_edges += len(cc) - 1

        pos_acc = pred_n_pcc_mst_edges / infr.real_n_pcc_mst_edges
        metrics = {
            'n_manual': infr.test_state['n_manual'],
            'n_auto': infr.test_state['n_auto'],
            'pos_acc': pos_acc,
            'n_merge_remain': infr.real_n_pcc_mst_edges - pred_n_pcc_mst_edges,
            'merge_remain': 1 - pos_acc,
            'n_errors': n_error_edges,
            'n_fn': n_fn,
            'n_fp': n_fp,
        }
        return metrics

    def __init__(infr, ibs, aids=[], nids=None, autoinit=False, verbose=False):
        infr.test_mode = False
        infr.verbose = verbose
        if infr.verbose >= 1:
            print('[infr] __init__')
        infr.ibs = ibs
        if aids == 'all':
            aids = ibs.get_valid_aids()
        infr.aids = None
        infr.method = 'graph'
        infr.aids_set = None
        infr.orig_name_labels = None
        infr.aid_to_node = None
        infr.node_to_aid = None

        # If not dirty, new feedback should dynamically maintain a consistent
        # state. If dirty it means we need to recompute connected compoments
        # before we can continue with dynamic review.
        infr.dirty = False

        infr.graph = None

        infr.review_graphs = {
            'match': None,
            'nomatch': None,
            'notcomp': None,
            'unreviewed': None,
        }
        infr.enable_inference = True
        infr.test_mode = False
        infr.edge_truth = {}

        # Criteria
        infr.refresh = None
        infr.term = None

        infr.init_bookkeeping()

        # TODO: rename to external_feedback? This should represent The feedback
        # read from a database. We do not need to do any updates to an external
        # database based on this data.
        infr.external_feedback = ut.ddict(list)

        # TODO: add all feedback to internal_feedback until we sync with the
        # database. Then merge it into external_feedback
        infr.internal_feedback = ut.ddict(list)

        infr.thresh = None
        infr.cm_list = None
        infr.vsone_matches = {}
        infr.qreq_ = None
        infr.nid_counter = None
        infr.queue = None
        infr.queue_params = {
            'pos_redundancy': 1,
            'neg_redundancy': 1,
            'complete_thresh': 1.0,
        }
        infr.add_aids(aids, nids)
        if autoinit:
            infr.initialize_graph()

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

    @classmethod
    def from_pairs(AnnotInference, aid_pairs, attrs=None, ibs=None, verbose=False):
        # infr.graph = G
        # infr.update_node_attributes(G)
        # aids = set(ut.flatten(aid_pairs))
        import networkx as nx
        G = nx.Graph()
        assert not any([a1 == a2 for a1, a2 in aid_pairs]), 'cannot have self-edges'
        G.add_edges_from(aid_pairs)
        if attrs is not None:
            for key in attrs.keys():
                nx.set_edge_attributes(G, key, ut.dzip(aid_pairs, attrs[key]))
        infr = AnnotInference.from_netx(G, ibs=ibs, verbose=verbose)
        return infr

    @classmethod
    def from_netx(AnnotInference, G, ibs=None, verbose=False):
        aids = list(G.nodes())
        nids = [-a for a in aids]
        infr = AnnotInference(ibs, aids, nids, autoinit=False, verbose=verbose)
        infr.initialize_graph(graph=G)
        infr.update_node_attributes()
        infr.refresh_bookkeeping()
        return infr

    @classmethod
    def from_qreq_(AnnotInference, qreq_, cm_list, autoinit=False):
        """
        Create a AnnotInference object using a precomputed query / results
        """
        # raise NotImplementedError('do not use')
        aids = ut.unique(ut.flatten([qreq_.qaids, qreq_.daids]))
        nids = qreq_.get_qreq_annot_nids(aids)
        ibs = qreq_.ibs
        infr = AnnotInference(ibs, aids, nids, verbose=False, autoinit=autoinit)
        infr.cm_list = cm_list
        infr.qreq_ = qreq_
        return infr

    def copy(infr):
        import copy
        # deep copy everything but ibs
        infr2 = AnnotInference(
            infr.ibs, copy.deepcopy(infr.aids),
            copy.deepcopy(infr.orig_name_labels), autoinit=False,
            verbose=infr.verbose)
        infr2.graph = infr.graph.copy()
        infr2.external_feedback = copy.deepcopy(infr.external_feedback)
        infr2.internal_feedback = copy.deepcopy(infr.internal_feedback)
        infr2.cm_list = copy.deepcopy(infr.cm_list)
        infr2.qreq_ = copy.deepcopy(infr.qreq_)
        infr2.nid_counter = infr.nid_counter
        infr2.thresh = infr.thresh
        infr2.aid_to_node = copy.deepcopy(infr.aid_to_node)
        infr2.review_graphs = {}

        infr2.recovery_ccs = copy.deepcopy(infr.recovery_ccs)
        infr2.recovery_cc = copy.deepcopy(infr.recovery_cc)
        infr2.recover_prev_neg_nids = copy.deepcopy(infr.recover_prev_neg_nids)
        infr2.pos_redun_nids = copy.deepcopy(infr.pos_redun_nids)
        infr2.neg_redun_nids = copy.deepcopy(infr.neg_redun_nids)

        for key, graph in infr.review_graphs.items():
            if graph is None:
                infr2.review_graphs[key] = None
            else:
                infr2.review_graphs[key] = graph.copy()
        return infr2

    def __nice__(infr):
        if infr.graph is None:
            return 'nAids=%r, G=None' % (len(infr.aids))
        else:
            return 'nAids=%r, nEdges=%r' % (len(infr.aids),
                                              infr.graph.number_of_edges())

    def _rectify_nids(infr, aids, nids):
        if nids is None:
            if infr.ibs is None:
                nids = [-aid for aid in aids]
            else:
                nids = infr.ibs.get_annot_nids(aids)
        elif ut.isscalar(nids):
            nids = [nids] * len(aids)
        return nids

    def remove_aids(infr, aids):
        remove_idxs = ut.take(ut.make_index_lookup(infr.aids), aids)
        ut.delete_items_by_index(infr.orig_name_labels, remove_idxs)
        ut.delete_items_by_index(infr.aids, remove_idxs)
        infr.graph.remove_nodes_from(aids)
        ut.delete_dict_keys(infr.aid_to_node, aids)
        ut.delete_dict_keys(infr.node_to_aid, aids)
        infr.aids_set = set(infr.aids)
        remove_edges = [(u, v) for u, v in infr.external_feedback.keys()
                        if u not in infr.aids_set or v not in infr.aids_set]
        ut.delete_dict_keys(infr.external_feedback, remove_edges)
        remove_edges = [(u, v) for u, v in infr.internal_feedback.keys()
                        if u not in infr.aids_set or v not in infr.aids_set]
        ut.delete_dict_keys(infr.internal_feedback, remove_edges)

        infr.pos_graph.remove_nodes_from(aids)
        infr.neg_graph.remove_nodes_from(aids)
        infr.incomp_graph.remove_nodes_from(aids)

    def add_aids(infr, aids, nids=None):
        """
        CommandLine:
            python -m ibeis.algo.hots.graph_iden add_aids --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> aids_ = [1, 2, 3, 4, 5, 6, 7, 9]
            >>> infr = AnnotInference(ibs=None, aids=aids_, autoinit=True)
            >>> aids = [2, 22, 7, 9, 8]
            >>> nids = None
            >>> infr.add_aids(aids, nids)
            >>> result = infr.aids
            >>> print(result)
            >>> assert len(infr.graph.node) == len(infr.aids)
            [1, 2, 3, 4, 5, 6, 7, 9, 22, 8]
        """
        nids = infr._rectify_nids(aids, nids)
        assert len(aids) == len(nids), 'must correspond'
        if infr.aids is None:
            nids = infr._rectify_nids(aids, nids)
            # Set object attributes
            infr.aids = aids
            infr.aids_set = set(infr.aids)
            infr.orig_name_labels = nids
        else:
            aid_to_idx = ut.make_index_lookup(infr.aids)
            orig_idxs = ut.dict_take(aid_to_idx, aids, None)
            new_flags = ut.flag_None_items(orig_idxs)
            new_aids = ut.compress(aids, new_flags)
            new_nids = ut.compress(nids, new_flags)
            # Extend object attributes
            infr.aids.extend(new_aids)
            infr.orig_name_labels.extend(new_nids)
            infr.aids_set.update(new_aids)
            infr.update_node_attributes(new_aids, new_nids)

    def update_node_attributes(infr, aids=None, nids=None):
        if aids is None:
            aids = infr.aids
            nids = infr.orig_name_labels
            infr.node_to_aid = {}
            infr.aid_to_node = {}
        assert aids is not None, 'must have aids'
        assert nids is not None, 'must have nids'
        node_to_aid = {aid: aid for aid in aids}
        aid_to_node = ut.invert_dict(node_to_aid)
        node_to_nid = {aid: nid for aid, nid in zip(aids, nids)}
        ut.assert_eq_len(node_to_nid, node_to_aid)
        infr.graph.add_nodes_from(aids)
        infr.set_node_attrs('aid', node_to_aid)
        infr.set_node_attrs('name_label', node_to_nid)
        infr.set_node_attrs('orig_name_label', node_to_nid)
        infr.node_to_aid.update(node_to_aid)
        infr.aid_to_node.update(aid_to_node)

        infr.pos_graph.add_nodes_from(aids)
        infr.neg_graph.add_nodes_from(aids)
        infr.incomp_graph.add_nodes_from(aids)

    def initialize_graph(infr, graph=None):
        if infr.verbose >= 1:
            print('[infr] initialize_graph')
        if graph is None:
            infr.graph = infr._graph_cls()
        else:
            infr.graph = graph

        infr.review_graphs['match'] = graph_iden_utils.DynConnGraph()
        infr.review_graphs['nomatch'] = infr._graph_cls()
        infr.review_graphs['notcomp'] = infr._graph_cls()
        infr.review_graphs['unreviewed'] = infr._graph_cls()

        if graph is not None:
            for u, v, d in graph.edges(data=True):
                decision = d.get('decision', 'unreviewed')
                if decision in {'match', 'nomatch', 'notcomp', 'unreviewed'}:
                    infr.review_graphs[decision].add_edge(u, v)

        infr.update_node_attributes()

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


def testdata_infr2(defaultdb='PZ_MTEST'):
    defaultdb = 'PZ_MTEST'
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaultdb)
    annots = ibs.annots()
    names = list(annots.group_items(annots.nids).values())[0:20]
    def dummy_phi(c, n):
        x = np.arange(n)
        phi = c * x / (c * x + 1)
        phi = phi / phi.sum()
        phi = np.diff(phi)
        return phi
    phis = {
        c: dummy_phi(c, 30)
        for c in range(1, 4)
    }
    aids = ut.flatten(names)
    infr = AnnotInference(ibs, aids, autoinit=True)
    infr.init_termination_criteria(phis)
    infr.init_refresh_criteria()

    # Partially review
    n1, n2, n3, n4 = names[0:4]
    for name in names[4:]:
        for a, b in ut.itertwo(name.aids):
            infr.add_feedback2((a, b), 'match')

    for name1, name2 in it.combinations(names[4:], 2):
        infr.add_feedback2((name1.aids[0], name2.aids[0]), 'nomatch')
    return infr


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
