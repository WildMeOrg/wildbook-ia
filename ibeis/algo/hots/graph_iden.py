# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt
import logging
import itertools as it
import six
from ibeis.algo.hots import viz_graph_iden
from ibeis.algo.hots import graph_iden_depmixin
from ibeis.algo.hots import graph_iden_mixins
from ibeis.algo.hots import graph_iden_utils
from ibeis.algo.hots.graph_iden_utils import e_, _dz
from ibeis.algo.hots.graph_iden_utils import group_name_edges
from ibeis.algo.hots.graph_iden_new import AnnotInfr2
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


DEBUG_CC = False
# DEBUG_CC = True


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrDummy(object):
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
        infr.set_edge_attrs('_dummy_edge', _dz(new_edges, [True]))

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
        infr.print('write_ibeis_staging_feedback %d' %
                   (len(infr.internal_feedback),), 1)
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
            conf_key = feedback_item.get('confidence', None)
            user_id = feedback_item.get('user_id', None)
            decision_int = ibs.const.REVIEW.CODE_TO_INT[decision_key]
            confidence_int = ibs.const.CONFIDENCE.CODE_TO_INT[conf_key]
            # confidence_int = infr.ibs.const.CONFIDENCE.CODE_TO_INT.get(
            #         confidence_key, None)
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
        infr.print('write_ibeis_annotmatch_feedback %r' % (
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
        infr.print('write_ibeis_name_assignment %d' % len(name_delta_df))
        aid_list = name_delta_df.index.values
        new_name_list = name_delta_df['new_name'].values
        infr.ibs.set_annot_names(aid_list, new_name_list)

    def get_ibeis_name_delta(infr, ignore_unknown=True):
        """
        Rectifies internal name_labels with the names stored in the name table.
        """
        infr.print('constructing name delta', 3)
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
        infr.print('had %d unknown labels' % (len(unknown_labels)), 3)
        infr.print('ignore_unknown = %r' % (ignore_unknown,), 3)
        infr.print('need to make %d new names' % (len(needs_assign)), 3)
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
        infr.print('finished making name delta', 3)
        return name_delta_df

    def read_ibeis_staging_feedback(infr):
        """
        Reads feedback from review staging table.
        """
        infr.print('read_ibeis_staging_feedback', 1)
        ibs = infr.ibs
        # annots = ibs.annots(infr.aids)
        review_ids = ibs.get_review_rowids_between(infr.aids)
        review_ids = sorted(review_ids)
        # aid_pairs = ibs.get_review_aid_tuple(review_ids)
        # flat_review_ids, cumsum = ut.invertible_flatten2(review_ids)

        infr.print('read %d staged reviews' % (len(review_ids)), 2)

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
        r"""
        Reads feedback from annotmatch table and returns the result.
        Internal state is not changed.

        Args:
            only_existing_edges (bool): if True only reads info existing edges

        CommandLine:
            python -m ibeis.algo.hots.graph_iden read_ibeis_annotmatch_feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> feedback = infr.read_ibeis_annotmatch_feedback()
            >>> items = feedback[(2, 3)]
            >>> result = ('feedback = %s' % (ut.repr2(feedback, nl=2),))
            >>> print(result)
            >>> assert len(feedback) >= 2, 'should contain at least 2 edges'
            >>> assert len(items) >= 1, '2-3 should have one review'
            >>> assert items[0]['decision'] == 'match', '2-3 must match'
        """
        infr.print('read_ibeis_annotmatch_feedback', 1)
        ibs = infr.ibs
        if only_existing_edges:
            aid_pairs = infr.graph.edges()
            am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        else:
            annots = ibs.annots(infr.aids)
            am_rowids, aid_pairs = annots.get_am_rowids_and_pairs()

        aids1 = ut.take_column(aid_pairs, 0)
        aids2 = ut.take_column(aid_pairs, 1)

        infr.print('read %d annotmatch rowids' % (len(am_rowids)), 2)
        infr.print('* checking truth', 2)

        # Use explicit truth state to mark truth
        truth = np.array(ibs.get_annotmatch_truth(am_rowids))
        infr.print('* checking tags', 2)
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
            infr.print('* checking split and joins', 2)
            # Use tags to infer truth
            props = ['SplitCase', 'JoinCase']
            flags_list = ibs.get_annotmatch_prop(props, am_rowids)
            is_split, is_merge = flags_list
            is_split = np.array(is_split).astype(np.bool)
            is_merge = np.array(is_merge).astype(np.bool)
            # truth[is_pb] = ibs.const.REVIEW.NON_MATCH
            truth[is_split] = ibs.const.REVIEW.NON_MATCH
            truth[is_merge] = ibs.const.REVIEW.MATCH

        infr.print('* making feedback dict', 2)

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
        infr.print('read %d annotmatch entries' % (len(feedback)), 1)
        return feedback

    def reset_staging_with_ensure(infr):
        """
        Make sure staging has all info that annotmatch has.
        """
        staging_feedback = infr.read_ibeis_staging_feedback()
        if len(staging_feedback) == 0:
            infr.internal_feedback = infr.read_ibeis_annotmatch_feedback()
            infr.write_ibeis_staging_feedback()
        else:
            infr.external_feedback = staging_feedback
        infr.internal_feedback = ut.ddict(list)
        # edge_delta_df = infr.match_state_delta(old='staging',
        # new='annotmatch')

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

    def _feedback_df(infr, key):
        if key == 'annotmatch':
            feedback = infr.read_ibeis_annotmatch_feedback()
        elif key == 'staging':
            feedback = infr.read_ibeis_staging_feedback()
        elif key == 'all':
            feedback = infr.all_feedback()
        elif key == 'internal':
            feedback = infr.internal_feedback
            df = infr._pandas_feedback_format()
        elif key == 'external':
            feedback = infr.external_feedback
        else:
            raise KeyError('key=%r' % (key,))
        df = infr._pandas_feedback_format(feedback)
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
            >>> infr.add_feedback((2, 3), 'match')
            >>> infr.add_feedback((5, 6), 'nomatch')
            >>> infr.add_feedback((5, 4), 'nomatch')
            >>> (edge_delta_df) = infr.match_state_delta()
            >>> result = ('edge_delta_df =\n%s' % (edge_delta_df,))
            >>> print(result)
        """
        old_feedback = infr._feedback_df(old)
        new_feedback = infr._feedback_df(new)
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

    def _check_edge(infr, edge):
        aid1, aid2 = edge
        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))

    @profile
    def add_feedback(infr, edge, decision, tags=None, user_id=None,
                     confidence=None, timestamp=None, verbose=None):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback((5, 6), 'match')
            >>> infr.add_feedback((5, 6), 'nomatch', ['Photobomb'])
            >>> infr.add_feedback((1, 2), 'notcomp')
            >>> print(ut.repr2(infr.internal_feedback, nl=2))
            >>> assert len(infr.external_feedback) == 0
            >>> assert len(infr.internal_feedback) == 2
            >>> assert len(infr.internal_feedback[(5, 6)]) == 2
            >>> assert len(infr.internal_feedback[(1, 2)]) == 1
        """
        prev_verbose = infr.verbose
        if verbose is not None:
            infr.verbose = verbose
        edge = aid1, aid2 = e_(*edge)

        if not infr.has_edge(edge):
            infr._check_edge(edge)
            infr.graph.add_edge(aid1, aid2)

        infr.print(
            'add_feedback {}, {}, decision={}, tags={}, user_id={},'
            ' confidence={}'.format(
                aid1, aid2, decision, tags, user_id, confidence), 1,
            color='underline')

        if decision == 'unreviewed':
            raise NotImplementedError('not done yet')
            feedback_item = None
            if edge in infr.external_feedback:
                raise ValueError('External edge reviews cannot be undone')
            if edge in infr.internal_feedback:
                del infr.internal_feedback[edge]
            for G in infr.review_graphs.values():
                if G.has_edge(*edge):
                    G.remove_edge(*edge)

        # Keep track of sequential reviews and set properties on global graph
        num_reviews = infr.get_edge_attr(edge, 'num_reviews', default=0)
        if timestamp is None:
            timestamp = ut.get_timestamp('int', isutc=True)
        feedback_item = {
            'decision': decision,
            'tags': tags,
            'timestamp': timestamp,
            'confidence': confidence,
            'user_id': user_id,
            'num_reviews': num_reviews + 1,
        }
        infr.internal_feedback[edge].append(feedback_item)
        infr.set_edge_attr(edge, feedback_item)
        if infr.refresh:
            infr.refresh.add(decision, user_id)

        if infr.enable_inference:
            # Update priority queue based on the new edge
            if infr.is_recovering():
                infr.inconsistent_inference(edge, decision)
            else:
                infr.consistent_inference(edge, decision)
        else:
            infr.dirty = True
            infr._add_review_edge(edge, decision)

        if infr.test_mode:
            if user_id.startswith('auto'):
                infr.test_state['n_auto'] += 1
            elif user_id == 'oracle':
                infr.test_state['n_manual'] += 1
            else:
                raise AssertionError('unknown user_id=%r' % (user_id,))
            infr.metrics_list.append(infr.measure_metrics())
        infr.verbose = prev_verbose

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
            >>> #infr.add_feedback((1, 2), 'unknown', tags=[])
            >>> infr.add_feedback((1, 2), 'notcomp', tags=[])
            >>> infr.apply_feedback_edges()
            >>> print('edges = ' + ut.repr4(infr.graph.edge))
            >>> result = str(infr)
            >>> print(result)
            <AnnotInference(nAids=6, nEdges=3)>
        """
        infr.print('apply_feedback_edges', 1)
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

        # Put pair orders in context of the graph
        edges = feedback_edges
        infr.print('_set_feedback_edges(nEdges=%d)' % (len(edges),), 3)
        # Ensure edges exist
        for edge in edges:
            if not infr.graph.has_edge(*edge):
                infr.graph.add_edge(*edge)

        for decision, es in ut.group_items(edges, decision_list).items():
            infr._add_review_edges_from(es, decision)

        # use UTC timestamps
        timestamp = ut.get_timestamp('int', isutc=True)
        infr.set_edge_attrs('decision', _dz(edges, decision_list))
        infr.set_edge_attrs('reviewed_tags', _dz(edges, tags_list))
        infr.set_edge_attrs('confidence', _dz(edges, confidence_list))
        infr.set_edge_attrs('user_id', _dz(edges, userid_list))
        infr.set_edge_attrs('num_reviews', _dz(edges, num_review_list))
        infr.set_edge_attrs('review_timestamp', _dz(edges, [timestamp]))
        infr.apply_nondynamic_update()

    def _del_feedback_edges(infr, edges=None):
        """ Delete all edges properties related to feedback """
        if edges is None:
            edges = list(infr.graph.edges())
        infr.print('_del_feedback_edges len(edges) = %r' % (len(edges)), 2)
        keys = ['decision', 'reviewed_tags', 'num_reviews',
                'reviewed_weight']
        ut.nx_delete_edge_attr(infr.graph, keys, edges)

    def remove_feedback(infr, apply=False):
        """ Deletes all feedback """
        infr.print('remove_feedback', 1)
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

    def reset_feedback(infr, mode='annotmatch', apply=False):
        """ Resets feedback edges to state of the SQL annotmatch table """
        infr.print('reset_feedback mode=%r' % (mode,), 1)
        if mode == 'annotmatch':
            infr.external_feedback = infr.read_ibeis_annotmatch_feedback()
        elif mode == 'staging':
            infr.external_feedback = infr.read_ibeis_staging_feedback()
        else:
            raise ValueError('no mode=%r' % (mode,))
        infr.internal_feedback = ut.ddict(list)
        if apply:
            infr.apply_feedback_edges()

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
        infr.print('reset_name_labels', 1)
        orig_names = infr.get_node_attrs('orig_name_label')
        infr.set_node_attrs('name_label', orig_names)

    def remove_name_labels(infr):
        """ Sets all annotation node name labels to be unknown """
        infr.print('remove_name_labels()', 1)
        # make distinct names for all nodes
        distinct_names = {
            node: -aid for node, aid in infr.get_node_attrs('aid').items()
        }
        infr.set_node_attrs('name_label', distinct_names)


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
        infr.print('_init_priority_queue', 1)
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

    def apply_nondynamic_update(infr, graph=None):
        r"""
        Recomputes all dynamic bookkeeping for a graph in any state.
        This ensures that subsequent dyanmic inference can be applied.
        """
        categories = infr.categorize_edges(graph)

        new_error_edges = set([])
        for nid, intern_edges in categories['inconsistent_internal'].items():
            recovery_cc = infr.pos_graph.component_nodes(nid)
            pos_subgraph = infr.pos_graph.subgraph(recovery_cc).copy()
            neg_subgraph = infr.neg_graph.subgraph(recovery_cc)
            inconsistent_edges = [
                e_(u, v) for u, v in neg_subgraph.edges()
                if infr.pos_graph.node_label(u) == infr.pos_graph.node_label(v)
            ]
            recover_hypothesis = dict(infr.hypothesis_errors(
                pos_subgraph, inconsistent_edges))
            new_error_edges.update(set(recover_hypothesis.keys()))

        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories['positive'].values()), ['same'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories['negative'].values()), ['diff'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories['notcomp'].values()), ['notcomp'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories['unreviewed'].values()), [None])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories['inconsistent_internal'].values()),
                    ['inconsistent_internal'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories['inconsistent_external'].values()),
                    ['inconsistent_external'])
        )
        # assert (
        #     set(ut.flatten(ut.flatten(categories['inconsistent_internal'].values()))) ==
        #     set(ut.flatten(infr.recovery_ccs))
        # )

        # Delete old hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ut.dzip(infr.error_edges, [None])
        )
        # Set new hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ut.dzip(new_error_edges, [True])
        )
        infr.print('new_error_edges = %r' % (new_error_edges,))
        infr.error_edges = new_error_edges

        # TODO: should this also update redundancy?
        # infr.refresh_bookkeeping()
        # def refresh_bookkeeping(infr):
        # TODO: need to ensure bookkeeping is taken care of
        infr.recovery_ccs = list(infr.inconsistent_components())
        if len(infr.recovery_ccs) > 0:
            infr.recovery_cc = infr.recovery_ccs[0]
            infr.recover_prev_neg_nids = list(
                infr.find_neg_outgoing_freq(infr.recovery_cc).keys()
            )
        else:
            infr.recovery_cc = None
            infr.recover_prev_neg_nids = None
        infr.pos_redun_nids = set(infr.find_pos_redun_nids())
        infr.neg_redun_nids = nx.Graph(list(infr.find_neg_redun_nids()))

    @profile
    def categorize_edges(infr, graph=None):
        r"""
        Non-dynamically computes the status of each edge in the graph.
        This is can be used to verify the dynamic computations and update when
        the dynamic state is lost.

        CommandLine:
            python -m ibeis.algo.hots.graph_iden categorize_edges --profile

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> from ibeis.algo.hots import demo_graph_iden
            >>> num_pccs = 250 if ut.get_argflag('--profile') else 100
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo_graph_iden.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> cat1 = infr.categorize_edges_old()['ne_categories']
            >>> cat2 = infr.categorize_edges()
            >>> assert cat1 == cat2
        """
        POSTV = 'match'
        NEGTV = 'nomatch'
        INCMP = 'notcomp'
        UNREV = 'unreviewed'
        states = (POSTV, NEGTV, INCMP, UNREV)

        rev_graph = {key: infr.review_graphs[key] for key in states}
        if graph is None or graph is infr.graph:
            graph = infr.graph
            nodes = None
        else:
            # Need to extract relevant subgraphs
            nodes = list(graph.nodes())
            for key in states:
                rev_graph[key] = rev_graph[key].subgraph(nodes)

        # TODO: Rebalance union find to ensure parents is a single lookup
        # infr.pos_graph._union_find.rebalance(nodes)
        # node_to_label = infr.pos_graph._union_find.parents
        node_to_label = infr.pos_graph._union_find

        # Get reviewed edges using fast lookup structures
        ne_to_edges = {
            key: group_name_edges(rev_graph[key], node_to_label)
            for key in states
        }

        # Use reviewed edges to determine status of PCCs (repr by name ids)
        # The next steps will rectify duplicates in these sets
        name_edges = {key: set(ne_to_edges[key].keys()) for key in states}

        # Positive and negative decisions override incomparable and unreviewed
        for key in (INCMP, UNREV):
            name_edges[key].difference_update(name_edges[POSTV])
            name_edges[key].difference_update(name_edges[NEGTV])

        # Negative edges within a PCC signals that an inconsistency exists
        # Remove inconsistencies from the name edges
        incon_internal_ne = name_edges[NEGTV].intersection(name_edges[POSTV])
        name_edges[POSTV].difference_update(incon_internal_ne)
        name_edges[NEGTV].difference_update(incon_internal_ne)

        if __debug__:
            assert all(n1 == n2 for n1, n2 in name_edges[POSTV]), (
                'All positive edges should be internal to a PCC')
            assert len(name_edges[INCMP].intersection(incon_internal_ne)) == 0
            assert len(name_edges[UNREV].intersection(incon_internal_ne)) == 0
            assert all(n1 == n2 for n1, n2 in incon_internal_ne), (
                'incon_internal edges should be internal to a PCC')

        # External inconsistentices are edges leaving inconsistent components
        incon_internal_nids = {n1 for n1, n2 in incon_internal_ne}
        incon_external_ne = set([])
        # Find all edges leaving an inconsistent PCC
        for key in (NEGTV, INCMP, UNREV):
            incon_external_ne.update({
                (nid1, nid2) for nid1, nid2 in name_edges[key]
                if nid1 in incon_internal_nids or nid2 in incon_internal_nids
            })
        for key in (NEGTV, INCMP, UNREV):
            name_edges[key].difference_update(incon_external_ne)

        # Inference between names is now complete.
        # Now we expand this inference and project the labels onto the
        # annotation edges corresponding to each name edge.

        # Version of union that accepts generators
        union = lambda gen: set.union(*gen)  # NOQA

        # Find edges within consistent PCCs
        positive = {
            nid1: union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (POSTV, INCMP, UNREV))
            for nid1, nid2 in name_edges[POSTV]
        }
        # Find edges between 1-negative-redundant consistent PCCs
        negative = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV, INCMP, UNREV))
            for nid1, nid2 in name_edges[NEGTV]
        }
        # Find edges internal to inconsistent PCCs
        incon_internal = {
            nid: union(
                ne_to_edges[key][(nid, nid)]
                for key in (POSTV, NEGTV, INCMP, UNREV))
            for nid in incon_internal_nids
        }
        # Find edges leaving inconsistent PCCs
        incon_external = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV, INCMP, UNREV))
            for nid1, nid2 in incon_external_ne
        }
        # Incomparable names cannot make inference about any other edges
        notcomparable = {
            (nid1, nid2): ne_to_edges[INCMP][(nid1, nid2)]
            for (nid1, nid2) in name_edges[INCMP]
        }
        # Unreviewed edges are between any name not known to be negative
        # (this ignores specific incomparable edges)
        unreviewed = {
            (nid1, nid2): ne_to_edges[UNREV][(nid1, nid2)]
            for (nid1, nid2) in name_edges[UNREV]
        }

        ne_categories = {
            'positive': positive,
            'negative': negative,
            'unreviewed': unreviewed,
            'notcomp': notcomparable,
            'inconsistent_internal': incon_internal,
            'inconsistent_external': incon_external,
        }
        return ne_categories


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
        r"""
        Determines if a PCC contains inconsistencies

        Args:
            cc (set): nodes in a PCC

        Returns:
            : bool: returns True unless cc contains any negative edges
        """
        return (len(cc) <= 2 or
                infr.neg_graph.subgraph(cc).number_of_edges() == 0)

    def inconsistent_components(infr, graph=None):
        """
        Generates inconsistent PCCs.
        These PCCs contain internal negative edges indicating an error exists.

        Note:
            If inference is enabled these will be accessible in
            constant time using `infr.recover_ccs`
        """
        # Find PCCs with negative edges
        for cc in infr.positive_components(graph):
            if not infr.is_consistent(cc):
                yield cc

    def consistent_components(infr, graph=None):
        r"""
        Generates consistent PCCs.
        These PCCs contain no internal negative edges.

        Yields:
            cc: set: nodes within the PCC
        """
        # Find PCCs without any negative edges
        for cc in infr.positive_components(graph):
            if infr.is_consistent(cc):
                yield cc

    @profile
    def positive_components(infr, graph=None):
        r"""
        Generates the positive connected compoments (PCCs) in the graph
        These will contain both consistent and inconsinstent PCCs.

        Yields:
            cc: set: nodes within the PCC
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
        infr.print('rectifying name lists', 3)
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
        """
        Reuses as many names as possible
        """
        # Determine which names can be reused
        from ibeis.scripts import name_recitifer
        infr.print('grouping names for rectification', 3)
        grouped_oldnames_ = [
            list(nx.get_node_attributes(subgraph, 'name_label').values())
            for count, subgraph in enumerate(cc_subgraphs)
        ]
        # Make sure negatives dont get priority
        grouped_oldnames = [
            [n for n in group if len(group) == 1 or n > 0]
            for group in grouped_oldnames_
        ]
        infr.print('begin rectification of %d grouped old names' % (
            len(grouped_oldnames)), 2)
        new_labels = name_recitifer.find_consistent_labeling(
            grouped_oldnames, verbose=infr.verbose >= 3)
        infr.print('done rectifying new names', 2)
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
        infr.print('relabel_using_reviews', 2)
        if graph is None:
            graph = infr.graph

        # Get subgraphs and check consistency
        cc_subgraphs = []
        num_inconsistent = 0
        for cc in infr.positive_components(graph=graph):
            cc_subgraphs.append(infr.graph.subgraph(cc))
            if not infr.is_consistent(cc):
                num_inconsistent += 1

        infr.print('num_inconsistent = %r' % (num_inconsistent,), 2)
        if infr.verbose >= 2:
            cc_sizes = list(map(len, cc_subgraphs))
            pcc_size_hist = ut.dict_hist(cc_sizes)
            pcc_size_stats = ut.get_stats(cc_sizes)
            if len(pcc_size_hist) < 8:
                infr.print('PCC size hist = %s' % (ut.repr2(pcc_size_hist),))
            infr.print('PCC size stats = %s' % (ut.repr2(pcc_size_stats),))

        if rectify:
            # Rectified relabeling, preserves grouping and labeling if possible
            new_labels = infr._rectified_relabel(cc_subgraphs)
        else:
            # Arbitrary relabeling, only preserves grouping
            if graph is infr.graph:
                # Use union find labels
                new_labels = {
                    count:
                    infr.pos_graph.node_label(next(iter(subgraph.nodes())))
                    for count, subgraph in enumerate(cc_subgraphs)
                }
            else:
                new_labels = {count: infr._next_nid()
                              for count, subgraph in enumerate(cc_subgraphs)}

        for count, subgraph in enumerate(cc_subgraphs):
            new_nid = new_labels[count]
            node_to_newlabel = ut.dzip(subgraph.nodes(), [new_nid])
            infr.set_node_attrs('name_label', node_to_newlabel)

        num_names = len(cc_subgraphs)
        infr.print('done relabeling', 3)
        return num_names, num_inconsistent

    def connected_component_status(infr):
        r"""
        Returns:
            dict: num_inconsistent, num_names_max

        CommandLine:
            python -m ibeis.algo.hots.graph_iden connected_component_status

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback((2, 3), 'nomatch')
            >>> infr.add_feedback((5, 6), 'nomatch')
            >>> infr.add_feedback((1, 2), 'match')
            >>> status = infr.connected_component_status()
            >>> print(ut.repr3(status))
        """
        infr.print('checking status', 3)

        num_inconsistent = len(infr.recovery_ccs)
        num_names_max = infr.pos_graph.number_of_components()

        status = dict(
            num_names_max=num_names_max,
            num_inconsistent=num_inconsistent,
        )
        infr.print('done checking status', 3)
        return status


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInference(ut.NiceRepr,
                     _AnnotInfrIBEIS,
                     _AnnotInfrFeedback, _AnnotInfrUpdates,
                     _AnnotInfrPriority, _AnnotInfrRelabel, _AnnotInfrDummy,
                     graph_iden_mixins._AnnotInfrHelpers,
                     graph_iden_mixins._AnnotInfrGroundtruth,
                     graph_iden_depmixin._AnnotInfrDepMixin,
                     AnnotInfr2,
                     viz_graph_iden._AnnotInfrViz):
    """
    class for maintaining state of an identification

    Terminology and Concepts:

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
        >>> infr.add_feedback((1, 4), 'nomatch')
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
        >>> infr.add_feedback((1, 4), 'nomatch')
        >>> try:
        >>>     infr.add_feedback((1, 10), 'nomatch')
        >>> except ValueError:
        >>>     pass
        >>> try:
        >>>     infr.add_feedback((11, 12), 'nomatch')
        >>> except ValueError:
        >>>     pass
        >>> infr.apply_feedback_edges()
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()
        infr = <AnnotInference(nAids=6, nEdges=0)>

    """

    _graph_cls = nx.Graph
    # _graph_cls = nx.DiGraph

    def init_test_mode(infr):
        infr.print('init_test_mode')
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

    def init_logging(infr):
        import collections
        infr.logs = collections.deque(maxlen=10000)

    def log_message(infr, msg, level=logging.NOTSET, color=None):
        if color is None:
            color = 'blue'
        if True:
            infr.logs.append((msg, color))
        if infr.verbose >= level:
            ut.cprint('[infr] ' + msg, color)

    print = log_message

    def dump_logs(infr):
        print('--- <LOG DUMP> ---')
        for msg, color in infr.logs:
            ut.cprint('[infr] ' + msg, color)
        print('--- <\LOG DUMP> ---')

    def __init__(infr, ibs, aids=[], nids=None, autoinit=False, verbose=False):
        # infr.verbose = verbose
        infr.verbose = 100
        infr.init_logging()
        infr.print('__init__', level=1)
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

        # Bookkeeping
        infr.error_edges = set([])
        infr.recovery_ccs = []
        # TODO: keep one main recovery cc but once it is done pop the next one
        # from recovery_ccs until none are left
        infr.recovery_cc = None
        infr.recover_prev_neg_nids = None
        # Set of PCCs that are positive redundant
        infr.pos_redun_nids = set([])
        # Represents the metagraph of negative edges between PCCs
        infr.neg_redun_nids = nx.Graph()

        # This should represent The feedback read from a database. We do not
        # need to do any updates to an external database based on this data.
        infr.external_feedback = ut.ddict(list)

        # Feedback that has not been synced with the external database.
        # Once we sync, this is merged into external feedback.
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

        infr2.recovery_ccs = copy.deepcopy(infr.recovery_ccs)
        infr2.recovery_cc = copy.deepcopy(infr.recovery_cc)
        infr2.recover_prev_neg_nids = copy.deepcopy(infr.recover_prev_neg_nids)
        infr2.pos_redun_nids = copy.deepcopy(infr.pos_redun_nids)
        infr2.neg_redun_nids = copy.deepcopy(infr.neg_redun_nids)

        infr2.review_graphs = copy.deepcopy(infr.review_graphs)
        infr2.error_edges = copy.deepcopy(infr.error_edges)
        return infr2

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
    def from_netx(AnnotInference, G, ibs=None, verbose=False, infer=True):
        aids = list(G.nodes())
        nids = [-a for a in aids]
        infr = AnnotInference(ibs, aids, nids, autoinit=False, verbose=verbose)
        infr.initialize_graph(graph=G)
        infr.update_node_attributes()
        if infer:
            infr.apply_nondynamic_update()
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

    def __nice__(infr):
        if infr.graph is None:
            return 'nAids=%r, G=None' % (len(infr.aids))
        else:
            return 'nAids={}, nEdges={}, nCCs={}'.format(
                len(infr.aids),
                infr.graph.number_of_edges(),
                infr.pos_graph.number_of_components()
            )

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
        infr.print('initialize_graph', 1)
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
            infr.add_feedback((a, b), 'match')

    for name1, name2 in it.combinations(names[4:], 2):
        infr.add_feedback((name1.aids[0], name2.aids[0]), 'nomatch')
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
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7 --graph --match=1,4 --nomatch=3,1,5,7
        python -m ibeis.algo.hots.graph_iden
        python -m ibeis.algo.hots.graph_iden --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
