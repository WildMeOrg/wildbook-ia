# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import networkx as nx
import pandas as pd
import utool as ut
import numpy as np
import vtool as vt
import six
from ibeis.algo.graph.nx_utils import e_
print, rrr, profile = ut.inject2(__name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class IBEISIO(object):
    """
    Direct interface into ibeis tables and delta statistics
    """

    def ibeis_delta_info(infr, edge_delta_df=None, name_delta_df=None):
        if name_delta_df is None:
            name_delta_df = infr.get_ibeis_name_delta()

        name_stats_df = infr.name_group_stats()
        name_delta_stats_df = infr.ibeis_name_group_delta_info()

        edge_delta_info = infr.ibeis_edge_delta_info(edge_delta_df)

        info = ut.odict([
            ('num_annots_with_names_changed' , len(name_delta_df)),
        ])
        info.update(edge_delta_info)
        for key, val in name_stats_df.iterrows():
            info['num_' + key] = int(val['size'])
        for key, val in name_delta_stats_df.iterrows():
            info['num_' + key] = int(val['size'])
        return info

    def ibeis_edge_delta_info(infr, edge_delta_df=None):
        if edge_delta_df is None:
            edge_delta_df = infr.match_state_delta(old='annotmatch', new='all')

        # Look at what changed
        tag_flags = edge_delta_df['old_tags'] != edge_delta_df['new_tags']
        state_flags = edge_delta_df['old_decision'] != edge_delta_df['new_decision']
        is_added = edge_delta_df['am_rowid'].isnull()
        info = ut.odict([
            ('num_edges_added' , is_added.sum()),
            ('num_edges_modified' , (~is_added).sum()),
            ('num_changed_decision_and_tags' , (
                (tag_flags & state_flags & ~is_added).sum())),
            ('num_changed_tags'              , (
                (tag_flags & ~state_flags & ~is_added).sum())),
            ('num_changed_decision'          , (
                (~tag_flags & state_flags & ~is_added).sum())),
            # 'num_non_pos_redundant': num_non_pos_redundant,
        ])
        return info

    def name_label_group_delta_info(infr):
        """
        If the name labeling delta is non-zero then you need to rectify names

        infr.relabel_using_reviews(rectify=False)
        """
        aids = infr.aids
        name_labels = list(infr.gen_node_values('name_label', aids))
        old_ccs = list(ut.group_items(aids, name_labels).values())
        new_ccs = list(infr.positive_components())
        return infr.name_group_delta_stats(old_ccs, new_ccs)

    def ibeis_name_group_delta_info(infr, verbose=None):
        aids = infr.aids
        new_names = list(infr.gen_node_values('name_label', aids))
        new_ccs = list(ut.group_items(aids, new_names).values())

        old_names = infr.ibs.get_annot_name_texts(
            aids, distinguish_unknowns=True)
        old_ccs = list(ut.group_items(aids, old_names).values())

        return infr.name_group_delta_stats(old_ccs, new_ccs, verbose)

    def name_group_stats(infr, verbose=None):
        stats = ut.odict()
        statsmap = ut.partial(lambda x: ut.stats_dict(map(len, x), size=True))
        stats['pos_redun'] = statsmap(infr.pos_redundant_pccs())
        stats['non_pos_redun'] = statsmap(infr.non_pos_redundant_pccs())
        stats['inconsistent'] = statsmap(infr.inconsistent_components())
        stats['consistent'] = statsmap(infr.consistent_components())
        df = pd.DataFrame.from_dict(stats, orient='index')
        df = df.loc[list(stats.keys())]
        if verbose:
            print('Name Group stats:')
            print(df.to_string(float_format='%.2f'))
        return df

    def name_group_delta_stats(infr, old_ccs, new_ccs, verbose=False):
        group_delta = ut.grouping_delta(old_ccs, new_ccs)

        stats = ut.odict()
        unchanged = group_delta['unchanged']
        splits = group_delta['splits']
        merges = group_delta['merges']
        hybrid = group_delta['hybrid']
        statsmap = ut.partial(lambda x: ut.stats_dict(map(len, x), size=True))
        stats['unchanged'] = statsmap(unchanged)
        stats['old_split'] = statsmap(splits['old'])
        stats['new_split'] = statsmap(ut.flatten(splits['new']))
        stats['old_merge'] = statsmap(ut.flatten(merges['old']))
        stats['new_merge'] = statsmap(merges['new'])
        stats['old_hybrid'] = statsmap(hybrid['old'])
        stats['new_hybrid'] = statsmap(hybrid['new'])
        df = pd.DataFrame.from_dict(stats, orient='index')
        df = df.loc[list(stats.keys())]
        if verbose:
            print('Name Group changes:')
            print(df.to_string(float_format='%.2f'))
        return df

    @profile
    def reset_labels_to_ibeis(infr):
        """ Sets to IBEIS de-facto labels if available """
        nids = infr.ibs.get_annot_nids(infr.aids)
        infr.set_node_attrs('name_label', ut.dzip(infr.aids, nids))

    def update_staging_to_annotmatch():
        print('Finding entries in annotmatch missing in staging')
        df = infr.match_state_delta('staging', 'annotmatch')
        # Find places that exist in annotmatch but not in staging
        flags = pd.isnull(df['old_decision'])
        missing_df = df[flags]
        alias = {
            'new_decision': 'decision',
            'new_tags': 'tags',
            'new_confidence': 'confidence',
            'new_user_id': 'user_id',
        }
        tmp = missing_df[list(alias.keys())].rename(columns=alias)
        missing_feedback = {k: [v] for k, v in tmp.to_dict('index').items()}
        feedback = missing_feedback
        infr._write_ibeis_staging_feedback(feedback)

        # am_fb = infr.read_ibeis_annotmatch_feedback()
        # staging_fb = infr.read_ibeis_staging_feedback()
        # set(am_fb.keys()) - set(staging_fb.keys())
        # set(staging_fb.keys()) == set(am_fb.keys())

    def _write_ibeis_staging_feedback(infr, feedback):
        infr.print('write_ibeis_staging_feedback %d' %
                   (len(feedback),), 1)
        aid_1_list = []
        aid_2_list = []
        decision_list = []
        timestamp_list = []
        tags_list = []
        confidence_list = []
        userid_list = []
        ibs = infr.ibs
        _iter = (
            (aid1, aid2, feedback_item)
            for (aid1, aid2), feedbacks in feedback.items()
            for feedback_item in feedbacks
        )
        for aid1, aid2, feedback_item in _iter:
            decision_key = feedback_item['decision']
            tags = feedback_item['tags']
            if tags is None:
                tags = []
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
            confidence_list.append(confidence_int)
            timestamp_list.append(timestamp)
            userid_list.append(user_id)
        review_id_list = ibs.add_review(
                aid_1_list, aid_2_list, decision_list,
                tags_list=tags_list,
                identity_list=userid_list,
                user_confidence_list=confidence_list,
                timestamp_list=timestamp_list)
        assert len(ut.find_duplicate_items(review_id_list)) == 0

    def write_ibeis_staging_feedback(infr):
        """
        Commit all reviews in internal_feedback into the staging table.  The
        edges are removed from interal_feedback and added to external feedback.
        The staging tables stores each review in the order it happened so
        history is fully reconstructable if staging is never deleted.
        """
        if len(infr.internal_feedback) == 0:
            infr.print('write_ibeis_staging_feedback 0', 1)
            return
        # Write internal feedback to disk
        infr._write_ibeis_staging_feedback(infr.internal_feedback)
        # Copy internal feedback into external
        for edge, feedbacks in infr.internal_feedback.items():
            infr.external_feedback[edge].extend(feedbacks)
        # Delete internal feedback
        infr.internal_feedback = ut.ddict(list)

    def write_ibeis_annotmatch_feedback(infr, edge_delta_df=None):
        """
        Commits the current state in external and internal into the annotmatch
        table. Annotmatch only stores the final review in the history of reviews.
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
                                edge_delta_df_['new_confidence'], None)
        new_timestamp = edge_delta_df_['new_timestamp']
        new_reviewer = edge_delta_df_['new_user_id']
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

        Returns:
            df: pd.DataFrame: data frame where each row specifies an aid
                and its `old_name` which is in the ibeis database and the
                `new_name` which is what we infer it should be renamed to.
        """
        infr.print('constructing name delta', 3)
        import pandas as pd
        graph = infr.graph
        node_to_new_label = nx.get_node_attributes(graph, 'name_label')
        aids = list(node_to_new_label.keys())
        old_names = infr.ibs.get_annot_name_texts(
            aids, distinguish_unknowns=True)
        # Indicate that unknown names should be replaced
        old_names = [None if n.startswith(infr.ibs.const.UNKNOWN) else n
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
        aid_list = list(node_to_new_label.keys())
        new_name_list = ut.take(label_to_name, node_to_new_label.values())
        old_name_list = infr.ibs.get_annot_name_texts(
            aid_list, distinguish_unknowns=True)
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

    def _rectify_annotmatch_direction(infr):
        ibs = infr
        ams = ibs._get_all_annotmatch_rowids()
        aids1 = np.array(ibs.get_annotmatch_aid1(ams))
        aids2 = np.array(ibs.get_annotmatch_aid2(ams))

        np.setdiff1d(aids1, infr.aids)
        np.setdiff1d(aids2, infr.aids)
        pass

    def read_ibeis_annotmatch_feedback(infr, only_existing_edges=False):
        r"""
        Reads feedback from annotmatch table and returns the result.
        Internal state is not changed.

        Args:
            only_existing_edges (bool): if True only reads info existing edges

        CommandLine:
            python -m ibeis.algo.graph.core read_ibeis_annotmatch_feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> feedback = infr.read_ibeis_annotmatch_feedback()
            >>> items = feedback[(2, 3)]
            >>> result = ('feedback = %s' % (ut.repr2(feedback, nl=2),))
            >>> print(result)
            >>> assert len(feedback) >= 2, 'should contain at least 2 edges'
            >>> assert len(items) >= 1, '2-3 should have one review'
            >>> assert items[0]['decision'] == POSTV, '2-3 must match'
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
            # truth[is_pb] = ibs.const.REVIEW.NEGATIVE
            truth[is_split] = ibs.const.REVIEW.NEGATIVE
            truth[is_merge] = ibs.const.REVIEW.POSITIVE

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
        timestamp = ut.dict_take_column(rectified_feedback, 'timestamp')
        user_id = ut.dict_take_column(rectified_feedback, 'user_id')
        df = pd.DataFrame([])
        df['decision'] = decision
        df['aid1'] = aids1
        df['aid2'] = aids2
        df['tags'] = tags
        df['confidence'] = confidence
        df['timestamp'] = timestamp
        df['user_id'] = user_id
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
            python -m ibeis.algo.graph.core match_state_delta

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.reset_feedback()
            >>> infr.add_feedback_from([(2, 3), POSTV) (5, 6), NEGTV)
            >>>                         (5, 4), NEGTV)]
            >>> (edge_delta_df) = infr.match_state_delta()
            >>> result = ('edge_delta_df =\n%s' % (edge_delta_df,))
            >>> print(result)
        """
        old_feedback = infr._feedback_df(old)
        new_feedback = infr._feedback_df(new)
        edge_delta_df = infr._make_state_delta(old_feedback, new_feedback)
        return edge_delta_df

    @staticmethod
    def _make_state_delta(old_feedback, new_feedback):
        r"""
        CommandLine:
            python -m ibeis.algo.graph.core _make_state_delta

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
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
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> import pandas as pd
            >>> columns = ['decision', 'aid1', 'aid2', 'am_rowid', 'tags']
            >>> old_feedback = pd.DataFrame([
            >>>     [NEGTV, 100, 101, 1000, []],
            >>>     [POSTV, 101, 102, 1001, []],
            >>>     [POSTV, 103, 104, 1002, []],
            >>>     [NEGTV, 101, 104, 1004, []],
            >>> ], columns=columns).set_index(['aid1', 'aid2'], drop=True)
            >>> new_feedback = pd.DataFrame([
            >>>     [POSTV, 101, 102, 1001, []],
            >>>     [NEGTV, 103, 104, 1002, []],
            >>>     [POSTV, 101, 104, 1004, []],
            >>>     [NEGTV, 102, 103, None, []],
            >>>     [NEGTV, 100, 103, None, []],
            >>>     [INCMP, 107, 109, None, []],
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
        data_columns += ['timestamp', 'user_id']
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
class IBEISGroundtruth(object):
    """
    Methods for generating training labels for classifiers
    """
    def ibeis_guess_if_comparable(infr, aid_pairs):
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
        # comp_by_viewpoint = (dists < tau / 8.1) | np.isnan(dists)
        comp_by_viewpoint = (dists < tau / 3) | np.isnan(dists)
        # comp_by_score = (scores > .1)
        # is_comp = comp_by_score | comp_by_viewpoint
        is_comp_guess = comp_by_viewpoint
        return is_comp_guess

    def ibeis_is_comparable(infr, aid_pairs, allow_guess=True):
        """
        Guesses by default when real comparable information is not available.
        """
        ibs = infr.ibs
        if allow_guess:
            # Guess if comparability information is unavailable
            is_comp_guess = infr.ibeis_guess_if_comparable(aid_pairs)
            is_comp = is_comp_guess.copy()
        else:
            is_comp = np.full(len(aid_pairs), np.nan)
        # But use information that we have
        am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        truths = ut.replace_nones(ibs.get_annotmatch_truth(am_rowids), np.nan)
        truths = np.asarray(truths)
        is_notcomp_have = truths == ibs.const.REVIEW.INCOMPARABLE
        is_comp_have = ((truths == ibs.const.REVIEW.POSITIVE) |
                        (truths == ibs.const.REVIEW.NEGATIVE))
        is_comp[is_notcomp_have] = False
        is_comp[is_comp_have] = True
        return is_comp

    def ibeis_is_photobomb(infr, aid_pairs):
        ibs = infr.ibs
        am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        am_tags = ibs.get_annotmatch_case_tags(am_rowids)
        is_pb = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
        return is_pb

    def ibeis_is_same(infr, aid_pairs):
        aids1, aids2 = np.asarray(aid_pairs).T
        nids1 = infr.ibs.get_annot_nids(aids1)
        nids2 = infr.ibs.get_annot_nids(aids2)
        is_same = (nids1 == nids2)
        return is_same


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.graph.mixin_ibeis
        python -m ibeis.algo.graph.mixin_ibeis --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
