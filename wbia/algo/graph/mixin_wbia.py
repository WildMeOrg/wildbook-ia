# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import networkx as nx
import pandas as pd
import utool as ut
import numpy as np
import vtool as vt  # NOQA
import six
from wbia.algo.graph import nx_utils as nxu
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

print, rrr, profile = ut.inject2(__name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class IBEISIO(object):
    """
    Direct interface into wbia tables and delta statistics
    """

    def add_annots(infr, aid_list):
        pass

    def wbia_delta_info(infr, edge_delta_df=None, name_delta_df=None):
        if name_delta_df is None:
            name_delta_df = infr.get_wbia_name_delta()

        name_stats_df = infr.name_group_stats()
        name_delta_stats_df = infr.wbia_name_group_delta_info()

        edge_delta_info = infr.wbia_edge_delta_info(edge_delta_df)

        info = ut.odict([('num_annots_with_names_changed', len(name_delta_df))])
        info.update(edge_delta_info)
        for key, val in name_stats_df.iterrows():
            info['num_' + key] = int(val['size'])
        for key, val in name_delta_stats_df.iterrows():
            info['num_' + key] = int(val['size'])
        return info

    def wbia_edge_delta_info(infr, edge_delta_df=None):
        if edge_delta_df is None:
            edge_delta_df = infr.match_state_delta(old='annotmatch', new='all')

        # Look at what changed
        tag_flags = edge_delta_df['old_tags'] != edge_delta_df['new_tags']
        state_flags = (
            edge_delta_df['old_evidence_decision']
            != edge_delta_df['new_evidence_decision']
        )
        is_added_to_am = edge_delta_df['am_rowid'].isnull()
        is_new = edge_delta_df['is_new']
        info = ut.odict(
            [
                # Technically num_edges_added only cares if the edge exists in the
                # annotmatch table.
                ('num_edges_added_to_am', is_added_to_am.sum()),
                ('num_edges_added', is_new.sum()),
                ('num_edges_modified', (~is_new).sum()),
                (
                    'num_changed_decision_and_tags',
                    ((tag_flags & state_flags & ~is_new).sum()),
                ),
                ('num_changed_tags', ((tag_flags & ~state_flags & ~is_new).sum())),
                ('num_changed_decision', ((~tag_flags & state_flags & ~is_new).sum())),
                # 'num_non_pos_redundant': num_non_pos_redundant,
            ]
        )
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
        df = infr.name_group_delta_stats(old_ccs, new_ccs)
        return df

    def wbia_name_group_delta_info(infr, verbose=None):
        """
        infr.relabel_using_reviews(rectify=False)
        """
        aids = infr.aids
        new_names = list(infr.gen_node_values('name_label', aids))
        new_ccs = list(ut.group_items(aids, new_names).values())

        old_names = infr.ibs.get_annot_name_texts(aids, distinguish_unknowns=True)
        old_ccs = list(ut.group_items(aids, old_names).values())

        df = infr.name_group_delta_stats(old_ccs, new_ccs, verbose)
        return df

    def name_group_stats(infr, verbose=None):
        stats = ut.odict()
        statsmap = ut.partial(lambda x: ut.stats_dict(map(len, x), size=True))
        stats['pos_redun'] = statsmap(infr.find_pos_redundant_pccs())
        stats['non_pos_redun'] = statsmap(infr.find_non_pos_redundant_pccs())
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

    def find_unjustified_splits(infr):
        """
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.mixin_helpers import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='GZ_Master1')
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> infr = wbia.AnnotInference(ibs, 'all', autoinit=True)
        >>> infr.reset_feedback('staging', apply=True)
        >>> infr.relabel_using_reviews(rectify=False)
        >>> unjustified = infr.find_unjustified_splits()
        >>> review_edges = []
        >>> for cc1, cc2 in unjustified:
        >>>     u = next(iter(cc1))
        >>>     v = next(iter(cc2))
        >>>     review_edges.append(nxu.e_(u, v))
        >>> infr.verbose = 100
        >>> infr.prioritize(
        >>>     edges=review_edges, scores=[1] * len(review_edges),
        >>>     reset=True,
        >>> )
        >>> infr.qt_review_loop()
        """
        ibs = infr.ibs
        annots = ibs.annots(infr.aids)
        ibs_ccs = [a.aids for a in annots.group(annots.nids)[1]]
        infr_ccs = list(infr.positive_components())
        delta = ut.grouping_delta(ibs_ccs, infr_ccs)

        hyrbid_splits = [ccs for ccs in delta['hybrid']['splits'] if len(ccs) > 1]
        pure_splits = delta['splits']['new']

        new_splits = hyrbid_splits + pure_splits
        unjustified = []
        for ccs in new_splits:
            for cc1, cc2 in ut.combinations(ccs, 2):
                edges = list(nxu.edges_between(infr.graph, cc1, cc2))
                df = infr.get_edge_dataframe(edges)
                if len(df) == 0 or not (df['evidence_decision'] == NEGTV).any(axis=0):
                    if len(df) > 0:
                        n_incmp = (df['evidence_decision'] == INCMP).sum()
                        if n_incmp > 0:
                            continue
                    unjustified.append((cc1, cc2))
                    # print('--------------------------------')
                    # print('No decision to justify splitting')
                    # print('cc1 = %r' % (cc1,))
                    # print('cc2 = %r' % (cc2,))
                    # if len(df):
                    #     df.index.names = ('aid1', 'aid2')
                    #     nids = np.array([
                    #         infr.pos_graph.node_labels(u, v)
                    #         for u, v in list(df.index)])
                    #     df = df.assign(nid1=nids.T[0], nid2=nids.T[1])
                    #     print(df)
        return unjustified

    @profile
    def reset_labels_to_wbia(infr):
        """ Sets to IBEIS de-facto labels if available """
        nids = infr.ibs.get_annot_nids(infr.aids)
        infr.set_node_attrs('name_label', ut.dzip(infr.aids, nids))

    def _prepare_write_wbia_staging_feedback(infr, feedback):
        r"""
        builds data that will be sent to ibs.add_review

        Returns:
            tuple: (aid_1_list, aid_2_list, add_review_kw)

        CommandLine:
            python -m wbia.algo.graph.mixin_wbia _prepare_write_wbia_staging_feedback

        Doctest:
            >>> from wbia.algo.graph.mixin_wbia import *  # NOQA
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', aids=list(range(1, 10)),
            >>>                             autoinit='annotmatch', verbose=4)
            >>> infr.add_feedback((6, 7), NEGTV, user_id='user:foobar')
            >>> infr.add_feedback((5, 8), NEGTV, tags=['photobomb'])
            >>> infr.add_feedback((4, 5), POSTV, confidence='absolutely_sure')
            >>> feedback = infr.internal_feedback
            >>> tup = infr._prepare_write_wbia_staging_feedback(feedback)
            >>> (aid_1_list, aid_2_list, add_review_kw) = tup
            >>> expected = set(ut.get_func_argspec(infr.ibs.add_review).args)
            >>> got = set(add_review_kw.keys())
            >>> overlap = ut.set_overlap_items(expected, got)
            >>> assert got.issubset(expected), ut.repr4(overlap, nl=2)
        """
        import uuid

        # Map what add_review expects to the keys used by feedback items
        add_review_alias = {
            'evidence_decision_list': 'evidence_decision',
            'meta_decision_list': 'meta_decision',
            'review_uuid_list': 'uuid',
            'identity_list': 'user_id',
            'user_confidence_list': 'confidence',
            'tags_list': 'tags',
            'review_client_start_time_posix': 'timestamp_c1',
            'review_client_end_time_posix': 'timestamp_c2',
            'review_server_start_time_posix': 'timestamp_s1',
            'review_server_end_time_posix': 'timestamp',
        }

        # Initialize kwargs we will pass to add_review
        aid_1_list = []
        aid_2_list = []

        add_review_kw = {}
        for k in add_review_alias:
            add_review_kw[k] = []

        # Translate data from feedback items into add_review format
        ibs = infr.ibs
        _iter = (
            (aid1, aid2, feedback_item)
            for (aid1, aid2), feedbacks in feedback.items()
            for feedback_item in feedbacks
        )
        for aid1, aid2, feedback_item in _iter:
            aid_1_list.append(aid1)
            aid_2_list.append(aid2)
            for review_key, fbkey in add_review_alias.items():
                value = feedback_item.get(fbkey, None)
                # Do mapping for particular keys
                if fbkey == 'tags' and value is None:
                    value = []
                elif fbkey == 'uuid' and value is None:
                    value = uuid.uuid4()
                elif fbkey == 'confidence':
                    value = ibs.const.CONFIDENCE.CODE_TO_INT[value]
                elif fbkey == 'evidence_decision':
                    value = ibs.const.EVIDENCE_DECISION.CODE_TO_INT[value]
                elif fbkey == 'meta_decision':
                    value = ibs.const.META_DECISION.CODE_TO_INT[value]
                add_review_kw[review_key].append(value)
        return aid_1_list, aid_2_list, add_review_kw

    def _write_wbia_staging_feedback(infr, feedback):
        """
        feedback = infr.internal_feedback
        ibs.staging.get_table_as_pandas('reviews')
        """
        infr.print('write_wbia_staging_feedback {}'.format(len(feedback)), 1)
        tup = infr._prepare_write_wbia_staging_feedback(feedback)
        aid_1_list, aid_2_list, add_review_kw = tup

        ibs = infr.ibs
        review_id_list = ibs.add_review(aid_1_list, aid_2_list, **add_review_kw)
        duplicates = ut.find_duplicate_items(review_id_list)
        if len(duplicates) != 0:
            raise AssertionError(
                'Staging should only be appended to but we found a duplicate'
                ' row. ' + str(duplicates)
            )

    def write_wbia_staging_feedback(infr):
        """
        Commit all reviews in internal_feedback into the staging table.  The
        edges are removed from interal_feedback and added to external feedback.
        The staging tables stores each review in the order it happened so
        history is fully reconstructable if staging is never deleted.

        This write function is done using the implicit delta maintained by
        infr.internal_feedback. Therefore, it take no args. This is generally
        called automatically by `infr.accept`.
        """
        if len(infr.internal_feedback) == 0:
            infr.print('write_wbia_staging_feedback 0', 1)
            return
        # Write internal feedback to disk
        infr._write_wbia_staging_feedback(infr.internal_feedback)
        # Copy internal feedback into external
        for edge, feedbacks in infr.internal_feedback.items():
            infr.external_feedback[edge].extend(feedbacks)
        # Delete internal feedback
        infr.internal_feedback = ut.ddict(list)

    def write_wbia_annotmatch_feedback(infr, edge_delta_df=None):
        """
        Commits the current state in external and internal into the annotmatch
        table. Annotmatch only stores the final review in the history of reviews.

        By default this will sync the current graph state to the annotmatch
        table. It computes the edge_delta under the hood, so if you already
        made one then you can pass it in for a little extra speed.

        Args:
            edge_delta_df (pd.DataFrame): precomputed using match_state_delta.
                if None it will be computed under the hood.

        """
        if edge_delta_df is None:
            edge_delta_df = infr.match_state_delta(old='annotmatch', new='all')
        infr.print('write_wbia_annotmatch_feedback %r' % (len(edge_delta_df)))
        ibs = infr.ibs
        edge_delta_df_ = edge_delta_df.reset_index()
        # Find the rows not yet in the annotmatch table
        is_add = edge_delta_df_['am_rowid'].isnull().values
        add_df = edge_delta_df_.loc[is_add]
        # Assign then a new annotmatch rowid
        add_ams = ibs.add_annotmatch_undirected(
            add_df['aid1'].values, add_df['aid2'].values
        )
        edge_delta_df_.loc[is_add, 'am_rowid'] = add_ams

        # Set residual matching data
        new_evidence_decisions = ut.take(
            ibs.const.EVIDENCE_DECISION.CODE_TO_INT,
            edge_delta_df_['new_evidence_decision'],
        )
        new_meta_decisions = ut.take(
            ibs.const.META_DECISION.CODE_TO_INT, edge_delta_df_['new_meta_decision']
        )
        new_tags = [
            '' if tags is None else ';'.join(tags) for tags in edge_delta_df_['new_tags']
        ]
        new_conf = ut.dict_take(
            ibs.const.CONFIDENCE.CODE_TO_INT, edge_delta_df_['new_confidence'], None
        )
        new_timestamp = edge_delta_df_['new_timestamp']
        new_reviewer = edge_delta_df_['new_user_id']
        am_rowids = edge_delta_df_['am_rowid'].values
        ibs.set_annotmatch_evidence_decision(am_rowids, new_evidence_decisions)
        ibs.set_annotmatch_meta_decision(am_rowids, new_meta_decisions)
        ibs.set_annotmatch_tag_text(am_rowids, new_tags)
        ibs.set_annotmatch_confidence(am_rowids, new_conf)
        ibs.set_annotmatch_reviewer(am_rowids, new_reviewer)
        ibs.set_annotmatch_posixtime_modified(am_rowids, new_timestamp)
        # ibs.set_annotmatch_count(am_rowids, new_timestamp) TODO

    def write_wbia_name_assignment(infr, name_delta_df=None, **kwargs):
        """
        Write the name delta to the annotations table.

        It computes the name delta under the hood, so if you already made one
        then you can pass it in for a little extra speed.

        Note:
            This will call infr.relabel_using_reviews(rectify=True) if
            name_delta_df is not given directly.

        Args:
            name_delta_df (pd.DataFrame): if None, the value is computed using
                `get_wbia_name_delta`. Note you should ensure this delta is made
                after nodes have been relabeled using reviews.
        """
        if name_delta_df is None:
            name_delta_df = infr.get_wbia_name_delta()
        infr.print('write_wbia_name_assignment id %d' % len(name_delta_df))
        aid_list = name_delta_df.index.values
        new_name_list = name_delta_df['new_name'].values
        infr.ibs.set_annot_names(aid_list, new_name_list, **kwargs)

    def get_wbia_name_delta(infr, ignore_unknown=True, relabel=True):
        """
        Rectifies internal name_labels with the names stored in the name table.

        Return a pandas dataframe indicating which names have changed for what
        annotations.

        Args:
            ignore_unknown (bool): if True does not return deltas for unknown
                annotations (those with degree 0).
            relabel (bool): if True, ensures that all nodes are labeled based
                on the current PCCs.

        Returns:
            pd.DataFrame - name_delta_df - data frame where each row specifies
                an aid and its `old_name` which is in the wbia database and
                the `new_name` which is what we infer it should be renamed to.

        Example:
            infr.write_wbia_name_assignment

        CommandLine:
            python -m wbia.algo.graph.mixin_wbia get_wbia_name_delta

        Doctest:
            >>> from wbia.algo.graph.mixin_wbia import *  # NOQA
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', aids=list(range(1, 10)),
            >>>                             autoinit='annotmatch', verbose=4)
            >>> pccs1 = list(infr.positive_components())
            >>> print('pccs1 = %r' % (pccs1,))
            >>> print('names = {}'.format(list(infr.gen_node_values('name_label', infr.aids))))
            >>> assert pccs1 == [{1, 2, 3, 4}, {5, 6, 7, 8}, {9}]
            >>> # Split a PCC and then merge two other PCCs
            >>> infr.add_feedback_from([(1, 2), (1, 3), (1, 4)], evidence_decision=NEGTV)
            >>> infr.add_feedback((6, 7), NEGTV)
            >>> infr.add_feedback((5, 8), NEGTV)
            >>> infr.add_feedback((4, 5), POSTV)
            >>> infr.add_feedback((7, 8), POSTV)
            >>> pccs2 = list(infr.positive_components())
            >>> print('pccs2 = %r' % (pccs2,))
            >>> pccs2 = sorted(pccs2)
            >>> assert pccs2 == [{9}, {1}, {2, 3, 4, 5, 6}, {7, 8}]
            >>> print(list(infr.gen_node_values('name_label', infr.aids)))
            >>> name_delta_df = infr.get_wbia_name_delta()
            >>> result = str(name_delta_df)
            >>> print(result)
                old_name       new_name
            aid
            1     06_410  IBEIS_PZ_0042
            5     07_061         06_410
            6     07_061         06_410

        Doctest:
            >>> from wbia.algo.graph.mixin_wbia import *  # NOQA
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', aids=list(range(1, 10)),
            >>>                             autoinit='annotmatch', verbose=4)
            >>> infr.add_feedback_from([(1, 2), (1, 3), (1, 4)], evidence_decision=NEGTV)
            >>> infr.add_feedback((4, 5), POSTV)
            >>> name_delta_df = infr.get_wbia_name_delta()
            >>> result = str(name_delta_df)
            >>> print(result)
                old_name new_name
            aid
            2     06_410   07_061
            3     06_410   07_061
            4     06_410   07_061

        Doctest:
            >>> from wbia.algo.graph.mixin_wbia import *  # NOQA
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', aids=list(range(1, 10)),
            >>>                             autoinit='annotmatch', verbose=4)
            >>> name_delta_df = infr.get_wbia_name_delta()
            >>> result = str(name_delta_df)
            >>> print(result)
            Empty DataFrame
            Columns: [old_name, new_name]
            Index: []
        """
        infr.print('constructing name delta', 3)

        if relabel:
            infr.relabel_using_reviews(rectify=True)

        import pandas as pd

        graph = infr.graph
        node_to_new_label = nx.get_node_attributes(graph, 'name_label')
        aids = list(node_to_new_label.keys())
        old_names = infr.ibs.get_annot_name_texts(aids, distinguish_unknowns=True)
        # Indicate that unknown names should be replaced
        old_names = [
            None if n.startswith(infr.ibs.const.UNKNOWN) else n for n in old_names
        ]
        new_labels = ut.take(node_to_new_label, aids)
        # Recycle as many old names as possible
        label_to_name, needs_assign, unknown_labels = infr._rectify_names(
            old_names, new_labels
        )
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
                node: label
                for node, label in node_to_new_label.items()
                if label not in unknown_labels_
            }
        aid_list = list(node_to_new_label.keys())
        new_name_list = ut.take(label_to_name, node_to_new_label.values())
        old_name_list = infr.ibs.get_annot_name_texts(aid_list, distinguish_unknowns=True)
        # Put into a dataframe for convinience
        name_delta_df_ = pd.DataFrame(
            {'old_name': old_name_list, 'new_name': new_name_list},
            columns=['old_name', 'new_name'],
            index=pd.Index(aid_list, name='aid'),
        )
        changed_flags = name_delta_df_['old_name'] != name_delta_df_['new_name']
        name_delta_df = name_delta_df_[changed_flags]
        infr.print('finished making name delta', 3)
        return name_delta_df

    def read_wbia_staging_feedback(infr, edges=None):
        """
        Reads feedback from review staging table.

        Args:
            infr (?):

        Returns:
            ?: feedback

        CommandLine:
            python -m wbia.algo.graph.mixin_wbia read_wbia_staging_feedback

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_wbia import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb('GZ_Master1')
            >>> infr = wbia.AnnotInference(ibs=ibs, aids='all')
            >>> feedback = infr.read_wbia_staging_feedback()
            >>> result = ('feedback = %s' % (ut.repr2(feedback),))
            >>> print(result)
        """
        # TODO: READ ONLY AFTER THE LATEST ANNOTMATCH TIME STAMP

        infr.print('read_wbia_staging_feedback', 1)
        ibs = infr.ibs

        from wbia.control.manual_review_funcs import hack_create_aidpair_index

        hack_create_aidpair_index(ibs)

        if edges:
            review_ids = ut.flatten(ibs.get_review_rowids_from_edges(edges))
        else:
            review_ids = ibs.get_review_rowids_between(infr.aids)

        review_ids = sorted(review_ids)

        infr.print('read %d staged reviews' % (len(review_ids)), 2)

        from wbia.control.manual_review_funcs import (
            # REVIEW_UUID,
            REVIEW_AID1,
            REVIEW_AID2,
            REVIEW_COUNT,
            REVIEW_EVIDENCE_DECISION,
            REVIEW_META_DECISION,
            REVIEW_USER_IDENTITY,
            REVIEW_USER_CONFIDENCE,
            REVIEW_TAGS,
            REVIEW_TIME_CLIENT_START,
            REVIEW_TIME_CLIENT_END,
            REVIEW_TIME_SERVER_START,
            REVIEW_TIME_SERVER_END,
        )

        add_review_alias = ut.odict(
            [
                (REVIEW_AID1, 'aid1'),
                (REVIEW_AID2, 'aid2'),
                # (REVIEW_UUID              , 'uuid'),
                (REVIEW_EVIDENCE_DECISION, 'evidence_decision'),
                (REVIEW_META_DECISION, 'meta_decision'),
                (REVIEW_USER_IDENTITY, 'user_id'),
                (REVIEW_USER_CONFIDENCE, 'confidence'),
                (REVIEW_TAGS, 'tags'),
                (REVIEW_TIME_CLIENT_START, 'timestamp_c1'),
                (REVIEW_TIME_CLIENT_END, 'timestamp_c2'),
                (REVIEW_TIME_SERVER_START, 'timestamp_s1'),
                (REVIEW_TIME_SERVER_END, 'timestamp'),
                (REVIEW_COUNT, 'num_reviews'),
            ]
        )
        columns = tuple(add_review_alias.keys())
        feedback_keys = list(add_review_alias.values())
        review_data = ibs.staging.get(ibs.const.REVIEW_TABLE, columns, review_ids)
        # table = infr.ibs.staging.get_table_as_pandas(
        #     ibs.const.REVIEW_TABLE, rowids=review_ids, columns=columns)

        lookup_decision = ibs.const.EVIDENCE_DECISION.INT_TO_CODE
        lookup_meta = ibs.const.META_DECISION.INT_TO_CODE
        lookup_conf = ibs.const.CONFIDENCE.INT_TO_CODE

        feedback = ut.ddict(list)
        for data in review_data:
            feedback_item = dict(zip(feedback_keys, data))
            aid1 = feedback_item.pop('aid1')
            aid2 = feedback_item.pop('aid2')
            edge = nxu.e_(aid1, aid2)

            tags = feedback_item['tags']
            feedback_item['meta_decision'] = lookup_meta[feedback_item['meta_decision']]
            feedback_item['evidence_decision'] = lookup_decision[
                feedback_item['evidence_decision']
            ]
            feedback_item['confidence'] = lookup_conf[feedback_item['confidence']]
            feedback_item['tags'] = [] if not tags else tags.split(';')

            feedback[edge].append(feedback_item)
        return feedback

    def read_wbia_annotmatch_feedback(infr, edges=None):
        r"""
        Reads feedback from annotmatch table and returns the result.
        Internal state is not changed.

        Args:
            only_existing_edges (bool): if True only reads info existing edges

        CommandLine:
            python -m wbia.algo.graph.core read_wbia_annotmatch_feedback

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> feedback = infr.read_wbia_annotmatch_feedback()
            >>> items = feedback[(2, 3)]
            >>> result = ('feedback = %s' % (ut.repr2(feedback, nl=2),))
            >>> print(result)
            >>> assert len(feedback) >= 2, 'should contain at least 2 edges'
            >>> assert len(items) == 1, '2-3 should have one review'
            >>> assert items[0]['evidence_decision'] == POSTV, '2-3 must match'
        """
        infr.print('read_wbia_annotmatch_feedback', 1)
        ibs = infr.ibs
        if edges is not None:
            matches = ibs.matches(edges=edges)
        else:
            matches = ibs.annots(infr.aids).matches()

        infr.print('read %d annotmatch rowids' % (len(matches)), 2)
        # Use explicit truth state to mark truth
        aids1 = matches.aid1
        aids2 = matches.aid2

        column_lists = {
            'evidence_decision': matches.evidence_decision_code,
            'meta_decision': matches.meta_decision_code,
            'timestamp_c1': [None] * len(matches),
            'timestamp_c2': [None] * len(matches),
            'timestamp_s1': [None] * len(matches),
            'timestamp': matches.posixtime_modified,
            'tags': matches.case_tags,
            'user_id': matches.reviewer,
            'confidence': matches.confidence_code,
            'num_reviews': matches.count,
        }

        feedback = ut.ddict(list)
        for aid1, aid2, row in zip(aids1, aids2, zip(*column_lists.values())):
            edge = nxu.e_(aid1, aid2)
            feedback_item = dict(zip(column_lists.keys(), row))
            feedback[edge].append(feedback_item)
        infr.print('read %d annotmatch entries' % (len(feedback)), 1)
        return feedback

    def reset_staging_with_ensure(infr):
        """
        Make sure staging has all info that annotmatch has.
        """
        staging_feedback = infr.read_wbia_staging_feedback()
        if len(staging_feedback) == 0:
            infr.internal_feedback = infr.read_wbia_annotmatch_feedback()
            infr.write_wbia_staging_feedback()
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
        am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
        rectified_feedback_ = infr._rectify_feedback(feedback)
        rectified_feedback = ut.take(rectified_feedback_, aid_pairs)
        decision = ut.dict_take_column(rectified_feedback, 'evidence_decision')
        tags = ut.dict_take_column(rectified_feedback, 'tags')
        confidence = ut.dict_take_column(rectified_feedback, 'confidence')
        timestamp_c1 = ut.dict_take_column(rectified_feedback, 'timestamp_c1')
        timestamp_c2 = ut.dict_take_column(rectified_feedback, 'timestamp_c2')
        timestamp_s1 = ut.dict_take_column(rectified_feedback, 'timestamp_s1')
        timestamp = ut.dict_take_column(rectified_feedback, 'timestamp')
        user_id = ut.dict_take_column(rectified_feedback, 'user_id')
        meta_decision = ut.dict_take_column(rectified_feedback, 'meta_decision')
        df = pd.DataFrame([])
        df['evidence_decision'] = decision
        df['meta_decision'] = meta_decision
        df['aid1'] = aids1
        df['aid2'] = aids2
        df['tags'] = [None if ts is None else [t.lower() for t in ts if t] for ts in tags]
        df['confidence'] = confidence
        df['timestamp_c1'] = timestamp_c1
        df['timestamp_c2'] = timestamp_c2
        df['timestamp_s1'] = timestamp_s1
        df['timestamp'] = timestamp
        df['user_id'] = user_id
        df['am_rowid'] = am_rowids
        df.set_index(['aid1', 'aid2'], inplace=True, drop=True)
        return df

    def _feedback_df(infr, key):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.core import *  # NOQA
            >>> import pandas as pd
            >>> infr = testdata_infr('testdb1')
            >>> assert 'meta_decision' in infr._feedback_df('annotmatch').columns
        """
        if key == 'annotmatch':
            feedback = infr.read_wbia_annotmatch_feedback()
        elif key == 'staging':
            feedback = infr.read_wbia_staging_feedback()
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

        By default this will return a pandas dataframe indicating which edges
        in the annotmatch table have changed and all new edges relative to the
        current infr.graph state.

        Notes:
            valid values for `old` and `new` are {'annotmatch', 'staging',
            'all', 'internal', or 'external'}.

            The args old/new='all' resolves to the internal graph state,
            'annotmatch' resolves to the on-disk annotmatch table, and
            'staging' resolves to the on-disk staging table (you can further
            separate all by specifying 'internal' or 'external').  You any of
            these old/new combinations to check differences in the state.
            However, the default values are what you use to sync the graph
            state to annotmatch.

        Args:
            old (str): indicates the old data (i.e. the place that will be
                written to)
            new (str): indicates the new data (i.e. the data to write)

        Returns:
            pd.DataFrame - edge_delta_df - indicates the old and new values
                of the changed edge attributes.

        CommandLine:
            python -m wbia.algo.graph.core match_state_delta

        Doctest:
            >>> from wbia.algo.graph.mixin_wbia import *  # NOQA
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', aids=list(range(1, 10)),
            >>>                             autoinit='annotmatch', verbose=4)
            >>> # Split a PCC and then merge two other PCCs
            >>> infr.add_feedback((1, 2), NEGTV)
            >>> infr.add_feedback((6, 7), NEGTV)
            >>> infr.add_feedback((5, 8), NEGTV)
            >>> infr.add_feedback((4, 5), POSTV)
            >>> infr.add_feedback((7, 8), POSTV)
            >>> edge_delta_df = infr.match_state_delta()
            >>> subset = edge_delta_df[['old_evidence_decision', 'new_evidence_decision']]
            >>> result = str(subset)
            >>> # if this doctest fails maybe PZ_MTEST has a non-determenistic reset?
            >>> print(result)
                      old_evidence_decision new_evidence_decision
            aid1 aid2
            1    2                    match               nomatch
            5    8               unreviewed               nomatch
            6    7               unreviewed               nomatch
            7    8                    match                 match
            4    5                      NaN                 match
        """
        old_feedback = infr._feedback_df(old)
        new_feedback = infr._feedback_df(new)
        edge_delta_df = infr._make_state_delta(old_feedback, new_feedback)
        return edge_delta_df

    @classmethod
    def _make_state_delta(cls, old_feedback, new_feedback):
        r"""
        CommandLine:
            python -m wbia.algo.graph.mixin_wbia IBEISIO._make_state_delta
            python -m wbia.algo.graph.mixin_wbia IBEISIO._make_state_delta:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.core import *  # NOQA
            >>> import pandas as pd
            >>> columns = ['evidence_decision', 'aid1', 'aid2', 'am_rowid', 'tags']
            >>> new_feedback = old_feedback = pd.DataFrame([
            >>> ], columns=columns).set_index(['aid1', 'aid2'], drop=True)
            >>> edge_delta_df = AnnotInference._make_state_delta(old_feedback,
            >>>                                                  new_feedback)
            >>> result = ('edge_delta_df =\n%s' % (edge_delta_df,))
            >>> print(result)
            edge_delta_df =
            Empty DataFrame
            Columns: [am_rowid, old_evidence_decision, new_evidence_decision, old_tags, new_tags, is_new]
            Index: []

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.core import *  # NOQA
            >>> import pandas as pd
            >>> columns = ['evidence_decision', 'meta_decision', 'aid1', 'aid2', 'am_rowid', 'tags']
            >>> old_feedback = pd.DataFrame([
            >>>     [NEGTV, 'diff', 100, 101, 1000, []],
            >>>     [POSTV, 'same', 101, 102, 1001, []],
            >>>     [POSTV, 'null', 103, 104, 1002, []],
            >>>     [NEGTV, 'null', 101, 104, 1004, []],
            >>> ], columns=columns).set_index(['aid1', 'aid2'], drop=True)
            >>> new_feedback = pd.DataFrame([
            >>>     [POSTV, 'null', 101, 102, 1001, []],
            >>>     [NEGTV, 'null', 103, 104, 1002, []],
            >>>     [POSTV, 'null', 101, 104, 1004, []],
            >>>     [NEGTV, 'null', 102, 103, None, []],
            >>>     [NEGTV, 'null', 100, 103, None, []],
            >>>     [INCMP, 'same', 107, 109, None, []],
            >>> ], columns=columns).set_index(['aid1', 'aid2'], drop=True)
            >>> edge_delta_df = AnnotInference._make_state_delta(old_feedback,
            >>>                                                  new_feedback)
            >>> result = ('edge_delta_df =\n%s' % (edge_delta_df,))
            >>> print(result)
                       am_rowid old_decision new_decision old_tags new_tags is_new
            aid1 aid2
            101  104     1004.0      nomatch        match       []       []  False
            103  104     1002.0        match      nomatch       []       []  False
            100  103        NaN          NaN      nomatch      NaN       []   True
            102  103        NaN          NaN      nomatch      NaN       []   True
            107  109        NaN          NaN      notcomp      NaN       []   True
        """
        import wbia
        import pandas as pd
        from six.moves import reduce
        import operator as op

        # Ensure input is in the expected format
        new_index = new_feedback.index
        old_index = old_feedback.index
        assert new_index.names == ['aid1', 'aid2'], 'not indexed on edges'
        assert old_index.names == ['aid1', 'aid2'], 'not indexed on edges'
        assert all(u < v for u, v in new_index.values), 'bad direction'
        assert all(u < v for u, v in old_index.values), 'bad direction'
        # Determine what edges have changed
        isect_edges = new_index.intersection(old_index)
        isect_new = new_feedback.loc[isect_edges]
        isect_old = old_feedback.loc[isect_edges]

        # If any important column is different we mark the row as changed
        data_columns = wbia.AnnotInference.feedback_data_keys
        important_columns = ['meta_decision', 'evidence_decision', 'tags']
        other_columns = ut.setdiff(data_columns, important_columns)
        if len(isect_edges) > 0:
            changed_gen = [isect_new[c] != isect_old[c] for c in important_columns]
            is_changed = reduce(op.or_, changed_gen)
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
            prep_new, how='outer', left_on=merge_keys, right_on=merge_keys
        )
        # Reorder the columns
        col_order = [
            'old_evidence_decision',
            'new_evidence_decision',
            'old_tags',
            'new_tags',
            'old_meta_decision',
            'new_meta_decision',
        ]
        edge_delta_df = merged_df.reindex(
            columns=(ut.setdiff(merged_df.columns.values, col_order) + col_order)
        )
        edge_delta_df.set_index(['aid1', 'aid2'], inplace=True, drop=True)
        edge_delta_df = edge_delta_df.assign(is_new=False)
        if len(add_edges):
            edge_delta_df.loc[add_edges, 'is_new'] = True
        return edge_delta_df

    def _debug_edge_gt(infr, edge):
        ibs = infr.ibs
        # Look at annotmatch and staging table for this edge
        matches = ibs.matches(edges=[edge])
        review_ids = ibs.get_review_rowids_between(edge)

        import pandas as pd

        pd.options.display.max_rows = 20
        pd.options.display.max_columns = 40
        pd.options.display.width = 160
        pd.options.display.float_format = lambda x: '%.4f' % (x,)

        df_a = ibs.db['annotmatch'].as_pandas(matches._rowids)
        df_s = ibs.staging['reviews'].as_pandas(review_ids)

        print('=====')

        print('AnnotMatch Raw')
        df_a = df_a.rename(
            columns={c: c.replace('annotmatch_', '') for c in df_a.columns}
        )
        df_s = df_s.rename(
            columns={
                'annot_rowid1': 'aid1',
                'annot_rowid2': 'aid2',
                'reviewer': 'user_id',
                'tag_text': 'tag',
                'posixtime_modified': 'ts_s2',
            }
        )
        print(df_a)

        print('AnnotMatch Feedback')
        print(infr._pandas_feedback_format(infr.read_wbia_staging_feedback([edge])))

        print('----')

        print('Staging Raw')
        df_s = df_s.rename(columns={c: c.replace('review_', '') for c in df_s.columns})
        df_s = df_s.rename(
            columns={
                'annot_1_rowid': 'aid1',
                'annot_2_rowid': 'aid2',
                'user_identity': 'user_id',
                'user_confidence': 'confidence',
                'client_start_time_posix': 'ts_c1',
                'client_end_time_posix': 'ts_c2',
                'server_end_time_posix': 'ts_s2',
                'server_start_time_posix': 'ts_s1',
            }
        )
        df_s = ut.pandas_reorder(
            df_s,
            [
                'rowid',
                'aid1',
                'aid2',
                'count',
                'evidence_decision',
                'meta_decision',
                'tags',
                'confidence',
                'user_id',
                'ts_s1',
                'ts_c1',
                'ts_c2',
                'ts_s2',
                'uuid',
            ],
        )
        print(df_s)

        print('Staging Feedback')
        print(infr._pandas_feedback_format(infr.read_wbia_annotmatch_feedback([edge])))
        print('____')


@six.add_metaclass(ut.ReloadingMetaclass)
class IBEISGroundtruth(object):
    """
    Methods for generating training labels for classifiers
    """

    @profile
    def wbia_guess_if_comparable(infr, aid_pairs):
        """
        Takes a guess as to which annots are not comparable based on scores and
        viewpoints. If either viewpoints is null assume they are comparable.
        """
        aid_pairs = np.asarray(aid_pairs)
        ibs = infr.ibs

        dists = ibs.get_annotedge_viewdist(aid_pairs)

        comp_by_viewpoint = (dists < 2) | np.isnan(dists)

        is_comp_guess = comp_by_viewpoint
        return is_comp_guess

    def wbia_is_comparable(infr, aid_pairs, allow_guess=True):
        """
        Guesses by default when real comparable information is not available.
        """
        ibs = infr.ibs
        if allow_guess:
            # Guess if comparability information is unavailable
            is_comp_guess = infr.wbia_guess_if_comparable(aid_pairs)
            is_comp = is_comp_guess.copy()
        else:
            is_comp = np.full(len(aid_pairs), np.nan)
        # But use information that we have
        am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        truths = ut.replace_nones(ibs.get_annotmatch_evidence_decision(am_rowids), np.nan)
        truths = np.asarray(truths)
        is_notcomp_have = truths == ibs.const.EVIDENCE_DECISION.INCOMPARABLE
        is_comp_have = (truths == ibs.const.EVIDENCE_DECISION.POSITIVE) | (
            truths == ibs.const.EVIDENCE_DECISION.NEGATIVE
        )
        is_comp[is_notcomp_have] = False
        is_comp[is_comp_have] = True
        return is_comp

    def wbia_is_photobomb(infr, aid_pairs):
        ibs = infr.ibs
        am_rowids = ibs.get_annotmatch_rowid_from_edges(aid_pairs)
        am_tags = ibs.get_annotmatch_case_tags(am_rowids)
        is_pb = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
        return is_pb

    def wbia_is_same(infr, aid_pairs):
        aids1, aids2 = np.asarray(aid_pairs).T
        nids1 = infr.ibs.get_annot_nids(aids1)
        nids2 = infr.ibs.get_annot_nids(aids2)
        is_same = nids1 == nids2
        return is_same


# VVVVV Non-complete non-general functions  VVVV


def _update_staging_to_annotmatch(infr):
    """
    BE VERY CAREFUL WITH THIS FUNCTION

    >>> import wbia
    >>> ibs = wbia.opendb('PZ_Master1')
    >>> infr = wbia.AnnotInference(ibs, aids=ibs.get_valid_aids())

    infr.reset_feedback('annotmatch', apply=True)
    infr.status()
    """
    print('Finding entries in annotmatch that are missing in staging')
    reverse_df = infr.match_state_delta('annotmatch', 'staging')
    if len(reverse_df) > 0:
        raise AssertionError(
            'Cannot update staging because ' 'some staging items have not been commited.'
        )
    df = infr.match_state_delta('staging', 'annotmatch')
    print(
        'There are {}/{} annotmatch items that do not exist in staging'.format(
            sum(df['is_new']), len(df)
        )
    )
    print(ut.repr4(infr.wbia_edge_delta_info(df)))

    # Find places that exist in annotmatch but not in staging
    flags = pd.isnull(df['old_evidence_decision'])
    missing_df = df[flags]
    alias = {'new_' + k: k for k in infr.feedback_data_keys}
    tmp = missing_df[list(alias.keys())].rename(columns=alias)
    missing_feedback = {k: [v] for k, v in tmp.to_dict('index').items()}
    feedback = missing_feedback

    infr._write_wbia_staging_feedback(feedback)

    # am_fb = infr.read_wbia_annotmatch_feedback()
    # staging_fb = infr.read_wbia_staging_feedback()
    # set(am_fb.keys()) - set(staging_fb.keys())
    # set(staging_fb.keys()) == set(am_fb.keys())


def fix_annotmatch_to_undirected_upper(ibs):
    """
    Enforce that all items in annotmatch are undirected upper

    import wbia
    # ibs = wbia.opendb('PZ_Master1')
    ibs = wbia.opendb('PZ_PB_RF_TRAIN')
    """
    df = ibs.db.get_table_as_pandas('annotmatch')
    df.set_index(['annot_rowid1', 'annot_rowid2'], inplace=True, drop=False)
    # We want everything in upper triangular form
    is_upper = df['annot_rowid1'] < df['annot_rowid2']
    is_lower = df['annot_rowid1'] > df['annot_rowid2']
    is_equal = df['annot_rowid1'] == df['annot_rowid2']
    assert not np.any(is_equal)

    print(is_lower.sum())
    print(is_upper.sum())

    upper_edges = ut.estarmap(nxu.e_, df[is_upper].index.tolist())
    lower_edges = ut.estarmap(nxu.e_, df[is_lower].index.tolist())
    both_edges = ut.isect(upper_edges, lower_edges)
    if len(both_edges) > 0:
        both_upper = both_edges
        both_lower = [tuple(e[::-1]) for e in both_edges]

        df1 = df.loc[both_upper].reset_index(drop=True)
        df2 = df.loc[both_lower].reset_index(drop=True)

        df3 = df1.copy()

        cols = [
            'annotmatch_evidence_decision',
            'annotmatch_meta_decision',
            'annotmatch_confidence',
            'annotmatch_tag_text',
            'annotmatch_posixtime_modified',
            'annotmatch_reviewer',
        ]

        ed_key = 'annotmatch_evidence_decision'

        for col in cols:
            idxs = np.where(pd.isnull(df3[col]))[0]
            df3.loc[idxs, col] = df2.loc[idxs, col]

        assert all(pd.isnull(df2.annotmatch_posixtime_modified)), 'should not happen'
        assert all(pd.isnull(df2.annotmatch_reviewer)), 'should not happen'
        assert all(pd.isnull(df2.annotmatch_confidence)), 'should not happen'

        flags = (
            (df3[ed_key] != df2[ed_key])
            & ~pd.isnull(df3[ed_key])
            & ~pd.isnull(df2[ed_key])
        )
        if any(flags & ~pd.isnull(df2.annotmatch_posixtime_modified)):
            assert False, 'need to rectify'

        tags2 = df2.annotmatch_tag_text.map(lambda x: {t for t in x.split(';') if t})
        tags3 = df3.annotmatch_tag_text.map(lambda x: {t for t in x.split(';') if t})

        # Merge the tags
        df3['annotmatch_tag_text'] = [
            ';'.join(sorted(t1.union(t2))) for t1, t2 in zip(tags3, tags2)
        ]

        delete_df = df3[pd.isnull(df3[ed_key])]

        df4 = df3[~pd.isnull(df3[ed_key])]
        ibs.set_annotmatch_evidence_decision(
            df4.annotmatch_rowid, [None if pd.isnull(x) else int(x) for x in df4[ed_key]],
        )
        ibs.set_annotmatch_tag_text(
            df4.annotmatch_rowid, df4.annotmatch_tag_text.tolist()
        )
        ibs.set_annotmatch_confidence(
            df4.annotmatch_rowid,
            [None if pd.isnull(x) else int(x) for x in df4.annotmatch_confidence],
        )
        ibs.set_annotmatch_reviewer(
            df4.annotmatch_rowid,
            [None if pd.isnull(x) else str(x) for x in df4.annotmatch_reviewer],
        )
        ibs.set_annotmatch_posixtime_modified(
            df4.annotmatch_rowid,
            [None if pd.isnull(x) else int(x) for x in df4.annotmatch_posixtime_modified],
        )

        ibs.delete_annotmatch(delete_df.annotmatch_rowid)

        # forwards_edge4 = [nxu.e_(u, v) for u, v in df4[['annot_rowid1', 'annot_rowid2']].values.tolist()]
        # forwards_rowids4 = ibs.get_annotmatch_rowid_from_superkey(forwards_edge4)
        backwards_edge4 = [
            nxu.e_(u, v)[::-1]
            for u, v in df4[['annot_rowid1', 'annot_rowid2']].values.tolist()
        ]
        backwards_rowids4 = ibs.get_annotmatch_rowid_from_superkey(backwards_edge4)
        ibs.delete_annotmatch(backwards_rowids4)

    # -------------------------

    # NOW WE HAVE RECIFIED DUPLICATE PAIRS AND THERE IS ONLY ONE AID PAIR PER DB
    # SO WE CAN SAFELY FLIP EVERYTHING TO BE IN UPPER TRIANGULAR MODE
    df = ibs.db.get_table_as_pandas('annotmatch')
    df.set_index(['annot_rowid1', 'annot_rowid2'], inplace=True, drop=False)
    # We want everything in upper triangular form
    is_upper = df['annot_rowid1'] < df['annot_rowid2']
    is_lower = df['annot_rowid1'] > df['annot_rowid2']
    is_equal = df['annot_rowid1'] == df['annot_rowid2']
    assert not np.any(is_equal)

    bad_lower_edges = df[is_lower].index.tolist()
    upper_edges = ut.estarmap(nxu.e_, df[is_upper].index.tolist())
    fix_lower_edges = ut.estarmap(nxu.e_, bad_lower_edges)
    both_edges = ut.isect(upper_edges, fix_lower_edges)
    assert len(both_edges) == 0, 'should not have any both edges anymore'

    lower_rowids = ibs.get_annotmatch_rowid_from_superkey(*list(zip(*bad_lower_edges)))

    assert not any(x is None for x in lower_rowids)

    # Ensure all edges are upper triangular in the database
    id_iter = lower_rowids
    colnames = ('annot_rowid1', 'annot_rowid2')
    ibs.db.set('annotmatch', colnames, fix_lower_edges, id_iter)

    df = ibs.db.get_table_as_pandas('annotmatch')
    df.set_index(['annot_rowid1', 'annot_rowid2'], inplace=True, drop=False)
    # We want everything in upper triangular form
    is_lower = df['annot_rowid1'] > df['annot_rowid2']
    assert is_lower.sum() == 0


def _is_staging_above_annotmatch(infr):
    """
    conversion step: make sure the staging db is ahead of match

    SeeAlso:
        _update_staging_to_annotmatch
    """
    ibs = infr.ibs
    n_stage = ibs.staging.get_row_count(ibs.const.REVIEW_TABLE)
    n_annotmatch = ibs.db.get_row_count(ibs.const.ANNOTMATCH_TABLE)
    return n_stage >= n_annotmatch
    # stage_fb = infr.read_wbia_staging_feedback()
    # match_fb = infr.read_wbia_annotmatch_feedback()
    # set(match_fb.keys()) - set(stage_fb.keys())
    # set(stage_fb.keys()) == set(match_fb.keys())


def needs_conversion(infr):
    # not sure what the criteria is exactly. probably depricate
    num_names = len(set(infr.get_node_attrs('name_label').values()))
    num_pccs = infr.pos_graph.number_of_components()
    return num_pccs == 0 and num_names > 0


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.mixin_wbia
        python -m wbia.algo.graph.mixin_wbia --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
