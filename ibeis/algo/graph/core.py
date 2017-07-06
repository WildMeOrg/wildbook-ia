# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np  # NOQA
import utool as ut
# import logging
import itertools as it
import copy
import six
import collections
from ibeis import constants as const
from ibeis.algo.graph import nx_dynamic_graph
# from ibeis.algo.graph import _dep_mixins
from ibeis.algo.graph import mixin_viz
from ibeis.algo.graph import mixin_helpers
from ibeis.algo.graph import mixin_dynamic
from ibeis.algo.graph import mixin_loops
from ibeis.algo.graph import mixin_matching
from ibeis.algo.graph import mixin_groundtruth
from ibeis.algo.graph import mixin_ibeis
from ibeis.algo.graph import nx_utils as nxu
import pandas as pd
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN
from ibeis.algo.graph.state import UNINFERABLE
from ibeis.algo.graph.state import SAME, DIFF, NULL
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


DEBUG_CC = False
# DEBUG_CC = True


def _rectify_decision(evidence_decision, meta_decision):
    # Default to the decision based on the media evidence
    decision = evidence_decision
    # Overwrite the graph decision with the meta decision if necessary
    if meta_decision == SAME:
        if decision in UNINFERABLE:
            decision = POSTV
        elif decision == NEGTV:
            raise ValueError('evidence=negative and meta=same')
    elif meta_decision == DIFF:
        if decision in UNINFERABLE:
            decision = NEGTV
        elif decision == POSTV:
            raise ValueError('evidence=positive and meta=diff')
    return decision


class Feedback(object):
    def _check_edge(infr, edge):
        aid1, aid2 = edge
        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))

    def add_feedback_from(infr, items, verbose=None, **kwargs):
        if verbose is None:
            verbose = infr.verbose > 5
        if isinstance(items, pd.DataFrame):
            if list(items.index.names) == ['aid1', 'aid2']:
                for edge, data in items.iterrows():
                    infr.add_feedback(edge=edge, verbose=verbose, **data)
            else:
                raise ValueError(
                    'Cannot interpret pd.DataFrame without edge index')
        else:
            # Dangerous if item length > 3
            for item in items:
                args = []
                if len(item) == 1:
                    # Case where items=[edge1, edge2]
                    if isinstance(item[0], int) or len(item[0]) != 2:
                        raise ValueError('invalid edge')
                if len(item) == 2:
                    # Case where items=[(edge1, state), (edge2, state)]
                    if ut.isiterable(item[0]):
                        edge = item[0]
                        args = item[1:]
                    else:
                        edge = item
                else:
                    raise ValueError('invalid edge')
                    # Case where items=[(u, v, state), (u, v, state)]
                if len(item) > 3:
                    raise ValueError('pass in data as a dataframe or '
                                     'use kwargs')
                infr.add_feedback(edge, *args, verbose=verbose, **kwargs)

    def edge_decision(infr, edge):
        r"""
        Gets a decision on an edge, either explicitly or implicitly

        CommandLine:
            python -m ibeis.algo.graph.core edge_decision

        Doctest:
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=1, p_incon=1)
            >>> decision = infr.edge_decision((1, 2))
            >>> print('decision = %r' % (decision,))
            >>> assert decision == POSTV
            >>> decision = infr.edge_decision((199, 299))
            >>> print('decision = %r' % (decision,))
            >>> assert decision == UNREV
        """
        evidence_decision = infr.get_edge_attr(edge, 'evidence_decision',
                                               on_missing='default',
                                               default=UNREV)
        meta_decision = infr.get_edge_attr(edge, 'meta_decision',
                                           on_missing='default', default=NULL)
        decision = _rectify_decision(evidence_decision, meta_decision)
        return decision

    def add_node_feedback(infr, aid, **attrs):
        infr.print('Writing annot aid=%r %s' % (aid, ut.repr2(attrs)))
        ibs = infr.ibs
        ibs.set_annot_quality_texts([aid], [attrs['quality_texts']])
        ibs.set_annot_yaw_texts([aid], [attrs['yaw_texts']])
        ibs.overwrite_annot_case_tags([aid], [attrs['case_tags']])
        ibs.set_annot_multiple([aid], [attrs['multiple']])

    @profile
    def add_feedback(infr, edge, evidence_decision=None, tags=None,
                     user_id=None, meta_decision=None, confidence=None,
                     timestamp_c1=None, timestamp_c2=None, timestamp_s1=None,
                     timestamp=None, verbose=None, priority=None):
        r"""
        Doctest:
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback((5, 6), POSTV)
            >>> infr.add_feedback((5, 6), NEGTV, tags=['photobomb'])
            >>> infr.add_feedback((1, 2), INCMP)
            >>> print(ut.repr2(infr.internal_feedback, nl=2))
            >>> assert len(infr.external_feedback) == 0
            >>> assert len(infr.internal_feedback) == 2
            >>> assert len(infr.internal_feedback[(5, 6)]) == 2
            >>> assert len(infr.internal_feedback[(1, 2)]) == 1
        """
        prev_verbose = infr.verbose
        if verbose is not None:
            infr.verbose = verbose
        edge = aid1, aid2 = nxu.e_(*edge)

        if not infr.has_edge(edge):
            if True:
                # Allow new aids
                if not infr.graph.has_node(aid1):
                    infr.add_aids([aid1])
                if not infr.graph.has_node(aid2):
                    infr.add_aids([aid2])
            infr._check_edge(edge)
            infr.graph.add_edge(aid1, aid2)

        if evidence_decision is None:
            evidence_decision = UNREV

        msg = 'add_feedback ({}, {}), '.format(aid1, aid2)
        loc = locals()
        msg += ', '.join([
            str(val)
            # key + '=' + str(val)
            for key, val in (
                (key, loc[key])
                for key in ['evidence_decision', 'tags', 'user_id',
                            'confidence', 'meta_decision'])
            if val is not None
        ])
        infr.print(msg, 2, color='white')

        decision = _rectify_decision(evidence_decision, meta_decision)

        if decision == UNREV:
            # Unreviewing an edge deletes anything not yet committed
            if edge in infr.external_feedback:
                raise ValueError('External edge reviews cannot be undone')
            if edge in infr.internal_feedback:
                del infr.internal_feedback[edge]

        # Remove the edge from the queue if it is in there.
        if infr.queue:
            if edge in infr.queue:
                del infr.queue[edge]

        # Keep track of sequential reviews and set properties on global graph
        num_reviews = infr.get_edge_attr(edge, 'num_reviews', default=0)

        if timestamp is None:
            timestamp = ut.get_timestamp('int', isutc=True)
        if meta_decision is None:
            meta_decision = const.META_DECISION.CODE.NULL
        if confidence is None:
            confidence = const.CONFIDENCE.CODE.UNKNOWN

        review_id = next(infr.review_counter)
        feedback_item = {
            'tags': tags,
            'evidence_decision': evidence_decision,
            'meta_decision': meta_decision,
            'timestamp_c1': timestamp_c1,
            'timestamp_c2': timestamp_c2,
            'timestamp_s1': timestamp_s1,
            'timestamp': timestamp,
            'confidence': confidence,
            'user_id': user_id,
            'num_reviews': num_reviews + 1,
            'review_id': review_id,
        }
        infr.internal_feedback[edge].append(feedback_item)
        infr.set_edge_attr(edge, feedback_item)

        if infr.test_mode:
            infr._dynamic_test_callback(edge, decision, user_id)

        if infr.params['inference.enabled']:
            assert infr.dirty is False, (
                'need to recompute before dynamic inference continues')
            # Update priority queue based on the new edge
            action = infr.add_review_edge(edge, decision)
            if infr.test_mode:
                infr.test_state['action'] = action
            if False:
                infr._print_debug_ccs()
        else:
            action = None
            infr.dirty = True
            infr._add_review_edge(edge, decision)

        if infr.refresh and infr.params['inference.enabled']:
            # only add to criteria if this wasn't requested as a fix edge
            if priority is not None and priority <= 1.0:
                meaningful = bool({'merge', 'split'} & set(action))
                infr.refresh.add(meaningful, user_id, decision)

        if infr.test_mode:
            infr.metrics_list.append(infr.measure_metrics())

        infr.verbose = prev_verbose

    def _print_debug_ccs(infr):
        assert all([ut.allsame(infr.node_labels(*cc))
                    for cc in infr.positive_components()])
        sorted_ccs = sorted([
            set(cc) for cc in infr.pos_graph.connected_components()
        ])
        msg = '[' + ', '.join([
            repr(cc)
            if infr.is_consistent(cc) else
            ut.highlight_text(repr(cc), 'red')
            for cc in sorted_ccs]) + ']'
        print(msg)

    @ut.classproperty
    def feedback_keys(Infr):
        """ edge attribute keys used for feedback """
        return Infr.feedback_data_keys + ['num_reviews', 'review_id']

    @ut.classproperty
    def feedback_data_keys(Infr):
        """ edge attribute keys used for feedback """
        return [
            'evidence_decision', 'tags', 'user_id',
            'meta_decision', 'timestamp_c1', 'timestamp_c2',
            'timestamp_s1', 'timestamp', 'confidence'
        ]

    @profile
    def apply_feedback_edges(infr):
        r"""
        Transforms the feedback dictionaries into nx graph edge attributes

        CommandLine:
            python -m ibeis.algo.graph.core apply_feedback_edges

        Doctest:
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.reset_feedback()
            >>> infr.params['inference.enabled'] = False
            >>> #infr.add_feedback((1, 2), 'unknown', tags=[])
            >>> infr.add_feedback((1, 2), INCMP, tags=[])
            >>> infr.apply_feedback_edges()
            >>> print('edges = ' + ut.repr4(infr.graph.edge))
            >>> result = str(infr)
            >>> print(result)
            <AnnotInference(nNodes=6, nEdges=3, nCCs=4)>
        """
        infr.print('apply_feedback_edges', 1)
        # Transforms dictionary feedback into numpy array
        edges = []
        attr_lists = {key: [] for key in infr.feedback_keys}
        for edge, vals in infr.all_feedback_items():
            # hack for feedback rectification
            feedback_item = infr._rectify_feedback_item(vals)
            feedback_item['review_id'] = next(infr.review_counter)
            feedback_item['num_reviews'] = len(vals)
            # if feedback_item['decision'] == 'unknown':
            #     continue
            set1 = set(feedback_item.keys())
            set2 = set(attr_lists.keys())
            if set1 != set2:
                raise AssertionError(
                    'Bad feedback keys: ' +
                    ut.repr2(ut.set_overlap_items(set1, set2, 'got', 'want'), nl=1)
                    # ut.repr2(sorted(feedback_item.keys()), sv=True) + ' ' +
                    # ut.repr2(sorted(attr_lists.keys()), sv=True)
                )
            for key, val in feedback_item.items():
                attr_lists[key].append(val)
            edges.append(edge)

        assert ut.allsame(list(map(len, attr_lists.values())))
        assert len(edges) == len(next(iter(attr_lists.values())))

        # Put pair orders in context of the graph
        infr.print('_set_feedback_edges(nEdges=%d)' % (len(edges),), 3)
        # Ensure edges exist
        for edge in edges:
            if not infr.graph.has_edge(*edge):
                infr.graph.add_edge(*edge)

        # take evidence_decision and meta_decision into account
        decisions = [
            _rectify_decision(ed, md) for ed, md in
            zip(attr_lists['evidence_decision'], attr_lists['meta_decision'])
        ]

        for state, es in ut.group_items(edges, decisions).items():
            infr._add_review_edges_from(es, state)

        for key, val_list in attr_lists.items():
            infr.set_edge_attrs(key, ut.dzip(edges, val_list))

        if infr.params['inference.enabled']:
            infr.apply_nondynamic_update()

    def _rectify_feedback(infr, feedback):
        return {edge: infr._rectify_feedback_item(vals)
                for edge, vals in feedback.items()}

    def _rectify_feedback_item(infr, vals):
        """ uses most recently use strategy """
        return vals[-1]

    def all_feedback_items(infr):
        for edge, vals in six.iteritems(infr.external_feedback):
            yield edge, vals
        for edge, vals in six.iteritems(infr.internal_feedback):
            yield edge, vals

    def all_feedback(infr):
        all_feedback = ut.ddict(list)
        all_feedback.update(infr.all_feedback_items())
        return all_feedback

    def clear_feedback(infr, edges=None):
        """ Delete all edges properties related to feedback """
        if edges is None:
            edges = infr.graph.edges()
        edges = list(edges)
        infr.print('clear_feedback len(edges) = %r' % (len(edges)), 2)
        infr.external_feedback = ut.ddict(list)
        infr.internal_feedback = ut.ddict(list)

        # Kill all feedback, remote edge labels, but leave graph edges alone
        keys = infr.feedback_keys + ['inferred_state']
        ut.nx_delete_edge_attr(infr.graph, keys, edges)

        # Move reviewed edges back into the unreviewed graph
        for key in (POSTV, NEGTV, INCMP):
            subgraph = infr.review_graphs[key]
            prev_edges = ut.compress(edges, list(subgraph.has_edges(edges)))
            subgraph.remove_edges_from(prev_edges)
            infr.review_graphs[UNREV].add_edges_from(prev_edges)

        infr.pos_redun_nids.clear()
        infr.neg_redun_nids.clear()
        infr.nid_to_errors.clear()

        if __debug__:
            infr.assert_disjoint_invariant()

    def clear_edges(infr):
        """
        Removes all edges from the graph and the feedback dictionaries.
        """
        for graph in infr.review_graphs.values():
            graph.remove_edges_from(list(graph.edges()))
        infr.graph.remove_edges_from(list(infr.graph.edges()))
        infr.pos_redun_nids.clear()
        infr.neg_redun_nids.clear()
        infr.nid_to_errors.clear()

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
        # stage_fb = infr.read_ibeis_staging_feedback()
        # match_fb = infr.read_ibeis_annotmatch_feedback()
        # set(match_fb.keys()) - set(stage_fb.keys())
        # set(stage_fb.keys()) == set(match_fb.keys())

    def needs_conversion(infr):
        # not sure what the criteria is exactly. probably depricate
        num_names = len(set(infr.get_node_attrs('name_label').values()))
        num_pccs = infr.pos_graph.number_of_components()
        return num_pccs == 0 and num_names > 0

    def reset_feedback(infr, mode='annotmatch', apply=True):
        """ Resets feedback edges to state of the SQL annotmatch table """
        infr.print('reset_feedback mode=%r' % (mode,), 1)
        infr.clear_feedback()
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
        infr.clear_edges()
        infr.clear_feedback()
        if state == 'empty':
            # Remove all edges, and component names
            infr.clear_name_labels()
        elif state == 'orig':
            raise NotImplementedError('unused')
            infr.reset_name_labels()
        else:
            raise ValueError('Unknown state=%r' % (state,))

    def reset_name_labels(infr):
        """ Resets all annotation node name labels to their initial values """
        infr.print('reset_name_labels', 1)
        orig_names = infr.get_node_attrs('orig_name_label')
        infr.set_node_attrs('name_label', orig_names)

    def clear_name_labels(infr):
        """ Sets all annotation node name labels to be unknown """
        infr.print('clear_name_labels()', 1)
        # make distinct names for all nodes
        distinct_names = {
            node: -aid for node, aid in infr.get_node_attrs('aid').items()
        }
        infr.set_node_attrs('name_label', distinct_names)


class NameRelabel(object):

    def node_label(infr, aid):
        return infr.pos_graph.node_label(aid)

    def node_labels(infr, *aids):
        return infr.pos_graph.node_labels(*aids)

    def _next_nid(infr):
        if getattr(infr, 'nid_counter', None) is None:
            nids = nx.get_node_attributes(infr.graph, 'name_label')
            infr.nid_counter = max(nids)
        infr.nid_counter += 1
        new_nid = infr.nid_counter
        return new_nid

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
        Relabels nodes in graph based on positive connected components

        This will change all of the names on the nodes to be consistent while
        preserving any existing names as best as possible. If rectify=False,
        this will be faster, but the old names will not be preserved and each
        PCC will be assigned an arbitrary name.

        Note:
            if something messes up you can call infr.reset_labels_to_ibeis() to
            reset node labels to their original values --- this will almost
            always put the graph in an inconsistent state --- but then you can
            this with rectify=True to fix everything up.

        Args:
            graph (nx.Graph, optional): only edges in `graph` are relabeled
                defaults to current graph.
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
                    infr.node_label(next(iter(subgraph.nodes())))
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
            python -m ibeis.algo.graph.core connected_component_status

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback_from([(2, 3), NEGTV) (5, 6), NEGTV)
            >>>                         (1, 2), POSTV)]
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


class MiscHelpers(object):

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
            python -m ibeis.algo.graph.core add_aids --show

        Doctest:
            >>> from ibeis.algo.graph.core import *  # NOQA
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

            if infr.graph is not None:
                infr.graph.add_nodes_from(aids)
                for subgraph in infr.review_graphs.values():
                    subgraph.add_nodes_from(aids)

    def update_node_attributes(infr, aids=None, nids=None):
        if aids is None:
            aids = infr.aids
            nids = infr.orig_name_labels
        assert aids is not None, 'must have aids'
        assert nids is not None, 'must have nids'
        node_to_aid = {aid: aid for aid in aids}
        node_to_nid = {aid: nid for aid, nid in zip(aids, nids)}
        ut.assert_eq_len(node_to_nid, node_to_aid)

        infr.graph.add_nodes_from(aids)
        for subgraph in infr.review_graphs.values():
            subgraph.add_nodes_from(aids)

        infr.set_node_attrs('aid', node_to_aid)
        infr.set_node_attrs('name_label', node_to_nid)
        infr.set_node_attrs('orig_name_label', node_to_nid)
        # TODO: depricate these, they will always be identity I think

    def initialize_graph(infr, graph=None):
        infr.print('initialize_graph', 1)
        if graph is None:
            infr.graph = infr._graph_cls()
        else:
            infr.graph = graph

        infr.review_graphs[POSTV] = nx_dynamic_graph.DynConnGraph()
        infr.review_graphs[NEGTV] = infr._graph_cls()
        infr.review_graphs[INCMP] = infr._graph_cls()
        infr.review_graphs[UNKWN] = infr._graph_cls()
        infr.review_graphs[UNREV] = infr._graph_cls()

        if graph is not None:
            for u, v, d in graph.edges(data=True):
                evidence_decision = d.get('evidence_decision', UNREV)
                meta_decision = d.get('meta_decision', NULL)
                decision = _rectify_decision(evidence_decision, meta_decision)
                if decision in {POSTV, NEGTV, INCMP, UNREV, UNKWN}:
                    infr.review_graphs[decision].add_edge(u, v)
                else:
                    raise ValueError('Unknown decision=%r' % (decision,))

        infr.update_node_attributes()

    @profile
    def log_message(infr, msg, level=1, color=None):
        if color is None:
            color = 'blue'
        if True:
            infr.logs.append((msg, color))
            if len(infr.logs) == infr.logs.maxlen:
                infr.log_index = max(infr.log_index - 1, 0)
        if infr.verbose >= level:
            ut.cprint('[infr] ' + msg, color)

    print = log_message

    def latest_logs(infr):
        index = infr.log_index
        infr.log_index = len(infr.logs)
        return [infr.logs[x][0] for x in range(index, len(infr.logs))]

    def dump_logs(infr):
        print('--- <LOG DUMP> ---')
        for msg, color in infr.logs:
            ut.cprint('[infr] ' + msg, color)
        print('--- <\LOG DUMP> ---')


class AltConstructors(object):
    _graph_cls = nx_dynamic_graph.NiceGraph
    # _graph_cls = nx.Graph
    # nx.Graph
    # _graph_cls = nx.DiGraph

    @classmethod
    def from_pairs(AnnotInference, aid_pairs, attrs=None, ibs=None, verbose=False):
        import networkx as nx
        G = AnnotInference._graph_cls()
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
        if ibs is not None:
            nids = None
        else:
            nids = [-a for a in aids]
        infr = AnnotInference(ibs, aids, nids, autoinit=False,
                              verbose=verbose)
        infr.initialize_graph(graph=G)
        # hack
        orig_name_labels = [infr.pos_graph.node_label(a) for a in aids]
        infr.orig_name_labels = orig_name_labels
        infr.set_node_attrs('orig_name_label',
                            ut.dzip(aids, orig_name_labels))
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

    def status(infr, extended=False):
        status_dict = ut.odict([
            ('nNodes', len(infr.aids)),
            ('nEdges', infr.graph.number_of_edges()),
            ('nCCs', infr.pos_graph.number_of_components()),
            ('nPostvEdges', infr.pos_graph.number_of_edges()),
            ('nNegtvEdges', infr.neg_graph.number_of_edges()),
            ('nIncmpEdges', infr.incomp_graph.number_of_edges()),
            ('nUnrevEdges', infr.unreviewed_graph.number_of_edges()),
            ('nPosRedunCCs', len(infr.pos_redun_nids)),
            ('nNegRedunPairs', infr.neg_redun_nids.number_of_edges()),
            ('nInconsistentCCs', len(infr.nid_to_errors)),
            #('nUnkwnEdges', infr.unknown_graph.number_of_edges()),
        ])
        if extended:
            def count_within_between(edges):
                n_within = 0
                n_between = 0
                for u, v in edges:
                    nid1, nid2 = infr.pos_graph.node_labels(u, v)
                    if nid1 == nid2:
                        n_within += 1
                    else:
                        n_between += 1
                return n_within, n_between

            a, b = count_within_between(infr.neg_graph.edges())
            status_dict['nNegEdgesWithin'] = a
            status_dict['nNegEdgesBetween'] = b

            a, b = count_within_between(infr.incomp_graph.edges())
            status_dict['nIncompEdgesWithin'] = a
            status_dict['nIncompEdgesBetween'] = b

            a, b = count_within_between(infr.unreviewed_graph.edges())
            status_dict['nUnrevEdgesWithin'] = a
            status_dict['nUrevEdgesBetween'] = b

        return status_dict

    def __nice__(infr):
        if infr.graph is None:
            return 'nAids=%r, G=None' % (len(infr.aids))
        else:
            fmt = 'nNodes={}, nEdges={}, nCCs={}'
            msg = fmt.format(
                len(infr.aids),
                infr.graph.number_of_edges(),
                infr.pos_graph.number_of_components(),
                # infr.incomp_graph.number_of_edges(),
                # infr.unreviewed_graph.number_of_edges(),
            )
            return msg
            # return 'nAids={}, nEdges={}, nCCs={}'.format(
            #     len(infr.aids),
            #     infr.graph.number_of_edges(),
            #     infr.pos_graph.number_of_components()
            # )


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInference(ut.NiceRepr,
                     # Old internal stuffs
                     AltConstructors,
                     MiscHelpers,
                     Feedback,
                     NameRelabel,
                     # New core algorithm stuffs
                     mixin_dynamic.NonDynamicUpdate,
                     mixin_dynamic.Recovery,
                     mixin_dynamic.Consistency,
                     mixin_dynamic.Redundancy,
                     mixin_dynamic.Priority,
                     mixin_dynamic.DynamicUpdate,
                     mixin_matching.CandidateSearch,
                     mixin_matching.InfrLearning,
                     mixin_matching.AnnotInfrMatching,
                     # General helpers
                     mixin_helpers.AssertInvariants,
                     mixin_helpers.DummyEdges,
                     mixin_helpers.Convenience,
                     mixin_helpers.AttrAccess,
                     # Algorithm loops and simulation
                     mixin_loops.SimulationHelpers,
                     mixin_loops.InfrReviewers,
                     mixin_loops.InfrLoops,
                     # Visualization
                     mixin_viz.GraphVisualization,
                     # plugging into IBEIS
                     mixin_groundtruth.Groundtruth,
                     mixin_ibeis.IBEISIO,
                     mixin_ibeis.IBEISGroundtruth,
                     # _dep_mixins._AnnotInfrDepMixin,
                     ):
    """
    class for maintaining state of an identification

    Terminology and Concepts:

    CommandLine:
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7
        ibeis AnnotInference:0 --show
        ibeis AnnotInference:1 --show
        ibeis AnnotInference:2 --show

    Doctest:
        >>> from ibeis.algo.graph.core import *  # NOQA
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
        infr = <AnnotInference(nNodes=6, nEdges=0, nCCs=6)>

    Example:
        >>> # SCRIPT
        >>> from ibeis.algo.graph.core import *  # NOQA
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
        >>> infr.ensure_mst()
        >>> infr.show_graph(use_image=use_image)
        >>> # Add some feedback
        >>> infr.add_feedback((1, 4), NEGTV)
        >>> infr.apply_feedback_edges()
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()

    Example:
        >>> # SCRIPT
        >>> from ibeis.algo.graph.core import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aids = [1, 2, 3, 4, 5, 6, 7, 9]
        >>> infr = AnnotInference(ibs, aids, autoinit=True)
        >>> result = ('infr = %s' % (infr,))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> use_image = False
        >>> infr.initialize_visual_node_attrs()
        >>> infr.ensure_mst()
        >>> # Add some feedback
        >>> infr.add_feedback((1, 4), NEGTV)
        >>> try:
        >>>     infr.add_feedback((1, 10), NEGTV)
        >>> except ValueError:
        >>>     pass
        >>> try:
        >>>     infr.add_feedback((11, 12), NEGTV)
        >>> except ValueError:
        >>>     pass
        >>> infr.apply_feedback_edges()
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()

    Ignore:
        >>> import ibeis
        >>> import utool as ut
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> infr = ibeis.AnnotInference(ibs, 'all')
        >>> class_ = infr
        >>> fpath = None
        >>> static_attrs = ut.check_static_member_vars(class_, fpath)
        >>> uninitialized = set(infr.__dict__.keys()) - set(static_attrs)

    """

    def __init__(infr, ibs, aids=[], nids=None, autoinit=True, verbose=False):
        """
        Ignore:
            pass
        """
        # infr.verbose = verbose
        infr.name = None
        infr.verbose = verbose

        # setup logging
        infr.logs = collections.deque(maxlen=10000)
        infr.log_index = 0

        # If not dirty, new feedback should dynamically maintain a consistent
        # state. If dirty it means we need to recompute connected compoments
        # before we can continue with dynamic review.
        infr.dirty = False
        infr.readonly = False

        # ibeis controller and initial nodes
        # TODO: aids can be abstracted as a property that simply looks at the
        # nodes in infr.graph.
        if isinstance(ibs, six.string_types):
            import ibeis
            ibs = ibeis.opendb(ibs)
        infr.ibs = ibs
        infr.aids = None
        infr.aids_set = None
        infr.orig_name_labels = None

        # Underlying graph structure
        infr.graph = None
        infr.review_graphs = {
            POSTV: None,
            NEGTV: None,
            INCMP: None,
            UNKWN: None,
            UNREV: None,
        }

        # Criterion
        infr.queue = None
        infr.refresh = None

        infr.review_counter = it.count(0)
        infr.nid_counter = None

        # Dynamic Properties (requires bookkeeping)
        infr.nid_to_errors = {}
        infr.recovery_ccs = []

        # Recover graph holds positive edges of inconsistent PCCs
        infr.recover_graph = nx_dynamic_graph.DynConnGraph()
        # Set of PCCs that are positive redundant
        infr.pos_redun_nids = set([])
        # Represents the metagraph of negative edges between PCCs
        infr.neg_redun_nids = infr._graph_cls()

        # This should represent The feedback read from a database. We do not
        # need to do any updates to an external database based on this data.
        infr.external_feedback = ut.ddict(list)
        # Feedback that has not been synced with the external database.
        # Once we sync, this is merged into external feedback.
        infr.internal_feedback = ut.ddict(list)

        # Bookkeeping
        infr.edge_truth = {}
        infr.task_probs = ut.ddict(dict)

        # Computer vision algorithms
        infr.ranker = None
        infr.verifiers = None

        infr.task_thresh = {
            'match_state': {
                POSTV: np.inf,
                NEGTV: np.inf,
                INCMP: np.inf,
            },
            'photobomb_state': {
                'pb': np.inf,
                'nopb': np.inf,
            }
        }

        # Parameters / Configurations / Callbacks
        infr.callbacks = {
            'request_review': None,
            'review_ready': None,
            'review_finished': None,
        }

        infr.params = {
            'manual.n_peek': 1,
            'manual.autosave': True,

            'algo.max_outer_loops': None,

            # Dynamic Inference
            'inference.enabled': True,
            'inference.update_attrs': True,

            # Termination / Refresh
            'refresh.window': 20,
            'refresh.patience': 72,
            'refresh.thresh': .1,

            # Redundancy
            # if redun.enabled is True, then redundant edges will be ignored by
            # # the priority queue and extra edges needed to achieve minimum
            # redundancy will be searched for if the queue is empty.
            'redun.enabled': True,
            'redun.pos': 2,  # positive-k
            'redun.neg': 2,  # negative-k
            'redun.enforce_pos': True,  # does positive augmentation

            # Autoreviewer params
            'autoreview.enabled': True,
            'autoreview.prioritize_nonpos': True,
        }

        infr._viz_image_config = {
            'in_image': False,
            'thumbsize': 221,
        }

        infr.verifier_params = {}  # TODO
        infr.ranker_params = {}

        # Developer modes (consoldate this)
        infr.test_mode = False
        infr.simulation_mode = False

        # Testing state
        infr.metrics_list = None
        infr.test_state = None
        infr.test_gt_pos_graph = None
        infr.nid_to_gt_cc = None
        infr.node_truth = None
        infr.real_n_pcc_mst_edges = None

        # External: Can we remove these?
        infr.cm_list = None
        infr.vsone_matches = {}
        infr.qreq_ = None
        infr.manual_wgt = None

        infr.print('__init__', level=1)
        if aids == 'all':
            aids = ibs.get_valid_aids()
        infr.add_aids(aids, nids)
        if autoinit:
            infr.initialize_graph()
            if isinstance(autoinit, six.string_types):
                infr.reset_feedback(autoinit)

    def subparams(infr, prefix):
        """
        Returns dict of params prefixed with <prefix>.
        The returned dict does not contain the prefix

        Doctest:
            >>> import ibeis
            >>> infr = ibeis.AnnotInference(None)
            >>> result = ut.repr2(infr.subparams('refresh'))
            >>> print(result)
            {'patience': 72, 'thresh': 0.1, 'window': 20}
        """
        prefix_ = prefix + '.'
        subparams = {k[len(prefix_):]: v for k, v in infr.params.items()
                     if k.startswith(prefix_)}
        return subparams

    def copy(infr):
        # shallow copy ibs
        infr2 = AnnotInference(
            infr.ibs, copy.deepcopy(infr.aids),
            copy.deepcopy(infr.orig_name_labels), autoinit=False,
            verbose=infr.verbose)
        # shallow copy classifiers
        infr2.classifiers = infr.verifiers

        infr2.graph = infr.graph.copy()
        infr2.external_feedback = copy.deepcopy(infr.external_feedback)
        infr2.internal_feedback = copy.deepcopy(infr.internal_feedback)
        infr2.cm_list = copy.deepcopy(infr.cm_list)
        infr2.qreq_ = copy.deepcopy(infr.qreq_)
        infr2.nid_counter = infr.nid_counter

        infr2.recover_graph = copy.deepcopy(infr.recover_graph)

        infr2.pos_redun_nids = copy.deepcopy(infr.pos_redun_nids)
        infr2.neg_redun_nids = copy.deepcopy(infr.neg_redun_nids)

        infr2.review_graphs = copy.deepcopy(infr.review_graphs)
        infr2.nid_to_errors = copy.deepcopy(infr.nid_to_errors)
        infr2.recovery_ccs = copy.deepcopy(infr.recovery_ccs)

        infr2.readonly = infr.readonly
        infr2.dirty = infr.dirty

        infr2.test_mode = infr.test_mode
        infr2.test_mode = infr.test_mode
        infr2.simulation_mode = infr.simulation_mode

        infr.queue = copy.deepcopy(infr.queue)

        infr.params = copy.deepcopy(infr.params)
        infr2._viz_image_config = infr._viz_image_config.copy()

        if infr.test_mode:
            infr2.test_state = copy.deepcopy(infr.test_state)
            infr2.metrics_list = copy.deepcopy(infr.metrics_list)
        return infr2

    def subgraph(infr, aids):
        """
        Makes a new inference object that is a subset of the original.

        Note, this is not robust, be careful. The subgraph should be treated as
        read only. Do not commit any reviews made from here.
        """
        orig_name_labels = list(infr.gen_node_values('orig_name_label', aids))
        infr2 = AnnotInference(infr.ibs, aids, orig_name_labels,
                               autoinit=False, verbose=infr.verbose)
        # deep copy the graph structure
        infr2.graph = infr.graph.subgraph(aids).copy()
        infr2.readonly = True
        infr2.classifiers = infr.verifiers

        infr.params = copy.deepcopy(infr.params)
        infr2._viz_image_config = infr._viz_image_config.copy()

        # infr2._viz_init_nodes = infr._viz_image_config
        # infr2._viz_image_config_dirty = infr._viz_image_config_dirty
        infr2.edge_truth = {
            e: infr.edge_truth[e] for e in infr2.graph.edges()
            if e in infr.edge_truth
        }

        # TODO: internal/external feedback

        infr2.nid_counter = infr.nid_counter
        infr2.dirty = True
        infr2.cm_list = None
        infr2.qreq_ = None

        # TODO:
        # infr2.nid_to_errors {}  # = copy.deepcopy(infr.nid_to_errors)
        # infr2.recover_graph = copy.deepcopy(infr.recover_graph)
        # infr2.pos_redun_nids = copy.deepcopy(infr.pos_redun_nids)
        # infr2.neg_redun_nids = copy.deepcopy(infr.neg_redun_nids)

        infr2.review_graphs = {}
        for k, g in infr.review_graphs.items():
            if g is None:
                infr2.review_graphs[k] = None
            elif k == POSTV:
                infr2.review_graphs[k] = g.subgraph(aids, dynamic=True)
            else:
                infr2.review_graphs[k] = g.subgraph(aids)
        return infr2

    def set_config(infr, config, **kw):
        pass


def testdata_infr(defaultdb='PZ_MTEST'):
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaultdb)
    aids = [1, 2, 3, 4, 5, 6]
    infr = AnnotInference(ibs, aids, autoinit=True)
    return infr


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show --aids=1,2,3,4,5,6,7 --graph --match=1,4 --nomatch=3,1,5,7
        python -m ibeis.algo.graph.core

        python -m ibeis.algo.graph all

        python -m ibeis.algo.graph.core --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
