# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import logging
import itertools as it
import six
from ibeis.algo.graph import nx_dynamic_graph
# from ibeis.algo.graph import _dep_mixins
from ibeis.algo.graph import mixin_viz
from ibeis.algo.graph import mixin_helpers
from ibeis.algo.graph import mixin_dynamic
from ibeis.algo.graph import mixin_loops
from ibeis.algo.graph import mixin_matching
from ibeis.algo.graph import mixin_ibeis
from ibeis.algo.graph.nx_utils import e_, _dz
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV
import networkx as nx
print, rrr, profile = ut.inject2(__name__)


DEBUG_CC = False
# DEBUG_CC = True


class Feedback(object):

    def _check_edge(infr, edge):
        aid1, aid2 = edge
        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))

    def add_feedback_from(infr, items):
        for item in items:
            infr.add_feedback(*item)

    @profile
    def add_feedback(infr, edge, decision, tags=None, user_id=None,
                     confidence=None, timestamp=None, verbose=None):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.add_feedback((5, 6), POSTV)
            >>> infr.add_feedback((5, 6), NEGTV, ['Photobomb'])
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
        edge = aid1, aid2 = e_(*edge)

        if not infr.has_edge(edge):
            if True:
                # Allow new aids
                if not infr.graph.has_node(aid1):
                    infr.add_aids([aid1])
                if not infr.graph.has_node(aid2):
                    infr.add_aids([aid2])
            infr._check_edge(edge)
            infr.graph.add_edge(aid1, aid2)

        if True:
            print('')

        msg = 'add_feedback {}, {} '.format(aid1, aid2)
        loc = locals()
        msg += ' '.join([
            key + '=' + str(val)
            for key, val in ((key, loc[key])
                             for key in 'decision tags user_id confidence'.split())
            if val is not None
        ])
        infr.print(
            msg,
            # 'add_feedback {}, {}, decision={}, tags={}, user_id={}, '
            # 'confidence={}'.format(
            #     aid1, aid2, decision, tags, user_id, confidence),
            1, color='white')

        if decision == UNREV:
            # raise NotImplementedError('not done yet')
            feedback_item = None
            if edge in infr.external_feedback:
                raise ValueError('External edge reviews cannot be undone')
            if edge in infr.internal_feedback:
                del infr.internal_feedback[edge]
            # for G in infr.review_graphs.values():
            #     if G.has_edge(*edge):
            #         G.remove_edge(*edge)

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
            assert infr.dirty is False, (
                'need to recompute before dynamic inference continues')
            # Update priority queue based on the new edge
            infr.add_review_edge(edge, decision)
            if False:
                infr._print_debug_ccs()
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

    @profile
    def apply_feedback_edges(infr, safe=True):
        r"""
        Transforms the feedback dictionaries into nx graph edge attributes

        CommandLine:
            python -m ibeis.algo.graph.core apply_feedback_edges

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.reset_feedback()
            >>> #infr.add_feedback((1, 2), 'unknown', tags=[])
            >>> infr.add_feedback((1, 2), INCMP, tags=[])
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
        infr.set_edge_attrs('tags', _dz(edges, tags_list))
        infr.set_edge_attrs('confidence', _dz(edges, confidence_list))
        infr.set_edge_attrs('user_id', _dz(edges, userid_list))
        infr.set_edge_attrs('num_reviews', _dz(edges, num_review_list))
        infr.set_edge_attrs('review_timestamp', _dz(edges, [timestamp]))

        if infr.enable_inference:
            infr.apply_nondynamic_update()

    def _del_feedback_edges(infr, edges=None):
        """ Delete all edges properties related to feedback """
        if edges is None:
            edges = list(infr.graph.edges())
        infr.print('_del_feedback_edges len(edges) = %r' % (len(edges)), 2)
        keys = ['decision', 'tags', 'num_reviews',
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
            python -m ibeis.algo.graph.core add_aids --show

        Example:
            >>> # ENABLE_DOCTEST
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

        infr.review_graphs[POSTV] = nx_dynamic_graph.DynConnGraph()
        infr.review_graphs[NEGTV] = infr._graph_cls()
        infr.review_graphs[INCMP] = infr._graph_cls()
        infr.review_graphs[UNREV] = infr._graph_cls()

        if graph is not None:
            for u, v, d in graph.edges(data=True):
                decision = d.get('decision', UNREV)
                if decision in {POSTV, NEGTV, INCMP, UNREV}:
                    infr.review_graphs[decision].add_edge(u, v)

        infr.update_node_attributes()

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


class AltConstructors(object):
    _graph_cls = nx_dynamic_graph.NiceGraph
    # _graph_cls = nx.Graph
    # nx.Graph
    # _graph_cls = nx.DiGraph

    @classmethod
    def from_pairs(AnnotInference, aid_pairs, attrs=None, ibs=None, verbose=False):
        # infr.graph = G
        # infr.update_node_attributes(G)
        # aids = set(ut.flatten(aid_pairs))
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
                     mixin_dynamic.Completeness,
                     mixin_dynamic.Priority,
                     mixin_dynamic.DynamicUpdate,
                     mixin_matching.CandidateSearch,
                     mixin_matching.InfrLearning,
                     mixin_matching.AnnotInfrMatching,
                     # General helpers
                     mixin_helpers.AssertInvariants,
                     mixin_helpers.DummyEdges,
                     mixin_helpers.AttrAccess,
                     # Algorithm loops and simulation
                     mixin_loops.SimulationHelpers,
                     mixin_loops.InfrReviewers,
                     mixin_loops.InfrLoops,
                     # Visualization
                     mixin_viz.GraphVisualization,
                     # plugging into IBEIS
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

    Example:
        >>> # ENABLE_DOCTEST
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
        infr = <AnnotInference(nAids=6, nEdges=0)>

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
        >>> infr.apply_mst()
        >>> infr.show_graph(use_image=use_image)
        >>> # Add some feedback
        >>> infr.add_feedback((1, 4), NEGTV)
        >>> infr.apply_feedback_edges()
        >>> infr.show_graph(use_image=use_image)
        >>> ut.show_if_requested()
        infr = <AnnotInference(nAids=6, nEdges=0)>

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
        >>> infr.apply_mst()
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
        infr = <AnnotInference(nAids=6, nEdges=0)>
    """

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
            POSTV: None,
            NEGTV: None,
            INCMP: None,
            UNREV: None,
        }

        infr.enable_inference = True
        infr.test_mode = False
        infr.edge_truth = {}

        # Criteria
        infr.refresh = None
        infr.term = None

        # Dynamic Properties (requires bookkeeping)
        infr.nid_to_errors = {}
        infr.recovery_ccs = []
        # TODO: keep one main recovery cc but once it is done pop the next one
        # from recovery_ccs until none are left
        infr.recovery_cc = None

        # graph holding positive edges of inconsistent PCCs
        # infr.recover_graph = infr._graph_cls()
        infr.recover_graph = nx_dynamic_graph.DynConnGraph()
        # infr._graph_cls()
        infr.recover_prev_neg_nids = None
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

        infr.thresh = None
        infr.cm_list = None
        infr.vsone_matches = {}
        infr.qreq_ = None
        infr.nid_counter = None
        infr.queue = None
        infr.queue_params = {
            'pos_redun': 1,
            'neg_redun': 1,
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

        infr2.recover_graph = copy.deepcopy(infr.recover_graph)
        infr2.recover_prev_neg_nids = copy.deepcopy(infr.recover_prev_neg_nids)

        infr2.pos_redun_nids = copy.deepcopy(infr.pos_redun_nids)
        infr2.neg_redun_nids = copy.deepcopy(infr.neg_redun_nids)

        infr2.review_graphs = copy.deepcopy(infr.review_graphs)
        infr2.nid_to_errors = copy.deepcopy(infr.nid_to_errors)
        return infr2


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
            infr.add_feedback((a, b), POSTV)

    for name1, name2 in it.combinations(names[4:], 2):
        infr.add_feedback((name1.aids[0], name2.aids[0]), NEGTV)
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
        python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show --aids=1,2,3,4,5,6,7 --graph --match=1,4 --nomatch=3,1,5,7
        python -m ibeis.algo.graph.core

        python -m ibeis.algo.graph all

        python -m ibeis.algo.graph.core --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
