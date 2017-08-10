# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import itertools as it
import networkx as nx
from ibeis import constants as const
from ibeis.algo.graph import nx_utils as nxu
from ibeis.algo.graph.state import (POSTV, NEGTV, INCMP, UNREV, UNKWN,
                                    UNINFERABLE)
from ibeis.algo.graph.state import (SAME, DIFF, NULL)  # NOQA
from ibeis.algo.graph.nx_utils import (edges_inside, edges_cross,
                                       edges_outgoing)
print, rrr, profile = ut.inject2(__name__)

DECISION_LEVEL = 4


class DynamicUpdate(object):
    """
    # 12 total possible states

    # details of these states.
    POSITIVE, WITHIN, CONSISTENT
        * pos-within never changes PCC status
        * never introduces inconsistency
        * might add pos-redun
    POSITIVE, WITHIN, INCONSISTENT
        * pos-within never changes PCC status
        * might fix inconsistent edge
    POSITIVE, BETWEEN, BOTH_CONSISTENT
        * pos-between edge always does merge
    POSITIVE, BETWEEN, ANY_INCONSISTENT
        * pos-between edge always does merge
        * pos-between never fixes inconsistency

    NEGATIVE, WITHIN, CONSISTENT
        * might split PCC, results will be consistent
        * might causes an inconsistency
    NEGATIVE, WITHIN, INCONSISTENT
        * might split PCC, results may be inconsistent
    NEGATIVE, BETWEEN, BOTH_CONSISTENT
        * might add neg-redun
    NEGATIVE, BETWEEN, ANY_INCONSISTENT
        * might add to incon-neg-external
        * neg-redun not tracked for incon.

    UNINFERABLE, WITHIN, CONSISTENT
        * might remove pos-redun
        * might split PCC, results will be consistent
    UNINFERABLE, WITHIN, INCONSISTENT
        * might split PCC, results may be inconsistent
    UNINFERABLE, BETWEEN, BOTH_CONSISTENT
        * might remove neg-redun
    UNINFERABLE, BETWEEN, ANY_INCONSISTENT
        * might remove incon-neg-external
    """

    def ensure_edges_from(infr, edges):
        """
        Finds edges that don't exist and adds them as unreviwed edges.
        Returns new edges that were added.
        """
        edges = list(edges)
        # Only add edges that don't exist
        new_edges = [e for e in edges if not infr.has_edge(e)]
        infr.graph.add_edges_from(new_edges,
                                  evidence_decision=UNREV,
                                  meta_decision=UNREV,
                                  decision=UNREV,
                                  num_reviews=0)
        # No inference is needed by expliclty creating unreviewed edges that
        # already implicitly existsed.
        infr._add_review_edges_from(new_edges, decision=UNREV)
        return new_edges

    @profile
    def add_review_edge(infr, edge, decision):
        """
        Adds edge to the dynamically connected graphs and updates dynamically
        inferrable edge attributes.
        """
        if decision == POSTV:
            action = infr._positive_decision(edge)
        elif decision == NEGTV:
            action = infr._negative_decision(edge)
        elif decision in UNINFERABLE:
            # incomparable and unreview have the same inference structure
            action = infr._uninferable_decision(edge, decision)
        else:
            raise AssertionError('Unknown decision=%r' % (decision,))
        return action

    def _add_review_edges_from(infr, edges, decision=UNREV):
        infr.print('add {} edges decision={}'.format(len(edges), decision), 1)
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edges_from(edges)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                G.remove_edges_from(edges)

    def _add_review_edge(infr, edge, decision):
        # infr.print('add review edge=%r, decision=%r' % (edge, decision), 20)
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edge(*edge)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                if G.has_edge(*edge):
                    G.remove_edge(*edge)

    @profile
    def _dynamic_test_callback(infr, edge, decision, user_id):
        was_gt_pos = infr.test_gt_pos_graph.has_edge(*edge)

        old_decision = infr.get_edge_attr(edge, 'decision', default=UNREV)
        true_decision = infr.edge_truth[edge]

        was_within_pred = infr.pos_graph.are_nodes_connected(*edge)
        was_within_gt = infr.test_gt_pos_graph.are_nodes_connected(*edge)
        was_reviewed = old_decision != UNREV
        is_within_gt = was_within_gt
        was_correct = old_decision == true_decision

        is_correct = true_decision == decision

        test_print = ut.partial(infr.print, level=2)
        def test_print(x, **kw):
            infr.print('[ACTION] ' + x, level=2, **kw)
        # test_print = lambda *a, **kw: None  # NOQA

        if 0:
            num = infr.recover_graph.number_of_components()
            old_data = infr.get_nonvisual_edge_data(edge)
            # print('old_data = %s' % (ut.repr4(old_data, stritems=True),))
            print('n_prev_reviews = %r' % (old_data['num_reviews'],))
            print('old_decision = %r' % (old_decision,))
            print('decision = %r' % (decision,))
            print('was_gt_pos = %r' % (was_gt_pos,))
            print('was_within_pred = %r' % (was_within_pred,))
            print('was_within_gt = %r' % (was_within_gt,))
            print('num inconsistent = %r' % (num,))
            # is_recovering = infr.is_recovering()

        if decision == POSTV:
            if is_correct:
                if not was_gt_pos:
                    infr.test_gt_pos_graph.add_edge(*edge)
        elif was_gt_pos:
            test_print("UNDID GOOD POSITIVE EDGE", color='darkred')
            infr.test_gt_pos_graph.remove_edge(*edge)
            is_within_gt = infr.test_gt_pos_graph.are_nodes_connected(*edge)

        split_gt = is_within_gt != was_within_gt
        if split_gt:
            test_print("SPLIT A GOOD MERGE", color='darkred')
            infr.test_state['n_true_merges'] -= 1

        confusion = infr.test_state['confusion']
        if confusion is None:
            # initialize dynamic confusion matrix
            # import pandas as pd
            states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)
            confusion = {r: {c: 0 for c in states} for r in states}
            # pandas takes a really long time doing this
            # confusion = pd.DataFrame(columns=states, index=states)
            # confusion[:] = 0
            # confusion.index.name = 'real'
            # confusion.columns.name = 'pred'
            infr.test_state['confusion'] = confusion

        if was_reviewed:
            confusion[true_decision][old_decision] -= 1
            confusion[true_decision][decision] += 1
        else:
            confusion[true_decision][decision] += 1

        test_action = None
        action_color = None

        if is_correct:
            # CORRECT DECISION
            if was_reviewed:
                if old_decision == decision:
                    test_action = 'correct duplicate'
                    action_color = 'darkyellow'
                else:
                    test_action = 'correction'
                    action_color = 'darkgreen'
                    if decision == POSTV:
                        if not was_within_gt:
                            test_action = 'correction redid merge'
                            action_color = 'darkgreen'
                            infr.test_state['n_true_merges'] += 1
            else:
                if decision == POSTV:
                    if not was_within_gt:
                        test_action = 'correct merge'
                        action_color = 'darkgreen'
                        infr.test_state['n_true_merges'] += 1
                    else:
                        test_action = 'correct redundant positive'
                        action_color = 'darkblue'
                else:
                    if decision == NEGTV:
                        test_action = 'correct negative'
                        action_color = 'teal'
                    else:
                        test_action = 'correct uninferrable'
                        action_color = 'teal'
        else:
            action_color = 'darkred'
            # INCORRECT DECISION
            if was_reviewed:
                if old_decision == decision:
                    test_action = 'incorrect duplicate'
                elif was_correct:
                    test_action = 'incorrect undid good edge'
            else:
                if decision == POSTV:
                    if was_within_pred:
                        test_action = 'incorrect redundant merge'
                    else:
                        test_action = 'incorrect new merge'
                else:
                    test_action = 'incorrect new mistake'

        infr.test_state['test_action'] = test_action
        infr.test_state['pred_decision'] = decision
        infr.test_state['true_decision'] = true_decision
        infr.test_state['user_id'] = user_id
        infr.test_state['recovering'] = (infr.recover_graph.has_node(edge[0]) or
                                         infr.recover_graph.has_node(edge[1]))

        infr.test_state['n_decision'] += 1
        if user_id.startswith('auto'):
            infr.test_state['n_auto'] += 1
        elif user_id == 'oracle':
            infr.test_state['n_manual'] += 1
        else:
            raise AssertionError('unknown user_id=%r' % (user_id,))

        test_print(test_action, color=action_color)
        assert test_action is not None, 'what happened?'

    def on_between(infr, edge, decision, nid1, nid2, merge_nid=None):
        """
        Callback when a review is made between two PCCs
        """
        if merge_nid is not None:
            # A merge occurred
            if infr.params['inference.update_attrs']:
                cc = infr.pos_graph.component(merge_nid)
                infr.set_node_attrs('name_label', ut.dzip(cc, [merge_nid]))
            # FIXME: this state is ugly
            action = ['merge']
        else:
            action = []
        return action

    def on_within(infr, edge, decision, nid, split_nids=None):
        """
        Callback when a review is made inside a PCC
        """
        if split_nids is not None:
            # A split occurred
            if infr.params['inference.update_attrs']:
                new_nid1, new_nid2 = split_nids
                cc1 = infr.pos_graph.component(new_nid1)
                cc2 = infr.pos_graph.component(new_nid2)
                infr.set_node_attrs('name_label', ut.dzip(cc1, [new_nid1]))
                infr.set_node_attrs('name_label', ut.dzip(cc2, [new_nid2]))
            action = ['split']
        else:
            action = []
        return action

    @profile
    def _positive_decision(infr, edge):
        r"""
        Logic for a dynamic positive decision.  A positive decision is evidence
        that two annots should be in the same PCC

        Note, this could be an incomparable edge, but with a meta_decision of
        same.

        Ignore:
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> kwargs = dict(num_pccs=3, p_incon=0, size=100)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> infr.apply_nondynamic_update()
            >>> cc1 = next(infr.positive_components())

            %timeit list(infr.pos_graph.subgraph(cc1, dynamic=True).edges())
            %timeit list(infr.pos_graph.subgraph(cc1, dynamic=False).edges())
            %timeit list(edges_inside(infr.pos_graph, cc1))
        """
        decision = POSTV
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1, incon2 = infr.recover_graph.has_nodes(edge)
        all_consistent = not (incon1 or incon2)
        was_within = nid1 == nid2

        print_ = ut.partial(infr.print, level=4)

        if was_within:
            infr._add_review_edge(edge, decision)
            if all_consistent:
                # print_('Internal consistent positive review')
                print_('pos-within-clean')
                infr.update_pos_redun(nid1, may_remove=False)
            else:
                # print_('Internal inconsistent positive review')
                print_('pos-within-dirty')
                infr._check_inconsistency(nid1)
            action = infr.on_within(edge, decision, nid1, None)
        else:
            # print_('Merge case')
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)

            if not all_consistent:
                # print_('Inconsistent merge',)
                print_('pos-between-dirty-merge')
                if not incon1:
                    recover_edges = list(edges_inside(infr.pos_graph, cc1))
                else:
                    recover_edges = list(edges_inside(infr.pos_graph, cc2))
                infr.recover_graph.add_edges_from(recover_edges)
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                infr.recover_graph.add_edge(*edge)
                new_nid = infr.pos_graph.node_label(edge[0])
            elif any(edges_cross(infr.neg_graph, cc1, cc2)):
                # print_('Merge creates inconsistency',)
                print_('pos-between-clean-merge-dirty')
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                new_nid = infr.pos_graph.node_label(edge[0])
                infr._new_inconsistency(new_nid)
            else:
                # print_('Consistent merge')
                print_('pos-between-clean-merge-clean')
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                new_nid = infr.pos_graph.node_label(edge[0])
                infr.update_extern_neg_redun(new_nid, may_remove=False)
                infr.update_pos_redun(new_nid, may_remove=False)
            action = infr.on_between(edge, decision, nid1, nid2,
                                     merge_nid=new_nid)
        return action

    @profile
    def _negative_decision(infr, edge):
        """
        Logic for a dynamic negative decision.  A negative decision is evidence
        that two annots should not be in the same PCC
        """
        decision = NEGTV
        nid1, nid2 = infr.node_labels(*edge)
        incon1, incon2 = infr.recover_graph.has_nodes(edge)
        all_consistent = not (incon1 or incon2)
        infr._add_review_edge(edge, decision)
        new_nid1, new_nid2 = infr.pos_graph.node_labels(*edge)

        was_within = nid1 == nid2
        was_split = was_within and new_nid1 != new_nid2

        print_ = ut.partial(infr.print, level=4)

        if was_within:
            if was_split:
                if all_consistent:
                    print_('neg-within-split-clean')
                    prev_neg_nids = infr._purge_redun_flags(nid1)
                    infr.update_neg_redun_to(new_nid1, prev_neg_nids)
                    infr.update_neg_redun_to(new_nid2, prev_neg_nids)
                    infr.update_neg_redun(new_nid1, new_nid2)
                    # infr.update_extern_neg_redun(new_nid1, may_remove=False)
                    # infr.update_extern_neg_redun(new_nid2, may_remove=False)
                    infr.update_pos_redun(new_nid1, may_remove=False)
                    infr.update_pos_redun(new_nid2, may_remove=False)
                else:
                    print_('neg-within-split-dirty')
                    if infr.recover_graph.has_edge(*edge):
                        infr.recover_graph.remove_edge(*edge)
                    infr._purge_error_edges(nid1)
                    infr._purge_redun_flags(nid1)
                    infr._check_inconsistency(new_nid1)
                    infr._check_inconsistency(new_nid2)
                # Signal that a split occurred
                action = infr.on_within(edge, decision, nid1,
                                        split_nids=(new_nid1, new_nid2))
            else:
                if all_consistent:
                    print_('neg-within-clean')
                    infr._purge_redun_flags(new_nid1)
                    infr._new_inconsistency(new_nid1)
                else:
                    print_('neg-within-dirty')
                    infr._check_inconsistency(new_nid1)
                action = infr.on_within(edge, decision, new_nid1)
        else:
            if all_consistent:
                print_('neg-between-clean')
                infr.update_neg_redun(new_nid1, new_nid2, may_remove=False)
            else:
                print_('neg-between-dirty')
                # nothing to do if a negative edge is added between two PCCs
                # where at least one is inconsistent
                pass
            action = infr.on_between(edge, decision, new_nid1, new_nid2)
        return action

    @profile
    def _uninferable_decision(infr, edge, decision):
        """
        Logic for a dynamic uninferable negative decision An uninferrable
        decision does not provide any evidence about PCC status and is either:
            incomparable, unreviewed, or unknown
        """
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1 = infr.recover_graph.has_node(edge[0])
        incon2 = infr.recover_graph.has_node(edge[1])
        all_consistent = not (incon1 or incon2)

        was_within = nid1 == nid2

        print_ = ut.partial(infr.print, level=4)

        overwrote_positive = infr.pos_graph.has_edge(*edge)
        overwrote_negative = infr.neg_graph.has_edge(*edge)

        try:
            prefix = {INCMP: 'incmp', UNREV: 'unrev',
                      UNKWN: 'unkown'}[decision]
        except KeyError:
            raise KeyError('decision can only be UNREV, INCMP, or UNKWN')

        infr._add_review_edge(edge, decision)

        if was_within:
            if overwrote_positive:
                # changed an existing positive edge
                if infr.recover_graph.has_edge(*edge):
                    infr.recover_graph.remove_edge(*edge)
                new_nid1, new_nid2 = infr.pos_graph.node_labels(*edge)
                was_split = new_nid1 != new_nid2
                if was_split:
                    old_nid = nid1
                    prev_neg_nids = infr._purge_redun_flags(old_nid)
                    if all_consistent:
                        print_('%s-within-pos-split-clean' % prefix)
                        # split case
                        infr.update_neg_redun_to(new_nid1, prev_neg_nids)
                        infr.update_neg_redun_to(new_nid2, prev_neg_nids)
                        # for other_nid in prev_neg_nids:
                        #     infr.update_neg_redun(new_nid1, other_nid)
                        #     infr.update_neg_redun(new_nid2, other_nid)
                        infr.update_neg_redun(new_nid1, new_nid2)
                        infr.update_pos_redun(new_nid1, may_remove=False)
                        infr.update_pos_redun(new_nid2, may_remove=False)
                    else:
                        print_('%s-within-pos-split-dirty' % prefix)
                        if infr.recover_graph.has_edge(*edge):
                            infr.recover_graph.remove_edge(*edge)
                        infr._purge_error_edges(nid1)
                        infr._check_inconsistency(new_nid1)
                        infr._check_inconsistency(new_nid2)
                    # Signal that a split occurred
                    action = infr.on_within(edge, decision, nid1,
                                            split_nids=(new_nid1, new_nid2))
                else:
                    if all_consistent:
                        print_('%s-within-pos-clean' % prefix)
                        infr.update_pos_redun(new_nid1, may_add=False)
                    else:
                        print_('%s-within-pos-dirty' % prefix)
                        # Overwriting a positive edge that is not a split
                        # in an inconsistent component, means no inference.
                    action = infr.on_within(edge, decision, new_nid1)
            elif overwrote_negative:
                print_('%s-within-neg-dirty' % prefix)
                assert not all_consistent
                infr._check_inconsistency(nid1)
                action = infr.on_within(edge, decision, new_nid1)
            else:
                if all_consistent:
                    print_('%s-within-clean' % prefix)
                else:
                    print_('%s-within-dirty' % prefix)
                action = infr.on_within(edge, decision, nid1)
        else:
            if overwrote_negative:
                if all_consistent:
                    # changed and existing negative edge only influences
                    # consistent pairs of PCCs
                    print_('incon-between-neg-clean')
                    infr.update_neg_redun(nid1, nid2, may_add=False)
                else:
                    print_('incon-between-neg-dirty')
            else:
                print_('incon-between')
            action = infr.on_between(edge, decision, nid1, nid2)
        return action


class Recovery(object):
    """ recovery funcs """

    def is_recovering(infr, edge=None):
        """
        Checks to see if the graph is inconsinsistent.

        Args:
            edge (None): If None, then returns True if the graph contains any
                inconsistency. Otherwise, returns True if the edge is related
                to an inconsistent component via a positive or negative
                connection.

        Returns:
            bool: flag

        CommandLine:
            python -m ibeis.algo.graph.mixin_dynamic is_recovering

        Doctest:
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=4, size=4, ignore_pair=True)
            >>> infr.ensure_cliques(meta_decision=SAME)
            >>> a, b, c, d = map(list, infr.positive_components())
            >>> assert infr.is_recovering() is False
            >>> infr.add_feedback((a[0], a[1]), NEGTV)
            >>> assert infr.is_recovering() is True
            >>> assert infr.is_recovering((a[2], a[3])) is True
            >>> assert infr.is_recovering((a[3], b[0])) is True
            >>> assert infr.is_recovering((b[0], b[1])) is False
            >>> infr.add_feedback((a[3], b[2]), NEGTV)
            >>> assert infr.is_recovering((b[0], b[1])) is True
            >>> assert infr.is_recovering((c[0], d[0])) is False
            >>> infr.add_feedback((b[2], c[0]), NEGTV)
            >>> assert infr.is_recovering((c[0], d[0])) is False
            >>> result = ut.repr4({
            >>>     'pccs': list(infr.positive_components()),
            >>>     'iccs': list(infr.inconsistent_components()),
            >>> }, nobr=True, si=True, itemsep='')
            >>> print(result)
            iccs: [{1,2,3,4}],
            pccs: [{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}],
        """
        if len(infr.recover_graph) == 0:
            # We can short-circuit if there is no inconsistency
            return False
        if edge is None:
            # By the short circuit we know the graph is inconsistent
            return True
        for nid in set(infr.node_labels(*edge)):
            # Is this edge part of a CC that has an error?
            if nid in infr.nid_to_errors:
                return True
            # Is this edge connected to a CC that has an error?
            cc = infr.pos_graph.component(nid)
            for nid2 in infr.find_external_neg_nids(cc):
                if nid2 in infr.nid_to_errors:
                    return True
        # If none of these conditions are true we are far enough away from the
        # inconsistency to ignore it.
        return False

    @profile
    def _purge_error_edges(infr, nid):
        """
        Removes all error edges associated with a PCC so they can be recomputed
        or resolved.
        """
        old_error_edges = infr.nid_to_errors.pop(nid, [])
        # Remove priority from old error edges
        if infr.params['inference.update_attrs']:
            infr.set_edge_attrs('maybe_error',
                                ut.dzip(old_error_edges, [None]))
        infr._remove_edge_priority(old_error_edges)
        was_clean = len(old_error_edges) > 0
        return was_clean

    @profile
    def _set_error_edges(infr, nid, new_error_edges):
        # flag error edges
        infr.nid_to_errors[nid] = new_error_edges
        # choose one and give it insanely high priority
        if infr.params['inference.update_attrs']:
            infr.set_edge_attrs('maybe_error',
                                ut.dzip(new_error_edges, [True]))
        infr._increase_priority(new_error_edges, 10)

    def maybe_error_edges(infr):
        return ut.iflatten(infr.nid_to_errors.values())

    def _new_inconsistency(infr, nid):
        cc = infr.pos_graph.component(nid)
        pos_edges = infr.pos_graph.edges(cc)
        infr.recover_graph.add_edges_from(pos_edges)
        num = infr.recover_graph.number_of_components()
        msg = 'New inconsistency {} total'.format(num)
        infr.print(msg, 2, color='red')
        infr._check_inconsistency(nid, cc=cc)

    @profile
    def _check_inconsistency(infr, nid, cc=None):
        """
        Check if a PCC contains an error
        """
        if cc is None:
            cc = infr.pos_graph.component(nid)
        was_clean = infr._purge_error_edges(nid)
        neg_edges = list(edges_inside(infr.neg_graph, cc))
        if neg_edges:
            pos_subgraph_ = infr.pos_graph.subgraph(cc, dynamic=False).copy()
            if not nx.is_connected(pos_subgraph_):
                print('cc = %r' % (cc,))
                print('pos_subgraph_ = %r' % (pos_subgraph_,))
                raise AssertionError('must be connected')
            hypothesis = dict(infr.hypothesis_errors(pos_subgraph_, neg_edges))
            assert len(hypothesis) > 0, 'must have at least one'
            infr._set_error_edges(nid, set(hypothesis.keys()))
            is_clean = False
        else:
            infr.recover_graph.remove_nodes_from(cc)
            num = infr.recover_graph.number_of_components()
            # num = len(list(nx.connected_components(infr.recover_graph)))
            msg = ('An inconsistent PCC recovered, '
                   '{} inconsistent PCC(s) remain').format(num)
            infr.print(msg, 2, color='green')
            infr.update_pos_redun(nid, force=True)
            infr.update_extern_neg_redun(nid, force=True)
            is_clean = True
        return (was_clean, is_clean)

    def _mincut_edge_weights(infr, edges_):
        conf_gen = infr.gen_edge_values('confidence', edges_,
                                        default='unspecified')
        conf_gen = ['unspecified' if c is None else c for c in conf_gen]
        code_to_conf = const.CONFIDENCE.CODE_TO_INT
        code_to_conf = {
            'absolutely_sure' : 4.0,
            'pretty_sure'     : 0.6,
            'not_sure'        : 0.2,
            'guessing'        : 0.0,
            'unspecified'     : 0.0,
        }
        confs = np.array(ut.take(code_to_conf, conf_gen))
        # confs = np.array([0 if c is None else c for c in confs])

        prob_gen = infr.gen_edge_values('prob_match', edges_, default=0)
        probs = np.array(list(prob_gen))

        nrev_gen = infr.gen_edge_values('num_reviews', edges_, default=0)
        nrev = np.array(list(nrev_gen))

        weight = nrev + probs + confs
        return weight

    @profile
    def hypothesis_errors(infr, pos_subgraph, neg_edges):
        if not nx.is_connected(pos_subgraph):
            raise AssertionError('Not connected' + repr(pos_subgraph))
        infr.print(
            'Find hypothesis errors in {} nodes with {} neg edges'.format(
                len(pos_subgraph), len(neg_edges)), 3)

        pos_edges = list(pos_subgraph.edges())

        neg_weight = infr._mincut_edge_weights(neg_edges)
        pos_weight = infr._mincut_edge_weights(pos_edges)

        capacity = 'weight'
        nx.set_edge_attributes(pos_subgraph, name=capacity, values=ut.dzip(pos_edges, pos_weight))

        # Solve a multicut problem for multiple pairs of terminal nodes.
        # Running multiple min-cuts produces a k-factor approximation
        maybe_error_edges = set([])
        for (s, t), join_weight in zip(neg_edges, neg_weight):
            cut_weight, parts = nx.minimum_cut(pos_subgraph, s, t,
                                               capacity=capacity)
            cut_edgeset = edges_cross(pos_subgraph, *parts)
            if join_weight < cut_weight:
                join_edgeset = {(s, t)}
                chosen = join_edgeset
                hypothesis = POSTV
            else:
                chosen = cut_edgeset
                hypothesis = NEGTV
            for edge in chosen:
                if edge not in maybe_error_edges:
                    maybe_error_edges.add(edge)
                    yield (edge, hypothesis)


class Consistency(object):
    def is_consistent(infr, cc):
        r"""
        Determines if a PCC contains inconsistencies

        Args:
            cc (set): nodes in a PCC

        Returns:
            flag: bool: returns True unless cc contains any negative edges

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=1, p_incon=1)
            >>> assert not infr.is_consistent(next(infr.positive_components()))
            >>> infr = demo.demodata_infr(num_pccs=1, p_incon=0)
            >>> assert infr.is_consistent(next(infr.positive_components()))
        """
        return len(cc) <= 2 or not any(edges_inside(infr.neg_graph, cc))

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

    def inconsistent_components(infr, graph=None):
        """
        Generates inconsistent PCCs.
        These PCCs contain internal negative edges indicating an error exists.
        """
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


class Priority(object):
    """
    Handles prioritization of edges for review.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
        >>> from ibeis.algo.graph import demo
        >>> infr = demo.demodata_infr(num_pccs=20)
    """

    def remaining_reviews(infr):
        assert infr.queue is not None
        return len(infr.queue)

    def _pop(infr, *args):
        """ Wraps queue so ordering is determenistic """
        (e, (p, _)) = infr.queue.pop(*args)
        return (e, -p)

    def _push(infr, edge, priority):
        """ Wraps queue so ordering is determenistic """
        PSEUDO_RANDOM_TIEBREAKER = False
        if PSEUDO_RANDOM_TIEBREAKER:
            # Make it so tiebreakers have a pseudo-random order
            chaotic = int.from_bytes(ut.digest_data(edge, alg='sha1')[:8], 'big')
            tiebreaker = (chaotic,) + edge
        else:
            tiebreaker = edge
        # tiebreaker = (chaotic(chaotic(u) + chaotic(v)), u, v)
        infr.queue[edge] = (-priority, tiebreaker)

    def _peek_many(infr, n):
        """ Wraps queue so ordering is determenistic """
        return [(k, -p) for (k, (p, _)) in infr.queue.peek_many(n)]

    def _remove_edge_priority(infr, edges):
        if infr.queue is None:
            return
        edges_ = [edge for edge in edges if edge in infr.queue]
        if len(edges_) > 0:
            infr.print('removed priority from {} edges'.format(len(edges_)), 5)
            infr.queue.delete_items(edges_)

    def _reinstate_edge_priority(infr, edges):
        if infr.queue is None:
            return
        edges_ = [edge for edge in edges if edge not in infr.queue]
        if len(edges_) > 0:
            # TODO: use whatever the current metric is
            metric = 'prob_match'
            infr.print('reprioritize {} edges'.format(len(edges_)), 5)
            priorities = infr.gen_edge_values(metric, edges_, default=1e-9)
            for edge, priority in zip(edges_, priorities):
                infr._push(edge, priority)

    def _increase_priority(infr, edges, amount=10):
        if infr.queue is None:
            return
        infr.print('increase priority of {} edges'.format(len(edges)), 5)
        metric = 'prob_match'
        priorities = infr.gen_edge_values(metric, edges, default=1e-9)
        for edge, base in zip(edges, priorities):
            infr.push(edge, base + amount)

    def remove_internal_priority(infr, cc):
        if infr.queue is not None:
            infr._remove_edge_priority(edges_inside(infr.graph, cc))

    def remove_external_priority(infr, cc):
        if infr.queue is not None:
            infr._remove_edge_priority(edges_outgoing(infr.graph, cc))

    def remove_between_priority(infr, cc1, cc2):
        if infr.queue is not None:
            infr._remove_edge_priority(edges_cross(infr.graph, cc1, cc2))

    def reinstate_between_priority(infr, cc1, cc2):
        if infr.queue is not None:
            # Reinstate the appropriate edges into the queue
            edges = edges_cross(infr.unreviewed_graph, cc1, cc2)
            infr._reinstate_edge_priority(edges)

    def reinstate_internal_priority(infr, cc):
        if infr.queue is not None:
            # Reinstate the appropriate edges into the queue
            edges = edges_inside(infr.unreviewed_graph, cc)
            infr._reinstate_edge_priority(edges)

    def reinstate_external_priority(infr, cc):
        if infr.queue is not None:
            # Reinstate the appropriate edges into the queue
            edges = edges_outgoing(infr.unreviewed_graph, cc)
            infr._reinstate_edge_priority(edges)

    @profile
    def prioritize(infr, metric=None, edges=None, scores=None,
                   force_inconsistent=True, reset=False):
        """
        Adds edges to the priority queue

        Doctest:
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=7, size=5)
            >>> infr.ensure_cliques(meta_decision=SAME)
            >>> # Add a negative edge inside a PCC
            >>> ccs = list(infr.positive_components())
            >>> edge1 = tuple(list(ccs[0])[0:2])
            >>> edge2 = tuple(list(ccs[1])[0:2])
            >>> infr.add_feedback(edge1, NEGTV)
            >>> infr.add_feedback(edge2, NEGTV)
            >>> num_new = infr.prioritize(reset=True)
            >>> order = infr._peek_many(np.inf)
            >>> scores = ut.take_column(order, 1)
            >>> assert scores[0] > 10
            >>> assert len(scores) == num_new, 'should prioritize two hypotheis edges'
            >>> unrev_edges = set(infr.unreviewed_graph.edges())
            >>> err_edges = set(ut.flatten(infr.nid_to_errors.values()))
            >>> edges = set(list(unrev_edges - err_edges)[0:2])
            >>> edges.update(list(err_edges)[0:2])
            >>> num_new = infr.prioritize(edges=edges, reset=True)
            >>> order2 = infr._peek_many(np.inf)
            >>> scores2 = np.array(ut.take_column(order2, 1))
            >>> assert np.all(scores2[0:2] > 10)
            >>> assert np.all(scores2[2:] < 10)

        Example:
            import ibeis
            infr = ibeis.AnnotInference('PZ_MTEST', aids='all', autoinit='staging')
            infr.verbose = 1000
            infr.load_published()
            incon_edges = set(ut.iflatten(infr.nid_to_errors.values()))
            assert len(incon_edges) > 0
            edges = list(infr.find_pos_redun_candidate_edges())
            assert len(set(incon_edges).intersection(set(edges))) == 0
            infr.add_candidate_edges(edges)

            infr.prioritize()
            print(ut.repr4(infr.status()))
        """
        if reset or infr.queue is None:
            infr.queue = ut.PriorityQueue()
        low = 1e-9
        if metric is None:
            metric = 'prob_match'

        # If edges are not explicilty specified get unreviewed and error edges
        # that are not redundant
        if edges is None:
            if scores is not None:
                raise ValueError('must provide edges with scores')
            unrev_edges = infr.unreviewed_graph.edges()
            edges = set(infr.filter_nonredun_edges(unrev_edges))

        infr.print('ensuring {} edge(s) get priority'.format(
            len(edges)), 5)

        if infr.params['inference.enabled'] and force_inconsistent:
            # Ensure that maybe_error edges are always prioritized
            maybe_error_edges = set(infr.maybe_error_edges())
            extra_edges = set(maybe_error_edges).difference(set(edges))
            extra_edges = list(extra_edges)
            infr.print('ensuring {} inconsistent edge(s) get priority'.format(
                len(extra_edges)), 5)

            if scores is not None:
                pgen = list(infr.gen_edge_values(metric, extra_edges, default=low))
                extra_scores = np.array(pgen)
                extra_scores[np.isnan(extra_scores)] = low

                scores = ut.aslist(scores) + ut.aslist(extra_scores)
            edges = ut.aslist(edges) + extra_edges

        # Ensure edges are in some arbitrary order
        edges = list(edges)

        # Ensure given scores do not have nan values
        if scores is None:
            pgen = infr.gen_edge_values(metric, edges, default=low)
            priorities = np.array(list(pgen))
            priorities[np.isnan(priorities)] = low
        else:
            priorities = np.asarray(scores)
            if np.any(np.isnan(priorities)):
                priorities[np.isnan(priorities)] = low

        if infr.params['inference.enabled']:
            # Increase priority of any flagged maybe_error edges
            err_flags = [e in maybe_error_edges for e in edges]
            priorities[err_flags] += 10

        # Push new items into the priority queue
        num_new = 0
        for edge, priority in zip(edges, priorities):
            if edge not in infr.queue:
                num_new += 1
            infr._push(edge, priority)

        infr.print('added %d edges to the queue' % (num_new,), 1)
        return num_new

    def push(infr, edge, priority=None):
        """
        Push an edge back onto the queue
        """
        if priority is None:
            priority = 'prob_match'
        if isinstance(priority, six.string_types):
            prob_match = infr.get_edge_attr(edge, priority, default=1e-9)
            priority = prob_match
        # Use edge-ids to break ties for determenistic behavior
        infr._push(edge, priority)

    @profile
    def pop(infr):
        """
        Main interface to the priority queue used by the algorithm loops.
        Pops the highest priority edge from the queue.
        """
        try:
            edge, priority = infr._pop()
        except IndexError:
            raise StopIteration('no more to review!')
        else:
            if infr.params['redun.enabled']:
                u, v = edge
                nid1, nid2 = infr.node_labels(u, v)
                pos_graph = infr.pos_graph
                pos_graph[nid1]
                if nid1 == nid2:
                    if nid1 not in infr.nid_to_errors:
                        # skip edges that increase local connectivity beyond
                        # redundancy thresholds.
                        k_pos = infr.params['redun.pos']
                        # Much faster to compute local connectivity on subgraph
                        cc = infr.pos_graph.component(nid1)
                        pos_subgraph = infr.pos_graph.subgraph(cc)
                        pos_conn = nx.connectivity.local_edge_connectivity(
                            pos_subgraph, u, v, cutoff=k_pos)
                        # Compute local connectivity
                        if pos_conn >= k_pos:
                            return infr.pop()
            if infr.params['queue.conf.thresh'] is not None:
                # Ignore reviews that would re-enforce a relationship that
                # already has high confidence.
                thresh_code = infr.params['queue.conf.thresh']
                thresh = const.CONFIDENCE.CODE_TO_INT[thresh_code]
                if priority < 10:
                    u, v = edge
                    nid1, nid2 = infr.node_labels(u, v)
                    if nid1 == nid2:
                        if infr.confidently_connected(u, v, thresh):
                            infr.pop()
                    else:
                        if infr.confidently_separated(u, v, thresh):
                            infr.pop()

            if getattr(infr, 'fix_mode_split', False):
                # only checking edges within a name
                nid1, nid2 = infr.pos_graph.node_labels(*edge)
                if nid1 != nid2:
                    return infr.pop()
            if getattr(infr, 'fix_mode_merge', False):
                # only checking edges within a name
                nid1, nid2 = infr.pos_graph.node_labels(*edge)
                if nid1 == nid2:
                    return infr.pop()
            if getattr(infr, 'fix_mode_predict', False):
                # No longer needed.
                pred = infr.get_edge_data(edge).get('pred', None)
                # only report cases where the prediction differs
                if priority < 10:
                    nid1, nid2 = infr.node_labels(*edge)
                    if nid1 == nid2:
                        u, v = edge
                        # Don't re-review confident CCs
                        thresh = const.CONFIDENCE.CODE_TO_INT['pretty_sure']
                        if infr.confidently_connected(u, v, thresh):
                            return infr.pop()
                    if pred == POSTV and nid1 == nid2:
                        # print('skip pos')
                        return infr.pop()
                    if pred == NEGTV and nid1 != nid2:
                        # print('skip neg')
                        return infr.pop()
                else:
                    print('in error recover mode')
            assert edge[0] < edge[1]
            return edge, priority

    def peek(infr):
        return infr.peek_many(n=1)[0]

    def peek_many(infr, n):
        """
        Peeks at the top n edges in the queue.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=7, size=5)
            >>> infr.refresh_candidate_edges()
            >>> infr.peek_many(50)
        """
        # Do pops that may invalidate pos redun edges internal to PCCs
        items = []
        count = 0
        # Pop the top n edges off the queue
        while len(infr.queue) > 0 and count < n:
            items.append(infr.pop())
            count += 1
        # Push them back because we are just peeking
        # (although we may have invalidated things based on local connectivity)
        for edge, priority in items:
            infr.push(edge, priority)
        return items

    def confidently_connected(infr, u, v, thresh=2):
        """
        Checks if u and v are conneted by edges above a confidence threshold
        """
        def satisfied(G, child, edge):
            decision = infr.edge_decision(edge)
            if decision != POSTV:
                return False
            data = G.get_edge_data(*edge)
            conf = data.get('confidence', 'unspecified')
            conf_int = const.CONFIDENCE.CODE_TO_INT[conf]
            conf_int = 0 if conf_int is None else conf_int
            return conf_int >= thresh
        for node in ut.bfs_conditional(infr.graph, u,
                                       yield_if=satisfied,
                                       continue_if=satisfied):
            if node == v:
                return True
        return False

    def confidently_separated(infr, u, v, thresh=2):
        """
        Checks if u and v are conneted by edges above a confidence threshold

        Doctest:
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.make_demo_infr(ccs=[(1, 2), (3, 4), (5, 6), (7, 8)])
            >>> infr.add_feedback((1, 5), NEGTV)
            >>> infr.add_feedback((5, 8), NEGTV)
            >>> infr.add_feedback((6, 3), NEGTV)
            >>> u, v = (1, 4)
            >>> thresh = 0
            >>> assert not infr.confidently_separated(u, v, thresh)
            >>> infr.add_feedback((2, 3), NEGTV)
            >>> assert not infr.confidently_separated(u, v, thresh)
        """

        def can_cross(G, edge, n_negs):
            """
            DFS state condition

            Args:
                edge (tuple): the edge we are trying to cross
                n_negs (int): the number of negative edges crossed so far

            Returns:
                flag, new_state -
                   flag (bool): True if the edge can be crossed
                   new_state: new state for future decisions in this path.
            """
            decision = infr.edge_decision(edge)
            # only cross positive or negative edges
            if decision in {POSTV, NEGTV}:
                # only cross a negative edge once
                willcross = (decision == NEGTV)
                if willcross and n_negs == 0:
                    data = G.get_edge_data(*edge)
                    # only cross edges above a threshold
                    conf = data.get('confidence', 'unspecified')
                    conf_int = const.CONFIDENCE.CODE_TO_INT[conf]
                    conf_int = 0 if conf_int is None else conf_int
                    flag = conf_int >= thresh
                    num = n_negs + willcross
                    return flag, num
            return False, n_negs

        # need to do DFS check for this. Make DFS only allowed to
        # cross a negative edge once.
        # def dfs_cond_rec(G, parent, state, visited=None):
        #     if visited is None:
        #         visited = set()
        #     visited.add(parent)
        #     for child in G.neighbors(parent):
        #         if child not in visited:
        #             edge = (parent, child)
        #             flag, new_state = can_cross(G, edge, state)
        #             if flag:
        #                 yield child
        #                 for _ in dfs_cond_rec(G, child, new_state, visited):
        #                     yield _

        # need to do DFS check for this. Make DFS only allowed to
        # cross a negative edge once.
        def dfs_cond_stack(G, source, state):
            # stack based version
            visited = {source}
            stack = [(source, iter(G[source]), state)]
            while stack:
                parent, children, state = stack[-1]
                try:
                    child = next(children)
                    if child not in visited:
                        edge = (parent, child)
                        flag, new_state = can_cross(G, edge, state)
                        if flag:
                            yield child
                            visited.add(child)
                            stack.append((child, iter(G[child]), new_state))
                except StopIteration:
                    stack.pop()

        for node in dfs_cond_stack(infr.graph, u, 0):
            if node == v:
                return True
        return False

    def generate_reviews(infr, pos_redun=None, neg_redun=None,
                         data=False):
        """
        Dynamic generator that yeilds high priority reviews
        """
        if pos_redun is not None:
            infr.params['redun.pos'] = pos_redun
        if neg_redun is not None:
            infr.params['redun.neg'] = neg_redun
        infr.prioritize()
        return infr._generate_reviews(data=data)

    def _generate_reviews(infr, data=False):
        if data:
            while True:
                edge, priority = infr.pop()
                yield edge, priority
        else:
            while True:
                edge, priority = infr.pop()
                yield edge


@six.add_metaclass(ut.ReloadingMetaclass)
class _RedundancyHelpers(object):
    """ methods for computing redundancy """

    def rand_neg_check_edges(infr, c1_nodes, c2_nodes):
        """
        Find enough edges to between two pccs to make them k-negative complete
        """
        k = infr.params['redun.neg']
        existing_edges = edges_cross(infr.graph, c1_nodes, c2_nodes)
        reviewed_edges = {
            edge: state
            for edge, state in infr.get_edge_attrs(
                'decision', existing_edges,
                default=UNREV).items()
            if state != UNREV
        }
        n_neg = sum([state == NEGTV for state in reviewed_edges.values()])
        if n_neg < k:
            # Find k random negative edges
            check_edges = existing_edges - set(reviewed_edges)
            if len(check_edges) < k:
                edges = it.starmap(nxu.e_, it.product(c1_nodes, c2_nodes))
                for edge in edges:
                    if edge not in reviewed_edges:
                        check_edges.add(edge)
                        if len(check_edges) == k:
                            break
        else:
            check_edges = {}
        return check_edges

    def find_external_neg_nids(infr, cc):
        """
        Find the nids with at least one negative edge external
        to this cc.
        """
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph
        out_neg_nids = set([])
        for u in cc:
            nid1 = pos_graph.node_label(u)
            for v in neg_graph.neighbors(u):
                nid2 = pos_graph.node_label(v)
                if nid1 == nid2 and v not in cc:
                    continue
                out_neg_nids.add(nid2)
        return out_neg_nids

    def find_external_neg_nid_freq(infr, cc):
        """
        Find the number of edges leaving `cc` and directed towards specific
        names.
        """
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph
        neg_nid_freq = ut.ddict(lambda: 0)
        for u in cc:
            nid1 = pos_graph.node_label(u)
            for v in neg_graph.neighbors(u):
                nid2 = pos_graph.node_label(v)
                if nid1 == nid2 and v not in cc:
                    continue
                neg_nid_freq[nid2] += 1
        return neg_nid_freq


@six.add_metaclass(ut.ReloadingMetaclass)
class Redundancy(_RedundancyHelpers):
    """ methods for computing redundancy """

    @profile
    def _purge_redun_flags(infr, nid):
        """
        Removes positive and negative redundancy from nids and all other PCCs
        touching nids respectively. Return the external PCC nids.
        """
        if not infr.params['redun.enabled']:
            return []
        if infr.neg_redun_nids.has_node(nid):
            prev_neg_nids = set(infr.neg_redun_nids.neighbors(nid))
        else:
            prev_neg_nids = []
        # infr.print('_purge, nid=%r, prev_neg_nids = %r' % (nid, prev_neg_nids,))
        # for other_nid in prev_neg_nids:
        #     flag = False
        #     if other_nid not in infr.pos_graph._ccs:
        #         flag = True
        #         infr.print('!!nid=%r did not update' % (other_nid,))
        #     if flag:
        #         assert flag, 'nids not maintained'
        for other_nid in prev_neg_nids:
            infr._set_neg_redun_flag((nid, other_nid), False)
        if nid in infr.pos_redun_nids:
            infr._set_pos_redun_flag(nid, False)
        # infr.update_pos_redun(nid, may_remove=True, may_add=False)
        # if nid in infr.
        return prev_neg_nids

    @profile
    def update_extern_neg_redun(infr, nid, may_add=True, may_remove=True,
                                force=False):
        """
        Checks if `nid` is negative redundant to any other `cc` it has at least
        one negative review to.
        """
        if not infr.params['redun.enabled']:
            return
        # infr.print('neg_redun external update nid={}'.format(nid), 5)
        k_neg = infr.params['redun.neg']
        cc1 = infr.pos_graph.component(nid)
        force = True
        if force:
            # TODO: non-force versions
            freqs = infr.find_external_neg_nid_freq(cc1)
            other_nids = []
            flags = []
            for other_nid, freq in freqs.items():
                if freq >= k_neg:
                    other_nids.append(other_nid)
                    flags.append(True)
                    # infr._set_neg_redun_flag((nid, other_nid), True)
                elif may_remove:
                    other_nids.append(other_nid)
                    flags.append(False)
                    # to_unflag.append(other_nid)
                    # infr._set_neg_redun_flag((nid, other_nid), False)

            if len(other_nids) > 0:
                infr._set_neg_redun_flags(nid, other_nids, flags)
            else:
                infr.print('neg_redun skip update nid=%r' % (nid,), 6)

    @profile
    def update_neg_redun_to(infr, nid1, other_nids, may_add=True, may_remove=True,
                            force=False):
        """
        Checks if nid1 is neg redundant to other_nids.
        Edges are either removed or added to the queue appropriately.
        """
        if not infr.params['redun.enabled']:
            return
        # infr.print('update_neg_redun', 5)
        force = True
        cc1 = infr.pos_graph.component(nid1)
        if not force:
            raise NotImplementedError('implement non-forced version')
        flags = []
        for nid2 in other_nids:
            cc2 = infr.pos_graph.component(nid2)
            need_add = infr.is_neg_redundant(cc1, cc2)
            flags.append(need_add)
        infr._set_neg_redun_flags(nid1, other_nids, flags)

    @profile
    def update_neg_redun(infr, nid1, nid2, may_add=True, may_remove=True,
                         force=False):
        """
        Checks if two PCCs are newly or no longer negative redundant.
        Edges are either removed or added to the queue appropriately.
        """
        infr.update_neg_redun_to(nid1, [nid2], may_add, may_remove, force)

    @profile
    def is_pos_redundant(infr, cc, k=None, relax=None, assume_connected=False):
        """
        Tests if a group of nodes is positive redundant.
        (ie. if the group is k-edge-connected)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.make_demo_infr(ccs=[(1, 2, 3, 4, 5)])
            >>> infr.params['redun.pos'] = 2
            >>> cc = infr.pos_graph.connected_to(1)
            >>> flag1 = infr.is_pos_redundant(cc)
            >>> infr.add_feedback((1, 5), POSTV)
            >>> flag2 = infr.is_pos_redundant(cc)
            >>> flags = [flag1, flag2]
            >>> print('flags = %r' % (flags,))
            >>> assert flags == [False, True]
        """
        if k is None:
            k = infr.params['redun.pos']
        if assume_connected and k == 1:
            return True  # assumes cc is connected
        if relax is None:
            relax = True
        pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False)
        if relax:
            # If we cannot add any more edges to the subgraph then we consider
            # it positive redundant.
            n_incomp = sum(1 for _ in nxu.edges_inside(infr.incomp_graph, cc))
            n_pos = pos_subgraph.number_of_edges()
            n_nodes = pos_subgraph.number_of_nodes()
            n_max = (n_nodes * (n_nodes - 1)) // 2
            if n_max == (n_pos + n_incomp):
                return True
        # In all other cases test edge-connectivity
        return nxu.is_k_edge_connected(pos_subgraph, k=k)

    @profile
    def is_neg_redundant(infr, cc1, cc2, k=None):
        r"""
        Tests if two groups of nodes are negative redundant
        (ie. have at least k negative edges between them).

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.make_demo_infr(ccs=[(1, 2), (3, 4)])
            >>> infr.params['redun.neg'] = 2
            >>> cc1 = infr.pos_graph.connected_to(1)
            >>> cc2 = infr.pos_graph.connected_to(3)
            >>> flag1 = infr.is_neg_redundant(cc1, cc2)
            >>> infr.add_feedback((1, 3), NEGTV)
            >>> flag2 = infr.is_neg_redundant(cc1, cc2)
            >>> infr.add_feedback((2, 4), NEGTV)
            >>> flag3 = infr.is_neg_redundant(cc1, cc2)
            >>> flags = [flag1, flag2, flag3]
            >>> print('flags = %r' % (flags,))
            >>> assert flags == [False, False, True]
        """
        if k is None:
            k = infr.params['redun.neg']
        neg_edge_gen = edges_cross(infr.neg_graph, cc1, cc2)
        # do a lazy count of negative edges
        for count, _ in enumerate(neg_edge_gen, start=1):
            if count >= k:
                return True
        return False

    def pos_redundancy(infr, cc):
        """ Returns how positive redundant a cc is """
        pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False)
        if nxu.is_complete(pos_subgraph):
            return np.inf
        else:
            return nx.edge_connectivity(pos_subgraph)

    def neg_redundancy(infr, cc1, cc2):
        """ Returns how negative redundant a cc is """
        neg_edge_gen = edges_cross(infr.neg_graph, cc1, cc2)
        num_neg = len(list(neg_edge_gen))
        if num_neg == len(cc1) or num_neg == len(cc2):
            return np.inf
        else:
            return num_neg

    # def pos_redun_edge_flag(infr, edge):
    #     """ Quickly check if edge is flagged as pos redundant """
    #     nid1, nid2 = infr.pos_graph.node_labels(*edge)
    #     return nid1 == nid2 and nid1 in infr.pos_redun_nids

    # def neg_redun_edge_flag(infr, edge):
    #     """ Quickly check if edge is flagged as neg redundant """
    #     nid1, nid2 = infr.pos_graph.node_labels(*edge)
    #     return infr.neg_redun_nids.has_edge(nid1, nid2)

    def is_redundant(infr, edge):
        nidu, nidv = infr.node_labels(*edge)
        if nidu == nidv:
            if nidu in infr.pos_redun_nids:
                return True
        elif nidu != nidv:
            if infr.neg_redun_nids.has_edge(nidu, nidv):
                return True
        return False

    def filter_nonredun_edges(infr, edges):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=1, size=4)
            >>> infr.clear_edges()
            >>> infr.ensure_cliques()
            >>> infr.clear_feedback()
            >>> print(ut.repr4(infr.status()))
            >>> nonredun_edges = list(infr.filter_nonredun_edges(
            >>>     infr.unreviewed_graph.edges()))
            >>> assert len(nonredun_edges) == 6
        """
        for edge in edges:
            if not infr.is_redundant(edge):
                yield edge

    @profile
    def negative_redundant_nids(infr, cc):
        """
        Get PCCs that are k-negative redundant with `cc`

            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> infr = demo.demodata_infr2()
            >>> node = 20
            >>> cc = infr.pos_graph.connected_to(node)
            >>> infr.params['redun.neg'] = 2
            >>> infr.negative_redundant_nids(cc)
        """
        neg_nid_freq = infr.find_external_neg_nid_freq(cc)
        # check for k-negative redundancy
        k_neg = infr.params['redun.neg']
        pos_graph = infr.pos_graph
        neg_nids = [
            nid2 for nid2, freq in neg_nid_freq.items()
            if (
                freq >= k_neg or
                freq == len(cc) or
                freq == len(pos_graph.connected_to(nid2))
            )
        ]
        return neg_nids

    def pos_redundant_pccs(infr, k=None, relax=None):
        if k is None:
            k = infr.params['redun.pos']
        for cc in infr.consistent_components():
            if infr.is_pos_redundant(cc, k=k, relax=relax):
                yield cc

    def non_pos_redundant_pccs(infr, k=None, relax=None):
        """
        Get PCCs that are not k-positive-redundant
        """
        if k is None:
            k = infr.params['redun.pos']
        for cc in infr.consistent_components():
            if not infr.is_pos_redundant(cc, k=k, relax=relax):
                yield cc

    def find_pos_redun_nids(infr):
        """ recomputes infr.pos_redun_nids """
        for cc in infr.pos_redundant_pccs():
            node = next(iter(cc))
            nid = infr.pos_graph.node_label(node)
            yield nid

    def find_neg_redun_nids(infr):
        """ recomputes edges in infr.neg_redun_nids """
        for cc in infr.consistent_components():
            node = next(iter(cc))
            nid1 = infr.pos_graph.node_label(node)
            for nid2 in infr.negative_redundant_nids(cc):
                if nid1 < nid2:
                    yield nid1, nid2

    @profile
    def update_pos_redun(infr, nid, may_add=True, may_remove=True,
                         force=False):
        """
        Checks if a PCC is newly, or no longer positive redundant.
        Edges are either removed or added to the queue appropriately.
        """
        if not infr.params['redun.enabled']:
            return

        # force = True
        # infr.print('update_pos_redun')
        need_add = False
        need_remove = False
        if force:
            cc = infr.pos_graph.component(nid)
            need_add = infr.is_pos_redundant(cc)
            need_remove = not need_add
        else:
            was_pos_redun = nid in infr.pos_redun_nids
            if may_add and not was_pos_redun:
                cc = infr.pos_graph.component(nid)
                need_add = infr.is_pos_redundant(cc)
            elif may_remove and not was_pos_redun:
                cc = infr.pos_graph.component(nid)
                need_remove = not infr.is_pos_redundant(cc)
        if need_add:
            infr._set_pos_redun_flag(nid, True)
        elif need_remove:
            infr._set_pos_redun_flag(nid, False)
        else:
            infr.print('pos_redun skip update nid=%r' % (nid,), 6)

    @profile
    def _set_pos_redun_flag(infr, nid, flag):
        """
        Flags or unflags an nid as positive redundant.
        """
        was_pos_redun = nid in infr.pos_redun_nids
        if flag:
            if not was_pos_redun:
                infr.print('pos_redun flag=T nid=%r' % (nid,), 5)
            else:
                infr.print('pos_redun flag=T nid=%r (already done)' % (nid,), 6)
            infr.pos_redun_nids.add(nid)
            cc = infr.pos_graph.component(nid)
            infr.remove_internal_priority(cc)
            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state',
                    ut.dzip(edges_inside(infr.graph, cc), ['same'])
                )
        else:
            if was_pos_redun:
                infr.print('pos_redun flag=F nid=%r' % (nid,), 5)
            else:
                infr.print('pos_redun flag=F nid=%r (already done)' % (nid,), 6)
            cc = infr.pos_graph.component(nid)
            infr.pos_redun_nids -= {nid}
            infr.reinstate_internal_priority(cc)
            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state',
                    ut.dzip(edges_inside(infr.graph, cc), [None])
                )

    @profile
    def _set_neg_redun_flags(infr, nid1, other_nids, flags):
        """
        Flags or unflags an nid1 as negative redundant with other nids.
        """
        needs_unflag = []
        needs_flag = []
        already_flagged = []
        already_unflagged = []
        cc1 = infr.pos_graph.component(nid1)
        other_nids = list(other_nids)

        # Determine what needs what
        for nid2, flag in zip(other_nids, flags):
            was_neg_redun = infr.neg_redun_nids.has_edge(nid1, nid2)
            if flag:
                if not was_neg_redun:
                    needs_flag.append(nid2)
                else:
                    already_flagged.append(nid2)
            else:
                if was_neg_redun:
                    needs_unflag.append(nid2)
                else:
                    already_unflagged.append(nid2)

        # Print summary of what will be done
        def _print_helper(what, others, already=False):
            if len(others) == 0:
                return
            n_other_thresh = 4
            if len(others) > n_other_thresh:
                omsg = '#others={}'.format(len(others))
            else:
                omsg = 'others={}'.format(others)
            amsg = '(already done)' if already else ''
            msg = '{} nid={}, {} {}'.format(what, nid1, omsg, amsg)
            infr.print(msg, 5 + already)

        _print_helper('neg_redun flag=T', needs_flag)
        _print_helper('neg_redun flag=T', already_flagged, already=True)
        _print_helper('neg_redun flag=F', needs_unflag)
        _print_helper('neg_redun flag=F', already_unflagged, already=True)

        # Do the flagging/unflagging
        for nid2 in needs_flag:
            infr.neg_redun_nids.add_edge(nid1, nid2)
        for nid2 in needs_unflag:
            infr.neg_redun_nids.remove_edge(nid1, nid2)

        # Update priorities and attributes
        if infr.params['inference.update_attrs'] or infr.queue is not None:
            all_flagged_edges = []
            # Unprioritize all edges between flagged nids
            for nid2 in it.chain(needs_flag, already_flagged):
                cc2 = infr.pos_graph.component(nid2)
                all_flagged_edges.extend(edges_cross(infr.graph, cc1, cc2))

        if infr.queue is not None or infr.params['inference.update_attrs']:
            all_unflagged_edges = []
            unrev_unflagged_edges = []
            unrev_graph = infr.unreviewed_graph
            # Reprioritize unreviewed edges between unflagged nids
            # Marked inferred state of all edges
            for nid2 in it.chain(needs_unflag, already_unflagged):
                cc2 = infr.pos_graph.component(nid2)
                if infr.queue is not None:
                    _edges = edges_cross(unrev_graph, cc1, cc2)
                    unrev_unflagged_edges.extend(_edges)
                if infr.params['inference.update_attrs']:
                    _edges = edges_cross(infr.graph, cc1, cc2)
                    all_unflagged_edges.extend(_edges)

            # Batch set prioritize
            infr._remove_edge_priority(all_flagged_edges)
            infr._reinstate_edge_priority(unrev_unflagged_edges)

            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state', ut.dzip(all_flagged_edges, ['diff'])
                )
                infr.set_edge_attrs(
                    'inferred_state', ut.dzip(all_unflagged_edges, [None])
                )

    @profile
    def _set_neg_redun_flag(infr, nid_edge, flag):
        """
        Flags or unflags an two nids as negative redundant.
        """
        nid1, nid2 = nid_edge
        infr._set_neg_redun_flags(nid1, [nid2], [flag])
        return
        was_neg_redun = infr.neg_redun_nids.has_edge(nid1, nid2)
        if flag:
            if not was_neg_redun:
                infr.print('flag_neg_redun nids=%r,%r' % (nid1, nid2), 5)
            else:
                infr.print('flag_neg_redun nids=%r,%r (already done)' % (
                    nid1, nid2), 6)

            infr.neg_redun_nids.add_edge(nid1, nid2)
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)
            infr.remove_between_priority(cc1, cc2)
            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state',
                    ut.dzip(edges_cross(infr.graph, cc1, cc2), ['diff'])
                )
        else:
            was_neg_redun = infr.neg_redun_nids.has_edge(nid1, nid2)
            if was_neg_redun:
                infr.print('unflag_neg_redun nids=%r,%r' % (nid1, nid2), 5)
            else:
                infr.print('unflag_neg_redun nids=%r,%r (already done)' % (
                    nid1, nid2), 6)
            try:
                infr.neg_redun_nids.remove_edge(nid1, nid2)
            except nx.exception.NetworkXError:
                pass
            # import utool
            # with utool.embed_on_exception_context:
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)
            infr.reinstate_between_priority(cc1, cc2)
            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state',
                    ut.dzip(edges_cross(infr.graph, cc1, cc2), [None])
                )


@six.add_metaclass(ut.ReloadingMetaclass)
class NonDynamicUpdate(object):

    @profile
    def apply_nondynamic_update(infr, graph=None):
        r"""
        Recomputes all dynamic bookkeeping for a graph in any state.
        This ensures that subsequent dyanmic inference can be applied.
        """
        categories = infr.categorize_edges(graph)

        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories[POSTV].values()), ['same'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories[NEGTV].values()), ['diff'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories[INCMP].values()), [INCMP])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories[UNKWN].values()), [UNKWN])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(ut.flatten(categories[UNREV].values()), [None])
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

        # Ensure bookkeeping is taken care of
        # * positive redundancy
        # * negative redundancy
        # * inconsistency
        infr.pos_redun_nids = set(infr.find_pos_redun_nids())
        infr.neg_redun_nids = infr._graph_cls(list(infr.find_neg_redun_nids()))
        infr.recover_graph.clear()
        nid_to_errors = {}
        for nid, intern_edges in categories['inconsistent_internal'].items():
            cc = infr.pos_graph.component_nodes(nid)
            pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False).copy()
            neg_edges = list(edges_inside(infr.neg_graph, cc))
            recover_hypothesis = dict(infr.hypothesis_errors(pos_subgraph,
                                                             neg_edges))
            nid_to_errors[nid] = set(recover_hypothesis.keys())
            infr.recover_graph.add_edges_from(pos_subgraph.edges())

        # Delete old hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ut.dzip(ut.flatten(infr.nid_to_errors.values()), [None])
        )
        # Set new hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ut.dzip(ut.flatten(nid_to_errors.values()), [True])
        )
        infr.nid_to_errors = nid_to_errors

        # no longer dirty
        if graph is None:
            infr.dirty = False

    @profile
    def categorize_edges(infr, graph=None):
        r"""
        Non-dynamically computes the status of each edge in the graph.
        This is can be used to verify the dynamic computations and update when
        the dynamic state is lost.

        CommandLine:
            python -m ibeis.algo.graph.mixin_dynamic categorize_edges --profile

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> num_pccs = 250 if ut.get_argflag('--profile') else 100
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> cat = infr.categorize_edges()
        """
        states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)
        rev_graph = {key: infr.review_graphs[key] for key in states}
        if graph is None or graph is infr.graph:
            graph = infr.graph
            nodes = None
        else:
            # Need to extract relevant subgraphs
            nodes = list(graph.nodes())
            for key in states:
                if key == POSTV:
                    rev_graph[key] = rev_graph[key].subgraph(nodes,
                                                             dynamic=False)
                else:
                    rev_graph[key] = rev_graph[key].subgraph(nodes)

        # TODO: Rebalance union find to ensure parents is a single lookup
        # infr.pos_graph._union_find.rebalance(nodes)
        # node_to_label = infr.pos_graph._union_find.parents
        node_to_label = infr.pos_graph._union_find

        # Get reviewed edges using fast lookup structures
        ne_to_edges = {
            key: nxu.group_name_edges(rev_graph[key], node_to_label)
            for key in states
        }

        # Use reviewed edges to determine status of PCCs (repr by name ids)
        # The next steps will rectify duplicates in these sets
        name_edges = {key: set(ne_to_edges[key].keys()) for key in states}

        # Positive and negative decisions override incomparable and unreviewed
        for key in UNINFERABLE:
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
            assert len(name_edges[UNKWN].intersection(incon_internal_ne)) == 0
            assert all(n1 == n2 for n1, n2 in incon_internal_ne), (
                'incon_internal edges should be internal to a PCC')

        # External inconsistentices are edges leaving inconsistent components
        incon_internal_nids = {n1 for n1, n2 in incon_internal_ne}
        incon_external_ne = set([])
        # Find all edges leaving an inconsistent PCC
        for key in (NEGTV,) + UNINFERABLE:
            incon_external_ne.update({
                (nid1, nid2) for nid1, nid2 in name_edges[key]
                if nid1 in incon_internal_nids or nid2 in incon_internal_nids
            })
        for key in (NEGTV,) + UNINFERABLE:
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
                for key in (POSTV,) + UNINFERABLE)
            for nid1, nid2 in name_edges[POSTV]
        }
        # Find edges between 1-negative-redundant consistent PCCs
        negative = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV,) + UNINFERABLE)
            for nid1, nid2 in name_edges[NEGTV]
        }
        # Find edges internal to inconsistent PCCs
        incon_internal = {
            nid: union(
                ne_to_edges[key][(nid, nid)]
                for key in (POSTV, NEGTV,) + UNINFERABLE)
            for nid in incon_internal_nids
        }
        # Find edges leaving inconsistent PCCs
        incon_external = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV,) + UNINFERABLE)
            for nid1, nid2 in incon_external_ne
        }
        # Unknown names may have been comparable but the reviewer did not
        # know and could not guess. Likely bad quality.
        unknown = {
            (nid1, nid2): ne_to_edges[UNKWN][(nid1, nid2)]
            for (nid1, nid2) in name_edges[UNKWN]
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
            POSTV: positive,
            NEGTV: negative,
            UNREV: unreviewed,
            INCMP: notcomparable,
            UNKWN: unknown,
            'inconsistent_internal': incon_internal,
            'inconsistent_external': incon_external,
        }
        return ne_categories


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.graph.dynamic_review
        python -m ibeis.algo.graph.dynamic_review --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
