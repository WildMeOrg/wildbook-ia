# -*- coding: utf-8 -*-
"""
TODO:
    Negative bookkeeping, needs a small re-organization fix.
    MOVE FROM neg_redun_metagraph TO neg_metagraph

    Instead of maintaining a graph that contains PCCS which are neg redundant
    to each other, the graph should maintain PCCs that have ANY negative edge
    between them (aka 1 neg redundant). Then that edge should store a flag
    indicating the strength / redundancy of that connection.
    A better idea might be to store both neg_redun_metagraph AND neg_metagraph.

    TODO: this (all neg-redun functionality can be easilly consolidated into
    the neg-metagraph-update. note, we have to allow inconsistent pccs to be in
    the neg redun graph, we just filter them out afterwords)

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import itertools as it
import networkx as nx
from wbia import constants as const
from wbia.algo.graph import nx_utils as nxu
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN, UNINFERABLE
from wbia.algo.graph.state import SAME, DIFF, NULL  # NOQA

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
        infr.graph.add_edges_from(
            new_edges,
            evidence_decision=UNREV,
            meta_decision=UNREV,
            decision=UNREV,
            num_reviews=0,
        )
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
        """
        Adds an edge to the appropriate data structure
        """
        # infr.print('add review edge=%r, decision=%r' % (edge, decision), 20)
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edge(*edge)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                if G.has_edge(*edge):
                    G.remove_edge(*edge)

    @profile
    def _get_current_decision(infr, edge):
        """
        Find if any data structure has the edge
        """
        for decision, G in infr.review_graphs.items():
            if G.has_edge(*edge):
                return decision
        return UNREV

    @profile
    def on_between(infr, edge, decision, prev_decision, nid1, nid2, merge_nid=None):
        """
        Callback when a review is made between two PCCs
        """
        action = ['between']

        infr._update_neg_metagraph(
            decision, prev_decision, nid1, nid2, merge_nid=merge_nid
        )

        if merge_nid is not None:
            # A merge occurred
            if infr.params['inference.update_attrs']:
                cc = infr.pos_graph.component(merge_nid)
                infr.set_node_attrs('name_label', ut.dzip(cc, [merge_nid]))
            # FIXME: this state is ugly
            action += ['merge']
        else:
            if decision == NEGTV:
                action += ['neg-evidence']
            elif decision == INCMP:
                action += ['incomp-evidence']
            else:
                action += ['other-evidence']
        return action

    @profile
    def on_within(infr, edge, decision, prev_decision, nid, split_nids=None):
        """
        Callback when a review is made inside a PCC
        """
        action = ['within']

        infr._update_neg_metagraph(
            decision, prev_decision, nid, nid, split_nids=split_nids
        )

        if split_nids is not None:
            # A split occurred
            if infr.params['inference.update_attrs']:
                new_nid1, new_nid2 = split_nids
                cc1 = infr.pos_graph.component(new_nid1)
                cc2 = infr.pos_graph.component(new_nid2)
                infr.set_node_attrs('name_label', ut.dzip(cc1, [new_nid1]))
                infr.set_node_attrs('name_label', ut.dzip(cc2, [new_nid2]))
            action += ['split']
        else:
            if decision == POSTV:
                action += ['pos-evidence']
            elif decision == INCMP:
                action += ['incomp-evidence']
            elif decision == NEGTV:
                action += ['neg-evidence']
            else:
                action += ['other-evidence']
        return action

    @profile
    def _update_neg_metagraph(
        infr, decision, prev_decision, nid1, nid2, merge_nid=None, split_nids=None
    ):
        """
        Update the negative metagraph based a new review

        TODO:
            we can likely consolidate lots of neg_redun_metagraph
            functionality into this function. Just check when the
            weights are above or under the threshold and update
            accordingly.
        """
        nmg = infr.neg_metagraph

        if decision == NEGTV and prev_decision != NEGTV:
            # New negative feedback. Add meta edge or increase weight
            if not nmg.has_edge(nid1, nid2):
                nmg.add_edge(nid1, nid2, weight=1)
            else:
                nmg.edges[nid1, nid2]['weight'] += 1
        elif decision != NEGTV and prev_decision == NEGTV:
            # Undid negative feedback. Remove meta edge or decrease weight.
            nmg.edges[nid1, nid2]['weight'] -= 1
            if nmg.edges[nid1, nid2]['weight'] == 0:
                nmg.remove_edge(nid1, nid2)

        if merge_nid:
            # Combine the negative edges between the merged PCCS
            assert split_nids is None
            # Find external nids marked as negative
            prev_edges = nmg.edges(nbunch=[nid1, nid2], data=True)
            # Map external neg edges onto new merged PCC
            # Accumulate weights between duplicate new name edges
            lookup = {nid1: merge_nid, nid2: merge_nid}
            ne_accum = {}
            for (u, v, d) in prev_edges:
                new_ne = infr.e_(lookup.get(u, u), lookup.get(v, v))
                if new_ne in ne_accum:
                    ne_accum[new_ne]['weight'] += d['weight']
                else:
                    ne_accum[new_ne] = d
            merged_edges = ((u, v, d) for (u, v), d in ne_accum.items())

            nmg.remove_nodes_from([nid1, nid2])
            nmg.add_node(merge_nid)
            nmg.add_edges_from(merged_edges)

        if split_nids:
            # Splitup the negative edges between the split PCCS
            assert merge_nid is None
            assert nid1 == nid2
            old_nid = nid1

            # Find the nodes we need to check against
            extern_nids = set(nmg.neighbors(old_nid))
            if old_nid in extern_nids:
                extern_nids.remove(old_nid)
                extern_nids.update(split_nids)

            # Determine how to split existing negative edges between the split
            # by going back to the original negative graph.
            split_edges = []
            for new_nid in split_nids:
                cc1 = infr.pos_graph.component(new_nid)
                for other_nid in extern_nids:
                    cc2 = infr.pos_graph.component(other_nid)
                    num = sum(
                        1
                        for _ in nxu.edges_between(
                            infr.neg_graph, cc1, cc2, assume_dense=False
                        )
                    )
                    if num:
                        split_edges.append((new_nid, other_nid, {'weight': num}))

            nmg.remove_node(old_nid)
            nmg.add_nodes_from(split_nids)
            nmg.add_edges_from(split_edges)

    @profile
    def _positive_decision(infr, edge):
        r"""
        Logic for a dynamic positive decision.  A positive decision is evidence
        that two annots should be in the same PCC

        Note, this could be an incomparable edge, but with a meta_decision of
        same.

        Ignore:
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> kwargs = dict(num_pccs=3, p_incon=0, size=100)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> infr.apply_nondynamic_update()
            >>> cc1 = next(infr.positive_components())

            %timeit list(infr.pos_graph.subgraph(cc1, dynamic=True).edges())
            %timeit list(infr.pos_graph.subgraph(cc1, dynamic=False).edges())
            %timeit list(nxu.edges_inside(infr.pos_graph, cc1))
        """
        decision = POSTV
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1, incon2 = infr.recover_graph.has_nodes(edge)
        all_consistent = not (incon1 or incon2)
        was_within = nid1 == nid2

        print_ = ut.partial(infr.print, level=4)
        prev_decision = infr._get_current_decision(edge)

        if was_within:
            infr._add_review_edge(edge, decision)
            if all_consistent:
                print_('pos-within-clean')
                infr.update_pos_redun(nid1, may_remove=False)
            else:
                print_('pos-within-dirty')
                infr._check_inconsistency(nid1)
            action = infr.on_within(edge, decision, prev_decision, nid1, None)
        else:
            # print_('Merge case')
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)

            if not all_consistent:
                print_('pos-between-dirty-merge')
                if not incon1:
                    recover_edges = list(nxu.edges_inside(infr.pos_graph, cc1))
                else:
                    recover_edges = list(nxu.edges_inside(infr.pos_graph, cc2))
                infr.recover_graph.add_edges_from(recover_edges)
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                infr.recover_graph.add_edge(*edge)
                new_nid = infr.pos_graph.node_label(edge[0])
            elif any(nxu.edges_cross(infr.neg_graph, cc1, cc2)):
                print_('pos-between-clean-merge-dirty')
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                new_nid = infr.pos_graph.node_label(edge[0])
                infr._new_inconsistency(new_nid)
            else:
                print_('pos-between-clean-merge-clean')
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                new_nid = infr.pos_graph.node_label(edge[0])
                infr.update_extern_neg_redun(new_nid, may_remove=False)
                infr.update_pos_redun(new_nid, may_remove=False)
            action = infr.on_between(
                edge, decision, prev_decision, nid1, nid2, merge_nid=new_nid
            )
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
        prev_decision = infr._get_current_decision(edge)

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
                    infr.update_neg_redun_to(new_nid1, [new_nid2])
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
                action = infr.on_within(
                    edge, decision, prev_decision, nid1, split_nids=(new_nid1, new_nid2)
                )
            else:
                if all_consistent:
                    print_('neg-within-clean')
                    infr._purge_redun_flags(new_nid1)
                    infr._new_inconsistency(new_nid1)
                else:
                    print_('neg-within-dirty')
                    infr._check_inconsistency(new_nid1)
                action = infr.on_within(edge, decision, prev_decision, new_nid1)
        else:
            if all_consistent:
                print_('neg-between-clean')
                infr.update_neg_redun_to(new_nid1, [new_nid2], may_remove=False)
            else:
                print_('neg-between-dirty')
                # nothing to do if a negative edge is added between two PCCs
                # where at least one is inconsistent
                pass
            action = infr.on_between(edge, decision, prev_decision, new_nid1, new_nid2)
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
        prev_decision = infr._get_current_decision(edge)

        print_ = ut.partial(infr.print, level=4)

        try:
            prefix = {INCMP: 'incmp', UNREV: 'unrev', UNKWN: 'unkown'}[decision]
        except KeyError:
            raise KeyError('decision can only be UNREV, INCMP, or UNKWN')

        infr._add_review_edge(edge, decision)

        if was_within:
            new_nid1, new_nid2 = infr.pos_graph.node_labels(*edge)
            if prev_decision == POSTV:
                # changed an existing positive edge
                if infr.recover_graph.has_edge(*edge):
                    infr.recover_graph.remove_edge(*edge)
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
                        #     infr.update_neg_redun_to(new_nid1, [other_nid])
                        #     infr.update_neg_redun_to(new_nid2, [other_nid])
                        infr.update_neg_redun_to(new_nid1, [new_nid2])
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
                    action = infr.on_within(
                        edge,
                        decision,
                        prev_decision,
                        nid1,
                        split_nids=(new_nid1, new_nid2),
                    )
                else:
                    if all_consistent:
                        print_('%s-within-pos-clean' % prefix)
                        infr.update_pos_redun(new_nid1, may_add=False)
                    else:
                        print_('%s-within-pos-dirty' % prefix)
                        # Overwriting a positive edge that is not a split
                        # in an inconsistent component, means no inference.
                    action = infr.on_within(edge, decision, prev_decision, new_nid1)
            elif prev_decision == NEGTV:
                print_('%s-within-neg-dirty' % prefix)
                assert not all_consistent
                infr._check_inconsistency(nid1)
                action = infr.on_within(edge, decision, prev_decision, new_nid1)
            else:
                if all_consistent:
                    print_('%s-within-clean' % prefix)
                else:
                    print_('%s-within-dirty' % prefix)
                action = infr.on_within(edge, decision, prev_decision, nid1)
        else:
            if prev_decision == NEGTV:
                if all_consistent:
                    # changed and existing negative edge only influences
                    # consistent pairs of PCCs
                    print_('incon-between-neg-clean')
                    infr.update_neg_redun_to(nid1, [nid2], may_add=False)
                else:
                    print_('incon-between-neg-dirty')
            else:
                print_('incon-between')
                # HACK, this sortof fixes inferred state not being set
                if infr.params['inference.update_attrs']:
                    if decision == INCMP:
                        pass
                        # if not infr.is_neg_redundant(cc1, cc2, k=1):
                        #     # TODO: verify that there isn't a negative inferred
                        #     # state
                        #     infr.set_edge_attrs(
                        #         'inferred_state', ut.dzip([edge], [INCMP])
                        #     )
            action = infr.on_between(edge, decision, prev_decision, nid1, nid2)
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
            python -m wbia.algo.graph.mixin_dynamic is_recovering

        Doctest:
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
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
            >>>     'pccs': sorted(list(infr.positive_components())),
            >>>     'iccs': sorted(list(infr.inconsistent_components())),
            >>> }, nobr=True, si=True, itemsep='')
            >>> print(result)
            iccs: [{1,2,3,4}],
            pccs: [{5,6,7,8},{9,10,11,12},{13,14,15,16},{1,2,3,4}],
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
            for nid2 in infr.find_neg_nids_to(cc):
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
            infr.set_edge_attrs('maybe_error', ut.dzip(old_error_edges, [None]))
        infr._remove_edge_priority(old_error_edges)
        was_clean = len(old_error_edges) > 0
        return was_clean

    @profile
    def _set_error_edges(infr, nid, new_error_edges):
        # flag error edges
        infr.nid_to_errors[nid] = new_error_edges
        # choose one and give it insanely high priority
        if infr.params['inference.update_attrs']:
            infr.set_edge_attrs('maybe_error', ut.dzip(new_error_edges, [True]))
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
        neg_edges = list(nxu.edges_inside(infr.neg_graph, cc))
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
            msg = (
                'An inconsistent PCC recovered, ' '{} inconsistent PCC(s) remain'
            ).format(num)
            infr.print(msg, 2, color='green')
            infr.update_pos_redun(nid, force=True)
            infr.update_extern_neg_redun(nid, force=True)
            is_clean = True
        return (was_clean, is_clean)

    def _mincut_edge_weights(infr, edges_):
        conf_gen = infr.gen_edge_values('confidence', edges_, default='unspecified')
        conf_gen = ['unspecified' if c is None else c for c in conf_gen]
        code_to_conf = const.CONFIDENCE.CODE_TO_INT
        code_to_conf = {
            'absolutely_sure': 4.0,
            'pretty_sure': 0.6,
            'not_sure': 0.2,
            'guessing': 0.0,
            'unspecified': 0.0,
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
                len(pos_subgraph), len(neg_edges)
            ),
            3,
        )

        pos_edges = list(pos_subgraph.edges())

        neg_weight = infr._mincut_edge_weights(neg_edges)
        pos_weight = infr._mincut_edge_weights(pos_edges)

        capacity = 'weight'
        nx.set_edge_attributes(
            pos_subgraph, name=capacity, values=ut.dzip(pos_edges, pos_weight)
        )

        # Solve a multicut problem for multiple pairs of terminal nodes.
        # Running multiple min-cuts produces a k-factor approximation
        maybe_error_edges = set([])
        for (s, t), join_weight in zip(neg_edges, neg_weight):
            cut_weight, parts = nx.minimum_cut(pos_subgraph, s, t, capacity=capacity)
            cut_edgeset = nxu.edges_cross(pos_subgraph, *parts)
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
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=1, p_incon=1)
            >>> assert not infr.is_consistent(next(infr.positive_components()))
            >>> infr = demo.demodata_infr(num_pccs=1, p_incon=0)
            >>> assert infr.is_consistent(next(infr.positive_components()))
        """
        return len(cc) <= 2 or not any(nxu.edges_inside(infr.neg_graph, cc))

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
            unique_labels = {pos_graph.node_label(node) for node in graph.nodes()}
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


@six.add_metaclass(ut.ReloadingMetaclass)
class _RedundancyComputers(object):
    """
    methods for computing redundancy

    These are used to compute redundancy bookkeeping structures.
    Thus, they should not use them in their calculations.
    """

    # def pos_redundancy(infr, cc):
    #     """ Returns how positive redundant a cc is """
    #     pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False)
    #     if nxu.is_complete(pos_subgraph):
    #         return np.inf
    #     else:
    #         return nx.edge_connectivity(pos_subgraph)

    # def neg_redundancy(infr, cc1, cc2):
    #     """ Returns how negative redundant a cc is """
    #     neg_edge_gen = nxu.edges_cross(infr.neg_graph, cc1, cc2)
    #     num_neg = len(list(neg_edge_gen))
    #     if num_neg == len(cc1) or num_neg == len(cc2):
    #         return np.inf
    #     else:
    #         return num_neg

    @profile
    def is_pos_redundant(infr, cc, k=None, relax=None, assume_connected=False):
        """
        Tests if a group of nodes is positive redundant.
        (ie. if the group is k-edge-connected)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
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
        Tests if two disjoint groups of nodes are negative redundant
        (ie. have at least k negative edges between them).

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
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
        neg_edge_gen = nxu.edges_cross(infr.neg_graph, cc1, cc2)
        # do a lazy count of negative edges
        for count, _ in enumerate(neg_edge_gen, start=1):
            if count >= k:
                return True
        return False

    def find_neg_nids_to(infr, cc):
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

    def find_neg_nid_freq_to(infr, cc):
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

    @profile
    def find_neg_redun_nids_to(infr, cc):
        """
        Get PCCs that are k-negative redundant with `cc`

            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> import wbia.plottool as pt
            >>> pt.qtensure()
            >>> infr = demo.demodata_infr2()
            >>> node = 20
            >>> cc = infr.pos_graph.connected_to(node)
            >>> infr.params['redun.neg'] = 2
            >>> infr.find_neg_redun_nids_to(cc)
        """
        neg_nid_freq = infr.find_neg_nid_freq_to(cc)
        # check for k-negative redundancy
        k_neg = infr.params['redun.neg']
        pos_graph = infr.pos_graph
        neg_nids = [
            nid2
            for nid2, freq in neg_nid_freq.items()
            if (
                freq >= k_neg
                or freq == len(cc)
                or freq == len(pos_graph.connected_to(nid2))
            )
        ]
        return neg_nids

    def find_pos_redundant_pccs(infr, k=None, relax=None):
        if k is None:
            k = infr.params['redun.pos']
        for cc in infr.consistent_components():
            if infr.is_pos_redundant(cc, k=k, relax=relax):
                yield cc

    def find_non_pos_redundant_pccs(infr, k=None, relax=None):
        """
        Get PCCs that are not k-positive-redundant
        """
        if k is None:
            k = infr.params['redun.pos']
        for cc in infr.consistent_components():
            if not infr.is_pos_redundant(cc, k=k, relax=relax):
                yield cc

    @profile
    def find_non_neg_redun_pccs(infr, k=None):
        """
        Get pairs of PCCs that are not complete.

        Ignore:
            >>> from wbia.algo.graph.mixin_matching import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_infr(pcc_sizes=[1, 1, 2, 3, 5, 8], ignore_pair=True)
            >>> non_neg_pccs = list(infr.find_non_neg_redun_pccs(k=2))
            >>> assert len(non_neg_pccs) == (6 * 5) / 2
        """
        if k is None:
            k = infr.params['redun.neg']
        # need to ensure pccs is static in case new user input is added
        pccs = list(infr.positive_components())
        # Loop through all pairs
        for cc1, cc2 in it.combinations(pccs, 2):
            if not infr.is_neg_redundant(cc1, cc2):
                yield cc1, cc2

    def find_pos_redun_nids(infr):
        """ recomputes infr.pos_redun_nids """
        for cc in infr.find_pos_redundant_pccs():
            node = next(iter(cc))
            nid = infr.pos_graph.node_label(node)
            yield nid

    def find_neg_redun_nids(infr):
        """ recomputes edges in infr.neg_redun_metagraph """
        for cc in infr.consistent_components():
            node = next(iter(cc))
            nid1 = infr.pos_graph.node_label(node)
            for nid2 in infr.find_neg_redun_nids_to(cc):
                if nid1 < nid2:
                    yield nid1, nid2


@six.add_metaclass(ut.ReloadingMetaclass)
class Redundancy(_RedundancyComputers):
    """ methods for dynamic redundancy book-keeping """

    # def pos_redun_edge_flag(infr, edge):
    #     """ Quickly check if edge is flagged as pos redundant """
    #     nid1, nid2 = infr.pos_graph.node_labels(*edge)
    #     return nid1 == nid2 and nid1 in infr.pos_redun_nids

    # def neg_redun_edge_flag(infr, edge):
    #     """ Quickly check if edge is flagged as neg redundant """
    #     nid1, nid2 = infr.pos_graph.node_labels(*edge)
    #     return infr.neg_redun_metagraph.has_edge(nid1, nid2)

    def is_flagged_as_redun(infr, edge):
        """
        Tests redundancy against bookkeeping structure against cache
        """
        nidu, nidv = infr.node_labels(*edge)
        if nidu == nidv:
            if nidu in infr.pos_redun_nids:
                return True
        elif nidu != nidv:
            if infr.neg_redun_metagraph.has_edge(nidu, nidv):
                return True
        return False

    def filter_edges_flagged_as_redun(infr, edges):
        """
        Returns only edges that are not flagged as redundant.
        Uses bookkeeping structures

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=1, size=4)
            >>> infr.clear_edges()
            >>> infr.ensure_cliques()
            >>> infr.clear_feedback()
            >>> print(ut.repr4(infr.status()))
            >>> nonredun_edges = list(infr.filter_edges_flagged_as_redun(
            >>>     infr.unreviewed_graph.edges()))
            >>> assert len(nonredun_edges) == 6
        """
        for edge in edges:
            if not infr.is_flagged_as_redun(edge):
                yield edge

    @profile
    def update_extern_neg_redun(infr, nid, may_add=True, may_remove=True, force=False):
        """
        Checks if `nid` is negative redundant to any other `cc` it has at least
        one negative review to.
        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        if not infr.params['redun.enabled']:
            return
        # infr.print('neg_redun external update nid={}'.format(nid), 5)
        k_neg = infr.params['redun.neg']
        cc1 = infr.pos_graph.component(nid)
        force = True
        if force:
            # TODO: non-force versions
            freqs = infr.find_neg_nid_freq_to(cc1)
            other_nids = []
            flags = []
            for other_nid, freq in freqs.items():
                if freq >= k_neg:
                    other_nids.append(other_nid)
                    flags.append(True)
                elif may_remove:
                    other_nids.append(other_nid)
                    flags.append(False)

            if len(other_nids) > 0:
                infr._set_neg_redun_flags(nid, other_nids, flags)
            else:
                infr.print('neg_redun skip update nid=%r' % (nid,), 6)

    @profile
    def update_neg_redun_to(
        infr, nid1, other_nids, may_add=True, may_remove=True, force=False
    ):
        """
        Checks if nid1 is neg redundant to other_nids.
        Edges are either removed or added to the queue appropriately.
        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        if not infr.params['redun.enabled']:
            return
        # infr.print('update_neg_redun_to', 5)
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
    def update_pos_redun(infr, nid, may_add=True, may_remove=True, force=False):
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
                    'inferred_state', ut.dzip(nxu.edges_inside(infr.graph, cc), ['same']),
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
                    'inferred_state', ut.dzip(nxu.edges_inside(infr.graph, cc), [None])
                )

    @profile
    def _set_neg_redun_flags(infr, nid1, other_nids, flags):
        """
        Flags or unflags an nid1 as negative redundant with other nids.
        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        needs_unflag = []
        needs_flag = []
        already_flagged = []
        already_unflagged = []
        cc1 = infr.pos_graph.component(nid1)
        other_nids = list(other_nids)

        # Determine what needs what
        for nid2, flag in zip(other_nids, flags):
            was_neg_redun = infr.neg_redun_metagraph.has_edge(nid1, nid2)
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
            infr.neg_redun_metagraph.add_edge(nid1, nid2)
        for nid2 in needs_unflag:
            infr.neg_redun_metagraph.remove_edge(nid1, nid2)

        # Update priorities and attributes
        if infr.params['inference.update_attrs'] or infr.queue is not None:
            all_flagged_edges = []
            # Unprioritize all edges between flagged nids
            for nid2 in it.chain(needs_flag, already_flagged):
                cc2 = infr.pos_graph.component(nid2)
                all_flagged_edges.extend(nxu.edges_cross(infr.graph, cc1, cc2))

        if infr.queue is not None or infr.params['inference.update_attrs']:
            all_unflagged_edges = []
            unrev_unflagged_edges = []
            unrev_graph = infr.unreviewed_graph
            # Reprioritize unreviewed edges between unflagged nids
            # Marked inferred state of all edges
            for nid2 in it.chain(needs_unflag, already_unflagged):
                cc2 = infr.pos_graph.component(nid2)
                if infr.queue is not None:
                    _edges = nxu.edges_cross(unrev_graph, cc1, cc2)
                    unrev_unflagged_edges.extend(_edges)
                if infr.params['inference.update_attrs']:
                    _edges = nxu.edges_cross(infr.graph, cc1, cc2)
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
    def _purge_redun_flags(infr, nid):
        """
        Removes positive and negative redundancy from nids and all other PCCs
        touching nids respectively. Return the external PCC nids.

        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        if not infr.params['redun.enabled']:
            return []
        if infr.neg_redun_metagraph.has_node(nid):
            prev_neg_nids = set(infr.neg_redun_metagraph.neighbors(nid))
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
            infr._set_neg_redun_flags(nid, [other_nid], [False])
        if nid in infr.pos_redun_nids:
            infr._set_pos_redun_flag(nid, False)
        return prev_neg_nids


@six.add_metaclass(ut.ReloadingMetaclass)
class NonDynamicUpdate(object):
    @profile
    def apply_nondynamic_update(infr, graph=None):
        r"""
        Recomputes all dynamic bookkeeping for a graph in any state.
        This ensures that subsequent dyanmic inference can be applied.

        CommandLine:
            python -m wbia.algo.graph.mixin_dynamic apply_nondynamic_update

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> num_pccs = 250
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> infr.apply_nondynamic_update()
            >>> infr.assert_neg_metagraph()

        """
        # Cluster edges by category
        ne_to_edges = infr.collapsed_meta_edges()
        categories = infr.categorize_edges(graph, ne_to_edges)

        infr.set_edge_attrs(
            'inferred_state', ut.dzip(ut.flatten(categories[POSTV].values()), ['same'])
        )
        infr.set_edge_attrs(
            'inferred_state', ut.dzip(ut.flatten(categories[NEGTV].values()), ['diff'])
        )
        infr.set_edge_attrs(
            'inferred_state', ut.dzip(ut.flatten(categories[INCMP].values()), [INCMP])
        )
        infr.set_edge_attrs(
            'inferred_state', ut.dzip(ut.flatten(categories[UNKWN].values()), [UNKWN])
        )
        infr.set_edge_attrs(
            'inferred_state', ut.dzip(ut.flatten(categories[UNREV].values()), [None])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(
                ut.flatten(categories['inconsistent_internal'].values()),
                ['inconsistent_internal'],
            ),
        )
        infr.set_edge_attrs(
            'inferred_state',
            ut.dzip(
                ut.flatten(categories['inconsistent_external'].values()),
                ['inconsistent_external'],
            ),
        )

        # Ensure bookkeeping is taken care of
        # * positive redundancy
        # * negative redundancy
        # * inconsistency
        infr.pos_redun_nids = set(infr.find_pos_redun_nids())
        infr.neg_redun_metagraph = infr._graph_cls(list(infr.find_neg_redun_nids()))

        # make a node for each PCC, and place an edge between any pccs with at
        # least one negative edge, with weight being the number of negative
        # edges. Self loops indicate inconsistency.
        infr.neg_metagraph = infr._graph_cls()
        infr.neg_metagraph.add_nodes_from(infr.pos_graph.component_labels())
        for (nid1, nid2), edges in ne_to_edges[NEGTV].items():
            infr.neg_metagraph.add_edge(nid1, nid2, weight=len(edges))

        infr.recover_graph.clear()
        nid_to_errors = {}
        for nid, intern_edges in categories['inconsistent_internal'].items():
            cc = infr.pos_graph.component_nodes(nid)
            pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False).copy()
            neg_edges = list(nxu.edges_inside(infr.neg_graph, cc))
            recover_hypothesis = dict(infr.hypothesis_errors(pos_subgraph, neg_edges))
            nid_to_errors[nid] = set(recover_hypothesis.keys())
            infr.recover_graph.add_edges_from(pos_subgraph.edges())

        # Delete old hypothesis
        infr.set_edge_attrs(
            'maybe_error', ut.dzip(ut.flatten(infr.nid_to_errors.values()), [None])
        )
        # Set new hypothesis
        infr.set_edge_attrs(
            'maybe_error', ut.dzip(ut.flatten(nid_to_errors.values()), [True])
        )
        infr.nid_to_errors = nid_to_errors

        # no longer dirty
        if graph is None:
            infr.dirty = False

    @profile
    def collapsed_meta_edges(infr, graph=None):
        """
        Collapse the grah such that each PCC is a node. Get a list of edges
        within/between each PCC.
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
                    rev_graph[key] = rev_graph[key].subgraph(nodes, dynamic=False)
                else:
                    rev_graph[key] = rev_graph[key].subgraph(nodes)

        # TODO: Rebalance union find to ensure parents is a single lookup
        # infr.pos_graph._union_find.rebalance(nodes)
        # node_to_label = infr.pos_graph._union_find.parents
        node_to_label = infr.pos_graph._union_find

        # Get reviewed edges using fast lookup structures
        ne_to_edges = {
            key: nxu.group_name_edges(rev_graph[key], node_to_label) for key in states
        }
        return ne_to_edges

    @profile
    def categorize_edges(infr, graph=None, ne_to_edges=None):
        r"""
        Non-dynamically computes the status of each edge in the graph.
        This is can be used to verify the dynamic computations and update when
        the dynamic state is lost.

        CommandLine:
            python -m wbia.algo.graph.mixin_dynamic categorize_edges --profile

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> num_pccs = 250 if ut.get_argflag('--profile') else 100
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> cat = infr.categorize_edges()
        """
        states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)

        if ne_to_edges is None:
            ne_to_edges = infr.collapsed_meta_edges(graph)

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
            assert all(
                n1 == n2 for n1, n2 in name_edges[POSTV]
            ), 'All positive edges should be internal to a PCC'
            assert len(name_edges[INCMP].intersection(incon_internal_ne)) == 0
            assert len(name_edges[UNREV].intersection(incon_internal_ne)) == 0
            assert len(name_edges[UNKWN].intersection(incon_internal_ne)) == 0
            assert all(
                n1 == n2 for n1, n2 in incon_internal_ne
            ), 'incon_internal edges should be internal to a PCC'

        # External inconsistentices are edges leaving inconsistent components
        incon_internal_nids = {n1 for n1, n2 in incon_internal_ne}
        incon_external_ne = set([])
        # Find all edges leaving an inconsistent PCC
        for key in (NEGTV,) + UNINFERABLE:
            incon_external_ne.update(
                {
                    (nid1, nid2)
                    for nid1, nid2 in name_edges[key]
                    if nid1 in incon_internal_nids or nid2 in incon_internal_nids
                }
            )
        for key in (NEGTV,) + UNINFERABLE:
            name_edges[key].difference_update(incon_external_ne)

        # Inference between names is now complete.
        # Now we expand this inference and project the labels onto the
        # annotation edges corresponding to each name edge.

        # Version of union that accepts generators
        union = lambda gen: set.union(*gen)  # NOQA

        # Find edges within consistent PCCs
        positive = {
            nid1: union(ne_to_edges[key][(nid1, nid2)] for key in (POSTV,) + UNINFERABLE)
            for nid1, nid2 in name_edges[POSTV]
        }
        # Find edges between 1-negative-redundant consistent PCCs
        negative = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)] for key in (NEGTV,) + UNINFERABLE
            )
            for nid1, nid2 in name_edges[NEGTV]
        }
        # Find edges internal to inconsistent PCCs
        incon_internal = {
            nid: union(
                ne_to_edges[key][(nid, nid)] for key in (POSTV, NEGTV,) + UNINFERABLE
            )
            for nid in incon_internal_nids
        }
        # Find edges leaving inconsistent PCCs
        incon_external = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)] for key in (NEGTV,) + UNINFERABLE
            )
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
        python -m wbia.algo.graph.mixin_dynamic
        python -m wbia.algo.graph.mixin_dynamic --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
