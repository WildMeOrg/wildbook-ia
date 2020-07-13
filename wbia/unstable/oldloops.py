# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def groundtruth_split_loop(infr):
    # TODO
    pass


def groundtruth_merge_loop(infr):
    """
    Finds edges to make sure the ground truth is merged
    """
    from wbia.algo.graph import nx_utils as nxu

    infr.print('==============================', color='white')
    infr.print('--- GROUNDTRUTH MERGE LOOP ---', color='white')
    assert infr.test_mode, 'only run this in test mode'

    group = ut.group_items(infr.aids, infr.orig_name_labels)
    fix_edges = []

    # Tell the oracle its time to get serious
    # infr.oracle.normal_accuracy = 1.0
    # infr.oracle.recover_accuracy = 1.0

    for gt_nid, aids in group.items():
        pos_sub = infr.pos_graph.subgraph(aids)
        aug_edges = nxu.edge_augmentation(pos_sub, k=1, partial=True)
        fix_edges.extend(aug_edges)

    if infr.test_mode:
        infr.ensure_edges_from(fix_edges)
        infr.apply_edge_truth(fix_edges)

    for edge in fix_edges:
        try:
            feedback = infr.request_user_review(edge)
        except ReviewCanceled:
            raise
        infr.add_feedback(edge=edge, **feedback)
        infr.recovery_review_loop(verbose=0)


def rereview_nonconf_auto(infr):
    infr.print('=========================', color='white')
    infr.print('--- REREVIEW NONCONF AUTO', color='white')
    # Enforce that a user checks any PCC that was auto-reviewed
    # but was unable to achieve k-positive-consistency
    for pcc in list(infr.non_pos_redundant_pccs(relax=False)):
        subgraph = infr.graph.subgraph(pcc)
        for u, v, data in subgraph.edges(data=True):
            edge = infr.e_(u, v)
            if data.get('user_id', '').startswith('auto'):
                try:
                    feedback = infr.request_user_review(edge)
                except ReviewCanceled:
                    raise
                infr.add_feedback(edge=edge, **feedback)
        infr.recovery_review_loop(verbose=0)


def recovery_review_loop(infr, verbose=1):
    if verbose:
        infr.print('=============================', color='white')
        infr.print('--- RECOVERY REVEIEW LOOP ---', color='white')
    while infr.is_recovering():
        edge, priority = infr.peek()
        try:
            feedback = infr.request_user_review(edge)
        except ReviewCanceled:
            # Place edge back on the queue
            if not infr.is_redundant(edge):
                infr.push(edge, priority)
            continue
        infr.add_feedback(edge=edge, **feedback)


class ReviewCanceled(Exception):
    pass

    # def request_user_review(infr, edge):
    #     if infr.simulation_mode:
    #         feedback = infr.request_oracle_review(edge)
    #     else:
    #         feedback = infr.manual_review(edge)
    #     return feedback

    # def manual_review(infr, edge):
    #     # OLD
    #     from wbia.viz import viz_graph2
    #     dlg = viz_graph2.AnnotPairDialog.as_dialog(
    #         infr=infr, edge=edge, standalone=False)
    #     # dlg.resize(700, 500)
    #     dlg.exec_()
    #     if dlg.widget.was_confirmed:
    #         feedback = dlg.widget.feedback_dict()
    #         feedback.pop('edge', None)
    #     else:
    #         raise ReviewCanceled('user canceled')
    #     dlg.close()
    #     # raise NotImplementedError('no user review')
    #     pass
