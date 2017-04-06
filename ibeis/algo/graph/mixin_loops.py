# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import pandas as pd
import itertools as it
from ibeis.algo.graph.state import (POSTV, NEGTV, INCMP, UNREV)
print, rrr, profile = ut.inject2(__name__)


class TerminationCriteria2(object):
    def __init__(term, phis):
        term.phis = phis


class RefreshCriteria2(object):
    """
    Determine when to re-query for candidate edges
    """
    def __init__(refresh):
        refresh.window = 100
        refresh.manual_decisions = []
        refresh.num_pos = 0
        refresh.frac_thresh = 3 / refresh.window
        refresh.pos_thresh = 2

    def reset(refresh):
        refresh.manual_decisions = []
        refresh.num_pos = 0

    def add(refresh, decision, user_id):
        decision_code = 1 if decision == POSTV else 0
        if user_id is not None and not user_id.startswith('auto'):
            refresh.manual_decisions.append(decision_code)
        if decision_code:
            refresh.num_pos += 1

    @property
    def pos_frac(refresh):
        return np.mean(refresh.manual_decisions[-refresh.window:])

    def check(refresh):
        return (refresh.pos_frac < refresh.frac_thresh and
                refresh.num_pos > refresh.pos_thresh)


class UserOracle(object):
    def __init__(oracle, accuracy, rng):
        if isinstance(rng, six.string_types):
            rng = sum(map(ord, rng))
        rng = ut.ensure_rng(rng, impl='python')

        oracle.accuracy = accuracy
        oracle.rng = rng
        oracle.states = {POSTV, NEGTV, INCMP}

    def review(oracle, edge, truth, force=False):
        feedback = {
            'user_id': 'oracle',
            'confidence': 'absolutely_sure',
            'decision': None,
            'tags': [],
        }
        error = oracle.accuracy < oracle.rng.random()
        if force:
            error = False
        if error:
            error_options = list(oracle.states - {truth} - {INCMP})
            observed = oracle.rng.choice(list(error_options))
        else:
            observed = truth
        if oracle.accuracy < 1.0:
            feedback['confidence'] = 'pretty_sure'
        if oracle.accuracy < .5:
            feedback['confidence'] = 'guessing'
        feedback['decision'] = observed
        if error:
            ut.cprint('MADE MANUAL ERROR edge=%r, truth=%r, observed=%r' %
                      (edge, truth, observed), 'red')
        return feedback


class InfrLoops(object):
    """
    Algorithm control flow loops
    """

    def inner_loop(infr):
        """
        Executes reviews until the queue is empty or needs refresh
        """

        infr.print('Start inner loop')
        for count in it.count(0):
            if len(infr.queue) == 0:
                infr.print('No more edges, need refresh')
                break
            edge, priority = infr.pop()
            if infr.is_recovering():
                infr.print('IN RECOVERY MODE priority=%r' % (priority,),
                           color='red')
            else:
                if infr.refresh.check():
                    infr.print('Refresh criteria flags refresh')
                    break

            flag = False
            if infr.enable_autoreview:
                flag, feedback = infr.try_auto_review(edge)
            if not flag:
                # if infr.enable_inference:
                #     flag, feedback = infr.try_implicit_review(edge)
                if not flag:
                    feedback = infr.request_user_review(edge)
            infr.add_feedback(edge=edge, **feedback)

            if infr.is_recovering():
                infr.recovery_review_loop()

    def recovery_review_loop(infr):
        while infr.is_recovering():
            edge, priority = infr.pop()
            num_reviews = infr.get_edge_attr(edge, 'num_reviews', default=0)
            try:
                feedback = infr.request_user_review(edge)
            except ReviewCanceled:
                if not infr.is_redundant(edge):
                    infr.queue[edge] = priority
                continue
            infr.print(
                'RECOVERY LOOP edge={}, decision={}, priority={}, '
                'n_reviews={}, len(recover_ccs)={}'.format(
                    edge, feedback['decision'], priority, num_reviews,
                    len(infr.recovery_ccs)),
                color='red'
            )
            infr.add_feedback(edge=edge, **feedback)

    def priority_review_loop(infr, max_loops):
        infr.refresh = RefreshCriteria2()
        for count in it.count(0):
            if count >= max_loops:
                infr.print('early stop')
                break
            infr.print('Outer loop iter %d ' % (count,))
            infr.refresh_candidate_edges()
            if not len(infr.queue):
                infr.print('Queue is empty. Terminate.')
                break
            infr.inner_loop()
            if infr.enable_inference:
                infr.assert_consistency_invariant()
                infr.print('HACK FIX REDUN', color='white')
                # Fix anything that is not positive/negative redundant
                real_queue = infr.queue
                # use temporary queue
                infr.queue = ut.PriorityQueue()
                infr.refresh_candidate_edges(ranking=False)
                infr.inner_loop()
                infr.queue = real_queue

    @profile
    def main_loop(infr, max_loops=np.inf):
        infr.print('Starting main loop', 1)
        infr.reset(state='empty')
        infr.remove_feedback(apply=True)

        infr.priority_review_loop(max_loops)

        if infr.enable_inference:
            # Enforce that a user checks any PCC that was auto-reviewed
            # but was unable to achieve k-positive-consistency
            for pcc in list(infr.non_pos_redundant_pccs()):
                subgraph = infr.graph.subgraph(pcc)
                for u, v, data in subgraph.edges(data=True):
                    edge = infr.e_(u, v)
                    if data.get('user_id', '').startswith('auto'):
                        try:
                            feedback = infr.request_user_review(edge)
                        except ReviewCanceled:
                            raise
                        infr.add_feedback(edge=edge, **feedback)
            # Check for inconsistency recovery
            infr.recovery_review_loop()

        if infr.enable_inference:
            infr.assert_consistency_invariant()
        # true_groups = list(map(set, infr.nid_to_gt_cc.values()))
        # pred_groups = list(infr.positive_connected_compoments())
        # from ibeis.algo.hots import simulate
        # comparisons = simulate.compare_groups(true_groups, pred_groups)
        # pred_merges = comparisons['pred_merges']
        # print(pred_merges)
        infr.print('Exiting main loop')


class ReviewCanceled(Exception):
    pass


class InfrReviewers(object):
    @profile
    def try_auto_review(infr, edge):
        review = {
            'user_id': 'auto_clf',
            'confidence': 'pretty_sure',
            'decision': None,
            'tags': [],
        }
        if infr.is_recovering():
            # Do not autoreview if we are in an inconsistent state
            infr.print('Must manually review inconsistent edge', 1)
            return False, review
        # Determine if anything passes the match threshold
        primary_task = 'match_state'
        data = infr.get_edge_data(*edge)
        decision_probs = pd.Series(data['task_probs'][primary_task])
        a, b = decision_probs.align(infr.task_thresh[primary_task])
        decision_flags = a > b
        # decision_probs > infr.task_thresh[primary_task]
        auto_flag = False
        if sum(decision_flags) == 1:
            # Check to see if it might be confounded by a photobomb
            pb_probs = data['task_probs']['photobomb_state']
            pb_thresh = infr.task_thresh['photobomb_state']['pb']
            confounded = pb_probs['pb'] > pb_thresh
            if not confounded:
                review['decision'] = decision_flags.argmax()
                truth = infr.match_state_gt(edge).idxmax()
                if review['decision'] != truth:
                    infr.print('AUTOMATIC ERROR edge=%r, truth=%r, decision=%r' %
                               (edge, truth, review['decision']), color='purple')
                auto_flag = True
        if auto_flag and infr.verbose > 1:
            infr.print('Automatic review success')

        return auto_flag, review

    def try_implicit_review(infr, edge):
        review = {}
        # Check if edge is implicitly negative
        if not infr.is_recovering():
            implicit_flag = (
                infr.check_prob_completeness(edge[0]) and
                infr.check_prob_completeness(edge[1])
            )
            if implicit_flag:
                review = {
                    'user_id': 'auto_implicit_complete',
                    'confidence': 'pretty_sure',
                    'decision': NEGTV,
                    'tags': [],
                }
        else:
            implicit_flag = False
        return implicit_flag, review

    def request_user_review(infr, edge):
        if infr.test_mode:
            true_state = infr.match_state_gt(edge)
            truth = true_state.idxmax()
            feedback = infr.oracle.review(
                edge, truth, infr.is_recovering())
        else:
            from ibeis.viz import viz_graph2
            dlg = viz_graph2.AnnotPairDialog.as_dialog(
                infr=infr, edge=edge, hack_write=False)
            # dlg.resize(700, 500)
            dlg.exec_()
            if dlg.widget.was_confirmed:
                feedback = dlg.widget.feedback_dict()
                feedback.pop('edge', None)
            else:
                raise ReviewCanceled('user canceled')
            # raise NotImplementedError('no user review')
        return feedback


class SimulationHelpers(object):
    def init_simulation(infr, oracle_accuracy=1.0, k_redun=2,
                        enable_autoreview=True, enable_inference=True,
                        classifiers=None, phis=None, complete_thresh=None,
                        match_state_thresh=None, name=None):

        infr.name = name

        infr.print('INIT SIMULATION', color='yellow')

        infr.classifiers = classifiers
        infr.enable_inference = enable_inference
        infr.enable_autoreview = enable_autoreview

        infr.queue_params['pos_redundancy'] = k_redun
        infr.queue_params['neg_redundancy'] = k_redun
        infr.queue_params['complete_thresh'] = complete_thresh

        infr.queue = ut.PriorityQueue()

        infr.oracle = UserOracle(oracle_accuracy, infr.name)
        infr.term = TerminationCriteria2(phis)

        infr.task_thresh = {
            'photobomb_state': pd.Series({
                'pb': .5,
                'notpb': .9,
            }),
            'match_state': pd.Series(match_state_thresh)
        }

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
        infr.print('real_n_pcc_mst_edges = %r' % (
            infr.real_n_pcc_mst_edges,), color='red')

    def measure_error_edges(infr):
        for edge, data in infr.edges(data=True):
            true_state = data['truth']
            pred_state = data.get('decision', UNREV)
            if pred_state != UNREV:
                if true_state != pred_state:
                    error = ut.odict([('real', true_state),
                                      ('pred', pred_state)])
                    yield edge, error

    @profile
    def measure_metrics(infr):
        real_pos_edges = []
        n_error_edges = 0
        pred_n_pcc_mst_edges = 0
        n_fn = 0
        n_fp = 0

        # TODO: dynamic measurement

        for edge, data in infr.edges(data=True):
            true_state = infr.edge_truth[edge]
            decision = data.get('decision', UNREV)
            if true_state == decision and true_state == POSTV:
                real_pos_edges.append(edge)
            elif decision != UNREV:
                if true_state != decision:
                    n_error_edges += 1
                    if true_state == POSTV:
                        n_fn += 1
                    elif true_state == NEGTV:
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.graph.loops
        python -m ibeis.algo.graph.loops --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
