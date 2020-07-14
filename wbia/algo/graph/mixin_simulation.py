# -*- coding: utf-8 -*-
"""
Mixin functionality for experiments, tests, and simulations.
This includes recordings measures used to generate plots in JC's thesis.
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import utool as ut
import ubelt as ub
import pandas as pd
import itertools as it
from wbia.algo.graph import nx_utils as nxu
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN, NULL

print, rrr, profile = ut.inject2(__name__)


class SimulationHelpers(object):
    def init_simulation(
        infr,
        oracle_accuracy=1.0,
        k_redun=2,
        enable_autoreview=True,
        enable_inference=True,
        classifiers=None,
        match_state_thresh=None,
        max_outer_loops=None,
        name=None,
    ):
        infr.print('INIT SIMULATION', color='yellow')

        infr.name = name
        infr.simulation_mode = True

        infr.verifiers = classifiers
        infr.params['inference.enabled'] = enable_inference
        infr.params['autoreview.enabled'] = enable_autoreview

        infr.params['redun.pos'] = k_redun
        infr.params['redun.neg'] = k_redun

        # keeps track of edges where the decision != the groundtruth
        infr.mistake_edges = set()

        infr.queue = ut.PriorityQueue()

        infr.oracle = UserOracle(oracle_accuracy, rng=infr.name)

        if match_state_thresh is None:
            match_state_thresh = {
                POSTV: 1.0,
                NEGTV: 1.0,
                INCMP: 1.0,
            }

        pb_state_thresh = None
        if pb_state_thresh is None:
            pb_state_thresh = {
                'pb': 0.5,
                'notpb': 0.9,
            }

        infr.task_thresh = {
            'photobomb_state': pd.Series(pb_state_thresh),
            'match_state': pd.Series(match_state_thresh),
        }
        infr.params['algo.max_outer_loops'] = max_outer_loops

    def init_test_mode(infr):
        from wbia.algo.graph import nx_dynamic_graph

        infr.print('init_test_mode')
        infr.test_mode = True
        # infr.edge_truth = {}
        infr.metrics_list = []
        infr.test_state = {
            'n_decision': 0,
            'n_algo': 0,
            'n_manual': 0,
            'n_true_merges': 0,
            'n_error_edges': 0,
            'confusion': None,
        }
        infr.test_gt_pos_graph = nx_dynamic_graph.DynConnGraph()
        infr.test_gt_pos_graph.add_nodes_from(infr.aids)
        infr.nid_to_gt_cc = ut.group_items(infr.aids, infr.orig_name_labels)
        infr.node_truth = ut.dzip(infr.aids, infr.orig_name_labels)

        # infr.real_n_pcc_mst_edges = sum(
        #     len(cc) - 1 for cc in infr.nid_to_gt_cc.values())
        # ut.cprint('real_n_pcc_mst_edges = %r' % (
        #     infr.real_n_pcc_mst_edges,), 'red')

        infr.metrics_list = []
        infr.nid_to_gt_cc = ut.group_items(infr.aids, infr.orig_name_labels)
        infr.real_n_pcc_mst_edges = sum(len(cc) - 1 for cc in infr.nid_to_gt_cc.values())
        infr.print(
            'real_n_pcc_mst_edges = %r' % (infr.real_n_pcc_mst_edges,), color='red'
        )

    def measure_error_edges(infr):
        for edge, data in infr.edges(data=True):
            true_state = data['truth']
            pred_state = data.get('evidence_decision', UNREV)
            if pred_state != UNREV:
                if true_state != pred_state:
                    error = ut.odict([('real', true_state), ('pred', pred_state)])
                    yield edge, error

    @profile
    def measure_metrics(infr):
        real_pos_edges = []

        n_true_merges = infr.test_state['n_true_merges']
        confusion = infr.test_state['confusion']

        n_tp = confusion[POSTV][POSTV]
        confusion[POSTV]
        columns = set(confusion.keys())
        reviewd_cols = columns - {UNREV}
        non_postv = reviewd_cols - {POSTV}
        non_negtv = reviewd_cols - {NEGTV}

        n_fn = sum(ut.take(confusion[POSTV], non_postv))
        n_fp = sum(ut.take(confusion[NEGTV], non_negtv))

        n_error_edges = sum(
            confusion[r][c] + confusion[c][r] for r, c in ut.combinations(reviewd_cols, 2)
        )
        # assert n_fn + n_fp == n_error_edges

        pred_n_pcc_mst_edges = n_true_merges

        if 0:
            import ubelt as ub

            for timer in ub.Timerit(10):
                with timer:
                    # Find undetectable errors
                    num_undetectable_fn = 0
                    for nid1, nid2 in infr.neg_redun_metagraph.edges():
                        cc1 = infr.pos_graph.component(nid1)
                        cc2 = infr.pos_graph.component(nid2)
                        neg_edges = nxu.edges_cross(infr.neg_graph, cc1, cc2)
                        for u, v in neg_edges:
                            real_nid1 = infr.node_truth[u]
                            real_nid2 = infr.node_truth[v]
                            if real_nid1 == real_nid2:
                                num_undetectable_fn += 1
                                break

                    # Find undetectable errors
                    num_undetectable_fp = 0
                    for nid in infr.pos_redun_nids:
                        cc = infr.pos_graph.component(nid)
                        if not ut.allsame(ut.take(infr.node_truth, cc)):
                            num_undetectable_fp += 1

            print('num_undetectable_fn = %r' % (num_undetectable_fn,))
            print('num_undetectable_fp = %r' % (num_undetectable_fp,))

        if 0:
            n_error_edges2 = 0
            n_fn2 = 0
            n_fp2 = 0
            for edge, data in infr.edges(data=True):
                decision = data.get('evidence_decision', UNREV)
                true_state = infr.edge_truth[edge]
                if true_state == decision and true_state == POSTV:
                    real_pos_edges.append(edge)
                elif decision != UNREV:
                    if true_state != decision:
                        n_error_edges2 += 1
                        if true_state == POSTV:
                            n_fn2 += 1
                        elif true_state == NEGTV:
                            n_fp2 += 1
            assert n_error_edges2 == n_error_edges
            assert n_tp == len(real_pos_edges)
            assert n_fn == n_fn2
            assert n_fp == n_fp2
            # pred_n_pcc_mst_edges2 = sum(
            #     len(cc) - 1 for cc in infr.test_gt_pos_graph.connected_components()
            # )
        if False:
            import networkx as nx

            # set(infr.test_gt_pos_graph.edges()) == set(real_pos_edges)
            pred_n_pcc_mst_edges = 0
            for cc in nx.connected_components(nx.Graph(real_pos_edges)):
                pred_n_pcc_mst_edges += len(cc) - 1
            assert n_true_merges == pred_n_pcc_mst_edges

        # Find all annotations involved in a mistake
        assert n_error_edges == len(infr.mistake_edges)
        direct_mistake_aids = {a for edge in infr.mistake_edges for a in edge}
        mistake_nids = set(infr.node_labels(*direct_mistake_aids))
        mistake_aids = set(
            ut.flatten([infr.pos_graph.component(nid) for nid in mistake_nids])
        )

        pos_acc = pred_n_pcc_mst_edges / infr.real_n_pcc_mst_edges
        metrics = {
            'n_decision': infr.test_state['n_decision'],
            'n_manual': infr.test_state['n_manual'],
            'n_algo': infr.test_state['n_algo'],
            'phase': infr.loop_phase,
            'pos_acc': pos_acc,
            'n_merge_total': infr.real_n_pcc_mst_edges,
            'n_merge_remain': infr.real_n_pcc_mst_edges - n_true_merges,
            'n_true_merges': n_true_merges,
            'recovering': infr.is_recovering(),
            # 'recovering2': infr.test_state['recovering'],
            'merge_remain': 1 - pos_acc,
            'n_mistake_aids': len(mistake_aids),
            'frac_mistake_aids': len(mistake_aids) / len(infr.aids),
            'n_mistake_nids': len(mistake_nids),
            'n_errors': n_error_edges,
            'n_fn': n_fn,
            'n_fp': n_fp,
            'refresh_support': len(infr.refresh.manual_decisions),
            'pprob_any': infr.refresh.prob_any_remain(),
            'mu': infr.refresh._ewma,
            'test_action': infr.test_state['test_action'],
            'action': infr.test_state.get('action', None),
            'user_id': infr.test_state['user_id'],
            'pred_decision': infr.test_state['pred_decision'],
            'true_decision': infr.test_state['true_decision'],
            'n_neg_redun': infr.neg_redun_metagraph.number_of_edges(),
            'n_neg_redun1': (
                infr.neg_metagraph.number_of_edges()
                - infr.neg_metagraph.number_of_selfloops()
            ),
        }

        return metrics

    def _print_previous_loop_statistics(infr, count):
        # Print stats about what happend in the this loop
        history = infr.metrics_list[-count:]
        recover_blocks = ut.group_items(
            [
                (k, sum(1 for i in g))
                for k, g in it.groupby(ut.take_column(history, 'recovering'))
            ]
        ).get(True, [])
        infr.print(
            ('Recovery mode entered {} times, ' 'made {} recovery decisions.').format(
                len(recover_blocks), sum(recover_blocks)
            ),
            color='green',
        )
        testaction_hist = ut.dict_hist(ut.take_column(history, 'test_action'))
        infr.print(
            'Test Action Histogram: {}'.format(ut.repr4(testaction_hist, si=True)),
            color='yellow',
        )
        if infr.params['inference.enabled']:
            action_hist = ut.dict_hist(
                ut.emap(frozenset, ut.take_column(history, 'action'))
            )
            infr.print(
                'Inference Action Histogram: {}'.format(ub.repr2(action_hist, si=True)),
                color='yellow',
            )
        infr.print(
            'Decision Histogram: {}'.format(
                ut.repr2(ut.dict_hist(ut.take_column(history, 'pred_decision')), si=True)
            ),
            color='yellow',
        )
        infr.print(
            'User Histogram: {}'.format(
                ut.repr2(ut.dict_hist(ut.take_column(history, 'user_id')), si=True)
            ),
            color='yellow',
        )

    @profile
    def _dynamic_test_callback(infr, edge, decision, prev_decision, user_id):
        was_gt_pos = infr.test_gt_pos_graph.has_edge(*edge)

        # prev_decision = infr.get_edge_attr(edge, 'decision', default=UNREV)
        # prev_decision = list(infr.edge_decision_from([edge]))[0]

        true_decision = infr.edge_truth[edge]

        was_within_pred = infr.pos_graph.are_nodes_connected(*edge)
        was_within_gt = infr.test_gt_pos_graph.are_nodes_connected(*edge)
        was_reviewed = prev_decision != UNREV
        is_within_gt = was_within_gt
        was_correct = prev_decision == true_decision

        is_correct = true_decision == decision
        # print('prev_decision = {!r}'.format(prev_decision))
        # print('decision = {!r}'.format(decision))
        # print('true_decision = {!r}'.format(true_decision))

        test_print = ut.partial(infr.print, level=2)

        def test_print(x, **kw):
            infr.print('[ACTION] ' + x, level=2, **kw)

        # test_print = lambda *a, **kw: None  # NOQA

        if 0:
            num = infr.recover_graph.number_of_components()
            old_data = infr.get_nonvisual_edge_data(edge)
            # print('old_data = %s' % (ut.repr4(old_data, stritems=True),))
            print('n_prev_reviews = %r' % (old_data['num_reviews'],))
            print('prev_decision = %r' % (prev_decision,))
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
            test_print('UNDID GOOD POSITIVE EDGE', color='darkred')
            infr.test_gt_pos_graph.remove_edge(*edge)
            is_within_gt = infr.test_gt_pos_graph.are_nodes_connected(*edge)

        split_gt = is_within_gt != was_within_gt
        if split_gt:
            test_print('SPLIT A GOOD MERGE', color='darkred')
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
            confusion[true_decision][prev_decision] -= 1
            confusion[true_decision][decision] += 1
        else:
            confusion[true_decision][decision] += 1

        test_action = None
        action_color = None

        if is_correct:
            # CORRECT DECISION
            if was_reviewed:
                if prev_decision == decision:
                    test_action = 'correct duplicate'
                    action_color = 'yellow'
                else:
                    infr.mistake_edges.remove(edge)
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
            infr.mistake_edges.add(edge)
            if was_reviewed:
                if prev_decision == decision:
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
        infr.test_state['recovering'] = infr.recover_graph.has_node(
            edge[0]
        ) or infr.recover_graph.has_node(edge[1])

        infr.test_state['n_decision'] += 1
        if user_id.startswith('algo'):
            infr.test_state['n_algo'] += 1
        elif user_id.startswith('user') or user_id == 'oracle':
            infr.test_state['n_manual'] += 1
        else:
            raise AssertionError('unknown user_id=%r' % (user_id,))

        test_print(test_action, color=action_color)
        assert test_action is not None, 'what happened?'


class UserOracle(object):
    def __init__(oracle, accuracy, rng):
        if isinstance(rng, six.string_types):
            rng = sum(map(ord, rng))
        rng = ut.ensure_rng(rng, impl='python')

        if isinstance(accuracy, tuple):
            oracle.normal_accuracy = accuracy[0]
            oracle.recover_accuracy = accuracy[1]
        else:
            oracle.normal_accuracy = accuracy
            oracle.recover_accuracy = accuracy
        # .5

        oracle.rng = rng
        oracle.states = {POSTV, NEGTV, INCMP}

    def review(oracle, edge, truth, infr, accuracy=None):
        feedback = {
            'user_id': 'user:oracle',
            'confidence': 'absolutely_sure',
            'evidence_decision': None,
            'meta_decision': NULL,
            'timestamp_s1': ut.get_timestamp('int', isutc=True),
            'timestamp_c1': ut.get_timestamp('int', isutc=True),
            'timestamp_c2': ut.get_timestamp('int', isutc=True),
            'tags': [],
        }
        is_recovering = infr.is_recovering()

        if accuracy is None:
            if is_recovering:
                accuracy = oracle.recover_accuracy
            else:
                accuracy = oracle.normal_accuracy

        # The oracle can get anything where the hardness is less than its
        # accuracy

        hardness = oracle.rng.random()
        error = accuracy < hardness

        if error:
            error_options = list(oracle.states - {truth} - {INCMP})
            observed = oracle.rng.choice(list(error_options))
        else:
            observed = truth
        if accuracy < 1.0:
            feedback['confidence'] = 'pretty_sure'
        if accuracy < 0.5:
            feedback['confidence'] = 'guessing'
        feedback['evidence_decision'] = observed
        if error:
            infr.print(
                'ORACLE ERROR real={} pred={} acc={:.2f} hard={:.2f}'.format(
                    truth, observed, accuracy, hardness
                ),
                2,
                color='red',
            )

            # infr.print(
            #     'ORACLE ERROR edge={}, truth={}, pred={}, rec={}, hardness={:.3f}'.format(edge, truth, observed, is_recovering, hardness),
            #     2, color='red')
        return feedback


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.mixin_simulation
        python -m wbia.algo.graph.mixin_simulation --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
