# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import pandas as pd
import itertools as it
from ibeis.algo.graph import nx_utils as nxu
from ibeis.algo.graph.state import (POSTV, NEGTV, INCMP, UNREV, NULL)
from ibeis.algo.graph.refresh import RefreshCriteria
print, rrr, profile = ut.inject2(__name__)


class InfrLoops(object):
    """
    Algorithm control flow loops
    """

    def main_gen(infr, max_loops=None, use_refresh=True):
        """
        The main outer loop

        Doctest:
            >>> from ibeis.algo.graph.mixin_loops import *
            >>> import ibeis
            >>> infr = ibeis.AnnotInference('testdb1', aids='all',
            >>>                             autoinit='staging', verbose=4)
            >>> infr.params['manual.n_peek'] = 10
            >>> infr.params['ranking.ntop'] = 1
            >>> infr.oracle = UserOracle(.99, rng=0)
            >>> infr.simulation_mode = False
            >>> infr.reset()
            >>> #infr.load_published()
            >>> gen = infr.main_gen()
            >>> while True:
            >>>     try:
            >>>         reviews = next(gen)
            >>>         edge, priority, data = reviews[0]
            >>>         feedback = infr.request_oracle_review(edge)
            >>>         infr.add_feedback(edge, **feedback)
            >>>     except StopIteration:
            >>>         break
        """
        infr.print('Starting main loop', 1)
        infr.print('infr.params = {}'.format(ut.repr3(infr.params)))
        if max_loops is None:
            max_loops = infr.params['algo.max_outer_loops']
            if max_loops is None:
                max_loops = np.inf

        if infr.test_mode:
            print('------------------ {} -------------------'.format(infr.name))

        # Initialize a refresh criteria
        infr.init_refresh()

        # Phase 0.1: Ensure the user sees something immediately
        if infr.params['algo.quickstart']:
            # quick startup. Yield a bunch of random edges
            num = infr.params['manual.n_peek']
            user_request = []
            for edge in ut.random_combinations(infr.aids, 2, num=num):
                user_request += [infr._make_review_tuple(edge, None)]
                yield user_request

        if infr.params['algo.hardcase']:
            # Check previously labeled edges that where the groundtruth and the
            # verifier disagree.
            for _ in infr.hardcase_review_gen():
                yield _

        # Phase 0.2: Ensure positive redundancy (this is generally quick)
        # so the user starts seeing real work after one random review is made
        # unless the graph is already positive redundant.
        if infr.params['redun.enabled'] and infr.params['redun.enforce_pos']:
            # Fix positive redundancy of anything within the loop
            for _ in infr.pos_redun_gen():
                yield _

        if infr.params['ranking.enabled']:
            for count in it.count(0):

                infr.print('Outer loop iter %d ' % (count,))

                # Phase 1: Try to merge PCCs by searching for LNBNN candidates
                for _ in infr.lnbnn_priority_gen(use_refresh):
                    yield _

                terminate = (infr.refresh.num_meaningful == 0)
                if terminate:
                    infr.print('Triggered break criteria', 1, color='red')

                # Phase 2: Ensure positive redundancy.
                if all(ut.take(infr.params, ['redun.enabled', 'redun.enforce_pos'])):
                    # Fix positive redundancy of anything within the loop
                    for _ in infr.pos_redun_gen():
                        yield _

                print('prob_any_remain = %r' % (infr.refresh.prob_any_remain(),))
                print('infr.refresh.num_meaningful = {!r}'.format(
                    infr.refresh.num_meaningful))

                if (count + 1) >= max_loops:
                    infr.print('early stop', 1, color='red')
                    break

                if terminate:
                    infr.print('break triggered')
                    break

        if all(ut.take(infr.params, ['redun.enabled', 'redun.enforce_neg'])):
            # Phase 3: Try to automatically acheive negative redundancy without
            # asking the user to do anything but resolve inconsistency.
            infr.print('Entering phase 3', 1, color='red')
            for _ in infr.neg_redun_gen():
                yield _

        infr.print('Terminate', 1, color='red')
        infr.print('Exiting main loop')

        if infr.params['inference.enabled']:
            infr.assert_consistency_invariant()

    def hardcase_review_gen(infr):
        """
        Re-review non-confident edges that vsone did not classify correctly
        """
        infr.print('==============================', color='white')
        infr.print('--- HARDCASE PRIORITY LOOP ---', color='white')
        verifiers = infr.learn_evaluation_verifiers()
        verif = verifiers['match_state']

        edges_ = list(infr.edges())
        real_ = list(infr.edge_decision_from(edges_))
        flags_ = [r in {POSTV, NEGTV, INCMP} for r in real_]
        real = ut.compress(real_, flags_)
        edges = ut.compress(edges_, flags_)

        hardness = 1 - verif.easiness(edges, real)

        if True:
            df = pd.DataFrame({'edges': edges, 'real': real})
            df['hardness'] = hardness

            pred = verif.predict(edges)
            df['pred'] = pred.values

            df.sort_values('hardness', ascending=False)
            infr.print('hardness analysis')
            infr.print(str(df))

            infr.print('infr status: ' + ut.repr4(infr.status()))

        # Don't re-review anything that was confidently reviewed
        # CONFIDENCE = infr.ibs.const.CONFIDENCE
        # CODE_TO_INT = CONFIDENCE.CODE_TO_INT.copy()
        # CODE_TO_INT[CONFIDENCE.CODE.UNKNOWN] = 0
        # conf = ut.take(CODE_TO_INT, infr.gen_edge_values(
        #     'confidence', edges, on_missing='default',
        #     default=CONFIDENCE.CODE.UNKNOWN))

        # This should only be run with certain params
        assert not infr.params['autoreview.enabled']
        assert not infr.params['redun.enabled']
        assert not infr.params['ranking.enabled']
        assert infr.params['inference.enabled']
        # infr.ibs.const.CONFIDENCE.CODE.PRETTY_SURE
        if infr.params['queue.conf.thresh'] is None:
            # != 'pretty_sure':
            infr.print('WARNING: should queue.conf.thresh = "pretty_sure"?')

        # work around add_candidate_edges
        infr.prioritize(metric='hardness', edges=edges,
                        scores=hardness)
        infr.set_edge_attrs('hardness', ut.dzip(edges, hardness))
        for _ in infr.inner_priority_gen(use_refresh=False):
            yield _

    def lnbnn_priority_gen(infr, use_refresh=True):
        infr.print('============================', color='white')
        infr.print('--- LNBNN PRIORITY LOOP ---', color='white')
        n_prioritized = infr.refresh_candidate_edges()
        if n_prioritized == 0:
            infr.print('LNBNN FOUND NO NEW EDGES')
            return
        if use_refresh:
            infr.refresh.clear()
        for _ in infr.inner_priority_gen(use_refresh):
            yield _

    def pos_redun_gen(infr):
        infr.print('===========================', color='white')
        infr.print('--- POSITIVE REDUN LOOP ---', color='white')
        new_edges = list(infr.find_pos_redun_candidate_edges())
        for count in it.count(0):
            infr.print('check pos-redun iter {}'.format(count))
            infr.queue.clear()
            infr.add_candidate_edges(new_edges)

            gen = infr.inner_priority_gen(use_refresh=False)
            # yield from gen
            for value in gen:
                yield value

            new_edges = list(infr.find_pos_redun_candidate_edges())
            if len(new_edges) == 0:
                infr.print(
                    'pos-redundancy achieved in {} iterations'.format(
                        count + 1))
                break
            infr.print('not pos-reduntant yet.', color='white')

    def neg_redun_gen(infr):
        infr.print('===========================', color='white')
        infr.print('--- NEGATIVE REDUN LOOP ---', color='white')
        import ubelt as ub

        infr.queue.clear()

        only_auto = infr.params['redun.neg.only_auto']

        for new_edges in ub.chunks(infr.find_neg_redun_candidate_edges(), 100):
            # Add chunks in a little at a time for faster response time
            infr.add_candidate_edges(new_edges)
            gen = infr.inner_priority_gen(use_refresh=False,
                                          only_auto=only_auto)
            # yield from gen
            for value in gen:
                yield value

    def inner_priority_gen(infr, use_refresh=False, only_auto=False):
        """
        Executes reviews until the queue is empty or needs refresh

        Args:
            user_refresh (bool): if True enables the refresh criteria.
                (set to True in Phase 1)
            only_auto (bool) if True, then the user wont be prompted with
                reviews unless the graph is inconsistent.
                (set to True in Phase 3)
        """
        if infr.refresh:
            infr.refresh.enabled = use_refresh
        infr.print('Start inner loop with {} items in the queue'.format(
            len(infr.queue)))
        for count in it.count(0):
            if infr.is_recovering():
                infr.print('Still recovering after %d iterations' % (count,),
                           3, color='turquoise')
            else:
                # Do not check for refresh if we are recovering
                if use_refresh and infr.refresh.check():
                    infr.print('Triggered refresh criteria after %d iterations' %
                               (count,), 1, color='yellow')
                    break

            # If the queue is empty break
            if len(infr.queue) == 0:
                infr.print('No more edges after %d iterations, need refresh' %
                           (count,), 1, color='yellow')
                break

            # Try to automatically do the next review.
            edge, priority = infr.peek()
            infr.print('next_review. edge={}'.format(edge), 100)

            inconsistent = infr.is_recovering(edge)

            feedback = None
            if infr.params['autoreview.enabled'] and not inconsistent:
                # Try to autoreview if we aren't in an inconsistent state
                feedback = infr.try_auto_review(edge)

            if feedback is not None:
                # Add feedback from the automated method
                infr.add_feedback(edge, priority=priority, **feedback)
            else:
                # We can't automatically review, ask for help
                if only_auto and not inconsistent:
                    # We are in auto only mode, skip manual review
                    # unless there is an inconsistency
                    infr.skip(edge)
                else:
                    if infr.simulation_mode:
                        # Use oracle feedback
                        feedback = infr.request_oracle_review(edge)
                        infr.add_feedback(edge, priority=priority, **feedback)
                    else:
                        # Yield to the user if we need to pause
                        user_request = infr.emit_manual_review(edge, priority)
                        yield user_request

        if infr.metrics_list:
            infr._print_previous_loop_statistics(count)

    def init_refresh(infr):
        refresh_params = infr.subparams('refresh')
        infr.refresh = RefreshCriteria(**refresh_params)

    def start_id_review(infr, max_loops=None, use_refresh=None):
        assert infr._gen is None, 'algo already running'
        # Just exhaust the main generator
        infr._gen = infr.main_gen(max_loops=max_loops, use_refresh=use_refresh)
        return infr._gen

    def main_loop(infr, max_loops=None, use_refresh=True):
        """ DEPRICATED """
        gen = infr.start_id_review(max_loops=max_loops, use_refresh=use_refresh)
        # To automatically run through the loop just exhaust the generator
        result = next(gen)
        assert result is None, 'need user interaction. cannot auto loop'


class InfrReviewers(object):
    @profile
    def try_auto_review(infr, edge):
        review = {
            'user_id': 'algo:auto_clf',
            'confidence': infr.ibs.const.CONFIDENCE.CODE.PRETTY_SURE,
            'evidence_decision': None,
            'meta_decision': NULL,
            'timestamp_s1': None,
            'timestamp_c1': None,
            'timestamp_c2': None,
            'tags': [],
        }
        if infr.is_recovering():
            # Do not autoreview if we are in an inconsistent state
            infr.print('Must manually review inconsistent edge', 3)
            return None
        # Determine if anything passes the match threshold
        primary_task = 'match_state'

        try:
            decision_probs = infr.task_probs[primary_task][edge]
        except KeyError:
            if infr.verifiers is None:
                return None
            if infr.verifiers.get(primary_task, None) is None:
                return None
            # Compute probs if they haven't been done yet
            infr.ensure_priority_scores([edge])
            try:
                decision_probs = infr.task_probs[primary_task][edge]
            except KeyError:
                return None

        primary_thresh = infr.task_thresh[primary_task]
        decision_flags = {k: decision_probs[k] > thresh
                          for k, thresh in primary_thresh.items()}
        hasone = sum(decision_flags.values()) == 1
        auto_flag = False
        if hasone:
            # Check to see if it might be confounded by a photobomb
            pb_probs = infr.task_probs['photobomb_state'][edge]
            # pb_probs = infr.task_probs['photobomb_state'].loc[edge]
            # pb_probs = data['task_probs']['photobomb_state']
            pb_thresh = infr.task_thresh['photobomb_state']['pb']
            confounded = pb_probs['pb'] > pb_thresh
            if not confounded:
                # decision = decision_flags.argmax()
                evidence_decision = ut.argmax(decision_probs)
                review['evidence_decision'] = evidence_decision
                truth = infr.match_state_gt(edge)
                if review['evidence_decision'] != truth:
                    infr.print('AUTOMATIC ERROR edge=%r, truth=%r, decision=%r' %
                               (edge, truth, review['evidence_decision']), 2,
                               color='darkred')
                auto_flag = True
        if auto_flag and infr.verbose > 1:
            infr.print('Automatic review success')

        if auto_flag:
            review
        else:
            return None

    def request_oracle_review(infr, edge, **kw):
        truth = infr.match_state_gt(edge)
        feedback = infr.oracle.review(edge, truth, infr, **kw)
        return feedback

    def _make_review_tuple(infr, edge, priority=None):
        """ Makes tuple to be sent back to the user """
        edge_data = infr.get_nonvisual_edge_data(
            edge, on_missing='default')
        # Extra information
        edge_data['nid_edge'] = infr.pos_graph.node_labels(*edge)
        edge_data['n_ccs'] = (
            len(infr.pos_graph.connected_to(edge[0])),
            len(infr.pos_graph.connected_to(edge[1]))
        )
        return (edge, priority, edge_data)

    def emit_manual_review(infr, edge, priority=None):
        """
        Emits a signal containing edges that need review. The callback should
        present them to a user, get feedback, and then call on_accpet.
        """
        infr.print('emit_manual_review', 100)
        # Emit a list of reviews that can be considered.
        # The first is the most important
        user_request = []
        user_request += [infr._make_review_tuple(edge, priority)]
        try:
            for edge_, priority in infr.peek_many(infr.params['manual.n_peek']):
                if edge == edge_:
                    continue
                user_request += [infr._make_review_tuple(edge_, priority)]
        except TypeError:
            pass

        # If registered, send the request via a callback.
        request_review = infr.callbacks.get('request_review', None)
        if request_review is not None:
            # Send these reviews to a user
            request_review(user_request)
        # Otherwise the current process must handle the request by return value
        return user_request

    def skip(infr, edge):
        infr.print('skipping edge={}'.format(edge), 100)
        try:
            del infr.queue[edge]
        except Exception:
            pass

    def accept(infr, feedback):
        """
        Called when user has completed feedback from qt or web
        """
        annot1_state = feedback.pop('annot1_state', None)
        annot2_state = feedback.pop('annot2_state', None)
        if annot1_state:
            infr.add_node_feedback(**annot1_state)
        if annot2_state:
            infr.add_node_feedback(**annot2_state)
        infr.add_feedback(**feedback)

        if infr.params['manual.autosave']:
            infr.write_ibeis_staging_feedback()

    def continue_review(infr):
        if infr._gen is None:
            return None
        try:
            user_request = next(infr._gen)
        except StopIteration:
            review_finished = infr.callbacks.get('review_finished', None)
            if review_finished is not None:
                review_finished()
            infr._gen = None
            user_request = None
        return user_request

    def qt_edge_reviewer(infr, edge=None):
        import guitool as gt
        gt.ensure_qapp()
        from ibeis.viz import viz_graph2
        infr.manual_wgt = viz_graph2.AnnotPairDialog(
            edge=edge, infr=infr, standalone=False,
            cfgdict=infr.verifier_params)
        if edge is not None:
            # infr.emit_manual_review(edge, priority=None)
            infr.manual_wgt.seek(0)
            # infr.manual_wgt.show()
        return infr.manual_wgt

    def qt_review_loop(infr):
        r"""
        TODO: The loop parts should be a non-mixin class

        Qt review loop entry point

        CommandLine:
            python -m ibeis.algo.graph.mixin_loops qt_review_loop --show

        Example:
            >>> # SCRIPT
            >>> import utool as ut
            >>> import ibeis
            >>> ibs = ibeis.opendb('PZ_MTEST')
            >>> infr = ibeis.AnnotInference(ibs, 'all', autoinit=True)
            >>> infr.ensure_mst()
            >>> # Add dummy priorities to each edge
            >>> infr.set_edge_attrs('prob_match', ut.dzip(infr.edges(), [1]))
            >>> infr.prioritize('prob_match', infr.edges(), reset=True)
            >>> infr.params['redun.enabled'] = False
            >>> win = infr.qt_review_loop()
            >>> import guitool as gt
            >>> gt.qtapp_loop(qwin=win, freq=10)
        """
        infr.qt_edge_reviewer()
        # infr.continue_review()
        return infr.manual_wgt


class SimulationHelpers(object):
    def init_simulation(infr, oracle_accuracy=1.0, k_redun=2,
                        enable_autoreview=True, enable_inference=True,
                        classifiers=None, match_state_thresh=None,
                        max_outer_loops=None, name=None):
        infr.print('INIT SIMULATION', color='yellow')

        infr.name = name
        infr.simulation_mode = True

        infr.verifiers = classifiers
        infr.params['inference.enabled'] = enable_inference
        infr.params['autoreview.enabled'] = enable_autoreview

        infr.params['redun.pos'] = k_redun
        infr.params['redun.neg'] = k_redun

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
                'pb': .5,
                'notpb': .9,
            }

        infr.task_thresh = {
            'photobomb_state': pd.Series(pb_state_thresh),
            'match_state': pd.Series(match_state_thresh)
        }
        infr.params['algo.max_outer_loops'] = max_outer_loops

    def init_test_mode(infr):
        from ibeis.algo.graph import nx_dynamic_graph
        infr.print('init_test_mode')
        infr.test_mode = True
        # infr.edge_truth = {}
        infr.metrics_list = []
        infr.test_state = {
            'n_decision': 0,
            'n_auto': 0,
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
        infr.real_n_pcc_mst_edges = sum(
            len(cc) - 1 for cc in infr.nid_to_gt_cc.values())
        infr.print('real_n_pcc_mst_edges = %r' % (
            infr.real_n_pcc_mst_edges,), color='red')

    def measure_error_edges(infr):
        for edge, data in infr.edges(data=True):
            true_state = data['truth']
            pred_state = data.get('evidence_decision', UNREV)
            if pred_state != UNREV:
                if true_state != pred_state:
                    error = ut.odict([('real', true_state),
                                      ('pred', pred_state)])
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

        n_error_edges = sum(confusion[r][c] + confusion[c][r] for r, c in
                            ut.combinations(reviewd_cols, 2))
        # assert n_fn + n_fp == n_error_edges

        pred_n_pcc_mst_edges = n_true_merges

        if 0:
            import ubelt as ub
            for timer in ub.Timerit(10):
                with timer:
                    # Find undetectable errors
                    num_undetectable_fn = 0
                    for nid1, nid2 in infr.neg_redun_nids.edges():
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

        pos_acc = pred_n_pcc_mst_edges / infr.real_n_pcc_mst_edges
        metrics = {
            'n_decision': infr.test_state['n_decision'],
            'n_manual': infr.test_state['n_manual'],
            'n_auto': infr.test_state['n_auto'],
            'pos_acc': pos_acc,
            'n_merge_total': infr.real_n_pcc_mst_edges,
            'n_merge_remain': infr.real_n_pcc_mst_edges - n_true_merges,
            'n_true_merges': n_true_merges,
            'recovering': infr.is_recovering(),
            # 'recovering2': infr.test_state['recovering'],
            'merge_remain': 1 - pos_acc,
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
        }

        return metrics

    def _print_previous_loop_statistics(infr, count):
        # Print stats about what happend in the this loop
        history = infr.metrics_list[-count:]
        recover_blocks = ut.group_items([
            (k, sum(1 for i in g))
            for k, g in it.groupby(ut.take_column(history, 'recovering'))
        ]).get(True, [])
        infr.print((
            'Recovery mode entered {} times, '
            'made {} recovery decisions.').format(
                len(recover_blocks), sum(recover_blocks)), color='green')
        testaction_hist = ut.dict_hist(ut.take_column(history, 'test_action'))
        infr.print(
            'Test Action Histogram: {}'.format(
                ut.repr4(testaction_hist, si=True)), color='yellow')
        if infr.params['inference.enabled']:
            action_hist = ut.dict_hist(
                ut.emap(frozenset, ut.take_column(history, 'action')))
            infr.print(
                'Inference Action Histogram: {}'.format(
                    ut.repr2(action_hist, si=True)), color='yellow')
        infr.print(
            'Decision Histogram: {}'.format(ut.repr2(ut.dict_hist(
                ut.take_column(history, 'pred_decision')
            ), si=True)), color='yellow')
        infr.print(
            'User Histogram: {}'.format(ut.repr2(ut.dict_hist(
                ut.take_column(history, 'user_id')
            ), si=True)), color='yellow')


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
        if accuracy < .5:
            feedback['confidence'] = 'guessing'
        feedback['evidence_decision'] = observed
        if error:
            infr.print(
                'ORACLE ERROR real={} pred={} acc={:.2f} hard={:.2f}'.format(
                    truth, observed, accuracy, hardness), 2, color='red')

            # infr.print(
            #     'ORACLE ERROR edge={}, truth={}, pred={}, rec={}, hardness={:.3f}'.format(edge, truth, observed, is_recovering, hardness),
            #     2, color='red')
        return feedback


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
