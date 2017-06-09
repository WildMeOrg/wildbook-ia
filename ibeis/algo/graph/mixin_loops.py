# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import pandas as pd
import itertools as it
from ibeis.algo.graph.state import (POSTV, NEGTV, INCMP, UNREV)
print, rrr, profile = ut.inject2(__name__)


class RefreshCriteria(object):
    """
    Determine when to re-query for candidate edges
    """
    def __init__(refresh, window=100, patience=50, thresh=.1):
        refresh.manual_decisions = []
        refresh.num_pos = 0
        refresh.num_meaningful = 0
        # refresh.frac_thresh = 3 / refresh.window
        # refresh.pos_thresh = 2
        refresh.window = window
        refresh._patience = patience
        refresh._prob_any_remain_thresh = thresh
        # refresh._prob_none_remain_thresh = thresh
        # refresh._ewma = None
        refresh._ewma = 1
        refresh.enabled = True

    def check(refresh):
        # if len(refresh.manual_decisions) > refresh.warmup:
        # return (refresh.pos_frac < refresh.frac_thresh and
        #         refresh.num_pos > refresh.pos_thresh)
        # return refresh._prob_none_remain() > refresh._prob_none_remain_thresh
        if not refresh.enabled:
            return False
        return refresh.prob_any_remain() < refresh._prob_any_remain_thresh

    def prob_any_remain(refresh, n_remain_edges=None):
        """
        CommandLine:
            python -m ibeis.algo.graph.mixin_loops prob_any_remain \
                    --num_pccs=40 --size=2 --patience=20 --window=20 --show

            python -m ibeis.algo.graph.mixin_loops prob_any_remain \
                    --num_pccs=40 --size=2 --patience=20 --window=20 \
                    --dpi=300 --figsize=7.4375,3.0 \
                    --dpath=~/latex/crall-thesis-2017 \
                    --save=figures5/poisson.png \
                    --diskshow

            --save poisson2.png \
                    --dpi=300 --figsize=7.4375,3.0 --diskshow --size=2

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_loops import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> demokw = ut.argparse_dict({'num_pccs': 50, 'size': 4})
            >>> refreshkw = ut.argparse_dict(
            >>>     {'window': 50, 'patience': 4, 'thresh': np.exp(-2)})
            >>> infr = demo.demodata_infr(size_std=0, **demokw)
            >>> edges = list(infr.dummy_matcher.find_candidate_edges(K=100))
            >>> scores = np.array(infr.dummy_matcher.predict_edges(edges))
            >>> print('edges = %r' % (ut.hash_data(edges),))
            >>> print('scores = %r' % (ut.hash_data(scores),))
            >>> sortx = scores.argsort()[::-1]
            >>> edges = ut.take(edges, sortx)
            >>> scores = scores[sortx]
            >>> ys = infr.match_state_df(edges)[POSTV].values
            >>> y_remainsum = ys[::-1].cumsum()[::-1]
            >>> refresh = RefreshCriteria(**refreshkw)
            >>> pprob_any = []
            >>> rfrac_any = []
            >>> n_real_list = []
            >>> xdata = []
            >>> for count, (edge, y) in enumerate(zip(edges, ys)):
            >>>     refresh.add(y, user_id='oracle')
            >>>     n_remain_edges = len(edges) - count
            >>>     n_real = y_remainsum[count]
            >>>     n_real_list.append(n_real)
            >>>     rfrac_any.append(y_remainsum[count] / y_remainsum[0])
            >>>     pprob_any.append(refresh.prob_any_remain())
            >>>     xdata.append(count + 1)
            >>>     if refresh.check():
            >>>         break
            >>> rprob_any = np.minimum(1, n_real_list)
            >>> rfrac_any = rfrac_any
            >>> xdata = xdata
            >>> ydatas = ut.odict([
            >>>     ('Est. probability any remain', pprob_any),
            >>>     #('real any remain', rprob_any),
            >>>     ('Fraction remaining', rfrac_any),
            >>> ])
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> from ibeis.scripts.thesis import TMP_RC
            >>> import matplotlib as mpl
            >>> mpl.rcParams.update(TMP_RC)
            >>> pt.multi_plot(
            >>>     xdata, ydatas,
            >>>     xlabel='# manual reviews', #ylabel='prob',
            >>>     rcParams=TMP_RC, marker='', ylim=(0, 1),
            >>>     use_legend=False,# legend_loc='upper right'
            >>> )
            >>> demokw = ut.map_keys({'num_pccs': '#PCC', 'size': 'PCC size'}, demokw)
            >>> thresh = refreshkw.pop('thresh')
            >>> refreshkw['span'] = refreshkw.pop('window')
            >>> pt.relative_text((.02, .58 + .0), ut.get_cfg_lbl(demokw, sep=' ')[1:], valign='bottom')
            >>> pt.relative_text((.02, .68 + .0), ut.get_cfg_lbl(refreshkw, sep=' ')[1:], valign='bottom')
            >>> legend = pt.gca().legend()
            >>> legend.get_frame().set_alpha(1.0)
            >>> pt.plt.plot([xdata[0], xdata[-1]], [thresh, thresh], 'g--', label='thresh')
            >>> ut.show_if_requested()

        Sympy:
            import sympy as sym
            mu, a, k = sym.symbols(['mu', 'a', 'k'])
            k = 0
            prob_no_event = sym.exp(-mu) * (mu ** k) / sym.factorial(k)
            prob_no_event ** a

        """
        prob_no_event_in_range = refresh._prob_none_remain(n_remain_edges)
        prob_event_in_range = 1 - prob_no_event_in_range
        return prob_event_in_range

    def _prob_none_remain(refresh, n_remain_edges=None):
        """
        mu = .3
        a = 3
        poisson_prob_k_events(0, mu)
        1 - poisson_prob_at_least_k_events(0, mu)

        poisson_prob_k_events(0, mu) ** a
        poisson_prob_k_events(0, mu * a)

        (1 - poisson_prob_at_least_k_events(0, mu)) ** a
        (1 - poisson_prob_at_least_k_events(0, mu * a))
        """
        import scipy as sp
        mu = refresh._ewma
        def poisson_prob_k_events(k, mu):
            return np.exp(-mu) * (mu ** k) / sp.math.factorial(k)
        # def  poisson_prob_at_least_k_events(k, mu):
        #     return sp.special.gammainc(k + 1, mu) / sp.math.factorial(k)
        a = refresh._patience
        prob_no_event_in_range = poisson_prob_k_events(0, mu * a)
        return prob_no_event_in_range

    def pred_num_positives(refresh, n_remain_edges):
        """
        Uses poisson process to estimate remaining positive reviews.

        Multipling mu * n_remain_edges gives a probabilistic upper bound on the
        number of errors remaning.  This only provides a real estimate if
        reviewing in a random order

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_loops import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=50, size=4, size_std=2)
            >>> edges = list(infr.dummy_matcher.find_candidate_edges(K=100))
            >>> #edges = ut.shuffle(sorted(edges), rng=321)
            >>> scores = np.array(infr.dummy_matcher.predict_edges(edges))
            >>> sortx = scores.argsort()[::-1]
            >>> edges = ut.take(edges, sortx)
            >>> scores = scores[sortx]
            >>> ys = infr.match_state_df(edges)[POSTV].values
            >>> y_remainsum = ys[::-1].cumsum()[::-1]
            >>> refresh = RefreshCriteria(window=250)
            >>> n_pred_list = []
            >>> n_real_list = []
            >>> xdata = []
            >>> for count, (edge, y) in enumerate(zip(edges, ys)):
            >>>     refresh.add(y, user_id='oracle')
            >>>     n_remain_edges = len(edges) - count
            >>>     n_pred = refresh.pred_num_positives(n_remain_edges)
            >>>     n_real = y_remainsum[count]
            >>>     if count == 2000:
            >>>         break
            >>>     n_real_list.append(n_real)
            >>>     n_pred_list.append(n_pred)
            >>>     xdata.append(count + 1)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> n_pred_list = n_pred_list[10:]
            >>> n_real_list = n_real_list[10:]
            >>> xdata = xdata[10:]
            >>> pt.multi_plot(xdata, [n_pred_list, n_real_list], marker='',
            >>>               label_list=['pred', 'real'], xlabel='review num',
            >>>               ylabel='pred remaining merges')
            >>> stop_point = xdata[np.where(y_remainsum[10:] == 0)[0][0]]
            >>> pt.gca().plot([stop_point, stop_point], [0, int(max(n_pred_list))], 'g-')
            >>> #idx = np.where(((scores == scores[ys].min()) & y)[10:])[0]
            >>> #idx = np.where((scores[10:] == 0) & ys[10:])[0]
            >>> #practical_point = np.where(y_remainsum == 0)[0][0]
            >>> #pt.gca().plot([stop_point, stop_point], [0, int(max(n_pred_list))], 'g-')
        """
        # variance and mean are the same
        mu = refresh._ewma
        # import scipy.stats
        # mu = refresh.pos_frac
        # rv = scipy.stats.poisson(mu)
        # sigma = np.sqrt(mu)
        # support = len(refresh.manual_decisions)
        # prob_at_least_k_events(1, mu)
        n_positives = mu * n_remain_edges
        return n_positives

    def clear(refresh):
        refresh.manual_decisions = []
        refresh.num_pos = 0
        refresh.num_meaningful = 0

    def add(refresh, meaningful, user_id, decision=None):
        if not refresh.enabled:
            return

        if user_id is not None and not user_id.startswith('auto'):
            refresh.manual_decisions.append(meaningful)
            m = meaningful
            # span corresponds roughly to window size
            # http://greenteapress.com/thinkstats2/html/thinkstats2013.html
            span = refresh.window
            alpha = 2 / (span + 1)
            refresh._ewma = (alpha * m) + (1 - alpha) * refresh._ewma
        refresh.num_meaningful += meaningful
        if decision == POSTV:
            refresh.num_pos += 1

    # def add(refresh, decision, user_id):
    #     decision_code = 1 if decision == POSTV else 0
    #     # dont add auto reviews to refresh criteria
    #     if user_id is not None and not user_id.startswith('auto'):
    #         refresh.manual_decisions.append(decision_code)
    #         x = decision_code
    #         # span corresponds roughly to window size
    #         # http://greenteapress.com/thinkstats2/html/thinkstats2013.html
    #         span = refresh.window
    #         alpha = 2 / (span + 1)
    #         if refresh._ewma is None:
    #             refresh._ewma = x
    #         refresh._ewma = (alpha * x) + (1 - alpha) * refresh._ewma
    #     if decision_code:
    #         refresh.num_pos += 1

    def ave(refresh, method='exp'):
        """
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_loops import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=40, size=4, size_std=2, ignore_pair=True)
            >>> edges = list(infr.dummy_matcher.find_candidate_edges(K=100))
            >>> scores = np.array(infr.dummy_matcher.predict_edges(edges))
            >>> #sortx = ut.shuffle(np.arange(len(edges)), rng=321)
            >>> sortx = scores.argsort()[::-1]
            >>> edges = ut.take(edges, sortx)
            >>> scores = scores[sortx]
            >>> ys = infr.match_state_df(edges)[POSTV].values
            >>> y_remainsum = ys[::-1].cumsum()[::-1]
            >>> refresh = RefreshCriteria(window=250)
            >>> ma1 = []
            >>> ma2 = []
            >>> reals = []
            >>> xdata = []
            >>> for count, (edge, y) in enumerate(zip(edges, ys)):
            >>>     refresh.add(y, user_id='oracle')
            >>>     ma1.append(refresh._ewma)
            >>>     ma2.append(refresh.pos_frac)
            >>>     n_real = y_remainsum[count] / (len(edges) - count)
            >>>     reals.append(n_real)
            >>>     xdata.append(count + 1)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> pt.multi_plot(xdata, [ma1, ma2, reals], marker='',
            >>>               label_list=['exp', 'win', 'real'], xlabel='review num',
            >>>               ylabel='mu')
        """
        if method == 'exp':
            # Compute exponentially weighted moving average
            span = refresh.window
            alpha = 2 / (span + 1)
            # Compute the whole thing
            iter_ = iter(refresh.manual_decisions)
            current = next(iter_)
            for x in iter_:
                current = (alpha * x) + (1 - alpha) * current
            return current
        elif method == 'window':
            return refresh.pos_frac

    @property
    def pos_frac(refresh):
        return np.mean(refresh.manual_decisions[-refresh.window:])


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

    def review(oracle, edge, truth, infr):
        feedback = {
            'user_id': 'oracle',
            'confidence': 'absolutely_sure',
            'decision': None,
            'tags': [],
        }
        is_recovering = infr.is_recovering()
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
        feedback['decision'] = observed
        if error:

            infr.print(
                '[ORACLE] MADE ERROR edge=%r, truth=%r, observed=%r, '
                'rec=%r, hardness=%r' % (
                    edge, truth, observed, is_recovering, hardness),
                2, color='red')
        return feedback


class InfrLoops(object):
    """
    Algorithm control flow loops
    """

    def groundtruth_split_loop(infr):
        # TODO
        pass

    @profile
    def groundtruth_merge_loop(infr):
        """
        Finds edges to make sure the ground truth is merged
        """
        infr.print('==============================')
        infr.print('--- GROUNDTRUTH MERGE LOOP ---')
        assert infr.test_mode, 'only run this in test mode'

        from ibeis.algo.graph import nx_utils
        group = ut.group_items(infr.aids, infr.orig_name_labels)
        fix_edges = []

        # Tell the oracle its time to get serious
        # infr.oracle.normal_accuracy = 1.0
        # infr.oracle.recover_accuracy = 1.0

        for gt_nid, aids in group.items():
            pos_sub = infr.pos_graph.subgraph(aids)
            aug_edges = nx_utils.edge_connected_augmentation(
                pos_sub, 1, return_anyway=True)
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

    def pos_redun_loop(infr):
        infr.print('===========================')
        infr.print('--- POSITIVE REDUN LOOP ---')
        new_edges = infr.find_pos_redun_candidate_edges()
        print('pos_redun_candidates = %r' % (len(new_edges),))
        infr.queue.clear()
        infr.add_candidate_edges(new_edges)
        infr.refresh.enabled = False
        infr.inner_priority_loop(use_refresh=False)

    @profile
    def rereview_nonconf_auto(infr):
        infr.print('=========================')
        infr.print('--- REREVIEW NONCONF AUTO')
        # Enforce that a user checks any PCC that was auto-reviewed
        # but was unable to achieve k-positive-consistency
        for pcc in list(infr.non_pos_redundant_pccs(relax_size=False)):
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

    @profile
    def recovery_review_loop(infr, verbose=1):
        if verbose:
            infr.print('=============================')
            infr.print('--- RECOVERY REVEIEW LOOP ---')
        while infr.is_recovering():
            edge, priority = infr.pop()
            try:
                feedback = infr.request_user_review(edge)
            except ReviewCanceled:
                # Place edge back on the queue
                if not infr.is_redundant(edge):
                    infr.queue[edge] = priority
                continue
            infr.add_feedback(edge=edge, **feedback)

    @profile
    def inner_priority_loop(infr, use_refresh=True):
        """
        Executes reviews until the queue is empty or needs refresh
        """
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
            try:
                infr.next_review()
            except StopIteration:
                assert len(infr.queue) == 0
                infr.print('No more edges after %d iterations, need refresh' %
                           (count,), 1, color='yellow')
                break

    def lnbnn_priority_loop(infr, use_refresh=True):
        infr.print('============================')
        infr.print('--- LNBNN PRIORITY LOOP ---')
        infr.refresh_candidate_edges()
        infr.refresh = RefreshCriteria(**infr._refresh_params)
        infr.inner_priority_loop(use_refresh)

    @profile
    def main_loop(infr, max_loops=None, use_refresh=True):
        """
        The main outer loop
        """
        infr.print('Starting main loop', 1)
        if max_loops is None:
            max_loops = infr._max_outer_loops
            if max_loops is None:
                max_loops = np.inf

        # Initialize a refresh criteria
        for count in it.count(0):
            if count >= max_loops:
                infr.print('early stop', 1, color='red')
                break
            infr.print('Outer loop iter %d ' % (count,))
            # Do priority loop over lnbnn candidates
            infr.lnbnn_priority_loop(use_refresh)
            print('prob_any_remain = %r' % (infr.refresh.prob_any_remain(),))

            terminate = (infr.refresh.num_meaningful == 0)
            print('infr.refresh.num_meaningful = %r' % (infr.refresh.num_meaningful,))
            if terminate:
                infr.print('Triggered termination criteria', 1, color='red')

            if infr.enable_redundancy:
                # Fix positive redundancy of anything within the loop
                infr.pos_redun_loop()

            print('prob_any_remain = %r' % (infr.refresh.prob_any_remain(),))
            print('infr.refresh.num_meaningful = %r' % (infr.refresh.num_meaningful,))

            if terminate:
                break

        # if infr.enable_redundancy:
        #     # Do a final fixup, check work of auto criteria
        #     infr.refresh.enabled = False
        #     # infr.rereview_nonconf_auto()
        #     print('infr.refresh.num_meaningful = %r' % (infr.refresh.num_meaningful,))
        #     # infr.recovery_review_loop()

        infr.print('Terminate.', 1, color='red')

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
            infr.print('Must manually review inconsistent edge', 3)
            return False, review
        # Determine if anything passes the match threshold
        primary_task = 'match_state'
        # data = infr.get_nonvisual_edge_data(edge)
        # decision_probs = infr.task_probs[primary_task].loc[edge]
        # decision_probs = pd.Series(data['task_probs'][primary_task])

        if False:
            decision_probs = pd.Series(infr.task_probs[primary_task][edge])
            a, b = decision_probs.align(infr.task_thresh[primary_task])
            decision_flags = a > b
            hasone = sum(decision_flags) == 1
            # decision = decision_flags.argmax()
        else:
            # TODO: don't autodecide if secondary classifiers are on
            decision_probs = infr.task_probs[primary_task][edge]
            primary_thresh = infr.task_thresh[primary_task]
            decision_flags = {k: decision_probs[k] > thresh
                              for k, thresh in primary_thresh.items()}
            hasone = sum(decision_flags.values()) == 1
            # decision = ut.argmax(decision_probs)
        # decision_probs > infr.task_thresh[primary_task]
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
                decision = ut.argmax(decision_probs)
                review['decision'] = decision
                truth = infr.match_state_gt(edge)
                if review['decision'] != truth:
                    infr.print('AUTOMATIC ERROR edge=%r, truth=%r, decision=%r' %
                               (edge, truth, review['decision']), 2, color='darkred')
                auto_flag = True
        if auto_flag and infr.verbose > 1:
            infr.print('Automatic review success')

        return auto_flag, review

    @profile
    def emit_or_review(infr, edge, priority):
        if infr.enable_autoreview:
            flag, feedback = infr.try_auto_review(edge)
            if flag:
                return feedback
        if infr.simulation_mode:
            feedback = infr.request_oracle_review(edge)
            return feedback
        else:
            infr.emit_manual_review(edge, priority)
            return None

    def qt_review_loop(infr, cfgdict=None):
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
            >>> cfgdict = {'ratio_thresh': .8, 'sv_on': False}
            >>> # Add dummy priorities to each edge
            >>> infr.set_edge_attrs('prob_match', ut.dzip(infr.edges(), [1]))
            >>> infr.prioritize('prob_match', infr.edges(), reset=True)
            >>> infr.enable_redundancy = False
            >>> win = infr.qt_review_loop(cfgdict=cfgdict)
            >>> import guitool as gt
            >>> gt.qtapp_loop(qwin=win, freq=10)
        """
        import guitool as gt
        gt.ensure_qapp()
        from ibeis.viz import viz_graph2
        infr.manual_wgt = viz_graph2.AnnotPairDialog(
            infr=infr, standalone=False, cfgdict=cfgdict)
        infr.manual_wgt.accepted.connect(infr.on_accept)
        infr.manual_wgt.skipped.connect(infr.continue_review)
        infr.manual_wgt.request.connect(infr.emit_manual_review)
        infr.continue_review()
        return infr.manual_wgt

    def emit_manual_review(infr, edge, priority=None):
        edge_data = infr.get_nonvisual_edge_data(edge, on_missing='default').copy()
        edge_data['nid_edge'] = infr.pos_graph.node_labels(*edge)
        edge_data['n_ccs'] = (
            len(infr.pos_graph.connected_to(edge[0])),
            len(infr.pos_graph.connected_to(edge[1]))
        )
        info_text = 'priority=%r' % (priority,)
        info_text += '\n' + ut.repr4(edge_data)
        infr.manual_wgt.set_edge(edge, info_text, external=True)
        infr.manual_wgt.show()

    @profile
    def next_review(infr):
        """
        does one review step
        """
        edge, priority = infr.pop()
        feedback = infr.emit_or_review(edge, priority)
        if feedback is None:
            # None feedback means we are waiting for a user response.
            return False
        # Add feedback from the automated method
        infr.add_feedback(edge, priority=priority, **feedback)
        return True

    def on_accept(infr, feedback, need_next=True):
        annot1_state = feedback.pop('annot1_state', None)
        annot2_state = feedback.pop('annot2_state', None)
        if annot1_state:
            infr.add_node_feedback(**annot1_state)
        if annot2_state:
            infr.add_node_feedback(**annot2_state)
        infr.add_feedback(**feedback)
        infr.write_ibeis_staging_feedback()
        if need_next:
            infr.continue_review()

    def continue_review(infr):
        try:
            while True:
                if not infr.next_review():
                    break
        except StopIteration:
            infr.on_queue_empty()

    def on_queue_empty(infr):
        # TODO: refresh criteria
        if infr.manual_wgt is not None:
            if infr.manual_wgt.isVisible():
                import guitool as gt
                gt.user_info(infr.manual_wgt, 'Review Complete')
        print('review lop complete')

    def manual_review(infr, edge):
        # OLD
        from ibeis.viz import viz_graph2
        dlg = viz_graph2.AnnotPairDialog.as_dialog(
            infr=infr, edge=edge, standalone=False)
        # dlg.resize(700, 500)
        dlg.exec_()
        if dlg.widget.was_confirmed:
            feedback = dlg.widget.feedback_dict()
            feedback.pop('edge', None)
        else:
            raise ReviewCanceled('user canceled')
        dlg.close()
        # raise NotImplementedError('no user review')
        pass

    @profile
    def request_oracle_review(infr, edge):
        truth = infr.match_state_gt(edge)
        feedback = infr.oracle.review(edge, truth, infr)
        return feedback

    def request_user_review(infr, edge):
        if infr.simulation_mode:
            feedback = infr.request_oracle_review(edge)
        else:
            feedback = infr.manual_review(edge)
        return feedback


class SimulationHelpers(object):
    def init_simulation(infr, oracle_accuracy=1.0, k_redun=2,
                        enable_autoreview=True, enable_inference=True,
                        classifiers=None, match_state_thresh=None,
                        max_outer_loops=None, name=None):
        infr.print('INIT SIMULATION', color='yellow')

        infr.name = name
        infr.simulation_mode = True

        infr.classifiers = classifiers
        infr.enable_inference = enable_inference
        infr.enable_autoreview = enable_autoreview

        infr.queue_params['pos_redun'] = k_redun
        infr.queue_params['neg_redun'] = k_redun

        infr.queue = ut.PriorityQueue()

        infr.oracle = UserOracle(oracle_accuracy, rng=infr.name)

        infr.task_thresh = {
            'photobomb_state': pd.Series({
                'pb': .5,
                'notpb': .9,
            }),
            'match_state': pd.Series(match_state_thresh)
        }
        infr._max_outer_loops = max_outer_loops

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
            pred_state = data.get('decision', UNREV)
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
            import ubelt
            for timer in ubelt.Timerit(10):
                with timer:
                    # Find undetectable errors
                    num_undetectable_fn = 0
                    from ibeis.algo.graph import nx_utils
                    for nid1, nid2 in infr.neg_redun_nids.edges():
                        cc1 = infr.pos_graph.component(nid1)
                        cc2 = infr.pos_graph.component(nid2)
                        neg_edges = nx_utils.edges_cross(infr.neg_graph, cc1, cc2)
                        for u, v in neg_edges:
                            real_nid1 = infr.node_truth[u]
                            real_nid2 = infr.node_truth[v]
                            if real_nid1 == real_nid2:
                                num_undetectable_fn += 1
                                break

                    # Find undetectable errors
                    num_undetectable_fp = 0
                    from ibeis.algo.graph import nx_utils
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
                decision = data.get('decision', UNREV)
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
            'merge_remain': 1 - pos_acc,
            'n_errors': n_error_edges,
            'n_fn': n_fn,
            'n_fp': n_fp,
            'pprob_any': infr.refresh.prob_any_remain(),
            'mu': infr.refresh._ewma,
            'action': infr.test_state['action'],
            'user_id': infr.test_state['user_id'],
            'pred_decision': infr.test_state['pred_decision'],
            'true_decision': infr.test_state['true_decision'],
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
