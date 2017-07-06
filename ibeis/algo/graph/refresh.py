# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
from ibeis.algo.graph.state import (POSTV, NEGTV, INCMP, UNREV, NULL)  # NOQA
print, rrr, profile = ut.inject2(__name__)


class RefreshCriteria(object):
    """
    Determine when to re-query for candidate edges
    """
    def __init__(refresh, window=100, patience=50, thresh=.1, method='poisson'):
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
        refresh.method = method

    def clear(refresh):
        refresh.manual_decisions = []
        refresh._ewma = 1
        refresh.num_pos = 0
        refresh.num_meaningful = 0

    def is_id_complete(refresh):
        return (refresh.num_meaningful == 0)

    def check(refresh):
        if not refresh.enabled:
            return False
        return refresh.prob_any_remain() < refresh._prob_any_remain_thresh

    def prob_any_remain(refresh, n_remain_edges=None):
        """
        CommandLine:
            python -m ibeis.algo.graph.mixin_loops prob_any_remain \
                    --num_pccs=40 --size=2 --patience=20 --window=20 --show

            python -m ibeis.algo.graph.mixin_loops prob_any_remain \
                    --method=poisson --num_pccs=40 --size=2 --patience=20 --window=20 --show

            python -m ibeis.algo.graph.mixin_loops prob_any_remain \
                    --method=binomial --num_pccs=40 --size=2 --patience=20 --window=20 --show

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
            >>>     {'window': 50, 'patience': 4, 'thresh': np.exp(-2), 'method': 'poisson'})
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
            >>>     refresh.add(y, user_id='user:oracle')
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
        poisson_prob_exactly_k_events(0, mu)
        1 - poisson_prob_more_than_k_events(0, mu)

        poisson_prob_exactly_k_events(0, mu) ** a
        poisson_prob_exactly_k_events(0, mu * a)

        poisson_prob_at_most_k_events(1, lam)
        poisson_prob_more_than_k_events(1, lam)

        poisson_prob_more_than_k_events(0, lam)
        poisson_prob_exactly_k_events(0, lam)

        (1 - poisson_prob_more_than_k_events(0, mu)) ** a
        (1 - poisson_prob_more_than_k_events(0, mu * a))

        import scipy.stats
        p = scipy.stats.distributions.poisson.pmf(0, lam)
        p = scipy.stats.distributions.poisson.pmf(0, lam)
        assert p == poisson_prob_exactly_k_events(0, lam)
        """
        import scipy as sp

        def poisson_prob_exactly_k_events(k, lam):
            return np.exp(-lam) * (lam ** k) / sp.math.factorial(k)

        def poisson_prob_at_most_k_events(k, lam):
            """ this is the cdf """
            k_ = int(np.floor(k))
            return np.exp(-lam) * sum((lam ** i) / sp.math.factorial(i) for i in range(0, k_ + 1))
            # return sp.special.gammaincc(k_ + 1, lam) / sp.math.factorial(k_)

        def poisson_prob_more_than_k_events(k, lam):
            k_ = int(np.floor(k))
            return sp.special.gammainc(k_ + 1, lam) / sp.math.factorial(k_)

        a = refresh._patience
        mu = refresh._ewma

        if refresh.method == 'poisson':
            lam = a * mu
            prob_no_event_in_range = np.exp(-lam)
            prob_no_event_in_range = poisson_prob_exactly_k_events(0, lam)
        elif refresh.method == 'binomial':
            prob_no_event_in_range = (1 - mu) ** a
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
            >>>     refresh.add(y, user_id='user:oracle')
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

    def add(refresh, meaningful, user_id, decision=None):
        if not refresh.enabled:
            return

        if user_id is not None and not user_id.startswith('algo'):
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
            >>>     refresh.add(y, user_id='user:oracle')
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.graph.refresh
        python -m ibeis.algo.graph.refresh --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
