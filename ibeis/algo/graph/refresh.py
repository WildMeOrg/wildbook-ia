# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
from ibeis.algo.graph.state import (POSTV, NEGTV, INCMP, UNREV, NULL)  # NOQA
print, rrr, profile = ut.inject2(__name__)


class RefreshCriteria(object):
    """
    Determine when to re-query for candidate edges.

    Models an upper bound on the probability that any of the next `patience`
    reviews will be label-changing (meaningful). Once this probability is below
    a threshold the criterion triggers. The model is either binomial or
    poisson.  They both work about the same. The binomial is a slightly better
    model.

    Does this by maintaining an estimate of the probability any particular
    review will be label-chaging using an exponentially weighted moving
    average. This is the rate parameter / individual event probability.

    """
    def __init__(refresh, window=20, patience=72, thresh=.1,
                 method='binomial'):
        refresh.window = window
        refresh._patience = patience
        refresh._prob_any_remain_thresh = thresh
        refresh.method = method
        refresh.manual_decisions = []
        refresh.num_meaningful = 0
        refresh._ewma = 1
        refresh.enabled = True

    def clear(refresh):
        refresh.manual_decisions = []
        refresh._ewma = 1
        refresh.num_meaningful = 0

    def check(refresh):
        if not refresh.enabled:
            return False
        return refresh.prob_any_remain() < refresh._prob_any_remain_thresh

    def prob_any_remain(refresh, n_remain_edges=None):
        """
        """
        prob_no_event_in_range = refresh._prob_none_remain(n_remain_edges)
        prob_event_in_range = 1 - prob_no_event_in_range
        return prob_event_in_range

    def _prob_none_remain(refresh, n_remain_edges=None):
        import scipy as sp

        def poisson_prob_exactly_k_events(k, lam):
            return np.exp(-lam) * (lam ** k) / sp.math.factorial(k)

        def poisson_prob_at_most_k_events(k, lam):
            """ this is the cdf """
            k_ = int(np.floor(k))
            return np.exp(-lam) * sum((lam ** i) / sp.math.factorial(i)
                                      for i in range(k_ + 1))
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
        else:
            raise KeyError('refresh.method = {!r}'.format(refresh.method))

        return prob_no_event_in_range

    def pred_num_positives(refresh, n_remain_edges):
        """
        Uses poisson process to estimate remaining positive reviews.

        Multipling mu * n_remain_edges gives a probabilistic upper bound on the
        number of errors remaning.  This only provides a real estimate if
        reviewing in a random order

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.refresh import *  # NOQA
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

    def ave(refresh, method='exp'):
        """
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.refresh import *  # NOQA
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


def demo_refresh():
    r"""
    CommandLine:
        python -m ibeis.algo.graph.refresh demo_refresh \
                --num_pccs=40 --size=2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.graph.refresh import *  # NOQA
        >>> demo_refresh()
        >>> ut.show_if_requested()
    """
    from ibeis.algo.graph import demo
    demokw = ut.argparse_dict({'num_pccs': 50, 'size': 4})
    refreshkw = ut.argparse_funckw(RefreshCriteria)
    # make an inference object
    infr = demo.demodata_infr(size_std=0, **demokw)
    edges = list(infr.dummy_matcher.find_candidate_edges(K=100))
    scores = np.array(infr.dummy_matcher.predict_edges(edges))
    sortx = scores.argsort()[::-1]
    edges = ut.take(edges, sortx)
    scores = scores[sortx]
    ys = infr.match_state_df(edges)[POSTV].values
    y_remainsum = ys[::-1].cumsum()[::-1]
    # Do oracle reviews and wait to converge
    refresh = RefreshCriteria(**refreshkw)
    xdata = []
    pprob_any = []
    rfrac_any = []
    for count, (edge, y) in enumerate(zip(edges, ys)):
        refresh.add(y, user_id='user:oracle')
        rfrac_any.append(y_remainsum[count] / y_remainsum[0])
        pprob_any.append(refresh.prob_any_remain())
        xdata.append(count + 1)
        if refresh.check():
            break
    xdata = xdata
    ydatas = ut.odict([
        ('Est. probability any remain', pprob_any),
        ('Fraction remaining', rfrac_any),
    ])

    ut.quit_if_noshow()
    import plottool as pt
    pt.qtensure()
    from ibeis.scripts.thesis import TMP_RC
    import matplotlib as mpl
    mpl.rcParams.update(TMP_RC)
    pt.multi_plot(
        xdata, ydatas, xlabel='# manual reviews', rcParams=TMP_RC, marker='',
        ylim=(0, 1), use_legend=False,
    )
    demokw = ut.map_keys({'num_pccs': '#PCC', 'size': 'PCC size'},
                         demokw)
    thresh = refreshkw.pop('thresh')
    refreshkw['span'] = refreshkw.pop('window')
    pt.relative_text((.02, .58 + .0), ut.get_cfg_lbl(demokw, sep=' ')[1:],
                     valign='bottom')
    pt.relative_text((.02, .68 + .0), ut.get_cfg_lbl(refreshkw, sep=' ')[1:],
                     valign='bottom')
    legend = pt.gca().legend()
    legend.get_frame().set_alpha(1.0)
    pt.plt.plot([xdata[0], xdata[-1]], [thresh, thresh], 'g--', label='thresh')


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
