# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, NULL  # NOQA

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

    def __init__(refresh, window=20, patience=72, thresh=0.1, method='binomial'):
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
            return np.exp(-lam) * sum(
                (lam ** i) / sp.math.factorial(i) for i in range(k_ + 1)
            )
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
            >>> from wbia.algo.graph.refresh import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=50, size=4, size_std=2)
            >>> edges = list(infr.dummy_verif.find_candidate_edges(K=100))
            >>> #edges = ut.shuffle(sorted(edges), rng=321)
            >>> scores = np.array(infr.dummy_verif.predict_edges(edges))
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
            >>> import wbia.plottool as pt
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
        >>> from wbia.algo.graph.refresh import *  # NOQA
        >>> from wbia.algo.graph import demo
        >>> infr = demo.demodata_infr(num_pccs=40, size=4, size_std=2, ignore_pair=True)
        >>> edges = list(infr.dummy_verif.find_candidate_edges(K=100))
        >>> scores = np.array(infr.dummy_verif.predict_edges(edges))
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
        >>> import wbia.plottool as pt
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
        return np.mean(refresh.manual_decisions[-refresh.window :])


def demo_refresh():
    r"""
    CommandLine:
        python -m wbia.algo.graph.refresh demo_refresh \
                --num_pccs=40 --size=2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.refresh import *  # NOQA
        >>> demo_refresh()
        >>> ut.show_if_requested()
    """
    from wbia.algo.graph import demo

    demokw = ut.argparse_dict({'num_pccs': 50, 'size': 4})
    refreshkw = ut.argparse_funckw(RefreshCriteria)
    # make an inference object
    infr = demo.demodata_infr(size_std=0, **demokw)
    edges = list(infr.dummy_verif.find_candidate_edges(K=100))
    scores = np.array(infr.dummy_verif.predict_edges(edges))
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
    ydatas = ut.odict(
        [('Est. probability any remain', pprob_any), ('Fraction remaining', rfrac_any)]
    )

    ut.quit_if_noshow()
    import wbia.plottool as pt

    pt.qtensure()
    from wbia.scripts.thesis import TMP_RC
    import matplotlib as mpl

    mpl.rcParams.update(TMP_RC)
    pt.multi_plot(
        xdata,
        ydatas,
        xlabel='# manual reviews',
        rcParams=TMP_RC,
        marker='',
        ylim=(0, 1),
        use_legend=False,
    )
    demokw = ut.map_keys({'num_pccs': '#PCC', 'size': 'PCC size'}, demokw)
    thresh = refreshkw.pop('thresh')
    refreshkw['span'] = refreshkw.pop('window')
    pt.relative_text(
        (0.02, 0.58 + 0.0), ut.get_cfg_lbl(demokw, sep=' ')[1:], valign='bottom'
    )
    pt.relative_text(
        (0.02, 0.68 + 0.0), ut.get_cfg_lbl(refreshkw, sep=' ')[1:], valign='bottom'
    )
    legend = pt.gca().legend()
    legend.get_frame().set_alpha(1.0)
    pt.plt.plot([xdata[0], xdata[-1]], [thresh, thresh], 'g--', label='thresh')


def _dev_iters_until_threshold():
    """
    INTERACTIVE DEVELOPMENT FUNCTION

    How many iterations of ewma until you hit the poisson / biniomal threshold

    This establishes a principled way to choose the threshold for the refresh
    criterion in my thesis. There are paramters --- moving parts --- that we
    need to work with: `a` the patience, `s` the span, and `mu` our ewma.

    `s` is a span paramter indicating how far we look back.

    `mu` is the average number of label-changing reviews in roughly the last
    `s` manual decisions.

    These numbers are used to estimate the probability that any of the next `a`
    manual decisions will be label-chanigng. When that probability falls below
    a threshold we terminate. The goal is to choose `a`, `s`, and the threshold
    `t`, such that the probability will fall below the threshold after a maximum
    of `a` consecutive non-label-chaning reviews. IE we want to tie the patience
    paramter (how far we look ahead) to how far we actually are willing to go.
    """
    import numpy as np
    import utool as ut
    import sympy as sym

    i = sym.symbols('i', integer=True, nonnegative=True, finite=True)
    # mu_i = sym.symbols('mu_i', integer=True, nonnegative=True, finite=True)
    s = sym.symbols('s', integer=True, nonnegative=True, finite=True)  # NOQA
    thresh = sym.symbols('tau', real=True, nonnegative=True, finite=True)  # NOQA
    alpha = sym.symbols('alpha', real=True, nonnegative=True, finite=True)  # NOQA
    c_alpha = sym.symbols('c_alpha', real=True, nonnegative=True, finite=True)
    # patience
    a = sym.symbols('a', real=True, nonnegative=True, finite=True)

    available_subs = {
        a: 20,
        s: a,
        alpha: 2 / (s + 1),
        c_alpha: (1 - alpha),
    }

    def subs(expr, d=available_subs):
        """ recursive expression substitution """
        expr1 = expr.subs(d)
        if expr == expr1:
            return expr1
        else:
            return subs(expr1, d=d)

    # mu is either the support for the poisson distribution
    # or is is the p in the binomial distribution
    # It is updated at timestep i based on ewma, assuming each incoming responce is 0
    mu_0 = 1.0
    mu_i = c_alpha ** i

    # Estimate probability that any event will happen in the next `a` reviews
    # at time `i`.
    poisson_i = 1 - sym.exp(-mu_i * a)
    binom_i = 1 - (1 - mu_i) ** a

    # Expand probabilities to be a function of i, s, and a
    part = ut.delete_dict_keys(available_subs.copy(), [a, s])
    mu_i = subs(mu_i, d=part)
    poisson_i = subs(poisson_i, d=part)
    binom_i = subs(binom_i, d=part)

    if True:
        # ewma of mu at time i if review is always not label-changing (meaningful)
        mu_1 = c_alpha * mu_0  # NOQA
        mu_2 = c_alpha * mu_1  # NOQA

    if True:
        i_vals = np.arange(0, 100)
        mu_vals = np.array([subs(mu_i).subs({i: i_}).evalf() for i_ in i_vals])  # NOQA
        binom_vals = np.array(
            [subs(binom_i).subs({i: i_}).evalf() for i_ in i_vals]
        )  # NOQA
        poisson_vals = np.array(
            [subs(poisson_i).subs({i: i_}).evalf() for i_ in i_vals]
        )  # NOQA

        # Find how many iters it actually takes my expt to terminate
        thesis_draft_thresh = np.exp(-2)
        np.where(mu_vals < thesis_draft_thresh)[0]
        np.where(binom_vals < thesis_draft_thresh)[0]
        np.where(poisson_vals < thesis_draft_thresh)[0]

    sym.pprint(sym.simplify(mu_i))
    sym.pprint(sym.simplify(binom_i))
    sym.pprint(sym.simplify(poisson_i))

    # Find the thresholds that force termination after `a` reviews have passed
    # do this by setting i=a
    poisson_thresh = poisson_i.subs({i: a})
    binom_thresh = binom_i.subs({i: a})

    print('Poisson thresh')
    print(sym.latex(sym.Eq(thresh, poisson_thresh)))
    print(sym.latex(sym.Eq(thresh, sym.simplify(poisson_thresh))))

    poisson_thresh.subs({a: 115, s: 30}).evalf()

    sym.pprint(sym.Eq(thresh, poisson_thresh))
    sym.pprint(sym.Eq(thresh, sym.simplify(poisson_thresh)))

    print('Binomial thresh')
    sym.pprint(sym.simplify(binom_thresh))

    sym.pprint(sym.simplify(poisson_thresh.subs({s: a})))

    def taud(coeff):
        return coeff * 360

    if 'poisson_cache' not in vars():
        poisson_cache = {}
        binom_cache = {}

    S, A = np.meshgrid(np.arange(1, 150, 1), np.arange(0, 150, 1))

    import wbia.plottool as pt

    SA_coords = list(zip(S.ravel(), A.ravel()))
    for sval, aval in ut.ProgIter(SA_coords):
        if (sval, aval) not in poisson_cache:
            poisson_cache[(sval, aval)] = float(
                poisson_thresh.subs({a: aval, s: sval}).evalf()
            )
    poisson_zdata = np.array(
        [poisson_cache[(sval, aval)] for sval, aval in SA_coords]
    ).reshape(A.shape)
    fig = pt.figure(fnum=1, doclf=True)
    pt.gca().set_axis_off()
    pt.plot_surface3d(
        S,
        A,
        poisson_zdata,
        xlabel='s',
        ylabel='a',
        rstride=3,
        cstride=3,
        zlabel='poisson',
        mode='wire',
        contour=True,
        title='poisson3d',
    )
    pt.gca().set_zlim(0, 1)
    pt.gca().view_init(elev=taud(1 / 16), azim=taud(5 / 8))
    fig.set_size_inches(10, 6)
    fig.savefig(
        'a-s-t-poisson3d.png',
        dpi=300,
        bbox_inches=pt.extract_axes_extents(fig, combine=True),
    )

    for sval, aval in ut.ProgIter(SA_coords):
        if (sval, aval) not in binom_cache:
            binom_cache[(sval, aval)] = float(
                binom_thresh.subs({a: aval, s: sval}).evalf()
            )
    binom_zdata = np.array(
        [binom_cache[(sval, aval)] for sval, aval in SA_coords]
    ).reshape(A.shape)
    fig = pt.figure(fnum=2, doclf=True)
    pt.gca().set_axis_off()
    pt.plot_surface3d(
        S,
        A,
        binom_zdata,
        xlabel='s',
        ylabel='a',
        rstride=3,
        cstride=3,
        zlabel='binom',
        mode='wire',
        contour=True,
        title='binom3d',
    )
    pt.gca().set_zlim(0, 1)
    pt.gca().view_init(elev=taud(1 / 16), azim=taud(5 / 8))
    fig.set_size_inches(10, 6)
    fig.savefig(
        'a-s-t-binom3d.png',
        dpi=300,
        bbox_inches=pt.extract_axes_extents(fig, combine=True),
    )

    # Find point on the surface that achieves a reasonable threshold

    # Sympy can't solve this
    # sym.solve(sym.Eq(binom_thresh.subs({s: 50}), .05))
    # sym.solve(sym.Eq(poisson_thresh.subs({s: 50}), .05))
    # Find a numerical solution
    def solve_numeric(expr, target, want, fixed, method=None, bounds=None):
        """
        Args:
            expr (Expr): symbolic expression
            target (float): numberic value
            fixed (dict): fixed values of the symbol

        expr = poisson_thresh
        expr.free_symbols
        fixed = {s: 10}

        solve_numeric(poisson_thresh, .05, {s: 30}, method=None)
        solve_numeric(poisson_thresh, .05, {s: 30}, method='Nelder-Mead')
        solve_numeric(poisson_thresh, .05, {s: 30}, method='BFGS')
        """
        import scipy.optimize

        # Find the symbol you want to solve for
        want_symbols = expr.free_symbols - set(fixed.keys())
        # TODO: can probably extend this to multiple params
        assert len(want_symbols) == 1, 'specify all but one var'
        assert want == list(want_symbols)[0]
        fixed_expr = expr.subs(fixed)

        def func(a1):
            expr_value = float(fixed_expr.subs({want: a1}).evalf())
            return (expr_value - target) ** 2

        # if method is None:
        #     method = 'Nelder-Mead'
        #     method = 'Newton-CG'
        #     method = 'BFGS'
        # Use one of the other params the startin gpoing
        a1 = list(fixed.values())[0]
        result = scipy.optimize.minimize(func, x0=a1, method=method, bounds=bounds)
        if not result.success:
            print('\n')
            print(result)
            print('\n')
        return result

    # Numeric measurments of thie line

    thresh_vals = [0.001, 0.01, 0.05, 0.1, 0.135]
    svals = np.arange(1, 100)

    target_poisson_plots = {}
    for target in ut.ProgIter(thresh_vals, bs=False, freq=1):
        poisson_avals = []
        for sval in ut.ProgIter(svals, 'poisson', freq=1):
            expr = poisson_thresh
            fixed = {s: sval}
            want = a
            aval = solve_numeric(expr, target, want, fixed, method='Nelder-Mead').x[0]
            poisson_avals.append(aval)
        target_poisson_plots[target] = (svals, poisson_avals)

    fig = pt.figure(fnum=3)
    for target, dat in target_poisson_plots.items():
        pt.plt.plot(*dat, label='prob={}'.format(target))
    pt.gca().set_xlabel('s')
    pt.gca().set_ylabel('a')
    pt.legend()
    pt.gca().set_title('poisson')
    fig.set_size_inches(5, 3)
    fig.savefig(
        'a-vs-s-poisson.png',
        dpi=300,
        bbox_inches=pt.extract_axes_extents(fig, combine=True),
    )

    target_binom_plots = {}
    for target in ut.ProgIter(thresh_vals, bs=False, freq=1):
        binom_avals = []
        for sval in ut.ProgIter(svals, 'binom', freq=1):
            aval = solve_numeric(
                binom_thresh, target, a, {s: sval}, method='Nelder-Mead'
            ).x[0]
            binom_avals.append(aval)
        target_binom_plots[target] = (svals, binom_avals)

    fig = pt.figure(fnum=4)
    for target, dat in target_binom_plots.items():
        pt.plt.plot(*dat, label='prob={}'.format(target))
    pt.gca().set_xlabel('s')
    pt.gca().set_ylabel('a')
    pt.legend()
    pt.gca().set_title('binom')
    fig.set_size_inches(5, 3)
    fig.savefig(
        'a-vs-s-binom.png',
        dpi=300,
        bbox_inches=pt.extract_axes_extents(fig, combine=True),
    )

    # ----
    if True:

        fig = pt.figure(fnum=5, doclf=True)
        s_vals = [1, 2, 3, 10, 20, 30, 40, 50]
        for sval in s_vals:
            pp = poisson_thresh.subs({s: sval})

            a_vals = np.arange(0, 200)
            pp_vals = np.array(
                [float(pp.subs({a: aval}).evalf()) for aval in a_vals]
            )  # NOQA

            pt.plot(a_vals, pp_vals, label='s=%r' % (sval,))
        pt.legend()
        pt.gca().set_xlabel('a')
        pt.gca().set_ylabel('poisson prob after a reviews')
        fig.set_size_inches(5, 3)
        fig.savefig(
            'a-vs-thresh-poisson.png',
            dpi=300,
            bbox_inches=pt.extract_axes_extents(fig, combine=True),
        )

        fig = pt.figure(fnum=6, doclf=True)
        s_vals = [1, 2, 3, 10, 20, 30, 40, 50]
        for sval in s_vals:
            pp = binom_thresh.subs({s: sval})
            a_vals = np.arange(0, 200)
            pp_vals = np.array(
                [float(pp.subs({a: aval}).evalf()) for aval in a_vals]
            )  # NOQA
            pt.plot(a_vals, pp_vals, label='s=%r' % (sval,))
        pt.legend()
        pt.gca().set_xlabel('a')
        pt.gca().set_ylabel('binom prob after a reviews')
        fig.set_size_inches(5, 3)
        fig.savefig(
            'a-vs-thresh-binom.png',
            dpi=300,
            bbox_inches=pt.extract_axes_extents(fig, combine=True),
        )

        # -------

        fig = pt.figure(fnum=5, doclf=True)
        a_vals = [1, 2, 3, 10, 20, 30, 40, 50]
        for aval in a_vals:
            pp = poisson_thresh.subs({a: aval})
            s_vals = np.arange(1, 200)
            pp_vals = np.array(
                [float(pp.subs({s: sval}).evalf()) for sval in s_vals]
            )  # NOQA
            pt.plot(s_vals, pp_vals, label='a=%r' % (aval,))
        pt.legend()
        pt.gca().set_xlabel('s')
        pt.gca().set_ylabel('poisson prob')
        fig.set_size_inches(5, 3)
        fig.savefig(
            's-vs-thresh-poisson.png',
            dpi=300,
            bbox_inches=pt.extract_axes_extents(fig, combine=True),
        )

        fig = pt.figure(fnum=5, doclf=True)
        a_vals = [1, 2, 3, 10, 20, 30, 40, 50]
        for aval in a_vals:
            pp = binom_thresh.subs({a: aval})
            s_vals = np.arange(1, 200)
            pp_vals = np.array(
                [float(pp.subs({s: sval}).evalf()) for sval in s_vals]
            )  # NOQA
            pt.plot(s_vals, pp_vals, label='a=%r' % (aval,))
        pt.legend()
        pt.gca().set_xlabel('s')
        pt.gca().set_ylabel('binom prob')
        fig.set_size_inches(5, 3)
        fig.savefig(
            's-vs-thresh-binom.png',
            dpi=300,
            bbox_inches=pt.extract_axes_extents(fig, combine=True),
        )

    # ---------------------
    # Plot out a table

    mu_i.subs({s: 75, a: 75}).evalf()
    poisson_thresh.subs({s: 75, a: 75}).evalf()

    sval = 50
    for target, dat in target_poisson_plots.items():
        slope = np.median(np.diff(dat[1]))
        aval = int(np.ceil(sval * slope))
        thresh = float(poisson_thresh.subs({s: sval, a: aval}).evalf())
        print('aval={}, sval={}, thresh={}, target={}'.format(aval, sval, thresh, target))

    for target, dat in target_binom_plots.items():
        slope = np.median(np.diff(dat[1]))
        aval = int(np.ceil(sval * slope))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.refresh
        python -m wbia.algo.graph.refresh --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
