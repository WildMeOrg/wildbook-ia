from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import utool as ut
import six  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[scorenorm]', DEBUG=False)


def learn_score_normalization(tp_support, tn_support, gridsize=1024,
                              adjust=8, return_all=False, monotonize=True,
                              clip_factor=(ut.PHI + 1)):
    r"""
    Takes collected data and applys parzen window density estimation and bayes rule.

    Args:
        tp_support (ndarray):
        tn_support (ndarray):
        gridsize       (int): default 512
        adjust         (int): default 8
        return_all     (bool): default False
        monotonize     (bool): default True
        clip_factor    (float): default phi ** 2

    Returns:
        tuple: (score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score, clip_score)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> tp_support = np.linspace(100, 10000, 512)
        >>> tn_support = np.linspace(0, 120, 512)
        >>> (score_domain, p_tp_given_score, clip_score) = learn_score_normalization(tp_support, tn_support)
        >>> result = int(p_tp_given_score.sum())
        >>> print(result)
        92
    """
    # Estimate true positive density
    score_tp_pdf = ut.estimate_pdf(tp_support, gridsize=gridsize, adjust=adjust)
    score_tn_pdf = ut.estimate_pdf(tn_support, gridsize=gridsize, adjust=adjust)
    # Find good maximum score (for domain not learning)
    #clip_score = 2000
    clip_score = find_score_maxclip(tp_support, tn_support, clip_factor)
    score_domain = np.linspace(0, clip_score, 1024)
    # Evaluate true negative density
    p_score_given_tp = score_tp_pdf.evaluate(score_domain)
    p_score_given_tn = score_tn_pdf.evaluate(score_domain)
    # Average to get probablity of any score
    p_score = (np.array(p_score_given_tp) + np.array(p_score_given_tn)) / 2.0
    # Apply bayes
    p_tp = .5
    p_tp_given_score = ut.bayes_rule(p_score_given_tp, p_tp, p_score)
    import vtool as vt
    if monotonize:
        p_tp_given_score = vt.ensure_monotone_strictly_increasing(
            p_tp_given_score, zerohack=True, onehack=True)
    if return_all:
        p_tn_given_score = 1 - p_tp_given_score
        return (score_domain, p_tp_given_score, p_tn_given_score,
                p_score_given_tp, p_score_given_tn, p_score, clip_score)
    else:
        return (score_domain, p_tp_given_score, clip_score)


def find_score_maxclip(tp_support, tn_support, clip_factor=ut.PHI + 1):
    """
    returns score to clip true positives past.

    Args:
        tp_support (ndarray):
        tn_support (ndarray):

    Returns:
        float: clip_score

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> tp_support = np.array([100, 200, 50000])
        >>> tn_support = np.array([10, 30, 110])
        >>> clip_score = find_score_maxclip(tp_support, tn_support)
        >>> result = str(clip_score)
        >>> print(result)
        287.983738762
    """
    max_true_positive_score = tp_support.max()
    max_true_negative_score = tn_support.max()
    if clip_factor is None:
        clip_score = max_true_positive_score
    else:
        overshoot_factor = max_true_positive_score / max_true_negative_score
        if overshoot_factor > clip_factor:
            clip_score = max_true_negative_score * clip_factor
        else:
            clip_score = max_true_positive_score
    return clip_score


# DEBUGGING FUNCTIONS


def test_score_normalization():
    """
    Shows how score normalization works with gaussian noise

    CommandLine:
        python -m vtool.score_normalization --test-test_score_normalization --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> test_score_normalization()
        >>> ut.show_if_requested()

    """
    import plottool as pt  # NOQA
    randstate = np.random.RandomState(seed=0)
    # Get a training sample
    tp_support = randstate.normal(loc=6.5, size=(256,))
    tn_support = randstate.normal(loc=2, size=(256,))

    # Print raw score statistics
    ut.print_stats(tp_support, lbl='tp_support')
    ut.print_stats(tn_support, lbl='tn_support')

    # Test (potentially multiple) normalizing configurations
    normkw_list = ut.util_dict.all_dict_combinations(
        {
            'monotonize': [True],  # [True, False],
            #'adjust': [1, 4, 8],
            'adjust': [1],
            #'adjust': [8],
        }
    )

    if len(normkw_list) > 32:
        raise AssertionError('Too many plots to test!')

    fnum = pt.next_fnum()
    #plot_support(tn_support, tp_support, fnum=fnum)

    for normkw in normkw_list:
        # Learn the appropriate normalization
        #normkw = {}  # dict(gridsize=1024, adjust=8, clip_factor=ut.PHI + 1, return_all=True)
        (score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn,
         p_score, clip_score) = learn_score_normalization(tp_support, tn_support, return_all=True, **normkw)

        assert clip_score > tn_support.max()

        inspect_pdfs(tn_support, tp_support, score_domain,
                     p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score, with_scores=True)

        pt.set_figtitle('ScoreNorm test' + ut.dict_str(normkw, newlines=False))
    locals_ = locals()
    return locals_


def inspect_pdfs(tn_support, tp_support, score_domain, p_tp_given_score,
                 p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score,
                 with_scores=False):
    import plottool as pt  # NOQA

    fnum = pt.next_fnum()
    nRows = 2 + with_scores
    pnum_ = pt.get_pnum_func(nRows=nRows, nCols=1)
    #pnum_ = pt.get_pnum_func(nRows=3, nCols=1)
    #def next_pnum():
    #    return pnum_(

    def generate_pnum():
        for px in range(nRows):
            yield pnum_(px)

    _pnumiter = generate_pnum().next

    pt.figure(fnum=fnum, pnum=pnum_(0))

    if with_scores:
        plot_support(tn_support, tp_support, fnum=fnum, pnum=_pnumiter())

    plot_prebayes_pdf(score_domain, p_score_given_tn, p_score_given_tp, p_score,
                      cfgstr='', fnum=fnum, pnum=_pnumiter())

    plot_postbayes_pdf(score_domain, p_tn_given_score, p_tp_given_score,
                       cfgstr='', fnum=fnum, pnum=_pnumiter())


def plot_support(tn_support, tp_support, fnum=None, pnum=(1, 1, 1)):
    r"""
    Args:
        tn_support (ndarray):
        tp_support (ndarray):
        fnum (int):  figure number
        pnum (tuple):  plot number

    CommandLine:
        python -m ibeis.model.hots.score_normalization --test-plot_support

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> tn_support = '?'
        >>> tp_support = '?'
        >>> fnum = None
        >>> pnum = (1, 1, 1)
        >>> result = plot_support(tn_support, tp_support, fnum, pnum)
        >>> print(result)
    """
    import plottool as pt  # NOQA
    if fnum is None:
        fnum = pt.next_fnum()
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED
    pt.plots.plot_sorted_scores(
        (tn_support, tp_support),
        ('trueneg scores', 'truepos scores'),
        score_colors=(false_color, true_color),
        #logscale=True,
        logscale=False,
        figtitle='sorted nscores',
        fnum=fnum,
        pnum=pnum)


def plot_prebayes_pdf(score_domain, p_score_given_tn, p_score_given_tp, p_score,
                      cfgstr='', fnum=None, pnum=(1, 1, 1)):
    import plottool as pt  # NOQA
    if fnum is None:
        fnum = pt.next_fnum()
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED
    unknown_color = pt.UNKNOWN_PURP

    pt.plots.plot_probabilities(
        (p_score_given_tn,  p_score_given_tp, p_score),
        ('p(score | tn)', 'p(score | tp)', 'p(score)'),
        prob_colors=(false_color, true_color, unknown_color),
        figtitle='pre_bayes pdf score',
        xdata=score_domain,
        fnum=fnum,
        pnum=pnum)


def plot_postbayes_pdf(score_domain, p_tn_given_score, p_tp_given_score,
                       cfgstr='', fnum=None, pnum=(1, 1, 1)):
    import plottool as pt  # NOQA
    if fnum is None:
        fnum = pt.next_fnum()
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED

    pt.plots.plot_probabilities(
        (p_tn_given_score, p_tp_given_score),
        ('p(tn | score)', 'p(tp | score)'),
        prob_colors=(false_color, true_color,),
        figtitle='post_bayes pdf score ' + cfgstr,
        xdata=score_domain, fnum=fnum, pnum=pnum)


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.score_normalization
        python -m vtool.score_normalization --allexamples
        python -m vtool.score_normalization --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
