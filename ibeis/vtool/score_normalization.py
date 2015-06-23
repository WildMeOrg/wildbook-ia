# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import utool as ut
import six
import scipy.interpolate
print, rrr, profile = utool.inject2(__name__, '[scorenorm]', DEBUG=False)


def check_unused_kwargs(kwargs, expected_keys):
    unused_keys = set(kwargs.keys()) - set(expected_keys)
    if len(unused_keys) > 0:
        print('unused kwargs keys = %r' % (unused_keys))


def testdata_score_normalier():
    randstate = np.random.RandomState(seed=0)
    # Get a training sample
    tp_support = randstate.normal(loc=6.5, size=(256,))
    tn_support = randstate.normal(loc=3.5, size=(256,))
    data   = np.hstack((tp_support, tn_support))
    labels = np.array([True] * len(tp_support) + [False] * len(tn_support))
    encoder = ScoreNormalizer()
    encoder.fit(data, labels)
    return encoder, data, labels


@six.add_metaclass(ut.ReloadingMetaclass)
class ScoreNormalizer(object):
    """
    Conforms to scikit-learn Estimator interface
    """
    def __init__(encoder, target_tpr=.95, **kwargs):
        """
        Args:
            target_tpr (float): target true positive rate
        """
        encoder.learn_kw = ut.update_existing(
            dict(
                gridsize=1024,
                adjust=8,
                monotonize=False,
                #monotonize=True,
                #clip_factor=(ut.PHI + 1),
                clip_factor=None,
                reverse=None,
            ), kwargs)
        check_unused_kwargs(kwargs, encoder.learn_kw.keys())
        # Target recall for learned threshold
        encoder.target_tpr = target_tpr
        # Support data
        encoder.support = dict(
            tp_support=None,
            tn_support=None,
        )
        # Learned score normalization
        encoder.score_domain     = None
        encoder.p_tp_given_score = None
        encoder.p_tn_given_score = None
        encoder.p_score_given_tn = None
        encoder.p_score_given_tp = None
        encoder.p_score = None
        # Learneed classification threshold
        encoder.learned_thresh   = None

    def fit(encoder, X, y, verbose=True):
        """
        Fits estimator to data.

        Args:
            X (ndarray):   1 dimensional scores
            y (ndarray): True or False labels

        CommandLine:
            python -m vtool.score_normalization --test-fit --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from vtool.score_normalization import *  # NOQA
            >>> import vtool as vt
            >>> encoder = ScoreNormalizer()
            >>> X, y = vt.tests.dummy.testdata_binary_scores()
            >>> result = encoder.fit(X, y)
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> encoder.visualize()
            >>> ut.show_if_requested()
        """
        encoder.learn_probabilities(X, y, verbose=verbose)
        encoder.learn_threshold(X, y, verbose=verbose)

    def learn_probabilities(encoder, X, y, verbose=True):
        tp_support = X.compress(y, axis=0).astype(np.float64)
        tn_support = X.compress(np.logical_not(y), axis=0).astype(np.float64)
        encoder.support['tp_support'] = tp_support
        encoder.support['tn_support'] = tn_support

        if encoder.learn_kw['reverse'] is None:
            # heuristic
            encoder.learn_kw['reverse'] = tp_support.mean() < tn_support.mean()
            print('[scorenorm] setting reverse = %r' % (encoder.learn_kw['reverse']))

        tup = learn_score_normalization(tp_support, tn_support, return_all=True,
                                        verbose=verbose, **encoder.learn_kw)
        (score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp,
         p_score_given_tn, p_score) = tup

        encoder.score_domain = score_domain
        encoder.p_tp_given_score = p_tp_given_score
        encoder.p_tn_given_score = p_tn_given_score
        encoder.p_score_given_tn = p_score_given_tn
        encoder.p_score_given_tp = p_score_given_tp
        encoder.p_score = p_score

        encoder.interp_fn = scipy.interpolate.interp1d(
            encoder.score_domain, encoder.p_tp_given_score, kind='linear',
            copy=False, assume_sorted=False)

    @staticmethod
    def _to_xy(tp_scores, tn_scores):
        """ helper """
        scores = np.hstack([tp_scores, tn_scores])
        labels = np.zeros(scores.size, dtype=np.bool)
        labels[0:len(tp_scores)] = True
        #np.array(([True] * len(tp_scores)) + ([False] * len(tn_scores), dtype=np.bool)
        return scores, labels

    def fit_partitioned(encoder, tp_scores, tn_scores, verbose=True):
        """ convinience func """
        X, y = encoder._to_xy(tp_scores, tn_scores)
        return encoder.fit(X, y, verbose=verbose)

    def learn_threshold(encoder, X, y, verbose=True):
        """
        Learns threshold that achieves the target true positive rate
        """
        # select a cutoff threshold
        #import sklearn.metrics
        import vtool as vt
        probs = encoder.normalize_scores(X)
        confusions = vt.ConfusionMetrics.from_scores_and_labels(probs, y, verbose=verbose)
        #fpr_curve, tpr_curve, thresholds = sklearn.metrics.roc_curve(y, probs, pos_label=True)
        # Select threshold that gives 95% recall (we should optimize this for a tradeoff)
        index = np.where(confusions.tpr > encoder.target_tpr)[0][0]
        encoder.learned_thresh = confusions.thresholds[index]
        score_thresh = encoder.inverse_normalize(encoder.learned_thresh)
        if verbose:
            print('[scorenorm] Learning threshold to achieve TPR=%r' % (encoder.target_tpr,))
            print('[scorenorm]   * learned_thresh = %r' % (encoder.learned_thresh,))
            print('[scorenorm]   * score_thresh = %r' % (score_thresh,))
            print('[scorenorm]   * fpr = %r' % (confusions.get_fpr_at_recall(encoder.target_tpr),))

    def inverse_normalize(encoder, probs):
        inverse_interp = scipy.interpolate.interp1d(encoder.p_tp_given_score, encoder.score_domain,
                                                    kind='linear', copy=False,
                                                    assume_sorted=False)
        scores = inverse_interp(probs)
        return scores

    def normalize_scores(encoder, X):
        is_iterable = ut.isiterable(X)
        if not is_iterable:
            X = np.array([X])
        prob = normalize_scores(encoder.score_domain, encoder.p_tp_given_score, X, interp_fn=encoder.interp_fn)
        if not is_iterable:
            prob = prob[0]
        return prob

    def predict(encoder, X):
        """ Predict true or false of ``X``. """
        prob = encoder.normalize_scores(X)
        pred = prob > encoder.learned_thresh
        return pred

    def get_accuracy(encoder, X, y):
        pred = encoder.predict(X)
        is_correct = pred == y
        accuracy = (is_correct).mean()
        return accuracy

    def get_error_indicies(encoder, X, y):
        """
        Returns the indicies of the most difficult type I and type II errors.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.score_normalization import *  # NOQA
            >>> encoder, X, y = testdata_score_normalier()
            >>> (fp_indicies, fn_indicies) = encoder.get_error_indicies(X, y)
            >>> fp_X = X.take(fp_indicies)[0:3]
            >>> fn_X = X.take(fn_indicies)[0:3]
            >>> result = 'fp_X = ' + ut.numpy_str2(fp_X) + '\nfn_X = ' + ut.numpy_str2(fn_X)
            >>> print(result)
            fp_X = np.array([ 6.196,  5.912,  5.804])
            fn_X = np.array([ 3.947,  4.277,  4.43 ])
        """
        prob = encoder.normalize_scores(X)
        pred = prob > encoder.learned_thresh
        is_correct = pred == y
        # Find misspredictions
        is_error = np.logical_not(is_correct)
        # get indexes of misspredictions
        error_indicies = np.where(is_error)[0]
        # Separate by Type I and Type II error
        error_y = y.take(error_indicies)
        fp_indicies_ = error_indicies.compress(np.logical_not(error_y))
        fn_indicies_ = error_indicies.compress(error_y)
        # Sort errors by difficulty
        fp_sortx = prob.take(fp_indicies_).argsort()[::-1]
        fn_sortx = prob.take(fn_indicies_).argsort()
        fp_indicies = fp_indicies_.take(fp_sortx)
        fn_indicies = fn_indicies_.take(fn_sortx)
        return fp_indicies, fn_indicies

    def get_correct_indices(encoder, X, y):
        r"""
        Args:
            X (ndarray):  data
            y (ndarray):  labels

        Returns:
            tuple: (fp_indicies, fn_indicies)

        CommandLine:
            python -m vtool.score_normalization --test-get_correct_indices

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.score_normalization import *  # NOQA
            >>> encoder, X, y = testdata_score_normalier()
            >>> (tp_indicies, tn_indicies) = encoder.get_correct_indices(X, y)
            >>> tp_X = X.take(tp_indicies)[0:3]
            >>> tn_X = X.take(tn_indicies)[0:3]
            >>> result = 'tp_X = ' + ut.numpy_str2(tp_X) + '\ntn_X = ' + ut.numpy_str2(tn_X)
            >>> print(result)
            tp_X = np.array([ 8.883,  8.77 ,  8.759])
            tn_X = np.array([ 0.727,  0.76 ,  0.841])
        """
        prob = encoder.normalize_scores(X)
        pred = prob > encoder.learned_thresh
        is_correct = pred == y
        # get indexes of misspredictions
        correct_indicies = np.where(is_correct)[0]
        # Separate by Type I and Type II error
        correct_y = y.take(correct_indicies)
        tp_indicies_ = correct_indicies.compress(correct_y)
        tn_indicies_ = correct_indicies.compress(np.logical_not(correct_y))
        # Sort by most correct cases
        tp_sortx = prob.take(tp_indicies_).argsort()[::-1]
        tn_sortx = prob.take(tn_indicies_).argsort()
        tp_indicies = tp_indicies_.take(tp_sortx)
        tn_indicies = tn_indicies_.take(tn_sortx)
        return tp_indicies, tn_indicies

    def get_confusion_indicies(encoder, X, y):
        """
        combination of get_correct_indices and get_error_indicies
        """
        prob = encoder.normalize_scores(X)
        pred = prob > encoder.learned_thresh
        # Find correct and misspredictions
        is_correct = pred == y
        is_error = np.logical_not(is_correct)
        correct_indicies = np.where(is_correct)[0]
        error_indicies = np.where(is_error)[0]
        # Separate by Correct and Type I and Type II error
        error_y = y.take(error_indicies)
        correct_y = y.take(correct_indicies)
        fp_indicies_ = error_indicies.compress(np.logical_not(error_y))
        fn_indicies_ = error_indicies.compress(error_y)
        tp_indicies_ = correct_indicies.compress(correct_y)
        tn_indicies_ = correct_indicies.compress(np.logical_not(correct_y))
        # Sort errors by difficulty and other cases by correctness
        fp_sortx = prob.take(fp_indicies_).argsort()[::-1]
        fn_sortx = prob.take(fn_indicies_).argsort()
        tp_sortx = prob.take(tp_indicies_).argsort()[::-1]
        tn_sortx = prob.take(tn_indicies_).argsort()
        indicies = ut.DynStruct()
        indicies.tp = tp_indicies_.take(tp_sortx)
        indicies.tn = tn_indicies_.take(tn_sortx)
        indicies.fp = fp_indicies_.take(fp_sortx)
        indicies.fn = fn_indicies_.take(fn_sortx)
        return indicies

    def visualize(encoder, **kwargs):
        """
        shows details about the score normalizer

        Kwargs:
            fnum
            figtitle
            interactive
            with_scores
            with_roc
            with_precision_recall
        """  # + ut.get_kwargs(inspect_pdfs)[0]
        import plottool as pt
        inspect_kw = ut.update_existing(
            dict(
                with_scores=True,
                with_roc=True,
                with_precision_recall=True,
                fnum=None,
                figtitle=None,
                interactive=None,
                use_stems=None,
            ), kwargs)
        prob_thresh = encoder.learned_thresh
        score_thresh = encoder.inverse_normalize(prob_thresh)
        inter = inspect_pdfs(
            encoder.support['tn_support'], encoder.support['tp_support'],
            encoder.score_domain, encoder.p_tp_given_score,
            encoder.p_tn_given_score, encoder.p_score_given_tp,
            encoder.p_score_given_tn, encoder.p_score, prob_thresh=prob_thresh,
            score_thresh=score_thresh, **inspect_kw)
        pt.adjust_subplots(bottom=.06, left=.06, right=.97, wspace=.25, hspace=.25, top=.9)
        return inter


def learn_score_normalization(tp_support, tn_support,
                              gridsize=1024,
                              adjust=8, return_all=False, monotonize=True,
                              clip_factor=(ut.PHI + 1), verbose=True, reverse=False):
    r"""
    Takes collected data and applys parzen window density estimation and bayes rule.

    #True positive scores must be larger than true negative scores.

    Args:
        tp_support (ndarray):
        tn_support (ndarray):
        gridsize       (int): default 512
        adjust         (int): default 8
        return_all     (bool): default False
        monotonize     (bool): default True
        clip_factor    (float): default phi ** 2

    Returns:
        tuple: (score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score)

    CommandLine:
        python -m vtool.score_normalization --test-learn_score_normalization

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> tp_support = np.linspace(100, 10000, 512)
        >>> tn_support = np.linspace(0, 120, 512)
        >>> gridsize = 1024
        >>> adjust = 8
        >>> return_all = False
        >>> monotonize = True
        >>> clip_factor = 2.6180339887499997
        >>> verbose = True
        >>> reverse = False
        >>> (score_domain, p_tp_given_score) = learn_score_normalization(tp_support, tn_support)
        >>> result = int(p_tp_given_score.sum())
        >>> print(result)
        92
    """
    # Estimate true positive density
    if verbose:
        print('[scorenorm] Learning normalization pdf')
        print('[scorenorm] * tp_support.shape=%r' % (tp_support.shape,))
        print('[scorenorm] * tn_support.shape=%r' % (tn_support.shape,))
        print('[scorenorm] * estimating true positive pdf, ')
        print('[scorenorm] * monotonize = %r' % (monotonize,))
        next_ = ut.next_counter(1)
        total = 8
    if verbose:
        print('[scorenorm] %d/%d estimating true negative pdf' % (next_(), total))
    score_tp_pdf = ut.estimate_pdf(tp_support, gridsize=gridsize, adjust=adjust)
    if verbose:
        print('[scorenorm] %d/%d estimating true negative pdf' % (next_(), total))
    score_tn_pdf = ut.estimate_pdf(tn_support, gridsize=gridsize, adjust=adjust)
    if verbose:
        print('[scorenorm] %d/%d estimating score domain' % (next_(), total))
    # Find good score domain range
    min_score, max_score = find_clip_range(tp_support, tn_support, clip_factor, reverse)
    score_domain = np.linspace(min_score, max_score, gridsize)
    # Evaluate true negative density
    if verbose:
        print('[scorenorm] %d/%d evaluating tp density' % (next_(), total))
    p_score_given_tp = score_tp_pdf.evaluate(score_domain)
    if verbose:
        print('[scorenorm] %d/%d evaluating tn density' % (next_(), total))
    p_score_given_tn = score_tn_pdf.evaluate(score_domain)
    if verbose:
        print('[scorenorm] %d/%d evaluating posterior probabilities' % (next_(), total))
    # Average to get probablity of any score
    p_score = (np.array(p_score_given_tp) + np.array(p_score_given_tn)) / 2.0
    # Apply bayes
    # FIXME: not always going to be equal probability of true and positive cases
    p_tp = .5
    p_tp_given_score = ut.bayes_rule(p_score_given_tp, p_tp, p_score)
    import vtool as vt
    if monotonize:
        if reverse:
            if verbose:
                print('[scorenorm] %d/%d monotonize decreasing' % (next_(), total))
            p_tp_given_score = vt.ensure_monotone_strictly_decreasing(
                p_tp_given_score, left_endpoint=1.0, right_endpoint=0.0)
        else:
            if verbose:
                print('[scorenorm] %d/%d monotonize increasing' % (next_(), total))
            p_tp_given_score = vt.ensure_monotone_strictly_increasing(
                p_tp_given_score, left_endpoint=0.0, right_endpoint=1.0)
    if return_all:
        if verbose:
            print('[scorenorm] %d/%d returning all' % (next_(), total))
        p_tn_given_score = 1 - p_tp_given_score
        return (score_domain, p_tp_given_score, p_tn_given_score,
                p_score_given_tp, p_score_given_tn, p_score)
    else:
        if verbose:
            print('[scorenorm] %d/%d returning minimum' % (next_(), total))
        return (score_domain, p_tp_given_score)


def find_clip_range(tp_support, tn_support, clip_factor=ut.PHI + 1, reverse=None):
    """
    TODO: generalize to arbitrary domains (not just 0->inf)

    Finds score to clip true positives past. This is useful when the highest
    true positive scores can be much larger than the highest true negative
    score.

    Args:
        tp_support (ndarray):
        tn_support (ndarray):
        clip_factor (float): factor of the true negative domain to search for true positives

    Returns:
        tuple: min_score, max_score

    CommandLine:
        python -m vtool.score_normalization --test-find_clip_range

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> tp_support = np.array([100, 200, 50000])
        >>> tn_support = np.array([10, 30, 110])
        >>> clip_factor = ut.PHI + 1
        >>> min_score, max_score = find_clip_range(tp_support, tn_support,  clip_factor)
        >>> result = '%.4f, %.4f' % ((min_score, max_score))
        >>> print(result)
        10.0000, 287.9837
    """
    if reverse is None:
        mean_tp_score = tp_support.mean()
        mean_tn_score = tn_support.mean()
        reverse = mean_tp_score < mean_tn_score

    if not reverse:
        # Normal case where higher scores is better
        high_scores = tp_support
        low_scores  = tn_support
    else:
        high_scores = tn_support
        low_scores  = tp_support

    max_high_score = high_scores.max()
    max_low_score  = low_scores.max()
    min_high_score = high_scores.min()
    min_low_score  = low_scores.min()
    abs_max_score = max(max_high_score, max_low_score)
    abs_min_score = min(min_high_score, min_low_score)

    if clip_factor is None:
        min_score = abs_min_score
        max_score = abs_max_score
        return min_score, max_score

    # FIXME: allow for true positive scores to be low, or not bounded at 0

    # Do not clip if true negatives can score higher than true positives
    if max_low_score < max_high_score:
        #overshoot_factor = (max_high_score - abs_min_score) / (max_low_score - abs_min_score)
        overshoot_factor = max_high_score / max_low_score
        if overshoot_factor > clip_factor:
            max_score = max_low_score * clip_factor
        else:
            max_score = max_high_score
    min_score = abs_min_score
    #if min_low_score < min_high_score:
    #    overshoot_factor = min_low_score / min_high_score
    #    if overshoot_factor > clip_factor:
    #        min_score = min_high_score * clip_factor
    #    else:
    #        min_score = min_low_score
    return min_score, max_score


def normalize_scores(score_domain, p_tp_given_score, scores, interp_fn=None):
    """
    Adjusts a raw scores to a probabilities based on a learned normalizer

    Args:
        score_domain (ndarray): input score domain
        p_tp_given_score (ndarray): learned probability mapping
        scores (ndarray): raw scores

    Returns:
        ndarray: probabilities

    CommandLine:
        python -m vtool.score_normalization --test-normalize_scores

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> score_domain = np.linspace(0, 10, 10)
        >>> p_tp_given_score = (score_domain ** 2) / (score_domain.max() ** 2)
        >>> scores = np.array([-1, 0.0, 0.01, 2.3, 8.0, 9.99, 10.0, 10.1, 11.1])
        >>> prob = normalize_scores(score_domain, p_tp_given_score, scores)
        >>> #np.set_printoptions(suppress=True)
        >>> result = ut.numpy_str(prob, precision=2, suppress_small=True)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot2(score_domain, p_tp_given_score, 'r-x', equal_aspect=False, label='learned probability')
        >>> pt.plot2(scores, prob, 'yo', equal_aspect=False, title='Normalized scores', pad=.2, label='query points')
        >>> pt.legend('upper left')
        >>> ut.show_if_requested()
        np.array([ 0.  ,  0.  ,  0.  ,  0.05,  0.64,  1.  ,  1.  ,  1.  ,  1.  ], dtype=np.float64)
    """
    #prob = np.zeros(len(scores))
    prob = np.zeros(len(scores))
    #prob = np.full(len(scores), np.nan)
    is_nan  = np.isnan(scores)
    # Check score domain bounds
    is_low  = scores < score_domain[0]
    is_high = scores > score_domain[-1]
    is_inbounds = np.logical_not(np.logical_or.reduce((is_nan, is_low, is_high)))
    # interpolate scores in the learned domain
    # we are garuenteed to have inbounds nonzero elements here
    if True:
        if interp_fn is None:
            # TODO build custom interpolator with correct bound checks
            interp_fn = scipy.interpolate.interp1d(score_domain, p_tp_given_score,
                                                   kind='linear', copy=False,
                                                   assume_sorted=True)
        prob[is_inbounds] = interp_fn(scores[is_inbounds])
    else:
        flags = score_domain <= scores[is_inbounds][:, None]
        left_indicies = np.array([np.nonzero(row)[0][-1] for row in flags])
        prob[is_inbounds] = p_tp_given_score[left_indicies]
    # currently just taking the min
    # fill in other values
    assert not np.any(is_nan), 'cannot normalize nan values'
    #if any(is_nan):
    #    # handle nans
    #    raise AssertionError('user normalize score list')
    #    prob[np.isnan(score_domain)] = -1.0
    # clip low scores at 0
    prob[is_low] = 0
    # clip high scores by between max probability and one
    prob[is_high] = (p_tp_given_score[-1] + 1.0) / 2.0
    return prob


# --------
# Plotting
# --------


def inspect_pdfs(tn_support, tp_support, score_domain, p_tp_given_score,
                 p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score,
                 prob_thresh=None, score_thresh=None,
                 with_scores=False, with_roc=False,
                 with_precision_recall=False, fnum=None, figtitle=None, interactive=None, use_stems=None):
    """
    Shows plots of learned thresholds
    """
    import plottool as pt  # NOQA

    if fnum is None:
        fnum = pt.next_fnum()

    with_normscore = with_scores
    with_prebayes = True
    with_postbayes = True

    nSubplots = (with_normscore + with_prebayes + with_postbayes +
                 with_scores + with_roc + with_precision_recall)
    if True:
        nRows, nCols = pt.get_square_row_cols(nSubplots)
    else:
        nRows = nSubplots
        nCols = 1
    _pnumiter = pt.make_pnum_nextgen(nRows=nRows, nCols=nCols, nSubplots=nSubplots)

    import plottool.interactions

    inter = plottool.interactions.ExpandableInteraction(fnum, _pnumiter)

    import vtool as vt
    scores = np.hstack([tn_support, tp_support])
    labels = np.array([False] * len(tn_support) + [True] * len(tp_support))

    #c
    # probs = encoder.normalize_scores(scores)
    probs = normalize_scores(score_domain, p_tp_given_score, scores)

    confusions = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)
    #print('fpr@.95 recall = %r' % (confusions.get_fpr_at_95_recall(),))
    print('fpp@95 recall = %05.2f%%' % (confusions.get_fpr_at_95_recall() * 100,))

    import plottool as pt  # NOQA
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED

    support_kw = dict(
        scores_lbls=('trueneg', 'truepos'),
        score_colors=(false_color, true_color),
    )
    support_sort_kw = dict(
        logscale=False,
        score_markers=['^', 'v'],
        markersizes=[5, 5],
        use_stems=use_stems,
        **support_kw
    )

    def _score_support_hist(fnum, pnum):
        pt.plot_score_histograms(
            (tn_support, tp_support),
            score_label='score',
            fnum=fnum,
            pnum=pnum,
            **support_kw)

    def _prob_support_hist(fnum, pnum):
        tp_probs = probs[labels]
        tn_probs = probs[np.logical_not(labels)]
        pt.plot_score_histograms(
            (tn_probs, tp_probs),
            score_label='prob',
            fnum=fnum,
            pnum=pnum,
            **support_kw)

    def _score_support_sorted(fnum, pnum):
        pt.plots.plot_sorted_scores(
            (tn_support, tp_support),
            fnum=fnum, pnum=pnum,
            score_label='score',
            threshold_value=score_thresh,
            **support_sort_kw
        )

    def _prob_support_sorted(fnum, pnum):
        tp_probs = probs[labels]
        tn_probs = probs[np.logical_not(labels)]
        pt.plots.plot_sorted_scores(
            (tn_probs, tp_probs),
            fnum=fnum, pnum=pnum,
            score_label='prob',
            threshold_value=prob_thresh,
            **support_sort_kw
        )
        #ax = pt.gca()
        #max_score = max(tn_support.max(), tp_support.max())
        #ax.set_ylim(-max_score, max_score)

    def _prebayes(fnum, pnum):
        plot_prebayes_pdf(score_domain, p_score_given_tn, p_score_given_tp, p_score,
                          cfgstr='', fnum=fnum, pnum=pnum)

    def _postbayes(fnum, pnum):
        plot_postbayes_pdf(score_domain, p_tn_given_score, p_tp_given_score, prob_thresh=prob_thresh,
                           cfgstr='', fnum=fnum, pnum=pnum)
    def _roc(fnum, pnum):
        confusions.draw_roc_curve(fnum=fnum, pnum=pnum)

    def _precision_recall(fnum, pnum):
        confusions.draw_precision_recall_curve(fnum=fnum, pnum=pnum)

    if with_scores:
        inter.append_plot(_score_support_sorted)
        #inter.append_plot(_score_support_hist)
    if with_prebayes:
        inter.append_plot(_prebayes)
    if with_postbayes:
        inter.append_plot(_postbayes)
    if with_normscore:
        inter.append_plot(_prob_support_sorted)
        #inter.append_plot(_prob_support_hist)
    if with_roc:
        inter.append_plot(_roc)
    if with_precision_recall:
        inter.append_plot(_precision_recall)

    inter.show_page()

    if figtitle is not None:
        pt.set_figtitle(figtitle)

    return inter


def plot_prebayes_pdf(score_domain, p_score_given_tn, p_score_given_tp, p_score,
                      cfgstr='', fnum=None, pnum=(1, 1, 1)):
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED
    #unknown_color = pt.UNKNOWN_PURP
    unknown_color = pt.PURPLE2
    #unknown_color = pt.GRAY

    pt.plots.plot_probabilities(
        (p_score_given_tn,  p_score_given_tp, p_score),
        ('p(score | tn)', 'p(score | tp)', 'p(score)'),
        prob_colors=(false_color, true_color, unknown_color),
        figtitle='pre_bayes pdf score',
        xdata=score_domain,
        fnum=fnum,
        pnum=pnum)


def plot_postbayes_pdf(score_domain, p_tn_given_score, p_tp_given_score, prob_thresh=None,
                       cfgstr='', fnum=None, pnum=(1, 1, 1)):
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED

    pt.plots.plot_probabilities(
        (p_tn_given_score, p_tp_given_score),
        ('p(tn | score)', 'p(tp | score)'),
        prob_colors=(false_color, true_color,),
        figtitle='post_bayes pdf score ' + cfgstr,
        xdata=score_domain, fnum=fnum, pnum=pnum,
        prob_thresh=prob_thresh)


# DEBUGGING FUNCTIONS


def test_score_normalization(tp_support, tn_support, with_scores=True,
                             verbose=True, with_roc=True,
                             with_precision_recall=False, figtitle=None, normkw_varydict=None):
    """
    Gives an overview of how well threshold can be learned from raw scores.

    CommandLine:
        python -m vtool.score_normalization --test-test_score_normalization --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> # Shows how score normalization works with gaussian noise
        >>> from vtool.score_normalization import *  # NOQA
        >>> verbose = True
        >>> randstate = np.random.RandomState(seed=0)
        >>> # Get a training sample
        >>> tp_support = randstate.normal(loc=6.5, size=(256,))
        >>> tn_support = randstate.normal(loc=3.5, size=(256,))
        >>> test_score_normalization(tp_support, tn_support, verbose=verbose)
        >>> ut.show_if_requested()

    """
    import plottool as pt  # NOQA

    # Print raw score statistics
    ut.print_stats(tp_support, lbl='tp_support')
    ut.print_stats(tn_support, lbl='tn_support')

    # Test (potentially multiple) normalizing configurations
    if normkw_varydict is None:
        normkw_varydict = {
            'monotonize': [False],  # [True, False],
            #'adjust': [1, 4, 8],
            'adjust': [1],
            #'adjust': [8],
        }
    normkw_list = ut.util_dict.all_dict_combinations(normkw_varydict)

    if len(normkw_list) > 32:
        raise AssertionError('Too many plots to test!')

    for normkw in normkw_list:
        # Learn the appropriate normalization
        tup = learn_score_normalization(tp_support, tn_support,
                                         return_all=True, verbose=verbose,
                                         **normkw)
        (score_domain,
         p_tp_given_score, p_tn_given_score,
         p_score_given_tp, p_score_given_tn,
         p_score) = tup

        if verbose:
            print('plotting pdfs')
        fnum = pt.next_fnum()

        inspect_pdfs(tn_support, tp_support, score_domain, p_tp_given_score,
                     p_tn_given_score, p_score_given_tp, p_score_given_tn,
                     p_score, with_scores=with_scores, with_roc=with_roc,
                     with_precision_recall=with_precision_recall, fnum=fnum)

        pt.adjust_subplots(hspace=.3, bottom=.05, left=.05)

        if figtitle is not None:
            pt.set_figtitle(figtitle)
        else:
            pt.set_figtitle('ScoreNorm test' + ut.dict_str(normkw, newlines=False))

    locals_ = locals()
    return locals_


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
