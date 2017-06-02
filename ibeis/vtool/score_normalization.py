# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool
import numpy as np
import utool as ut
import six
import scipy.interpolate
from functools import partial
print, rrr, profile = utool.inject2(__name__, '[scorenorm]', DEBUG=False)


def check_unused_kwargs(kwargs, expected_keys):
    unused_keys = set(kwargs.keys()) - set(expected_keys)
    if len(unused_keys) > 0:
        print('unused kwargs keys = %r' % (unused_keys))


def testdata_score_normalier(tp_bumps=[(6.5, 256)], tn_bumps=[(3.5, 256)], tp_scale=1.0, tn_scale=1.0, min_clip=None, **kwargs):
    rng = np.random.RandomState(seed=0)
    # Get a training sample
    tp_support = np.hstack([rng.normal(loc=loc, scale=tp_scale, size=(size,)) for loc, size in tp_bumps])
    tn_support = np.hstack([rng.normal(loc=loc, scale=tn_scale, size=(size,)) for loc, size in tn_bumps])
    if min_clip is not None:
        tp_support[tp_support < min_clip] = min_clip
        tn_support[tn_support < min_clip] = min_clip
    data   = np.hstack((tp_support, tn_support))
    labels = np.array([True] * len(tp_support) + [False] * len(tn_support))
    encoder = ScoreNormalizer(**kwargs)
    encoder.fit(data, labels)
    return encoder, data, labels


def get_left_area(ydata, xdata, index_list):
    """ area to the left of each index point """
    left_area = np.array([np.trapz(ydata[:ix + 1], xdata[:ix + 1])
                          for ix in index_list])
    return left_area


def get_right_area(ydata, xdata, index_list):
    """ area to the right of each index point """
    right_area = np.array([np.trapz(ydata[ix:], xdata[ix:])
                           for ix in index_list])
    return right_area


class ScoreNormVisualizeClass(object):
    """
    # HACK; eventually move all individual plots into a class structure
    """

    def _hack_vizlearn(encoder, **kwargs):
        if 'target_tpr' in kwargs:
            print('_HACK VIZLERAN TARGET')
            verbose = ut.VERBOSE
            score_thresh = encoder.learn_threshold(verbose=verbose, **kwargs)
            prob_thresh = encoder.learned_thresh
            #prob_thresh = encoder.normalize_scores(score_thresh)
        else:
            #prob_thresh = encoder.learned_thresh
            #score_thresh = encoder.inverse_normalize(prob_thresh)
            print('_HACK VIZLERAN2')
            score_thresh = encoder.learn_threshold2()
            prob_thresh = encoder.normalize_scores(score_thresh)
        return score_thresh, prob_thresh

    def _plot_score_support_hist(encoder, fnum, pnum=(1, 1, 1), **kwargs):
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        tup = encoder.get_partitioned_support()
        tp_support, tn_support, part_attrs = tup
        score_thresh, prob_thresh = encoder._hack_vizlearn(**kwargs)

        true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
        false_color = pt.FALSE_RED
        support_kw = dict(
            score_lbls=('trueneg', 'truepos'),
            score_colors=(false_color, true_color),
            titlesuf=kwargs.get('titlesuf', '')
        )
        score_range = kwargs.get('score_range', None)
        pt.plot_score_histograms(
            (tn_support, tp_support),
            score_thresh=score_thresh,
            score_label='score',
            fnum=fnum,
            pnum=pnum,
            bin_width=kwargs.get('bin_width', None),
            num_bins=kwargs.get('num_bins', None),
            overlay_prob_given_list=(encoder.p_score_given_tn, encoder.p_score_given_tp),
            overlay_score_domain=encoder.score_domain,
            xlim=score_range,
            **support_kw)

    def _plot_roc(encoder, fnum, pnum, **kwargs):
        import vtool as vt
        import plottool as pt  # NOQA
        tup = encoder.get_partitioned_support()
        tp_support, tn_support, part_attrs = tup

        scores = np.hstack([tn_support, tp_support])
        labels = np.array([False] * len(tn_support) + [True] * len(tp_support))

        # probs = encoder.normalize_scores(scores)
        probs = normalize_scores(encoder.score_domain, encoder.p_tp_given_score, scores)

        confusions = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)

        score_thresh, prob_thresh = encoder._hack_vizlearn(**kwargs)

        #target_tpr = None
        target_tpr = confusions.get_metric_at_thresh('tpr', prob_thresh)
        #print('target_tpr = %r' % (target_tpr,))
        ROCInteraction = vt.interact_roc_factory(confusions, target_tpr,
                                                 show_operating_point=True)
        fnum = pt.ensure_fnum(fnum)
        ROCInteraction.static_plot(fnum, pnum, **kwargs)

    def _plot_prebayes(encoder, fnum, pnum, **kwargs):
        score_thresh, prob_thresh = encoder._hack_vizlearn(**kwargs)
        plot_prebayes_pdf(
            encoder.score_domain, encoder.p_score_given_tn, encoder.p_score_given_tp, encoder.p_score,
            score_thresh=score_thresh,
            cfgstr='', fnum=fnum, pnum=pnum)

    def _plot_postbayes(encoder, fnum, pnum, **kwargs):
        score_thresh, prob_thresh = encoder._hack_vizlearn(**kwargs)
        plot_postbayes_pdf(encoder.score_domain, encoder.p_tn_given_score, encoder.p_tp_given_score,
                           prob_thresh=prob_thresh,
                           score_thresh=score_thresh,
                           cfgstr='', fnum=fnum, pnum=pnum)


@six.add_metaclass(ut.ReloadingMetaclass)
class ScoreNormalizer(ut.Cachable, ScoreNormVisualizeClass):
    """
    Conforms to scikit-learn Estimator interface

    CommandLine:
        python -m vtool.score_normalization --test-ScoreNormalizer --show --cmd

    Kwargs:
        tpr (float): target true positive rate (default .90)
        fpr (float): target false positive rate (default None)
        gridsize=1024,
        adjust=8,
        monotonize=False, if True ensures inferred probability curves are monotonic
        clip_factor=None,
        reverse (bool): True if lower scores are better, False if higher scores
            are better (default=None)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> import vtool as vt
        >>> encoder = ScoreNormalizer()
        >>> X, y = vt.tests.dummy.testdata_binary_scores()
        >>> attrs = {'index': np.arange(len(y)) * ((2 * y) - 1)}
        >>> encoder.fit(X, y, attrs)
        >>> ut.quit_if_noshow()
        >>> encoder.visualize()
        >>> ut.show_if_requested()
    """
    def __init__(encoder, **kwargs):
        encoder.learn_kw = ut.update_existing(
            dict(
                gridsize=1024,
                adjust=8,
                monotonize=False,
                #monotonize=True,
                #clip_factor=(ut.PHI + 1),
                clip_factor=None,
                reverse=None,
                p_tp_method='eq',
            ), kwargs)
        #check_unused_kwargs(kwargs, encoder.learn_kw.keys())
        encoder.thresh_kw = ut.update_existing(
            dict(
                # Target recall for learned threshold
                tpr=None,
                fpr=None,
            ), kwargs)
        assert not any(key.startswith('target_') for key in kwargs), (
            'old interface of target_<metric> used just use <metric>')
        if not any(encoder.thresh_kw.values()):
            encoder.thresh_kw['tpr'] = .90
        # Support data
        encoder.support = dict(
            X=None,
            y=None,
            attrs=None,
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
        # Learned interpolation function
        encoder.interp_fn = None

    def __getstate__(encoder):
        """

        CommandLine:
            python -m vtool.score_normalization --test-__getstate__

        Example:
            >>> # ENABLE_DOCTEST
            >>> from vtool.score_normalization import *  # NOQA
            >>> encoder = ScoreNormalizer()
            >>> from six.moves import cPickle as pickle
            >>> dump = pickle.dumps(encoder)
            >>> encoder2 = pickle.loads(dump)
        """
        state_dict = encoder.__dict__.copy()
        state_dict['interp_fn'] = None
        return state_dict

    def get_prefix(encoder):
        return 'ScoreNorm_'

    def __setstate__(encoder, state_dict):
        encoder.__dict__.update(state_dict)
        encoder._update_interp_fn()

    def fit(encoder, X, y, attrs=None, verbose=False, finite_only=True):
        """
        Fits estimator to data

        Args:
            X (ndarray): one dimensional scores
            y (ndarray): binary labels
            attrs (dict): dictionary of data attributes
        """
        # Record support
        encoder.support['X'] = X
        encoder.support['y'] = y
        encoder.support['attrs'] = {} if attrs is None else attrs
        encoder.learn_probabilities(verbose=verbose)
        try:
            encoder.learn_threshold(verbose=verbose)
        except Exception as ex:
            ut.printex(ex, 'could not learn thresh', iswarning=True)

    @staticmethod
    # @ut.apply_docstr(flatten_scores)
    def _to_xy(tp_scores, tn_scores, part_attrs=None):
        return flatten_scores(tp_scores, tn_scores, part_attrs)

    @staticmethod
    # @ut.apply_docstr(partition_scores)
    def _to_partitioned(X, y, attrs={}):
        return partition_scores(X, y, attrs)

    def fit_partitioned(encoder, tp_scores, tn_scores, part_attrs=None,
                        **kwargs):
        """ convinience func to fit only scores that have been separated
        instead of labeled"""
        fitargs = flatten_scores(tp_scores, tn_scores, part_attrs)
        return encoder.fit(*fitargs, **kwargs)

    def get_partitioned_support(encoder):
        """ convinience get prepartitioned data """
        X, y, attrs = encoder.get_support()
        return partition_scores(X, y, attrs)

    def get_support(encoder, finite_only=True):
        """
        return X, y, and attrs
        """
        X = encoder.support['X']
        y = encoder.support['y']
        attrs = encoder.support['attrs']
        if finite_only:
            mask = np.isfinite(X)
            X = X.compress(mask)
            y = y.compress(mask)
            attrs = ut.map_dict_vals(partial(np.compress, mask), attrs)
        return X, y, attrs

    def learn_probabilities(encoder, verbose=False):
        """
        Kernel density estimation
        """
        #X, y = encoder.get_support()
        tp_support, tn_support, part_attrs = encoder.get_partitioned_support()
        # heuristic
        encoder.learn_kw['reverse'] = tp_support.mean() < tn_support.mean()
        if verbose:
            print('[scorenorm] setting reverse = %r' %
                  (encoder.learn_kw['reverse']))

        tup = learn_score_normalization(tp_support, tn_support, return_all=True,
                                        verbose=verbose, **encoder.learn_kw)
        # unpack
        (score_domain, p_tp_given_score, p_tn_given_score,
         p_score_given_tp, p_score_given_tn, p_score) = tup

        encoder.score_domain = score_domain
        encoder.p_tp_given_score = p_tp_given_score
        encoder.p_tn_given_score = p_tn_given_score
        encoder.p_score_given_tn = p_score_given_tn
        encoder.p_score_given_tp = p_score_given_tp
        encoder.p_score = p_score
        encoder._update_interp_fn()

    def _update_interp_fn(encoder):
        """
        Internal call to update interpolation function. Used when learning and
        when loading from cache.
        """
        if encoder.p_tp_given_score is not None:
            encoder.interp_fn = scipy.interpolate.interp1d(
                encoder.score_domain, encoder.p_tp_given_score, kind='linear',
                copy=False, assume_sorted=False)

    def learn_threshold2(encoder):
        """
        Finds a cutoff where the probability of a truepos stats becoming
        greater than probability of trueneg

        CommandLine:
            python -m vtool.score_normalization --exec-learn_threshold2 --show

        Example:
            >>> from vtool.score_normalization import *  # NOQA
            >>> import vtool as vt
            >>> #encoder, X, y = testdata_score_normalier([(3.5, 256), (9.5, 1024), (15.5, 2048)], [(6.5, 256), (12.5, 5064), (18.5, 128)], adjust=1, p_tp_method='ratio')
            >>> encoder, X, y = testdata_score_normalier([(3.5, 64), (9.5, 1024), (15.5, 5064)], [(6.5, 256), (12.5, 2048), (18.5, 128)], adjust=1, p_tp_method='ratio')
            >>> #encoder, X, y = testdata_score_normalier(adjust=1)
            >>> #encoder, X, y = testdata_score_normalier([(3.5, 2048)], [(30.5, 128)], tn_scale=.1, adjust=1)
            >>> #encoder, X, y = testdata_score_normalier([(0, 64)], [(-.1, 12)], adjust=8, min_clip=0)
            >>> locals_ = ut.exec_func_src(encoder.learn_threshold2)
            >>> exec(ut.execstr_dict(locals_))
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.ensureqt()
            >>> #pt.plot(xdata[0:-2], np.diff(np.diff(closeness)))
            >>> #maxima_x, maxima_y, argmaxima = vt.hist_argmaxima(closeness)
            >>> fnum = 100
            >>> pt.multi_plot(xdata, [tp_curve, tn_curve, closeness, ],
            >>>               label_list=['p(tp | s)', 'p(tn | s)', 'closeness', ], marker='',
            >>>               linewidth_list=[4, 4, 1,], title='intersection points',
            >>>               pnum=(4, 1, 1), fnum=fnum, xmax=xdata.max(), xmin=0)
            >>> pt.plot(xdata[argmaxima], closeness[argmaxima], 'rx', label='closeness maxima')
            >>> pt.plot(x_submax, y_submax, 'o', label='chosen')
            >>> #pt.plot(xdata[argmaxima], curveness[argmaxima], 'rx', label='curveness maxima')
            >>> pt.legend()
            >>> #pt.plot(x_submax, y_submax, 'o')
            >>> pt.plot(xdata[argmaxima], tp_curve[argmaxima], 'rx')
            >>> pt.plot(xdata[argmaxima], tn_curve[argmaxima], 'rx')
            >>> pt.plot(xdata[argmaxima], tp_curve[argmaxima], 'rx')
            >>> pt.plot(xdata[argmaxima], tn_curve[argmaxima], 'rx')
            >>> #pt.plot(xdata[argmaxima], encoder.interp_fn(x_submax), 'rx')
            >>> _mkinterp = ut.partial(
            >>>     scipy.interpolate.interp1d, kind='linear', copy=False,
            >>>     assume_sorted=False, bounds_error=False)
            >>> _interp_sgtn = _mkinterp(xdata, tn_curve)
            >>> _interp_sgtp = _mkinterp(xdata, tp_curve)
            >>> pt.plot(x_submax, _interp_sgtn(x_submax), 'go')
            >>> pt.plot(x_submax, _interp_sgtp(x_submax), 'bx')
            >>> #
            >>> pt.multi_plot(xdata[argmaxima], [tp_area, fp_area, tn_area, fn_area], title='intersection areas',
            >>>               label_list=['tp_area', 'fp_area', 'tn_area', 'fn_area'], markers=['o', 'd', 'o', '.'],
            >>>               pnum=(4, 1, 2), fnum=fnum, xmax=xdata.max(), xmin=0)
            >>> #
            >>> pt.multi_plot(xdata[argmaxima], [lr_pos, lr_neg, acc], title='intersection quality (liklihood ratios)',
            >>>               label_list=['lr_pos=tp/fp', 'lr_neg=fn/tn', 'acc'], markers=['o', 'o', '*'],
            >>>               pnum=(4, 1, 3), fnum=fnum, xmax=xdata.max(), xmin=0)
            >>> #
            >>> pnum_ = pt.make_pnum_nextgen(4, 3, start=9)
            >>> encoder._plot_score_support_hist(fnum=fnum, pnum=pnum_())
            >>> #encoder._plot_prebayes(fnum=fnum, pnum=pnum_())
            >>> encoder._plot_postbayes(fnum=fnum, pnum=pnum_())
            >>> encoder._plot_roc(fnum=fnum, pnum=pnum_())
            >>> pt.adjust_subplots(hspace=.5, top=.95, bottom=.08)
            >>> pt.show_if_requested()
        """
        import vtool as vt
        if False:
            # New stuff with area should make this irrelevant
            #weights = encoder.p_score_given_tp
            weights = encoder.p_score_given_tn
            values = encoder.score_domain
            mean = np.average(values, weights=weights)
            std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
            score_cutoff = mean + (2.5 * std)
            cutx = np.sum(encoder.score_domain < score_cutoff)
            xdata = encoder.score_domain[:cutx]
            tp_curve = encoder.p_score_given_tp[:cutx]
            tn_curve = encoder.p_score_given_tn[:cutx]
        else:
            xdata = encoder.score_domain
            tp_curve = encoder.p_score_given_tp
            tn_curve = encoder.p_score_given_tn
            if 0:
                _p = np.where(xdata > 50)[0][0]
                xdata = xdata[0:_p]
                tp_curve = tp_curve[0:_p]
                tn_curve = tn_curve[0:_p]
            #print('xdata = %r' % (xdata,))
            #print('tp_curve = %r' % (tp_curve,))
            #print('tn_curve = %r' % (tn_curve,))
            #tp_curve[:] = .1
            #tn_curve[:] = .5

        # Find locations of intersection
        closeness = -np.abs(tp_curve - tn_curve)
        closeness = closeness - closeness.min()
        #closeness
        #print('closeness = %r' % (closeness,))
        #argmaxima = vt.hist_argmaxima2(closeness)
        #print('argmaxima = %r' % (argmaxima,))

        argmaxima = vt.hist_argmaxima2(closeness)
        #argmaxima = np.arange(2, len(closeness) - 2)
        #curvature = -np.gradient(np.gradient(closeness))
        #curveness = (curvature - curvature.min()) / (curvature.max() - curvature.min())
        # Remove maxima points with almost no curvature
        #if False:
        #    if len(argmaxima) > 1:
        #        #curveness[argmaxima]
        #        curvature_  = curvature[argmaxima]
        #        valid = curvature_ > 1e-5
        #        #valid = curvature[argmaxima] > np.median(curvature[argmaxima])
        #        #valid = curvature[argmaxima] > np.median(curvature[argmaxima])
        #        if np.any(valid):
        #            argmaxima = argmaxima[valid]

        #argmaxima2 = vt.hist_argmaxima2(-deriv_no2)
        #if len(np.intersect1d(argmaxima2, argmaxima)) > 0:
        #    argmaxima = np.intersect1d(argmaxima2, argmaxima)

        # Now find which intersection points are "best"
        # TODO: have user specify metric that they care about
        # https://en.wikipedia.org/wiki/Confusion_matrix
        if True:
            # Use area under curves to determine the probability density of tp,
            # fp, tn, fn at each candidate threshold.
            tp_area = get_right_area(tp_curve, xdata, argmaxima)
            fp_area = get_right_area(tn_curve, xdata, argmaxima)
            tn_area = get_left_area(tn_curve, xdata, argmaxima)
            fn_area = get_left_area(tp_curve, xdata, argmaxima)  # NOQA
            # Choose the location of intersection that performs best on some test
            # statistic. (Positive likelihood ratio)
            lr_pos_ = tp_area / fp_area
            lr_neg_ = fn_area / tn_area
            # Accuracy is (tp + tn) / total
            acc = (tp_area + tn_area) / 2
            #print('lr_neg = %r' % (lr_neg,))
            #print('lr_pos = %r' % (lr_pos,))
            # Normalize likelihood into range 0 to 1
            pos_norm = max(1, vt.safe_max(lr_pos_, fill=1, finite=True, nans=False))
            neg_norm = max(1, vt.safe_max(lr_neg_, fill=1, finite=True, nans=False))
            lr_pos = lr_pos_ / pos_norm  # NOQA
            lr_neg = lr_neg_ / neg_norm  # NOQA

            #chosen_metric = lr_pos
            chosen_metric = acc
            # Invalidate impossible values
            isvalid = np.isfinite(chosen_metric)
            if np.any(isvalid):
                valid_argmaxima = argmaxima[isvalid]
                chosen_metric = chosen_metric[isvalid]
            else:
                valid_argmaxima = argmaxima

            # Invalidate values based on "reasonable" heuristics
            #reasonable_tp = tp_area > .1
            #reasonable_fp = tn_area > .1
            #reasonable_flags = np.logical_and(reasonable_tp, reasonable_fp)
            #if np.any(reasonable_flags):
            #    lr_pos = lr_pos.compress(reasonable_flags)
            #    lr_neg = lr_neg.compress(reasonable_flags)

            # Choose a finite argmax
            sortx = chosen_metric.argsort()[::-1]
            closeness_argmax = valid_argmaxima[sortx[0]]
            #maxpos = valid_argmaxima[lr_pos.argmax()]
            #print('lr_pos.argmax = %r' % (lr_pos.argmax,))
            # Hack for infinity and nans. bring thems out of the 0 and 1 range, but only by a bit.
            #lr_pos[np.isnan(lr_pos)] = -.1
            #lr_neg[np.isnan(lr_neg)] = -.1
            #lr_pos[np.isinf(lr_pos)] = 1.1
            #lr_neg[np.isinf(lr_neg)] = 1.1
        else:
            closeness_argmax = closeness.argmax()

        if closeness_argmax == len(closeness) - 1:
            y_submax = closeness[-2:-1]
            x_submax = xdata[-2:-1]
        elif closeness_argmax == 0:
            y_submax = closeness[0:1]
            x_submax = xdata[0:1]
        else:
            # argmaxima, hist_, centers = maxpos, closeness, xdata
            x_submax, y_submax = vt.interpolate_submaxima(np.array([closeness_argmax]), closeness, xdata)
        score_thresh = x_submax[0]
        if ut.get_argflag('--debug-scorethresh') and not getattr(encoder, 'block', False):
            encoder.block = True
            ut.exec_func_doctest(encoder.learn_threshold2,
                                 start_sentinal='import plottool as pt',
                                 end_sentinal='pt.show_if_requested()')
            encoder.block = False
        return score_thresh

    def learn_threshold(encoder, verbose=False, **thresh_kw):
        """
        Learns cutoff threshold that achieves the target confusion metric
        Typically a desired false positive rate (recall) is specified
        """
        import vtool as vt
        # select a cutoff threshold
        #import sklearn.metrics
        if len(thresh_kw) > 0:
            _thresh_kw = ut.map_dict_keys(lambda x: x.replace('target_', ''), thresh_kw)
        else:
            _thresh_kw = encoder.thresh_kw
        # Select threshold that gives target confusion
        _selected_items = [item for item in _thresh_kw.items()
                           if item[1] is not None]
        assert len(_selected_items) == 1, (
            'Can only specify one desired confusion metric')
        # choose how to optimize the threshold
        metric, value = _selected_items[0]
        # Get classifier confusions (maybe dont need probs here)
        X, y, attrs = encoder.get_support()
        probs = encoder.normalize_scores(X)

        if False:
            confusions_score = vt.ConfusionMetrics.from_scores_and_labels(
                -X, y, verbose=verbose)

            confusions_prob = vt.ConfusionMetrics.from_scores_and_labels(
                probs, y, verbose=verbose)

            _score_thresh = confusions_score.get_thresh_at_metric(metric, value)
            _prob_thresh = confusions_prob.get_thresh_at_metric(metric, value)
            _inv_score = encoder.inverse_normalize(_prob_thresh)
            _inv_prob = encoder.normalize_scores(-_score_thresh)

            _inv_prob - _prob_thresh
            _inv_score - (-_score_thresh)

        confusions = vt.ConfusionMetrics.from_scores_and_labels(
            probs, y, verbose=verbose)

        prob_thresh = confusions.get_thresh_at_metric(metric, value)

        #target_value = confusions.get_metric_at_thresh(metric, prob_thresh)
        #check_thresh = confusions.get_thresh_at_metric(metric, target_value)
        score_thresh = encoder.inverse_normalize(prob_thresh)
        if verbose:
            print('[scorenorm] Learning threshold to achieve %s=%.5f' % (
                metric.upper(), value,))
            if encoder.learned_thresh is not None:
                print('[scorenorm]   * learned_thresh = %.5f' % (
                    encoder.learned_thresh,))
            else:
                print('[scorenorm]   * learned_thresh = %r' % (encoder.learned_thresh,))
            print('[scorenorm]   * score_thresh = %.5f' % (
                score_thresh,))
            if metric == 'tpr':
                print('[scorenorm]   * fpr = %r' % (
                    confusions.get_fpr_at_recall(value),))
            elif metric == 'fpr':
                print('[scorenorm]   * tpr = %r' % (
                    confusions.get_recall_at_fpr(value),))

        # TODO: maybe do not change state?
        encoder.learned_thresh = prob_thresh
        return score_thresh

    def inverse_normalize(encoder, probs):
        inverse_interp = scipy.interpolate.interp1d(
            encoder.p_tp_given_score, encoder.score_domain, kind='linear',
            copy=False, assume_sorted=False)
        scores = inverse_interp(probs)
        return scores

    def normalize_scores(encoder, X):
        is_iterable = ut.isiterable(X)
        if not is_iterable:
            X = np.array([X])
        prob = normalize_scores(
            encoder.score_domain, encoder.p_tp_given_score,
            X, interp_fn=encoder.interp_fn)
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
        r"""
        Returns the indicies of the most difficult type I and type II errors.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.score_normalization import *  # NOQA
            >>> encoder, X, y = testdata_score_normalier()
            >>> (fp_indicies, fn_indicies) = encoder.get_error_indicies(X, y)
            >>> fp_X = X.take(fp_indicies)[0:3]
            >>> fn_X = X.take(fn_indicies)[0:3]
            >>> result =    'fp_X = ' + ut.numpy_str2(fp_X)
            >>> result += '\nfn_X = ' + ut.numpy_str2(fn_X)
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
            >>> result =    'tp_X = ' + ut.numpy_str2(tp_X)
            >>> result += '\ntn_X = ' + ut.numpy_str2(tn_X)
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
        r"""
        shows details about the score normalizer

        Kwargs:
            fnum
            figtitle
            with_hist
            interactive
            with_scores
            with_roc
            with_precision_recall

        CommandLine:
            python -m vtool.score_normalization --exec-ScoreNormalizer.visualize:0 --show
            python -m vtool.score_normalization --exec-ScoreNormalizer.visualize:1 --show

        Example0:
            >>> # UNSTABLE_DOCTEST
            >>> from vtool.score_normalization import *  # NOQA
            >>> import vtool as vt
            >>> encoder = ScoreNormalizer()
            >>> X, y = vt.tests.dummy.testdata_binary_scores()
            >>> encoder.fit(X, y)
            >>> kwargs = dict(
            >>>     with_pr=True, interactive=True, with_roc=True,
            >>>     with_hist=True)
            >>> encoder.visualize(**kwargs)
            >>> ut.show_if_requested()

        Example1:
            >>> # UNSTABLE_DOCTEST
            >>> from vtool.score_normalization import *  # NOQA
            >>> import vtool as vt
            >>> encoder = ScoreNormalizer()
            >>> X, y = vt.tests.dummy.testdata_binary_scores()
            >>> encoder.fit(X, y)
            >>> kwargs = dict(
            >>>     with_pr=True, interactive=True, with_roc=True, with_hist=True,
            >>>     with_scores=False, with_prebayes=False, with_postbayes=False)
            >>> encoder.visualize(target_tpr=.95, **kwargs)
            >>> ut.show_if_requested()
        """
        #import plottool as pt
        default_kw = dict(
            with_scores=False,
            with_roc=True,
            #with_roc=False,
            #with_precision_recall=True,
            with_precision_recall=False,
            #with_hist=False,
            with_hist=True,
            fnum=None,
            figtitle=None,
            #interactive=None,
            interactive=False,
            use_stems=None,
            attr_callback=None,
            with_prebayes=True,
            with_postbayes=True,
            score_range=None,
            bin_width=None,
            logscale=False,
        )
        alias_dict = {'with_pr': 'with_precision_recall'}
        # inspect_kw = ut.update_existing(default_kw, kwargs, alias_dict=alias_dict)
        inspect_kw = ut.update_dict(default_kw, kwargs, alias_dict=alias_dict)
        print('inspect_kw = %r' % (inspect_kw,))
        other_kw = ut.delete_dict_keys(kwargs.copy(), list(inspect_kw.keys()) +
                                       list(alias_dict.keys()))

        score_thresh, prob_thresh = encoder._hack_vizlearn(**other_kw)

        tup = encoder.get_partitioned_support()
        tp_support, tn_support, part_attrs = tup
        inter = inspect_pdfs(
            tn_support, tp_support,
            encoder.score_domain, encoder.p_tp_given_score,
            encoder.p_tn_given_score, encoder.p_score_given_tp,
            encoder.p_score_given_tn, encoder.p_score,
            prob_thresh=prob_thresh,
            score_thresh=score_thresh, part_attrs=part_attrs,
            thresh_kw=encoder.thresh_kw,
            **inspect_kw)
        return inter


def partition_scores(X, y, attrs=None):
    """
    convinience helper to translate partitioned to unpartitioned data

    Args:
        tp_scores (ndarray):
        tn_scores (ndarray):
        attrs (dict): (default = None)

    Returns:
        tuple: (scores, labels, attrs)

    CommandLine:
        python -m vtool.score_normalization --test-partition_scores

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> X = np.array([5, 6, 6, 7, 1, 2, 2])
        >>> attrs = {'qaid': np.array([21, 24, 25, 26, 11, 14, 15])}
        >>> y = np.array([1, 1, 1, 1, 0, 0, 0], dtype=np.bool)
        >>> tup = partition_scores(X, y, attrs)
        >>> resdict = ut.odict(zip(
        >>>     ['tp_scores', 'tn_scores', 'part_attrs'], tup))
        >>> result = ut.dict_str(resdict, nobraces=True, with_dtype=False,
        >>>                      explicit=1, nl=2)
        >>> print(result)
        tp_scores=np.array([5, 6, 6, 7]),
        tn_scores=np.array([1, 2, 2]),
        part_attrs={
            False: {'qaid': np.array([11, 14, 15])},
            True: {'qaid': np.array([21, 24, 25, 26])},
        },

    """
    import vtool as vt
    import operator
    # Make partitioning
    unique_labels, groupxs = vt.group_indices(y)
    _grouper = partial(vt.apply_grouping, groupxs=groupxs)
    # Group data
    X_parts = _grouper(X)
    # Group attributes
    _nested_attrs = ut.map_dict_vals(_grouper, attrs)
    def _getitem(a, b):
        return operator.getitem(b, a)
    part_attrs = {
        label: ut.map_dict_vals(partial(_getitem, lblx), _nested_attrs)
        for lblx, label in enumerate(unique_labels)
    }
    assert len(unique_labels) == 2, 'exepcted two groups'
    assert not unique_labels[0], 'expected true negatives to be first'
    assert unique_labels[1], 'expected true positives to be second'
    #
    tn_scores, tp_scores = X_parts
    return tp_scores, tn_scores, part_attrs


def flatten_scores(tp_scores, tn_scores, part_attrs=None):
    """
    convinience helper to translate partitioned to unpartitioned data

    Args:
        tp_scores (ndarray):
        tn_scores (ndarray):
        part_attrs (dict): (default = None)

    Returns:
        tuple: (scores, labels, attrs)

    CommandLine:
        python -m vtool.score_normalization --test-flatten_scores

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> tp_scores = np.array([5, 6, 6, 7])
        >>> tn_scores = np.array([1, 2, 2])
        >>> part_attrs = {
        ...     1: {'qaid': [21, 24, 25, 26]},
        ...     0: {'qaid': [11, 14, 15]},
        ... }
        >>> tup = flatten_scores(
        ... tp_scores, tn_scores, part_attrs)
        >>> (X, y, attrs) = tup
        >>> y = y.astype(np.int)
        >>> resdict = ut.odict(zip(['X', 'y', 'attrs'], [X, y, attrs]))
        >>> result = ut.dict_str(resdict, nobraces=True, with_dtype=False,
        >>>                      explicit=1, nl=1)
        >>> print(result)
        X=np.array([5, 6, 6, 7, 1, 2, 2]),
        y=np.array([1, 1, 1, 1, 0, 0, 0]),
        attrs={'qaid': np.array([21, 24, 25, 26, 11, 14, 15])},
    """
    scores = np.hstack([tp_scores, tn_scores])
    labels = np.zeros(scores.size, dtype=np.bool)
    labels[0:len(tp_scores)] = True
    if part_attrs is None:
        return scores, labels
    else:
        tp_attrs = part_attrs[1]
        tn_attrs = part_attrs[0]
        assert (tp_attrs is None) == (tn_attrs is None), (
            'must specify both or none')
        assert sorted(tp_attrs.keys()) == sorted(tn_attrs.keys()), (
            'dicts do not agree')
        # attrs = ut.dict_isect_combine(tp_attrs, tn_attrs, combine_op=np.append)
        from functools import partial
        combine_op = partial(np.append, axis=0)
        attrs = ut.dict_isect_combine(tp_attrs, tn_attrs, combine_op=combine_op)
        num_attrs = np.array(list(map(len, attrs.values())))
        assert np.all(num_attrs == len(scores)), (
            'num_attrs=%r must agree with data. len(scores)=%r' % (
                num_attrs, len(scores)))
        return scores, labels, attrs


def learn_score_normalization(tp_support, tn_support, gridsize=1024, adjust=8,
                              return_all=False, monotonize=True,
                              clip_factor=(ut.PHI + 1), verbose=False,
                              reverse=False, p_tp_method='eq'):
    r"""
    Takes collected data and applys parzen window density estimation and bayes rule.

    #True positive scores must be larger than true negative scores.
    FIXME: might be an issue with pdfs summing to 1 here.

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
        >>> result = '%.2f' % (np.diff(p_tp_given_score).sum())
        >>> print(result)
        0.99
    """
    import vtool as vt
    if verbose:
        print('[scorenorm] Learning normalization pdf')
        print('[scorenorm] * tp_support.shape=%r' % (tp_support.shape,))
        print('[scorenorm] * tn_support.shape=%r' % (tn_support.shape,))
        print('[scorenorm] * estimating true positive pdf, ')
        print('[scorenorm] * monotonize = %r' % (monotonize,))
        print('stats.tp_support = ' + ut.get_stats_str(tp_support, use_nan=True))
        print('stats.tn_support = ' + ut.get_stats_str(tn_support, use_nan=True))
        next_ = ut.next_counter(1)
        total = 8
    # import utool
    # utool.embed()
    # Find good score domain range
    if True:
        min_score, max_score = find_clip_range(tp_support, tn_support, clip_factor, reverse)
    else:
        min_score = min(tp_support.min(), tn_support.min())
        max_score = min(tp_support.max(), tn_support.max())
    score_domain = np.linspace(min_score, max_score, gridsize)
    # Estimate true positive/negative density
    if verbose:
        print('[scorenorm] %d/%d estimating true negative pdf' % (next_(), total))
    score_tp_pdf = vt.estimate_pdf(tp_support, gridsize=gridsize, adjust=adjust)
    #assert score_tp_pdf.bw != 0, 'error bandwidth estimated to be 0'
    if verbose:
        print('[scorenorm] %d/%d estimating true negative pdf' % (next_(), total))
    score_tn_pdf = vt.estimate_pdf(tn_support, gridsize=gridsize, adjust=adjust)
    #assert score_tn_pdf.bw != 0, 'error bandwidth estimated to be 0'
    if verbose:
        print('[scorenorm] %d/%d estimating score domain' % (next_(), total))
    # Evaluate true negative density
    if verbose:
        print('[scorenorm] %d/%d evaluating tp density' % (next_(), total))
    p_score_given_tp = score_tp_pdf.evaluate(score_domain)
    if verbose:
        print('[scorenorm] %d/%d evaluating tn density' % (next_(), total))
    p_score_given_tn = score_tn_pdf.evaluate(score_domain)

    # Not sure why the pdfs returned from statsmodels dont integrate to 1

    if verbose:
        print('[sn.pre]stats.score_domain = ' + ut.get_stats_str(score_domain, use_nan=True, precision=5))
        print('[sn.pre]stats:p_score_given_tn = ' + ut.get_stats_str(p_score_given_tn, use_nan=True, precision=5))
        print('[sn.pre]stats:p_score_given_tp = ' + ut.get_stats_str(p_score_given_tp, use_nan=True, precision=5))
        #print('stats.tn_support = ' + ut.get_stats_str(tn_support, use_nan=True))

    problems = []

    try:
        assert not np.any(np.isnan(p_score_given_tp)), ('Need more positive support')
    except AssertionError as ex:  # NOQA
        print('[sn.pre]stats:tpsupport = ' + ut.get_stats_str(score_tp_pdf.support, use_nan=True, precision=5))
        problems += [str(ex)]

    try:
        assert not np.any(np.isnan(p_score_given_tn)), ('Need more negative support')
    except AssertionError as ex:  # NOQA
        print('[sn.pre]stats:tnsupport = ' + ut.get_stats_str(score_tn_pdf.support, use_nan=True, precision=5))
        problems += [str(ex)]

    if problems:
        raise AssertionError(', '.join(problems))

    if True:
        # Make sure we still have probability functions
        area_tp = np.trapz(p_score_given_tp, score_domain)
        area_tn = np.trapz(p_score_given_tn, score_domain)
        if verbose:
            print('pre.area_tp = %r' % (area_tp,))
            print('pre.area_tn = %r' % (area_tn,))

        # normalize to ensure
        p_score_given_tp = p_score_given_tp / area_tp
        p_score_given_tn = p_score_given_tn / area_tn

        area_tp = np.trapz(p_score_given_tp, score_domain)
        area_tn = np.trapz(p_score_given_tn, score_domain)
        #if ut.DEBUG2:
        if verbose:
            print('norm.area_tp = %r' % (area_tp,))
            print('norm.area_tn = %r' % (area_tn,))

        assert np.isclose(area_tp, 1.0), area_tp
        assert np.isclose(area_tn, 1.0), area_tn

    if verbose:
        print('[scorenorm] %d/%d evaluating posterior probabilities' % (next_(), total))
        print('p_tp_method = %r' % (p_tp_method,))

    # For inbalanced data there are several methods we might want to use to
    # calculate p_tp.  not always going to be equal probability of true and
    # positive cases
    if p_tp_method == 'eq':
        p_tp = .5
    elif p_tp_method == 'ratio':
        p_tp = len(tp_support) / (len(tp_support) + len(tn_support))
    else:
        raise NotImplementedError('p_tp_method = %r' % (p_tp_method,))
    # Average to get probablity of any score
    p_score = (np.array(p_score_given_tp) + np.array(p_score_given_tn)) / 2.0
    # Apply bayes
    p_tp_given_score = ut.bayes_rule(p_score_given_tp, p_tp, p_score)
    if ut.DEBUG2:
        assert np.isclose(np.trapz(p_score, score_domain), 1.0)
        assert np.isclose(np.trapz(p_score, p_tp_given_score), 1.0)
    if np.any(np.isnan(p_tp_given_score)):
        p_tp_given_score = vt.interpolate_nans(p_tp_given_score)
    if verbose:
        # np.trapz(p_tp_given_score / np.trapz(p_tp_given_score, score_domain), score_domain)
        print('stats:p_score_given_tn = ' + ut.get_stats_str(p_score_given_tn, newlines=0, use_nan=True, precision=5))
        print('stats:p_score_given_tp = ' + ut.get_stats_str(p_score_given_tp, newlines=0, use_nan=True, precision=5))
        print('stats:p_score = ' + ut.get_stats_str(p_score, newlines=0, use_nan=True, precision=5))
        print('stats:p_tp_given_score = ' + ut.get_stats_str(p_tp_given_score, newlines=0, use_nan=True, precision=5))
    if monotonize:
        if reverse:
            if verbose:
                print('[scorenorm] %d/%d monotonize decreasing' % (next_(), total))
            p_tp_given_score = vt.ensure_monotone_strictly_decreasing(
                p_tp_given_score, left_endpoint=1.0, right_endpoint=0.0)
        else:
            if verbose:
                print('[scorenorm] %d/%d monotonize increasing' % (next_(), total))
            #if False:
            #    flags = ~np.isnan(p_tp_given_score)
            #    pt.plot(score_domain[flags], p_tp_given_score[flags])
            #    pt.plot(score_domain, p_tp_given_score)
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


# DEBUGGING FUNCTIONS


def test_score_normalization(tp_support, tn_support, with_scores=True,
                             verbose=True, with_roc=True,
                             with_precision_recall=False, figtitle=None,
                             normkw_varydict=None):
    """
    Gives an overview of how well threshold can be learned from raw scores.

    DEPRICATE

    CommandLine:
        python -m vtool.score_normalization --test-test_score_normalization --show

    Example:
        >>> # GUI_DOCTEST
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

        if figtitle is not None:
            pt.set_figtitle(figtitle)
        else:
            pt.set_figtitle('ScoreNorm test' + ut.dict_str(normkw, newlines=False))

    locals_ = locals()
    return locals_


# --------
# Plotting
# --------


def plot_prebayes_pdf(score_domain, p_score_given_tn, p_score_given_tp, p_score,
                      cfgstr='', fnum=None, pnum=(1, 1, 1), **kwargs):
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
        #figtitle='pre_bayes pdf score',
        figtitle='p(score | truth)',
        xdata=score_domain,
        fnum=fnum,
        pnum=pnum, **kwargs)


def plot_postbayes_pdf(score_domain, p_tn_given_score, p_tp_given_score,
                       score_thresh=None, prob_thresh=None, cfgstr='',
                       fnum=None, pnum=(1, 1, 1)):
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED

    pt.plots.plot_probabilities(
        (p_tn_given_score, p_tp_given_score),
        ('p(tn | score)', 'p(tp | score)'),
        prob_colors=(false_color, true_color,),
        #figtitle='post_bayes pdf score ' + cfgstr,
        figtitle='p(truth | score)' + cfgstr,
        xdata=score_domain, fnum=fnum, pnum=pnum,
        score_thresh=score_thresh,
        prob_thresh=prob_thresh)


def inspect_pdfs(tn_support, tp_support,
                 score_domain, p_tp_given_score,
                 p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score,
                 prob_thresh=None, score_thresh=None, with_scores=False,
                 with_roc=False, with_precision_recall=False, with_hist=False,
                 fnum=None, figtitle=None, interactive=None, use_stems=None,
                 part_attrs=None, thresh_kw=None, attr_callback=None,
                 with_prebayes=True, with_postbayes=True, score_range=None,
                 **kwargs):
    r"""
    Shows plots of learned thresholds

    CommandLine:
        python -m vtool.score_normalization --test-ScoreNormalizer --show
        python -m vtool.score_normalization --exec-ScoreNormalizer.visualize --show
    """
    import plottool as pt  # NOQA
    from plottool.interactions import ExpandableInteraction
    from plottool.abstract_interaction import AbstractInteraction
    import vtool as vt
    import plottool as pt  # NOQA

    if fnum is None:
        fnum = pt.next_fnum()

    with_normscore = with_scores
    #with_prebayes = True
    #with_postbayes = True

    nSubplots = (with_normscore + with_prebayes + with_postbayes +
                 with_scores + with_roc + with_precision_recall +
                 with_hist)
    if nSubplots == 0:
        raise ValueError('Must choose at least one subplot')
    nRows, nCols = pt.get_square_row_cols(nSubplots)
    _pnumiter = pt.make_pnum_nextgen(nRows=nRows, nCols=nCols,
                                     nSubplots=nSubplots)

    #print('Always interactive even if: interactive = %r' % (interactive,))
    # Make a plottool interaction
    inter = ExpandableInteraction(fnum, _pnumiter)

    scores = np.hstack([tn_support, tp_support])
    labels = np.array([False] * len(tn_support) + [True] * len(tp_support))

    # probs = encoder.normalize_scores(scores)
    probs = normalize_scores(score_domain, p_tp_given_score, scores)

    confusions = vt.ConfusionMetrics.from_scores_and_labels(
        probs, labels)
    # Hack change confusion prob thresholds to score thresholds
    if False:
        # Fixme: assume sorted
        inverse_interp = scipy.interpolate.interp1d(
            p_tp_given_score, score_domain, kind='linear',
            copy=False, assume_sorted=False)
        confusions._orig_thresholds = confusions.thresholds
        confusions.thresholds = inverse_interp(confusions.thresholds)
        confusions._hackscores = scores

    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED

    support_kw = dict(
        score_lbls=kwargs.get('score_lbls', ('trueneg', 'truepos')),
        score_colors=(false_color, true_color),
        logscale=kwargs.get('logscale', False),
    )
    support_sort_kw = dict(
        score_markers=['^', 'v'],
        markersizes=[5, 5],
        use_stems=use_stems,
        **support_kw
    )

    class SortedScoreSupportInteraction(AbstractInteraction):
        def __init__(data_pdf, **kwargs):
            super(SortedScoreSupportInteraction, data_pdf).__init__(**kwargs)
            data_pdf.tn_support = tn_support
            data_pdf.tp_support = tp_support
            data_pdf.part_attrs = part_attrs
            data_pdf.attr_callback = attr_callback
            data_pdf.sel_mode = 'tn'

        def toggle_mode(data_pdf):
            if data_pdf.sel_mode == 'tn':
                data_pdf.sel_mode = 'tp'
            else:
                data_pdf.sel_mode = 'tn'
            print('TOGGLE data_pdf.sel_mode = %r' % (self.sel_mode,))

        @staticmethod
        def static_plot(fnum, pnum):
            pt.plots.plot_sorted_scores(
                (tn_support, tp_support),
                fnum=fnum, pnum=pnum,
                score_label='score',
                thresh=score_thresh,
                **support_sort_kw
            )

        def plot(data_pdf, fnum, pnum):
            data_pdf.static_plot(fnum, pnum)

        def on_key_press(data_pdf, event):
            #print('event = %r' % (event,))
            #print('event.key = %r' % (event.key,))
            if event.key == 't':
                data_pdf.toggle_mode()

        def on_click_inside(data_pdf, event, ex):
            import vtool as vt
            #ax = event.inaxes
            #for l in ax.get_lines():
            #    print(l.get_label())
            tp_index, tp_dist = vt.closest_point(event.ydata, data_pdf.tp_support[:, None])
            tn_index, tn_dist = vt.closest_point(event.ydata, data_pdf.tn_support[:, None])
            print('closest tp_index = %r, %r' % (tp_index, tp_dist))
            print('closest tn_index = %r, %r' % (tn_index, tn_dist))
            SEL_TP = data_pdf.sel_mode == 'tp'
            print('data_pdf.sel_mode = %r' % (self.sel_mode,))
            if SEL_TP:
                tp_attrs = data_pdf.part_attrs[True]
                if len(tp_attrs) == 0:
                    print('No positive attrs')
                subattrs = ut.get_dict_column(tp_attrs, tp_index)
            else:
                tn_attrs = data_pdf.part_attrs[False]
                if len(tn_attrs) == 0:
                    print('No negative attrs')
                subattrs = ut.get_dict_column(tn_attrs, tn_index)
            print('subattrs = %r' % (subattrs,))
            if data_pdf.attr_callback is not None:
                print('Executing callback')
                data_pdf.attr_callback(**subattrs)
            #dists = vt.L1(event.ydata, data_pdf.tp_support[:, None])
            #index = dists.argsort()[0]
            #event.xdata
            # Find the nearest label
            pass

    #target_tpr = None
    target_tpr = confusions.get_metric_at_thresh('tpr', prob_thresh)
    #print('target_tpr = %r' % (target_tpr,))
    ROCInteraction = vt.interact_roc_factory(confusions, target_tpr,
                                             show_operating_point=True)

    def _score_support_hist(fnum, pnum):
        overlay_score_domain = None
        score_thresh_ = None
        if kwargs.get('histoverlay', True):
            overlay_score_domain = score_domain
            score_thresh_ = score_thresh

        print('support_kw = %r' % (support_kw,))

        pt.plot_score_histograms(
            (tn_support, tp_support),
            score_thresh=score_thresh_,
            score_label='score',
            fnum=fnum,
            pnum=pnum,
            bin_width=kwargs.get('bin_width', None),
            num_bins=kwargs.get('num_bins', None),
            overlay_prob_given_list=(p_score_given_tn, p_score_given_tp),
            overlay_score_domain=overlay_score_domain,
            xlim=score_range,
            histnorm=kwargs.get('histnorm', False),
            **support_kw)

    def _prob_support_hist(fnum, pnum):
        tp_probs = probs[labels]
        tn_probs = probs[np.logical_not(labels)]
        pt.plot_score_histograms(
            (tn_probs, tp_probs),
            score_label='prob',
            fnum=fnum,
            pnum=pnum,
            bin_width=kwargs.get('bin_width', None),
            num_bins=kwargs.get('num_bins', None),
            **support_kw)

    def _prob_support_sorted(fnum, pnum):
        tp_probs = probs[labels]
        tn_probs = probs[np.logical_not(labels)]
        pt.plots.plot_sorted_scores(
            (tn_probs, tp_probs),
            fnum=fnum, pnum=pnum,
            score_label='prob',
            thresh=prob_thresh,
            **support_sort_kw
        )
        #ax = pt.gca()
        #max_score = max(tn_support.max(), tp_support.max())
        #ax.set_ylim(-max_score, max_score)

    def _prebayes(fnum, pnum):
        plot_prebayes_pdf(
            score_domain, p_score_given_tn, p_score_given_tp, p_score,
            score_thresh=score_thresh,
            cfgstr='', fnum=fnum, pnum=pnum)

    def _postbayes(fnum, pnum):
        plot_postbayes_pdf(score_domain, p_tn_given_score, p_tp_given_score,
                           prob_thresh=prob_thresh,
                           score_thresh=score_thresh,
                           cfgstr='', fnum=fnum, pnum=pnum)

    def _precision_recall(fnum, pnum):
        confusions.draw_precision_recall_curve(fnum=fnum, pnum=pnum)

    if with_scores:
        inter.append_plot(SortedScoreSupportInteraction)
    if with_hist:
        inter.append_plot(_score_support_hist)
    if with_prebayes:
        inter.append_plot(_prebayes)
    if with_postbayes:
        inter.append_plot(_postbayes)
    if with_normscore:
        inter.append_plot(_prob_support_sorted)
        #inter.append_plot(_prob_support_hist)
    if with_roc:
        inter.append_plot(ROCInteraction)
    if with_precision_recall:
        inter.append_plot(_precision_recall)

    inter.start()

    if figtitle is not None:
        pt.set_figtitle(figtitle)

    return inter


def estimate_pdf(data, gridsize=1024, adjust=1):
    """
    estimate_pdf

    References;
        http://statsmodels.sourceforge.net/devel/generated/statsmodels.nonparametric.kde.KDEUnivariate.html
        https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    Args:
        data (ndarray): 1 dimensional data of float64
        gridsize(int): domain size
        adjust(int): smoothing factor

    Returns:
        ndarray: data_pdf

    CommandLine:
        python -m vtool.score_normalization estimate_pdf --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.score_normalization import *  # NOQA
        >>> import vtool as vt
        >>> rng = np.random.RandomState(0)
        >>> data = rng.randn(1000)
        >>> data_pdf = vt.estimate_pdf(data)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> #pt.plot(data_pdf.support, data_pdf.cdf)
        >>> #pt.plot(data_pdf.support, data_pdf.density)
        >>> pt.plot(data_pdf.support[:-1], np.diff(data_pdf.cdf))
        >>> #pt.plot(data_pdf.cumhazard)
        >>> ut.show_if_requested()

    Ignore:
        mx = data_pdf.support.max()
        mn = data_pdf.support.min()
        scipy.integrate.quad(data_pdf.evaluate, mn, mx)

        assert np.isclose(np.sum(np.diff(data_pdf.support)[0] * data_pdf.density), 1)
        assert np.isclose(np.trapz(data_pdf.density, data_pdf.support), 1)

        np.trapz(data_pdf.density, data_pdf.support)
    """
    import utool as ut
    import numpy as np
    import statsmodels.nonparametric.kde
    import statsmodels.nonparametric.bandwidths
    #import scipy.stats as spstats
    #import statsmodelskde.score_samples(support[:, None])

    if True:
        # Ensure that a non-zero bandwidth is chosen
        # bw_choices = ['scott', 'silverman', 'normal_reference']
        # bw = bw_choices[1]
        for bw in ['silverman', 'scott']:
            bw_value = statsmodels.nonparametric.bandwidths.select_bandwidth(data, bw, None)
            if bw_value > 0:
                break
        if bw_value == 0:
            sorted_diffs = np.diff(sorted(data))
            nonzero_diffs = sorted_diffs[sorted_diffs > 0]
            if len(nonzero_diffs) > 0:
                median_diff = np.median(nonzero_diffs)
                bw_value = np.sqrt(median_diff)
            else:
                # use a very small value
                bw_value = 1e-9
    try:

        if False:
            # Alternate implementation in case statsmodels breaks
            class TempPdf():
                def __init__(data_pdf, data, bw_value, gridsize):
                    from sklearn.neighbors.kde import KernelDensity
                    kde = KernelDensity(kernel='gaussian', bandwidth=bw_value)
                    kde.fit(data[:, None])
                    data_pdf.kde = kde
                    data_pdf.support = np.linspace(data.min(), data.max(), gridsize)
                    data_pdf.density = data_pdf.evaluate(data_pdf.support)
                    # import scipy as sp
                    data_pdf.cdf = data_pdf.density.cumsum()
                    # data_pdf.cdf = sp.integrate.cumtrapz(data_pdf.density,
                                                           # data_pdf.support)

                def evaluate(data_pdf, scores):
                    return np.exp(data_pdf.kde.score_samples(scores[:, None]))
            data_pdf = TempPdf(data, bw_value, gridsize)

        data_pdf = statsmodels.nonparametric.kde.KDEUnivariate(data)
        fitkw = dict(kernel='gau',
                     bw=bw_value,
                     fft=True,
                     weights=None,
                     adjust=adjust,
                     cut=3,
                     gridsize=gridsize,
                     clip=(-np.inf, np.inf),)
        data_pdf.fit(**fitkw)
    except Exception as ex:
        ut.printex(ex, '! Exception while estimating kernel density',
                   keys=['data', 'gridsize', 'bw_value', 'adjust'])
        raise

    return data_pdf


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
