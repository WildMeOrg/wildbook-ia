# -*- coding: utf-8 -*-
"""
Module for -- Confusion matrix, contingency, error matrix,

References:
    http://en.wikipedia.org/wiki/Confusion_matrix
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import numpy as np
import scipy.interpolate
(print, rrr, profile) = ut.inject2(__name__)


def testdata_scores_labels():
    scores = [2,  3,  4,  6,  9,  9, 13, 17, 19, 22, 22, 23, 26, 26, 34, 59, 63, 75, 80, 81, 89]
    labels = [0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1]
    return scores, labels


def nan_to_num(arr, num):
    arr[np.isnan(arr)] = num
    return arr


@six.add_metaclass(ut.ReloadingMetaclass)
class ConfusionMetrics(object):
    r"""
    Can compute average percision using the PASCAL definition

    References:
        http://www.flinders.edu.au/science_engineering/fms/School-CSEM/publications/tech_reps-research_artfcts/TRRA_2007.pdf
        http://www.alta.asn.au/events/altss_w2003_proc/altss/courses/powers/Bookmaker-all/200302-ICCS-Bookmaker.pdfcs
        http://www.cs.bris.ac.uk/Publications/Papers/1000704.pdf
        http://en.wikipedia.org/wiki/Information_retrieval
        http://en.wikipedia.org/wiki/Precision_and_recall
        https://en.wikipedia.org/wiki/Confusion_matrix
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

    SeeAlso:
        sklearn.metrics.ranking._binary_clf_curve

    Notes:
        From oxford:
        Precision is defined as the ratio of retrieved positive images to the
          total number retrieved.
        Recall is defined as the ratio of the number of retrieved positive
          images to the total number of positive images in the corpus.

    Ignore:
        varname_list = 'tp, fp, fn, tn, fpr, tpr, tpa'.split(', ')
        lines = ['self.{varname} = {varname}'.format(varname=varname) for varname in varname_list]
        print(ut.indent('\n'.join(lines)))

    CommandLine:
        python -m vtool.confusion ConfusionMetrics --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.confusion import *  # NOQA
        >>> scores, labels = testdata_scores_labels()
        >>> c = self = confusions = ConfusionMetrics().fit(scores, labels)
        >>> assert np.all(c.n_pos == c.n_tp + c.n_fn)
        >>> assert np.all(c.n_neg == c.n_tn + c.n_fp)
        >>> assert np.all(np.isclose(c.rp + c.rn, 1.0))
        >>> assert np.all(np.isclose(c.pp + c.pn, 1.0))
        >>> assert np.all(np.isclose(c.fpr, 1 - c.tnr))
        >>> assert np.all(np.isclose(c.fnr, 1 - c.tpr))
        >>> assert np.all(np.isclose(c.tpr, c.tp / c.rp))
        >>> assert np.all(np.isclose(c.tpa, c.tp / c.pp))
        >>> assert np.all(np.isclose(c.jacc, c.tp / (c.tp + c.fn + c.fp)))
        >>> assert np.all(np.isclose(c.mcc, np.sqrt(c.mk * c.bm)))
        >>> assert np.all(np.isclose(
        >>>     c.acc, (c.tpr + c.c * (1 - c.fpr)) / (1 + c.c)))
        >>> assert np.all(np.isclose(c.ppv, c.recall * c.prev / c.bias))
        >>> assert np.all(np.isclose(
        >>>    c.wracc, 4 * c.c * (c.tpr - c.fpr) / (1 + c.c) ** 2))
        >>> ut.quit_if_noshow()
        >>> confusions.draw_roc_curve()
        >>> ut.show_if_requested()
    """
    aliases = {
        'n_tp': {'n_hit', 'n_true_pos'},
        'n_tn': {'n_reject', 'n_true_neg'},
        'n_fp': {'n_false_alarm', 'n_false_pos'},
        'n_fn': {'n_miss', 'n_false_neg'},
        # -----
        'rp': {'real_pos', 'prev', 'prevalence'},
        'rn': {'real_neg'},
        'pp': {'pred_pos', 'bias'},
        'pn': {'pred_neg'},
        # -----
        'cs': {'class_odds', 'skew'},
        'cv': {'cost_ratio'},
        'cn': {'cost_pos'},
        'cn': {'cost_neg'},
        # -----
        'tp': {'true_pos', 'hit'},
        'tn': {'true_neg', 'reject'},
        'fp': {'false_pos', 'type1_error', 'false_alarm'},
        'fn': {'false_neg', 'type2_error', 'miss'},
        # -----
        'fpr': {'false_pos_rate', 'fallout'},
        'fnr': {'false_neg_rate', 'miss_rate'},
        'tpr': {'true_pos_rate', 'recall', 'sensitivity', 'hit_rate'},
        'tnr': {'true_neg_rate', 'inv_recall, specificity'},
        # -----
        'tpa': {'true_pos_acc', 'pos_predict_value', 'precision', 'ppv'},
        'tna': {'true_neg_acc', 'neg_predict_value', 'inv_precision', 'npv'},
        # -----
        'mk': {'markedness', 'deltaP', 'r_P'},
        'bm': {'informedness', 'bookmaker_informedness', 'deltaP\'', 'r_R'},
        # -----
        'mcc': {'matthews_correlation_coefficient'},
        'jacc': {'jaccard_coefficient'},
        'acc': {'accuracy', 'rand_accuracy', 'tea', 'ter'},
        'wracc': {'weighted_relative_accuracy'},
    }

    # the same things are called by lots of different names
    paper_alias = [
        ['dtp', 'determinant'],
        ['lr', 'liklihood-ratio'],
        ['nlr', 'negative-liklihood-ratio'],
        ['bmg', 'bookmarkG', 'bookmark_geometric_mean', 'mcc?'],
        ['evenness_R', 'PrevG2'],
        ['evenness_P', 'BiasG2'],
        ['rh', 'real_harmonic_mean'],
        ['ph', 'pred_harminic_mean'],
    ]

    # And they related to each other in interesting ways
    paper_relations = {
        'N'      : ['A + B + C + D'],
        'dtp'    : ['A * D - B * C'],
        'mk'     : ['dtp / (bias * (1 - bias))',
                    'dtp / biasG ** 2'],
        'bm'     : ['dtp / (prev * (1 - prev))'],
        'BiasG2' : ['bias * 1 - bias'],
        'lr'     : ['tpr / (1 - tnr)'],
        'nlr'    : ['tnr / (1 - tpr)'],
        'BMG'    : ['dtp / evenness_G'],
        'IBias'  : ['1 - Bias'],
        'etp'    : ['rp * pp', 'expected_true_positives'],
        'etn'    : ['rn * pn', 'expected_true_negatives'],
        'rh' : ['2 * rp * rn / (rp + rn)', 'real_harmonic_mean'],
        'ph' : ['2 * pp * pn / (pp + pn)', 'pred_harminic_mean'],
        'dp'     : ['tp - etp', 'dtp', '-dtn', '-(tn - etn)'],
        'deltap' : ['dtp - dtn', '2 * dp'],
        'kappa': ['deltap / (deltap + (fp + fn) / 2)']
    }

    # ROC Plot: tpr vs fpr
    # PN Plot: TP vs FP

    minimizing_metrics = {'fpr', 'fnr', 'fp', 'fn'}

    inv_aliases = {
        alias_key: std_key
        for std_key, alias_vals in aliases.items()
        for alias_key in set.union(alias_vals, {std_key})
    }

    def __init__(self):
        # Scalars
        self.n_pos = None
        self.n_neg = None
        self.n_samples = None

        # Threshold based
        self.thresholds = None
        self.n_tp = None
        self.n_fp = None
        self.n_fn = None
        self.n_tn = None

        # Can be set to weight the cost of errors
        self.cp = 1.0
        self.cn = 1.0

    # ----

    @property
    def cs(self):
        """ class ratio """
        return self.rn / self.rp

    @property
    def cv(self):
        """ ratio of cost of making a mistake"""
        return self.cn / self.cp

    @property
    def c(self):
        return self.cs * self.cv

    # -----

    @property
    def tp(self):
        """ true positive probability """
        return self.n_tp / self.n_samples

    @property
    def tn(self):
        """ true negative probability """
        return self.n_tn / self.n_samples

    @property
    def fp(self):
        """ false positive probability """
        return self.n_fp / self.n_samples

    @property
    def fn(self):
        """ false negative probability """
        return self.n_fn / self.n_samples

    # ----

    @property
    def rp(self):
        """ real positive probability """
        # return (self.tp + self.fn)
        return self.n_pos / self.n_samples

    @property
    def rn(self):
        """ real negative probability """
        # return (self.fp + self.tn)
        return self.n_neg / self.n_samples

    @property
    def pp(self):
        """ predicted positive probability """
        return (self.tp + self.fp)

    @property
    def pn(self):
        """ predicted negative probability """
        return (self.fn + self.tn)

    # ----

    @property
    def fpr(self):
        """ fallout, false positive rate """
        return self.n_fp / self.n_neg

    @property
    def fnr(self):
        """ miss rate, false negative rate """
        return self.n_fn / self.n_pos

    @property
    def tpr(self):
        """ sensitivity, recall, hit rate, tpr """
        return self.n_tp / self.n_pos

    @property
    def tnr(self):
        """ true negative rate, inverse recall """
        return self.n_tn / self.n_neg

    # ----

    @property
    def tpa(self):
        """ miss rate, false negative rate """
        with np.errstate(invalid='ignore'):
            return nan_to_num(self.n_tp / (self.n_tp + self.n_fp), 1.0)

    @property
    def tna(self):
        """ negative predictive value, inverse precision """
        with np.errstate(invalid='ignore'):
            return nan_to_num(self.n_tn / (self.n_tn + self.n_fn), 1.0)

    # ----

    @property
    def bm(self):
        """ bookmaker informedness """
        return self.tpr + self.tnr - 1

    @property
    def mk(self):
        """ markedness """
        return self.tpa + self.tna - 1

    # ---- other measures

    @property
    def auc_trap(self):
        # per threshold trapazoidal auc metric
        return (self.tpr + self.tnr) / 2

    @property
    def acc(self):
        """ accuracy """
        return self.tp + self.tn

    @property
    def sqrd_error(self):
        """ squared error """
        return np.sqrt(self.fpr ** 2 + self.fnr ** 2)

    @property
    def mcc(self):
        """ matthews correlation coefficient

        Also true that:
            mcc == np.sqrt(self.bm * self.mk)
        """
        mcc_numer = (self.tp * self.tn - self.fp * self.fn)
        mcc_denom = np.sqrt((self.tp + self.fp) * (self.tp + self.fn) *
                            (self.tn + self.fp) * (self.tn + self.fn))
        with np.errstate(invalid='ignore'):
            mcc = nan_to_num(mcc_numer / mcc_denom, 0.0)
        return mcc

    @property
    def jacc(self):
        """ jaccard coefficient """
        return self.n_tp / (self.n_samples - self.n_tn)
        # return self.tp / (self.tp + self.fn + self.fp)

    @property
    def wracc(self):
        """ weighted relative accuracy """
        return 4 * (self.recall - self.bias) * self.prev

    # --- alias names currently needed for compatability

    def __dir__(self):
        attrs = dir(object)
        attrs += list(self.__class__.__dict__.keys())
        attrs += list(self.__dict__.keys())
        attrs += self.inv_aliases.keys()
        attrs = sorted(set(attrs))
        return attrs

    def __getattr__(self, attr):
        try:
            std_attr = self.inv_aliases[attr]
        except KeyError:
            raise AttributeError(attr)
        return getattr(self, std_attr)

    # @property
    # def recall(self):
    #     return self.tpr

    # @property
    # def precision(self):
    #     return self.tpa

    # @property
    # def fallout(self):
    #     return self.fpr

    # --------------
    # Construtors
    # --------------

    def fit(self, scores, labels, verbose=False):
        import sklearn.metrics
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        # must be binary
        labels = labels.astype(np.bool)
        if verbose:
            print('[confusion] building confusion metrics.')
            print('[confusion]  * scores.shape=%r, scores.dtype=%r' %
                  (scores.shape, scores.dtype))
            print('[confusion]  * labels.shape=%r, labels.dtype=%r' %
                  (labels.shape, labels.dtype))

        # sklearn has much faster implementation
        # n_fp - count the number of false positives with score >= threshold[i]
        # n_tp - count the number of true positives with score >= threshold[i]
        n_fp, n_tp, thresholds = sklearn.metrics.ranking._binary_clf_curve(
            labels, scores, pos_label=1)

        n_samples = len(labels)
        n_pos = labels.sum()
        n_neg = n_samples - n_pos

        # Scalars
        self.n_samples = n_samples
        self.n_pos = n_pos
        self.n_neg = n_neg

        # Threshold based
        self.thresholds = thresholds
        self.n_tp = n_tp
        self.n_fp = n_fp
        self.n_fn = n_pos - n_tp
        self.n_tn = n_neg - n_fp
        return self

    @classmethod
    def from_scores_and_labels(cls, scores, labels, verbose=False):
        self = cls().fit(scores, labels, verbose=verbose)
        return self

    @classmethod
    def from_tp_and_tn_scores(cls, tp_scores, tn_scores, verbose=False):
        scores = np.hstack([tp_scores, tn_scores])
        labels = np.array([True] * len(tp_scores) + [False] * len(tn_scores))
        self = cls().fit(scores, labels, verbose=verbose)
        return self

    # -------------------------------
    # Threshold-less Summary Measures
    # -------------------------------

    def get_ave_precision(self):
        precision = self.precision
        recall = self.recall
        recall_domain, p_interp = interpolate_precision_recall(precision, recall)
        return p_interp.sum() / p_interp.size

    @property
    def auc(self):
        """
        The AUC is a standard measure used to evaluate a binary classifier and
          represents the probability that a random correct case will
          receive a higher score than a random incorrect case.

        References:
            https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
        """
        # TODO: change name to represent it is a total measure
        import sklearn.metrics
        return sklearn.metrics.auc(self.fpr, self.tpr)

    # ---------------------
    # Threshold Choosers / Info
    # ---------------------

    def get_fpr_at_recall(self, target_recall):
        indicies = np.where(self.recall >= target_recall)[0]
        assert len(indicies) > 0, 'no recall at target level'
        func = scipy.interpolate.interp1d(self.recall, self.fpr)
        interp_fpr = func(target_recall)
        ## interpolate to target recall
        #right_index  = indicies[0]
        #right_recall = self.recall[right_index]
        #left_index   = right_index - 1
        #left_recall  = self.recall[left_index]
        #stepsize = right_recall - left_recall
        #alpha = (target_recall - left_recall) / stepsize
        #left_fpr   = self.fpr[left_index]
        #right_fpr  = self.fpr[right_index]
        #interp_fpp = (left_fpr * (1 - alpha)) + (right_fpr * (alpha))
        return interp_fpr

    def get_recall_at_fpr(self, target_fpr):
        indicies = np.where(self.fpr >= target_fpr)[0]
        assert len(indicies) > 0, 'no false positives at target level'
        func = scipy.interpolate.interp1d(self.fpr, self.tpr)
        interp_tpr = func(target_fpr)
        return interp_tpr

    def get_thresh_at_metric_max(self, metric):
        """
        metric = 'mcc'
        metric = 'fnr'
        """
        metric_values = getattr(self, metric)
        if False:
            idx = metric_values.argmax()
            thresh = self.thresholds[idx]
        else:
            # interpolated version
            import vtool as vt
            thresh, max_value = vt.argsubmax(metric_values, self.thresholds)
        return thresh

    def get_thresh_at_metric(self, metric, value, prefer_max=None):
        r"""
        Gets a threshold for a binary classifier using a target metric and value

        Args:
            metric (str): name of metric like tpr or fpr
            value (float): corresponding numeric value

        Returns:
            float: thresh

        CommandLine:
            python -m vtool.confusion get_thresh_at_metric
            python -m vtool.confusion --exec-interact_roc_factory --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.confusion import *  # NOQA
            >>> scores, labels = testdata_scores_labels()
            >>> self = ConfusionMetrics().fit(scores, labels)
            >>> metric = 'tpr'
            >>> value = .85
            >>> thresh = self.get_thresh_at_metric(metric, value)
            >>> print('%s = %r' % (metric, value,))
            >>> result = ('thresh = %s' % (str(thresh),))
            >>> print(result)
            thresh = 22.5
        """
        if value == 'max':
            return self.get_thresh_at_metric_max(metric)
        # if value == 'min':
        #     return self.get_thresh_at_metric_min(metric)
        # TODO: Use interpoloation here and make tpr vs fpr a smooth funciton
        metric = self.inv_aliases[metric]
        metric_values = getattr(self, metric)
        if metric == 'fpr':
            # hack
            if len(metric_values) <= 1:
                return 1.0
        # prefer_max = metric not in self.minimizing_metrics
        if prefer_max is None:
            prefer_max = metric not in {'fpr'}
        thresh = interpolate_replbounds(metric_values, self.thresholds, value,
                                        prefer_max=prefer_max)
        return thresh

    def get_metric_at_thresh(self, metric, thresh):
        r"""
        Args:
            metric (str): name of a metric
            thresh (float): desired threshold

        Returns:
            float : value - metric value

        CommandLine:
            python -m vtool.confusion --exec-get_metric_at_threshold

        Ignore:
            >>> self = cfms
            >>> metric = 'fpr'
            >>> thresh = 0

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.confusion import *  # NOQA
            >>> scores, labels = testdata_scores_labels()
            >>> self = ConfusionMetrics().fit(scores, labels)
            >>> metric = 'tpr'
            >>> thresh = .8
            >>> thresh = [0, .1, .9, 1.0]
            >>> value = self.get_metric_at_threshold(metric, thresh)
            >>> result = ('(None, None) = %s' % (str((None, None)),))
            >>> print(result)
        """
        was_scalar = ut.isscalar(thresh)
        if was_scalar:
            thresh = [thresh]
        else:
            thresh = np.asarray(thresh)
        # Assert decreasing
        assert self.thresholds[0] > self.thresholds[-1]
        sortx = np.argsort(self.thresholds)
        thresh_ = np.clip(thresh, self.thresholds[-1], self.thresholds[0])
        r = np.searchsorted(self.thresholds, thresh_, side='left', sorter=sortx)
        index_list = sortx[r]
        # index_list = [np.where(self.thresholds <= t)[0][0] for t in thresh]
        # sortx[r]
        # index_list = []
        # for t in thresh:
        #     try:
        #         index = np.nonzero(self.thresholds <= t)[0][0]
        #     except IndexError:
        #         print('warning: index error in get_metric_at_thresh t=%r' % (t,))
        #         index = len(self.thresholds) - 1
        #     index_list.append(index)
        # # value = self.__dict__[metric][index]
        value = [getattr(self, metric)[index] for index in index_list]
        if was_scalar:
            value = value[0]
        return value

    get_threshold_at_metric = get_thresh_at_metric
    get_metric_at_threshold = get_metric_at_thresh

    # --------------
    # Visualizations
    # --------------

    def draw_roc_curve(self, **kwargs):
        return draw_roc_curve(self.fpr, self.tpr, **kwargs)

    def draw_precision_recall_curve(self, nSamples=11, **kwargs):
        precision = self.precision
        recall = self.recall
        recall_domain, p_interp = interpolate_precision_recall(precision, recall, nSamples)
        return draw_precision_recall_curve(recall_domain, p_interp, **kwargs)

    def plot_vs(self, x_metric, y_metric):
        """
        x_metric = 'thresholds'
        y_metric = 'fpr'
        """
        import plottool as pt
        pt.qt4ensure()
        xdata = self.thresholds
        ydata_list = [getattr(self, y_metric)]
        pt.multi_plot(xdata, ydata_list, label_list=[y_metric],
                      xlabel=x_metric, marker='',
                      ylabel=y_metric, use_legend=True)

    def plot_metrics(self):
        import plottool as pt
        metrics = [
            'mcc',
            'acc',
            'auc_trap'
            # 'tpa', 'tpr',
            # 'acc', 'sqrd_error',
            # 'auc_trap',
            # 'mk', 'bm'
        ]
        metrics = [
            'fnr',
            'fpr',
            'tpr',
            'tnr',
        ]
        xdata = self.thresholds
        ydata_list = [getattr(self, m) for m in metrics]
        pt.multi_plot(xdata, ydata_list, label_list=metrics,
                      xlabel='threshold', marker='',
                      ylabel='metric', use_legend=True)

    def show_mcc(self):
        import plottool as pt
        pt.multi_plot(self.thresholds, [self.mcc], xlabel='threshold', marker='',
                      ylabel='MCC')
        pass


@profile
def interpolate_replbounds(xdata, ydata, pt, prefer_max=True):
    """
    xdata = np.array([.1, .2, .3, .4, .5])
    ydata = np.array([.1, .2, .3, .4, .5])
    pt = .35

    FIXME:
        if duplicate xdata is given bad things happen.
    BUG:
        in scipy.interpolate.interp1d
        If there is a duplicate xdata, then assume_sorted=False will
        sort ydata by xdata, but xdata should retain its initial ordering
        in places of ambuguity. Currently it does not.
    Args:
        xdata (ndarray):
        ydata (ndarray):
        pt (ndarray):

    Returns:
        float: interp_vals

    CommandLine:
        python -m vtool.confusion --exec-interpolate_replbounds

    Example:
        >>> from vtool.confusion import *  # NOQA
        >>> scores, labels = testdata_scores_labels()
        >>> self = ConfusionMetrics().fit(scores, labels)
        >>> xdata = self.tpr
        >>> ydata = self.thresholds
        >>> pt = 1.0
        >>> #xdata = self.fpr
        >>> #ydata = self.thresholds
        >>> #pt = 0.0
        >>> thresh = interpolate_replbounds(xdata, ydata, pt, prefer_max=True)
        >>> print('thresh = %r' % (thresh,))
        >>> thresh = interpolate_replbounds(xdata, ydata, pt, prefer_max=False)
        >>> print('thresh = %r' % (thresh,))

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.confusion import *  # NOQA
        >>> xdata = np.array([0.7,  0.8,  0.8,  0.9,  0.9, 0.9])
        >>> ydata = np.array([34,    26,   23,   22,   19,  17])
        >>> pt = np.array([.85, 1.0, -1.0])
        >>> interp_vals = interpolate_replbounds(xdata, ydata, pt)
        >>> result = ('interp_vals = %s' % (str(interp_vals),))
        >>> print(result)
        interp_vals = [ 22.5  17.   34. ]
    """
    if not ut.issorted(xdata):
        if ut.issorted(xdata[::-1]):
            xdata = xdata[::-1]
            ydata = ydata[::-1]
        else:
            raise AssertionError('need to sort xdata and ydata in function')
            sortx = np.lexsort(np.vstack([np.arange(len(xdata)), xdata]))
            xdata = xdata.take(sortx, axis=0)
            ydata = ydata.take(sortx, axis=0)

    is_scalar = not ut.isiterable(pt)
    #print('----')
    #print('xdata = %r' % (xdata,))
    #print('ydata = %r' % (ydata,))
    if is_scalar:
        pt = np.array([pt])
    #ut.ensure_iterable(pt)
    minval = xdata.min()
    maxval = xdata.max()
    argx_min_list = np.argwhere(xdata == minval)
    argx_max_list = np.argwhere(xdata == maxval)
    argx_min = argx_min_list.min()
    argx_max = argx_max_list.max()
    lower_mask = pt < xdata[argx_min]
    upper_mask = pt > xdata[argx_max]
    interp_mask = ~np.logical_or(lower_mask, upper_mask)
    #if isinstance(pt, np.ndarray):
    dtype = np.result_type(np.float32, ydata.dtype)
    interp_vals = np.empty(pt.shape, dtype=dtype)
    interp_vals[lower_mask] = ydata[argx_min]
    interp_vals[upper_mask] = ydata[argx_max]

    # TODO: fix duplicate values depending on if higher or lower numbers are
    # desirable
    if True:
        # Grouping should be ok because xdata should be sorted
        # therefore groupxs are consecutive
        import vtool as vt
        unique_vals, groupxs = vt.group_indices(xdata)
        grouped_ydata = vt.apply_grouping(ydata, groupxs)
        if prefer_max:
            sub_idxs = [idxs[np.argmax(ys)] for idxs, ys in zip(groupxs, grouped_ydata)]
        else:
            sub_idxs = [idxs[np.argmin(ys)] for idxs, ys in zip(groupxs, grouped_ydata)]
        sub_idxs = np.array(sub_idxs)
        xdata = xdata[sub_idxs]
        ydata = ydata[sub_idxs]

    if np.any(interp_mask):
        # FIXME: allow assume_sorted = False
        func = scipy.interpolate.interp1d(xdata, ydata, kind='linear', assume_sorted=True)
        interp_vals[interp_mask] = func(pt[interp_mask])
    if is_scalar:
        interp_vals = interp_vals[0]
    # interpolate to target recall
    #right_index  = indicies[0]
    #right_recall = self.recall[right_index]
    #left_index   = right_index - 1
    #left_recall  = self.recall[left_index]
    #stepsize = right_recall - left_recall
    #alpha = (target_recall - left_recall) / stepsize
    #left_fpr   = self.fpr[left_index]
    #right_fpr  = self.fpr[right_index]
    #interp_fpp = (left_fpr * (1 - alpha)) + (right_fpr * (alpha))
    return interp_vals


def interpolate_precision_recall(precision, recall, nSamples=11):
    """
    Interpolates precision as a function of recall p_{interp}(r)

    Reduce wiggles in average precision curve by taking interpolated values
    along a uniform sample.

    References:
        http://en.wikipedia.org/wiki/Information_retrieval#Average_precision
        http://en.wikipedia.org/wiki/Information_retrieval#Mean_Average_precision

    CommandLine:
        python -m vtool.confusion --test-interpolate_precision_recall --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.confusion import *  # NOQA
        >>> scores, labels = testdata_scores_labels()
        >>> nSamples = 11
        >>> confusions = ConfusionMetrics().fit(scores, labels)
        >>> precision = confusions.precision
        >>> recall = confusions.recall
        >>> recall_domain, p_interp = interpolate_precision_recall(confusions.precision, recall, nSamples=11)
        >>> result = ut.numpy_str(p_interp, precision=1)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> draw_precision_recall_curve(recall_domain, p_interp)
        >>> ut.show_if_requested()
        np.array([ 1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  0.9,  0.9,  0.8,  0.6], dtype=np.float64)
    """
    if precision is None:
        return None, None

    recall_domain = np.linspace(0, 1, nSamples)
    if False:
        # normal interpolation
        func = scipy.interpolate.interp1d(recall, precision,
                                          bounds_error=False,
                                          fill_value=precision.max())
        p_interp = func(recall_domain)
    else:
        # Pascal interpolation
        #candidate_masks = recall >= recall_domain[:, None]
        #candidates_idxs_ = [np.where(mask)[0] for mask in candidate_masks]
        #chosen_idx = [-1 if len(idxs) == 0 else idxs.min() for idxs in  candidates_idxs_]
        #p_interp = precision[chosen_idx]
        def p_interp(r):
            precision_candidates = precision[recall >= r]
            if len(precision_candidates) == 0:
                return 0
            return precision_candidates.max()

        p_interp = np.array([p_interp(r) for r in recall_domain])
    return recall_domain, p_interp


def interact_roc_factory(confusions, target_tpr=None, show_operating_point=False):
    r"""
    Args:
        confusions (Confusions):

    CommandLine:
        python -m vtool.confusion --exec-interact_roc_factory --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.confusion import *  # NOQA
        >>> scores, labels = testdata_scores_labels()
        >>> print('scores = %r' % (scores,))
        >>> confusions = ConfusionMetrics().fit(scores, labels)
        >>> print(ut.make_csv_table(
        >>>   [confusions.fpr, confusions.tpr, confusions.thresholds],
        >>>   ['fpr', 'tpr', 'thresh']))
        >>> ut.quit_if_noshow()
        >>> ROCInteraction = interact_roc_factory(confusions, target_tpr=.4, show_operating_point=True)
        >>> inter = ROCInteraction()
        >>> inter.show_page()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    from plottool.abstract_interaction import AbstractInteraction

    class ROCInteraction(AbstractInteraction):
        """
        References:
            http://scipy-central.org/item/38/1/roc-curve-demo

        Notes:
            Sensitivity = true positive rate
            Specificity = true negative rate
        """
        def __init__(self, **kwargs):
            print('ROC Interact')
            super(ROCInteraction, self).__init__(**kwargs)
            self.confusions = confusions
            self.target_fpr = None
            self.show_operating_point = show_operating_point

        @staticmethod
        def static_plot(fnum, pnum, **kwargs):
            #print('ROC Interact2')
            kwargs['thresholds'] = kwargs.get('thresholds', confusions.thresholds)
            kwargs['show_operating_point'] = kwargs.get('show_operating_point',
                                                        show_operating_point)
            #ut.embed()
            confusions.draw_roc_curve(fnum=fnum, pnum=pnum,
                                      target_tpr=target_tpr, **kwargs)

        def plot(self, fnum, pnum):
            #print('ROC Interact3')
            self.static_plot(fnum, pnum, target_fpr=self.target_fpr,
                             show_operating_point=self.show_operating_point)

        def on_click_inside(self, event, ex):
            self.target_fpr = event.xdata
            self.show_page()
            self.draw()

        def on_drag(self, event):
            # FIXME: blit
            if False:
                #print('Dragging ' + str(event.x) + ' ' + str(event.y))
                self.target_fpr = event.xdata
                self.show_page()
                #self.draw()
                if event.inaxes is not None:
                    self.fig.canvas.blit(event.inaxes.bbox)
                    #[blit(ax) event.canvas.figure.axes]

    return ROCInteraction


def draw_roc_curve(fpr, tpr, fnum=None, pnum=None, marker='', target_tpr=None,
                   target_fpr=None, thresholds=None, color=None, name=None,
                   label=None, show_operating_point=False):
    r"""
    Args:
        fpr (?):
        tpr (?):
        fnum (int):  figure number(default = None)
        pnum (tuple):  plot number(default = None)
        marker (str): (default = '-x')
        target_tpr (None): (default = None)
        target_fpr (None): (default = None)
        thresholds (None): (default = None)
        color (None): (default = None)
        show_operating_point (bool): (default = False)

    CommandLine:
        python -m vtool.confusion --exec-draw_roc_curve --show --lightbg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.confusion import *  # NOQA
        >>> scores, labels = testdata_scores_labels()
        >>> confusions = ConfusionMetrics().fit(scores, labels)
        >>> fpr = confusions.fpr
        >>> tpr = confusions.tpr
        >>> thresholds = confusions.thresholds
        >>> fnum = None
        >>> pnum = None
        >>> marker = 'x'
        >>> target_tpr = .85
        >>> target_fpr = None
        >>> color = None
        >>> show_operating_point = True
        >>> draw_roc_curve(fpr, tpr, fnum, pnum, marker, target_tpr, target_fpr,
        >>>   thresholds, color, show_operating_point)
        >>> ut.show_if_requested()
    """
    import plottool as pt
    import sklearn.metrics
    if fnum is None:
        fnum = pt.next_fnum()

    # if color is None:
    #     color = (0.4, 1.0, 0.4) if pt.is_default_dark_bg() else  (0.1, 0.4, 0.4)

    roc_auc = sklearn.metrics.auc(fpr, tpr)

    title_suffix = ''

    if target_fpr is not None:
        #func = scipy.interpolate.interp1d(fpr, tpr, kind='linear', assume_sorted=False)
        #func = scipy.interpolate.interp1d(xdata, ydata, kind='nearest', assume_sorted=False)
        #interp_vals[interp_mask] = func(pt[interp_mask])
        target_fpr = np.clip(target_fpr, 0, 1)
        interp_tpr = interpolate_replbounds(fpr, tpr, target_fpr)
        choice_tpr = interp_tpr
        choice_fpr = target_fpr
    elif target_tpr is not None:
        target_tpr = np.clip(target_tpr, 0, 1)
        interp_fpr = interpolate_replbounds(tpr, fpr, target_tpr)
        choice_tpr = target_tpr
        choice_fpr = interp_fpr
    else:
        choice_tpr = None
        choice_fpr = None

    if choice_fpr is not None:
        choice_thresh = 0
        if thresholds is not None:
            try:
                index = np.nonzero(tpr >= choice_tpr)[0][0]
            except IndexError:
                index = len(thresholds) - 1
            choice_thresh = thresholds[index]
        #percent = ut.scalar_str(choice_tpr * 100).split('.')[0]
        #title_suffix = ', FPR%s=%05.2f%%' % (percent, choice_fpr)
        title_suffix = ''
        if show_operating_point:
            title_suffix = ', fpr=%.2f, tpr=%.2f, thresh=%.2f' % (
                choice_fpr, choice_tpr, choice_thresh)
    else:
        title_suffix = ''

    #if recall_domain is None:
    #    ave_p = np.nan
    #else:
    #    ave_p = p_interp.sum() / p_interp.size
    title = 'Receiver operating characteristic'
    if name and not label:
        title += ' (%s)' % (name,)
    if not label:
        title += '\n' + 'AUC=%.3f' % (roc_auc,)
    else:
        label += ' AUC=%.3f' % (roc_auc,)

    title += title_suffix

    label_list = None
    if label:
        label_list = [label]

    pt.multi_plot(fpr, [tpr], label_list=label_list, marker=marker,
                  color=color, fnum=fnum, pnum=pnum, title=title,
                  xlabel='False Positive Rate',
                  ylabel='True Positive Rate')

    # pt.plot2(fpr, tpr, marker=marker,
    #          x_label='False Positive Rate',
    #          y_label='True Positive Rate',
    #          unitbox=True, flipx=False, color=color, fnum=fnum, pnum=pnum,
    #          title=title)

    if False:
        # Interp does not work right because of duplicate values
        # in xdomain
        line_ = np.linspace(.11, .9, 20)
        #np.append([np.inf], np.diff(fpr)) > 0
        #np.append([np.inf], np.diff(tpr)) > 0
        unique_tpr_idxs = np.nonzero(np.append([np.inf], np.diff(tpr)) > 0)[0]
        unique_fpr_idxs = np.nonzero(np.append([np.inf], np.diff(fpr)) > 0)[0]

        pt.plt.plot(
            line_,
            interpolate_replbounds(fpr[unique_fpr_idxs], tpr[unique_fpr_idxs],
                                   line_), 'b-x')
        pt.plt.plot(
            interpolate_replbounds(tpr[unique_tpr_idxs], fpr[unique_tpr_idxs],
                                   line_), line_, 'r-x')
    if choice_fpr is not None:
        pt.plot(choice_fpr, choice_tpr, 'o', color=pt.PINK)


def draw_precision_recall_curve(recall_domain, p_interp, title_pref=None,
                                fnum=1, pnum=None,
                                color=None):
    import plottool as pt
    if color is None:
        color = (0.4, 1.0, 0.4) if pt.is_default_dark_bg() else  (0.1, 0.4, 0.4)
    if recall_domain is None:
        recall_domain = np.array([])
        p_interp = np.array([])
    if recall_domain is None:
        ave_p = -1.0  # np.nan
    else:
        ave_p = p_interp.sum() / p_interp.size

    pt.plot2(recall_domain, p_interp, marker='o--',
              x_label='recall', y_label='precision', unitbox=True,
              flipx=False, color=color, fnum=fnum, pnum=pnum,
              title='Interplated Precision Vs Recall\n' + 'avep = %.3f'  % ave_p)
    #print('Interplated Precision')
    #print(ut.list_str(list(zip(recall_domain, p_interp))))
    #fig.show()


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.confusion
        python -m vtool.confusion --allexamples
        python -m vtool.confusion --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
