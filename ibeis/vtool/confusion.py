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
import sklearn.metrics
import scipy.interpolate
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[confusion]', DEBUG=False)


def testdata_scores_labels():
    scores = [2,  3,  4,  6,  9,  9, 13, 17, 19, 22, 22, 23, 26, 26, 34, 59, 63, 75, 80, 81, 89]
    labels = [0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1]
    return scores, labels


def interpolate_replbounds(xdata, ydata, pt):
    """
    xdata = np.array([.1, .2, .3, .4, .5])
    ydata = np.array([.1, .2, .3, .4, .5])
    pt = .35

    # FIXME: if duplicate xdata is given bad things happen.
    """
    is_scalar = not ut.isiterable(pt)
    #print('----')
    #print('xdata = %r' % (xdata,))
    #print('ydata = %r' % (ydata,))
    if is_scalar:
        pt = np.array([pt])
    #ut.ensure_iterable(pt)
    argx_min = xdata.argmin()
    argx_max = xdata.argmax()
    lower_mask = pt < xdata[argx_min]
    upper_mask = pt > xdata[argx_max]
    interp_mask = ~np.logical_or(lower_mask, upper_mask)
    #if isinstance(pt, np.ndarray):
    interp_vals = np.empty(pt.shape, dtype=ydata.dtype)
    interp_vals[lower_mask] = ydata[argx_min]
    interp_vals[upper_mask] = ydata[argx_max]
    if np.any(interp_mask):
        func = scipy.interpolate.interp1d(xdata, ydata, kind='linear', assume_sorted=False)
        #func = scipy.interpolate.interp1d(xdata, ydata, kind='nearest', assume_sorted=False)
        interp_vals[interp_mask] = func(pt[interp_mask])
    if is_scalar:
        interp_vals = interp_vals[0]
    return interp_vals
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


@six.add_metaclass(ut.ReloadingMetaclass)
class ConfusionMetrics(object):
    """
    Ignore:
        varname_list = 'tp, fp, fn, tn, fpr, tpr, ppv'.split(', ')
        lines = ['self.{varname} = {varname}'.format(varname=varname) for varname in varname_list]
        print(ut.indent('\n'.join(lines)))

    """
    def __init__(self, thresholds, tp, fp, fn, tn, fpr, tpr, ppv):
        self.thresholds = thresholds
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.fpr = fpr
        self.tpr = tpr
        self.ppv = ppv

    # --------------
    # Construtors
    # --------------

    @classmethod
    def from_scores_and_labels(cls, scores, labels, verbose=False):
        self = get_confusion_metrics(scores, labels, verbose=verbose)
        return self

    @classmethod
    def from_tp_and_tn_scores(cls, tp_scores, tn_scores, verbose=False):
        scores = np.hstack([tp_scores, tn_scores])
        labels = np.array([True] * len(tp_scores) + [False] * len(tn_scores))
        self = get_confusion_metrics(scores, labels, verbose=verbose)
        return self

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

    def get_fpr_at_95_recall(self):
        target_recall = .95
        fpr95 = self.get_fpr_at_recall(target_recall)
        return fpr95

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
        return interp_tpr

    def get_threshold_at_metric(self, metric, value):
        # TODO: Use interpoloation here and make tpr vs fpr a smooth funciton
        try:
            metric_values = self.__dict__[metric]
            if metric_values[0] > metric_values[-1]:
                index = np.nonzero(metric_values <= value)[0][0]
            else:
                index = np.nonzero(metric_values >= value)[0][0]
        except IndexError:
            index = len(self.thresholds) - 1
        thresh = self.thresholds[index]
        return thresh

    def get_metric_at_threshold(self, metric, thresh):
        r"""
        Args:
            metric (?):
            thresh (?):

        Returns:
            tuple: (None, None)

        CommandLine:
            python -m vtool.confusion --exec-get_metric_at_threshold

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.confusion import *  # NOQA
            >>> scores, labels = testdata_scores_labels()
            >>> self = get_confusion_metrics(scores, labels)
            >>> metric = 'tpr'
            >>> thresh = .8
            >>> (None, None) = self.get_metric_at_threshold(metric, thresh)
            >>> result = ('(None, None) = %s' % (str((None, None)),))
            >>> print(result)
        """
        # Assert decreasing
        assert self.thresholds[0] > self.thresholds[-1]
        try:
            index = np.nonzero(self.thresholds <= thresh)[0][0]
        except IndexError:
            index = len(self.thresholds) - 1
        value = self.__dict__[metric][index]
        return value

    @property
    def precision(self):
        return self.ppv

    @property
    def recall(self):
        return self.tpr

    def get_ave_precision(self):
        precision = self.precision
        recall = self.recall
        recall_domain, p_interp = interpolate_precision_recall(precision, recall)
        return p_interp.sum() / p_interp.size

    @property
    def auc(self):
        return sklearn.metrics.auc(self.fpr, self.tpr)


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
        >>> confusions = get_confusion_metrics(scores, labels)
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
        func = scipy.interpolate.interp1d(recall, precision, bounds_error=False, fill_value=precision.max())
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


def get_confusion_metrics(scores, labels, verbose=False):
    """
    gets average percision using the PASCAL definition

    FIXME: Use only the groundtruth that could have been matched in the
    database. (shouldn't be an issue until we start using daid subsets)

    References:
        http://en.wikipedia.org/wiki/Information_retrieval
        http://en.wikipedia.org/wiki/Precision_and_recall
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

    Notes:
        From oxford:
        Precision is defined as the ratio of retrieved positive images to the total number retrieved.
        Recall is defined as the ratio of the number of retrieved positive images to the total
        number of positive images in the corpus.

    CommandLine:
        python -m vtool.confusion --exec-get_confusion_metrics --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.confusion import *  # NOQA
        >>> scores, labels = testdata_scores_labels()
        >>> confusions = get_confusion_metrics(scores, labels)
        >>> ut.quit_if_noshow()
        >>> confusions.draw_roc_curve()
        >>> ut.show_if_requested()
    """
    import sklearn.metrics
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    labels = labels.astype(np.bool)
    if verbose:
        print('[confusion] building confusion metrics.')
        print('[confusion]  * scores.shape=%r, scores.dtype=%r' % (scores.shape, scores.dtype))
        print('[confusion]  * labels.shape=%r, labels.dtype=%r' % (labels.shape, labels.dtype))
        print('[confusion]  * size(scores) = %r' % (ut.get_object_size_str(scores),))
        print('[confusion]  * size(labels) = %r' % (ut.get_object_size_str(labels),))
    #memtrack = ut.MemoryTracker(disable=False)
    #memtrack.report('[CONFUSION]')

    #ut.embed()
    #memtrack.report('[SKLEARN]')
    # TODO
    #sklearn.metrics.precision_recall_curve(labels, probs)
    #sklearn.metrics.classification_report

    #if False:
    #    max_rank =  max(true_ranks.max(), false_ranks.max())
    #    if max_rank is None:
    #        max_rank = 0
    #    thresholds = np.arange(max_rank + 1)

    if False:
        # Hey this is n**2 in memory!
        #"""
        #s = np.array([.3, .9, .4, .2])
        #t = np.array([1.0, .5, .6, .1])
        #s[:, None] >= t
        #(s[:, None] >= t).sum(axis=0)
        #"""
        pass
        #false_positive_rate, true_positive_rate = fpr, tpr
        # True positive rate (sensitivity)
        # TPR = TP / (TP + FN)
        #true_ranks = np.where(labels)[0]
        #false_ranks = np.where(np.logical_not(labels))[0]
        #true_scores = scores.compress(labels)
        #false_scores = scores.compress(np.logical_not(labels))
        #nGroundTruth = len(true_scores)
        #nGroundFalse = len(false_scores)
        # Count the number of true positives at each threshold position
        # the number of documents we got right
        #truepos_curve = (true_scores[:, None] >= thresholds).sum(axis=0)
        # Count the number of false positives at each threshold position
        # the number of documents we should not have retrieved
        #falsepos_curve = (false_scores[:, None] >= thresholds).sum(axis=0)
        # Count the number of documents we should have retrieved but did not
        #falseneg_curve = nGroundTruth - truepos_curve
        #trueneg_curve  = nGroundFalse - falsepos_curve
        #tp = truepos_curve
        #fp = falsepos_curve
        #fn = falseneg_curve
        #tn = trueneg_curve
    else:
        #tp_ = np.array([(true_scores >= t).sum() for t in thresholds])
        #fp_ = np.array([(false_scores >= t).sum() for t in thresholds])
        # sklearn has much faster implementation
        fp, tp, thresholds = sklearn.metrics.ranking._binary_clf_curve(labels, scores, pos_label=1)
        nGroundTruth = labels.sum()
        fn = nGroundTruth - tp
        tn = nGroundTruth - fp

        # TODO: if this breaks check implmentation in
        #fpr_curve, tpr_curve, thresholds = sklearn.metrics.roc_curve(labels, scores, pos_label=1)
        fpr = fp / fp[-1]
        tpr = tp / tp[-1]

    def get_precision(tp, fp):
        """ precision -- positive predictive value (PPV) """
        precision = tp / (tp + fp)
        #precision = np.nan_to_num(precision)
        return precision

    def get_recall(tp, fn):
        """ recall, true positive rate, sensitivity, hit rate """
        recall = tp / (tp + fn)
        #recall = np.nan_to_num(recall)
        return recall

    ppv = precision = get_precision(tp, fp)
    precision[np.isnan(precision)] = 1.0
    #precision = np.nan_to_num(precision)
    #recall    = get_recall(tp, fn)
    # False positive rate (fall-out)
    #fall_out = fp / (fp + tn)

    #memtrack.report('[SUPERFLOUS]')
    #fpr = fpr_curve
    #tpr = tpr_curve

    #assert np.allclose(tpr_curve, recall)
    #assert np.allclose(fpr_curve, fall_out)
    #memtrack.report('[MAKE]')

    confusions = ConfusionMetrics(thresholds, tp, fp, fn, tn, fpr, tpr, ppv)
    #memtrack.report('[RETURN]')

    #print(tpr_curve)
    #print(recall)
    return confusions


def interact_roc_factory(confusions, target_tpr=None):
    r"""
    Args:
        confusions (?):

    CommandLine:
        python -m vtool.confusion --exec-interact_roc_factory --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.confusion import *  # NOQA
        >>> scores, labels = testdata_scores_labels()
        >>> confusions = get_confusion_metrics(scores, labels)
        >>> ut.quit_if_noshow()
        >>> ROCInteraction = interact_roc_factory(confusions, target_tpr=.92)
        >>> inter = ROCInteraction()
        >>> inter.show_page()
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

        @staticmethod
        def static_plot(fnum, pnum, **kwargs):
            #print('ROC Interact2')
            kwargs['thresholds'] = kwargs.get('thresholds', confusions.thresholds)
            confusions.draw_roc_curve(fnum=fnum, pnum=pnum, target_tpr=target_tpr, **kwargs)

        def plot(self, fnum, pnum):
            #print('ROC Interact3')
            self.static_plot(fnum, pnum, target_fpr=self.target_fpr)

        def on_click_inside(self, event, ex):
            self.target_fpr = event.xdata
            self.show_page()
            self.draw()

    return ROCInteraction


def draw_roc_curve(fpr, tpr, fnum=None, pnum=None, marker='-', target_tpr=None,
                   target_fpr=None, thresholds=None, color=(0.4, 1.0, 0.4)):
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    title_suffix = ''

    if target_fpr is not None:
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
                index = np.nonzero(tpr > choice_tpr)[0][0]
            except IndexError:
                index = len(thresholds) - 1
            choice_thresh = thresholds[index]
        #percent = ut.scalar_str(choice_tpr * 100).split('.')[0]
        #title_suffix = ', FPR%s=%05.2f%%' % (percent, choice_fpr)
        title_suffix = ', fpr=%.2f, tpr=%.2f, thresh=%.2f' % (
            choice_fpr, choice_tpr, choice_thresh)
    else:
        title_suffix = ''

    #if recall_domain is None:
    #    ave_p = np.nan
    #else:
    #    ave_p = p_interp.sum() / p_interp.size
    title = 'Receiver operating characteristic\n' + 'auc=%.3f' % (roc_auc,)
    title += title_suffix

    pt.plot2(fpr, tpr, marker=marker,
             x_label='False Positive Rate',
             y_label='True Positive Rate',
             unitbox=True, flipx=False, color=color, fnum=fnum, pnum=pnum,
             title=title)

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
            interpolate_replbounds(fpr[unique_fpr_idxs], tpr[unique_fpr_idxs], line_),
            'b-x')
        pt.plt.plot(
            interpolate_replbounds(tpr[unique_tpr_idxs], fpr[unique_tpr_idxs], line_),
            line_,
            'r-x')
    if choice_fpr is not None:
        pt.plot(choice_fpr, choice_tpr, 'o', color=pt.PINK)


def draw_precision_recall_curve(recall_domain, p_interp, title_pref=None,
                                fnum=1, pnum=None, color=(0.4, 1.0, 0.4)):
    import plottool as pt
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
