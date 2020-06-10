# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

# import warning
import numpy as np
import utool as ut
import pandas as pd

from sklearn.utils.validation import check_array

# from sklearn.utils import check_random_state
from six.moves import zip

# from sklearn.model_selection._split import (_BaseKFold, KFold)
from sklearn.model_selection._split import _BaseKFold

print, rrr, profile = ut.inject2(__name__)


# from sklearn.utils.fixes import bincount
bincount = np.bincount


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds cross-validator with Grouping

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of GroupKFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(StratifiedGroupKFold, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None, groups=None):
        """
        Args:
            self (?):
            X (ndarray):  data
            y (ndarray):  labels(default = None)
            groups (None): (default = None)

        Returns:
            ?: test_folds

        CommandLine:
            python -m wbia.algo.verif.sklearn_utils _make_test_folds

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.verif.sklearn_utils import *  # NOQA
            >>> import utool as ut
            >>> rng = ut.ensure_rng(0)
            >>> groups = [1, 1, 3, 4, 2, 2, 7, 8, 8]
            >>> y      = [1, 1, 1, 1, 2, 2, 2, 3, 3]
            >>> X = np.empty((len(y), 0))
            >>> self = StratifiedGroupKFold(random_state=rng)
            >>> skf_list = list(self.split(X=X, y=y, groups=groups))
        """
        # if self.shuffle:
        #     rng = check_random_state(self.random_state)
        # else:
        #     rng = self.random_state
        n_splits = self.n_splits
        y = np.asarray(y)
        n_samples = y.shape[0]

        import utool as ut

        # y_counts = bincount(y_inversed)
        # min_classes_ = np.min(y_counts)
        # if np.all(self.n_splits > y_counts):
        #     raise ValueError("All the n_groups for individual classes"
        #                      " are less than n_splits=%d."
        #                      % (self.n_splits))
        # if self.n_splits > min_classes_:
        #     warnings.warn(("The least populated class in y has only %d"
        #                    " members, which is too few. The minimum"
        #                    " number of groups for any class cannot"
        #                    " be less than n_splits=%d."
        #                    % (min_classes_, self.n_splits)), Warning)

        unique_y, y_inversed = np.unique(y, return_inverse=True)
        n_classes = max(unique_y) + 1
        unique_groups, group_idxs = ut.group_indices(groups)
        # grouped_ids = list(grouping.keys())
        grouped_y = ut.apply_grouping(y, group_idxs)
        grouped_y_counts = np.array(
            [bincount(y_, minlength=n_classes) for y_ in grouped_y]
        )

        target_freq = grouped_y_counts.sum(axis=0)
        target_ratio = target_freq / target_freq.sum()

        # Greedilly choose the split assignment that minimizes the local
        # * squared differences in target from actual frequencies
        # * and best equalizes the number of items per fold
        # Distribute groups with most members first
        split_freq = np.zeros((n_splits, n_classes))
        # split_ratios = split_freq / split_freq.sum(axis=1)
        split_ratios = np.ones(split_freq.shape) / split_freq.shape[1]
        split_diffs = ((split_freq - target_ratio) ** 2).sum(axis=1)
        sortx = np.argsort(grouped_y_counts.sum(axis=1))[::-1]
        grouped_splitx = []
        for count, group_idx in enumerate(sortx):
            # print('---------\n')
            group_freq = grouped_y_counts[group_idx]
            cand_freq = split_freq + group_freq
            cand_ratio = cand_freq / cand_freq.sum(axis=1)[:, None]
            cand_diffs = ((cand_ratio - target_ratio) ** 2).sum(axis=1)
            # Compute loss
            losses = []
            # others = np.nan_to_num(split_diffs)
            other_diffs = np.array(
                [
                    sum(split_diffs[x + 1 :]) + sum(split_diffs[:x])
                    for x in range(n_splits)
                ]
            )
            # penalize unbalanced splits
            ratio_loss = other_diffs + cand_diffs
            # penalize heavy splits
            freq_loss = split_freq.sum(axis=1)
            freq_loss = freq_loss / freq_loss.sum()
            losses = ratio_loss + freq_loss
            # print('group_freq = %r' % (group_freq,))
            # print('freq_loss = %s' % (ut.repr2(freq_loss, precision=2),))
            # print('ratio_loss = %s' % (ut.repr2(ratio_loss, precision=2),))
            # -------
            splitx = np.argmin(losses)
            # print('losses = %r, splitx=%r' % (losses, splitx))
            split_freq[splitx] = cand_freq[splitx]
            split_ratios[splitx] = cand_ratio[splitx]
            split_diffs[splitx] = cand_diffs[splitx]
            grouped_splitx.append(splitx)

            # if count > 4:
            #     break
            # else:
            #     print('split_freq = \n' +
            #           ut.repr2(split_freq, precision=2, suppress_small=True))
            #     print('target_ratio = \n' +
            #           ut.repr2(target_ratio, precision=2, suppress_small=True))
            #     print('split_ratios = \n' +
            #           ut.repr2(split_ratios, precision=2, suppress_small=True))
            #     print(ut.dict_hist(grouped_splitx))

        # final_ratio_loss = ((split_ratios - target_ratio) ** 2).sum(axis=1)
        # print('split_freq = \n' +
        #       ut.repr2(split_freq, precision=3, suppress_small=True))
        # print('target_ratio = \n' +
        #       ut.repr2(target_ratio, precision=3, suppress_small=True))
        # print('split_ratios = \n' +
        #       ut.repr2(split_ratios, precision=3, suppress_small=True))
        # print(ut.dict_hist(grouped_splitx))

        test_folds = np.empty(n_samples, dtype=np.int)
        for group_idx, splitx in zip(sortx, grouped_splitx):
            idxs = group_idxs[group_idx]
            test_folds[idxs] = splitx

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y, groups)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(StratifiedGroupKFold, self).split(X, y, groups)


def temp(samples):
    from sklearn import model_selection
    from wbia.algo.verif import sklearn_utils

    def check_balance(idxs):
        # from sklearn.utils.fixes import bincount
        print('-------')
        for count, (test, train) in enumerate(idxs):
            print('split %r' % (count))
            groups_train = set(groups.take(train))
            groups_test = set(groups.take(test))
            n_group_isect = len(groups_train.intersection(groups_test))
            y_train_freq = bincount(y.take(train))
            y_test_freq = bincount(y.take(test))
            y_test_ratio = y_test_freq / y_test_freq.sum()
            y_train_ratio = y_train_freq / y_train_freq.sum()
            balance_error = np.sum((y_test_ratio - y_train_ratio) ** 2)
            print('n_group_isect = %r' % (n_group_isect,))
            print('y_test_ratio = %r' % (y_test_ratio,))
            print('y_train_ratio = %r' % (y_train_ratio,))
            print('balance_error = %r' % (balance_error,))

    X = np.empty((len(samples), 0))
    y = samples.encoded_1d().values
    groups = samples.group_ids

    n_splits = 3

    splitter = model_selection.GroupShuffleSplit(n_splits=n_splits)
    idxs = list(splitter.split(X=X, y=y, groups=groups))
    check_balance(idxs)

    splitter = model_selection.GroupKFold(n_splits=n_splits)
    idxs = list(splitter.split(X=X, y=y, groups=groups))
    check_balance(idxs)

    splitter = model_selection.StratifiedKFold(n_splits=n_splits)
    idxs = list(splitter.split(X=X, y=y, groups=groups))
    check_balance(idxs)

    splitter = sklearn_utils.StratifiedGroupKFold(n_splits=n_splits)
    idxs = list(splitter.split(X=X, y=y, groups=groups))
    check_balance(idxs)


def testdata_ytrue(p_classes, p_wrong, size, rng):
    classes_ = list(range(len(p_classes)))
    # Generate samples at specified fractions
    y_true = rng.choice(classes_, size=size, p=p_classes)
    return y_true


def testdata_ypred(y_true, p_wrong, rng):
    # Make mistakes at specified rate
    classes_ = list(range(len(p_wrong)))
    y_pred = np.array(
        [y if rng.rand() > p_wrong[y] else rng.choice(classes_) for y in y_true]
    )
    return y_pred


def classification_report2(
    y_true, y_pred, target_names=None, sample_weight=None, verbose=True
):
    """
    References:
        https://csem.flinders.edu.au/research/techreps/SIE07001.pdf
        https://www.mathworks.com/matlabcentral/fileexchange/5648-bm-cm-?requestedDomain=www.mathworks.com
        Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC and CEN
            Error Measures in MultiClass Prediction

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.verif.sklearn_utils import *  # NOQA
        >>> y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        >>> y_pred = [1, 2, 1, 3, 1, 2, 2, 3, 2, 2, 3, 3, 2, 3, 3, 3, 1, 3]
        >>> target_names = None
        >>> sample_weight = None
        >>> verbose = True
        >>> report = classification_report2(y_true, y_pred, verbose=verbose)

    Ignore:
        >>> size = 100
        >>> rng = np.random.RandomState(0)
        >>> p_classes = np.array([.90, .05, .05][0:2])
        >>> p_classes = p_classes / p_classes.sum()
        >>> p_wrong   = np.array([.03, .01, .02][0:2])
        >>> y_true = testdata_ytrue(p_classes, p_wrong, size, rng)
        >>> rs = []
        >>> for x in range(17):
        >>>     p_wrong += .05
        >>>     y_pred = testdata_ypred(y_true, p_wrong, rng)
        >>>     report = classification_report2(y_true, y_pred, verbose='hack')
        >>>     rs.append(report)
        >>> import wbia.plottool as pt
        >>> pt.qtensure()
        >>> df = pd.DataFrame(rs).drop(['raw'], axis=1)
        >>> delta = df.subtract(df['target'], axis=0)
        >>> sqrd_error = np.sqrt((delta ** 2).sum(axis=0))
        >>> print('Error')
        >>> print(sqrd_error.sort_values())
        >>> ys = df.to_dict(orient='list')
        >>> pt.multi_plot(ydata_list=ys)
    """
    import sklearn.metrics
    from sklearn.preprocessing import LabelEncoder

    if target_names is None:
        unique_labels = np.unique(np.hstack([y_true, y_pred]))
        if len(unique_labels) == 1 and (unique_labels[0] == 0 or unique_labels[0] == 1):
            target_names = np.array([False, True])
            y_true_ = y_true
            y_pred_ = y_pred
        else:
            lb = LabelEncoder()
            lb.fit(unique_labels)
            y_true_ = lb.transform(y_true)
            y_pred_ = lb.transform(y_pred)
            target_names = lb.classes_
    else:
        y_true_ = y_true
        y_pred_ = y_pred

    # Real data is on the rows,
    # Pred data is on the cols.

    cm = sklearn.metrics.confusion_matrix(y_true_, y_pred_, sample_weight=sample_weight)
    confusion = cm  # NOQA

    k = len(cm)  # number of classes
    N = cm.sum()  # number of examples

    real_total = cm.sum(axis=1)
    pred_total = cm.sum(axis=0)

    # the number of "positive" cases **per class**
    n_pos = real_total  # NOQA
    # the number of times a class was predicted.
    n_neg = N - n_pos  # NOQA

    # number of true positives per class
    n_tps = np.diag(cm)
    # number of true negatives per class
    n_fps = (cm - np.diagflat(np.diag(cm))).sum(axis=0)

    tprs = n_tps / real_total  # true pos rate (recall)
    tpas = n_tps / pred_total  # true pos accuracy (precision)

    unused = (real_total + pred_total) == 0

    fprs = n_fps / n_neg  # false pose rate
    fprs[unused] = np.nan
    # tnrs = 1 - fprs

    rprob = real_total / N
    pprob = pred_total / N

    if len(cm) == 2:
        [[A, B], [C, D]] = cm
        (A * D - B * C) / np.sqrt((A + C) * (B + D) * (A + B) * (C + D))

        # c2 = vt.ConfusionMetrics().fit(scores, y)

    # bookmaker is analogous to recall, but unbiased by class frequency
    rprob_mat = np.tile(rprob, [k, 1]).T - (1 - np.eye(k))
    bmcm = cm.T / rprob_mat
    bms = np.sum(bmcm.T, axis=0) / N

    # markedness is analogous to precision, but unbiased by class frequency
    pprob_mat = np.tile(pprob, [k, 1]).T - (1 - np.eye(k))
    mkcm = cm / pprob_mat
    mks = np.sum(mkcm.T, axis=0) / N

    mccs = np.sign(bms) * np.sqrt(np.abs(bms * mks))

    perclass_data = ut.odict(
        [
            ('precision', tpas),
            ('recall', tprs),
            ('fpr', fprs),
            ('markedness', mks),
            ('bookmaker', bms),
            ('mcc', mccs),
            ('support', real_total),
        ]
    )

    tpa = np.nansum(tpas * rprob)
    tpr = np.nansum(tprs * rprob)

    fpr = np.nansum(fprs * rprob)

    mk = np.nansum(mks * rprob)
    bm = np.nansum(bms * pprob)

    # The simple mean seems to do the best
    mccs_ = mccs[~np.isnan(mccs)]
    if len(mccs_) == 0:
        mcc_combo = np.nan
    else:
        mcc_combo = np.nanmean(mccs_)

    combined_data = ut.odict(
        [
            ('precision', tpa),
            ('recall', tpr),
            ('fpr', fpr),
            ('markedness', mk),
            ('bookmaker', bm),
            # ('mcc', np.sign(bm) * np.sqrt(np.abs(bm * mk))),
            ('mcc', mcc_combo),
            # np.sign(bm) * np.sqrt(np.abs(bm * mk))),
            ('support', real_total.sum()),
        ]
    )

    # Not sure how to compute this. Should it agree with the sklearn impl?
    if verbose == 'hack':
        verbose = False
        mcc_known = sklearn.metrics.matthews_corrcoef(
            y_true, y_pred, sample_weight=sample_weight
        )
        mcc_raw = np.sign(bm) * np.sqrt(np.abs(bm * mk))

        import scipy as sp

        def gmean(x, w=None):
            if w is None:
                return sp.stats.gmean(x)
            return np.exp(np.nansum(w * np.log(x)) / np.nansum(w))

        def hmean(x, w=None):
            if w is None:
                return sp.stats.hmean(x)
            return 1 / (np.nansum(w * (1 / x)) / np.nansum(w))

        def amean(x, w=None):
            if w is None:
                return np.mean(x)
            return np.nansum(w * x) / np.nansum(w)

        report = {
            'target': mcc_known,
            'raw': mcc_raw,
        }

        # print('%r <<<' % (mcc_known,))
        means = {
            'a': amean,
            # 'h': hmean,
            'g': gmean,
        }
        weights = {
            'p': pprob,
            'r': rprob,
            '': None,
        }
        for mean_key, mean in means.items():
            for w_key, w in weights.items():
                # Hack of very wrong items
                if mean_key == 'g':
                    if w_key in ['r', 'p', '']:
                        continue
                if mean_key == 'g':
                    if w_key in ['r']:
                        continue
                m = mean(mccs, w)
                r_key = '{} {}'.format(mean_key, w_key)
                report[r_key] = m
                # print(r_key)
                # print(np.abs(m - mcc_known))

        # print(ut.repr4(report, precision=8))
        return report
        # print('mcc_known = %r' % (mcc_known,))
        # print('mcc_combo1 = %r' % (mcc_combo1,))
        # print('mcc_combo2 = %r' % (mcc_combo2,))
        # print('mcc_combo3 = %r' % (mcc_combo3,))

    # if target_names is None:
    #     target_names = list(range(k))
    index = pd.Index(target_names, name='class')

    perclass_df = pd.DataFrame(perclass_data, index=index)
    # combined_df = pd.DataFrame(combined_data, index=['ave/sum'])
    combined_df = pd.DataFrame(combined_data, index=['combined'])

    metric_df = pd.concat([perclass_df, combined_df])
    metric_df.index.name = 'class'
    metric_df.columns.name = 'metric'

    pred_id = ['%s' % m for m in target_names]
    real_id = ['%s' % m for m in target_names]
    confusion_df = pd.DataFrame(confusion, columns=pred_id, index=real_id)

    confusion_df = confusion_df.append(
        pd.DataFrame([confusion.sum(axis=0)], columns=pred_id, index=['Σp'])
    )
    confusion_df['Σr'] = np.hstack([confusion.sum(axis=1), [0]])
    confusion_df.index.name = 'real'
    confusion_df.columns.name = 'pred'

    if np.all(confusion_df - np.floor(confusion_df) < 0.000001):
        confusion_df = confusion_df.astype(np.int)
    confusion_df.iloc[(-1, -1)] = N
    if np.all(confusion_df - np.floor(confusion_df) < 0.000001):
        confusion_df = confusion_df.astype(np.int)
    # np.nan

    if verbose:
        cfsm_str = confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))
        print('Confusion Matrix (real × pred) :')
        print(ut.hz_str('    ', cfsm_str))

        # ut.cprint('\nExtended Report', 'turquoise')
        print('\nEvaluation Metric Report:')
        float_precision = 2
        float_format = '%.' + str(float_precision) + 'f'
        ext_report = metric_df.to_string(float_format=float_format)
        print(ut.hz_str('    ', ext_report))

    report = {
        'metrics': metric_df,
        'confusion': confusion_df,
    }

    # FIXME: What is the difference between sklearn multiclass-MCC
    # and BM * MK MCC?

    def matthews_corrcoef(y_true, y_pred, sample_weight=None):
        from sklearn.metrics.classification import (
            _check_targets,
            LabelEncoder,
            confusion_matrix,
        )

        y_type, y_true, y_pred = _check_targets(y_true, y_pred)
        if y_type not in {'binary', 'multiclass'}:
            raise ValueError('%s is not supported' % y_type)
        lb = LabelEncoder()
        lb.fit(np.hstack([y_true, y_pred]))
        y_true = lb.transform(y_true)
        y_pred = lb.transform(y_pred)
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        t_sum = C.sum(axis=1)
        p_sum = C.sum(axis=0)
        n_correct = np.trace(C)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
        if np.isnan(mcc):
            return 0.0
        else:
            return mcc

    try:
        # mcc = sklearn.metrics.matthews_corrcoef(
        #     y_true, y_pred, sample_weight=sample_weight)
        mcc = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
        # These scales are chosen somewhat arbitrarily in the context of a
        # computer vision application with relatively reasonable quality data
        # https://stats.stackexchange.com/questions/118219/how-to-interpret
        mcc_significance_scales = ut.odict(
            [
                (1.0, 'perfect'),
                (0.9, 'very strong'),
                (0.7, 'strong'),
                (0.5, 'significant'),
                (0.3, 'moderate'),
                (0.2, 'weak'),
                (0.0, 'negligible'),
            ]
        )
        for k, v in mcc_significance_scales.items():
            if np.abs(mcc) >= k:
                if verbose:
                    print('classifier correlation is %s' % (v,))
                break
        if verbose:
            float_precision = 2
            print(("MCC' = %." + str(float_precision) + 'f') % (mcc,))
        report['mcc'] = mcc
    except ValueError:
        pass
    return report


def predict_from_probs(probs, method='argmax', target_names=None, **kwargs):
    """
    Predictions are returned as indices into columns or target_names

    Doctest:
        >>> from wbia.algo.verif.sklearn_utils import *
        >>> rng = np.random.RandomState(0)
        >>> probs = pd.DataFrame(rng.rand(10, 3), columns=['a', 'b', 'c'])
        >>> pred1 = predict_from_probs(probs, 'argmax')
        >>> pred2 = predict_from_probs(probs, 'argmax', target_names=probs.columns)
        >>> threshes = probs.loc[0]
        >>> pred3 = predict_from_probs(probs, threshes.values, force=True,
        >>>                            target_names=probs.columns)
    """
    import six

    if isinstance(method, six.string_types) and method == 'argmax':
        if isinstance(probs, pd.DataFrame):
            pred_enc = pd.Series(probs.values.argmax(axis=1), index=probs.index)
        else:
            pred_enc = probs.argmax(axis=1)
    else:
        threshes = method
        pred_enc = predict_with_thresh(probs, threshes, target_names, **kwargs)
    return pred_enc


def predict_with_thresh(
    probs, threshes, target_names=None, force=False, multi=True, return_flags=False
):
    """

    if force is true, everything will make a prediction, even if nothing passes
    the thresholds. In that case it will use argmax.

    if more than one thing passes the thresold we take the highest one if
    multi=True, and return nan otherwise.

    Doctest:
        >>> from wbia.algo.verif.sklearn_utils import *
        >>> probs = np.array([
        >>>     [0.5, 0.5, 0.0],
        >>>     [0.4, 0.5, 0.1],
        >>>     [1.0, 0.0, 0.0],
        >>>     [0.3, 0.3, 0.4],
        >>>     [0.1, 0.3, 0.6],
        >>>     [0.1, 0.6, 0.3],
        >>>     [0.6, 0.1, 0.3],])
        >>> threshes = [.5, .5, .5]
        >>> pred_enc = predict_with_thresh(probs, threshes)
        >>> a = predict_with_thresh(probs, [.5, .5, .5])
        >>> b = predict_with_thresh(probs, [.5, .5, .5], force=True)
        >>> assert np.isnan(a).sum() == 3
        >>> assert np.isnan(b).sum() == 0
    """
    df_index = None
    if isinstance(probs, pd.DataFrame):
        df_index = probs.index
        if target_names is None and isinstance(threshes, dict):
            target_names = probs.columns.tolist()
        probs = probs.values

    if isinstance(threshes, dict):
        if target_names is None:
            raise ValueError('need target names to use a dict of threshes')
        threshes = ut.take(threshes, target_names)

    # if force:
    #     bin_flags = (probs >= threshes)
    bin_flags = probs > threshes
    num_states = bin_flags.sum(axis=1)

    no_predict = num_states == 0
    multi_predict = num_states > 1

    pred_enc = bin_flags.argmax(axis=1)

    if np.any(no_predict):
        if force or return_flags:
            pred_enc[no_predict] = probs[no_predict].argmax(axis=1)
        else:
            pred_enc = pred_enc.astype(np.float)
            pred_enc[no_predict] = np.nan

    if np.any(multi_predict):
        if multi or return_flags:
            pred_enc[multi_predict] = probs[multi_predict].argmax(axis=1)
        else:
            pred_enc = pred_enc.astype(np.float)
            pred_enc[multi_predict] = np.nan

    if df_index is not None:
        pred_enc = pd.Series(pred_enc, index=df_index)
        # pred = pred_enc.apply(lambda x: target_names[x])
    if return_flags:
        flags = np.ones(len(probs), dtype=np.bool)
        if not force:
            flags[no_predict] = False
        if not multi:
            flags[no_predict] = False
        return pred_enc, flags
    else:
        return pred_enc


def predict_proba_df(clf, X_df, class_names=None):
    """
    Calls sklearn classifier predict_proba but then puts results in a dataframe
    using the same index as X_df and incorporating all possible class_names
    given
    """
    if class_names is not None:
        columns = ut.take(class_names, clf.classes_)
    else:
        columns = None
    if len(X_df) == 0:
        return pd.DataFrame(columns=columns)
    try:
        probs = clf.predict_proba(X_df)
    except ValueError:
        # solves a problem when values are infinity for whatever reason
        X = X_df.values.copy()
        X[~np.isfinite(X)] = np.nan
        probs = clf.predict_proba(X)

    probs_df = pd.DataFrame(probs, columns=columns, index=X_df.index)
    # add in zero probability for classes without training data
    if class_names is not None:
        missing = ut.setdiff(class_names, columns)
        if missing:
            for classname in missing:
                probs_df = probs_df.assign(**{classname: np.zeros(len(probs_df))})
    return probs_df


class PrefitEstimatorEnsemble(object):
    """

    hacks around limitations of sklearn.ensemble.VotingClassifier

    """

    def __init__(self, clf_list, voting='soft', weights=None):
        self.clf_list = clf_list
        self.voting = voting
        self.weights = None

        classes_list = [clf.classes_ for clf in clf_list]
        if ut.allsame(classes_list):
            self.classes_ = classes_list[0]
            self.class_idx_mappers = None
        else:
            # Need to make a mapper from individual clf classes to ensemble
            self.class_idx_mappers = []
            classes_ = sorted(set.union(*map(set, classes_list)))
            for clf in clf_list:
                # For each index of the clf classes, find that index in the
                # ensemble classes. Eg. class y=4 might be at cx=1 and ex=0
                mapper = np.empty(len(clf.classes_), dtype=np.int)
                for cx, y in enumerate(clf.classes_):
                    ex = classes_.index(y)
                    mapper[cx] = ex
                self.class_idx_mappers.append(mapper)
            self.classes_ = np.array(classes_)

        for clf in clf_list:
            clf.classes_
            pass

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        if self.class_idx_mappers is None:
            probas = np.asarray([clf.predict_proba(X) for clf in self.clf_list])
        else:
            n_estimators = len(self.clf_list)
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            probas = np.zeros((n_estimators, n_samples, n_classes))
            for ex, (clf, mapper) in enumerate(
                zip(self.clf_list, self.class_idx_mappers)
            ):
                proba = clf.predict_proba(X)
                # Use mapper to map indicies of clf classes to ensemble classes
                probas[ex][:, mapper] = proba
        return probas

    def predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError(
                'predict_proba is not available when' ' voting=%r' % self.voting
            )
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions.astype('int'),
            )
        return maj

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clf_list]).T


def voting_ensemble(clf_list, voting='hard'):
    """
    hack to construct a VotingClassifier from pretrained classifiers
    TODO: contribute similar functionality to sklearn
    """
    eclf = PrefitEstimatorEnsemble(clf_list, voting=voting)
    # classes_ = ut.list_getattr(clf_list, 'classes_')
    # if not ut.allsame(classes_):
    #     for clf in clf_list:
    #         print(clf.predict_proba(X_train))
    #         pass
    #     # Note: There is a corner case where one fold doesn't get any labels of
    #     # a certain class. Because y_train is an encoded integer, the
    #     # clf.classes_ attribute will cause predictions to agree with other
    #     # classifiers trained on the same labels. Therefore, the voting
    #     # classifer will still work. But
    #     raise ValueError(
    #         'Classifiers predict different things. classes_={}'.format(
    #             classes_)
    #     )
    # estimators = [('clf%d' % count, clf) for count, clf in enumerate(clf_list)]
    # eclf = sklearn.ensemble.VotingClassifier(estimators=estimators,
    #                                          voting=voting)
    # eclf.classes_ = clf_list[0].classes_
    # eclf.estimators_ = clf_list
    return eclf
