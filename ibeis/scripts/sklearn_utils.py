from __future__ import print_function, division
# import warnings
import numpy as np
import utool as ut
import pandas as pd

from sklearn.utils.validation import check_array
# from sklearn.utils import check_random_state
from sklearn.externals.six.moves import zip
from sklearn.utils.fixes import bincount
# from sklearn.model_selection._split import (_BaseKFold, KFold)
from sklearn.model_selection._split import (_BaseKFold,)


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
        grouped_y_counts = np.array([
            bincount(y_, minlength=n_classes) for y_ in grouped_y])

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
            other_diffs = np.array([
                sum(split_diffs[x + 1:]) + sum(split_diffs[:x])
                for x in range(n_splits)
            ])
            # penalize unbalanced splits
            ratio_loss = other_diffs + cand_diffs
            # penalize heavy splits
            freq_loss = split_freq.sum(axis=1)
            freq_loss = freq_loss / freq_loss.sum()
            losses = ratio_loss + freq_loss
            # print('group_freq = %r' % (group_freq,))
            # print('freq_loss = %s' % (ut.repr2(freq_loss, precision=2),))
            # print('ratio_loss = %s' % (ut.repr2(ratio_loss, precision=2),))
            #-------
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
    from ibeis.scripts import sklearn_utils
    def check_balance(idxs):
        from sklearn.utils.fixes import bincount
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
    y = samples.encoded_1d()
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


def classification_report2(y_true, y_pred, target_names=None,
                           sample_weight=None, verbose=True):
    import sklearn.metrics
    from sklearn.preprocessing import LabelEncoder

    if target_names is None:
        lb = LabelEncoder()
        lb.fit(np.hstack([y_true, y_pred]))
        y_true_ = lb.transform(y_true)
        y_pred_ = lb.transform(y_pred)
        target_names = lb.classes_
    else:
        y_true_ = y_true
        y_pred_ = y_pred

    cm = sklearn.metrics.confusion_matrix(
        y_true_, y_pred_, sample_weight=sample_weight)
    confusion = cm  # NOQA

    k = len(cm)
    N = cm.sum()

    real_total = cm.sum(axis=1)
    pred_total = cm.sum(axis=0)

    n_tps = np.diag(cm)
    tprs = n_tps / real_total
    tpas = n_tps / pred_total

    rprob = real_total / N
    pprob = pred_total / N

    # bookmaker is analogous to recall
    rprob_mat = np.tile(rprob, [k, 1]).T - (1 - np.eye(k))
    bmcm = cm.T / rprob_mat
    bms = np.sum(bmcm.T, axis=0) / N

    # markedness is analogous to precision
    pprob_mat = np.tile(pprob, [k, 1]).T - (1 - np.eye(k))
    mkcm = cm / pprob_mat
    mks = np.sum(mkcm.T, axis=0) / N

    perclass_data = ut.odict([
        ('precision', tpas),
        ('recall', tprs),
        ('markedness', mks),
        ('bookmaker', bms),
        ('mcc', np.sign(bms) * np.sqrt(np.abs(bms * mks))),
        ('support', real_total),
    ])
    tpa = np.nansum(tpas * rprob)
    tpr = np.nansum(tprs * rprob)
    mk = np.nansum(mks * rprob)
    bm = np.nansum(bms * pprob)

    combined_data = ut.odict([
        ('precision', tpa),
        ('recall', tpr),
        ('markedness', mk),
        ('bookmaker', bm),
        ('mcc', np.sign(bm) * np.sqrt(np.abs(bm * mk))),
        ('support', real_total.sum())
    ])

    if target_names is None:
        target_names = list(range(k))
    index = pd.Series(target_names, name='class')

    perclass_df = pd.DataFrame(perclass_data, index=index)
    combined_df = pd.DataFrame(combined_data, index=['ave/sum'])
    metric_df = pd.concat([perclass_df, combined_df])
    metric_df.index.name = 'class'
    metric_df.columns.name = 'metric'

    pred_id = ['%s' % m for m in target_names]
    real_id = ['%s' % m for m in target_names]
    confusion_df = pd.DataFrame(confusion, columns=pred_id, index=real_id)
    confusion_df = confusion_df.append(pd.DataFrame(
        [confusion.sum(axis=0)], columns=pred_id, index=['Σp']))
    confusion_df['Σr'] = np.hstack([confusion.sum(axis=1), [np.nan]])
    confusion_df.index.name = 'real'
    confusion_df.columns.name = 'pred'

    if verbose:
        cfsm_str = confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))
        print('Confusion Matrix (real × pred) :')
        print(ut.hz_str('    ', cfsm_str))

        # ut.cprint('\nExtended Report', 'turquoise')
        print('\nEvaluation Metric Report:')
        precision = 2
        float_format = '%.' + str(precision) + 'f'
        ext_report = metric_df.to_string(float_format=float_format)
        print(ut.hz_str('    ', ext_report))
    return metric_df, confusion_df
