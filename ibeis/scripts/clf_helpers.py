# -*- coding: utf-8 -*-
"""
This module is a work in progress, as such concepts are subject to change.

MAIN IDEA:
    `MultiTaskSamples` serves as a structure to contain and manipulate a set of
    samples with potentially many different types of labels and features.
"""
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import numpy as np
from six.moves import range
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
print, rrr, profile = ut.inject2(__name__)


@ut.reloadable_class
class MultiTaskSamples(ut.NiceRepr):
    """
    Handles samples (i.e. feature-label pairs) with a combination of
    non-mutually exclusive subclassification labels

    CommandLine:
        python -m ibeis.scripts.clf_helpers MultiTaskSamples

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.clf_helpers import *  # NOQA
        >>> samples = MultiTaskSamples()
        >>> tasks_to_indicators = ut.odict([
        >>>     ('task1', ut.odict([
        >>>         ('state1', [0, 0, 0, 1]),
        >>>         ('state2', [0, 0, 1, 0]),
        >>>         ('state3', [1, 1, 0, 0]),
        >>>     ])),
        >>>     ('task2', ut.odict([
        >>>         ('state4', [0, 0, 0, 1]),
        >>>         ('state5', [1, 1, 1, 0]),
        >>>     ]))
        >>> ])
        >>> samples.apply_indicators(tasks_to_indicators)
    """
    def __init__(samples):

        # TODO:
        # samples.index - aid_pairs
        # samples.subtasks
        samples.index = None
        samples.X_dict = None
        samples.subtasks = ut.odict()

    def apply_indicators(samples, tasks_to_indicators):
        samples.n_tasks = len(tasks_to_indicators)
        for task_name, indicator in tasks_to_indicators.items():
            labels = MultiClassLabels.from_indicators(
                indicator, task_name=task_name)
            samples.subtasks[task_name] = labels

    @ut.memoize
    def encoded_2d(samples):
        encoded_2d = pd.concat([v.encoded_df for k, v in samples.items()], axis=1).values
        return encoded_2d

    def class_name_basis(samples):
        """ corresponds with indexes returned from encoded1d """
        class_name_basis = [(b, a) for a, b in ut.product(*[
            v.class_names for k, v in samples.items()][::-1])]
        return class_name_basis

    def class_idx_basis_2d(samples):
        """ 2d-index version of class_name_basis """
        class_idx_basis_2d = [(b, a) for a, b in ut.product(*[
            range(v.n_classes) for k, v in samples.items()][::-1])]
        return class_idx_basis_2d

    def class_idx_basis_1d(samples):
        """ 1d-index version of class_name_basis """
        n_states = np.prod([v.n_classes for k, v in samples.items()])
        class_idx_basis_1d = np.arange(n_states, dtype=np.int)
        return class_idx_basis_1d

    @ut.memoize
    def encoded_1d(samples):
        """ Returns a unique label for each combination of samples """
        # from sklearn.preprocessing import MultiLabelBinarizer
        encoded_2d = samples.encoded_2d()
        class_space = [v.n_classes for k, v in samples.items()]
        offsets = np.array([1] + np.cumprod(class_space).tolist()[:-1])[None, :]
        encoded_1d = (offsets * encoded_2d).sum(axis=1)
        # e = MultiLabelBinarizer()
        # bin_coeff = e.fit_transform(encoded_2d)
        # bin_basis = (2 ** np.arange(bin_coeff.shape[1]))[None, :]
        # # encoded_1d = (bin_coeff * bin_basis).sum(axis=1)
        # encoded_1d = (bin_coeff * bin_basis[::-1]).sum(axis=1)
        # # vt.unique_rows(sklearn.preprocessing.MultiLabelBinarizer().fit_transform(encoded_2d))
        # [v.encoded_df.values for k, v in samples.items()]
        # encoded_df_1d = pd.concat([v.encoded_df for k, v in samples.items()], axis=1)
        return encoded_1d

    def __nice__(samples):
        return 'nS=%r, nT=%r' % (len(samples), samples.n_tasks)

    def __len__(samples):
        return samples.n_samples

    def print_info(samples):
        for task_name, labels in samples.items():
            labels.print_info()
        print('hist(all) = %s' % (ut.repr4(samples.make_histogram())))
        print('len(all) = %s' % (len(samples)))

    def make_histogram(samples):
        """ label histogram """
        class_name_basis = samples.class_name_basis()
        class_idx_basis_1d = samples.class_idx_basis_1d()
        # print('class_idx_basis_1d = %r' % (class_idx_basis_1d,))
        # print(samples.encoded_1d())
        multi_task_idx_hist = ut.dict_hist(
            samples.encoded_1d(), labels=class_idx_basis_1d)
        multi_task_hist = ut.map_keys(
            lambda k: class_name_basis[k], multi_task_idx_hist)
        return multi_task_hist

    def items(samples):
        for task_name, labels in samples.subtasks.items():
            yield task_name, labels

    # def take(samples, idxs):
    #     mask = ut.index_to_boolmask(idxs, len(samples))
    #     return samples.compress(mask)


@ut.reloadable_class
class MultiClassLabels(ut.NiceRepr):
    """
    Used by samples to encode a single set of mutually exclusive labels.  These
    can either be binary or multiclass.

        import pandas as pd
        pd.options.display.max_rows = 10
        # pd.options.display.max_rows = 20
        pd.options.display.max_columns = 40
        pd.options.display.width = 160
    """
    def __init__(labels):
        # Helper Info
        labels.task_name = None
        labels.class_names = None
        labels.classes_ = None
        labels.n_samples = None
        labels.n_classes = None
        # Core data
        labels.indicator_df = None
        labels.encoded_df = None

    @property
    def target_type(labels):
        return sklearn.utils.multiclass.type_of_target(labels.y_enc)

    def one_vs_rest_task_names(labels):
        return [labels.task_name + '(' + labels.class_names[k] + '-v-rest)'
                for k in range(labels.n_classes)]

    def gen_one_vs_rest_labels(labels):
        """
        >>> from ibeis.scripts.clf_helpers import *  # NOQA
        >>> indicator = ut.odict([
        >>>         ('state1', [0, 0, 0, 1]),
        >>>         ('state2', [0, 0, 1, 0]),
        >>>         ('state3', [1, 1, 0, 0]),
        >>>     ])
        >>> labels = MultiClassLabels.from_indicators(indicator, 'task1')
        >>> sublabels = list(labels.gen_one_vs_rest_labels())
        >>> sublabel = sublabels[0]
        """
        if labels.target_type == 'binary':
            yield labels
            raise StopIteration()
        task_names_1vR = labels.one_vs_rest_task_names()
        for k in range(labels.n_classes):
            class_name = labels.class_names[k]
            task_name = task_names_1vR[k]
            indicator_df = pd.DataFrame()
            indicator_df['not-' + class_name] = 1 - labels.indicator_df[class_name]
            indicator_df[class_name] = labels.indicator_df[class_name]
            # indicator = labels.encoded_df == k
            # indicator.rename(columns={indicator.columns[0]: class_name}, inplace=True)
            n_samples = len(indicator_df)
            sublabel = MultiClassLabels()
            sublabel.indicator_df = indicator_df
            sublabel.class_names = indicator_df.columns.values
            # if len(indicator_df.columns) == 1:
            #     sublabel.encoded_df = pd.DataFrame(
            #         indicator_df.values.T[0],
            #         columns=[task_name]
            #     )
            # else:
            sublabel.encoded_df = pd.DataFrame(
                indicator_df.values.argmax(axis=1),
                columns=[task_name]
            )
            sublabel.task_name = task_name
            sublabel.n_samples = n_samples
            sublabel.n_classes = len(sublabel.class_names)
            # if sublabel.n_classes == 1:
            #     sublabel.n_classes = 2  # 1 column means binary case
            sublabel.classes_ = np.arange(sublabel.n_classes)

            # sublabel = MultiClassLabels.from_indicators(indicator, task_name=subname)
            yield sublabel

    @property
    def y_bin(labels):
        return labels.indicator_df.values

    @property
    def y_enc(labels):
        return labels.encoded_df.values.ravel()

    @classmethod
    def from_indicators(MultiClassLabels, indicator, task_name=None):
        import six
        labels = MultiClassLabels()
        n_samples = len(six.next(six.itervalues(indicator)))
        # if index is None:
        #     index = pd.Series(np.arange(n_samples), name='index')
        indicator_df = pd.DataFrame(indicator)
        assert np.all(indicator_df.sum(axis=1).values), (
            'states in the same task must be mutually exclusive')
        labels.indicator_df = indicator_df
        labels.class_names = indicator_df.columns.values
        labels.encoded_df = pd.DataFrame(
            indicator_df.values.argmax(axis=1),
            columns=[task_name]
        )
        labels.task_name = task_name
        labels.n_samples = n_samples
        labels.n_classes = len(labels.class_names)
        if labels.n_classes == 1:
            labels.n_classes = 2  # 1 column means binary case
        labels.classes_ = np.arange(labels.n_classes)
        return labels

    def __nice__(labels):
        parts = []
        if labels.task_name is not None:
            parts.append(labels.task_name)
        parts.append('nD=%r' % (labels.n_samples))
        parts.append('nC=%r' % (labels.n_classes))
        return ' '.join(parts)

    def __len__(labels):
        return labels.n_samples

    def make_histogram(labels):
        class_idx_hist = ut.dict_hist(labels.y_enc)
        class_hist = ut.map_keys(
            lambda idx: labels.class_names[idx], class_idx_hist)
        return class_hist

    def print_info(labels):
        print('hist(%s) = %s' % (labels.task_name, ut.repr4(labels.make_histogram())))
        print('len(%s) = %s' % (labels.task_name, len(labels)))


@ut.reloadable_class
class ClfResult(object):
    """
    Handles evaluation statistics for a multiclass classifier trained on a
    specific dataset with specific labels.
    """
    def __init__(res):
        pass

    @classmethod
    def make_single(ClfResult, clf, X_df, test_idx, labels):
        """
        Make a result for a single cross validiation subset
        """
        X_test = X_df.values[test_idx]
        clf_probs = clf.predict_proba(X_test)
        index = pd.Series(test_idx, name='test_idx')
        # Ensure shape corresponds with all classes

        def align_cols(arr, arr_cols, target_cols):
            alignx = ut.list_alignment(arr_cols, target_cols, missing=True)
            aligned_arrT = ut.none_take(arr.T, alignx)
            aligned_arrT = ut.replace_nones(aligned_arrT, np.zeros(len(arr)))
            aligned_arr = np.vstack(aligned_arrT).T
            return aligned_arr

        res = ClfResult()
        res.class_names = ut.lmap(str, labels.class_names)
        res.probs_df = pd.DataFrame(
            align_cols(clf_probs, clf.classes_, labels.classes_), index=index,
            columns=['p_' + n for n in res.class_names],
        )
        res.target_bin_df = pd.DataFrame(
            data=labels.y_bin[test_idx], index=index,
            columns=['is_' + n for n in res.class_names],
        )
        res.target_enc_df = pd.DataFrame(
            data=labels.y_enc[test_idx], index=index,
            columns=['class_idx'],
        )

        if hasattr(clf, 'estimators_'):
            # The n-th estimator in the OVR classifier predicts the prob of the
            # n-th class (as label 1).
            probs_hat = np.hstack([est.predict_proba(X_test)[:, 1:2]
                                   for est in clf.estimators_])
            res.probhats_df = pd.DataFrame(
                align_cols(probs_hat, clf.classes_, labels.classes_), index=index,
                columns=['phat_' + n for n in res.class_names],
            )
            # In the OVR-case, ideally things will sum to 1, but when they
            # don't normalization happens. An Z-value of more than 1 means
            # overconfidence, and under 0 means underconfidence.
            res.confidence_ratio = res.probhats_df.sum(axis=1)
        else:
            res.probhats_df = None

        return res

    @classmethod
    def combine_results(ClfResult, res_list, labels=None):
        """
        Combine results from cross validation runs into a single result
        representing the performance of the entire dataset
        """
        # Ensure that res_lists are not overlapping
        idx_sets = [set(_res.probs_df.index.values) for _res in res_list]
        assert not any([s1.intersection(s2)
                        for s1, s2 in ut.combinations(idx_sets, 2)])
        # Combine them with pandas
        res = ClfResult()
        res0 = res_list[0]
        # res.samples = res0.samples
        res.class_names = res0.class_names
        # Combine all dataframe properties
        combo_df_attrs = [
            'probs_df',
            'probhats_df',
            'target_bin_df',
            'target_enc_df',
        ]
        for attr in combo_df_attrs:
            if getattr(res0, attr) is not None:
                combo_attr = pd.concat([getattr(r, attr) for r in res_list])
                setattr(res, attr, combo_attr)
        return res

    def make_meta(res, samples):
        """
        samples = self.samples
        """
        meta = {}
        meta['easiness'] = np.array(ut.ziptake(res.probs_df.values,
                                               res.target_enc_df.values)).ravel()
        meta['hardness'] = 1 - meta['easiness']
        meta['aid1'] = samples.aid_pairs.T[0].take(res.probs_df.index.values)
        meta['aid2'] = samples.aid_pairs.T[1].take(res.probs_df.index.values)
        meta['pred'] = res.probs_df.values.argmax(axis=1)
        meta['target'] = res.target_enc_df.values.ravel()
        meta['failed'] = meta['pred'] != meta['target']
        meta = pd.DataFrame(meta)
        res.meta = meta
        res.meta.take(res.meta['easiness'].argsort())
        return res.meta

    def missing_classes(res):
        # Find classes that were never predicted
        unique_predictions = np.unique(res.probs_df.values.argmax(axis=1))
        n_classes = len(res.class_names)
        missing_classes = ut.index_complement(unique_predictions, n_classes)
        return missing_classes

    def augment_if_needed(res):
        missing_classes = res.missing_classes()
        n_classes = len(res.class_names)
        y_test_enc_aug = res.target_enc_df.values
        y_test_bin_aug = res.target_bin_df.values
        clf_probs_aug = res.probs_df.values
        sample_weight = np.ones(len(y_test_enc_aug))
        n_missing = len(missing_classes)

        if res.probhats_df is not None:
            clf_probhats_aug = res.probhats_df.values
        else:
            clf_probhats_aug = None

        # Check if augmentation is necessary
        if n_missing > 0:
            missing_bin = np.zeros((n_missing, n_classes))
            missing_bin[(np.arange(n_missing), missing_classes)] = 1.0
            missing_enc = np.array(missing_classes)[:, None]
            y_test_enc_aug = np.vstack([y_test_enc_aug, missing_enc])
            y_test_bin_aug = np.vstack([y_test_bin_aug, missing_bin])
            clf_probs_aug = np.vstack([clf_probs_aug, missing_bin])
            # make sample weights where dummies are significantly downweighted
            sample_weight = np.hstack([sample_weight, np.full(n_missing, 1e-9)])

            if res.probhats_df is not None:
                clf_probhats_aug = np.vstack([clf_probhats_aug, missing_bin])

        res.clf_probs = clf_probs_aug
        res.clf_probhats = clf_probhats_aug
        res.y_test_enc = y_test_enc_aug
        res.y_test_bin = y_test_bin_aug
        res.sample_weight = sample_weight

    def extended_clf_report(res):
        res.augment_if_needed()
        pred_enc = res.clf_probs.argmax(axis=1)
        y_pred = pred_enc
        y_true = res.y_test_enc
        sample_weight = res.sample_weight
        confusion = cm = sklearn.metrics.confusion_matrix(  # NOQA
            y_true, y_pred, sample_weight=sample_weight)

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
        tpa = tpas.dot(rprob)
        tpr = tprs.dot(rprob)
        mk = mks.dot(rprob)
        bm = bms.dot(pprob)

        combined_data = ut.odict([
            ('precision', tpa),
            ('recall', tpr),
            ('markedness', mk),
            ('bookmaker', bm),
            ('mcc', np.sign(bm) * np.sqrt(np.abs(bm * mk))),
            ('support', real_total.sum())
        ])

        index = pd.Series(res.class_names, name='class')

        perclass_df = pd.DataFrame(perclass_data, index=index)
        combined_df = pd.DataFrame(combined_data, index=['ave/sum'])
        df = pd.concat([perclass_df, combined_df])

        pred_id = ['p(%s)' % m for m in res.class_names]
        real_id = ['r(%s)' % m for m in res.class_names]
        confusion_df = pd.DataFrame(confusion, columns=pred_id, index=real_id)
        confusion_df = confusion_df.append(pd.DataFrame([confusion.sum(axis=0)], columns=pred_id, index=['Σp']))
        confusion_df['Σr'] = np.hstack([confusion.sum(axis=1), ['-']])
        cfsm_str = confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))
        print('Confusion Matrix (real×pred) :')
        print(ut.hz_str('    ', cfsm_str))

        # ut.cprint('\nExtended Report', 'turquoise')
        print('\nEvaluation Metric Report:')
        precision = 2
        float_format = '%.' + str(precision) + 'f'
        ext_report = df.to_string(float_format=float_format)
        print(ut.hz_str('    ', ext_report))

        # FIXME: What is the difference between sklearn multiclass-MCC
        # and BM * MK MCC?
        mcc = sklearn.metrics.matthews_corrcoef(
            res.y_test_enc, pred_enc, sample_weight=res.sample_weight)
        # These scales are chosen somewhat arbitrarily in the context of a
        # computer vision application with relatively reasonable quality data
        mcc_significance_scales = ut.odict([
            (1.0, 'perfect'),
            (0.9, 'very strong'),
            (0.7, 'strong'),
            (0.5, 'significant'),
            (0.3, 'moderate'),
            (0.2, 'weak'),
            (0.0, 'negligible'),
        ])
        for k, v in mcc_significance_scales.items():
            if np.abs(mcc) >= k:
                print('classifier correlation is %s' % (v,))
                break
        print(('MCC\' = %.' + str(precision) + 'f') % (mcc,))

        # import utool
        # utool.embed()

    def print_report(res):
        res.augment_if_needed()
        pred_enc = res.clf_probs.argmax(axis=1)

        res.extended_clf_report()

        # p, r, f1, s = sklearn.metrics.precision_recall_fscore_support(
        #     y_true=res.y_test_enc, y_pred=pred_enc,
        #     sample_weight=res.sample_weight,
        # )

        report = sklearn.metrics.classification_report(
            y_true=res.y_test_enc, y_pred=pred_enc,
            target_names=res.class_names,
            sample_weight=res.sample_weight,
        )
        # confusion = sklearn.metrics.confusion_matrix(
        #     res.y_test_enc, pred_enc, sample_weight=res.sample_weight)
        # mcc = sklearn.metrics.matthews_corrcoef(
        #     res.y_test_enc, pred_enc, sample_weight=res.sample_weight)
        # pred_id = [m for m in res.class_names]
        # real_id = ['gt ' + m for m in res.class_names]
        # confusion_df = pd.DataFrame(confusion, columns=pred_id, index=real_id)
        # cfsm_str = confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))
        # print('MCC = %.4f' % (mcc,))
        print('Precision/Recall Report:')
        print(report)
        # print('Confusion Matrix:')
        # print(ut.hz_str('    ', cfsm_str))

    def report_thresholds(res):
        import vtool as vt
        y_test_bin = res.target_bin_df.values
        clf_probs = res.probs_df.values

        # The maximum allowed false positive rate
        # We expect that we will make 1 error every 1,000 decisions
        # thresh_df['foo'] = [1, 2, 3]
        # thresh_df['foo'][res.class_names[k]] = 1

        # for k in [2, 0, 1]:
        for k in range(y_test_bin.shape[1]):
            thresh_dict = ut.odict()
            class_name = res.class_names[k]
            probs, labels = clf_probs.T[k], y_test_bin.T[k]
            cfms = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)
            self = cfms  # NOQA

            encoder = vt.ScoreNormalizer()
            encoder.fit(probs, labels)
            maxsep_thresh = encoder.inverse_normalize(encoder.learn_threshold2()).tolist()

            threshes = ut.odict([
                # (class_name + '@tpr=1', cfms.get_thresh_at_metric('tpr', 1)),
                # (class_name + '@fpr=0', cfms.get_thresh_at_metric('fpr', 0)),
                (class_name + '@fpr=.01', cfms.get_thresh_at_metric('fpr', .01)),
                (class_name + '@fpr=.001', cfms.get_thresh_at_metric('fpr', .001)),
                # (class_name + '@fpr=.0001', cfms.get_thresh_at_metric('fpr', .0001)),
                (class_name + '@max(mcc)', cfms.get_thresh_at_metric_max('mcc')),
                (class_name + '@max(acc)', cfms.get_thresh_at_metric_max('acc')),
                # (class_name + '@max(mk)', cfms.get_thresh_at_metric_max('mk')),
                # (class_name + '@max(bm)', cfms.get_thresh_at_metric_max('bm')),
                (class_name + '@max(sep*)', maxsep_thresh),
            ])
            for key, thresh in threshes.items():
                thresh_dict[key] = ut.odict()
                thresh_dict[key]['thresh'] = thresh
                for metric in ['fpr', 'tpr', 'mcc', 'acc', 'ppv', 'bm', 'mk']:
                    thresh_dict[key][metric] = cfms.get_metric_at_threshold(metric, thresh)
            thresh_df = pd.DataFrame.from_dict(thresh_dict, orient='index')
            thresh_df = thresh_df.loc[list(threshes.keys())]
            print('\n')
            print('1vR Thresholds for ' + class_name)
            print(thresh_df.to_string(float_format=lambda x: '%.4f' % (x,)))
            # chosen_type = class_name + '@fpr=0'
            # pos_threshes[class_name] = thresh_df.loc[chosen_type]['thresh']

        pos_threshes = {}
        # neg_threshes = {}
        # What is the lowest threshold such that something
        for k in range(y_test_bin.shape[1]):
            class_name = res.class_names[k]
            probs, labels = clf_probs.T[k], y_test_bin.T[k]
            cfms = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)
            pos_threshes[class_name] = cfms.get_thresh_at_metric('fpr', 1E-4,
                                                                 prefer_max=False)
            # neg_threshes[class_name] = cfms.get_thresh_at_metric('fnr', 1E-1,
            #                                                      prefer_max=True)
            # neg_threshes[class_name] = cfms.get_thresh_at_metric('fnr', 1E-5,
            #                                                      prefer_max=True)
            # neg_threshes[class_name] = cfms.thresholds[np.where(cfms.tp > 0)[0][0]]
            # neg_threshes[class_name] = cfms.thresholds[np.where(cfms.fn > 0)[0][-1]]

        print('pos_threshes = %r' % (pos_threshes,))
        # print('neg_threshes = %r' % (neg_threshes,))
        # Actually we need negative pos_threshes?
        # So if ALL probs are under pos_threshes then its ok
        # See how many automated decisions can be made
        pos_ts = np.array(ut.take(pos_threshes, res.class_names))
        # TODO, choose neg thresholds
        neg_ts = np.array([.5] * len(res.class_names))

        above_pos_thresh = clf_probs > pos_ts[None, :]
        under_neg_thresh = clf_probs < neg_ts[None, :]
        # auto_chosen = clf_probs > pos_ts[None, :]
        # assert np.all(auto_chosen.sum(axis=1) <= 1)

        # Choose samples where all but one class is under the negative
        # threshold and that class is above a positive threshold
        can_autodecide = ((above_pos_thresh.sum(axis=1) > 0) &
                          (under_neg_thresh.sum(axis=1) >= len(res.class_names) - 1))
        print('Can make automated decisions on %d/%d = %.2f%% of the data' % (
            can_autodecide.sum(), len(can_autodecide),
            can_autodecide.sum() / len(can_autodecide)))

        auto_probs = clf_probs[can_autodecide]
        auto_truth_bin = y_test_bin[can_autodecide]
        auto_truth_enc = auto_truth_bin.argmax(axis=1)
        auto_pred_enc = auto_probs.argmax(axis=1)
        print('Autoclassify Confusion Matrix:\n')
        print(sklearn.metrics.confusion_matrix(auto_truth_enc, auto_pred_enc))
        print('Autoclassify MCC: ' + str(sklearn.metrics.matthews_corrcoef(auto_truth_enc, auto_pred_enc)))
        print('Autoclassify AUC(Macro): ' + str(sklearn.metrics.roc_auc_score(auto_truth_bin, auto_probs)))

        # print('hist of auto_truth labels' + str(ut.dict_hist(auto_pred_enc)))
        # thresh_df = pd.DataFrame.from_dict(thresh_dict, orient='columns')

        # # ut.qt4ensure()
        # # ROCInteraction = vt.interact_roc_factory(cfms, target_tpr,
        # #                                          show_operating_point=True)
        # # import plottool as pt
        # # fnum = pt.ensure_fnum(k)
        # # ROCInteraction.static_plot(fnum, None, name=str(k))
        # if False:
        #     X = probs
        #     y = labels
        #     encoder = vt.ScoreNormalizer()
        #     encoder.fit(probs, labels)
        #     learn_thresh = encoder.learn_threshold2()
        #     encoder.inverse_normalize(learn_thresh)
        # # encoder.visualize(fnum=k)
        # pass

    def roc_scores_ovr_hat(res):
        res.augment_if_needed()
        for k in range(len(res.class_names)):
            class_k_truth = res.y_test_bin.T[k]
            class_k_probs = res.probhats_df.values.T[k]
            auc = sklearn.metrics.roc_auc_score(class_k_truth, class_k_probs)
            yield auc

    def roc_scores_ovr(res):
        res.augment_if_needed()
        for k in range(res.y_test_bin.shape[1]):
            class_k_truth = res.y_test_bin.T[k]
            class_k_probs = res.clf_probs.T[k]
            auc = sklearn.metrics.roc_auc_score(class_k_truth, class_k_probs)
            yield auc

    def roc_score(res):
        res.augment_if_needed()
        auc_learn = sklearn.metrics.roc_auc_score(res.y_test_bin, res.clf_probs)
        return auc_learn


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts.samples
        python -m ibeis.scripts.samples --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
