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
    def target_type(samples):
        return sklearn.utils.multiclass.type_of_target(samples.y_enc)

    def gen_one_vs_rest_labels(samples):
        if samples.target_type == 'binary':
            yield samples
            raise StopIteration()
        for k in range(samples.n_classes):
            class_name = samples.class_names[k]
            task_name = samples.task_name + '(' + class_name + '-v-rest)'
            indicator_df = samples.indicator_df[[class_name]]
            # indicator = labels.encoded_df == k
            # indicator.rename(columns={indicator.columns[0]: class_name}, inplace=True)
            n_samples = len(indicator_df)
            sublabel = MultiClassLabels()
            sublabel.indicator_df = indicator_df
            sublabel.class_names = indicator_df.columns.values
            if len(indicator_df.columns) == 1:
                sublabel.encoded_df = pd.DataFrame(
                    indicator_df.values.T[0],
                    columns=[task_name]
                )
            else:
                sublabel.encoded_df = pd.DataFrame(
                    indicator_df.values.argmax(axis=1),
                    columns=[task_name]
                )
            sublabel.task_name = task_name
            sublabel.n_samples = n_samples
            sublabel.n_classes = len(sublabel.class_names)
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


@ut.reloadable_class
class ClfResult(object):
    """
    cls = ClfResult

    TODO: use Markedness and Informedness
    http://www.flinders.edu.au/science_engineering/fms/School-CSEM/publications/tech_reps-research_artfcts/TRRA_2007.pdf
    """
    def __init__(res):
        pass

    @classmethod
    def make_single(ClfResult, test_idx, clf_probs, pred_classes, labels):
        """
        Make a result for a single cross validiation subset
        """
        res = ClfResult()

        # Ensure shape corresponds with all classes
        alignx = ut.list_alignment(pred_classes, labels.classes_, missing=True)
        aligned_probs_ = ut.none_take(clf_probs.T, alignx)
        aligned_probs_ = ut.replace_nones(aligned_probs_, np.zeros(len(clf_probs)))
        aligned_probs = np.vstack(aligned_probs_).T

        class_names = ut.lmap(str, labels.class_names)
        res.class_names = class_names
        index = pd.Series(test_idx, name='test_idx')

        res.probs_df = pd.DataFrame(
            aligned_probs, index=index,
            columns=['p_' + n for n in class_names],
        )
        res.target_bin_df = pd.DataFrame(
            data=labels.y_bin[test_idx], index=index,
            columns=['is_' + n for n in class_names],
        )
        res.target_enc_df = pd.DataFrame(
            data=labels.y_enc[test_idx], index=index,
            columns=['class_idx'],
        )
        return res

    @classmethod
    def combine_results(cls, res_list, labels=None):
        """
        Combine results from cross validation runs into a single result
        representing the performance of the entire dataset
        """
        # Ensure that res_lists are not overlapping
        idx_sets = [set(_res.probs_df.index.values) for _res in res_list]
        assert not any([s1.intersection(s2)
                        for s1, s2 in ut.combinations(idx_sets, 2)])
        # Combine them with pandas
        res = cls()
        res0 = res_list[0]
        # res.samples = res0.samples
        res.class_names = res0.class_names
        res.probs_df = pd.concat([r.probs_df for r in res_list])
        res.target_bin_df = pd.concat([r.target_bin_df for r in res_list])
        res.target_enc_df = pd.concat([r.target_enc_df for r in res_list])

        return res

    def make_meta(res, samples):
        """
        samples = self.samples
        """
        meta = {}
        meta['easiness'] = np.array(ut.ziptake(res.probs_df.values, res.target_enc_df.values)).ravel()
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
        return y_test_enc_aug, y_test_bin_aug, clf_probs_aug, sample_weight

    def print_report(res):
        (y_test_enc_aug, y_test_bin_aug,
         clf_probs_aug, sample_weight) = res.augment_if_needed()

        pred_enc = clf_probs_aug.argmax(axis=1)

        p, r, f1, s = sklearn.metrics.precision_recall_fscore_support(
            y_true=y_test_enc_aug, y_pred=pred_enc,
            sample_weight=sample_weight,
        )

        # invp, invr, _, _ = sklearn.metrics.precision_recall_fscore_support(
        #     y_true=1 - y_test_enc_aug, y_pred=1 - pred_enc,
        #     sample_weight=sample_weight,
        # )
        report = sklearn.metrics.classification_report(
            y_true=y_test_enc_aug, y_pred=pred_enc,
            target_names=res.class_names,
            sample_weight=sample_weight,
        )
        confusion = sklearn.metrics.confusion_matrix(y_test_enc_aug, pred_enc,
                                                     sample_weight=sample_weight)

        mcc = sklearn.metrics.matthews_corrcoef(y_test_enc_aug, pred_enc,
                                                sample_weight=sample_weight)
        print('MCC = %.4f' % (mcc,))
        print('Confusion Matrix:')
        confusion_df = pd.DataFrame(confusion, columns=[m for m in res.class_names],
                                    index=['gt ' + m for m in res.class_names])
        print(ut.hz_str('    ', confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))))
        print('Precision/Recall Report:')
        print(report)

    def roc_score(res):
        (y_test_enc_aug, y_test_bin_aug,
         clf_probs_aug, sample_weight) = res.augment_if_needed()
        auc_learn = sklearn.metrics.roc_auc_score(y_test_bin_aug, clf_probs_aug)
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
