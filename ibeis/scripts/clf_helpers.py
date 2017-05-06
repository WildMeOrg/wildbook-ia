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


def classification_report2(y_true, y_pred, target_names=None,
                                   sample_weight=None):
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

    if target_names is None:
        target_names = list(range(k))
    index = pd.Series(target_names, name='class')

    perclass_df = pd.DataFrame(perclass_data, index=index)
    combined_df = pd.DataFrame(combined_data, index=['ave/sum'])
    df = pd.concat([perclass_df, combined_df])

    pred_id = ['p(%s)' % m for m in target_names]
    real_id = ['r(%s)' % m for m in target_names]
    confusion_df = pd.DataFrame(confusion, columns=pred_id, index=real_id)
    confusion_df = confusion_df.append(pd.DataFrame(
        [confusion.sum(axis=0)], columns=pred_id, index=['Σp']))
    confusion_df['Σr'] = np.hstack([confusion.sum(axis=1), ['-']])
    cfsm_str = confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))
    print('Confusion Matrix (real × pred) :')
    print(ut.hz_str('    ', cfsm_str))

    # ut.cprint('\nExtended Report', 'turquoise')
    print('\nEvaluation Metric Report:')
    precision = 2
    float_format = '%.' + str(precision) + 'f'
    ext_report = df.to_string(float_format=float_format)
    print(ut.hz_str('    ', ext_report))


def predict_proba_df(clf, X_df, class_names=None):
    """
    Calls sklearn classifier predict_proba but then puts results in a dataframe
    using the same index as X_df and incorporating all possible class_names
    given
    """
    import utool
    if class_names is not None:
        columns = ut.take(class_names, clf.classes_)
    else:
        columns = None
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
                probs_df = probs_df.assign(**{
                    classname: np.zeros(len(probs_df))})
    return probs_df


def voting_ensemble(clf_list, voting='hard'):
    """
    hack to construct a VotingClassifier from pretrained classifiers
    TODO: contribute similar functionality to sklearn
    """
    estimators = [('clf%d' % count, clf) for count, clf in enumerate(clf_list)]
    eclf = sklearn.ensemble.VotingClassifier(estimators=estimators,
                                             voting=voting)
    assert ut.allsame(ut.list_getattr(clf_list, 'classes_'))
    eclf.classes_ = clf_list[0].classes_
    eclf.estimators_ = clf_list
    return eclf


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
    def __init__(samples, index):
        samples.index = index
        samples.subtasks = ut.odict()

    def apply_indicators(samples, tasks_to_indicators):
        samples.n_tasks = len(tasks_to_indicators)
        for task_name, indicator in tasks_to_indicators.items():
            labels = MultiClassLabels.from_indicators(
                indicator, task_name=task_name, index=samples.index)
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

    def __getitem__(samples, task_key):
        return samples.subtasks[task_key]

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

    def stratified_kfold_indices(samples, **xval_kw):
        """ TODO: check xval label frequency """
        skf = sklearn.model_selection.StratifiedKFold(**xval_kw)
        skf_iter = skf.split(X=np.empty((len(samples), 0)),
                             y=samples.encoded_1d())
        skf_list = list(skf_iter)
        return skf_list


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
        labels.n_samples = None
        labels.n_classes = None
        labels.class_names = None
        labels.classes_ = None
        # Core data
        labels.indicator_df = None
        labels.encoded_df = None

    def lookup_class_idx(labels, class_name):
        return ut.dzip(labels.class_names, labels.classes_)[class_name]

    @classmethod
    def from_indicators(MultiClassLabels, indicator, index=None, task_name=None):
        import six
        labels = MultiClassLabels()
        n_samples = len(six.next(six.itervalues(indicator)))
        # if index is None:
        #     index = pd.Series(np.arange(n_samples), name='index')
        indicator_df = pd.DataFrame(indicator, index=index)
        assert np.all(indicator_df.sum(axis=1).values), (
            'states in the same task must be mutually exclusive')
        labels.indicator_df = indicator_df
        labels.class_names = indicator_df.columns.values
        labels.encoded_df = pd.DataFrame(
            indicator_df.values.argmax(axis=1),
            columns=[task_name],
            index=index,
        )
        labels.task_name = task_name
        labels.n_samples = n_samples
        labels.n_classes = len(labels.class_names)
        if labels.n_classes == 1:
            labels.n_classes = 2  # 1 column means binary case
        labels.classes_ = np.arange(labels.n_classes)
        return labels

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
            index = labels.indicator_df.index
            indicator_df = pd.DataFrame()
            indicator_df['not-' + class_name] = 1 - labels.indicator_df[class_name]
            indicator_df[class_name] = labels.indicator_df[class_name]
            indicator_df.index = index
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
                columns=[task_name],
                index=index
            )
            sublabel.task_name = task_name
            sublabel.n_samples = n_samples
            sublabel.n_classes = len(sublabel.class_names)
            # if sublabel.n_classes == 1:
            #     sublabel.n_classes = 2  # 1 column means binary case
            sublabel.classes_ = np.arange(sublabel.n_classes)

            # sublabel = MultiClassLabels.from_indicators(indicator,
            # task_name=subname, index=samples.index)
            yield sublabel

    @property
    def y_bin(labels):
        return labels.indicator_df.values

    @property
    def y_enc(labels):
        return labels.encoded_df.values.ravel()

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
class ClfProblem(ut.NiceRepr):
    pass
    # TODO
    # def learn_single_clf
    def get_xval_kw(pblm):
        # xvalkw = dict(n_splits=10, shuffle=True,
        xval_kw = {
            # 'n_splits': 10,
            'n_splits': 2,
            'shuffle': True,
            'random_state': 3953056901,
        }
        return xval_kw

    def get_clf_params(pblm, clf_key):
        est_type = clf_key.split('-')[0]
        if est_type in {'RF', 'RandomForest'}:
            est_kw1 = {
                # 'max_depth': 4,
                'bootstrap': True,
                'class_weight': None,
                'max_features': 'sqrt',
                # 'max_features': None,
                'missing_values': np.nan,
                'min_samples_leaf': 5,
                'min_samples_split': 2,
                'n_estimators': 256,
                'criterion': 'entropy',
            }
            est_kw2 = {
                'random_state': 3915904814,
                'verbose': 0,
                'n_jobs': -1,
            }
        elif est_type in {'SVC', 'SVM'}:
            est_kw1 = dict(kernel='linear')
            est_kw2 = {}
        return est_kw1, est_kw2

    def get_clf_partial(pblm, clf_key):
        tup = clf_key.split('-')
        wrap_type = None if len(tup) == 1 else tup[1]
        est_type = tup[0]
        multiclass_wrapper = {
            None: ut.identity,
            'OVR': sklearn.multiclass.OneVsRestClassifier,
            'OVO': sklearn.multiclass.OneVsOneClassifier,
        }[wrap_type]
        est_class = {
            'RF': sklearn.ensemble.RandomForestClassifier,
            'SVC': sklearn.svm.SVC,
        }[est_type]

        est_kw1, est_kw2 = pblm.get_clf_params(est_type)
        est_params = ut.merge_dicts(est_kw1, est_kw2)

        def clf_partial():
            return multiclass_wrapper(est_class(**est_params))
        return clf_partial

    def set_pandas_options(pblm):
        # pd.options.display.max_rows = 10
        pd.options.display.max_rows = 20
        pd.options.display.max_columns = 40
        pd.options.display.width = 160
        pd.options.display.float_format = lambda x: '%.4f' % (x,)

    def set_pandas_options_low(pblm):
        # pd.options.display.max_rows = 10
        pd.options.display.max_rows = 5
        pd.options.display.max_columns = 40
        pd.options.display.width = 160
        pd.options.display.float_format = lambda x: '%.4f' % (x,)

    def set_pandas_options_normal(pblm):
        # pd.options.display.max_rows = 10
        pd.options.display.max_rows = 20
        pd.options.display.max_columns = 40
        pd.options.display.width = 160
        pd.options.display.float_format = lambda x: '%.4f' % (x,)

    def learn_evaluation_classifiers(pblm, task_keys=None, clf_keys=None,
                                     data_keys=None, cfg_prefix=''):
        """
        Evaluates by learning classifiers using cross validation.
        Do not use this to learn production classifiers.

        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --show
        """
        pblm.task_clfs = ut.AutoVivification()
        pblm.task_combo_res = ut.AutoVivification()

        if task_keys is None:
            task_keys = list(pblm.samples.subtasks.keys())
        if data_keys is None:
            task_keys = list(pblm.samples.X_dict.keys())
        if clf_keys is None:
            clf_keys = ['RF']

        Prog = ut.ProgPartial(freq=1, adjust=False, prehack='%s')
        task_prog = Prog(task_keys, label='Task')
        for task_key in task_prog:
            dataset_prog = Prog(data_keys, label='Data')
            for data_key in dataset_prog:
                clf_prog = Prog(clf_keys, label='CLF')
                for clf_key in clf_prog:
                    pblm._ensure_evaluation_clf(task_key, data_key, clf_key,
                                                cfg_prefix)

    def _ensure_evaluation_clf(pblm, task_key, data_key, clf_key, cfg_prefix):
        """
        Learns and caches an evaluation (cross-validated) classifier and tests
        and caches the results.
        """
        # TODO: add in params used to construct features into the cfgstr
        est_kw1, est_kw2 = pblm.get_clf_params(clf_key)
        xval_kw = pblm.get_xval_kw()
        param_id = ut.get_dict_hashid(est_kw1)
        xval_id = ut.get_dict_hashid(xval_kw)
        cfgstr = '_'.join([cfg_prefix, param_id, xval_id, task_key,
                           data_key, clf_key])

        cacher_kw = dict(appname='vsone_rf_train', enabled=1, verbose=1)
        cacher_clf = ut.Cacher('eval_clf_v13_0', cfgstr=cfgstr, **cacher_kw)
        cacher_res = ut.Cacher('eval_res_v13_0', cfgstr=cfgstr, **cacher_kw)

        clf_list = cacher_clf.tryload()
        if not clf_list:
            clf_list = pblm._train_evaluation_clf(task_key, data_key, clf_key)
            cacher_clf.save(clf_list)

        res_list = cacher_res.tryload()
        if not res_list:
            res_list = pblm._test_evaulation_clf(task_key, data_key, clf_list)
            cacher_res.save(res_list)

        labels = pblm.samples.subtasks[task_key]
        combo_res = ClfResult.combine_results(res_list, labels)
        pblm.task_clfs[task_key][clf_key][data_key] = clf_list
        pblm.task_combo_res[task_key][clf_key][data_key] = combo_res

    def learn_deploy_classifiers(pblm, task_keys=None, clf_key=None,
                                 data_key=None):
        """
        Learns on data without any train/validation split
        """
        if pblm.verbose > 0:
            print('[pblm] learn_deploy_classifiers')
        if clf_key is None:
            clf_key = pblm.default_clf_key
        if data_key is None:
            data_key = pblm.default_data_key
        if task_keys is None:
            task_keys = list(pblm.samples.subtasks.keys())

        Prog = ut.ProgPartial(freq=1, adjust=False, prehack='%s')
        task_prog = Prog(task_keys, label='Task')
        deploy_task_clfs = {}
        for task_key in task_prog:
            clf = pblm._train_deploy_clf(task_key, data_key, clf_key)
            deploy_task_clfs[task_key] = clf
        pblm.deploy_task_clfs = deploy_task_clfs
        return deploy_task_clfs

    def _train_deploy_clf(pblm, task_key, data_key, clf_key):
        X_df = pblm.samples.X_dict[data_key]
        labels = pblm.samples.subtasks[task_key]
        assert np.all(labels.encoded_df.index == X_df.index)
        clf_partial = pblm.get_clf_partial(clf_key)
        print('Training deployment {} classifier on {} for {}'.format(
            clf_key, data_key, task_key))
        clf = clf_partial()
        index = X_df.index
        X = X_df.loc[index].values
        y = labels.encoded_df.loc[index].values
        clf.fit(X, y)
        return clf

    def _train_evaluation_clf(pblm, task_key, data_key, clf_key):
        """
        Learns a cross-validated classifier on the dataset

            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem()
            >>> pblm.load_features()
            >>> pblm.load_samples()
            >>> data_key = 'learn(all)'
            >>> task_key = 'photobomb_state'
            >>> task_key = 'match_state'
            >>> clf_key = 'RF-OVR'
            >>> clf_key = 'RF'
        """
        X_df = pblm.samples.X_dict[data_key]
        labels = pblm.samples.subtasks[task_key]
        assert np.all(labels.encoded_df.index == X_df.index)

        clf_partial = pblm.get_clf_partial(clf_key)

        xval_kw = pblm.get_xval_kw()

        clf_list = []
        skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)
        skf_prog = ut.ProgIter(skf_list, label='skf-learn')
        for train_idx, test_idx in skf_prog:
            assert (X_df.iloc[train_idx].index.tolist() ==
                    ut.take(pblm.samples.index, train_idx))
            # train_uv = X_df.iloc[train_idx].index
            # X_train = X_df.loc[train_uv]
            # y_train = labels.encoded_df.loc[train_uv]
            X_train = X_df.iloc[train_idx].values
            y_train = labels.encoded_df.iloc[train_idx].values
            clf = clf_partial()
            clf.fit(X_train, y_train)
            clf_list.append(clf)
        return clf_list

    def _test_evaulation_clf(pblm, task_key, data_key, clf_list):
        """ Test a cross-validated classifier on the dataset """
        X_df = pblm.samples.X_dict[data_key]
        labels = pblm.samples.subtasks[task_key]
        xval_kw = pblm.get_xval_kw()

        res_list = []
        skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)
        skf_prog = ut.ProgIter(zip(clf_list, skf_list), length=len(skf_list),
                               label='skf-test')
        for clf, (train_idx, test_idx) in skf_prog:
            res = ClfResult.make_single(clf, X_df, test_idx, labels, data_key)
            res_list.append(res)
        return res_list


@ut.reloadable_class
class ClfResult(ut.NiceRepr):
    r"""
    Handles evaluation statistics for a multiclass classifier trained on a
    specific dataset with specific labels.
    """

    # Attributes that identify the task and data the classifier is evaluated on
    _key_attrs = ['task_key', 'data_key', 'class_names']

    # Attributes about results and labels of individual samples
    _datafame_attrs = ['probs_df', 'probhats_df', 'target_bin_df',
                       'target_enc_df']

    def __init__(res):
        pass

    def __nice__(res):
        return '%s, %s' % (res.task_key, res.data_key)

    @property
    def index(res):
        return res.probs_df.index

    @classmethod
    def make_single(ClfResult, clf, X_df, test_idx, labels, data_key):
        """
        Make a result for a single cross validiation subset
        """
        X_df_test = X_df.iloc[test_idx]
        index = X_df_test.index
        # clf_probs = clf.predict_proba(X_df_test)

        # index = pd.Series(test_idx, name='test_idx')
        # Ensure shape corresponds with all classes

        def align_cols(arr, arr_cols, target_cols):
            import utool as ut
            alignx = ut.list_alignment(arr_cols, target_cols, missing=True)
            aligned_arrT = ut.none_take(arr.T, alignx)
            aligned_arrT = ut.replace_nones(aligned_arrT, np.zeros(len(arr)))
            aligned_arr = np.vstack(aligned_arrT).T
            return aligned_arr

        res = ClfResult()
        res.task_key = labels.task_name
        res.data_key = data_key
        res.class_names = ut.lmap(str, labels.class_names)

        # res.probs_df = pd.DataFrame(
        #     align_cols(clf_probs, clf.classes_, labels.classes_), index=index,
        #     columns=['p_' + n for n in res.class_names],
        # )
        res.probs_df = predict_proba_df(clf, X_df_test, res.class_names)
        res.target_bin_df = labels.indicator_df.iloc[test_idx]
        res.target_enc_df = labels.encoded_df.iloc[test_idx]

        if hasattr(clf, 'estimators_') and labels.n_classes > 2:
            # The n-th estimator in the OVR classifier predicts the prob of the
            # n-th class (as label 1).
            probs_hat = np.hstack([est.predict_proba(X_df_test)[:, 1:2]
                                   for est in clf.estimators_])
            res.probhats_df = pd.DataFrame(
                align_cols(probs_hat, clf.classes_, labels.classes_),
                index=index, columns=res.class_names,
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
        for r1, r2 in ut.combinations(res_list, 2):
            assert len(r1.index.intersection(r2.index)) == 0, (
                'ClfResult dataframes must be disjoint')
        # sanity check
        for r in res_list:
            assert np.all(r.index == r.probs_df.index)
            assert np.all(r.index == r.target_bin_df.index)
            assert np.all(r.index == r.target_enc_df.index)

        # Combine them with pandas
        res = ClfResult()
        res0 = res_list[0]
        # Transfer single attributes (which should all be the same)
        for attr in ClfResult._key_attrs:
            val = getattr(res0, attr)
            setattr(res, attr, val)
            assert all([getattr(r, attr) == val for r in res_list]), (
                'ClfResult with different key attributes are incompatible')
        # Combine dataframe properties (which should all have disjoint indices)
        for attr in ClfResult._datafame_attrs:
            if getattr(res0, attr) is not None:
                combo_attr = pd.concat([getattr(r, attr) for r in res_list])
                setattr(res, attr, combo_attr)
            else:
                setattr(res, attr, None)

        for attr in ClfResult._datafame_attrs:
            val = getattr(res, attr)
            if val is not None:
                assert np.all(res.index == val.index), 'index got weird'

        return res

    def make_meta(res, samples):
        """
        samples = pblm.samples
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
        target_names = res.class_names
        classification_report2(y_true, y_pred, target_names=target_names,
                               sample_weight=sample_weight)

        # confusion = cm = sklearn.metrics.confusion_matrix(  # NOQA
        #     y_true, y_pred, sample_weight=sample_weight)

        # k = len(cm)
        # N = cm.sum()

        # real_total = cm.sum(axis=1)
        # pred_total = cm.sum(axis=0)

        # n_tps = np.diag(cm)
        # tprs = n_tps / real_total
        # tpas = n_tps / pred_total

        # rprob = real_total / N
        # pprob = pred_total / N

        # # bookmaker is analogous to recall
        # rprob_mat = np.tile(rprob, [k, 1]).T - (1 - np.eye(k))
        # bmcm = cm.T / rprob_mat
        # bms = np.sum(bmcm.T, axis=0) / N

        # # markedness is analogous to precision
        # pprob_mat = np.tile(pprob, [k, 1]).T - (1 - np.eye(k))
        # mkcm = cm / pprob_mat
        # mks = np.sum(mkcm.T, axis=0) / N

        # perclass_data = ut.odict([
        #     ('precision', tpas),
        #     ('recall', tprs),
        #     ('markedness', mks),
        #     ('bookmaker', bms),
        #     ('mcc', np.sign(bms) * np.sqrt(np.abs(bms * mks))),
        #     ('support', real_total),
        # ])
        # tpa = tpas.dot(rprob)
        # tpr = tprs.dot(rprob)
        # mk = mks.dot(rprob)
        # bm = bms.dot(pprob)

        # combined_data = ut.odict([
        #     ('precision', tpa),
        #     ('recall', tpr),
        #     ('markedness', mk),
        #     ('bookmaker', bm),
        #     ('mcc', np.sign(bm) * np.sqrt(np.abs(bm * mk))),
        #     ('support', real_total.sum())
        # ])

        # index = pd.Series(res.class_names, name='class')

        # perclass_df = pd.DataFrame(perclass_data, index=index)
        # combined_df = pd.DataFrame(combined_data, index=['ave/sum'])
        # df = pd.concat([perclass_df, combined_df])

        # pred_id = ['p(%s)' % m for m in res.class_names]
        # real_id = ['r(%s)' % m for m in res.class_names]
        # confusion_df = pd.DataFrame(confusion, columns=pred_id, index=real_id)
        # confusion_df = confusion_df.append(pd.DataFrame([confusion.sum(axis=0)], columns=pred_id, index=['Σp']))
        # confusion_df['Σr'] = np.hstack([confusion.sum(axis=1), ['-']])
        # cfsm_str = confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))
        # print('Confusion Matrix (real × pred) :')
        # print(ut.hz_str('    ', cfsm_str))

        # # ut.cprint('\nExtended Report', 'turquoise')
        # print('\nEvaluation Metric Report:')
        precision = 2
        # float_format = '%.' + str(precision) + 'f'
        # ext_report = df.to_string(float_format=float_format)
        # print(ut.hz_str('    ', ext_report))

        # FIXME: What is the difference between sklearn multiclass-MCC
        # and BM * MK MCC?
        try:
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
        except ValueError:
            pass

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

    @profile
    def get_pos_threshes(res, metric='fpr', value=1E-4, prefer_max=False):
        import vtool as vt
        y_test_bin = res.target_bin_df.values
        clf_probs = res.probs_df.values
        pos_threshes = {}
        for k in range(y_test_bin.shape[1]):
            class_name = res.class_names[k]
            probs, labels = clf_probs.T[k], y_test_bin.T[k]
            cfms = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)
            pos_threshes[class_name] = cfms.get_thresh_at_metric(
                metric, value, prefer_max=prefer_max)
        return pos_threshes

    def report_thresholds(res):
        import vtool as vt
        y_test_bin = res.target_bin_df.values
        # y_test_enc = y_test_bin.argmax(axis=1)
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

            # encoder = vt.ScoreNormalizer()
            # encoder.fit(probs, labels)
            # maxsep_thresh = encoder.inverse_normalize(encoder.learn_threshold2()).tolist()

            threshes = ut.odict([
                # (class_name + '@tpr=1', cfms.get_thresh_at_metric('tpr', 1)),
                # (class_name + '@fpr=0', cfms.get_thresh_at_metric('fpr', 0)),
                (class_name + '@fpr=.01', cfms.get_thresh_at_metric('fpr', .01)),
                (class_name + '@fpr=.001', cfms.get_thresh_at_metric('fpr', 1E-4)),
                # (class_name + '@fpr=.0001', cfms.get_thresh_at_metric('fpr', .0001)),
                # (class_name + '@max(mcc)', cfms.get_thresh_at_metric_max('mcc')),
                # (class_name + '@max(acc)', cfms.get_thresh_at_metric_max('acc')),
                # (class_name + '@max(mk)', cfms.get_thresh_at_metric_max('mk')),
                # (class_name + '@max(bm)', cfms.get_thresh_at_metric_max('bm')),
                # (class_name + '@max(sep*)', maxsep_thresh),
            ])
            for key, thresh in threshes.items():
                thresh_dict[key] = ut.odict()
                thresh_dict[key]['thresh'] = thresh
                for metric in ['fpr', 'tpr', 'tpa', 'bm', 'mk', 'mcc']:
                    thresh_dict[key][metric] = cfms.get_metric_at_threshold(metric, thresh)
            thresh_df = pd.DataFrame.from_dict(thresh_dict, orient='index')
            thresh_df = thresh_df.loc[list(threshes.keys())]
            print('\n1vR Thresholds for ' + class_name)
            print(thresh_df.to_string(float_format=lambda x: '%.4f' % (x,)))
            # chosen_type = class_name + '@fpr=0'
            # pos_threshes[class_name] = thresh_df.loc[chosen_type]['thresh']

        pos_threshes = res.get_pos_threshes()

        print('pos_threshes = %s' % (ut.repr2(pos_threshes, precision=4),))
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

        perclass_autodecide_num_total = np.array([
            (can_autodecide[y_test_bin.T[k]].sum(), y_test_bin.T[k].sum())
            for k in range(y_test_bin.shape[1])
        ])
        num, total = perclass_autodecide_num_total.sum(axis=0)
        # TODO: put this in context of how true/false predictions
        print('Auto %r thresholds passed by %d/%d = %.2f%%' % (
            res.task_key, num, total, num / total))
        for k in range(y_test_bin.shape[1]):
            num, total = perclass_autodecide_num_total[k]
            print(' * %d/%d = %.2f%% of class %r' % (
                num, total, num / total, res.class_names[k]))

        auto_probs = clf_probs[can_autodecide]
        auto_truth_bin = y_test_bin[can_autodecide]
        auto_truth_enc = auto_truth_bin.argmax(axis=1)

        class_xs, groupxs = vt.group_indices(auto_truth_enc)

        auto_pred_enc = auto_probs.argmax(axis=1)
        print('Autoclassify Confusion Matrix:\n')
        print(sklearn.metrics.confusion_matrix(auto_truth_enc, auto_pred_enc))
        try:
            print('Autoclassify MCC: ' + str(sklearn.metrics.matthews_corrcoef(auto_truth_enc, auto_pred_enc)))
        except ValueError:
            pass
        print('Autoclassify AUC(Macro): ' + str(sklearn.metrics.roc_auc_score(auto_truth_bin, auto_probs)))
        # return pos_threshes

        # print('hist of auto_truth labels' + str(ut.dict_hist(auto_pred_enc)))
        # thresh_df = pd.DataFrame.from_dict(thresh_dict, orient='columns')

    def ishow_roc(res):
        import vtool as vt
        import plottool as pt
        y_test_bin = res.target_bin_df.values
        clf_probs = res.probs_df.values
        ut.qtensure()

        # The maximum allowed false positive rate
        # We expect that we will make 1 error every 1,000 decisions
        # thresh_df['foo'] = [1, 2, 3]
        # thresh_df['foo'][res.class_names[k]] = 1

        # for k in [2, 0, 1]:
        for k in range(y_test_bin.shape[1]):
            if y_test_bin.shape[1] == 2 and k == 0:
                # only show one in the binary case
                continue
            class_name = res.class_names[k]
            probs, labels = clf_probs.T[k], y_test_bin.T[k]
            confusions = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)

            ROCInteraction = vt.interact_roc_factory(confusions,
                                                     show_operating_point=True)
            fnum = pt.ensure_fnum(k)
            # ROCInteraction.static_plot(fnum, None, name=class_name)
            inter = ROCInteraction(fnum=fnum, pnum=None, name=class_name)
            inter.start()
        # if False:
        #     X = probs
        #     y = labels
        #     encoder = vt.ScoreNormalizer()
        #     encoder.fit(probs, labels)
        #     learn_thresh = encoder.learn_threshold2()
        #     encoder.inverse_normalize(learn_thresh)
        # encoder.visualize(fnum=k)
        pass

    def show_roc(res, class_name, **kwargs):
        import vtool as vt
        labels = res.target_bin_df[class_name].values
        probs = res.probs_df[class_name].values
        confusions = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)
        confusions.draw_roc_curve(**kwargs)

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
