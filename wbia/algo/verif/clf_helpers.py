# -*- coding: utf-8 -*-
"""
This module is a work in progress, as such concepts are subject to change.

MAIN IDEA:
    `MultiTaskSamples` serves as a structure to contain and manipulate a set of
    samples with potentially many different types of labels and features.
"""
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import ubelt as ub
import numpy as np
from wbia import dtool as dt
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.ensemble
import sklearn.pipeline
import sklearn.neural_network
from wbia.algo.verif import sklearn_utils
from six.moves import range

print, rrr, profile = ut.inject2(__name__)


class XValConfig(dt.Config):
    _param_info_list = [
        # ut.ParamInfo('type', 'StratifiedKFold'),
        ut.ParamInfo('type', 'StratifiedGroupKFold'),
        ut.ParamInfo('n_splits', 3),
        ut.ParamInfo(
            'shuffle', True, hideif=lambda cfg: cfg['type'] == 'StratifiedGroupKFold'
        ),
        ut.ParamInfo(
            'random_state',
            3953056901,
            hideif=lambda cfg: cfg['type'] == 'StratifiedGroupKFold',
        ),
    ]


@ut.reloadable_class
class ClfProblem(ut.NiceRepr):
    def __init__(pblm):
        pblm.deploy_task_clfs = None
        pblm.eval_task_clfs = None
        pblm.xval_kw = XValConfig()
        pblm.eval_task_clfs = None
        pblm.task_combo_res = None
        pblm.verbose = True

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

    def learn_evaluation_classifiers(pblm, task_keys=None, clf_keys=None, data_keys=None):
        """
        Evaluates by learning classifiers using cross validation.
        Do not use this to learn production classifiers.

        python -m wbia.algo.verif.vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --show

        Example:

        CommandLine:
            python -m clf_helpers learn_evaluation_classifiers

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.verif.clf_helpers import *  # NOQA
            >>> pblm = IrisProblem()
            >>> pblm.setup()
            >>> pblm.verbose = True
            >>> pblm.eval_clf_keys = ['Logit', 'RF']
            >>> pblm.eval_task_keys = ['iris']
            >>> pblm.eval_data_keys = ['learn(all)']
            >>> result = pblm.learn_evaluation_classifiers()
            >>> res = pblm.task_combo_res['iris']['Logit']['learn(all)']
            >>> res.print_report()
            >>> res = pblm.task_combo_res['iris']['RF']['learn(all)']
            >>> res.print_report()
            >>> print(result)
        """
        pblm.eval_task_clfs = ut.AutoVivification()
        pblm.task_combo_res = ut.AutoVivification()

        if task_keys is None:
            task_keys = pblm.eval_task_keys
        if data_keys is None:
            data_keys = pblm.eval_data_keys
        if clf_keys is None:
            clf_keys = pblm.eval_clf_keys

        if task_keys is None:
            task_keys = [pblm.primary_task_key]
        if data_keys is None:
            data_keys = [pblm.default_data_key]
        if clf_keys is None:
            clf_keys = [pblm.default_clf_key]

        if pblm.verbose:
            ut.cprint('[pblm] learn_evaluation_classifiers', color='blue')
            ut.cprint('[pblm] task_keys = {}'.format(task_keys))
            ut.cprint('[pblm] data_keys = {}'.format(data_keys))
            ut.cprint('[pblm] clf_keys = {}'.format(clf_keys))

        Prog = ut.ProgPartial(freq=1, adjust=False, prehack='%s')
        task_prog = Prog(task_keys, label='Task')
        for task_key in task_prog:
            dataset_prog = Prog(data_keys, label='Data')
            for data_key in dataset_prog:
                clf_prog = Prog(clf_keys, label='CLF')
                for clf_key in clf_prog:
                    pblm._ensure_evaluation_clf(task_key, data_key, clf_key)

    def _ensure_evaluation_clf(pblm, task_key, data_key, clf_key, use_cache=True):
        """
        Learns and caches an evaluation (cross-validated) classifier and tests
        and caches the results.

        data_key = 'learn(sum,glob)'
        clf_key = 'RF'
        """
        # TODO: add in params used to construct features into the cfgstr
        if hasattr(pblm.samples, 'sample_hashid'):
            ibs = pblm.infr.ibs
            sample_hashid = pblm.samples.sample_hashid()

            feat_dims = pblm.samples.X_dict[data_key].columns.values.tolist()
            # cfg_prefix = sample_hashid + pblm.qreq_.get_cfgstr() + feat_cfgstr

            est_kw1, est_kw2 = pblm._estimator_params(clf_key)
            param_id = ut.get_dict_hashid(est_kw1)
            xval_id = pblm.xval_kw.get_cfgstr()
            cfgstr = '_'.join(
                [
                    sample_hashid,
                    param_id,
                    xval_id,
                    task_key,
                    data_key,
                    clf_key,
                    ut.hashid_arr(feat_dims, 'feats'),
                ]
            )
            fname = 'eval_clfres_' + ibs.dbname
        else:
            fname = 'foo'
            feat_dims = None
            cfgstr = 'bar'
            use_cache = False

        # TODO: ABI class should not be caching
        cacher_kw = dict(appname='vsone_rf_train', enabled=use_cache, verbose=1)
        cacher_clf = ub.Cacher(fname, cfgstr=cfgstr, meta=[feat_dims], **cacher_kw)

        data = cacher_clf.tryload()
        if not data:
            data = pblm._train_evaluation_clf(task_key, data_key, clf_key)
            cacher_clf.save(data)
        clf_list, res_list = data

        labels = pblm.samples.subtasks[task_key]
        combo_res = ClfResult.combine_results(res_list, labels)
        pblm.eval_task_clfs[task_key][clf_key][data_key] = clf_list
        pblm.task_combo_res[task_key][clf_key][data_key] = combo_res

    def _train_evaluation_clf(pblm, task_key, data_key, clf_key, feat_dims=None):
        """
        Learns a cross-validated classifier on the dataset

        Ignore:
            >>> from wbia.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem()
            >>> pblm.load_features()
            >>> pblm.load_samples()
            >>> data_key = 'learn(all)'
            >>> task_key = 'photobomb_state'
            >>> clf_key = 'RF-OVR'
            >>> task_key = 'match_state'
            >>> data_key = pblm.default_data_key
            >>> clf_key = pblm.default_clf_key
        """
        X_df = pblm.samples.X_dict[data_key]
        labels = pblm.samples.subtasks[task_key]
        assert np.all(labels.encoded_df.index == X_df.index)

        clf_partial = pblm._get_estimator(clf_key)
        xval_kw = pblm.xval_kw.asdict()

        clf_list = []
        res_list = []
        skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)
        skf_prog = ut.ProgIter(skf_list, label='skf-train-eval')
        for train_idx, test_idx in skf_prog:
            X_df_train = X_df.iloc[train_idx]
            assert X_df_train.index.tolist() == ut.take(pblm.samples.index, train_idx)
            # train_uv = X_df.iloc[train_idx].index
            # X_train = X_df.loc[train_uv]
            # y_train = labels.encoded_df.loc[train_uv]

            if feat_dims is not None:
                X_df_train = X_df_train[feat_dims]

            X_train = X_df_train.values
            y_train = labels.encoded_df.iloc[train_idx].values.ravel()

            clf = clf_partial()
            clf.fit(X_train, y_train)

            # Note: There is a corner case where one fold doesn't get any
            # labels of a certain class. Because y_train is an encoded integer,
            # the clf.classes_ attribute will cause predictions to agree with
            # other classifiers trained on the same labels.

            # Evaluate results
            res = ClfResult.make_single(
                clf, X_df, test_idx, labels, data_key, feat_dims=feat_dims
            )
            clf_list.append(clf)
            res_list.append(res)
        return clf_list, res_list

    def _external_classifier_result(
        pblm, clf, task_key, data_key, feat_dims=None, test_idx=None
    ):
        """
        Given an external classifier (ensure its trained on disjoint data)
        evaluate all data on it.

        Args:
            test_idx (list): subset of this classifier to test on
                (defaults to all if None)
        """
        X_df = pblm.samples.X_dict[data_key]
        if test_idx is None:
            test_idx = np.arange(len(X_df))
        labels = pblm.samples.subtasks[task_key]

        res = ClfResult.make_single(
            clf, X_df, test_idx, labels, data_key, feat_dims=feat_dims
        )

        return res

    def learn_deploy_classifiers(pblm, task_keys=None, clf_key=None, data_key=None):
        """
        Learns on data without any train/validation split
        """
        if pblm.verbose > 0:
            ut.cprint('[pblm] learn_deploy_classifiers', color='blue')
        if clf_key is None:
            clf_key = pblm.default_clf_key
        if data_key is None:
            data_key = pblm.default_data_key
        if task_keys is None:
            task_keys = list(pblm.samples.supported_tasks())

        if pblm.deploy_task_clfs is None:
            pblm.deploy_task_clfs = ut.AutoVivification()

        Prog = ut.ProgPartial(freq=1, adjust=False, prehack='%s')
        task_prog = Prog(task_keys, label='Task')
        task_clfs = {}
        for task_key in task_prog:
            clf = pblm._train_deploy_clf(task_key, data_key, clf_key)
            task_clfs[task_key] = clf
            pblm.deploy_task_clfs[task_key][clf_key][data_key] = clf

        return task_clfs

    def _estimator_params(pblm, clf_key):
        est_type = clf_key.split('-')[0]
        if est_type in {'RF', 'RandomForest'}:
            est_kw1 = {
                # 'max_depth': 4,
                'bootstrap': True,
                'class_weight': None,
                'criterion': 'entropy',
                'max_features': 'sqrt',
                # 'max_features': None,
                'min_samples_leaf': 5,
                'min_samples_split': 2,
                # 'n_estimators': 64,
                'n_estimators': 256,
            }
            # Hack to only use missing values if we have the right sklearn
            if 'missing_values' in ut.get_func_kwargs(
                sklearn.ensemble.RandomForestClassifier.__init__
            ):
                est_kw1['missing_values'] = np.nan
            est_kw2 = {
                'random_state': 3915904814,
                'verbose': 0,
                'n_jobs': -1,
            }
        elif est_type in {'SVC', 'SVM'}:
            est_kw1 = dict(kernel='linear')
            est_kw2 = {}
        elif est_type in {'Logit', 'LogisticRegression'}:
            est_kw1 = {}
            est_kw2 = {}
        elif est_type in {'MLP'}:
            est_kw1 = dict(
                activation='relu',
                alpha=1e-05,
                batch_size='auto',
                beta_1=0.9,
                beta_2=0.999,
                early_stopping=False,
                epsilon=1e-08,
                hidden_layer_sizes=(10, 10),
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=200,
                momentum=0.9,
                nesterovs_momentum=True,
                power_t=0.5,
                random_state=3915904814,
                shuffle=True,
                solver='lbfgs',
                tol=0.0001,
                validation_fraction=0.1,
                warm_start=False,
            )
            est_kw2 = dict(verbose=False)
        else:
            raise KeyError('Unknown Estimator')
        return est_kw1, est_kw2

    def _get_estimator(pblm, clf_key):
        """
        Returns sklearn classifier
        """
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
            'Logit': sklearn.linear_model.LogisticRegression,
            'MLP': sklearn.neural_network.MLPClassifier,
        }[est_type]

        est_kw1, est_kw2 = pblm._estimator_params(est_type)
        est_params = ut.merge_dicts(est_kw1, est_kw2)

        # steps = []
        # steps.append((est_type, est_class(**est_params)))
        # if wrap_type is not None:
        #     steps.append((wrap_type, multiclass_wrapper))
        if est_type == 'MLP':

            def clf_partial():
                pipe = sklearn.pipeline.Pipeline(
                    [
                        (
                            'inputer',
                            sklearn.preprocessing.Imputer(
                                missing_values='NaN', strategy='mean', axis=0
                            ),
                        ),
                        # ('scale', sklearn.preprocessing.StandardScaler),
                        ('est', est_class(**est_params)),
                    ]
                )
                return multiclass_wrapper(pipe)

        elif est_type == 'Logit':

            def clf_partial():
                pipe = sklearn.pipeline.Pipeline(
                    [
                        (
                            'inputer',
                            sklearn.preprocessing.Imputer(
                                missing_values='NaN', strategy='mean', axis=0
                            ),
                        ),
                        ('est', est_class(**est_params)),
                    ]
                )
                return multiclass_wrapper(pipe)

        else:

            def clf_partial():
                return multiclass_wrapper(est_class(**est_params))

        return clf_partial

    def _train_deploy_clf(pblm, task_key, data_key, clf_key):
        X_df = pblm.samples.X_dict[data_key]
        labels = pblm.samples.subtasks[task_key]
        assert np.all(labels.encoded_df.index == X_df.index)
        clf_partial = pblm._get_estimator(clf_key)
        print(
            'Training deployment {} classifier on {} for {}'.format(
                clf_key, data_key, task_key
            )
        )
        clf = clf_partial()
        index = X_df.index
        X = X_df.loc[index].values
        y = labels.encoded_df.loc[index].values.ravel()
        clf.fit(X, y)
        return clf

    def _optimize_rf_hyperparams(pblm, data_key=None, task_key=None):
        """
        helper script I've only run interactively

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            #>>> pblm = OneVsOneProblem.from_empty('GZ_Master1')
            >>> pblm.load_samples()
            >>> pblm.load_features()
            >>> pblm.build_feature_subsets()
            >>> data_key=None
            >>> task_key=None
        """
        from sklearn.model_selection import RandomizedSearchCV  # NOQA
        from sklearn.model_selection import GridSearchCV  # NOQA
        from sklearn.ensemble import RandomForestClassifier
        from wbia.algo.verif import sklearn_utils

        if data_key is None:
            data_key = pblm.default_data_key
        if task_key is None:
            task_key = pblm.primary_task_key

        # Load data
        X = pblm.samples.X_dict[data_key].values
        y = pblm.samples.subtasks[task_key].y_enc
        groups = pblm.samples.group_ids

        # Define estimator and parameter search space
        grid = {
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced'],
            'criterion': ['entropy', 'gini'],
            # 'max_features': ['sqrt', 'log2'],
            'max_features': ['sqrt'],
            'min_samples_leaf': list(range(2, 11)),
            'min_samples_split': list(range(2, 11)),
            'n_estimators': [8, 64, 128, 256, 512, 1024],
        }
        est = RandomForestClassifier(missing_values=np.nan)
        if False:
            # debug
            params = ut.util_dict.all_dict_combinations(grid)[0]
            est.set_params(verbose=10, n_jobs=1, **params)
            est.fit(X=X, y=y)

        cv = sklearn_utils.StratifiedGroupKFold(n_splits=3)

        if True:
            n_iter = 25
            SearchCV = ut.partial(RandomizedSearchCV, n_iter=n_iter)
        else:
            n_iter = ut.prod(map(len, grid.values()))
            SearchCV = GridSearchCV

        search = SearchCV(est, grid, cv=cv, verbose=10)

        n_cpus = ut.num_cpus()
        thresh = n_cpus * 1.5
        n_jobs_est = 1
        n_jobs_ser = min(n_cpus, n_iter)
        if n_iter < thresh:
            n_jobs_est = int(max(1, thresh / n_iter))
        est.set_params(n_jobs=n_jobs_est)
        search.set_params(n_jobs=n_jobs_ser)

        search.fit(X=X, y=y, groups=groups)

        res = search.cv_results_.copy()
        alias = ut.odict(
            [
                ('rank_test_score', 'rank'),
                ('mean_test_score', 'μ-test'),
                ('std_test_score', 'σ-test'),
                ('mean_train_score', 'μ-train'),
                ('std_train_score', 'σ-train'),
                ('mean_fit_time', 'fit_time'),
                ('params', 'params'),
            ]
        )
        res = ut.dict_subset(res, alias.keys())
        cvresult_df = pd.DataFrame(res).rename(columns=alias)
        cvresult_df = cvresult_df.sort_values('rank').reset_index(drop=True)
        params = pd.DataFrame.from_dict(cvresult_df['params'].values.tolist())
        print('Varied params:')
        print(ut.repr4(ut.map_vals(set, params.to_dict('list'))))
        print('Ranked Params')
        print(params)
        print('Ranked scores on development set:')
        print(cvresult_df)
        print('Best parameters set found on hyperparam set:')
        print('best_params_ = %s' % (ut.repr4(search.best_params_),))

        print('Fastest params')
        cvresult_df.loc[cvresult_df['fit_time'].idxmin()]['params']

    def _dev_calib(pblm):
        """
        interactive script only
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import log_loss, brier_score_loss

        # Load data
        data_key = pblm.default_data_key
        task_key = pblm.primary_task_key
        X = pblm.samples.X_dict[data_key].values
        y = pblm.samples.subtasks[task_key].y_enc
        groups = pblm.samples.group_ids

        # Split into test/train/valid
        cv = sklearn_utils.StratifiedGroupKFold(n_splits=2)
        test_idx, train_idx = next(cv.split(X, y, groups))
        # valid_idx = train_idx[0::2]
        # train_idx = train_idx[1::2]
        # train_valid_idx = np.hstack([train_idx, valid_idx])

        # Train Uncalibrated RF
        est_kw = pblm._estimator_params('RF')[0]
        uncal_clf = RandomForestClassifier(**est_kw)
        uncal_clf.fit(X[train_idx], y[train_idx])
        uncal_probs = uncal_clf.predict_proba(X[test_idx]).T[1]
        uncal_score = log_loss(y[test_idx] == 1, uncal_probs)
        uncal_brier = brier_score_loss(y[test_idx] == 1, uncal_probs)

        # Train Calibrated RF
        method = 'isotonic' if len(test_idx) > 2000 else 'sigmoid'
        precal_clf = RandomForestClassifier(**est_kw)
        # cv = sklearn_utils.StratifiedGroupKFold(n_splits=3)
        cal_clf = CalibratedClassifierCV(precal_clf, cv=2, method=method)
        cal_clf.fit(X[train_idx], y[train_idx])
        cal_probs = cal_clf.predict_proba(X[test_idx]).T[1]
        cal_score = log_loss(y[test_idx] == 1, cal_probs)
        cal_brier = brier_score_loss(y[test_idx] == 1, cal_probs)

        print('cal_brier = %r' % (cal_brier,))
        print('uncal_brier = %r' % (uncal_brier,))

        print('uncal_score = %r' % (uncal_score,))
        print('cal_score = %r' % (cal_score,))

        import wbia.plottool as pt

        ut.qtensure()
        pt.figure()
        ax = pt.gca()

        y_test = y[test_idx] == 1
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, uncal_probs, n_bins=10
        )

        ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            's-',
            label='%s (%1.3f)' % ('uncal-RF', uncal_brier),
        )

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, cal_probs, n_bins=10
        )
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            's-',
            label='%s (%1.3f)' % ('cal-RF', cal_brier),
        )
        pt.legend()


@ut.reloadable_class
class ClfResult(ut.NiceRepr):
    r"""
    Handles evaluation statistics for a multiclass classifier trained on a
    specific dataset with specific labels.
    """

    # Attributes that identify the task and data the classifier is evaluated on
    _key_attrs = ['task_key', 'data_key', 'class_names']

    # Attributes about results and labels of individual samples
    _datafame_attrs = ['probs_df', 'probhats_df', 'target_bin_df', 'target_enc_df']

    def __init__(res):
        pass

    def __nice__(res):
        return '{}, {}, {}'.format(res.task_key, res.data_key, len(res.index))

    @property
    def index(res):
        return res.probs_df.index

    @classmethod
    def make_single(ClfResult, clf, X_df, test_idx, labels, data_key, feat_dims=None):
        """
        Make a result for a single cross validiation subset
        """
        X_df_test = X_df.iloc[test_idx]
        if feat_dims is not None:
            X_df_test = X_df_test[feat_dims]
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
        res.feat_dims = feat_dims

        res.probs_df = sklearn_utils.predict_proba_df(clf, X_df_test, res.class_names)
        res.target_bin_df = labels.indicator_df.iloc[test_idx]
        res.target_enc_df = labels.encoded_df.iloc[test_idx]

        if hasattr(clf, 'estimators_') and labels.n_classes > 2:
            # The n-th estimator in the OVR classifier predicts the prob of the
            # n-th class (as label 1).
            probs_hat = np.hstack(
                [est.predict_proba(X_df_test)[:, 1:2] for est in clf.estimators_]
            )
            res.probhats_df = pd.DataFrame(
                align_cols(probs_hat, clf.classes_, labels.classes_),
                index=index,
                columns=res.class_names,
            )
            # In the OVR-case, ideally things will sum to 1, but when they
            # don't normalization happens. An Z-value of more than 1 means
            # overconfidence, and under 0 means underconfidence.
            res.confidence_ratio = res.probhats_df.sum(axis=1)
        else:
            res.probhats_df = None
        return res

    def compress(res, flags):
        res2 = ClfResult()
        res2.task_key = res.task_key
        res2.data_key = res.data_key
        res2.class_names = res.class_names
        res2.probs_df = res.probs_df[flags]
        res2.target_bin_df = res.target_bin_df[flags]
        res2.target_enc_df = res.target_enc_df[flags]
        if res.probhats_df is None:
            res2.probhats_df = None
        else:
            res2.probhats_df = res.probhats_df[flags]
            # res2.confidence_ratio = res.confidence_ratio[flags]
        return res2

    @classmethod
    def combine_results(ClfResult, res_list, labels=None):
        """
        Combine results from cross validation runs into a single result
        representing the performance of the entire dataset
        """
        # Ensure that res_lists are not overlapping
        for r1, r2 in ut.combinations(res_list, 2):
            assert (
                len(r1.index.intersection(r2.index)) == 0
            ), 'ClfResult dataframes must be disjoint'
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
            assert all(
                [getattr(r, attr) == val for r in res_list]
            ), 'ClfResult with different key attributes are incompatible'
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

    def hardness_analysis(res, samples, infr=None, method='argmax'):
        """
        samples = pblm.samples

        # TODO MWE with sklearn data

            # ClfResult.make_single(ClfResult, clf, X_df, test_idx, labels,
            # data_key, feat_dims=None):

            import sklearn.datasets
            iris = sklearn.datasets.load_iris()

            # TODO: make this setup simpler
            pblm = ClfProblem()
            task_key, clf_key, data_key = 'iris', 'RF', 'learn(all)'
            X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
            samples = MultiTaskSamples(X_df.index)
            samples.apply_indicators({'iris': {name: iris.target == idx
                         for idx, name in enumerate(iris.target_names)}})
            samples.X_dict = {'learn(all)': X_df}

            pblm.samples = samples
            pblm.xval_kw['type'] = 'StratifiedKFold'
            clf_list, res_list = pblm._train_evaluation_clf(
                task_key, data_key, clf_key)
            labels = pblm.samples.subtasks[task_key]
            res = ClfResult.combine_results(res_list, labels)


        res.get_thresholds('mcc', 'maximize')

        predict_method = 'argmax'

        """
        meta = {}
        easiness = ut.ziptake(res.probs_df.values, res.target_enc_df.values)

        # pred = sklearn_utils.predict_from_probs(res.probs_df, predict_method)
        if method == 'max-mcc':
            method = res.get_thresholds('mcc', 'maximize')
        pred = sklearn_utils.predict_from_probs(res.probs_df, method, force=True)

        meta['easiness'] = np.array(easiness).ravel()
        meta['hardness'] = 1 - meta['easiness']
        meta['aid1'] = res.probs_df.index.get_level_values(0)
        meta['aid2'] = res.probs_df.index.get_level_values(1)
        # meta['aid1'] = samples.aid_pairs.T[0].take(res.probs_df.index.values)
        # meta['aid2'] = samples.aid_pairs.T[1].take(res.probs_df.index.values)
        # meta['pred'] = res.probs_df.values.argmax(axis=1)
        meta['pred'] = pred.values
        meta['real'] = res.target_enc_df.values.ravel()
        meta['failed'] = meta['pred'] != meta['real']
        meta = pd.DataFrame(meta)
        meta = meta.set_index(['aid1', 'aid2'], drop=False)

        if infr is not None:
            ibs = infr.ibs
            edges = list(meta.index.tolist())
            conf_dict = infr.get_edge_attrs(
                'confidence',
                edges,
                on_missing='filter',
                default=ibs.const.CONFIDENCE.CODE.UNKNOWN,
            )
            conf_df = pd.DataFrame.from_dict(conf_dict, orient='index')
            conf_df = conf_df[0].map(ibs.const.CONFIDENCE.CODE_TO_INT)
            meta = meta.assign(real_conf=conf_df)
            meta['real_conf'] = np.nan_to_num(meta['real_conf']).astype(np.int)

        meta = meta.sort_values('hardness', ascending=False)
        res.meta = meta
        return res.meta

    def missing_classes(res):
        # Find classes that were never predicted
        unique_predictions = np.unique(res.probs_df.values.argmax(axis=1))
        n_classes = len(res.class_names)
        missing_classes = ut.index_complement(unique_predictions, n_classes)
        return missing_classes

    def augment_if_needed(res):
        """
        Adds in dummy values for missing classes
        """
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
            # make sample weights where dummies have no weight
            sample_weight = np.hstack([sample_weight, np.full(n_missing, 0)])

            if res.probhats_df is not None:
                clf_probhats_aug = np.vstack([clf_probhats_aug, missing_bin])

        res.clf_probs = clf_probs_aug
        res.clf_probhats = clf_probhats_aug
        res.y_test_enc = y_test_enc_aug
        res.y_test_bin = y_test_bin_aug
        res.sample_weight = sample_weight

    def extended_clf_report(res, verbose=True):
        res.augment_if_needed()
        pred_enc = res.clf_probs.argmax(axis=1)
        y_pred = pred_enc
        y_true = res.y_test_enc
        sample_weight = res.sample_weight
        target_names = res.class_names
        report = sklearn_utils.classification_report2(
            y_true,
            y_pred,
            target_names=target_names,
            sample_weight=sample_weight,
            verbose=verbose,
        )
        return report

    def print_report(res):
        res.augment_if_needed()
        pred_enc = res.clf_probs.argmax(axis=1)
        res.extended_clf_report()
        report = sklearn.metrics.classification_report(
            y_true=res.y_test_enc,
            y_pred=pred_enc,
            target_names=res.class_names,
            sample_weight=res.sample_weight,
        )
        print('Precision/Recall Report:')
        print(report)

    def get_thresholds(res, metric='mcc', value='maximize'):
        """
        get_metric = 'thresholds'
        at_metric = metric = 'mcc'
        at_value = value = 'maximize'

        a = []
        b = []
        for x in np.linspace(0, 1, 1000):
            a += [cfms.get_metric_at_metric('thresholds', 'fpr', x, subindex=True)]
            b += [cfms.get_thresh_at_metric('fpr', x)]
        a = np.array(a)
        b = np.array(b)
        d = (a - b)
        print((d.min(), d.max()))
        """
        threshes = {}
        for class_name in res.class_names:
            cfms = res.confusions(class_name)
            thresh = cfms.get_metric_at_metric('thresh', metric, value)
            threshes[class_name] = thresh
        return threshes

    @profile
    def get_pos_threshes(
        res,
        metric='fpr',
        value=1e-4,
        maximize=False,
        warmup=200,
        priors=None,
        min_thresh=0.5,
    ):
        """
        Finds a threshold that achieves the desired `value` for the desired
        metric, while maximizing or minimizing the threshold.

        For positive classification you want to minimize the threshold.
        Priors can be passed in to augment probabilities depending on support.
        By default a class prior is 1 for threshold minimization and 0 for
        maximization.
        """
        pos_threshes = {}
        if priors is None:
            priors = {name: float(not maximize) for name in res.class_names}
        for class_name in res.class_names:
            cfms = res.confusions(class_name)

            learned_thresh = cfms.get_metric_at_metric('thresh', metric, value)
            # learned_thresh = cfms.get_thresh_at_metric(
            #     metric, value, maximize=maximize)

            prior_thresh = priors[class_name]
            n_support = cfms.n_pos

            if warmup is not None:
                """
                python -m wbia.plottool.draw_func2 plot_func --show --range=0,1 \
                        --func="lambda x: np.maximum(0, (x - .6) / (1 - .6))"
                """
                # If n_support < warmup: then interpolate to learned thresh
                nmax = warmup if isinstance(warmup, int) else warmup[class_name]
                # alpha varies from 0 to 1
                alpha = min(nmax, n_support) / nmax
                # transform alpha through nonlinear function (similar to ReLU)
                p = 0.6  # transition point
                alpha = max(0, (alpha - p) / (1 - p))
                thresh = prior_thresh * (1 - alpha) + learned_thresh * (alpha)
            else:
                thresh = learned_thresh
            pos_threshes[class_name] = max(min_thresh, thresh)
        return pos_threshes

    def report_thresholds(res, warmup=200):
        # import vtool as vt
        ut.cprint('Threshold Report', 'yellow')
        y_test_bin = res.target_bin_df.values
        # y_test_enc = y_test_bin.argmax(axis=1)
        # clf_probs = res.probs_df.values

        # The maximum allowed false positive rate
        # We expect that we will make 1 error every 1,000 decisions
        # thresh_df['foo'] = [1, 2, 3]
        # thresh_df['foo'][res.class_names[k]] = 1

        # for k in [2, 0, 1]:
        choice_mv = ut.odict(
            [
                ('@fpr=.01', ('fpr', 0.01)),
                ('@fpr=.001', ('fpr', 0.001)),
                ('@fpr=.0001', ('fpr', 1e-4)),
                ('@fpr=.0000', ('fpr', 0)),
                ('@max(mcc)', ('mcc', 'max')),
                # (class_name + '@max(acc)', ('acc', 'max')),
                # (class_name + '@max(mk)', ('mk', 'max')),
                # (class_name + '@max(bm)', ('bm', 'max')),
            ]
        )
        for k in range(y_test_bin.shape[1]):
            thresh_dict = ut.odict()
            class_name = res.class_names[k]
            cfms = res.confusions(class_name)
            # probs, labels = clf_probs.T[k], y_test_bin.T[k]
            # cfms = vt.ConfusionMetrics().fit(probs, labels)

            for k, mv in choice_mv.items():
                metric, value = mv
                idx = cfms.get_index_at_metric(metric, value)
                key = class_name + k
                thresh_dict[key] = ut.odict()
                for metric in ['thresh', 'fpr', 'tpr', 'tpa', 'bm', 'mk', 'mcc']:
                    thresh_dict[key][metric] = cfms.get_metric_at_index(metric, idx)
            thresh_df = pd.DataFrame.from_dict(thresh_dict, orient='index')
            thresh_df = thresh_df.loc[list(thresh_dict.keys())]
            if cfms.n_pos > 0 and cfms.n_neg > 0:
                print('Raw 1vR {} Thresholds'.format(class_name))
                print(ut.indent(thresh_df.to_string(float_format='{:.4f}'.format)))
                # chosen_type = class_name + '@fpr=0'
                # pos_threshes[class_name] = thresh_df.loc[chosen_type]['thresh']

        for choice_k, choice_mv in iter(choice_mv.items()):
            metric, value = choice_mv
            pos_threshes = res.get_pos_threshes(metric, value, warmup=warmup)
            print('Choosing threshold based on %s' % (choice_k,))
            res.report_auto_thresholds(pos_threshes)

    def report_auto_thresholds(res, threshes, verbose=True):
        report_lines = []
        print_ = report_lines.append
        print_(
            'Chosen thresholds = %s'
            % (ut.repr2(threshes, nl=1, precision=4, align=True),)
        )

        res.augment_if_needed()
        target_names = res.class_names
        sample_weight = res.sample_weight
        y_true = res.y_test_enc.ravel()
        y_pred, can_autodecide = sklearn_utils.predict_from_probs(
            res.clf_probs,
            threshes,
            res.class_names,
            force=False,
            multi=False,
            return_flags=True,
        )
        can_autodecide[res.sample_weight == 0] = False

        auto_pred = y_pred[can_autodecide].astype(np.int)
        auto_true = y_true[can_autodecide].ravel()
        auto_probs = res.clf_probs[can_autodecide]

        total_cases = int(sample_weight.sum())
        print_('Will autodecide for %r/%r cases' % (can_autodecide.sum(), (total_cases)))

        def frac_str(a, b):
            return '{:}/{:} = {:.2f}%'.format(int(a), int(b), a / b)

        y_test_bin = res.target_bin_df.values
        supported_class_idxs = [k for k, y in enumerate(y_test_bin.T) if y.sum() > 0]

        print_(' * Auto-Decide Per-Class Summary')
        for k in supported_class_idxs:
            # Look at fail/succs in threshold
            name = res.class_names[k]
            # number of times this class appears overall
            n_total_k = (y_test_bin.T[k]).sum()
            # get the cases where this class was predicted
            auto_true_k = auto_true == k
            auto_pred_k = auto_pred == k
            # number of cases auto predicted
            n_pred_k = auto_pred_k.sum()
            # number of times auto was right
            n_tp = (auto_true_k & auto_pred_k).sum()
            # number of times auto was wrong
            n_fp = (~auto_true_k & auto_pred_k).sum()
            fail_str = frac_str(n_fp, n_pred_k)
            pass_str = frac_str(n_tp, n_total_k)
            fmtstr = '\n'.join(
                [
                    '{name}:',
                    '    {n_total_k} samples existed, and did {n_pred_k} auto predictions',
                    '    got {pass_str} right',
                    '    made {fail_str} errors',
                ]
            )
            print_(ut.indent(fmtstr.format(**locals())))

        report = sklearn_utils.classification_report2(
            y_true,
            y_pred,
            target_names=target_names,
            sample_weight=can_autodecide.astype(np.float),
            verbose=False,
        )
        print_(' * Auto-Decide Confusion')
        print_(ut.indent(str(report['confusion'])))
        print_(' * Auto-Decide Metrics')
        print_(ut.indent(str(report['metrics'])))
        if 'mcc' in report:
            print_(ut.indent(str(report['mcc'])))

        try:
            auto_truth_bin = res.y_test_bin[can_autodecide]
            for k in supported_class_idxs:
                auto_truth_k = auto_truth_bin.T[k]
                auto_probs_k = auto_probs.T[k]
                if auto_probs_k.sum():
                    auc = sklearn.metrics.roc_auc_score(auto_truth_k, auto_probs_k)
                    print_(
                        ' * Auto AUC(Macro): {:.4f} for class={}'.format(
                            auc, res.class_names[k]
                        )
                    )
        except ValueError:
            pass
        report = '\n'.join(report_lines)
        if verbose:
            print(report)
        return report

    def confusions(res, class_name):
        import vtool as vt

        y_test_bin = res.target_bin_df.values
        clf_probs = res.probs_df.values
        k = res.class_names.index(class_name)
        probs, labels = clf_probs.T[k], y_test_bin.T[k]
        confusions = vt.ConfusionMetrics().fit(probs, labels)
        return confusions

    def ishow_roc(res):
        import vtool as vt
        import wbia.plottool as pt

        ut.qtensure()
        y_test_bin = res.target_bin_df.values
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
            confusions = res.confusions(class_name)
            ROCInteraction = vt.interact_roc_factory(
                confusions, show_operating_point=True
            )
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
        confusions = vt.ConfusionMetrics().fit(probs, labels)
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

    def confusions_ovr(res):
        # one_vs_rest confusions
        import vtool as vt

        res.augment_if_needed()
        for k in range(res.y_test_bin.shape[1]):
            class_k_truth = res.y_test_bin.T[k]
            class_k_probs = res.clf_probs.T[k]
            cfms = vt.ConfusionMetrics().fit(class_k_probs, class_k_truth)
            # auc = sklearn.metrics.roc_auc_score(class_k_truth, class_k_probs)
            yield res.class_names[k], cfms

    def roc_score(res):
        res.augment_if_needed()
        auc_learn = sklearn.metrics.roc_auc_score(res.y_test_bin, res.clf_probs)
        return auc_learn


@ut.reloadable_class
class MultiTaskSamples(ut.NiceRepr):
    """
    Handles samples (i.e. feature-label pairs) with a combination of
    non-mutually exclusive subclassification labels

    CommandLine:
        python -m wbia.algo.verif.clf_helpers MultiTaskSamples

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.verif.clf_helpers import *  # NOQA
        >>> samples = MultiTaskSamples([0, 1, 2, 3])
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

    # def set_simple_scores(samples, simple_scores):
    #     if simple_scores is not None:
    #         edges = ut.emap(tuple, samples.aid_pairs.tolist())
    #         assert (edges == simple_scores.index.tolist())
    #     samples.simple_scores = simple_scores

    # def set_feats(samples, X_dict):
    #     if X_dict is not None:
    #         edges = ut.emap(tuple, samples.aid_pairs.tolist())
    #         for X in X_dict.values():
    #             assert np.all(edges == X.index.tolist())
    #     samples.X_dict = X_dict

    def supported_tasks(samples):
        for task_key, labels in samples.subtasks.items():
            labels = samples.subtasks[task_key]
            if labels.has_support():
                yield task_key

    def apply_indicators(samples, tasks_to_indicators):
        """
        Adds labels for a specific task

        Args:
            tasks_to_indicators (dict): takes the form:
                {
                   `my_task_name1' {
                       'class1': [list of bools indicating class membership]
                       ...
                       'classN': [list of bools indicating class membership]
                   }
                   ...
                   `my_task_nameN': ...
               }
        """
        n_samples = None
        samples.n_tasks = len(tasks_to_indicators)
        for task_name, indicator in tasks_to_indicators.items():
            labels = MultiClassLabels.from_indicators(
                indicator, task_name=task_name, index=samples.index
            )
            samples.subtasks[task_name] = labels
            if n_samples is None:
                n_samples = labels.n_samples
            elif n_samples != labels.n_samples:
                raise ValueError('numer of samples is different')
        samples.n_samples = n_samples

    def apply_encoded_labels(samples, y_enc, class_names, task_name):
        """
        Adds labels for a specific task. Alternative to `apply_indicators`

        Args:
            y_enc (list): integer label indicating the class for each sample
            class_names (list): list of strings indicating the class-domain
            task_name (str): key for denoting this specific task
        """
        # convert to indicator structure and use that
        tasks_to_indicators = ut.odict(
            [
                (
                    task_name,
                    ut.odict(
                        [
                            (name, np.array(y_enc) == i)
                            for i, name in enumerate(class_names)
                        ]
                    ),
                )
            ]
        )
        samples.apply_indicators(tasks_to_indicators)

    # @ut.memoize
    def encoded_2d(samples):
        encoded_2d = pd.concat([v.encoded_df for k, v in samples.items()], axis=1)
        return encoded_2d

    def class_name_basis(samples):
        """ corresponds with indexes returned from encoded1d """
        class_name_basis = [
            t[::-1]
            for t in ut.product(*[v.class_names for k, v in samples.items()][::-1])
        ]
        # class_name_basis = [(b, a) for a, b in ut.product(*[
        #     v.class_names for k, v in samples.items()][::-1])]
        return class_name_basis

    def class_idx_basis_2d(samples):
        """ 2d-index version of class_name_basis """
        class_idx_basis_2d = [
            (b, a)
            for a, b in ut.product(
                *[range(v.n_classes) for k, v in samples.items()][::-1]
            )
        ]
        return class_idx_basis_2d

    def class_idx_basis_1d(samples):
        """ 1d-index version of class_name_basis """
        n_states = np.prod([v.n_classes for k, v in samples.items()])
        class_idx_basis_1d = np.arange(n_states, dtype=np.int)
        return class_idx_basis_1d

    # @ut.memoize
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
            samples.encoded_1d().values, labels=class_idx_basis_1d
        )
        multi_task_hist = ut.map_keys(lambda k: class_name_basis[k], multi_task_idx_hist)
        return multi_task_hist

    def items(samples):
        for task_name, labels in samples.subtasks.items():
            yield task_name, labels

    # def take(samples, idxs):
    #     mask = ut.index_to_boolmask(idxs, len(samples))
    #     return samples.compress(mask)

    @property
    def group_ids(samples):
        return None

    def stratified_kfold_indices(samples, **xval_kw):
        """
        TODO: check xval label frequency


        """
        from sklearn import model_selection

        X = np.empty((len(samples), 0))
        y = samples.encoded_1d().values
        groups = samples.group_ids

        type_ = xval_kw.pop('type', 'StratifiedGroupKFold')
        if type_ == 'StratifiedGroupKFold':
            assert groups is not None
            # FIXME: The StratifiedGroupKFold could be implemented better.
            splitter = sklearn_utils.StratifiedGroupKFold(**xval_kw)
            skf_list = list(splitter.split(X=X, y=y, groups=groups))
        elif type_ == 'StratifiedKFold':
            splitter = model_selection.StratifiedKFold(**xval_kw)
            skf_list = list(splitter.split(X=X, y=y))
        return skf_list

    def subsplit_indices(samples, subset_idx, **xval_kw):
        """ split an existing set """
        from sklearn import model_selection

        X = np.empty((len(subset_idx), 0))
        y = samples.encoded_1d().values[subset_idx]
        groups = samples.group_ids[subset_idx]

        xval_kw_ = xval_kw.copy()
        if 'n_splits' not in xval_kw_:
            xval_kw_['n_splits'] = 3
        type_ = xval_kw_.pop('type', 'StratifiedGroupKFold')
        if type_ == 'StratifiedGroupKFold':
            assert groups is not None
            # FIXME: The StratifiedGroupKFold could be implemented better.
            splitter = sklearn_utils.StratifiedGroupKFold(**xval_kw_)
            rel_skf_list = list(splitter.split(X=X, y=y, groups=groups))
        elif type_ == 'StratifiedKFold':
            splitter = model_selection.StratifiedKFold(**xval_kw_)
            rel_skf_list = list(splitter.split(X=X, y=y))

        # map back into original coords
        skf_list = [
            (subset_idx[rel_idx1], subset_idx[rel_idx2])
            for rel_idx1, rel_idx2 in rel_skf_list
        ]

        for idx1, idx2 in skf_list:
            assert len(np.intersect1d(subset_idx, idx1)) == len(idx1)
            assert len(np.intersect1d(subset_idx, idx2)) == len(idx2)
            # assert
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
        labels.default_class = None

    def has_support(labels):
        return len(labels.make_histogram()) > 1

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
        assert np.all(
            indicator_df.sum(axis=1).values
        ), 'states in the same task must be mutually exclusive'
        labels.indicator_df = indicator_df
        labels.class_names = indicator_df.columns.values
        labels.encoded_df = pd.DataFrame(
            indicator_df.values.argmax(axis=1), columns=[task_name], index=index,
        )
        labels.task_name = task_name
        labels.n_samples = n_samples
        labels.n_classes = len(labels.class_names)
        if labels.n_classes == 1:
            labels.n_classes = 2  # 1 column means binary case
        labels.classes_ = np.arange(labels.n_classes)
        labels.default_class_name = labels.class_names[1]
        return labels

    @property
    def target_type(labels):
        return sklearn.utils.multiclass.type_of_target(labels.y_enc)

    def one_vs_rest_task_names(labels):
        return [
            labels.task_name + '(' + labels.class_names[k] + '-v-rest)'
            for k in range(labels.n_classes)
        ]

    def gen_one_vs_rest_labels(labels):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.verif.clf_helpers import *  # NOQA
            >>> indicator = ut.odict([
            >>>         ('state1', [0, 0, 0, 1]),
            >>>         ('state2', [0, 0, 1, 0]),
            >>>         ('state3', [1, 1, 0, 0]),
            >>>     ])
            >>> labels = MultiClassLabels.from_indicators(indicator, task_name='task1')
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
                indicator_df.values.argmax(axis=1), columns=[task_name], index=index
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
        class_hist = ut.map_keys(lambda idx: labels.class_names[idx], class_idx_hist)
        return class_hist

    def print_info(labels):
        print('hist(%s) = %s' % (labels.task_name, ut.repr4(labels.make_histogram())))
        print('len(%s) = %s' % (labels.task_name, len(labels)))


class IrisProblem(ClfProblem):
    """
    Simple demo using the abstract clf problem to work on the iris dataset.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.verif.clf_helpers import *  # NOQA
            >>> pblm = IrisProblem()
            >>> pblm.setup()
            >>> pblm.samples

    """

    def setup(pblm):
        import sklearn.datasets

        iris = sklearn.datasets.load_iris()

        pblm.primary_task_key = 'iris'
        pblm.default_data_key = 'learn(all)'
        pblm.default_clf_key = 'RF'

        X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        samples = MultiTaskSamples(X_df.index)
        samples.apply_indicators(
            {
                'iris': {
                    name: iris.target == idx for idx, name in enumerate(iris.target_names)
                }
            }
        )
        samples.X_dict = {'learn(all)': X_df}

        pblm.samples = samples
        pblm.xval_kw['type'] = 'StratifiedKFold'


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.verif.samples
        python -m wbia.algo.verif.samples --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
