# -*- coding: utf-8 -*-
"""
TODO:

* Use depcache to compute match objects (ideally via the infr object)

* Find thresholds to maximize score metric (mcc, auc)

* Get end-to-end system test working with simulated reviewer

* Autoselect features:
    * Learn RF
    * prune bottom N features
    * loop until only X features remain


* allow random forests / whatever classifier to be trained according to one of the following ways:
    * Multiclass - naitively output multiclass labels
    * One-vs-Rest - Use sklearns 1-v-Rest framework
    * One-vs-One - Use sklearns 1-v-1 framework

"""
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import numpy as np
import vtool as vt
import dtool as dt
import copy
from six.moves import zip, range  # NOQA
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass
import sklearn.ensemble
from ibeis.scripts import clf_helpers
print, rrr, profile = ut.inject2(__name__)


class PairSampleConfig(dt.Config):
    _param_info_list = [
        ut.ParamInfo('top_gt', 4),
        ut.ParamInfo('mid_gt', 2),
        ut.ParamInfo('bot_gt', 2),
        ut.ParamInfo('rand_gt', 2),
        ut.ParamInfo('top_gf', 3),
        ut.ParamInfo('mid_gf', 2),
        ut.ParamInfo('bot_gf', 1),
        ut.ParamInfo('rand_gf', 2),
    ]


class PairFeatureConfig(dt.Config):
    _param_info_list = [
        ut.ParamInfo('indices', slice(0, 26, 5)),
        ut.ParamInfo('sum', True),
        ut.ParamInfo('std', True),
        ut.ParamInfo('mean', True),
        ut.ParamInfo('med', True),
    ]


class VsOneAssignConfig(dt.Config):
    _param_info_list = vt.matching.VSONE_ASSIGN_CONFIG


@ut.reloadable_class
class OneVsOneProblem(object):
    """
    Keeps information about the one-vs-one pairwise classification problem

    CommandLine:
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --show

    Example:
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> self = OneVsOneProblem()
        >>> self.load_features()
        >>> self.load_samples()
    """
    def __init__(self):
        import ibeis
        # ut.aug_sysargv('--db PZ_Master1')
        qreq_ = ibeis.testdata_qreq_(
            defaultdb='PZ_PB_RF_TRAIN',
            a=':mingt=3,species=primary',
            # t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum',
            # t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum,QRH=True',
            t='default:K=3,Knorm=1,score_method=csum,prescore_method=csum,QRH=True',
        )
        hyper_params = dt.Config.from_dict(dict(
            subsample=None,
            pair_sample=PairSampleConfig(),
            vsone_assign=VsOneAssignConfig(),
            pairwise_feats=PairFeatureConfig(), ),
            tablename='HyperParams'
        )
        if qreq_.qparams.featweight_enabled:
            hyper_params.vsone_assign['weight'] = 'fgweights'
        else:
            hyper_params.vsone_assign['weight'] = None
        assert qreq_.qparams.can_match_samename is True
        assert qreq_.qparams.prescore_method == 'csum'
        self.hyper_params = hyper_params
        self.qreq_ = qreq_
        self.ibs = qreq_.ibs

    def set_pandas_options(self):
        # pd.options.display.max_rows = 10
        pd.options.display.max_rows = 20
        pd.options.display.max_columns = 40
        pd.options.display.width = 160
        pd.options.display.float_format = lambda x: '%.4f' % (x,)

    def load_samples(self):
        self.samples = AnnotPairSamples(
            self.ibs, copy.deepcopy(self.raw_aid_pairs),
            copy.deepcopy(self.raw_simple_scores),
            copy.deepcopy(self.raw_X_dict))

    def load_features(self):
        qreq_ = self.qreq_
        dbname = qreq_.ibs.get_dbname()
        vsmany_hashid = qreq_.get_cfgstr(hash_pipe=True, with_input=True)
        hyper_params = self.hyper_params
        features_hashid = ut.hashstr27(vsmany_hashid + hyper_params.get_cfgstr())
        cfgstr = '_'.join(['devcache', str(dbname), features_hashid])
        cacher = ut.Cacher('pairwise_data_v8', cfgstr=cfgstr,
                           appname='vsone_rf_train', enabled=1)
        data = cacher.tryload()
        if not data:
            data = build_features(qreq_, hyper_params)
            cacher.save(data)
        aid_pairs, simple_scores, X_dict, match = data
        self.raw_aid_pairs = aid_pairs
        self.raw_X_dict = X_dict
        self.raw_simple_scores = simple_scores

    def evaluate_classifiers(self):
        """
        CommandLine:
            python -m ibeis.scripts.script_vsone evaluate_classifiers
            python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --show
            python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_MTEST --show
            python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_Master1 --show
            python -m ibeis.scripts.script_vsone evaluate_classifiers --db GZ_Master1 --show

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> self = OneVsOneProblem()
            >>> self.evaluate_classifiers()
        """
        self.set_pandas_options()

        ut.cprint('\n--- LOADING DATA ---', 'blue')
        self.load_features()
        self.load_samples()

        # self.samples.print_info()
        ut.cprint('\n--- CURATING DATA ---', 'blue')
        self.reduce_dataset_size()
        self.samples.print_info()
        print('---------------')

        ut.cprint('\n--- FEATURE INFO ---', 'blue')
        self.build_feature_subsets()

        if 0:
            for data_key in self.samples.X_dict.keys():
                print('\nINFO(samples.X_dict[%s])' % (data_key,))
                print(ut.indent(AnnotPairFeatInfo(self.samples.X_dict[data_key]).get_infostr()))

        task_keys = list(self.samples.subtasks.keys())
        # task_keys = ut.setdiff(task_keys, ['photobomb_state'])

        data_keys = list(self.samples.X_dict.keys())
        clf_keys = ['RF', 'RF-OVR']
        # clf_keys = ['RF', 'SVC']
        # clf_keys = ['RF']
        # clf_keys = ['RF-OVR']

        # task_keys = [
        #     'photobomb_state',
        #     'match_state',
        # ]

        # data_keys = [
        #     'learn(sum,glob,3,+view)',
        #     'learn(sum,glob,3)',
        #     'learn(sum,glob)',
        #     # 'learn(all)',
        #     # 'learn(local)',
        # ]

        # Remove any tasks that cant be done
        for task_key in task_keys[:]:
            labels = self.samples.subtasks[task_key]
            if len(labels.make_histogram()) < 2:
                print('No data to train task_key = %r' % (task_key,))
                task_keys.remove(task_key)

        ut.cprint('\n--- EVALUTE SIMPLE SCORES ---', 'blue')
        self.evaluate_simple_scores(task_keys)

        ut.cprint('\n--- LEARN CROSS-VALIDATED RANDOM FORESTS ---', 'blue')
        self.learn_evaluation_classifiers(task_keys, clf_keys, data_keys)

        selected_data_keys = []

        ut.cprint('\n--- EVALUATE LEARNED CLASSIFIERS ---', 'blue')
        from utool.experimental.pandas_highlight import to_string_monkey

        # For each task / classifier type
        for task_key in task_keys:
            ut.cprint('--- TASK = %s' % (ut.repr2(task_key),), 'turquoise')
            self.report_simple_scores(task_key)
            for clf_key in clf_keys:
                # Combine results over datasets
                print('clf_key = %s' % (ut.repr2(clf_key),))
                data_combo_res = self.task_combo_res[task_key][clf_key]
                df_auc_ovr = pd.DataFrame(dict([
                    (datakey, list(data_combo_res[datakey].roc_scores_ovr()))
                    for datakey in data_keys
                ]),
                    index=self.samples.subtasks[task_key].one_vs_rest_task_names()
                )
                ut.cprint('[%s] ROC-AUC(OVR) Scores' % (clf_key,), 'yellow')
                print(to_string_monkey(df_auc_ovr, highlight_cols='all'))

                if clf_key.endswith('-OVR'):
                    # Report un-normalized ovr measures if they available
                    ut.cprint('[%s] ROC-AUC(OVR_hat) Scores' % (clf_key,), 'yellow')
                    df_auc_ovr_hat = pd.DataFrame(dict([
                        (datakey, list(data_combo_res[datakey].roc_scores_ovr_hat()))
                        for datakey in data_keys
                    ]),
                        index=self.samples.subtasks[task_key].one_vs_rest_task_names()
                    )
                    print(to_string_monkey(df_auc_ovr_hat, highlight_cols='all'))

                roc_scores = {datakey: [data_combo_res[datakey].roc_score()]
                              for datakey in data_keys}
                df_auc = pd.DataFrame(roc_scores)
                ut.cprint('[%s] ROC-AUC(MacroAve) Scores' % (clf_key,), 'yellow')
                print(to_string_monkey(df_auc, highlight_cols='all'))

                # best_data_key = 'learn(sum,glob,3)'
                best_data_key = df_auc.columns[df_auc.values.argmax(axis=1)[0]]
                selected_data_keys.append(best_data_key)
                combo_res = data_combo_res[best_data_key]
                ut.cprint('[%s] BEST DataKey = %r' % (clf_key, best_data_key,), 'darkgreen')
                with ut.Indenter('[%s] ' % (best_data_key,)):
                    combo_res.extended_clf_report()
                res = combo_res
                if 0:
                    res.report_thresholds()
                if 0:
                    importance_datakeys = set([
                        # 'learn(all)'
                    ] + [best_data_key])

                    for data_key in importance_datakeys:
                        self.report_classifier_importance(task_key, data_key)

        # ut.cprint('\n--- FEATURE INFO ---', 'blue')
        # for best_data_key in selected_data_keys:
        #     print('data_key=(%s)' % (best_data_key,))
        #     print(ut.indent(AnnotPairFeatInfo(
        #           self.samples.X_dict[best_data_key]).get_infostr()))

        # TODO: view failure / success cases
        # Need to show and potentially fix misclassified examples
        if False:
            self.samples.aid_pairs
            combo_res.target_bin_df
            res = combo_res
            samples = self.samples
            meta = res.make_meta(samples).copy()
            import ibeis
            aid_pairs = ut.lzip(meta['aid1'], meta['aid2'])
            attrs = meta.drop(['aid1', 'aid2'], 1).to_dict(orient='list')
            ibs = self.qreq_.ibs
            infr = ibeis.AnnotInference.from_pairs(aid_pairs, attrs, ibs=ibs, verbose=3)
            infr.reset_feedback('staging')
            infr.reset_labels_to_ibeis()
            infr.apply_feedback_edges()
            infr.relabel_using_reviews()
            # x = [c for c in infr.consistent_compoments()]
            # cc = x[ut.argmax(ut.lmap(len, x))]
            # keep = list(cc.nodes())
            # infr.remove_aids(ut.setdiff(infr.aids, keep))
            infr.start_qt_interface()
            return

    def learn_evaluation_classifiers(self, task_keys=None, clf_keys=None, data_keys=None):
        """
        Evaluates by learning classifiers using cross validation.
        Do not use this to learn production classifiers.

        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --show
        """
        self.task_clfs = ut.AutoVivification()
        # self.task_res_list = ut.AutoVivification()
        self.task_combo_res = ut.AutoVivification()

        if task_keys is None:
            task_keys = list(self.samples.subtasks.keys())
        if data_keys is None:
            task_keys = list(self.samples.X_dict.keys())
        if clf_keys is None:
            clf_keys = ['RF']

        Prog = ut.ProgPartial(freq=1, adjust=False, prehack='%s')

        annot_cfgstr = self.samples.make_annotpair_vhashid()
        cfg_prefix = annot_cfgstr + self.qreq_.get_cfgstr()

        def cached_clf(task_key, data_key, clf_key):
            # TODO: add in params to cfgstr
            est_kw1, est_kw2, xval_kw = self.get_clf_params(clf_key)
            param_id = ut.get_dict_hashid(est_kw1)
            xval_id = ut.get_dict_hashid(xval_kw)
            cfgstr = '_'.join([cfg_prefix, param_id, xval_id, task_key,
                               data_key, clf_key])
            cacher = ut.Cacher('rf_clf_v9', cfgstr=cfgstr,
                               appname='vsone_rf_train', enabled=1,
                               verbose=1)
            data = cacher.tryload()
            if not data:
                data = self.learn_single_evaulation_clf(task_key, data_key, clf_key)
                cacher.save(data)
            clf_list, res_list = data
            labels = self.samples.subtasks[task_key]
            combo_res = clf_helpers.ClfResult.combine_results(res_list, labels)
            # combo_res.extended_clf_report()
            self.task_clfs[task_key][clf_key][data_key] = clf_list
            # self.task_res_list[task_key][clf_key][data_key]  = res_list
            self.task_combo_res[task_key][clf_key][data_key] = combo_res

        task_prog = Prog(task_keys, label='Task')
        for task_key in task_prog:
            dataset_prog = Prog(data_keys, label='Data')
            for data_key in dataset_prog:
                clf_prog = Prog(clf_keys, label='CLF')
                for clf_key in clf_prog:
                    cached_clf(task_key, data_key, clf_key)

    def get_clf_params(self, clf_key):
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

        # xvalkw = dict(n_splits=10, shuffle=True,
        xval_kw = {
            # 'n_splits': 10,
            'n_splits': 3,
            'shuffle': True,
            'random_state': 3953056901,
        }
        return est_kw1, est_kw2, xval_kw

    def learn_single_evaulation_clf(self, task_key, data_key, clf_key):
        """
        Learns a cross-validated classifier on the dataset

            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> self = OneVsOneProblem()
            >>> self.load_features()
            >>> self.load_samples()
            >>> data_key = 'learn(all)'
            >>> task_key = 'photobomb_state'
            >>> task_key = 'match_state'
            >>> clf_key = 'RF-OVR'
            >>> clf_key = 'RF'
        """
        X_df = self.samples.X_dict[data_key]
        labels = self.samples.subtasks[task_key]

        tup = clf_key.split('-')
        if len(tup) == 1:
            multiclass_wrapper = ut.identity
        else:
            multiclass_wrapper = {
                'OVR': sklearn.multiclass.OneVsRestClassifier,
                'OVO': sklearn.multiclass.OneVsOneClassifier,
            }[tup[1]]

        est_type = tup[0]
        if est_type in {'RF', 'RandomForest'}:
            est_class = sklearn.ensemble.RandomForestClassifier
        elif est_type in {'SVM', 'SVC'}:
            est_class = sklearn.svm.SVC
        else:
            raise KeyError('est_type = %s' % (est_type,))

        est_kw1, est_kw2, xval_kw = self.get_clf_params(est_type)
        est_params = ut.merge_dicts(est_kw1, est_kw2)

        def clf_new():
            return multiclass_wrapper(est_class(**est_params))

        clf_list = []
        res_list = []

        """ TODO: check xval label frequency """
        skf = sklearn.model_selection.StratifiedKFold(**xval_kw)
        skf_iter = skf.split(X=np.empty((len(self.samples), 0)),
                             y=self.samples.encoded_1d())
        skf_list = list(skf_iter)
        skf_prog = ut.ProgIter(skf_list, label='skf')
        for train_idx, test_idx in skf_prog:
            X_train = X_df.values[train_idx]
            y_train = labels.y_enc[train_idx]

            clf = clf_new()
            clf.fit(X_train, y_train)

            # Evaluate on testing data
            res = clf_helpers.ClfResult.make_single(clf, X_df, test_idx, labels)
            res_list.append(res)
            clf_list.append(clf)
        return clf_list, res_list

    def reduce_dataset_size(self):
        """
        Reduce the size of the dataset for development speed

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> self = OneVsOneProblem()
            >>> self.load_features()
            >>> self.load_samples()
            >>> self.reduce_dataset_size()
        """
        from six import next
        labels = next(iter(self.samples.subtasks.values()))
        ut.assert_eq(len(labels), len(self.samples), verbose=False)

        if True:
            # Remove singletons
            unique_aids = np.unique(self.samples.aid_pairs)
            nids = self.ibs.get_annot_nids(unique_aids)
            singleton_nids = set([nid for nid, v in ut.dict_hist(nids).items() if v == 1])
            nid_flags = [nid in singleton_nids for nid in nids]
            singleton_aids = set(ut.compress(unique_aids, nid_flags))
            mask = [not (a1 in singleton_aids or a2 in singleton_aids)
                     for a1, a2 in self.samples.aid_pairs]
            print('Removing %d pairs based on singleton' % (len(mask) - sum(mask)))
            self.samples = samples2 = self.samples.compress(mask)
            # samples2.print_info()
            # print('---------------')
            labels = next(iter(samples2.subtasks.values()))
            ut.assert_eq(len(labels), len(samples2), verbose=False)
            self.samples = samples2

        if True:
            # Remove anything 1vM didn't get
            mask = (self.samples.simple_scores['score_lnbnn_1vM'] > 0).values
            print('Removing %d pairs based on LNBNN failure' % (len(mask) - sum(mask)))
            self.samples = samples3 = self.samples.compress(mask)
            # samples3.print_info()
            # print('---------------')
            labels = next(iter(samples3.subtasks.values()))
            ut.assert_eq(len(labels), len(samples3), verbose=False)
            self.samples = samples3

        from sklearn.utils import random

        if False:
            # Choose labels to balance
            labels = self.samples.subtasks['match_state']
            unique_labels, groupxs = ut.group_indices(labels.y_enc)
            #
            # unique_labels, groupxs = ut.group_indices(self.samples.encoded_1d())

            # Take approximately the same number of examples from each class type
            n_take = int(np.round(np.median(list(map(len, groupxs)))))
            # rng = np.random.RandomState(0)
            rng = random.check_random_state(0)
            sample_idxs = [
                random.choice(idxs, min(len(idxs), n_take), replace=False,
                              random_state=rng)
                for idxs in groupxs
            ]
            idxs = sorted(ut.flatten(sample_idxs))
            mask = ut.index_to_boolmask(idxs, len(self.samples))
            print('Removing %d pairs for class balance' % (len(mask) - sum(mask)))
            self.samples = samples4 = self.samples.compress(mask)
            # samples4.print_info()
            # print('---------------')
            labels = next(iter(samples4.subtasks.values()))
            ut.assert_eq(len(labels), len(samples4), verbose=False)
            self.samples = samples4
            # print('hist(y) = ' + ut.repr4(self.samples.make_histogram()))

            # print('Reducing dataset size for class balance')
            # X = self.samples.X_dict['learn(all)']
            # Find the data with the most null / 0 values
            # nullness = (X == 0).sum(axis=1) + pd.isnull(X).sum(axis=1)
            # nullness = pd.isnull(X).sum(axis=1)
            # nullness = nullness.reset_index(drop=True)
            # false_nullness = (nullness)[~samples.is_same()]
            # sortx = false_nullness.argsort()[::-1]
            # false_nullness_ = false_nullness.iloc[sortx]
            # Remove a few to make training more balanced / faster
            # class_hist = self.samples.make_histogram()
            # num_remove = max(class_hist['match'] - class_hist['nomatch'], 0)
            # if num_remove > 0:
            #     to_remove = false_nullness_.iloc[:num_remove]
            #     mask = ~np.array(ut.index_to_boolmask(to_remove.index, len(self.samples)))
            #     self.samples = self.samples.compress(mask)
            #     print('hist(y) = ' + ut.repr4(self.samples.make_histogram()))

        # if 0:
        #     print('Random dataset size reduction for development')
        #     rng = np.random.RandomState(1851057325)
        #     num = len(self.samples)
        #     to_keep = rng.choice(np.arange(num), 1000)
        #     mask = np.array(ut.index_to_boolmask(to_keep, num))
        #     self.samples = self.samples.compress(mask)
        #     class_hist = self.samples.make_histogram()
        #     print('hist(y) = ' + ut.repr4(class_hist))
        labels = next(iter(self.samples.subtasks.values()))
        ut.assert_eq(len(labels), len(self.samples), verbose=False)

    def build_feature_subsets(self):
        """
        Try to identify a useful subset of features to reduce problem
        dimensionality
        """
        X_dict = self.samples.X_dict
        X = X_dict['learn(all)']
        featinfo = AnnotPairFeatInfo(X)
        # print('RAW FEATURE INFO (learn(all)):')
        # print(ut.indent(featinfo.get_infostr()))
        if False:
            # measures_ignore = ['weighted_lnbnn', 'lnbnn', 'weighted_norm_dist',
            #                    'fgweights']
            # Use only local features
            cols = featinfo.select_columns([
                ('measure_type', '==', 'local'),
                # ('local_sorter', 'in', ['weighted_ratio']),
                # ('local_measure', 'not in', measures_ignore),
            ])
            X_dict['learn(local)'] = featinfo.X[sorted(cols)]

        if False:
            # measures_ignore = ['weighted_lnbnn', 'lnbnn', 'weighted_norm_dist',
            #                    'fgweights']
            # Use only local features
            cols = featinfo.select_columns([
                ('measure_type', '==', 'local'),
                ('local_sorter', 'in', ['weighted_ratio']),
                # ('local_measure', 'not in', measures_ignore),
            ])
            X_dict['learn(local,1)'] = featinfo.X[sorted(cols)]

        if False:
            # Use only summary stats
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            X_dict['learn(sum)']  = featinfo.X[sorted(cols)]

        if True:
            # Use summary and global
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            cols.update(featinfo.select_columns([
                ('measure_type', '==', 'global'),
            ]))
            X_dict['learn(sum,glob)'] = featinfo.X[sorted(cols)]

            if True:
                # Remove view columns
                view_cols = featinfo.select_columns([
                    ('measure_type', '==', 'global'),
                    ('measure', 'in', ['yaw_1', 'yaw_2', 'yaw_delta']),
                ])
                cols = set.difference(cols, view_cols)
                X_dict['learn(sum,glob,-view)'] = featinfo.X[sorted(cols)]

        if True:
            # Only allow very specific summary features
            summary_cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
                ('summary_op', 'in', ['len']),
            ])
            summary_cols.update(featinfo.select_columns([
                ('measure_type', '==', 'summary'),
                ('summary_op', 'in', ['sum']),
                ('summary_measure', 'in', [
                    'weighted_ratio', 'ratio',
                    'norm_dist', 'weighted_norm_dist',
                    'fgweights',
                    'weighted_lnbnn_norm_dist', 'lnbnn_norm_dist',
                    'norm_y2', 'norm_y1',
                    # 'norm_x1', 'norm_x2',
                    'scale1', 'scale2',
                    # 'weighted_norm_dist',
                    # 'weighted_lnbnn_norm_dist',
                ]),
            ]))
            summary_cols.update(featinfo.select_columns([
                ('measure_type', '==', 'summary'),
                ('summary_op', 'in', ['mean']),
                ('summary_measure', 'in', [
                    'sver_err_xy', 'sver_err_ori',
                    # 'sver_err_scale',
                    'norm_y1', 'norm_y2',
                    'norm_x1', 'norm_x2',
                    'ratio',
                ]),
            ]))
            summary_cols.update(featinfo.select_columns([
                ('measure_type', '==', 'summary'),
                ('summary_op', 'in', ['std']),
                ('summary_measure', 'in', [
                    'norm_y1', 'norm_y2',
                    'norm_x1', 'norm_x2',
                    'scale1', 'scale2',
                    'sver_err_ori', 'sver_err_xy',
                    # 'sver_err_scale',
                    # 'match_dist',
                    'norm_dist', 'ratio'
                ]),
            ]))

            global_cols = featinfo.select_columns([
                ('measure_type', '==', 'global'),
                ('measure', 'not in', [
                    'gps_2[0]', 'gps_2[1]',
                    'gps_1[0]', 'gps_1[1]',
                    # 'time_1', 'time_2',
                ]),
                # NEED TO REMOVE YAW BECAUSE WE USE IT IN CONSTRUCTING LABELS
                ('measure', 'not in', [
                    'yaw_1', 'yaw_2', 'yaw_delta'
                ]),
            ])

            if 0:
                cols = set([])
                cols.update(summary_cols)
                cols.update(global_cols)
                X_dict['learn(sum,glob,3)'] = featinfo.X[sorted(cols)]

            if 0:
                cols = set([])
                cols.update(summary_cols)
                cols.update(global_cols)
                cols.update(featinfo.select_columns([
                    ('measure_type', '==', 'global'),
                    # Add yaw back in if not_comp is explicitly labeled
                    ('measure', 'in', [
                        'yaw_1', 'yaw_2', 'yaw_delta'
                    ]),
                ]))
                X_dict['learn(sum,glob,3,+view)'] = featinfo.X[sorted(cols)]

            # if 0:
            #     summary_cols_ = summary_cols.copy()
            #     summary_cols_ = [c for c in summary_cols_ if 'lnbnn' not in c]
            #     cols = set([])
            #     cols.update(summary_cols_)
            #     cols.update(global_cols)
            #     X_dict['learn(sum,glob,4)'] = featinfo.X[sorted(cols)]

            # if 0:
            #     cols = set([])
            #     cols.update(summary_cols)
            #     cols.update(global_cols)
            #     cols.update(featinfo.select_columns([
            #         ('measure_type', '==', 'local'),
            #         ('local_sorter', 'in', ['weighted_ratio', 'lnbnn_norm_dist']),
            #         ('local_measure', 'in', ['weighted_ratio']),
            #         ('local_rank', '<', 20),
            #         ('local_rank', '>', 0),
            #     ]))
            #     X_dict['learn(loc,sum,glob,5)'] = featinfo.X[sorted(cols)]
        self.samples.X_dict = X_dict

    def evaluate_simple_scores(self, task_keys=None):
        """
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> self = OneVsOneProblem()
            >>> self.set_pandas_options()
            >>> self.load_features()
            >>> self.load_samples()
            >>> self.evaluate_simple_scores()
        """
        score_dict = self.samples.simple_scores.copy()
        if True:
            # Remove scores that arent worth reporting
            for k in list(score_dict.keys())[:]:
                ignore = [
                    'sum(norm_x', 'sum(norm_y',
                    'sum(sver_err', 'sum(scale',
                    'sum(match_dist)',
                    'sum(weighted_norm_dist',
                ]
                if self.qreq_.qparams.featweight_enabled:
                    ignore.extend([
                        # 'sum(norm_dist)',
                        # 'sum(ratio)',
                        # 'sum(lnbnn)',
                        # 'sum(lnbnn_norm_dist)'
                    ])
                flags = [part in k for part in ignore]
                if any(flags):
                    del score_dict[k]

        if task_keys is None:
            task_keys = list(self.samples.subtasks.keys())

        simple_aucs = {}
        for task_key in task_keys:
            task_aucs = {}
            labels = self.samples.subtasks[task_key]
            for sublabels in labels.gen_one_vs_rest_labels():
                sublabel_aucs = {}
                for scoretype in score_dict.keys():
                    scores = score_dict[scoretype].values
                    auc = sklearn.metrics.roc_auc_score(sublabels.y_enc, scores)
                    sublabel_aucs[scoretype] = auc
                # task_aucs[sublabels.task_key] = sublabel_aucs
                task_aucs[sublabels.task_name.replace(task_key, '')] = sublabel_aucs
            simple_aucs[task_key] = task_aucs
        self.simple_aucs = simple_aucs

    def report_simple_scores(self, task_key):
        force_keep = ['score_lnbnn_1vM']
        simple_aucs = self.simple_aucs
        from utool.experimental.pandas_highlight import to_string_monkey
        n_keep = 6
        df_simple_auc = pd.DataFrame.from_dict(simple_aucs[task_key], orient='index')
        # Take only a subset of the columns that scored well in something
        rankings = df_simple_auc.values.argsort(axis=1).argsort(axis=1)
        rankings = rankings.shape[1] - rankings - 1
        ordered_ranks = np.array(vt.ziptake(rankings.T, rankings.argsort(axis=0).T)).T
        sortx = np.lexsort(ordered_ranks[::-1])
        keep_cols = df_simple_auc.columns[sortx][0:n_keep]
        extra = np.setdiff1d(force_keep, np.intersect1d(keep_cols, force_keep))
        keep_cols = keep_cols[:len(keep_cols) - len(extra)].tolist() + extra.tolist()
        # Now print them
        ut.cprint('\n[None] ROC-AUC of simple scoring measures for %s' % (task_key,), 'yellow')
        print(to_string_monkey(df_simple_auc[keep_cols], highlight_cols='all'))

    def report_classifier_importance(self, task_key, clf_key, data_key):
        ut.qt4ensure()
        import plottool as pt  # NOQA

        X = self.samples.X_dict[data_key]
        # Take average feature importance
        ut.cprint('MARGINAL IMPORTANCE INFO for %s on task %s' % (data_key, task_key), 'yellow')
        print(' Caption:')
        print(' * The NaN row ensures that `weight` always sums to 1')
        print(' * `num` indicates how many dimensions the row groups')
        print(' * `ave_w` is the average importance a single feature in the row')
        # with ut.Indenter('[%s] ' % (data_key,)):
        if True:
            clf_list = self.task_clfs[task_key][clf_key][data_key]
            feature_importances = np.mean([
                clf_.feature_importances_ for clf_ in clf_list
            ], axis=0)
            importances = ut.dzip(X.columns, feature_importances)

            featinfo = AnnotPairFeatInfo(X, importances)

            featinfo.print_margins('feature')
            featinfo.print_margins('measure_type')
            featinfo.print_margins('summary_op')
            featinfo.print_margins('summary_measure')
            featinfo.print_margins('global_measure')
            # featinfo.print_margins([('measure_type', '==', 'summary'),
            #                     ('summary_op', '==', 'sum')])
            # featinfo.print_margins([('measure_type', '==', 'summary'),
            #                     ('summary_op', '==', 'mean')])
            # featinfo.print_margins([('measure_type', '==', 'summary'),
            #                     ('summary_op', '==', 'std')])
            # featinfo.print_margins([('measure_type', '==', 'global')])
            featinfo.print_margins('local_measure')
            featinfo.print_margins('local_sorter')
            featinfo.print_margins('local_rank')
            # ut.fix_embed_globals()
            # pt.wordcloud(importances)


@ut.reloadable_class
class AnnotPairSamples(clf_helpers.MultiTaskSamples):
    """
    Manages the different ways to assign samples (i.e. feat-label pairs) to
    1-v-1 classification

    CommandLine:
        python -m ibeis.scripts.script_vsone AnnotPairSamples

    Example:
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> self = OneVsOneProblem()
        >>> self.load_features()
        >>> samples = AnnotPairSamples(self.ibs, self.raw_aid_pairs, self.raw_simple_scores)
        >>> print(samples)
        >>> samples.print_info()
    """
    def __init__(samples, ibs, aid_pairs, simple_scores, X_dict):
        super(AnnotPairSamples, samples).__init__()
        samples.ibs = ibs
        samples.aid_pairs = aid_pairs
        samples.simple_scores = simple_scores
        samples.X_dict = X_dict
        samples.annots1 = ibs.annots(aid_pairs.T[0], asarray=True)
        samples.annots2 = ibs.annots(aid_pairs.T[1], asarray=True)
        samples.n_samples = len(aid_pairs)
        samples.apply_multi_task_multi_label()
        # samples.apply_multi_task_binary_label()

    @ut.memoize
    def make_annotpair_vhashid(samples):
        qvuuids = samples.annots1.visual_uuids
        dvuuids = samples.annots2.visual_uuids
        vsone_uuids = [
            ut.combine_uuids(uuids)
            for uuids in ut.ProgIter(zip(qvuuids, dvuuids), length=len(qvuuids),
                                     label='hashing ids')
        ]
        annot_pair_visual_hashid = ut.hashstr_arr27(vsone_uuids, '', pathsafe=True)
        return annot_pair_visual_hashid

    def compress(samples, flags):
        assert len(flags) == len(samples), 'mask has incorrect size'
        aid_pairs = samples.aid_pairs.compress(flags, axis=0)
        simple_scores = samples.simple_scores[flags]
        X_dict = ut.map_vals(lambda val: val[flags], samples.X_dict)
        ibs = samples.ibs
        new_labels = AnnotPairSamples(ibs, aid_pairs, simple_scores, X_dict)
        return new_labels

    @ut.memoize
    def is_same(samples):
        is_same = samples.annots1.nids == samples.annots2.nids
        samples.simple_scores
        return is_same

    @ut.memoize
    def is_photobomb(samples):
        am_rowids = samples.ibs.get_annotmatch_rowid_from_edges(samples.aid_pairs)
        am_tags = samples.ibs.get_annotmatch_case_tags(am_rowids)
        is_pb = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
        return is_pb

    @ut.memoize
    def is_comparable(samples):
        # If we don't have actual comparability information just guess
        # Start off by guessing
        is_comp_guess = samples.guess_if_comparable()
        is_comp = is_comp_guess.copy()

        # But use information that we have
        am_rowids = samples.ibs.get_annotmatch_rowid_from_edges(samples.aid_pairs)
        truths = np.array(ut.replace_nones(samples.ibs.get_annotmatch_truth(am_rowids), np.nan))
        is_notcomp_have = truths == samples.ibs.const.TRUTH_NOT_COMP
        is_comp_have = (truths == samples.ibs.const.TRUTH_MATCH) | (truths == samples.ibs.const.TRUTH_NOT_MATCH)
        is_comp[is_notcomp_have] = False
        is_comp[is_comp_have] = True
        # num_guess = (~(is_notcomp_have  | is_comp_have)).sum()
        # num_have = len(is_notcomp_have) - num_guess
        return is_comp

    def guess_if_comparable(samples):
        """
        Takes a guess as to which annots are not comparable based on scores and
        viewpoints. If either viewpoints is null assume they are comparable.
        """
        simple_scores = samples.simple_scores
        key = 'sum(weighted_ratio)'
        if key not in simple_scores:
            key = 'sum(ratio)'
        scores = simple_scores[key].values
        yaws1 = samples.annots1.yaws_asfloat
        yaws2 = samples.annots2.yaws_asfloat
        dists = vt.ori_distance(yaws1, yaws2)
        tau = np.pi * 2
        is_comp = (scores > .1) | (dists < tau / 8.1) | np.isnan(dists)
        return is_comp

    def apply_multi_task_multi_label(samples):
        # multioutput-multiclass / multi-task
        tasks_to_indicators = ut.odict([
            ('match_state', ut.odict([
                ('nomatch', ~samples.is_same() & samples.is_comparable()),
                ('match',    samples.is_same() & samples.is_comparable()),
                ('notcomp', ~samples.is_comparable()),
            ])),
            ('photobomb_state', ut.odict([
                ('notpb', ~samples.is_photobomb()),
                ('pb',   samples.is_photobomb()),
            ]))
        ])
        samples.apply_indicators(tasks_to_indicators)

    def apply_multi_task_binary_label(samples):
        # multioutput-multiclass / multi-task
        tasks_to_indicators = ut.odict([
            ('same_state', ut.odict([
                ('notsame', ~samples.is_same()),
                ('same',     samples.is_same())
                # ('nomatch', ~samples.is_same() | ~samples.is_comparable()),
                # ('match',    samples.is_same() & samples.is_comparable()),
            ])),
            ('photobomb_state', ut.odict([
                ('notpb', ~samples.is_photobomb()),
                ('pb',   samples.is_photobomb()),
            ]))
        ])
        samples.apply_indicators(tasks_to_indicators)

    def apply_single_task_multi_label(samples):
        is_comp = samples.is_comparable()
        is_same = samples.is_same()
        is_pb   = samples.is_photobomb()
        tasks_to_indicators = ut.odict([
            ('match_pb_state', ut.odict([
                ('is_notcomp',       ~is_comp & ~is_pb),
                ('is_match',          is_same & is_comp & ~is_pb),
                ('is_nomatch',       ~is_same & is_comp & ~is_pb),
                ('is_notcomp_pb',    ~is_comp & is_pb),
                ('is_match_pb',       is_same & is_comp & is_pb),
                ('is_nomatch_pb',    ~is_same & is_comp & is_pb),
            ])),
        ])
        samples.apply_indicators(tasks_to_indicators)


@ut.reloadable_class
class AnnotPairFeatInfo(object):
    """
    Used to compute marginal importances over groups of features used in the
    pairwise one-vs-one scoring algorithm
    """
    def __init__(featinfo, X, importances=None):
        featinfo.X = X
        featinfo.importances = importances
        featinfo._summary_keys = ['sum', 'mean', 'med', 'std', 'len']

    def select_columns(featinfo, criteria, op='and'):
        if op == 'and':
            cols = set(featinfo.X.columns)
            update = cols.intersection_update
        elif op == 'or':
            cols = set([])
            update = cols.update
        else:
            raise Exception(op)
        for group_id, op, value in criteria:
            found = featinfo.find(group_id, op, value)
            update(found)
        return cols

    def find(featinfo, group_id, op, value):
        import six
        if isinstance(op, six.text_type):
            opdict = ut.get_comparison_operators()
            op = opdict.get(op)
        grouper = getattr(featinfo, group_id)
        found = []
        for col in featinfo.X.columns:
            value1 = grouper(col)
            if value1 is None:
                # Only filter out/in comparable things
                found.append(col)
            else:
                try:
                    if value1 is not None:
                        if isinstance(value, int):
                            value1 = int(value1)
                        elif isinstance(value, list):
                            if len(value) > 0 and isinstance(value[0], int):
                                value1 = int(value1)
                    if op(value1, value):
                        found.append(col)
                except:
                    pass
        return found

    def group_importance(featinfo, item):
        name, keys = item
        num = len(keys)
        weight = sum(ut.take(featinfo.importances, keys))
        ave_w = weight / num
        tup = ave_w, weight, num
        # return tup
        df = pd.DataFrame([tup], columns=['ave_w', 'weight', 'num'],
                          index=[name])
        return df

    def print_margins(featinfo, group_id, ignore_trivial=True):
        X = featinfo.X
        if isinstance(group_id, list):
            cols = featinfo.select_columns(criteria=group_id)
            _keys = [(c, [c]) for c in cols]
            try:
                _weights = pd.concat(ut.lmap(featinfo.group_importance, _keys))
            except ValueError:
                _weights = []
                pass
            nice = str(group_id)
        else:
            grouper = getattr(featinfo, group_id)
            _keys = ut.group_items(X.columns, ut.lmap(grouper, X.columns))
            _weights = pd.concat(ut.lmap(featinfo.group_importance, _keys.items()))
            nice = ut.get_funcname(grouper).replace('_', ' ')
            nice = ut.pluralize(nice)
        try:
            _weights = _weights.iloc[_weights['ave_w'].argsort()[::-1]]
        except Exception:
            pass
        if not ignore_trivial or len(_weights) > 1:
            ut.cprint('\nMarginal importance of ' + nice, 'white')
            print(_weights)

    def group_counts(featinfo, item):
        name, keys = item
        num = len(keys)
        tup = (num,)
        # return tup
        df = pd.DataFrame([tup], columns=['num'], index=[name])
        return df

    def print_counts(featinfo, group_id):
        X = featinfo.X
        grouper = getattr(featinfo, group_id)
        _keys = ut.group_items(X.columns, ut.lmap(grouper, X.columns))
        _weights = pd.concat(ut.lmap(featinfo.group_counts, _keys.items()))
        _weights = _weights.iloc[_weights['num'].argsort()[::-1]]
        nice = ut.get_funcname(grouper).replace('_', ' ')
        nice = ut.pluralize(nice)
        print('\nCounts of ' + nice)
        print(_weights)

    def measure(featinfo, key):
        return key[key.find('(') + 1:-1]

    def feature(featinfo, key):
        return key

    def measure_type(featinfo, key):
        if key.startswith('global'):
            return 'global'
        if key.startswith('loc'):
            return 'local'
        if any(key.startswith(p) for p in featinfo._summary_keys):
            return 'summary'

    def summary_measure(featinfo, key):
        if any(key.startswith(p) for p in featinfo._summary_keys):
            return featinfo.measure(key)

    def local_measure(featinfo, key):
        if key.startswith('loc'):
            return featinfo.measure(key)

    def global_measure(featinfo, key):
        if key.startswith('global'):
            return featinfo.measure(key)

    def summary_op(featinfo, key):
        for p in featinfo._summary_keys:
            if key.startswith(p):
                return key[0:key.find('(')]

    def local_sorter(featinfo, key):
        if key.startswith('loc'):
            return key[key.find('[') + 1:key.find(',')]

    def local_rank(featinfo, key):
        if key.startswith('loc'):
            return key[key.find(',') + 1:key.find(']')]

    def get_infostr(featinfo):
        """
        Summarizes the types (global, local, summary) of features in X based on
        standardized dimension names.
        """
        grouped_keys = ut.ddict(list)
        for key in featinfo.X.columns:
            type_ = featinfo.measure_type(key)
            grouped_keys[type_].append(key)

        info_items = ut.odict([
            ('global_measures', ut.lmap(featinfo.global_measure,
                                        grouped_keys['global'])),

            ('local_sorters', set(map(featinfo.local_sorter,
                                       grouped_keys['local']))),
            ('local_ranks', set(map(featinfo.local_rank,
                                     grouped_keys['local']))),
            ('local_measures', set(map(featinfo.local_measure,
                                        grouped_keys['local']))),

            ('summary_measures', set(map(featinfo.summary_measure,
                                          grouped_keys['summary']))),
            ('summary_ops', set(map(featinfo.summary_op,
                                     grouped_keys['summary']))),
        ])

        import textwrap
        def _wrap(list_):
            unwrapped = ', '.join(sorted(list_))
            indent = (' ' * 4)
            lines_ = textwrap.wrap(unwrapped, width=80 - len(indent))
            lines = ['    ' + line for line in lines_]
            return lines

        lines = []
        for item  in info_items.items():
            key, list_ = item
            if len(list_):
                title = key.replace('_', ' ').title()
                if key.endswith('_measures'):
                    groupid = key.replace('_measures', '')
                    num = len(grouped_keys[groupid])
                    title = title + ' (%d)' % (num,)
                lines.append(title + ':')
                if key == 'summary_measures':
                    other = info_items['local_measures']
                    if other.issubset(list_) and len(other) > 0:
                        remain = list_ - other
                        lines.extend(_wrap(['<same as local_measures>'] + list(remain)))
                    else:
                        lines.extend(_wrap(list_))
                else:
                    lines.extend(_wrap(list_))

        infostr = '\n'.join(lines)
        return infostr
        # print(infostr)


def build_features(qreq_, hyper_params):
    """
    Cached output of one-vs-one matches

    Example:
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> self = OneVsOneProblem()
        >>> qreq_ = self.qreq_
        >>> hyper_params = self.hyper_params
    """
    import pandas as pd
    import vtool as vt
    import ibeis

    # ==================================
    # Compute or load one-vs-one results
    # ==================================
    # Get a set of training pairs
    ibs = qreq_.ibs
    cm_list = qreq_.execute()
    infr = ibeis.AnnotInference.from_qreq_(qreq_, cm_list, autoinit=True)

    # Per query choose a set of correct, incorrect, and random training pairs
    aid_pairs_ = infr._cm_training_pairs(rng=np.random.RandomState(42),
                                         **hyper_params.pair_sample)
    pb_aid_pairs = photobomb_samples(ibs)
    # TODO: try to add in more non-comparable samples
    aid_pairs = pb_aid_pairs + aid_pairs_
    aid_pairs = ut.lmap(tuple, vt.unique_rows(np.array(aid_pairs)).tolist())
    # Keep only a random subset
    assert hyper_params.subsample is None

    # NEW WAY TO DO MATCHING
    config = hyper_params.vsone_assign
    # TODO: ensure feat/chip configs are resepected
    matches = infr.exec_vsone_subset(aid_pairs, config=config)

    # Ensure matches know about relavent metadata
    for match in matches:
        vt.matching.ensure_metadata_normxy(match.annot1)
        vt.matching.ensure_metadata_normxy(match.annot2)
    global_keys = ['yaw', 'qual', 'gps', 'time']
    for match in ut.ProgIter(matches, label='setup globals'):
        match.add_global_measures(global_keys)
    for match in ut.ProgIter(matches, label='setup locals'):
        match.add_local_measures()

    if True:
        # augment match correspondences with LNBNN scores
        qreq_.load_indexer()
        matches = batch_apply_lnbnn(matches, qreq_, inplace=True)

    # Pass back just one match to play with
    for match in matches:
        if len(match.fm) > 10:
            break

    aid_pairs = np.array([(m.annot1['aid'], m.annot2['aid']) for m in matches])
    # ---------------
    # Grab some simple scores
    simple_scores = pd.DataFrame([
        m._make_local_summary_feature_vector(sum=True, mean=False, std=False)
        for m in matches])
    ADD_VSMANY_LNBNN_TO_SIMPLE_SCORES = True
    if ADD_VSMANY_LNBNN_TO_SIMPLE_SCORES:
        # Ensure that all annots exist in the graph
        expt_aids = ut.unique(ut.flatten(aid_pairs))
        infr.add_aids(expt_aids)
        infr.graph.add_edges_from(aid_pairs)
        # TEST ORIGINAL LNBNN SCORE SEP
        infr.apply_match_scores()
        edge_data = [infr.graph.get_edge_data(u, v) for u, v in aid_pairs]
        lnbnn_score_list = [0 if d is None else d.get('score', 0)
                            for d in edge_data]
        lnbnn_score_list = np.nan_to_num(lnbnn_score_list)
        simple_scores = simple_scores.assign(score_lnbnn_1vM=lnbnn_score_list)
    simple_scores[pd.isnull(simple_scores)] = 0

    # ---------------
    sorters = [
        'ratio',
        # 'lnbnn', 'lnbnn_norm_dist',
        'norm_dist', 'match_dist'
    ]
    if qreq_.qparams.featweight_enabled:
        sorters += [
            'weighted_ratio',
            # 'weighted_lnbnn',
        ]

    measures = list(match.local_measures.keys())
    pairfeat_cfg = hyper_params.pairwise_feats

    # Try different feature constructions
    print('Building pairwise features')
    pairwise_feats = pd.DataFrame([
        m.make_feature_vector(sorters=sorters, keys=measures, **pairfeat_cfg)
        for m in ut.ProgIter(matches, label='making pairwise feats')
    ])
    pairwise_feats[pd.isnull(pairwise_feats)] = np.nan
    X_all = pairwise_feats.copy()
    X_dict = {'learn(all)': X_all}
    return aid_pairs, simple_scores, X_dict, match


def demo_single_pairwise_feature_vector():
    r"""
    CommandLine:
        python -m ibeis.scripts.script_vsone demo_single_pairwise_feature_vector

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> match = demo_single_pairwise_feature_vector()
        >>> print(match)
    """
    import vtool as vt
    import ibeis
    ibs = ibeis.opendb('testdb1')
    qaid, daid = 1, 2
    annot1 = ibs.annots([qaid])[0]._make_lazy_dict()
    annot2 = ibs.annots([daid])[0]._make_lazy_dict()

    vt.matching.ensure_metadata_normxy(annot1)
    vt.matching.ensure_metadata_normxy(annot2)

    match = vt.PairwiseMatch(annot1, annot2)
    cfgdict = {'checks': 200, 'symmetric': False}
    match.assign(cfgdict=cfgdict)
    match.apply_ratio_test({'ratio_thresh': .638}, inplace=True)
    match.apply_sver(inplace=True)

    match.add_global_measures(['yaw', 'qual', 'gps', 'time'])
    match.add_local_measures()

    # sorters = ['ratio', 'norm_dist', 'match_dist']
    match.make_feature_vector()
    return match


def batch_apply_lnbnn(matches, qreq_, inplace=False):
    from ibeis.algo.hots import nn_weights
    indexer = qreq_.indexer
    if not inplace:
        matches_ = [match.copy() for match in matches]
    else:
        matches_ = matches
    K = qreq_.qparams.K
    Knorm = qreq_.qparams.Knorm
    normalizer_rule  = qreq_.qparams.normalizer_rule

    print('Stacking vecs for batch matching')
    offset_list = np.cumsum([0] + [match_.fm.shape[0] for match_ in matches_])
    stacked_vecs = np.vstack([
        match_.matched_vecs2()
        for match_ in ut.ProgIter(matches_, lablel='stacking matched vecs')
    ])

    vecs = stacked_vecs
    num = (K + Knorm)
    idxs, dists = indexer.batch_knn(vecs, num, chunksize=8192,
                                    label='lnbnn scoring')

    idx_list = [idxs[l:r] for l, r in ut.itertwo(offset_list)]
    dist_list = [dists[l:r] for l, r in ut.itertwo(offset_list)]
    iter_ = zip(matches_, idx_list, dist_list)
    prog = ut.ProgIter(iter_, nTotal=len(matches_), label='lnbnn scoring')
    for match_, neighb_idx, neighb_dist in prog:
        qaid = match_.annot2['aid']
        norm_k = nn_weights.get_normk(qreq_, qaid, neighb_idx, Knorm, normalizer_rule)
        ndist = vt.take_col_per_row(neighb_dist, norm_k)
        vdist = match_.local_measures['match_dist']
        lnbnn_dist = nn_weights.lnbnn_fn(vdist, ndist)
        lnbnn_clip_dist = np.clip(lnbnn_dist, 0, np.inf)
        match_.local_measures['lnbnn_norm_dist'] = ndist
        match_.local_measures['lnbnn'] = lnbnn_dist
        match_.local_measures['lnbnn_clip'] = lnbnn_clip_dist
        match_.fs = lnbnn_dist
    return matches_


def photobomb_samples(ibs):
    """
    import ibeis
    ibs = ibeis.opendb('PZ_Master1')
    """
    # all_annots = ibs.annots()
    am_rowids = ibs._get_all_annotmatch_rowids()
    am_tags = ibs.get_annotmatch_case_tags(am_rowids)

    # ut.dict_hist(ut.flatten(am_tags))
    am_flags = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
    am_rowids_ = ut.compress(am_rowids, am_flags)
    aids1 = ibs.get_annotmatch_aid1(am_rowids_)
    aids2 = ibs.get_annotmatch_aid2(am_rowids_)

    if False:
        a1 = ibs.annots(aids1, asarray=True)
        a2 = ibs.annots(aids2, asarray=True)
        flags = a1.nids == a2.nids
        a1_ = a1.compress(flags)
        a2_ = a2.compress(flags)
        import guitool as gt
        ut.qt4ensure()
        gt.ensure_qapp()
        from vtool import inspect_matches
        import vtool as vt
        i = 1
        annot1 = a1_[i]._make_lazy_dict()
        annot2 = a2_[i]._make_lazy_dict()

        def on_context():
            from ibeis.gui import inspect_gui
            return inspect_gui.make_annotpair_context_options(
                ibs, annot1['aid'], annot1['aid'], None)

        match = vt.PairwiseMatch(annot1, annot2)
        self = inspect_matches.MatchInspector(match=match,
                                              on_context=on_context)
        self.show()
    return list(zip(aids1, aids2))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts.script_vsone
        python -m ibeis.scripts.script_vsone --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
