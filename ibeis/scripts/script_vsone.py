# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import numpy as np
import vtool as vt
import dtool as dt
from six.moves import zip, range  # NOQA
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
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


class OneVsOneProblem(object):
    """
    Keeps information about the one-vs-one pairwise classification problem

    Example:
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> self = OneVsOneProblem()
        >>> self.load_features()
    """
    def __init__(self):
        import ibeis
        # ut.aug_sysargv('--db PZ_Master1')
        qreq_ = ibeis.testdata_qreq_(
            defaultdb='PZ_PB_RF_TRAIN',
            a=':mingt=4,species=primary',
            # t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum',
            t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum,QRH=True',
        )
        assert qreq_.qparams.can_match_samename is True
        assert qreq_.qparams.prescore_method == 'csum'
        self.qreq_ = qreq_
        self.ibs = qreq_.ibs

    def evaluate_classifiers(self):
        """
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
        import pandas as pd
        # pd.options.display.max_rows = 10
        pd.options.display.max_rows = 80
        pd.options.display.max_columns = 40
        pd.options.display.width = 160

        self.load_features()
        self.load_labels()

        # self.labels.print_info()
        self.reduce_dataset_size()
        self.build_feature_subsets()
        self.labels.print_info()

        # _all_dfs = []
        # self.load_multiclass_scores()
        # df_simple = self.evaluate_simple_scores()
        # _all_dfs.append(df_simple)

        self.task_clfs = ut.ddict(dict)
        self.task_res_list = ut.ddict(dict)
        self.task_combo_res = ut.ddict(dict)

        task_list = [
            'match_state',
            # 'photobomb_state'
        ]

        self.datakey_list = [
            'learn(sum,glob,3)',
        ]

        cacher = ut.Cacher('pair_clf_v2', cfgstr='tmp' + self.qreq_.get_cfgstr() + str(task_list),
                           appname='vsone_rf_train', enabled=1)
        data = cacher.tryload()
        if not data:
            task_prog = ut.ProgIter(task_list, label='task')
            for task_name in task_prog:
                task_prog.ensure_newline()
                self.evaluate_task(task_name)
            data = (self.task_clfs, self.task_res_list, self.task_combo_res)
            cacher.save(data)
        else:
            self.task_clfs, self.task_res_list, self.task_combo_res = data

        for task_name in task_list:
            self.report_classifier_accuracy(task_name)
            # self.report_classifier_importance(task_name)

        for task_name in task_list:
            data_combo_res = self.task_combo_res[task_name]
            best_data_key = 'learn(sum,glob,3)'
            combo_res = data_combo_res[best_data_key]
            combo_res.print_report()

        # TODO: view failure / success cases
        # Need to show and potentially fix misclassified examples
        if True:
            self.labels.aid_pairs
            combo_res.target_bin_df
            res = combo_res
            labels = self.labels
            meta = res.make_meta(labels).copy()
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

            win = infr.start_qt_interface()
            import guitool as gt
            gt.qtapp_loop(qwin=win, freq=10)

        # _all_dfs.append(df_rf)
        # df_all = pd.concat(_all_dfs, axis=1)

        # # Add in the simple scores
        # from utool.experimental.pandas_highlight import to_string_monkey
        # print(to_string_monkey(df_all, highlight_cols=np.arange(len(df_all.columns))))

        # self.report_classifier_importance()

        # best_name = df_all.columns[df_all.values.argmax()]
        # pt.show_if_requested()
        # import utool
        # utool.embed()
        # print('rat_sver_rf_auc = %r' % (rat_sver_rf_auc,))
        # columns = ['Method', 'AUC']
        # data = [
        #     ['1vM-LNBNN',       vsmany_lnbnn_auc],
        #     ['1v1-LNBNN',       vsone_sver_lnbnn_auc],
        #     ['1v1-RAT',         rat_auc],
        #     ['1v1-RAT+SVER',    rat_sver_auc],
        #     ['1v1-RAT+SVER+RF', rat_sver_rf_auc],
        # ]
        # table = pd.DataFrame(data, columns=columns)
        # error = 1 - table['AUC']
        # orig = 1 - vsmany_lnbnn_auc
        # import tabulate
        # table = table.assign(percent_error_decrease=(orig - error) / orig * 100)
        # col_to_nice = {
        #     'percent_error_decrease': '% error decrease',
        # }
        # header = [col_to_nice.get(c, c) for c in table.columns]
        # print(tabulate.tabulate(table.values, header, tablefmt='orgtbl'))

    def evaluate_task(self, task_name):
        """
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --show

        TODO: use Markedness and Informedness
        http://www.flinders.edu.au/science_engineering/fms/School-CSEM/publications/tech_reps-research_artfcts/TRRA_2007.pdf
        """
        X_dict = self.X_dict
        task_labels = self.labels.subtasks[task_name]
        dataset_prog = ut.ProgIter(self.datakey_list, label='X_set')
        for name in dataset_prog:
            X_df = X_dict[name]
            clf_list, res_list = self.evaluate_single_rf(X_df, task_labels)

            print('\nDataset Name = %r' % (name,))
            print('Task Name = %r' % (task_name,))
            combo_res = ClfResult.combine_results(res_list, task_labels)
            # combo_res.print_report()

            self.task_clfs[task_name][name]      = clf_list
            self.task_res_list[task_name][name]  = res_list
            self.task_combo_res[task_name][name] = combo_res

    def evaluate_single_rf(self, X_df, task_labels):
        """
        X_df = self.X_dict['learn(all)']
        task_labels = self.labels.subtasks['photobomb_state']
        """
        rf_params = self.get_rf_params()

        clf_list = []
        res_list = []

        skf_list = self.get_crossval_idxs()
        for train_idx, test_idx in ut.ProgIter(skf_list, label='skf'):
            X_train = X_df.values[train_idx]
            y_train = task_labels.y_enc[train_idx]

            clf = sklearn.ensemble.RandomForestClassifier(**rf_params)
            clf.fit(X_train, y_train)

            # Evaluate on testing data
            X_test = X_df.values[test_idx]
            clf_probs = clf.predict_proba(X_test)

            pred_classes = clf.classes_

            res = ClfResult.make_single(test_idx, clf_probs, pred_classes, task_labels)
            res_list.append(res)
            clf_list.append(clf)
        return clf_list, res_list

    def load_labels(self):
        self.labels = PairLabels(self.ibs, self.aid_pairs, self.simple_scores)

    def load_features(self):
        qreq_ = self.qreq_
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

        dbname = qreq_.ibs.get_dbname()
        vsmany_hashid = qreq_.get_cfgstr(hash_pipe=True, with_input=True)
        features_hashid = ut.hashstr27(vsmany_hashid + hyper_params.get_cfgstr())
        cfgstr = '_'.join(['devcache', str(dbname), features_hashid])

        cacher = ut.Cacher('pairwise_data_v5', cfgstr=cfgstr,
                           appname='vsone_rf_train', enabled=1)
        data = cacher.tryload()
        if not data:
            data = build_features(qreq_, hyper_params)
        cacher.save(data)

        aid_pairs, simple_scores, X_dict, y, match = data
        self.aid_pairs = aid_pairs
        self.raw_X_dict = X_dict
        self.simple_scores = simple_scores
        self.match = match

    def load_multiclass_scores(self):
        # convert simple scores to multiclass scores
        import vtool as vt
        self.multiclass_scores = {}
        for key in self.simple_scores.keys():
            scores = self.simple_scores[key].values
            # Hack scores into the range 0 to 1
            normer = vt.ScoreNormalizer(adjust=8, monotonize=True)
            normer.fit(scores, y=self.labels.is_same())
            normed_scores = normer.normalize_scores(scores)
            # Create a dimension for each class
            # but only populate two of the dimensions
            class_idxs = ut.take(self.labels.text_to_class, ['nomatch', 'match'])
            pred = np.zeros((len(scores), len(self.labels.class_names)))
            pred[:, class_idxs[0]] = 1 - normed_scores
            pred[:, class_idxs[1]] = normed_scores
            self.multiclass_scores[key] = pred

    def reduce_dataset_size(self):
        """
        Reduce the size of the dataset for development speed

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> self = OneVsOneProblem()
            >>> self.load_features()
            >>> self.load_labels()
        """
        raw_X_dict = self.raw_X_dict
        X = raw_X_dict['learn(all)']

        if True:
            labels = self.labels
            # Remove singletons
            unique_aids = np.unique(labels.aid_pairs)
            nids = self.ibs.get_annot_nids(unique_aids)
            singleton_nids = set([nid for nid, v in ut.dict_hist(nids).items() if v == 1])
            nid_flags = [nid in singleton_nids for nid in nids]
            singleton_aids = set(ut.compress(unique_aids, nid_flags))
            flags = [not (a1 in singleton_aids or a2 in singleton_aids)
                     for a1, a2 in labels.aid_pairs]
            self.labels = labels.compress(flags)

        if False:
            # Remove anything 1vM didn't get
            mask = (self.simple_scores['score_lnbnn_1vM'] > 0).values
            self.labels = self.labels.compress(mask)
            simple_scores = self.simple_scores[mask]
            self.labels.print_info()
            X = X[mask]
            raw_X_dict['learn(all)'] = X

        if False:
            print('Reducing dataset size for class balance')
            labels = self.labels
            # Find the data with the most null / 0 values
            # nullness = (X == 0).sum(axis=1) + pd.isnull(X).sum(axis=1)
            nullness = pd.isnull(X).sum(axis=1)
            nullness = nullness.reset_index(drop=True)
            false_nullness = (nullness)[~labels.is_same()]
            sortx = false_nullness.argsort()[::-1]
            false_nullness_ = false_nullness.iloc[sortx]
            # Remove a few to make training more balanced / faster
            class_hist = labels.make_histogram()
            num_remove = max(class_hist['match'] - class_hist['nomatch'], 0)
            if num_remove > 0:
                to_remove = false_nullness_.iloc[:num_remove]
                mask = ~np.array(ut.index_to_boolmask(to_remove.index, len(labels)))
                self.labels = self.labels.compress(mask)
                simple_scores = simple_scores[mask]
                print('hist(y) = ' + ut.repr4(labels.make_histogram()))
                X = X[mask]
                raw_X_dict['learn(all)'] = X

        if 0:
            print('Reducing dataset size for development')
            rng = np.random.RandomState(1851057325)
            num = len(self.labels)
            to_keep = rng.choice(np.arange(num), 1000)
            mask = np.array(ut.index_to_boolmask(to_keep, num))
            self.labels = self.labels.compress(mask)
            simple_scores = simple_scores[mask]
            class_hist = self.labels.make_histogram()
            print('hist(y) = ' + ut.repr4(class_hist))
            X = X[mask]
            raw_X_dict['learn(all)'] = X

    def build_feature_subsets(self):
        """
        Try to identify a useful subset of features to reduce problem
        dimensionality
        """
        X_dict = self.raw_X_dict.copy()
        X = X_dict['learn(all)']
        featinfo = PairFeatInfo(X)
        if 1:
            measures_ignore = ['weighted_lnbnn', 'lnbnn', 'weighted_norm_dist',
                               'fgweights']
            # Use only local features
            cols = featinfo.select_columns([
                ('measure_type', '==', 'local'),
                ('local_sorter', 'in', ['weighted_ratio']),
                ('local_measure', 'not in', measures_ignore),
            ])
            X_dict['learn(local)'] = featinfo.X[sorted(cols)]

        if False:
            # Use only summary stats
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            X_dict['learn(sum)']  = featinfo.X[sorted(cols)]

        if 0:
            # Use summary and global
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            cols.update(featinfo.select_columns([
                ('measure_type', '==', 'global'),
            ]))
            X_dict['learn(sum,glob)'] = featinfo.X[sorted(cols)]

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

            if False:
                cols = set([])
                cols.update(summary_cols)
                cols.update(featinfo.select_columns([
                    ('measure_type', '==', 'global'),
                ]))
                X_dict['learn(sum,glob,2)'] = featinfo.X[sorted(cols)]

            if 1:
                cols = set([])
                cols.update(summary_cols)
                cols.update(global_cols)
                X_dict['learn(sum,glob,3)'] = featinfo.X[sorted(cols)]

            if 0:
                summary_cols_ = summary_cols.copy()
                summary_cols_ = [c for c in summary_cols_ if 'lnbnn' not in c]
                cols = set([])
                cols.update(summary_cols_)
                cols.update(global_cols)
                X_dict['learn(sum,glob,4)'] = featinfo.X[sorted(cols)]

            if 0:
                cols = set([])
                cols.update(summary_cols)
                cols.update(global_cols)
                cols.update(featinfo.select_columns([
                    ('measure_type', '==', 'local'),
                    ('local_sorter', 'in', ['weighted_ratio', 'lnbnn_norm_dist']),
                    ('local_measure', 'in', ['weighted_ratio']),
                    ('local_rank', '<', 20),
                    ('local_rank', '>', 0),
                ]))
                X_dict['learn(loc,sum,glob,5)'] = featinfo.X[sorted(cols)]
        self.X_dict = X_dict

    def evaluate_simple_scores(self):
        score_dict = self.multiclass_scores.copy()
        if False:
            # Remove scores that arent worth reporting
            for k in list(score_dict.keys())[:]:
                ignore = [
                    'sum(norm_x', 'sum(norm_y',
                    'sum(sver_err', 'sum(scale', 'sum(match_dist)',
                ]
                if self.qreq_.qparams.featweight_enabled:
                    ignore.extend(['sum(norm_dist)', 'sum(ratio)', 'sum(lnbnn)',
                                   'sum(lnbnn_norm_dist)'])
                flags = [part in k for part in ignore]
                if any(flags):
                    del score_dict[k]

        # Sort AUC by values
        simple_aucs = pd.DataFrame(dict([
            (k, [sklearn.metrics.roc_auc_score(self.labels.y_bin, score_dict[k])])
            for k in score_dict.keys()
        ]))
        simple_auc_dict = ut.dzip(simple_aucs.columns, simple_aucs.values[0])
        simple_auc_dict = ut.sort_dict(simple_auc_dict, 'vals', reverse=True)

        # Simple printout of aucs
        # we dont need cross validation because there is no learning here
        print('\nAUC of simple scoring measures:')
        print(ut.align(ut.repr4(simple_auc_dict, precision=8), ':'))

        simple_auc_dict.values()
        simple_keys = list(simple_auc_dict.keys())
        simple_vals = list(simple_auc_dict.values())
        idxs = ut.argsort(list(simple_auc_dict.values()))[::-1][0:2]
        idx_ = simple_keys.index('score_lnbnn_1vM')
        if idx_ in idxs:
            idxs.remove(idx_)
        idxs = [idx_] + idxs
        best_simple_cols = ut.take(simple_keys, idxs)
        best_simple_aucs = ut.take(simple_vals, idxs)
        df_simple = pd.DataFrame([best_simple_aucs], columns=best_simple_cols)
        return df_simple

    def get_rf_params(self):
        rf_params = {
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
        rf_settings = {
            'random_state': 3915904814,
            'verbose': 0,
            'n_jobs': -1,
        }
        rf_params.update(rf_settings)
        return rf_params

    def get_crossval_idxs(self):
        """
        # TODO: check xval label frequency
        """
        # ---------------
        # Setup cross-validation
        labels = self.labels
        # xvalkw = dict(n_splits=10, shuffle=True,
        xvalkw = dict(n_splits=3, shuffle=True,
                      random_state=np.random.RandomState(42))
        skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
        skf_iter = skf.split(X=np.empty((len(labels), 0)), y=labels.encoded_1d())
        skf_list = list(skf_iter)
        return skf_list

    def report_classifier_importance(self, task_name):
        ut.qt4ensure()
        import plottool as pt  # NOQA

        for name, X in self.X_dict.items():
            # if name != best_name:
            #     continue
            # Take average feature importance
            print('IMPORTANCE INFO FOR %s DATA FOR %s' % (name, task_name))
            with ut.Indenter('[%s] ' % (name,)):
                clf_list = self.task_clfs[task_name][name]
                feature_importances = np.mean([
                    clf_.feature_importances_ for clf_ in clf_list
                ], axis=0)
                importances = ut.dzip(X.columns, feature_importances)

                featinfo = PairFeatInfo(X, importances)

                # featinfo.print_margins('feature')
                featinfo.print_margins('measure_type')
                # featinfo.print_margins('summary_op')
                # featinfo.print_margins('summary_measure')
                featinfo.print_margins('global_measure')
                # featinfo.print_margins([('measure_type', '==', 'summary'),
                #                     ('summary_op', '==', 'sum')])
                # featinfo.print_margins([('measure_type', '==', 'summary'),
                #                     ('summary_op', '==', 'mean')])
                # featinfo.print_margins([('measure_type', '==', 'summary'),
                #                     ('summary_op', '==', 'std')])
                # featinfo.print_margins([('measure_type', '==', 'global')])
                # featinfo.print_margins('global_measure')
                featinfo.print_margins('local_measure')
                featinfo.print_margins('local_sorter')
                featinfo.print_margins('local_rank')
                # ut.fix_embed_globals()
                # pt.wordcloud(importances)

    def report_classifier_accuracy(self, task_name):
        roc_scores = {}
        for name in self.datakey_list:
            # X = self.X_dict[name]
            combo_res = self.task_combo_res[task_name][name]
            roc_scores[name] = [combo_res.roc_score()]
        df_rf = pd.DataFrame(roc_scores)
        from utool.experimental.pandas_highlight import to_string_monkey
        print(to_string_monkey(
            df_rf, highlight_cols=np.arange(len(df_rf.columns))))


class ClfResult(object):
    """
    cls = ClfResult
    """
    def __init__(res):
        pass

    @classmethod
    def make_single(ClfResult, test_idx, clf_probs, pred_classes, task_labels):
        """
        Make a result for a single cross validiation subset
        """
        res = ClfResult()

        # Ensure shape corresponds with all classes
        alignx = ut.list_alignment(pred_classes, task_labels.classes_, missing=True)
        aligned_probs_ = ut.none_take(clf_probs.T, alignx)
        aligned_probs_ = ut.replace_nones(aligned_probs_, np.zeros(len(clf_probs)))
        aligned_probs = np.vstack(aligned_probs_).T

        class_names = ut.lmap(str, task_labels.class_names)
        res.class_names = class_names
        index = pd.Series(test_idx, name='test_idx')

        res.probs_df = pd.DataFrame(
            aligned_probs, index=index,
            columns=['p_' + n for n in class_names],
        )
        res.target_bin_df = pd.DataFrame(
            data=task_labels.y_bin[test_idx], index=index,
            columns=['is_' + n for n in class_names],
        )
        res.target_enc_df = pd.DataFrame(
            data=task_labels.y_enc[test_idx], index=index,
            columns=['class_idx'],
        )
        return res

    @classmethod
    def combine_results(cls, res_list, task_labels=None):
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
        # res.labels = res0.labels
        res.class_names = res0.class_names
        res.probs_df = pd.concat([r.probs_df for r in res_list])
        res.target_bin_df = pd.concat([r.target_bin_df for r in res_list])
        res.target_enc_df = pd.concat([r.target_enc_df for r in res_list])

        return res

    def make_meta(res, labels):
        """
        labels = self.labels
        """
        meta = {}
        meta['easiness'] = np.array(ut.ziptake(res.probs_df.values, res.target_enc_df.values)).ravel()
        meta['hardness'] = 1 - meta['easiness']
        meta['aid1'] = labels.aid_pairs.T[0].take(res.probs_df.index.values)
        meta['aid2'] = labels.aid_pairs.T[1].take(res.probs_df.index.values)
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

        # mcc = sklearn.metrics.matthews_corrcoef(y_test_enc_aug, pred_enc,
        #                                         sample_weight=sample_weight)
        # print('MCC = %r' % (mcc,))

        print('Confusion Matrix:')
        print(pd.DataFrame(confusion, columns=[m for m in res.class_names],
                           index=['gt ' + m for m in res.class_names]))
        print(report)

    def roc_score(res):
        (y_test_enc_aug, y_test_bin_aug,
         clf_probs_aug, sample_weight) = res.augment_if_needed()
        auc_learn = sklearn.metrics.roc_auc_score(y_test_bin_aug, clf_probs_aug)
        return auc_learn


class MultiClassLabels(ut.NiceRepr):
    """
    Handles labels that encode mutually exclusive classes

        import pandas as pd
        pd.options.display.max_rows = 10
        # pd.options.display.max_rows = 20
        pd.options.display.max_columns = 40
        pd.options.display.width = 160

    """
    def __init__(task_labels):
        # Helper Info
        task_labels.task_name = None
        task_labels.class_names = None
        task_labels.classes_ = None
        task_labels.n_samples = None
        task_labels.n_classes = None
        # Core data
        task_labels.indicator_df = None
        task_labels.encoded_df = None

    @property
    def y_bin(task_labels):
        return task_labels.indicator_df.values

    @property
    def y_enc(task_labels):
        return task_labels.encoded_df.values.ravel()

    @classmethod
    def from_indicators(MultiClassLabels, indicator, task_name=None):
        import six
        task_labels = MultiClassLabels()
        n_samples = len(six.next(six.itervalues(indicator)))
        # if index is None:
        #     index = pd.Series(np.arange(n_samples), name='index')
        indicator_df = pd.DataFrame(indicator)
        assert np.all(indicator_df.sum(axis=1).values), (
            'states in the same task must be mutually exclusive')
        task_labels.indicator_df = indicator_df
        task_labels.class_names = indicator_df.columns.values
        task_labels.encoded_df = pd.DataFrame(
            indicator_df.values.argmax(axis=1),
            columns=[task_name]
        )
        task_labels.task_name = task_name
        task_labels.n_samples = n_samples
        task_labels.n_classes = len(task_labels.class_names)
        task_labels.classes_ = np.arange(task_labels.n_classes)
        return task_labels

    def __nice__(task_labels):
        parts = []
        if task_labels.task_name is not None:
            parts.append(task_labels.task_name)
        parts.append('nD=%r' % (task_labels.n_samples))
        parts.append('nC=%r' % (task_labels.n_classes))
        return ' '.join(parts)

    def __len__(task_labels):
        return task_labels.n_samples

    def make_histogram(task_labels):
        class_idx_hist = ut.dict_hist(task_labels.y_enc)
        class_hist = ut.map_keys(
            lambda idx: task_labels.class_names[idx], class_idx_hist)
        return class_hist

    def print_info(task_labels):
        print('hist(%s) = %s' % (task_labels.task_name, ut.repr4(task_labels.make_histogram())))


class MultiTaskLabels(ut.NiceRepr):
    """
    Handles a combination of non-mutually exclusive subclassification labels
    """
    def __init__(labels):
        labels.subtasks = ut.odict()

    def apply_indicators(labels, tasks_to_indicators):
        labels.n_tasks = len(tasks_to_indicators)
        for task_name, indicator in tasks_to_indicators.items():
            task_labels = MultiClassLabels.from_indicators(
                indicator, task_name=task_name)
            labels.subtasks[task_name] = task_labels

    @ut.memoize
    def encoded_2d(labels):
        encoded_2d = pd.concat([v.encoded_df for k, v in labels.items()], axis=1).values
        return encoded_2d

    def class_name_basis(labels):
        """ corresponds with indexes returned from encoded1d """
        class_name_basis = [(b, a) for a, b in ut.product(*[
            v.class_names for k, v in labels.items()][::-1])]
        return class_name_basis

    @ut.memoize
    def encoded_1d(labels):
        """ Returns a unique label for each combination of labels """
        # from sklearn.preprocessing import MultiLabelBinarizer
        encoded_2d = labels.encoded_2d()
        class_space = [v.n_classes for k, v in labels.items()]
        offsets = np.array([1] + np.cumprod(class_space).tolist()[:-1])[None, :]
        encoded_1d = (offsets * encoded_2d).sum(axis=1)
        # e = MultiLabelBinarizer()
        # bin_coeff = e.fit_transform(encoded_2d)
        # bin_basis = (2 ** np.arange(bin_coeff.shape[1]))[None, :]
        # # encoded_1d = (bin_coeff * bin_basis).sum(axis=1)
        # encoded_1d = (bin_coeff * bin_basis[::-1]).sum(axis=1)
        # # vt.unique_rows(sklearn.preprocessing.MultiLabelBinarizer().fit_transform(encoded_2d))
        # [v.encoded_df.values for k, v in labels.items()]
        # encoded_df_1d = pd.concat([v.encoded_df for k, v in labels.items()], axis=1)
        return encoded_1d

    def __nice__(labels):
        return 'nS=%r, nT=%r' % (len(labels), labels.n_tasks)

    def __len__(labels):
        return labels.n_samples

    def print_info(labels):
        for task_name, task_labels in labels.items():
            task_labels.print_info()
        print('hist(all) = %s' % (ut.repr4(labels.make_histogram())))

    def make_histogram(labels):
        class_name_basis = labels.class_name_basis()
        multi_task_idx_hist = ut.dict_hist(labels.encoded_1d())
        multi_task_hist = ut.map_keys(
            lambda k: class_name_basis[k], multi_task_idx_hist)
        return multi_task_hist

    def items(labels):
        for task_name, task_labels in labels.subtasks.items():
            yield task_name, task_labels

    # def take(labels, idxs):
    #     mask = ut.index_to_boolmask(idxs, len(labels))
    #     return labels.compress(mask)


class PairLabels(MultiTaskLabels):
    """
    Manages the different ways to assign labels to 1-v-1 classification labels

    CommandLine:
        python -m ibeis.scripts.script_vsone PairLabels

    Example:
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> self = OneVsOneProblem()
        >>> self.load_features()
        >>> labels = PairLabels(self.ibs, self.aid_pairs, self.simple_scores)
        >>> print(labels)
        >>> labels.print_info()
    """
    def __init__(labels, ibs, aid_pairs, simple_scores):
        super(PairLabels, labels).__init__()
        labels.ibs = ibs
        labels.aid_pairs = aid_pairs
        labels.simple_scores = simple_scores
        labels.annots1 = ibs.annots(aid_pairs.T[0], asarray=True)
        labels.annots2 = ibs.annots(aid_pairs.T[1], asarray=True)
        labels.n_samples = len(aid_pairs)
        labels.apply_multi_task_multi_label()

    def compress(labels, flags):
        aid_pairs = labels.aid_pairs.compress(flags, axis=0)
        simple_scores = labels.simple_scores[flags]
        ibs = labels.ibs
        new_labels = PairLabels(ibs, aid_pairs, simple_scores)
        return new_labels

    @ut.memoize
    def is_same(labels):
        is_same = labels.annots1.nids == labels.annots2.nids
        labels.simple_scores
        return is_same

    @ut.memoize
    def is_photobomb(labels):
        am_rowids = labels.ibs.get_annotmatch_rowid_from_edges(labels.aid_pairs)
        am_tags = labels.ibs.get_annotmatch_case_tags(am_rowids)
        is_pb = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
        return is_pb

    @ut.memoize
    def is_comparable(labels):
        # If we don't have actual comparability information just guess
        return labels.guess_if_comparable()

    def guess_if_comparable(labels):
        """
        Takes a guess as to which annots are not comparable based on scores and
        viewpoints. If either viewpoints is null assume they are comparable.
        """
        simple_scores = labels.simple_scores
        key = 'sum(weighted_ratio)'
        if key not in simple_scores:
            key = 'sum(ratio)'
        scores = simple_scores[key].values
        yaws1 = labels.annots1.yaws_asfloat
        yaws2 = labels.annots2.yaws_asfloat
        dists = vt.ori_distance(yaws1, yaws2)
        tau = np.pi * 2
        is_comp = (scores > .1) | (dists < tau / 8.1) | np.isnan(dists)
        return is_comp

    def apply_multi_task_multi_label(labels):
        # multioutput-multiclass / multi-task
        tasks_to_indicators = ut.odict([
            ('match_state', ut.odict([
                ('nomatch', ~labels.is_same() & labels.is_comparable()),
                ('match',    labels.is_same() & labels.is_comparable()),
                ('notcomp', ~labels.is_comparable()),
            ])),
            ('photobomb_state', ut.odict([
                (False, ~labels.is_photobomb()),
                (True,   labels.is_photobomb()),
            ]))
        ])
        labels.apply_indicators(tasks_to_indicators)

    def apply_single_task_multi_label(labels):
        is_comp = labels.is_comparable()
        is_same = labels.is_same()
        is_pb   = labels.is_photobomb()
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
        labels.apply_indicators(tasks_to_indicators)


@ut.reloadable_class
class PairFeatInfo(object):
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

    def print_margins(featinfo, group_id):
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
        print('\nImportance of ' + nice)
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

    def summary_op(featinfo, key):
        for p in featinfo._summary_keys:
            if key.startswith(p):
                return key[0:key.find('(')]

    def summary_measure(featinfo, key):
        if any(key.startswith(p) for p in featinfo._summary_keys):
            return featinfo.measure(key)

    def local_sorter(featinfo, key):
        if key.startswith('loc'):
            return key[key.find('[') + 1:key.find(',')]

    def local_rank(featinfo, key):
        if key.startswith('loc'):
            return key[key.find(',') + 1:key.find(']')]

    def local_measure(featinfo, key):
        if key.startswith('loc'):
            return featinfo.measure(key)

    def global_measure(featinfo, key):
        if key.startswith('global'):
            return featinfo.measure(key)


def build_features(qreq_, hyper_params):
    import pandas as pd

    # ==================================
    # Compute or load one-vs-one results
    # ==================================
    cached_data, infr = bigcache_vsone(qreq_, hyper_params)
    # key_ = 'SV_LNBNN'
    key_ = 'RAT_SV'

    # for key in list(cached_data.keys()):
    #     if key != 'SV_LNBNN':
    #         del cached_data[key]

    matches = cached_data[key_]
    # Pass back just one match to play with
    for match in matches:
        if len(match.fm) > 10:
            break

    # setup truth targets
    y = np.array([m.annot1['nid'] == m.annot2['nid'] for m in matches])
    aid_pairs = np.array([(m.annot1['aid'], m.annot2['aid']) for m in matches])

    # ---------------
    # Try just using simple scores
    # vt.rrrr()
    # for m in matches:
    #     m.rrr(0, reload_module=False)
    simple_scores = pd.DataFrame([
        m._make_local_summary_feature_vector(sum=True, mean=False, std=False)
        for m in matches])

    if True:
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

    X_dict = {
        'learn(all)': X_all
    }

    return aid_pairs, simple_scores, X_dict, y, match


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


def vsone_(qreq_, query_aids, data_aids, qannot_cfg, dannot_cfg,
           configured_obj_annots, hyper_params):
    # Do vectorized preload before constructing lazy dicts
    # Then make sure the lazy dicts point to this subset
    unique_obj_annots = list(configured_obj_annots.values())
    for annots in ut.ProgIter(unique_obj_annots, 'vectorized preload'):
        annots.set_caching(True)
        annots.chip_size
        annots.vecs
        annots.kpts
        annots.yaw
        annots.qual
        annots.gps
        annots.time
        if qreq_.qparams.featweight_enabled:
            annots.fgweights
    # annots._internal_attrs.clear()

    # Make convinient lazy dict representations (after loading pre info)
    configured_lazy_annots = ut.ddict(dict)
    for config, annots in configured_obj_annots.items():
        annot_dict = configured_lazy_annots[config]
        for _annot in ut.ProgIter(annots.scalars(), label='make lazy dict'):
            annot = _annot._make_lazy_dict()
            annot_dict[_annot.aid] = annot

    unique_lazy_annots = ut.flatten(
        [x.values() for x in configured_lazy_annots.values()])

    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    for annot in ut.ProgIter(unique_lazy_annots, label='lazy flann'):
        vt.matching.ensure_metadata_flann(annot, flann_params)
        vt.matching.ensure_metadata_normxy(annot)

    for annot in ut.ProgIter(unique_lazy_annots, 'preload kpts'):
        annot['kpts']
    for annot in ut.ProgIter(unique_lazy_annots, 'preload normxy'):
        annot['norm_xys']
    for annot in ut.ProgIter(unique_lazy_annots, 'preload vecs'):
        annot['vecs']

    # Extract pairs of annot objects (with shared caches)
    lazy_annots1 = ut.take(configured_lazy_annots[qannot_cfg], query_aids)
    lazy_annots2 = ut.take(configured_lazy_annots[dannot_cfg], data_aids)

    # TODO: param search over grid
    #     'use_sv': [0, 1],
    #     'use_fg': [0, 1],
    #     'use_ratio_test': [0, 1],
    matches_RAT = [vt.PairwiseMatch(annot1, annot2)
                   for annot1, annot2 in zip(lazy_annots1, lazy_annots2)]

    # Construct global measurements
    global_keys = ['yaw', 'qual', 'gps', 'time']
    for match in ut.ProgIter(matches_RAT, label='setup globals'):
        match.add_global_measures(global_keys)

    # Preload flann for only specific annots
    for match in ut.ProgIter(matches_RAT, label='preload FLANN'):
        match.annot1['flann']

    cfgdict = hyper_params.vsone_assign
    # Find one-vs-one matches
    # cfgdict = {'checks': 20, 'symmetric': False}
    for match in ut.ProgIter(matches_RAT, label='assign vsone'):
        match.assign(cfgdict=cfgdict)

    # gridsearch_ratio_thresh()
    # vt.matching.gridsearch_match_operation(matches_RAT, 'apply_ratio_test', {
    #     'ratio_thresh': np.linspace(.6, .7, 50)
    # })
    for match in ut.ProgIter(matches_RAT, label='apply ratio thresh'):
        match.apply_ratio_test({'ratio_thresh': .638}, inplace=True)

    # TODO gridsearch over sv params
    # vt.matching.gridsearch_match_operation(matches_RAT, 'apply_sver', {
    #     'xy_thresh': np.linspace(0, 1, 3)
    # })
    matches_RAT_SV = [
        match.apply_sver(inplace=True)
        for match in ut.ProgIter(matches_RAT, label='sver')
    ]

    # Add keypoint spatial information to local features
    for match in matches_RAT_SV:
        match.add_local_measures()
        # key_ = 'norm_xys'
        # norm_xy1 = match.annot1[key_].take(match.fm.T[0], axis=1)
        # norm_xy2 = match.annot2[key_].take(match.fm.T[1], axis=1)
        # match.local_measures['norm_x1'] = norm_xy1[0]
        # match.local_measures['norm_y1'] = norm_xy1[1]
        # match.local_measures['norm_x2'] = norm_xy2[0]
        # match.local_measures['norm_y2'] = norm_xy2[1]

        # match.local_measures['scale1'] = vt.get_scales(
        #     match.annot1['kpts'].take(match.fm.T[0], axis=0))
        # match.local_measures['scale2'] = vt.get_scales(
        #     match.annot2['kpts'].take(match.fm.T[1], axis=0))

    # Create another version where we find global normalizers for the data
    # qreq_.load_indexer()
    # matches_SV_LNBNN = batch_apply_lnbnn(matches_RAT_SV, qreq_, inplace=True)

    # if 'weight' in cfgdict:
    #     for match in matches_SV_LNBNN[::-1]:
    #         lnbnn_dist = match.local_measures['lnbnn']
    #         ndist = match.local_measures['lnbnn_norm_dist']
    #         weights = match.local_measures[cfgdict['weight']]
    #         match.local_measures['weighted_lnbnn'] = weights * lnbnn_dist
    #         match.local_measures['weighted_lnbnn_norm_dist'] = weights * ndist
    #         match.fs = match.local_measures['weighted_lnbnn']

    cached_data = {
        # 'RAT': matches_RAT,
        'RAT_SV': matches_RAT_SV,
        # 'SV_LNBNN': matches_SV_LNBNN,
    }
    return cached_data


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
        lnbnn_dist = np.clip(lnbnn_dist, 0, np.inf)
        match_.local_measures['lnbnn_norm_dist'] = ndist
        match_.local_measures['lnbnn'] = lnbnn_dist
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


def photobombing_subset():
    """
    CommandLine:
        python -m ibeis.scripts.script_vsone photobombing_subset
    """
    import ibeis
    # pair_sample = ut.odict([
    #     ('top_gt', 4), ('mid_gt', 2), ('bot_gt', 2), ('rand_gt', 2),
    #     ('top_gf', 3), ('mid_gf', 2), ('bot_gf', 1), ('rand_gf', 2),
    # ])
    qreq_ = ibeis.testdata_qreq_(
        defaultdb='PZ_Master1',
        a=':mingt=2,species=primary',
        # t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum',
        t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum,QRH=True',
    )
    ibs = qreq_.ibs
    # cm_list = qreq_.execute()
    # infr = ibeis.AnnotInference.from_qreq_(qreq_, cm_list, autoinit=True)
    # aid_pairs_ = infr._cm_training_pairs(rng=np.random.RandomState(42),
    #                                      **pair_sample)

    # # ut.dict_hist(ut.flatten(am_tags))
    # am_rowids = ibs._get_all_annotmatch_rowids()
    # am_tags = ibs.get_annotmatch_case_tags(am_rowids)
    # am_flags = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
    # am_rowids_ = ut.compress(am_rowids, am_flags)
    # aids1 = ibs.get_annotmatch_aid1(am_rowids_)
    # aids2 = ibs.get_annotmatch_aid2(am_rowids_)
    # pb_aids_pairs = list(zip(aids1, aids2))

    # # aids = unique_pb_aids = ut.unique(ut.flatten(pb_aids_pairs))
    # # ut.compress(unique_pb_aids, ibs.is_aid_unknown(unique_pb_aids))

    # assert len(pb_aids_pairs) > 0

    # # Keep only a random subset
    # subset_idxs = list(range(len(aid_pairs_)))
    # rng = np.random.RandomState(3104855634)
    # num_max = len(pb_aids_pairs)
    # if num_max < len(subset_idxs):
    #     subset_idxs = rng.choice(subset_idxs, size=num_max, replace=False)
    #     subset_idxs = sorted(subset_idxs)
    # aid_pairs_ = ut.take(aid_pairs_, subset_idxs)

    # aid_pairs_ += pb_aids_pairs
    # unique_aids = ut.unique(ut.flatten(aid_pairs_))

    # a1 = ibs.filter_annots_general(unique_aids, is_known=True, verbose=True, min_pername=2, has_none=['photobomb'])
    # a2 = ibs.filter_annots_general(unique_aids, has_any=['photobomb'], verbose=True, is_known=True)
    # a = sorted(set(a1 + a2))
    # ibs.print_annot_stats(a)
    # len(a)

    a = [8, 27, 30, 86, 87, 90, 92, 94, 99, 103, 104, 106, 111, 217, 218, 242,
         298, 424, 425, 456, 464, 465, 472, 482, 529, 559, 574, 585, 588, 592,
         598, 599, 601, 617, 630, 645, 661, 664, 667, 694, 723, 724, 759, 768,
         843, 846, 861, 862, 866, 933, 934, 980, 987, 1000, 1003, 1005, 1011,
         1017, 1020, 1027, 1059, 1074, 1076, 1080, 1095, 1096, 1107, 1108,
         1192, 1203, 1206, 1208, 1220, 1222, 1223, 1224, 1256, 1278, 1293,
         1294, 1295, 1296, 1454, 1456, 1474, 1484, 1498, 1520, 1521, 1548,
         1563, 1576, 1593, 1669, 1675, 1680, 1699, 1748, 1751, 1811, 1813,
         1821, 1839, 1927, 1934, 1938, 1952, 1992, 2003, 2038, 2054, 2066,
         2080, 2103, 2111, 2170, 2171, 2175, 2192, 2216, 2227, 2240, 2250,
         2253, 2266, 2272, 2288, 2292, 2314, 2329, 2341, 2344, 2378, 2397,
         2417, 2429, 2444, 2451, 2507, 2551, 2552, 2553, 2581, 2628, 2640,
         2642, 2646, 2654, 2667, 2686, 2733, 2743, 2750, 2759, 2803, 2927,
         3008, 3054, 3077, 3082, 3185, 3205, 3284, 3306, 3334, 3370, 3386,
         3390, 3393, 3401, 3448, 3508, 3542, 3597, 3614, 3680, 3684, 3695,
         3707, 3727, 3758, 3765, 3790, 3812, 3813, 3818, 3858, 3860, 3874,
         3875, 3887, 3892, 3915, 3918, 3924, 3927, 3929, 3933, 3941, 3952,
         3955, 3956, 3959, 4004, 4059, 4073, 4076, 4089, 4094, 4124, 4126,
         4128, 4182, 4189, 4217, 4222, 4229, 4257, 4266, 4268, 4288, 4289,
         4296, 4306, 4339, 4353, 4376, 4403, 4428, 4455, 4487, 4494, 4515,
         4517, 4524, 4541, 4544, 4556, 4580, 4585, 4597, 4604, 4629, 4639,
         4668, 4671, 4672, 4675, 4686, 4688, 4693, 4716, 4730, 4731, 4749,
         4772, 4803, 4820, 4823, 4832, 4833, 4836, 4900, 4902, 4909, 4924,
         4936, 4938, 4939, 4944, 5004, 5006, 5034, 5043, 5044, 5055, 5064,
         5072, 5115, 5131, 5150, 5159, 5165, 5167, 5168, 5174, 5218, 5235,
         5245, 5249, 5309, 5319, 5334, 5339, 5344, 5347, 5378, 5379, 5384,
         5430, 5447, 5466, 5509, 5546, 5587, 5588, 5621, 5640, 5663, 5676,
         5682, 5685, 5687, 5690, 5707, 5717, 5726, 5732, 5733, 5791, 5830,
         5863, 5864, 5869, 5870, 5877, 5879, 5905, 5950, 6008, 6110, 6134,
         6160, 6167, 6234, 6238, 6265, 6344, 6345, 6367, 6384, 6386, 6437,
         6495, 6533, 6538, 6569, 6587, 6626, 6634, 6643, 6659, 6661, 6689,
         6714, 6725, 6739, 6754, 6757, 6759, 6763, 6781, 6830, 6841, 6843,
         6893, 6897, 6913, 6930, 6932, 6936, 6944, 6976, 7003, 7022, 7037,
         7052, 7058, 7074, 7103, 7107, 7108, 7113, 7143, 7183, 7185, 7187,
         7198, 7200, 7202, 7207, 7222, 7275, 7285, 7388, 7413, 7421, 7425,
         7429, 7445, 7487, 7507, 7508, 7528, 7615, 7655, 7696, 7762, 7786,
         7787, 7796, 7797, 7801, 7807, 7808, 7809, 7826, 7834, 7835, 7852,
         7861, 7874, 7881, 7901, 7902, 7905, 7913, 7918, 7941, 7945, 7990,
         7999, 8007, 8009, 8017, 8018, 8019, 8034, 8041, 8057, 8058, 8079,
         8080, 8086, 8089, 8092, 8094, 8100, 8105, 8109, 8147, 8149, 8153,
         8221, 8264, 8302, 8303, 8331, 8366, 8367, 8370, 8376, 8474, 8501,
         8504, 8506, 8507, 8514, 8531, 8532, 8534, 8538, 8563, 8564, 8587,
         8604, 8608, 8751, 8771, 8792, 9175, 9204, 9589, 9726, 9841, 10674,
         12122, 12305, 12796, 12944, 12947, 12963, 12966, 13098, 13099, 13101,
         13103, 13109, 13147, 13157, 13168, 13194, 13236, 13253, 13255, 13410,
         13450, 13474, 13477, 13481, 13508, 13630, 13670, 13727, 13741, 13819,
         13820, 13908, 13912, 13968, 13979, 14007, 14009, 14010, 14019, 14066,
         14067, 14072, 14074, 14148, 14153, 14224, 14230, 14237, 14239, 14241,
         14274, 14277, 14290, 14293, 14308, 14309, 14313, 14319, 14668, 14670,
         14776, 14918, 14920, 14924, 15135, 15157, 15318, 15319, 15490, 15518,
         15531, 15777, 15903, 15913, 16004, 16012, 16013, 16014, 16020, 16215,
         16221, 16235, 16240, 16259, 16273, 16279, 16284, 16289, 16316, 16322,
         16329, 16336, 16364, 16389, 16706, 16897, 16898, 16903, 16949, 17094,
         17101, 17137, 17200, 17222, 17290, 17327, 17336]

    from ibeis.dbio import export_subset
    export_subset.export_annots(ibs, a, 'PZ_PB_RF_TRAIN')

    # closed_aids = ibs.annots(unique_aids).get_name_image_closure()

    # annots = ibs.annots(unique_aids)
    # closed_gt_aids = ut.unique(ut.flatten(ibs.get_annot_groundtruth(unique_aids)))
    # closed_gt_aids = ut.unique(ut.flatten(ibs.get_annot_groundtruth(unique_aids)))
    # closed_img_aids = ut.unique(ut.flatten(ibs.get_annot_otherimage_aids(unique_aids)))

    # ibs.print_annot_stats(unique_aids)
    # all_annots = ibs.annots()


def bigcache_vsone(qreq_, hyper_params):
    """
    Cached output of one-vs-one matches
    """
    import vtool as vt
    import ibeis
    # Get a set of training pairs
    ibs = qreq_.ibs
    cm_list = qreq_.execute()
    infr = ibeis.AnnotInference.from_qreq_(qreq_, cm_list, autoinit=True)

    # Per query choose a set of correct, incorrect, and random training pairs
    aid_pairs_ = infr._cm_training_pairs(rng=np.random.RandomState(42),
                                         **hyper_params.pair_sample)

    aid_pairs_ = vt.unique_rows(np.array(aid_pairs_), directed=False).tolist()

    pb_aid_pairs_ = photobomb_samples(ibs)
    # TODO: handle non-comparability / photobombs

    aid_pairs_ = pb_aid_pairs_ + aid_pairs_

    # []

    # ======================================
    # Compute one-vs-one scores and local_measures
    # ======================================

    # Prepare lazy attributes for annotations
    qreq_ = infr.qreq_
    ibs = qreq_.ibs
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_data_config2
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)

    # Remove any pairs missing features
    if dannot_cfg == qannot_cfg:
        unique_annots = ibs.annots(np.unique(np.array(aid_pairs_)),
                                   config=dannot_cfg)
        bad_aids = unique_annots.compress(~np.array(unique_annots.num_feats) >
                                          0).aids
        bad_aids = set(bad_aids)
    else:
        annots1_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 0)),
                              config=qannot_cfg)
        annots2_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 1)),
                              config=dannot_cfg)
        bad_aids1 = annots1_.compress(~np.array(annots1_.num_feats) > 0).aids
        bad_aids2 = annots2_.compress(~np.array(annots2_.num_feats) > 0).aids
        bad_aids = set(bad_aids1 + bad_aids2)
    subset_idxs = np.where([not (a1 in bad_aids or a2 in bad_aids)
                            for a1, a2 in aid_pairs_])[0]
    # Keep only a random subset
    if hyper_params.subsample:
        rng = np.random.RandomState(3104855634)
        num_max = hyper_params.subsample
        if num_max < len(subset_idxs):
            subset_idxs = rng.choice(subset_idxs, size=num_max, replace=False)
            subset_idxs = sorted(subset_idxs)

    # Take the current selection
    aid_pairs = ut.take(aid_pairs_, subset_idxs)
    query_aids = ut.take_column(aid_pairs, 0)
    data_aids = ut.take_column(aid_pairs, 1)

    # Determine a unique set of annots per config
    configured_aids = ut.ddict(set)
    configured_aids[qannot_cfg].update(query_aids)
    configured_aids[dannot_cfg].update(data_aids)

    # Make efficient annot-object representation
    configured_obj_annots = {}
    for config, aids in configured_aids.items():
        annots = ibs.annots(sorted(list(aids)), config=config)
        configured_obj_annots[config] = annots

    annots1 = configured_obj_annots[qannot_cfg].loc(query_aids)
    annots2 = configured_obj_annots[dannot_cfg].loc(data_aids)

    # Get hash based on visual annotation appearence of each pair
    # as well as algorithm configurations used to compute those properties
    qvuuids = annots1.visual_uuids
    dvuuids = annots2.visual_uuids
    qcfgstr = annots1._config.get_cfgstr()
    dcfgstr = annots2._config.get_cfgstr()
    annots_cfgstr = ut.hashstr27(qcfgstr) + ut.hashstr27(dcfgstr)
    vsone_uuids = [
        ut.combine_uuids(uuids, salt=annots_cfgstr)
        for uuids in ut.ProgIter(zip(qvuuids, dvuuids), length=len(qvuuids),
                                 label='hashing ids')
    ]

    # Combine into a big cache for the entire 1-v-1 matching run
    big_uuid = ut.hashstr_arr27(vsone_uuids, '', pathsafe=True)
    cacher = ut.Cacher('vsone_v2', cfgstr=str(big_uuid), appname='vsone_rf_train')

    cached_data = cacher.tryload()
    if cached_data is not None:
        # Caching doesn't work 100% for PairwiseMatch object, so we need to do
        # some postprocessing
        configured_lazy_annots = ut.ddict(dict)
        for config, annots in configured_obj_annots.items():
            annot_dict = configured_lazy_annots[config]
            for _annot in ut.ProgIter(annots.scalars(), label='make lazy dict'):
                annot_dict[_annot.aid] = _annot._make_lazy_dict()

        # Extract pairs of annot objects (with shared caches)
        lazy_annots1 = ut.take(configured_lazy_annots[qannot_cfg], query_aids)
        lazy_annots2 = ut.take(configured_lazy_annots[dannot_cfg], data_aids)

        # Create a set of PairwiseMatches with the correct annot properties
        matches = [vt.PairwiseMatch(annot1, annot2)
                   for annot1, annot2 in zip(lazy_annots1, lazy_annots2)]

        # Updating a new matches dictionary ensure the annot1/annot2 properties
        # are set correctly
        for key, cached_matches in list(cached_data.items()):
            fixed_matches = [match.copy() for match in matches]
            for fixed, internal in zip(fixed_matches, cached_matches):
                dict_ = internal.__dict__
                ut.delete_dict_keys(dict_, ['annot1', 'annot2'])
                fixed.__dict__.update(dict_)
            cached_data[key] = fixed_matches
    else:
        cached_data = vsone_(qreq_, query_aids, data_aids, qannot_cfg,
                             dannot_cfg, configured_obj_annots, hyper_params)
        cacher.save(cached_data)
    return cached_data, infr


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
