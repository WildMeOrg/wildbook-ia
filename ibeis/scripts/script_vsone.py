# -*- coding: utf-8 -*-
"""
TODO:

* Get end-to-end system test working with simulated reviewer

* Autoselect features:
    * Learn RF
    * prune bottom N features
    * loop until only X features remain
"""
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import itertools as it
import numpy as np
import vtool as vt
import dtool as dt
import copy  # NOQA
from six.moves import zip
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass
import sklearn.ensemble
from ibeis.scripts import clf_helpers
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP
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
        ut.ParamInfo('summary_ops', {'sum', 'std', 'mean', 'len', 'med'}),
        ut.ParamInfo('local_keys', None),
        # ut.ParamInfo('local_keys', [...]),
        ut.ParamInfo('sorters', [
            'ratio', 'norm_dist', 'match_dist'
            # 'lnbnn', 'lnbnn_norm_dist',
        ]),

        # ut.ParamInfo('sum', True),
        # ut.ParamInfo('std', True),
        # ut.ParamInfo('mean', True),
        # ut.ParamInfo('len', True),
        # ut.ParamInfo('med', True),
    ]


class VsOneAssignConfig(dt.Config):
    _param_info_list = vt.matching.VSONE_ASSIGN_CONFIG


@ut.reloadable_class
class OneVsOneProblem(clf_helpers.ClfProblem):
    """
    Keeps information about the one-vs-one pairwise classification problem

    CommandLine:
        python -m ibeis.scripts.script_vsone evaluate_classifiers
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --show
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --profile
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_MTEST --show
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_Master1 --show
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db GZ_Master1 --show

    Example:
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST')
        >>> pblm.load_samples()
        >>> pblm.load_features()
    """
    appname = 'vsone_rf_train'

    def __init__(pblm, infr=None, verbose=None):
        super(OneVsOneProblem, pblm).__init__()
        if verbose is None:
            verbose = 2
        pblm.verbose = verbose
        pblm.primary_task_key = 'match_state'
        pblm.default_data_key = 'learn(sum,glob)'
        pblm.default_clf_key = 'RF'

        pblm.infr = infr
        pblm.qreq_ = None

        hyper_params = dt.Config.from_dict(ut.odict([
            ('subsample', None),
            ('pair_sample', PairSampleConfig()),
            ('vsone_assign', VsOneAssignConfig()),
            ('pairwise_feats', PairFeatureConfig()),
            ('sample_search', dict(K=4, Knorm=1, requery=True,
                                   score_method='csum', prescore_method='csum')),
        ]),
            tablename='HyperParams'
        )
        pblm.hyper_params = hyper_params

    @classmethod
    def from_aids(OneVsOneProblem, ibs, aids, verbose=None):
        pblm = OneVsOneProblem(ibs=ibs, qaid_override=aids, daid_override=aids,
                               verbose=verbose)
        return pblm

    @classmethod
    def from_empty(OneVsOneProblem, defaultdb=None):
        if defaultdb is None:
            defaultdb = 'PZ_PB_RF_TRAIN'
            # defaultdb = 'GZ_Master1'
            # defaultdb = 'PZ_MTEST'
        import ibeis
        ibs, aids = ibeis.testdata_aids(defaultdb)

        # TODO: If the graph structure is defined, this should load the most
        # recent state, so the infr object has all the right edges.  If the
        # graph structure is not defined, it should apply the conversion
        # method.
        infr = ibeis.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        assert infr._is_staging_above_annotmatch()
        infr.reset_feedback('staging', apply=True)
        if infr.ibs.dbname == 'PZ_MTEST':
            assert False, 'need to do conversion'
        # if infr.needs_conversion():
        #     infr.ensure_mst()
        pblm = OneVsOneProblem(infr=infr)
        return pblm

    def _fix_hyperparams(pblm, qreq_):
        if qreq_.qparams.featweight_enabled:
            pblm.hyper_params.vsone_assign['weight'] = 'fgweights'
            pblm.hyper_params.pairwise_feats['sorters'] = ut.unique(
                pblm.hyper_params.pairwise_feats['sorters'] +
                [
                    'weighted_ratio',
                    # 'weighted_lnbnn'
                ]
            )
        else:
            pblm.hyper_params.vsone_assign['weight'] = None

    @profile
    def make_training_pairs(pblm):
        infr = pblm.infr
        ibs = pblm.infr.ibs
        if pblm.verbose > 0:
            print('[pblm] gather match-state hard cases')

        cfgdict = pblm.hyper_params['sample_search']
        aids = ibs.filter_annots_general(
            infr.aids, min_pername=3, species='primary')
        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      verbose=False)
        pblm._fix_hyperparams(qreq_)

        use_cache = True
        cfgstr = qreq_.get_cfgstr(with_input=True)
        cacher = ut.Cacher('pairwise_sample_v1', cfgstr=cfgstr,
                           appname=pblm.appname, enabled=use_cache,
                           verbose=pblm.verbose + 10)
        assert qreq_.qparams.can_match_samename is True
        assert qreq_.qparams.prescore_method == 'csum'
        assert pblm.hyper_params.subsample is None
        data = cacher.tryload()
        if data is None:
            cm_list = qreq_.execute()
            infr._set_vsmany_info(qreq_, cm_list)

            # Sample hard moderate and easy positives / negative
            # For each query, choose same, different, and random training pairs
            rng = np.random.RandomState(42)
            aid_pairs_ = infr._cm_training_pairs(
                rng=rng, **pblm.hyper_params.pair_sample)
            cacher.save(aid_pairs_)
            data = aid_pairs_
        aid_pairs_ = data

        if pblm.verbose > 0:
            print('[pblm] gather photobomb and incomparable cases')
        pb_aid_pairs = infr.photobomb_samples()
        incomp_aid_pairs = list(infr.incomp_graph.edges())

        # Simplify life by using undirected pairs
        aid_pairs = pb_aid_pairs + aid_pairs_ + incomp_aid_pairs
        aid_pairs = sorted(set(it.starmap(infr.e_, aid_pairs)))
        return aid_pairs

    def load_samples(pblm):
        r"""
        CommandLine:
            python -m ibeis.scripts.script_vsone load_samples --profile

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> #pblm = OneVsOneProblem.from_empty('PZ_MTEST')
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm.load_samples()
            >>> samples = pblm.samples
            >>> samples.print_info()
        """
        # Get a set of training pairs
        if pblm.verbose > 0:
            print('[pblm] load_samples')
        aid_pairs = pblm.make_training_pairs()
        pblm.samples = AnnotPairSamples(pblm.infr.ibs, aid_pairs, pblm.infr)
        # simple_scores=copy.deepcopy(pblm.raw_simple_scores),
        # X_dict=copy.deepcopy(pblm.raw_X_dict),

    @profile
    def load_features(pblm, use_cache=True):
        """
        CommandLine:
            python -m ibeis.scripts.script_vsone load_features --profile

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm.load_samples()
            >>> pblm.load_features()
        """
        if pblm.verbose > 0:
            print('[pblm] load_features')
        infr = pblm.infr
        dbname = infr.ibs.get_dbname()
        aid_pairs = ut.lmap(tuple, pblm.samples.aid_pairs.tolist())
        hyper_params = pblm.hyper_params
        edge_hashid = pblm.samples.edge_hashid()
        feat_cfgstr = hyper_params.get_cfgstr()
        feat_hashid = ut.hashstr27(edge_hashid + feat_cfgstr)
        # print('features_hashid = %r' % (features_hashid,))
        cfgstr = '_'.join(['devcache', str(dbname), feat_hashid])
        # use_cache = False
        cacher = ut.Cacher('pairwise_data_v14', cfgstr=cfgstr,
                           appname=pblm.appname, enabled=use_cache,
                           verbose=pblm.verbose)
        data = cacher.tryload()
        if not data:
            config = hyper_params.vsone_assign
            pairfeat_cfg = hyper_params.pairwise_feats
            need_lnbnn = False
            if need_lnbnn:
                raise NotImplementedError('not done yet')
                if infr.qreq_ is None:
                    pass
                    # cfgdict = pblm.hyper_params['sample_search']
                    # ibs = pblm.infr.ibs
                    # infr = pblm.infr
                    # aids = ibs.filter_annots_general(
                    #     infr.aids, min_pername=3, species='primary')
                    # qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                    #                               verbose=False)
                    # infr.qreq_ = qreq_
            matches, X_all = infr._make_pairwise_features(
                aid_pairs, config=config, pairfeat_cfg=pairfeat_cfg,
                need_lnbnn=need_lnbnn)

            # # Pass back just one match to play with
            # for match in matches:
            #     if len(match.fm) > 10:
            #         break

            # ---------------
            # Construct simple scores to learning comparison
            simple_scores = pd.DataFrame([
                m._make_local_summary_feature_vector(summary_ops={'sum', 'len'})
                for m in ut.ProgIter(matches, 'make simple scores')],
                index=X_all.index,
            )

            if True:
                # The main idea here is to load lnbnn scores for the pairwise
                # matches so we can compare them to the outputs of the pairwise
                # classifeir.
                # TODO: separate this into different cache
                # Add vsmany_lnbnn to simple scores
                cfgdict = pblm.hyper_params['sample_search']
                ibs = pblm.infr.ibs
                infr = pblm.infr
                aids = ibs.filter_annots_general(
                    infr.aids, min_pername=3, species='primary')
                qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                              verbose=False)
                cm_list = qreq_.execute()
                edge_to_data = infr._get_cm_edge_data(aid_pairs, cm_list=cm_list)
                edge_data = ut.take(edge_to_data, aid_pairs)
                lnbnn_score_list = [d.get('score', 0) for d in edge_data]
                lnbnn_score_list = [0 if s is None else s
                                    for s in lnbnn_score_list]

                # infr.add_aids(ut.unique(ut.flatten(aid_pairs)))
                # # Ensure that all annots exist in the graph
                # infr.graph.add_edges_from(aid_pairs)
                # # test original lnbnn score sep
                # infr.apply_match_scores()
                # edge_data = [infr.graph.get_edge_data(u, v) for u, v in aid_pairs]
                # lnbnn_score_list = [0 if d is None else d.get('score', 0)
                #                     for d in edge_data]
                # lnbnn_score_list = np.nan_to_num(lnbnn_score_list)
                simple_scores = simple_scores.assign(
                    score_lnbnn_1vM=lnbnn_score_list)
            simple_scores[pd.isnull(simple_scores)] = 0
            data = simple_scores, X_all
            cacher.save(data)
        simple_scores, X_all = data
        assert X_all.index.tolist() == aid_pairs, 'index disagrees'
        pblm.raw_X_dict = {'learn(all)': X_all}
        pblm.raw_simple_scores = simple_scores
        # simple_scores=,
        pblm.samples.set_feats(
            copy.deepcopy(pblm.raw_simple_scores),
            copy.deepcopy(pblm.raw_X_dict)
        )

    def evaluate_classifiers(pblm):
        """
        CommandLine:
            python -m ibeis.scripts.script_vsone evaluate_classifiers

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> #pblm = OneVsOneProblem.from_empty('GZ_Master1')
            >>> pblm.evaluate_classifiers()
        """
        pblm.setup_evaluation()

        ut.cprint('\n--- EVALUATE LEARNED CLASSIFIERS ---', 'blue')
        # For each task / classifier type
        for task_key in pblm.eval_task_keys:
            pblm.task_evaluation_report(task_key)

    def setup_evaluation(pblm):
        pblm.set_pandas_options()

        ut.cprint('\n--- LOADING DATA ---', 'blue')
        pblm.load_samples()
        # pblm.samples.print_info()
        pblm.load_features()

        # pblm.samples.print_info()
        ut.cprint('\n--- CURATING DATA ---', 'blue')
        # pblm.reduce_dataset_size()
        pblm.samples.print_info()
        print('---------------')

        ut.cprint('\n--- FEATURE INFO ---', 'blue')
        pblm.build_feature_subsets()

        if 1:
            for data_key in pblm.samples.X_dict.keys():
                print('\nINFO(samples.X_dict[%s])' % (data_key,))
                featinfo = AnnotPairFeatInfo(pblm.samples.X_dict[data_key])
                print(ut.indent(featinfo.get_infostr()))

        task_keys = list(pblm.samples.subtasks.keys())
        task_keys = [pblm.primary_task_key]
        # task_keys = ut.setdiff(task_keys, ['photobomb_state'])

        data_keys = list(pblm.samples.X_dict.keys())
        # clf_keys = ['RF', 'RF-OVR', 'SVC']
        clf_keys = ['RF']

        # Remove any tasks that cant be done
        for task_key in task_keys[:]:
            labels = pblm.samples.subtasks[task_key]
            if len(labels.make_histogram()) < 2:
                print('No data to train task_key = %r' % (task_key,))
                task_keys.remove(task_key)

        ut.cprint('\n--- EVALUTE SIMPLE SCORES ---', 'blue')
        pblm.evaluate_simple_scores(task_keys)

        ut.cprint('\n--- LEARN CROSS-VALIDATED RANDOM FORESTS ---', 'blue')
        pblm.learn_evaluation_classifiers(task_keys, clf_keys, data_keys)

        pblm.eval_task_keys = task_keys
        pblm.eval_clf_keys = clf_keys
        pblm.eval_data_keys = data_keys

    def task_evaluation_report(pblm, task_key):
        """
        clf_keys = [pblm.default_clf_key]
        """
        # selected_data_keys = ut.ddict(list)
        from utool.experimental.pandas_highlight import to_string_monkey
        clf_keys = pblm.eval_clf_keys
        data_keys = pblm.eval_data_keys
        ut.cprint('--- TASK = %s' % (ut.repr2(task_key),), 'turquoise')
        labels = pblm.samples.subtasks[task_key]
        if hasattr(pblm, 'simple_aucs'):
            pblm.report_simple_scores(task_key)
        for clf_key in clf_keys:
            # Combine results over datasets
            print('clf_key = %s' % (ut.repr2(clf_key),))
            data_combo_res = pblm.task_combo_res[task_key][clf_key]
            df_auc_ovr = pd.DataFrame(dict([
                (datakey, list(data_combo_res[datakey].roc_scores_ovr()))
                for datakey in data_keys
            ]),
                index=labels.one_vs_rest_task_names()
            )
            ut.cprint('[%s] ROC-AUC(OVR) Scores' % (clf_key,), 'yellow')
            print(to_string_monkey(df_auc_ovr, highlight_cols='all'))

            if clf_key.endswith('-OVR') and labels.n_classes > 2:
                # Report un-normalized ovr measures if they available
                ut.cprint('[%s] ROC-AUC(OVR_hat) Scores' % (clf_key,),
                          'yellow')
                df_auc_ovr_hat = pd.DataFrame(dict([
                    (datakey,
                     list(data_combo_res[datakey].roc_scores_ovr_hat()))
                    for datakey in data_keys
                ]),
                    index=labels.one_vs_rest_task_names()
                )
                print(to_string_monkey(df_auc_ovr_hat,
                                       highlight_cols='all'))

            roc_scores = dict(
                [(datakey, [data_combo_res[datakey].roc_score()])
                 for datakey in data_keys])
            df_auc = pd.DataFrame(roc_scores)
            ut.cprint('[%s] ROC-AUC(MacroAve) Scores' % (clf_key,),
                      'yellow')
            print(to_string_monkey(df_auc, highlight_cols='all'))

            # best_data_key = 'learn(sum,glob,3)'
            best_data_key = df_auc.columns[df_auc.values.argmax(axis=1)[0]]

            # selected_data_keys[task_key].append(best_data_key)

            combo_res = data_combo_res[best_data_key]
            ut.cprint('[%s] BEST DataKey = %r' % (clf_key, best_data_key,),
                      'darkgreen')
            with ut.Indenter('[%s] ' % (best_data_key,)):
                combo_res.extended_clf_report()
            res = combo_res
            if 1:
                pos_threshes = res.report_thresholds()  # NOQA
            if 0:
                importance_datakeys = set([
                    # 'learn(all)'
                ] + [best_data_key])

                for data_key in importance_datakeys:
                    pblm.report_classifier_importance(task_key, clf_key,
                                                      data_key)

        # ut.cprint('\n--- FEATURE INFO ---', 'blue')
        # for best_data_key in selected_data_keys:
        #     print('data_key=(%s)' % (best_data_key,))
        #     print(ut.indent(AnnotPairFeatInfo(
        #           pblm.samples.X_dict[best_data_key]).get_infostr()))

        # TODO: view failure / success cases
        # Need to show and potentially fix misclassified examples
        if False:
            pblm.samples.aid_pairs
            combo_res.target_bin_df
            res = combo_res
            samples = pblm.samples
            meta = res.make_meta(samples).copy()
            import ibeis
            aid_pairs = ut.lzip(meta['aid1'], meta['aid2'])
            attrs = meta.drop(['aid1', 'aid2'], 1).to_dict(orient='list')
            ibs = pblm.qreq_.ibs
            infr = ibeis.AnnotInference.from_pairs(aid_pairs, attrs, ibs=ibs,
                                                   verbose=3)
            infr.reset_feedback('staging')
            infr.reset_labels_to_ibeis()
            infr.apply_feedback_edges()
            infr.relabel_using_reviews()
            # x = [c for c in infr.consistent_components()]
            # cc = x[ut.argmax(ut.lmap(len, x))]
            # keep = list(cc.nodes())
            # infr.remove_aids(ut.setdiff(infr.aids, keep))
            infr.start_qt_interface()
            return

    def extra_report(pblm, task_probs, is_auto, want_samples):
        task_key = 'photobomb_state'
        probs = task_probs[task_key]
        labels = want_samples[task_key]
        y_true = labels.encoded_df.loc[probs.index.tolist()]
        y_pred = probs.idxmax(axis=1).apply(labels.lookup_class_idx)
        target_names = probs.columns
        print('----------------------')
        print('Want Photobomb Report')
        clf_helpers.classification_report2(
            y_true, y_pred, target_names=target_names)

        # Make labels for entire set
        task_key = 'match_state'
        primary_probs = task_probs[task_key]
        primary_labels = want_samples[task_key]
        y_true_enc = primary_labels.encoded_df
        y_true = y_true_enc.loc[primary_probs.index.tolist()]
        y_pred = primary_probs.idxmax(axis=1).apply(
            primary_labels.lookup_class_idx)
        target_names = primary_probs.columns
        print('----------------------')
        print('Want Match Report')
        clf_helpers.classification_report2(
            y_true, y_pred, target_names=target_names)
        print('----------------------')
        print('Autoclassification Report')
        auto_edges = is_auto[is_auto].index
        clf_helpers.classification_report2(
            y_true.loc[auto_edges], y_pred.loc[auto_edges],
            target_names=target_names)
        print('----------------------')

    def auto_decisions_at_threshold(pblm, primary_task, task_probs,
                                    task_thresh, task_keys, clf_key,
                                    data_key):
        # task_thresh = {}
        # for task_key in task_keys:
        #     metric, value = operating_points[task_key]
        #     res = pblm.task_combo_res[task_key][clf_key][data_key]
        #     task_thresh[task_key] = res.get_pos_threshes(metric, value)
        # print('Using thresolds %s' % (ut.repr3(task_thresh, precision=4)))

        # Find edges that pass positive thresh and have max liklihood
        task_pos_flags = {}
        for task_key in task_keys:
            thresh = pd.Series(task_thresh[task_key])
            probs = task_probs[task_key]
            ismax_flags = probs.values.argsort(axis=1) == (probs.shape[1] - 1)
            pos_flags_df = probs > thresh
            pos_flags_df = pos_flags_df & ismax_flags
            if __debug__:
                assert all(f < 2 for f in pos_flags_df.sum(axis=1).unique()), (
                    'unsupported multilabel decision')
            task_pos_flags[task_key] = pos_flags_df

        # Define the primary task and which tasks confound it
        # Restrict auto-decisions based on if the main task is likely to be
        # confounded. (basically restrict based on photobombs)
        task_confounders = {
            'match_state': [('photobomb_state', ['pb'])],
        }
        primary_pos_flags = task_pos_flags[primary_task]

        # Determine classes that are very unlikely or likely to be confounded
        # Either: be safe, don't decide on anything that *is* confounding, OR
        # be even safer, don't decide on anything that *could* be confounding
        task_confounder_flags = pd.DataFrame()
        primary_confounders = task_confounders[primary_task]
        for task_key, confounding_classes in primary_confounders:
            pos_flags = task_pos_flags[task_key]
            nonconfounding_classes = pos_flags.columns.difference(
                confounding_classes)
            likely = pos_flags[confounding_classes].any(axis=1)
            unlikely = pos_flags[nonconfounding_classes].any(axis=1)
            flags = likely if True else likely | ~unlikely
            task_confounder_flags[task_key] = flags

        # A sample is confounded in general if is confounded by any task
        is_confounded = task_confounder_flags.any(axis=1)
        # Automatic decisions are applied to positive and unconfounded samples
        primary_auto_flags = primary_pos_flags.__and__(~is_confounded, axis=0)

        # print('Autodecision info after pos threshold')
        # print('Number positive-decisions\n%s' % primary_pos_flags.sum(axis=0))
        # # print('Percent positive-decisions\n%s' % (
        # #     100 * primary_pos_flags.sum(axis=0) / len(primary_pos_flags)))
        # # print('Total %s, Percent %.2f%%' % (primary_pos_flags.sum(axis=0).sum(),
        # #       100 * primary_pos_flags.sum(axis=0).sum() /
        # #       len(primary_pos_flags)))
        # print('Revoked autodecisions based on confounders:\n%s'  %
        #         primary_pos_flags.__and__(is_confounded, axis=0).sum())
        # print('Making #auto-decisions %s' % ut.map_dict_vals(
        #     sum, primary_auto_flags))
        return primary_auto_flags

    def make_deploy_features(pblm, infr, edges, data_key):
        """
        Create pairwise features for annotations in a test inference object
        based on the features used to learn here
        """
        candidate_edges = list(edges)
        # Parse the data_key to build the appropriate feature
        featinfo = AnnotPairFeatInfo(pblm.samples.X_dict[data_key])
        # Do one-vs-one scoring on candidate edges
        # Find the kwargs to make the desired feature subset
        pairfeat_cfg, global_keys = featinfo.make_pairfeat_cfg()
        need_lnbnn = any('lnbnn' in key for key in pairfeat_cfg['local_keys'])
        # print(featinfo.get_infostr())
        print('Building need features')
        config = pblm.hyper_params.vsone_assign
        matches, X = infr._make_pairwise_features(
            candidate_edges, config=config, pairfeat_cfg=pairfeat_cfg,
            need_lnbnn=need_lnbnn)
        assert np.all(featinfo.X.columns == X.columns), (
            'inconsistent feature dimensions')
        return X

    def predict_proba_deploy(pblm, X, task_keys):
        # import pandas as pd
        task_probs = {}
        for task_key in task_keys:
            print('[pblm] predicting %s probabilities' % (task_key,))
            clf = pblm.deploy_task_clfs[task_key]
            labels = pblm.samples.subtasks[task_key]
            probs_df = clf_helpers.predict_proba_df(
                clf, X, labels.class_names)
            # columns = ut.take(labels.class_names, clf.classes_)
            # probs_df = pd.DataFrame(
            #     clf.predict_proba(X),
            #     columns=columns, index=X.index
            # )
            # # add in zero probability for classes without training data
            # missing = ut.setdiff(labels.class_names, columns)
            # if missing:
            #     for classname in missing:
            #         # print('classname = %r' % (classname,))
            #         probs_df = probs_df.assign(**{
            #             classname: np.zeros(len(probs_df))})
            task_probs[task_key] = probs_df
        return task_probs

    @profile
    def predict_proba_evaluation(pblm, infr, want_edges, task_keys=None,
                                 clf_key=None, data_key=None):
        """
        Note: Ideally we should use a completely independant dataset to test.
        However, due to lack of labeled photobombs and notcomparable cases we
        can cheat a little. A subset of want_edges were previously used in
        training, but there is one classifier that never saw it. We use this
        classifier to predict on that case. For completely unseen data we use
        the average probability of all classifiers.

        NOTE: Using the cross-validated training data to select thresholds
        breaks these test independence assumptions. You really should use a
        completely disjoint test set.
        """
        if clf_key is None:
            clf_key = pblm.default_clf_key
        if data_key is None:
            data_key = pblm.default_data_key
        if task_keys is None:
            task_keys = [pblm.primary_task_key]
        # Construct the matches
        # TODO: move probability predictions into the depcache
        config = pblm.hyper_params.vsone_assign
        prob_cfgstr = '_'.join([
            infr.ibs.dbname,
            ut.hashstr3(np.array(want_edges)),
            data_key,
            clf_key,
            repr(task_keys),
            config.get_cfgstr()
        ])
        cacher2 = ut.Cacher('full_eval_probs', prob_cfgstr,
                            appname=pblm.appname, verbose=20)
        data2 = cacher2.tryload()
        if not data2:
            # Choose a classifier for each task
            res_dict = dict([
                (task_key, pblm.task_combo_res[task_key][clf_key][data_key])
                for task_key in task_keys
            ])
            assert ut.allsame([res.probs_df.index for res in res_dict.values()]), (
                'inconsistent combined result indices')

            # Normalize and align combined result sample edges
            res0 = next(iter(res_dict.values()))
            train_uv = np.array(res0.probs_df.index.tolist())
            assert np.all(train_uv.T[0] < train_uv.T[1]), (
                'edges must be in lower triangular form')
            assert len(vt.unique_row_indexes(train_uv)) == len(train_uv), (
                'edges must be unique')
            assert (sorted(ut.lmap(tuple, train_uv.tolist())) ==
                    sorted(ut.lmap(tuple, pblm.samples.aid_pairs.tolist())))
            want_uv = np.array(want_edges)

            # Determine which edges need/have probabilities
            want_uv_, train_uv_ = vt.structure_rows(want_uv, train_uv)
            unordered_have_uv_ = np.intersect1d(want_uv_, train_uv_)
            need_uv_ = np.setdiff1d(want_uv_, unordered_have_uv_)
            flags = vt.flag_intersection(train_uv_, unordered_have_uv_)
            # Re-order have_edges to agree with test_idx
            have_uv_ = train_uv_[flags]
            need_uv, have_uv = vt.unstructure_rows(need_uv_, have_uv_)

            # Convert to tuples for pandas lookup. bleh...
            have_edges = ut.lmap(tuple, have_uv.tolist())
            need_edges = ut.lmap(tuple, need_uv.tolist())
            want_edges = ut.lmap(tuple, want_uv.tolist())
            assert set(have_edges) & set(need_edges) == set([])
            assert set(have_edges) | set(need_edges) == set(want_edges)

            infr.classifiers = pblm
            matches, X_need = infr._pblm_pairwise_features(need_edges,
                                                           data_key)
            # Make an ensemble of the evaluation classifiers
            # (todo: use a classifier that hasn't seen any of this data)
            task_need_probs = {}
            for task_key in task_keys:
                print('Predicting %s probabilities' % (task_key,))
                clf_list = pblm.eval_task_clfs[task_key][clf_key][data_key]
                labels = pblm.samples.subtasks[task_key]
                eclf = clf_helpers.voting_ensemble(clf_list, voting='soft')
                eclf_probs = clf_helpers.predict_proba_df(eclf, X_need,
                                                          labels.class_names)
                task_need_probs[task_key] = eclf_probs

            # Combine probabilities --- get probabilites for each sample
            # edges = have_edges + need_edges
            task_probs = {}
            for task_key in task_keys:
                eclf_probs = task_need_probs[task_key]
                have_probs = res_dict[task_key].probs_df.loc[have_edges]
                task_probs[task_key] = pd.concat([have_probs, eclf_probs])
                assert have_probs.index.intersection(eclf_probs.index).size == 0
            data2 = task_probs
            cacher2.save(data2)
        task_probs = data2
        return task_probs

    def reduce_dataset_size(pblm):
        """
        Reduce the size of the dataset for development speed

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty()
            >>> pblm.load_samples()
            >>> pblm.load_features()
            >>> pblm.reduce_dataset_size()
        """
        from six import next
        labels = next(iter(pblm.samples.subtasks.values()))
        ut.assert_eq(len(labels), len(pblm.samples), verbose=False)

        if 0:
            # Remove singletons
            unique_aids = np.unique(pblm.samples.aid_pairs)
            nids = pblm.ibs.get_annot_nids(unique_aids)
            singleton_nids = set([nid for nid, v in ut.dict_hist(nids).items() if v == 1])
            nid_flags = [nid in singleton_nids for nid in nids]
            singleton_aids = set(ut.compress(unique_aids, nid_flags))
            mask = [not (a1 in singleton_aids or a2 in singleton_aids)
                     for a1, a2 in pblm.samples.aid_pairs]
            print('Removing %d pairs based on singleton' % (len(mask) - sum(mask)))
            pblm.samples = samples2 = pblm.samples.compress(mask)
            # samples2.print_info()
            # print('---------------')
            labels = next(iter(samples2.subtasks.values()))
            ut.assert_eq(len(labels), len(samples2), verbose=False)
            pblm.samples = samples2

        if 0:
            # Remove anything 1vM didn't get
            mask = (pblm.samples.simple_scores['score_lnbnn_1vM'] > 0).values
            print('Removing %d pairs based on LNBNN failure' % (len(mask) - sum(mask)))
            pblm.samples = samples3 = pblm.samples.compress(mask)
            # samples3.print_info()
            # print('---------------')
            labels = next(iter(samples3.subtasks.values()))
            ut.assert_eq(len(labels), len(samples3), verbose=False)
            pblm.samples = samples3

        from sklearn.utils import random

        if False:
            # Choose labels to balance
            labels = pblm.samples.subtasks['match_state']
            unique_labels, groupxs = ut.group_indices(labels.y_enc)
            #
            # unique_labels, groupxs = ut.group_indices(pblm.samples.encoded_1d())

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
            mask = ut.index_to_boolmask(idxs, len(pblm.samples))
            print('Removing %d pairs for class balance' % (len(mask) - sum(mask)))
            pblm.samples = samples4 = pblm.samples.compress(mask)
            # samples4.print_info()
            # print('---------------')
            labels = next(iter(samples4.subtasks.values()))
            ut.assert_eq(len(labels), len(samples4), verbose=False)
            pblm.samples = samples4
            # print('hist(y) = ' + ut.repr4(pblm.samples.make_histogram()))

        # if 0:
        #     print('Random dataset size reduction for development')
        #     rng = np.random.RandomState(1851057325)
        #     num = len(pblm.samples)
        #     to_keep = rng.choice(np.arange(num), 1000)
        #     mask = np.array(ut.index_to_boolmask(to_keep, num))
        #     pblm.samples = pblm.samples.compress(mask)
        #     class_hist = pblm.samples.make_histogram()
        #     print('hist(y) = ' + ut.repr4(class_hist))
        labels = next(iter(pblm.samples.subtasks.values()))
        ut.assert_eq(len(labels), len(pblm.samples), verbose=False)

    def build_feature_subsets(pblm):
        """
        Try to identify a useful subset of features to reduce problem
        dimensionality
        """
        if pblm.verbose:
            print('[pblm] build_feature_subsets')
        X_dict = pblm.samples.X_dict
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

            if 1:
                cols = featinfo.select_columns([
                    ('measure_type', '==', 'summary'),
                ])
                cols.update(featinfo.select_columns([
                    ('measure_type', '==', 'global'),
                ]))
                cols = [c for c in cols if 'lnbnn' not in c]
                X_dict['learn(sum,glob,4)'] = featinfo.X[sorted(cols)]

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
        pblm.samples.X_dict = X_dict

    def print_featinfo(pblm, data_key=None):
        if data_key is None:
            data_key = pblm.default_data_key
        X = pblm.samples.X_dict[data_key]
        featinfo = AnnotPairFeatInfo(X)
        print(featinfo.get_infostr())

    def evaluate_simple_scores(pblm, task_keys=None):
        """
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty()
            >>> pblm.set_pandas_options()
            >>> pblm.load_samples()
            >>> pblm.load_features()
            >>> pblm.evaluate_simple_scores()
        """
        score_dict = pblm.samples.simple_scores.copy()
        if True:
            # Remove scores that arent worth reporting
            for k in list(score_dict.keys())[:]:
                ignore = [
                    'sum(norm_x', 'sum(norm_y',
                    'sum(sver_err', 'sum(scale',
                    'sum(match_dist)',
                    'sum(weighted_norm_dist',
                ]
                # if pblm.qreq_.qparams.featweight_enabled:
                #     ignore.extend([
                #         # 'sum(norm_dist)',
                #         # 'sum(ratio)',
                #         # 'sum(lnbnn)',
                #         # 'sum(lnbnn_norm_dist)'
                #     ])
                flags = [part in k for part in ignore]
                if any(flags):
                    del score_dict[k]

        if task_keys is None:
            task_keys = list(pblm.samples.subtasks.keys())

        simple_aucs = {}
        for task_key in task_keys:
            task_aucs = {}
            labels = pblm.samples.subtasks[task_key]
            for sublabels in labels.gen_one_vs_rest_labels():
                sublabel_aucs = {}
                for scoretype in score_dict.keys():
                    scores = score_dict[scoretype].values
                    auc = sklearn.metrics.roc_auc_score(sublabels.y_enc, scores)
                    sublabel_aucs[scoretype] = auc
                # task_aucs[sublabels.task_key] = sublabel_aucs
                task_aucs[sublabels.task_name.replace(task_key, '')] = sublabel_aucs
            simple_aucs[task_key] = task_aucs
        pblm.simple_aucs = simple_aucs

    def report_simple_scores(pblm, task_key):
        force_keep = ['score_lnbnn_1vM']
        simple_aucs = pblm.simple_aucs
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

    def report_classifier_importance(pblm, task_key, clf_key, data_key):
        r"""
        CommandLine:
            python -m ibeis.scripts.script_vsone report_classifier_importance --show

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('GZ_Master1')
            >>> data_key = pblm.default_data_key
            >>> clf_key = pblm.default_clf_key
            >>> task_key = pblm.primary_task_key
            >>> pblm.setup_evaluation()
            >>> importances = pblm.report_classifier_importance(task_key, clf_key, data_key)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> text = importances
            >>> pt.wordcloud(importances)
            >>> ut.show_if_requested()
        """
        # ut.qtensure()
        # import plottool as pt  # NOQA

        if clf_key != 'RF':
            print('Cannot only report importance for RF not %r' % (clf_key,))
            return

        X = pblm.samples.X_dict[data_key]
        # Take average feature importance
        ut.cprint('MARGINAL IMPORTANCE INFO for %s on task %s' % (data_key, task_key), 'yellow')
        print(' Caption:')
        print(' * The NaN row ensures that `weight` always sums to 1')
        print(' * `num` indicates how many dimensions the row groups')
        print(' * `ave_w` is the average importance a single feature in the row')
        # with ut.Indenter('[%s] ' % (data_key,)):

        clf_list = pblm.eval_task_clfs[task_key][clf_key][data_key]
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
        return importances

    def report_classifier_importance2(pblm, clf, data_key=None):
        if data_key is None:
            data_key = pblm.default_data_key
        X = pblm.samples.X_dict[data_key]
        assert len(clf.feature_importances_) == len(X.columns)
        importances = ut.dzip(X.columns, clf.feature_importances_)
        featinfo = AnnotPairFeatInfo(X, importances)
        featinfo.print_margins('feature')
        featinfo.print_margins('measure_type')
        featinfo.print_margins('summary_op')
        featinfo.print_margins('summary_measure')
        featinfo.print_margins('global_measure')

    def demo_classes(pblm):
        r"""
        CommandLine:
            python -m ibeis.scripts.script_vsone demo_classes --saveparts --save=classes.png --clipwhite

            python -m ibeis.scripts.script_vsone demo_classes --saveparts --save=figures/classes.png --clipwhite --dpath=~/latex/crall-iccv-2017

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='PZ_PB_RF_TRAIN')
            >>> pblm.load_features()
            >>> pblm.load_samples()
            >>> pblm.build_feature_subsets()
            >>> pblm.demo_classes()
            >>> ut.show_if_requested()
        """
        task_key = 'match_state'
        labels = pblm.samples.subtasks[task_key]
        pb_labels = pblm.samples.subtasks['photobomb_state']
        classname_offset = {
            POSTV: 0,
            NEGTV: 0,
            INCMP: 0,
        }
        class_name = POSTV
        class_name = NEGTV
        class_name = INCMP

        feats = pblm.samples.X_dict['learn(sum,glob)']

        offset = 0
        class_to_edge = {}
        for class_name in labels.class_names:
            print('Find example of %r' % (class_name,))
            # Find an example of each class (that is not a photobomb)
            pbflags = pb_labels.indicator_df['notpb']
            flags = labels.indicator_df[class_name]
            assert np.all(pbflags.index == flags.index)
            flags = flags & pbflags
            ratio = feats['sum(ratio)']
            if class_name == INCMP:
                flags &= feats['global(yaw_delta)'] > 3
                # flags &= feats['sum(ratio)'] > 0
            if class_name == NEGTV:
                low = ratio[flags].max()
                flags &= feats['sum(ratio)'] >= low
            if class_name == POSTV:
                low = ratio[flags].median() / 2
                high = ratio[flags].median()
                flags &= feats['sum(ratio)'] < high
                flags &= feats['sum(ratio)'] > low
            # flags &= pblm.samples.simple_scores[flags]['score_lnbnn_1vM'] > 0
            idxs = np.where(flags)[0]
            print('Found %d candidates' % (len(idxs)))
            offset = classname_offset[class_name]
            idx = idxs[offset]
            series = labels.indicator_df.iloc[idx]
            assert series[class_name]
            edge = series.name
            class_to_edge[class_name] = edge

        import plottool as pt
        import guitool as gt
        gt.ensure_qapp()
        pt.qtensure()

        fnum = 1
        pt.figure(fnum=fnum, pnum=(1, 3, 1))
        pnum_ = pt.make_pnum_nextgen(1, 3)

        # classname_alias = {
        #     POSTV: 'positive',
        #     NEGTV: 'negative',
        #     INCMP: 'incomparable',
        # }

        ibs = pblm.infr.ibs
        for class_name in class_to_edge.keys():
            edge = class_to_edge[class_name]
            aid1, aid2 = edge
            # alias = classname_alias[class_name]
            print('class_name = %r' % (class_name,))
            annot1 = ibs.annots([aid1])[0]._make_lazy_dict()
            annot2 = ibs.annots([aid2])[0]._make_lazy_dict()
            vt.matching.ensure_metadata_normxy(annot1)
            vt.matching.ensure_metadata_normxy(annot2)
            match = vt.PairwiseMatch(annot1, annot2)
            cfgdict = pblm.hyper_params.vsone_assign.asdict()
            match.apply_all(cfgdict)
            pt.figure(fnum=fnum, pnum=pnum_())
            match.show(show_ell=False, show_ori=False)
            # pt.set_title(alias)


@ut.reloadable_class
class AnnotPairSamples(clf_helpers.MultiTaskSamples):
    """
    Manages the different ways to assign samples (i.e. feat-label pairs) to
    1-v-1 classification

    CommandLine:
        python -m ibeis.scripts.script_vsone AnnotPairSamples

    Example:
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> pblm = OneVsOneProblem.from_empty()
        >>> pblm.load_samples()
        >>> samples = AnnotPairSamples(pblm.ibs, pblm.raw_simple_scores, {})
        >>> print(samples)
        >>> samples.print_info()
        >>> print(samples.sample_hashid())
        >>> encode_index = samples.subtasks['match_state'].encoded_df.index
        >>> indica_index = samples.subtasks['match_state'].indicator_df.index
        >>> assert np.all(samples.index == encode_index)
        >>> assert np.all(samples.index == indica_index)
    """
    def __init__(samples, ibs, aid_pairs, infr=None):
        super(AnnotPairSamples, samples).__init__(aid_pairs)
        samples.aid_pairs = np.array(aid_pairs)
        samples.infr = infr
        samples.ibs = ibs
        samples.annots1 = ibs.annots(samples.aid_pairs.T[0], asarray=True)
        samples.annots2 = ibs.annots(samples.aid_pairs.T[1], asarray=True)
        samples.n_samples = len(aid_pairs)
        samples.X_dict = None
        samples.simple_scores = None
        samples.apply_multi_task_multi_label()
        # samples.apply_multi_task_binary_label()

    def edge_uuids(samples):
        qvuuids = samples.annots1.visual_uuids
        dvuuids = samples.annots2.visual_uuids
        edge_uuids = [ut.combine_uuids(uuids)
                       for uuids in zip(qvuuids, dvuuids)]
        return edge_uuids

    def edge_hashid(samples):
        edge_hashid = ut.hashstr_arr27(samples.edge_uuids(), 'edges',
                                       hashlen=32, pathsafe=True)
        return edge_hashid

    @ut.memoize
    def sample_hashid(samples):
        visual_hash = samples.edge_hashid()
        label_hash = ut.hashstr_arr27(samples.encoded_1d(), 'labels',
                                      pathsafe=True)
        sample_hash = visual_hash + '_' + label_hash
        return sample_hash

    def set_feats(samples, simple_scores, X_dict):
        edges = ut.lmap(tuple, samples.aid_pairs.tolist())
        if simple_scores is not None:
            assert (edges == simple_scores.index.tolist())

        if X_dict is not None:
            for X in X_dict.values():
                assert np.all(edges == X.index.tolist())
        samples.X_dict = X_dict
        samples.simple_scores = simple_scores

    @property
    def primary_task(samples):
        primary_task_key = 'match_state'
        primary_task =  samples.subtasks[primary_task_key]
        return primary_task

    def compress(samples, flags):
        """
        flags = np.zeros(len(samples), dtype=np.bool)
        flags[0] = True
        flags[3] = True
        flags[4] = True
        flags[-1] = True
        """
        assert len(flags) == len(samples), 'mask has incorrect size'
        infr = samples.infr
        simple_scores = samples.simple_scores[flags]
        X_dict = ut.map_vals(lambda val: val[flags], samples.X_dict)
        aid_pairs = samples.aid_pairs[flags]
        ibs = samples.ibs
        new_labels = AnnotPairSamples(ibs, aid_pairs, infr)
        new_labels.set_feats(simple_scores, X_dict)
        return new_labels

    @ut.memoize
    def is_same(samples):
        infr = samples.infr
        edges = samples.aid_pairs
        def _check(u, v):
            nid1, nid2 = infr.pos_graph.node_labels(u, v)
            if nid1 == nid2:
                return True
            elif infr.neg_redun_nids.has_edge(nid1, nid2):
                return False
            else:
                return None
        flags = [_check(*edge) for edge in edges]
        return np.array(flags, dtype=np.bool)
        # return samples.infr.is_same(samples.aid_pairs)

    @ut.memoize
    def is_photobomb(samples):
        infr = samples.infr
        edges = samples.aid_pairs
        flags = [
            'photobomb' in d['tags'] if d is not None else None
            for d in ut.lstarmap(infr.graph.get_edge_data, edges)
        ]
        return np.array(flags, dtype=np.bool)
        # return samples.infr.is_photobomb(samples.aid_pairs)

    @ut.memoize
    def is_comparable(samples):
        infr = samples.infr
        edges = samples.aid_pairs
        def _check(u, v):
            if infr.incomp_graph.has_edge(u, v):
                return False
            elif infr.pos_graph.has_edge(u, v):
                return True
            elif infr.neg_graph.has_edge(u, v):
                return True
            return np.nan
        flags = np.array([_check(*edge) for edge in edges])
        # hack guess if comparable based on viewpoint
        guess_flags = np.isnan(flags)
        need_edges = edges[guess_flags]
        need_flags = infr.ibeis_guess_if_comparable(need_edges)
        flags[guess_flags] = need_flags
        return np.array(flags, dtype=np.bool)
        # return samples.infr.is_comparable(samples.aid_pairs, allow_guess=True)

    def apply_multi_task_multi_label(samples):
        # multioutput-multiclass / multi-task
        tasks_to_indicators = ut.odict([
            ('match_state', ut.odict([
                (NEGTV, ~samples.is_same() & samples.is_comparable()),
                (POSTV,  samples.is_same() & samples.is_comparable()),
                (INCMP, ~samples.is_comparable()),
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
                # (NEGTV, ~samples.is_same() | ~samples.is_comparable()),
                # (POSTV,    samples.is_same() & samples.is_comparable()),
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

    @property
    def group_ids(samples):
        """
        Prevents samples with the same group-id from appearing in the same
        cross validation fold. For us this means any pair within the same
        name or between the same names will have the same groupid.
        """
        infr = samples.infr
        name_edges = np.array([
            infr.e_(*infr.pos_graph.node_labels(u, v))
            for u, v in samples.aid_pairs])
        # Edges within the same name or between the same name, must be grouped
        # together. This will prevent identity-specific effects.
        group_ids = vt.get_undirected_edge_ids(name_edges)
        return group_ids


AnnotPairFeatInfo = vt.AnnotPairFeatInfo


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
