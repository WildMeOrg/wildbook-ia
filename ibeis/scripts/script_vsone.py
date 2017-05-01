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
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass
import sklearn.ensemble
from ibeis.scripts import clf_helpers
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP
from six.moves import zip
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
    """ for building pairwise feature dimensions """
    _param_info_list = [
        ut.ParamInfo('indices', slice(0, 26, 5)),
        ut.ParamInfo('summary_ops', {'sum', 'std', 'mean', 'len', 'med'}),
        ut.ParamInfo('local_keys', None),
        ut.ParamInfo('sorters', [
            'ratio', 'norm_dist', 'match_dist'
            # 'lnbnn', 'lnbnn_norm_dist',
        ]),
        # ut.ParamInfo('bin_key', None, valid_values=[None, 'ratio']),
        ut.ParamInfo('bin_key', 'ratio', valid_values=[None, 'ratio']),
        ut.ParamInfo('bins', [.5, .6, .7, .8])
        # ut.ParamInfo('need_lnbnn', False),

        # ut.ParamInfo('med', True),
    ]


class VsOneMatchConfig(dt.Config):
    _param_info_list = vt.matching.VSONE_DEFAULT_CONFIG


class VsOneFeatConfig(dt.Config):
    """ keypoint params """
    _param_info_list = vt.matching.VSONE_FEAT_CONFIG


@ut.reloadable_class
class OneVsOneProblem(clf_helpers.ClfProblem):
    """
    Keeps information about the one-vs-one pairwise classification problem

    CommandLine:
        python -m ibeis.scripts.script_vsone evaluate_classifiers
        python -m ibeis.scripts.script_vsone evaluate_classifiers --db PZ_PB_RF_TRAIN
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

        pblm.raw_X_dict = None
        pblm.raw_simple_scores = None
        pblm.samples = None
        pblm.simple_aucs = None
        pblm.eval_task_keys = None
        pblm.eval_clf_keys = None
        pblm.eval_data_keys = None

        pblm.verbose = verbose
        pblm.primary_task_key = 'match_state'
        pblm.default_data_key = 'learn(sum,glob)'
        pblm.default_clf_key = 'RF'

        pblm.infr = infr
        pblm.qreq_ = None

        hyper_params = dt.Config.from_dict(ut.odict([
            ('subsample', None),
            ('pair_sample', PairSampleConfig()),
            ('vsone_kpts', VsOneFeatConfig()),
            ('vsone_match', VsOneMatchConfig()),
            ('pairwise_feats', PairFeatureConfig()),
            ('sample_search', dict(K=4, Knorm=1, requery=True,
                                   score_method='csum', prescore_method='csum')),
        ]),
            tablename='HyperParams'
        )
        maxbin = max(hyper_params['pairwise_feats']['bins'])
        hyper_params['vsone_match']['ratio_thresh'] = maxbin
        hyper_params['vsone_match']['sv_on'] = True

        if False:
            # For giraffes
            hyper_params['vsone_match']['checks'] = 80
            hyper_params['vsone_match']['sver_xy_thresh'] = .02
            hyper_params['vsone_match']['sver_ori_thresh'] = 3

        species = infr.ibs.get_primary_database_species()
        if species == 'zebra_plains' or True:
            hyper_params['vsone_match']['Knorm'] = 3
            hyper_params['vsone_match']['symmetric'] = True
            hyper_params['vsone_kpts']['augment_orientation'] = True

        pblm.hyper_params = hyper_params

    @classmethod
    def from_aids(OneVsOneProblem, ibs, aids, verbose=None):
        pblm = OneVsOneProblem(ibs=ibs, qaid_override=aids, daid_override=aids,
                               verbose=verbose)
        return pblm

    def _update_girm(self):
        # ibs2 = ibeis.opendb('NNP_MasterGIRM_core')
        import ibeis
        defaultdb = 'GIRM_Master1'
        ibs, aids = ibeis.testdata_aids(defaultdb)
        infr = ibeis.AnnotInference(ibs=ibs, aids=aids, autoinit=True)

        if False:
            df = infr._feedback_df('annotmatch')
            match_ams = df[(df['decision'] == 'match')]['am_rowid']
            aids1 = match_ams.index.get_level_values(0).values
            aids2 = match_ams.index.get_level_values(1).values
            ibs.get_annotmatch_truth(match_ams)

        # Hack to precompute data for slightly faster viz
        qreq2_ = ibs.new_query_request(
            infr.aids, infr.aids, cfgdict={}, verbose=False)
        qreq2_.ensure_chips()
        qreq2_.ensure_features()


        infr.reset_feedback('annotmatch', apply=True)

        # The annotmatches do not agree with the names

        # Assume name labels are correct, fix the annot matches
        from ibeis.algo.graph import nx_utils
        node_to_label = infr.get_node_attrs('orig_name_label')
        label_to_nodes = ut.group_items(node_to_label.keys(),
                                        node_to_label.values())

        bad_edges = []
        priorities = []
        dummy = 1.5
        for cc1, cc2 in it.combinations(label_to_nodes.values(), 2):
            edges = nx_utils.edges_cross(infr.graph, set(cc1), set(cc2))
            datas = [infr.get_edge_data(e) for e in edges]
            bad = [e for e, d in zip(edges, datas)
                   if d.get('decision') == POSTV]
            # if len(bad) > 1:
            priorities.extend([dummy] * len(bad))
            dummy += 1
            bad_edges.extend(bad)
        print(len(bad_edges))

        infr.enable_redundancy = False
        infr.fix_mode_merge = False
        infr.fix_mode_predict = True
        infr.enable_inference = False
        infr.fix_mode_split = True
        infr.classifiers = None

        infr.set_edge_attrs('disagrees', ut.dzip(bad_edges, priorities))
        infr.prioritize('disagrees', bad_edges, reset=True)
        # infr.apply_nondynamic_update()
        infr.verbose = 10

        win = infr.qt_review_loop()

        infr.enable_inference = True
        win = infr.qt_review_loop()

        df = infr.match_state_delta('annotmatch', 'staging')
        df = infr.match_state_delta('staging', 'all')


    @classmethod
    def from_empty(OneVsOneProblem, defaultdb=None):
        """
        >>> from ibeis.scripts.script_vsone import *  # NOQA
        >>> defaultdb = 'GIRM_Master1'
        >>> pblm = OneVsOneProblem.from_empty(defaultdb)
        """
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
        if infr.ibs.dbname not in {'GIRM_Master1', 'NNP_MasterGIRM_core'}:
            assert infr._is_staging_above_annotmatch()
        infr.reset_feedback('staging', apply=True)
        if infr.ibs.dbname == 'PZ_MTEST':
            # assert False, 'need to do conversion'
            infr.ensure_mst()
        # if infr.needs_conversion():
        #     infr.ensure_mst()
        pblm = OneVsOneProblem(infr=infr)
        return pblm

    def _fix_hyperparams(pblm, qreq_):
        if qreq_.qparams.featweight_enabled:
            pblm.hyper_params.vsone_match['weight'] = 'fgweights'
            pblm.hyper_params.pairwise_feats['sorters'] = ut.unique(
                pblm.hyper_params.pairwise_feats['sorters'] +
                [
                    'weighted_ratio',
                    # 'weighted_lnbnn'
                ]
            )
        else:
            pblm.hyper_params.vsone_match['weight'] = None

    @profile
    def make_training_pairs(pblm):
        infr = pblm.infr
        ibs = pblm.infr.ibs
        if pblm.verbose > 0:
            print('[pblm] gather match-state hard cases')

        cfgdict = pblm.hyper_params['sample_search'].copy()
        # Use the same keypoints for vsone and vsmany for comparability
        cfgdict.update(pblm.hyper_params['vsone_kpts'])

        aids = ibs.filter_annots_general(
            infr.aids, min_pername=3, species='primary')

        infr.relabel_using_reviews(rectify=False)
        custom_nid_lookup = infr.get_node_attrs('name_label', aids)
        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      verbose=False,
                                      custom_nid_lookup=custom_nid_lookup)
        pblm._fix_hyperparams(qreq_)

        use_cache = False
        use_cache = True
        cfgstr = qreq_.get_cfgstr(with_input=True)
        cacher1 = ut.Cacher('pairsample_1_v5', cfgstr=cfgstr,
                            appname=pblm.appname, enabled=use_cache,
                            verbose=pblm.verbose + 10)
        assert qreq_.qparams.can_match_samename is True
        assert qreq_.qparams.prescore_method == 'csum'
        assert pblm.hyper_params.subsample is None
        data = cacher1.tryload()
        if data is None:
            cm_list = qreq_.execute()
            infr._set_vsmany_info(qreq_, cm_list)

            # Sample hard moderate and easy positives / negative
            # For each query, choose same, different, and random training pairs
            rng = np.random.RandomState(42)
            aid_pairs_ = infr._cm_training_pairs(
                rng=rng, **pblm.hyper_params.pair_sample)
            cacher1.save(aid_pairs_)
            data = aid_pairs_
        aid_pairs_ = data

        # TODO: it would be nice to have a ibs database proprty that changes
        # whenever any value in a primary table changes
        cacher2 = ut.Cacher('pairsample_2_v5', cfgstr=cfgstr,
                            appname=pblm.appname, enabled=use_cache,
                            verbose=pblm.verbose + 10)
        data = cacher2.tryload()
        if data is None:
            if pblm.verbose > 0:
                print('[pblm] gather photobomb and incomparable cases')
            pb_aid_pairs = infr.photobomb_samples()
            incomp_aid_pairs = list(infr.incomp_graph.edges())
            data = pb_aid_pairs, incomp_aid_pairs
            cacher2.save(data)
        pb_aid_pairs, incomp_aid_pairs = data

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
        # TODO: features should also rely on global attributes
        # fornow using edge+label hashids is good enough
        sample_hashid = pblm.samples.sample_hashid()
        feat_cfgstr = hyper_params.get_cfgstr()
        feat_hashid = ut.hashstr27(sample_hashid + feat_cfgstr)
        # print('features_hashid = %r' % (features_hashid,))
        cfgstr = '_'.join(['devcache', str(dbname), feat_hashid])
        # use_cache = False
        cacher = ut.Cacher('pairwise_data_v17', cfgstr=cfgstr,
                           appname=pblm.appname, enabled=use_cache,
                           verbose=pblm.verbose)
        data = cacher.tryload()
        if not data:
            config = {}
            config.update(hyper_params.vsone_match)
            config.update(hyper_params.vsone_kpts)
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

        # hack to fix feature validity
        if np.any(np.isinf(X_all['global(speed)'])):
            flags = np.isinf(X_all['global(speed)'])
            numer = X_all.loc[flags, 'global(gps_delta)']
            denom = X_all.loc[flags, 'global(time_delta)']
            newvals = np.full(len(numer), np.nan)
            newvals[(numer == 0) & (denom == 0)] = 0
            X_all.loc[flags, 'global(speed)'] = newvals

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
            python -m ibeis.scripts.script_vsone evaluate_classifiers --db GZ_Master1
            python -m ibeis.scripts.script_vsone evaluate_classifiers --db GIRM_Master1

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
        pblm.samples.print_info()
        print('---------------')

        ut.cprint('\n--- FEATURE INFO ---', 'blue')
        pblm.build_feature_subsets()

        pblm.samples.print_featinfo()

        task_keys = pblm.eval_task_keys
        clf_keys = pblm.eval_clf_keys
        data_keys = pblm.eval_data_keys
        if task_keys is None:
            task_keys = list(pblm.samples.subtasks.keys())
        if clf_keys is None:
            clf_keys = ['RF']
        if data_keys is None:
            data_keys = list(pblm.samples.X_dict.keys())
        pblm.eval_task_keys = task_keys
        pblm.eval_clf_keys = clf_keys
        pblm.eval_data_keys = data_keys

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
                    pblm.report_importance(task_key, clf_key,
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
            meta = res.hardness_analysis(samples).copy()
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
        config = {}
        config.update(pblm.hyper_params.vsone_match)
        config.update(pblm.hyper_params.vsone_kpts)
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
        prob_cfgstr = '_'.join([
            infr.ibs.dbname,
            ut.hashstr3(np.array(want_edges)),
            data_key,
            clf_key,
            repr(task_keys),
            pblm.hyper_params.vsone_match.get_cfgstr(),
            pblm.hyper_params.vsone_kpts.get_cfgstr(),
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

    def build_feature_subsets(pblm):
        """
        Try to identify a useful subset of features to reduce problem
        dimensionality

        CommandLine:
            python -m ibeis.scripts.script_vsone build_feature_subsets --db GZ_Master1
            python -m ibeis.scripts.script_vsone build_feature_subsets --db PZ_PB_RF_TRAIN

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST')
            >>> pblm.load_samples()
            >>> pblm.load_features()
            >>> pblm.build_feature_subsets()
            >>> pblm.samples.print_featinfo()
        """
        if pblm.verbose:
            print('[pblm] build_feature_subsets')
        X_dict = pblm.samples.X_dict
        X = X_dict['learn(all)']
        featinfo = AnnotPairFeatInfo(X)

        if False:
            # Use only summary stats
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            X_dict['learn(sum)']  = featinfo.X[sorted(cols)]

        if True:
            # Use summary and global single thresholds with raw unaries
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            cols.update(featinfo.select_columns([
                ('measure_type', '==', 'global'),
                ('measure', 'not in', [
                    'qual_1', 'qual_2', 'yaw_1', 'yaw_2',
                    'gps_1[0]', 'gps_2[0]', 'gps_1[1]', 'gps_2[1]',
                    'time_1', 'time_2'
                ])
            ]))
            X_dict['learn(sum,glob)'] = featinfo.X[sorted(cols)]

            if True:
                # Remove view columns
                view_cols = featinfo.select_columns([
                    ('measure_type', '==', 'global'),
                    ('measure', 'in', ['yaw_1', 'yaw_2', 'yaw_delta',
                                       'min_yaw', 'max_yaw']),
                ])
                cols = set.difference(cols, view_cols)
                X_dict['learn(sum,glob,-view)'] = featinfo.X[sorted(cols)]

        if True:
            # Use summary and global single thresholds with raw unaries
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
                ('summary_binval', '==', '0.625'),
            ])
            cols.update(featinfo.select_columns([
                ('measure_type', '==', 'global'),
                ('measure', 'not in', [
                    'qual_1', 'qual_2', 'yaw_1', 'yaw_2',
                    'gps_1[0]', 'gps_2[0]', 'gps_1[1]', 'gps_2[1]',
                    'time_1', 'time_2'
                ])
            ]))
            X_dict['learn(sum,glob,single)'] = featinfo.X[sorted(cols)]

        if False:
            # Use summary and global single thresholds with raw unaries
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            cols.update(featinfo.select_columns([
                ('measure_type', '==', 'global'),
                ('measure', 'not in', [
                    'min_qual', 'max_qual', 'min_yaw', 'max_yaw']),
            ]))
            # cols = [c for c in cols if 'lnbnn' not in c]
            X_dict['learn(sum,rawglob)'] = featinfo.X[sorted(cols)]

        pblm.samples.X_dict = X_dict

    def evaluate_simple_scores(pblm, task_keys=None):
        """
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty()
            >>> pblm.set_pandas_options()
            >>> pblm.load_samples()
            >>> pblm.load_features()
            >>> pblm.evaluate_simple_scores()
        """
        if task_keys is None:
            task_keys = [pblm.primary_task_key]

        score_dict = pblm.samples.simple_scores.copy()
        if True:
            # Remove scores that arent worth reporting
            for k in list(score_dict.keys())[:]:
                ignore = [
                    'sum(norm_x', 'sum(norm_y',  # ))
                    'sum(sver_err', 'sum(scale',  # ))
                    'sum(match_dist)',
                    'sum(weighted_norm_dist',  # )
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

        # pblm.task_simple_res = ut.AutoVivification()
        # for simple_key in score_dict.keys():
        #     X = score_dict[[simple_key]]
        #     labels = pblm.samples.subtasks[task_key]
        #     ClfResult.make_single(clf, X, test_idx, labels, data_key)
        #     pblm.task_simple_res[task_key][simple_key]

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

    def report_simple_scores(pblm, task_key=None):
        if task_key is None:
            task_key = pblm.primary_task_key
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

    def feature_importance(pblm, task_key=None, clf_key=None, data_key=None):
        r"""
        CommandLine:
            python -m ibeis.scripts.script_vsone report_importance --show
            python -m ibeis.scripts.script_vsone report_importance --show --db PZ_PB_RF_TRAIN

        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('GZ_Master1')
            >>> data_key = pblm.default_data_key
            >>> clf_key = pblm.default_clf_key
            >>> task_key = pblm.primary_task_key
            >>> pblm.setup_evaluation()
            >>> featinfo = pblm.feature_info(task_key, clf_key, data_key)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> text = importances
            >>> pt.wordcloud(featinfo.importances)
            >>> ut.show_if_requested()
        """
        if data_key is None:
            data_key = pblm.default_data_key
        if clf_key is None:
            clf_key = pblm.default_clf_key
        if task_key is None:
            task_key = pblm.primary_task_key

        X = pblm.samples.X_dict[data_key]
        clf_list = pblm.eval_task_clfs[task_key][clf_key][data_key]
        feature_importances = np.mean([
            clf_.feature_importances_ for clf_ in clf_list
        ], axis=0)
        importances = ut.dzip(X.columns, feature_importances)
        return importances

    def report_importance(pblm, task_key, clf_key, data_key):
        # ut.qtensure()
        # import plottool as pt  # NOQA
        if clf_key != 'RF':
            print('Can only report importance for RF not %r' % (clf_key,))
            return

        importances = pblm.feature_info(task_key, clf_key, data_key)
        featinfo = AnnotPairFeatInfo(importances=importances)

        # Take average feature importance
        ut.cprint('MARGINAL IMPORTANCE INFO for %s on task %s' % (
            data_key, task_key), 'yellow')
        print(' Caption:')
        print(' * The NaN row ensures that `weight` always sums to 1')
        print(' * `num` indicates how many dimensions the row groups')
        print(' * `ave_w` is the average importance a single feature in the row')
        # with ut.Indenter('[%s] ' % (data_key,)):

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
            cfgdict = pblm.hyper_params.vsone_match.asdict()
            match.apply_all(cfgdict)
            pt.figure(fnum=fnum, pnum=pnum_())
            match.show(show_ell=False, show_ori=False)
            # pt.set_title(alias)

    def find_opt_ratio(pblm):
        """
        script to help find the correct value for the ratio threshold

            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm = OneVsOneProblem.from_empty('GZ_Master1')
        """
        # Find best ratio threshold
        pblm.load_samples()
        infr = pblm.infr
        edges = ut.lmap(tuple, pblm.samples.aid_pairs.tolist())
        task = pblm.samples['match_state']
        pos_idx = task.class_names.tolist().index(POSTV)

        config = {'ratio_thresh': 1.0, 'sv_on': False}
        matches = infr._exec_pairwise_match(edges, config)

        import plottool as pt
        pt.qtensure()
        thresholds = np.linspace(0, 1.0, 100)
        pos_truth = task.y_bin.T[pos_idx]
        ratio_fs = [m.local_measures['ratio'] for m in matches]

        aucs = []
        # Given the current correspondences: Find the optimal
        # correspondence threshold.
        for thresh in ut.ProgIter(thresholds, 'computing thresh'):
            scores = np.array([fs[fs < thresh].sum() for fs in ratio_fs])
            roc = sklearn.metrics.roc_auc_score(pos_truth, scores)
            aucs.append(roc)
        aucs = np.array(aucs)
        opt_auc = aucs.max()
        opt_thresh = thresholds[aucs.argmax()]

        if True:
            pt.plt.plot(thresholds, aucs, 'r-', label='')
            pt.plt.plot(opt_thresh, opt_auc, 'ro',
                        label='L opt=%r' % (opt_thresh,))
            pt.set_ylabel('auc')
            pt.set_xlabel('ratio threshold')
            pt.legend()

        # colors = {
        #     1: 'r',
        #     2: 'b',
        #     3: 'g',
        # }
        # def predict_truth(ratio_fs, opt_thresh, pos_truth):
        #     # Filter correspondence using thresh then sum their scores
        #     new_ratio_fs = [fs < opt_thresh for fs in ratio_fs]
        #     scores = np.array([fs.sum() for fs in new_ratio_fs])
        #     # Find the point (summed score threshold) that maximizes MCC
        #     fpr, tpr, points = sklearn.metrics.roc_curve(pos_truth, scores)
        #     mccs = np.array([sklearn.metrics.matthews_corrcoef(
        #         pos_truth, scores > point) for point in points])
        #     opt_point = points[mccs.argmax()]
        #     pos_pred = scores > opt_point
        #     return pos_pred
        # thresholds = np.linspace(0, 1.0, 100)
        # pos_truth = task.y_bin.T[pos_idx]
        # ratio_fs = [m.local_measures['ratio'] for m in matches]
        # thresh_levels = []
        # for level in range(1, 3 + 1):
        #     if ut.allsame(pos_truth):
        #         print('breaking')
        #         break
        #     print('level = %r' % (level,))
        #     aucs = []
        #     # Given the current correspondences: Find the optimal
        #     # correspondence threshold.
        #     for thresh in ut.ProgIter(thresholds, 'computing thresh'):
        #         scores = np.array([fs[fs < thresh].sum() for fs in ratio_fs])
        #         roc = sklearn.metrics.roc_auc_score(pos_truth, scores)
        #         aucs.append(roc)
        #     aucs = np.array(aucs)
        #     opt_auc = aucs.max()
        #     opt_thresh = thresholds[aucs.argmax()]
        #     thresh_levels.append(opt_thresh)

        #     if True:
        #         color = colors[level]
        #         pt.plt.plot(thresholds, aucs, color + '-', label='L%d' % level)
        #         pt.plt.plot(opt_thresh, opt_auc, color + 'o',
        #                     label='L%d opt=%r' % (level, opt_thresh,))

        #     # Remove the positive samples that this threshold fails on
        #     pred = predict_truth(ratio_fs, opt_thresh, pos_truth)
        #     flags = pred != pos_truth | ~pos_truth

        #     ratio_fs = ut.compress(ratio_fs, flags)
        #     pos_truth = pos_truth.compress(flags)

        # submax_thresh, submax_roc = vt.argsubmax(aucs, thresholds)

        # Now find all pairs that would be correctly classified using this
        # threshold

        # ratio_fs = thresh_ratio_fs
        # rocs = []
        # for thresh in ut.ProgIter(thresholds, 'computing thresh'):
        #     scores = np.array([fs[fs < thresh].sum() for fs in ratio_fs])
        #     roc = sklearn.metrics.roc_auc_score(pos_truth, scores)
        #     rocs.append(roc)
        # submax_thresh, submax_roc = vt.argsubmax(rocs, thresholds)
        # pt.plt.plot(thresholds, rocs, 'b-', label='L2')
        # pt.plt.plot(submax_thresh, submax_roc, 'bo', label='L2 opt=%r' % (submax_thresh,))

    def simple_confusion(pblm, score_key=None, task_key=None,
                         target_class=None):
        if score_key is None:
            score_key = 'score_lnbnn_1vM'
        if task_key is None:
            task_key = pblm.primary_task_key
        task = pblm.samples[task_key]
        if target_class is None:
            target_class = task.default_class_name

        target_class_idx = task.lookup_class_idx(target_class)
        scores = pblm.samples.simple_scores[score_key]
        y = task.y_bin.T[target_class_idx]
        conf = vt.ConfusionMetrics.from_scores_and_labels(scores, y)
        conf.label = score_key
        return conf

    def qt_review_hardcases(pblm):
        """
        Example:
            >>> from ibeis.scripts.script_vsone import *  # NOQA
            >>> #pblm = OneVsOneProblem.from_empty('GZ_Master1')
            >>> pblm = OneVsOneProblem.from_empty('GIRM_Master1')
            >>> #pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm.evaluate_classifiers()
            >>> win = pblm.qt_review_hardcases()

        Ignore:
            >>> # TEST to ensure we can priorizite reviewed edges without inference
            >>> import networkx as nx
            >>> from ibeis.algo.graph import demo
            >>> kwargs = dict(num_pccs=6, p_incon=.4, size_std=2)
            >>> infr = demo.demodata_infr(**kwargs)
            >>> infr.queue_params['pos_redun'] = 1
            >>> infr.queue_params['neg_redun'] = 1
            >>> infr.apply_nondynamic_update()
            >>> edges = list(infr.edges())
            >>> prob_match = ut.dzip(edges, infr.dummy_matcher.predict(edges))
            >>> infr.set_edge_attrs('prob_match', prob_match)
            >>> infr.enable_redundancy = True
            >>> infr.prioritize('prob_match', edges)
            >>> order = []
            >>> while True:
            >>>     order.append(infr.pop())
            >>> print(len(order))
        """
        task_key = pblm.primary_task_key
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key
        res = pblm.task_combo_res[task_key][clf_key][data_key]

        samples = pblm.samples
        infr = pblm.infr
        ibs = infr.ibs
        unsure_cases = res.hardness_analysis(samples, infr)
        # Remove very confidenct cases
        # CONFIDENCE = ibs.const.CONFIDENCE
        # flags = unsure_cases['real_conf'] < CONFIDENCE.CODE_TO_INT['pretty_sure']

        if True:
            flags = unsure_cases['real_conf'] < 2
            unsure_cases = unsure_cases[flags]

        # only review big ccs
        if False:
            n_other1 = np.array([len(infr.pos_graph.connected_to(a))
                                 for a in unsure_cases['aid1']])
            n_other2 = np.array([len(infr.pos_graph.connected_to(a))
                                 for a in unsure_cases['aid2']])
            unsure_cases = unsure_cases[(n_other2 > 10) & (n_other1 > 10)]

        infr.enable_redundancy = False
        infr.fix_mode_split = False
        infr.fix_mode_merge = False
        infr.fix_mode_predict = True
        infr.classifiers = None

        # TODO: force it to re-review non-confident edges with the hardness
        # as priority ignoring the connectivity criteria
        edges = unsure_cases.index.tolist()
        infr.ensure_edges(edges)

        # Assign probs to edges for propper weighting
        pred_edges = [e for e in infr.edges() if e in res.probs_df.index]
        prob_matches = res.probs_df[POSTV].loc[pred_edges].to_dict()
        infr.set_edge_attrs('prob_match', prob_matches)

        # Assign hardness to hard cases
        # infr.set_edge_attrs('hardness', unsure_cases['hardness'].to_dict())

        # Only review failure cases
        unsure_cases = unsure_cases[unsure_cases['failed']]
        unsure_cases = unsure_cases.sort_values('hardness', ascending=False)

        infr.set_edge_attrs('hardness', unsure_cases['hardness'].to_dict())
        infr.set_edge_attrs('probs', res.probs_df.loc[edges].to_dict('index'))
        for key in ['pred', 'real']:
            vals = unsure_cases[key].map(ibs.const.REVIEW.INT_TO_CODE)
            infr.set_edge_attrs(key, vals.to_dict())
        infr.prioritize('hardness', unsure_cases['hardness'].to_dict(), reset=True)
        infr.apply_nondynamic_update()

        cfgdict = pblm.hyper_params['vsone_match'].asdict()
        cfgdict.update(pblm.hyper_params['vsone_kpts'].asdict())

        win = infr.qt_review_loop(cfgdict=cfgdict)
        # gt.qtapp_loop(qwin=infr.manual_wgt, freq=10)
        return win


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
        assert aid_pairs is not None
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

    def compress(samples, flags):
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
        assert edges is not None
        tags = [None if d is None else d.get('tags')
                for d in map(infr.get_edge_data, edges)]
        flags = [None if t is None else 'photobomb' in t
                 for t in tags]
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
        samples['match_state'].default_class_name = POSTV
        samples['photobomb_state'].default_class_name = 'pb'

    def apply_multi_task_binary_label(samples):
        assert False
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
        assert False
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

    def print_featinfo(samples):
        for data_key in samples.X_dict.keys():
            print('\nINFO(samples.X_dict[%s])' % (data_key,))
            featinfo = AnnotPairFeatInfo(samples.X_dict[data_key])
            print(ut.indent(featinfo.get_infostr()))


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
