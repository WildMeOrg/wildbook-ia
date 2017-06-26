# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import ubelt as ub
import itertools as it
import numpy as np
import vtool as vt
import dtool as dt
import six  # NOQA
import hashlib
import copy  # NOQA
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass
import sklearn.ensemble
from ibeis.algo.verif import clf_helpers
from ibeis.algo.verif import sklearn_utils
from ibeis.algo.verif import pairfeat
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP
from os.path import join, exists, basename
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


@ut.reloadable_class
class OneVsOneProblem(clf_helpers.ClfProblem):
    """
    Keeps information about the one-vs-one pairwise classification problem

    CommandLine:
        python -m ibeis.algo.verif.vsone evaluate_classifiers
        python -m ibeis.algo.verif.vsone evaluate_classifiers --db PZ_PB_RF_TRAIN
        python -m ibeis.algo.verif.vsone evaluate_classifiers --db PZ_PB_RF_TRAIN --profile
        python -m ibeis.algo.verif.vsone evaluate_classifiers --db PZ_MTEST --show
        python -m ibeis.algo.verif.vsone evaluate_classifiers --db PZ_Master1 --show
        python -m ibeis.algo.verif.vsone evaluate_classifiers --db GZ_Master1 --show

    Example:
        >>> from ibeis.algo.verif.vsone import *  # NOQA
        >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST')
        >>> pblm.load_samples()
        >>> pblm.load_features()
    """
    appname = 'vsone_rf_train'

    def __init__(pblm, infr=None, verbose=None, **params):
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
            ('vsone_kpts', pairfeat.VsOneFeatConfig()),
            ('vsone_match', pairfeat.VsOneMatchConfig()),
            ('pairwise_feats', pairfeat.PairFeatureConfig()),
            ('need_lnbnn', False),
            # ('sample_method', 'lnbnn'),
            ('sample_method', 'random'),
            ('sample_search', dict(
                K=4, Knorm=1, requery=True, score_method='csum',
                prescore_method='csum')),
        ]),
            tablename='HyperParams'
        )

        bins = hyper_params['pairwise_feats']['bins']
        hyper_params['vsone_match']['ratio_thresh'] = max(bins)
        hyper_params['vsone_match']['thresh_bins'] = bins
        hyper_params['vsone_match']['sv_on'] = True

        # if False:
        #     # For giraffes
        #     hyper_params['vsone_match']['checks'] = 80
        #     hyper_params['vsone_match']['sver_xy_thresh'] = .02
        #     hyper_params['vsone_match']['sver_ori_thresh'] = 3

        species = infr.ibs.get_primary_database_species()
        if species == 'zebra_plains' or True:
            hyper_params['vsone_match']['Knorm'] = 3
            hyper_params['vsone_match']['symmetric'] = True
            hyper_params['vsone_kpts']['augment_orientation'] = True

        if infr.ibs.has_species_detector(species):
            hyper_params.vsone_match['weight'] = 'fgweights'
            hyper_params.pairwise_feats['sorters'] = ut.unique(
                hyper_params.pairwise_feats['sorters'] +
                [
                    # 'weighted_ratio_score',
                    # 'weighted_lnbnn'
                ]
            )
        else:
            hyper_params.vsone_match['weight'] = None

        # global_keys = ['yaw', 'qual', 'gps', 'time']
        global_keys = ['view', 'qual', 'gps', 'time']
        match_config = {}
        match_config.update(hyper_params['vsone_kpts'])
        match_config.update(hyper_params['vsone_match'])
        pairfeat_cfg = hyper_params['pairwise_feats'].asdict()
        need_lnbnn = hyper_params['need_lnbnn']
        # need_lnbnn = pairfeat_cfg.pop('need_lnbnn', False)
        pblm.feat_extract_config = {
            'global_keys': global_keys,
            'match_config': match_config,
            'pairfeat_cfg': pairfeat_cfg,
            'need_lnbnn': need_lnbnn,
        }

        pblm.hyper_params = hyper_params
        updated = pblm.hyper_params.update2(params)
        if updated:
            print('Externally updated params = %r' % (updated,))

    @classmethod
    def from_aids(OneVsOneProblem, ibs, aids, verbose=None, **params):
        # TODO: If the graph structure is defined, this should load the most
        # recent state, so the infr object has all the right edges.  If the
        # graph structure is not defined, it should apply the conversion
        # method.
        import ibeis
        infr = ibeis.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        # if infr.ibs.dbname not in {'GIRM_Master1', 'NNP_MasterGIRM_core'}:
        #     assert infr._is_staging_above_annotmatch()
        infr.reset_feedback('staging', apply=True)
        if infr.ibs.dbname == 'PZ_MTEST':
            # assert False, 'need to do conversion'
            infr.ensure_mst()
        # if infr.needs_conversion():
        #     infr.ensure_mst()
        pblm = OneVsOneProblem(infr=infr, **params)
        return pblm

    @classmethod
    def from_empty(OneVsOneProblem, defaultdb=None, **params):
        """
        >>> from ibeis.algo.verif.vsone import *  # NOQA
        >>> defaultdb = 'GIRM_Master1'
        >>> pblm = OneVsOneProblem.from_empty(defaultdb)
        """
        if defaultdb is None:
            defaultdb = 'PZ_PB_RF_TRAIN'
            # defaultdb = 'GZ_Master1'
            # defaultdb = 'PZ_MTEST'
        import ibeis
        ibs, aids = ibeis.testdata_aids(defaultdb)
        pblm = OneVsOneProblem.from_aids(ibs, aids, **params)
        return pblm

    def _make_lnbnn_qreq(pblm, aids=None):
        # This is the qreq used to do LNBNN sampling and to compute simple
        # LNBNN scores.
        infr = pblm.infr
        ibs = pblm.infr.ibs

        if aids is None:
            aids = ibs.filter_annots_general(
                infr.aids, min_pername=3, species='primary')

        cfgdict = pblm.hyper_params['sample_search'].copy()
        # Use the same keypoints for vsone and vsmany for comparability
        cfgdict.update(pblm.hyper_params['vsone_kpts'])
        if cfgdict['augment_orientation']:
            # Do query-side only if augment ori is on for 1vs1
            cfgdict['augment_orientation'] = False
            cfgdict['query_rotation_heuristic'] = True

        infr.relabel_using_reviews(rectify=False)
        custom_nid_lookup = infr.get_node_attrs('name_label', aids)
        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      verbose=False,
                                      custom_nid_lookup=custom_nid_lookup)

        assert qreq_.qparams.can_match_samename is True
        assert qreq_.qparams.prescore_method == 'csum'
        assert pblm.hyper_params.subsample is None
        return qreq_

    def make_lnbnn_training_pairs(pblm):
        infr = pblm.infr
        ibs = pblm.infr.ibs
        if pblm.verbose > 0:
            print('[pblm] gather lnbnn match-state cases')

        aids = ibs.filter_annots_general(
            infr.aids, min_pername=3, species='primary')
        qreq_ = pblm._make_lnbnn_qreq(aids)

        use_cache = False
        use_cache = True
        cfgstr = qreq_.get_cfgstr(with_input=True)
        cacher1 = ub.Cacher('pairsample_1_v6' + ibs.get_dbname(),
                            cfgstr=cfgstr, appname=pblm.appname,
                            enabled=use_cache, verbose=pblm.verbose)

        # make sure changes is names doesn't change the pair sample so I can
        # iterate a bit faster. Ensure this is turned off later.
        import datetime
        deadline = datetime.date(year=2017, month=8, day=1)
        nowdate = datetime.datetime.now().date()
        HACK_LOAD_STATIC_DATASET = nowdate < deadline
        HACK_LOAD_STATIC_DATASET = False

        data = cacher1.tryload()

        if HACK_LOAD_STATIC_DATASET:
            # Just take anything, I dont care if it changed
            if data is None:
                print('HACKING STATIC DATASET')
                infos = []
                for fpath in cacher1.existing_versions():
                    finfo = ut.get_file_info(fpath)
                    finfo['fname'] = basename(fpath)
                    finfo['fpath'] = fpath
                    infos.append(finfo)
                df = pd.DataFrame(infos)
                if len(df):
                    df = df.drop(['owner', 'created', 'last_accessed'], axis=1)
                    df = df.sort_values('last_modified').reindex()
                    fpath = df['fpath'][0]
                    print(df)
                    print('HACKING STATIC DATASET')
                    data = ut.load_data(fpath)

        if data is None:
            print('Using LNBNN to compute pairs')
            cm_list = qreq_.execute()
            infr._set_vsmany_info(qreq_, cm_list)  # hack

            # Sample hard moderate and easy positives / negative
            # For each query, choose same, different, and random training pairs
            rng = np.random.RandomState(42)
            aid_pairs_ = infr._cm_training_pairs(
                rng=rng, **pblm.hyper_params.pair_sample)
            cacher1.save(aid_pairs_)
            data = aid_pairs_
            print('Finished using LNBNN to compute pairs')
        else:
            print('Loaded previous LNBNN pairs')
        aid_pairs_ = data
        return aid_pairs_

    def make_randomized_training_pairs(pblm):
        """
        Randomized sample that does not require LNBNN
        """

        if pblm.verbose > 0:
            print('[pblm] Using randomized training pairs')
        # from ibeis.algo.graph import nx_utils as nxu
        infr = pblm.infr
        infr.status()

        pair_sample = pblm.hyper_params.pair_sample

        n_pos = sum(ut.take(
            pair_sample, ['top_gt', 'mid_gt', 'bot_gt', 'rand_gt']))
        n_neg = sum(ut.take(
            pair_sample, ['top_gf', 'mid_gf', 'bot_gf', 'rand_gf']))

        # LNBNN makes 48729 given a set of 6474, so about 8 examples per annot

        multipler = (n_pos + n_neg) // 2
        n_target = len(infr.aids) * multipler

        def edgeset(iterable):
            return set(it.starmap(infr.e_, iterable))

        pos_edges = edgeset(infr.pos_graph.edges())
        neg_edges = edgeset(infr.neg_graph.edges())
        aid_pairs = pos_edges.union(neg_edges)

        n_need = n_target - len(aid_pairs)

        per_cc = int(n_need / infr.pos_graph.number_of_components() / 2)
        per_cc = max(2, per_cc)

        rng = ut.ensure_rng(2039141610)

        # User previous explicit reviews
        pccs = list(map(frozenset, infr.positive_components()))
        for cc in pccs:
            pos_pairs = edgeset(ut.random_combinations(cc, 2, per_cc, rng=rng))
            aid_pairs.update(pos_pairs)

        n_need = n_target - len(aid_pairs)

        rng = ut.ensure_rng(282695095)
        per_pair = 1
        for cc1, cc2 in ut.random_combinations(pccs, 2, rng=rng):
            neg_pairs = edgeset(ut.random_product((cc1, cc2), num=per_pair,
                                                  rng=rng))
            aid_pairs.update(neg_pairs)
            if len(aid_pairs) >= n_target:
                break

        n_need = n_target - len(aid_pairs)
        return aid_pairs

    @profile
    def make_training_pairs(pblm):
        """
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST')
            >>> pblm.make_training_pairs()
        """
        infr = pblm.infr
        if pblm.verbose > 0:
            print('[pblm] gathering training pairs')

        sample_method = pblm.hyper_params['sample_method']

        if sample_method == 'lnbnn':
            aid_pairs_ = pblm.make_lnbnn_training_pairs()
        elif sample_method == 'random':
            aid_pairs_ = pblm.make_randomized_training_pairs()
        else:
            raise KeyError('Unknown sample_method={}'.format(sample_method))

        pb_aid_pairs = infr.photobomb_samples()
        incomp_aid_pairs = list(infr.incomp_graph.edges())

        # Simplify life by using undirected pairs
        aid_pairs = pb_aid_pairs + list(aid_pairs_) + incomp_aid_pairs
        # Ensure this is sorted
        aid_pairs = sorted(set(it.starmap(infr.e_, aid_pairs)))
        return aid_pairs

    @profile
    def load_samples(pblm):
        r"""
        CommandLine:
            python -m ibeis.algo.verif.vsone load_samples --profile

        Example:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> #pblm = OneVsOneProblem.from_empty('PZ_MTEST')
            >>> #pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm = OneVsOneProblem.from_empty('PZ_Master1')
            >>> pblm.load_samples()
            >>> samples = pblm.samples
            >>> samples.print_info()
        """
        # Get a set of training pairs
        if pblm.verbose > 0:
            ut.cprint('[pblm] load_samples', color='blue')
        aid_pairs = pblm.make_training_pairs()
        pblm.samples = AnnotPairSamples(pblm.infr.ibs, aid_pairs, pblm.infr)
        if pblm.verbose > 0:
            ut.cprint('[pblm] apply_multi_task_multi_label', color='blue')
        pblm.samples.apply_multi_task_multi_label()

    @profile
    def load_features(pblm, use_cache=True, with_simple=False):
        """
        CommandLine:
            python -m ibeis.algo.verif.vsone load_features --profile

        Example:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> #pblm = OneVsOneProblem.from_empty('GZ_Master1')
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm.load_samples()
            >>> pblm.load_features(with_simple=False)
        """
        if pblm.verbose > 0:
            ut.cprint('[pblm] load_features', color='blue')

        ibs = pblm.infr.ibs
        edges = ut.emap(tuple, pblm.samples.aid_pairs.tolist())
        feat_extract_config = pblm.feat_extract_config
        extr = pairfeat.PairwiseFeatureExtractor(ibs, **feat_extract_config)
        X_all = extr.transform(edges)

        pblm.raw_X_dict = {'learn(all)': X_all}
        pblm.samples.set_feats(copy.deepcopy(pblm.raw_X_dict))

        if with_simple:
            pblm.load_simple_scores()

    def load_simple_scores(pblm):
        if pblm.verbose > 0:
            ut.cprint('[pblm] load_simple_scores', color='blue')

        infr = pblm.infr
        ibs = infr.ibs

        aid_pairs = ut.emap(tuple, pblm.samples.aid_pairs.tolist())

        hyper_params = pblm.hyper_params
        sample_hashid = pblm.samples.sample_hashid()
        feat_cfgstr = hyper_params.get_cfgstr()
        feat_hashid = ut.hashstr27(sample_hashid + feat_cfgstr)
        # print('features_hashid = %r' % (features_hashid,))
        cfgstr = '_'.join(['devcache', str(ibs.dbname), feat_hashid])

        cacher = ub.Cacher('simple_scores_' + ibs.dbname,
                           cfgstr=cfgstr,
                           appname=pblm.appname, enabled=0,
                           verbose=pblm.verbose)
        data = cacher.tryload()
        if data is None:
            # ---------------
            X_all = pblm.raw_X_dict['learn(all)']
            featinfo = vt.AnnotPairFeatInfo(X_all)
            simple_cols = featinfo.find('summary_op', '==', 'sum')
            simple_cols += featinfo.find('summary_op', '==', 'len', hack=False)

            # Select simple scores out of the full feat vectors
            simple_scores = X_all[simple_cols]

            if True:
                # The main idea here is to load lnbnn scores for the pairwise
                # matches so we can compare them to the outputs of the pairwise
                # classifier.
                # TODO: separate this into different cache
                # Add vsmany_lnbnn to simple scoren

                # aids = ibs.filter_annots_general(
                #     infr.aids, min_pername=3, species='primary')

                # Only query the aids in the sampled set
                aids = sorted(set(ut.flatten(aid_pairs)))
                qreq_ = pblm._make_lnbnn_qreq(aids)

                cm_list = qreq_.execute()
                edge_to_data = infr._get_cm_edge_data(aid_pairs, cm_list=cm_list)
                edge_data = ut.take(edge_to_data, aid_pairs)
                lnbnn_score_list = [d.get('score', 0) for d in edge_data]
                lnbnn_score_list = [0 if s is None else s
                                    for s in lnbnn_score_list]

                simple_scores = simple_scores.assign(
                    score_lnbnn_1vM=lnbnn_score_list)

            simple_scores[pd.isnull(simple_scores)] = 0
            data = simple_scores
            cacher.save(data)
        simple_scores = data

        pblm.raw_simple_scores = simple_scores
        pblm.samples.set_simple_scores(copy.deepcopy(pblm.raw_simple_scores))

    def ensure_deploy_classifiers(pblm, dpath='.'):
        classifiers = {}
        task_keys = list(pblm.samples.supported_tasks())
        for task_key in task_keys:
            deploy_info = Deployer(dpath).ensure(pblm, task_key)
            classifiers[task_key] = deploy_info
        return classifiers

    def deploy_all(pblm, dpath='.', publish=False):
        task_keys = list(pblm.samples.supported_tasks())
        for task_key in task_keys:
            pblm.deploy(dpath, task_key=task_key, publish=publish)

    def deploy(pblm, dpath='.', task_key=None, publish=False):
        """
        Trains and saves a classifier for deployment

        Notes:
            A deployment consists of the following information
                * The classifier itself
                * Information needed to construct the input to the classifier
                    - TODO: can this be encoded as an sklearn pipeline?
                * Metadata concerning what data the classifier was trained with
                * PUBLISH TO /media/hdd/PUBLIC/models/pairclf

        Ignore:
            pblm.evaluate_classifiers(with_simple=False)
            res = pblm.task_combo_res[pblm.primary_task_key]['RF']['learn(sum,glob)']
        """
        return Deployer(dpath=dpath, pblm=pblm).deploy(task_key, publish)

    def setup(pblm, with_simple=False):
        pblm.set_pandas_options()

        ut.cprint('\n[pblm] --- LOADING DATA ---', 'blue')
        pblm.load_samples()
        # pblm.samples.print_info()
        pblm.load_features(with_simple=with_simple)

        # pblm.samples.print_info()
        ut.cprint('\n[pblm] --- CURATING DATA ---', 'blue')
        pblm.samples.print_info()
        print('---------------')

        ut.cprint('\n[pblm] --- FEATURE INFO ---', 'blue')
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
        unsupported = set(task_keys) - set(pblm.samples.supported_tasks())
        for task_key in unsupported:
            print('No data to train task_key = %r' % (task_key,))
            task_keys.remove(task_key)

    def setup_evaluation(pblm, with_simple=False):
        pblm.setup(with_simple=with_simple)

        task_keys = pblm.eval_task_keys
        clf_keys = pblm.eval_clf_keys
        data_keys = pblm.eval_data_keys

        if pblm.samples.simple_scores is not None:
            ut.cprint('\n--- EVALUTE SIMPLE SCORES ---', 'blue')
            pblm.evaluate_simple_scores(task_keys)
        else:
            print('no simple scores')
            print('...skipping simple evaluation')

        ut.cprint('\n--- LEARN CROSS-VALIDATED RANDOM FORESTS ---', 'blue')
        pblm.learn_evaluation_classifiers(task_keys, clf_keys, data_keys)

    def report_evaluation(pblm):
        """
        CommandLine:
            python -m ibeis.algo.verif.vsone report_evaluation --db PZ_MTEST

        Example:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='PZ_MTEST',
            >>>                                   sample_method='random')
            >>> pblm.eval_clf_keys = ['MLP', 'Logit', 'RF']
            >>> pblm.eval_data_keys = ['learn(sum,glob)']
            >>> pblm.setup_evaluation(with_simple=False)
            >>> pblm.report_evaluation()
        """
        ut.cprint('\n--- EVALUATE LEARNED CLASSIFIERS ---', 'blue')
        # For each task / classifier type
        for task_key in pblm.eval_task_keys:
            pblm.task_evaluation_report(task_key)

    def evaluate_classifiers(pblm, with_simple=False):
        """
        CommandLine:
            python -m ibeis.algo.verif.vsone evaluate_classifiers
            python -m ibeis.algo.verif.vsone evaluate_classifiers --db PZ_MTEST
            python -m ibeis.algo.verif.vsone evaluate_classifiers --db GZ_Master1
            python -m ibeis.algo.verif.vsone evaluate_classifiers --db GIRM_Master1

        Example:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='PZ_MTEST',
            >>>                                   sample_method='random')
            >>> pblm.default_clf_key = 'Logit'
            >>> pblm.evaluate_classifiers()
        """
        pblm.setup_evaluation(with_simple=with_simple)
        pblm.report_evaluation()

    def task_evaluation_report(pblm, task_key):
        """
        clf_keys = [pblm.default_clf_key]
        """
        # selected_data_keys = ut.ddict(list)
        from utool.experimental.pandas_highlight import to_string_monkey
        clf_keys = pblm.eval_clf_keys
        data_keys = pblm.eval_data_keys
        print('data_keys = %r' % (data_keys,))
        ut.cprint('--- TASK = %s' % (ut.repr2(task_key),), 'turquoise')
        labels = pblm.samples.subtasks[task_key]
        if getattr(pblm, 'simple_aucs', None) is not None:
            pblm.report_simple_scores(task_key)
        for clf_key in clf_keys:
            # Combine results over datasets
            print('clf_key = %s' % (ut.repr2(clf_key),))
            data_combo_res = pblm.task_combo_res[task_key][clf_key]
            df_auc_ovr = pd.DataFrame(dict([
                (datakey, list(data_combo_res[datakey].roc_scores_ovr()))
                for datakey in data_keys
            ]), index=labels.one_vs_rest_task_names())
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
                ]), index=labels.one_vs_rest_task_names())
                print(to_string_monkey(df_auc_ovr_hat, highlight_cols='all'))

            roc_scores = dict(
                [(datakey, [data_combo_res[datakey].roc_score()])
                 for datakey in data_keys])
            df_auc = pd.DataFrame(roc_scores)
            ut.cprint('[%s] ROC-AUC(MacroAve) Scores' % (clf_key,), 'yellow')
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
        #     print(ut.indent(vt.AnnotPairFeatInfo(
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
            # cc = x[ut.argmax(ut.emap(len, x))]
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

    @profile
    def predict_proba_deploy(pblm, X, task_keys):
        # import pandas as pd
        task_probs = {}
        for task_key in task_keys:
            print('[pblm] predicting %s probabilities' % (task_key,))
            clf = pblm.deploy_task_clfs[task_key]
            labels = pblm.samples.subtasks[task_key]
            class_names = labels.class_names
            probs_df = sklearn_utils.predict_proba_df(clf, X, class_names)
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
            ut.hash_data(np.array(want_edges)),
            data_key,
            clf_key,
            repr(task_keys),
            pblm.hyper_params.vsone_match.get_cfgstr(),
            pblm.hyper_params.vsone_kpts.get_cfgstr(),
        ])
        cacher2 = ub.Cacher('full_eval_probs2' + pblm.infr.ibs.dbname,
                            prob_cfgstr, appname=pblm.appname, verbose=20)
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
            assert (sorted(ut.emap(tuple, train_uv.tolist())) ==
                    sorted(ut.emap(tuple, pblm.samples.aid_pairs.tolist())))
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
            have_edges = ut.emap(tuple, have_uv.tolist())
            need_edges = ut.emap(tuple, need_uv.tolist())
            want_edges = ut.emap(tuple, want_uv.tolist())
            assert set(have_edges) & set(need_edges) == set([])
            assert set(have_edges) | set(need_edges) == set(want_edges)

            infr.classifiers = pblm
            data_info = pblm.feat_extract_info[data_key]
            X_need = infr._cached_pairwise_features(need_edges, data_info)
            # X_need = infr._pblm_pairwise_features(need_edges, data_key)
            # Make an ensemble of the evaluation classifiers
            # (todo: use a classifier that hasn't seen any of this data)
            task_need_probs = {}
            for task_key in task_keys:
                print('Predicting %s probabilities' % (task_key,))
                clf_list = pblm.eval_task_clfs[task_key][clf_key][data_key]
                labels = pblm.samples.subtasks[task_key]
                import utool
                with utool.embed_on_exception_context:
                    eclf = sklearn_utils.voting_ensemble(clf_list, voting='soft')
                eclf_probs = sklearn_utils.predict_proba_df(eclf, X_need,
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
            python -m ibeis.algo.verif.vsone build_feature_subsets --db GZ_Master1
            python -m ibeis.algo.verif.vsone build_feature_subsets --db PZ_PB_RF_TRAIN

            python -m ibeis Chap4._setup_pblm --db GZ_Master1 --eval
            python -m ibeis Chap4._setup_pblm --db PZ_Master1 --eval

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST')
            >>> pblm.load_samples()
            >>> pblm.load_features()
            >>> pblm.build_feature_subsets()
            >>> pblm.samples.print_featinfo()
        """
        if pblm.verbose:
            ut.cprint('[pblm] build_feature_subsets', color='blue')
        # orig_dict = pblm.samples.X_dict
        # X = orig_dict
        X = pblm.raw_X_dict['learn(all)']
        featinfo = vt.AnnotPairFeatInfo(X)

        X_dict = ut.odict()
        pblm.feat_extract_info = {}
        def register_data_key(data_key, cols):
            feat_dims = sorted(cols)
            data_info = (pblm.feat_extract_config, feat_dims)
            pblm.feat_extract_info[data_key] = data_info
            X_dict[data_key] = X[feat_dims]

        register_data_key('learn(all)', list(X.columns))

        if False:
            # Use only summary stats
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
            ])
            register_data_key('learn(sum)', cols)

        if True:
            # Use summary and global single thresholds with raw unaries
            cols = featinfo.select_columns([
                ('measure_type', '==', 'summary'),
                ('summary_measure', '!=', 'ratio')
            ])
            cols.update(featinfo.select_columns([
                ('measure_type', '==', 'global'),
                ('measure', 'not in', {
                    'qual_1', 'qual_2', 'yaw_1', 'yaw_2',
                    'view_1', 'view_2', 'gps_1[0]', 'gps_2[0]', 'gps_1[1]',
                    'gps_2[1]', 'time_1', 'time_2'
                })
            ]))
            register_data_key('learn(sum,glob)', cols)

            # if True:
            #     # Use summary and global single thresholds with raw unaries
            #     sumcols1 = featinfo.select_columns([
            #         ('summary_op', 'in', {'med'})
            #     ])
            #     m1_cols = set.difference(cols, set(sumcols1))
            #     register_data_key('learn(sum-1,glob)', m1_cols)

            #     sumcols2 = featinfo.select_columns([
            #         ('summary_op', 'in', {'std'})
            #     ])
            #     m2_cols = set.difference(cols, set(sumcols2))
            #     register_data_key('learn(sum-2,glob)', m2_cols)

            if True:
                # Use summary and global single thresholds with raw unaries
                sumcols1 = featinfo.select_columns([
                    ('summary_op', 'in', {'med'})
                ])
                m1_cols = set.difference(cols, set(sumcols1))
                register_data_key('learn(sum-1,glob)', m1_cols)

                sumcols2 = featinfo.select_columns([
                    ('summary_op', 'in', {'std'})
                ])
                m2_cols = set.difference(cols, set(sumcols2))
                register_data_key('learn(sum-2,glob)', m2_cols)

            if True:
                # Use summary and global single thresholds with raw unaries
                multibin_cols = featinfo.select_columns([
                    ('summary_binval', '>', 0.625)
                ])
                onebin_cols = set.difference(cols, set(multibin_cols))
                register_data_key('learn(sum,glob,onebin)', onebin_cols)

            if True:
                # Remove view columns
                view_cols = featinfo.select_columns([
                    ('measure_type', '==', 'global'),
                    ('measure', 'in', [
                        'yaw_1', 'yaw_2', 'delta_yaw', 'min_yaw', 'max_yaw'
                        'view_1', 'view_2', 'delta_view', 'min_view', 'max_view'
                    ]),
                ])
                noview_cols = set.difference(cols, view_cols)
                register_data_key('learn(sum,glob,-view)', noview_cols)

        pblm.samples.set_feats(X_dict)

    def evaluate_simple_scores(pblm, task_keys=None):
        """
            >>> from ibeis.algo.verif.vsone import *  # NOQA
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
        from utool.experimental.pandas_highlight import to_string_monkey
        if task_key is None:
            task_key = pblm.primary_task_key
        force_keep = ['score_lnbnn_1vM']
        simple_aucs = pblm.simple_aucs
        n_keep = 5
        df_simple_auc = pd.DataFrame.from_dict(simple_aucs[task_key],
                                               orient='index')
        # Take only a subset of the columns that scored well in something
        rankings = df_simple_auc.values.argsort(axis=1).argsort(axis=1)
        rankings = rankings.shape[1] - rankings - 1
        ordered_ranks = np.array(vt.ziptake(rankings.T,
                                            rankings.argsort(axis=0).T)).T
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
            python -m ibeis.algo.verif.vsone report_importance --show
            python -m ibeis.algo.verif.vsone report_importance --show --db PZ_PB_RF_TRAIN

        Example:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
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
        featinfo = vt.AnnotPairFeatInfo(importances=importances)

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
        featinfo = vt.AnnotPairFeatInfo(X, importances)
        featinfo.print_margins('feature')
        featinfo.print_margins('measure_type')
        featinfo.print_margins('summary_op')
        featinfo.print_margins('summary_measure')
        featinfo.print_margins('global_measure')

    def prune_features(pblm):
        """
        References:
            http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
            http://alexperrier.github.io/jekyll/update/2015/08/27/feature-importance-random-forests-gini-accuracy.html
            https://arxiv.org/abs/1407.7502
            https://github.com/glouppe/phd-thesis

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='PZ_MTEST')
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='PZ_PB_RF_TRAIN')
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='PZ_Master1')

        Ignore:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='GZ_Master1')
            >>> pblm.setup_evaluation()
        """
        # from sklearn.feature_selection import SelectFromModel
        from ibeis.algo.verif import clf_helpers

        task_key = pblm.primary_task_key
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key

        labels = pblm.samples.subtasks[task_key]
        # X = pblm.samples.X_dict[data_key]

        feat_dims = pblm.samples.X_dict[data_key].columns.tolist()
        n_dims = []
        reports = []
        sub_reports = []
        feats = []

        n_worst = 3
        min_feats = 10

        iter_ = range(min_feats, len(feat_dims), n_worst)
        prog = ut.ProgIter(iter_, label='prune')
        for _ in prog:
            prog.ensure_newline()
            clf_list, res_list = pblm._train_evaluation_clf(task_key, data_key,
                                                            clf_key, feat_dims)
            combo_res = clf_helpers.ClfResult.combine_results(res_list, labels)
            rs = [res.extended_clf_report(verbose=0) for res in res_list]
            report = combo_res.extended_clf_report(verbose=0)

            n_dims.append(len(feat_dims))
            feats.append(feat_dims[:])
            reports.append(report)
            sub_reports.append(rs)

            clf_importances = np.array(
                [clf_.feature_importances_ for clf_ in clf_list])
            feature_importances = np.mean(clf_importances, axis=0)
            importances = ut.dzip(feat_dims, feature_importances)

            # remove the worst features
            worst_features = ut.argsort(importances)[0:n_worst]
            for f in worst_features:
                feat_dims.remove(f)

        # mccs = [r['mcc'] for r in reports]

        mccs2 = np.array([[r['mcc'] for r in rs] for rs in sub_reports])
        u = mccs2.mean(axis=0)
        s = mccs2.std(axis=0)

        import plottool as pt
        pt.qtensure()
        pt.plot(n_dims, u, label='mean')
        ax = pt.gca()
        ax.fill_between(n_dims, u - s, u + s)
        pt.plot(n_dims, mccs2.T[0], label='mcc1')
        pt.plot(n_dims, mccs2.T[1], label='mcc2')
        pt.plot(n_dims, mccs2.T[2], label='mcc3')
        ax.legend()

    def demo_classes(pblm):
        r"""
        CommandLine:
            python -m ibeis.algo.verif.vsone demo_classes --saveparts --save=classes.png --clipwhite

            python -m ibeis.algo.verif.vsone demo_classes --saveparts --save=figures/classes.png --clipwhite --dpath=~/latex/crall-iccv-2017

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.verif.vsone import *  # NOQA
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
                # flags &= feats['global(delta_yaw)'] > 3
                flags &= feats['global(delta_view)'] > 2
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

            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm = OneVsOneProblem.from_empty('GZ_Master1')
        """
        # Find best ratio threshold
        pblm.load_samples()
        infr = pblm.infr
        edges = ut.emap(tuple, pblm.samples.aid_pairs.tolist())
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

    # def simple_confusion(pblm, score_key=None, task_key=None,
    #                      target_class=None):
    #     if score_key is None:
    #         score_key = 'score_lnbnn_1vM'
    #     if task_key is None:
    #         task_key = pblm.primary_task_key
    #     task = pblm.samples[task_key]
    #     if target_class is None:
    #         target_class = task.default_class_name

    #     target_class_idx = task.lookup_class_idx(target_class)
    #     scores = pblm.samples.simple_scores[score_key]
    #     y = task.y_bin.T[target_class_idx]
    #     conf = vt.ConfusionMetrics.from_scores_and_labels(scores, y)
    #     conf.label = score_key
    #     return conf

    def qt_review_hardcases(pblm):
        """
        Example:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
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
class Deployer(object):

    fname_parts = ['vsone', '{species}', '{task_key}',
                   '{clf_key}', '{n_dims}', '{hashid}']

    fname_fmtstr = '.'.join(fname_parts)

    meta_suffix = '.meta.json'

    publish_info = {
        'remote': 'lev.cs.rpi.edu',
        'path': '/media/hdd/PUBLIC/models/pairclf',
    }

    # published = {
    #     'zebra_plains': {
    #         'match_state': 'vsone.zebra_plains.match_state.RF.131.eurizlstehqjvlsu.cPkl',
    #         # 'photobomb_state': None,
    #     },
    #     'zebra_grevys': {},
    # }
    published = {
        'zebra_plains': {
            'match_state': 'vsone.zebra_plains.match_state.RF.131.eurizlstehqjvlsu.cPkl',
        },
    }

    def __init__(self, dpath='.', pblm=None):
        self.dpath = dpath
        self.pblm = pblm

    def _load_published(self, ibs, species, task_key):
        """
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> self = Deployer()
        """

        base_url = 'https://{remote}/public/models/pairclf'.format(
            **self.publish_info)

        task_fnames = self.published[species]
        fname = task_fnames[task_key]

        grabkw = dict(appname='ibeis', check_hash=False, verbose=0)

        # meta_url = base_url + '/' + fname + self.meta_suffix
        # meta_fpath = ut.grab_file_url(meta_url, **grabkw)

        deploy_url = base_url + '/' + fname
        deploy_fpath = ut.grab_file_url(deploy_url, **grabkw)

        verif = self._load_verifier(ibs, deploy_fpath, task_key)
        return verif

    def _load_verifier(self, ibs, deploy_fpath, task_key):
        from ibeis.algo.verif import verifier
        deploy_info = ut.load_data(deploy_fpath)
        verif = verifier.Verifier(ibs, deploy_info=deploy_info)
        if task_key is not None:
            assert verif.metadata['task_key'] == task_key, (
                'bad saved clf at fpath={}'.format(deploy_fpath))
        return verif

    def load_published(self, ibs, species):
        task_fnames = self.published[species]
        classifiers = {
            task_key: self._load_published(ibs, species, task_key)
            for task_key in task_fnames.keys()
        }
        return classifiers

    def find_pretrained(self):
        import glob
        import parse
        fname_fmt = self.fname_fmtstr + '.cPkl'
        task_clf_candidates = ut.ddict(list)
        globstr = self.fname_parts[0] + '.*.cPkl'
        for fpath in glob.iglob(join(self.dpath, globstr)):
            fname = basename(fpath)
            result = parse.parse(fname_fmt, fname)
            if result:
                task_key = result.named['task_key']
                task_clf_candidates[task_key].append(fpath)
        return task_clf_candidates

    def find_latest_remote(self):
        """
        Used to update the published dict
        """
        base_url = 'https://{remote}/public/models/pairclf'.format(
            **self.publish_info)
        import requests
        import bs4
        resp = requests.get(base_url)
        soup = bs4.BeautifulSoup(resp.text, 'html.parser')
        table = soup.findAll('table')[0]

        def parse_bs_table(table):
            n_columns = 0
            n_rows = 0
            column_names = []
            # Find number of rows and columns
            # we also find the column titles if we can
            for row in table.find_all('tr'):
                td_tags = row.find_all('td')
                if len(td_tags) > 0:
                    n_rows += 1
                    if n_columns == 0:
                        n_columns = len(td_tags)
                # Handle column names if we find them
                th_tags = row.find_all('th')
                if len(th_tags) > 0 and len(column_names) == 0:
                    for th in th_tags:
                        column_names.append(th.get_text())

            # Safeguard on Column Titles
            if len(column_names) > 0 and len(column_names) != n_columns:
                raise Exception("Column titles do not match the number of columns")
            columns = column_names if len(column_names) > 0 else range(0, n_columns)
            df = pd.DataFrame(columns=columns, index=list(range(0, n_rows)))
            row_marker = 0
            for row in table.find_all('tr'):
                column_marker = 0
                columns = row.find_all('td')
                for column in columns:
                    df.iat[row_marker, column_marker] = column.get_text().strip()
                    column_marker += 1
                if len(columns) > 0:
                    row_marker += 1
            return df
        df = parse_bs_table(table)
        # Find all available models
        df = df[df['Name'].map(lambda x: x.endswith('.cPkl'))]
        # df = df[df['Last modified'].map(len) > 0]

        fname_fmt = self.fname_fmtstr + '.cPkl'
        task_clf_candidates = ut.ddict(list)
        import parse
        for idx, row in df.iterrows():
            fname = basename(row['Name'])
            result = parse.parse(fname_fmt, fname)
            if result:
                task_key = result.named['task_key']
                species = result.named['species']
                task_clf_candidates[(species, task_key)].append(idx)

        task_clf_fnames = ut.ddict(dict)
        for key, idxs in task_clf_candidates.items():
            species, task_key = key
            # Find the classifier most recently created
            max_idx = ut.argmax(df.loc[idxs]['Last modified'].tolist())
            fname = df.loc[idxs[max_idx]]['Name']
            task_clf_fnames[species][task_key] = fname

        print('published = ' + ut.repr2(task_clf_fnames, nl=2))
        return task_clf_fnames

    def find_latest_local(self):
        """

            >>> self = Deployer()
            >>> self.find_pretrained()
            >>> self.find_latest_local()
        """
        from os.path import getctime
        task_clf_candidates = self.find_pretrained()
        task_clf_fpaths = {}
        for task_key, fpaths in task_clf_candidates.items():
            # Find the classifier most recently created
            fpath = fpaths[ut.argmax(map(getctime, fpaths))]
            task_clf_fpaths[task_key] = fpath
        return task_clf_fpaths

    def _make_deploy_metadata(self, task_key=None):
        pblm = self.pblm
        if pblm.samples is None:
            pblm.setup()

        if task_key is None:
            task_key = pblm.primary_task_key

        # task_keys = list(pblm.samples.supported_tasks())
        clf_key = pblm.default_clf_key
        data_key = pblm.default_data_key

        # Save the classifie
        data_info = pblm.feat_extract_info[data_key]
        feat_extract_config, feat_dims = data_info

        samples = pblm.samples
        labels = samples.subtasks[task_key]

        edge_hashid = samples.edge_set_hashid()
        label_hashid = samples.task_label_hashid(task_key)
        tasksamp_hashid = samples.task_sample_hashid(task_key)

        annot_hashid = ut.hashid_arr(samples._unique_annots.visual_uuids,
                                     'annots')

        # species = pblm.infr.ibs.get_primary_database_species(
        #     samples._unique_annots.aid)
        species = '+'.join(sorted(set(samples._unique_annots.species)))

        metadata = {
            'tasksamp_hashid': tasksamp_hashid,
            'edge_hashid': edge_hashid,
            'label_hashid': label_hashid,
            'annot_hashid': annot_hashid,
            'class_hist': labels.make_histogram(),
            'class_names': labels.class_names,
            'data_info': data_info,
            'task_key': task_key,
            'species': species,
            'data_key': data_key,
            'clf_key': clf_key,
            'n_dims': len(feat_dims),
            # 'aid_pairs': samples.aid_pairs,
        }

        meta_cfgstr = ut.repr2(metadata, kvsep=':', itemsep='', si=True)
        hashid = ut.hash_data(meta_cfgstr)[0:16]

        deploy_fname = (self.fname_fmtstr.format(hashid=hashid, **metadata) +
                        '.cPkl')

        deploy_metadata = metadata.copy()
        deploy_metadata['hashid'] = hashid
        deploy_metadata['fname'] = deploy_fname
        return deploy_metadata, deploy_fname

    def _make_deploy_info(self, task_key=None):
        pblm = self.pblm
        if pblm.samples is None:
            pblm.setup()

        if task_key is None:
            task_key = pblm.primary_task_key

        deploy_metadata, deploy_fname = self._make_deploy_metadata(task_key)
        clf_key = deploy_metadata['clf_key']
        data_key = deploy_metadata['data_key']

        clf = None
        if pblm.deploy_task_clfs:
            clf = pblm.deploy_task_clfs[task_key][clf_key][data_key]
        if not clf:
            pblm.learn_deploy_classifiers([task_key], clf_key, data_key)
            clf = pblm.deploy_task_clfs[task_key][clf_key][data_key]

        deploy_info = {
            'clf': clf,
            'metadata': deploy_metadata,
        }
        return deploy_info

    def ensure(self, task_key):
        _, fname = self._make_deploy_metadata(task_key=task_key)
        fpath = join(self.dpath, fname)
        if exists(fpath):
            deploy_info = ut.load_data(fpath)
            assert bool(deploy_info['clf']), 'must have clf'
        else:
            deploy_info = self.deploy(task_key=task_key)
            assert exists(fpath), 'must now exist'

    def deploy(self, task_key=None, publish=False):
        """
        Trains and saves a classifier for deployment

        Notes:
            A deployment consists of the following information
                * The classifier itself
                * Information needed to construct the input to the classifier
                    - TODO: can this be encoded as an sklearn pipeline?
                * Metadata concerning what data the classifier was trained with
                * PUBLISH TO /media/hdd/PUBLIC/models/pairclf

        Example:
            >>> from ibeis.algo.verif.vsone import *  # NOQA
            >>> params = dict(sample_method='random')
            >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST', **params)
            >>> pblm.setup(with_simple=False)
            >>> task_key = pblm.primary_task_key
            >>> self = Deployer(dpath='.', pblm=pblm)
            >>> deploy_info = self.deploy()

        Ignore:
            pblm.evaluate_classifiers(with_simple=False)
            res = pblm.task_combo_res[pblm.primary_task_key]['RF']['learn(sum,glob)']
        """
        deploy_info = self._make_deploy_info(task_key=task_key)
        deploy_fname = deploy_info['metadata']['fname']

        meta_fname = deploy_fname + self.meta_suffix
        deploy_fpath = join(self.dpath, deploy_fname)
        meta_fpath = join(self.dpath, meta_fname)

        ut.save_json(meta_fpath, deploy_info['metadata'])
        ut.save_data(deploy_fpath, deploy_info)

        if publish:
            user = ut.get_user_name()
            remote_uri = '{user}@{remote}:{path}'.format(user=user,
                                                         **self.publish_info)

            ut.rsync(meta_fpath, remote_uri + '/' + meta_fname)
            ut.rsync(deploy_fpath, remote_uri + '/' + deploy_fname)
        return deploy_info


@ut.reloadable_class
class AnnotPairSamples(clf_helpers.MultiTaskSamples):
    """
    Manages the different ways to assign samples (i.e. feat-label pairs) to
    1-v-1 classification

    CommandLine:
        python -m ibeis.algo.verif.vsone AnnotPairSamples

    Example:
        >>> from ibeis.algo.verif.vsone import *  # NOQA
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

    @profile
    def __init__(samples, ibs, aid_pairs, infr=None, apply=False):
        assert aid_pairs is not None
        super(AnnotPairSamples, samples).__init__(aid_pairs)
        samples.aid_pairs = np.array(aid_pairs)
        samples.infr = infr
        samples.ibs = ibs
        _unique_annots = ibs.annots(np.unique(samples.aid_pairs)).view()
        samples._unique_annots = _unique_annots
        samples.annots1 = _unique_annots.view(samples.aid_pairs.T[0])
        samples.annots2 = _unique_annots.view(samples.aid_pairs.T[1])
        samples.n_samples = len(aid_pairs)
        samples.X_dict = None
        samples.simple_scores = None
        if apply:
            samples.apply_multi_task_multi_label()
        # samples.apply_multi_task_binary_label()

    # @profile
    # def edge_hashids(samples):
    #     qvuuids = samples.annots1.visual_uuids
    #     dvuuids = samples.annots2.visual_uuids
    #     # edge_uuids = [ut.combine_uuids(uuids)
    #     #                for uuids in zip(qvuuids, dvuuids)]
    #     edge_hashids = [make_edge_hashid(uuid1, uuid2) for uuid1, uuid2 in zip(qvuuids, dvuuids)]
    #     # edge_uuids = [combine_2uuids(uuid1, uuid2)
    #     #                for uuid1, uuid2 in zip(qvuuids, dvuuids)]
    #     return edge_hashids

    # @profile
    # def edge_hashid(samples):
    #     edge_hashids = samples.edge_hashids()
    #     edge_hashid = ut.hashstr_arr27(edge_hashids, 'edges', hashlen=32,
    #                                    pathsafe=True)
    #     return edge_hashid

    @profile
    def edge_set_hashid(samples):
        """
        Faster than using ut.combine_uuids, because we condense and don't
        bother casting back to UUIDS, and we just directly hash.
        """
        qvuuids = samples.annots1.visual_uuids
        dvuuids = samples.annots2.visual_uuids
        hasher = hashlib.sha1()
        for uuid1, uuid2 in zip(qvuuids, dvuuids):
            hasher.update(uuid1.bytes)
            hasher.update(uuid2.bytes)
            hasher.update(b'-')
        edge_bytes = hasher.digest()
        edge_hash = ut.util_hash.convert_bytes_to_bigbase(edge_bytes)
        edge_hashstr = edge_hash[0:16]
        edge_hashid = 'e{}-{}'.format(len(samples), edge_hashstr)
        return edge_hashid

    @ut.memoize
    @profile
    def sample_hashid(samples):
        visual_hash = samples.edge_set_hashid()
        # visual_hash = samples.edge_hashid()
        ut.hashid_arr(samples.encoded_1d().values, 'labels')
        label_hash = ut.hash_data(samples.encoded_1d().values)[0:16]
        sample_hash = visual_hash + '_' + label_hash
        return sample_hash

    def task_label_hashid(samples, task_key):
        labels = samples.subtasks[task_key]
        label_hashid = ut.hashid_arr(labels.y_enc, 'labels')
        return label_hashid

    @ut.memoize
    @profile
    def task_sample_hashid(samples, task_key):
        labels = samples.subtasks[task_key]
        edge_hashid = samples.edge_set_hashid()
        label_hashid = samples.task_label_hashid(task_key)
        tasksamp_hashstr = ut.hash_data([edge_hashid, label_hashid])[0:16]
        tasksamp_hashid = 'tasksamp-{},{}-{}'.format(
            len(samples), labels.n_classes, tasksamp_hashstr)
        return tasksamp_hashid

    def set_simple_scores(samples, simple_scores):
        if simple_scores is not None:
            edges = ut.emap(tuple, samples.aid_pairs.tolist())
            assert (edges == simple_scores.index.tolist())
        samples.simple_scores = simple_scores

    def set_feats(samples, X_dict):
        if X_dict is not None:
            edges = ut.emap(tuple, samples.aid_pairs.tolist())
            for X in X_dict.values():
                assert np.all(edges == X.index.tolist())
        samples.X_dict = X_dict

    @profile
    def compress(samples, flags):
        assert len(flags) == len(samples), 'mask has incorrect size'
        infr = samples.infr
        simple_scores = samples.simple_scores[flags]
        X_dict = ut.map_vals(lambda val: val[flags], samples.X_dict)
        aid_pairs = samples.aid_pairs[flags]
        ibs = samples.ibs
        new_labels = AnnotPairSamples(ibs, aid_pairs, infr, apply=True)
        new_labels.set_feats(X_dict)
        new_labels.set_simple_scores(simple_scores)
        return new_labels

    @ut.memoize
    @profile
    def is_same(samples):
        infr = samples.infr
        edges = samples.aid_pairs
        nodes = np.unique(edges)
        labels = infr.pos_graph.node_labels(*nodes)
        lookup = dict(ut.dzip(nodes, labels))
        # def _check2(u, v):
        #     nid1, nid2 = lookup[u], lookup[v]
        #     if nid1 == nid2:
        #         return True
        #     else:
        #         return False
        #     # elif infr.neg_redun_nids.has_edge(nid1, nid2):
        #     #     return False
        #     # else:
        #     #     return None
        # flags = [_check2(u, v) for (u, v) in edges]
        flags = [lookup[u] == lookup[v] for (u, v) in edges]
        # def _check(u, v):
        #     nid1, nid2 = infr.pos_graph.node_labels(u, v)
        #     if nid1 == nid2:
        #         return True
        #     elif infr.neg_redun_nids.has_edge(nid1, nid2):
        #         return False
        #     else:
        #         return None
        # flags = [_check(u, v) for (u, v) in edges]
        return np.array(flags, dtype=np.bool)
        # return samples.infr.is_same(samples.aid_pairs)

    @ut.memoize
    @profile
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
    @profile
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

    @profile
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

    @profile
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

    @profile
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
            featinfo = vt.AnnotPairFeatInfo(samples.X_dict[data_key])
            print(ut.indent(featinfo.get_infostr()))


def demo_single_pairwise_feature_vector():
    r"""
    CommandLine:
        python -m ibeis.algo.verif.vsone demo_single_pairwise_feature_vector

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.verif.vsone import *  # NOQA
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

    # match.add_global_measures(['yaw', 'qual', 'gps', 'time'])
    match.add_global_measures(['view', 'qual', 'gps', 'time'])
    match.add_local_measures()

    # sorters = ['ratio', 'norm_dist', 'match_dist']
    match.make_feature_vector()
    return match


# @profile
# def make_edge_hashid(uuid1, uuid2):
#     """
#     Slightly faster than using ut.combine_uuids, because we condense and don't
#     bother casting back to UUIDS
#     """
#     sep_str = '-'
#     sep_byte = six.b(sep_str)
#     pref = six.b('{}2'.format(sep_str))
#     combined_bytes = pref + sep_byte.join([uuid1.bytes, uuid2.bytes])
#     bytes_sha1 = hashlib.sha1(combined_bytes)
#     # Digest them into a hash
#     hashbytes_20 = bytes_sha1.digest()
#     hashbytes_16 = hashbytes_20[0:16]
#     # uuid_ = uuid.UUID(bytes=hashbytes_16)
#     return hashbytes_16


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.verif.vsone
        python -m ibeis.algo.verif.vsone --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
