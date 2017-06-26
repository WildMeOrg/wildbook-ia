
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import plottool as pt
import utool as ut
from ibeis.algo.verif import vsone
from ibeis.scripts._thesis_helpers import DBInputs
from ibeis.scripts.thesis import Sampler
from ibeis.scripts._thesis_helpers import Tabular, upper_one
from ibeis.scripts._thesis_helpers import TMP_RC, W, H, DPI
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
import numpy as np  # NOQA
import pandas as pd
import ubelt as ub
import matplotlib as mpl
from os.path import basename, join, splitext, exists  # NOQA
import ibeis.constants as const
import vtool as vt
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA
(print, rrr, profile) = ut.inject2(__name__)


def turk_pz():
    import ibeis

    ibs = ibeis.opendb('GZ_Master1')
    infr = ibeis.AnnotInference(ibs, aids='all')
    infr.reset_feedback('staging', apply=True)
    infr.relabel_using_reviews(rectify=True)
    # infr.apply_nondynamic_update()
    print(ut.repr4(infr.status()))

    infr.ibeis_delta_info()

    infr.match_state_delta()
    infr.get_ibeis_name_delta()

    infr.relabel_using_reviews(rectify=True)
    infr.write_ibeis_annotmatch_feedback()
    infr.write_ibeis_name_assignment()

    pass


@ut.reloadable_class
class VerifierExpt(DBInputs):
    """
    Collect data from experiments to visualize

    Ignore:
        >>> from ibeis.scripts.thesis import *
        >>> fpath = ut.glob(ut.truepath('~/Desktop/mtest_plots'), '*.pkl')[0]
        >>> self = ut.load_data(fpath)
    """
    base_dpath = ut.truepath('~/Desktop/pair_expts')

    task_nice_lookup = {
        'match_state': const.REVIEW.CODE_TO_NICE,
        'photobomb_state': {
            'pb': 'Photobomb',
            'notpb': 'Not Photobomb',
        }
    }

    def _setup(self):
        r"""
        CommandLine:
            python -m ibeis VerifierExpt._setup --db GZ_Master1

            python -m ibeis VerifierExpt._setup --db PZ_Master1 --eval
            python -m ibeis VerifierExpt._setup --db PZ_MTEST
            python -m ibeis VerifierExpt._setup --db PZ_PB_RF_TRAIN

            python -m ibeis VerifierExpt.measure_all --db PZ_PB_RF_TRAIN

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='GZ_Master1')
            >>> self = VerifierExpt(dbname)
            >>> self._setup()

        Ignore:
            from ibeis.scripts.thesis import *
            self = VerifierExpt('PZ_Master1')

            from ibeis.scripts.thesis import *
            self = VerifierExpt('PZ_PB_RF_TRAIN')

            self.ibs.print_annot_stats(aids, prefix='P')
        """
        import ibeis
        self._precollect()
        ibs = self.ibs

        if ibs.dbname == 'PZ_Master1':
            # FIND ALL PHOTOBOMB / INCOMPARABLE CASES
            if False:
                infr = ibeis.AnnotInference(ibs, aids='all')
                infr.reset_feedback('staging', apply=True)
                print(ut.repr4(infr.status()))

                pblm = vsone.OneVsOneProblem.from_aids(ibs, self.aids_pool)
                pblm.load_samples()
                pblm.samples.print_info()

            aids = self.aids_pool
        else:
            aids = self.aids_pool

        pblm = vsone.OneVsOneProblem.from_aids(ibs, aids)
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key
        pblm.eval_task_keys = ['match_state', 'photobomb_state']
        pblm.eval_data_keys = [data_key]
        pblm.eval_clf_keys = [clf_key]

        if ut.get_argflag('--eval'):
            pblm.eval_task_keys = ['photobomb_state', 'match_state']
            # pblm.eval_task_keys = ['match_state']
            pblm.eval_data_keys = None
            pblm.evaluate_classifiers()
            pblm.eval_data_keys = [data_key]
        else:
            pblm.setup_evaluation()

        if False:
            pblm.infr
            pblm.load_samples()

        # pblm.evaluate_classifiers()
        ibs = pblm.infr.ibs
        pblm.samples.print_info()

        species_code = ibs.get_database_species(pblm.infr.aids)[0]
        if species_code == 'zebra_plains':
            species = 'Plains Zebras'
        if species_code == 'zebra_grevys':
            species = 'Grévy\'s Zebras'
        dbcode = '{}_{}'.format(ibs.dbname, len(pblm.samples))

        self.pblm = pblm
        self.dbcode = dbcode
        self.eval_task_keys = pblm.eval_task_keys
        self.species = species
        self.data_key = data_key
        self.clf_key = clf_key

        # RESET DPATH BASED ON SAMPLE?
        # MAYBE SYMLINK TO NEW DPATH?
        from os.path import expanduser
        dpath = expanduser(self.base_dpath + '/' + self.dbcode)
        link = expanduser(self.base_dpath + '/' + self.dbname)
        ut.ensuredir(dpath)
        self.real_dpath = dpath
        try:
            self.link = ut.symlink(dpath, link, overwrite=True)
        except Exception:
            if exists(dpath):
                newpath = ut.non_existing_path(dpath, suffix='_old')
                ut.move(link, newpath)
                self.link = ut.symlink(dpath, link)

    def measure_all(self):
        r"""
        CommandLine:
            python -m ibeis VerifierExpt.measure_all --db PZ_PB_RF_TRAIN
            python -m ibeis VerifierExpt.measure_all --db PZ_MTEST
            python -m ibeis VerifierExpt.measure_all

            python -m ibeis VerifierExpt.measure_all --db GZ_Master1

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = VerifierExpt(dbname)
            >>>     self.measure_all()
        """
        self._setup()
        pblm = self.pblm

        expt_name = 'sample_info'
        results = {
            'graph': pblm.infr.graph,
            'aid_pool': self.aids_pool,
            'pblm_aids': pblm.infr.aids,
            'encoded_labels2d': pblm.samples.encoded_2d(),
            'subtasks': pblm.samples.subtasks,
            'multihist': pblm.samples.make_histogram(),
        }
        self.expt_results[expt_name] = results
        ut.save_data(join(str(self.dpath), expt_name + '.pkl'), results)

        importance = {
            task_key: pblm.feature_importance(task_key=task_key)
            for task_key in pblm.eval_task_keys
        }

        task = pblm.samples['match_state']
        scores = pblm.samples.simple_scores['score_lnbnn_1vM']
        y = task.indicator_df[task.default_class_name]
        lnbnn_xy = pd.concat([scores, y], axis=1)

        results = {
            'lnbnn_xy': lnbnn_xy,
            'task_combo_res': self.pblm.task_combo_res,
            'importance': importance,
            'data_key': self.data_key,
            'clf_key': self.clf_key,
        }
        expt_name = 'all'
        self.expt_results[expt_name] = results
        ut.save_data(join(str(self.dpath), expt_name + '.pkl'), results)

        task_key = 'match_state'
        if task_key in pblm.eval_task_keys:
            self.measure_hard_cases(task_key)

        task_key = 'photobomb_state'
        if task_key in pblm.eval_task_keys:
            self.measure_hard_cases(task_key)

        self.measure_rerank()
        self.measure_prune()

        if ut.get_argflag('--draw'):
            self.draw_all()

    def draw_all(self):
        r"""
        CommandLine:
            python -m ibeis VerifierExpt.draw_all --db PZ_MTEST
            python -m ibeis VerifierExpt.draw_all --db PZ_PB_RF_TRAIN
            python -m ibeis VerifierExpt.draw_all --db GZ_Master1
            python -m ibeis VerifierExpt.draw_all --db PZ_Master1

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = VerifierExpt(dbname)
            >>>     self.draw_all()
        """
        results = self.ensure_results('all')
        eval_task_keys = set(results['task_combo_res'].keys())

        self.write_sample_info()

        task_key = 'photobomb_state'
        if task_key in eval_task_keys:
            self.write_importance(task_key)
            self.write_metrics(task_key)
            self.write_metrics2(task_key)
            self.draw_roc(task_key)
            self.draw_mcc_thresh(task_key)

        task_key = 'match_state'
        if task_key in eval_task_keys:
            self.draw_class_score_hist()
            self.draw_roc(task_key)
            self.draw_mcc_thresh(task_key)

            self.draw_wordcloud(task_key)
            self.write_importance(task_key)
            self.write_metrics(task_key)

            self.draw_rerank()

            if not ut.get_argflag('--noprune'):
                self.draw_prune()

        if not ut.get_argflag('--nodraw'):
            task_key = 'match_state'
            if task_key in eval_task_keys:
                self.draw_hard_cases(task_key)

            task_key = 'photobomb_state'
            if task_key in eval_task_keys:
                self.draw_hard_cases(task_key)

    def measure_prune(self):
        """
        >>> from ibeis.scripts.thesis import *
        >>> self = VerifierExpt('GZ_Master1')
        >>> self = VerifierExpt('PZ_Master1')
        >>> self = VerifierExpt('PZ_MTEST')
        """
        # from sklearn.feature_selection import SelectFromModel
        from ibeis.scripts import clf_helpers
        if getattr(self, 'pblm', None) is None:
            self._setup()

        pblm = self.pblm
        task_key = pblm.primary_task_key
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key

        featinfo = vt.AnnotPairFeatInfo(pblm.samples.X_dict[data_key])
        print(featinfo.get_infostr())

        labels = pblm.samples.subtasks[task_key]
        # X = pblm.samples.X_dict[data_key]

        feat_dims = pblm.samples.X_dict[data_key].columns.tolist()
        n_orig = len(feat_dims)
        n_dims = []
        reports = []
        sub_reports = []
        mdis_list = []

        prune_rate = 1
        min_feats = 1

        n_steps_needed = int(np.ceil((n_orig - min_feats) / prune_rate))
        prog = ub.ProgIter(range(n_steps_needed), label='prune')
        for _ in prog:
            prog.ensure_newline()
            clf_list, res_list = pblm._train_evaluation_clf(task_key, data_key,
                                                            clf_key, feat_dims)
            combo_res = clf_helpers.ClfResult.combine_results(res_list, labels)
            rs = [res.extended_clf_report(verbose=0) for res in res_list]
            report = combo_res.extended_clf_report(verbose=0)

            # Measure mean decrease in impurity
            clf_mdi = np.array(
                [clf_.feature_importances_ for clf_ in clf_list])
            mean_mdi = ut.dzip(feat_dims, np.mean(clf_mdi, axis=0))

            # Record state
            n_dims.append(len(feat_dims))
            reports.append(report)
            sub_reports.append(rs)
            mdis_list.append(mean_mdi)

            # remove the worst features
            sorted_featdims = ub.argsort(mean_mdi)
            n_have = len(sorted_featdims)
            n_remove = (n_have - max(n_have - prune_rate, min_feats))
            worst_features = sorted_featdims[0:n_remove]
            for f in worst_features:
                feat_dims.remove(f)

        results = {
            'n_dims': n_dims,
            'reports': reports,
            'sub_reports': sub_reports,
            'mdis_list': mdis_list,
        }
        expt_name = 'prune'
        self.expt_results[expt_name] = results
        ut.save_data(join(str(self.dpath), expt_name + '.pkl'), results)

    def measure_rerank(self):
        """
            >>> from ibeis.scripts.thesis import *
            >>> defaultdb = 'PZ_Master1'
            >>> defaultdb = 'GZ_Master1'
            >>> self = VerifierExpt(defaultdb)
            >>> self._setup()
            >>> self.measure_rerank()
        """
        if getattr(self, 'pblm', None) is None:
            self._setup()

        pblm = self.pblm
        infr = pblm.infr
        ibs = pblm.infr.ibs

        # NOTE: this is not the aids_pool for PZ_Master1
        aids = pblm.infr.aids

        qaids, daids_list, info_list = Sampler._varied_inputs(ibs, aids)

        if pblm.hyper_params['vsone_kpts']['augment_orientation']:
            # HACK
            cfgdict = {
                'query_rotation_heuristic': True,
            }
        else:
            cfgdict = {}
        daids = daids_list[0]
        info = info_list[0]

        # Execute the ranking algorithm
        qaids = sorted(qaids)
        daids = sorted(daids)
        qreq_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict)
        cm_list = qreq_.execute()
        cm_list = [cm.extend_results(qreq_) for cm in cm_list]

        # Measure LNBNN rank probabilities
        top = 20
        rerank_pairs = []
        for cm in cm_list:
            pairs = [infr.e_(cm.qaid, daid) for daid in cm.get_top_aids(top)]
            rerank_pairs.extend(pairs)
        rerank_pairs = list(set(rerank_pairs))

        probs = pblm.predict_proba_evaluation(infr, rerank_pairs)['match_state']
        pos_probs = probs[POSTV]

        clf_name_ranks = []
        lnbnn_name_ranks = []
        infr = pblm.infr
        for cm in cm_list:
            daids = cm.get_top_aids(top)
            edges = [infr.e_(cm.qaid, daid) for daid in daids]
            dnids = cm.dnid_list[ut.take(cm.daid2_idx, daids)]
            scores = pos_probs.loc[edges].values

            sortx = np.argsort(scores)[::-1]
            clf_ranks = np.where(cm.qnid == dnids[sortx])[0]
            if len(clf_ranks) == 0:
                clf_rank = len(cm.unique_nids) - 1
            else:
                clf_rank = clf_ranks[0]
            lnbnn_rank = cm.get_name_ranks([cm.qnid])[0]
            clf_name_ranks.append(clf_rank)
            lnbnn_name_ranks.append(lnbnn_rank)

        bins = np.arange(len(qreq_.dnids))
        hist = np.histogram(lnbnn_name_ranks, bins=bins)[0]
        lnbnn_cdf = (np.cumsum(hist) / sum(hist))

        bins = np.arange(len(qreq_.dnids))
        hist = np.histogram(clf_name_ranks, bins=bins)[0]
        clf_cdf = (np.cumsum(hist) / sum(hist))

        results = [
            (lnbnn_cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})),
            (clf_cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})),
        ]
        expt_name = 'rerank'
        self.expt_results[expt_name] = results
        ut.save_data(join(str(self.dpath), expt_name + '.pkl'), results)

    def measure_hard_cases(self, task_key):
        """
        Find a failure case for each class

        CommandLine:
            python -m ibeis VerifierExpt.measure hard_cases GZ_Master1 match_state
            python -m ibeis VerifierExpt.measure hard_cases GZ_Master1 photobomb_state
            python -m ibeis VerifierExpt.draw hard_cases GZ_Master1 match_state
            python -m ibeis VerifierExpt.draw hard_cases GZ_Master1 photobomb_state

            python -m ibeis VerifierExpt.measure hard_cases PZ_Master1 match_state
            python -m ibeis VerifierExpt.measure hard_cases PZ_Master1 photobomb_state
            python -m ibeis VerifierExpt.draw hard_cases PZ_Master1 match_state
            python -m ibeis VerifierExpt.draw hard_cases PZ_Master1 photobomb_state

            python -m ibeis VerifierExpt.measure hard_cases PZ_MTEST match_state
            python -m ibeis VerifierExpt.draw hard_cases PZ_MTEST photobomb_state

        Ignore:
            >>> task_key = 'match_state'
            >>> task_key = 'photobomb_state'
            >>> from ibeis.scripts.thesis import *
            >>> self = VerifierExpt('GZ_Master1')
            >>> self._setup()
        """
        if getattr(self, 'pblm', None) is None:
            print('Need to setup before measuring hard cases')
            self._setup()
        print('Measuring hard cases')

        pblm = self.pblm

        front = mid = back = 8

        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]

        print('task_key = %r' % (task_key,))
        if task_key == 'photobomb_state':
            method = 'max-mcc'
            method = res.get_thresholds('mcc', 'maximize')
            print('Using thresholds: ' + ut.repr4(method))
        else:
            method = 'argmax'
            print('Using argmax')

        case_df = res.hardness_analysis(pblm.samples, pblm.infr, method=method)
        # group = case_df.sort_values(['real_conf', 'easiness'])
        case_df = case_df.sort_values(['easiness'])

        # failure_cases = case_df[(case_df['real_conf'] > 0) & case_df['failed']]
        failure_cases = case_df[case_df['failed']]
        if len(failure_cases) == 0:
            print('No reviewed failures exist. Do pblm.qt_review_hardcases')

        print('There are {} failure cases'.format(len(failure_cases)))
        print('With average hardness {}'.format(
            ut.repr2(ut.stats_dict(failure_cases['hardness']), strkeys=True,
                     precision=2)))

        cases = []
        for (pred, real), group in failure_cases.groupby(('pred', 'real')):

            group = group.sort_values(['easiness'])
            flags = ut.flag_percentile_parts(group['easiness'], front, mid, back)
            subgroup = group[flags]

            print('Selected {} r({})-p({}) cases'.format(
                len(subgroup), res.class_names[real], res.class_names[pred]
            ))
            # ut.take_percentile_parts(group['easiness'], front, mid, back)

            # Prefer examples we have manually reviewed before
            # group = group.sort_values(['real_conf', 'easiness'])
            # subgroup = group[0:num_top]

            for idx, case in subgroup.iterrows():
                edge = tuple(ut.take(case, ['aid1', 'aid2']))
                cases.append({
                    'edge': edge,
                    'real': res.class_names[case['real']],
                    'pred': res.class_names[case['pred']],
                    'failed': case['failed'],
                    'easiness': case['easiness'],
                    'real_conf': case['real_conf'],
                    'probs': res.probs_df.loc[edge].to_dict(),
                    'edge_data': pblm.infr.get_edge_data(edge),
                })

        print('Selected %d cases in total' % (len(cases)))

        # Augment cases with their one-vs-one matches
        infr = pblm.infr
        data_key = self.data_key
        config = pblm.feat_extract_info[data_key][0]['match_config']
        edges = [case['edge'] for case in cases]
        matches = infr._exec_pairwise_match(edges, config)

        def _prep_annot(annot):
            # Load data needed for plot into annot dictionary
            annot['aid']
            annot['rchip']
            annot['kpts']
            # Cast the lazy dict to a real one
            return {k: annot[k] for k in annot.evaluated_keys()}

        for case, match in zip(cases, matches):
            # store its chip fpath and other required info
            match.annot1 = _prep_annot(match.annot1)
            match.annot2 = _prep_annot(match.annot2)
            case['match'] = match

        fpath = join(str(self.dpath), task_key + '_hard_cases.pkl')
        ut.save_data(fpath, cases)
        print('Hard case space on disk: {}'.format(ut.get_file_nBytes_str(fpath)))

        # if False:
        #     ybin_df = res.target_bin_df
        #     flags = ybin_df['pb'].values
        #     pb_edges = ybin_df[flags].index.tolist()

        #     matches = infr._exec_pairwise_match(pb_edges, config)
        #     prefix = 'training_'

        #     subdir = 'temp_cases_{}'.format(task_key)
        #     dpath = join(str(self.dpath), subdir)
        #     ut.ensuredir(dpath)

        #     tbl = pblm.infr.ibs.db.get_table_as_pandas('annotmatch')
        #     tagged_tbl = tbl[~pd.isnull(tbl['annotmatch_tag_text']).values]
        #     ttext = tagged_tbl['annotmatch_tag_text']
        #     flags = ['photobomb' in t.split(';') for t in ttext]
        #     pb_table = tagged_tbl[flags]
        #     am_pb_edges = set(
        #         ut.estarmap(infr.e_, zip(pb_table.annot_rowid1.tolist(),
        #                                  pb_table.annot_rowid2.tolist())))

        #     # missing = am_pb_edges - set(pb_edges)
        #     # matches = infr._exec_pairwise_match(missing, config)
        #     # prefix = 'missing_'

        #     # infr.relabel_using_reviews()
        #     # infr.apply_nondynamic_update()

        #     # infr.verbose = 100
        #     # for edge in missing:
        #     #     print(edge[0] in infr.aids)
        #     #     print(edge[1] in infr.aids)

        #     # fix = [
        #     #     (1184, 1185),
        #     #     (1376, 1378),
        #     #     (1377, 1378),
        #     # ]

        #     #     fb = infr.current_feedback(edge).copy()
        #     #     fb = ut.dict_subset(fb, ['decision', 'tags', 'confidence'],
        #     #                         default=None)
        #     #     fb['user_id'] = 'jon_fixam'
        #     #     fb['confidence'] = 'pretty_sure'
        #     #     fb['tags'] += ['photobomb']
        #     #     infr.add_feedback(edge, **fb)

        #     for c, match in enumerate(ut.ProgIter(matches)):
        #         edge = match.annot1['aid'], match.annot2['aid']

        #         fig = pt.figure(fnum=1, clf=True)
        #         ax = pt.gca()
        #         # Draw with feature overlay
        #         match.show(ax, vert=False, heatmask=True, show_lines=True,
        #                    show_ell=False, show_ori=False, show_eig=False,
        #                    line_lw=1, line_alpha=.1,
        #                    modifysize=True)
        #         fname = prefix + '_'.join(ut.emap(str, edge))
        #         ax.set_xlabel(fname)
        #         fpath = join(str(dpath), fname + '.jpg')
        #         vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        #     # visualize real photobomb cases
        return cases

    def draw_hard_cases(self, task_key):
        """
        draw hard cases with and without overlay

        python -m ibeis VerifierExpt.draw hard_cases GZ_Master1 match_state
        python -m ibeis VerifierExpt.draw hard_cases PZ_Master1 match_state
        python -m ibeis VerifierExpt.draw hard_cases PZ_Master1 photobomb_state
        python -m ibeis VerifierExpt.draw hard_cases GZ_Master1 photobomb_state



            >>> from ibeis.scripts.thesis import *
            >>> self = VerifierExpt('PZ_MTEST')
            >>> task_key = 'match_state'
            >>> self.draw_hard_cases(task_key)
        """
        cases = self.ensure_results(task_key + '_hard_cases')
        print('Loaded {} {} hard cases'.format(len(cases), task_key))

        subdir = 'cases_{}'.format(task_key)
        dpath = join(str(self.dpath), subdir)
        # ut.delete(dpath)
        ut.ensuredir(dpath)
        code_to_nice = self.task_nice_lookup[task_key]

        mpl.rcParams.update(TMP_RC)

        pz_gt_errors = {  # NOQA
            # The true state of these pairs are:
            NEGTV: [
                (239, 3745),
                (484, 519),
                (802, 803),
            ],
            INCMP: [
                (4652, 5245),
                (4405, 5245),
                (4109, 5245),
                (16192, 16292),
            ],
            POSTV: [
                (6919, 7192),
            ]
        }

        prog = ut.ProgIter(cases, 'draw {} hard case'.format(task_key),
                           bs=False)
        for case in prog:
            aid1, aid2 = case['edge']
            match = case['match']
            real_name, pred_name = case['real'], case['pred']
            real_nice, pred_nice = ut.take(code_to_nice, [real_name, pred_name])
            fname = 'fail_{}_{}_{}_{}'.format(real_name, pred_name, aid1, aid2)
            # Build x-label
            _probs = case['probs']
            probs = ut.odict((v, _probs[k])
                             for k, v in code_to_nice.items()
                             if k in _probs)
            probstr = ut.repr2(probs, precision=2, strkeys=True, nobr=True)
            xlabel = 'real={}, pred={},\n{}'.format(real_nice, pred_nice,
                                                    probstr)
            fig = pt.figure(fnum=1000, clf=True)
            ax = pt.gca()
            # Draw with feature overlay
            match.show(ax, vert=False, heatmask=True,
                       show_lines=False,
                       # show_lines=True, line_lw=1, line_alpha=.1,
                       # ell_alpha=.3,
                       show_ell=False, show_ori=False, show_eig=False,
                       modifysize=True)
            ax.set_xlabel(xlabel)
            # ax.get_xaxis().get_label().set_fontsize(24)
            ax.get_xaxis().get_label().set_fontsize(24)

            fpath = join(str(dpath), fname + '.jpg')
            vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def write_metrics2(self, task_key='match_state'):
        """
        CommandLine:
            python -m ibeis VerifierExpt.draw metrics PZ_PB_RF_TRAIN match_state
            python -m ibeis VerifierExpt.draw metrics2 PZ_Master1 photobomb_state
            python -m ibeis VerifierExpt.draw metrics2 GZ_Master1 photobomb_state

            python -m ibeis VerifierExpt.draw metrics2 GZ_Master1 photobomb_state
        """
        results = self.ensure_results('all')
        task_combo_res = results['task_combo_res']
        data_key = results['data_key']
        clf_key = results['clf_key']

        res = task_combo_res[task_key][clf_key][data_key]

        from ibeis.scripts import sklearn_utils
        threshes = res.get_thresholds('mcc', 'max')
        y_pred = sklearn_utils.predict_from_probs(res.probs_df, threshes,
                                                  force=True)
        y_true = res.target_enc_df

        # pred_enc = res.clf_probs.argmax(axis=1)
        # y_pred = pred_enc
        res.augment_if_needed()
        sample_weight = res.sample_weight
        target_names = res.class_names

        report = sklearn_utils.classification_report2(
            y_true, y_pred, target_names, sample_weight, verbose=False)
        metric_df = report['metrics']
        confusion_df = report['confusion']

        print(metric_df)
        print(confusion_df)

        # df = self.task_confusion[task_key]
        df = confusion_df
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.rename_axis(self.task_nice_lookup[task_key], 1)
        df.index.name = None
        df.columns.name = None

        colfmt = '|l|' + 'r' * (len(df) - 1) + '|l|'
        tabular = Tabular(df, colfmt=colfmt, hline=True)
        tabular.groupxs = [list(range(len(df) - 1)), [len(df) - 1]]
        latex_str = tabular.as_tabular()

        sum_pred = df.index[-1]
        sum_real = df.columns[-1]
        latex_str = latex_str.replace(sum_pred, r'$\sum$ predicted')
        latex_str = latex_str.replace(sum_real, r'$\sum$ real')
        confusion_tex = ut.align(latex_str, '&', pos=None)
        print(confusion_tex)

        df = metric_df
        # df = self.task_metrics[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.rename_axis({'mcc': 'MCC'}, 1)
        df = df.drop(['markedness', 'bookmaker', 'fpr'], axis=1)
        df.index.name = None
        df.columns.name = None
        df['support'] = df['support'].astype(np.int)
        df.columns = ut.emap(upper_one, df.columns)

        import re
        tabular = Tabular(df, colfmt='numeric')
        top, header, mid, bot = tabular.as_parts()
        lines = mid[0].split('\n')
        newmid = [lines[0:-1], lines[-1:]]
        tabular.parts = (top, header, newmid, bot)
        latex_str = tabular.as_tabular()
        latex_str = re.sub(' -0.00 ', '  0.00 ', latex_str)
        metrics_tex = latex_str
        print(metrics_tex)

        dpath = str(self.dpath)
        confusion_fname = 'confusion2_{}'.format(task_key)
        metrics_fname = 'eval_metrics2_{}'.format(task_key)

        ut.write_to(join(dpath, confusion_fname + '.tex'), confusion_tex)
        ut.write_to(join(dpath, metrics_fname + '.tex'), metrics_tex)

        fpath1 = ut.render_latex(confusion_tex, dpath=dpath,
                                 fname=confusion_fname)
        fpath2 = ut.render_latex(metrics_tex, dpath=dpath, fname=metrics_fname)
        return fpath1, fpath2

    def write_metrics(self, task_key='match_state'):
        """
        CommandLine:
            python -m ibeis VerifierExpt.draw metrics PZ_PB_RF_TRAIN match_state
            python -m ibeis VerifierExpt.draw metrics GZ_Master1 photobomb_state

            python -m ibeis VerifierExpt.draw metrics PZ_Master1,GZ_Master1 photobomb_state,match_state

        Ignore:
            >>> from ibeis.scripts.thesis import *
            >>> self = VerifierExpt('PZ_Master1')
            >>> task_key = 'match_state'
        """
        results = self.ensure_results('all')
        task_combo_res = results['task_combo_res']
        data_key = results['data_key']
        clf_key = results['clf_key']

        res = task_combo_res[task_key][clf_key][data_key]
        res.augment_if_needed()
        pred_enc = res.clf_probs.argmax(axis=1)
        y_pred = pred_enc
        y_true = res.y_test_enc
        sample_weight = res.sample_weight
        target_names = res.class_names

        from ibeis.scripts import sklearn_utils
        report = sklearn_utils.classification_report2(
            y_true, y_pred, target_names, sample_weight, verbose=False)
        metric_df = report['metrics']
        confusion_df = report['confusion']

        print(metric_df)
        print(confusion_df)

        # df = self.task_confusion[task_key]
        df = confusion_df
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.rename_axis(self.task_nice_lookup[task_key], 1)
        df.index.name = None
        df.columns.name = None

        colfmt = '|l|' + 'r' * (len(df) - 1) + '|l|'
        tabular = Tabular(df, colfmt=colfmt, hline=True)
        tabular.groupxs = [list(range(len(df) - 1)), [len(df) - 1]]
        latex_str = tabular.as_tabular()

        sum_pred = df.index[-1]
        sum_real = df.columns[-1]
        latex_str = latex_str.replace(sum_pred, r'$\sum$ predicted')
        latex_str = latex_str.replace(sum_real, r'$\sum$ real')
        confusion_tex = ut.align(latex_str, '&', pos=None)
        print(confusion_tex)

        df = metric_df
        # df = self.task_metrics[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.rename_axis({'mcc': 'MCC'}, 1)
        df = df.drop(['markedness', 'bookmaker', 'fpr'], axis=1)
        df.index.name = None
        df.columns.name = None
        df['support'] = df['support'].astype(np.int)
        df.columns = ut.emap(upper_one, df.columns)

        import re
        tabular = Tabular(df, colfmt='numeric')
        top, header, mid, bot = tabular.as_parts()
        lines = mid[0].split('\n')
        newmid = [lines[0:-1], lines[-1:]]
        tabular.parts = (top, header, newmid, bot)
        latex_str = tabular.as_tabular()
        latex_str = re.sub(' -0.00 ', '  0.00 ', latex_str)
        metrics_tex = latex_str
        print(metrics_tex)

        dpath = str(self.dpath)
        confusion_fname = 'confusion_{}'.format(task_key)
        metrics_fname = 'eval_metrics_{}'.format(task_key)

        ut.write_to(join(dpath, confusion_fname + '.tex'), confusion_tex)
        ut.write_to(join(dpath, metrics_fname + '.tex'), metrics_tex)

        fpath1 = ut.render_latex(confusion_tex, dpath=dpath,
                                 fname=confusion_fname)
        fpath2 = ut.render_latex(metrics_tex, dpath=dpath, fname=metrics_fname)
        return fpath1, fpath2

    def write_sample_info(self):
        """
        python -m ibeis VerifierExpt.draw sample_info GZ_Master1

        """
        results = self.ensure_results('sample_info')
        # results['aid_pool']
        # results['encoded_labels2d']
        # results['multihist']
        import ibeis
        infr = ibeis.AnnotInference.from_netx(results['graph'])
        info = ut.odict()
        info['n_names'] = infr.pos_graph.number_of_components(),
        info['n_aids'] = len(results['pblm_aids']),
        info['known_n_incomparable'] = infr.incomp_graph.number_of_edges()
        subtasks = results['subtasks']

        task = subtasks['match_state']
        flags = (task.encoded_df == task.class_names.tolist().index(INCMP))
        incomp_edges = task.encoded_df[flags.values].index.tolist()
        nid_edges = [infr.pos_graph.node_labels(*e) for e in incomp_edges]
        nid_edges = vt.ensure_shape(np.array(nid_edges), (None, 2))

        n_true = nid_edges.T[0] == nid_edges.T[1]
        info['incomp_info'] = {
            'inside_pcc': n_true.sum(),
            'betweeen_pcc': (~n_true).sum(),
        }

        for task_key, task in subtasks.items():
            info[task_key + '_hist'] = task.make_histogram()
        info_str = ut.repr4(info)
        fname = 'sample_info.txt'
        ut.write_to(join(str(self.dpath), fname), info_str)

    def write_importance(self, task_key):
        """
        python -m ibeis VerifierExpt.draw importance GZ_Master1,PZ_Master1 match_state

        python -m ibeis VerifierExpt.draw importance GZ_Master1 match_state
        python -m ibeis VerifierExpt.draw importance PZ_Master1 match_state

        python -m ibeis VerifierExpt.draw importance GZ_Master1 photobomb_state
        python -m ibeis VerifierExpt.draw importance PZ_Master1 photobomb_state
        """
        # Print info for latex table
        results = self.ensure_results('all')
        importances = results['importance'][task_key]
        vals = importances.values()
        items = importances.items()
        top_dims = ut.sortedby(items, vals)[::-1]
        lines = []
        num_top = 10
        for k, v in top_dims[:num_top]:
            k = k.replace('_', '\\_')
            lines.append('\\tt{{{}}} & ${:.4f}$ \\\\'.format(k, v))
        latex_str = '\n'.join(ut.align_lines(lines, '&'))

        fname = 'feat_importance_{}'.format(task_key)

        print('TOP {} importances for {}'.format(num_top, task_key))
        print('# of dimensions: %d' % (len(importances)))
        print(latex_str)
        print()
        extra_ = ut.codeblock(
            r'''
            \begin{{table}}[h]
                \centering
                \caption{{Top {}/{} dimensions for {}}}
                \begin{{tabular}}{{lr}}
                    \toprule
                    Dimension & Importance \\
                    \midrule
                    {}
                    \bottomrule
                \end{{tabular}}
            \end{{table}}
            '''
        ).format(num_top, len(importances), task_key.replace('_', '-'), latex_str)

        fpath = ut.render_latex(extra_, dpath=self.dpath, fname=fname)
        ut.write_to(join(str(self.dpath), fname + '.tex'), latex_str)
        return fpath

    def draw_prune(self):
        """
        CommandLine:
            python -m ibeis VerifierExpt.draw importance GZ_Master1

            python -m ibeis VerifierExpt.draw importance PZ_Master1 photobomb_state
            python -m ibeis VerifierExpt.draw importance PZ_Master1 match_state

            python -m ibeis VerifierExpt.draw prune GZ_Master1,PZ_Master1
            python -m ibeis VerifierExpt.draw prune PZ_Master1

        >>> from ibeis.scripts.thesis import *
        >>> self = VerifierExpt('PZ_Master1')
        >>> self = VerifierExpt('GZ_Master1')
        >>> self = VerifierExpt('PZ_MTEST')
        """

        task_key = 'match_state'
        expt_name = 'prune'
        results = self.ensure_results(expt_name)

        n_dims = results['n_dims']
        mdis_list = results['mdis_list']
        sub_reports = results['sub_reports']

        # mccs = [r['mcc'] for r in reports]
        # mccs2 = np.array([[r['mcc'] for r in rs] for rs in sub_reports])
        # pos_mccs = np.array([[r['metrics']['mcc'][POSTV] for r in rs]
        # for rs in sub_reports])
        ave_mccs = np.array([[r['metrics']['mcc']['ave/sum'] for r in rs]
                             for rs in sub_reports])

        import plottool as pt

        mpl.rcParams.update(TMP_RC)
        fig = pt.figure(fnum=1, doclf=True)
        pt.multi_plot(n_dims, {'mean': ave_mccs.mean(axis=1)},
                      rcParams=TMP_RC,
                      marker='',
                      force_xticks=[min(n_dims)],
                      # num_xticks=5,
                      ylabel='MCC',
                      xlabel='# feature dimensions',
                      ymin=.5,
                      ymax=1, xmin=1, xmax=n_dims[0], fnum=1, use_legend=False)
        ax = pt.gca()
        ax.invert_xaxis()
        fig.set_size_inches([W / 2, H])
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

        # Find the point at which accuracy starts to fall
        u = ave_mccs.mean(axis=1)
        # middle = ut.take_around_percentile(u, .5, len(n_dims) // 2.2)
        # thresh = middle.mean() - (middle.std() * 6)
        # print('thresh = %r' % (thresh,))
        # idx = np.where(u < thresh)[0][0]
        idx = u.argmax()

        fig = pt.figure(fnum=2)
        n_to_mid = ut.dzip(n_dims, mdis_list)
        pruned_importance = n_to_mid[n_dims[idx]]
        pt.wordcloud(pruned_importance, ax=fig.axes[0])
        fname = 'wc_{}_pruned.png'.format(task_key)
        fig_fpath = join(str(self.dpath), fname)
        vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

        vals = pruned_importance.values()
        items = pruned_importance.items()
        top_dims = ut.sortedby(items, vals)[::-1]
        lines = []
        num_top = 10
        for k, v in top_dims[:num_top]:
            k = k.replace('_', '\\_')
            lines.append('\\tt{{{}}} & ${:.4f}$ \\\\'.format(k, v))
        latex_str = '\n'.join(ut.align_lines(lines, '&'))

        increase = u[idx] - u[0]

        print(latex_str)
        print()
        extra_ = ut.codeblock(
            r'''
            \begin{{table}}[h]
                \centering
                \caption{{Pruned top {}/{} dimensions for {} increases MCC by {:.4f}}}
                \begin{{tabular}}{{lr}}
                    \toprule
                    Dimension & Importance \\
                    \midrule
                    {}
                    \bottomrule
                \end{{tabular}}
            \end{{table}}
            '''
        ).format(num_top, len(pruned_importance), task_key.replace('_', '-'),
                 increase, latex_str)
        # topinfo = vt.AnnotPairFeatInfo(list(pruned_importance.keys()))

        fname = 'pruned_feat_importance_{}'.format(task_key)
        fpath = ut.render_latex(extra_, dpath=self.dpath, fname=fname)
        ut.write_to(join(str(self.dpath), fname + '.tex'), latex_str)

        print(ut.repr4(ut.sort_dict(n_to_mid[n_dims[idx]], 'vals', reverse=True)))
        print(ut.repr4(ut.sort_dict(n_to_mid[n_dims[-1]], 'vals', reverse=True)))

    def measure_thresh(self, pblm):
        task_key = 'match_state'
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        infr = pblm.infr

        truth_colors = infr._get_truth_colors()

        cfms = res.confusions(POSTV)
        fig = pt.figure(fnum=1, doclf=True)  # NOQA
        ax = pt.gca()
        ax.plot(cfms.thresholds, cfms.n_fp, label='positive', color=truth_colors[POSTV])

        cfms = res.confusions(NEGTV)
        ax.plot(cfms.thresholds, cfms.n_fp, label='negative', color=truth_colors[NEGTV])

        # cfms = res.confusions(INCMP)
        # if len(cfms.thresholds) == 1:
        #     cfms.thresholds = [0, 1]
        #     cfms.n_fp = np.array(cfms.n_fp.tolist() * 2)
        # ax.plot(cfms.thresholds, cfms.n_fp, label='incomparable',
        #         color=pt.color_funcs.darken_rgb(truth_colors[INCMP], .15))
        ax.set_xlabel('thresholds')
        ax.set_ylabel('n_fp')

        ax.set_ylim(0, 20)
        ax.legend()

        cfms.plot_vs('fpr', 'thresholds')

    def _draw_score_hist(self, freqs, xlabel, fnum):
        """ helper """
        bins, freq0, freq1 = ut.take(freqs, ['bins', 'neg_freq', 'pos_freq'])
        width = np.diff(bins)[0]
        xlim = (bins[0] - (width / 2), bins[-1] + (width / 2))
        fig = pt.multi_plot(
            bins, (freq0, freq1), label_list=('negative', 'positive'),
            color_list=(pt.FALSE_RED, pt.TRUE_BLUE),
            kind='bar', width=width, alpha=.7, edgecolor='none',
            xlabel=xlabel, ylabel='frequency', fnum=fnum, pnum=(1, 1, 1),
            rcParams=TMP_RC, stacked=True,
            ytickformat='%.2f', xlim=xlim,
            # title='LNBNN positive separation'
        )
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9)
        fig.set_size_inches([W, H])
        return fig

    def draw_rerank(self):
        mpl.rcParams.update(TMP_RC)

        expt_name = 'rerank'
        results = self.ensure_results(expt_name)

        cdfs, infos = list(zip(*results))
        lnbnn_cdf = cdfs[0]
        clf_cdf = cdfs[1]
        fig = pt.figure(fnum=1)
        plot_cmcs([lnbnn_cdf, clf_cdf], ['ranking', 'rank+clf'], fnum=1)
        fig.set_size_inches([W, H * .6])
        qsizes = ut.take_column(infos, 'qsize')
        dsizes = ut.take_column(infos, 'dsize')
        assert ut.allsame(qsizes) and ut.allsame(dsizes)
        nonvaried_text = 'qsize={}, dsize={}'.format(qsizes[0], dsizes[0])
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())

        fpath = join(str(self.dpath), expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fpath)
        return fpath

    def draw_class_score_hist(self):
        """ Plots distribution of positive and negative scores """
        task_key = 'match_state'

        results = self.ensure_results('all')
        task_combo_res = results['task_combo_res']
        data_key = results['data_key']
        clf_key = results['clf_key']

        res = task_combo_res[task_key][clf_key][data_key]

        y = res.target_bin_df[POSTV]
        scores = res.probs_df[POSTV]
        bins = np.linspace(0, 1, 100)
        pos_freq = np.histogram(scores[y], bins)[0]
        neg_freq = np.histogram(scores[~y], bins)[0]
        pos_freq = pos_freq / pos_freq.sum()
        neg_freq = neg_freq / neg_freq.sum()

        score_hist_pos = {
            'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}

        lnbnn_xy = results['lnbnn_xy']
        scores = lnbnn_xy['score_lnbnn_1vM'].values
        y = lnbnn_xy[POSTV].values

        # Get 95% of the data at least
        maxbin = scores[scores.argsort()][-max(1, int(len(scores) * .05))]
        bins = np.linspace(0, max(maxbin, 10), 100)
        pos_freq = np.histogram(scores[y], bins)[0]
        neg_freq = np.histogram(scores[~y], bins)[0]
        pos_freq = pos_freq / pos_freq.sum()
        neg_freq = neg_freq / neg_freq.sum()
        score_hist_lnbnn = {
            'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}

        fig1 = self._draw_score_hist(score_hist_pos, 'positive probability', 1)
        fig2 = self._draw_score_hist(score_hist_lnbnn, 'LNBNN score', 2)

        fname = 'score_hist_pos_{}.png'.format(data_key)
        vt.imwrite(join(str(self.dpath), fname),
                   pt.render_figure_to_image(fig1, dpi=DPI))

        fname = 'score_hist_lnbnn.png'
        vt.imwrite(join(str(self.dpath), fname),
                   pt.render_figure_to_image(fig2, dpi=DPI))

    def draw_mcc_thresh(self, task_key):
        """
        python -m ibeis VerifierExpt.draw mcc_thresh GZ_Master1 match_state
        python -m ibeis VerifierExpt.draw mcc_thresh PZ_Master1 match_state

        python -m ibeis VerifierExpt.draw mcc_thresh GZ_Master1 photobomb_state
        python -m ibeis VerifierExpt.draw mcc_thresh PZ_Master1 photobomb_state

        """
        mpl.rcParams.update(TMP_RC)

        results = self.ensure_results('all')
        data_key = results['data_key']
        clf_key = results['clf_key']

        task_combo_res = results['task_combo_res']

        code_to_nice = self.task_nice_lookup[task_key]

        if task_key == 'photobomb_state':
            classes = ['pb']
        elif task_key == 'match_state':
            classes = [POSTV, NEGTV, INCMP]

        res = task_combo_res[task_key][clf_key][data_key]

        roc_curves = []

        for class_name in classes:
            c1 = res.confusions(class_name)
            if len(c1.thresholds) <= 2:
                continue
            class_nice = code_to_nice[class_name]
            idx = c1.mcc.argmax()
            t = c1.thresholds[idx]
            mcc = c1.mcc[idx]
            roc_curves += [
                {'label': class_nice + ', t={:.2f}, mcc={:.2f}'.format(t, mcc),
                 'thresh': c1.thresholds, 'mcc': c1.mcc},
            ]

        fig = pt.figure(fnum=1)  # NOQA
        ax = pt.gca()
        for data in roc_curves:
            ax.plot(data['thresh'], data['mcc'], label='%s' % (data['label']))
        ax.set_xlabel('threshold')
        ax.set_ylabel('MCC')
        # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
        ax.legend()
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9)
        fig.set_size_inches([W, H])

        fname = 'mcc_thresh_{}.png'.format(task_key)
        fig_fpath = join(str(self.dpath), fname)
        vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fig_fpath)

    def draw_roc(self, task_key):
        """
        python -m ibeis VerifierExpt.draw roc GZ_Master1 photobomb_state
        python -m ibeis VerifierExpt.draw roc GZ_Master1 match_state
        """
        mpl.rcParams.update(TMP_RC)

        results = self.ensure_results('all')
        data_key = results['data_key']
        clf_key = results['clf_key']

        task_combo_res = results['task_combo_res']
        lnbnn_xy = results['lnbnn_xy']

        if task_key == 'match_state':
            scores = lnbnn_xy['score_lnbnn_1vM'].values
            y = lnbnn_xy[POSTV].values
            # task_key = 'match_state'
            target_class = POSTV
            res = task_combo_res[task_key][clf_key][data_key]
            c2 = vt.ConfusionMetrics.from_scores_and_labels(scores, y)
            c3 = res.confusions(target_class)
            roc_curves = [
                {'label': 'LNBNN', 'fpr': c2.fpr, 'tpr': c2.tpr, 'auc': c2.auc},
                {'label': 'learned', 'fpr': c3.fpr, 'tpr': c3.tpr, 'auc': c3.auc},
            ]

            at_metric = 'tpr'
            for at_value in [.25, .5, .75]:
                info = ut.odict()
                for want_metric in ['fpr', 'n_false_pos', 'n_true_pos']:
                    key = '{}_@_{}={:.2f}'.format(want_metric, at_metric, at_value)
                    info[key] = c3.get_metric_at_metric(want_metric, at_metric, at_value)
                print(ut.repr4(info, align=True, precision=8))
        else:
            target_class = 'pb'
            res = task_combo_res[task_key][clf_key][data_key]
            c1 = res.confusions(target_class)
            roc_curves = [
                {'label': 'learned', 'fpr': c1.fpr, 'tpr': c1.tpr, 'auc': c1.auc},
            ]

        fig = pt.figure(fnum=1)  # NOQA
        ax = pt.gca()
        for data in roc_curves:
            ax.plot(data['fpr'], data['tpr'],
                    label='%s AUC=%.2f' % (data['label'], data['auc']))
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('true positive rate')
        # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
        ax.legend()
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9)
        fig.set_size_inches([W, H])

        fname = 'roc_{}.png'.format(task_key)
        fig_fpath = join(str(self.dpath), fname)
        vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def draw_wordcloud(self, task_key):
        import plottool as pt
        results = self.ensure_results('all')
        importances = results['importance'][task_key]

        fig = pt.figure(fnum=1)
        pt.wordcloud(importances, ax=fig.axes[0])

        fname = 'wc_{}.png'.format(task_key)
        fig_fpath = join(str(self.dpath), fname)
        vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

    @classmethod
    def draw_tagged_pair(VerifierExpt):
        import ibeis
        # ibs = ibeis.opendb(defaultdb='GZ_Master1')
        ibs = ibeis.opendb(defaultdb='PZ_Master1')

        query_tag = 'leftrightface'

        rowids = ibs._get_all_annotmatch_rowids()
        texts = ['' if t is None else t for t in ibs.get_annotmatch_tag_text(rowids)]
        tags = [[] if t is None else t.split(';') for t in texts]
        print(ut.repr4(ut.dict_hist(ut.flatten(tags))))

        flags = [query_tag in t.lower() for t in texts]
        filtered_rowids = ut.compress(rowids, flags)
        edges = ibs.get_annotmatch_aids(filtered_rowids)

        # The facematch leftright side example
        # edge = (5161, 5245)

        edge = edges[0]
        # for edge in ut.InteractiveIter(edges):
        infr = ibeis.AnnotInference(ibs=ibs, aids=edge, verbose=10)
        infr.reset_feedback('annotmatch', apply=True)
        match = infr._exec_pairwise_match([edge])[0]

        if False:
            # Fix the example tags
            infr.add_feedback(
                edge, 'match', tags=['facematch', 'leftrightface'],
                user_id='qt-hack', confidence='pretty_sure')
            infr.write_ibeis_staging_feedback()
            infr.write_ibeis_annotmatch_feedback()
            pass

        # THE DEPCACHE IS BROKEN FOR ANNOTMATCH APPARENTLY! >:(
        # Redo matches
        feat_keys = ['vecs', 'kpts', '_feats', 'flann']
        match.annot1._mutable = True
        match.annot2._mutable = True
        for key in feat_keys:
            if key in match.annot1:
                del match.annot1[key]
            if key in match.annot2:
                del match.annot2[key]
        match.apply_all({})

        fig = pt.figure(fnum=1, clf=True)
        ax = pt.gca()

        mpl.rcParams.update(TMP_RC)
        match.show(ax, vert=False,
                   heatmask=True,
                   show_lines=False,
                   show_ell=False,
                   show_ori=False,
                   show_eig=False,
                   # ell_alpha=.3,
                   modifysize=True)
        # ax.set_xlabel(xlabel)

        self = VerifierExpt()

        fname = 'custom_match_{}_{}_{}'.format(query_tag, *edge)
        dpath = ut.truepath(self.base_dpath)
        fpath = join(str(dpath), fname + '.jpg')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def custom_single_hard_case(self):
        """
        Example:
            >>> from ibeis.scripts.thesis import *
            >>> defaultdb = 'PZ_PB_RF_TRAIN'
            >>> #defaultdb = 'GZ_Master1'
            >>> defaultdb = 'PZ_MTEST'
            >>> self = VerifierExpt.collect(defaultdb)
            >>> self.dbname = 'PZ_PB_RF_TRAIN'
        """
        task_key = 'match_state'
        edge = (383, 503)
        for _case in self.hard_cases[task_key]:
            if _case['edge'] == edge:
                case = _case
                break

        import ibeis
        ibs = ibeis.opendb(self.dbname)

        from ibeis import core_annots
        config = {
            'augment_orientation': True,
            'ratio_thresh': .8,
        }
        config['checks'] = 80
        config['sver_xy_thresh'] = .02
        config['sver_ori_thresh'] = 3
        config['Knorm'] = 3
        config['symmetric'] = True
        config = ut.hashdict(config)

        aid1, aid2 = case['edge']
        real_name = case['real']
        pred_name = case['pred']
        match = case['match']
        code_to_nice = self.task_nice_lookup[task_key]
        real_nice, pred_nice = ut.take(code_to_nice,
                                       [real_name, pred_name])
        fname = 'fail_{}_{}_{}_{}'.format(real_nice, pred_nice, aid1, aid2)
        # Draw case
        probs = case['probs'].to_dict()
        order = list(code_to_nice.values())
        order = ut.setintersect(order, probs.keys())
        probs = ut.map_dict_keys(code_to_nice, probs)
        probstr = ut.repr2(probs, precision=2, strkeys=True, nobr=True,
                           key_order=order)
        xlabel = 'real={}, pred={},\n{}'.format(real_nice, pred_nice,
                                                probstr)

        match_list = ibs.depc.get('pairwise_match', ([aid1], [aid2]),
                                       'match', config=config)
        match = match_list[0]
        configured_lazy_annots = core_annots.make_configured_annots(
            ibs, [aid1], [aid2], config, config, preload=True)
        match.annot1 = configured_lazy_annots[config][aid1]
        match.annot2 = configured_lazy_annots[config][aid2]
        match.config = config

        fig = pt.figure(fnum=1, clf=True)
        ax = pt.gca()

        mpl.rcParams.update(TMP_RC)
        match.show(ax, vert=False,
                   heatmask=True,
                   show_lines=False,
                   show_ell=False,
                   show_ori=False,
                   show_eig=False,
                   # ell_alpha=.3,
                   modifysize=True)
        ax.set_xlabel(xlabel)

        subdir = 'cases_{}'.format(task_key)
        dpath = join(str(self.dpath), subdir)
        fpath = join(str(dpath), fname + '_custom.jpg')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))


def prepare_cdfs(cdfs, labels):
    cdfs = vt.pad_vstack(cdfs, fill_value=1)
    # Sort so the best is on top
    sortx = np.lexsort(cdfs.T[::-1])[::-1]
    cdfs = cdfs[sortx]
    labels = ut.take(labels, sortx)
    return cdfs, labels


def plot_cmcs(cdfs, labels, fnum=1, pnum=(1, 1, 1), ymin=.4):
    cdfs, labels = prepare_cdfs(cdfs, labels)
    # Truncte to 20 ranks
    num_ranks = min(cdfs.shape[-1], 20)
    xdata = np.arange(1, num_ranks + 1)
    cdfs_trunc = cdfs[:, 0:num_ranks]
    label_list = ['%6.2f%% - %s' % (cdf[0] * 100, lbl)
                  for cdf, lbl in zip(cdfs_trunc, labels)]

    # ymin = .4
    num_yticks = (10 - int(ymin * 10)) + 1

    pt.multi_plot(
        xdata, cdfs_trunc, label_list=label_list,
        xlabel='rank', ylabel='match probability',
        use_legend=True, legend_loc='lower right', num_yticks=num_yticks,
        ymax=1, ymin=ymin, ypad=.005, xmin=.9, num_xticks=5,
        xmax=num_ranks + 1 - .5,
        pnum=pnum, fnum=fnum,
        rcParams=TMP_RC,
    )
    return pt.gcf()
