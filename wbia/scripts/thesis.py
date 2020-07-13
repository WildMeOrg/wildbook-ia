# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
from wbia.algo.verif import vsone
from wbia.scripts._thesis_helpers import DBInputs
from wbia.scripts._thesis_helpers import Tabular, upper_one, ave_str
from wbia.scripts._thesis_helpers import TMP_RC, W, H, DPI
import wbia.constants as const
from wbia.algo.graph import nx_utils as nxu
import ubelt as ub
import pandas as pd
import numpy as np
from os.path import basename, join, splitext, exists  # NOQA
import utool as ut
import wbia.plottool as pt
import vtool as vt
import pathlib
import matplotlib as mpl
import random
import sys
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA

(print, rrr, profile) = ut.inject2(__name__)


@ut.reloadable_class
class Chap5(DBInputs):
    """
    python -m wbia Chap5.measure all GZ_Master1
    python -m wbia Chap5.measure all PZ_Master1
    python -m wbia Chap5.draw all GZ_Master1
    python -m wbia Chap5.draw all PZ_Master1 --comp Leviathan

    python -m wbia Chap5.draw error_graph_analysis GZ_Master1
    python -m wbia Chap5.draw error_graph_analysis PZ_Master1 --comp Leviathan


    """

    base_dpath = ut.truepath('~/latex/crall-thesis-2017/figures5')

    def measure_all(self):
        self.measure_dbstats()
        self.measure_simulation()

    def draw_all(self):
        r"""
        CommandLine:
            python -m wbia Chap5.draw all GZ_Master1
            python -m wbia Chap5.draw error_graph_analysis GZ_Master1

            python -m wbia Chap5.draw all PZ_Master1
            python -m wbia Chap5.draw error_graph_analysis PZ_Master1

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap4('GZ_Master1')
        """
        self.ensure_results('simulation')
        self.draw_simulation()
        self.draw_refresh()
        self.write_dbstats()
        self.write_error_tables()
        # python -m wbia Chap5.draw error_graph_analysis GZ_Master1

    def _precollect(self):
        if self.ibs is None:
            _Chap5 = ut.fix_super_reload(Chap5, self)
            super(_Chap5, self)._precollect()

        # Split data into a training and testing test
        ibs = self.ibs
        annots = ibs.annots(self.aids_pool)
        names = list(annots.group_items(annots.nids).values())
        ut.shuffle(names, rng=321)
        train_names, test_names = names[0::2], names[1::2]
        train_aids, test_aids = map(ut.flatten, (train_names, test_names))

        self.test_train = train_aids, test_aids

        params = {}
        if ibs.dbname == 'PZ_MTEST':
            params['sample_method'] = 'random'

        self.pblm = vsone.OneVsOneProblem.from_aids(ibs, train_aids, **params)

        # ut.get_nonconflicting_path(dpath, suffix='_old')
        self.const_dials = {
            # 'oracle_accuracy' : (0.98, 1.0),
            # 'oracle_accuracy' : (0.98, .98),
            'oracle_accuracy': (0.99, 0.99),
            'k_redun': 2,
            'max_outer_loops': np.inf,
            # 'max_outer_loops' : 1,
        }

        if ibs.dbname == 'GZ_Master1':
            self.thresh_targets = {
                'graph': ('fpr', 0.0014),
                'rankclf': ('fpr', 0.001),
            }
        elif ibs.dbname == 'PZ_Master1':
            self.thresh_targets = {
                # 'graph': ('fpr', .03),
                # 'rankclf': ('fpr', .01),
                'graph': ('fpr', 0.0014),
                'rankclf': ('fpr', 0.001),
            }
        else:
            self.thresh_targets = {
                'graph': ('fpr', 0.002),
                'rankclf': ('fpr', 0),
            }

        config = ut.dict_union(self.const_dials, self.thresh_targets)
        cfg_prefix = '{}_{}'.format(len(test_aids), len(train_aids))
        self._setup_links(cfg_prefix, config)

    def _setup(self):
        """
        python -m wbia Chap5._setup

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> #self = Chap5('GZ_Master1')
            >>> self = Chap5('PZ_Master1')
            >>> #self = Chap5('PZ_MTEST')
            >>> self._setup()
        """
        self._precollect()
        train_aids, test_aids = self.test_train

        task_key = 'match_state'
        pblm = self.pblm
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key

        pblm.eval_data_keys = [data_key]
        pblm.setup(with_simple=False)
        pblm.learn_evaluation_classifiers()

        res = pblm.task_combo_res[task_key][clf_key][data_key]
        # pblm.report_evaluation()

        # if False:
        #     pblm.learn_evaluation_classifiers(task_keys=['photobomb_state'])
        #     pb_res = pblm.task_combo_res['photobomb_state'][clf_key][data_key]
        #     pb_res  # TODO?

        if True:
            # Remove results that are photobombs for now
            # res = pblm.task_combo_res['photobomb_state'][clf_key][data_key]
            pb_task = pblm.samples.subtasks['photobomb_state']
            import utool

            with utool.embed_on_exception_context:
                flags = pb_task.indicator_df.loc[res.index]['notpb'].values
                notpb_res = res.compress(flags)
                res = notpb_res

        # TODO: need more principled way of selecting thresholds
        graph_thresh = res.get_pos_threshes(*self.thresh_targets['graph'])
        rankclf_thresh = res.get_pos_threshes(*self.thresh_targets['rankclf'])

        print('\n--- Graph thresholds ---')
        graph_report = res.report_auto_thresholds(graph_thresh, verbose=0)

        print('\n --- Ranking thresholds ---')
        rankclf_report = res.report_auto_thresholds(rankclf_thresh, verbose=0)

        ut.writeto(
            join(self.dpath, 'thresh_reports.txt'),
            '\n'.join(
                [
                    '============',
                    'Graph report',
                    '------------',
                    graph_report,
                    '',
                    '============',
                    'Rank CLF report',
                    '------------',
                    rankclf_report,
                ]
            ),
        )

        # Load or create the deploy classifiers
        clf_dpath = ut.ensuredir((self.dpath, 'clf'))
        classifiers = pblm.ensure_deploy_classifiers(dpath=clf_dpath)

        sim_params = {
            'test_aids': test_aids,
            'train_aids': train_aids,
            'classifiers': classifiers,
            'graph_thresh': graph_thresh,
            'rankclf_thresh': rankclf_thresh,
            'const_dials': self.const_dials,
        }
        self.pblm = pblm
        self.sim_params = sim_params
        return sim_params

    def _thresh_test(self):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> self = Chap5('PZ_Master1')
            >>> self = Chap5('GZ_Master1')
        """
        import wbia

        self.ensure_setup()

        task_key = 'match_state'
        pblm = self.pblm
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key
        res = pblm.task_combo_res[task_key][clf_key][data_key]
        pblm.report_evaluation()

        if True:
            # Remove results that are photobombs for now
            # res = pblm.task_combo_res['photobomb_state'][clf_key][data_key]
            pb_task = pblm.samples.subtasks['photobomb_state']
            import utool

            with utool.embed_on_exception_context:
                flags = pb_task.indicator_df.loc[res.index]['notpb'].values
                notpb_res = res.compress(flags)
                res = notpb_res

        """
        PLAN:
            Draw an LNBNN sample.
            Estimate probabilities on sample.

            for each fpr on validation,
                find threshold
                find fpr at that threshold for the lnbnn sample

            plot the predicted fpr vs the true fpr to show that this is
            difficult to predict.
        """

        ibs = self.ibs
        sim_params = self.sim_params
        classifiers = sim_params['classifiers']
        test_aids = sim_params['test_aids']
        const_dials = sim_params['const_dials']

        graph_thresh = sim_params['graph_thresh']

        verbose = 1

        # ----------
        # Graph test
        dials1 = ut.dict_union(
            const_dials,
            {
                'name': 'graph',
                'enable_inference': True,
                'match_state_thresh': graph_thresh,
            },
        )

        infr1 = wbia.AnnotInference(
            ibs=ibs, aids=test_aids, autoinit=True, verbose=verbose
        )

        estimate = []
        thresh = []
        confusions = []

        target_values = [0, 0.001, 0.0012, 0.0014, 0.0016, 0.002]
        for value in target_values:
            match_thresh = res.get_pos_threshes('fpr', value=value)
            estimate.append(value)
            thresh.append(match_thresh)

            infr1.enable_auto_prioritize_nonpos = True
            infr1._refresh_params['window'] = 20
            infr1._refresh_params['thresh'] = np.exp(-2)
            infr1._refresh_params['patience'] = 20
            infr1.init_simulation(classifiers=classifiers, **dials1)
            infr1.init_test_mode()
            infr1.reset(state='empty')
            infr1.task_thresh['match_state'] = match_thresh
            infr1.enable_fixredun = False
            infr1.main_loop(max_loops=1)

            # for e in infr1.edges():
            #     decision = infr1.get_edge_data(e).get('decision', UNREV)
            #     truth = infr1.get_edge_data(e).get('truth', None)
            c = pd.DataFrame(infr1.test_state['confusion'])
            confusions.append(c)

        actual_fpr = []
        actual_nums = []
        for c, est, t in zip(estimate, confusions, thresh):
            print(t)
            print(est)
            print(c)
            # n_total = c.sum().sum()
            # n_true = np.diag(c).sum()
            # n_false = n_total - n_true
            # tpa = n_true / n_total
            # fpr = n_false / n_total
            # pos_fpr = (c.loc[POSTV].sum() - c.loc[POSTV][POSTV]) / c.loc[POSTV].sum()
            N = c.sum(axis=0)
            TP = pd.Series(np.diag(c), index=c.index)  # NOQA
            FP = (c - np.diagflat(np.diag(c))).sum(axis=0)
            fpr = FP / N
            # tpas.append(tpa)
            actual_fpr.append(fpr)
            actual_nums.append(N)

        for class_name in [NEGTV, POSTV]:
            fnum = 1
            x = [t[class_name] for t in thresh]
            # class_nums = np.array([n[class_name] for n in actual_nums])
            class_actual = np.array([a[class_name] for a in actual_fpr])
            class_est = target_values

            pt.figure(fnum=fnum)
            pt.plot(x, class_est, 'x--', label='est ' + class_name)
            pt.plot(x, class_actual, 'o--', label='actual ' + class_name)
            pt.legend()

    @profile
    def measure_simulation(self):
        """
        CommandLine:
            python -m wbia Chap5.measure simulation GZ_Master1
            python -m wbia Chap5.measure simulation PZ_Master1

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap5('GZ_Master1')
        """
        import wbia

        self.ensure_setup()

        ibs = self.ibs
        sim_params = self.sim_params
        classifiers = sim_params['classifiers']
        test_aids = sim_params['test_aids']

        rankclf_thresh = sim_params['rankclf_thresh']
        graph_thresh = sim_params['graph_thresh']

        const_dials = sim_params['const_dials']

        sim_results = {}
        verbose = 1

        # ----------
        # Graph test
        dials1 = ut.dict_union(
            const_dials,
            {
                'name': 'graph',
                'enable_inference': True,
                'match_state_thresh': graph_thresh,
            },
        )
        infr1 = wbia.AnnotInference(
            ibs=ibs, aids=test_aids, autoinit=True, verbose=verbose
        )
        infr1.enable_auto_prioritize_nonpos = True
        infr1._refresh_params['window'] = 20
        infr1._refresh_params['thresh'] = np.exp(-2)
        infr1._refresh_params['patience'] = 20

        infr1.init_simulation(classifiers=classifiers, **dials1)
        infr1.init_test_mode()

        infr1.reset(state='empty')
        infr1.main_loop()

        sim_results['graph'] = self._collect_sim_results(infr1, dials1)

        # --------
        # Rank+CLF
        dials2 = ut.dict_union(
            const_dials,
            {
                'name': 'rank+clf',
                'enable_inference': False,
                'match_state_thresh': rankclf_thresh,
            },
        )
        infr2 = wbia.AnnotInference(
            ibs=ibs, aids=test_aids, autoinit=True, verbose=verbose
        )
        infr2.init_simulation(classifiers=classifiers, **dials2)
        infr2.init_test_mode()
        infr2.enable_redundancy = False
        infr2.enable_autoreview = True
        infr2.reset(state='empty')

        infr2.main_loop(max_loops=1, use_refresh=False)

        sim_results['rank+clf'] = self._collect_sim_results(infr2, dials2)

        # ------------
        # Ranking test
        dials3 = ut.dict_union(
            const_dials,
            {'name': 'ranking', 'enable_inference': False, 'match_state_thresh': None},
        )
        infr3 = wbia.AnnotInference(
            ibs=ibs, aids=test_aids, autoinit=True, verbose=verbose
        )
        infr3.init_simulation(classifiers=None, **dials3)
        infr3.init_test_mode()
        infr3.enable_redundancy = False
        infr3.enable_autoreview = False
        infr3.reset(state='empty')

        infr3.main_loop(max_loops=1, use_refresh=False)

        sim_results['ranking'] = self._collect_sim_results(infr3, dials3)

        # ------------
        # Dump experiment output to disk
        expt_name = 'simulation'
        self.expt_results[expt_name] = sim_results
        ut.ensuredir(self.dpath)
        ut.save_data(join(self.dpath, expt_name + '.pkl'), sim_results)

        # metrics_df = pd.DataFrame.from_dict(graph_expt_data['metrics'])
        # for user, group in metrics_df.groupby('user_id'):
        #     print('actions of user = %r' % (user,))
        #     user_actions = group['test_action']
        #     print(ut.repr4(ut.dict_hist(user_actions), stritems=True))

        # self.draw_simulation()
        # ut.show_if_requested()
        pass

    def _collect_sim_results(self, infr, dials):
        pred_confusion = pd.DataFrame(infr.test_state['confusion'])
        pred_confusion.index.name = 'real'
        pred_confusion.columns.name = 'pred'
        print('Edge confusion')
        print(pred_confusion)

        expt_data = {
            'real_ccs': list(infr.nid_to_gt_cc.values()),
            'pred_ccs': list(infr.pos_graph.connected_components()),
            'graph': infr.graph.copy(),
            'dials': dials,
            'refresh_thresh': infr.refresh._prob_any_remain_thresh,
            'metrics': infr.metrics_list,
        }
        return expt_data

    @profile
    def measure_dbstats(self):
        """
        python -m wbia Chap5.draw dbstats GZ_Master1

        python -m wbia Chap5.measure dbstats PZ_Master1
        python -m wbia Chap5.draw dbstats PZ_Master1

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap5('GZ_Master1')
        """
        self.ensure_setup()
        classifiers = self.sim_params['classifiers']
        clf_meta = classifiers['match_state']['metadata'].copy()
        clf_meta.pop('data_info')

        def ibs_stats(aids):
            pccs = self.ibs.group_annots_by_name(aids)[0]
            nper_annot = ut.emap(len, pccs)
            return {
                'n_annots': len(aids),
                'n_names': len(pccs),
                'annot_size_mean': np.mean(nper_annot),
                'annot_size_std': np.std(nper_annot),
            }

        train_aids = self.sim_params['train_aids']
        test_aids = self.sim_params['test_aids']
        dbstats = {
            'testing': ibs_stats(test_aids),
            'training': ibs_stats(train_aids),
        }
        traininfo = dbstats['training']
        traininfo['class_hist'] = clf_meta['class_hist']
        traininfo['n_training_pairs'] = sum(clf_meta['class_hist'].values())

        infr = self.pblm.infr
        pblm_pccs = list(self.pblm.infr.positive_components())
        pblm_nper_annot = ut.emap(len, pblm_pccs)
        traininfo['pblm_info'] = {
            'n_annots': infr.graph.number_of_nodes(),
            'n_names': len(pblm_pccs),
            'annot_size_mean': np.mean(pblm_nper_annot),
            'annot_size_std': np.std(pblm_nper_annot),
            'notes': ut.textblock(
                """
                if this (the real training data) is different from the parents
                (wbia) info, that means the staging database is ahead of
                annotmatch. Report the wbia one for clarity. Num annots should
                always be the same though.
                """
            ),
        }

        expt_name = 'dbstats'
        self.expt_results[expt_name] = dbstats
        ut.save_data(join(self.dpath, expt_name + '.pkl'), dbstats)

    def write_dbstats(self):
        """
        # TODO: write info about what dataset was used

        CommandLine:
            python -m wbia Chap5.measure dbstats PZ_Master1
            python -m wbia Chap5.measure dbstats PZ_Master1

            python -m wbia Chap5.measure simulation GZ_Master1
            python -m wbia Chap5.draw dbstats --db GZ_Master1 --diskshow

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap5('GZ_Master1')
        """
        dbstats = self.ensure_results('dbstats')

        d = ut.odict()
        keys = ['training', 'testing']
        for k in keys:
            v = dbstats[k]
            k = k.capitalize()
            size_str = ave_str(v['annot_size_mean'], v['annot_size_std'])
            r = d[k] = ut.odict()
            r['Names'] = v['n_names']
            r['Annots'] = v['n_annots']
            r['Annots size'] = size_str
            r['Training edges'] = v.get('n_training_pairs', '-')
        df = pd.DataFrame.from_dict(d, orient='index').loc[list(d.keys())]
        tabular = Tabular(df)
        tabular.colfmt = 'numeric'
        tabular.caption = self.species_nice.capitalize()
        print(tabular.as_table())
        print(tabular.as_tabular())

        ut.writeto(join(self.dpath, 'dbstats.tex'), tabular.as_tabular())
        fpath = ut.render_latex(tabular.as_table(), dpath=self.dpath, fname='dbstats')
        return fpath

    def print_error_analysis(self):
        """
        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap5('GZ_Master1')
            >>> self = Chap5('PZ_Master1')
        """
        sim_results = self.ensure_results('simulation')

        key = 'graph'
        real_ccs = sim_results[key]['real_ccs']
        pred_ccs = sim_results[key]['pred_ccs']

        delta = ut.grouping_delta(pred_ccs, real_ccs, pure=False)
        splits = delta['splits']
        merges = delta['merges']

        graph = sim_results[key]['graph']
        ignore = [
            'timestamp',
            'num_reviews',
            'confidence',
            'default_priority',
            'review_id',
        ]

        print('\nsplits = ' + ut.repr4(splits))
        print('\nmerges = ' + ut.repr4(merges))

        def print_edge_df(df, parts):
            if len(df):
                order = ['truth', 'decision', 'tags', 'prob_match']
                order = df.columns.intersection(order)
                neworder = ut.partial_order(df.columns, order)
                df = df.reindex(neworder, axis=1)

                df_str = df.to_string()
                cols = ['blue', 'red', 'green', 'teal']
                df_str = ut.highlight_multi_regex(
                    df_str,
                    {
                        ut.regex_or(ut.regex_word(str(a)) for a in part): col
                        for part, col in zip(parts, cols)
                    },
                )
                print(df_str)
            else:
                print(df)

        for parts in merges:
            print('\n\n')
            print('Merge Row: ' + ut.repr2(parts))
            sub = graph.subgraph(ut.flatten(parts))
            df = nxu.edge_df(graph, sub.edges(), ignore=ignore)
            print_edge_df(df, parts)
        for parts in splits:
            print('\n\n')
            print('Split Row: ' + ut.repr2(parts))
            sub = graph.subgraph(ut.flatten(parts))
            df = nxu.edge_df(graph, sub.edges(), ignore=ignore)
            print_edge_df(df, parts)

    def draw_error_graph_analysis(self):
        """
        CommandLine:
            python -m wbia Chap5.draw error_graph_analysis GZ_Master1
            python -m wbia Chap5.draw error_graph_analysis PZ_Master1

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap5('GZ_Master1')
            >>> self = Chap5('PZ_Master1')
        """
        import wbia
        import wbia.plottool as pt

        sim_results = self.ensure_results('simulation')
        key = 'graph'

        ignore = [
            'timestamp',
            'num_reviews',
            'default_priority',
            'confidence',
            'review_id',
        ]

        task_keys = [
            'match_state',
            'photobomb_state',
        ]
        task_nice_lookup = {
            'match_state': const.EVIDENCE_DECISION.CODE_TO_NICE,
            'photobomb_state': {'pb': 'Photobomb', 'notpb': 'Not Photobomb'},
        }

        mpl.rcParams.update(TMP_RC)

        # Load simulation end state with predicted and real PCCs
        real_ccs = sim_results[key]['real_ccs']
        pred_ccs = sim_results[key]['pred_ccs']
        graph = sim_results[key]['graph']

        # Manage data using a read-only inference object
        ibs = wbia.opendb(db=self.dbname)
        infr = wbia.AnnotInference.from_netx(graph, ibs=ibs)
        infr.readonly = True
        infr._viz_image_config['thumbsize'] = 700
        infr._viz_image_config['grow'] = True
        infr.load_latest_classifiers(join(self.dpath, 'clf'))
        infr.relabel_using_reviews(rectify=False)

        # For each node, mark its real and predicted ids
        infr.set_node_attrs(
            'real_id', {aid: nid for nid, cc in enumerate(real_ccs) for aid in cc}
        )
        infr.set_node_attrs(
            'pred_id', {aid: nid for nid, cc in enumerate(pred_ccs) for aid in cc}
        )

        # from networkx.utils import arbitrary_element as arbitrary

        # Gather a sample of error groups
        n = 20
        delta = ut.grouping_delta(pred_ccs, real_ccs, pure=False)
        sampled_errors = ut.odict(
            [
                ('merge', ut.strided_sample(delta['merges'], n)),
                ('split', ut.strided_sample(delta['splits'], n)),
            ]
        )

        for k, v in sampled_errors.items():
            print('Sampled {} {} cases'.format(len(v), k))

        err_items = []
        for case_type, cases in sampled_errors.items():
            for case in cases:
                case_aids = set(ut.flatten(case))
                # For each case find what edges need fixing
                if case_type == 'merge':
                    error_edges = infr.find_pos_augment_edges(case_aids, k=1)
                else:
                    edges = list(nxu.edges_between(graph, case_aids))
                    _df = infr.get_edge_dataframe(edges)
                    flags = (_df.truth != _df.decision) & (_df.truth == NEGTV)
                    error_edges = _df.index[flags].tolist()
                for edge in error_edges:
                    edge = infr.e_(*edge)
                    err_items.append((case_type, case, error_edges, edge))
        err_items_df = pd.DataFrame(
            err_items, columns=['case_type', 'case', 'error_edges', 'edge']
        )

        edges = err_items_df['edge'].tolist()
        err_df = infr.get_edge_dataframe(edges)
        err_df = err_df.drop(err_df.columns.intersection(ignore), axis=1)
        # Lookup the probs for each state
        task_probs = infr._make_task_probs(edges)
        probs_df = pd.concat(task_probs, axis=1)  # NOQA

        dpath = ut.ensuredir((self.dpath, 'errors'))
        fnum = 1

        fig = pt.figure(fnum=fnum, pnum=(2, 1, 2))
        ax = pt.gca()
        pt.adjust_subplots(
            top=1, right=1, left=0, bottom=0.15, hspace=0.01, wspace=0, fig=fig
        )

        subitems = err_items_df
        # subitems = err_items_df[err_items_df.case_type == 'merge'].iloc[-2:]
        for _, (case_type, case, error_edges, edge) in subitems.iterrows():
            aids = ut.total_flatten(case)

            if case_type == 'split':
                colorby = 'real_id'
            if case_type == 'merge':
                colorby = 'pred_id'

            infr.show_error_case(aids, edge, error_edges, colorby=colorby)

            edge_info = err_df.loc[edge].to_dict()

            xlabel = case_type.capitalize() + ' case. '
            code_to_nice = task_nice_lookup['match_state']
            real_code = infr.match_state_gt(edge)
            pred_code = edge_info['decision']

            real_nice = 'real={}'.format(code_to_nice[real_code])

            if edge_info['user_id'] == 'auto_clf':
                xlabel += 'Reviewed automatically'
            elif edge_info['user_id'] == 'oracle':
                xlabel += 'Reviewed manually'
            else:
                if pred_code is None:
                    xlabel += 'Edge did not appear in candidate set'
                else:
                    xlabel += 'Edge was a candidate, but not reviewed'

            if pred_code is None:
                pred_nice = 'pred=None'
            else:
                pred_nice = 'pred={}'.format(code_to_nice[pred_code])
            xlabel += '\n{}, {}'.format(real_nice, pred_nice)

            for task_key in task_keys:
                tprobs = task_probs[task_key]
                _probs = tprobs.loc[edge].to_dict()
                code_to_nice = task_nice_lookup[task_key]
                probs = ut.odict(
                    (v, _probs[k]) for k, v in code_to_nice.items() if k in _probs
                )
                probstr = ut.repr2(probs, precision=2, strkeys=True, nobr=True)
                xlabel += '\n' + probstr
            xlabel = xlabel.lstrip('\n')

            fig = pt.gcf()
            ax = fig.axes[0]
            ax.set_xlabel(xlabel)
            fig.set_size_inches([W, H * 2])

            parts = [ut.repr2(sorted(p), itemsep='', nobr=True) for p in case]
            case_id = ','.join(list(map(str, map(len, parts))))
            case_id += '_' + ut.hash_data('-'.join(parts))[0:8]
            eid = '{},{}'.format(*edge)
            fname = case_type + '_' + case_id + '_edge' + eid + '.png'
            fpath = join(dpath, fname)
            vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def write_error_tables(self):
        """
        CommandLine:
            python -m wbia Chap5.draw error_tables PZ_Master1
            python -m wbia Chap5.draw error_tables GZ_Master1

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> from wbia.scripts.thesis import _ranking_hist, _ranking_cdf
            >>> self = Chap5('GZ_Master1')
        """
        sim_results = self.ensure_results('simulation')
        keys = ['ranking', 'rank+clf', 'graph']
        infos = {}
        for key in keys:
            # print('!!!!!!!!!!!!')
            # print('key = %r' % (key,))
            expt_data = sim_results[key]
            info = self._get_error_sizes(expt_data, allow_hist=False)
            info['correct']['n_pred_pccs'] = '-'
            info['correct']['size_pred_pccs'] = '-'
            infos[key] = info

        dfs = {}
        with_aves = 0
        for key in keys:
            info = infos[key]

            table = ut.odict()
            types = ['correct', 'split', 'merge']
            for t in types:
                caseinfo = info[t]
                casetable = ut.odict()
                casetable['pred PCCs'] = caseinfo['n_pred_pccs']
                casetable['pred PCC size'] = caseinfo['size_pred_pccs']
                casetable['real PCCs'] = caseinfo['n_real_pccs']
                casetable['real PCC size'] = caseinfo['size_real_pccs']
                if with_aves:
                    casetable['small size'] = caseinfo.get('ave_small', '-')
                    casetable['large size'] = caseinfo.get('ave_large', '-')
                table[t] = ut.map_keys(upper_one, casetable)

            df = pd.DataFrame.from_dict(table, orient='index')
            df = df.loc[list(table.keys())]
            dfs[key] = df

        df = pd.concat(ut.take(dfs, keys), axis=0, keys=keys)
        tabular = Tabular(df, index=True, escape=True, colfmt='numeric')
        error_size_text = tabular.as_tabular()
        print(error_size_text)

        # Inspect error sizes only for the graph
        caseinfo = infos['graph']
        table = ut.odict()
        types = ['split', 'merge']
        for t in types:
            caseinfo = info[t]
            casetable = ut.odict()
            casetable['error groups'] = caseinfo.get('n_errgroups', '-')
            casetable['group size'] = caseinfo.get('errgroup_size', '-')
            casetable['small PCC size'] = caseinfo.get('ave_small', '-')
            casetable['large PCC size'] = caseinfo.get('ave_large', '-')
            casetable = ut.map_keys(upper_one, casetable)
            table[t] = ut.map_keys(upper_one, casetable)
        df = pd.DataFrame.from_dict(table, orient='index')
        df = df.loc[list(table.keys())]

        tabular = Tabular(df, index=True, escape=True, colfmt='numeric')
        error_group_text = tabular.as_tabular()
        print(error_group_text)

        fname = 'error_size_details'
        ut.write_to(join(self.dpath, fname + '.tex'), error_size_text)
        ut.render_latex(
            error_size_text, self.dpath, fname, preamb_extra=['\\usepackage{makecell}']
        )

        fname = 'error_group_details'
        ut.write_to(join(self.dpath, fname + '.tex'), error_group_text)
        ut.render_latex(
            error_group_text, self.dpath, fname, preamb_extra=['\\usepackage{makecell}']
        )

    def _get_error_sizes(self, expt_data, allow_hist=False):
        real_ccs = expt_data['real_ccs']
        pred_ccs = expt_data['pred_ccs']
        graph = expt_data['graph']
        # delta_df = ut.grouping_delta_stats(pred_ccs, real_ccs)
        # print(delta_df)

        delta = ut.grouping_delta(pred_ccs, real_ccs)
        unchanged = delta['unchanged']
        splits = delta['splits']['new']
        merges = delta['merges']['old']

        # hybrids can be done by first splitting and then merging
        hybrid_splits = delta['hybrid']['splits']
        hybrid_merges = delta['hybrid']['merges']

        all_merges = merges + hybrid_merges
        all_splits = splits + hybrid_splits

        def ave_size(sets):
            lens = list(map(len, sets))
            hist = ut.dict_hist(lens)
            if allow_hist and len(hist) <= 2:
                return ut.repr4(hist, nl=0)
            else:
                mu = np.mean(lens)
                sigma = np.std(lens)
                return ave_str(mu, sigma, precision=1)

        def unchanged_measures(unchanged):
            pred = true = unchanged
            unchanged_info = ut.odict(
                [
                    ('n_pred_pccs', len(pred)),
                    ('size_pred_pccs', ave_size(pred)),
                    ('n_real_pccs', len(true)),
                    ('size_real_pccs', ave_size(true)),
                ]
            )
            return unchanged_info

        def get_bad_edges(ccs, bad_decision, ret_ccs=False):
            for cc1, cc2 in ut.combinations(ccs, 2):
                cc1 = frozenset(cc1)
                cc2 = frozenset(cc2)
                bad_edges = []
                cross = nxu.edges_cross(graph, cc1, cc2)
                for edge in cross:
                    d = graph.get_edge_data(*edge)
                    if d['decision'] == bad_decision:
                        if ret_ccs:
                            bad_edges.append((cc1, cc2, edge))
                        else:
                            bad_edges.append(edge)
                yield bad_edges

        def split_measures(splits):
            # Filter out non-split hybrids
            splits = [s for s in splits if len(s) > 1]
            pred = ut.lmap(ut.flatten, splits)
            true = ut.flatten(splits)
            baddies = []
            smalls = []
            larges = []
            for split in splits:
                split = ut.sortedby(split, ut.lmap(len, split))
                smalls.append((split[0]))
                larges.append((split[1]))
                b = list(get_bad_edges(split, POSTV))
                baddies.append(b)

            split_info = ut.odict(
                [
                    ('n_pred_pccs', len(pred)),
                    ('size_pred_pccs', ave_size(pred)),
                    ('n_real_pccs', len(true)),
                    ('size_real_pccs', ave_size(true)),
                    ('n_errgroups', len(splits)),
                    ('errgroup_size', ave_size(splits)),
                    ('ave_small', ave_size(smalls)),
                    ('ave_large', ave_size(larges)),
                ]
            )
            return split_info

        def merge_measures(merges):
            # Filter out non-merge hybrids
            merges = [s for s in merges if len(s) > 1]
            true = ut.lmap(ut.flatten, merges)
            pred = ut.flatten(merges)
            baddies = []
            n_neg_redun = 0
            n_bad_pairs = 0
            n_bad_pccs = 0
            smalls = []
            larges = []
            for merge in merges:
                merge = ut.sortedby(merge, ut.lmap(len, merge))

                smalls.append((merge[0]))
                larges.append((merge[1]))

                b = list(get_bad_edges(merge, NEGTV))
                b2 = list(get_bad_edges(merge, NEGTV, ret_ccs=True))

                baddies.append(b)
                bad_neg_redun = max(map(len, b))
                n_bad_pairs += sum(map(any, b))
                n_bad_pccs += len(set(ut.flatten(ut.take_column(ut.flatten(b2), [0, 1]))))
                if bad_neg_redun >= 2:
                    n_neg_redun += 1

            merge_info = ut.odict(
                [
                    ('n_pred_pccs', len(pred)),
                    ('size_pred_pccs', ave_size(pred)),
                    ('n_real_pccs', len(true)),
                    ('size_real_pccs', ave_size(true)),
                    ('n_errgroups', len(merges)),
                    ('errgroup_size', ave_size(merges)),
                    ('ave_incon_edges', ave_size(ut.lmap(ut.flatten, baddies))),
                    ('n_bad_pairs', n_bad_pairs),
                    ('n_bad_pccs', n_bad_pccs),
                    ('n_neg_redun', n_neg_redun),
                    ('ave_small', ave_size(smalls)),
                    ('ave_large', ave_size(larges)),
                ]
            )
            return merge_info

        # def hybrid_measures(hybrid):
        #     pred = hybrid['old']
        #     true = hybrid['new']
        #     hybrid_info = ut.odict([
        #         ('n_pred_pccs', len(pred)),
        #         ('size_pred_pccs', ave_size(pred)),
        #         ('n_real_pccs', len(true)),
        #         ('size_real_pccs', ave_size(true)),
        #     ])
        #     return hybrid_info

        info = {
            'correct': unchanged_measures(unchanged),
            'split': split_measures(all_splits),
            'merge': merge_measures(all_merges),
        }
        return info

    def draw_simulation(self):
        """
        CommandLine:
            python -m wbia Chap5.draw simulation PZ_MTEST --diskshow
            python -m wbia Chap5.draw simulation GZ_Master1 --diskshow
            python -m wbia Chap5.draw simulation PZ_Master1 --diskshow

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap5('GZ_Master')
        """
        sim_results = self.ensure_results('simulation')

        keys = ['ranking', 'rank+clf', 'graph']
        colors = ut.dzip(keys, ['red', 'orange', 'b'])

        def _metrics(col):
            return {k: ut.take_column(v['metrics'], col) for k, v in sim_results.items()}

        fnum = 1

        xdatas = _metrics('n_manual')
        xmax = max(map(max, xdatas.values()))
        xpad = (1.01 * xmax) - xmax

        pnum_ = pt.make_pnum_nextgen(nSubplots=2)

        mpl.rcParams.update(TMP_RC)

        pt.figure(fnum=fnum, pnum=pnum_())
        ax = pt.gca()
        ydatas = _metrics('merge_remain')
        for key in keys:
            ax.plot(xdatas[key], ydatas[key], label=key, color=colors[key])
        ax.set_ylim(0, 1)
        ax.set_xlim(-xpad, xmax + xpad)
        ax.set_xlabel('# manual reviews')
        ax.set_ylabel('fraction of merges remain')
        ax.legend()

        pt.figure(fnum=fnum, pnum=pnum_())
        ax = pt.gca()
        ydatas = _metrics('n_errors')
        for key in keys:
            ax.plot(xdatas[key], ydatas[key], label=key, color=colors[key])
        ax.set_ylim(0, max(map(max, ydatas.values())) * 1.01)
        ax.set_xlim(-xpad, xmax + xpad)
        ax.set_xlabel('# manual reviews')
        ax.set_ylabel('# errors')
        ax.legend()

        fig = pt.gcf()  # NOQA
        fig.set_size_inches([W, H * 0.75])
        pt.adjust_subplots(wspace=0.25, fig=fig)

        fpath = join(self.dpath, 'simulation.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fpath)

    def draw_refresh(self):
        """
        CommandLine:
            python -m wbia Chap5.draw refresh GZ_Master1 --diskshow
            python -m wbia Chap5.draw refresh PZ_Master1 --diskshow
        """
        sim_results = self.ensure_results('simulation')

        keys = ['ranking', 'rank+clf', 'graph']
        colors = ut.dzip(keys, ['red', 'orange', 'b'])

        def _metrics(col):
            return {k: ut.take_column(v['metrics'], col) for k, v in sim_results.items()}

        fnum = 1
        xdatas = _metrics('n_manual')
        pnum_ = pt.make_pnum_nextgen(nSubplots=1)

        mpl.rcParams.update(TMP_RC)

        pt.figure(fnum=fnum, pnum=pnum_())
        ax = pt.gca()
        ydatas = _metrics('pprob_any')

        # fix the visual inconsistency that doesn't matter in practice
        # flags = _metrics('refresh_support')
        key = 'graph'
        ax.plot(xdatas[key], ydatas[key], label=key, color=colors[key])
        ax.set_xlabel('# manual reviews')
        ax.set_ylabel('P(C=1)')
        # ax.legend()

        fpath = join(self.dpath, 'refresh.png')
        fig = pt.gcf()  # NOQA
        fig.set_size_inches([W, H * 0.5])
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_simulation2(self):
        """
        CommandLine:
            python -m wbia Chap5.draw_simulation2 --db PZ_MTEST --show
            python -m wbia Chap5.draw_simulation2 --db GZ_Master1 --show
            python -m wbia Chap5.draw_simulation2 --db PZ_Master1 --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='GZ_Master1')
            >>> self = Chap5(dbname)
            >>> self.draw_simulation2()
            >>> ut.show_if_requested()
        """
        mpl.rcParams.update(TMP_RC)
        sim_results = self.ensure_results('simulation')

        expt_data = sim_results['graph']
        metrics_df = pd.DataFrame.from_dict(expt_data['metrics'])

        fnum = 1  # NOQA
        overshow = {
            'phase': True,
            'pred': False,
            'auto': True,
            'real': True,
            'error': True,
            'recover': True,
        }
        if overshow['auto']:
            xdata = metrics_df['n_decision']
            xlabel = '# decisions'
        else:
            xdata = metrics_df['n_manual']
            xlabel = '# manual reviews'

        def plot_intervals(flags, color=None, low=0, high=1):
            ax = pt.gca()
            idxs = np.where(flags)[0]
            ranges = ut.group_consecutives(idxs)
            bounds = [(min(a), max(a)) for a in ranges if len(a) > 0]
            xdata_ = xdata.values

            xs, ys = [xdata_[0]], [low]
            for a, b in bounds:
                x1, x2 = xdata_[a], xdata_[b]
                # if x1 == x2:
                x1 -= 0.5
                x2 += 0.5
                xs.extend([x1, x1, x2, x2])
                ys.extend([low, high, high, low])
            xs.append(xdata_[-1])
            ys.append(low)
            ax.fill_between(xs, ys, low, alpha=0.6, color=color)

        def overlay_actions(ymax=1):
            """
            Draws indicators that detail the algorithm state at given
            timestamps.
            """
            phase = metrics_df['phase'].map(lambda x: x.split('_')[0])
            is_correct = (
                metrics_df['test_action'].map(lambda x: x.startswith('correct')).values
            )
            recovering = metrics_df['recovering'].values
            is_auto = metrics_df['user_id'].map(lambda x: x.startswith('algo:')).values
            ppos = metrics_df['pred_decision'].map(lambda x: x == POSTV).values
            rpos = metrics_df['true_decision'].map(lambda x: x == POSTV).values
            # ymax = max(metrics_df['n_errors'])

            num = sum(overshow.values())
            steps = np.linspace(0, 1, num + 1) * ymax
            i = -1

            def stacked_interval(data, color, i):
                plot_intervals(data, color, low=steps[i], high=steps[i + 1])

            if overshow['auto']:
                i += 1
                pt.absolute_text(
                    (0.2, steps[i : i + 2].mean()), 'is_auto(auto=gold,manual=blue)'
                )
                stacked_interval(is_auto, 'gold', i)
                stacked_interval(~is_auto, 'blue', i)

            if overshow['pred']:
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'pred_pos')
                stacked_interval(ppos, 'aqua', low=steps[i], high=steps[i + 1])
                # stacked_interval(~ppos, 'salmon', i)

            if overshow['real']:
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'real_pos')
                stacked_interval(rpos, 'lime', i)
                # stacked_interval(~ppos, 'salmon', i)

            if overshow['error']:
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'is_error')
                # stacked_interval(is_correct, 'blue', low=steps[i], high=steps[i + 1])
                stacked_interval(~is_correct, 'red', i)

            if overshow['recover']:
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'is_recovering')
                stacked_interval(recovering, 'orange', i)

            if overshow['phase']:
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'phase')
                stacked_interval(phase == 'ranking', 'red', i)
                stacked_interval(phase == 'posredun', 'green', i)
                stacked_interval(phase == 'negredun', 'blue', i)

        pnum_ = pt.make_pnum_nextgen(nRows=2, nSubplots=8)

        ydatas = ut.odict([('Graph', metrics_df['merge_remain'])])
        pt.multi_plot(
            xdata,
            ydatas,
            marker='',
            markersize=1,
            xlabel=xlabel,
            ylabel='fraction of merge remaining',
            ymin=0,
            rcParams=TMP_RC,
            use_legend=True,
            fnum=1,
            pnum=pnum_(),
        )
        # overlay_actions(1)

        ykeys = ['n_errors']
        pt.multi_plot(
            xdata,
            metrics_df[ykeys].values.T,
            xlabel=xlabel,
            ylabel='# of errors',
            marker='',
            markersize=1,
            ymin=0,
            rcParams=TMP_RC,
            fnum=1,
            pnum=pnum_(),
            use_legend=False,
        )
        overlay_actions(max(metrics_df['n_errors']))

        pt.multi_plot(
            xdata,
            [metrics_df['pprob_any']],
            label_list=['P(C=1)'],
            xlabel=xlabel,
            ylabel='refresh criteria',
            marker='',
            ymin=0,
            ymax=1,
            rcParams=TMP_RC,
            fnum=1,
            pnum=pnum_(),
            use_legend=False,
        )
        ax = pt.gca()
        thresh = expt_data['refresh_thresh']
        ax.plot([min(xdata), max(xdata)], [thresh, thresh], '-g', label='refresh thresh')
        ax.legend()
        # overlay_actions(1)

        ykeys = ['n_fn', 'n_fp']
        pt.multi_plot(
            xdata,
            metrics_df[ykeys].values.T,
            label_list=ykeys,
            xlabel=xlabel,
            ylabel='# of errors',
            marker='x',
            markersize=1,
            ymin=0,
            rcParams=TMP_RC,
            ymax=max(metrics_df['n_errors']),
            fnum=1,
            pnum=pnum_(),
            use_legend=True,
        )

        xdata = metrics_df['n_manual']
        xlabel = '# manual reviews'
        ydatas = ut.odict([('Graph', metrics_df['merge_remain'])])
        pt.multi_plot(
            xdata,
            ydatas,
            marker='',
            markersize=1,
            xlabel=xlabel,
            ylabel='fraction of merge remaining',
            ymin=0,
            rcParams=TMP_RC,
            use_legend=True,
            fnum=1,
            pnum=pnum_(),
        )
        # overlay_actions(1)

        ykeys = ['n_errors']
        pt.multi_plot(
            xdata,
            metrics_df[ykeys].values.T,
            xlabel=xlabel,
            ylabel='# of errors',
            marker='',
            markersize=1,
            ymin=0,
            rcParams=TMP_RC,
            fnum=1,
            pnum=pnum_(),
            use_legend=False,
        )
        overlay_actions(max(metrics_df['n_errors']))

        pt.multi_plot(
            xdata,
            [metrics_df['pprob_any']],
            label_list=['P(C=1)'],
            xlabel=xlabel,
            ylabel='refresh criteria',
            marker='',
            ymin=0,
            ymax=1,
            rcParams=TMP_RC,
            fnum=1,
            pnum=pnum_(),
            use_legend=False,
        )
        ax = pt.gca()
        thresh = expt_data['refresh_thresh']
        ax.plot([min(xdata), max(xdata)], [thresh, thresh], '-g', label='refresh thresh')
        ax.legend()
        # overlay_actions(1)

        ykeys = ['n_fn', 'n_fp']
        pt.multi_plot(
            xdata,
            metrics_df[ykeys].values.T,
            label_list=ykeys,
            xlabel=xlabel,
            ylabel='# of errors',
            marker='x',
            markersize=1,
            ymin=0,
            rcParams=TMP_RC,
            ymax=max(metrics_df['n_errors']),
            fnum=1,
            pnum=pnum_(),
            use_legend=True,
        )

        # fpath = join(self.dpath, expt_name + '2' + '.png')
        # fig = pt.gcf()  # NOQA
        # fig.set_size_inches([W * 1.5, H * 1.1])
        # vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        # if ut.get_argflag('--diskshow'):
        #     ut.startfile(fpath)
        # fig.save_fig

        # if 1:
        #     pt.figure(fnum=fnum, pnum=(2, 2, 4))
        #     overlay_actions(ymax=1)
        pt.set_figtitle(self.dbname)


@ut.reloadable_class
class Chap4(DBInputs):
    """
    Collect data from experiments to visualize

    TODO: redo save/loading of measurments

    Ignore:
        >>> from wbia.scripts.thesis import *
        >>> fpath = ut.glob(ut.truepath('~/Desktop/mtest_plots'), '*.pkl')[0]
        >>> self = ut.load_data(fpath)
    """

    base_dpath = ut.truepath('~/latex/crall-thesis-2017/figures4')

    task_nice_lookup = {
        'match_state': const.EVIDENCE_DECISION.CODE_TO_NICE,
        'photobomb_state': {'pb': 'Photobomb', 'notpb': 'Not Photobomb'},
    }

    def _setup(self):
        r"""
        CommandLine:
            python -m wbia Chap4._setup --db GZ_Master1

            python -m wbia Chap4._setup --db PZ_Master1 --eval
            python -m wbia Chap4._setup --db PZ_MTEST
            python -m wbia Chap4._setup --db PZ_PB_RF_TRAIN

            python -m wbia Chap4.measure_all --db PZ_PB_RF_TRAIN

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='GZ_Master1')
            >>> self = Chap4(dbname)
            >>> self._setup()

        Ignore:
            from wbia.scripts.thesis import *
            self = Chap4('PZ_Master1')

            from wbia.scripts.thesis import *
            self = Chap4('PZ_PB_RF_TRAIN')

            self.ibs.print_annot_stats(aids, prefix='P')
        """
        import wbia

        self._precollect()
        ibs = self.ibs

        if ibs.dbname == 'PZ_Master1':
            # FIND ALL PHOTOBOMB / INCOMPARABLE CASES
            if False:
                infr = wbia.AnnotInference(ibs, aids='all')
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
            species = "Grévy's Zebras"
        dbcode = '{}_{}'.format(ibs.dbname, len(pblm.samples))

        self.pblm = pblm
        self.dbcode = dbcode
        self.eval_task_keys = pblm.eval_task_keys
        self.species = species
        self.data_key = data_key
        self.clf_key = clf_key

        # config = pblm.hyper_params
        # self._setup_links(cfg_prefix, config)

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
            python -m wbia Chap4.measure_all --db PZ_PB_RF_TRAIN
            python -m wbia Chap4.measure_all --db PZ_MTEST
            python -m wbia Chap4.measure_all

            python -m wbia Chap4.measure_all --db GZ_Master1

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = Chap4(dbname)
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
            python -m wbia Chap4.draw_all --db PZ_MTEST
            python -m wbia Chap4.draw_all --db PZ_PB_RF_TRAIN
            python -m wbia Chap4.draw_all --db GZ_Master1
            python -m wbia Chap4.draw_all --db PZ_Master1

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = Chap4(dbname)
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
        >>> from wbia.scripts.thesis import *
        >>> self = Chap4('GZ_Master1')
        >>> self = Chap4('PZ_Master1')
        >>> self = Chap4('PZ_MTEST')
        """
        # from sklearn.feature_selection import SelectFromModel
        from wbia.scripts import clf_helpers

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
            clf_list, res_list = pblm._train_evaluation_clf(
                task_key, data_key, clf_key, feat_dims
            )
            combo_res = clf_helpers.ClfResult.combine_results(res_list, labels)
            rs = [res.extended_clf_report(verbose=0) for res in res_list]
            report = combo_res.extended_clf_report(verbose=0)

            # Measure mean decrease in impurity
            clf_mdi = np.array([clf_.feature_importances_ for clf_ in clf_list])
            mean_mdi = ut.dzip(feat_dims, np.mean(clf_mdi, axis=0))

            # Record state
            n_dims.append(len(feat_dims))
            reports.append(report)
            sub_reports.append(rs)
            mdis_list.append(mean_mdi)

            # remove the worst features
            sorted_featdims = ub.argsort(mean_mdi)
            n_have = len(sorted_featdims)
            n_remove = n_have - max(n_have - prune_rate, min_feats)
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
        >>> from wbia.scripts.thesis import *
        >>> defaultdb = 'PZ_Master1'
        >>> defaultdb = 'GZ_Master1'
        >>> self = Chap4(defaultdb)
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

        verifiers = infr.learn_evaluation_verifiers()
        probs = verifiers['match_state'].predict_proba_df(rerank_pairs)
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
        lnbnn_cdf = np.cumsum(hist) / sum(hist)

        bins = np.arange(len(qreq_.dnids))
        hist = np.histogram(clf_name_ranks, bins=bins)[0]
        clf_cdf = np.cumsum(hist) / sum(hist)

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
            python -m wbia Chap4.measure hard_cases GZ_Master1 match_state
            python -m wbia Chap4.measure hard_cases GZ_Master1 photobomb_state
            python -m wbia Chap4.draw hard_cases GZ_Master1 match_state
            python -m wbia Chap4.draw hard_cases GZ_Master1 photobomb_state

            python -m wbia Chap4.measure hard_cases PZ_Master1 match_state
            python -m wbia Chap4.measure hard_cases PZ_Master1 photobomb_state
            python -m wbia Chap4.draw hard_cases PZ_Master1 match_state
            python -m wbia Chap4.draw hard_cases PZ_Master1 photobomb_state

            python -m wbia Chap4.measure hard_cases PZ_MTEST match_state
            python -m wbia Chap4.draw hard_cases PZ_MTEST photobomb_state

            python -m wbia Chap4.measure hard_cases MantaMatcher match_state

        Ignore:
            >>> task_key = 'match_state'
            >>> task_key = 'photobomb_state'
            >>> from wbia.scripts.thesis import *
            >>> self = Chap4('GZ_Master1')
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
        print(
            'With average hardness {}'.format(
                ut.repr2(
                    ut.stats_dict(failure_cases['hardness']), strkeys=True, precision=2
                )
            )
        )

        cases = []
        for (pred, real), group in failure_cases.groupby(('pred', 'real')):

            group = group.sort_values(['easiness'])
            flags = ut.flag_percentile_parts(group['easiness'], front, mid, back)
            subgroup = group[flags]

            print(
                'Selected {} r({})-p({}) cases'.format(
                    len(subgroup), res.class_names[real], res.class_names[pred]
                )
            )
            # ut.take_percentile_parts(group['easiness'], front, mid, back)

            # Prefer examples we have manually reviewed before
            # group = group.sort_values(['real_conf', 'easiness'])
            # subgroup = group[0:num_top]

            for idx, case in subgroup.iterrows():
                edge = tuple(ut.take(case, ['aid1', 'aid2']))
                cases.append(
                    {
                        'edge': edge,
                        'real': res.class_names[case['real']],
                        'pred': res.class_names[case['pred']],
                        'failed': case['failed'],
                        'easiness': case['easiness'],
                        'real_conf': case['real_conf'],
                        'probs': res.probs_df.loc[edge].to_dict(),
                        'edge_data': pblm.infr.get_edge_data(edge),
                    }
                )

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

        python -m wbia Chap4.draw hard_cases GZ_Master1 match_state
        python -m wbia Chap4.draw hard_cases PZ_Master1 match_state
        python -m wbia Chap4.draw hard_cases PZ_Master1 photobomb_state
        python -m wbia Chap4.draw hard_cases GZ_Master1 photobomb_state



            >>> from wbia.scripts.thesis import *
            >>> self = Chap4('PZ_MTEST')
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
            NEGTV: [(239, 3745), (484, 519), (802, 803)],
            INCMP: [(4652, 5245), (4405, 5245), (4109, 5245), (16192, 16292)],
            POSTV: [(6919, 7192)],
        }

        prog = ut.ProgIter(cases, 'draw {} hard case'.format(task_key), bs=False)
        for case in prog:
            aid1, aid2 = case['edge']
            match = case['match']
            real_name, pred_name = case['real'], case['pred']
            real_nice, pred_nice = ut.take(code_to_nice, [real_name, pred_name])
            fname = 'fail_{}_{}_{}_{}'.format(real_name, pred_name, aid1, aid2)
            # Build x-label
            _probs = case['probs']
            probs = ut.odict(
                (v, _probs[k]) for k, v in code_to_nice.items() if k in _probs
            )
            probstr = ut.repr2(probs, precision=2, strkeys=True, nobr=True)
            xlabel = 'real={}, pred={},\n{}'.format(real_nice, pred_nice, probstr)
            fig = pt.figure(fnum=1000, clf=True)
            ax = pt.gca()
            # Draw with feature overlay
            match.show(
                ax,
                vert=False,
                heatmask=True,
                show_lines=False,
                # show_lines=True, line_lw=1, line_alpha=.1,
                # ell_alpha=.3,
                show_ell=False,
                show_ori=False,
                show_eig=False,
                modifysize=True,
            )
            ax.set_xlabel(xlabel)
            # ax.get_xaxis().get_label().set_fontsize(24)
            ax.get_xaxis().get_label().set_fontsize(24)

            fpath = join(str(dpath), fname + '.jpg')
            vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def write_metrics2(self, task_key='match_state'):
        """
        CommandLine:
            python -m wbia Chap4.draw metrics PZ_PB_RF_TRAIN match_state
            python -m wbia Chap4.draw metrics2 PZ_Master1 photobomb_state
            python -m wbia Chap4.draw metrics2 GZ_Master1 photobomb_state

            python -m wbia Chap4.draw metrics2 GZ_Master1 photobomb_state
        """
        results = self.ensure_results('all')
        task_combo_res = results['task_combo_res']
        data_key = results['data_key']
        clf_key = results['clf_key']

        res = task_combo_res[task_key][clf_key][data_key]

        from wbia.scripts import sklearn_utils

        threshes = res.get_thresholds('mcc', 'max')
        y_pred = sklearn_utils.predict_from_probs(res.probs_df, threshes, force=True)
        y_true = res.target_enc_df

        # pred_enc = res.clf_probs.argmax(axis=1)
        # y_pred = pred_enc
        res.augment_if_needed()
        sample_weight = res.sample_weight
        target_names = res.class_names

        report = sklearn_utils.classification_report2(
            y_true, y_pred, target_names, sample_weight, verbose=False
        )
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

        fpath1 = ut.render_latex(confusion_tex, dpath=dpath, fname=confusion_fname)
        fpath2 = ut.render_latex(metrics_tex, dpath=dpath, fname=metrics_fname)
        return fpath1, fpath2

    def write_metrics(self, task_key='match_state'):
        """
        CommandLine:
            python -m wbia Chap4.draw metrics PZ_PB_RF_TRAIN match_state
            python -m wbia Chap4.draw metrics GZ_Master1 photobomb_state

            python -m wbia Chap4.draw metrics PZ_Master1,GZ_Master1 photobomb_state,match_state

        Ignore:
            >>> from wbia.scripts.thesis import *
            >>> self = Chap4('PZ_Master1')
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

        from wbia.scripts import sklearn_utils

        report = sklearn_utils.classification_report2(
            y_true, y_pred, target_names, sample_weight, verbose=False
        )
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

        fpath1 = ut.render_latex(confusion_tex, dpath=dpath, fname=confusion_fname)
        fpath2 = ut.render_latex(metrics_tex, dpath=dpath, fname=metrics_fname)
        return fpath1, fpath2

    def write_sample_info(self):
        """
        python -m wbia Chap4.draw sample_info GZ_Master1

        """
        results = self.ensure_results('sample_info')
        # results['aid_pool']
        # results['encoded_labels2d']
        # results['multihist']
        import wbia

        infr = wbia.AnnotInference.from_netx(results['graph'])
        info = ut.odict()
        info['n_names'] = (infr.pos_graph.number_of_components(),)
        info['n_aids'] = (len(results['pblm_aids']),)
        info['known_n_incomparable'] = infr.incomp_graph.number_of_edges()
        subtasks = results['subtasks']

        task = subtasks['match_state']
        flags = task.encoded_df == task.class_names.tolist().index(INCMP)
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
        python -m wbia Chap4.draw importance GZ_Master1,PZ_Master1 match_state

        python -m wbia Chap4.draw importance GZ_Master1 match_state
        python -m wbia Chap4.draw importance PZ_Master1 match_state

        python -m wbia Chap4.draw importance GZ_Master1 photobomb_state
        python -m wbia Chap4.draw importance PZ_Master1 photobomb_state
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
            k = feat_alias(k)
            k = k.replace('_', '\\_')
            lines.append('\\tt{{{}}} & ${:.4f}$ \\\\'.format(k, v))
        latex_str = '\n'.join(ut.align_lines(lines, '&'))

        fname = 'feat_importance_{}'.format(task_key)

        print('TOP {} importances for {}'.format(num_top, task_key))
        print('# of dimensions: %d' % (len(importances)))
        print(latex_str)
        print()
        extra_ = ut.codeblock(
            r"""
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
            """
        ).format(num_top, len(importances), task_key.replace('_', '-'), latex_str)

        fpath = ut.render_latex(extra_, dpath=self.dpath, fname=fname)
        ut.write_to(join(str(self.dpath), fname + '.tex'), latex_str)
        return fpath

    def draw_prune(self):
        """
        CommandLine:
            python -m wbia Chap4.draw importance GZ_Master1

            python -m wbia Chap4.draw importance PZ_Master1 photobomb_state
            python -m wbia Chap4.draw importance PZ_Master1 match_state

            python -m wbia Chap4.draw prune GZ_Master1,PZ_Master1
            python -m wbia Chap4.draw prune PZ_Master1

        >>> from wbia.scripts.thesis import *
        >>> self = Chap4('PZ_Master1')
        >>> self = Chap4('GZ_Master1')
        >>> self = Chap4('PZ_MTEST')
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
        ave_mccs = np.array(
            [[r['metrics']['mcc']['ave/sum'] for r in rs] for rs in sub_reports]
        )

        import wbia.plottool as pt

        mpl.rcParams.update(TMP_RC)
        fig = pt.figure(fnum=1, doclf=True)
        pt.multi_plot(
            n_dims,
            {'mean': ave_mccs.mean(axis=1)},
            rcParams=TMP_RC,
            marker='',
            force_xticks=[min(n_dims)],
            # num_xticks=5,
            ylabel='MCC',
            xlabel='# feature dimensions',
            ymin=0.5,
            ymax=1,
            xmin=1,
            xmax=n_dims[0],
            fnum=1,
            use_legend=False,
        )
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
            k = feat_alias(k)
            k = k.replace('_', '\\_')
            lines.append('\\tt{{{}}} & ${:.4f}$ \\\\'.format(k, v))
        latex_str = '\n'.join(ut.align_lines(lines, '&'))

        increase = u[idx] - u[0]

        print(latex_str)
        print()
        extra_ = ut.codeblock(
            r"""
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
            """
        ).format(
            num_top,
            len(pruned_importance),
            task_key.replace('_', '-'),
            increase,
            latex_str,
        )
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
            bins,
            (freq0, freq1),
            label_list=('negative', 'positive'),
            color_list=(pt.FALSE_RED, pt.TRUE_BLUE),
            kind='bar',
            width=width,
            alpha=0.7,
            edgecolor='none',
            xlabel=xlabel,
            ylabel='frequency',
            fnum=fnum,
            pnum=(1, 1, 1),
            rcParams=TMP_RC,
            stacked=True,
            ytickformat='%.2f',
            xlim=xlim,
            # title='LNBNN positive separation'
        )
        pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
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
        fig.set_size_inches([W, H * 0.6])
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

        score_hist_pos = {'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}

        lnbnn_xy = results['lnbnn_xy']
        scores = lnbnn_xy['score_lnbnn_1vM'].values
        y = lnbnn_xy[POSTV].values

        # Get 95% of the data at least
        maxbin = scores[scores.argsort()][-max(1, int(len(scores) * 0.05))]
        bins = np.linspace(0, max(maxbin, 10), 100)
        pos_freq = np.histogram(scores[y], bins)[0]
        neg_freq = np.histogram(scores[~y], bins)[0]
        pos_freq = pos_freq / pos_freq.sum()
        neg_freq = neg_freq / neg_freq.sum()
        score_hist_lnbnn = {'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}

        fig1 = self._draw_score_hist(score_hist_pos, 'positive probability', 1)
        fig2 = self._draw_score_hist(score_hist_lnbnn, 'LNBNN score', 2)

        fname = 'score_hist_pos_{}.png'.format(data_key)
        vt.imwrite(join(str(self.dpath), fname), pt.render_figure_to_image(fig1, dpi=DPI))

        fname = 'score_hist_lnbnn.png'
        vt.imwrite(join(str(self.dpath), fname), pt.render_figure_to_image(fig2, dpi=DPI))

    def draw_mcc_thresh(self, task_key):
        """
        python -m wbia Chap4.draw mcc_thresh GZ_Master1 match_state
        python -m wbia Chap4.draw mcc_thresh PZ_Master1 match_state

        python -m wbia Chap4.draw mcc_thresh GZ_Master1 photobomb_state
        python -m wbia Chap4.draw mcc_thresh PZ_Master1 photobomb_state

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
                {
                    'label': class_nice + ', t={:.2f}, mcc={:.2f}'.format(t, mcc),
                    'thresh': c1.thresholds,
                    'mcc': c1.mcc,
                },
            ]

        fig = pt.figure(fnum=1)  # NOQA
        ax = pt.gca()
        for data in roc_curves:
            ax.plot(data['thresh'], data['mcc'], label='%s' % (data['label']))
        ax.set_xlabel('threshold')
        ax.set_ylabel('MCC')
        # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
        ax.legend()
        pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
        fig.set_size_inches([W, H])

        fname = 'mcc_thresh_{}.png'.format(task_key)
        fig_fpath = join(str(self.dpath), fname)
        vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fig_fpath)

    def draw_roc(self, task_key):
        """
        python -m wbia Chap4.draw roc GZ_Master1 photobomb_state
        python -m wbia Chap4.draw roc GZ_Master1 match_state
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
            c2 = vt.ConfusionMetrics().fit(scores, y)
            c3 = res.confusions(target_class)
            roc_curves = [
                {'label': 'LNBNN', 'fpr': c2.fpr, 'tpr': c2.tpr, 'auc': c2.auc},
                {'label': 'learned', 'fpr': c3.fpr, 'tpr': c3.tpr, 'auc': c3.auc},
            ]

            at_metric = 'tpr'
            for at_value in [0.25, 0.5, 0.75]:
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
            ax.plot(
                data['fpr'],
                data['tpr'],
                label='%s AUC=%.2f' % (data['label'], data['auc']),
            )
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('true positive rate')
        # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
        ax.legend()
        pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
        fig.set_size_inches([W, H])

        fname = 'roc_{}.png'.format(task_key)
        fig_fpath = join(str(self.dpath), fname)
        vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def draw_wordcloud(self, task_key):
        import wbia.plottool as pt

        results = self.ensure_results('all')
        importances = ut.map_keys(feat_alias, results['importance'][task_key])

        fig = pt.figure(fnum=1)
        pt.wordcloud(importances, ax=fig.axes[0])

        fname = 'wc_{}.png'.format(task_key)
        fig_fpath = join(str(self.dpath), fname)
        vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

    @classmethod
    def draw_tagged_pair(cls):
        import wbia

        # ibs = wbia.opendb(defaultdb='GZ_Master1')
        ibs = wbia.opendb(defaultdb='PZ_Master1')

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
        infr = wbia.AnnotInference(ibs=ibs, aids=edge, verbose=10)
        infr.reset_feedback('annotmatch', apply=True)
        match = infr._exec_pairwise_match([edge])[0]

        if False:
            # Fix the example tags
            infr.add_feedback(
                edge,
                'match',
                tags=['facematch', 'leftrightface'],
                user_id='qt-hack',
                confidence='pretty_sure',
            )
            infr.write_wbia_staging_feedback()
            infr.write_wbia_annotmatch_feedback()
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
        match.show(
            ax,
            vert=False,
            heatmask=True,
            show_lines=False,
            show_ell=False,
            show_ori=False,
            show_eig=False,
            # ell_alpha=.3,
            modifysize=True,
        )
        # ax.set_xlabel(xlabel)

        self = cls()

        fname = 'custom_match_{}_{}_{}'.format(query_tag, *edge)
        dpath = pathlib.Path(ut.truepath(self.base_dpath))
        fpath = join(str(dpath), fname + '.jpg')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def custom_single_hard_case(self):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> defaultdb = 'PZ_PB_RF_TRAIN'
            >>> #defaultdb = 'GZ_Master1'
            >>> defaultdb = 'PZ_MTEST'
            >>> self = Chap4.collect(defaultdb)
            >>> self.dbname = 'PZ_PB_RF_TRAIN'
        """
        task_key = 'match_state'
        edge = (383, 503)
        for _case in self.hard_cases[task_key]:
            if _case['edge'] == edge:
                case = _case
                break

        import wbia

        ibs = wbia.opendb(self.dbname)

        from wbia import core_annots

        config = {
            'augment_orientation': True,
            'ratio_thresh': 0.8,
        }
        config['checks'] = 80
        config['sver_xy_thresh'] = 0.02
        config['sver_ori_thresh'] = 3
        config['Knorm'] = 3
        config['symmetric'] = True
        config = ut.hashdict(config)

        aid1, aid2 = case['edge']
        real_name = case['real']
        pred_name = case['pred']
        match = case['match']
        code_to_nice = self.task_nice_lookup[task_key]
        real_nice, pred_nice = ut.take(code_to_nice, [real_name, pred_name])
        fname = 'fail_{}_{}_{}_{}'.format(real_nice, pred_nice, aid1, aid2)
        # Draw case
        probs = case['probs'].to_dict()
        order = list(code_to_nice.values())
        order = ut.setintersect(order, probs.keys())
        probs = ut.map_dict_keys(code_to_nice, probs)
        probstr = ut.repr2(probs, precision=2, strkeys=True, nobr=True, key_order=order)
        xlabel = 'real={}, pred={},\n{}'.format(real_nice, pred_nice, probstr)

        match_list = ibs.depc.get(
            'pairwise_match', ([aid1], [aid2]), 'match', config=config
        )
        match = match_list[0]
        configured_lazy_annots = core_annots.make_configured_annots(
            ibs, [aid1], [aid2], config, config, preload=True
        )
        match.annot1 = configured_lazy_annots[config][aid1]
        match.annot2 = configured_lazy_annots[config][aid2]
        match.config = config

        fig = pt.figure(fnum=1, clf=True)
        ax = pt.gca()

        mpl.rcParams.update(TMP_RC)
        match.show(
            ax,
            vert=False,
            heatmask=True,
            show_lines=False,
            show_ell=False,
            show_ori=False,
            show_eig=False,
            # ell_alpha=.3,
            modifysize=True,
        )
        ax.set_xlabel(xlabel)

        subdir = 'cases_{}'.format(task_key)
        dpath = join(str(self.dpath), subdir)
        fpath = join(str(dpath), fname + '_custom.jpg')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))


@ut.reloadable_class
class Chap3Measures(object):
    def measure_baseline(self):
        """
        >>> from wbia.scripts.thesis import *
        >>> self = Chap3('GZ_Master1')
        >>> self._precollect()
        """
        ibs = self.ibs
        qaids, daids_list, info_list = Sampler._varied_inputs(
            self.ibs, self.aids_pool, denc_per_name=[1], extra_dbsize_fracs=[1]
        )
        cfgdict = {}
        daids = daids_list[0]
        info = info_list[0]
        cdf = _ranking_cdf(ibs, qaids, daids, cfgdict)
        results = [(cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict}))]

        expt_name = 'baseline'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_foregroundness_intra(self):
        ibs = self.ibs
        samples = Sampler._intra_enc(ibs, self.aids_pool)
        # qaids, daids_list, info_list = sample.expand()

        results = []
        for sample in samples:
            qaids = sample.qaids
            daids = sample.daids
            info = {'qsize': len(qaids), 'dsize': len(daids)}
            grid = ut.all_dict_combinations({'featweight_enabled': [False, True]})
            for cfgdict in grid:
                hist = _ranking_hist(ibs, qaids, daids, cfgdict)
                info = ut.update_dict(info.copy(), {'pcfg': cfgdict})
                results.append((hist, info))

        expt_name = 'foregroundness_intra'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def draw_foregroundness_intra(self):
        """
        python -m wbia Chap3.measure foregroundness_intra --dbs=GZ_Master1,PZ_Master1
        python -m wbia Chap3.draw foregroundness_intra --dbs=GZ_Master1,PZ_Master1 --diskshow
        """
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        expt_name = 'foregroundness_intra'

        mpl.rcParams.update(TMP_RC)
        results = self.ensure_results(expt_name)
        hists, infos = list(zip(*results))
        pcfgs = ut.take_column(infos, 'pcfg')
        df = pd.DataFrame.from_records(infos)
        df['hists'] = hists
        df['fg_on'] = ut.take_column(pcfgs, 'featweight_enabled')

        cdfs = []
        labels = []

        for fg, group in df.groupby(('fg_on')):
            labels.append('fg=T' if fg else 'fg=F')
            hists = vt.pad_vstack(group['hists'], fill_value=0)
            hist = hists.sum(axis=0)
            cdf = np.cumsum(hist) / sum(hist)
            cdfs.append(cdf)
            qsize = str(group['qsize'].sum())
            u, s = group['dsize'].mean(), group['dsize'].std()
            dsize = ave_str(u, s, precision=1)

        fig = plot_cmcs(cdfs, labels, ymin=0.5)
        fig.set_size_inches([W, H * 0.6])
        nonvaried_text = 'qsize={:s}, dsize={:s}'.format(qsize, dsize)
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def measure_foregroundness(self):
        ibs = self.ibs
        qaids, daids_list, info_list = Sampler._varied_inputs(
            self.ibs,
            self.aids_pool,
            denc_per_name=[1],
            extra_dbsize_fracs=[1],
            method='same_occur'
            # method='same_enc'
        )
        daids = daids_list[0]
        info = info_list[0]

        results = []
        cfgdict1 = {'fg_on': False}
        cdf = _ranking_cdf(ibs, qaids, daids, cfgdict1)
        results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict1})))

        cfgdict2 = {'fg_on': True}
        cdf = _ranking_cdf(ibs, qaids, daids, cfgdict2)
        results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict2})))

        expt_name = 'foregroundness'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_invar(self):
        ibs = self.ibs
        qaids, daids_list, info_list = Sampler._varied_inputs(
            self.ibs, self.aids_pool, denc_per_name=[1], extra_dbsize_fracs=[1]
        )
        daids = daids_list[0]
        info = info_list[0]

        cfgdict_list = [
            {
                'affine_invariance': True,
                'rotation_invariance': False,
                'query_rotation_heuristic': False,
            },
            # {'affine_invariance':  True, 'rotation_invariance':  True, 'query_rotation_heuristic': False},
            # {'affine_invariance': False, 'rotation_invariance':  True, 'query_rotation_heuristic': False},
            {
                'affine_invariance': False,
                'rotation_invariance': False,
                'query_rotation_heuristic': False,
            },
            {
                'affine_invariance': True,
                'rotation_invariance': False,
                'query_rotation_heuristic': True,
            },
            {
                'affine_invariance': False,
                'rotation_invariance': False,
                'query_rotation_heuristic': True,
            },
        ]
        results = []
        for cfgdict in cfgdict_list:
            cdf = _ranking_cdf(ibs, qaids, daids, cfgdict)
            results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})))

        expt_name = 'invar'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_smk(self):
        """
        python -m wbia Chap3.measure smk --dbs=GZ_Master1,PZ_Master1
        python -m wbia Chap3.draw smk --dbs=GZ_Master1,PZ_Master1 --diskshow
        """
        from wbia.algo.smk.smk_pipeline import SMKRequest

        # ibs = wbia.opendb('PZ_MTEST')
        ibs = self.ibs
        qaids, daids_list, info_list = Sampler._varied_inputs(
            self.ibs, self.aids_pool, denc_per_name=[1], extra_dbsize_fracs=[1]
        )
        daids = daids_list[0]
        info = info_list[0]
        results = []

        # SMK pipeline
        config = {'nAssign': 1, 'num_words': 8000, 'sv_on': True}
        qreq_ = SMKRequest(ibs, qaids, daids, config)
        qreq_.ensure_data()
        cm_list = qreq_.execute()
        cm_list = [cm.extend_results(qreq_) for cm in cm_list]
        name_ranks = [cm.get_name_ranks([cm.qnid])[0] for cm in cm_list]
        bins = np.arange(len(qreq_.dnids))
        hist = np.histogram(name_ranks, bins=bins)[0]
        cdf = np.cumsum(hist) / sum(hist)
        results.append((cdf, ut.update_dict(info.copy(), {'pcfg': config})))

        # LNBNN pipeline
        cfgdict = {}
        cdf = _ranking_cdf(ibs, qaids, daids, cfgdict)
        results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})))

        expt_name = 'smk'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_nsum(self):
        """
        python -m wbia Chap3.measure nsum --dbs=GZ_Master1,PZ_Master1
        python -m wbia Chap3.draw nsum --dbs=GZ_Master1,PZ_Master1 --diskshow

        from wbia.scripts.thesis import *
        self = Chap3('GZ_Master1')
        self = Chap3('PZ_Master1')
        self = Chap3('PZ_MTEST')
        self._precollect()
        """
        ibs = self.ibs
        qaids, daids_list, info_list = Sampler._varied_inputs(
            self.ibs, self.aids_pool, denc_per_name=[1, 2, 3], extra_dbsize_fracs=[1]
        )

        base = {'query_rotation_heuristic': True}
        cfgdict1 = ut.dict_union(
            base, {'score_method': 'nsum', 'prescore_method': 'nsum'}
        )
        cfgdict2 = ut.dict_union(
            base, {'score_method': 'csum', 'prescore_method': 'csum'}
        )
        results = []
        for count, (daids, info) in enumerate(zip(daids_list, info_list), start=1):
            cdf1 = _ranking_cdf(ibs, qaids, daids, cfgdict1)
            results.append((cdf1, ut.update_dict(info.copy(), {'pcfg': cfgdict1})))
            cdf2 = _ranking_cdf(ibs, qaids, daids, cfgdict2)
            results.append((cdf2, ut.update_dict(info.copy(), {'pcfg': cfgdict2})))

        if False:
            self._precollect()
            ibs = self.ibs
            qaids, daids_list, info_list = Sampler._varied_inputs(
                self.ibs, self.aids_pool, denc_per_name=[1, 2, 3], extra_dbsize_fracs=[1]
            )
            # Check dpername issue
            base = {'query_rotation_heuristic': False, 'K': 1, 'sv_on': True}
            cfgdict1 = ut.dict_union(
                base, {'score_method': 'nsum', 'prescore_method': 'nsum'}
            )
            cfgdict2 = ut.dict_union(
                base, {'score_method': 'csum', 'prescore_method': 'csum'}
            )

            qaids = [2491]

            info = {}
            daids = daids_list[0]
            a = ibs.annots(daids)
            daids = a.compress(ut.flag_unique_items(a.nids)).aids

            while True:
                qreq1_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict1)
                qreq2_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict2)
                cm_list1 = qreq1_.execute(use_cache=False)
                cm_list2 = qreq2_.execute(use_cache=False)
                cm1 = cm_list1[0]
                cm2 = cm_list2[0]
                assert cm1 == cm2

            # cm_list1 = [cm.extend_results(qreq1_) for cm in cm_list1]
            # cm_list2 = [cm.extend_results(qreq2_) for cm in cm_list2]

            # cm_list1 = [cm.compress_results() for cm in cm_list1]
            # cm_list2 = [cm.compress_results() for cm in cm_list2]

            name_ranks1 = [cm.get_name_ranks([cm.qnid])[0] for cm in cm_list1]
            name_ranks2 = [cm.get_name_ranks([cm.qnid])[0] for cm in cm_list2]

            idxs = np.where(np.array(name_ranks1) != np.array(name_ranks2))[0]
            print('idxs = %r' % (idxs,))
            print('ranks1 = {}'.format(ut.take(name_ranks1, idxs)))
            print('ranks2 = {}'.format(ut.take(name_ranks2, idxs)))
            if len(idxs) > 0:
                cm1 = cm_list1[idxs[0]]  # NOQA
                cm2 = cm_list2[idxs[0]]  # NOQA

        expt_name = 'nsum'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_dbsize(self):
        ibs = self.ibs
        qaids, daids_list, info_list = Sampler._varied_inputs(
            self.ibs, self.aids_pool, denc_per_name=[1, 2], extra_dbsize_fracs=[0, 1.0]
        )
        cfgdict = {
            'query_rotation_heuristic': True,
        }
        results = []
        for daids, info in zip(daids_list, info_list):
            info = info.copy()
            cdf = _ranking_cdf(ibs, qaids, daids, cfgdict)
            results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})))

        expt_name = 'dsize'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)
        cdfs, infos = zip(*results)

    def measure_kexpt(self):
        ibs = self.ibs
        qaids, daids_list, info_list = Sampler._varied_inputs(
            self.ibs, self.aids_pool, denc_per_name=[1, 2], extra_dbsize_fracs=[0, 1.0]
        )
        cfg_grid = {
            'query_rotation_heuristic': True,
            'K': [1, 2, 4, 6],
        }
        results = []
        for cfgdict in ut.all_dict_combinations(cfg_grid):
            for daids, info in zip(daids_list, info_list):
                cdf = _ranking_cdf(ibs, qaids, daids, cfgdict)
                results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})))

        expt_name = 'kexpt'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_dbstats(self):
        if self.ibs is None:
            self._precollect()

        # self.ibs.print_annot_stats(self.aids_pool)
        annots = self.ibs.annots(self.aids_pool)

        encounters = annots.group(annots.encounter_text)[1]
        nids = ut.take_column(self.ibs._annot_groups(encounters).nids, 0)
        nid_to_enc = ut.group_items(encounters, nids)

        single_encs = {nid: e for nid, e in nid_to_enc.items() if len(e) == 1}
        multi_encs = {nid: e for nid, e in nid_to_enc.items() if len(e) > 1}
        multi_aids = ut.flatten(ut.flatten(multi_encs.values()))

        enc_deltas = []
        for encs_ in nid_to_enc.values():
            a = encs_[0]
            times = a.image_unixtimes_asfloat
            delta = times.max() - times.min()
            enc_deltas.append(delta)
        max_enc_timedelta = max(enc_deltas)
        print('max enc timedelta = %r' % (ut.get_unix_timedelta_str(max_enc_timedelta)))

        multi_stats = self.ibs.get_annot_stats_dict(multi_aids)
        multi_stats['enc_per_name']

        enc_info = ut.odict()
        enc_info['species_nice'] = self.species_nice
        enc_info['n_singleton_names'] = len(single_encs)
        enc_info['n_resighted_names'] = len(multi_encs)
        enc_info['n_encounter_per_resighted_name'] = ave_str(
            *ut.take(multi_stats['enc_per_name'], ['mean', 'std']), precision=1
        )
        n_annots_per_enc = ut.lmap(len, encounters)
        enc_info['n_annots_per_encounter'] = ave_str(
            np.mean(n_annots_per_enc), np.std(n_annots_per_enc), precision=1,
        )
        enc_info['n_annots'] = sum(n_annots_per_enc)

        # qual_info = ut.odict()
        qual_info = ut.dict_hist(annots.quality_texts)
        qual_info['None'] = qual_info.pop('UNKNOWN', 0)
        qual_info['None'] += qual_info.pop(None, 0)
        qual_info['species_nice'] = self.species_nice

        view_info = ut.dict_hist(annots.viewpoint_code)
        view_info['None'] = view_info.pop('UNKNOWN', 0)
        view_info['None'] += view_info.pop(None, 0)
        view_info['species_nice'] = self.species_nice

        info = {
            'enc': enc_info,
            'qual': qual_info,
            'view': view_info,
        }

        expt_name = ut.get_stack_frame().f_code.co_name.replace('measure_', '')
        expt_name = 'dbstats'
        self.expt_results[expt_name] = info
        ut.save_data(join(self.dpath, expt_name + '.pkl'), info)
        return info


@ut.reloadable_class
class Chap3Draw(object):
    def draw_baseline(self):
        mpl.rcParams.update(TMP_RC)
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        results = self.ensure_results(expt_name)
        cdfs, infos = list(zip(*results))
        baseline_cdf = cdfs[0]
        fig = plot_cmcs([baseline_cdf], ['baseline'], fnum=1)
        fig.set_size_inches([W, H * 0.6])
        qsizes = ut.take_column(infos, 'qsize')
        dsizes = ut.take_column(infos, 'dsize')
        assert ut.allsame(qsizes) and ut.allsame(dsizes)
        nonvaried_text = 'qsize={}, dsize={}'.format(qsizes[0], dsizes[0])
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_smk(self):
        """
        wbia Chap3.measure smk --dbs=GZ_Master1,PZ_Master1
        wbia Chap3.draw smk --dbs=GZ_Master1,PZ_Master1
        """
        mpl.rcParams.update(TMP_RC)
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        results = self.ensure_results(expt_name)
        cdfs, infos = list(zip(*results))
        labels = ['smk', 'baseline']
        fig = plot_cmcs(cdfs, labels, fnum=1, ymin=0.5)
        fig.set_size_inches([W, H * 0.6])
        qsizes = ut.take_column(infos, 'qsize')
        dsizes = ut.take_column(infos, 'dsize')
        assert ut.allsame(qsizes) and ut.allsame(dsizes)
        nonvaried_text = 'qsize={}, dsize={}'.format(qsizes[0], dsizes[0])
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_foregroundness(self):
        """
        wbia Chap3.measure foregroundness --dbs=GZ_Master1,PZ_Master1
        wbia Chap3.draw foregroundness --dbs=GZ_Master1,PZ_Master1
        """
        mpl.rcParams.update(TMP_RC)
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        results = self.ensure_results(expt_name)
        cdfs, infos = list(zip(*results))
        labels = ['fg=F', 'fg=T']
        fig = plot_cmcs(cdfs, labels, fnum=1, ymin=0.5)
        fig.set_size_inches([W, H * 0.6])
        qsizes = ut.take_column(infos, 'qsize')
        dsizes = ut.take_column(infos, 'dsize')
        assert ut.allsame(qsizes) and ut.allsame(dsizes)
        nonvaried_text = 'qsize={}, dsize={}'.format(qsizes[0], dsizes[0])
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_invar(self):
        """
        wbia Chap3.measure invar --dbs=GZ_Master1,PZ_Master1
        wbia Chap3.draw invar --dbs=GZ_Master1,PZ_Master1
        """
        mpl.rcParams.update(TMP_RC)
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        results = self.ensure_results(expt_name)
        ALIAS_KEYS = ut.invert_dict(
            {
                'RI': 'rotation_invariance',
                'AI': 'affine_invariance',
                'QRH': 'query_rotation_heuristic',
            }
        )
        results = [
            (c, i) for c, i in results if not i['pcfg'].get('rotation_invariance', False)
        ]
        cdfs, infos = list(zip(*results))
        pcfgs = ut.take_column(infos, 'pcfg')

        for p in pcfgs:
            del p['rotation_invariance']

        labels = [ut.get_cfg_lbl(ut.map_keys(ALIAS_KEYS, pcfg))[1:] for pcfg in pcfgs]
        labels = ut.lmap(label_alias, labels)
        fig = plot_cmcs(cdfs, labels, fnum=1, ymin=0.5)
        fig.set_size_inches([W, H * 0.6])
        qsizes = ut.take_column(infos, 'qsize')
        dsizes = ut.take_column(infos, 'dsize')
        assert ut.allsame(qsizes) and ut.allsame(dsizes)
        nonvaried_text = 'qsize={}, dsize={}'.format(qsizes[0], dsizes[0])
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_nsum(self):
        """
        wbia Chap3.measure nsum --dbs=GZ_Master1,PZ_Master1
        wbia Chap3.draw nsum --dbs=GZ_Master1,PZ_Master1
        """
        mpl.rcParams.update(TMP_RC)
        # expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        expt_name = 'nsum'
        results = self.ensure_results(expt_name)
        cdfs, infos = list(zip(*results))
        # pcfgs = ut.take_column(infos, 'pcfg')
        alias = {
            'nsum': 'fmech',
            'csum': 'amech',
        }
        labels = [
            alias[x['pcfg']['score_method']] + ',dpername={}'.format(x['t_dpername'])
            for x in infos
        ]
        fig = plot_cmcs(cdfs, labels, fnum=1, ymin=0.5)
        qsizes = ut.take_column(infos, 'qsize')
        dsizes = ut.take_column(infos, 'dsize')
        assert ut.allsame(qsizes) and ut.allsame(dsizes)
        nonvaried_text = 'qsize={}, dsize={}'.format(qsizes[0], dsizes[0])
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())
        fig.set_size_inches([W, H * 0.6])
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_nsum_simple(self):
        """
        wbia Chap3.measure nsum --dbs=GZ_Master1,PZ_Master1
        wbia Chap3.draw nsum --dbs=GZ_Master1,PZ_Master1

        Ignore:
            >>> from wbia.scripts.thesis import *  # NOQA
            >>> self = Chap3('PZ_Master1')
        """
        raise Exception('hacked')
        mpl.rcParams.update(TMP_RC)
        # expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        fpath = '/home/joncrall/latex/crall-thesis-2017/figures3/PZ_Master1/nsum.pkl'
        results = ut.load_data(fpath)
        # results = self.ensure_results(expt_name)
        cdfs, infos = list(zip(*results))
        # pcfgs = ut.take_column(infos, 'pcfg')
        alias = {
            'nsum': 'fmech',
            'csum': 'amech',
        }
        labels = [
            alias[x['pcfg']['score_method']] + ',dpername={}'.format(x['t_dpername'])
            for x in infos
        ]
        # hack
        cdfs = cdfs[::2]
        labels = labels[::2]
        infos = infos[::2]
        fig = plot_cmcs(cdfs, labels, fnum=1, ymin=0.5)
        qsizes = ut.take_column(infos, 'qsize')
        dsizes = ut.take_column(infos, 'dsize')
        assert ut.allsame(qsizes) and ut.allsame(dsizes)
        nonvaried_text = 'qsize={}, dsize={}'.format(qsizes[0], dsizes[0])
        pt.relative_text('lowerleft', nonvaried_text, ax=pt.gca())
        fig.set_size_inches([W, H * 0.6])
        ut.ensuredir(self.dpath)
        fpath = join(self.dpath, 'nsum_simple.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_kexpt(self):
        """
        wbia Chap3.measure kexpt --dbs=GZ_Master1,PZ_Master1
        wbia Chap3.draw kexpt --dbs=GZ_Master1,PZ_Master1 --diskshow
        """
        mpl.rcParams.update(TMP_RC)
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        results = self.ensure_results(expt_name)
        # results = self.expt_results[expt_name]
        cdfs, infos = list(zip(*results))
        pcfgs = ut.take_column(infos, 'pcfg')
        df = pd.DataFrame.from_records(infos)
        df['cdfs'] = cdfs
        df['K'] = ut.take_column(pcfgs, 'K')
        import wbia.plottool as pt

        # groups = list(df.groupby(('dsize', 't_denc_pername')))
        df = df[df['K'] != 10]

        fig = pt.figure(fnum=1)
        groups = list(df.groupby(('dsize')))
        pnum_ = pt.make_pnum_nextgen(nCols=1, nSubplots=len(groups))
        for val, df_group in groups:
            # print('---')
            # print(df_group)
            relevant_df = df_group[['K', 'qsize', 'dsize', 't_dpername']]
            relevant_df = relevant_df.rename(columns={'t_dpername': 'dpername'})
            relevant_cfgs = [
                ut.order_dict_by(d, relevant_df.columns.tolist())
                for d in relevant_df.to_dict('records')
            ]
            nonvaried_kw, varied_kws = ut.partition_varied_cfg_list(relevant_cfgs)
            labels_ = [ut.get_cfg_lbl(kw)[1:] for kw in varied_kws]
            cdfs_ = df_group['cdfs'].values
            plot_cmcs(cdfs_, labels_, fnum=1, pnum=pnum_(), ymin=0.5)
            ax = pt.gca()
            nonvaried_text = ut.get_cfg_lbl(nonvaried_kw)[1:]
            # ax.set_title(nonvaried_text)
            pt.relative_text('lowerleft', nonvaried_text, ax=ax)

        pt.adjust_subplots(
            top=0.9, bottom=0.1, left=0.12, right=0.9, hspace=0.3, wspace=0.2
        )
        fig.set_size_inches([W, H * 1.9])
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_all(self):
        """
        CommandLine:
            python -m wbia Chap3.draw_all --dbs=GZ_Master1,PZ_Master1,GIRM_Master1

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *  # NOQA
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = Chap3(dbname)
            >>>     self.draw_all()
        """
        self.ensure_results()

        # if 'baseline' in self.expt_results:
        #     self.draw_baseline()

        if 'PZ' in self.dbname or 'GZ' in self.dbname:
            expts = ['foregroundness', 'invar', 'smk', 'nsum', 'kexpt']
            for expt_name in expts:
                if expt_name in self.expt_results:
                    try:
                        getattr(self, 'draw_' + expt_name)()
                    except KeyError:
                        getattr(self, 'measure_' + expt_name)()
                        getattr(self, 'draw_' + expt_name)()
            # if 'invar' in self.expt_results:
            #     self.draw_invar()
            # if 'smk' in self.expt_results:
            #     self.draw_smk()
            # if 'nsum' in self.expt_results:
            #     self.draw_nsum()
            # if 'kexpt' in self.expt_results:
            #     self.draw_kexpt()

    def draw_time_distri(self):
        """
        CommandLine:
            python -m wbia Chap3.draw_time_distri --dbs=GZ_Master1,PZ_Master1,GIRM_MasterV
            python -m wbia Chap3.draw_time_distri --dbs=GIRM_Master1
            python -m wbia Chap3.draw_time_distri --dbs=GZ_Master1
            python -m wbia Chap3.draw_time_distri --dbs=PZ_Master1
            python -m wbia Chap3.draw_time_distri --dbs=humpbacks_fb

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *  # NOQA
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = Chap3(dbname)
            >>>     self.draw_time_distri()
        """
        import matplotlib as mpl

        mpl.rcParams.update(TMP_RC)
        if self.ibs is None:
            self._precollect()
        ibs = self.ibs
        annots = ibs.annots(self.aids_pool)
        images = ibs.images(set(annots.gids))
        unixtimes_ = images.unixtime_asfloat
        num_nan = np.isnan(unixtimes_).sum()
        num_total = len(unixtimes_)
        unixtimes = unixtimes_[~np.isnan(unixtimes_)]

        mintime = vt.safe_min(unixtimes)
        maxtime = vt.safe_max(unixtimes)
        unixtime_domain = np.linspace(mintime, maxtime, 1000)

        import matplotlib as mpl

        mpl.rcParams.update(TMP_RC)

        from sklearn.neighbors.kde import KernelDensity

        bw = ut.get_argval('--bw', default=None)
        day = 60 * 60 * 24
        if bw is not None:
            pass
        elif 'GIRM' in self.dbname:
            bw = day / 4
        elif 'GZ' in self.dbname:
            bw = day * 30
        elif 'PZ' in self.dbname:
            bw = day * 30
        elif 'humpbacks_fb' in self.dbname:
            bw = day * 30
        else:
            from sklearn.model_selection import RandomizedSearchCV

            space = np.linspace(day, day * 14, 14).tolist()
            grid_params = {'bandwidth': space}
            searcher = ut.partial(RandomizedSearchCV, n_iter=5, n_jobs=8)
            print('Searching for best bandwidth')
            grid = searcher(
                KernelDensity(kernel='gaussian'), grid_params, cv=2, verbose=0
            )
            grid.fit(unixtimes[:, None])
            bw = grid.best_params_['bandwidth']
            print('bw = %r' % (bw,))
            print('bw(days) = %r' % (bw / day,))

        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(unixtimes[:, None])
        log_density = kde.score_samples(unixtime_domain[:, None])
        density = np.exp(log_density)
        ydata = density.copy()
        # emphasize smaller values
        ydata /= ydata.max()
        ydata = np.sqrt(ydata)
        xdata = unixtime_domain
        xdata_ts = ut.lmap(ut.unixtime_to_datetimeobj, xdata)

        pt.multi_plot(
            xdata_ts,
            [ydata],
            label_list=['time'],
            alpha=0.7,
            fnum=1,
            pnum=(1, 1, 1),
            ymin=0,
            fill=True,
            marker='',
            xlabel='Date',
            ylabel='# images',
            num_xticks=5,
            use_legend=False,
        )
        infos = []
        if num_nan > 0:
            infos.append('#nan={}'.format(num_nan))
            infos.append('#total={}'.format(num_total))
        else:
            infos.append('#total={}'.format(num_total))
        text = '\n'.join(infos)
        pt.relative_text((0.02, 0.02), text, halign='left', valign='top')

        ax = pt.gca()
        fig = pt.gcf()
        ax.set_yticks([])
        if False:
            icon = ibs.get_database_icon()
            pt.overlay_icon(
                icon,
                coords=(0, 1),
                bbox_alignment=(0, 1),
                as_artist=1,
                max_asize=(100, 200),
            )
        pt.adjust_subplots(top=0.9, bottom=0.1, left=0.12, right=0.9)
        fig.set_size_inches([W, H * 0.4])
        fpath = join(self.dpath, 'timedist.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath


@ut.reloadable_class
class Chap3(DBInputs, Chap3Draw, Chap3Measures):
    base_dpath = ut.truepath('~/latex/crall-thesis-2017/figures3')

    def _setup(self):
        self._precollect()

    @classmethod
    def run_all(cls):
        """
        CommandLine:
            python -m wbia Chap3.run_all
        """
        agg_dbnames = ['PZ_Master1', 'GZ_Master1', 'GIRM_Master1', 'humpbacks_fb']
        agg_dbnames = agg_dbnames[::-1]

        for dbname in agg_dbnames:
            self = cls(dbname)
            self.measure_all()
            self.draw_time_distri()

        cls.agg_dbstats()
        cls.draw_agg_baseline()

    def measure_all(self):
        """
        Example:
            from wbia.scripts.thesis import *
            self = Chap3('PZ_Master1')
            self.measure_all()
            self = Chap3('GZ_Master1')
            self.measure_all()
            self = Chap3('GIRM_Master1')
            self.measure_all()
        """
        if self.ibs is None:
            self._precollect()
        self.measure_baseline()
        if self.dbname in {'PZ_Master1', 'GZ_Master1'}:
            self.measure_foregroundness()
            self.measure_smk()
            self.measure_nsum()
            # self.measure_dbsize()
            self.measure_kexpt()
            self.measure_invar()

    @classmethod
    def agg_dbstats(cls):
        """
        CommandLine:
            python -m wbia Chap3.agg_dbstats
            python -m wbia Chap3.measure_dbstats

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *  # NOQA
            >>> result = Chap3.agg_dbstats()
            >>> print(result)
        """
        agg_dbnames = ['PZ_Master1', 'GZ_Master1', 'GIRM_Master1', 'humpbacks_fb']
        infos = ut.ddict(list)
        for dbname in agg_dbnames:
            self = cls(dbname)
            info = self.ensure_results('dbstats')
            infos['enc'].append(info['enc'])
            infos['qual'].append(info['qual'])
            infos['view'].append(info['view'])
            # labels.append(self.species_nice.capitalize())

        alias = {
            'species_nice': 'database',
            'n_singleton_names': 'names (singleton)',
            'n_resighted_names': 'names (resighted)',
            'n_encounter_per_resighted_name': 'encounters per name (resighted)',
            'n_annots_per_encounter': 'annots per encounter',
            'n_annots': 'annots',
        }
        alias = ut.map_vals(upper_one, alias)
        df = pd.DataFrame(infos['enc']).rename(columns=alias)
        df = df.set_index('Database')
        df.index.name = None
        df.index = ut.emap(upper_one, df.index)
        alias = ut.map_vals(upper_one, alias)
        tabular = Tabular(df, colfmt='numeric')
        tabular.theadify = 16
        enc_text = tabular.as_tabular()
        print(enc_text)

        df = pd.DataFrame(infos['qual'])
        df = df.rename(columns={'species_nice': 'Database'})
        df = df.reindex(
            ut.partial_order(
                df.columns, ['Database', 'excellent', 'good', 'ok', 'poor', 'None']
            ),
            axis=1,
        )
        df = df.set_index('Database')
        df.index.name = None
        df.index = ut.emap(upper_one, df.index)
        df[pd.isnull(df)] = 0
        df = df.astype(np.int)
        df.columns = ut.emap(upper_one, df.columns)
        tabular = Tabular(df, colfmt='numeric')
        qual_text = tabular.as_tabular()
        print(qual_text)

        df = pd.DataFrame(infos['view'])
        df = df.rename(
            columns={
                'species_nice': 'Database',
                'back': 'B',
                'left': 'L',
                'right': 'R',
                'front': 'F',
                'backleft': 'BL',
                'backright': 'BR',
                'frontright': 'FR',
                'frontleft': 'FL',
            }
        )
        order = ut.partial_order(
            df.columns, ['Database', 'BL', 'L', 'FL', 'F', 'FR', 'R', 'BR', 'B', 'None']
        )
        df = df.reindex(order, axis=1)
        df = df.set_index('Database')
        df.index.name = None
        df.index = ut.emap(upper_one, df.index)
        df[pd.isnull(df)] = 0
        df = df.astype(np.int)
        tabular = Tabular(df, colfmt='numeric')
        view_text = tabular.as_tabular()
        print(view_text)

        ut.render_latex(
            enc_text,
            dpath=self.base_dpath,
            fname='agg-enc',
            preamb_extra=['\\usepackage{makecell}'],
        )

        ut.render_latex(
            view_text,
            dpath=self.base_dpath,
            fname='agg-view',
            preamb_extra=['\\usepackage{makecell}'],
        )

        ut.render_latex(
            qual_text,
            dpath=self.base_dpath,
            fname='agg-qual',
            preamb_extra=['\\usepackage{makecell}'],
        )

        ut.write_to(join(cls.base_dpath, 'agg-enc.tex'), enc_text)
        ut.write_to(join(cls.base_dpath, 'agg-view.tex'), view_text)
        ut.write_to(join(cls.base_dpath, 'agg-qual.tex'), qual_text)

    @classmethod
    def draw_agg_baseline(cls):
        """
        CommandLine:
            python -m wbia Chap3.draw_agg_baseline --diskshow

        Example:
            >>> # SCRIPT
            >>> from wbia.scripts.thesis import *  # NOQA
            >>> Chap3.draw_agg_baseline()
        """
        agg_dbnames = ['GZ_Master1', 'PZ_Master1', 'GIRM_Master1', 'humpbacks_fb']
        cdfs = []
        labels = []
        for dbname in agg_dbnames:
            self = cls(dbname)
            results = self.ensure_results('baseline')
            cdf, config = results[0]
            dsize = config['dsize']
            qsize = config['t_n_names']
            baseline_cdf = results[0][0]
            cdfs.append(baseline_cdf)
            labels.append('{},qsize={},dsize={}'.format(self.species_nice, qsize, dsize))
            # labels.append(self.species_nice.capitalize())

        mpl.rcParams.update(TMP_RC)
        fig = plot_cmcs(cdfs, labels, fnum=1, ymin=0.5)
        fig.set_size_inches([W, H * 1.5])
        fpath = join(cls.base_dpath, 'agg-baseline.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fpath)


class Sampler(object):
    @staticmethod
    def _same_occur_split(ibs, aids):
        """
        >>> from wbia.scripts.thesis import *
        >>> self = Chap3('PZ_Master1')
        >>> self._precollect()
        """
        annots = ibs.annots(aids)
        # occurrences = ibs._annot_groups(annots.group(annots.occurrence_text)[1])
        encounters = ibs._annot_groups(annots.group(annots.encounter_text)[1])

        nid_to_splits = ut.ddict(list)
        # Find the biggest occurrences and pick an annotation from that
        # occurrence to be sampled
        occur_to_encs = ut.group_items(
            encounters, ut.take_column(encounters.occurrence_text, 0)
        )
        occur_encs = ut.sortedby(
            list(occur_to_encs.values()), list(map(len, occur_to_encs.values()))
        )[::-1]
        for encs in occur_encs:
            for enc in encs:
                sortx = ut.argsort(enc.qualities)[::-1]
                annot = enc[sortx[0]]
                if len(nid_to_splits[annot.nid]) < 2:
                    nid_to_splits[annot.nid].append(annot.aid)

        rng = ut.ensure_rng(0)
        pyrng = random.Random(rng.randint(sys.maxsize))

        qaids = []
        dname_encs = []
        confusor_pool = []
        for nid, aids_ in nid_to_splits.items():
            if len(aids_) < 2:
                confusor_pool.extend(aids_)
            else:
                pyrng.shuffle(aids_)
                qaids.append(aids_[0])
                dname_encs.append([[aids_[1]]])
        confusor_pool = ut.shuffle(confusor_pool, rng=0)
        return qaids, dname_encs, confusor_pool

    @staticmethod
    def _intra_enc(ibs, aids):
        # Make a query / database set for each occurrence
        # ibs = self.ibs
        # aids = self.aids_pool
        annots = ibs.annots(aids)
        # occurrences = ibs._annot_groups(annots.group(annots.occurrence_text)[1])
        encounters = ibs._annot_groups(annots.group(annots.encounter_text)[1])

        # rng = ut.ensure_rng(0)
        # pyrng = random.Random(rng.randint(sys.maxsize))

        # Find the biggest occurrences and pick an annotation from that
        # occurrence to be sampled
        occurrences = ut.group_items(
            encounters, ut.take_column(encounters.occurrence_text, 0)
        )
        occurrences = ut.map_vals(ibs._annot_groups, occurrences)

        occur_nids = {o: set(ut.flatten(encs.nids)) for o, encs in occurrences.items()}

        # Need to find multiple disjoint exact covers of the nids
        # Greedy solution because this is NP-hard
        from wbia.algo.graph import nx_dynamic_graph

        G = nx_dynamic_graph.DynConnGraph()
        G.add_nodes_from(occur_nids.keys())
        occur_ids = ut.sortedby(occur_nids.keys(), ut.lmap(len, occur_nids.values()))[
            ::-1
        ]
        current_combos = {
            frozenset(G.connected_to(o1)): occur_nids[o1] for o1 in occur_ids
        }
        for o1, o2 in ut.combinations(occur_ids, 2):
            if G.node_label(o1) == G.node_label(o2):
                continue
            cc1 = frozenset(G.connected_to(o1))
            cc2 = frozenset(G.connected_to(o2))
            nids1 = current_combos[cc1]
            nids2 = current_combos[cc2]
            if nids1.isdisjoint(nids2):
                G.add_edge(o1, o2)
                del current_combos[cc1]
                del current_combos[cc2]
                current_combos[frozenset(cc1.union(cc2))] = nids1.union(nids2)

        # Pick the top few occurrence groups with the most names
        grouped_occurs = list(map(frozenset, G.connected_components()))
        group_size = ut.lmap(len, list(grouped_occurs))
        top_groups = ut.sortedby(grouped_occurs, group_size)[::-1][0:4]

        samples = []

        for os in top_groups:
            encs = ut.flatten(occurrences[o].aids for o in os)
            encs = ut.lmap(ibs.annots, encs)
            qaids = []
            daids = []
            for enc in encs:
                if len(enc) == 1:
                    daids.extend(enc.aids)
                else:
                    daids.extend(enc.aids)
                    qaids.extend(enc.aids)
            sample = SplitSample(qaids, daids)
            samples.append(sample)
        return samples

    @staticmethod
    def _same_enc_split(ibs, aids):
        """
        >>> from wbia.scripts.thesis import *
        >>> self = Chap3('PZ_Master1')
        >>> self._precollect()
        """
        annots = ibs.annots(aids)
        # occurrences = ibs._annot_groups(annots.group(annots.occurrence_text)[1])
        encounters = ibs._annot_groups(annots.group(annots.encounter_text)[1])

        rng = ut.ensure_rng(0)
        pyrng = random.Random(rng.randint(sys.maxsize))

        nid_to_splits = ut.ddict(list)
        # Find the biggest occurrences and pick an annotation from that
        # occurrence to be sampled
        occur_to_encs = ut.group_items(
            encounters, ut.take_column(encounters.occurrence_text, 0)
        )
        occur_encs = ut.sortedby(
            list(occur_to_encs.values()), list(map(len, occur_to_encs.values()))
        )[::-1]
        for encs in occur_encs:
            for enc in encs:
                nid = enc.nids[0]
                if len(nid_to_splits[nid]) == 0:
                    chosen = pyrng.sample(enc.aids, min(len(enc), 2))
                    nid_to_splits[nid].extend(chosen)

        qaids = []
        dname_encs = []
        confusor_pool = []
        for nid, aids_ in nid_to_splits.items():
            if len(aids_) < 2:
                confusor_pool.extend(aids_)
            else:
                pyrng.shuffle(aids_)
                qaids.append(aids_[0])
                dname_encs.append([[aids_[1]]])
        confusor_pool = ut.shuffle(confusor_pool, rng=0)
        return qaids, dname_encs, confusor_pool

    def _rand_splits(ibs, aids, qenc_per_name, denc_per_name_, annots_per_enc):
        """ This can be used for cross validation """
        # Find a split of query/database encounters and confusors
        from wbia.init.filter_annots import encounter_crossval

        enc_splits, nid_to_confusors = encounter_crossval(
            ibs,
            aids,
            qenc_per_name=1,
            annots_per_enc=1,
            denc_per_name=denc_per_name_,
            rebalance=True,
            rng=0,
            early=True,
        )

        qname_encs, dname_encs = enc_splits[0]
        qaids = sorted(ut.flatten(ut.flatten(qname_encs)))
        confusor_pool = ut.flatten(ut.flatten(nid_to_confusors.values()))
        confusor_pool = ut.shuffle(confusor_pool, rng=0)
        return qaids, dname_encs, confusor_pool

    @staticmethod
    def _alt_splits(
        ibs, aids, qenc_per_name, denc_per_name_, annots_per_enc, viewpoint_aware=False
    ):
        """
        This cannot be used for cross validation

        Notes:

            (a) single encounter experiments are structured somewhat like this:
                (of course this script is more general than this)

                * For each name with more than one encounter

                * Choose a random encounter, and select the highest quality
                annotation as the single query annotation.

                * For each other encounter the best annotation that is comparable
                (close to the same viewpoint) to the query. If no other encounter
                satisfies this then skip this name (dont add a query or database
                annotation).

                * Of the remaining encounters choose a random annotation to belong
                to the database.

                * For each other name, that was not selected to form a
                query/database pair, add all annotations to the database as
                distractors.

            (b) with multiple exemplars in the database:

                * Follow the same steps above, but now if there are not at
                least N valid database encounters, we ignore the query/database
                pair.

                * Multiple sets of daids are generated (each with a different
                number of exempars per query), but the query set remains the
                same and consistent across different runs of this experiment.
        """
        # Group annotations by encounter
        # from wbia.other import ibsfuncs
        # primary_view = ibsfuncs.get_primary_species_viewpoint(ibs.get_primary_database_species())

        annots = ibs.annots(aids)
        encounters = ibs._annot_groups(annots.group(annots.encounter_text)[1])
        enc_nids = ut.take_column(encounters.nids, 0)
        nid_to_encs = ut.group_items(encounters, enc_nids)
        rng = ut.ensure_rng(0)
        pyrng = random.Random(rng.randint(sys.maxsize))

        n_need = qenc_per_name + denc_per_name_

        confusor_encs = {}
        sample_splits = {}

        def choose_best(enc, num):
            if len(enc) > num:
                sortx = ut.argsort(enc.qualities)[::-1]
                subenc = enc.take(sortx[0:num])
            else:
                subenc = enc
            return subenc

        def _only_comparable(qsubenc, avail_dencs):
            from vtool import _rhomb_dist

            qviews = set(ut.flatten(qsubenc.viewpoint_code))
            comparable_encs = []
            for denc in avail_dencs:
                comparable = []
                for daid, dview in zip(denc.aids, denc.viewpoint_code):
                    for qview in qviews:
                        dist = _rhomb_dist.VIEW_CODE_DIST[(qview, dview)]
                        if np.isnan(dist) or dist < 2:
                            comparable.append(daid)
                if comparable:
                    comparable_encs.append(ibs.annots(comparable))
            return comparable_encs

        for nid, encs in nid_to_encs.items():
            if len(encs) < n_need:
                confusor_encs[nid] = encs
            else:
                if viewpoint_aware:
                    # Randomly choose queries
                    avail_qxs = list(range(len(encs)))
                    qencxs = pyrng.sample(avail_qxs, qenc_per_name)
                    qencs = ut.take(encs, qencxs)
                    qsubenc = ibs._annot_groups(
                        [choose_best(enc, annots_per_enc) for enc in qencs]
                    )

                    # Ensure the db annots are comparable to at least one query
                    avail_dencs = ut.take(encs, ut.setdiff(avail_qxs, qencxs))
                    comparable_encs = _only_comparable(qsubenc, avail_dencs)

                    if len(comparable_encs) >= denc_per_name_:
                        # If we still have enough, sample daids
                        dencs = pyrng.sample(comparable_encs, denc_per_name_)
                        dsubenc = ibs._annot_groups(
                            [choose_best(enc, annots_per_enc) for enc in dencs]
                        )
                        sample_splits[nid] = (qsubenc.aids, dsubenc.aids)
                    else:
                        # If we don't add to confusors
                        confusor_encs[nid] = encs
                else:
                    # For each name choose a query / database encounter.
                    chosen_encs = pyrng.sample(encs, n_need)
                    # Choose high quality annotations from each encounter
                    best_subencs = [
                        choose_best(enc, annots_per_enc) for enc in chosen_encs
                    ]
                    # ibs._annot_groups(best_subencs).aids
                    qsubenc = ibs._annot_groups(best_subencs[0:qenc_per_name])
                    dsubenc = ibs._annot_groups(best_subencs[qenc_per_name:])
                    sample_splits[nid] = (qsubenc.aids, dsubenc.aids)

        # if viewpoint_aware:
        #     for qenc, denc in sample_splits.values():
        #         q = ibs.annots(ut.flatten(qenc))
        #         d = ibs.annots(ut.flatten(denc))
        #         print(q.viewpoint_code, d.viewpoint_code)

        # make confusor encounters subject to the same constraints
        confusor_pool = []
        confname_encs = []
        for encs in confusor_encs.values():
            # new
            # chosen_encs = pyrng.sample(encs, min(len(encs), denc_per_name_))
            # rand_subencs = [pyrng.sample(enc.aids, annots_per_enc)
            #                 for enc in chosen_encs]
            # confname_encs.append(rand_subencs)

            # old
            confusor_pool.extend(ut.flatten([enc[0:annots_per_enc].aids for enc in encs]))

        qaids = ut.total_flatten(ut.take_column(sample_splits.values(), 0))
        dname_encs = ut.take_column(sample_splits.values(), 1)
        return qaids, dname_encs, confname_encs, confusor_pool

    @staticmethod
    def _varied_inputs(
        ibs,
        aids,
        denc_per_name=[1],
        extra_dbsize_fracs=None,
        method='alt',
        viewpoint_aware=None,
    ):
        """
        Vary num per name and total number of annots

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *  # NOQA
            >>> self = Chap3('PZ_Master1')
            >>> self._precollect()
            >>> ibs = self.ibs
            >>> aids = self.aids_pool
            >>> print('--------')
            >>> qaids, daids_list, info_list = self._varied_inputs(ibs, aids, [1], [1], method='same_occur')
            >>> print('qsize = %r' % (len(qaids),))
            >>> for info in info_list:
            >>>     print(ut.repr4(info))
            >>> print('--------')
            >>> qaids, daids_list, info_list = self._varied_inputs(ibs, aids, [1], [1], method='same_enc')
            >>> print('qsize = %r' % (len(qaids),))
            >>> for info in info_list:
            >>>     print(ut.repr4(info))
            >>> print('--------')
            >>> qaids, daids_list, info_list = self._varied_inputs(ibs, aids, [1, 2], [0])
            >>> print('qsize = %r' % (len(qaids),))
            >>> for info in info_list:
            >>>     print(ut.repr4(info))
            >>> print('--------')
            >>> qaids, daids_list, info_list = self._varied_inputs(ibs, aids, [3], [0, 1])
            >>> print('qsize = %r' % (len(qaids),))
            >>> for info in info_list:
            >>>     print(ut.repr4(info))

        Ignore:
            ibs, aids = self.ibs, self.aids_pool
            denc_per_name = 1
            denc_per_name = [1]
            extra_dbsize_fracs = None

            extra_dbsize_fracs = [0, .5, 1]
        """
        qenc_per_name = 1
        annots_per_enc = 1
        denc_per_name_ = max(denc_per_name)

        if method == 'alt':
            if viewpoint_aware is None:
                viewpoint_aware = False
            qaids, dname_encs, confname_encs, confusor_pool = Sampler._alt_splits(
                ibs,
                aids,
                qenc_per_name,
                denc_per_name_,
                annots_per_enc,
                viewpoint_aware=viewpoint_aware,
            )
        elif method == 'same_occur':
            assert viewpoint_aware is None, 'cannot specify viewpoint_aware here'
            assert denc_per_name_ == 1
            assert annots_per_enc == 1
            assert qenc_per_name == 1
            qaids, dname_encs, confusor_pool = Sampler._same_occur_split(ibs, aids)
        elif method == 'same_enc':
            assert viewpoint_aware is None, 'cannot specify viewpoint_aware here'
            qaids, dname_encs, confusor_pool = Sampler._same_enc_split(ibs, aids)
        else:
            raise KeyError(method)

        # Vary the number of database encounters in each sample
        target_daids_list = []
        target_info_list_ = []
        for num in denc_per_name:
            dname_encs_ = ut.take_column(dname_encs, slice(0, num))
            dnames_ = ut.lmap(ut.flatten, dname_encs_)
            daids_ = ut.flatten(dnames_)
            target_daids_list.append(daids_)
            name_lens = ut.lmap(len, dnames_)
            dpername = name_lens[0] if ut.allsame(name_lens) else np.mean(name_lens)
            target_info_list_.append(
                ut.odict(
                    [
                        ('qsize', len(qaids)),
                        ('t_n_names', len(dname_encs_)),
                        ('t_dpername', dpername),
                        ('t_denc_pername', num),
                        ('t_dsize', len(daids_)),
                    ]
                )
            )

        # confusor_names_matter = True
        # if confusor_names_matter:
        #     extra_pools = [
        #         ut.total_flatten(ut.take_column(confname_encs, slice(0, num)))
        #         for num in denc_per_name
        #     ]
        #     dbsize_list = ut.lmap(len, target_daids_list)
        #     max_dsize = max(dbsize_list)
        #     for num, daids_ in zip(denc_per_name, target_daids_list):
        #         num_take = max_dsize - len(daids_)
        #         print('num_take = %r' % (num_take,))

        #         confname_encs_ = ut.total_flatten(ut.take_column(confname_encs, slice(0, num)))
        #         confusor_pool_ = ut.total_flatten(confname_encs_)
        #         if num_take > len(confusor_pool_):
        #             # we need to siphon off valid queries to use them as
        #             # confusors
        #             raise AssertionError(
        #                 'have={}, need={}, not enough confusors for num={}'.format(
        #                     len(confusor_pool_), num_take, num
        #                 ))

        # Append confusors to maintain a constant dbsize in each base sample
        dbsize_list = ut.lmap(len, target_daids_list)
        max_dsize = max(dbsize_list)
        n_need = max_dsize - min(dbsize_list)
        n_extra_avail = len(confusor_pool) - n_need
        # assert len(confusor_pool) > n_need, 'not enough confusors'
        padded_daids_list = []
        padded_info_list_ = []
        for num, daids_, info_ in zip(
            denc_per_name, target_daids_list, target_info_list_
        ):
            num_take = max_dsize - len(daids_)

            assert num_take < len(confusor_pool), 'not enough confusors'
            pad_aids = confusor_pool[:num_take]

            new_aids = daids_ + pad_aids
            info_ = info_.copy()
            info_['n_pad'] = len(pad_aids)
            info_['pad_dsize'] = len(new_aids)
            padded_info_list_.append(info_)
            padded_daids_list.append(new_aids)

        # Vary the dbsize by appending extra confusors
        if extra_dbsize_fracs is None:
            extra_dbsize_fracs = [1.0]
        extra_fracs = np.array(extra_dbsize_fracs)
        n_extra_list = np.unique(extra_fracs * n_extra_avail).astype(np.int)
        daids_list = []
        info_list = []
        for n in n_extra_list:
            for daids_, info_ in zip(padded_daids_list, padded_info_list_):
                extra_aids = confusor_pool[len(confusor_pool) - n :]
                daids = sorted(daids_ + extra_aids)
                daids_list.append(daids)
                info = info_.copy()
                info['n_extra'] = len(extra_aids)
                info['dsize'] = len(daids)
                info_list.append(info)

        import pandas as pd

        verbose = 0
        if verbose:
            print(pd.DataFrame.from_records(info_list))
            print('#qaids = %r' % (len(qaids),))
            print('num_need = %r' % (n_need,))
            print('max_dsize = %r' % (max_dsize,))
            if False:
                for daids in daids_list:
                    ibs.print_annotconfig_stats(qaids, daids)
        return qaids, daids_list, info_list

    pass


class SplitSample(ut.NiceRepr):
    def __init__(sample, qaids, daids):
        sample.qaids = qaids
        sample.daids = daids

    def __nice__(sample):
        return 'nQaids={}, nDaids={}'.format(len(sample.qaids), len(sample.daids))


def _ranking_hist(ibs, qaids, daids, cfgdict):
    # Execute the ranking algorithm
    qaids = sorted(qaids)
    daids = sorted(daids)
    qreq_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict)
    cm_list = qreq_.execute()
    cm_list = [cm.extend_results(qreq_) for cm in cm_list]
    name_ranks = [cm.get_name_ranks([cm.qnid])[0] for cm in cm_list]
    # Measure rank probabilities
    bins = np.arange(len(qreq_.dnids))
    hist = np.histogram(name_ranks, bins=bins)[0]
    return hist


def _ranking_cdf(ibs, qaids, daids, cfgdict):
    hist = _ranking_hist(ibs, qaids, daids, cfgdict)
    cdf = np.cumsum(hist) / sum(hist)
    return cdf


def label_alias(k):
    k = k.replace('True', 'T')
    k = k.replace('False', 'F')
    return k


def feat_alias(k):
    # presentation values for feature dimension
    # k = k.replace('weighted_', 'wgt_')
    # k = k.replace('norm_x', 'x')
    # k = k.replace('norm_y', 'y')
    return k


def prepare_cdfs(cdfs, labels):
    cdfs = vt.pad_vstack(cdfs, fill_value=1)
    # Sort so the best is on top
    sortx = np.lexsort(cdfs.T[::-1])[::-1]
    cdfs = cdfs[sortx]
    labels = ut.take(labels, sortx)
    return cdfs, labels


def plot_cmcs(cdfs, labels, fnum=1, pnum=(1, 1, 1), ymin=0.4):
    cdfs, labels = prepare_cdfs(cdfs, labels)
    # Truncte to 20 ranks
    num_ranks = min(cdfs.shape[-1], 20)
    xdata = np.arange(1, num_ranks + 1)
    cdfs_trunc = cdfs[:, 0:num_ranks]
    label_list = [
        '%6.2f%% - %s' % (cdf[0] * 100, lbl) for cdf, lbl in zip(cdfs_trunc, labels)
    ]

    # ymin = .4
    num_yticks = (10 - int(ymin * 10)) + 1

    pt.multi_plot(
        xdata,
        cdfs_trunc,
        label_list=label_list,
        xlabel='rank',
        ylabel='match probability',
        use_legend=True,
        legend_loc='lower right',
        num_yticks=num_yticks,
        ymax=1,
        ymin=ymin,
        ypad=0.005,
        xmin=0.9,
        num_xticks=5,
        xmax=num_ranks + 1 - 0.5,
        pnum=pnum,
        fnum=fnum,
        rcParams=TMP_RC,
    )
    return pt.gcf()


def plot_cmcs2(cdfs, labels, fnum=1, **kwargs):
    fig = pt.figure(fnum=fnum)
    plot_cmcs(cdfs, labels, fnum=fnum, **kwargs)
    pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
    fig.set_size_inches([W, H])
    return fig


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.scripts.thesis
        python -m wbia.scripts.thesis --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
