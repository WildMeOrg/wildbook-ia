# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import wbia.plottool as pt
import utool as ut
from wbia.algo.verif import vsone
from wbia.scripts._thesis_helpers import DBInputs
from wbia.scripts.thesis import Sampler  # NOQA
from wbia.scripts._thesis_helpers import Tabular, upper_one, ave_str
from wbia.scripts._thesis_helpers import dbname_to_species_nice
from wbia.scripts._thesis_helpers import TMP_RC, W, H, DPI
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
import numpy as np  # NOQA
import pandas as pd
import ubelt as ub  # NOQA
import itertools as it
import matplotlib as mpl
from os.path import basename, join, splitext, exists  # NOQA
import wbia.constants as const
import vtool as vt
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

(print, rrr, profile) = ut.inject2(__name__)


CLF = 'VAMP'
LNBNN = 'LNBNN'


def turk_pz():
    import wbia

    ibs = wbia.opendb('GZ_Master1')
    infr = wbia.AnnotInference(ibs, aids='all')
    infr.reset_feedback('staging', apply=True)
    infr.relabel_using_reviews(rectify=True)
    # infr.apply_nondynamic_update()
    print(ut.repr4(infr.status()))

    infr.wbia_delta_info()

    infr.match_state_delta()
    infr.get_wbia_name_delta()

    infr.relabel_using_reviews(rectify=True)
    infr.write_wbia_annotmatch_feedback()
    infr.write_wbia_name_assignment()
    pass


@ut.reloadable_class
class GraphExpt(DBInputs):
    r"""

    TODO:
        - [ ] Experimental analysis of duration of each phase and state of
            graph.

        - [ ] Experimental analysis of phase 3, including how far we can get
            with automatic decision making and do we discover new merges?  If
            there are potential merges, can we run phase iii with exactly the
            same ordering as before:  ordering by probability for automatically
            decidable and then by positive probability for others.  This should
            work for phase 3 and therefore allow a clean combination of the
            three phases and our termination criteria.  I just thought of this
            so don't really have it written cleanly above.

        - [ ] Experimental analysis of choice of automatic decision thresholds.
            by lowering the threshold we increase the risk of mistakes.  Each
            mistake costs some number of manual reviews (perhaps 2-3), but if
            the frequency of errors is low then we could be saving ourselves a
            lot of manual reviews.

        \item OTHER SPECIES

    CommandLine:
        python -m wbia GraphExpt.measure all PZ_MTEST

    Ignore:
        >>> from wbia.scripts.postdoc import *
        >>> self = GraphExpt('PZ_MTEST')
        >>> self._precollect()
        >>> self._setup()
    """

    base_dpath = ut.truepath('~/Desktop/graph_expt')

    def _precollect(self):
        if self.ibs is None:
            _GraphExpt = ut.fix_super_reload(GraphExpt, self)
            super(_GraphExpt, self)._precollect()

        # Split data into a training and testing test
        ibs = self.ibs
        annots = ibs.annots(self.aids_pool)
        names = list(annots.group_items(annots.nids).values())
        ut.shuffle(names, rng=321)
        train_names, test_names = names[0::2], names[1::2]
        train_aids, test_aids = map(ut.flatten, (train_names, test_names))

        self.test_train = train_aids, test_aids

        params = {}
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

        config = ut.dict_union(self.const_dials)
        cfg_prefix = '{}_{}'.format(len(test_aids), len(train_aids))
        self._setup_links(cfg_prefix, config)

    def _setup(self):
        """
        python -m wbia GraphExpt._setup

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.postdoc import *
            >>> #self = GraphExpt('GZ_Master1')
            >>> self = GraphExpt('PZ_MTEST')
            >>> self = GraphExpt('PZ_Master1')
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

        # TODO: need more principled way of selecting thresholds
        # graph_thresh = res.get_pos_threshes('fpr', 0.01)
        graph_thresh = res.get_pos_threshes('fpr', 0.001)
        # rankclf_thresh = res.get_pos_threshes(fpr=0.01)

        # Load or create the deploy classifiers
        clf_dpath = ut.ensuredir((self.dpath, 'clf'))
        classifiers = pblm.ensure_deploy_classifiers(dpath=clf_dpath)

        sim_params = {
            'test_aids': test_aids,
            'train_aids': train_aids,
            'classifiers': classifiers,
            'graph_thresh': graph_thresh,
            # 'rankclf_thresh': rankclf_thresh,
            'const_dials': self.const_dials,
        }
        self.pblm = pblm
        self.sim_params = sim_params
        return sim_params

    def measure_all(self):
        self.measure_graphsim()

    @profile
    def measure_graphsim(self):
        """
        CommandLine:
            python -m wbia GraphExpt.measure graphsim GZ_Master1
            1

        Ignore:
            >>> from wbia.scripts.postdoc import *
            >>> #self = GraphExpt('PZ_MTEST')
            >>> #self = GraphExpt('GZ_Master1')
            >>> self = GraphExpt.measure('graphsim', 'PZ_Master1')
            >>> self = GraphExpt.measure('graphsim', 'GZ_Master1')
            >>> self = GraphExpt.measure('graphsim', 'PZ_MTEST')
        """
        import wbia

        self.ensure_setup()

        ibs = self.ibs
        sim_params = self.sim_params
        classifiers = sim_params['classifiers']
        test_aids = sim_params['test_aids']

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
        infr1.params['refresh.window'] = 20
        infr1.params['refresh.thresh'] = 0.052
        infr1.params['refresh.patience'] = 72
        infr1.params['redun.enforce_pos'] = True
        infr1.params['redun.enforce_neg'] = True

        infr1.init_simulation(classifiers=classifiers, **dials1)
        infr1.init_test_mode()

        infr1.reset(state='empty')

        # if False:
        #     infr = infr1
        #     infr.init_refresh()
        #     n_prioritized = infr.refresh_candidate_edges()
        #     gen = infr.lnbnn_priority_gen(use_refresh=True)
        #     next(gen)
        #     edge = (25, 118)

        list(infr1.main_gen())
        # infr1.main_loop()

        sim_results['graph'] = self._collect_sim_results(infr1, dials1)

        # ------------
        # Dump experiment output to disk
        expt_name = 'graphsim'
        self.expt_results[expt_name] = sim_results
        ut.ensuredir(self.dpath)
        ut.save_data(join(self.dpath, expt_name + '.pkl'), sim_results)

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

    def draw_graphsim(self):
        """
        CommandLine:

            python -m wbia GraphExpt.measure graphsim GZ_Master1
            python -m wbia GraphExpt.draw graphsim GZ_Master1 --diskshow

            python -m wbia GraphExpt.draw graphsim PZ_MTEST --diskshow
            python -m wbia GraphExpt.draw graphsim GZ_Master1 --diskshow
            python -m wbia GraphExpt.draw graphsim PZ_Master1 --diskshow

        Ignore:
            >>> from wbia.scripts.postdoc import *
            >>> self = GraphExpt('GZ_Master1')
            >>> self = GraphExpt('PZ_MTEST')
        """
        sim_results = self.ensure_results('graphsim')

        metric_nice = {
            'n_errors': '# errors',
            'n_manual': '# manual reviews',
            'frac_mistake_aids': 'fraction error annots',
            'merge_remain': 'fraction of merges remain',
        }

        # keys = ['ranking', 'rank+clf', 'graph']
        # keycols = ['red', 'orange', 'b']
        keys = ['graph']
        keycols = ['b']
        colors = ut.dzip(keys, keycols)

        dfs = {k: pd.DataFrame(v['metrics']) for k, v in sim_results.items()}

        n_aids = sim_results['graph']['graph'].number_of_nodes()
        df = dfs['graph']
        df['frac_mistake_aids'] = df.n_mistake_aids / n_aids
        # mdf = pd.concat(dfs.values(), keys=dfs.keys())

        import xarray as xr

        panel = xr.concat(
            [xr.DataArray(df, dims=('ts', 'metric')) for df in dfs.values()],
            dim=pd.Index(list(dfs.keys()), name='key'),
        )

        xmax = panel.sel(metric='n_manual').values.max()
        xpad = (1.01 * xmax) - xmax

        pnum_ = pt.make_pnum_nextgen(nSubplots=2)
        mpl.rcParams.update(TMP_RC)

        fnum = 1
        pt.figure(fnum=fnum, pnum=pnum_())
        ax = pt.gca()

        xkey, ykey = 'n_manual', 'merge_remain'
        datas = panel.sel(metric=[xkey, ykey])
        for key in keys:
            ax.plot(*datas.sel(key=key).values.T, label=key, color=colors[key])
        ax.set_ylim(0, 1)
        ax.set_xlim(-xpad, xmax + xpad)
        ax.set_xlabel(metric_nice[xkey])
        ax.set_ylabel(metric_nice[ykey])
        ax.legend()

        pt.figure(fnum=fnum, pnum=pnum_())
        ax = pt.gca()
        xkey, ykey = 'n_manual', 'frac_mistake_aids'
        datas = panel.sel(metric=[xkey, ykey])
        for key in keys:
            ax.plot(*datas.sel(key=key).values.T, label=key, color=colors[key])
        ax.set_ylim(0, datas.T[1].max() * 1.01)
        ax.set_xlim(-xpad, xmax + xpad)
        ax.set_xlabel(metric_nice[xkey])
        ax.set_ylabel(metric_nice[ykey])
        ax.legend()

        fig = pt.gcf()  # NOQA
        fig.set_size_inches([W, H * 0.75])
        pt.adjust_subplots(wspace=0.25, fig=fig)

        fpath = join(self.dpath, 'simulation.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fpath)

    def draw_graphsim2(self):
        """
        CommandLine:
            python -m wbia GraphExpt.draw graphsim2 --db PZ_MTEST --diskshow
            python -m wbia GraphExpt.draw graphsim2 GZ_Master1 --diskshow
            python -m wbia GraphExpt.draw graphsim2 PZ_Master1 --diskshow

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.thesis import *
            >>> dbname = ut.get_argval('--db', default='GZ_Master1')
            >>> self = GraphExpt(dbname)
            >>> self.draw_graphsim2()
            >>> ut.show_if_requested()
        """
        mpl.rcParams.update(TMP_RC)
        sim_results = self.ensure_results('graphsim')

        expt_data = sim_results['graph']
        metrics_df = pd.DataFrame.from_dict(expt_data['metrics'])

        # n_aids = sim_results['graph']['graph'].number_of_nodes()
        # metrics_df['frac_mistake_aids'] = metrics_df.n_mistake_aids / n_aids

        fnum = 1  # NOQA
        default_flags = {
            'phase': True,
            'pred': False,
            'user': True,
            'real': True,
            'error': 0,
            'recover': 1,
        }

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

        def overlay_actions(ymax=1, kw=None):
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

            if kw is None:
                kw = default_flags

            num = sum(kw.values())
            steps = np.linspace(0, 1, num + 1) * ymax
            i = -1

            def stacked_interval(data, color, i):
                plot_intervals(data, color, low=steps[i], high=steps[i + 1])

            if kw.get('user', False):
                i += 1
                pt.absolute_text(
                    (0.2, steps[i : i + 2].mean()), 'user(algo=gold,manual=blue)'
                )
                stacked_interval(is_auto, 'gold', i)
                stacked_interval(~is_auto, 'blue', i)

            if kw.get('pred', False):
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'pred_pos')
                stacked_interval(ppos, 'aqua', low=steps[i], high=steps[i + 1])
                # stacked_interval(~ppos, 'salmon', i)

            if kw.get('real', False):
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'real_merge')
                stacked_interval(rpos, 'lime', i)
                # stacked_interval(~ppos, 'salmon', i)

            if kw.get('error', False):
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'is_error')
                # stacked_interval(is_correct, 'blue', low=steps[i], high=steps[i + 1])
                stacked_interval(~is_correct, 'red', i)

            if kw.get('recover', False):
                i += 1
                pt.absolute_text((0.2, steps[i : i + 2].mean()), 'is_recovering')
                stacked_interval(recovering, 'orange', i)

            if kw.get('phase', False):
                i += 1
                pt.absolute_text(
                    (0.2, steps[i : i + 2].mean()), 'phase(1=yellow, 2=aqua, 3=pink)'
                )
                stacked_interval(phase == 'ranking', 'yellow', i)
                stacked_interval(phase == 'posredun', 'aqua', i)
                stacked_interval(phase == 'negredun', 'pink', i)
                # stacked_interval(phase == 'ranking', 'red', i)
                # stacked_interval(phase == 'posredun', 'green', i)
                # stacked_interval(phase == 'negredun', 'blue', i)

        def accuracy_plot(xdata, xlabel):
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

        def error_plot(xdata, xlabel):
            # ykeys = ['n_errors']
            ykeys = ['frac_mistake_aids']
            pt.multi_plot(
                xdata,
                metrics_df[ykeys].values.T,
                xlabel=xlabel,
                ylabel='fraction error annots',
                marker='',
                markersize=1,
                ymin=0,
                rcParams=TMP_RC,
                fnum=1,
                pnum=pnum_(),
                use_legend=False,
            )

        def refresh_plot(xdata, xlabel):
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
            ax.plot(
                [min(xdata), max(xdata)], [thresh, thresh], '-g', label='refresh thresh'
            )
            ax.legend()

        def error_breakdown_plot(xdata, xlabel):
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

        def neg_redun_plot(xdata, xlabel):
            n_pred = len(sim_results['graph']['pred_ccs'])
            z = (n_pred * (n_pred - 1)) / 2

            metrics_df['p_neg_redun'] = metrics_df['n_neg_redun'] / z
            metrics_df['p_neg_redun1'] = metrics_df['n_neg_redun1'] / z

            ykeys = ['p_neg_redun', 'p_neg_redun1']
            pt.multi_plot(
                xdata,
                metrics_df[ykeys].values.T,
                label_list=ykeys,
                xlabel=xlabel,
                ylabel='% neg-redun-meta-edges',
                marker='x',
                markersize=1,
                ymin=0,
                rcParams=TMP_RC,
                ymax=max(metrics_df['p_neg_redun1']),
                fnum=1,
                pnum=pnum_(),
                use_legend=True,
            )

        pnum_ = pt.make_pnum_nextgen(nRows=2, nSubplots=6)

        # --- ROW 1 ---
        xdata = metrics_df['n_decision']
        xlabel = '# decisions'

        accuracy_plot(xdata, xlabel)
        # overlay_actions(1)

        error_plot(xdata, xlabel)
        overlay_actions(max(metrics_df['frac_mistake_aids']))
        # overlay_actions(max(metrics_df['n_errors']))

        # refresh_plot(xdata, xlabel)
        # overlay_actions(1, {'phase': True})

        # error_breakdown_plot(xdata, xlabel)
        neg_redun_plot(xdata, xlabel)

        # --- ROW 2 ---
        xdata = metrics_df['n_manual']
        xlabel = '# manual reviews'

        accuracy_plot(xdata, xlabel)
        # overlay_actions(1)

        error_plot(xdata, xlabel)
        overlay_actions(max(metrics_df['frac_mistake_aids']))
        # overlay_actions(max(metrics_df['n_errors']))

        # refresh_plot(xdata, xlabel)
        # overlay_actions(1, {'phase': True})

        # error_breakdown_plot(xdata, xlabel)
        neg_redun_plot(xdata, xlabel)

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

        fig = pt.gcf()  # NOQA
        fig.set_size_inches([W * 2, H * 2.5])
        fig.suptitle(self.dbname)
        pt.adjust_subplots(hspace=0.25, wspace=0.25, fig=fig)
        fpath = join(self.dpath, 'graphsim2.png')
        fig.savefig(fpath, dpi=DPI)
        # vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fpath)


def draw_match_states():
    import wbia

    infr = wbia.AnnotInference('PZ_Master1', 'all')

    if infr.ibs.dbname == 'PZ_Master1':
        # [UUID('0cb1ebf5-2a4f-4b80-b172-1b449b8370cf'),
        #  UUID('cd644b73-7978-4a5f-b570-09bb631daa75')]
        chosen = {
            POSTV: (17095, 17225),
            NEGTV: (3966, 5080),
            INCMP: (3197, 8455),
        }
    else:
        infr.reset_feedback('staging')
        chosen = {
            POSTV: list(infr.pos_graph.edges())[0],
            NEGTV: list(infr.neg_graph.edges())[0],
            INCMP: list(infr.incmp_graph.edges())[0],
        }
    import wbia.plottool as pt
    import vtool as vt

    for key, edge in chosen.items():
        match = infr._make_matches_from(
            [edge], config={'match_config': {'ratio_thresh': 0.7}}
        )[0]
        with pt.RenderingContext(dpi=300) as ctx:
            match.show(heatmask=True, show_ell=False, show_ori=False, show_lines=False)
        vt.imwrite('matchstate_' + key + '.jpg', ctx.image)


def entropy_potential(infr, u, v, decision):
    """
    Returns the number of edges this edge would invalidate

    from wbia.algo.graph import demo
    infr = demo.demodata_infr(pcc_sizes=[5, 2, 4, 2, 2, 1, 1, 1])
    infr.refresh_candidate_edges()
    infr.params['redun.neg'] = 1
    infr.params['redun.pos'] = 1
    infr.apply_nondynamic_update()

    ut.qtensure()
    infr.show(show_cand=True, groupby='name_label')

    u, v = 1, 7
    decision = 'positive'
    """
    nid1, nid2 = infr.pos_graph.node_labels(u, v)

    # Cases for K=1
    if decision == 'positive' and nid1 == nid2:
        # The actual reduction is the number previously needed to make the cc
        # k-edge-connected vs how many its needs now.

        # In the same CC does nothing
        # (unless k > 1, in which case check edge connectivity)
        return 0
    elif decision == 'positive' and nid1 != nid2:
        # Between two PCCs reduces the number of PCCs by one
        n_ccs = infr.pos_graph.number_of_components()

        # Find needed negative redundency when appart
        if infr.neg_redun_metagraph.has_node(nid1):
            neg_redun_set1 = set(infr.neg_redun_metagraph.neighbors(nid1))
        else:
            neg_redun_set1 = set()

        if infr.neg_redun_metagraph.has_node(nid2):
            neg_redun_set2 = set(infr.neg_redun_metagraph.neighbors(nid2))
        else:
            neg_redun_set2 = set()

        # The number of negative edges needed before we place this edge
        # is the number of PCCs that each PCC doesnt have a negative edge to
        # yet

        n_neg_need1 = n_ccs - len(neg_redun_set1) - 1
        n_neg_need2 = n_ccs - len(neg_redun_set2) - 1
        n_neg_need_before = n_neg_need1 + n_neg_need2

        # After we join them we take the union of their negative redundancy
        # (really we should check if it changes after)
        # and this is now the new number of negative edges that would be needed
        neg_redun_after = neg_redun_set1.union(neg_redun_set2) - {nid1, nid2}
        n_neg_need_after = (n_ccs - 2) - len(neg_redun_after)

        neg_entropy = n_neg_need_before - n_neg_need_after  # NOQA


def _find_good_match_states(infr, ibs, edges):
    pos_edges = list(infr.pos_graph.edges())
    timedelta = ibs.get_annot_pair_timedelta(*zip(*edges))
    edges = ut.take(pos_edges, ut.argsort(timedelta))[::-1]
    infr.qt_edge_reviewer(edges)

    neg_edges = ut.shuffle(list(infr.neg_graph.edges()))
    infr.qt_edge_reviewer(neg_edges)

    if infr.incomp_graph.number_of_edges() > 0:
        incmp_edges = list(infr.incomp_graph.edges())
        if False:
            ibs = infr.ibs
            # a1, a2 = map(ibs.annots, zip(*incmp_edges))
            # q1 = np.array(ut.replace_nones(a1.qual, np.nan))
            # q2 = np.array(ut.replace_nones(a2.qual, np.nan))
            # edges = ut.compress(incmp_edges,
            #                     ((q1 > 3) | np.isnan(q1)) &
            #                     ((q2 > 3) | np.isnan(q2)))

            # a = ibs.annots(asarray=True)
            # flags = [t is not None and 'right' == t for t in a.viewpoint_code]
            # r = a.compress(flags)
            # flags = [q is not None and q > 4 for q in r.qual]

            rights = ibs.filter_annots_general(
                view='right',
                minqual='excellent',
                require_quality=True,
                require_viewpoint=True,
            )
            lefts = ibs.filter_annots_general(
                view='left',
                minqual='excellent',
                require_quality=True,
                require_viewpoint=True,
            )

            if False:
                edges = list(infr._make_rankings(3197, rights))
                infr.qt_edge_reviewer(edges)

            edges = list(ut.random_product((rights, lefts), num=10, rng=0))
            infr.qt_edge_reviewer(edges)

        for edge in incmp_edges:
            infr._make_matches_from([edge])[0]
            # infr._debug_edge_gt(edge)


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
        '%6.3f%% - %s' % (cdf[0] * 100, lbl) for cdf, lbl in zip(cdfs_trunc, labels)
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


@ut.reloadable_class
class VerifierExpt(DBInputs):
    """
    Collect data from experiments to visualize

    python -m wbia VerifierExpt.measure all PZ_Master1.GZ_Master1,GIRM_Master1,MantaMatcher,RotanTurtles,humpbacks_fb,LF_ALL
    python -m wbia VerifierExpt.measure all GIRM_Master1,PZ_Master1,LF_ALL
    python -m wbia VerifierExpt.measure all LF_ALL
    python -m wbia VerifierExpt.measure all PZ_Master1

    python -m wbia VerifierExpt.measure all MantaMatcher
    python -m wbia VerifierExpt.draw all MantaMatcher

    python -m wbia VerifierExpt.draw rerank PZ_Master1

    python -m wbia VerifierExpt.measure all RotanTurtles
    python -m wbia VerifierExpt.draw all RotanTurtles

    Ignore:
        >>> from wbia.scripts.postdoc import *
        >>> fpath = ut.glob(ut.truepath('~/Desktop/mtest_plots'), '*.pkl')[0]
        >>> self = ut.load_data(fpath)
    """

    # base_dpath = ut.truepath('~/Desktop/pair_expts')
    base_dpath = ut.truepath('~/latex/crall-iccvw-2017/figures')

    agg_dbnames = [
        'PZ_Master1',
        'GZ_Master1',
        # 'LF_ALL',
        'MantaMatcher',
        'RotanTurtles',
        'humpbacks_fb',
        'GIRM_Master1',
    ]

    task_nice_lookup = {
        'match_state': const.EVIDENCE_DECISION.CODE_TO_NICE,
        'photobomb_state': {'pb': 'Photobomb', 'notpb': 'Not Photobomb'},
    }

    def _setup(self, quick=False):
        r"""
        CommandLine:
            python -m wbia VerifierExpt._setup --db GZ_Master1

            python -m wbia VerifierExpt._setup --db PZ_Master1 --eval
            python -m wbia VerifierExpt._setup --db PZ_MTEST
            python -m wbia VerifierExpt._setup --db PZ_PB_RF_TRAIN

            python -m wbia VerifierExpt.measure_all --db PZ_PB_RF_TRAIN

            python -m wbia VerifierExpt.measure all GZ_Master1
            python -m wbia VerifierExpt.measure all RotanTurtles --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.postdoc import *
            >>> dbname = ut.get_argval('--db', default='GZ_Master1')
            >>> self = VerifierExpt(dbname)
            >>> self._setup()

        Ignore:
            from wbia.scripts.postdoc import *
            self = VerifierExpt('PZ_Master1')

            from wbia.scripts.postdoc import *
            self = VerifierExpt('PZ_PB_RF_TRAIN')

            from wbia.scripts.postdoc import *
            self = VerifierExpt('LF_ALL')

            self = VerifierExpt('RotanTurtles')
            task = pblm.samples.subtasks['match_state']
            ind_df = task.indicator_df
            dist = ibs.get_annotedge_viewdist(ind_df.index.tolist())
            np.all(ind_df[dist > 1]['notcomp'])

            self.ibs.print_annot_stats(aids, prefix='P')
        """
        self._precollect()

        print('VerifierExpt _setup()')

        ibs = self.ibs

        aids = self.aids_pool

        # pblm = vsone.OneVsOneProblem.from_aids(ibs, aids, sample_method='random')
        pblm = vsone.OneVsOneProblem.from_aids(
            ibs,
            aids,
            sample_method='lnbnn+random',
            # sample_method='random',
            n_splits=10,
        )

        data_key = 'learn(sum)'  # tests without global features
        # data_key = 'learn(sum,glob)'  # tests with global features
        # data_key = pblm.default_data_key  # same as learn(sum,glob)

        clf_key = pblm.default_clf_key

        pblm.eval_task_keys = ['match_state']

        # test with and without globals
        pblm.eval_data_keys = ['learn(sum)', 'learn(sum,glob)']
        # pblm.eval_data_keys = [data_key]
        pblm.eval_clf_keys = [clf_key]

        ibs = pblm.infr.ibs
        # pblm.samples.print_info()

        species_code = ibs.get_database_species(pblm.infr.aids)[0]
        if species_code == 'zebra_plains':
            species = 'Plains Zebras'
        if species_code == 'zebra_grevys':
            species = "Grévy's Zebras"
        else:
            species = species_code

        self.pblm = pblm
        self.species = species
        self.data_key = data_key
        self.clf_key = clf_key

        if quick:
            return

        pblm.setup_evaluation(with_simple=True)
        pblm.report_evaluation()
        self.eval_task_keys = pblm.eval_task_keys

        cfg_prefix = '{}'.format(len(pblm.samples))
        config = pblm.hyper_params
        self._setup_links(cfg_prefix, config)
        print('Finished setup')

    @classmethod
    def agg_dbstats(cls):
        """
        CommandLine:
            python -m wbia VerifierExpt.agg_dbstats
            python -m wbia VerifierExpt.measure_dbstats

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.postdoc import *  # NOQA
            >>> result = VerifierExpt.agg_dbstats()
            >>> print(result)
        """
        dfs = []
        for dbname in cls.agg_dbnames:
            self = cls(dbname)
            info = self.ensure_results('dbstats', nocompute=False)
            sample_info = self.ensure_results('sample_info', nocompute=False)

            # info = self.measure_dbstats()
            outinfo = info['outinfo']

            task = sample_info['subtasks']['match_state']
            y_ind = task.indicator_df
            outinfo['Positive'] = (y_ind[POSTV]).sum()
            outinfo['Negative'] = (y_ind[NEGTV]).sum()
            outinfo['Incomparable'] = (y_ind[INCMP]).sum()
            if outinfo['Database'] == 'mantas':
                outinfo['Database'] = 'manta rays'
            dfs.append(outinfo)
            # labels.append(self.species_nice.capitalize())

        df = pd.DataFrame(dfs)
        print('df =\n{!r}'.format(df))
        df = df.set_index('Database')
        df.index.name = None

        tabular = Tabular(df, colfmt='numeric')
        tabular.theadify = 16
        enc_text = tabular.as_tabular()
        print(enc_text)

        ut.write_to(join(cls.base_dpath, 'agg-dbstats.tex'), enc_text)

        _ = ut.render_latex(
            enc_text,
            dpath=self.base_dpath,
            fname='agg-dbstats',
            preamb_extra=['\\usepackage{makecell}'],
        )
        _
        # ut.startfile(_)

    @classmethod
    def agg_results(cls, task_key):
        """

        python -m wbia VerifierExpt.agg_results
        python -m wbia VerifierExpt.agg_results --link link-paper-final

        GZ_Master1,LF_ALL,MantaMatcher,RotanTurtles,humpbacks_fb,GIRM_Master1

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.postdoc import *  # NOQA
            >>> task_key = 'match_state'
            >>> result = VerifierExpt.agg_results(task_key)
            >>> print(result)
        """
        cls.agg_dbstats()
        dbnames = cls.agg_dbnames

        all_results = ut.odict([])
        for dbname in cls.agg_dbnames:
            self = cls(dbname)
            info = self.ensure_results('all')
            all_results[dbname] = info

        rerank_results = ut.odict([])
        for dbname in cls.agg_dbnames:
            self = cls(dbname)
            info = self.ensure_results('rerank')
            rerank_results[dbname] = info

        rank_curves = ub.AutoOrderedDict()

        rank1_cmc_table = pd.DataFrame(columns=[LNBNN, CLF])
        rank5_cmc_table = pd.DataFrame(columns=[LNBNN, CLF])

        n_dbs = len(all_results)
        color_cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color'][:n_dbs]

        color_cycle = ['r', 'b', 'purple', 'orange', 'deeppink', 'g']
        markers = pt.distinct_markers(n_dbs)

        dbprops = ub.AutoDict()
        for n, dbname in enumerate(dbnames):
            dbprops[dbname]['color'] = color_cycle[n]
            dbprops[dbname]['marker'] = markers[n]

        def highlight_metric(metric, data1, data2):
            # Highlight the bigger one for each metric
            for d1, d2 in it.permutations([data1, data2], 2):
                text = '{:.3f}'.format(d1[metric])
                if d1[metric] >= d2[metric]:
                    d1[metric + '_tex'] = '\\mathbf{' + text + '}'
                    d1[metric + '_text'] = text + '*'
                else:
                    d1[metric + '_tex'] = text
                    d1[metric + '_text'] = text

        for dbname in dbnames:
            results = all_results[dbname]
            data_key = results['data_key']
            clf_key = results['clf_key']

            lnbnn_data = results['lnbnn_data']
            task_combo_res = results['task_combo_res']
            res = task_combo_res[task_key][clf_key][data_key]
            nice = dbname_to_species_nice(dbname)

            # ranking results
            results = rerank_results[dbname]
            cdfs, infos = list(zip(*results))
            lnbnn_cdf, clf_cdf = cdfs
            cdfs = {
                CLF: clf_cdf,
                LNBNN: lnbnn_cdf,
            }

            rank1_cmc_table.loc[nice, LNBNN] = lnbnn_cdf[0]
            rank1_cmc_table.loc[nice, CLF] = clf_cdf[0]

            rank5_cmc_table.loc[nice, LNBNN] = lnbnn_cdf[4]
            rank5_cmc_table.loc[nice, CLF] = clf_cdf[4]

            # Check the ROC for only things in the top of the LNBNN ranked lists
            # nums = [1, 2, 3, 4, 5, 10, 20, np.inf]
            nums = [1, 5, np.inf]
            for num in nums:

                ranks = lnbnn_data['rank_lnbnn_1vM'].values
                sub_data = lnbnn_data[ranks <= num]
                scores = sub_data['score_lnbnn_1vM'].values
                y = sub_data[POSTV].values
                probs = res.probs_df[POSTV].loc[sub_data.index].values

                cfsm_vsm = vt.ConfusionMetrics().fit(scores, y)
                cfsm_clf = vt.ConfusionMetrics().fit(probs, y)
                algo_confusions = {LNBNN: cfsm_vsm, CLF: cfsm_clf}

                datas = []
                for algo in {LNBNN, CLF}:
                    cfms = algo_confusions[algo]
                    data = {
                        'dbname': dbname,
                        'species': nice,
                        'fpr': cfms.fpr,
                        'tpr': cfms.tpr,
                        'auc': cfms.auc,
                        'cmc0': cdfs[algo][0],
                        'cmc': cdfs[algo],
                        'color': dbprops[dbname]['color'],
                        'marker': dbprops[dbname]['marker'],
                        'tpr@fpr=0': cfms.get_metric_at_metric(
                            'tpr', 'fpr', 0, tiebreaker='minthresh'
                        ),
                        'thresh@fpr=0': cfms.get_metric_at_metric(
                            'thresh', 'fpr', 0, tiebreaker='minthresh'
                        ),
                    }
                    rank_curves[num][algo][dbname] = data
                    datas.append(data)
                # Highlight the bigger one for each metric
                highlight_metric('auc', *datas)
                highlight_metric('tpr@fpr=0', *datas)
                highlight_metric('cmc0', *datas)

        rank_auc_tables = ut.ddict(lambda: pd.DataFrame(columns=[LNBNN, CLF]))
        rank_tpr_tables = ut.ddict(lambda: pd.DataFrame(columns=[LNBNN, CLF]))
        rank_tpr_thresh_tables = ut.ddict(lambda: pd.DataFrame(columns=[LNBNN, CLF]))
        for num in rank_curves.keys():
            rank_auc_df = rank_auc_tables[num]
            rank_auc_df.index.name = 'AUC@rank<={}'.format(num)
            rank_tpr_df = rank_tpr_tables[num]
            rank_tpr_df.index.name = 'tpr@fpr=0&rank<={}'.format(num)
            rank_thesh_df = rank_tpr_thresh_tables[num]
            rank_thesh_df.index.name = 'thresh@fpr=0&rank<={}'.format(num)
            for algo in rank_curves[num].keys():
                for dbname in rank_curves[num][algo].keys():
                    data = rank_curves[num][algo][dbname]
                    nice = data['species']
                    rank_auc_df.loc[nice, algo] = data['auc']
                    rank_tpr_df.loc[nice, algo] = data['tpr@fpr=0']
                    rank_thesh_df.loc[nice, algo] = data['thresh@fpr=0']

        from utool.experimental.pandas_highlight import to_string_monkey

        nums = [1]
        for rank in nums:
            print('-----')
            print('AUC at rank = {!r}'.format(rank))
            rank_auc_df = rank_auc_tables[rank]
            print(to_string_monkey(rank_auc_df, 'all'))

        print('===============')
        for rank in nums:
            print('-----')
            print('TPR at rank = {!r}'.format(rank))
            rank_tpr_df = rank_tpr_tables[rank]
            print(to_string_monkey(rank_tpr_df, 'all'))

        def _bf_best(df):
            df = df.copy()
            for rx in range(len(df)):
                col = df.iloc[rx]
                for cx in ut.argmax(col.values, multi=True):
                    val = df.iloc[rx, cx]
                    df.iloc[rx, cx] = '\\mathbf{{{:.3f}}}'.format(val)
            return df

        if True:
            # Tables
            rank1_auc_table = rank_auc_tables[1]
            rank1_tpr_table = rank_tpr_tables[1]
            # all_stats = pd.concat(ut.emap(_bf_best, [auc_table, rank1_cmc_table, rank5_cmc_table]), axis=1)
            column_parts = [
                ('Rank $1$ AUC', rank1_auc_table),
                ('Rank $1$ TPR', rank1_tpr_table),
                ('Pos. @ Rank $1$', rank1_cmc_table),
            ]

            all_stats = pd.concat(
                ut.emap(_bf_best, ut.take_column(column_parts, 1)), axis=1
            )
            all_stats.index.name = None
            colfmt = 'l|' + '|'.join(['rr'] * len(column_parts))
            multi_header = (
                [None]
                + [(2, 'c|', name) for name in ut.take_column(column_parts, 0)[0:-1]]
                + [(2, 'c', name) for name in ut.take_column(column_parts, 0)[-1:]]
            )

            from wbia.scripts import _thesis_helpers

            tabular = _thesis_helpers.Tabular(all_stats, colfmt=colfmt, escape=False)
            tabular.add_multicolumn_header(multi_header)
            tabular.precision = 3
            tex_text = tabular.as_tabular()

            # HACKS
            import re

            num_pat = ut.named_field('num', r'[0-9]*\.?[0-9]*')
            tex_text = re.sub(
                re.escape('\\mathbf{$') + num_pat + re.escape('$}'),
                '$\\mathbf{' + ut.bref_field('num') + '}$',
                tex_text,
            )
            print(tex_text)
            # tex_text = tex_text.replace('\\mathbf{$', '$\\mathbf{')
            # tex_text = tex_text.replace('$}', '}$')

            ut.write_to(join(cls.base_dpath, 'agg-results-all.tex'), tex_text)

            _ = ut.render_latex(
                tex_text,
                dpath=cls.base_dpath,
                fname='agg-results-all',
                preamb_extra=['\\usepackage{makecell}'],
            )
            # ut.startfile(_)

        if True:
            # Tables
            rank1_auc_table = rank_auc_tables[1]
            rank1_tpr_table = rank_tpr_tables[1]
            print(
                '\nrank1_auc_table =\n{}'.format(to_string_monkey(rank1_auc_table, 'all'))
            )
            print(
                '\nrank1_tpr_table =\n{}'.format(to_string_monkey(rank1_tpr_table, 'all'))
            )
            print(
                '\nrank1_cmc_table =\n{}'.format(to_string_monkey(rank1_cmc_table, 'all'))
            )

            # Tables
            rank1_auc_table = rank_auc_tables[1]
            rank1_tpr_table = rank_tpr_tables[1]
            # all_stats = pd.concat(ut.emap(_bf_best, [auc_table, rank1_cmc_table, rank5_cmc_table]), axis=1)
            column_parts = [
                ('Rank $1$ AUC', rank1_auc_table),
                # ('Rank $1$ TPR', rank1_tpr_table),
                ('Pos. @ Rank $1$', rank1_cmc_table),
            ]

            all_stats = pd.concat(
                ut.emap(_bf_best, ut.take_column(column_parts, 1)), axis=1
            )
            all_stats.index.name = None
            colfmt = 'l|' + '|'.join(['rr'] * len(column_parts))
            multi_header = (
                [None]
                + [(2, 'c|', name) for name in ut.take_column(column_parts, 0)[0:-1]]
                + [(2, 'c', name) for name in ut.take_column(column_parts, 0)[-1:]]
            )

            from wbia.scripts import _thesis_helpers

            tabular = _thesis_helpers.Tabular(all_stats, colfmt=colfmt, escape=False)
            tabular.add_multicolumn_header(multi_header)
            tabular.precision = 3
            tex_text = tabular.as_tabular()

            # HACKS
            import re

            num_pat = ut.named_field('num', r'[0-9]*\.?[0-9]*')
            tex_text = re.sub(
                re.escape('\\mathbf{$') + num_pat + re.escape('$}'),
                '$\\mathbf{' + ut.bref_field('num') + '}$',
                tex_text,
            )
            print(tex_text)
            print(tex_text)
            # tex_text = tex_text.replace('\\mathbf{$', '$\\mathbf{')
            # tex_text = tex_text.replace('$}', '}$')
            ut.write_to(join(cls.base_dpath, 'agg-results.tex'), tex_text)

            _ = ut.render_latex(
                tex_text,
                dpath=cls.base_dpath,
                fname='agg-results',
                preamb_extra=['\\usepackage{makecell}'],
            )
            _
            # ut.startfile(_)

        method = 2
        if method == 2:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.unicode'] = True
            # mpl.rcParams['axes.labelsize'] = 12
            mpl.rcParams['legend.fontsize'] = 12

            mpl.rcParams['xtick.color'] = 'k'
            mpl.rcParams['ytick.color'] = 'k'
            mpl.rcParams['axes.labelcolor'] = 'k'
            # mpl.rcParams['text.color'] = 'k'

            nums = [1, np.inf]
            nums = [1]
            for num in nums:
                chunked_dbnames = list(ub.chunks(dbnames, 2))
                for fnum, dbname_chunk in enumerate(chunked_dbnames, start=1):
                    fig = pt.figure(fnum=fnum)  # NOQA
                    fig.clf()
                    ax = pt.gca()
                    for dbname in dbname_chunk:
                        data1 = rank_curves[num][CLF][dbname]
                        data2 = rank_curves[num][LNBNN][dbname]

                        data1['label'] = 'TPR=${tpr}$ {algo} {species}'.format(
                            algo=CLF, tpr=data1['tpr@fpr=0_tex'], species=data1['species']
                        )
                        data1['ls'] = '-'
                        data1['chunk_marker'] = '^'
                        data1['color'] = dbprops[dbname]['color']

                        data2['label'] = 'TPR=${tpr}$ {algo} {species}'.format(
                            algo=LNBNN,
                            tpr=data2['tpr@fpr=0_tex'],
                            species=data2['species'],
                        )
                        data2['ls'] = '--'
                        data2['chunk_marker'] = 'v'
                        data2['color'] = dbprops[dbname]['color']

                        for d in [data1, data2]:
                            ax.plot(
                                d['fpr'], d['tpr'], d['ls'], color=d['color'], zorder=10
                            )

                        for d in [data1, data2]:
                            ax.plot(
                                0,
                                d['tpr@fpr=0'],
                                d['ls'],
                                marker=d['chunk_marker'],
                                markeredgecolor='k',
                                markersize=8,
                                # fillstyle='none',
                                color=d['color'],
                                label=d['label'],
                                zorder=100,
                            )

                    ax.set_xlabel('false positive rate')
                    ax.set_ylabel('true positive rate')
                    ax.set_ylim(0, 1)
                    ax.set_xlim(-0.05, 0.5)
                    # ax.set_title('ROC with ranks $<= {}$'.format(num))
                    ax.legend(loc='lower right')
                    pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
                    fig.set_size_inches([W * 0.7, H])

                    fname = 'agg_roc_rank_{}_chunk_{}_{}.png'.format(num, fnum, task_key)
                    fig_fpath = join(str(cls.base_dpath), fname)
                    vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

            chunked_dbnames = list(ub.chunks(dbnames, 2))
            for fnum, dbname_chunk in enumerate(chunked_dbnames, start=1):
                fig = pt.figure(fnum=fnum)  # NOQA
                fig.clf()
                ax = pt.gca()
                for dbname in dbname_chunk:
                    data1 = rank_curves[num][CLF][dbname]
                    data2 = rank_curves[num][LNBNN][dbname]

                    data1['label'] = 'pos@rank1=${cmc0}$ {algo} {species}'.format(
                        algo=CLF, cmc0=data1['cmc0_tex'], species=data1['species']
                    )
                    data1['ls'] = '-'
                    data1['chunk_marker'] = '^'
                    data1['color'] = dbprops[dbname]['color']

                    data2['label'] = 'pos@rank1=${cmc0}$ {algo} {species}'.format(
                        algo=LNBNN, cmc0=data2['cmc0_tex'], species=data2['species']
                    )
                    data2['ls'] = '--'
                    data2['chunk_marker'] = 'v'
                    data2['color'] = dbprops[dbname]['color']

                    for d in [data1, data2]:
                        ax.plot(d['fpr'], d['tpr'], d['ls'], color=d['color'])

                    for d in [data1, data2]:
                        ax.plot(
                            d['cmc'],
                            d['ls'],
                            # marker=d['chunk_marker'],
                            # markeredgecolor='k',
                            # markersize=8,
                            # fillstyle='none',
                            color=d['color'],
                            label=d['label'],
                        )

                ax.set_xlabel('rank')
                ax.set_ylabel('match probability')
                ax.set_ylim(0, 1)
                ax.set_xlim(1, 20)
                ax.set_xticks([1, 5, 10, 15, 20])
                # ax.set_title('ROC with ranks $<= {}$'.format(num))
                ax.legend(loc='lower right')
                pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
                fig.set_size_inches([W * 0.7, H])

                fname = 'agg_cmc_chunk_{}_{}.png'.format(fnum, task_key)
                fig_fpath = join(str(cls.base_dpath), fname)
                vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

        if method == 1:

            # Does going from rank 1 to rank inf generally improve deltas?
            # -rank_tpr_tables[np.inf].diff(axis=1) - -rank_tpr_tables[1].diff(axis=1)

            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.unicode'] = True
            # mpl.rcParams['axes.labelsize'] = 12
            mpl.rcParams['legend.fontsize'] = 12

            mpl.rcParams['xtick.color'] = 'k'
            mpl.rcParams['ytick.color'] = 'k'
            mpl.rcParams['axes.labelcolor'] = 'k'
            # mpl.rcParams['text.color'] = 'k'

            def method1_roc(roc_curves, algo, other):
                ax = pt.gca()

                for dbname in dbnames:
                    data = roc_curves[algo][dbname]
                    ax.plot(data['fpr'], data['tpr'], color=data['color'])

                for dbname in dbnames:
                    data = roc_curves[algo][dbname]
                    other_data = roc_curves[other][dbname]
                    other_tpr = other_data['tpr@fpr=0']
                    species = data['species']
                    tpr = data['tpr@fpr=0']
                    tpr_text = '{:.3f}'.format(tpr)
                    if tpr >= other_tpr:
                        if mpl.rcParams['text.usetex']:
                            tpr_text = '\\mathbf{' + tpr_text + '}'
                        else:
                            tpr_text = tpr_text + '*'
                    label = 'TPR=${tpr}$ {species}'.format(tpr=tpr_text, species=species)
                    ax.plot(
                        0,
                        data['tpr@fpr=0'],
                        marker=data['marker'],
                        label=label,
                        color=data['color'],
                    )

                if algo:
                    algo = algo.rstrip() + ' '

                algo = ''
                ax.set_xlabel(algo + 'false positive rate')
                ax.set_ylabel('true positive rate')
                ax.set_ylim(0, 1)
                ax.set_xlim(-0.005, 0.5)

                # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
                ax.legend(loc='lower right')
                pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
                fig.set_size_inches([W * 0.7, H])

            nums = [1, np.inf]
            # nums = [1]
            for num in nums:
                algos = {CLF, LNBNN}
                for fnum, algo in enumerate(algos, start=1):
                    roc_curves = rank_curves[num]
                    other = next(iter(algos - {algo}))
                    fig = pt.figure(fnum=fnum)  # NOQA
                    method1_roc(roc_curves, algo, other)
                    fname = 'agg_roc_rank_{}_{}_{}.png'.format(num, algo, task_key)
                    fig_fpath = join(str(cls.base_dpath), fname)
                    vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

            # -------------

            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.unicode'] = True

            mpl.rcParams['xtick.color'] = 'k'
            mpl.rcParams['ytick.color'] = 'k'
            mpl.rcParams['axes.labelcolor'] = 'k'
            mpl.rcParams['text.color'] = 'k'

            def method1_cmc(cmc_curves):
                ax = pt.gca()
                color_cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']
                markers = pt.distinct_markers(len(cmc_curves))
                for data, marker, color in zip(cmc_curves.values(), markers, color_cycle):
                    species = data['species']

                    if mpl.rcParams['text.usetex']:
                        cmc0_text = data['cmc0_tex']
                        label = 'pos@rank1=${}$ {species}'.format(
                            cmc0_text, species=species
                        )
                    else:
                        cmc0_text = data['cmc0_text']
                        label = 'pos@rank1={} {species}'.format(
                            cmc0_text, species=species
                        )

                    ranks = np.arange(1, len(data['cmc']) + 1)
                    ax.plot(ranks, data['cmc'], marker=marker, color=color, label=label)

                ax.set_xlabel('rank')
                ax.set_ylabel('match probability')
                ax.set_ylim(0, 1)
                ax.set_xlim(1, 20)
                ax.set_xticks([1, 5, 10, 15, 20])

                # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
                ax.legend(loc='lower right')
                pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
                fig.set_size_inches([W * 0.7, H])

            fig = pt.figure(fnum=1)  # NOQA
            # num doesnt actually matter here
            num = 1
            cmc_curves = rank_curves[num][CLF]
            method1_cmc(cmc_curves)
            fname = 'agg_cmc_clf_{}.png'.format(task_key)
            fig_fpath = join(str(cls.base_dpath), fname)
            vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

            fig = pt.figure(fnum=2)  # NOQA
            cmc_curves = rank_curves[num][LNBNN]
            method1_cmc(cmc_curves)
            fname = 'agg_cmc_lnbnn_{}.png'.format(task_key)
            fig_fpath = join(str(cls.base_dpath), fname)
            vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))

        if True:
            # Agg metrics

            agg_y_pred = []
            agg_y_true = []
            agg_sample_weight = []
            agg_class_names = None

            for dbname, results in all_results.items():
                task_combo_res = results['task_combo_res']
                res = task_combo_res[task_key][clf_key][data_key]

                res.augment_if_needed()
                y_true = res.y_test_enc
                incmp_enc = ut.aslist(res.class_names).index(INCMP)
                if sum(y_true == incmp_enc) < 500:
                    continue

                # Find auto thresholds
                print('-----')
                print('dbname = {!r}'.format(dbname))
                for k in range(res.y_test_bin.shape[1]):
                    class_k_truth = res.y_test_bin.T[k]
                    class_k_probs = res.clf_probs.T[k]
                    cfms_ovr = vt.ConfusionMetrics().fit(class_k_probs, class_k_truth)
                    # auc = sklearn.metrics.roc_auc_score(class_k_truth, class_k_probs)
                    state = res.class_names[k]
                    # for state, cfms_ovr in res.confusions_ovr():
                    if state == POSTV:
                        continue
                    tpr = cfms_ovr.get_metric_at_metric(
                        'tpr', 'fpr', 0, tiebreaker='minthresh'
                    )
                    # thresh = cfsm_scores_rank.get_metric_at_metric(
                    #     'thresh', 'fpr', 0, tiebreaker='minthresh')
                    print('state = {!r}'.format(state))
                    print('tpr = {:.3f}'.format(tpr))
                    print('+--')
                print('-----')

                # aggregate results
                y_pred = res.clf_probs.argmax(axis=1)
                agg_y_true.extend(y_true.tolist())
                agg_y_pred.extend(y_pred.tolist())
                agg_sample_weight.extend(res.sample_weight.tolist())
                assert (
                    agg_class_names is None or agg_class_names == res.class_names
                ), 'classes are inconsistent'
                agg_class_names = res.class_names

            from wbia.algo.verif import sklearn_utils

            agg_report = sklearn_utils.classification_report2(
                agg_y_true, agg_y_pred, agg_class_names, agg_sample_weight, verbose=False
            )
            metric_df = agg_report['metrics']
            confusion_df = agg_report['confusion']
            # multiclass_mcc = agg_report['mcc']
            # df.loc['combined', 'MCC'] = multiclass_mcc

            multiclass_mcc = agg_report['mcc']
            metric_df.loc['combined', 'mcc'] = multiclass_mcc

            print(metric_df)
            print(confusion_df)

            dpath = str(self.base_dpath)
            confusion_fname = 'agg_confusion_{}'.format(task_key)
            metrics_fname = 'agg_eval_metrics_{}'.format(task_key)

            # df = self.task_confusion[task_key]
            df = confusion_df.copy()
            df = df.rename_axis(self.task_nice_lookup[task_key], 0)
            df = df.rename_axis(self.task_nice_lookup[task_key], 1)
            df.columns.name = None

            df.index.name = 'Real'

            colfmt = '|l|' + 'r' * (len(df) - 1) + '|l|'
            tabular = Tabular(df, colfmt=colfmt, hline=True)
            tabular.groupxs = [list(range(len(df) - 1)), [len(df) - 1]]

            tabular.add_multicolumn_header([None, (3, 'c|', 'Predicted'), None])

            latex_str = tabular.as_tabular()

            sum_pred = df.index[-1]
            sum_real = df.columns[-1]
            latex_str = latex_str.replace(sum_pred, r'$\sum$ predicted')
            latex_str = latex_str.replace(sum_real, r'$\sum$ real')
            confusion_tex = ut.align(latex_str, '&', pos=None)
            print(confusion_tex)

            ut.render_latex(confusion_tex, dpath=self.base_dpath, fname=confusion_fname)

            df = metric_df
            # df = self.task_metrics[task_key]
            df = df.rename_axis(self.task_nice_lookup[task_key], 0)
            df = df.rename_axis({'mcc': 'MCC'}, 1)
            df = df.rename_axis({'combined': 'Combined'}, 1)
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

            confusion_tex = confusion_tex.replace('Incomparable', 'Incomp.')
            confusion_tex = confusion_tex.replace('predicted', 'pred')
            metrics_tex = metrics_tex.replace('Incomparable', 'Incomp.')

            ut.write_to(join(dpath, confusion_fname + '.tex'), confusion_tex)
            ut.write_to(join(dpath, metrics_fname + '.tex'), metrics_tex)

            ut.render_latex(confusion_tex, dpath=dpath, fname=confusion_fname)
            ut.render_latex(metrics_tex, dpath=dpath, fname=metrics_fname)

        old_cmc = rank1_cmc_table[LNBNN]
        new_cmc = rank1_cmc_table[CLF]
        cmc_diff = new_cmc - old_cmc
        cmc_change = cmc_diff / old_cmc
        improved = cmc_diff > 0
        print('{} / {} datasets saw CMC improvement'.format(sum(improved), len(cmc_diff)))
        print('CMC average absolute diff: {}'.format(cmc_diff.mean()))
        print('CMC average percent change: {}'.format(cmc_change.mean()))

        print('Average AUC:\n{}'.format(rank1_auc_table.mean(axis=0)))

        print('Average TPR:\n{}'.format(rank1_tpr_table.mean(axis=0)))

        old_tpr = rank1_tpr_table[LNBNN]
        new_tpr = rank1_tpr_table[CLF]
        tpr_diff = new_tpr - old_tpr
        tpr_change = tpr_diff / old_tpr
        improved = tpr_diff > 0
        print('{} / {} datasets saw TPR improvement'.format(sum(improved), len(tpr_diff)))
        print('TPR average absolute diff: {}'.format(tpr_diff.mean()))
        print('TPR average percent change: {}'.format(tpr_change.mean()))

    @profile
    def measure_dbstats(self):
        """
        python -m wbia VerifierExpt.measure dbstats GZ_Master1
        python -m wbia VerifierExpt.measure dbstats PZ_Master1
        python -m wbia VerifierExpt.measure dbstats MantaMatcher
        python -m wbia VerifierExpt.measure dbstats RotanTurtles

        Ignore:
            >>> from wbia.scripts.postdoc import *
            >>> #self = VerifierExpt('GZ_Master1')
            >>> self = VerifierExpt('MantaMatcher')
        """
        if self.ibs is None:
            self._precollect()
        ibs = self.ibs

        # self.ibs.print_annot_stats(self.aids_pool)
        # encattr = 'static_encounter'
        encattr = 'encounter_text'
        # encattr = 'aids'

        annots = ibs.annots(self.aids_pool)

        encounters = annots.group2(getattr(annots, encattr))

        nids = ut.take_column(encounters.nids, 0)
        nid_to_enc = ut.group_items(encounters, nids)

        single_encs = {nid: e for nid, e in nid_to_enc.items() if len(e) == 1}
        multi_encs = {
            nid: self.ibs._annot_groups(e) for nid, e in nid_to_enc.items() if len(e) > 1
        }

        multi_annots = ibs.annots(ut.flatten(ut.flatten(multi_encs.values())))
        single_annots = ibs.annots(ut.flatten(ut.flatten(single_encs.values())))

        def annot_stats(annots, encattr):
            encounters = annots.group2(getattr(annots, encattr))
            nid_to_enc = ut.group_items(encounters, ut.take_column(encounters.nids, 0))
            nid_to_nenc = ut.map_vals(len, nid_to_enc)
            n_enc_per_name = list(nid_to_nenc.values())
            n_annot_per_enc = ut.lmap(len, encounters)

            enc_deltas = []
            for encs_ in nid_to_enc.values():
                times = [np.mean(a.image_unixtimes_asfloat) for a in encs_]
                for tup in ut.combinations(times, 2):
                    delta = max(tup) - min(tup)
                    enc_deltas.append(delta)
                #     pass
                # delta = times.max() - times.min()
                # enc_deltas.append(delta)

            annot_info = ut.odict()
            annot_info['n_names'] = len(nid_to_enc)
            annot_info['n_annots'] = len(annots)
            annot_info['n_encs'] = len(encounters)
            annot_info['enc_time_deltas'] = ut.get_stats(enc_deltas)
            annot_info['n_enc_per_name'] = ut.get_stats(n_enc_per_name)
            annot_info['n_annot_per_enc'] = ut.get_stats(n_annot_per_enc)
            # print(ut.repr4(annot_info, si=True, nl=1, precision=2))
            return annot_info

        enc_info = ut.odict()
        enc_info['all'] = annot_stats(annots, encattr)
        del enc_info['all']['enc_time_deltas']
        enc_info['multi'] = annot_stats(multi_annots, encattr)
        enc_info['single'] = annot_stats(single_annots, encattr)
        del enc_info['single']['n_encs']
        del enc_info['single']['n_enc_per_name']
        del enc_info['single']['enc_time_deltas']

        qual_info = ut.dict_hist(annots.quality_texts)
        qual_info['None'] = qual_info.pop('UNKNOWN', 0)
        qual_info['None'] += qual_info.pop(None, 0)

        view_info = ut.dict_hist(annots.viewpoint_code)
        view_info['None'] = view_info.pop('unknown', 0)
        view_info['None'] += view_info.pop(None, 0)

        info = ut.odict([])
        info['species_nice'] = self.species_nice
        info['enc'] = enc_info
        info['qual'] = qual_info
        info['view'] = view_info

        print('Annotation Pool DBStats')
        print(ut.repr4(info, si=True, nl=3, precision=2))

        def _ave_str2(d):
            try:
                return ave_str(*ut.take(d, ['mean', 'std']))
            except Exception:
                return 0

        outinfo = ut.odict(
            [
                ('Database', info['species_nice']),
                ('Annots', enc_info['all']['n_annots']),
                ('Names (singleton)', enc_info['single']['n_names']),
                ('Names (resighted)', enc_info['multi']['n_names']),
                (
                    'Enc per name (resighted)',
                    _ave_str2(enc_info['multi']['n_enc_per_name']),
                ),
                ('Annots per encounter', _ave_str2(enc_info['all']['n_annot_per_enc'])),
            ]
        )
        info['outinfo'] = outinfo

        df = pd.DataFrame([outinfo])
        df = df.set_index('Database')
        df.index.name = None
        df.index = ut.emap(upper_one, df.index)

        tabular = Tabular(df, colfmt='numeric')
        tabular.theadify = 16
        enc_text = tabular.as_tabular()
        print(enc_text)

        # ut.render_latex(enc_text, dpath=self.dpath, fname='dbstats',
        #                         preamb_extra=['\\usepackage{makecell}'])
        # ut.startfile(_)

        # expt_name = ut.get_stack_frame().f_code.co_name.replace('measure_', '')
        expt_name = 'dbstats'
        self.expt_results[expt_name] = info
        ut.ensuredir(self.dpath)
        ut.save_data(join(self.dpath, expt_name + '.pkl'), info)
        return info

    def measure_all(self):
        r"""
        CommandLine:
            python -m wbia VerifierExpt.measure all GZ_Master1,MantaMatcher,RotanTurtles,LF_ALL
            python -m wbia VerifierExpt.measure all GZ_Master1

        Ignore:
            from wbia.scripts.postdoc import *
            self = VerifierExpt('PZ_MTEST')
            self.measure_all()
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

        # importance = {
        #     task_key: pblm.feature_importance(task_key=task_key)
        #     for task_key in pblm.eval_task_keys
        # }

        task = pblm.samples['match_state']
        scores = pblm.samples.simple_scores['score_lnbnn_1vM']
        lnbnn_ranks = pblm.samples.simple_scores['rank_lnbnn_1vM']
        y = task.indicator_df[task.default_class_name]
        lnbnn_data = pd.concat([scores, lnbnn_ranks, y], axis=1)

        results = {
            'lnbnn_data': lnbnn_data,
            'task_combo_res': self.pblm.task_combo_res,
            # 'importance': importance,
            'data_key': self.data_key,
            'clf_key': self.clf_key,
        }
        expt_name = 'all'
        self.expt_results[expt_name] = results
        ut.save_data(join(str(self.dpath), expt_name + '.pkl'), results)

        task_key = 'match_state'
        self.measure_hard_cases(task_key)

        self.measure_dbstats()

        self.measure_rerank()

        if ut.get_argflag('--draw'):
            self.draw_all()

    def draw_all(self):
        r"""
        CommandLine:
            python -m wbia VerifierExpt.draw_all --db PZ_MTEST
            python -m wbia VerifierExpt.draw_all --db PZ_PB_RF_TRAIN
            python -m wbia VerifierExpt.draw_all --db GZ_Master1
            python -m wbia VerifierExpt.draw_all --db PZ_Master1

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.postdoc import *
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = VerifierExpt(dbname)
            >>>     self.draw_all()
        """
        results = self.ensure_results('all')
        eval_task_keys = set(results['task_combo_res'].keys())
        print('eval_task_keys = {!r}'.format(eval_task_keys))
        task_key = 'match_state'

        if ut.get_argflag('--cases'):
            self.draw_hard_cases(task_key)

        self.write_sample_info()

        self.draw_roc(task_key)
        self.draw_rerank()

        self.write_metrics(task_key)

        self.draw_class_score_hist()
        self.draw_mcc_thresh(task_key)

    def draw_roc(self, task_key='match_state'):
        """
        python -m wbia VerifierExpt.draw roc GZ_Master1 photobomb_state
        python -m wbia VerifierExpt.draw roc GZ_Master1 match_state

        python -m wbia VerifierExpt.draw roc PZ_MTEST
        """
        mpl.rcParams.update(TMP_RC)

        results = self.ensure_results('all')
        data_key = results['data_key']
        clf_key = results['clf_key']

        task_combo_res = results['task_combo_res']
        lnbnn_data = results['lnbnn_data']

        task_key = 'match_state'

        scores = lnbnn_data['score_lnbnn_1vM'].values
        y = lnbnn_data[POSTV].values
        # task_key = 'match_state'
        target_class = POSTV
        res = task_combo_res[task_key][clf_key][data_key]
        cfsm_vsm = vt.ConfusionMetrics().fit(scores, y)
        cfsm_clf = res.confusions(target_class)
        roc_curves = [
            {
                'label': LNBNN,
                'fpr': cfsm_vsm.fpr,
                'tpr': cfsm_vsm.tpr,
                'auc': cfsm_vsm.auc,
            },
            {'label': CLF, 'fpr': cfsm_clf.fpr, 'tpr': cfsm_clf.tpr, 'auc': cfsm_clf.auc},
        ]

        rank_clf_roc_curve = ut.ddict(list)
        rank_lnbnn_roc_curve = ut.ddict(list)

        roc_info_lines = []

        # Check the ROC for only things in the top of the LNBNN ranked lists
        if True:
            rank_auc_df = pd.DataFrame()
            rank_auc_df.index.name = '<=rank'
            nums = [1, 2, 3, 4, 5, 10, 20, np.inf]
            for num in nums:
                ranks = lnbnn_data['rank_lnbnn_1vM'].values
                sub_data = lnbnn_data[ranks <= num]

                scores = sub_data['score_lnbnn_1vM'].values
                y = sub_data[POSTV].values
                probs = res.probs_df[POSTV].loc[sub_data.index].values

                cfsm_scores_rank = vt.ConfusionMetrics().fit(scores, y)
                cfsm_probs_rank = vt.ConfusionMetrics().fit(probs, y)

                # if num == np.inf:
                #     num = 'inf'
                rank_auc_df.loc[num, LNBNN] = cfsm_scores_rank.auc
                rank_auc_df.loc[num, CLF] = cfsm_probs_rank.auc

                rank_lnbnn_roc_curve[num] = {
                    'label': LNBNN,
                    'fpr': cfsm_scores_rank.fpr,
                    'tpr': cfsm_scores_rank.tpr,
                    'auc': cfsm_scores_rank.auc,
                    'tpr@fpr=0': cfsm_scores_rank.get_metric_at_metric(
                        'tpr', 'fpr', 0, tiebreaker='minthresh'
                    ),
                    'thresh@fpr=0': cfsm_scores_rank.get_metric_at_metric(
                        'thresh', 'fpr', 0, tiebreaker='minthresh'
                    ),
                }

                rank_clf_roc_curve[num] = {
                    'label': CLF,
                    'fpr': cfsm_probs_rank.fpr,
                    'tpr': cfsm_probs_rank.tpr,
                    'auc': cfsm_probs_rank.auc,
                    'tpr@fpr=0': cfsm_probs_rank.get_metric_at_metric(
                        'tpr', 'fpr', 0, tiebreaker='minthresh'
                    ),
                    'thresh@fpr=0': cfsm_probs_rank.get_metric_at_metric(
                        'thresh', 'fpr', 0, tiebreaker='minthresh'
                    ),
                }

            auc_text = 'AUC when restricting to the top `num` LNBNN ranks:'
            auc_text += '\n' + str(rank_auc_df)
            print(auc_text)
            roc_info_lines += [auc_text]

        if True:
            tpr_info = []
            at_metric = 'tpr'
            for at_value in [0.25, 0.5, 0.75]:
                info = ut.odict()
                for want_metric in ['fpr', 'n_false_pos', 'n_true_pos', 'thresh']:
                    key = '{}_@_{}={:.3f}'.format(want_metric, at_metric, at_value)
                    info[key] = cfsm_clf.get_metric_at_metric(
                        want_metric, at_metric, at_value, tiebreaker='minthresh'
                    )
                    if key.startswith('n_'):
                        info[key] = int(info[key])
                tpr_info += [(ut.repr4(info, align=True, precision=8))]

            tpr_text = 'Metric TPR relationships\n' + '\n'.join(tpr_info)
            print(tpr_text)

            roc_info_lines += [tpr_text]

            fpr_info = []
            at_metric = 'fpr'
            for at_value in [0, 0.001, 0.01, 0.1]:
                info = ut.odict()
                for want_metric in ['tpr', 'n_false_pos', 'n_true_pos', 'thresh']:
                    key = '{}_@_{}={:.3f}'.format(want_metric, at_metric, at_value)
                    info[key] = cfsm_clf.get_metric_at_metric(
                        want_metric, at_metric, at_value, tiebreaker='minthresh'
                    )
                    if key.startswith('n_'):
                        info[key] = int(info[key])
                fpr_info += [(ut.repr4(info, align=True, precision=8))]

            fpr_text = 'Metric FPR relationships\n' + '\n'.join(fpr_info)
            print(fpr_text)

            roc_info_lines += [fpr_text]

        roc_info_text = '\n\n'.join(roc_info_lines)
        ut.writeto(join(self.dpath, 'roc_info.txt'), roc_info_text)
        # print(roc_info_text)

        fig = pt.figure(fnum=1)  # NOQA
        ax = pt.gca()
        for data in roc_curves:
            ax.plot(
                data['fpr'],
                data['tpr'],
                label='AUC={:.3f} {}'.format(data['auc'], data['label']),
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
        print('wrote roc figure to fig_fpath= {!r}'.format(fig_fpath))

        for num in [1, 2, 5, np.inf]:
            roc_curves_ = [rank_clf_roc_curve[num], rank_lnbnn_roc_curve[num]]
            fig = pt.figure(fnum=1)  # NOQA
            ax = pt.gca()
            for data in roc_curves_:
                ax.plot(
                    data['fpr'],
                    data['tpr'],
                    label='AUC={:.3f} TPR={:.3f} {}'.format(
                        data['auc'], data['tpr@fpr=0'], data['label']
                    ),
                )
            ax.set_xlabel('false positive rate')
            ax.set_ylabel('true positive rate')
            ax.set_title('ROC@rank<={num}'.format(num=num))
            ax.legend()
            pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
            fig.set_size_inches([W, H])

            fname = 'rank_{}_roc_{}.png'.format(num, task_key)
            fig_fpath = join(str(self.dpath), fname)
            vt.imwrite(fig_fpath, pt.render_figure_to_image(fig, dpi=DPI))
            print('wrote roc figure to fig_fpath= {!r}'.format(fig_fpath))

    def draw_rerank(self):
        mpl.rcParams.update(TMP_RC)

        expt_name = 'rerank'
        results = self.ensure_results(expt_name)

        cdfs, infos = list(zip(*results))
        lnbnn_cdf = cdfs[0]
        clf_cdf = cdfs[1]
        fig = pt.figure(fnum=1)
        plot_cmcs([lnbnn_cdf, clf_cdf], ['ranking', 'rank+clf'], fnum=1, ymin=0)
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

    def measure_rerank(self):
        """
        >>> from wbia.scripts.postdoc import *
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

        # These are not gaurenteed to be comparable
        if ibs.dbname == 'RotanTurtles':
            # HACK
            viewpoint_aware = True
        else:
            viewpoint_aware = False

        from wbia.scripts import thesis

        qaids, daids_list, info_list = thesis.Sampler._varied_inputs(
            ibs, aids, viewpoint_aware=viewpoint_aware
        )
        daids = daids_list[0]
        info = info_list[0]

        # ---------------------------
        # Execute the ranking algorithm
        qaids = sorted(qaids)
        daids = sorted(daids)
        cfgdict = pblm._make_lnbnn_pcfg()
        qreq_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict)
        cm_list = qreq_.execute()
        cm_list = [cm.extend_results(qreq_) for cm in cm_list]

        # ---------------------------
        # Measure LNBNN rank probabilities
        top = 20
        rerank_pairs = []
        for cm in cm_list:
            pairs = [infr.e_(cm.qaid, daid) for daid in cm.get_top_aids(top)]
            rerank_pairs.extend(pairs)
        rerank_pairs = list(set(rerank_pairs))

        # ---------------------------
        # Re-rank the those top ranks
        verif = pblm._make_evaluation_verifiers()['match_state']
        # verif = infr.learn_evaluation_verifiers()['match_state']
        probs = verif.predict_proba_df(rerank_pairs)
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

    def ranking_hyperparamm_search(self):
        """
        >>> from wbia.scripts.postdoc import *
        >>> self = VerifierExpt('humpbacks_fb')

        >>> self = VerifierExpt('MantaMatcher')

        >>> self = VerifierExpt('RotanTurtles')
        """
        ut.set_num_procs(4)
        if getattr(self, 'pblm', None) is None:
            self._setup(quick=True)
        # ut.set_process_title(self.dbname + 'hyperparam_search')
        pblm = self.pblm
        ibs = pblm.infr.ibs
        aids = pblm.infr.aids

        def measure_name_cmc(cm_list, nbins=None):
            lnbnn_name_ranks = []
            for cm in cm_list:
                lnbnn_rank = cm.get_name_ranks([cm.qnid])[0]
                lnbnn_name_ranks.append(lnbnn_rank)
            if nbins is None:
                nbins = max(1, max(lnbnn_name_ranks)) + 1
            bins = np.arange(nbins)
            hist = np.histogram(lnbnn_name_ranks, bins=bins)[0]
            cdf = np.cumsum(hist) / sum(hist)
            return cdf

        # These are not gaurenteed to be comparable
        viewpoint_aware = ibs.dbname == 'RotanTurtles'

        from wbia.scripts import thesis

        qaids, daids_list, info_list = thesis.Sampler._varied_inputs(
            ibs, aids, viewpoint_aware=viewpoint_aware
        )
        daids = daids_list[0]

        # ---------------------------
        # Execute the ranking algorithm
        qaids = sorted(qaids)
        daids = sorted(daids)
        cfgdict = pblm._make_lnbnn_pcfg()
        qreq_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict)
        cm_list = qreq_.execute()
        cm_list = [cm.extend_results(qreq_) for cm in cm_list]

        measure_name_cmc(cm_list)[0]

        # ---------------------------
        # Sample a small dataset of borderline pairs to do hyperparamter
        # optimization on
        # target_n_qaids = 64
        # target_n_daids = 128
        # target_n_daids = 64

        # target_n_qaids = 16
        # target_n_daids = 64

        target_n_qaids = 32
        target_n_daids = 128

        border = pd.DataFrame(object(), index=qaids, columns=['qaid', 'rank', 'daids'])
        for cm in cm_list:
            gt_aids = cm.get_top_gt_aids(ibs=ibs)
            gt_ranks = np.array(cm.get_annot_ranks(gt_aids))
            gt_scores = cm.get_annot_scores(gt_aids)
            border.loc[cm.qaid, 'qaid'] = cm.qaid
            border.loc[cm.qaid, 'rank'] = gt_ranks.min()
            border.loc[cm.qaid, 'score'] = min(gt_scores)
            border.loc[cm.qaid, 'daids'] = gt_aids.tolist()

        # Remove very hard and very easy cases
        border = border[border['rank'] > 0]
        border = border[border['score'] > -np.inf]

        # Take things near the middle
        border = border.sort_values('score')
        p1 = int((len(border) / 2) - (target_n_qaids / 2))
        p2 = p1 + target_n_qaids
        selected = border.iloc[p1:p2]

        # vals, bins = np.histogram(border['rank'],
        #                           bins=np.arange(border['rank'].max()))
        # rank_hist = pd.Series(vals, index=bins[:-1])
        # rank_cumhist = rank_hist.cumsum()
        # rank_thresh = np.where(rank_cumhist <= target_n_qaids)[0][-1]
        # selected = border[border['rank'] <= rank_thresh]

        needed_daids = sorted(ut.flatten(selected['daids'].values))
        avail_daids = set(daids) - set(needed_daids)
        n_need = target_n_daids - len(needed_daids)
        confuse_daids = ut.shuffle(list(avail_daids), rng=1393290)[:n_need]

        # ---------------------------
        # Rerun ranking on the smaller dataset

        """
        pip install git+git://github.com/pandas-dev/pandas.git@master

        (26, 876),
        """

        sel_qaids = sorted(selected['qaid'].values)
        sel_daids = sorted(needed_daids + confuse_daids)

        assert len(sel_qaids) > 0, 'no query chips'

        baseline_cfgdict = pblm._make_lnbnn_pcfg()
        baseline_cfgdict['sv_on'] = True
        baseline_cfgdict['affine_invariance'] = False

        ibs._parallel_chips = False

        def to_optimize(dial):
            cfgdict = baseline_cfgdict.copy()
            cfgdict.update(dial)
            qreq_ = ibs.new_query_request(sel_qaids, sel_daids, cfgdict=cfgdict)
            cm_list = qreq_.execute()
            cm_list = [cm.extend_results(qreq_) for cm in cm_list]
            # Optimize this score
            score = measure_name_cmc(cm_list)[0]
            return score

        dials = {
            # 'medianblur_thresh': [0, 10, 20, 30, 40, 50],
            # 'adapteq_ksize': [4, 6, 8, 16, 32],
            # 'adapteq_ksize': [4, 8, 32],
            'adapteq_ksize': [8, 16],
            # 'adapteq_limit': [1, 2, 3, 6],
            # 'adapteq_limit': [1, 6],
            'adapteq_limit': [1, 2],
            # 'affine_invariance': [True, False]
        }

        dials = {
            'adapteq_ksize': [16],
            'adapteq_limit': [2],
            # 'medianblur_thresh': [40, 50, 60, 100],
            'medianblur_thresh': [40, 45, 50],
        }

        combos = list(ut.all_dict_combinations(dials))
        # combos += [
        #     {'adapteq': False},
        #     {'adapteq': False, 'medianblur': False},
        # ]
        combos = ut.shuffle(combos, rng=32132)

        grid_output = {}
        dial = {}
        grid_output['baseline'] = to_optimize(dial)
        print('SO FAR: ')
        print(
            ut.align(ut.repr4(ut.sort_dict(grid_output, 'vals'), precision=4), ':', pos=2)
        )

        for dial in combos:
            score = to_optimize(dial)
            print('FINISHED: ')
            print('score = {!r}'.format(score))
            print('dial = {!r}'.format(dial))
            hashdial = ut.hashdict(dial)
            grid_output[hashdial] = score

            print('SO FAR: ')
            print(
                ut.align(
                    ut.repr4(ut.sort_dict(grid_output, 'vals'), precision=4), ':', pos=2
                )
            )

    def measure_hard_cases(self, task_key):
        """
        Find a failure case for each class

        CommandLine:
            python -m wbia VerifierExpt.measure hard_cases GZ_Master1 match_state
            python -m wbia VerifierExpt.measure hard_cases GZ_Master1 photobomb_state
            python -m wbia VerifierExpt.draw hard_cases GZ_Master1 match_state
            python -m wbia VerifierExpt.draw hard_cases GZ_Master1 photobomb_state

            python -m wbia VerifierExpt.measure hard_cases PZ_Master1 match_state
            python -m wbia VerifierExpt.measure hard_cases PZ_Master1 photobomb_state
            python -m wbia VerifierExpt.draw hard_cases PZ_Master1 match_state
            python -m wbia VerifierExpt.draw hard_cases PZ_Master1 photobomb_state

            python -m wbia VerifierExpt.measure hard_cases PZ_MTEST match_state
            python -m wbia VerifierExpt.draw hard_cases PZ_MTEST photobomb_state

            python -m wbia VerifierExpt.draw hard_cases RotanTurtles match_state
            python -m wbia VerifierExpt.draw hard_cases MantaMatcher match_state

        Ignore:
            >>> task_key = 'match_state'
            >>> task_key = 'photobomb_state'
            >>> from wbia.scripts.postdoc import *
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
        config = pblm.feat_extract_info[data_key][0]
        edges = [case['edge'] for case in cases]
        matches = infr._make_matches_from(edges, config=config)
        match = matches[0]
        match.config

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

    def draw_hard_cases(self, task_key='match_state'):
        """
        draw hard cases with and without overlay

        python -m wbia VerifierExpt.draw hard_cases GZ_Master1 match_state
        python -m wbia VerifierExpt.draw hard_cases PZ_Master1 match_state
        python -m wbia VerifierExpt.draw hard_cases PZ_Master1 photobomb_state
        python -m wbia VerifierExpt.draw hard_cases GZ_Master1 photobomb_state

        python -m wbia VerifierExpt.draw hard_cases RotanTurtles match_state

            >>> from wbia.scripts.postdoc import *
            >>> self = VerifierExpt('PZ_MTEST')
            >>> task_key = 'match_state'
            >>> self.draw_hard_cases(task_key)
        """
        REWORK = False
        REWORK = True

        if REWORK:
            # HACK
            if self.ibs is None:
                self._precollect()

        cases = self.ensure_results(task_key + '_hard_cases')
        print('Loaded {} {} hard cases'.format(len(cases), task_key))

        subdir = 'cases_{}'.format(task_key)
        dpath = join(str(self.dpath), subdir)
        # ut.delete(dpath)
        ut.ensuredir(dpath)
        code_to_nice = self.task_nice_lookup[task_key]

        mpl.rcParams.update(TMP_RC)

        prog = ut.ProgIter(cases, 'draw {} hard case'.format(task_key), bs=False)
        for case in prog:
            aid1, aid2 = case['edge']
            match = case['match']
            real_name, pred_name = case['real'], case['pred']
            real_nice, pred_nice = ut.take(code_to_nice, [real_name, pred_name])

            if real_nice != 'Negative':
                continue

            fname = 'fail_{}_{}_{}_{}'.format(real_name, pred_name, aid1, aid2)
            # Build x-label
            _probs = case['probs']
            probs = ut.odict()
            for k, v in code_to_nice.items():
                if k in _probs:
                    probs[v] = _probs[k]
            probstr = ut.repr2(probs, precision=2, strkeys=True, nobr=True)
            xlabel = 'real={}, pred={},\n{}'.format(real_nice, pred_nice, probstr)
            fig = pt.figure(fnum=1000, clf=True)
            ax = pt.gca()

            # if REWORK:
            #     ibs = self.ibs
            #     annots = ibs.annots([aid1, aid2])
            #     imgs = ibs.images(annots.gids)
            #     xlabel += '\nimg: ' + '-vs-'.join(map(repr, imgs.gnames))
            #     xlabel += '\nname: ' + '-vs-'.join(map(repr, annots.name))
            #     import datetime
            #     delta = ut.get_timedelta_str(datetime.timedelta(seconds=np.diff(annots.image_unixtimes_asfloat)[0]))
            #     xlabel += '\ntimeΔ: ' + delta
            #     xlabel += '\nedge: ' + str(tuple(annots.aids))

            if REWORK:
                ibs = self.ibs
                match.annot1['rchip'] = ibs.annots(match.annot1['aid'], config={}).rchip[
                    0
                ]
                match.annot2['rchip'] = ibs.annots(match.annot2['aid'], config={}).rchip[
                    0
                ]
                # match.annot1['rchip'] = ibs.annots(match.annot1['aid'], config={'medianblur': True, 'adapt_eq': True}).rchip[0]
                # match.annot2['rchip'] = ibs.annots(match.annot2['aid'], config={'medianblur': True, 'adapt_eq': True}).rchip[0]

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

    def write_metrics(self, task_key='match_state'):
        """
        Writes confusion matricies

        CommandLine:
            python -m wbia VerifierExpt.draw metrics PZ_PB_RF_TRAIN match_state
            python -m wbia VerifierExpt.draw metrics GZ_Master1 photobomb_state

            python -m wbia VerifierExpt.draw metrics PZ_Master1,GZ_Master1 photobomb_state,match_state

        Ignore:
            >>> from wbia.scripts.postdoc import *
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

        from wbia.algo.verif import sklearn_utils

        report = sklearn_utils.classification_report2(
            y_true, y_pred, target_names, sample_weight, verbose=False
        )
        metric_df = report['metrics']
        confusion_df = report['confusion']

        multiclass_mcc = report['mcc']
        metric_df.loc['combined', 'mcc'] = multiclass_mcc

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
        df = df.rename_axis({'combined': 'Combined'}, 1)
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

        confusion_tex = confusion_tex.replace('Incomparable', 'Incomp.')
        confusion_tex = confusion_tex.replace('predicted', 'pred')

        ut.write_to(join(dpath, confusion_fname + '.tex'), confusion_tex)
        ut.write_to(join(dpath, metrics_fname + '.tex'), metrics_tex)

        fpath1 = ut.render_latex(confusion_tex, dpath=dpath, fname=confusion_fname)
        fpath2 = ut.render_latex(metrics_tex, dpath=dpath, fname=metrics_fname)
        return fpath1, fpath2

    def write_sample_info(self):
        """
        python -m wbia VerifierExpt.draw sample_info GZ_Master1

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
            ytickformat='%.3f',
            xlim=xlim,
            # title='LNBNN positive separation'
        )
        pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9)
        fig.set_size_inches([W, H])
        return fig

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

        lnbnn_data = results['lnbnn_data']
        scores = lnbnn_data['score_lnbnn_1vM'].values
        y = lnbnn_data[POSTV].values

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
        python -m wbia VerifierExpt.draw mcc_thresh GZ_Master1 match_state
        python -m wbia VerifierExpt.draw mcc_thresh PZ_Master1 match_state

        python -m wbia VerifierExpt.draw mcc_thresh GZ_Master1 photobomb_state
        python -m wbia VerifierExpt.draw mcc_thresh PZ_Master1 photobomb_state

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
                    'label': class_nice + ', t={:.3f}, mcc={:.3f}'.format(t, mcc),
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
            # show_ell=False,
            show_ell=False,
            show_ori=False,
            show_eig=False,
            # ell_alpha=.3,
            modifysize=True,
        )
        # ax.set_xlabel(xlabel)

        self = cls()

        fname = 'custom_match_{}_{}_{}'.format(query_tag, *edge)
        dpath = ut.truepath(self.base_dpath)
        fpath = join(str(dpath), fname + '.jpg')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

    def custom_single_hard_case(self):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.postdoc import *
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.scripts.postdoc
        python -m wbia.scripts.postdoc --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
