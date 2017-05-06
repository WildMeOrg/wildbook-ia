from ibeis.scripts.script_vsone import OneVsOneProblem
import numpy as np
import utool as ut
import plottool as pt
import vtool as vt
import pathlib
import matplotlib as mpl
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP  # NOQA

TMP_RC = {
    'legend.fontsize': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.facecolor': 'w',
    'font.family': 'DejaVu Sans',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}


@ut.reloadable_class
class Chap3(object):
    """
    """
    def __init__(self):
        self.dpath = ut.truepath('~/latex/crall-thesis-2017/figures_new3')

    @classmethod
    def collect(Chap3, defaultdb='PZ_MTEST'):
        """
        Example:
            >>> from ibeis.scripts.thesis import *
            >>> defaultdb = 'PZ_Master1'
            >>> defaultdb = 'GZ_Master1'
            >>> defaultdb = 'PZ_Master0'
            >>> self = Chap3.collect('PZ_MTEST')
            >>> self = Chap3.collect('PZ_PB_RF_TRAIN')
            >>> self = Chap3.collect('PZ_Master1')
            >>> #self = Chap3.collect('GZ_Master1')
        """
        import ibeis
        self = Chap3()
        self.dbdir = ibeis.sysres.get_args_dbdir(defaultdb=defaultdb)
        self._init()
        return self

    def _init(self):
        import ibeis
        from ibeis.init import main_helpers
        ibs = ibeis.opendb(dbdir=self.dbdir)
        if ibs.dbname == 'PZ_Master0':
            aids = ibs.filter_annots_general(require_timestamp=True, is_known=True,
                                             # require_viewpoint=True,
                                             view='left',
                                             # view_ext=2,  # FIXME
                                             min_pername=2, minqual='poor')
        elif ibs.dbname == 'GZ_Master1':
            aids = ibs.filter_annots_general(require_timestamp=True,
                                             is_known=True,
                                             # require_viewpoint=True,
                                             # view='right',
                                             # view_ext2=2,
                                             # view_ext1=2,
                                             minqual='poor')
            # flags = ['left' not in text for text in ibs.annots(aids).yaw_texts]
            # aids = ut.compress(aids, flags)
        ibs.print_annot_stats(aids, prefix='P')
        main_helpers.monkeypatch_encounters(ibs, aids, minutes=30)
        self.ibs = ibs
        self.aids_pool = aids
        if False:
            # check encounter stats
            annots = ibs.annots(aids)
            annots_list = annots.group(annots.encounter_text)[1]
            # singletons = [a for a in annots_list if len(a) == 1]
            multitons = [a for a in annots_list if len(a) > 1]
            deltas = []
            for a in multitons:
                times = a.image_unixtimes_asfloat
                deltas.append(max(times) - min(times))
            ut.lmap(ut.get_posix_timedelta_str, sorted(deltas))

    # def _baseline_inputs(self):
    #     aids = self.ibs.filter_annots_general(self.aids_pool, minqual='ok',
    #                                           view='primary')
    #     annots = self.ibs.annots(aids)
    #     encounters = annots.group(annots.encounter_text)[1]
    #     # Choose annotation per encounter
    #     rng = np.random.RandomState(0)
    #     annots2 = self.ibs.annots([rng.choice(enc.aids) for enc in encounters])
    #     names = annots2.group(annots2.nids)[1]
    #     singletons = [a for a in names if len(a) == 1]
    #     multitons = [a for a in names if len(a) > 1]
    #     qaids = []
    #     daids = []
    #     for a in multitons:
    #         qaid, daid = rng.choice(a, 2, replace=False)
    #         qaids.append(qaid)
    #         daids.append(daid)
    #     daids.extend(ut.flatten(singletons))
    #     ibs.print_annotconfig_stats(qaids, daids)

    def _inputs(self):
        from ibeis.init.filter_annots import encounter_crossval
        # Sample a dataset
        ibs = self.ibs
        # aids = self.ibs.filter_annots_general(self.aids_pool, minqual='ok',
        #                                       view='primary')
        # aids = self.ibs.filter_annots_general(self.aids_pool, minqual='poor')
        aids = self.aids_pool
        expanded_aids = encounter_crossval(self.ibs, aids, qenc_per_name=1,
                                           annots_per_enc=1, denc_per_name=1,
                                           rebalance=True, rng=0)
        qaids, daids = expanded_aids[0]
        # if True:
        #     print_cfg = dict(per_multiple=False, use_hist=False)
        #     ibs.print_annotconfig_stats(qaids, daids, **print_cfg)
        return ibs, qaids, daids

    def _vary_dpername_inputs(self):
        from ibeis.init.filter_annots import encounter_crossval
        # Sample a dataset
        ibs = self.ibs
        # aids = self.ibs.filter_annots_general(self.aids_pool, minqual='poor')
        aids = self.aids_pool
        denc_per_name = 3
        enc_splits, nid_to_confusors = encounter_crossval(
            self.ibs, aids, qenc_per_name=1, annots_per_enc=1,
            denc_per_name=denc_per_name, rebalance=True, rng=0, early=True)
        qencs, dencs = enc_splits[0]
        qaids = ut.flatten(ut.flatten(qencs))

        confusor_pool = ut.flatten(ut.flatten(nid_to_confusors.values()))
        confusor_pool = ut.shuffle(confusor_pool, rng=0)

        target_daids_list = []
        # Keep the number of annots in the database (more or less) constant
        for num in range(1, denc_per_name + 1):
            denc_ = ut.take_column(dencs, list(range(num)))
            daids_ = ut.flatten(ut.flatten(denc_))
            target_daids_list.append(daids_)

        dbsize_list = ut.lmap(len, target_daids_list)
        min_dsize = min(dbsize_list)
        max_dsize = max(dbsize_list)
        num_need = max_dsize - min_dsize
        num_extra = len(confusor_pool) - num_need
        if len(confusor_pool) < num_need:
            print('Warning: not enough confusors to pad dbsize')

        confusor_daids_list = []
        for dsize in dbsize_list:
            num_take = max_dsize - dsize + num_extra
            confusor_daids_list.append(confusor_pool[:num_take])

        daids_list = [a + b for a, b in zip(target_daids_list,
                                            confusor_daids_list)]
        print('#qaids = %r' % (len(qaids),))
        print('num_need = %r' % (num_need,))
        print('max_dsize = %r' % (max_dsize,))
        print('num_extra = %r' % (num_extra,))
        print(list(map(len, daids_list)))
        if True:
            print_cfg = dict(per_multiple=False, use_hist=False)
            for daids in daids_list:
                ibs.print_annotconfig_stats(qaids, daids, **print_cfg)
        return ibs, qaids, daids_list

    def _vary_dbsize_inputs(self):
        """
        Vary num per name and total number of annots
        """
        from ibeis.init.filter_annots import encounter_crossval
        # Sample a dataset
        ibs = self.ibs
        # aids = self.ibs.filter_annots_general(self.aids_pool, minqual='poor')
        aids = self.aids_pool
        denc_per_name = 2
        enc_splits, nid_to_confusors = encounter_crossval(
            self.ibs, aids, qenc_per_name=1, annots_per_enc=1,
            denc_per_name=denc_per_name, rebalance=True, rng=0, early=True)
        qencs, dencs = enc_splits[0]
        qaids = sorted(ut.flatten(ut.flatten(qencs)))

        confusor_pool = ut.flatten(ut.flatten(nid_to_confusors.values()))
        confusor_pool = ut.shuffle(confusor_pool, rng=0)

        target_daids_list = []
        # Keep the number of annots in the database (more or less) constant
        for num in range(1, denc_per_name + 1):
            denc_ = ut.take_column(dencs, list(range(num)))
            daids_ = ut.flatten(ut.flatten(denc_))
            target_daids_list.append(daids_)

        dbsize_list = ut.lmap(len, target_daids_list)
        min_dsize = min(dbsize_list)
        max_dsize = max(dbsize_list)
        num_need = max_dsize - min_dsize
        num_extra = len(confusor_pool) - num_need
        if len(confusor_pool) < num_need:
            print('Warning: not enough confusors to pad dbsize')

        confusor_daids_list = []
        for dsize in dbsize_list:
            num_take = max_dsize - dsize
            confusor_daids_list.append(confusor_pool[:num_take])

        extra_fracs = [0, .5, 1]
        nextra_list = ut.unique([int(num_extra * frac) for frac in extra_fracs])
        print('nextra_list = %r' % (nextra_list,))

        daids_list_ = [a + b for a, b in zip(target_daids_list,
                                             confusor_daids_list)]
        print(list(map(len, daids_list_)))

        daids_list = [sorted(daids + confusor_pool[len(confusor_pool) - n:])
                      for n in nextra_list for daids in daids_list_]
        print(list(map(len, daids_list)))

        print('nextra_list = %r' % (nextra_list,))
        print('#qaids = %r' % (len(qaids),))
        print('num_need = %r' % (num_need,))
        print('max_dsize = %r' % (max_dsize,))
        print('num_extra = %r' % (num_extra,))
        if True:
            print_cfg = dict(per_multiple=False, use_hist=False)
            for daids in daids_list:
                ibs.print_annotconfig_stats(qaids, daids, **print_cfg)
        return ibs, qaids, daids_list

    def _exec_ranking(self, ibs, qaids, daids, cfgdict):
        # ibs, qaids, daids = self._inputs()
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
        cdf = (np.cumsum(hist) / sum(hist))
        return cdf

    def plot_cmcs(self, cdfs, labels):
        cdfs = np.array(cdfs)
        # Sort so the best is on top
        sortx = np.lexsort(cdfs.T[::-1])[::-1]
        cdfs = cdfs[sortx]
        labels = ut.take(labels, sortx)
        # Truncte to 20 ranks
        num_ranks = min(cdfs.shape[-1], 20)
        xdata = np.arange(1, num_ranks + 1)
        cdfs_trunc = cdfs[:, 0:num_ranks]
        label_list = ['%6.2f%% - %s' % (cdf[0] * 100, lbl)
                      for cdf, lbl in zip(cdfs_trunc, labels)]
        pt.multi_plot(
            xdata, cdfs_trunc, label_list=label_list,
            xlabel='rank', ylabel='match probability',
            use_legend=True, legend_loc='lower right', num_yticks=6, ymax=1,
            ymin=.5, ypad=.005, xmin=.9, num_xticks=5, xmax=num_ranks + 1 - .5,
        )

    def baseline(self):
        cdfs, labels = zip(*[self.measure_baseline()])
        self.plot_cmcs(cdfs, labels)

    def foregroundness(self):
        """
        Example:
            >>> from ibeis.scripts.thesis import *
            >>> self = Chap3.collect('PZ_PB_RF_TRAIN')
            >>> self = Chap3.collect('PZ_MTEST')
            >>> self = Chap3.collect('GZ_Master1')
        """
        cdfs, labels = zip(*[
            self.measure_baseline(),
            self.measure_foregroundness()
        ])
        self.plot_cmcs(cdfs, labels)

    def measure_baseline(self):
        ibs, qaids, daids = self._inputs()
        cfgdict = {}
        cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
        return cdf, 'baseline'

    def measure_foregroundness(self):
        ibs, qaids, daids = self._inputs()
        cfgdict = {'fg_on': False}
        cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
        return cdf, 'without foregroundness'

    def measure_invariance(self):
        ALIAS_KEYS = ut.invert_dict({
            'RI': 'rotation_invariance',
            'AI': 'affine_invariance',
            'QRH': 'query_rotation_heuristic',
        })
        invar = [
            {'affine_invariance':  True, 'rotation_invariance': False, 'query_rotation_heuristic': False},
            {'affine_invariance':  True, 'rotation_invariance':  True, 'query_rotation_heuristic': False},
            {'affine_invariance': False, 'rotation_invariance':  True, 'query_rotation_heuristic': False},
            {'affine_invariance': False, 'rotation_invariance': False, 'query_rotation_heuristic': False},
            {'affine_invariance':  True, 'rotation_invariance': False, 'query_rotation_heuristic':  True},
            {'affine_invariance': False, 'rotation_invariance': False, 'query_rotation_heuristic':  True},
        ]
        ibs, qaids, daids = self._inputs()
        pairs = []
        for cfgdict in invar:
            cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
            label = ut.get_cfg_lbl(ut.map_keys(ALIAS_KEYS, cfgdict))[1:]
            label = label.replace('True', 'T').replace('False', 'F')
            # TODO: put a baseline label on whichever of these corresponds to
            # baseline
            pairs.append((cdf, label))
        # pairs.append(self.measure_baseline())
        cdfs, labels = zip(*pairs)
        # self.plot_cmcs(cdfs, labels)

    def measure_nsum(self):
        ibs, qaids, daids_list = self._vary_dpername_inputs()
        cfgdict1 = {
            'score_method': 'nsum',
            'score_method': 'nsum',
            'query_rotation_heuristic': True,
        }
        cfgdict2 = {
            'score_method': 'csum',
            'score_method': 'csum',
            'query_rotation_heuristic': True,
        }
        pairs = []
        for count, daids in enumerate(daids_list, start=1):
            cdf1 = self._exec_ranking(ibs, qaids, daids, cfgdict1)
            pairs.append((cdf1, 'nsum,dpername=%d' % count))
            cdf2 = self._exec_ranking(ibs, qaids, daids, cfgdict2)
            pairs.append((cdf2, 'csum,dpername=%d' % count))
        cdfs, labels = zip(*pairs)
        # self.plot_cmcs(cdfs, labels)
        # pt.gca().set_title('#qaids=%r #daids=%r' % (len(qaids), len(daids)))

    def measure_k_dbsize(self):
        ibs, qaids, daids_list = self._vary_dbsize_inputs()
        cfg_grid = {
            'query_rotation_heuristic': True,
            'K': [1, 2, 4, 6, 10],
        }
        pairs = []
        for cfgdict in ut.all_dict_combinations(cfg_grid):
            for daids in daids_list:
                cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
                label = 'K=%r, dsize=%r' % (cfgdict['K'], len(daids))
                pairs.append((cdf, label))
        cdfs, labels = zip(*pairs)

    def measure_smk(self):
        from ibeis.algo.smk.smk_pipeline import SMKRequest
        # ibs = ibeis.opendb('PZ_MTEST')
        ibs, qaids, daids = self._inputs()
        config = {'nAssign': 1, 'num_words': 8000, 'sv_on': True}
        qreq_ = SMKRequest(ibs, qaids, daids, config)
        qreq_.ensure_data()
        cm_list = qreq_.execute()
        cm_list = [cm.extend_results(qreq_) for cm in cm_list]
        name_ranks = [cm.get_name_ranks([cm.qnid])[0] for cm in cm_list]
        # Measure rank probabilities
        bins = np.arange(len(qreq_.dnids))
        hist = np.histogram(name_ranks, bins=bins)[0]
        cdf = (np.cumsum(hist) / sum(hist))
        pairs = [(cdf, 'smk')]
        pairs.append(self.measure_baseline())

        # cdfs, labels = zip(*pairs)
        # self.plot_cmcs(cdfs, labels)
        # pt.gca().set_title('#qaids=%r #daids=%r' % (len(qaids), len(daids)))

    def measure_all(self):
        """
        from ibeis.scripts.thesis import *
        self = Chap3.collect('PZ_Master0')
        self = Chap3.collect('GZ_Master1')
        self = Chap3.collect('PZ_Master1')
        """
        self.measure_baseline()
        self.measure_foregroundness()
        self.measure_smk()
        self.measure_nsum()
        self.measure_k_dbsize()
        self.measure_invariance()


# @ut.reloadable_class
class Chap4(object):
    """
    Collect data from experiments to visualize

    Ignore:
        >>> from ibeis.scripts.thesis import *
        >>> fpath = ut.glob(ut.truepath('~/Desktop/mtest_plots'), '*.pkl')[0]
        >>> self = ut.load_data(fpath)
    """

    @classmethod
    def collect(Chap4, defaultdb):
        r"""
        CommandLine:
            python -m ibeis.scripts.thesis Chap4.collect --db PZ_PB_RF_TRAIN
            python -m ibeis.scripts.thesis Chap4.collect --db PZ_MTEST
            python -m ibeis.scripts.thesis Chap4.collect

            python -m ibeis.scripts.thesis Chap4.collect --db GZ_Master1

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> defaultdb = 'PZ_PB_RF_TRAIN'
            >>> #defaultdb = 'GZ_Master1'
            >>> defaultdb = 'PZ_MTEST'
            >>> self = Chap4.collect(defaultdb)
            >>> self.draw()
        """
        pblm = OneVsOneProblem.from_empty(defaultdb)
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key
        pblm.eval_task_keys = ['match_state', 'photobomb_state']
        pblm.eval_data_keys = [data_key]
        pblm.eval_clf_keys = [clf_key]
        pblm.setup_evaluation()

        # pblm.evaluate_classifiers()
        ibs = pblm.infr.ibs
        pblm.samples.print_info()

        species_code = ibs.get_database_species(pblm.infr.aids)[0]
        if species_code == 'zebra_plains':
            species = 'Plains Zebras'
        if species_code == 'zebra_grevys':
            species = 'Grévy\'s Zebras'
        dbcode = '{}_{}'.format(ibs.dbname, len(pblm.samples))

        self = Chap4()
        self.eval_task_keys = pblm.eval_task_keys
        self.species = species
        self.dbcode = dbcode
        self.data_key = data_key
        self.clf_key = clf_key

        self.task_nice_lookup = {
            'match_state': ibs.const.REVIEW.CODE_TO_NICE,
            'photobomb_state': {
                'pb': 'Phototomb',
                'notpb': 'Not Phototomb',
            }
        }

        if ibs.dbname == 'PZ_MTEST':
            self.dpath = ut.truepath('~/Desktop/' + self.dbcode)
        else:
            base = '~/latex/crall-thesis-2017/figures_pairclf'
            self.dpath = ut.truepath(base + '/' + self.dbcode)
        self.dpath = pathlib.Path(self.dpath)
        ut.ensuredir(self.dpath)

        #-----------
        # COLLECTION
        #-----------
        task_key = 'match_state'
        if task_key in pblm.eval_task_keys:
            self.build_importance_data(pblm, task_key)
            self.build_roc_data_positive(pblm)
            self.build_score_freq_positive(pblm)
            self.build_hard_cases(pblm, task_key, num_top=4)
            self.build_metrics(pblm, task_key)

        task_key = 'photobomb_state'
        if task_key in pblm.eval_task_keys:
            # self.build_roc_data_photobomb(pblm)
            self.build_importance_data(pblm, task_key)
            self.build_metrics(pblm, task_key)

        fname = 'collected_data.pkl'
        ut.save_data(str(self.dpath.joinpath(fname)), self)
        return self

    def __init__(self):
        self.dpath = ut.truepath('~/latex/crall-thesis-2017/figures_pairclf')
        self.dpath = pathlib.Path(self.dpath)
        self.species = None
        self.dbcode = None
        self.data_key = None
        self.clf_key = None
        # info
        self.eval_task_keys = None
        self.task_importance = {}
        self.task_rocs = {}
        self.hard_cases = {}
        self.task_confusion = {}
        self.task_metrics = {}
        self.task_nice_lookup = None

        self.score_hist_lnbnn = None
        self.score_hist_pos = None

    def draw(self):
        task_key = 'photobomb_state'
        if task_key in self.eval_task_keys:
            self.write_importance(task_key)
            self.write_metrics(task_key)

        task_key = 'match_state'
        if task_key in self.eval_task_keys:

            # self.build_score_freq_positive(pblm)
            self.draw_class_score_hist()
            self.draw_roc(task_key)

            self.draw_wordcloud(task_key)
            self.write_importance(task_key)
            self.write_metrics(task_key)

            if not ut.get_argflag('--nodraw'):
                self.draw_hard_cases(task_key)

    def build_metrics(self, pblm, task_key):
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        res.augment_if_needed()
        pred_enc = res.clf_probs.argmax(axis=1)
        y_pred = pred_enc
        y_true = res.y_test_enc
        sample_weight = res.sample_weight
        target_names = res.class_names

        from ibeis.scripts import sklearn_utils
        metric_df, confusion_df = sklearn_utils.classification_report2(
            y_true, y_pred, target_names, sample_weight, verbose=False)
        self.task_confusion[task_key] = confusion_df
        self.task_metrics[task_key] = metric_df
        return ut.partial(self.write_metrics, task_key)

    def write_metrics(self, task_key='match_state'):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap4.write_metrics --db PZ_PB_RF_TRAIN --task-key=match_state
            python -m ibeis.scripts.thesis Chap4.write_metrics --db GZ_Master1 --task-key=match_state

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> kwargs = ut.argparse_funckw(Chap4.write_metrics)
            >>> defaultdb = 'GZ_Master1'
            >>> defaultdb = 'PZ_PB_RF_TRAIN'
            >>> self, pblm = precollect(defaultdb)
            >>> task_key = kwargs['task_key']
            >>> self.build_metrics(pblm, task_key)
            >>> self.write_metrics(task_key)
        """
        df = self.task_confusion[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.rename_axis(self.task_nice_lookup[task_key], 1)
        df.index.name = None
        df.columns.name = None

        latex_str = df.to_latex(
            float_format=lambda x: '' if np.isnan(x) else str(int(x)),
        )
        sum_pred = df.index[-1]
        sum_real = df.columns[-1]
        latex_str = latex_str.replace(sum_pred, '$\sum$ predicted')
        latex_str = latex_str.replace(sum_real, '$\sum$ real')
        colfmt = '|l|' + 'r' * (len(df) - 1) + '|l|'
        newheader = '\\begin{tabular}{%s}' % (colfmt,)
        latex_str = '\n'.join([newheader] + latex_str.split('\n')[1:])
        lines = latex_str.split('\n')
        lines = lines[0:-4] + ['\\midrule'] + lines[-4:]
        latex_str = '\n'.join(lines)
        latex_str = latex_str.replace('midrule', 'hline')

        fname = 'confusion_{}.tex'.format(task_key)
        print(latex_str)
        ut.write_to(str(self.dpath.joinpath(fname)), latex_str)
        # sum_real = '\\sum real'

        df = self.task_metrics[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.drop(['markedness', 'bookmaker'], axis=1)
        df.index.name = None
        df.columns.name = None
        df['support'] = df['support'].astype(np.int)
        latex_str = df.to_latex(
            float_format=lambda x: '%.2f' % (x)
        )
        lines = latex_str.split('\n')
        lines = lines[0:-4] + ['\\midrule'] + lines[-4:]
        latex_str = '\n'.join(lines)
        print(latex_str)
        fname = 'eval_metrics_{}.tex'.format(task_key)
        ut.write_to(str(self.dpath.joinpath(fname)), latex_str)

    def write_importance(self, task_key):
        # Print info for latex table
        importances = self.task_importance[task_key]
        vals = importances.values()
        items = importances.items()
        top_dims = ut.sortedby(items, vals)[::-1]
        lines = []
        for k, v in top_dims[:5]:
            k = feat_alias(k)
            k = k.replace('_', '\\_')
            lines.append('{} & {:.4f} \\\\'.format(k, v))
        latex_str = '\n'.join(ut.align_lines(lines, '&'))

        fname = 'feat_importance_{}.tex'.format(task_key)
        ut.write_to(str(self.dpath.joinpath(fname)), latex_str)

        print('TOP 5 importances for ' + task_key)
        print('# of dimensions: %d' % (len(importances)))
        print()

    def build_importance_data(self, pblm, task_key):
        self.task_importance[task_key] = pblm.feature_importance(task_key=task_key)

    def build_score_freq_positive(self, pblm):
        task_key = 'match_state'
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        y = res.target_bin_df[POSTV]
        scores = res.probs_df[POSTV]
        bins = np.linspace(0, 1, 100)
        pos_freq = np.histogram(scores[y], bins)[0]
        neg_freq = np.histogram(scores[~y], bins)[0]
        pos_freq = pos_freq / pos_freq.sum()
        neg_freq = neg_freq / neg_freq.sum()
        freqs = {'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}
        self.score_hist_pos = freqs

        scores = pblm.samples.simple_scores['score_lnbnn_1vM']
        y = pblm.samples[task_key].indicator_df[POSTV].loc[scores.index]
        # Get 95% of the data at least
        maxbin = scores[scores.argsort()][-max(1, int(len(scores) * .05))]
        bins = np.linspace(0, max(maxbin, 10), 100)
        pos_freq = np.histogram(scores[y], bins)[0]
        neg_freq = np.histogram(scores[~y], bins)[0]
        pos_freq = pos_freq / pos_freq.sum()
        neg_freq = neg_freq / neg_freq.sum()
        freqs = {'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}
        self.score_hist_lnbnn = freqs

    def build_roc_data_positive(self, pblm):
        task_key = 'match_state'
        target_class = POSTV
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        c2 = pblm.simple_confusion('score_lnbnn_1vM', task_key=task_key)
        c3 = res.confusions(target_class)
        self.task_rocs[task_key] = {
            'target_class': target_class,
            'curves': [
                {'label': 'LNBNN', 'fpr': c2.fpr, 'tpr': c2.tpr, 'auc': c2.auc},
                {'label': 'learned', 'fpr': c3.fpr, 'tpr': c3.tpr, 'auc': c3.auc},
            ]
        }

    def build_roc_data_photobomb(self, pblm):
        task_key = 'photobomb_state'
        target_class = 'pb'
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        c1 = res.confusions(target_class)
        self.task_rocs[task_key] = {
            'target_class': target_class,
            'curves': [
                {'label': 'learned', 'fpr': c1.fpr, 'tpr': c1.tpr, 'auc': c1.auc},
            ]
        }

    def build_hard_cases(self, pblm, task_key, num_top=2):
        """ Find a failure case for each class """
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        case_df = res.hardness_analysis(pblm.samples, pblm.infr)
        # group = case_df.sort_values(['real_conf', 'easiness'])
        case_df = case_df.sort_values(['easiness'])

        # failure_cases = case_df[(case_df['real_conf'] > 0) & case_df['failed']]
        failure_cases = case_df[case_df['failed']]
        if len(failure_cases) == 0:
            print('No reviewed failures exist. Do pblm.qt_review_hardcases')

        cases = []
        for (pred, real), group in failure_cases.groupby(('pred', 'real')):
            # Prefer examples we have manually reviewed before
            group = group.sort_values(['real_conf', 'easiness'])
            for idx in range(min(num_top, len(group))):
                case = group.iloc[idx]
                edge = tuple(ut.take(case, ['aid1', 'aid2']))
                cases.append({
                    'edge': edge,
                    'real': res.class_names[real],
                    'pred': res.class_names[pred],
                    'probs': res.probs_df.loc[edge]
                })

        # Augment cases with their one-vs-one matches
        infr = pblm.infr
        config = pblm.hyper_params['vsone_match'].asdict()
        config.update(pblm.hyper_params['vsone_kpts'])
        edges = [case['edge'] for case in cases]
        matches = infr._exec_pairwise_match(edges, config)
        for case, match in zip(cases, matches):
            # TODO: decouple the match from the database
            # store its chip fpath and other required info
            case['match'] = match

        self.hard_cases[task_key] = cases

    def draw_hard_cases(self, task_key):
        """ draw hard cases with and without overlay """
        subdir = 'cases_{}'.format(task_key)
        dpath = self.dpath.joinpath(subdir)
        ut.ensuredir(dpath)
        code_to_nice = self.task_nice_lookup[task_key]

        for case in ut.ProgIter(self.hard_cases[task_key], 'draw hard case'):
            aid1, aid2 = case['edge']
            real_name = case['real']
            pred_name = case['pred']
            match = case['match']
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
            fig = pt.figure(fnum=1, clf=True)
            ax = pt.gca()
            # Draw with feature overlay
            match.show(ax, vert=False,
                       heatmask=True,
                       show_lines=False,
                       show_ell=False,
                       show_ori=False,
                       show_eig=False,
                       # ell_alpha=.3,
                       modifysize=True)
            ax.set_xlabel(xlabel)
            # fpath = str(dpath.joinpath(fname + '_overlay.jpg'))
            fpath = str(dpath.joinpath(fname + '.jpg'))
            self.savefig(fig, fpath)
            # Draw without feature overlay
            # ax.cla()
            # match.show(ax, vert=False, overlay=False, modifysize=True)
            # ax.set_xlabel(xlabel)
            # fpath = str(dpath.joinpath(fname + '.jpg'))
            # self.savefig(fig, fpath)

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
        fig.set_size_inches([7.4375,  3.125])
        return fig

    def draw_class_score_hist(self):
        """ Plots distribution of positive and negative scores """
        freqs = self.score_hist_pos
        fig1 = self._draw_score_hist(freqs, 'positive probability', 1)

        freqs = self.score_hist_lnbnn
        fig2 = self._draw_score_hist(freqs, 'LNBNN score', 2)

        fname = 'score_hist_pos_{}.png'.format(self.data_key)
        self.savefig(fig1, str(self.dpath.joinpath(fname)))

        fname = 'score_hist_lnbnn.png'
        self.savefig(fig2, str(self.dpath.joinpath(fname)))

    def draw_roc(self, task_key):
        mpl.rcParams.update(TMP_RC)

        roc_data = self.task_rocs[task_key]

        fig = pt.figure(fnum=1)  # NOQA
        ax = pt.gca()
        for data in roc_data['curves']:
            ax.plot(data['fpr'], data['tpr'],
                    label='%s AUC=%.2f' % (data['label'], data['auc']))
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('true positive rate')
        # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
        ax.legend()
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9)
        fig.set_size_inches([7.4375,  3.125])

        fname = 'roc_{}.png'.format(task_key)
        self.savefig(fig, str(self.dpath.joinpath(fname)))

    def draw_wordcloud(self, task_key):
        import plottool as pt
        importances = ut.map_keys(feat_alias, self.task_importance[task_key])

        fig = pt.figure(fnum=1)
        pt.wordcloud(importances, ax=fig.axes[0])

        fname = 'wc_{}.png'.format(task_key)
        fig_fpath = str(self.dpath.joinpath(fname))
        self.savefig(fig, fig_fpath)

    def savefig(self, fig, fpath):
        image = pt.render_figure_to_image(fig, dpi=256)
        # image = vt.clipwhite(image)
        vt.imwrite(fpath, image)


def feat_alias(k):
    # presentation values for feature dimension
    k = k.replace('weighted_', 'wgt_')
    k = k.replace('norm_x', 'x')
    k = k.replace('norm_y', 'y')
    k = k.replace('yaw', 'view')
    return k


def test_mcc():
    num = 100
    xdata = np.linspace(0, 1, num * 2)
    ydata = np.linspace(1, -1, num * 2)
    pt.plt.plot(xdata, ydata, '--k',
                label='linear')

    y_true = [1] * num + [0] * num
    y_pred = y_true[:]
    import sklearn.metrics
    xs = []
    for i in range(0, len(y_true)):
        y_pred[-i] = 1 - y_pred[-i]
        xs.append(sklearn.metrics.matthews_corrcoef(y_true, y_pred))

    import plottool as pt
    pt.plot(xdata, xs, label='change one class at a time')

    y_true = ut.flatten(zip([1] * num, [0] * num))
    y_pred = y_true[:]
    import sklearn.metrics
    xs = []
    for i in range(0, len(y_true)):
        y_pred[-i] = 1 - y_pred[-i]
        xs.append(sklearn.metrics.matthews_corrcoef(y_true, y_pred))

    pt.plot(xdata, xs, label='change classes evenly')
    pt.gca().legend()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts.thesis
        python -m ibeis.scripts.thesis --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
