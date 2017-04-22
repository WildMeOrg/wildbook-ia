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

def chapter4_collect(defaultdb):
    """
    CommandLine:
        python -m ibeis.scripts.thesis chapter4_collect --db PZ_PB_RF_TRAIN
        python -m ibeis.scripts.thesis chapter4_collect --db GZ_Master1
        python -m ibeis.scripts.thesis chapter4_collect --db PZ_MTEST
        python -m ibeis.scripts.thesis chapter4_collect

    Example:
        >>> from ibeis.scripts.thesis import *
        >>> #defaultdb = 'PZ_PB_RF_TRAIN'
        >>> #defaultdb = 'GZ_Master1'
        >>> defaultdb = 'PZ_MTEST'
        >>> self = chapter4_collect(defaultdb)
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
        data_code = 'PZ_%d' % len(pblm.samples)
    if species_code == 'zebra_grevys':
        species = 'Grévy\'s Zebras'
        data_code = 'GZ_%d' % len(pblm.samples)

    self = ExptChapter4()
    self.eval_task_keys = pblm.eval_task_keys
    self.species = species
    self.data_code = data_code
    self.data_key = data_key
    self.clf_key = clf_key

    self.task_nice_lookup = {
        'match_state': ibs.const.REVIEW.CODE_TO_NICE,
        'photobomb_state': {
            'pb': 'Phototomb'
        }
    }

    if ibs.dbname == 'PZ_MTEST':
        self.dpath = ut.truepath('~/Desktop/mtest_plots')
        self.dpath = pathlib.Path(self.dpath)
        ut.ensuredir(self.dpath)
        # ut.vd(self.dpath)

    #-----------
    # COLLECTION
    #-----------
    task_key = 'match_state'
    if task_key in pblm.eval_task_keys:
        self.build_importance_data(pblm, task_key)
        self.build_roc_data_positive(pblm)
        self.build_score_freq_positive(pblm)
        self.build_hard_cases(pblm, task_key, num_top=2)
        self.build_metrics(pblm, task_key)

    task_key = 'photobomb_state'
    if task_key in pblm.eval_task_keys:
        # self.build_roc_data_photobomb(pblm)
        self.build_importance_data(pblm, task_key)
        self.build_metrics(pblm, task_key)

    fname = 'collected_data_{}.pkl'.format(self.data_code)
    ut.save_data(str(self.dpath.joinpath(fname)), self)
    return self


# @ut.reloadable_class
class ExptChapter4(object):
    """
    Collect data from experiments to visualize

    Ignore:
        >>> from ibeis.scripts.thesis import *
        >>> fpath = ut.glob(ut.truepath('~/Desktop/mtest_plots'), '*.pkl')[0]
        >>> self = ut.load_data(fpath)
    """

    def draw(self):
        task_key = 'match_state'
        if task_key in self.eval_task_keys:
            self.draw_hard_cases(task_key)

            # self.build_score_freq_positive(pblm)
            self.draw_class_score_hist()
            self.draw_roc(task_key)

            self.draw_wordcloud(task_key)
            self.print_top_importance(task_key)
            self.print_metrics(task_key)

        task_key = 'photobomb_state'
        if task_key in self.eval_task_keys:
            self.print_top_importance(task_key)
            self.print_metrics(task_key)

    def __init__(self):
        self.dpath = ut.truepath('~/latex/crall-thesis-2017/figures_pairclf')
        self.dpath = pathlib.Path(self.dpath)
        self.species = None
        self.data_code = None
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

    def build_metrics(self, pblm, task_key):
        import sklearn.metrics
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

    def print_metrics(self, task_key):
        df = self.task_confusion[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.rename_axis(self.task_nice_lookup[task_key], 1)

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
        # sum_real = '\\sum real'
        print(latex_str)

        df = self.task_metrics[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.drop(['markedness', 'bookmaker'], axis=1)
        df['support'] = df['support'].astype(np.int)
        latex_str = df.to_latex(
            float_format=lambda x: '%.2f' % (x)
        )
        print(latex_str)

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
            for idx in range(num_top):
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
        subdir = 'cases_{}_{}'.format(task_key, self.data_code)
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
            match.show(ax, vert=False, ell_alpha=.3, modifysize=True)
            ax.set_xlabel(xlabel)
            fpath = str(dpath.joinpath(fname + '_overlay.jpg'))
            self.savefig(fig, fpath)
            # Draw without feature overlay
            ax.cla()
            match.show(ax, vert=False, overlay=False, modifysize=True)
            ax.set_xlabel(xlabel)
            fpath = str(dpath.joinpath(fname + '.jpg'))
            self.savefig(fig, fpath)

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

        fname = 'score_hist_pos_{}_{}.png'.format(
            self.data_key, self.data_code)
        self.savefig(fig1, str(self.dpath.joinpath(fname)))

        fname = 'score_hist_lnbnn_{}_{}.png'.format(
            self.data_key, self.data_code)
        self.savefig(fig2, str(self.dpath.joinpath(fname)))

    def draw_roc(self, task_key):
        mpl.rcParams.update(TMP_RC)

        roc_data = self.task_rocs[task_key]
        target_class = class_alias(roc_data['target_class'])

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

        fname = 'roc_{}_{}.png'.format(task_key, self.data_code)
        self.savefig(fig, str(self.dpath.joinpath(fname)))

    def draw_wordcloud(self, task_key):
        import plottool as pt
        importances = ut.map_keys(feat_alias, self.task_importance[task_key])

        fig = pt.figure(fnum=1)
        pt.wordcloud(importances, ax=fig.axes[0])

        fname = 'wc_{}_{}.png'.format(task_key, self.data_code)
        fig_fpath = str(self.dpath.joinpath(fname))
        self.savefig(fig, fig_fpath)

    def print_top_importance(self, task_key):
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
        print('TOP 5 importances for ' + task_key)
        print('# of dimensions: %d' % (len(importances)))
        print('\n'.join(ut.align_lines(lines, '&')))

    def savefig(self, fig, fpath):
        image = pt.render_figure_to_image(fig, dpi=256)
        # image = vt.clipwhite(image)
        vt.imwrite(fpath, image)


def class_alias(k):
    if k == 'match':
        return 'Positive'
    if k == 'pb':
        return 'Photobomb'

def feat_alias(k):
    # presentation values for feature dimension
    k = k.replace('weighted_', 'wgt_')
    k = k.replace('norm_x', 'x')
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
