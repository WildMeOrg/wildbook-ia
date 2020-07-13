# -*- coding: utf-8 -*-
"""
./dev.py -t custom:affine_invariance=False,adapteq=True,fg_on=False --db Elephants_drop1_ears --allgt --index=0:10 --guiview  # NOQA
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import numpy as np
import utool as ut
import vtool as vt
from wbia.expt import draw_helpers
from six.moves import map, range, reduce

print, rrr, profile = ut.inject2(__name__)


def scorediff(ibs, testres, f=None, verbose=None):
    r"""
    Args:
        ibs (wbia.IBEISController):  image analysis api
        testres (wbia.TestResult):  test result object
        f (None): (default = None)
        verbose (bool):  verbosity flag(default = None)

    CommandLine:
        python -m wbia.expt.experiment_drawing scorediff --db PZ_Master1 -a timectrl -t best --show
        python -m wbia.expt.experiment_drawing scorediff --db PZ_MTEST -a default -t best --show

        python -m wbia.expt.experiment_drawing scorediff --db humpbacks_fb \
            -a default:has_any=hasnotch,mingt=2 \
            -t default:proot=BC_DTW,decision=max,crop_dim_size=500,crop_enabled=True,use_te_scorer=False,manual_extract=True,ignore_notch=True,te_net=annot_simple --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_drawing import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs, testres = main_helpers.testdata_expts(defaultdb, a=['timectrl'], t=['best'])
        >>> f = ut.get_argval(('--filt', '-f'), type_=list, default=[''])
        >>> scorediff(ibs, testres, f=f, verbose=ut.VERBOSE)
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt

    for cfgx in range(testres.nConfig):
        cm_list = testres.cfgx2_qreq_[cfgx].execute()
        aid_list = [cm.qaid for cm in cm_list]
        score_diffs = []
        top_scores = []
        for cm in cm_list:
            annot_scores = sorted(cm.annot_score_list, key=lambda x: -x)
            if False:
                # We want to measure just the top 2 regardless of what they are
                diff = annot_scores[0] - annot_scores[1]
            else:
                try:
                    # We want to measure just the top 2 wrt truth
                    top_true_aid = cm.get_top_gt_aids(ibs)[0]
                    top_false_aid = cm.get_top_gf_aids(ibs)[0]
                    gt_score = cm.get_annot_scores([top_true_aid])[0]
                    gf_score = cm.get_annot_scores([top_false_aid])[0]
                    diff = gt_score - gf_score
                except IndexError:
                    continue
                    # diff = 0  # np.nan
            top_scores.append(annot_scores[0])
            score_diffs.append(diff)
        score_diffs = np.array(score_diffs)
        top_scores = np.array(top_scores)

        succ = score_diffs > 0
        fail = score_diffs <= 0

        # succ = testres.get_truth2_prop()[0]['gt']['rank'][:, 0] == 0
        # fail = testres.get_truth2_prop()[0]['gt']['rank'][:, 0] != 0

        # fail = np.where(testres.get_truth2_prop()[0]['gt']['score'][:, 0] != 0)
        width = 0.25
        succ_max = max(2, score_diffs[succ].max())
        fail_max = max(2, -score_diffs[fail].min())
        num_succ_bins = int(succ_max / width)
        num_fail_bins = int(fail_max / width)
        succ_bins = np.linspace(0, succ_max, num_succ_bins + 1)
        fail_bins = np.linspace(-fail_max, 0, num_fail_bins + 1)
        print('succ_bins = %r' % (succ_bins,))
        print('fail_bins = %r' % (fail_bins,))
        succ_hist, succ_edges = np.histogram(score_diffs[succ], bins=succ_bins)
        fail_hist, fail_edges = np.histogram(score_diffs[fail], bins=fail_bins)
        # bin_max = (score_diffs.max() - score_diffs.min()) / 50

        ydata_list = [succ_hist, fail_hist]
        bins_list = [succ_bins, fail_bins]
        # xdata_list = [(bins[:-1] + bins[1:]) / 2 for bins in bins_list]
        xdata_list = [bins[:-1] for bins in bins_list]
        width_list = [np.diff(bins)[0] for bins in bins_list]

        # nbins = 8
        # bin_width = (score_diffs.mean() + score_diffs.std()) / nbins
        # bin_width = int(np.ceil((score_diffs.mean() + score_diffs.std() / 4) / nbins))
        # bin_width = 1
        # bins = np.arange(nbins) * bin_width
        # succ_hist, succ_edges = np.histogram(score_diffs[succ], bins=bins)
        # fail_hist, fail_edges = np.histogram(score_diffs[fail], bins=bins)
        # hist, edges = np.histogram(score_diffs)
        # print('hist = %r' % (hist,))
        # ymax = max(max(succ_hist), max(fail_hist)) * 1.1

        species = ibs.get_dominant_species(testres.qaids)
        species_nice = ibs.get_species_nice(
            ibs.get_species_rowids_from_text(ibs.get_dominant_species(testres.qaids))
        )
        species_nice = {
            'zebra_grevys': "Grevy's Zebras",
            'zebra_plains': 'Plains Zebras',
            'giraffe_masai': 'Masai Giraffes',
        }.get(species, species_nice)

        pt.multi_plot(
            xdata_list,
            ydata_list,
            label_list=['correct match at rank 1', 'correct match above rank 1'],
            width_list=width_list,
            fnum=pt.next_fnum(),
            color_list=[pt.TRUE_BLUE, pt.FALSE_RED],
            kind='bar',
            alpha=0.7,
            stacked=True,
            edgecolor='none',
            xlabel='difference between the best correct score and the best incorrect score',
            ylabel='frequency (number of query annotations)',
            title='Score differences for %s' % (species_nice,),
            xmin=-10,
            xmax=30,
            use_legend=True,
        )

        # fnum = pt.next_fnum()
        # pt.plot_score_histograms(
        #     [score_diffs[succ], score_diffs[fail]],
        #     fnum=fnum,
        #     score_colors=[pt.TRUE_BLUE, pt.FALSE_RED],
        #     score_lbls=['correct', 'incorrect'],
        #     bin_width=.5,
        #     # histnorm='percent',
        # )

        # pt.draw_histogram(succ_edges, succ_hist, xlabel='1st - 2nd score',
        #                   autolabel=False, color='blue', title='Success Cases',
        #                   ymax=ymax, pnum=(1, 2, 1), fnum=fnum)
        # pt.draw_histogram(fail_edges, fail_hist, xlabel='1st - 2nd score',
        #                   autolabel=False, title='Failure Cases', ymax=ymax,
        #                   pnum=(1, 2, 2), fnum=fnum)

        from wbia.plottool.abstract_interaction import AbstractInteraction

        class SortedScoreSupportInteraction(AbstractInteraction):
            @staticmethod
            def static_plot(fnum, pnum):
                s = 20
                pt.plt.scatter(
                    top_scores[succ], score_diffs[succ], marker='x', color='b', s=s
                )
                pt.plt.scatter(
                    top_scores[fail], score_diffs[fail], marker='x', color='r', s=s
                )

                flags = ibs.filterflags_annot_tags(
                    aid_list, has_any=['photobomb', 'scenerymatch']
                )
                pt.plt.scatter(
                    top_scores[succ * flags],
                    score_diffs[succ * flags],
                    marker='o',
                    color='y',
                    s=s * 2,
                    facecolors='none',
                )
                pt.plt.scatter(
                    top_scores[fail * flags],
                    score_diffs[fail * flags],
                    marker='o',
                    color='y',
                    s=s * 2,
                    facecolors='none',
                )
                pt.set_xlabel('top score')
                pt.set_ylabel('score diff')

            def on_click_inside(self, event, ex):
                import vtool as vt

                # ax = event.inaxes
                # for l in ax.get_lines():
                #    print(l.get_label())
                pts = np.array([top_scores, score_diffs]).T
                idx, dist = vt.closest_point(np.array([event.xdata, event.ydata]), pts)
                print('idx = %r' % (idx,))
                print('dist = %r' % (dist,))
                aid = aid_list[idx]
                print('aid = %r' % (aid,))
                if event.button == 3:  # right-click
                    cm = cm_list[idx]
                    from wbia.gui import inspect_gui

                    qaid = aid
                    qreq_ = testres.cfgx2_qreq_[cfgx]
                    ibs = testres.ibs

                    if len(cm.daid_list) > 0:
                        daid = cm.get_top_aids()[0]

                        options = [
                            ('Interact Analysis', lambda: cm.ishow_analysis(qreq_))
                        ]

                        options += inspect_gui.get_aidpair_context_menu_options(
                            ibs, qaid, daid, cm, qreq_=qreq_
                        )
                    else:
                        print(' no matches for this one')
                    # update_callback=self.show_page,
                    # backend_callback=None, aid_list=aid_list)

                    # from wbia.viz.interact import interact_chip
                    # options = interact_chip.build_annot_context_options(
                    #    testres.ibs, aid, refresh_func=self.show_page, config2_=.extern_query_config2)
                    self.show_popup_menu(options, event)

        # x = SortedScoreSupportInteraction()
        # x.start()
        # pt.interactions.zoom_factory()


# @devcmd('scores', 'score', 'namescore_roc')
def draw_annot_scoresep(ibs, testres, f=None, verbose=None):
    """
    Draws the separation between true positive and true negative name scores.

    TODO:
        plot the difference between the top true score and the next best false score?

    CommandLine:
        ib
        python -m wbia draw_annot_scoresep --show
        python -m wbia draw_annot_scoresep --db PZ_MTEST --allgt -w --show --serial
        python -m wbia draw_annot_scoresep -t scores --db PZ_MTEST --allgt --show
        python -m wbia draw_annot_scoresep -t scores --db PZ_Master0 --allgt --show
        python -m wbia draw_annot_scoresep --db PZ_Master1 -a timectrl -t best --show
        python -m wbia draw_annot_scoresep --db PZ_Master1 -a timectrl -t best --show -f :without_tag=photobomb

    Paper:
        python -m wbia draw_annot_scoresep --dbdir lev/media/hdd/golden/GGR-IBEIS  -a timectrl --save gz_scoresep.png
        python -m wbia draw_annot_scoresep --dbdir lev/media/hdd/golden/GZGC -a timectrl:species=zebra_plains --save pz_scoresep.png
        python -m wbia draw_annot_scoresep --dbdir lev/media/hdd/golden/GZGC -a timectrl1h:species=giraffe_masai --save girm_scoresep.png


    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_drawing import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs, testres = main_helpers.testdata_expts(defaultdb, a=['timectrl'], t=['best'])
        >>> f = ut.get_argval(('--filt', '-f'), type_=list, default=[''])
        >>> draw_annot_scoresep(ibs, testres, f=f, verbose=ut.VERBOSE)
        >>> ut.show_if_requested()

    Ignore:
        import IPython
        IPython.get_ipython().magic('pylab qt4')
    """
    import wbia.plottool as pt
    import vtool as vt
    from wbia.expt import cfghelpers

    if ut.VERBOSE:
        print('[dev] draw_annot_scoresep')
    if f is None:
        f = ['']
    filt_cfg = ut.flatten(cfghelpers.parse_cfgstr_list2(f, strict=False))[0]
    print('filt_cfg = %r' % (filt_cfg,))

    # assert len(testres.cfgx2_qreq_) == 1, 'can only specify one config here'
    test_qaids = testres.get_test_qaids()

    # TODO: option to group configs with same pcfg and different acfg

    def load_annot_scores(testres, cfgx, filt_cfg):
        qaids = testres.cfgx2_qaids[cfgx]
        gt_rawscore = testres.get_infoprop_mat('qx2_gt_raw_score', qaids).T[cfgx]
        gf_rawscore = testres.get_infoprop_mat('qx2_gf_raw_score', qaids).T[cfgx]

        gt_daid = testres.get_infoprop_mat('qx2_gt_aid', qaids).T[cfgx]
        gf_daid = testres.get_infoprop_mat('qx2_gf_aid', qaids).T[cfgx]

        # FIXME: may need to specify which cfg is used in the future
        isvalid = testres.case_sample2(filt_cfg, qaids=qaids, return_mask=True).T[cfgx]

        isvalid[np.isnan(gf_rawscore)] = False
        isvalid[np.isnan(gt_rawscore)] = False

        tp_nscores = gt_rawscore[isvalid]
        tn_nscores = gf_rawscore[isvalid]

        # ---
        tn_qaids = tp_qaids = test_qaids[isvalid]
        tn_daids = gf_daid[isvalid]
        tp_daids = gt_daid[isvalid]

        part_attrs = {
            1: {'qaid': tp_qaids, 'daid': tn_daids},
            0: {'qaid': tn_qaids, 'daid': tp_daids},
        }
        return tp_nscores, tn_nscores, part_attrs

    join_acfgs = True
    if join_acfgs:
        groupxs = testres.get_cfgx_groupxs()
    else:
        groupxs = list(zip(range(len(testres.cfgx2_qreq_))))
    grouped_qreqs = ut.apply_grouping(testres.cfgx2_qreq_, groupxs)
    cfgx2_shortlbl = testres.get_short_cfglbls(join_acfgs=join_acfgs)

    grouped_scores = []
    for cfgxs in groupxs:
        # testres.print_pcfg_info()
        score_group = []
        for cfgx in cfgxs:
            print('Loading cached chipmatches')
            tp_scores, tn_scores, part_attrs = load_annot_scores(testres, cfgx, filt_cfg)
            score_group.append((tp_scores, tn_scores, part_attrs))
        grouped_scores.append(score_group)

    def attr_callback(qaid):
        print('callback qaid = %r' % (qaid,))
        testres.interact_individual_result(qaid)
        reconstruct_str = (
            'python -m wbia.dev -e cases '
            + testres.reconstruct_test_flags()
            + ' --qaid '
            + str(qaid)
            + ' --show'
        )
        print('Independent reconstruct')
        print(reconstruct_str)

    fpr = ut.get_argval('--fpr', type_=float, default=None)
    tpr = ut.get_argval('--tpr', type_=float, default=None if fpr is not None else 0.85)

    for score_group, lbl in zip(grouped_scores, cfgx2_shortlbl):
        tp_nscores = np.hstack(ut.take_column(score_group, 0))
        tn_nscores = np.hstack(ut.take_column(score_group, 1))
        combine_attrs = ut.partial(
            ut.dict_union_combine,
            combine_op=ut.partial(ut.dict_union_combine, combine_op=np.append),
        )
        part_attrs = reduce(combine_attrs, ut.take_column(score_group, 2))
        # encoder = vt.ScoreNormalizer(adjust=8, tpr=.85)
        encoder = vt.ScoreNormalizer(
            # adjust=8,
            adjust=1.5,
            # fpr=fpr, tpr=tpr,
            monotonize=True,
            verbose=verbose,
        )
        tp_scores = tp_nscores
        tn_scores = tn_nscores
        name_scores, labels, attrs = encoder._to_xy(tp_nscores, tn_nscores, part_attrs)

        encoder.fit(name_scores, labels, attrs, verbose=verbose)
        # encoder.visualize(figtitle='Learned Name Score Normalizer\n' + qreq_.get_cfgstr())
        # encoder.visualize(figtitle='Learned Name Score Normalizer\n' + qreq_.get_cfgstr(), fnum=cfgx)
        # pt.set_figsize(w=30, h=10, dpi=256)

        plotname = ''
        figtitle = testres.make_figtitle(plotname, filt_cfg=filt_cfg)

        encoder.visualize(
            figtitle=figtitle,
            #
            with_scores=False,
            with_prebayes=False,
            with_postbayes=False,
            #
            histnorm='percent',
            histoverlay=False,
            with_hist=True,
            with_roc=False,
            attr_callback=attr_callback,
            # bin_width=.125,
            # bin_width=.05,
            # logscale=dict(linthreshx=1, linthreshy=1, basex=2, basey=0),
            # score_range=(0, 14),
            # score_range=(0, 10),
            score_range=(0, 6),
            # bin_width=.5,
            bin_width=0.125,
            score_lbls=('incorrect', 'correct'),
            verbose=verbose,
        )

        icon = ibs.get_database_icon()
        if False and icon is not None:
            pt.overlay_icon(
                icon,
                coords=(1, 0),
                bbox_alignment=(1, 0),
                as_artist=1,
                max_asize=(1000, 2000),
            )

        if ut.get_argflag('--contextadjust'):
            pt.adjust_subplots(left=0.1, bottom=0.25, wspace=0.2, hspace=0.2)
            pt.adjust_subplots(use_argv=True)
        # pt.set_figsize(w=30, h=10, dpi=256)
        pt.set_figtitle(ibs.get_dbname() + ' ' + lbl)

    locals_ = locals()
    return locals_


def draw_casetag_hist(
    ibs, testres, f=None, with_wordcloud=not ut.get_argflag('--no-wordcloud')
):
    r"""
    Args:
        ibs (wbia.IBEISController):  wbia controller object
        testres (TestResult):  test result object

    CommandLine:
        wbia --tf -draw_casetag_hist --show

        # Experiments I tagged
        wbia --tf -draw_casetag_hist -a timectrl -t invarbest --db PZ_Master1  --show

        wbia -e taghist -a timectrl -t best --db PZ_Master1  --show

        wbia -e taghist -a timequalctrl -t invarbest --db PZ_Master1  --show
        wbia -e taghist -a timequalctrl:minqual=good -t invarbest --db PZ_Master1  --show
        wbia -e taghist -a timequalctrl:minqual=good -t invarbest --db PZ_Master1  --show --filt :fail=True

        # Do more tagging
        wbia -e cases -a timequalctrl:minqual=good -t invarbest --db PZ_Master1 \
            --filt :orderby=gfscore,reverse=1,min_gtrank=1,max_gf_tags=0 --show
        wbia -e print -a timequalctrl:minqual=good -t invarbest --db PZ_Master1 --show
        wbia -e cases -a timequalctrl -t invarbest --db PZ_Master1 \
            --filt :orderby=gfscore,reverse=1,max_gf_tags=0,:fail=True,min_gf_timedelta=12h --show

        wbia -e cases -a timequalctrl -t invarbest --db PZ_Master1 \
            --filt :orderby=gfscore,reverse=1,max_gf_tags=0,:fail=True,min_gf_timedelta=12h --show
        python -m wbia -e taghist --db PZ_Master1 -a timectrl -t best \
            --filt :fail=True --no-wordcloud --hargv=tags  --prefix "Failure Case " --label PZTags  --figsize=10,3  --left=.2


    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_drawing import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts('PZ_Master1', a=['timequalcontrolled'])
        >>> f = ut.get_argval(('--filt', '-f'), type_=list, default=[''])
        >>> draw_casetag_hist(ibs, testres, f=f)
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt
    from wbia import tag_funcs
    from wbia.expt import cfghelpers

    # All unfiltered tags
    all_tags = testres.get_all_tags()
    if True:
        # Remove gf tags below a thresh and gt tags above a thresh
        gt_tags = testres.get_gt_tags()
        gf_tags = testres.get_gf_tags()
        truth2_prop, prop2_mat = testres.get_truth2_prop()
        score_thresh = testres.find_score_thresh_cutoff()
        print('score_thresh = %r' % (score_thresh,))
        # TODO: I want the point that the prob true is greater than prob false
        gt_is_problem = truth2_prop['gt']['score'] < score_thresh
        gf_is_problem = truth2_prop['gf']['score'] >= score_thresh
        other_is_problem = ~np.logical_or(gt_is_problem, gf_is_problem)

        def zipmask(_tags, _flags):
            return [
                [item if flag else [] for item, flag in zip(list_, flags)]
                for list_, flags in zip(_tags, _flags)
            ]

        def combinetags(tags1, tags2):
            import utool as ut

            return [ut.list_zipflatten(t1, t2) for t1, t2 in zip(tags1, tags2)]

        gt_problem_tags = zipmask(gt_tags, gt_is_problem)
        gf_problem_tags = zipmask(gf_tags, gf_is_problem)
        other_problem_tags = zipmask(all_tags, other_is_problem)
        all_tags = reduce(
            combinetags, [gt_problem_tags, gf_problem_tags, other_problem_tags]
        )
    if not ut.get_argflag('--fulltag'):
        all_tags = [tag_funcs.consolodate_annotmatch_tags(tags) for tags in all_tags]
    # Get tags that match the filter
    if f is None:
        f = ['']
    filt_cfg = ut.flatten(cfghelpers.parse_cfgstr_list2(f, strict=False))[0]
    case_pos_list = testres.case_sample2(filt_cfg)
    case_qx_list = ut.unique_ordered(case_pos_list.T[0])
    selected_tags = ut.take(all_tags, case_qx_list)
    flat_tags_list = list(map(ut.flatten, selected_tags))
    WITH_NOTAGS = False
    if WITH_NOTAGS:
        flat_tags_list_ = [
            tags if len(tags) > 0 else ['NoTag'] for tags in flat_tags_list
        ]
    else:
        flat_tags_list_ = flat_tags_list
    WITH_TOTAL = False
    if WITH_TOTAL:
        total = [['Total']] * len(case_qx_list)
        flat_tags_list_ += total
    WITH_WEIGHTS = True
    if WITH_WEIGHTS:
        flat_weights_list = [
            [] if len(tags) == 0 else [1.0 / len(tags)] * len(tags)
            for tags in flat_tags_list_
        ]
    flat_tags = list(map(str, ut.flatten(flat_tags_list_)))
    if WITH_WEIGHTS:
        weight_list = ut.flatten(flat_weights_list)
    else:
        weight_list = None
    fnum = None
    pnum_ = pt.make_pnum_nextgen(nRows=1, nCols=with_wordcloud + 1)
    fnum = pt.ensure_fnum(fnum)
    pt.word_histogram2(
        flat_tags, weight_list=weight_list, fnum=fnum, pnum=pnum_(), xlabel='Case'
    )
    icon = ibs.get_database_icon()
    if icon is not None:
        pt.overlay_icon(icon, coords=(1, 1), bbox_alignment=(1, 1))
    if with_wordcloud:
        pt.wordcloud(' '.join(flat_tags), fnum=fnum, pnum=pnum_())
    # figtitle = testres.make_figtitle('Tag Histogram', filt_cfg=filt_cfg)
    figtitle = testres.make_figtitle('Case Histogram', filt_cfg=filt_cfg)
    figtitle += ' #cases=%r' % (len(case_qx_list))
    pt.set_figtitle(figtitle)
    if ut.get_argflag('--contextadjust'):
        # pt.adjust_subplots(left=.1, bottom=.25, wspace=.2, hspace=.2)
        # pt.adjust_subplots(wspace=.01)
        pt.adjust_subplots(use_argv=True, wspace=0.01, bottom=0.3)


def draw_rank_surface(ibs, testres, verbose=None, fnum=None):
    r"""
    Draws n dimensional data + a score / rank
    The rank is always on the y axis.

    The first dimension is on the x axis.
    The second dimension is split over multiple plots.
    The third dimension becomes multiple lines.
    May need to clean this scheme up a bit.

    Args:
        ibs (wbia.IBEISController):  wbia controller object
        testres (TestResult):  test result object

    CommandLine:
        wbia --tf draw_rank_surface --db PZ_Master1 -a varysize_td -t CircQRH_K --show

        wbia --tf draw_rank_surface --show -t best -a varysize --db PZ_Master1 --show

        wbia --tf draw_rank_surface --show -t CircQRH_K -a varysize_td --db PZ_Master1 --show
        wbia --tf draw_rank_surface --show -t CircQRH_K -a varysize_td --db PZ_Master1 --show

        wbia --tf draw_rank_surface --show  -t candidacy_k -a varysize  --db PZ_Master1 --show --param-keys=K,dcfg_sample_per_name,dcfg_sample_size
        wbia --tf draw_rank_surface --show  -t best \
            -a varynannots_td varynannots_td:qmin_pername=3,dpername=2  \
            --db PZ_Master1 --show --param-keys=dcfg_sample_per_name,dcfg_sample_size
        wbia --tf draw_rank_surface --show  -t best -a varynannots_td  --db PZ_Master1 --show --param-keys=dcfg_sample_size

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_drawing import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST')
        >>> result = draw_rank_surface(ibs, testres)
        >>> ut.show_if_requested()
        >>> print(result)
    """
    import wbia.plottool as pt
    from wbia.expt import annotation_configs

    if verbose is None:
        verbose = ut.VERBOSE
    # percent_le1_list = 100 * rank_le1_list / len(testres.qaids)
    cfgx2_cumhist_percent, edges = testres.get_rank_percentage_cumhist(bins='dense')
    percent_le1_list = cfgx2_cumhist_percent.T[0]
    # testres.cfgx2_lbl
    # testres.get_param_basis('dcfg_sample_per_name')
    # testres.get_param_basis('dcfg_sample_size')
    # K_basis = testres.get_param_basis('K')
    # K_cfgx_lists = [testres.get_cfgx_with_param('K', K) for K in K_basis]
    # param_key_list = testres.get_all_varied_params()

    # Extract the requested keys
    default_param_key_list = ['K', 'dcfg_sample_per_name', 'dcfg_sample_size']
    param_key_list = ut.get_argval(
        '--param-keys', type_=list, default=default_param_key_list
    )

    # param_key_list = ['K', 'dcfg_sample_per_name', 'len(daids)']
    basis_dict = {}
    cfgx_lists_dict = {}
    for key in param_key_list:
        _basis = testres.get_param_basis(key)
        _cfgx_list = [testres.get_cfgx_with_param(key, val) for val in _basis]
        # Grid of config indexes using param key as reference
        cfgx_lists_dict[key] = _cfgx_list
        basis_dict[key] = _basis

    if verbose:
        print('basis_dict = ' + ut.repr2(basis_dict, nl=1, hack_liststr=True))
        print(
            'e.g. cfgx_lists_dict[1] contains indicies of configs where K = basis_dict["K"][1]'
        )
        print('cfx_lists_dict = ' + ut.repr2(cfgx_lists_dict, nl=2, hack_liststr=True))

    # const_key = 'K'

    if len(param_key_list) == 1:
        const_key = None
        const_basis = [None]
        basis_dict[None] = [0]
        # const_basis_cfgx_lists
        # Create a single empty dimension for a single pnum
        # correct, but not conceptually right
        cfgx_lists_dict[None] = ut.list_transpose(cfgx_lists_dict[param_key_list[0]])
    elif len(param_key_list) > 1:
        # Hold a key constant if more than 1 subplot
        const_key = param_key_list[1]
        # const_key = 'dcfg_sample_per_name'
        const_basis = basis_dict[const_key]
        # pnum_ = pt.make_pnum_nextgen(*pt.get_square_row_cols(len(basis_dict[const_key]), max_cols=1))
        # ymax = percent_le1_list.max()
        # ymin = percent_le1_list.min()
    else:
        assert False

    const_basis_cfgx_lists = cfgx_lists_dict[const_key]

    if len(param_key_list) > 2:
        # Use consistent markers and colors when varying a lot of params
        # num_other_params = len(basis_dict[param_key_list[-1]])
        # num_other_params = len(basis_dict[const_key])
        num_other_params = len(basis_dict[ut.setdiff(param_key_list, const_key)[-1]])
        color_list = pt.distinct_colors(num_other_params)
        marker_list = pt.distinct_markers(num_other_params)
    else:
        color_list = pt.distinct_colors(1)
        marker_list = pt.distinct_markers(1)

    fnum = pt.ensure_fnum(fnum)

    nd_labels_full = [key for key in param_key_list if key != const_key]

    # setup args for all plots
    pnum_ = pt.make_pnum_nextgen(*pt.get_square_row_cols(len(basis_dict[const_key])))
    for const_idx, const_val in enumerate(const_basis):
        pnum = pnum_()
        if verbose:
            print('---- NEXT PNUM=%r --- ' % (pnum,))
            print('const_key = %r' % (const_key,))
            print('const_val = %r' % (const_val,))
            print('const_idx = %r' % (const_idx,))
        const_basis_cfgx_list = const_basis_cfgx_lists[const_idx]
        rank_list = ut.take(percent_le1_list, const_basis_cfgx_list)
        # Figure out what the values are for other dimensions
        agree_param_vals = dict(
            [
                (
                    key,
                    [
                        testres.get_param_val_from_cfgx(cfgx, key)
                        for cfgx in const_basis_cfgx_list
                    ],
                )
                for key in nd_labels_full
            ]
        )

        # Make a list of points that need plotting
        known_nd_data = np.array(list(agree_param_vals.values())).T
        known_target_points = np.array(rank_list)

        nd_labels_ = nd_labels_full[:]

        if len(nd_labels_) == 1:
            # hack for nonvaried params
            empty_dim = np.zeros((len(known_nd_data), 1))
            known_nd_data = np.hstack([known_nd_data, empty_dim])
            nd_labels_ += [None]

        # short ndlabels
        nd_labels = [
            annotation_configs.shorten_to_alias_labels(key) for key in nd_labels_
        ]
        target_label = annotation_configs.shorten_to_alias_labels(key)

        target_label = 'accuracy (%)'

        # hack
        ymin = 30 if known_target_points.min() > 30 and False else 0
        num_yticks = 8 if ymin == 30 else 11

        if const_key is None:
            title = 'accuracy'
        else:
            title = (
                'accuracy when '
                + annotation_configs.shorten_to_alias_labels(const_key)
                + '=%r' % (const_val,)
            )
        if verbose:
            print('title = %r' % (title,))
            # print('nd_labels = %r' % (nd_labels,))
            print('target_label = %r' % (target_label,))
            print('known_nd_data = %r' % (known_nd_data,))
            # print('known_target_points = %r' % (known_target_points,))

        # PLOT3D = not ut.get_argflag('--no3dsurf')
        # PLOT3D = ut.get_argflag('--no2dsurf')
        PLOT3D = ut.get_argflag('--3dsurf')
        if PLOT3D:
            pt.plot_search_surface(
                known_nd_data,
                known_target_points,
                nd_labels,
                target_label,
                title=title,
                fnum=fnum,
                pnum=pnum,
            )
        else:
            # Convert known nd data into a multiplot-ish format
            nonconst_basis_vals = np.unique(known_nd_data.T[1])
            # Find which colors will not be used
            nonconst_key = nd_labels_[1]
            nonconst_basis = np.array(basis_dict[nonconst_key])
            nonconst_covers_basis = np.in1d(nonconst_basis, nonconst_basis_vals)
            # I dont remember what was trying to happen here
            nonconst_color_list = ut.compress(color_list, nonconst_covers_basis)
            nonconst_marker_list = ut.compress(marker_list, nonconst_covers_basis)

            pt.plot_multiple_scores(
                known_nd_data,
                known_target_points,
                nd_labels,
                target_label,
                title=title,
                color_list=nonconst_color_list,
                marker_list=nonconst_marker_list,
                fnum=fnum,
                pnum=pnum,
                num_yticks=num_yticks,
                ymin=ymin,
                ymax=100,
                ypad=0.5,
                xpad=0.05,
                legend_loc='lower right',
                # **FONTKW
            )

    fig = pt.gcf()
    ax = fig.axes[0]
    pt.plt.sca(ax)
    # pt.figure(fnum=fnum, pnum=pnum0)
    icon = ibs.get_database_icon()
    if icon is not None:
        # pt.overlay_icon(icon, coords=(0, 0), bbox_alignment=(0, 0))
        pt.overlay_icon(icon, coords=(0.001, 0.001), bbox_alignment=(0, 0))

    nd_labels = [
        annotation_configs.shorten_to_alias_labels(key) for key in nd_labels_full
    ]
    plotname = 'Effect of ' + ut.conj_phrase(nd_labels, 'and') + ' on accuracy'
    figtitle = testres.make_figtitle(plotname)
    # hack
    if 1 or verbose:
        testres.print_unique_annot_config_stats()
    # pt.set_figtitle(figtitle, size=14)
    pt.set_figtitle(figtitle)
    # HACK FOR FIGSIZE
    fig = pt.gcf()
    fig.set_size_inches(14, 4)
    pt.adjust_subplots(
        left=0.05, bottom=0.08, top=0.80, right=0.95, wspace=0.2, hspace=0.3
    )
    if ut.get_argflag('--contextadjust'):
        pt.adjust_subplots(left=0.1, bottom=0.25, wspace=0.2, hspace=0.2)
        pt.adjust_subplots(use_argv=True)


def temp_num_exmaples_cmc():
    import wbia

    ibs1, testres1 = wbia.testdata_expts(
        dbdir=ut.truepath('~/lev/media/hdd/golden/GGR-IBEIS'),
        t='Ell:K=1',
        # t='best:prescore_method=nsum',
        # a='timectrl'
        a='timectrl:species=zebra_grevys,qmin_pername=3,dsample_per_name=[1,2],dsize=1939',
    )

    ibs2, testres2 = wbia.testdata_expts(
        dbdir=ut.truepath('~/lev/media/hdd/golden/GZGC'),
        # t='best:prescore_method=nsum',
        t='CircQRH:K=3',
        a='timectrl:species=zebra_plains,qmin_pername=3,dsample_per_name=[1,2],dsize=1187'
        # a='timectrl:species=zebra_plains,sample_per_name=3'
    )

    key = 'qx2_gt_rank'
    ydata_list = []
    label_list = []
    num_ranks = 5

    testres_list = [testres1, testres2]
    ibs_list = [ibs1, ibs2]

    # ibs = ibs1
    # testres = testres1
    for ibs, testres in zip(ibs_list, testres_list):
        cfgx2_cmc, edges = testres.get_rank_percentage_cumhist(
            bins='dense', key=key, join_acfgs=True
        )

        species = ibs.get_dominant_species(testres.qaids)
        species_nice = ibs.get_species_nice(
            ibs.get_species_rowids_from_text(ibs.get_dominant_species(testres.qaids))
        )
        species_nice = {
            'zebra_grevys': "Grevy's Zebras",
            'zebra_plains': 'Plains Zebras',
            'giraffe_masai': 'Masai Giraffes',
        }.get(species, species_nice)

        cmcs = cfgx2_cmc.T[:num_ranks].T
        lbls = testres.get_varied_labels(shorten=True, join_acfgs=True)
        # lbls = ['%6.2f%% @ rank #1' % (cmc[0],) + ' - ' + lbl + ' - ' + species_nice
        #         for lbl, cmc in zip(lbls, cmcs)]
        lbls = [
            species_nice + ' - ' + lbl.replace('dpername', 'exemplars')
            for lbl, cmc in zip(lbls, cmcs)
        ]

        ydata_list.extend(cmcs)
        label_list.extend(lbls)

    sortx = np.argsort(ut.take_column(ydata_list, 0))[::-1]
    ydata_list = ut.take(ydata_list, sortx)
    label_list = ut.take(label_list, sortx)

    xdata = list(range(1, num_ranks + 1))

    import wbia.plottool as pt

    ymin = 30
    xpad = 0.9  # if kind == 'plot' else .5
    num_yticks = 8 if ymin == 30 else 11
    # ymin = 30 if cfgx2_cumhist_percent.min() > 30 and False else 0

    fig = pt.multi_plot(
        xdata,
        ydata_list,
        marker_list=pt.distinct_markers(len(ydata_list)),
        label_list=label_list,
        num_xticks=num_ranks,
        xlabel='rank',
        ylabel='recognition rate (%)',
        legend_loc='lower right',
        num_yticks=num_yticks,
        ymax=100,
        ymin=ymin,
        ypad=0.5,
        xmin=xpad,
        xmax=num_ranks + 1 - xpad,
        use_legend=True,
        title='Cumulative match characteristics',
        legendsize=10,
        titlesize=12,
        labelsize=12,
    )
    fig.set_size_inches(*((1.0 * np.array([4, 3])).tolist()))
    pt.adjust_subplots(bottom=0.15, top=0.9, left=0.2)
    fpath = 'cmc-dpername-combined.png'
    fig.savefig(fpath, transparent=True, edgecolor='none')
    import vtool as vt

    vt.clipwhite_ondisk(fpath, fpath, verbose=True)
    ut.startfile(fpath)


def temp_multidb_cmc():
    """
    Plots multiple database CMC curves in the same plot for the AI for social
    good paper
    """
    import wbia

    ibs1, testres1 = wbia.testdata_expts(
        dbdir=ut.truepath('~/lev/media/hdd/golden/GGR-IBEIS'),
        t='Ell:K=4',
        # t='best:prescore_method=nsum',
        a='timectrl'
        # a='timectrl:species=zebra_grevys,qmin_pername=3,dsample_per_name=2'
    )
    ibs2, testres2 = wbia.testdata_expts(
        dbdir=ut.truepath('~/lev/media/hdd/golden/GZGC'),
        # t='best:prescore_method=nsum',
        t='CircQRH:K=3',
        a='timectrl:species=zebra_plains'
        # a='timectrl:species=zebra_plains,qmin_pername=3,dsample_per_name=2'
        # a='timectrl:species=zebra_plains,sample_per_name=3'
    )
    ibs3, testres3 = wbia.testdata_expts(
        dbdir=ut.truepath('~/lev/media/hdd/golden/GZGC'),
        t='Ell:K=2',
        a='timectrl1h:species=giraffe_masai'
        # a='timectrl1h:species=giraffe_masai,qmin_pername=2,dsample_per_name=2'
    )

    testres_list = [testres1, testres2, testres3]
    ibs_list = [ibs1, ibs2, ibs3]

    key = 'qx2_gt_rank'
    ydata_list = []
    label_list = []
    num_ranks = 5

    for ibs, testres in zip(ibs_list, testres_list):
        cfgx2_cmc, edges = testres.get_rank_percentage_cumhist(
            bins='dense', key=key, join_acfgs=True
        )
        cmc = cfgx2_cmc[0][:num_ranks]

        species = ibs.get_dominant_species(testres.qaids)
        species_nice = ibs.get_species_nice(
            ibs.get_species_rowids_from_text(ibs.get_dominant_species(testres.qaids))
        )
        species_nice = {
            'zebra_grevys': "Grevy's Zebras",
            'zebra_plains': 'Plains Zebras',
            'giraffe_masai': 'Masai Giraffes',
        }.get(species, species_nice)
        label = ('%6.2f%% @ rank #1' % (cmc[0],)) + ' - ' + species_nice
        label_list.append(label)
        ydata_list.append(cmc)

    sortx = np.argsort(ut.take_column(ydata_list, 0))[::-1]
    ydata_list = ut.take(ydata_list, sortx)
    label_list = ut.take(label_list, sortx)

    xdata = list(range(1, num_ranks + 1))

    import wbia.plottool as pt

    ymin = 30
    xpad = 0.9  # if kind == 'plot' else .5
    num_yticks = 8 if ymin == 30 else 11
    # ymin = 30 if cfgx2_cumhist_percent.min() > 30 and False else 0

    fig = pt.multi_plot(
        xdata,
        ydata_list,
        marker_list=pt.distinct_markers(len(ydata_list)),
        label_list=label_list,
        num_xticks=num_ranks,
        xlabel='rank',
        ylabel='recognition rate (%)',
        legend_loc='lower right',
        num_yticks=num_yticks,
        ymax=100,
        ymin=ymin,
        ypad=0.5,
        xmin=xpad,
        xmax=num_ranks + 1 - xpad,
        use_legend=True,
        title='Cumulative match characteristics',
    )
    fig.set_size_inches(*((1.5 * np.array([4, 3])).tolist()))
    pt.adjust_subplots(bottom=0.15)
    fig.savefig('cmc-combined.png', transparent=True, edgecolor='none')
    import vtool as vt

    vt.clipwhite_ondisk('cmc-combined.png', 'cmc-combined.png', verbose=True)
    ut.startfile('cmc-combined.png')


def draw_rank_cmc(
    ibs,
    testres,
    verbose=False,
    test_cfgx_slice=None,
    group_queries=False,
    draw_icon=True,
    numranks=5,
    kind='cmc',
    cdfzoom=True,
    **kwargs
):
    # numranks=3, kind='bar', cdfzoom=False):
    r"""
    Args:
        ibs (wbia.IBEISController):  wbia controller object
        testres (TestResult):

    TODO:
        # Cross-validated results with timectrl
        python -m wbia draw_rank_cmc --db PZ_MTEST --show -a timectrl:xval=True -t invar --kind=cmc

    CommandLine:
        python -m wbia draw_rank_cmc
        python -m wbia draw_rank_cmc --db PZ_MTEST --show -a timectrl -t default --kind=cmc

        python -m wbia draw_rank_cmc --db PZ_MTEST --show -a :proot=smk,num_words=64000
        python -m wbia draw_rank_cmc --db PZ_MTEST --show -a ctrl -t best:prescore_method=csum
        python -m wbia draw_rank_cmc --db PZ_MTEST --show -a timectrl -t invar --kind=cmc --cdfzoom
        python -m wbia draw_rank_cmc --db PZ_MTEST --show -a varypername_td   -t CircQRH_ScoreMech:K=3
        #wbia -e rank_cmc --db lynx -a default:qsame_imageset=True,been_adjusted=True,excluderef=True -t default:K=1 --show

        python -m wbia.dev -e draw_rank_cmc --db lynx -a default:qsame_imageset=True,been_adjusted=True,excluderef=True -t default:K=1 --show

        python -m wbia --tf draw_rank_cmc -t best -a timectrl --db PZ_Master1 --show

        python -m wbia --tf draw_rank_cmc --db PZ_Master1 --show -t best \
            -a timectrl:qhas_any=\(needswork,correctable,mildviewpoint\),qhas_none=\(viewpoint,photobomb,error:viewpoint,quality\) \
            --acfginfo --veryverbtd

        wbia --tf draw_match_cases --db GZ_ALL -a ctrl \
            -t default:K=1,resize_dim=[width],dim_size=[700,750] \
            -f :sortdsc=gfscore,without_tag=scenerymatch,disagree=True \
            --show

        wbia --tf autogen_ipynb --db GZ_ALL --ipynb -a ctrl \
            -t default:K=1,resize_dim=[width],dim_size=[600,700,750] \
             default:K=1,resize_dim=[area],dim_size=[450,550,600,650]

        wbia draw_rank_cmc --db GZ_ALL -a ctrl -t default --show
        wbia draw_match_cases --db GZ_ALL -a ctrl -t default -f :fail=True --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_drawing import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> #ibs, testres = main_helpers.testdata_expts(
        >>> #    'seaturtles', a='default2:qhas_any=(left),sample_occur=True,occur_offset=[0,1,2,3,4,5,6,7,8],num_names=None')
        >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST')
        >>> kwargs = ut.argparse_funckw(draw_rank_cmc)
        >>> result = draw_rank_cmc(ibs, testres, **kwargs)
        >>> ut.show_if_requested()
        >>> print(result)
    """
    import wbia.plottool as pt

    if group_queries:
        key = 'qnx2_gt_name_rank'
        target_label = 'accuracy (% per name)'
    else:
        key = 'qx2_gt_rank'
        target_label = 'accuracy (% per annotation)'

    join_acfgs = True
    cfgx2_cumhist_percent, edges = testres.get_rank_percentage_cumhist(
        bins='dense', key=key, join_acfgs=join_acfgs
    )

    # Do this twice, but no sep the first time
    cfglbl_list = testres.get_varied_labels(shorten=True, join_acfgs=join_acfgs)
    cfglbl_to_score = ut.dzip(cfglbl_list, cfgx2_cumhist_percent.T[0])
    cfglbl_to_score = ut.sort_dict(cfglbl_to_score, part='vals')
    print('accuracy @rank1: ' + ut.repr4(cfglbl_to_score, strkeys=True, precision=4))

    cfglbl_list = testres.get_varied_labels(
        shorten=True, join_acfgs=join_acfgs, sep=kwargs.get('sep', '')
    )
    label_list = [
        ('%6.2f%%' % (percent,)) + ' - ' + label
        for label, percent in zip(cfglbl_list, cfgx2_cumhist_percent.T[0])
    ]

    cmap_seed = ut.get_argval('--prefix', type_=str, default=None)
    color_list = pt.distinct_colors(len(label_list), cmap_seed=cmap_seed)

    marker_list = pt.distinct_markers(len(label_list))
    test_cfgx_slice = ut.get_argval(
        '--test_cfgx_slice', type_='fuzzy_subset', default=test_cfgx_slice
    )
    if test_cfgx_slice is not None:
        print('test_cfgx_slice = %r' % (test_cfgx_slice,))
        cfgx2_cumhist_percent = np.array(ut.take(cfgx2_cumhist_percent, test_cfgx_slice))
        label_list = ut.take(label_list, test_cfgx_slice)
        color_list = ut.take(color_list, test_cfgx_slice)
        marker_list = ut.take(marker_list, test_cfgx_slice)

    # Order cdf list by rank0
    # sortx = cfgx2_cumhist_percent.T[0].argsort()[::-1]
    sortx = vt.argsort_records(cfgx2_cumhist_percent.T)[::-1]
    label_list = ut.take(label_list, sortx)
    cfgx2_cumhist_percent = np.array(ut.take(cfgx2_cumhist_percent, sortx))
    color_list = ut.take(color_list, sortx)
    marker_list = ut.take(marker_list, sortx)
    #

    if verbose:
        testres.print_unique_annot_config_stats(ibs)

    numranks = ut.get_argval('--numranks', type_=int, default=numranks)

    if numranks is None:
        numranks = len(cfgx2_cumhist_percent.T)

    maxpos = min(len(cfgx2_cumhist_percent.T), numranks)
    cfgx2_cumhist_short = cfgx2_cumhist_percent[:, 0:maxpos]
    edges_short = edges[0 : min(len(edges), numranks + 1)]

    # twoplots = not ut.get_argflag('--cdfzoom')
    twoplots = not cdfzoom
    pnum_ = pt.make_pnum_nextgen(nRows=twoplots + 1, nCols=1)

    fnum = pt.ensure_fnum(None)
    # target_label = '% groundtrue matches ≤ rank'

    ymin = 30 if cfgx2_cumhist_percent.min() > 30 and False else 0
    num_yticks = 8 if ymin == 30 else 11

    kind = ut.get_argval('--kind', default=kind)
    if kind is None:
        kind = 'bar'
    elif kind == 'cmc':
        kind = 'plot'

    plotname = 'Cumulative Match Characteristic (CMC)'
    plotname = ut.get_argval('--plotname', default=plotname)
    figtitle = testres.make_figtitle(plotname)

    xpad = 0.9 if kind == 'plot' else 0.5

    cumhistkw = dict(
        xlabel='rank',
        ylabel=target_label,
        color_list=color_list,
        marker_list=marker_list,
        fnum=fnum,
        # legend_loc='lower right',
        legend_loc='lower right',
        num_yticks=num_yticks,
        ymax=100,
        ymin=ymin,
        ypad=0.5,
        xmin=xpad,
        kind=kind,
        title=figtitle,
        # figtitle=figtitle,
        # xpad=.05,
        # **FONTKW
    )
    cumhistkw.update(kwargs)

    # if not twoplots:
    minval = numranks if numranks <= 10 else 10

    pt.plot_rank_cumhist(
        cfgx2_cumhist_short,
        edges=edges_short,
        label_list=label_list,
        num_xticks=max(5, min(minval, (numranks // 20) + 2)),
        # legend_alpha=.85,
        legend_alpha=0.92,
        # legendsize=12,
        xmax=numranks + 1 - xpad,
        use_legend=not twoplots,
        pnum=pnum_(),
        **cumhistkw
    )

    if twoplots:
        del cumhistkw['title']
        numranks2 = len(cfgx2_cumhist_percent.T)
        ax1 = pt.gca()
        pt.plot_rank_cumhist(
            cfgx2_cumhist_percent,
            edges=edges,
            label_list=label_list,
            num_xticks=numranks2,
            use_legend=twoplots,
            pnum=pnum_(),
            xmax=numranks2 + 1 - xpad,
            **cumhistkw
        )
        ax2 = pt.gca()
        # pt.zoom_effect01(ax1, ax2, 1, numranks2, fc='w')
        # pt.zoom_effect01(ax1, ax2, 1, numranks, fc='w')
        pt.zoom_effect01(ax1, ax2, 1, numranks, ec='k', fc='w')

    icon = ibs.get_database_icon()
    # print('draw_icon = %r' % (draw_icon,))
    if draw_icon and icon is not None:
        # ax = pt.gca()
        # ax.get_xlim()
        pt.overlay_icon(icon, bbox_alignment=(0, 0), as_artist=True)
        # pt.overlay_icon(icon, bbox_alignment=(0, 0), as_artist=False, max_asize=(.5, .5))
        # pt.overlay_icon(icon, bbox_alignment=(0, 1), as_artist=False)
        # max_asize=(2, 4))
        pass
        # ax.get_ylim()

    # ax = pt.gca()
    # ax.grid(True)
    # fig = pt.gcf()
    # import utool as ut
    # HACK FOR FIGSIZE
    # fig.set_size_inches(15, 7)
    if ut.get_argflag('--contextadjust') or True:
        pt.adjust_subplots(left=0.05, bottom=0.08, wspace=0.0, hspace=0.15)
        pt.adjust_subplots(use_argv=True)


@profile
def draw_case_timedeltas(ibs, testres, falsepos=None, truepos=None, verbose=False):
    r"""

    CommandLine:
        python -m wbia.dev -e draw_case_timedeltas --show
        python -m wbia.dev -e draw_case_timedeltas --show -t default \
            -a unctrl:num_names=1,name_offset=[1,2]
        python -m wbia.dev -e draw_case_timedeltas --show -t default \
            -a unctrl:num_names=1,name_offset=[1,2],joinme=1
        python -m wbia.dev -e draw_case_timedeltas --show -t default \
            -a unctrl:num_names=1,name_offset=[1,2] \
               unctrl:num_names=1,name_offset=[3,0]


        python -m wbia.dev -e timedelta_hist --show -t baseline \
            -a unctrl ctrl:force_const_size=True unctrl:force_const_size=True \
            --consistent --db PZ_MTEST

        # Testing
        python -m wbia.dev -e timedelta_hist --show -t baseline \
            -a unctrl ctrl:force_const_size=True unctrl:force_const_size=True \
            --consistent --db PZ_Master1
        python -m wbia.dev -e timedelta_hist --show -t baseline \
            -a unctrl ctrl:sample_rule_ref=max_timedelta --db PZ_Master1 \
            --aidcfginfo

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_drawing import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST')
        >>> draw_case_timedeltas(ibs, testres)
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt
    import datetime

    plotkw = {}
    plotkw['markersize'] = 12
    plotkw['marker_list'] = []
    # plotkw['linestyle'] = '--'

    if verbose:
        testres.print_unique_annot_config_stats(ibs)

    join_acfgs = True

    # cfgx2_shortlbl = testres.get_short_cfglbls()
    cfgx2_shortlbl = testres.get_short_cfglbls(join_acfgs=join_acfgs)

    if falsepos is None:
        falsepos = ut.get_argflag('--falsepos')
    if truepos is None:
        truepos = ut.get_argflag('--truepos')

    X_data_list = []
    X_label_list = []

    truth2_prop, prop2_mat = testres.get_truth2_prop(join_acfg=join_acfgs)
    # is_failure = prop2_mat['is_failure']
    is_success = prop2_mat['is_success']

    for cfgx, lbl in enumerate(cfgx2_shortlbl):
        gt_timedelta = truth2_prop['gt']['timedelta'].T
        gf_timedelta = truth2_prop['gf']['timedelta'].T
        # gt_f_td = gt_timedelta[cfgx][is_failure.T[cfgx]]
        # gf_f_td = gf_timedelta[cfgx][is_failure.T[cfgx]]
        gt_s_td = gt_timedelta[cfgx][is_success.T[cfgx]]
        gf_s_td = gf_timedelta[cfgx][is_success.T[cfgx]]
        if not falsepos or truepos:
            X_data_list += [
                gt_s_td,
                # gf_s_td
            ]
            X_label_list += [
                'TP ' + lbl,
                # 'FP ' + lbl
            ]
        if falsepos:
            X_data_list += [gf_s_td]
            X_label_list += ['FP ' + lbl]
        plotkw['marker_list'] += pt.distinct_markers(
            1, style='polygon', offset=cfgx, total=len(cfgx2_shortlbl)
        )

    numnan_list = [(~np.isfinite(X)).sum() for X in X_data_list]
    xdata_list = [X[~np.isnan(X)] for X in X_data_list]
    max_score = max([0 if len(xdata) == 0 else xdata.max() for xdata in xdata_list])

    bins = [
        datetime.timedelta(seconds=0).total_seconds(),
        datetime.timedelta(minutes=1).total_seconds(),
        datetime.timedelta(hours=1).total_seconds(),
        datetime.timedelta(days=1).total_seconds(),
        datetime.timedelta(weeks=1).total_seconds(),
        datetime.timedelta(days=356).total_seconds(),
        # np.inf,
        max(datetime.timedelta(days=356 * 10).total_seconds(), max_score + 1),
    ]

    # HISTOGRAM
    # if False:
    freq_list = [np.histogram(xdata, bins)[0] for xdata in xdata_list]
    timedelta_strs = [
        ut.get_timedelta_str(datetime.timedelta(seconds=b), exclude_zeros=True)
        for b in bins
    ]
    bin_labels = [
        length + ' - ' + height for length, height in ut.iter_window(timedelta_strs)
    ]
    bin_labels[-1] = '> 1 year'
    bin_labels[0] = '< 1 minute'
    WITH_NAN = True
    if WITH_NAN:
        freq_list = [
            np.append(freq, [numnan]) for freq, numnan in zip(freq_list, numnan_list)
        ]
        bin_labels += ['nan']

    # Make PIE chart
    fnum = None
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum)
    pnum_ = pt.make_pnum_nextgen(*pt.get_square_row_cols(len(freq_list)))
    bin_labels[0]
    # python -m wbia.dev -e timedelta_hist -t baseline -a
    # ctrl:force_const_size=True uncontrolled:force_const_size=True
    # --consistent --db GZ_ALL  --show
    colors = pt.distinct_colors(len(bin_labels))
    if WITH_NAN:
        colors[-1] = pt.GRAY

    for count, freq in enumerate(freq_list):
        pt.figure(fnum=fnum, pnum=pnum_())
        mask = freq > 0
        masked_freq = freq.compress(mask, axis=0)
        masked_lbls = ut.compress(bin_labels, mask)
        masked_colors = ut.compress(colors, mask)
        explode = [0] * len(masked_freq)
        size = masked_freq.sum()
        masked_percent = masked_freq * 100 / size
        pt.plt.pie(
            masked_percent,
            explode=explode,
            autopct='%1.1f%%',
            labels=masked_lbls,
            colors=masked_colors,
        )
        ax = pt.gca()
        ax.set_xlabel(X_label_list[count] + '\nsize=%d' % (size,))
        ax.set_aspect('equal')

    if ut.get_argflag('--contextadjust'):
        pt.adjust_subplots(left=0.08, bottom=0.1, top=0.9, wspace=0.3, hspace=0.1)
        pt.adjust_subplots(use_argv=True)


@profile
def draw_match_cases(
    ibs,
    testres,
    metadata=None,
    f=None,
    show_in_notebook=False,
    annot_modes=None,
    figsize=None,
    case_pos_list=None,
    verbose=None,
    interact=None,
    figdir=None,
    **kwargs
):
    r"""
    Args:
        ibs (wbia.IBEISController):  wbia controller object
        testres (TestResult):  test result object
        metadata (None): (default = None)

    CommandLine:
        python -m wbia --tf draw_match_cases
        python -m wbia.dev -e draw_match_cases --figdir=figure
        python -m wbia.dev -e draw_match_cases --db PZ_Master1 -a ctrl \
            -t default --filt :fail=True,min_gtrank=5,gtrank_lt=20 --render

        # Shows the best results
        python -m wbia.dev -e cases --db PZ_Master1 \
            -a timectrl -t invarbest
            --filt :sortasc=gtscore,success=True,index=200:201 --show

        # Shows failures sorted by gt score
        python -m wbia.dev -e cases --db PZ_Master1 \
            -a timectrl -t invarbest \
            --filt :sortdsc=gfscore,min_gtrank=1 --show

        # Find the untagged photobomb and scenery cases
        python -m wbia.dev -e cases --db PZ_Master1 -a timectrl \
            -t invarbest --show --filt \
            :orderby=gfscore,reverse=1,min_gtrank=1,max_gf_td=24h,max_gf_tags=0

        # Find untagged failures
        python -m wbia.dev -e cases --db PZ_Master1 -a timectrl \
            -t invarbest \
            --filt :orderby=gfscore,reverse=1,min_gtrank=1,max_gf_tags=0 --show

        # Show disagreement cases
        wbia --tf draw_match_cases --db PZ_MTEST -a default:size=20 \
            -t default:K=[1,4] \
            --filt :disagree=True,index=0:4 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_drawing import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST')
        >>> filt_cfg = main_helpers.testdata_filtcfg()
        >>> metadata = None
        >>> figdir = ut.get_argval(('--figdir', '--dpath'), type_=str, default=None)
        >>> analysis_fpath_list = draw_match_cases(ibs, testres, metadata,
        >>>                                        f=filt_cfg, figdir=figdir)
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt

    if ut.NOT_QUIET:
        ut.colorprint('[expt] Drawing individual results', 'yellow')

    if True:
        import matplotlib as mpl
        from wbia.scripts.thesis import TMP_RC

        mpl.rcParams.update(TMP_RC)

    # ### ARGUMENT PARSING AND RECTIFICATION ###
    cmdaug = ut.get_argval('--cmdaug', type_=str, help_='candhack')

    show = ut.get_argflag('--show')
    if interact is None:
        interact = ut.get_argflag('--interact')

    if figdir is None:
        figdir = ibs.get_fig_dir()
    show_kwargs = {'N': 3, 'ori': True, 'ell_alpha': 0.9}
    # show analysis
    show_kwargs['show_query'] = False
    show_kwargs['viz_name_score'] = kwargs.get('viz_name_score', True)
    show_kwargs['show_timedelta'] = True
    show_kwargs['show_gf'] = True
    show_kwargs['colorbar_'] = False
    show_kwargs['with_figtitle'] = show_in_notebook
    show_kwargs['fastmode'] = True
    if annot_modes is None:
        annot_modes = ut.get_argval('--annotmodes', default=[0])
    filt_cfg = f
    if case_pos_list is None:
        case_pos_list = testres.case_sample2(filt_cfg, verbose=verbose)  # NOQA
    #########################

    # ### DIRECTORY SETUP ###
    figdir = ut.truepath(figdir)
    case_figdir = join(figdir, 'cases_' + ibs.get_dbname())
    ut.ensuredir(case_figdir)
    if ut.get_argflag(('--view-fig-directory', '--vf')):
        ut.view_directory(case_figdir)
    # Common directory
    indiv_results_figdir = ut.ensuredir((case_figdir, 'indiv_results'))
    top_rank_analysis_dir = ut.ensuredir((case_figdir, 'top_rank_analysis'))
    dump = False
    DO_COPY_QUEUE = True
    if DO_COPY_QUEUE:
        cpq = draw_helpers.IndividualResultsCopyTaskQueue()
    if ut.NOT_QUIET:
        print('case_figdir = %r' % (case_figdir,))
    ##########################

    # ### INTERACTIVE SETUP ###
    def toggle_annot_mode():
        for ix in range(len(annot_modes)):
            # See viz_qres.py for more annot_mode info
            # TODO: use viz_qres to get the number of possible annot modes.
            annot_modes[ix] = annot_modes[ix] + 1 % 4

    def toggle_fast_mode():
        show_kwargs['fastmode'] = not show_kwargs['fastmode']
        print("show_kwargs['fastmode'] = %r" % (show_kwargs['fastmode'],))

    custom_actions = [
        ('present', ['s'], 'present', pt.present),
        ('toggle_annot_mode', ['a'], 'toggle_annot_mode', toggle_annot_mode),
        (
            'toggle_fast_mode',
            ['f'],
            'toggle_fast_mode',
            toggle_fast_mode,
            'Fast mode lowers drwaing quality',
        ),
    ]
    ##########################

    cfgx2_qreq_ = testres.cfgx2_qreq_
    qaids = testres.get_test_qaids()
    cfgx2_shortlbl = testres.get_short_cfglbls()
    qx_list, cfgx_list = case_pos_list.T
    # Get configs needed for each query
    qx2_cfgxs = ut.group_items(cfgx_list, qx_list)

    if show_in_notebook:
        cfg_colors = pt.distinct_colors(len(testres.cfgx2_qreq_))

    unique_qx = ut.unique(qx_list)
    if interact:
        _iter = ut.InteractiveIter(
            unique_qx, enabled=interact, custom_actions=custom_actions
        )
    else:
        _iter = ut.ProgIter(unique_qx, lbl='drawing cases')

    fnum = pt.ensure_fnum(None)
    fpaths_list = []
    analysis_fpath_list = []
    for count, qx in enumerate(_iter):
        cfgxs = qx2_cfgxs[qx]
        qreq_list = ut.take(cfgx2_qreq_, cfgxs)
        # TODO: try to get away with not reloading query results or loading
        # them in batch if possible
        # It actually doesnt take that long. the drawing is what hurts
        # TODO: be able to load old results even if they are currently invalid
        qaid = qaids[qx]
        cm_list = [qreq_.execute(qaids=[qaid])[0] for qreq_ in qreq_list]
        fpaths_list.append([])

        truth2_prop, prop2_mat = testres.get_truth2_prop()

        if 0 or ut.VERBOSE:
            print('=== QUERY INFO ===')
            print('=== QUERY INFO ===')
            print('qaid = %r' % (qaid,))
            print('qx = %r' % (qx,))
            print('cfgxs = %r' % (cfgxs,))
            # print testres info about this item
            take_cfgs = ut.partial(ut.take, index_list=cfgxs)
            take_qx = ut.partial(ut.take, index_list=qx)
            truth_cfgs = ut.hmap_vals(take_qx, truth2_prop)
            truth_item = ut.hmap_vals(take_cfgs, truth_cfgs, max_depth=1)
            prop_cfgs = ut.hmap_vals(take_qx, prop2_mat)
            prop_item = ut.hmap_vals(take_cfgs, prop_cfgs, max_depth=0)
            print('truth2_prop[item] = ' + ut.repr3(truth_item, nl=2))
            print('prop2_mat[item] = ' + ut.repr3(prop_item, nl=1))

        if show_in_notebook:
            # hack to show vertical line in notebook separate configs
            fnum = fnum + 1
            pt.imshow(np.zeros((1, 200), dtype=np.uint8), fnum=fnum)

        for count2, (cfgx, cm, qreq_) in enumerate(zip(cfgxs, cm_list, qreq_list)):
            if show_in_notebook:
                fnum = fnum + 1
            else:
                fnum = cfgx if interact else 1
            # cm = cm.extend_results(qreq_)
            # Get row and column index
            cfgstr = testres.get_cfgstr(cfgx)
            query_lbl = cfgx2_shortlbl[cfgx]
            qres_dpath = 'qaid={qaid}'.format(qaid=cm.qaid)
            individ_results_dpath = join(indiv_results_figdir, qres_dpath)
            ut.ensuredir(individ_results_dpath)
            # Draw Result
            # try to shorten query labels a bit
            query_lbl = query_lbl.replace(' ', '').replace("'", '')
            _query_lbl = query_lbl
            qres_fname = query_lbl + '.png'

            analysis_fpath = join(individ_results_dpath, qres_fname)

            if show or interact or show_in_notebook:
                needs_draw = True
            else:
                needs_draw = not (dump and ut.checkpath(analysis_fpath))

            if needs_draw:
                bar_label = 'Case: Query %r / %r, Config %r / %r --- qaid=%d, cfgx=%r' % (
                    count + 1,
                    len(qx_list),
                    count2 + 1,
                    len(cfgxs),
                    qaid,
                    cfgx,
                )
                print('bar_label = %r' % (bar_label,))
                if show_in_notebook:
                    # hack to show vertical line in notebook
                    if len(cfg_colors) > 0:
                        bar = np.zeros((1, 400, 3), dtype=np.uint8) + (
                            np.array(cfg_colors[cfgx]) * 255
                        )
                        fnum = fnum + 1
                        fig, ax = pt.imshow(bar, fnum=fnum)
                        pt.set_xlabel(bar_label, ax=ax)
                        pt.plt.show()  # Need to show when doing notebook
                for annot_mode in annot_modes:
                    show_kwargs['annot_mode'] = annot_mode
                    if show_in_notebook:
                        fnum = fnum + 1
                    if interact:
                        cm.ishow_analysis(
                            qreq_, figtitle=_query_lbl, fnum=fnum, **show_kwargs
                        )
                    else:
                        cm.show_analysis(
                            qreq_, figtitle=_query_lbl, fnum=fnum, **show_kwargs
                        )
                    if show_in_notebook:
                        _query_lbl = ''  # only show the query label once
                        if figsize is not None:
                            fig = pt.gcf()
                            fig.set_size_inches(*figsize)
                            fig.set_dpi(256)
                        pt.plt.show()
                if cmdaug is not None:
                    # Hack for candidacy
                    analysis_fpath = join(figdir, 'figuresC/case_%s.png' % (cmdaug,))
                    print('analysis_fpath = %r' % (analysis_fpath,))
                if show_in_notebook:
                    pt.plt.show()

                # elif not show:
                if False:
                    fig = pt.gcf()
                    print('analysis_fpath = %r' % (analysis_fpath,))
                    fig.savefig(analysis_fpath)
                    vt.clipwhite_ondisk(
                        analysis_fpath, analysis_fpath, verbose=ut.VERBOSE
                    )
                    if DO_COPY_QUEUE:
                        if cmdaug is None:
                            cpq.append_copy_task(analysis_fpath, top_rank_analysis_dir)
            analysis_fpath_list.append(analysis_fpath)
            fpaths_list[-1].append(analysis_fpath)
            if metadata is not None:
                metadata.set_global_data(
                    cfgstr, cm.qaid, 'analysis_fpath', analysis_fpath
                )

        # if some condition of of batch sizes
        if DO_COPY_QUEUE:
            flush_freq = 4
            if count % flush_freq == (flush_freq - 1):
                cpq.flush_copy_tasks()

    if DO_COPY_QUEUE:
        # Copy summary images to query_analysis folder
        cpq.flush_copy_tasks()

    return analysis_fpath_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.expt.experiment_drawing
        python -m wbia.expt.experiment_drawing --allexamples
        python -m wbia.expt.experiment_drawing --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
