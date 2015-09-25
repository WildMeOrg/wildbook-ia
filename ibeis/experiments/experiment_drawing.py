# -*- coding: utf-8 -*-
"""
./dev.py -t custom:affine_invariance=False,adapteq=True,fg_on=False --db Elephants_drop1_ears --allgt --index=0:10 --guiview
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
#from ibeis import params
import utool as ut
from ibeis.experiments import experiment_storage
from ibeis.model.hots import match_chips4 as mc4
from os.path import join, dirname, split, basename, splitext
from plottool import draw_func2 as df2
import vtool as vt
import re
#from plottool import plot_helpers as ph
from six.moves import map, range, input  # NOQA
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[expt_drawres]')
SKIP_TO = ut.get_argval(('--skip-to', '--skipto'), type_=int, default=None)
#SAVE_FIGURES = ut.get_argflag(('--save-figures', '--sf'))
SAVE_FIGURES = not ut.get_argflag(('--nosave-figures', '--nosf'))

SHOW                 = ut.get_argflag('--show')

# only triggered if dump_extra is on
DUMP_PROBCHIP = False
DUMP_REGCHIP = False


#fontkw = dict(legendsize=8, labelsize=10, ticksize=8, titlesize=8)
FONTKW = dict(legendsize=12, labelsize=12, ticksize=12, titlesize=14)


#@devcmd('scores', 'score', 'namescore_roc')
#def annotationmatch_scores(ibs, qaid_list, daid_list=None):
def annotationmatch_scores(ibs, test_result, f=None):
    """
    TODO: plot the difference between the top true score and the next best false score
    CommandLine:
        ib
        python dev.py -t scores --db PZ_MTEST --allgt -w --show
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg fg_on:True
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db GZ_ALL --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db PZ_Master0 --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db GZ_ALL --allgt -w --show

        python dev.py -t scores --db GZ_ALL --allgt -w --show --cfg codename='vsmany'
        python dev.py -t scores --db PZ_Master0 --allgt -w --show --cfg codename='vsmany'

        python dev.py -t scores --db PZ_Master0 --allgt --show
        python dev.py -t scores --db PZ_MTEST --allgt --show

        python dev.py -t scores --db PZ_Master0 --show --controlled
        python dev.py -t scores --db PZ_Master0 --show --controlled --cfg bar_l2_on:True lnbnn_on:False
        python dev.py -t scores --db PZ_Master0 --show --controlled --cfg fg_on:False
        python dev.py -t scores --db PZ_Master0 --show --controlled --cfg fg_on:False

        python -m ibeis.dev -e scores -t default  -a uncontrolled --db PZ_MTEST --show
        python -m ibeis.dev -e scores -t default -a timecontrolled --db PZ_Master1 --show
        python -m ibeis.dev -e scores -t default:AQH=False,AI=False -a timecontrolled --db PZ_Master1 --show --onlygood

        python -m ibeis.dev -e cases -a timecontrolled -t default:AQH=False,AI=False --db PZ_Master1 --qaid 1253 --show
        python -m ibeis.dev -e cases -a timecontrolled -t invarbest --db PZ_Master1 --qaid 574 --show

        python -m ibeis.dev -e scores -t invarbest -a timecontrolled:require_quality=True --db PZ_Master1 --filt :onlygood=False,smallfptime=False --show
        python -m ibeis.dev -e scores -t invarbest -a timecontrolled:require_quality=True --db PZ_Master1 --filt :fail=False,min_gf_timedelta=86400 --show
        python -m ibeis.dev -e scores -t invarbest -a timecontrolled:require_quality=True --db PZ_Master1 --filt :fail=False,min_gf_timedelta=24h --show


    Example:
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST', a=['uncontrolled'])
        >>> annotationmatch_scores(ibs, test_result)
        >>> ut.show_if_requested()
    """
    import plottool as pt
    import vtool as vt
    if ut.VERBOSE:
        print('[dev] annotationmatch_scores')
    #from ibeis.experiments import cfghelpers
    from ibeis.init import main_helpers

    filt_cfg = main_helpers.testdata_filtcfg(default=f)

    assert len(test_result.cfgx2_qreq_) == 1, 'can only specify one config here'
    cfgx = 0
    qreq_ = test_result.cfgx2_qreq_[cfgx]
    common_qaids = test_result.get_common_qaids()
    gt_rawscore = test_result.get_infoprop_mat('qx2_gt_raw_score').T[cfgx]
    gf_rawscore = test_result.get_infoprop_mat('qx2_gf_raw_score').T[cfgx]

    # FIXME: may need to specify which cfg is used in the future
    isvalid = test_result.case_sample2(filt_cfg, return_mask=True).T[cfgx]

    tp_nscores = gt_rawscore[isvalid]
    tn_nscores = gf_rawscore[isvalid]
    tn_qaids = tp_qaids = common_qaids[isvalid]

    #encoder = vt.ScoreNormalizer(target_tpr=.7)
    #print(qreq_.get_cfgstr())
    part_attrs = {1: {'qaid': tp_qaids},
                  0: {'qaid': tn_qaids}}

    def attr_callback(qaid):
        print('callback qaid = %r' % (qaid,))
        test_result.interact_individual_result(qaid)
        reconstruct_str = 'python -m ibeis.dev -e cases ' + test_result.reconstruct_test_flags() + ' --qaid ' + str(qaid) + ' --show'
        print('Independent reconstruct')
        print(reconstruct_str)

    #encoder = vt.ScoreNormalizer(adjust=8, tpr=.85)
    fpr = ut.get_argval('--fpr', type_=float, default=None)
    tpr = ut.get_argval('--tpr', type_=float, default=None if fpr is not None else .85)
    encoder = vt.ScoreNormalizer(adjust=8, fpr=fpr, tpr=tpr, monotonize=True)
    tp_scores = tp_nscores
    tn_scores = tn_nscores
    name_scores, labels, attrs = encoder._to_xy(tp_nscores, tn_nscores, part_attrs)
    encoder.fit(name_scores, labels, attrs)
    #encoder.visualize(figtitle='Learned Name Score Normalizer\n' + qreq_.get_cfgstr())

    plotname = ''
    figtitle = test_result.make_figtitle(plotname, filt_cfg=filt_cfg)

    encoder.visualize(
        figtitle=figtitle,
        #
        with_scores=False,
        with_prebayes=False,
        with_postbayes=False,
        #
        with_hist=True,
        with_roc=True,
        attr_callback=attr_callback,
        bin_width=.125,
    )

    if ut.get_argflag('--contextadjust'):
        pt.adjust_subplots(left=.1, bottom=.25, wspace=.2, hspace=.2)
        pt.adjust_subplots2(use_argv=True)

    locals_ = locals()
    return locals_


def draw_casetag_hist(ibs, test_result, f=None, with_wordcloud=not ut.get_argflag('--no-wordcloud')):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        test_result (TestResult):  test result object

    CommandLine:
        python -m ibeis.experiments.experiment_drawing --exec-draw_casetag_hist --show

        # Experiments I tagged
        python -m ibeis.experiments.experiment_drawing --exec-draw_casetag_hist -a timecontrolled -t invarbest --db PZ_Master1  --show

        ibeis -e taghist -a timequalctrl -t invarbest --db PZ_Master1  --show
        ibeis -e taghist -a timequalctrl:minqual=good -t invarbest --db PZ_Master1  --show
        ibeis -e taghist -a timequalctrl:minqual=good -t invarbest --db PZ_Master1  --show --filt :fail=True

        # Do more tagging
        ibeis -e cases -a timequalctrl:minqual=good -t invarbest --db PZ_Master1 --filt :orderby=gfscore,reverse=1,min_gtrank=1,max_gf_tags=0 --show
        ibeis -e print -a timequalctrl:minqual=good -t invarbest --db PZ_Master1 --show
        ibeis -e cases -a timequalctrl -t invarbest --db PZ_Master1 --filt :orderby=gfscore,reverse=1,max_gf_tags=0,:fail=True,min_gf_timedelta=12h --show

        ibeis -e cases -a timequalctrl -t invarbest --db PZ_Master1 --filt :orderby=gfscore,reverse=1,max_gf_tags=0,:fail=True,min_gf_timedelta=12h --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_Master1', a=['timequalcontrolled'])
        >>> draw_casetag_hist(ibs, test_result)
        >>> ut.show_if_requested()
    """
    from ibeis.init import main_helpers
    #from ibeis.experiments import cfghelpers
    import plottool as pt
    from ibeis import ibsfuncs
    filt_cfg = main_helpers.testdata_filtcfg(f)
    case_pos_list = test_result.case_sample2(filt_cfg)
    all_tags = test_result.get_all_tags()

    all_tags = [ibsfuncs.consolodate_tags(case_tags) for case_tags in all_tags]

    selected_tags = ut.list_take(all_tags, case_pos_list.T[0])
    flat_tags = list(map(str, ut.flatten(ut.flatten(selected_tags))))
    #print(ut.dict_str(ut.dict_hist(flat_tags), key_order_metric='val'))
    pnum_ = pt.make_pnum_nextgen(nRows=1, nCols=with_wordcloud + 1)
    fnum = None
    fnum = pt.ensure_fnum(fnum)
    pt.word_histogram2(flat_tags, fnum=fnum, pnum=pnum_(), xlabel='Tag')

    if with_wordcloud:
        pt.wordcloud(' '.join(flat_tags), fnum=fnum, pnum=pnum_())

    figtitle = test_result.make_figtitle('Tag Histogram', filt_cfg=filt_cfg)

    pt.set_figtitle(figtitle)

    if ut.get_argflag('--contextadjust'):
        #pt.adjust_subplots(left=.1, bottom=.25, wspace=.2, hspace=.2)
        #pt.adjust_subplots(wspace=.01)
        pt.adjust_subplots2(use_argv=True, wspace=.01)


@profile
def draw_individual_cases(ibs, test_result, metadata=None, f=None,
                          show_in_notebook=False, annot_modes=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        test_result (TestResult):
        metadata (None): (default = None)

    CommandLine:
        python -m ibeis.dev -e draw_individual_cases --figdir=individual_results
        python -m ibeis.dev -e draw_individual_cases --db PZ_Master1 -a controlled -t default --figdir=figures --vf --vh2 --show
        python -m ibeis.dev -e draw_individual_cases --db PZ_Master1 -a controlled -t default --filt :fail=True,min_gtrank=5,gtrank_lt=20 --render

        python -m ibeis.dev -e print --db PZ_Master1 -a timecontrolled -t invarbest
        python -m ibeis.dev -e cases --db PZ_Master1 -a timecontrolled -t invarbest --filt : --show

        # Shows the best results
        python -m ibeis.dev -e cases --db PZ_Master1 -a timecontrolled -t invarbest --filt :orderby=gfscore,reverse=1 --show

        # Shows failures sorted by gt score
        python -m ibeis.dev -e cases --db PZ_Master1 -a timecontrolled -t invarbest --filt :orderby=gfscore,reverse=1,min_gtrank=1 --show


        # Find the untagged photobomb and scenery cases
        python -m ibeis.dev -e cases --db PZ_Master1 -a timecontrolled -t invarbest --filt :orderby=gfscore,reverse=1,min_gtrank=1,max_gf_timedelta=24h,max_gf_tags=0 --show

        # Find untagged failures
        python -m ibeis.dev -e cases --db PZ_Master1 -a timecontrolled -t invarbest --filt :orderby=gfscore,reverse=1,min_gtrank=1,max_gf_tags=0 --show

        python -m ibeis.dev -e cases --db PZ_Master1 -a timecontrolled -t invarbest --filt :fail=True,min_gtrank=5,gtrank_lt=20 --render


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST')
        >>> metadata = None
        >>> analysis_fpath_list = draw_individual_cases(ibs, test_result, metadata)
        >>> #ut.show_if_requested()
    """
    if ut.NOT_QUIET:
        ut.colorprint('[expt] Drawing individual results', 'yellow')
    import plottool as pt
    cfgx2_qreq_ = test_result.cfgx2_qreq_

    # Get selected rows and columns for individual rank investigation
    #qaids = test_result.qaids

    #=================================
    # TODO:
    # Get a better (stratified) sample of the hard cases that incorporates the known failure cases
    # (Show a photobomb, scenery match, etc...)
    # This is just one config, because showing everything should also be an
    # option so we can find these errors
    #-------------
    # TODO;
    # Time statistics on incorrect results
    #=================================
    # Sel rows index into qx_list
    # Sel cols index into cfgx2 maps
    #_viewkw = dict(view_interesting=True)
    sel_rows, sel_cols, flat_case_labels = get_individual_result_sample(test_result, filt_cfg=f)
    if flat_case_labels is None:
        flat_case_labels = [None] * len(sel_rows)

    show_kwargs = {
        'N': 3,
        'ori': True,
        'ell_alpha': .9,
    }
    # SHOW ANALYSIS
    show_kwargs['show_query'] = False
    show_kwargs['viz_name_score'] = True
    show_kwargs['show_timedelta'] = True
    show_kwargs['show_gf'] = True
    #show_kwargs['with_figtitle'] = True
    show_kwargs['with_figtitle'] = show_in_notebook
    #show_kwargs['with_figtitle'] = show_in_notebook
    if annot_modes is None:
        if SHOW:
            annot_modes = [1]
        else:
            annot_modes = [1]
    #show_kwargs['annot_mode'] = 1 if not SHOW else 0

    cpq = IndividualResultsCopyTaskQueue()

    figdir = ibs.get_fig_dir()
    figdir = ut.truepath(ut.get_argval(('--figdir', '--dpath'), type_=str, default=figdir))
    #figdir = join(figdir, 'cases_' + test_result.get_fname_aug(withinfo=False))
    figdir = join(figdir, 'cases_' + ibs.get_dbname())
    ut.ensuredir(figdir)

    if ut.get_argflag(('--view-fig-directory', '--vf')):
        ut.view_directory(figdir)

    DRAW_ANALYSIS = True
    DRAW_BLIND = False and not SHOW
    #DUMP_EXTRA = ut.get_argflag('--dump-extra')
    #DRAW_QUERY_CHIP = DUMP_EXTRA
    #DRAW_QUERY_GROUNDTRUTH = DUMP_EXTRA
    #DRAW_QUERY_RESULT_CONTEXT  = DUMP_EXTRA

    # Common directory
    individual_results_figdir = join(figdir, 'individual_results')
    ut.ensuredir(individual_results_figdir)

    if DRAW_ANALYSIS:
        top_rank_analysis_dir = join(figdir, 'top_rank_analysis')
        ut.ensuredir(top_rank_analysis_dir)

    if DRAW_BLIND:
        blind_results_figdir  = join(figdir, 'blind_results')
        ut.ensuredir(blind_results_figdir)

    qaids = test_result.get_common_qaids()
    ibs.get_annot_semantic_uuids(ut.list_take(qaids, sel_rows))  # Ensure semantic uuids are in the APP cache.
    #samplekw = dict(per_group=5)
    #case_pos_list = test_result.get_case_positions('failure', samplekw=samplekw)
    #failure_qx_list = ut.unique_keep_order2(case_pos_list.T[0])
    #sel_rows = (np.array(failure_qx_list).tolist())
    #sel_cols = (list(range(test_result.nConfig)))

    def toggle_annot_mode():
        for ix in range(len(annot_modes)):
            annot_modes[ix] = (annot_modes[ix] + 1 % 3)
        #show_kwargs['annot_mode'] = (show_kwargs['annot_mode'] + 1) % 3
        #print('show_kwargs[annot_mode] = %r' % (show_kwargs['annot_mode'] ,))

    custom_actions = [
        ('present', ['s'], 'present', pt.present),
        ('toggle_annot_mode', ['f'], 'toggle_annot_mode', toggle_annot_mode),
    ]

    analysis_fpath_list = []

    def reset():
        if not SHOW:
            try:
                pt.fig_presenter.reset()
            except Exception as ex:
                if ut.VERBOSE:
                    ut.prinex(ex)
                pass

    #overwrite = False
    overwrite = ut.get_argflag('--overwrite')

    cfgx2_shortlbl = test_result.get_short_cfglbls(friendly=True)

    if ut.NOT_QUIET:
        print('figdir = %r' % (figdir,))
    fpaths_list = []

    fnum_start = None
    fnum = pt.ensure_fnum(fnum_start)

    if show_in_notebook:
        cfg_colors = pt.distinct_colors(len(test_result.cfgx2_qreq_))

    for count, qx in enumerate(ut.InteractiveIter(sel_rows, enabled=SHOW, custom_actions=custom_actions)):
        if SHOW:
            try:
                case_labels = flat_case_labels[count]
                print('case_labels = %r' % (case_labels,))
            except IndexError:
                print('flat_case_labels are known to be messed up')
                pass
        qreq_list = ut.list_take(cfgx2_qreq_, sel_cols)
        #qres_list = [load_qres(ibs, qaids[qx], qreq_) for qreq_ in qreq_list]
        # TODO: try to get away with not reloading query results or loading
        # them in batch if possible
        # It actually doesnt take that long. the drawing is what hurts
        qres_list = [qreq_.load_cached_qres(qaids[qx]) for qreq_ in qreq_list]
        fpaths_list.append([])

        if show_in_notebook:
            # hack to show vertical line in notebook separate configs
            fnum = fnum + 1
            pt.imshow(np.zeros((1, 200), dtype=np.uint8), fnum=fnum)

        for cfgx, qres, qreq_ in zip(sel_cols, qres_list, qreq_list):
            if show_in_notebook:
                fnum = fnum + 1
            else:
                fnum = cfgx if SHOW else 1
            # Get row and column index
            cfgstr = test_result.get_cfgstr(cfgx)
            query_lbl = cfgx2_shortlbl[cfgx]
            if False:
                qres_dpath = qres.get_fname(ext='', hack27=True)
            else:
                qres_dpath = 'qaid={qaid}'.format(qaid=qres.qaid)
            individ_results_dpath = join(individual_results_figdir, qres_dpath)
            ut.ensuredir(individ_results_dpath)
            # Draw Result
            # try to shorten query labels a bit
            query_lbl = query_lbl.replace(' ', '').replace('\'', '')
            qres_fname = query_lbl + '.png'
            #qres.show(ibs, 'analysis', figtitle=query_lbl, fnum=fnum, **show_kwargs)
            if DRAW_ANALYSIS:
                analysis_fpath = join(individ_results_dpath, qres_fname)
                #print('analysis_fpath = %r' % (analysis_fpath,))
                if SHOW or overwrite or not ut.checkpath(analysis_fpath) or show_in_notebook:
                    if show_in_notebook:
                        # hack to show vertical line in notebook
                        if len(cfg_colors) > 0:
                            bar = np.zeros((1, 400, 3), dtype=np.uint8) + (np.array(cfg_colors[cfgx]) * 255)
                            fnum = fnum + 1
                            pt.imshow(bar, fnum=fnum)
                    for annot_mode in annot_modes:
                        show_kwargs['annot_mode'] = annot_mode
                        if show_in_notebook:
                            # hack to show vertical line
                            fnum = fnum + 1
                        if SHOW:
                            qres.ishow_analysis(ibs, figtitle=query_lbl, fnum=fnum, qreq_=qreq_, **show_kwargs)
                        else:
                            qres.show_analysis(ibs, figtitle=query_lbl, fnum=fnum, qreq_=qreq_, **show_kwargs)

                    ## So hacky
                    #if ut.get_argflag('--tight'):
                    #    #pt.plt.tight_layout()
                    #    pass
                    fig = pt.gcf()
                    fig.savefig(analysis_fpath)
                    vt.clipwhite_ondisk(analysis_fpath, analysis_fpath, verbose=ut.VERBOSE)
                    cpq.append_copy_task(analysis_fpath, top_rank_analysis_dir)
                    #fig, fnum = prepare_figure_for_save(fnum, dpi, figsize, fig)
                    #analysis_fpath_ = pt.save_figure(fpath=analysis_fpath, **dumpkw)
                    #reset()
                analysis_fpath_list.append(analysis_fpath)
                fpaths_list[-1].append(analysis_fpath)
                if metadata is not None:
                    metadata.set_global_data(cfgstr, qres.qaid, 'analysis_fpath', analysis_fpath)

            # BLIND CASES - draws results without labels to see if we can determine what happened using doubleblind methods
            if DRAW_BLIND:
                pt.clf()
                best_gt_aid = qres.get_top_groundtruth_aid(ibs=ibs)
                qres.show_name_matches(
                    ibs, best_gt_aid, show_matches=False,
                    show_name_score=False, show_name_rank=False,
                    show_annot_score=False, fnum=fnum, qreq_=qreq_,
                    **show_kwargs)
                blind_figtitle = 'BLIND ' + query_lbl
                pt.set_figtitle(blind_figtitle)
                blind_fpath = join(individ_results_dpath, blind_figtitle) + '.png'
                pt.gcf().savefig(blind_fpath)
                #blind_fpath = pt.custom_figure.save_figure(fpath=blind_fpath, **dumpkw)
                cpq.append_copy_task(blind_fpath, blind_results_figdir)
                if metadata is not None:
                    metadata.set_global_data(cfgstr, qres.qaid, 'blind_fpath', blind_fpath)
                #reset()

            # REMOVE DUMP_FIG
            #extra_kw = dict(config2_=qreq_.get_external_query_config2(), subdir=subdir, **dumpkw)
            #if DRAW_QUERY_CHIP:
            #    _show_chip(ibs, qres.qaid, individual_results_figdir, 'QUERY_', **extra_kw)
            #    _show_chip(ibs, qres.qaid, individual_results_figdir, 'QUERY_CXT_', in_image=True, **extra_kw)

            #if DRAW_QUERY_GROUNDTRUTH:
            #    gtaids = ibs.get_annot_groundtruth(qres.qaid)
            #    for aid in gtaids:
            #        rank = qres.get_aid_ranks(aid)
            #        _show_chip(ibs, aid, individual_results_figdir, 'GT_CXT_', rank=rank, in_image=True, **extra_kw)

            #if DRAW_QUERY_RESULT_CONTEXT:
            #    topids = qres.get_top_aids(num=3)
            #    for aid in topids:
            #        rank = qres.get_aid_ranks(aid)
            #        _show_chip(ibs, aid, individual_results_figdir, 'TOP_CXT_', rank=rank, in_image=True, **extra_kw)

        # if some condition of of batch sizes
        flush_freq = 4
        if count % flush_freq == (flush_freq - 1):
            cpq.flush_copy_tasks()

    # Copy summary images to query_analysis folder
    cpq.flush_copy_tasks()

    make_individual_latex_figures(ibs, fpaths_list, flat_case_labels, cfgx2_shortlbl, figdir, analysis_fpath_list)
    return analysis_fpath_list


def make_individual_latex_figures(ibs, fpaths_list, flat_case_labels, cfgx2_shortlbl, figdir, analysis_fpath_list):
    # HACK MAKE LATEX CONVINENCE STUFF
    #print('LATEX HACK')
    if len(fpaths_list) == 0:
        print('nothing to render')
        return
    RENDER = ut.get_argflag('--render')
    DUMP_FIGDEF = ut.get_argflag(('--figdump', '--dump-figdef', '--figdef'))

    if not (DUMP_FIGDEF or RENDER):  # HACK
        return

    latex_code_blocks = []
    latex_block_keys = []

    caption_prefix = ut.get_argval('--cappref', type_=str, default='')
    caption_suffix = ut.get_argval('--capsuf', type_=str, default='')
    cmdaug = ut.get_argval('--cmdaug', type_=str, default='custom')

    selected = None

    for case_idx, (fpaths, labels) in enumerate(zip(fpaths_list, flat_case_labels)):
        if labels is None:
            labels = [cmdaug]
        if len(fpaths) < 4:
            nCols = len(fpaths)
        else:
            nCols = 2

        _cmdname = ibs.get_dbname() + ' Case ' + ' '.join(labels) + '_' + str(case_idx)
        #print('_cmdname = %r' % (_cmdname,))
        cmdname = ut.latex_sanatize_command_name(_cmdname)
        label_str = cmdname
        if len(caption_prefix) == 0:
            caption_str = ut.escape_latex('Casetags: ' + ut.list_str(labels, nl=False, strvals=True) + ', db=' + ibs.get_dbname() + '. ')
        else:
            caption_str = ''

        use_sublbls = len(cfgx2_shortlbl) > 1
        if use_sublbls:
            caption_str += ut.escape_latex('Each figure shows a different configuration: ')
            sublbls = ['(' + chr(97 + count) + ') ' for count in range(len(cfgx2_shortlbl))]
        else:
            #caption_str += ut.escape_latex('This figure depicts correct and incorrect matches from configuration: ')
            sublbls = [''] * len(cfgx2_shortlbl)
        def wrap_tt(text):
            return r'{\tt ' + text + '}'
        _shortlbls = cfgx2_shortlbl
        _shortlbls = list(map(ut.escape_latex, _shortlbls))
        # Adjust spacing for breaks
        #tex_small_space = r''
        tex_small_space = r'\hspace{0pt}'
        # Remove query specific config flags in individual results
        _shortlbls = [re.sub('\\bq[^,]*,?', '', shortlbl) for shortlbl in _shortlbls]
        # Let config strings be broken over newlines
        _shortlbls = [re.sub('\\+', tex_small_space + '+' + tex_small_space, shortlbl) for shortlbl in _shortlbls]
        _shortlbls = [re.sub(', *', ',' + tex_small_space, shortlbl) for shortlbl in _shortlbls]
        _shortlbls = list(map(wrap_tt, _shortlbls))
        cfgx2_texshortlbl = ['\n    ' + lbl + shortlbl for lbl, shortlbl in zip(sublbls, _shortlbls)]

        caption_str += ut.conj_phrase(cfgx2_texshortlbl, 'and') + '.\n    '
        caption_str = '\n    ' + caption_prefix + caption_str + caption_suffix
        caption_str = caption_str.rstrip()
        figure_str  = ut.get_latex_figure_str(fpaths,
                                                nCols=nCols,
                                                label_str=label_str,
                                                caption_str=caption_str,
                                                use_sublbls=None,
                                                use_frame=True)
        latex_block = ut.latex_newcommand(cmdname, figure_str)
        latex_block = '\n%----------\n' + latex_block
        latex_code_blocks.append(latex_block)
        latex_block_keys.append(cmdname)

    # HACK
    remove_fpath = ut.truepath('~/latex/crall-candidacy-2015') + '/'

    latex_fpath = join(figdir, 'latex_cases.tex')

    if selected is not None:
        selected_keys = selected
    else:
        selected_keys = latex_block_keys

    selected_blocks = ut.dict_take(dict(zip(latex_block_keys, latex_code_blocks)), selected_keys)

    figdef_block = '\n'.join(selected_blocks)
    figcmd_block = '\n'.join(['\\' + key for key in latex_block_keys])

    selected_block = figdef_block + '\n\n' + figcmd_block

    # HACK: need full paths to render
    selected_block_renderable = selected_block
    selected_block = selected_block.replace(remove_fpath, '')
    if RENDER:
        ut.render_latex_text(selected_block_renderable)

    if DUMP_FIGDEF:
        ut.writeto(latex_fpath, selected_block)

    if DUMP_FIGDEF or RENDER:
        ut.print_code(selected_block, 'latex')
    #else:
    #    print('STANDARD LATEX RESULTS')
    #    cmdname = ibs.get_dbname() + 'Results'
    #    latex_block  = ut.get_latex_figure_str2(analysis_fpath_list, cmdname, nCols=1)
    #    ut.print_code(latex_block, 'latex')


def get_individual_result_sample(test_result, filt_cfg=None, **kwargs):
    """
    The selected rows are the query annotation you are interested in viewing
    The selected cols are the parameter configuration you are interested in viewing

    Args:
        test_result (TestResult):  test result object
        filt_cfg (dict): config dict

    Kwargs:
        all, hard, hard2, easy, interesting, hist

    Returns:
        tuple: (sel_rows, sel_cols, flat_case_labels)

    CommandLine:
        python -m ibeis.experiments.experiment_drawing --exec-get_individual_result_sample --db PZ_Master1 -a controlled
        python -m ibeis.experiments.experiment_drawing --exec-get_individual_result_sample --db PZ_Master1 -a controlled --filt :fail=True,min_gtrank=5,gtrank_lt=20


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST')
        >>> filt_cfg = {'fail': True, 'success': True, 'min_gtrank': 5, 'max_gtrank': 40}
        >>> sel_rows, sel_cols, flat_case_labels = get_individual_result_sample(test_result, filt_cfg)
        >>> result = ('(sel_rows, sel_cols, flat_case_labels) = %s' % (str((sel_rows, sel_cols, flat_case_labels)),))
        >>> print(result)
    """
    #from ibeis.experiments import cfghelpers
    #sample_cfgstr_list = ut.get_argval('--filt', type_=list, default=None)
    #from ibeis.experiments import cfghelpers

    #if sample_cfgstr_list is None:
    if filt_cfg is None or isinstance(filt_cfg, list):
        # Hack to check if specified on command line
        from ibeis.init import main_helpers
        filt_cfg = main_helpers.testdata_filtcfg(default=filt_cfg)

    cfg_list = test_result.cfg_list
    #qaids = test_result.qaids
    qaids = test_result.get_common_qaids()

    view_all          = kwargs.get('all', ut.get_argflag(('--view-all', '--va')))
    view_hard         = kwargs.get('hard', ut.get_argflag(('--view-hard', '--vh')))
    view_hard2        = kwargs.get('hard2', ut.get_argflag(('--view-hard2', '--vh2')))
    view_easy         = kwargs.get('easy', ut.get_argflag(('--view-easy', '--vz')))
    view_interesting  = kwargs.get('interesting', ut.get_argflag(('--view-interesting', '--vn')))
    hist_sample       = kwargs.get('hist', ut.get_argflag(('--hs', '--hist-sample')))
    view_differ_cases = kwargs.get('differcases', ut.get_argflag(('--diff-cases', '--dc')))
    view_cases        = kwargs.get('cases', ut.get_argflag(('--view-cases', '--vc')))

    if ut.get_argval('--qaid', type_=str, default=None) is not None:
        # hack
        view_all = True

    #sel_cols = params.args.sel_cols  # FIXME
    #sel_rows = params.args.sel_rows  # FIXME
    #sel_cols = [] if sel_cols is None else sel_cols
    #sel_rows = [] if sel_rows is None else sel_rows
    sel_rows = []
    sel_cols = []
    flat_case_labels = None
    if ut.NOT_QUIET:
        print('remember to inspect with --show --sel-rows (-r) and --sel-cols (-c) ')
        print('other options:')
        print('   --vf - view figure dir')
        print('   --va - view all (--filt :)')
        print('   --vh - view hard (--filt :fail=True)')
        print('   --ve - view easy (--filt :success=True)')
        print('   --vn - view iNteresting')
        print('   --hs - hist sample')
        print(' --filt - result filtering config (new way to do this func)')
        print('   --gv, --guiview - gui result inspection')
    if len(sel_rows) > 0 and len(sel_cols) == 0:
        sel_cols = list(range(len(cfg_list)))
    if len(sel_cols) > 0 and len(sel_rows) == 0:
        sel_rows = list(range(len(qaids)))
    if view_all:
        sel_rows = list(range(len(qaids)))
        sel_cols = list(range(len(cfg_list)))
    if view_hard:
        new_hard_qx_list = test_result.get_new_hard_qx_list()
        sel_rows.extend(np.array(new_hard_qx_list).tolist())
        sel_cols.extend(list(range(len(cfg_list))))
    # sample-cases

    def convert_case_pos_to_cfgx(case_pos_list, case_labels_list):
        # Convert to all cfgx format
        qx_list = ut.unique_keep_order2(np.array(case_pos_list).T[0])
        ut.dict_take(ut.group_items(case_pos_list, case_pos_list.T[0]), qx_list)
        if case_labels_list is not None:
            grouped_labels = ut.dict_take(ut.group_items(case_labels_list, case_pos_list.T[0]), qx_list)
            flat_case_labels = list(map(ut.unique_keep_order2, map(ut.flatten, grouped_labels)))
        else:
            flat_case_labels = None
        new_rows = np.array(qx_list).tolist()
        new_cols = list(range(len(cfg_list)))
        return new_rows, new_cols, flat_case_labels

    if view_differ_cases:
        # Cases that passed on config but failed another
        case_pos_list, case_labels_list = test_result.case_type_sample(1, with_success=True, min_success_diff=1)
        new_rows, new_cols, flat_case_labels = convert_case_pos_to_cfgx(case_pos_list, case_labels_list)
        sel_rows.extend(new_rows)
        sel_cols.extend(new_cols)

    if view_cases:
        case_pos_list, case_labels_list = test_result.case_type_sample(1, with_success=False)
        new_rows, new_cols, flat_case_labels = convert_case_pos_to_cfgx(case_pos_list, case_labels_list)
        sel_rows.extend(new_rows)
        sel_cols.extend(new_cols)

    if view_hard2:
        # TODO handle returning case_pos_list
        #samplekw = ut.argparse_dict(dict(per_group=5))
        samplekw = ut.argparse_dict(dict(per_group=None))
        case_pos_list = test_result.get_case_positions(mode='failure', samplekw=samplekw)
        failure_qx_list = ut.unique_keep_order2(case_pos_list.T[0])
        sel_rows.extend(np.array(failure_qx_list).tolist())
        sel_cols.extend(list(range(len(cfg_list))))

    if view_easy:
        new_hard_qx_list = test_result.get_new_hard_qx_list()
        new_easy_qx_list = np.setdiff1d(np.arange(len(qaids)), new_hard_qx_list).tolist()
        sel_rows.extend(new_easy_qx_list)
        sel_cols.extend(list(range(len(cfg_list))))
    if view_interesting:
        interesting_qx_list = test_result.get_interesting_ranks()
        sel_rows.extend(interesting_qx_list)
        # TODO: grab the best scoring and most interesting configs
        if len(sel_cols) == 0:
            sel_cols.extend(list(range(len(cfg_list))))
    if hist_sample:
        # Careful if there is more than one config
        config_rand_bin_qxs = test_result.get_rank_histogram_qx_sample(size=10)
        sel_rows = np.hstack(ut.flatten(config_rand_bin_qxs))
        # TODO: grab the best scoring and most interesting configs
        if len(sel_cols) == 0:
            sel_cols.extend(list(range(len(cfg_list))))

    if filt_cfg is not None:
        # NEW WAY OF SAMPLING
        case_pos_list = test_result.case_sample2(filt_cfg)
        new_rows, new_cols, flat_case_labels = convert_case_pos_to_cfgx(case_pos_list, None)
        sel_rows.extend(new_rows)
        sel_cols.extend(new_cols)
        pass

    sel_rows = ut.unique_keep_order2(sel_rows)
    sel_cols = ut.unique_keep_order2(sel_cols)
    sel_cols = list(sel_cols)
    sel_rows = list(sel_rows)

    sel_rowxs = ut.get_argval('-r', type_=list, default=None)
    sel_colxs = ut.get_argval('-c', type_=list, default=None)

    if sel_rowxs is not None:
        sel_rows = ut.list_take(sel_rows, sel_rowxs)
        print('sel_rows = %r' % (sel_rows,))

    if sel_colxs is not None:
        sel_cols = ut.list_take(sel_cols, sel_colxs)

    if ut.NOT_QUIET:
        print('Returning Case Selection')
        print('len(sel_rows) = %r/%r' % (len(sel_rows), len(qaids)))
        print('len(sel_cols) = %r/%r' % (len(sel_cols), len(cfg_list)))

    return sel_rows, sel_cols, flat_case_labels


def draw_rank_surface(ibs, test_result, verbose=False, fnum=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        test_result (TestResult):  test result object

    CommandLine:
        python -m ibeis.dev -e draw_rank_surface --show  -t candidacy_k -a varysize  --db PZ_MTEST --show
        python -m ibeis.dev -e draw_rank_surface --show  -t candidacy_k -a varysize  --db PZ_Master0 --show
        python -m ibeis.dev -e draw_rank_surface --show  -t candidacy_k -a varysize2  --db PZ_Master0 --show

        python -m ibeis.dev -e draw_rank_surface -t candidacy_k -a varysize:qsize=500,dsize=[500,1000,1500,2000,2500,3000] --db PZ_Master1 --show


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST')
        >>> result = draw_rank_surface(ibs, test_result)
        >>> ut.show_if_requested()
        >>> print(result)
    """
    rank_le1_list = test_result.get_rank_cumhist(bins='dense')[0].T[0]
    percent_le1_list = 100 * rank_le1_list / len(test_result.qaids)
    #test_result.cfgx2_lbl
    #test_result.get_param_basis('dcfg_sample_per_name')
    #test_result.get_param_basis('dcfg_sample_size')
    #K_basis = test_result.get_param_basis('K')
    #K_cfgx_lists = [test_result.get_cfgx_with_param('K', K) for K in K_basis]

    #param_key_list = test_result.get_all_varied_params()
    param_key_list = ['K', 'dcfg_sample_per_name', 'dcfg_sample_size']
    #param_key_list = ['K', 'dcfg_sample_per_name', 'len(daids)']
    basis_dict      = {}
    cfgx_lists_dict = {}
    for key in param_key_list:
        _basis = test_result.get_param_basis(key)
        _cfgx_list = [test_result.get_cfgx_with_param(key, val) for val in _basis]
        cfgx_lists_dict[key] = _cfgx_list
        basis_dict[key] = _basis
    if verbose:
        print('basis_dict = ' + ut.dict_str(basis_dict, nl=1, hack_liststr=True))
        print('cfx_lists_dict = ' + ut.dict_str(cfgx_lists_dict, nl=2, hack_liststr=True))

    # Hold a key constant
    import plottool as pt
    #const_key = 'K'
    const_key = 'dcfg_sample_per_name'
    #pnum_ = pt.make_pnum_nextgen(*pt.get_square_row_cols(len(basis_dict[const_key]), max_cols=1))
    pnum_ = pt.make_pnum_nextgen(*pt.get_square_row_cols(len(basis_dict[const_key])))
    #ymax = percent_le1_list.max()
    #ymin = percent_le1_list.min()
    # Use consistent markers and colors.
    color_list = pt.distinct_colors(len(basis_dict[param_key_list[-1]]))
    marker_list = pt.distinct_markers(len(basis_dict[param_key_list[-1]]))

    fnum = pt.ensure_fnum(fnum)

    for const_idx, const_val in enumerate(basis_dict[const_key]):
        const_basis_cfgx_list = cfgx_lists_dict[const_key][const_idx]
        rank_list = ut.list_take(percent_le1_list, const_basis_cfgx_list)
        # Figure out what the values are for other dimensions
        agree_param_vals = dict([
            (key, [test_result.get_param_val_from_cfgx(cfgx, key) for cfgx in const_basis_cfgx_list])
            for key in param_key_list if key != const_key])

        from ibeis.experiments import annotation_configs
        nd_labels = list(agree_param_vals.keys())
        nd_labels = [annotation_configs.shorten_to_alias_labels(key) for key in nd_labels]
        target_label = annotation_configs.shorten_to_alias_labels(key)

        #target_label = 'num rank ≤ 1'
        #label_list = [ut.scalar_str(percent, precision=2) + '% - ' + label for percent, label in zip(cdf_list.T[0], label_list)]
        #target_label = '% groundtrue matches = rank 1'
        target_label = 'accuracy (%)'
        known_nd_data = np.array(list(agree_param_vals.values())).T
        known_target_points = np.array(rank_list)

        ymin = 30 if known_target_points.min() > 30 and False else 0
        num_yticks = 8 if ymin == 30 else 11

        #title = ('% Ranks = 1 when ' + annotation_configs.shorten_to_alias_labels(const_key) + '=%r' % (const_val,))
        title = ('accuracy when ' + annotation_configs.shorten_to_alias_labels(const_key) + '=%r' % (const_val,))
        #PLOT3D = not ut.get_argflag('--no3dsurf')
        PLOT3D = ut.get_argflag('--no2dsurf')
        if PLOT3D:
            pt.plot_search_surface(known_nd_data, known_target_points, nd_labels, target_label, title=title, fnum=fnum, pnum=pnum_())
        else:
            nonconst_basis_vals = np.unique(known_nd_data.T[1])
            # Find which colors will not be used
            nonconst_covers_basis = np.in1d(basis_dict[param_key_list[-1]], nonconst_basis_vals)
            nonconst_color_list = ut.list_compress(color_list, nonconst_covers_basis)
            nonconst_marker_list = ut.list_compress(marker_list, nonconst_covers_basis)
            pt.plot_multiple_scores(known_nd_data, known_target_points,
                                    nd_labels, target_label, title=title,
                                    color_list=nonconst_color_list,
                                    marker_list=nonconst_marker_list, fnum=fnum,
                                    pnum=pnum_(), num_yticks=num_yticks, ymin=ymin, ymax=100, ypad=.5,
                                    xpad=.05, legend_loc='lower right', **FONTKW)

    plotname = 'Effect of ' + ut.conj_phrase(nd_labels, 'and') + ' on Accuracy'
    figtitle = test_result.make_figtitle(plotname)

    #if ut.get_argflag('--save'):
    # hack
    if verbose:
        test_result.print_unique_annot_config_stats()

    #if test_result.has_constant_daids():
    #    print('test_result.common_acfg = ' + ut.dict_str(test_result.common_acfg))
    #    annotconfig_stats_strs, locals_ = ibs.get_annotconfig_stats(test_result.qaids, test_result.cfgx2_daids[0])

    pt.set_figtitle(figtitle, size=14)

    # HACK FOR FIGSIZE
    fig = pt.gcf()
    fig.set_size_inches(14, 4)
    pt.adjust_subplots(left=.05, bottom=.08, top=.80, right=.95, wspace=.2, hspace=.3)

    if ut.get_argflag('--contextadjust'):
        pt.adjust_subplots(left=.1, bottom=.25, wspace=.2, hspace=.2)
        pt.adjust_subplots2(use_argv=True)


def draw_rank_cdf(ibs, test_result, verbose=False, test_cfgx_slice=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        test_result (TestResult):

    CommandLine:

        python -m ibeis.dev -e draw_rank_cdf
        python -m ibeis.dev -e draw_rank_cdf --db PZ_MTEST --show
        python -m ibeis.dev -e draw_rank_cdf --db PZ_MTEST --show -a varysize -t default
        python -m ibeis.dev -e draw_rank_cdf --db PZ_MTEST --show -a controlled:qsize=1 controlled:qsize=3
        python -m ibeis.dev -e draw_rank_cdf -t candidacy_baseline --db PZ_MTEST -a controlled --show
        python -m ibeis.dev -e draw_rank_cdf -t candidacy_baseline --db PZ_Master0 --controlled --show
        python -m ibeis.dev -e draw_rank_cdf -t candidacy_namescore --db PZ_Master0 --controlled --show

        python -m ibeis.dev -e draw_rank_cdf -t candidacy_namescore --db PZ_MTEST --controlled --show

        python -m ibeis.dev -e draw_rank_cdf -t candidacy_baseline --db PZ_MTEST -a controlled --show

        python -m ibeis -tm exptdraw --exec-draw_rank_cdf -t candidacy_baseline -a controlled --db PZ_MTEST --show

        python -m ibeis.dev -e rank_cdf -t baseline:AQH=True,AI=False baseline:AQH=False,AI=True baseline:AQH=False,AI=False -a largetimedelta --db PZ_Master1 --show

        python -m ibeis.dev -e draw_rank_cdf -t candidacy_invariance -a controlled --db PZ_Master1 --show
        \
           --save invar_cumhist_{db}_a_{a}_t_{t}.png --dpath=~/code/ibeis/results  --adjust=.15 --dpi=256 --clipwhite --diskshow

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST')
        >>> result = draw_rank_cdf(ibs, test_result)
        >>> ut.show_if_requested()
        >>> print(result)
    """
    import plottool as pt
    #cdf_list, edges = test_result.get_rank_cumhist(bins='dense')
    cfgx2_cumhist_percent, edges = test_result.get_rank_percentage_cumhist(bins='dense')
    label_list = test_result.get_short_cfglbls(friendly=True)
    label_list = [
        ('%6.2f' % (percent,))
        #ut.scalar_str(percent, precision=2)
        + '% - ' + label
        for percent, label in zip(cfgx2_cumhist_percent.T[0], label_list)]
    color_list = pt.distinct_colors(len(label_list))
    marker_list = pt.distinct_markers(len(label_list))
    test_cfgx_slice = ut.get_argval('--test_cfgx_slice', type_='fuzzy_subset', default=test_cfgx_slice)
    if test_cfgx_slice is not None:
        print('test_cfgx_slice = %r' % (test_cfgx_slice,))
        cfgx2_cumhist_percent = np.array(ut.list_take(cfgx2_cumhist_percent,
                                                      test_cfgx_slice))
        label_list = ut.list_take(label_list, test_cfgx_slice)
        color_list = ut.list_take(color_list, test_cfgx_slice)
        marker_list = ut.list_take(marker_list, test_cfgx_slice)
    # Order cdf list by rank0
    sortx = cfgx2_cumhist_percent.T[0].argsort()[::-1]
    label_list = ut.list_take(label_list, sortx)
    cfgx2_cumhist_percent = np.array(ut.list_take(cfgx2_cumhist_percent, sortx))
    color_list = ut.list_take(color_list, sortx)
    marker_list = ut.list_take(marker_list, sortx)
    #

    figtitle = test_result.make_figtitle('Cumulative Rank Histogram')

    if verbose:
        test_result.print_unique_annot_config_stats(ibs)

    import vtool as vt
    # Find where the functions no longer change
    freq_deriv = np.diff(cfgx2_cumhist_percent.T[:-1].T)
    reverse_deriv_cumsum = freq_deriv[:, ::-1].cumsum(axis=0)
    reverse_changing_pos = np.array(ut.replace_nones(
        vt.find_first_true_indices(reverse_deriv_cumsum > 0), np.nan))
    nonzero_poses = (len(cfgx2_cumhist_percent.T) - 1) - reverse_changing_pos
    maxrank = np.nanmax(nonzero_poses)

    maxrank = 5
    #maxrank = ut.get_argval('--maxrank', type_=int, default=maxrank)

    if maxrank is not None:
        maxpos = min(len(cfgx2_cumhist_percent.T), maxrank)
        cfgx2_cumhist_short = cfgx2_cumhist_percent[:, 0:maxpos]
        edges_short = edges[0:min(len(edges), maxrank + 1)]

    USE_ZOOM = ut.get_argflag('--use-zoom')
    pnum_ = pt.make_pnum_nextgen(nRows=USE_ZOOM + 1, nCols=1)

    fnum = pt.ensure_fnum(None)
    target_label = 'accuracy (%)'
    #target_label = '% groundtrue matches ≤ rank'

    ymin = 30 if cfgx2_cumhist_percent.min() > 30 and False else 0
    num_yticks = 8 if ymin == 30 else 11

    cumhistkw = dict(
        xlabel='rank', ylabel=target_label, color_list=color_list,
        marker_list=marker_list, fnum=fnum, legend_loc='lower right',
        num_yticks=num_yticks, ymax=100, ymin=ymin, ypad=.5,
        xmin=.5, xmax=maxrank + .5,
        #xpad=.05,
        **FONTKW)

    pt.plot_rank_cumhist(
        cfgx2_cumhist_short, edges=edges_short, label_list=label_list,
        num_xticks=maxrank,
        use_legend=True, pnum=pnum_(), **cumhistkw)

    if USE_ZOOM:
        ax1 = pt.gca()
        pt.plot_rank_cumhist(
            cfgx2_cumhist_percent, edges=edges, label_list=label_list,
            num_xticks=maxrank, use_legend=False, pnum=pnum_(), **cumhistkw)
        ax2 = pt.gca()
        pt.zoom_effect01(ax1, ax2, 1, maxrank, fc='w')
    pt.set_figtitle(figtitle, size=14)
    # HACK FOR FIGSIZE

    fig = pt.gcf()
    #import utool as ut
    fig.set_size_inches(15, 7)
    if ut.get_argflag('--contextadjust'):
        pt.adjust_subplots(left=.05, bottom=.08, wspace=.0, hspace=.15)
        pt.adjust_subplots2(use_argv=True)
    #pt.set_figtitle(figtitle, size=10)


def make_metadata_custom_api(metadata):
    r"""
    CommandLine:
        python -m ibeis.experiments.experiment_drawing --test-make_metadata_custom_api --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> import guitool
        >>> guitool.ensure_qapp()
        >>> metadata_fpath = '/media/raid/work/Elephants_drop1_ears/_ibsdb/figures/result_metadata.shelf'
        >>> metadata = experiment_storage.ResultMetadata(metadata_fpath, autoconnect=True)
        >>> wgt = make_metadata_custom_api(metadata)
        >>> ut.quit_if_noshow()
        >>> wgt.show()
        >>> wgt.raise_()
        >>> guitool.qtapp_loop(wgt, frequency=100)
    """
    import guitool
    from guitool.__PYQT__ import QtCore

    class MetadataViewer(guitool.APIItemWidget):
        def __init__(wgt, parent=None, tblnice='Result Metadata Viewer', **kwargs):
            guitool.APIItemWidget.__init__(wgt, parent=parent, tblnice=tblnice, **kwargs)
            wgt.connect_signals_and_slots()

        @guitool.slot_(QtCore.QModelIndex)
        def _on_doubleclick(wgt, qtindex):
            print('[wgt] _on_doubleclick: ')
            col = qtindex.column()
            if wgt.api.col_edit_list[col]:
                print('do nothing special for editable columns')
                return
            model = qtindex.model()
            colname = model.get_header_name(col)
            if colname.endswith('fpath'):
                print('showing fpath')
                fpath = model.get_header_data(colname, qtindex)
                ut.startfile(fpath)

        def connect_signals_and_slots(wgt):
            #wgt.view.clicked.connect(wgt._on_click)
            wgt.view.doubleClicked.connect(wgt._on_doubleclick)
            #wgt.view.pressed.connect(wgt._on_pressed)
            #wgt.view.activated.connect(wgt._on_activated)

    guitool.ensure_qapp()
    #cfgstr_list = metadata
    col_name_list, column_list = metadata.get_square_data()

    # Priority of column names
    colname_priority = ['qaids', 'qx2_gt_rank', 'qx2_gt_timedelta',
                        'qx2_gf_timedelta',  'analysis_fpath',
                        'qx2_gt_raw_score', 'qx2_gf_raw_score']
    colname_priority += sorted(ut.setdiff_ordered(col_name_list, colname_priority))
    sortx = ut.priority_argsort(col_name_list, colname_priority)
    col_name_list = ut.list_take(col_name_list, sortx)
    column_list = ut.list_take(column_list, sortx)

    col_lens = list(map(len, column_list))
    print('col_name_list = %r' % (col_name_list,))
    print('col_lens = %r' % (col_lens,))
    assert len(col_lens) > 0, 'no columns'
    assert col_lens[0] > 0, 'no rows'
    assert all([len_ == col_lens[0] for len_ in col_lens]), 'inconsistant data'
    col_types_dict = {}
    col_getter_dict = dict(zip(col_name_list, column_list))
    col_bgrole_dict = {}
    col_ider_dict = {}
    col_setter_dict = {}
    col_nice_dict = {name: name.replace('qx2_', '') for name in col_name_list}
    col_nice_dict.update({
        'qx2_gt_timedelta': 'GT TimeDelta',
        'qx2_gf_timedelta': 'GF TimeDelta',
        'qx2_gt_rank': 'GT Rank',
    })
    editable_colnames = []
    sortby = 'qaids'
    get_thumb_size = lambda: 128
    col_width_dict = {}
    custom_api = guitool.CustomAPI(
        col_name_list, col_types_dict, col_getter_dict,
        col_bgrole_dict, col_ider_dict, col_setter_dict,
        editable_colnames, sortby, get_thumb_size,
        sort_reverse=True,
        col_width_dict=col_width_dict,
        col_nice_dict=col_nice_dict
    )
    #headers = custom_api.make_headers(tblnice='results')
    #print(ut.dict_str(headers))
    wgt = MetadataViewer()
    wgt.connect_api(custom_api)
    return wgt


def make_test_result_custom_api(ibs, test_result):
    import guitool
    guitool.ensure_qapp()
    cfgx = 0
    cfgres_info = test_result.cfgx2_cfgresinfo[cfgx]
    qaids = test_result.qaids
    gt_aids = cfgres_info['qx2_gt_aid']
    gf_aids = cfgres_info['qx2_gf_aid']
    qx2_gt_timedelta = ibs.get_annot_pair_timdelta(qaids, gt_aids)
    qx2_gf_timedelta = ibs.get_annot_pair_timdelta(qaids, gf_aids)
    col_name_list = [
        'qaids',
        'qx2_gt_aid',
        'qx2_gf_aid',
        'qx2_gt_timedelta',
        'qx2_gf_timedelta',
    ]
    col_types_dict = {}
    col_getter_dict = {}
    col_getter_dict.update(**cfgres_info)
    col_getter_dict['qaids'] = test_result.qaids
    col_getter_dict['qx2_gt_timedelta'] = qx2_gt_timedelta
    col_getter_dict['qx2_gf_timedelta'] = qx2_gf_timedelta
    col_bgrole_dict = {}
    col_ider_dict = {}
    col_setter_dict = {}
    editable_colnames = []
    sortby = 'qaids'
    get_thumb_size = lambda: 128
    col_width_dict = {}

    custom_api = guitool.CustomAPI(
        col_name_list, col_types_dict, col_getter_dict,
        col_bgrole_dict, col_ider_dict, col_setter_dict,
        editable_colnames, sortby, get_thumb_size, True, col_width_dict)
    #headers = custom_api.make_headers(tblnice='results')
    #print(ut.dict_str(headers))
    wgt = guitool.APIItemWidget()
    wgt.connect_api(custom_api)
    return wgt


def _show_chip(ibs, aid, individual_results_figdir, prefix, rank=None, in_image=False, seen=set([]), config2_=None, **dumpkw):
    print('[PRINT_RESULTS] show_chip(aid=%r) prefix=%r' % (aid, prefix))
    import plottool as pt
    from ibeis import viz
    # only dump a chip that hasn't been dumped yet
    if aid in seen:
        print('[PRINT_RESULTS] SEEN SKIPPING')
        return
    fulldir = join(individual_results_figdir, dumpkw['subdir'])
    if DUMP_PROBCHIP:
        # just copy it
        probchip_fpath = ibs.get_annot_probchip_fpath([aid], config2_=config2_)[0]
        ut.copy(probchip_fpath, fulldir, overwrite=False)
    if DUMP_REGCHIP:
        chip_fpath = ibs.get_annot_chip_fpath([aid], config2_=config2_)[0]
        ut.copy(chip_fpath, fulldir, overwrite=False)

    viz.show_chip(ibs, aid, in_image=in_image, config2_=config2_)
    if rank is not None:
        prefix += 'rank%d_' % rank
    fname = prefix + ibs.annotstr(aid)
    df2.set_figtitle(fname)
    seen.add(aid)
    if ut.VERBOSE:
        print('[expt] dumping fig to individual_results_figdir=%s' % individual_results_figdir)
    #fpath_clean = ph.dump_figure(individual_results_figdir, **dumpkw)
    fpath_ = join(individual_results_figdir, fname)
    pt.gcf().savefig(fpath_)
    return fpath_


class IndividualResultsCopyTaskQueue(object):
    def __init__(self):
        self.cp_task_list = []

    def append_copy_task(self, fpath_orig, dstdir=None):
        """ helper which copies a summary figure to root dir """
        fname_orig, ext = splitext(basename(fpath_orig))
        outdir = dirname(fpath_orig)
        fdir_clean, cfgdir = split(outdir)
        if dstdir is None:
            dstdir = fdir_clean
        #aug = cfgdir[0:min(len(cfgdir), 10)]
        aug = cfgdir
        fname_fmt = '{aug}_{fname_orig}{ext}'
        fmt_dict = {'aug': aug, 'fname_orig': fname_orig, 'ext': ext}
        fname_clean = ut.long_fname_format(fname_fmt, fmt_dict, ['fname_orig'], max_len=128)
        fdst_clean = join(dstdir, fname_clean)
        self.cp_task_list.append((fpath_orig, fdst_clean))

    def flush_copy_tasks(self):
        # Execute all copy tasks and empty the lists
        if ut.NOT_QUIET:
            print('[DRAW_RESULT] copying %r summaries' % (len(self.cp_task_list)))
        for src, dst in self.cp_task_list:
            ut.copy(src, dst, verbose=False)
        del self.cp_task_list[:]


@profile
def draw_case_timedeltas(ibs, test_result, falsepos=None, truepos=None, verbose=False):
    r"""

    CommandLine:
        python -m ibeis.dev -e draw_case_timedeltas

        python -m ibeis.dev -e timedelta_hist -t baseline -a uncontrolled controlled:force_const_size=True uncontrolled:force_const_size=True --consistent --db PZ_Master1 --show
        python -m ibeis.dev -e timedelta_hist -t baseline -a uncontrolled controlled:sample_rule_ref=max_timedelta --db PZ_Master1 --show --aidcfginfo


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST')
        >>> draw_case_timedeltas(ibs, test_result)
        >>> ut.show_if_requested()
    """
    #category_poses = test_result.partition_case_types()
    # TODO: Split up into cfgxs
    plotkw = FONTKW.copy()
    plotkw['markersize'] = 12
    plotkw['marker_list'] = []
    #plotkw['linestyle'] = '--'
    import plottool as pt

    if verbose:
        test_result.print_unique_annot_config_stats(ibs)

    truth2_prop, prop2_mat = test_result.get_truth2_prop()
    is_failure = prop2_mat['is_failure']
    is_success = prop2_mat['is_success']
    X_data_list = []
    X_label_list = []
    cfgx2_shortlbl = test_result.get_short_cfglbls(friendly=True)
    if falsepos is None:
        falsepos = ut.get_argflag('--falsepos')
    if truepos is None:
        truepos  = ut.get_argflag('--truepos')
    for cfgx, lbl in enumerate(cfgx2_shortlbl):
        gt_f_td = truth2_prop['gt']['timedelta'].T[cfgx][is_failure.T[cfgx]]  # NOQA
        gf_f_td = truth2_prop['gf']['timedelta'].T[cfgx][is_failure.T[cfgx]]  # NOQA
        gt_s_td = truth2_prop['gt']['timedelta'].T[cfgx][is_success.T[cfgx]]
        gf_s_td = truth2_prop['gf']['timedelta'].T[cfgx][is_success.T[cfgx]]  # NOQA
        #X_data_list  += [np.append(gt_f_td, gt_s_td), np.append(gf_f_td, gf_s_td)]
        #X_label_list += ['GT ' + lbl, 'GF ' + lbl]
        #X_data_list  += [gt_s_td, gt_f_td, gf_f_td, gf_s_td]
        #X_label_list += ['TP ' + lbl, 'FN ' + lbl, 'TN ' + lbl, 'FP ' + lbl]
        if not falsepos or truepos:
            X_data_list  += [
                gt_s_td,
                #gf_s_td
            ]
            X_label_list += [
                'TP ' + lbl,
                #'FP ' + lbl
            ]
        if falsepos:
            X_data_list  += [
                gf_s_td
            ]
            X_label_list += [
                'FP ' + lbl
            ]
        plotkw['marker_list'] += pt.distinct_markers(1, style='polygon',
                                                     offset=cfgx,
                                                     total=len(cfgx2_shortlbl))
        #plotkw['marker_list'] += pt.distinct_markers(1, style='astrisk', offset=cfgx, total=len(cfgx2_shortlbl))

    # TODO WRAP IN VTOOL
    # LEARN MULTI PDF
    #gridsize = 1024
    #adjust = 64
    numnan_list = [(~np.isfinite(X)).sum() for X in X_data_list]
    xdata_list = [X[~np.isnan(X)] for X in X_data_list]
    #import vtool as vt
    #xdata_pdf_list = [vt.estimate_pdf(xdata, gridsize=gridsize, adjust=adjust) for xdata in xdata_list]  # NOQA
    #min_score = min([xdata.min() for xdata in xdata_list])
    max_score = max([xdata.max() for xdata in xdata_list])
    #xdata_domain = np.linspace(min_score, max_score, gridsize)  # NOQA
    #pxdata_list = [pdf.evaluate(xdata_domain) for pdf in xdata_pdf_list]

    ## VISUALIZE MULTI PDF

    #import vtool as vt
    #encoder = vt.ScoreNormalizerUnsupervised(gt_f_td)
    #encoder.visualize()

    #import plottool as pt
    ##is_timedata = False
    #is_timedelta = True
    #pt.plot_probabilities(pxdata_list, X_label_list, xdata=xdata_domain)
    #ax = pt.gca()

    import datetime

    bins = [
        datetime.timedelta(seconds=0).total_seconds(),
        datetime.timedelta(minutes=1).total_seconds(),
        datetime.timedelta(hours=1).total_seconds(),
        datetime.timedelta(days=1).total_seconds(),
        datetime.timedelta(weeks=1).total_seconds(),
        datetime.timedelta(days=356).total_seconds(),
        #np.inf,
        max(datetime.timedelta(days=356 * 10).total_seconds(), max_score + 1),
    ]

    # HISTOGRAM
    #if False:
    freq_list = [np.histogram(xdata, bins)[0] for xdata in xdata_list]
    timedelta_strs = [ut.get_timedelta_str(datetime.timedelta(seconds=b), exclude_zeros=True) for b in bins]
    bin_labels = [l + ' - ' + h for l, h in ut.iter_window(timedelta_strs)]
    bin_labels[-1] = '> 1 year'
    bin_labels[0] = '< 1 minute'
    WITH_NAN = True
    if WITH_NAN:
        freq_list = [np.append(freq, [numnan]) for freq, numnan in zip(freq_list , numnan_list)]
        bin_labels += ['nan']

    # Convert to percent
    #freq_list = [100 * freq / len(is_success) for freq in freq_list]

    PIE = True

    if PIE:
        fnum = None
        fnum = pt.ensure_fnum(fnum)
        pt.figure(fnum=fnum)
        pnum_ = pt.make_pnum_nextgen(*pt.get_square_row_cols(len(freq_list)))
        bin_labels[0]
        # python -m ibeis.dev -e timedelta_hist -t baseline -a controlled:force_const_size=True uncontrolled:force_const_size=True --consistent --db GZ_ALL  --show

        colors = pt.distinct_colors(len(bin_labels))
        if WITH_NAN:
            colors[-1] = pt.GRAY

        for count, freq in enumerate(freq_list):
            pt.figure(fnum=fnum, pnum=pnum_())
            mask = freq > 0
            masked_freq   = freq.compress(mask, axis=0)
            masked_lbls   = ut.list_compress(bin_labels, mask)
            masked_colors = ut.list_compress(colors, mask)
            explode = [0] * len(masked_freq)
            size = masked_freq.sum()
            masked_percent = (masked_freq * 100 / size)
            pt.plt.pie(masked_percent, explode=explode, autopct='%1.1f%%', labels=masked_lbls, colors=masked_colors)
            ax = pt.gca()
            ax.set_xlabel(X_label_list[count] + '\nsize=%d' % (size,))
            ax.set_aspect('equal')

        if ut.get_argflag('--contextadjust'):
            pt.adjust_subplots2(left=.08, bottom=.1, top=.9, wspace=.3, hspace=.1)
            pt.adjust_subplots2(use_argv=True)
    else:
        pass
        #xints = np.arange(len(bin_labels))
        #pt.multi_plot(xints, freq_list, label_list=X_label_list, xpad=1, ypad=.5, **plotkw)
        #ax = pt.gca()

        #xtick_labels = [''] + bin_labels + ['']

        #ax.set_xticklabels(xtick_labels)
        #ax.set_xlabel('timedelta')
        ##ax.set_ylabel('Frequency')
        #ax.set_ylabel('% true positives')

        #plotname = 'Timedelta histogram of correct matches'
        #figtitle = test_result.make_figtitle(plotname)
        #ax.set_title(figtitle)
        #pt.gcf().autofmt_xdate()

        #if ut.get_argflag('--contextadjust'):
        #    pt.adjust_subplots(left=.2, bottom=.2, wspace=.0, hspace=.15)
        #    pt.adjust_subplots2(use_argv=True)


@profile
def draw_results(ibs, test_result):
    r"""
    Draws results from an experiment harness run.
    Rows store different qaids (query annotation ids)
    Cols store different configurations (algorithm parameters)

    Args:
        test_result (experiment_storage.TestResult):

    CommandLine:
        python dev.py -t custom:rrvsone_on=True,constrained_coeff=0 custom --qaid 12 --db PZ_MTEST --show --va
        python dev.py -t custom:rrvsone_on=True,constrained_coeff=.3 custom --qaid 12 --db PZ_MTEST --show --va --noqcache
        python dev.py -t custom:rrvsone_on=True custom --qaid 4 --db PZ_MTEST --show --va --noqcache

        python dev.py -t custom:rrvsone_on=True,grid_scale_factor=1 custom --qaid 12 --db PZ_MTEST --show --va --noqcache
        python dev.py -t custom:rrvsone_on=True,grid_scale_factor=1,grid_steps=1 custom --qaid 12 --db PZ_MTEST --show --va --noqcache

    CommandLine:
        python dev.py -t best --db seals2 --allgt --vz --fig-dname query_analysis_easy --show
        python dev.py -t best --db seals2 --allgt --vh --fig-dname query_analysis_hard --show

        python dev.py -t pyrscale --db PZ_MTEST --allgt --vn --fig-dname query_analysis_interesting --show
        python dev.py -t pyrscale --db testdb3 --allgt --vn --fig-dname query_analysis_interesting --vf
        python dev.py -t pyrscale --db testdb3 --allgt --vn --fig-dname query_analysis_interesting --vf --quality


        python -m ibeis.experiments.experiment_drawing --test-draw_results --show --vn
        python -m ibeis.experiments.experiment_drawing --test-draw_results --show --vn --db PZ_MTEST
        python -m ibeis.experiments.experiment_drawing --test-draw_results --show --db PZ_MTEST --draw-rank-cdf

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST')
        >>> # execute function
        >>> result = draw_results(ibs, test_result)
        >>> # verify results
        >>> print(result)
    """
    print(' --- DRAW RESULTS ---')

    # It is very inefficient to turn off caching when view_all is true
    if not mc4.USE_CACHE:
        print('WARNING: view_all specified with USE_CACHE == False')
        print('WARNING: we will try to turn cache on when reloading results')
        #mc4.USE_CACHE = True

    figdir = ibs.get_fig_dir()
    ut.ensuredir(figdir)

    if ut.get_argflag(('--view-fig-directory', '--vf')):
        ut.view_directory(figdir)

    figdir_suffix = ut.get_argval('--fig-dname', type_=str, default=None)
    if figdir_suffix is not None:
        figdir = join(figdir, figdir_suffix)
        ut.ensuredir(figdir)
    #gx2_gt_timedelta
    #    cfgres_info['qx2_gf_timedelta'] = qx2_gf_timedelta

    metadata_fpath = join(figdir, 'result_metadata.shelf')
    metadata = experiment_storage.ResultMetadata(metadata_fpath)
    #metadata.rrr()
    metadata.connect()
    metadata.sync_test_results(test_result)
    #cfgstr = qreq_.get_cfgstr()
    #cfg_metadata = ensure_item(metadata, cfgstr, {})
    #avuuids = ibs.get_annot_visual_uuids(qaids)
    #avuuid2_ax = ensure_item(cfg_metadata, 'avuuid2_ax', {})
    #cfg_columns = ensure_item(cfg_metadata, 'columns', {})
    #import guitool

    ut.argv_flag_dec(draw_rank_cdf)(ibs, test_result)

    VIZ_INDIVIDUAL_RESULTS = True
    if VIZ_INDIVIDUAL_RESULTS:
        draw_individual_cases(ibs, test_result, metadata=metadata)

    metadata.write()
    if ut.get_argflag(('--guiview', '--gv')):
        import guitool
        guitool.ensure_qapp()
        #wgt = make_test_result_custom_api(ibs, test_result)
        wgt = make_metadata_custom_api(metadata)
        wgt.show()
        wgt.raise_()
        guitool.qtapp_loop(wgt, frequency=100)
    metadata.close()

    if ut.NOT_QUIET:
        print('[DRAW_RESULT] EXIT EXPERIMENT HARNESS')


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.experiments.experiment_drawing
        python -m ibeis.experiments.experiment_drawing --allexamples
        python -m ibeis.experiments.experiment_drawing --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
