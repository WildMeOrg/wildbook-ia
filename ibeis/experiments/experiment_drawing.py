# -*- coding: utf-8 -*-
"""
./dev.py -t custom:affine_invariance=False,adapteq=True,fg_on=False --db Elephants_drop1_ears --allgt --index=0:10 --guiview
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from ibeis import params
import utool as ut
from ibeis.experiments import experiment_storage
from ibeis.model.hots import match_chips4 as mc4
from os.path import join, dirname, split, basename, splitext
from plottool import draw_func2 as df2
from plottool import plot_helpers as ph
from six.moves import map, range, input  # NOQA
import vtool as vt
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[expt_drawres]')
SKIP_TO = ut.get_argval(('--skip-to', '--skipto'), type_=int, default=None)
#SAVE_FIGURES = ut.get_argflag(('--save-figures', '--sf'))
SAVE_FIGURES = not ut.get_argflag(('--nosave-figures', '--nosf'))

QUALITY              = ut.get_argflag('--quality')
SHOW                 = ut.get_argflag('--show')

# only triggered if dump_extra is on
DUMP_PROBCHIP = False
DUMP_REGCHIP = False


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
    colname_priority = ['qaids', 'qx2_gt_rank', 'qx2_gt_timedelta', 'qx2_gf_timedelta',  'analysis_fpath', 'qx2_gt_raw_score', 'qx2_gf_raw_score']
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


def get_diffranks(rank_mat, qaids):
    """ Find rows which scored differently over the various configs
    FIXME: duplicated
    """
    isdiff_flags = [not np.all(row == row[0]) for row in rank_mat]
    diff_aids    = ut.list_compress(qaids, isdiff_flags)
    diff_rank    = rank_mat.compress(isdiff_flags, axis=0)
    diff_qxs     = np.where(isdiff_flags)[0]
    return diff_aids, diff_rank, diff_qxs


def get_interesting_ranks(rank_mat, qaids):
    # find the rows that vary greatest with the parameter settings
    diff_aids, diff_rank, diff_qxs = get_diffranks(rank_mat, qaids)
    if False:
        rankcategory = np.log(diff_rank + 1)
    else:
        rankcategory = diff_rank.copy()
        rankcategory[diff_rank == 0]  = 0
        rankcategory[diff_rank > 0]   = 1
        rankcategory[diff_rank > 2]   = 2
        rankcategory[diff_rank > 5]   = 3
        rankcategory[diff_rank > 50]  = 4
        rankcategory[diff_rank > 100] = 5
    row_rankcategory_std = np.std(rankcategory, axis=1)
    row_rankcategory_mean = np.mean(rankcategory, axis=1)
    row_sortx = vt.argsort_multiarray([row_rankcategory_std, row_rankcategory_mean], reverse=True)

    interesting_qx_list = diff_qxs.take(row_sortx).tolist()
    #print("INTERSETING MEASURE")
    #print(interesting_qx_list)
    #print(row_rankcategory_std)
    #print(ut.list_take(qaids, row_sortx))
    #print(diff_rank.take(row_sortx, axis=0))
    return interesting_qx_list


def _show_chip(ibs, aid, individual_results_figdir, prefix, rank=None, in_image=False, seen=set([]), config2_=None, **dumpkw):
    print('[PRINT_RESULTS] show_chip(aid=%r) prefix=%r' % (aid, prefix))
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
    df2.set_figtitle(prefix + ibs.annotstr(aid))
    seen.add(aid)
    if ut.VERBOSE:
        print('[expt] dumping fig to individual_results_figdir=%s' % individual_results_figdir)

    fpath_clean = ph.dump_figure(individual_results_figdir, **dumpkw)
    return fpath_clean


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
        print('[DRAW_RESULT] copying %r summaries' % (len(self.cp_task_list)))
        for src, dst in self.cp_task_list:
            ut.copy(src, dst, verbose=False)
        del self.cp_task_list[:]


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
        >>> from ibeis.experiments.experiment_printres import *  # NOQA
        >>> from ibeis.experiments import experiment_harness
        >>> import ibeis
        >>> # build test data
        >>> species = ibeis.const.Species.ZEB_PLAIN
        >>> #ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> ibs = ibeis.opendb(defaultdb='testdb3')
        >>> #test_cfg_name_list = ['pyrscale']
        >>> test_cfg_name_list = ['custom', 'custom:sv_on=False']
        >>> qaids = ibs.get_valid_aids(species=species, hasgt=True)
        >>> daids = ibs.get_valid_aids(species=species)
        >>> test_result = experiment_harness.run_test_configurations(ibs, qaids, daids, test_cfg_name_list)
        >>> # execute function
        >>> result = draw_results(ibs, test_result)
        >>> # verify results
        >>> print(result)
    """
    print(' --- DRAW RESULTS ---')
    import plottool as pt

    # It is very inefficient to turn off caching when view_all is true
    if not mc4.USE_CACHE:
        print('WARNING: view_all specified with USE_CACHE == False')
        print('WARNING: we will try to turn cache on when reloading results')
        #mc4.USE_CACHE = True

    qaids = test_result.qaids
    daids = test_result.daids
    rank_mat = test_result.get_rank_mat()
    interesting_qx_list = get_interesting_ranks(rank_mat, qaids)

    (cfg_list, cfgx2_lbl, cfgx2_qreq_) = ut.dict_take(
        test_result.__dict__, ['cfg_list', 'cfgx2_lbl', 'cfgx2_qreq_'])

    figdir = ibs.get_fig_dir()
    ut.ensuredir(figdir)

    if ut.get_argflag(('--view-fig-directory', '--vf')):
        ut.view_directory(figdir)

    figdir_suffix = ut.get_argval('--fig-dname', type_=str, default=None)
    if figdir_suffix is not None:
        figdir = join(figdir, figdir_suffix)
        ut.ensuredir(figdir)

    individual_results_figdir = join(figdir, 'individual_results')
    aggregate_results_figdir  = join(figdir, 'aggregate_results')
    blind_results_figdir  = join(figdir, 'blind_results')
    top_rank_analysis_dir = join(figdir, 'top_rank_analysis')
    ut.ensuredir(individual_results_figdir)
    ut.ensuredir(aggregate_results_figdir)
    ut.ensuredir(top_rank_analysis_dir)
    ut.ensuredir(blind_results_figdir)
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

    dumpkw = {
        'quality'   : QUALITY,
        'overwrite' : True,
        'verbose'   : 0,
    }

    @ut.argv_flag_dec
    def draw_rank_cdf():
        cdf_list, edges = test_result.get_rank_cumhist(bins='dense')
        lbl_list = test_result.get_short_cfglbls()
        # Order cdf list by rank0
        lbl_list = ut.sortedby(lbl_list, cdf_list.T[0], reverse=True)
        cdf_list = np.array(ut.sortedby(cdf_list.tolist(), cdf_list.T[0], reverse=True))
        #
        figtitle = 'Cumulative Histogram of GT-Ranks for db=' + (ibs.get_dbname())
        #cdf_list = config_cdfs
        maxrank = ut.get_argval('--maxrank', type_=int, default=None)
        if maxrank is not None:
            cdf_list = cdf_list[:, 0:min(len(cdf_list.T), maxrank)]
            edges = edges[0:min(len(edges), maxrank + 1)]
        fig = pt.plot_rank_cumhist(cdf_list, lbl_list, edges=edges, figtitle=figtitle)  # NOQA
        #ut.show_if_requested()
        #rank_cdf_fpath = ph.dump_figure(aggregate_results_figdir, reset=not SHOW, subdir=None, **dumpkw)
        #print(rank_cdf_fpath)
        #rank_cdf_fpath  # NOQA
        #if SHOW:
        #    pt.plt.show()
    draw_rank_cdf()

    VIZ_INDIVIDUAL_RESULTS = True
    if VIZ_INDIVIDUAL_RESULTS:
        #_viewkw = dict(view_interesting=True)
        _viewkw = {}
        # Get selected rows and columns for individual rank investigation
        new_hard_qx_list = test_result.get_new_hard_qx_list()
        sel_rows, sel_cols = get_sel_rows_and_cols(
            qaids, cfg_list, new_hard_qx_list, interesting_qx_list, test_result, **_viewkw)

        show_kwargs = {
            'N': 3,
            'ori': True,
            'ell_alpha': .9,
        }

        cpq = IndividualResultsCopyTaskQueue()

        def load_qres(ibs, qaid, daids, qreq_):
            """ Load / Execute the query w/ correct config """
            # TODO: try to get away with not reloading query results or loading
            # them in batch if possible
            qreq_.set_external_qaids([qaid])
            qres = ibs._query_chips4(
                [qaid], daids, use_cache=True, use_bigcache=False,
                qreq_=qreq_)[qaid]
            return qres

        for count, r in enumerate(ut.InteractiveIter(sel_rows, enabled=SHOW)):
            qreq_list = ut.list_take(cfgx2_qreq_, sel_cols)
            qres_list = [load_qres(ibs, qaids[r], daids, qreq_) for qreq_ in qreq_list]

            for cfgx, qres, qreq_ in zip(sel_cols, qres_list, qreq_list):
                fnum = cfgx if SHOW else 1
                # Get row and column index
                cfgstr = test_result.get_cfgstr(cfgx)
                query_lbl = cfgx2_lbl[cfgx]
                qres_cfg = qres.get_fname(ext='')
                subdir = qres_cfg
                # Draw Result
                # try to shorten query labels a bit
                query_lbl = query_lbl.replace(' ', '').replace('\'', '')
                #qres.show(ibs, 'analysis', figtitle=query_lbl, fnum=fnum, **show_kwargs)

                # SHOW ANALYSIS
                DRAW_ANALYSIS = True
                if DRAW_ANALYSIS:
                    if SHOW:
                        #show_kwargs['show_query'] = False
                        show_kwargs['viz_name_score'] = True
                        show_kwargs['show_timedelta'] = True
                        qres.ishow_analysis(ibs, figtitle=query_lbl, fnum=fnum, annot_mode=1, qreq_=qreq_, **show_kwargs)
                    else:
                        show_kwargs['show_query'] = False
                        #show_kwargs['viz_name_score'] = False
                        show_kwargs['viz_name_score'] = True
                        show_kwargs['show_timedelta'] = True
                        qres.show_analysis(ibs, figtitle=query_lbl, fnum=fnum, annot_mode=1, qreq_=qreq_, **show_kwargs)
                    analysis_fpath = ph.dump_figure(individual_results_figdir, reset=not SHOW, subdir=subdir, **dumpkw)
                    metadata.set_global_data(cfgstr, qres.qaid, 'analysis_fpath', analysis_fpath)
                    cpq.append_copy_task(analysis_fpath, top_rank_analysis_dir)

                # BLIND CASES - draws results without labels to see if we can determine what happened using doubleblind methods
                DRAW_BLIND = not SHOW
                if DRAW_BLIND:
                    pt.clf()
                    best_gt_aid = qres.get_top_groundtruth_aid(ibs=ibs)
                    qres.show_name_matches(ibs, best_gt_aid,
                                           show_matches=False,
                                           show_name_score=False,
                                           show_name_rank=False,
                                           show_annot_score=False, fnum=fnum,
                                           qreq_=qreq_, **show_kwargs)
                    pt.set_figtitle('BLIND ' + query_lbl)
                    blind_fpath = ph.dump_figure(individual_results_figdir, reset=not SHOW, subdir=subdir, **dumpkw)
                    cpq.append_copy_task(blind_fpath, blind_results_figdir)
                    metadata.set_global_data(cfgstr, qres.qaid, 'blind_fpath', blind_fpath)

                DUMP_EXTRA = ut.get_argflag('--dump-extra')
                DRAW_QUERY_CHIP = DUMP_EXTRA
                extra_kw = dict(config2_=qreq_.get_external_query_config2(), subdir=subdir, **dumpkw)
                if DRAW_QUERY_CHIP:
                    _show_chip(ibs, qres.qaid, individual_results_figdir, 'QUERY_', **extra_kw)
                    _show_chip(ibs, qres.qaid, individual_results_figdir, 'QUERY_CXT_', in_image=True, **extra_kw)

                DRAW_QUERY_GROUNDTRUTH = DUMP_EXTRA
                if DRAW_QUERY_GROUNDTRUTH:
                    gtaids = ibs.get_annot_groundtruth(qres.qaid)
                    for aid in gtaids:
                        rank = qres.get_aid_ranks(aid)
                        _show_chip(ibs, aid, individual_results_figdir, 'GT_CXT_', rank=rank, in_image=True, **extra_kw)

                DRAW_QUERY_RESULT_CONTEXT  = DUMP_EXTRA
                if DRAW_QUERY_RESULT_CONTEXT:
                    topids = qres.get_top_aids(num=3)
                    for aid in topids:
                        rank = qres.get_aid_ranks(aid)
                        _show_chip(ibs, aid, individual_results_figdir, 'TOP_CXT_', rank=rank, in_image=True, **extra_kw)

            # if some condition of of batch sizes
            flush_freq = 4
            if count % flush_freq == (flush_freq - 1):
                cpq.flush_copy_tasks()

        # Copy summary images to query_analysis folder
        cpq.flush_copy_tasks()

    metadata.write()
    #ut.embed()
    #if ut.is_developer():
    if ut.get_argflag(('--guiview', '--gv')):
        import guitool
        guitool.ensure_qapp()
        #wgt = make_test_result_custom_api(ibs, test_result)
        wgt = make_metadata_custom_api(metadata)
        wgt.show()
        wgt.raise_()
        guitool.qtapp_loop(wgt, frequency=100)
    #ut.embed()
    metadata.close()

    if ut.NOT_QUIET:
        print('[DRAW_RESULT] EXIT EXPERIMENT HARNESS')


def get_sel_rows_and_cols(qaids, cfg_list, new_hard_qx_list, interesting_qx_list, test_result,
                          view_all=ut.get_argflag(('--view-all', '--va')),
                          view_hard=ut.get_argflag(('--view-hard', '--vh')),
                          view_easy=ut.get_argflag(('--view-easy', '--vz')),
                          view_interesting=ut.get_argflag(('--view-interesting', '--vn')),
                          **kwargs):
    """
    The selected rows are the query annotation you are interested in viewing
    The selected cols are the parameter configuration you are interested in viewing
    """
    sel_cols = params.args.sel_cols  # FIXME
    sel_rows = params.args.sel_rows  # FIXME
    sel_cols = [] if sel_cols is None else sel_cols
    sel_rows = [] if sel_rows is None else sel_rows
    #sel_rows = []
    #sel_cols = []
    if ut.NOT_QUIET:
        print('remember to inspect with --show --sel-rows (-r) and --sel-cols (-c) ')
        print('other options:')
        print('   --vf - view figure dir')
        print('   --va - view all')
        print('   --vh - view hard')
        print('   --ve - view easy')
        print('   --vn - view iNteresting')
        print('   --hs - hist sample')
        print('   --gv, --guiview - gui result inspection')
    if len(sel_rows) > 0 and len(sel_cols) == 0:
        sel_cols = list(range(len(cfg_list)))
    if len(sel_cols) > 0 and len(sel_rows) == 0:
        sel_rows = list(range(len(qaids)))
    if view_all:
        sel_rows = list(range(len(qaids)))
        sel_cols = list(range(len(cfg_list)))
    if view_hard:
        sel_rows.extend(np.array(new_hard_qx_list).tolist())
        sel_cols.extend(list(range(len(cfg_list))))
    if view_easy:
        new_easy_qx_list = np.setdiff1d(np.arange(len(qaids)), new_hard_qx_list).tolist()
        sel_rows.extend(new_easy_qx_list)
        sel_cols.extend(list(range(len(cfg_list))))
    if view_interesting:
        sel_rows.extend(interesting_qx_list)
        # TODO: grab the best scoring and most interesting configs
        if len(sel_cols) == 0:
            sel_cols.extend(list(range(len(cfg_list))))
    if kwargs.get('hist_sample', ut.get_argflag(('--hs', '--hist-sample'))):
        # Careful if there is more than one config
        config_rand_bin_qxs = test_result.get_rank_histogram_qx_sample(size=10)
        sel_rows = np.hstack(ut.flatten(config_rand_bin_qxs))
        # TODO: grab the best scoring and most interesting configs
        if len(sel_cols) == 0:
            sel_cols.extend(list(range(len(cfg_list))))
    sel_rows = ut.unique_keep_order2(sel_rows)
    sel_cols = ut.unique_keep_order2(sel_cols)
    sel_cols = list(sel_cols)
    sel_rows = list(sel_rows)
    return sel_rows, sel_cols


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
