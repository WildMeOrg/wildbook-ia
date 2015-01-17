"""
development module storing my "development state"

TODO:
    * figure out what packages I use have lisencing issues.
        - Reimplement them or work around them.
"""
from __future__ import absolute_import, division, print_function
from ibeis.model.hots import hstypes
from uuid import UUID
import utool as ut
import copy
import six  # NOQA
import numpy as np  # NOQA
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[devcases]')


def myquery():
    r"""

    BUG::
        THERE IS A BUG SOMEWHERE: HOW IS THIS POSSIBLE?
        if everything is weightd ) how di the true positive even get a score
        while the true negative did not
        qres_copy.filtkey_list = ['ratio', 'fg', 'homogerr', 'distinctiveness']
        CORRECT STATS
        {
            'max'  : [0.832, 0.968, 0.604, 0.000],
            'min'  : [0.376, 0.524, 0.000, 0.000],
            'mean' : [0.561, 0.924, 0.217, 0.000],
            'std'  : [0.114, 0.072, 0.205, 0.000],
            'nMin' : [1, 1, 1, 51],
            'nMax' : [1, 1, 1, 1],
            'shape': (52, 4),
        }
        INCORRECT STATS
        {
            'max'  : [0.759, 0.963, 0.264, 0.000],
            'min'  : [0.379, 0.823, 0.000, 0.000],
            'mean' : [0.506, 0.915, 0.056, 0.000],
            'std'  : [0.125, 0.039, 0.078, 0.000],
            'nMin' : [1, 1, 1, 24],
            'nMax' : [1, 1, 1, 1],
            'shape': (26, 4),
        #   score_diff,  tp_score,  tn_score,       p,   K,  clip_fraction,  fg_power,  homogerr_power
             0.494,     0.494,     0.000,  73.000,   2,          0.500,     0.100,          10.000

    see how seperability changes as we very things

    CommandLine:
        python -m ibeis.model.hots.devcases --test-myquery
        python -m ibeis.model.hots.devcases --test-myquery --show --index 0
        python -m ibeis.model.hots.devcases --test-myquery --show --index 1
        python -m ibeis.model.hots.devcases --test-myquery --show --index 2

    References:
        http://en.wikipedia.org/wiki/Pareto_distribution <- look into

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.devcases import *  # NOQA
        >>> ut.dev_ipython_copypaster(myquery) if ut.inIPython() else myquery()
        >>> pt.show_if_requested()
    """
    from ibeis.model.hots import special_query  # NOQA
    from ibeis import viz  # NOQA
    import plottool as pt
    index = ut.get_argval('--index', int, 0)
    ibs, aid1, aid2, tn_aid = testdata_my_exmaples(index)
    qaids = [aid1]
    daids = [aid2] + [tn_aid]
    qvuuid = ibs.get_annot_visual_uuids(aid1)

    cfgdict_vsone = dict(
        sv_on=True,
        #sv_on=False,
        #codename='vsone_unnorm_dist_ratio_extern_distinctiveness',
        codename='vsone_unnorm_ratio_extern_distinctiveness',
        sver_weighting=True,
    )

    use_cache   = False
    save_qcache = False

    qres_list, qreq_ = ibs.query_chips(qaids, daids, cfgdict=cfgdict_vsone,
                                       return_request=True, use_cache=use_cache,
                                       save_qcache=save_qcache, verbose=True)

    qreq_.load_distinctiveness_normalizer()
    qres = qres_list[0]
    top_aids = qres.get_top_aids()  # NOQA
    qres_orig = qres  # NOQA

    def test_config(qreq_, qres_orig, cfgdict):
        """ function to grid search over """
        qres_copy = copy.deepcopy(qres_orig)
        qreq_vsone_ = qreq_
        qres_vsone = qres_copy
        filtkey = hstypes.FiltKeys.DISTINCTIVENESS
        newfsv_list, newscore_aids = special_query.get_extern_distinctiveness(qreq_, qres_copy, **cfgdict)
        special_query.apply_new_qres_filter_scores(qreq_vsone_, qres_vsone, newfsv_list, newscore_aids, filtkey)
        tp_score  = qres_copy.aid2_score[aid2]
        tn_score  = qres_copy.aid2_score[tn_aid]
        return qres_copy, tp_score, tn_score

    #[.01, .1, .2, .5, .6, .7, .8, .9, 1.0]),
    FiltKeys = hstypes.FiltKeys
    grid_basis = [
        #ut.DimensionBasis('p', np.linspace(1, 100.0, 50)),
        ut.DimensionBasis(
            'p',
            # higher seems better but effect flattens out
            # current_best = 73
            # This seems to imply that anything with a distinctivness less than
            # .9 is not relevant
            #[73.0]
            [.5, 1.0, 2]
            #[1, 20, 73]
        ),
        ut.DimensionBasis(
            # the score seems to significantly drop off when k>2
            # but then has a spike at k=8
            # best is k=2
            'K',
            [2]
            #[2, 3, 4, 5, 7, 8, 9, 16],
        ),
        #ut.DimensionBasis('clip_fraction', ),
        #ut.DimensionBasis('clip_fraction', np.linspace(.01, .11, 100)),
        ut.DimensionBasis(
            'clip_fraction',
            # THERE IS A VERY CLEAR SPIKE AT .09
            [.09],
            #[.09, 1.0],
            #np.linspace(.05, .15, 10),
        ),
        #ut.DimensionBasis(FiltKeys.FG + '_power', ),
        ut.DimensionBasis(
            FiltKeys.FG + '_power',
            # the forground power seems to be very influential in scoring
            # it seems higher is better but effect flattens out
            # the reason it seems to be better is because it zeros out weights
            [.1, 1.0, 2.0]
            #np.linspace(.01, 30.0, 10)
        ),
        ut.DimensionBasis(
            FiltKeys.HOMOGERR + '_power',
            # current_best = 2.5
            #[2.5]
            [.1, 1.0, 2.0]
            #np.linspace(.1, 10, 5)
            #np.linspace(.1, 10, 30)
        ),
    ]
    gridsearch = ut.GridSearch(grid_basis, label='qvuuid=%r' % (qvuuid,))
    print('Begin Grid Search')
    for cfgdict in ut.ProgressIter(gridsearch, lbl='GridSearch'):
        qres_copy, tp_score, tn_score = test_config(qreq_, qres_orig, cfgdict)
        gridsearch.append_result(tp_score, tn_score)
    print('Finish Grid Search')

    # Get best result
    best_cfgdict = gridsearch.get_rank_cfgdict()
    qres_copy, tp_score, tn_score = test_config(qreq_, qres_orig, best_cfgdict)

    # Examine closely what you can do with scores
    if False:
        qres_copy = copy.deepcopy(qres_orig)
        qreq_vsone_ = qreq_
        filtkey = hstypes.FiltKeys.DISTINCTIVENESS
        newfsv_list, newscore_aids = special_query.get_extern_distinctiveness(qreq_, qres_copy, **cfgdict)
        ut.embed()
        def make_new_chipmatch(qres_copy):
            assert ut.listfind(qres_copy.filtkey_list, filtkey) is None
            weight_filters = hstypes.WEIGHT_FILTERS
            weight_filtxs, nonweight_filtxs = special_query.index_partition(qres_copy.filtkey_list, weight_filters)

            aid2_fsv = {}
            aid2_fs = {}
            aid2_score = {}

            for new_fsv_vsone, daid in zip(newfsv_list, newscore_aids):
                pass
                break
                #scorex_vsone  = ut.listfind(qres_copy.filtkey_list, filtkey)
                #if scorex_vsone is None:
                # TODO: add spatial verification as a filter score
                # augment the vsone scores
                # TODO: paramaterize
                weighted_ave_score = True
                if weighted_ave_score:
                    # weighted average scoring
                    new_fs_vsone = special_query.weighted_average_scoring(new_fsv_vsone, weight_filtxs, nonweight_filtxs)
                else:
                    # product scoring
                    new_fs_vsone = special_query.product_scoring(new_fsv_vsone)
                new_score_vsone = new_fs_vsone.sum()
                aid2_fsv[daid]   = new_fsv_vsone
                aid2_fs[daid]    = new_fs_vsone
                aid2_score[daid] = new_score_vsone
            return aid2_fsv, aid2_fs, aid2_score

        # Look at plot of query products
        for new_fsv_vsone, daid in zip(newfsv_list, newscore_aids):
            new_fs_vsone = special_query.product_scoring(new_fsv_vsone)
            scores_list = np.array(new_fs_vsone)[:, None].T
            pt.plot_sorted_scores(scores_list, logscale=False, figtitle=str(daid))
        pt.iup()
        special_query.apply_new_qres_filter_scores(qreq_vsone_, qres_copy, newfsv_list, newscore_aids, filtkey)

    # PRINT INFO
    import functools
    #ut.rrrr()
    get_stats_str = functools.partial(ut.get_stats_str, axis=0, newlines=True, precision=3)
    tp_stats_str = ut.align(get_stats_str(qres_copy.aid2_fsv[aid2]), ':')
    tn_stats_str = ut.align(get_stats_str(qres_copy.aid2_fsv[tn_aid]), ':')
    info_str_list = []
    info_str_list.append('qres_copy.filtkey_list = %r' % (qres_copy.filtkey_list,))
    info_str_list.append('CORRECT STATS')
    info_str_list.append(tp_stats_str)
    info_str_list.append('INCORRECT STATS')
    info_str_list.append(tn_stats_str)
    info_str = '\n'.join(info_str_list)
    print(info_str)

    # SHOW BEST RESULT
    #qres_copy.ishow_top(ibs, fnum=pt.next_fnum())
    #qres_orig.ishow_top(ibs, fnum=pt.next_fnum())

    # Text Informatio
    param_lbl = 'p'
    param_stats_str = gridsearch.get_dimension_stats_str(param_lbl)
    print(param_stats_str)

    csvtext = gridsearch.get_csv_results(10)
    print(csvtext)

    # Paramter visuzliation
    fnum = pt.next_fnum()
    # plot paramter influence
    param_label_list = gridsearch.get_param_lbls()
    pnum_ = pt.get_pnum_func(2, len(param_label_list))
    for px, param_label in enumerate(param_label_list):
        gridsearch.plot_dimension(param_label, fnum=fnum, pnum=pnum_(px))
    # plot match figure
    pnum2_ = pt.get_pnum_func(2, 2)
    qres_copy.show_matches(ibs, aid2, fnum=fnum, pnum=pnum2_(2))
    qres_copy.show_matches(ibs, tn_aid, fnum=fnum, pnum=pnum2_(3))
    # Add figure labels
    figtitle = 'Effect of parameters on vsone separation for a single case'
    subtitle = 'qvuuid = %r' % (qvuuid)
    figtitle += '\n' + subtitle
    pt.set_figtitle(figtitle)
    # Save Figure
    #fig_fpath = pt.save_figure(usetitle=True)
    #print(fig_fpath)
    # Write CSV Results
    #csv_fpath = fig_fpath + '.csv.txt'
    #ut.write_to(csv_fpath, csvtext)

    #qres_copy.ishow_top(ibs)
    #from matplotlib import pyplot as plt
    #plt.show()
    #print(ut.list_str()))
    # TODO: plot max variation dims
    #import plottool as pt
    #pt.plot(p_list, diff_list)
    """
    viz.show_chip(ibs, aid1)
    import plottool as pt
    pt.update()
    """


def get_dev_test_fpaths(index):
    ibs, aid1, aid2, tn_aid = testdata_my_exmaples(index)
    fpath1, fpath2, fpath3 = ibs.get_annot_chip_fpaths([aid1, aid2, tn_aid])
    return fpath1, fpath2, fpath3


def testdata_my_exmaples(index):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.devcases import *  # NOQA
        >>> index = 1
    """
    import ibeis
    from uuid import UUID
    ibs = ibeis.opendb('GZ_ALL')
    vsone_pair_examples = [
        [UUID('8415b50f-2c98-0d52-77d6-04002ff4d6f8'), UUID('308fc664-7990-91ad-0576-d2e8ea3103d0')],
        [UUID('490f76bf-7616-54d5-576a-8fbc907e46ae'), UUID('2046509f-0a9f-1470-2b47-5ea59f803d4b')],
        [UUID('5cdf68ab-be49-ee3f-94d8-5483772c8618'), UUID('879977a7-b841-d223-dd91-761dfa58d486')],
    ]
    gf_mapping = {
        UUID('8415b50f-2c98-0d52-77d6-04002ff4d6f8'): [UUID('38211759-8fa7-875b-1f3e-39a630653f66')],
        UUID('490f76bf-7616-54d5-576a-8fbc907e46ae'): [UUID('58920d6e-31ba-307c-2ac8-e56aff2b2b9e')],  # other bad_aid is actually a good partial match
        UUID('5cdf68ab-be49-ee3f-94d8-5483772c8618'): [UUID('5a8c8ad7-873a-e6ed-98df-56a452e0a93e')],
    }

    #ibs.get_annot_visual_uuids([36, 3])

    vuuid_pair = vsone_pair_examples[index]
    vuuid1, vuuid2 = vuuid_pair
    aid1, aid2 = ibs.get_annot_aids_from_visual_uuid(vuuid_pair)
    assert aid1 is not None
    assert aid2 is not None
    #daids = ibs.get_valid_aids()

    tn_vuuid = gf_mapping.get(vuuid1)
    if tn_vuuid is None:
        qaids = [aid1]
        find_close_incorrect_match(ibs, qaids)
        print('baste the result in gf_mapping')
        return

    tn_aids = ibs.get_annot_aids_from_visual_uuid(tn_vuuid)
    tn_aid = tn_aids[0]
    return ibs, aid1, aid2, tn_aid


def find_close_incorrect_match(ibs, qaids):
    use_cache = False
    save_qcache = False
    cfgdict_vsmany = dict(index_method='single',
                          pipeline_root='vsmany',)
    qres_vsmany_list, qreq_vsmany_ = ibs.query_chips(
        qaids, ibs.get_valid_aids(), cfgdict=cfgdict_vsmany,
        return_request=True, use_cache=use_cache, save_qcache=save_qcache,
        verbose=True)
    qres_vsmany = qres_vsmany_list[0]
    qres_vsmany.ishow_top(ibs)
    top_aids = qres_vsmany.get_top_aids()
    top_nids = ibs.get_annot_nids(top_aids)
    qaid = qaids[0]
    qnid = ibs.get_annot_nids(qaid)
    is_groundfalse = [nid != qnid for nid in top_nids]
    top_gf_aids = ut.filter_items(top_aids, is_groundfalse)
    #top_gt_aids = ut.filterfalse_items(top_aids, is_groundfalse)
    top_gf_vuuids = ibs.get_annot_visual_uuids(top_gf_aids)
    qvuuid = ibs.get_annot_visual_uuids(qaid)
    gf_mapping = {qvuuid: top_gf_vuuids[0:1]}
    print('gf_mapping = ' + ut.dict_str(gf_mapping))
    pass


def show_power_law_plots():
    """

    CommandLine:
        python -m ibeis.model.hots.devcases --test-show_power_law_plots --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> #%pylab qt4
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.devcases import *  # NOQA
        >>> show_power_law_plots()
        >>> pt.show_if_requested()
    """
    import numpy as np
    import plottool as pt
    xdata = np.linspace(0, 1, 1000)
    ydata = xdata
    fnum = 1
    powers = [.01, .1, .5, 1, 2, 30, 70, 100, 1000]
    nRows, nCols = pt.get_square_row_cols(len(powers), fix=True)
    pnum_next = pt.make_pnum_nextgen(nRows, nCols)
    for p in powers:
        plotkw = dict(
            fnum=fnum,
            marker='g-',
            linewidth=2,
            pnum=pnum_next(),
            title='p=%r' % (p,)
        )
        ydata_ = ydata ** p
        pt.plot2(xdata, ydata_, **plotkw)
    pt.set_figtitle('power laws y = x ** p')


def get_gzall_small_test():
    """
    ibs.get_annot_visual_uuids([qaid, aid])
    """
    #aid_list = [839, 999, 1047, 209, 307, 620, 454, 453, 70, 1015, 939, 1021,
    #              306, 742, 1010, 802, 619, 1041, 27, 420, 740, 1016, 140, 992,
    #              1043, 662, 816, 793, 994, 867, 534, 986, 783, 858, 937, 60,
    #              879, 1044, 528, 459, 639]
    debug_examples = [
        UUID('308fc664-7990-91ad-0576-d2e8ea3103d0'),
    ]
    #vsone_pair_examples
    debug_examples

    ignore_vuuids = [
        UUID('be6fe4d6-ae87-0f8f-269f-e9f706b69e41'),  # OUT OF PLANE
        UUID('c3394b28-e7f2-2da6-1a49-335b748acf9e'),  # HUGE OUT OF PLANE, foal (vsmany gets rank3)
        UUID('490f76bf-7616-54d5-576a-8fbc907e46ae'),
        UUID('2046509f-0a9f-1470-2b47-5ea59f803d4b'),
    ]
    vuuid_list = [
        UUID('e9a9544d-083d-6c30-b00f-d6806824a21a'),
        UUID('153306d8-e9f8-b5a6-a06d-90ddb7de6c17'),
        UUID('04908b6f-b775-46f1-e9ec-fd834d9fe046'),
        UUID('20817244-12d1-bcf4-dac8-b787c064e6b4'),
        UUID('7ad8b4fc-f057-bac9-0b1a-fa05db09b685'),
        UUID('df418005-ce26-e439-ab43-00f56447a3c8'),
        UUID('2ca8c5b0-ae45-a1a2-11fb-3497dfd58736'),
        UUID('7c6fd123-ad9c-7360-8b7c-b8d694ac4057'),
        UUID('98e87153-4437-562b-9f77-d7c58495cfea'),
        UUID('7bb9b77a-7ad5-44f3-4352-bf6bf901323a'),
        UUID('863786d9-853d-859d-f726-13796fc0a257'),
        UUID('9c1b04bc-7af1-bd3f-85d3-7c06c8b0d0a7'),
        UUID('c0775a6d-f3a9-f1d2-1a92-cfc4817ecedf'),
        UUID('52668d79-8065-bae4-29ca-0f393e8b0331'),
        UUID('86bf60e5-20a8-0e8d-590c-836ac4723d23'),
        UUID('2046509f-0a9f-1470-2b47-5ea59f803d4b'),
        UUID('a0444615-1264-8768-8a4e-fbc2cafb76ce'),
        UUID('308fc664-7990-91ad-0576-d2e8ea3103d0'),  # testcase
        UUID('4bd156a3-0315-72fd-d181-309b92f21d58'),
        UUID('04815be5-fee7-f34d-e2cd-6130914e2071'),
        UUID('815a8276-8812-35a3-d1e5-963c2047edc5'),
        UUID('2d94da8d-0d97-815d-d350-bf3ab1caaf23'),
        UUID('9732e459-4c73-c8d5-3911-59c6e66d81f8'),
        UUID('38e39dda-bae3-ce19-f7d3-a50fc1c3554d'),
        UUID('1509cad7-e368-6d95-9047-552d054ddabd'),
        UUID('2091fa5b-bf9d-25ba-b539-9156202dd522'),
        UUID('fc94609f-b378-9877-d0ac-433993e7f6bd'),
        UUID('914f1c91-c22b-a4b5-77f8-59423bc6d99d'),
        UUID('249f7615-95e2-ea66-649f-ec8021e5aa41'),
        UUID('2de71fb1-dd7c-a7de-2e0d-0be399286d09'),
        UUID('94010938-cb14-c209-5488-5372b81d1eb1'),
        UUID('d75a9205-efc4-a078-a533-bdde5345b74a'),
        UUID('99a1e02a-0e3d-cd9a-b410-902f5d8cf308'),
        UUID('193ade79-eff2-f888-7f15-27a399c505b0'),
        UUID('190bf50d-9729-48c3-47b6-acefc6f3ef03'),
        UUID('5345c6dc-bc52-43ec-d792-0e7c9e7ec3b5'),
        UUID('10757fe8-8fd3-ad59-f550-c941da967b82'),
        UUID('89859efb-a233-5e43-fb5e-c36e9d446a1e'),
        UUID('265cf095-64f6-e5dd-8f7d-2a82f627b7d1'),
        UUID('4b19968e-f813-f238-0dcc-6a54f1943d57')]
    return vuuid_list, ignore_vuuids


def load_gztest(ibs):
    r"""
    CommandLine:
        python -m ibeis.model.hots.special_query --test-load_gztest

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.devcases import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('GZ_ALL')
    """
    from os.path import join
    from ibeis.model.hots import match_chips4 as mc4
    dir_ = ut.get_module_dir(mc4)
    eval_text = ut.read_from(join(dir_,  'GZ_TESTTUP.txt'))
    testcases = eval(eval_text)
    count_dict = ut.count_dict_vals(testcases)
    print(ut.dict_str(count_dict))

    testtup_list = ut.flatten(ut.dict_take_list(testcases, ['vsone_wins',
                                                            'vsmany_outperformed',
                                                            'vsmany_dominates',
                                                            'vsmany_wins']))
    qaid_list = [testtup.qaid_t for testtup in testtup_list]
    visual_uuids = ibs.get_annot_visual_uuids(qaid_list)
    visual_uuids


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.devcases
        python -m ibeis.model.hots.devcases --allexamples
        python -m ibeis.model.hots.devcases --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
