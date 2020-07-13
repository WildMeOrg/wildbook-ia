# -*- coding: utf-8 -*-
"""
development module storing my "development state"

TODO:
    * figure out what packages I use have lisencing issues.
        - Reimplement them or work around them.

    Excplitict Negative Matches between chips
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from wbia.algo.hots import hstypes
from uuid import UUID
import utool as ut
import copy
import numpy as np

print, rrr, profile = ut.inject2(__name__)


def fix_pz_master():
    r"""
    CommandLine:
        python -m wbia.algo.hots.devcases --test-fix_pz_master --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.devcases import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = fix_pz_master()
        >>> # verify results
        >>> print(result)
    """
    # ReROI aid4513+name
    """
    82 97 117 118 157 213 299 336 351 368 392 415 430 434 441 629 664 679 682
    685 695 835 1309 1317 1327 1382 1455 1816 3550 4067 4125 4131 4141 4191 4239
    4242 4256 4257 4258 4439 4489 4625 4797 4800 4809 4812 4813 4819 4829 4831
    4864 4865 4871 4877 4879 4885 4886 4891 4897 4899 4902 4904 4909 4912 4931
    4933 4934 4935 4936 4952 5053 5073 5248 5249 5367 5776 5999 6150 6699 6882
    7197 7225 7231 7237 7247 7254 7263 7280 7298 7357 7373 7385 7412 7424 7426
    7464 7470 7472 7473 7479 7480 7507 7533 7535 7537 7578 7589 7629 7666 7669
    7672 7720 7722 7734 7740 7754 7778 7792 7796 7798 7807 7812 7813 7829 7840
    7846 7875 7876 7888 7889 7896 7899 7900 7901 7908 7909 7911 7915 7916 7917
    7925 7931 7936 7938 7944 7947 7951 7954 7961 7965 7966 7978 7979 7981 7988
    7992 7998 8019 8044 8045 8047 8051 8052 8059 8062 8063 8064 8066 8069 8074
    8075 8083 8088 8093 8094 8095 8100 8102 8103 8105 8113 8116 8118 8119 8120
    8123 8126 8128 8139 8144 8151 8152 8153 8154 8155 8157 8165 8170 8178 8180
    8186 8188 8189 8195 8197 8198 8201 8202 8206 8213 8216 8226 8228 8231 8238
    8245 8258 8265 8269 8276 8281 8285 8287 8297 8301 8305 8306 8308 8312 8318
    8319 8321 8329 8332 8345 8349 8357 8361 8365 8367 8372 8373 8381 8386 8389
    8392 8398 8399 8400 8402 8403 8406 8407 8412 8423 8424 8426 8427 8428 8429
    8431 8439 8444 8446 8447 8450 8456 8457 8461 8463 8464 8466 8471 8472 8477
    8481 8486 8489 8490 8494 8497 8499 8500 8501 8503 8506 8508 8535 8536 8537
    8538 8539 8540 8541 8544 8545 8550 8552 8554 8555 8557 8558 8559 8564 8567
    8568 8574 8575 8582 8584 8587 8589 8591 8592 8593 8596 8597 8601 8602 8605
    8607 8608 8616 8617 8618 8619 8620 8621 8622 8629 8637 8639 8647 8662 8664
    8665 8666 8673 8674 8676 8689 8691 8692 8693 8694 8699 8700 8702 8703 8712
    8714 8715 8719 8724 8733 8734 8736
    """

    qaids_str = """
    82 117 118 213 299 336 351 368 392 430 434 441 495 616 629 664 679 682 685 695 835 915 1317 1382 3550 4239 4242 4246 4256 4257 4258 4439 4445 4447 4489 4706 4800 4812 4813 4819 4828 4829 4831 4864 4865 4871 4877 4879 4885 4886 4891 4897 4899 4902 4904 4909 4912 4933 4934 4936 4952 5053 5073 5248 5367 5776 7197 7225 7254 7263 7298 7309 7470 7473 7479 7507 7535 7570 7589 7666 7672 7722 7734 7740 7754 7760 7796 7798 7807 7813 7829 7840 7875 7876 7888 7889 7896 7899 7900 7901 7908 7909 7911 7916 7917 7925 7931 7934 7936 7938 7944 7947 7951 7954 7961 7964 7965 7966 7978 7979 7981 7988 7992 7998 8019 8044 8045 8047 8051 8052 8058 8059 8062 8063 8064 8066 8074 8075 8083 8088 8094 8095 8100 8101 8102 8103 8105 8111 8113 8116 8119 8120 8121 8123 8126 8128 8144 8151 8152 8153 8155 8156 8157 8165 8170 8180 8186 8188 8198 8201 8206 8213 8216 8226 8228 8231 8238 8258 8276 8281 8285 8287 8295 8297 8301 8305 8306 8308 8312 8318 8319 8329 8332 8355 8357 8361 8365 8367 8372 8373 8381 8386 8388 8389 8392 8398 8399 8402 8403 8406 8407 8412 8424 8425 8426 8428 8429 8439 8442 8444 8446 8447 8449 8450 8452 8456 8457 8461 8463 8464 8466 8470 8471 8481 8486 8489 8490 8494 8497 8499 8500 8501 8503 8506 8508 8535 8536 8537 8538 8539 8540 8544 8545 8550 8554 8555 8557 8558 8559 8563 8564 8567 8574 8575 8582 8584 8587 8589 8592 8593 8596 8597 8600 8601 8604 8605 8607 8608 8616 8617 8618 8619 8620 8621 8622 8623 8629 8637 8639 8647 8662 8665 8666 8673 8674 8676 8691 8693 8694 8699 8700 8702 8703 8712 8714 8715 8719 8724 8731 8733 8734 8736
    """

    qaids_ = map(int, filter(len, qaids_str.replace('\n', ' ').split(' ')))

    import wbia

    wbia._preload()
    from wbia.gui import inspect_gui
    from wbia import guitool

    ibs = wbia.opendb('PZ_Master0')
    daids = ibs.get_valid_aids(minqual='poor')
    qaids = ibs.filter_junk_annotations(qaids_)

    # qaids = qaids[64:128]
    qreq_ = ibs.new_query_request(qaids, daids)
    # qreq_.lazy_load()
    qres_list = ibs.query_chips(qreq_=qreq_, verbose=True)

    qres_wgt = inspect_gui.launch_review_matches_interface(
        ibs, qres_list, dodraw=ut.show_was_requested()
    )

    if ut.show_was_requested():
        guitool.guitool_main.qtapp_loop()
    return qres_wgt


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
        #   score_diff,  tp_score,  tn_score,       p,   K,  dcvs_clip_max,  fg_power,  homogerr_power
             0.494,     0.494,     0.000,  73.000,   2,          0.500,     0.100,          10.000

    see how seperability changes as we very things

    CommandLine:
        python -m wbia.algo.hots.devcases --test-myquery
        python -m wbia.algo.hots.devcases --test-myquery --show --index 0
        python -m wbia.algo.hots.devcases --test-myquery --show --index 1
        python -m wbia.algo.hots.devcases --test-myquery --show --index 2

    References:
        http://en.wikipedia.org/wiki/Pareto_distribution <- look into

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.devcases import *  # NOQA
        >>> ut.dev_ipython_copypaster(myquery) if ut.inIPython() else myquery()
        >>> pt.show_if_requested()
    """
    from wbia.algo.hots import special_query  # NOQA
    from wbia.algo.hots import distinctiveness_normalizer  # NOQA
    from wbia import viz  # NOQA
    import wbia.plottool as pt

    index = ut.get_argval('--index', int, 0)
    ibs, aid1, aid2, tn_aid = testdata_my_exmaples(index)
    qaids = [aid1]
    daids = [aid2] + [tn_aid]
    qvuuid = ibs.get_annot_visual_uuids(aid1)

    cfgdict_vsone = dict(
        sv_on=True,
        # sv_on=False,
        # codename='vsone_unnorm_dist_ratio_extern_distinctiveness',
        codename='vsone_unnorm_ratio_extern_distinctiveness',
        sver_output_weighting=True,
    )

    use_cache = False
    save_qcache = False

    qres_list, qreq_ = ibs.query_chips(
        qaids,
        daids,
        cfgdict=cfgdict_vsone,
        return_request=True,
        use_cache=use_cache,
        save_qcache=save_qcache,
        verbose=True,
    )

    qreq_.load_distinctiveness_normalizer()
    qres = qres_list[0]
    top_aids = qres.get_top_aids()  # NOQA
    qres_orig = qres  # NOQA

    def try_config(qreq_, qres_orig, cfgdict):
        """ function to grid search over """
        qres_copy = copy.deepcopy(qres_orig)
        qreq_vsone_ = qreq_
        qres_vsone = qres_copy
        filtkey = hstypes.FiltKeys.DISTINCTIVENESS
        newfsv_list, newscore_aids = special_query.get_extern_distinctiveness(
            qreq_, qres_copy, **cfgdict
        )
        special_query.apply_new_qres_filter_scores(
            qreq_vsone_, qres_vsone, newfsv_list, newscore_aids, filtkey
        )
        tp_score = qres_copy.aid2_score[aid2]
        tn_score = qres_copy.aid2_score[tn_aid]
        return qres_copy, tp_score, tn_score

    # [.01, .1, .2, .5, .6, .7, .8, .9, 1.0]),
    # FiltKeys = hstypes.FiltKeys
    # FIXME: Use other way of doing gridsearch
    grid_basis = distinctiveness_normalizer.DCVS_DEFAULT.get_grid_basis()
    gridsearch = ut.GridSearch(grid_basis, label='qvuuid=%r' % (qvuuid,))
    print('Begin Grid Search')
    for cfgdict in ut.ProgressIter(gridsearch, lbl='GridSearch'):
        qres_copy, tp_score, tn_score = try_config(qreq_, qres_orig, cfgdict)
        gridsearch.append_result(tp_score, tn_score)
    print('Finish Grid Search')

    # Get best result
    best_cfgdict = gridsearch.get_rank_cfgdict()
    qres_copy, tp_score, tn_score = try_config(qreq_, qres_orig, best_cfgdict)

    # Examine closely what you can do with scores
    if False:
        qres_copy = copy.deepcopy(qres_orig)
        qreq_vsone_ = qreq_
        filtkey = hstypes.FiltKeys.DISTINCTIVENESS
        newfsv_list, newscore_aids = special_query.get_extern_distinctiveness(
            qreq_, qres_copy, **cfgdict
        )
        ut.embed()

        def make_cm_very_old_tuple(qres_copy):
            assert ut.listfind(qres_copy.filtkey_list, filtkey) is None
            weight_filters = hstypes.WEIGHT_FILTERS
            weight_filtxs, nonweight_filtxs = special_query.index_partition(
                qres_copy.filtkey_list, weight_filters
            )

            aid2_fsv = {}
            aid2_fs = {}
            aid2_score = {}

            for new_fsv_vsone, daid in zip(newfsv_list, newscore_aids):
                pass
                break
                # scorex_vsone  = ut.listfind(qres_copy.filtkey_list, filtkey)
                # if scorex_vsone is None:
                # TODO: add spatial verification as a filter score
                # augment the vsone scores
                # TODO: paramaterize
                weighted_ave_score = True
                if weighted_ave_score:
                    # weighted average scoring
                    new_fs_vsone = special_query.weighted_average_scoring(
                        new_fsv_vsone, weight_filtxs, nonweight_filtxs
                    )
                else:
                    # product scoring
                    new_fs_vsone = special_query.product_scoring(new_fsv_vsone)
                new_score_vsone = new_fs_vsone.sum()
                aid2_fsv[daid] = new_fsv_vsone
                aid2_fs[daid] = new_fs_vsone
                aid2_score[daid] = new_score_vsone
            return aid2_fsv, aid2_fs, aid2_score

        # Look at plot of query products
        for new_fsv_vsone, daid in zip(newfsv_list, newscore_aids):
            new_fs_vsone = special_query.product_scoring(new_fsv_vsone)
            scores_list = np.array(new_fs_vsone)[:, None].T
            pt.plot_sorted_scores(scores_list, logscale=False, figtitle=str(daid))
        pt.iup()
        special_query.apply_new_qres_filter_scores(
            qreq_vsone_, qres_copy, newfsv_list, newscore_aids, filtkey
        )

    # PRINT INFO
    import functools

    # ut.rrrr()
    get_stats_str = functools.partial(
        ut.get_stats_str, axis=0, newlines=True, precision=3
    )
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
    # qres_copy.ishow_top(ibs, fnum=pt.next_fnum())
    # qres_orig.ishow_top(ibs, fnum=pt.next_fnum())

    # Text Informatio
    param_lbl = 'dcvs_power'
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
    # fig_fpath = pt.save_figure(usetitle=True)
    # print(fig_fpath)
    # Write CSV Results
    # csv_fpath = fig_fpath + '.csv.txt'
    # ut.write_to(csv_fpath, csvtext)

    # qres_copy.ishow_top(ibs)
    # from matplotlib import pyplot as plt
    # plt.show()
    # print(ut.repr2()))
    # TODO: plot max variation dims
    # import wbia.plottool as pt
    # pt.plot(p_list, diff_list)
    """
    viz.show_chip(ibs, aid1)
    import wbia.plottool as pt
    pt.update()
    """


def get_dev_test_fpaths(index):
    ibs, aid1, aid2, tn_aid = testdata_my_exmaples(index)
    fpath1, fpath2, fpath3 = ibs.get_annot_chip_fpath([aid1, aid2, tn_aid])
    return fpath1, fpath2, fpath3


def testdata_my_exmaples(index):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.devcases import *  # NOQA
        >>> index = 1
    """
    import wbia
    from uuid import UUID

    ibs = wbia.opendb('GZ_ALL')
    vsone_pair_examples = [
        [
            UUID('8415b50f-2c98-0d52-77d6-04002ff4d6f8'),
            UUID('308fc664-7990-91ad-0576-d2e8ea3103d0'),
        ],
        [
            UUID('490f76bf-7616-54d5-576a-8fbc907e46ae'),
            UUID('2046509f-0a9f-1470-2b47-5ea59f803d4b'),
        ],
        [
            UUID('5cdf68ab-be49-ee3f-94d8-5483772c8618'),
            UUID('879977a7-b841-d223-dd91-761dfa58d486'),
        ],
    ]
    gf_mapping = {
        UUID('8415b50f-2c98-0d52-77d6-04002ff4d6f8'): [
            UUID('38211759-8fa7-875b-1f3e-39a630653f66')
        ],
        UUID('490f76bf-7616-54d5-576a-8fbc907e46ae'): [
            UUID('58920d6e-31ba-307c-2ac8-e56aff2b2b9e')
        ],  # other bad_aid is actually a good partial match
        UUID('5cdf68ab-be49-ee3f-94d8-5483772c8618'): [
            UUID('5a8c8ad7-873a-e6ed-98df-56a452e0a93e')
        ],
    }

    # ibs.get_annot_visual_uuids([36, 3])

    vuuid_pair = vsone_pair_examples[index]
    vuuid1, vuuid2 = vuuid_pair
    aid1, aid2 = ibs.get_annot_aids_from_visual_uuid(vuuid_pair)
    assert aid1 is not None
    assert aid2 is not None
    # daids = ibs.get_valid_aids()

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
    cfgdict_vsmany = dict(index_method='single', pipeline_root='vsmany',)
    qres_vsmany_list, qreq_vsmany_ = ibs.query_chips(
        qaids,
        ibs.get_valid_aids(),
        cfgdict=cfgdict_vsmany,
        return_request=True,
        use_cache=use_cache,
        save_qcache=save_qcache,
        verbose=True,
    )
    qres_vsmany = qres_vsmany_list[0]
    qres_vsmany.ishow_top(ibs)
    top_aids = qres_vsmany.get_top_aids()
    top_nids = ibs.get_annot_nids(top_aids)
    qaid = qaids[0]
    qnid = ibs.get_annot_nids(qaid)
    is_groundfalse = [nid != qnid for nid in top_nids]
    top_gf_aids = ut.compress(top_aids, is_groundfalse)
    # top_gt_aids = ut.filterfalse_items(top_aids, is_groundfalse)
    top_gf_vuuids = ibs.get_annot_visual_uuids(top_gf_aids)
    qvuuid = ibs.get_annot_visual_uuids(qaid)
    gf_mapping = {qvuuid: top_gf_vuuids[0:1]}
    print('gf_mapping = ' + ut.repr2(gf_mapping))
    pass


def show_power_law_plots():
    """

    CommandLine:
        python -m wbia.algo.hots.devcases --test-show_power_law_plots --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> #%pylab qt4
        >>> from wbia.algo.hots.devcases import *  # NOQA
        >>> show_power_law_plots()
        >>> pt.show_if_requested()
    """
    import numpy as np
    import wbia.plottool as pt

    xdata = np.linspace(0, 1, 1000)
    ydata = xdata
    fnum = 1
    powers = [0.01, 0.1, 0.5, 1, 2, 30, 70, 100, 1000]
    nRows, nCols = pt.get_square_row_cols(len(powers), fix=True)
    pnum_next = pt.make_pnum_nextgen(nRows, nCols)
    for p in powers:
        plotkw = dict(
            fnum=fnum, marker='g-', linewidth=2, pnum=pnum_next(), title='p=%r' % (p,)
        )
        ydata_ = ydata ** p
        pt.plot2(xdata, ydata_, **plotkw)
    pt.set_figtitle('power laws y = x ** p')


def get_gzall_small_test():
    """
    ibs.get_annot_visual_uuids([qaid, aid])
    """
    # aid_list = [839, 999, 1047, 209, 307, 620, 454, 453, 70, 1015, 939, 1021,
    #              306, 742, 1010, 802, 619, 1041, 27, 420, 740, 1016, 140, 992,
    #              1043, 662, 816, 793, 994, 867, 534, 986, 783, 858, 937, 60,
    #              879, 1044, 528, 459, 639]
    debug_examples = [
        UUID('308fc664-7990-91ad-0576-d2e8ea3103d0'),
    ]
    # vsone_pair_examples
    debug_examples

    ignore_vuuids = [
        UUID('be6fe4d6-ae87-0f8f-269f-e9f706b69e41'),  # OUT OF PLANE
        UUID(
            'c3394b28-e7f2-2da6-1a49-335b748acf9e'
        ),  # HUGE OUT OF PLANE, foal (vsmany gets rank3)
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
        UUID('4b19968e-f813-f238-0dcc-6a54f1943d57'),
    ]
    return vuuid_list, ignore_vuuids


def get_pz_master_testcase():
    aid_uuid_list = [
        (
            7944,
            UUID('b315d75f-a54f-5abf-18e5-7e353c113876'),
            'small chip area.  fgweights should not be dialated here',
        )(8490, UUID('316571aa-f675-ea1a-2674-0cb9a0f00426'), 'had corrupted chip')
    ]
    aid_uuid_list


def load_gztest(ibs):
    r"""
    CommandLine:
        python -m wbia.algo.hots.special_query --test-load_gztest

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.devcases import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('GZ_ALL')
    """
    from os.path import join
    from wbia.algo.hots import match_chips4 as mc4

    dir_ = ut.get_module_dir(mc4)
    eval_text = ut.read_from(join(dir_, 'GZ_TESTTUP.txt'))
    testcases = eval(eval_text)
    count_dict = ut.count_dict_vals(testcases)
    print(ut.repr2(count_dict))

    testtup_list = ut.flatten(
        ut.dict_take_list(
            testcases,
            ['vsone_wins', 'vsmany_outperformed', 'vsmany_dominates', 'vsmany_wins'],
        )
    )
    qaid_list = [testtup.qaid_t for testtup in testtup_list]
    visual_uuids = ibs.get_annot_visual_uuids(qaid_list)
    visual_uuids


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.devcases
        python -m wbia.algo.hots.devcases --allexamples
        python -m wbia.algo.hots.devcases --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
