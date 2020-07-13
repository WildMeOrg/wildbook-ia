# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def test_scoremech():
    import utool as ut
    import wbia
    from wbia.algo.hots import _pipeline_helpers as plh  # NOQA

    base = {'query_rotation_heuristic': False, 'sv_on': False}
    base = {'query_rotation_heuristic': True, 'sv_on': False, 'K': 1}
    base = {'query_rotation_heuristic': True, 'sv_on': False, 'K': 1}
    cfgdict1 = ut.dict_union(base, {'score_method': 'nsum', 'prescore_method': 'nsum'})
    cfgdict2 = ut.dict_union(base, {'score_method': 'csum', 'prescore_method': 'csum'})

    qaids = [13]
    daids = [
        1,
        5,
        11,
        19,
        22,
        27,
        31,
        36,
        39,
        42,
        43,
        46,
        50,
        53,
        55,
        58,
        64,
        68,
        73,
        79,
        81,
        84,
        85,
        89,
        95,
        98,
        99,
        105,
        108,
        111,
        114,
        119,
    ]

    ibs = wbia.opendb('PZ_MTEST')
    qreq1_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict1)
    qreq2_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict2)
    cm_list1 = qreq1_.execute()
    cm_list2 = qreq2_.execute()

    cm1, cm2 = cm_list1[0], cm_list2[0]

    ai1 = cm1.pandas_annot_info().set_index(['daid', 'dnid'], drop=True)
    ai2 = cm2.pandas_annot_info().set_index(['daid', 'dnid'], drop=True)
    ai1 = ai1.rename(columns={c: c + '1' for c in ai2.columns})
    ai2 = ai2.rename(columns={c: c + '2' for c in ai2.columns})
    # print(ai1.join(ai2))

    ni1 = cm1.pandas_name_info().set_index(['dnid'], drop=True)
    ni2 = cm2.pandas_name_info().set_index(['dnid'], drop=True)
    ni1 = ni1.rename(columns={c: c + '1' for c in ni2.columns})
    ni2 = ni2.rename(columns={c: c + '2' for c in ni2.columns})
    print(ni1.join(ni2))

    cm1 == cm2

    from wbia.algo.hots.chip_match import check_arrs_eq

    cm1.score_list == cm2.score_list
    cm1.name_score_list == cm2.name_score_list
    cm2.annot_score_list == cm2.annot_score_list
    assert check_arrs_eq(cm1.fm_list, cm2.fm_list)
    assert check_arrs_eq(cm1.fsv_list, cm2.fsv_list)

    # assert np.all(cm1.daid_list == cm2.daid_list)

    cm1.score_list
    cm2.score_list

    cm, qreq_ = cm1, qreq1_
    cm.evaluate_nsum_name_score(qreq_)
    cm.evaluate_maxcsum_name_score(qreq_)
    # fmech = cm.algo_name_scores['nsum']
    # amech = cm.algo_name_scores['maxcsum']
    if qreq_.qparams.K == 1 and qreq_.qparams.query_rotation_heuristic is False:
        import numpy as np

        assert np.all(cm.algo_name_scores['nsum'] == cm.algo_name_scores['maxcsum'])

    from wbia.algo.hots import name_scoring

    # name_scoring.compute_fmech_score(cm, qreq_=qreq_, hack_single_ori=False)
    name_scoring.compute_fmech_score(cm, qreq_=qreq_)

    # else:
    #     qreq1_, args1 = plh.testdata_pre('spatial_verification', 'PZ_MTEST',
    #                                      a=(qaids, daids), p=cfgdict1)
    #     qreq2_, args2 = plh.testdata_pre('spatial_verification', 'PZ_MTEST',
    #                                      a=(qaids, daids), p=cfgdict2)
    #     cm_list1 = args1.cm_list_FILT
    #     cm_list2 = args2.cm_list_FILT
