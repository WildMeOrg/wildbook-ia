from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import six
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_debug]')


def check_wx2_rvecs(wx2_rvecs, verbose=True):
    flag = True
    for wx, rvecs in six.iteritems(wx2_rvecs):
        shape = rvecs.shape
        if shape[0] == 0:
            print('word[wx={wx}] has no rvecs')
            flag = False
        if np.any(np.isnan(rvecs)):
            print('word[wx={wx}] has nans')
            flag = False
    if verbose:
        if flag:
            print('check_wx2_rvecs passed')
        else:
            print('check_wx2_rvecs failed')
    return flag


def check_invindex(invindex, verbose=True):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    """
    daids = invindex.daids
    daid2_gamma = invindex.daid2_gamma
    check_daid2_gamma(daid2_gamma, verbose=verbose)
    assert daid2_gamma.shape[0] == daids.shape[0]
    if verbose:
        print('each aid has a gamma')


def check_daid2_gamma(daid2_gamma, verbose=True):
    assert not np.any(np.isnan(daid2_gamma)), 'gammas are nan'
    if verbose:
        print('database gammas are not nan')
        print('database gamma stats:')
        print(utool.common_stats(daid2_gamma, newlines=True))


def wx2_rvecs_stats(wx2_rvecs):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_rvecs = compute_query_repr(annots_df, qaid, invindex)
    >>> print(utool.dict_str(wx2_rvecs_stats(wx2_qrvecs)))
    """
    stats_ = utool.mystats(map(len, wx2_rvecs))
    return stats_


def test_gamma_cache():
    from ibeis.model.hots import smk_debug
    from ibeis.model.hots import smk_index
    from ibeis.model.hots import smk
    ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    with_internals = True
    invindex = smk_index.index_data_annots(annots_df, daids, words, with_internals)
    idx2_daid  = invindex.idx2_daid
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight
    daids      = invindex.daids
    daid2_gamma1 = smk_index.compute_data_gamma_(idx2_daid, wx2_drvecs,
                                                 wx2_weight, daids,
                                                 use_cache=True)
    daid2_gamma2 = smk_index.compute_data_gamma_(idx2_daid, wx2_drvecs,
                                                 wx2_weight, daids,
                                                 use_cache=False)
    daid2_gamma3 = smk_index.compute_data_gamma_(idx2_daid, wx2_drvecs,
                                                 wx2_weight, daids,
                                                 use_cache=True)
    smk_debug.check_daid2_gamma(daid2_gamma1)
    smk_debug.check_daid2_gamma(daid2_gamma2)
    smk_debug.check_daid2_gamma(daid2_gamma3)
    if not np.all(daid2_gamma2 == daid2_gamma3):
        raise AssertionError('caching error in gamma')
    if not np.all(daid2_gamma1 == daid2_gamma2):
        raise AssertionError('cache outdated in gamma')



def check_dtype(annots_df):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> import ibeis
    >>> ibs = ibeis.opendb('PZ_MTEST')
    >>> annots_df = make_annot_df(ibs)
    """

    #utool.printex(Exception('check'), keys=[
    #    'annots_df.index'
    #]
    #)
    vecs = annots_df['vecs']
    kpts = annots_df['kpts']
    locals_ = locals()
    key_list = [
        'annots_df.index.dtype',
        'annots_df.columns.dtype',
        'annots_df.columns',
        'vecs.index.dtype',
        'kpts.index.dtype',
        #'vecs',
        #'kpts',
    ]
    utool.print_keys(key_list)
