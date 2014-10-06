#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import six
import ibeis
import pandas as pd
from ibeis.model.hots.smk import smk_index
from ibeis.model.hots.smk import smk_match
from ibeis.model.hots.smk import pandas_helpers as pdh
from ibeis.model.hots import query_request
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_debug]')


def testdata_printops(**kwargs):
    """ test print options. doesnt take up too much screen
    """
    print('[smk_debug] testdata_printops')
    np.set_printoptions(precision=4)
    pd.set_option('display.max_rows', 7)
    pd.set_option('display.max_columns', 7)
    pd.set_option('isplay.notebook_repr_html', True)


def testdata_ibeis(**kwargs):
    """ builds ibs for testing

    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
        >>> kwargs = {}
    """
    print(' === Test Data IBEIS ===')
    from ibeis.model.hots.smk import smk_debug
    smk_debug.testdata_printops(**kwargs)
    print('[smk_debug] testdata_ibeis')
    ibeis.ensure_pz_mtest()
    ibs = ibeis.opendb('PZ_MTEST')
    ibs._default_config()
    #aggregate = False
    aggregate = kwargs.get('aggregate', utool.get_argflag(('--agg', '--aggregate')))
    #aggregate = not kwargs.get('aggregate', utool.get_argflag(('--noagg', '--noaggregate')))
    nWords = utool.get_argval(('--nWords', '--nCentroids'), int, default=8E3)
    nAssign = utool.get_argval(('--nAssign', '--K'), int, default=10)
    # Configs
    ibs.cfg.query_cfg.pipeline_root = 'smk'
    ibs.cfg.query_cfg.smk_cfg.aggregate = aggregate
    ibs.cfg.query_cfg.smk_cfg.nWords = nWords
    ibs.cfg.query_cfg.smk_cfg.alpha = 3
    ibs.cfg.query_cfg.smk_cfg.thresh = 0
    ibs.cfg.query_cfg.smk_cfg.nAssign = nAssign
    return ibs


def testdata_ibeis2(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    print('[smk_debug] testdata_ibeis2')
    ibs = smk_debug.testdata_ibeis(**kwargs)
    valid_aids = ibs.get_valid_aids()
    # Training/Database/Search set
    taids = valid_aids[:]
    daids  = valid_aids
    #daids = valid_aids[1:10]
    #daids = valid_aids[0:3]
    #qaids = valid_aids[0::2]
    #qaids = valid_aids[0:2]
    #qaids = [37]  # NOQA new test case for PZ_MTEST
    #qaids = [valid_aids[0], valid_aids[4]]
    qaids = [valid_aids[0]]
    return ibs, taids, daids, qaids


def testdata_dataframe(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    ibs, taids, daids, qaids = smk_debug.testdata_ibeis2(**kwargs)
    print('[smk_debug] testdata_dataframe')
    # Pandas Annotation Dataframe
    annots_df = smk_index.make_annot_df(ibs)
    nWords = ibs.cfg.query_cfg.smk_cfg.nWords
    return ibs, annots_df, taids, daids, qaids, nWords


def testdata_words(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, taids, daids, qaids, nWords = smk_debug.testdata_dataframe(**kwargs)
    print('[smk_debug] testdata_words')
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    return ibs, annots_df, daids, qaids, words


def testdata_raw_internals0():
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, words = smk_debug.testdata_words()
    print('[smk_debug] testdata_raw_internals0')
    with_internals = False
    invindex = smk_index.index_data_annots(annots_df, daids, words, with_internals)
    return ibs, annots_df, daids, qaids, invindex


def testdata_raw_internals1():
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_raw_internals0()
    print('[smk_debug] testdata_raw_internals1')
    #ibs.cfg.query_cfg.smk_cfg.printme3()
    words  = invindex.words
    wordflann = invindex.wordflann
    idx2_vec  = invindex.idx2_dvec
    nAssign = ibs.cfg.query_cfg.smk_cfg.nAssign
    _dbargs = (wordflann, words, idx2_vec, nAssign)
    (wx2_idxs, wx2_maws, idx2_wxs) = smk_index.assign_to_words_(*_dbargs)
    invindex.wx2_idxs = wx2_idxs
    invindex.wx2_maws = wx2_maws
    invindex.idx2_wxs = idx2_wxs
    return ibs, annots_df, daids, qaids, invindex, wx2_idxs


def testdata_raw_internals2():
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, wx2_idxs = smk_debug.testdata_raw_internals1()
    print('[smk_debug] testdata_raw_internals2')
    #ibs.cfg.query_cfg.smk_cfg.printme3()
    words     = invindex.words
    wx_series = np.arange(len(words))  # .index
    idx2_aid  = invindex.idx2_daid
    idx2_vec  = invindex.idx2_dvec
    idx2_fx  = invindex.idx2_dfx
    wx2_maws  = invindex.wx2_maws
    aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    wx2_idf = smk_index.compute_word_idf_(
        wx_series, wx2_idxs, idx2_aid, daids)
    wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws = smk_index.compute_residuals_(
        words, wx2_idxs, wx2_maws, idx2_vec, idx2_aid, idx2_fx, aggregate)
    invindex.wx2_maws  = wx2_maws
    return ibs, annots_df, invindex, wx2_idxs, wx2_idf, wx2_rvecs, wx2_aids


def testdata_query_repr():
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    from ibeis.model.hots.smk import smk_index
    ibs, annots_df, daids, qaids, invindex, wx2_idxs = smk_debug.testdata_raw_internals1()
    print('[smk_debug] testdata_query_repr')
    words     = invindex.words
    wx_series = np.arange(len(words))  # .index
    idx2_aid  = invindex.idx2_daid
    wx2_idf = smk_index.compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
    qaid = qaids[0]
    #qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    invindex.wx2_idf = wx2_idf
    return ibs, annots_df, qaid, invindex


def testsdata_sccw_sum():
    from ibeis.model.hots.smk import smk_debug
    from ibeis.model.hots.smk import smk_index

    ibs, annots_df, qaid, invindex = smk_debug.testdata_query_repr()
    aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    alpha     = ibs.cfg.query_cfg.smk_cfg.alpha
    thresh    = ibs.cfg.query_cfg.smk_cfg.thresh
    nAssign   = ibs.cfg.query_cfg.smk_cfg.nAssign
    wx2_idf   = invindex.wx2_idf
    words     = invindex.words
    wordflann = invindex.wordflann
    qfx2_vec  = annots_df['vecs'][qaid].values
    # Assign query to (multiple) words
    _wx2_qfxs, wx2_maws, qfx2_wxs = smk_index.assign_to_words_(
        wordflann, words, qfx2_vec, nAssign=nAssign)
    # Hack to make implementing asmk easier, very redundant
    qfx2_aid = np.array([qaid] * len(qfx2_wxs), dtype=smk_index.INTEGER_TYPE)
    qfx2_qfx = np.arange(len(qfx2_vec))
    # Compute query residuals
    wx2_qrvecs, wx2_qaids, wx2_qfxs, wx2_maws = smk_index.compute_residuals_(
        words, _wx2_qfxs, wx2_maws, qfx2_vec, qfx2_aid, qfx2_qfx, aggregate)
    # Compute query sccw
    if utool.VERBOSE:
        print('[smk_index] Query TF alpha=%r, thresh=%r' % (alpha, thresh))
    wx_sublist  = np.array(wx2_qrvecs.keys(), dtype=smk_index.INDEX_TYPE)
    idf_list    = [wx2_idf[wx]    for wx in wx_sublist]
    rvecs_list  = [wx2_qrvecs[wx] for wx in wx_sublist]
    maws_list   = [wx2_maws[wx]   for wx in wx_sublist]
    return idf_list, rvecs_list, maws_list, alpha, thresh


def testdata_internals(**kwargs):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    from ibeis.model.hots.smk import smk_index
    ibs, annots_df, daids, qaids, words = smk_debug.testdata_words(**kwargs)
    print('[smk_debug] testdata_internals')
    with_internals = True
    aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    #if True:
    #    _args1 = (annots_df, daids, words, with_internals, True, alpha, thresh)
    #    _args2 = (annots_df, daids, words, with_internals, False, alpha, thresh)
    #    invindex1 = smk_index.index_data_annots(*_args1)
    #    invindex2 = smk_index.index_data_annots(*_args2)
    #    #utool.flatten(invindex1.wx2_aids.values.tolist())
    #    invindex = invindex1
    #    smk_debug.invindex_dbgstr(invindex)
    #    invindex = invindex2
    #    smk_debug.invindex_dbgstr(invindex)
    #else:
    _args = (annots_df, daids, words, with_internals, aggregate, alpha, thresh)
    invindex = smk_index.index_data_annots(*_args)
    return ibs, annots_df, daids, qaids, invindex


def testdata_match_kernel(**kwargs):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    from ibeis.model.hots.smk import smk_index
    ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals(**kwargs)
    print('[smk_debug] testdata_match_kernel')
    qaid = qaids[0]
    aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    nAssign = ibs.cfg.query_cfg.smk_cfg.nAssign
    print('+------------')
    print('[smk_debug] aggregate = %r' % (aggregate,))
    print('[smk_debug] alpha = %r' % (alpha,))
    print('[smk_debug] thresh = %r' % (thresh,))
    print('L------------')
    qindex = smk_index.new_qindex(annots_df, qaid, invindex, aggregate, alpha,
                                  thresh, nAssign)
    #qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    return ibs, invindex, qindex


def testdata_nonagg_rvec():
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, wx2_idxs = smk_debug.testdata_raw_internals1()
    words     = invindex.words
    idx2_vec  = invindex.idx2_dvec
    wx2_maws  = invindex.wx2_maws
    idx2_daid  = invindex.idx2_daid
    wx_sublist = np.array(list(wx2_idxs.keys()))
    idxs_list  = [wx2_idxs[wx].astype(np.int32) for wx in wx_sublist]
    maws_list  = [wx2_maws[wx] for wx in wx_sublist]
    aids_list  = [idx2_daid.take(idxs) for idxs in idxs_list]
    return words, wx_sublist, aids_list, idxs_list, idx2_vec, maws_list


def check_invindex_wx2(invindex):
    words = invindex.words
    #wx2_idf = invindex.wx2_idf
    wx2_rvecs = invindex.wx2_drvecs
    #wx2_idxs   = invindex.wx2_idxs
    wx2_aids   = invindex.wx2_aids  # needed for asmk
    wx2_fxs    = invindex.wx2_fxs   # needed for asmk
    check_wx2(words, wx2_rvecs, wx2_aids, wx2_fxs)


def wx_len_stats(wx2_xxx):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, taids, daids, qaids, nWords = smk_debug.testdata_dataframe()
        >>> invindex = index_data_annots(annots_df, daids, words)
        >>> qaid = qaids[0]
        >>> wx2_qrvecs, wx2_qaids, wx2_qfxs, query_sccw = new_qindex(annots_df, qaid, invindex)
        >>> print(utool.dict_str(wx2_rvecs_stats(wx2_qrvecs)))
    """
    import utool
    if wx2_xxx is None:
        return 'None'
    if isinstance(wx2_xxx, dict):
        #len_list = [len(xxx) for xxx in ]
        val_list = wx2_xxx.values()
    else:
        val_list = wx2_xxx
    try:
        len_list = [len(xxx) for xxx in val_list]
        statdict = utool.get_stats(len_list)
        return utool.dict_str(statdict, strvals=True, newlines=False)
    except Exception as ex:
        utool.printex(ex)
        for count, xxx in wx2_xxx:
            try:
                len(xxx)
            except Exception:
                print('failed on count=%r' % (count,))
                print('failed on xxx=%r' % (xxx,))
                pass
        raise


def check_wx2(words=None, wx2_rvecs=None, wx2_aids=None, wx2_fxs=None):
    """ provides debug info for mappings from word indexes to values
    """
    if words is None:
        nWords = max(wx2_rvecs.keys()) + 1
    else:
        nWords = len(words)

    print('[smk_debug] checking wx2 for %d words' % (nWords))
    def missing_word(wx2_xxx, wx=None):
        return (wx2_xxx is not None) and (wx not in wx2_xxx)

    def missing_word_or_None(wx2_xxx, wx=None):
        return (wx2_xxx is None) or (wx not in wx2_xxx)

    def same_size_or_None(wx2_xxx1, wx2_xxx2, wx=None):
        if (wx2_xxx1 is None or wx2_xxx2 is None):
            return True
        if missing_word(wx2_xxx1, wx) and missing_word(wx2_xxx2, wx):
            return True
        return len(wx2_xxx1[wx]) == len(wx2_xxx2[wx])

    nMissing = 0
    for wx in range(nWords):
        if (missing_word(wx2_fxs, wx) or missing_word(wx2_aids, wx) or missing_word(wx2_rvecs, wx)):
            assert missing_word_or_None(wx2_aids, wx), 'in one but not others'
            assert missing_word_or_None(wx2_rvecs, wx), 'in one but not others'
            assert missing_word_or_None(wx2_fxs, wx), 'in one but not others'
            nMissing += 1
        assert same_size_or_None(wx2_aids, wx2_rvecs, wx=None)
        assert same_size_or_None(wx2_aids, wx2_fxs, wx=None)
        assert same_size_or_None(wx2_rvecs, wx2_fxs, wx=None)

    print('[smk_debug] %d words had 0 members' % nMissing)
    print(' lenstats(wx2_rvecs) = ' + wx_len_stats(wx2_rvecs))
    print(' lenstats(wx2_aids)  = ' + wx_len_stats(wx2_aids))
    print(' lenstats(wx2_fxs)   = ' + wx_len_stats(wx2_fxs))


def check_wx2_rvecs(wx2_rvecs, verbose=True):
    flag = True
    for wx, rvecs in six.iteritems(wx2_rvecs):
        shape = rvecs.shape
        if shape[0] == 0:
            print('word[wx={wx}] has no rvecs'.format(wx=wx))
            flag = False
        if np.any(np.isnan(rvecs)):
            #rvecs[:] = 1 / np.sqrt(128)
            print('word[wx={wx}] has nans'.format(wx=wx))
            flag = False
    if verbose:
        if flag:
            print('check_wx2_rvecs passed')
        else:
            print('check_wx2_rvecs failed')
    return flag


def check_wx2_rvecs2(invindex, wx2_rvecs=None, wx2_idxs=None, idx2_vec=None, verbose=True):
    words = invindex.words
    if wx2_rvecs is None:
        if verbose:
            print('[smk_debug] check_wx2_rvecs2 inverted index')
        wx2_rvecs = invindex.wx2_drvecs
        wx2_idxs = invindex.wx2_idxs
        idx2_vec = invindex.idx2_dvec
    else:
        if verbose:
            print('[smk_debug] check_wx2_rvecs2 queryrepr index')
    flag = True
    nan_wxs = []
    no_wxs = []
    for wx, rvecs in six.iteritems(wx2_rvecs):
        shape = rvecs.shape
        if shape[0] == 0:
            #print('word[wx={wx}] has no rvecs'.format(wx=wx))
            no_wxs.append(wx)
        for sx in range(shape[0]):
            if np.any(np.isnan(pdh.ensure_values(rvecs)[sx])):
                #rvecs[:] = 1 / np.sqrt(128)
                #print('word[wx={wx}][sx={sx}] has nans'.format(wx=wx))
                nan_wxs.append((wx, sx))
    if verbose:
        print('[smk_debug] %d words had no residuals' % len(no_wxs))
        print('[smk_debug] %d words have nans' % len(nan_wxs))
    failed_wx = []
    for count, (wx, sx) in enumerate(nan_wxs):
        rvec = pdh.ensure_values(wx2_rvecs[wx])[sx]
        idxs = wx2_idxs[wx][sx]
        dvec = pdh.ensure_values(idx2_vec)[idxs]
        word = pdh.ensure_values(words)[wx]
        truth = (word == dvec)
        if not np.all(truth):
            failed_wx.append(wx)
            if verbose:
                print('+=====================')
                print('Bad RVEC #%d' % count)
                print('[smk_debug] wx=%r, sx=%r was nan and not equal to its word' % (wx, sx))
                print('[smk_debug] rvec=%r ' % (rvec,))
                print('[smk_debug] dvec=%r ' % (dvec,))
                print('[smk_debug] word=%r ' % (word,))
                print('[smk_debug] truth=%r ' % (truth,))
            flag = False
    if len(failed_wx) == 0:
        if verbose:
            print('[smk_debug] all nan rvecs were equal to their words')
    return flag


def check_invindex(invindex, verbose=True):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, taids, daids, qaids, nWords = smk_debug.testdata_dataframe()
        >>> words = learn_visual_words(annots_df, taids, nWords)
        >>> invindex = index_data_annots(annots_df, daids, words)
    """
    daids = invindex.daids
    daid2_sccw = invindex.daid2_sccw
    check_daid2_sccw(daid2_sccw, verbose=verbose)
    assert daid2_sccw.shape[0] == daids.shape[0]
    if verbose:
        print('each aid has a sccw')


def check_daid2_sccw(daid2_sccw, verbose=True):
    daid2_sccw_values = pdh.ensure_values(daid2_sccw)
    assert not np.any(np.isnan(daid2_sccw_values)), 'sccws are nan'
    if verbose:
        print('database sccws are not nan')
        print('database sccw stats:')
        print(utool.get_stats_str(daid2_sccw_values, newlines=True))


def test_sccw_cache():
    ibs, annots_df, taids, daids, qaids, nWords = testdata_dataframe()
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    with_internals = True
    invindex = smk_index.index_data_annots(annots_df, daids, words, with_internals)
    idx2_daid  = invindex.idx2_daid
    wx2_drvecs = invindex.wx2_drvecs
    wx2_idf = invindex.wx2_idf
    daids      = invindex.daids
    daid2_sccw1 = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs,
                                                       wx2_idf, daids,
                                                       use_cache=True)
    daid2_sccw2 = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs,
                                                       wx2_idf, daids,
                                                       use_cache=False)
    daid2_sccw3 = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs,
                                                       wx2_idf, daids,
                                                       use_cache=True)
    check_daid2_sccw(daid2_sccw1)
    check_daid2_sccw(daid2_sccw2)
    check_daid2_sccw(daid2_sccw3)
    if not np.all(daid2_sccw2 == daid2_sccw3):
        raise AssertionError('caching error in sccw')
    if not np.all(daid2_sccw1 == daid2_sccw2):
        raise AssertionError('cache outdated in sccw')


def check_dtype(annots_df):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
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


def check_rvecs_list_eq(rvecs_list, rvecs_list2):
    """
    Example:
        >>> rvecs_list = smk_speed.compute_nonagg_rvec_listcomp(*_args1)  # 125 ms
        >>> rvecs_list2 = smk_speed.compute_nonagg_residuals_forloop(*_args1)
    """
    assert len(rvecs_list) == len(rvecs_list2)
    for rvecs, rvecs2 in zip(rvecs_list, rvecs_list2):
        try:
            assert len(rvecs) == len(rvecs2)
            assert rvecs.shape == rvecs2.shape
            #assert np.all(rvecs == rvecs2)
            np.testing.assert_equal(rvecs, rvecs2, verbose=True)
        except AssertionError:
            utool.print_keys([rvecs, rvecs2])
            raise


def display_info(ibs, invindex, annots_df):
    from vtool import clustering2 as clustertool
    ################
    from ibeis.dev import dbinfo
    print(ibs.get_infostr())
    dbinfo.get_dbinfo(ibs, verbose=True)
    ################
    print('Inverted Index Stats: vectors per word')
    print(utool.get_stats_str(map(len, invindex.wx2_idxs.values())))
    ################
    #qfx2_vec     = annots_df['vecs'][1]
    centroids    = invindex.words
    num_pca_dims = 2  # 3
    whiten       = False
    kwd = dict(num_pca_dims=num_pca_dims,
               whiten=whiten,)
    #clustertool.rrr()
    def makeplot_(fnum, prefix, data, labels='centroids', centroids=centroids):
        return clustertool.plot_centroids(data, centroids, labels=labels,
                                          fnum=fnum, prefix=prefix + '\n', **kwd)
    makeplot_(1, 'centroid vecs', centroids)
    #makeplot_(2, 'database vecs', invindex.idx2_dvec)
    #makeplot_(3, 'query vecs', qfx2_vec)
    #makeplot_(4, 'database vecs', invindex.idx2_dvec)
    #makeplot_(5, 'query vecs', qfx2_vec)
    #################


def check_daid2_chipmatch(daid2_chipmatch, verbose=True):
    ## Concatenate into full fmfsfk reprs
    #def concat_chipmatch(cmtup):
    #    fm_list = [_[0] for _ in cmtup]
    #    fs_list = [_[1] for _ in cmtup]
    #    fk_list = [_[2] for _ in cmtup]
    #    assert len(fm_list) == len(fs_list)
    #    assert len(fk_list) == len(fs_list)
    #    chipmatch = (np.vstack(fm_list), np.hstack(fs_list), np.hstack(fk_list))
    #    assert len(chipmatch[0]) == len(chipmatch[1])
    #    assert len(chipmatch[2]) == len(chipmatch[1])
    #    return chipmatch
    ##daid2_chipmatch = {}
    ##for daid, cmtup in six.iteritems(daid2_chipmatch_):
    ##    daid2_chipmatch[daid] = concat_chipmatch(cmtup)
    print('[smk_debug] checking %d chipmatches' % len(daid2_chipmatch))
    featmatches = 0
    daid2_fm, daid2_fs, daid2_fk = daid2_chipmatch
    for daid in six.iterkeys(daid2_fm):
        chipmatch = (daid2_fm[daid], daid2_fs[daid], daid2_fk[daid])
        try:
            assert len(chipmatch) == 3, (
                'chipmatch = %r' % (chipmatch.shape,))
            (fm, fs, fk) = chipmatch
            featmatches += len(fm)
            assert len(fm) == len(fs), (
                'fm.shape = %r, fs.shape=%r' % (fm.shape, fs.shape))
            assert len(fk) == len(fs), (
                'fk.shape = %r, fs.shape=%r' % (fk.shape, fs.shape))
            assert fm.shape[1] == 2
        except AssertionError as ex:
            utool.printex(ex, keys=[
                'daid',
                'chipmatch',
            ])
            raise
    print('[smk_debug] checked %d featmatches in %d chipmatches' % (featmatches, len(daid2_chipmatch)))


def dictinfo(dict_):
    keys = list(dict_.keys())
    vals = list(dict_.values())
    num_keys  = len(keys)
    key_types = list(set(map(type, keys)))
    val_types = list(set(map(type, vals)))

    fmtstr_ = '\n' + utool.unindent('''
    * num_keys  = {num_keys}
    * key_types = {key_types}
    * val_types = {val_types}
    '''.strip('\n'))

    if len(val_types) == 1:
        if val_types[0] == np.ndarray:
            val_shape_stats = utool.get_stats(set(map(np.shape, vals)), axis=0)
            val_shape_stats_str = utool.dict_str(val_shape_stats, strvals=True, newlines=False)
            val_dtypes = set([val.dtype for val in vals])
            fmtstr_ += utool.unindent('''
            * val_shape_stats = {val_shape_stats_str}
            * val_dtypes = {val_dtypes}
            '''.strip('\n'))
        elif val_types[0] == list:
            val_len_stats =  utool.get_stats(set(map(len, vals)))
            val_len_stats_str = utool.dict_str(val_len_stats, strvals=True, newlines=False)
            depth = utool.list_depth(vals)
            val_types = set(utool.list_deep_types(vals))
            fmtstr_ += utool.unindent('''
            * list_depth = {depth}
            * val_len_stats = {val_len_stats_str}
            * deep_types = {val_types}
            '''.strip('\n'))

    fmtstr = fmtstr_.format(**locals())
    return utool.indent(fmtstr)


def invindex_dbgstr(invindex):
    """
    >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    >>> ibs, annots_df, daids, qaids, invindex = testdata_raw_internals0()
    >>> invindex_dbgstr(invindex)
    """
    print('+--- INVINDEX DBGSTR ---')
    locals_ = {'invindex': invindex}
    #print(dictinfo(invindex.wx2_fxs))

    key_list = [
        'invindex.words.shape',
        'invindex.words.dtype',
        'invindex.daids.dtype',
        'invindex.idx2_dvec.shape',
        'invindex.idx2_dvec.dtype',
        'invindex.idx2_daid.shape',
        'invindex.idx2_daid.dtype',
        'invindex.idx2_dfx.shape',
        (dictinfo, 'invindex.daid2_sccw'),
        (dictinfo, 'invindex.wx2_drvecs'),
        (dictinfo, 'invindex.wx2_idf'),
        (dictinfo, 'invindex.wx2_aids'),
        (dictinfo, 'invindex.wx2_fxs'),
        (dictinfo, 'invindex.wx2_idxs'),
    ]
    keystr_list = utool.parse_locals_keylist(locals_, key_list)
    append = keystr_list.append
    def stats_(arr):
        return wx_len_stats(arr)

    append('lenstats(invindex.wx2_idxs) = ' + stats_(invindex.wx2_idxs))
    #append('lenstats(invindex.wx2_idf) = ' + stats_(invindex.wx2_idf))
    append('lenstats(invindex.wx2_drvecs) = ' + stats_(invindex.wx2_drvecs))
    append('lenstats(invindex.wx2_aids) = ' + stats_(invindex.wx2_aids))
    def mapval(func, dict_):
        if isinstance(dict_, dict):
            return map(func, six.itervalues(dict_))
        else:
            return map(func, dict_)
    def isunique(aids):
        return len(set(aids)) == len(aids)

    if invindex.wx2_aids is not None:
        wx_series = list(invindex.wx2_aids.keys())
        aids_list = list(invindex.wx2_aids.values())
        nAids_list = map(len, aids_list)
        invindex.wx2_aids
        append('sum(mapval(len, invindex.wx2_aids))) = ' + str(sum(nAids_list)))
        probably_asmk = all(mapval(isunique, invindex.wx2_aids))
        if probably_asmk:
            append('All wx2_aids are unique. aggregate probably is True')
        else:
            append('Some wx2_aids are duplicates. aggregate probably is False')
        maxkey = wx_series[np.array(nAids_list).argmax()]
        print('wx2_aids[maxkey=%r] = \n' % (maxkey,) + str(invindex.wx2_aids[maxkey]))
    dbgstr = '\n'.join(keystr_list)
    print(dbgstr)
    print('L--- END INVINDEX DBGSTR ---')


def query_smk_test(annots_df, invindex, qreq_):
    """
    ibeis interface
    Example:
        >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
        >>> from ibeis.model.hots import query_request  # NOQA
        >>> from ibeis.model.hots.smk import smk_match  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals()
        >>> qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
        >>> qaid2_qres_ = smk_match.query_smk(annots_df, invindex, qreq_)

    Dev::
        qres = qaid2_qres_[qaids[0]]
        fig = qres.show_top(ibs)

    """
    from ibeis.model.hots import pipeline
    from ibeis.model.hots.smk import smk_match  # NOQA
    qaids = qreq_.get_external_qaids()
    qaid2_chipmatch = {}
    qaid2_scores    = {}
    aggregate = qreq_.qparams.aggregate
    alpha     = qreq_.qparams.alpha
    thresh    = qreq_.qparams.thresh
    lbl = '[smk_match] asmk query: ' if aggregate else '[smk_match] smk query: '
    mark, end_ = utool.log_progress(lbl, len(qaids), flushfreq=1,
                                    writefreq=1, with_totaltime=True,
                                    backspace=False)
    withinfo = True
    for count, qaid in enumerate(qaids):
        mark(count)
        daid2_score, daid2_chipmatch = smk_match.query_inverted_index(
            annots_df, qaid, invindex, withinfo, aggregate, alpha, thresh)
        qaid2_scores[qaid]    = daid2_score
        qaid2_chipmatch[qaid] = daid2_chipmatch
    end_()
    try:
        filt2_meta = {}
        qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, filt2_meta, qreq_)
    except Exception as ex:
        utool.printex(ex)
        utool.qflag()
        raise
    return qaid2_qres_


def main():
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    print('+------------')
    print('SMK_DEBUG MAIN')
    print('+------------')
    from ibeis.model.hots import pipeline
    ibs, annots_df, taids, daids, qaids, nWords = testdata_dataframe()
    # Query using SMK
    #qaid = qaids[0]
    qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    nWords    = qreq_.qparams.nWords
    aggregate = qreq_.qparams.aggregate
    alpha     = qreq_.qparams.alpha
    thresh    = qreq_.qparams.thresh
    nAssign   = qreq_.qparams.nAssign
    #aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    #alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    #thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    print('+------------')
    print('SMK_DEBUG PARAMS')
    print('[smk_debug] aggregate = %r' % (aggregate,))
    print('[smk_debug] alpha   = %r' % (alpha,))
    print('[smk_debug] thresh  = %r' % (thresh,))
    print('[smk_debug] nWords  = %r' % (nWords,))
    print('[smk_debug] nAssign = %r' % (nAssign,))
    print('L------------')
    # Learn vocabulary
    #words = qreq_.words = smk_index.learn_visual_words(annots_df, taids, nWords)
    # Index a database of annotations
    #qreq_.invindex = smk_index.index_data_annots(annots_df, daids, words, aggregate, alpha, thresh)
    qreq_.ibs = ibs
    # Smk Mach
    print('+------------')
    print('SMK_DEBUG MATCH KERNEL')
    print('+------------')
    qaid2_scores, qaid2_chipmatch_SMK = smk_match.selective_match_kernel(qreq_)
    SVER = utool.get_argflag('--sver')
    if SVER:
        print('+------------')
        print('SMK_DEBUG SVER? YES!')
        print('+------------')
        qaid2_chipmatch_SVER_ = pipeline.spatial_verification(qaid2_chipmatch_SMK, qreq_)
        qaid2_chipmatch = qaid2_chipmatch_SVER_
    else:
        print('+------------')
        print('SMK_DEBUG SVER? NO')
        print('+------------')
        qaid2_chipmatch = qaid2_chipmatch_SMK
    print('+------------')
    print('SMK_DEBUG DISPLAY RESULT')
    print('+------------')
    filt2_meta = {}
    qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, filt2_meta, qreq_)
    for count, (qaid, qres) in enumerate(six.iteritems(qaid2_qres_)):
        print('+================')
        #qres = qaid2_qres_[qaid]
        qres.show_top(ibs, fnum=count)
        for aid in qres.aid2_score.keys():
            smkscore = qaid2_scores[qaid][aid]
            sumscore = qres.aid2_score[aid]
            if not utool.almost_eq(smkscore, sumscore):
                print('scorediff aid=%r, smkscore=%r, sumscore=%r' % (aid, smkscore, sumscore))

        scores = qaid2_scores[qaid]
        #print(scores)
        print(qres.get_inspect_str(ibs))
        print('L================')
        #utool.embed()
    #print(qres.aid2_fs)
    #daid2_totalscore, chipmatch = smk_index.query_inverted_index(annots_df, qaid, invindex)
    ## Pack into QueryResult
    #qaid2_chipmatch = {qaid: chipmatch}
    #qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, {}, qreq_)
    ## Show match
    #daid2_totalscore.sort(axis=1, ascending=False)
    #print(daid2_totalscore)

    #daid2_totalscore2, chipmatch = query_inverted_index(annots_df, daids[0], invindex)
    #print(daid2_totalscore2)
    #display_info(ibs, invindex, annots_df)
    print('finished main')
    return locals()


if __name__ == '__main__':
    print('\n\n\n\n\n\n')
    import multiprocessing
    from plottool import draw_func2 as df2
    np.set_printoptions(precision=2)
    pd.set_option('display.max_rows', 7)
    pd.set_option('display.max_columns', 7)
    pd.set_option('isplay.notebook_repr_html', True)
    multiprocessing.freeze_support()  # for win32
    main_locals = main()
    main_execstr = utool.execstr_dict(main_locals, 'main_locals')
    exec(main_execstr)
    exec(df2.present())
