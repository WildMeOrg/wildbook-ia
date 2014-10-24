#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import six
import ibeis
from ibeis.model.hots import hstypes
from ibeis.model.hots.smk import smk_index
from ibeis.model.hots.smk import smk_repr
from ibeis.model.hots.smk import smk_match
from ibeis.model.hots.smk import smk_scoring
from ibeis.model.hots import query_request
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_debug]')


def testdata_printops(**kwargs):
    """ test print options. doesnt take up too much screen
    """
    print('[smk_debug] testdata_printops')
    np.set_printoptions(precision=4)


def testdata_ibeis(**kwargs):
    """
    Step 1

    builds ibs for testing

    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
        >>> kwargs = {}
    """
    print(' === Test Data IBEIS ===')
    from ibeis.model.hots.smk import smk_debug
    smk_debug.testdata_printops(**kwargs)
    print('kwargs = ' + utool.dict_str(kwargs))
    print('[smk_debug] testdata_ibeis')
    get_argflag = utool.get_argflag
    get_argval = utool.get_argval
    db = kwargs.get('db', get_argval('--db', str, 'PZ_MTEST'))
    if db == 'PZ_MTEST':
        ibeis.ensure_pz_mtest()
    ibs = ibeis.opendb()
    ibs._default_config()
    aggregate = kwargs.get('aggregate', get_argflag(('--agg', '--aggregate')))
    nWords    = kwargs.get(   'nWords',  get_argval(('--nWords', '--nCentroids'), int, default=8E3))
    nAssign   = kwargs.get(  'nAssign',  get_argval(('--nAssign', '--K'), int, default=10))
    # Configs
    ibs.cfg.query_cfg.pipeline_root = 'smk'
    ibs.cfg.query_cfg.smk_cfg.aggregate = aggregate
    ibs.cfg.query_cfg.smk_cfg.nWords = nWords
    ibs.cfg.query_cfg.smk_cfg.smk_alpha = 3
    ibs.cfg.query_cfg.smk_cfg.smk_thresh = 0
    ibs.cfg.query_cfg.smk_cfg.nAssign = nAssign
    ibs.cfg.query_cfg.smk_cfg.printme3()
    return ibs


def testdata_ibeis2(**kwargs):
    """
    Step 2

    selects training and test set

    """
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
    qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    qreq_.ibs = ibs  # Hack
    return ibs, taids, daids, qaids, qreq_


def testdata_dataframe(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    ibs, taids, daids, qaids, qreq_ = smk_debug.testdata_ibeis2(**kwargs)
    print('[smk_debug] testdata_dataframe')
    # Pandas Annotation Dataframe
    annots_df = smk_repr.make_annot_df(ibs)
    nWords = qreq_.qparams.nWords
    return ibs, annots_df, taids, daids, qaids, qreq_, nWords


def testdata_words(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, taids, daids, qaids, qreq_, nWords = smk_debug.testdata_dataframe(**kwargs)
    print('[smk_debug] testdata_words')
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    return ibs, annots_df, daids, qaids, qreq_, words


def testdata_raw_internals0(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, qreq_, words = smk_debug.testdata_words(**kwargs)
    qparams = qreq_.qparams
    print('[smk_debug] testdata_raw_internals0')
    with_internals = False
    invindex = smk_repr.index_data_annots(annots_df, daids, words, qparams, with_internals)
    return ibs, annots_df, daids, qaids, invindex, qreq_


def testdata_raw_internals1(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_raw_internals0(**kwargs)
    qparams = qreq_.qparams
    print('[smk_debug] testdata_raw_internals1')
    #ibs.cfg.query_cfg.smk_cfg.printme3()
    words  = invindex.words
    wordflann = invindex.wordflann
    idx2_vec  = invindex.idx2_dvec
    nAssign = 1  # 1 for database
    massign_sigma = qparams.massign_sigma
    massign_alpha = qparams.massign_alpha
    massign_equal_weights = qparams.massign_equal_weights
    # TODO: Extract args from function via inspect
    _dbargs = (wordflann, words, idx2_vec, nAssign, massign_alpha,
               massign_sigma, massign_equal_weights)
    (wx2_idxs, wx2_maws, idx2_wxs) = smk_index.assign_to_words_(*_dbargs)
    invindex.wx2_idxs = wx2_idxs
    invindex.wx2_maws = wx2_maws
    invindex.idx2_wxs = idx2_wxs
    return ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams


def testdata_raw_internals1_5(**kwargs):
    """
    contains internal data up to idf weights

    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1(**kwargs)
    print('[smk_debug] testdata_raw_internals1_5')
    #ibs.cfg.query_cfg.smk_cfg.printme3()
    words     = invindex.words
    wx_series = np.arange(len(words))
    idx2_aid  = invindex.idx2_daid
    wx2_idf = smk_index.compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
    invindex.wx2_idf = wx2_idf
    return ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams


def testdata_raw_internals2(**kwargs):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1_5(**kwargs)
    print('[smk_debug] testdata_raw_internals2')
    aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    idx2_vec  = invindex.idx2_dvec
    idx2_fx   = invindex.idx2_dfx
    wx2_maws  = invindex.wx2_maws
    idx2_aid  = invindex.idx2_daid
    words     = invindex.words
    wx2_idf   = invindex.wx2_idf
    wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws, wx2_dflags = smk_index.compute_residuals_(
        words, wx2_idxs, wx2_maws, idx2_vec, idx2_aid, idx2_fx, aggregate)
    invindex.wx2_maws  = wx2_maws
    invindex.wx2_dflags = wx2_dflags
    return ibs, annots_df, invindex, wx2_idxs, wx2_idf, wx2_rvecs, wx2_aids, qparams


def testdata_query_repr(**kwargs):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1_5(**kwargs)
    print('[smk_debug] testdata_query_repr')
    qaid = qaids[0]
    #qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    return ibs, annots_df, qaid, invindex


def testdata_sccw_sum(**kwargs):
    from ibeis.model.hots.smk import smk_debug
    from ibeis.model.hots.smk import smk_index

    ibs, annots_df, qaid, invindex = smk_debug.testdata_query_repr(**kwargs)
    aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    smk_alpha     = ibs.cfg.query_cfg.smk_cfg.smk_alpha
    smk_thresh    = ibs.cfg.query_cfg.smk_cfg.smk_thresh

    nAssign =  ibs.cfg.query_cfg.smk_cfg.nAssign
    massign_sigma = ibs.cfg.query_cfg.smk_cfg.massign_sigma
    massign_alpha = ibs.cfg.query_cfg.smk_cfg.massign_alpha
    massign_equal_weights = ibs.cfg.query_cfg.smk_cfg.massign_equal_weights
    nAssign   = ibs.cfg.query_cfg.smk_cfg.nAssign
    wx2_idf   = invindex.wx2_idf
    words     = invindex.words
    wordflann = invindex.wordflann
    #qfx2_vec  = annots_df['vecs'][qaid].values
    qfx2_vec  = annots_df['vecs'][qaid]
    # Assign query to (multiple) words
    _wx2_qfxs, wx2_maws, qfx2_wxs = smk_index.assign_to_words_(
        wordflann, words, qfx2_vec, nAssign, massign_alpha, massign_sigma, massign_equal_weights)
    # Hack to make implementing asmk easier, very redundant
    qfx2_aid = np.array([qaid] * len(qfx2_wxs), dtype=hstypes.INTEGER_TYPE)
    qfx2_qfx = np.arange(len(qfx2_vec))
    # Compute query residuals
    wx2_qrvecs, wx2_qaids, wx2_qfxs, wx2_maws, wx2_flags = smk_index.compute_residuals_(
        words, _wx2_qfxs, wx2_maws, qfx2_vec, qfx2_aid, qfx2_qfx, aggregate)
    # Compute query sccw
    if utool.VERBOSE:
        print('[smk_index] Query TF smk_alpha=%r, smk_thresh=%r' % (smk_alpha, smk_thresh))
    wx_sublist  = np.array(wx2_qrvecs.keys(), dtype=hstypes.INDEX_TYPE)
    idf_list    = [wx2_idf[wx]    for wx in wx_sublist]
    rvecs_list  = [wx2_qrvecs[wx] for wx in wx_sublist]
    maws_list   = [wx2_maws[wx]   for wx in wx_sublist]
    flags_list  = [wx2_flags[wx]  for wx in wx_sublist]
    return idf_list, rvecs_list, flags_list, maws_list, smk_alpha, smk_thresh


def testdata_internals_full(**kwargs):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
        >>> kwargs = {}
    """
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, qreq_, words = smk_debug.testdata_words(**kwargs)
    print('[smk_debug] testdata_internals_full')
    with_internals = True
    qparams = qreq_.qparams
    _args = (annots_df, daids, words, qparams, with_internals)
    invindex = smk_repr.index_data_annots(*_args)
    return ibs, annots_df, daids, qaids, invindex, qreq_


def testdata_match_kernel_L2(**kwargs):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_internals_full(**kwargs)
    print('[smk_debug] testdata_match_kernel_L2')
    qparams = qreq_.qparams
    qaid = qaids[0]
    qindex = smk_repr.new_qindex(annots_df, qaid, invindex, qparams)
    return ibs, invindex, qindex, qparams


def testdata_nonagg_rvec():
    from ibeis.model.hots.smk import smk_debug
    ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
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
        >>> from ibeis.model.hots.smk import smk_repr
        >>> ibs, annots_df, taids, daids, qaids, qreq_, nWords = smk_debug.testdata_dataframe()
        >>> qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
        >>> qparams = qreq_.qparams
        >>> invindex = smk_repr.index_data_annots(annots_df, daids, words)
        >>> qaid = qaids[0]
        >>> wx2_qrvecs, wx2_qaids, wx2_qfxs, query_sccw = smk_repr.new_qindex(annots_df, qaid, invindex, qparams)
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


def check_wx2_idxs(wx2_idxs, nWords):
    wx_list = list(wx2_idxs.keys())
    missing_vals, missing_indicies, duplicate_items = utool.debug_consec_list(wx_list)
    empty_wxs = [wx for wx, idxs in six.iteritems(wx2_idxs) if len(idxs) == 0]
    print('[smk_debug] num indexes with no support: %r' % len(missing_vals))
    print('[smk_debug] num indexes with empty idxs: %r' % len(empty_wxs))


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
            if np.any(np.isnan(rvecs[sx])):
                #rvecs[:] = 1 / np.sqrt(128)
                #print('word[wx={wx}][sx={sx}] has nans'.format(wx=wx))
                nan_wxs.append((wx, sx))
    if verbose:
        print('[smk_debug] %d words had no residuals' % len(no_wxs))
        print('[smk_debug] %d words have nans' % len(nan_wxs))
    if not (wx2_rvecs is None or wx2_idxs is None or idx2_vec is None):
        failed_wx = []
        for count, (wx, sx) in enumerate(nan_wxs):
            rvec = wx2_rvecs[wx][sx]
            idxs = wx2_idxs[wx][sx]
            dvec = idx2_vec[idxs]
            word = words[wx]
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


def assert_single_assigned_maws(maws_list):
    try:
        assert all([np.all(np.array(maws) == 1) for maws in maws_list]), 'cannot multiassign database'
    except AssertionError:
        print(maws_list)
        raise


def check_data_smksumm(aididf_list, aidrvecs_list):
    #sccw_list = []
    try:
        for count, (idf_list, rvecs_list) in enumerate(zip(aididf_list, aidrvecs_list)):
            assert len(idf_list) == len(rvecs_list), 'one list for each word'
            #sccw = smk_scoring.sccw_summation(rvecs_list, idf_list, None, smk_alpha, smk_thresh)
    except Exception as ex:
        utool.printex(ex)
        #utool.embed()
        raise


def check_invindex(invindex, verbose=True):
    """
    Example:
        >>> from ibeis.model.hots.smk import smk_index
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, taids, daids, qaids, qreq_, nWords = smk_debug.testdata_dataframe()
        >>> words = smk_index.learn_visual_words(annots_df, taids, nWords)
        >>> qparams = qreq_.qparams
        >>> invindex = smk_repr.index_data_annots(annots_df, daids, words, qparams)
    """
    daids = invindex.daids
    daid2_sccw = invindex.daid2_sccw
    check_daid2_sccw(daid2_sccw, verbose=verbose)
    assert daid2_sccw.shape[0] == daids.shape[0]
    if verbose:
        print('each aid has a sccw')


def check_daid2_sccw(daid2_sccw, verbose=True):
    daid2_sccw_values = daid2_sccw
    assert not np.any(np.isnan(daid2_sccw_values)), 'sccws are nan'
    if verbose:
        print('database sccws are not nan')
        print('database sccw stats:')
        print(utool.get_stats_str(daid2_sccw_values, newlines=True))


def test_sccw_cache():
    ibs, annots_df, taids, daids, qaids, qreq_, nWords = testdata_dataframe()
    smk_alpha  = ibs.cfg.query_cfg.smk_cfg.smk_alpha
    smk_thresh = ibs.cfg.query_cfg.smk_cfg.smk_thresh
    qparams = qreq_.qparams
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    with_internals = True
    invindex = smk_repr.index_data_annots(annots_df, daids, words, qparams, with_internals)
    idx2_daid  = invindex.idx2_daid
    wx2_drvecs = invindex.wx2_drvecs
    wx2_idf    = invindex.wx2_idf
    wx2_aids   = invindex.wx2_aids
    wx2_maws   = invindex.wx2_maws
    daids      = invindex.daids
    daid2_sccw1 = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_aids,
                                               wx2_idf, wx2_maws, smk_alpha,
                                               smk_thresh, use_cache=True)
    daid2_sccw2 = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_aids,
                                               wx2_idf, wx2_maws, smk_alpha,
                                               smk_thresh, use_cache=False)
    daid2_sccw3 = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_aids,
                                                wx2_idf, wx2_maws, smk_alpha,
                                                smk_thresh, use_cache=True)
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
        >>> rvecs_list, flag_list = smk_residual.compute_nonagg_rvecs(*_args1)  # 125 ms
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


def vector_normal_stats(vectors):
    import numpy.linalg as npl
    norm_list = npl.norm(vectors, axis=1)
    #norm_list2 = np.sqrt((vectors ** 2).sum(axis=1))
    #assert np.all(norm_list == norm_list2)
    norm_stats = utool.get_stats(norm_list)
    print('normal_stats:' + utool.dict_str(norm_stats, newlines=False))


def vector_stats(vectors, name):
    print('-- Vector Stats --')
    print(' * vectors = %r' % name)
    key_list = utool.codeblock(
        '''
        vectors.shape
        vectors.dtype
        vectors.max()
        vectors.min()
        '''
    ).split('\n')
    utool.print_keys(key_list)
    vector_normal_stats(vectors)


def sift_stats():
    import ibeis
    ibs = ibeis.opendb('PZ_Mothers')
    aid_list = ibs.get_valid_aids()
    stacked_sift = np.vstack(ibs.get_annot_desc(aid_list))
    vector_stats(stacked_sift, 'sift')
    # We see that SIFT vectors are actually normalized
    # Between 0 and 512 and clamped to uint8
    vector_stats(stacked_sift.astype(np.float32) / 512.0, 'sift')


def OLDdump_word_patches(ibs, invindex):
    """
    messy function

    Dev:
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=64000)
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=64000)
        >>> tup = smk_debug.testdata_raw_internals0(db='PZ_MTEST', nWords=64000)
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = tup
        >>> compute_data_internals_(invindex, qreq_.qparams)
    """
    from vtool import patch as ptool
    from vtool import image as gtool
    import utool
    from os.path import join
    from os.path import basename
    invindex.idx2_wxs = np.array(invindex.idx2_wxs)

    flatdir = False
    testmode = False
    print('[smk_debug] Dumping word patches to figure directory')

    vocabdir = join(ibs.get_fig_dir(), 'vocab_patches2' + ('_flat' if flatdir else ''))
    utool.ensuredir(ibs.get_fig_dir())
    utool.ensuredir(vocabdir)
    if utool.get_argflag('--vf'):
        utool.view_directory(vocabdir)

    def dump_annot_word_patches(aid):
        chip = ibs.get_annot_chips(aid)
        chip_kpts = ibs.get_annot_kpts(aid)
        nid = ibs.get_annot_nids(aid)
        idx_list = np.where(invindex.idx2_daid == aid)[0]
        fx_list = invindex.idx2_dfx[idx_list]
        wxs_list = invindex.idx2_wxs[idx_list]
        #maws_list = invindex.idx2_wxs[idxs]
        patches, subkpts = ptool.get_warped_patches(chip, chip_kpts)
        for fx, wxs, patch in zip(fx_list, wxs_list, patches):
            assert len(wxs) == 1
            for k, wx in enumerate(wxs):
                word_dname = 'wx=%06d' % wx
                patch_fname = 'patch_nid=%04d_aid=%04d_fx=%04d_k=%d' % (nid, aid, fx, k)
                if flatdir:
                    fpath = join(vocabdir, word_dname + '_' + patch_fname)
                else:
                    word_dpath = join(vocabdir, word_dname)
                    utool.ensuredir(word_dpath)
                    fpath = join(word_dpath, patch_fname)
                if testmode:
                    print(fpath)
                else:
                    gtool.imwrite(fpath, patch, fallback=True)

    for aid in utool.progiter(invindex.daids):
        dump_annot_word_patches(aid)

    def nest_large_directory(dpath):
        dpath = '/media/raid/work/GZ_ALL/_ibsdb/figures/vocab_patches'
        fpath_list = sorted(utool.ls(dpath))
        max_files = 1000
        fpath_chunks = list(utool.ichunks(fpath_list, max_files))
        for count, src_list in enumerate(utool.progiter(fpath_chunks)):
            print('')
            chunk_dpath = join(dpath, 'chunk%d' % count)
            dst_list = [join(chunk_dpath, basename(fpath)) for fpath in src_list]
            utool.ensuredir(chunk_dpath)
            utool.move_list(src_list, dst_list)

    def select_subdirs(dpath):
        dpath1 = '/media/raid/work/GZ_ALL/_ibsdb/figures/vocab_patches'
        dpath_list = sorted(utool.glob(dpath1, '*', recursive=True, with_files=False))
        dpath_list1 = list(filter(lambda x: x.find('wx=') > -1, dpath_list))

        subfiles = [utool.ls(dpath_) for dpath_ in dpath_list1]
        subfiles_len = list(map(len, subfiles))

        dpath_list2 = utool.sortedby(dpath_list1, subfiles_len, reverse=True)

        seldpath = join(dpath1, '..', 'selected_vocab_patches/')

        for dpath2 in utool.progiter(dpath_list2[0:50]):
            print('')
            utool.copy(dpath2, join(seldpath, basename(dpath2)))

        # stack for show
        from plottool import draw_func2 as df2
        #df2.rrr()
        bigpatch_list = []
        for dpath in utool.ls(seldpath):
            try:
                fpath_list = utool.ls(dpath)
                patch_list = [gtool.imread(fpath) for fpath in fpath_list]
                #img_list = patch_list
                #bigpatch = df2.stack_image_recurse(patch_list)
                #bigpatch = df2.stack_image_list(patch_list, vert=False)
                bigpatch = df2.stack_square_images(patch_list)
                bigpatch_list.append(bigpatch)
                bigpatch_fpath = join(seldpath, basename(dpath) + '_patches.png')
                gtool.imwrite(bigpatch_fpath, bigpatch)
            except IndexError:
                pass
                #df2.imshow(bigpatch)
                #df2.present()
                pass


def vizualize_vocabulary(ibs, invindex):
    """
    cleaned up version of dump_word_patches

    Dev:
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        #>>> tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=64000)
        #>>> tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=64000)
        #>>> smk_debug.rrr()
        #>>> tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=8000)
        >>> tup = smk_debug.testdata_raw_internals0(db='PZ_Master0', nWords=64000)
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = tup
        >>> compute_data_internals_(invindex, qreq_.qparams, delete_rawvecs=False)
    """
    from vtool import patch as ptool
    from vtool import image as gtool
    import six
    import utool
    import scipy.stats.mstats as spms
    from os.path import join
    from os.path import basename
    import scipy.spatial.distance as spdist
    invindex.idx2_wxs = np.array(invindex.idx2_wxs)

    print('[smk_debug] Vizualizing vocabulary')

    # DUMPING PART --- dumps patches to disk
    figdir = ibs.get_fig_dir()
    utool.ensuredir(figdir)
    if utool.get_argflag('--vf'):
        utool.view_directory(figdir)

    ax2_idxs = [np.where(invindex.idx2_daid == aid)[0] for aid in utool.progiter(
        invindex.daids, 'Building Forward Index: ', freq=100)]

    def euclidean_dist(vecs1, vec2):
        return np.sqrt(((vecs1.astype(np.float32) - vec2.astype(np.float32)) ** 2).sum(1))

    def dict_where0(dict_):
        keys = np.array(dict_.keys())
        flags = np.array(list(map(len, dict_.values()))) == 0
        indices = np.where(flags)[0]
        return keys[indices]

    def compute_word_statistics(invindex):
        wx2_idxs  = invindex.wx2_idxs
        idx2_dvec = invindex.idx2_dvec
        words     = invindex.words
        wx2_pdist = {}
        wx2_wdist = {}
        wx2_nMembers = {}
        wx2_pdist_stats = {}
        wx2_wdist_stats = {}
        wordidx_iter = utool.progiter(six.iteritems(wx2_idxs), lbl='Word Dists: ', num=len(wx2_idxs), freq=200)

        for _item in wordidx_iter:
            wx, idxs = _item
            dvecs = idx2_dvec.take(idxs, axis=0)
            word = words[wx:wx + 1]
            wx2_pdist[wx] = spdist.pdist(dvecs)  # pairwise dist between words
            wx2_wdist[wx] = euclidean_dist(dvecs, word)  # dist to word center
            wx2_nMembers[wx] = len(idxs)

        for wx, pdist in utool.progiter(six.iteritems(wx2_pdist), lbl='Word pdist Stats: ', num=len(wx2_idxs), freq=2000):
            wx2_pdist_stats[wx] = utool.get_stats(pdist)

        for wx, wdist in utool.progiter(six.iteritems(wx2_wdist), lbl='Word wdist Stats: ', num=len(wx2_idxs), freq=2000):
            wx2_wdist_stats[wx] = utool.get_stats(wdist)

        utool.print_stats(wx2_nMembers.values(), 'word members')
        return wx2_pdist, wx2_wdist, wx2_nMembers, wx2_pdist_stats, wx2_wdist_stats
        #word_pdist = spdist.pdist(invindex.words)

    # Compute Word Statistics
    (wx2_pdist, wx2_wdist, wx2_nMembers, wx2_pdist_stats, wx2_wdist_stats) = compute_word_statistics(invindex)

    wx2_prad = {wx: pdist_stats['max'] for wx, pdist_stats in six.iteritems(wx2_pdist_stats) if 'max' in pdist_stats}
    wx2_wrad = {wx: wdist_stats['max'] for wx, wdist_stats in six.iteritems(wx2_wdist_stats) if 'max' in wdist_stats}

    def select_by_metric(wx2_metric, per_quantile=20):
        # sample a few words around the quantile points
        metric_list = np.array(list(wx2_metric.values()))
        wx_list = np.array(list(wx2_nMembers.keys()))
        metric_quantiles = spms.mquantiles(metric_list)
        metric_quantiles = np.array(metric_quantiles.tolist() + [metric_list.max(), metric_list.min()])
        wx_interest = []
        for scalar in metric_quantiles:
            dist = (metric_list - scalar) ** 2
            wx_quantile = wx_list[dist.argsort()[0:per_quantile]]
            wx_interest.extend(wx_quantile.tolist())
        overlap = len(wx_interest) - len(set(wx_interest))
        if overlap > 0:
            print('warning: overlap=%r' % overlap)
        return wx_interest

    wx_sample1 = select_by_metric(wx2_nMembers)
    wx_sample2 = select_by_metric(wx2_prad)
    wx_sample3 = select_by_metric(wx2_wrad)

    wx_sample = wx_sample1 + wx_sample2 + wx_sample3
    overlap123 = len(wx_sample) - len(set(wx_sample))
    print('overlap123 = %r' % overlap123)
    wx_sample  = set(wx_sample)
    print('len(wx_sample) = %r' % len(wx_sample))

    def make_scatterplots():
        from plottool import draw_func2 as df2

        def wx2_avepdist(wx):
            return wx2_pdist_stats[wx]['mean'] if wx in wx2_pdist_stats and 'mean' in wx2_pdist_stats[wx] else -1
        def wx2_avewdist(wx):
            return wx2_wdist_stats[wx]['mean'] if wx in wx2_wdist_stats and 'mean' in wx2_wdist_stats[wx] else -1

        wx2_idf = invindex.wx2_idf

        # data
        wx_list  =  list(wx2_idf.keys())
        idf_list =  [wx2_idf[wx] for wx in wx_list]
        nPoints_list =  [wx2_nMembers[wx] if wx in wx2_nMembers else -1 for wx in wx_list]
        avepdist_list = [wx2_avepdist(wx) for wx in wx_list]
        avewdist_list = [wx2_avewdist(wx) for wx in wx_list]

        def draw_scatterplot(datax, datay, xlabel, ylabel, color, fnum=None):
            datac = [color for _ in range(len(datax))]
            assert len(datax) == len(datay), '%r %r' % (len(datax), len(datay))
            df2.figure(fnum=fnum, doclf=True, docla=True)
            df2.plt.scatter(datax, datay,  c=datac, s=20, marker='o', alpha=.9)
            ax = df2.gca()
            title = '%s vs %s.\nnWords=%r. db=%r' % (xlabel, ylabel, len(datax), ibs.get_dbname())
            df2.set_xlabel(xlabel)
            df2.set_ylabel(ylabel)
            ax.set_ylim(min(datay) - 1, max(datay) + 1)
            ax.set_xlim(min(datax) - 1, max(datax) + 1)
            df2.dark_background()
            df2.set_figtitle(title)
            figpath = join(figdir, title)
            df2.save_figure(fnum, figpath)

        df2.reset()
        draw_scatterplot(idf_list, avepdist_list, 'idf', 'mean(pdist)', df2.WHITE, fnum=1)
        draw_scatterplot(idf_list, avewdist_list, 'idf', 'mean(wdist)', df2.PINK, fnum=3)
        draw_scatterplot(nPoints_list, avepdist_list, 'nPointsInWord', 'mean(pdist)', df2.GREEN, fnum=2)
        draw_scatterplot(avepdist_list, avewdist_list, 'mean(pdist)', 'mean(wdist)', df2.YELLOW, fnum=4)
        draw_scatterplot(nPoints_list, avewdist_list, 'nPointsInWord', 'mean(wdist)', df2.ORANGE, fnum=5)
        draw_scatterplot(idf_list, nPoints_list, 'idf', 'nPointsInWord', df2.LIGHT_BLUE, fnum=6)
        #df2.present()
    make_scatterplots()

    def make_wordfigures():
        """
        Builds mosaics of patches assigned to words in sample
        ouptuts them to disk
        """

        def get_word_dname(wx):
            stats_ = wx2_wdist_stats[wx]
            wname_clean = 'wx=%06d' % wx
            stats1 = 'max={max},min={min},mean={mean},'.format(**stats_)
            stats2 = 'std={std},nMaxMin=({nMax},{nMin}),shape={shape}'.format(**stats_)
            fname_fmt = wname_clean + '_{stats1}{stats2}'
            fmt_dict = dict(stats1=stats1, stats2=stats2)
            word_dname = utool.long_fname_format(fname_fmt, fmt_dict, ['stats2', 'stats1'], max_len=250, hashlen=4)
            return word_dname

        vocabdir = join(figdir, 'vocab_patches2')
        utool.ensuredir(vocabdir)
        wx2_dpath = {wx: join(vocabdir, get_word_dname(wx)) for wx in wx_sample}

        for dpath in utool.progiter(list(wx2_dpath.values()),
                                    lbl='Ensuring word_dpath: ', freq=200):
            utool.ensuredir(dpath)

        #
        # DUMP PATCHES IN WX_SAMPLE

        # Write each patch from each annotation to disk
        patchdump_iter = utool.progiter(zip(invindex.daids, ax2_idxs), freq=1,
                                        lbl='Dumping Selected Patches: ', num=len(invindex.daids))
        for aid, idxs in patchdump_iter:
            wxs_list  = invindex.idx2_wxs[idxs]
            if len(set(utool.flatten(wxs_list)).intersection(set(wx_sample))) == 0:
                # skip this annotation
                continue
            fx_list   = invindex.idx2_dfx[idxs]
            chip      = ibs.get_annot_chips(aid)
            chip_kpts = ibs.get_annot_kpts(aid)
            nid       = ibs.get_annot_nids(aid)
            #maws_list = invindex.idx2_wxs[idxs]
            patches, subkpts = ptool.get_warped_patches(chip, chip_kpts)
            for fx, wxs, patch in zip(fx_list, wxs_list, patches):
                assert len(wxs) == 1, 'did you multiassign the database? If so implement it here too'
                for k, wx in enumerate(wxs):
                    if wx not in wx_sample:
                        continue
                    patch_fname = 'patch_nid=%04d_aid=%04d_fx=%04d_k=%d' % (nid, aid, fx, k)
                    fpath = join(wx2_dpath[wx], patch_fname)
                    #gtool.imwrite(fpath, patch, fallback=True)
                    gtool.imwrite_fallback(fpath, patch)

        # COLLECTING PART --- collects patches in word folders
        #vocabdir

        seldpath = vocabdir + '_selected'
        utool.ensurepath(seldpath)
        # stack for show
        from plottool import draw_func2 as df2
        import parse
        for wx, dpath in utool.progiter(six.iteritems(wx2_dpath), lbl='Dumping Word Images:', num=len(wx2_dpath), freq=1, backspace=False):
            #df2.rrr()
            fpath_list = utool.ls(dpath)
            fname_list = [basename(fpath_) for fpath_ in fpath_list]
            patch_list = [gtool.imread(fpath_) for fpath_ in fpath_list]
            # color each patch by nid
            nid_list = [int(parse.parse('{}_nid={nid}_{}', fname)['nid']) for fname in fname_list]
            nid_set = set(nid_list)
            nid_list = np.array(nid_list)
            if len(nid_list) == len(nid_set):
                # no duplicate names
                newpatch_list = patch_list
            else:
                # duplicate names. do coloring
                sortx = nid_list.argsort()
                patch_list = np.array(patch_list, dtype=object)[sortx]
                fname_list = np.array(fname_list, dtype=object)[sortx]
                nid_list = nid_list[sortx]
                colors = (255 * np.array(df2.distinct_colors(len(nid_set)))).astype(np.int32)
                color_dict = dict(zip(nid_set, colors))
                wpad, hpad = 3, 3
                newshape_list = [tuple((np.array(patch.shape) + (wpad * 2, hpad * 2, 0)).tolist()) for patch in patch_list]
                color_list = [color_dict[nid_] for nid_ in nid_list]
                newpatch_list = [np.zeros(shape) + color[None, None] for shape, color in zip(newshape_list, color_list)]
                for patch, newpatch in zip(patch_list, newpatch_list):
                    newpatch[wpad:-wpad, hpad:-hpad, :] = patch
                #img_list = patch_list
                #bigpatch = df2.stack_image_recurse(patch_list)
            #bigpatch = df2.stack_image_list(patch_list, vert=False)
            bigpatch = df2.stack_square_images(newpatch_list)
            bigpatch_fpath = join(seldpath, basename(dpath) + '_patches.png')

            #
            def dictstr(dict_):
                str_ = utool.dict_str(dict_, newlines=False)
                str_ = str_.replace('\'', '').replace(': ', '=').strip('{},')
                return str_

            figtitle = '\n'.join([
                'wx=%r' % wx,
                'stat(pdist): %s' % dictstr(wx2_pdist_stats[wx]),
                'stat(wdist): %s' % dictstr(wx2_wdist_stats[wx]),
            ])
            wx2_nMembers[wx]

            df2.figure(fnum=1, doclf=True, docla=True)
            fig, ax = df2.imshow(bigpatch, figtitle=figtitle)
            #fig.show()
            df2.set_figtitle(figtitle)
            df2.adjust_subplots(top=.878, bottom=0)
            df2.save_figure(1, bigpatch_fpath)
            #gtool.imwrite(bigpatch_fpath, bigpatch)
    make_wordfigures()
    if True:
        import sys
        sys.exit(1)


def get_cached_vocabs():
    from os.path import join
    import parse
    # Parse some of the training data from fname
    parse_str = '{}nC={num_cent},{}_DPTS(({num_dpts},{dim}){}'
    smkdir = utool.get_app_resource_dir('smk')
    fname_list = utool.glob(smkdir, 'akmeans*')
    fpath_list = [join(smkdir, fname) for fname in fname_list]
    result_list = [parse.parse(parse_str, fpath) for fpath in fpath_list]
    nCent_list = [int(res['num_cent']) for res in result_list]
    nDpts_list = [int(res['num_dpts']) for res in result_list]
    key_list = zip(nCent_list, nDpts_list)
    fpath_sorted = utool.sortedby(fpath_list, key_list, reverse=True)
    return fpath_sorted


def view_vocabs():
    """
    looks in vocab cachedir and prints info / vizualizes the vocabs
    """
    from vtool import clustering2 as clustertool
    import numpy as np

    fpath_sorted = get_cached_vocabs()

    num_pca_dims = 2  # 3
    whiten       = False
    kwd = dict(num_pca_dims=num_pca_dims,
               whiten=whiten,)

    def view_vocab(fpath):
        # QUANTIZED AND FLOATING POINT STATS
        centroids = utool.load_cPkl(fpath)
        print('viewing vocat fpath=%r' % (fpath,))
        vector_stats(centroids, 'centroids')
        #centroids_float = centroids.astype(np.float64) / 255.0
        centroids_float = centroids.astype(np.float64) / 512.0
        vector_stats(centroids_float, 'centroids_float')

        fig = clustertool.plot_centroids(centroids, centroids, labels='centroids',
                                         fnum=1, prefix='centroid vecs\n', **kwd)
        fig.show()

    for count, fpath in enumerate(fpath_sorted):
        if count > 0:
            break
        view_vocab(fpath)


def check_qaid2_chipmatch(qaid2_chipmatch, qaids, verbose=True):
    try:
        assert isinstance(qaid2_chipmatch, dict), 'type(qaid2_chipmatch) = %r' % type(qaid2_chipmatch)
        qaid_list = list(qaid2_chipmatch.keys())
        _qaids = set(qaids)
        assert _qaids == set(qaid_list), 'something is wrong'
        print('has correct key. (len(keys) = %r)' % len(_qaids))
        chipmatch_list = list(qaid2_chipmatch.values())
        for count, daid2_chipmatch in enumerate(chipmatch_list):
            check_daid2_chipmatch(daid2_chipmatch)
    except Exception as ex:
        utool.printex(ex, keys=['qaid2_chipmatch', 'daid2_chipmatch', 'count'])
        raise


def check_daid2_chipmatch(daid2_chipmatch, verbose=True):
    print('[smk_debug] checking %d chipmatches' % len(daid2_chipmatch))
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
    if not isinstance(dict_, dict):
        return 'expected dict got %r' % type(dict_)

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
            # each key holds an ndarray
            val_shape_stats = utool.get_stats(set(map(np.shape, vals)), axis=0)
            val_shape_stats_str = utool.dict_str(val_shape_stats, strvals=True, newlines=False)
            val_dtypes = list(set([val.dtype for val in vals]))
            fmtstr_ += utool.unindent('''
            * val_shape_stats = {val_shape_stats_str}
            * val_dtypes = {val_dtypes}
            '''.strip('\n'))
        elif val_types[0] == list:
            # each key holds a list
            val_len_stats =  utool.get_stats(set(map(len, vals)))
            val_len_stats_str = utool.dict_str(val_len_stats, strvals=True, newlines=False)
            depth = utool.list_depth(vals)
            deep_val_types = list(set(utool.list_deep_types(vals)))
            fmtstr_ += utool.unindent('''
            * list_depth = {depth}
            * val_len_stats = {val_len_stats_str}
            * deep_types = {deep_val_types}
            '''.strip('\n'))
            if len(deep_val_types) == 1:
                if deep_val_types[0] == np.ndarray:
                    deep_val_dtypes = list(set([val.dtype for val in vals]))
                    fmtstr_ += utool.unindent('''
                    * deep_val_dtypes = {deep_val_dtypes}
                    ''').strip('\n')
        elif val_types[0] in [np.uint8, np.int8, np.int32, np.int64, np.float16, np.float32, np.float64]:
            # each key holds a scalar
            val_stats = utool.get_stats(vals)
            fmtstr_ += utool.unindent('''
            * val_stats = {val_stats}
            ''').strip('\n')

    fmtstr = fmtstr_.format(**locals())
    return utool.indent(fmtstr)


def invindex_dbgstr(invindex):
    """
    >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    >>> ibs, annots_df, daids, qaids, invindex = testdata_raw_internals0()
    >>> invindex_dbgstr(invindex)
    """
    print('+--- INVINDEX DBGSTR ---')
    print('called by %r' % (utool.get_caller_name(),))
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
        >>> from ibeis.model.hots.smk import smk_match
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_internals_full()
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
    smk_alpha     = qreq_.qparams.smk_alpha
    smk_thresh    = qreq_.qparams.smk_thresh
    lbl = '[smk_match] asmk query: ' if aggregate else '[smk_match] smk query: '
    mark, end_ = utool.log_progress(lbl, len(qaids), flushfreq=1,
                                    writefreq=1, with_totaltime=True,
                                    backspace=False)
    withinfo = True
    for count, qaid in enumerate(qaids):
        mark(count)
        daid2_score, daid2_chipmatch = smk_match.query_inverted_index(
            annots_df, qaid, invindex, withinfo, aggregate, smk_alpha, smk_thresh)
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


def dbstr_qindex(qindex_=None):
    qindex = utool.get_localvar_from_stack('qindex')
    common_wxs = utool.get_localvar_from_stack('common_wxs')
    wx2_qaids = utool.get_localvar_from_stack('wx2_qaids')
    qindex.query_sccw
    qmaws_list  = [qindex.wx2_maws[wx] for wx in common_wxs]
    qaids_list  = [qindex.wx2_qaids[wx] for wx in common_wxs]
    qfxs_list   = [qindex.wx2_qfxs[wx] for wx in common_wxs]
    qrvecs_list = [qindex.wx2_qrvecs[wx] for wx in common_wxs]
    qaids_list  = [wx2_qaids[wx] for wx in common_wxs]
    print('-- max --')
    print('list_depth(qaids_list) = %d' % utool.list_depth(qaids_list, max))
    print('list_depth(qmaws_list) = %d' % utool.list_depth(qmaws_list, max))
    print('list_depth(qfxs_list) = %d' % utool.list_depth(qfxs_list, max))
    print('list_depth(qrvecs_list) = %d' % utool.list_depth(qrvecs_list, max))
    print('-- min --')
    print('list_depth(qaids_list) = %d' % utool.list_depth(qaids_list, min))
    print('list_depth(qmaws_list) = %d' % utool.list_depth(qmaws_list, min))
    print('list_depth(qfxs_list) = %d' % utool.list_depth(qfxs_list, min))
    print('list_depth(qrvecs_list) = %d' % utool.list_depth(qrvecs_list, min))
    print('-- sig --')
    print('list_depth(qaids_list) = %r' % utool.depth_profile(qaids_list))
    print('list_depth(qmaws_list) = %r' % utool.depth_profile(qmaws_list))
    print('list_depth(qfxs_list) = %r' % utool.depth_profile(qfxs_list))
    print('list_depth(qrvecs_list) = %r' % utool.depth_profile(utool.depth_profile(qrvecs_list)))
    print(qfxs_list[0:3])
    print(qaids_list[0:3])
    print(qmaws_list[0:3])


def main():
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_debug import *  # NOQA
    """
    print('+------------')
    print('SMK_DEBUG MAIN')
    print('+------------')
    from ibeis.model.hots import pipeline
    ibs, annots_df, taids, daids, qaids, qreq_, nWords = testdata_dataframe()
    # Query using SMK
    #qaid = qaids[0]
    nWords    = qreq_.qparams.nWords
    aggregate = qreq_.qparams.aggregate
    smk_alpha  = qreq_.qparams.smk_alpha
    smk_thresh = qreq_.qparams.smk_thresh
    nAssign   = qreq_.qparams.nAssign
    #aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    #smk_alpha = ibs.cfg.query_cfg.smk_cfg.smk_alpha
    #smk_thresh = ibs.cfg.query_cfg.smk_cfg.smk_thresh
    print('+------------')
    print('SMK_DEBUG PARAMS')
    print('[smk_debug] aggregate = %r' % (aggregate,))
    print('[smk_debug] smk_alpha   = %r' % (smk_alpha,))
    print('[smk_debug] smk_thresh  = %r' % (smk_thresh,))
    print('[smk_debug] nWords  = %r' % (nWords,))
    print('[smk_debug] nAssign = %r' % (nAssign,))
    print('L------------')
    # Learn vocabulary
    #words = qreq_.words = smk_index.learn_visual_words(annots_df, taids, nWords)
    # Index a database of annotations
    #qreq_.invindex = smk_repr.index_data_annots(annots_df, daids, words, aggregate, smk_alpha, smk_thresh)
    qreq_.ibs = ibs
    # Smk Mach
    print('+------------')
    print('SMK_DEBUG MATCH KERNEL')
    print('+------------')
    qaid2_scores, qaid2_chipmatch_SMK = smk_match.execute_smk_L5(qreq_)
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


def get_test_float_norm_rvecs(num=1000, dim=None):
    import numpy.linalg as npl
    from ibeis.model.hots import hstypes
    if dim is None:
        dim = hstypes.VEC_DIM
    rvecs_float = np.random.normal(size=(num, dim))
    rvecs_norm_float = rvecs_float / npl.norm(rvecs_float, axis=1)[:, None]
    return rvecs_norm_float


def get_test_rvecs(num=1000, dim=None, nanrows=None):
    from ibeis.model.hots import hstypes
    max_ = hstypes.RVEC_MAX
    min_ = hstypes.RVEC_MIN
    dtype = hstypes.RVEC_TYPE
    if dim is None:
        dim = hstypes.VEC_DIM
    dtype_range = max_ - min_
    rvecs_float = np.random.normal(size=(num, dim))
    rvecs = ((dtype_range * rvecs_float) - hstypes.RVEC_MIN).astype(dtype)
    if nanrows is not None:
        rvecs[nanrows] = np.nan

    """
    dtype = np.int8
    max_ = 128
    min_ = -128
    nanrows = 1

    import numpy.ma as ma
    if dtype not in [np.float16, np.float32, np.float64]:
        rvecs.view(ma.MaskedArray)


    np.ma.array([1,2,3,4,5], dtype=int)

    """
    return rvecs


def get_test_maws(rvecs):
    from ibeis.model.hots import hstypes
    return (np.random.rand(rvecs.shape[0])).astype(hstypes.FLOAT_TYPE)


def testdata_match_kernel_L0():
    from ibeis.model.hots.smk import smk_debug
    from ibeis.model.hots import hstypes
    np.random.seed(0)
    smk_alpha = 3.0
    smk_thresh = 0.0
    num_qrvecs_per_word = [0, 1, 3, 4, 5]
    num_drvecs_per_word = [0, 1, 2, 4, 6]
    qrvecs_list = [smk_debug.get_test_rvecs(n, dim=2) for n in num_qrvecs_per_word]
    drvecs_list = [smk_debug.get_test_rvecs(n, dim=2) for n in num_drvecs_per_word]
    daids_list  = [list(range(len(rvecs))) for rvecs in drvecs_list]
    qaids_list  = [[42] * len(rvecs) for rvecs in qrvecs_list]
    qmaws_list  = [smk_debug.get_test_maws(rvecs) for rvecs in qrvecs_list]
    dmaws_list  = [np.ones(rvecs.shape[0], dtype=hstypes.FLOAT_TYPE) for rvecs in drvecs_list]
    idf_list = [1.0 for _ in qrvecs_list]
    daid2_sccw  = {daid: 1.0 for daid in range(10)}
    query_sccw = smk_scoring.sccw_summation(qrvecs_list, idf_list, qmaws_list, smk_alpha, smk_thresh)
    qaid2_sccw  = {42: query_sccw}
    core1 = smk_alpha, smk_thresh, query_sccw, daids_list, daid2_sccw
    core2 = qrvecs_list, drvecs_list, qmaws_list, dmaws_list, idf_list
    extra = qaid2_sccw, qaids_list
    return core1, core2, extra


def testdata_similarity_function():
    from ibeis.model.hots.smk import smk_debug
    qrvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
    drvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
    return qrvecs_list, drvecs_list


def testdata_apply_weights():
    from ibeis.model.hots.smk import smk_debug
    from ibeis.model.hots import hstypes
    qrvecs_list, drvecs_list = smk_debug.testdata_similarity_function()
    simmat_list = smk_scoring.similarity_function(qrvecs_list, drvecs_list)
    qmaws_list  = [smk_debug.get_test_maws(rvecs) for rvecs in qrvecs_list]
    dmaws_list  = [np.ones(rvecs.shape[0], dtype=hstypes.FLOAT_TYPE) for rvecs in qrvecs_list]
    idf_list = [1 for _ in qrvecs_list]
    return simmat_list, qmaws_list, dmaws_list, idf_list


def testdata_selectivity_function():
    from ibeis.model.hots.smk import smk_debug
    smk_alpha = 3
    smk_thresh = 0
    simmat_list, qmaws_list, dmaws_list, idf_list = smk_debug.testdata_apply_weights()
    wsim_list = smk_scoring.apply_weights(simmat_list, qmaws_list, dmaws_list, idf_list)
    return wsim_list, smk_alpha, smk_thresh


if __name__ == '__main__':
    print('\n\n\n\n\n\n')
    import multiprocessing
    from plottool import draw_func2 as df2
    mode = utool.get_argval('--mode', int, default=0)
    if mode == 0 or utool.get_argflag('--view-vocabs'):
        view_vocabs()
    else:
        np.set_printoptions(precision=2)
        multiprocessing.freeze_support()  # for win32
        main_locals = main()
        main_execstr = utool.execstr_dict(main_locals, 'main_locals')
        exec(main_execstr)
    exec(df2.present())
