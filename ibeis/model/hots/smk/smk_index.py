from __future__ import absolute_import, division, print_function
import six
import utool
import numpy as np
import pandas as pd
from six.moves import zip  # NOQA
from vtool import clustering2 as clustertool
from vtool import nearest_neighbors as nntool
from ibeis.model.hots.smk import smk_core
from ibeis.model.hots.smk import smk_speed
#from ibeis.model.hots.smk import smk_match
from ibeis.model.hots.smk import pandas_helpers as pdh
from ibeis.model.hots.smk.hstypes import INTEGER_TYPE, FLOAT_TYPE
from ibeis.model.hots.smk.pandas_helpers import VEC_COLUMNS, KPT_COLUMNS
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_index]')

USE_CACHE_WORDS = not utool.get_flag('--nocache-words')


@six.add_metaclass(utool.ReloadingMetaclass)
class InvertedIndex(object):
    """
    class which holds inverted index state information
    """
    def __init__(invindex, words, wordflann, idx2_vec, idx2_aid, idx2_fx, _daids):
        invindex.wordflann   = wordflann
        invindex.words       = words     # visual word centroids
        invindex.daids       = _daids    # indexed annotation ids
        invindex.idx2_dvec   = idx2_vec  # stacked index -> descriptor vector (currently sift)
        invindex.idx2_daid   = idx2_aid  # stacked index -> annot id
        invindex.idx2_dfx    = idx2_fx   # stacked index -> feature index (wrt daid)
        invindex.wx2_idxs    = None     # word index -> stacked indexes
        invindex.wx2_aids    = None     # word index -> aggregate aids
        invindex.wx2_fxs     = None     # word index -> aggregate aids
        invindex.wx2_drvecs  = None     # word index -> residual vectors
        invindex.wx2_weight  = None     # word index -> idf (wx normalize)
        invindex.daid2_gamma = None     # word index -> gamma (daid normalizer)

    #def get_cfgstr(invindex):
    #    lbl = 'InvIndex'
    #    hashstr = utool.hashstr(repr(invindex.wx2_drvecs))
    #    return '_{lbl}({hashstr})'.format(lbl=lbl, hashstr=hashstr)


@profile
def make_annot_df(ibs):
    """
    Creates a panda dataframe using an ibeis controller
    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs = smk_debug.testdata_ibeis()
    >>> annots_df = make_annot_df(ibs)
    >>> print(utool.hashstr(repr(annots_df.values)))
    j12n+x93m4c!4un3

    #>>> from ibeis.model.hots.smk import smk_debug
    #>>> smk_debug.rrr()
    #>>> smk_debug.check_dtype(annots_df)
    """
    aid_list = ibs.get_valid_aids()
    kpts_list = ibs.get_annot_kpts(aid_list)
    vecs_list = ibs.get_annot_desc(aid_list)
    aid_series = pdh.IntSeries(np.array(aid_list, dtype=INTEGER_TYPE), name='aid')
    kpts_df = pdh.pandasify_list2d(kpts_list, aid_series, KPT_COLUMNS, 'fx', 'kpts')
    vecs_df = pdh.pandasify_list2d(vecs_list, aid_series, VEC_COLUMNS, 'fx', 'vecs')
    # Pandas Annotation Dataframe
    annots_df = pd.concat([kpts_df, vecs_df], axis=1)
    return annots_df


@profile
def learn_visual_words(annots_df, taids, nWords, use_cache=USE_CACHE_WORDS):
    """
    Computes visual words
    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk_debug.testdata_dataframe()
    >>> use_cache = USE_CACHE_WORDS
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> print(words.shape)
    (8000, 128)
    """
    train_vecs_df = annots_df['vecs'][taids]
    train_vecs = np.vstack(train_vecs_df.values)
    print('Training %d word vocabulary with %d annots and %d descriptors' %
          (nWords, len(taids), len(train_vecs)))
    _words = clustertool.cached_akmeans(train_vecs, nWords, max_iters=100,
                                        use_cache=use_cache, appname='smk')
    wx_series = pdh.IntIndex(np.arange(len(_words)), name='wx')
    words = pd.DataFrame(_words, index=wx_series, columns=VEC_COLUMNS)
    return words


@profile
def index_data_annots(annots_df, daids, words, with_internals=True,
                      aggregate=False, alpha=3, thresh=0):
    """
    Builds the initial inverted index from a dataframe, daids, and words.
    Optionally builds the internals of the inverted structure
    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, daids, qaids, words = smk_debug.testdata_words()
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)

    #>>> print(utool.hashstr(repr(list(invindex.__dict__.values()))))
    #v8+i5i8+55j0swio
    """
    print('[smk_index] index_data_annots')
    vecs_list = annots_df['vecs'][daids]
    flann_params = {}
    _words = words.values
    wordflann = nntool.flann_cache(_words, flann_params=flann_params,
                                   appname='smk')
    _daids = pdh.ensure_numpy(daids)
    _vecs_list = pdh.ensure_numpy(vecs_list)
    _idx2_dvec, _idx2_daid, _idx2_dfx = nntool.invertable_stack(_vecs_list, _daids)

    # Pandasify
    idx_series = pdh.IntIndex(np.arange(len(_idx2_daid)), name='idx')
    idx2_dfx   = pdh.IntSeries(_idx2_dfx, index=idx_series, name='fx')
    idx2_daid  = pdh.IntSeries(_idx2_daid, index=idx_series, name='aid')
    idx2_dvec  = pd.DataFrame(_idx2_dvec, index=idx_series, columns=VEC_COLUMNS)

    invindex = InvertedIndex(words, wordflann, idx2_dvec, idx2_daid, idx2_dfx, daids)
    if with_internals:
        compute_data_internals_(invindex, aggregate, alpha, thresh)
    return invindex


@profile
def compute_data_internals_(invindex, aggregate=False, alpha=3, thresh=0):
    """
    13 seconds
    Builds each of the inverted index internals.

    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_raw_internals0()
    >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    >>> alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> compute_data_internals_(invindex, aggregate, alpha, thresh)
    """
    print('[smk_index] compute_data_internals_')
    print('[smk_index] aggregate = %r' % (aggregate,))
    print('[smk_index] alpha = %r' % (alpha,))
    print('[smk_index] thresh = %r' % (thresh,))
    # Get information
    idx2_vec  = invindex.idx2_dvec
    idx2_dfx  = invindex.idx2_dfx
    idx2_daid = invindex.idx2_daid
    daids     = invindex.daids
    wordflann = invindex.wordflann
    words     = invindex.words
    wx_series = invindex.words.index
    # Compute word assignments
    wx2_idxs, idx2_wx = assign_to_words_(
        wordflann, words, idx2_vec, idx_name='idx', dense=True)  # 7.1 %
    # Compute word weights
    wx2_weight = compute_word_idf_(
        wx_series, wx2_idxs, idx2_daid, daids)  # 2.4 %
    # Compute residual vectors and inverse mappings
    wx2_drvecs, wx2_aids, wx2_fxs = compute_residuals_(
        words, wx2_idxs, idx2_vec, idx2_daid, idx2_dfx, aggregate)  # 42.9%
    # Compute annotation normalization factor
    daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_aids,
                                      wx2_weight, daids, alpha, thresh)  # 47.5%
    # Store information
    invindex.idx2_wx    = idx2_wx    # stacked index -> word index
    invindex.wx2_idxs   = wx2_idxs
    invindex.wx2_weight = wx2_weight
    invindex.wx2_drvecs = wx2_drvecs
    invindex.wx2_aids   = wx2_aids  # needed for asmk
    invindex.wx2_fxs    = wx2_fxs  # needed for asmk
    invindex.daid2_gamma = daid2_gamma


@profile
def assign_to_words_(wordflann, words, idx2_vec, idx_name='idx', dense=True,
                     nAssign=1):
    """
    Time: 19 seconds

    Assigns descriptor-vectors to nearest word. Returns forward and inverted index.

    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_raw_internals0()
    >>> words  = invindex.words
    >>> wordflann = invindex.wordflann
    >>> idx2_vec  = invindex.idx2_dvec
    >>> dense = True
    >>> nAssign = ibs.cfg.query_cfg.smk_cfg.nAssign
    >>> idx_name, series_name = 'idx', 'wx2_idxs'
    >>> _dbargs = (wordflann, words, idx2_vec, idx_name, dense, nAssign)
    >>> wx2_idxs, idx2_wx = assign_to_words_(*_dbargs)
    """
    wx_series  = words.index
    idx_series = idx2_vec.index
    idx_series_values = idx_series.values
    idx2_vec_values = pdh.ensure_numpy(idx2_vec)
    # Find each vectors nearest word
    #TODO: multiple assignment
    _idx2_wx, _idx2_wdist = wordflann.nn_index(idx2_vec_values, 1)
    PANDAS_GROUP = True
    # Pandas grouping seems to be faster in this instance
    if PANDAS_GROUP:
        word_assignments = pd.DataFrame(_idx2_wx, index=idx2_vec.index, columns=['wx'])  # 107 us
        # Compute inverted index
        word_group = word_assignments.groupby('wx')  # 44.5 us
        _wx2_idxs = word_group['wx'].indices  # 13.6 us
    else:
        wx_list, groupxs = smk_speed.group_indicies(_idx2_wx)  # 5.52 ms
        idxs_list = [idx_series_values.take(xs) for xs in groupxs]  # 2.9 ms
        _wx2_idxs = dict(zip(wx_list, idxs_list))  # 753 us

    wx2_idxs = pdh.pandasify_dict1d(_wx2_idxs, wx_series, idx_name, ('wx2_' + idx_name + 's'), dense=dense)  # 97.4 %
    idx2_wx = pdh.IntSeries(_idx2_wx, index=idx_series, name='wx')
    return wx2_idxs, idx2_wx


#@utool.cached_func('idf_', appname='smk', key_argx=[1, 2, 3])
@profile
def compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids):
    """
    Returns the inverse-document-frequency weighting for each word

    internals step 2

    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs = smk_debug.testdata_raw_internals1()
    >>> wx_series = invindex.words.index
    >>> idx2_aid = invindex.idx2_daid
    >>> wx2_idf = compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
    >>> print(wx2_idf.shape)
    (8000,)

    #>>> wx2_idxs = invindex.wx2_idxs
    """
    mark, end_ = utool.log_progress('[smk_index] Word IDFs: ', len(wx_series), flushfreq=500, writefreq=50)
    mark(0)
    idx2_aid_values = idx2_aid.values
    wx2_idxs_values = wx2_idxs.values
    #with utool.Timer('method 1'):  # 0.16s
    INDEX_TYPE = np.int32
    idxs_list = [idxs.values.astype(INDEX_TYPE) for idxs in wx2_idxs_values]  # 11%
    aids_list = [idx2_aid_values.take(idxs) if len(idxs) > 0 else [] for idxs in idxs_list]
    nTotalDocs = daids.shape[0]
    nDocsWithWord_list = [len(pd.unique(aids)) for aids in aids_list]  # 68%
    # compute idf half of tf-idf weighting
    idf_list = [np.log(nTotalDocs / nDocsWithWord).astype(FLOAT_TYPE)
                if nDocsWithWord > 0 else 0.0
                for nDocsWithWord in nDocsWithWord_list]  # 17.8 ms   # 13%
    #with utool.Timer('method 2'):  # 5.04s
    #    wx2_idf = {}
    #    for count, wx in enumerate(wx_series):
    #        nDocsWithWord = len(pd.unique(idx2_aid.take(wx2_idxs.get(wx, []))))
    #        idf = 0 if nDocsWithWord == 0 else np.log(nTotalDocs / nDocsWithWord)
    #        wx2_idf[wx] = idf
    end_()
    wx2_idf = pdh.IntSeries(idf_list, index=wx_series, name='idf')
    return wx2_idf


#@utool.cached_func('residuals', appname='smk')
@profile
def compute_residuals_(words, wx2_idxs, idx2_vec, idx2_aid, idx2_fx, aggregate):
    """
    11.8874 seconds

    Computes residual vectors based on word assignments
    returns mapping from word index to a set of residual vectors

    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs = smk_debug.testdata_raw_internals1()
    >>> words     = invindex.words
    >>> idx2_aid  = invindex.idx2_daid
    >>> idx2_fx   = invindex.idx2_dfx
    >>> idx2_vec  = invindex.idx2_dvec
    >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    >>> wx2_rvecs, wx2_aids = compute_residuals_(words, wx2_idxs, idx2_vec, idx2_aid, idx2_fx, aggregate)
    """
    words_values    = words.values
    idx2_aid_values = idx2_aid.values
    idx2_vec_values = idx2_vec.values
    idx2_fx_values  = idx2_fx.values
    wx2_idxs_values = wx2_idxs.values
    wx_sublist      = wx2_idxs.index
    idxs_list  = [idxsdf.values.astype(np.int32) for idxsdf in wx2_idxs_values]   # 13 ms
    # Prealloc output
    if utool.VERBOSE:
        print('[smk_index] Residual Vectors for %d words. aggregate=%r' %
              (len(wx2_idxs), aggregate,))
    # Nonaggregated residuals
    _args1 = (words_values, wx_sublist, idxs_list, idx2_vec_values)
    rvecs_list = smk_speed.compute_nonagg_rvec_listcomp(*_args1)  # 125 ms  11%
    if aggregate:
        # Aggregate over words of the same aid
        #agg_list = smk_speed.compute_agg_rvecs(rvecs_list, idxs_list, idx2_aid)
        #aggaids_list = [tup[0] for tup in agg_list]
        #aggvecs_list = [tup[1] for tup in agg_list]
        #aggidxs_list = [tup[1] for tup in agg_list]
        tup = smk_speed.compute_agg_rvecs(rvecs_list, idxs_list, idx2_aid)  # 38%
        (aggvecs_list, aggaids_list, aggidxs_list) = tup
        aggfxs_list = [[idx2_fx_values.take(idxs) for idxs in aggidxs]
                       for aggidxs in aggidxs_list]
        _args2 = (wx_sublist, aggvecs_list, aggaids_list, aggfxs_list)
        # Make aggregate dataframes
        wx2_aggvecs, wx2_aggaids, wx2_aggfxs = pdh.pandasify_agg_list(*_args2)  # 617 ms  47%
        return wx2_aggvecs, wx2_aggaids, wx2_aggfxs
    else:
        # Make residuals dataframes
        aids_list = [idx2_aid_values.take(idxs) for idxs in idxs_list]
        # compatibility hack
        fxs_list  = [[idx2_fx_values[idx:idx + 1] for idx in idxs]  for idxs in idxs_list]
        _args3 = (wx_sublist, wx2_idxs_values, rvecs_list, aids_list, fxs_list)
        wx2_rvecs, wx2_aids, wx2_fxs = pdh.pandasify_rvecs_list(*_args3)  # 405 ms
        return wx2_rvecs, wx2_aids, wx2_fxs


#@utool.cached_func('gamma', appname='smk', key_argx=[1, 2])
@profile
def compute_data_gamma_(idx2_daid, wx2_rvecs, wx2_aids, wx2_weight, daids,
                        alpha=3, thresh=0):
    """
    5.5 seconds
    Internals step4

    Computes gamma normalization scalar for the database annotations
    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, invindex, wx2_idxs, wx2_idf, wx2_rvecs, wx2_aids = smk_debug.testdata_raw_internals2()
    >>> alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> idx2_daid  = invindex.idx2_daid
    >>> wx2_weight = wx2_idf
    >>> daids      = invindex.daids
    >>> use_cache  = USE_CACHE_GAMMA and False
    >>> daid2_gamma = compute_data_gamma_(idx2_daid, wx2_rvecs, wx2_aids, wx2_weight, daids, use_cache=use_cache)
    """
    # Gropuing by aid and words
    if utool.VERBOSE:
        print('[smk_index] Compute Gamma alpha=%r, thresh=%r: ' % (alpha, thresh))
    wx_series = wx2_rvecs.index
    if utool.VERBOSE:
        mark1, end1_ = utool.log_progress(
            '[smk_index] Gamma Group: ', len(wx_series), flushfreq=100, writefreq=50)
    wx_series = wx2_rvecs.index
    rvecs_values_list = [rvecs.values for rvecs in wx2_rvecs.values]
    aids_list  = pdh.ensure_numpy(wx2_aids)
    daid2_wx2_drvecs = utool.ddict(lambda: utool.ddict(list))
    # Group by daids first and then by word index
    for wx, aids, rvecs in zip(wx_series.values, aids_list, rvecs_values_list):
        for aid, rvec in zip(aids, rvecs):
            daid2_wx2_drvecs[aid][wx].append(rvec)
        # Stack all rvecs from this word
        for aid in set(aids):
            daid2_wx2_drvecs[aid][wx] = np.vstack(daid2_wx2_drvecs[aid][wx])  # 19%

    if utool.VERBOSE:
        end1_()

    # For every daid, compute its gamma using pregrouped rvecs
    # Summation over words for each aid
    if utool.VERBOSE:
        mark2, end2_ = utool.log_progress(
            '[smk_index] Gamma Sum: ', len(daid2_wx2_drvecs), flushfreq=100, writefreq=25)
    wx2_weight_values = wx2_weight.values
    gamma_list = []
    for aid, wxvecs_list in six.iteritems(daid2_wx2_drvecs):
        wx_list = wxvecs_list.keys()
        weight_list = wx2_weight_values[wx_list]
        wx2_weight_values[wx_list]
        rvecs_list = list(wxvecs_list.values())
        assert len(weight_list) == len(rvecs_list), 'one list for each word'
        gamma = smk_core.gamma_summation2(rvecs_list, weight_list, alpha, thresh)  # 66.8 %
        #weight_list = np.ones(weight_list.size)
        gamma_list.append(gamma)

    daid2_gamma = pdh.IntSeries(gamma_list, index=daids, name='gamma')
    if utool.VERBOSE:
        end2_()
    return daid2_gamma


@profile
def compute_query_repr(annots_df, qaid, invindex, aggregate=False, alpha=3, thresh=0):
    """
    26.2054 s
    Gets query read for computations

    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, qaid, invindex = smk_debug.testdata_query_repr()
    >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    >>> alpha     = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh    = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> query_repr = compute_query_repr(annots_df, qaid, invindex, aggregate, alpha, thresh)
    >>> (wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma) = query_repr
    >>> assert smk_debug.check_wx2_rvecs(wx2_qrvecs), 'has nan'
    >>> invindex_dbgstr.invindex_dbgstr(invindex)


    idx2_vec = qfx2_vec
    idx2_aid = qfx2_aid
    idx2_fx = qfx2_qfx
    wx2_idxs = wx2_qfxs1
    """
    if utool.VERBOSE:
        print('[smk_index] Query Repr qaid=%r' % (qaid,))
    wx2_weight = invindex.wx2_weight
    words = invindex.words
    wordflann = invindex.wordflann
    qfx2_vec = annots_df['vecs'][qaid]
    # Assign query to words
    wx2_qfxs1, qfx2_wx = assign_to_words_(wordflann, words, qfx2_vec, idx_name='fx', dense=False)  # 71.9 %
    # Hack to make implementing asmk easier, very redundant
    qfx2_aid = pdh.IntSeries([qaid] * len(qfx2_wx), index=qfx2_wx.index, name='qfx2_aid')
    qfx2_qfx = qfx2_wx.index
    # Compute query residuals
    wx2_qrvecs, wx2_qaids, wx2_qfxs = compute_residuals_(
        words, wx2_qfxs1, qfx2_vec, qfx2_aid, qfx2_qfx, aggregate)  # 24.8
    # Compute query gamma
    if utool.VERBOSE:
        print('[smk_index] Query Gamma alpha=%r, thresh=%r' % (alpha, thresh))
    weight_list = wx2_weight.values[wx2_qrvecs.index.values.astype(np.int32)]
    rvecs_list  = [rvecs.values for rvecs in wx2_qrvecs.values]
    query_gamma = smk_core.gamma_summation2(rvecs_list, weight_list, alpha, thresh)
    assert query_gamma > 0, 'query gamma is not positive!'
    return wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma
