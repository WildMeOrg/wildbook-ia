"""
smk index
"""
from __future__ import absolute_import, division, print_function
import six
import utool
import numpy as np
import numpy.linalg as npl  # NOQA
import pandas as pd
from vtool import clustering2 as clustertool
from vtool import nearest_neighbors as nntool
from ibeis.model.hots import smk_core
from ibeis.model.hots import pandas_helpers as pdh
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_index]')

FLOAT_TYPE = np.float32
INTEGER_TYPE = np.int32
VEC_TYPE = np.uint8
VEC_DIM = 128
VEC_COLUMNS  = pdh.IntIndex(np.arange(VEC_DIM, dtype=INTEGER_TYPE), name='vec')
KPT_COLUMNS = pd.Index(['xpos', 'ypos', 'a', 'c', 'd', 'theta'], name='kpt')
USE_CACHE_WORDS = not utool.get_flag('--nocache-words')


@six.add_metaclass(utool.ReloadingMetaclass)
class InvertedIndex(object):
    def __init__(invindex, words, wordflann, idx2_vec, idx2_aid, idx2_fx, _daids):
        invindex.wordflann  = wordflann
        invindex.words      = words     # visual word centroids
        invindex.daids     = _daids    # indexed annotation ids
        invindex.idx2_dvec  = idx2_vec  # stacked index -> descriptor vector (currently sift)
        invindex.idx2_daid  = idx2_aid  # stacked index -> annot id
        invindex.idx2_dfx   = idx2_fx   # stacked index -> feature index (wrt daid)
        invindex.wx2_idxs   = None      # word index -> stacked indexes
        invindex.wx2_drvecs = None      # word index -> residual vectors
        invindex.wx2_weight = None      # word index -> idf (wx normalize)
        invindex.daid2_gamma = None     # word index -> gamma (daid normalizer)
        #
        invindex.wx2_aggvecs = None     # word index -> aggregate vectors
        invindex.wx2_aggaids = None     # word index -> aggregate aids
        #invindex.compute_data_internals()

    def get_cfgstr(invindex):
        lbl = 'InvIndex'
        hashstr = utool.hashstr(repr(invindex.wx2_drvecs))
        return '_{lbl}({hashstr})'.format(lbl=lbl, hashstr=hashstr)

    def compute_data_internals(invindex):
        compute_data_internals_(invindex)

    def compute_word_weights(invindex, wx2_idxs):
        idx2_aid = invindex.idx2_daid
        _daids   = invindex.daids
        wx_series = invindex.words.index
        wx2_weight = compute_word_idf_(wx_series, wx2_idxs, idx2_aid, _daids)
        return wx2_weight

    def inverted_assignments(invindex, idx2_vec, idx_name='idx', dense=True):
        wordflann = invindex.wordflann
        words     = invindex.words
        wx2_idxs, idx2_wx = inverted_assignments_(wordflann, words, idx2_vec, idx_name, dense)
        return wx2_idxs, idx2_wx

    def compute_residuals(invindex, idx2_vec, wx2_idxs):
        words = invindex.words
        wx2_rvecs = compute_residuals_(words, idx2_vec, wx2_idxs)
        return wx2_rvecs

    def compute_data_gamma(invindex):
        idx2_daid  = invindex.idx2_daid
        wx2_drvecs = invindex.wx2_drvecs
        wx2_weight = invindex.wx2_weight
        daids      = invindex.daids
        daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight, daids)
        return daid2_gamma


@profile
def make_annot_df(ibs):
    """
    Creates a panda dataframe using an ibeis controller
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> import ibeis
    >>> ibs = ibeis.opendb('PZ_MTEST')
    >>> annots_df = make_annot_df(ibs)

    #>>> from ibeis.model.hots import smk_debug
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
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> use_cache = USE_CACHE_WORDS
    >>> words = learn_visual_words(annots_df, taids, nWords)
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
def index_data_annots(annots_df, daids, words, with_internals=True):
    """
    Create inverted index for database annotations
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = True
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    """
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
        invindex.compute_data_internals()
    return invindex


@profile
def inverted_assignments_(wordflann, words, idx2_vec, idx_name='idx', dense=True):
    """ Assigns vectors to nearest word
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words, False)
    >>> wordflann = invindex.wordflann
    >>> words = invindex.words
    >>> idx2_vec = invindex.idx2_dvec
    >>> idx_name, series_name = 'idx', 'wx2_idxs'
    >>> dense = True
    >>> wx2_idxs, idx2_wx = inverted_assignments_(wordflann, words, idx2_vec)
    """
    #TODO: multiple assignment
    wx_series  = words.index
    idx_series = idx2_vec.index
    _idx2_vec = pdh.ensure_numpy(idx2_vec)
    _idx2_wx, _idx2_wdist = wordflann.nn_index(_idx2_vec, 1)
    # TODO: maybe we can use multiindex here?
    #idx_wx_mindex = pd.MultiIndex.from_arrays(((idx_series.values, _idx2_wx)),
    #                                          names=(idx_name, 'wx'))
    #idx2_wx = pdh.IntSeries(_idx2_wx, index=idx_wx_mindex, name='wx')
    idx2_wx = pdh.IntSeries(_idx2_wx, index=idx_series, name='wx')
    word_assignments = pd.DataFrame(_idx2_wx, index=idx2_vec.index, columns=['wx'])
    word_group = word_assignments.groupby('wx')
    # TODO Ensure every wx is here.
    # each word should at least have an empty list
    _wx2_idxs = word_group['wx'].indices
    series_name = ('wx2_' + idx_name + 's')
    """
    dict_ = _wx2_idxs
    keys = wx_series
    val_name = idx_name
    """
    wx2_idxs = pdh.pandasify_dict1d(_wx2_idxs, wx_series, idx_name, series_name, dense=dense)
    return wx2_idxs, idx2_wx


@utool.cached_func('idf', appname='smk', key_argx=[1, 2, 3])
def compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids):
    """ Returns the inverse-document-frequency weighting for each word
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> wx_series = words.index
    >>> with_internals = False
    >>> invindex  = index_data_annots(annots_df, daids, words, with_internals)
    >>> idx2_aid  = invindex.idx2_daid
    >>> daids     = invindex.daids
    >>> wordflann = invindex.wordflann
    >>> wx2_idxs, idx2_wx = inverted_assignments_(wordflann, words, idx2_vec)
    >>> wx2_idf = compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
    """
    nTotalDocs = daids.shape[0]
    wx2_idf = pdh.IntSeries(np.empty(len(wx_series)), index=wx_series, name='idf')
    mark, end_ = utool.log_progress('computing word idfs: ', len(wx_series), flushfreq=500, writefreq=50)
    for count, wx in enumerate(wx_series):
        mark(count)
        nDocsWithWord = len(pd.unique(idx2_aid.take(wx2_idxs.get(wx, []))))
        idf = 0 if nDocsWithWord == 0 else np.log(nTotalDocs / nDocsWithWord)
        wx2_idf[wx] = idf
    end_()
    return wx2_idf


def normalize_vecs_inplace(vecs):
    # Normalize residuals
    norm_ = npl.norm(vecs, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(vecs, norm_.reshape(norm_.size, 1), out=vecs)


def aggregate_rvecs(rvecs):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> rvecs = (255 * np.random.rand(4, 128)).astype(FLOAT_TYPE)
    """
    if rvecs.shape[0] == 1:
        return rvecs
    rvecs_agg = np.empty((1, rvecs.shape[1]), dtype=rvecs.dtype)
    rvecs.sum(axis=0, out=rvecs_agg[0])
    normalize_vecs_inplace(rvecs_agg)
    return rvecs_agg


def get_norm_rvecs(vecs, word):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> vecs = (255 * np.random.rand(4, 128)).astype(VEC_TYPE)
    >>> word = (255 * np.random.rand(1, 128)).astype(VEC_TYPE)
    """
    # Compute residuals of assigned vectors
    rvecs_n = word.astype(dtype=FLOAT_TYPE) - vecs.astype(dtype=FLOAT_TYPE)
    normalize_vecs_inplace(rvecs_n)
    return rvecs_n


#@profile
@utool.cached_func('residuals', appname='smk')
def compute_residuals_(words, idx2_vec, wx2_idxs, aggregate=False):
    """
    Computes residual vectors based on word assignments
    returns mapping from word index to a set of residual vectors

    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> wx_series = words.index
    >>> with_internals = False
    >>> invindex  = index_data_annots(annots_df, daids, words, with_internals)
    >>> idx2_vec  = invindex.idx2_dvec
    >>> daids     = invindex.daids
    >>> idx2_aid = invindex.idx2_daid
    >>> aggregate = False
    >>> wordflann = invindex.wordflann
    >>> wx2_idxs, idx2_wx = inverted_assignments_(wordflann, words, idx2_vec, dense=False)
    >>> wx2_rvecs = compute_residuals_(words, idx2_vec, wx2_idxs)
    """
    idx2_aid = None
    wx_keys = wx2_idxs.index
    words_values = words.values
    # Prealloc output
    mark, end_ = utool.log_progress('compute residual: ', len(wx_keys), flushfreq=500, writefreq=50)
    # For each word get vecs assigned to it
    wx2_rvecs = pdh.IntSeries(np.empty(len(wx_keys), dtype=pd.DataFrame), index=wx_keys, name='rvec')
    for count, wx in enumerate(wx_keys):
        mark(count)
        idxs = wx2_idxs[wx].values
        vecs = idx2_vec.take(idxs).values
        word = words_values[wx:wx + 1]
        rvecs_n = get_norm_rvecs(vecs, word)
        wx2_rvecs[wx] = pd.DataFrame(rvecs_n, index=idxs, columns=VEC_COLUMNS)
    end_()
    #else:
    #    wx2_aggvecs = pdh.IntSeries(np.empty(len(wx_keys), dtype=pd.DataFrame), index=wx_keys, name='aggvecs')
    #    wx2_aggaids = pdh.IntSeries(np.empty(len(wx_keys), dtype=pd.Series), index=wx_keys, name='aggaids')
    #    for count, wx in enumerate(wx_keys):
    #        mark(count)
    #        idxs = wx2_idxs[wx].values
    #        vecs = idx2_vec.take(idxs).values
    #        word = words_values[wx:wx + 1]
    #        rvecs_n = get_norm_rvecs(vecs, word)
    #        if idx2_aid is not None:
    #            # Need to compute aggregate residual for each aid for each word
    #            aids = pdh.IntSeries(idx2_aid.values[idxs])
    #            group_aids, groupxs = pdh.group_indicies(aids)
    #            group_aggvecs = np.vstack([aggregate_rvecs(rvecs_n[xs]) for xs in groupxs])
    #            wx2_aggvecs[wx] = pd.DataFrame(group_aggvecs, index=group_aids, columns=VEC_COLUMNS)
    #            wx2_aggaids[wx] = group_aids
    #    x = 0
    #    max_ = 0
    #    for count, wx in enumerate(wx_keys):
    #        if len(wx2_aggaids[wx]) != len(wx2_idxs[wx]):
    #            print(wx)
    #            max__ = len(wx2_idxs[wx])
    #            if max__ > max_:
    #                max_ = max__
    #                x = wx
    if aggregate:
        wx2_aggvecs = pdh.IntSeries(np.empty(len(wx_keys), dtype=pd.DataFrame), index=wx_keys, name='aggvecs')
        wx2_aggaids = pdh.IntSeries(np.empty(len(wx_keys), dtype=pd.Series), index=wx_keys, name='aggaids')
        for wx in wx_keys:
            rvecs_n = wx2_rvecs[wx].values
            idxs = wx2_idxs[wx].values
            aids = pdh.IntSeries(idx2_aid.values[idxs], name='aids')
            group_aids, groupxs = pdh.group_indicies(aids)
            group_aggvecs = np.vstack([aggregate_rvecs(rvecs_n[xs]) for xs in groupxs])
            wx2_aggvecs[wx] = pd.DataFrame(group_aggvecs, index=group_aids, columns=VEC_COLUMNS)
            wx2_aggaids[wx] = group_aids
    return wx2_rvecs


@utool.cached_func('gamma', appname='smk', key_argx=[1, 2])
def compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight, daids):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = True
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> idx2_daid  = invindex.idx2_daid
    >>> wx2_drvecs = invindex.wx2_drvecs
    >>> wx2_weight = invindex.wx2_weight
    >>> daids      = invindex.daids
    >>> use_cache  = USE_CACHE_GAMMA and False
    >>> daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight, daids, use_cache=use_cache)
    """
    #cfgstr = utool.hashstr(repr(wx2_drvecs) + repr(wx2_weight))
    #cacher  = utool.Cacher('gamma', cfgstr=cfgstr, appname='smk')
    #data    = cacher.tryload()
    #if data is not None:
    #    return data
    # Gropuing by aid and words
    mark1, end1_ = utool.log_progress(
        'data gamma grouping: ', len(wx2_drvecs), flushfreq=100, writefreq=50)
    daid2_wx2_drvecs = utool.ddict(dict)
    for count, wx in enumerate(wx2_drvecs.index):
        mark1(count)
        group  = wx2_drvecs[wx].groupby(idx2_daid)
        for daid, vecs in group:
            daid2_wx2_drvecs[daid][wx] = vecs
    end1_()
    # Summation over words for each aid
    mark2, end2_ = utool.log_progress(
        'computing data gamma: ', len(daid2_wx2_drvecs), flushfreq=100, writefreq=25)
    daid2_gamma = pdh.IntSeries(np.empty(daids.shape[0]), index=daids, name='gamma')
    for count, (daid, wx2_drvecs) in enumerate(six.iteritems(daid2_wx2_drvecs)):
        mark2(count)
        wx2_rvecs = wx2_drvecs
        daid2_gamma[daid] = smk_core.gamma_summation(wx2_rvecs, wx2_weight)
    end2_()
    # Cache save
    #cacher.save(daid2_gamma)
    return daid2_gamma


@profile
def compute_data_internals_(invindex):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> compute_data_internals_(invindex)
    """
    idx2_vec  = invindex.idx2_dvec
    idx2_daid = invindex.idx2_daid
    daids     = invindex.daids
    wx2_idxs, idx2_wx = invindex.inverted_assignments(idx2_vec, idx_name='idx', dense=True)
    wx2_weight = invindex.compute_word_weights(wx2_idxs)
    wx2_drvecs = invindex.compute_residuals(idx2_vec, wx2_idxs)
    daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight, daids)
    invindex.idx2_wx    = idx2_wx    # stacked index -> word index
    invindex.wx2_idxs   = wx2_idxs
    invindex.wx2_weight = wx2_weight
    invindex.wx2_drvecs = wx2_drvecs
    invindex.daid2_gamma = daid2_gamma


@profile
def compute_query_repr(annots_df, qaid, invindex):
    """
    Gets query read for computations

    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)

    wordflann = invindex.wordflann

    idx_name  = 'fx'
    dense     = False
    idx2_vec = qfx2_vec
    wx2_idxs = wx2_qfxs
    """
    qfx2_vec = annots_df['vecs'][qaid]
    wx2_qfxs, qfx2_wx = invindex.inverted_assignments(qfx2_vec, idx_name='fx', dense=False)
    wx2_qrvecs = invindex.compute_residuals(qfx2_vec, wx2_qfxs)
    return wx2_qfxs, wx2_qrvecs


@profile
def query_inverted_index(annots_df, qaid, invindex, withinfo=True):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk_debug
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    >>> assert smk_debug.check_wx2_rvecs(wx2_qrvecs)
    >>> query_inverted_index(annots_df, qaid, invindex)
    >>> withinfo = False
    >>> daid2_totalscore = query_inverted_index(annots_df, qaid, invindex, withinfo=withinfo)
    """
    # Get query words / residuals
    wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    # Compute match kernel for all database aids
    match_kernel = utool.cached_func('match_kernel', appname='smk', key_argx=list(range(5)))(smk_core.match_kernel)
    daid2_totalscore, daid2_wx2_scoremat = match_kernel(
        wx2_qrvecs, wx2_qfxs, invindex, qaid, withinfo)
    # Build chipmatches if daid2_wx2_scoremat is not None
    if withinfo:
        assert daid2_wx2_scoremat is not None
        chipmatch = build_chipmatch(daid2_wx2_scoremat, invindex.idx2_dfx)
        return daid2_totalscore, chipmatch
    else:
        return daid2_totalscore


#@profile
#def convert_scoremat_to_fmfsfk(scoremat):
#    scoremat_column_values = scoremat.columns.values
#    qfxs = scoremat.index.values
#    dfxs = idx2_dfx_values.take(scoremat_column_values)
#    if len(qfxs) == 0 or len(dfxs) == 0:
#        continue
#    fm_ = np.dstack(np.meshgrid(qfxs, dfxs, indexing='ij')).reshape((qfxs.size * dfxs.size, 2))
#    #fm_.shape =
#    scoremat_values = scoremat.values
#    fs_ = scoremat_values.flatten()
#    if scoremat_values.shape[0] > 1 and scoremat_values.shape[1] > 1:
#        break
#    thresh = 0.001
#    valid = fs_ > thresh
#    fm = fm_[valid]
#    fs = fs_[valid]
#    fk = np.ones(len(fm), dtype=np.int32)


@profile
def build_chipmatch(daid2_wx2_scoremat, idx2_dfx, thresh=0):
    """
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    >>> withinfo = True
    >>> daid2_totalscore, daid2_wx2_scoremat = smk_core.match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid, withinfo)
    >>> idx2_dfx = invindex.idx2_dfx
    >>> out = build_chipmatch(daid2_wx2_scoremat, idx2_dfx)

    #if CYTH
    cdef:
        dict daid_fm, daid_fs, daid_fk
        tuple item, chipmatch
        object scoremat
        list fm_accum, fs_accum, fk_accum
        Py_ssize_t count
        np.ndarray[np.int64_t, ndim=2] fm, fm_
        np.ndarray[np.float64_t, ndim=1] fs_, fs
        np.ndarray[np.int32_t, ndim=1] fk
        np.ndarray[np.int64_t, ndim=1] qfxs
        np.ndarray[int, ndim=1] dfxs
        np.ndarray[np.int64_t, ndim=1] scoremat_column_values
        np.ndarray[np.float64_t, ndim=2] scoremat_values
        np.ndarray[np.uint8_t, cast=True] valid
        np.float64_t thresh
    #endif
    """
    daid_fm = {}
    daid_fs = {}
    daid_fk = {}
    "#CYTH: exclude benchmark"
    mark, end_ = utool.log_progress('accumulating match info: ',
                                    len(daid2_wx2_scoremat), flushfreq=100,
                                    writefreq=25)
    idx2_dfx_values = idx2_dfx.values
    for count, item in enumerate(six.iteritems(daid2_wx2_scoremat)):
        daid, wx2_scoremat = item
        #mark(count)
        fm_accum = []
        fs_accum = []
        fk_accum = []
        for wx, scoremat in wx2_scoremat.iteritems():
            if scoremat.shape[0] > 1 and scoremat.shape[1] > 1:
                break
            scoremat_column_values = scoremat.columns.values
            qfxs = scoremat.index.values
            dfxs = idx2_dfx_values.take(scoremat_column_values)
            fm_ = np.dstack(np.meshgrid(qfxs, dfxs, indexing='ij')).reshape((qfxs.size * dfxs.size, 2))
            scoremat_values = scoremat.values
            fs_ = scoremat_values.flatten()
            thresh = 0.001
            valid = fs_ > thresh
            fm = fm_[valid]
            fs = fs_[valid]
            fk = np.ones(len(fm), dtype=INTEGER_TYPE)
            fm_accum.append(fm)
            fs_accum.append(fs)
            fk_accum.append(fk)
        daid_fm[daid] = np.vstack(fm_accum)
        daid_fs[daid] = np.hstack(fs_accum).T
        daid_fk[daid] = np.hstack(fk_accum).T
    end_()
    chipmatch = (daid_fm, daid_fs, daid_fk,)
    return chipmatch


@profile
def query_smk(ibs, annots_df, invindex, qreq_):
    """
    ibeis interface
    """
    from ibeis.model.hots import pipeline
    qaids = qreq_.get_external_qaids()
    qaid2_chipmatch = {}
    qaid2_scores = {}
    mark, end_ = utool.log_progress('query: ', len(qaids), flushfreq=1,
                                    writefreq=1, with_totaltime=True,
                                    backspace=False)
    for count, qaid in enumerate(qaids):
        mark(count)
        daid2_score, chipmatch = query_inverted_index(
            annots_df, qaid, invindex, withinfo=True)
        qaid2_scores[qaid] = daid2_score
        qaid2_chipmatch[qaid] = chipmatch
    end_()
    qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, {}, qreq_)
    #,
    #qaid2_scores=qaid2_scores)
    return qaid2_qres_

import cyth
if cyth.DYNAMIC:
    exec(cyth.import_cyth_execstr(__name__))
else:
    pass
    # <AUTOGEN_CYTH>
    # </AUTOGEN_CYTH>
