"""
smk index
"""
from __future__ import absolute_import, division, print_function
import ibeis
import six
import vtool
import utool
import numpy as np
import numpy.linalg as npl  # NOQA
import pandas as pd
from vtool import clustering2 as clustertool
from vtool import nearest_neighbors as nntool
import smk_core

VEC_DIM = 128
VEC_COLUMNS  = pd.Int64Index(range(VEC_DIM), name='vec')
KPT_COLUMNS = pd.Index(['xpos', 'ypos', 'a', 'c', 'd', 'theta'], name='kpt')
USE_CACHE_WORDS = not utool.get_flag('--nocache-words')
USE_CACHE_GAMMA = not utool.get_flag('--nocache-gamma')


def ensure_numpy(data):
    return data.values if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) else data


def pandasify_dict1d(dict_, keys, val_name, series_name, dense=True):
    """ Turns dict into heirarchy of series """
    if dense:
        key2_series = pd.Series(
            {key: pd.Series(dict_.get(key, []), name=val_name,)
             for key in keys},
            index=keys, name=series_name)
    else:
        key2_series = pd.Series(
            {key: pd.Series(dict_.get(key), name=val_name,)
             for key in keys},
            index=keys, name=dict_.keys())
    return key2_series


def pandasify_dict2d(dict_, keys, key2_index, columns, series_name):
    """ Turns dict into heirarchy of dataframes """
    key2_df = pd.Series(
        {key: pd.DataFrame(dict_[key], index=key2_index[key], columns=columns,)
         for key in keys},
        index=keys, name=series_name)
    return key2_df


def pandasify_list2d(list_, keys, columns, val_name, series_name):
    """ Turns dict into heirarchy of dataframes """
    key2_df = pd.Series(
        [pd.DataFrame(item,
                      index=pd.Int64Index(np.arange(len(item)), name=val_name),
                      columns=columns,) for item in list_],
        index=keys, name=series_name)
    return key2_df


def make_annot_df(ibs):
    """
    Creates a panda dataframe using an ibeis controller
    >>> from smk_index import *  # NOQA
    >>> ibs = ibeis.opendb('PZ_MTEST')
    >>> annots_df = make_annot_df(ibs)
    """
    aid_list = ibs.get_valid_aids()
    kpts_list = ibs.get_annot_kpts(aid_list)
    vecs_list = ibs.get_annot_desc(aid_list)
    aid_series = pd.Series(aid_list, name='aid')
    kpts_df = pandasify_list2d(kpts_list, aid_series, KPT_COLUMNS, 'fx', 'kpts')
    vecs_df = pandasify_list2d(vecs_list, aid_series, VEC_COLUMNS, 'fx', 'vecs')
    # Pandas Annotation Dataframe
    annots_df = pd.concat([kpts_df, vecs_df], axis=1)
    return annots_df


def learn_visual_words(annots_df, taids, nWords, use_cache=USE_CACHE_WORDS):
    """
    Computes visual words
    >>> from smk_index import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> use_cache = USE_CACHE_WORDS
    >>> words = learn_visual_words(annots_df, taids, nWords)
    """
    train_vecs_df = annots_df['vecs'][taids]
    train_vecs = np.vstack(train_vecs_df.values)
    print('Training %d word vocabulary with %d annots and %d descriptors' %
          (nWords, len(taids), len(train_vecs)))
    _words = clustertool.cached_akmeans(train_vecs, nWords, max_iters=100,
                                        use_cache=use_cache, appname='smk')
    wx_series = pd.Int64Index(np.arange(len(_words)), name='wx')
    words = pd.DataFrame(_words, index=wx_series, columns=VEC_COLUMNS)
    return words


def index_data_annots(annots_df, daids, words, with_internals=True):
    """
    Create inverted index for database annotations
    >>> from smk_index import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = True
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    """
    vecs_list = annots_df['vecs'][daids]
    flann_params = {}
    _words = words.values
    wordflann = nntool.flann_cache(_words, flann_params=flann_params,
                                   appname='smk')
    _daids = ensure_numpy(daids)
    _vecs_list = ensure_numpy(vecs_list)
    _idx2_dvec, _idx2_daid, _idx2_dfx = nntool.invertable_stack(_vecs_list, _daids)

    # Pandasify
    idx_series = pd.Int64Index(np.arange(len(_idx2_daid)), name='idx')
    idx2_dfx   = pd.Series(_idx2_dfx, index=idx_series, name='fx')
    idx2_daid  = pd.Series(_idx2_daid, index=idx_series, name='aid')
    idx2_dvec  = pd.DataFrame(_idx2_dvec, index=idx_series, columns=VEC_COLUMNS)

    invindex = InvertedIndex(words, wordflann, idx2_dvec, idx2_daid, idx2_dfx, daids)
    if with_internals:
        invindex.compute_internals()
    return invindex


def inverted_assignments_(wordflann, words, idx2_vec, idx_name='idx', dense=True):
    """ Assigns vectors to nearest word
    >>> from smk_index import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words, False)
    >>> wordflann = invindex.wordflann
    >>> words = invindex.words
    >>> idx2_vec = invindex.idx2_dvec
    >>> idx_name='idx'
    >>> series_name = 'wx2_idxs'
    >>> dense = True
    >>> wx2_idxs, idx2_wx = inverted_assignments_(wordflann, words, idx2_vec)
    """
    #TODO: multiple assignment
    wx_series  = words.index
    idx_series = idx2_vec.index
    _idx2_vec = ensure_numpy(idx2_vec)
    _idx2_wx, _idx2_wdist = wordflann.nn_index(_idx2_vec, 1)
    # TODO: maybe we can use multiindex here?
    #idx_wx_mindex = pd.MultiIndex.from_arrays(((idx_series.values, _idx2_wx)),
    #                                          names=(idx_name, 'wx'))
    #idx2_wx = pd.Series(_idx2_wx, index=idx_wx_mindex, name='wx')
    idx2_wx = pd.Series(_idx2_wx, index=idx_series, name='wx')
    word_assignments = pd.DataFrame(_idx2_wx, index=idx2_vec.index, columns=['wx'])
    word_group = word_assignments.groupby('wx')
    # TODO Ensure every wx is here.
    # each word should at least have an empty list
    _wx2_idxs = word_group['wx'].indices
    series_name = ('wx2_' + idx_name + 's')
    wx2_idxs = pandasify_dict1d(_wx2_idxs, wx_series, idx_name, series_name, dense=dense)
    return wx2_idxs, idx2_wx


def compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids):
    """ Returns the inverse-document-frequency weighting for each word
    >>> from smk_index import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> wx_series = words.index
    >>> with_internals = False
    >>> invindex  = index_data_annots(annots_df, daids, words, with_internals)
    >>> idx2_aid  = invindex.idx2_daid
    >>> daids     = invindex._daids
    >>> wordflann = invindex.wordflann
    >>> wx2_idxs, idx2_wx = inverted_assignments_(wordflann, words, idx2_vec)
    >>> wx2_idf = compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
    """
    nTotalDocs = daids.shape[0]
    wx2_idf = pd.Series(np.empty(len(wx_series)), index=wx_series, name='idf')
    mark, end_ = utool.log_progress('computing word idfs: ', len(wx_series), flushfreq=500)
    for count, wx in enumerate(wx_series):
        mark(count)
        nDocsWithWord = len(pd.unique(idx2_aid.take(wx2_idxs.get(wx, []))))
        idf = 0 if nDocsWithWord == 0 else np.log(nTotalDocs / nDocsWithWord)
        wx2_idf[wx] = idf
    end_()
    return wx2_idf


def compute_residuals_(words, idx2_vec, wx2_idxs):
    """ Computes residual vectors based on word assignments
    returns mapping from word index to a set of residual vectors
    >>> from smk_index import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> wx_series = words.index
    >>> with_internals = False
    >>> invindex  = index_data_annots(annots_df, daids, words, with_internals)
    >>> idx2_vec  = invindex.idx2_dvec
    >>> daids     = invindex._daids
    >>> wordflann = invindex.wordflann
    >>> wx2_idxs, idx2_wx = inverted_assignments_(wordflann, words, idx2_vec)
    >>> wx2_rvecs = compute_residuals_(words, idx2_vec, wx2_idxs)
    """
    wx_series = words.index
    _words = words.values
    wx2_rvecs = pd.Series(np.empty(len(wx_series), dtype=pd.DataFrame), index=wx_series, name='rvec')
    mark, end_ = utool.log_progress('computing residual: ', len(wx_series), flushfreq=500)
    for count, wx in enumerate(wx_series):
        mark(count)
        # for each word
        idxs = wx2_idxs[wx]
        # Get vecs assigned to it
        vecs = idx2_vec.take(idxs).values.astype(dtype=np.float64)
        word = _words[wx].astype(dtype=np.float64)
        # compute residuals of all vecs assigned to this word
        tiled_words = np.tile(word, (vecs.shape[0], 1))
        residuals = tiled_words - vecs
        # normalize residuals
        residuals_n = vtool.linalg.normalize_rows(residuals)
        #residuals_n[np.isnan(residuals_n)] = 1.0 / VEC_DIM
        rvecs = pd.DataFrame(residuals_n, index=idxs, columns=VEC_COLUMNS,)
        wx2_rvecs[wx] = rvecs
    end_()
    return wx2_rvecs


def compute_internals_(invindex):
    """
    >>> from smk_index import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> compute_internals_(invindex)
    """
    idx2_vec  = invindex.idx2_dvec
    idx2_daid = invindex.idx2_daid
    daids     = invindex._daids
    wx2_idxs, idx2_wx = invindex.inverted_assignments(idx2_vec, idx_name='idx')
    wx2_weight = invindex.compute_word_weights(wx2_idxs)
    wx2_drvecs = invindex.compute_residuals(idx2_vec, wx2_idxs)
    daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight, daids)
    invindex.idx2_wx    = idx2_wx    # stacked index -> word index
    invindex.wx2_idxs   = wx2_idxs
    invindex.wx2_weight = wx2_weight
    invindex.wx2_drvecs = wx2_drvecs
    invindex.daid2_gamma = daid2_gamma


def compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight, daids, use_cache=USE_CACHE_GAMMA):
    """
    >>> from smk_index import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = True
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> idx2_daid  = invindex.idx2_daid
    >>> wx2_drvecs = invindex.wx2_drvecs
    >>> wx2_weight = invindex.wx2_weight
    >>> daids      = invindex._daids
    >>> use_cache  = USE_CACHE_GAMMA
    >>> daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight, daids, use_cache=use_cache)
    """
    hashstr = utool.hashstr(repr(wx2_drvecs) + repr(wx2_weight))
    cache_key = hashstr
    if use_cache:
        try:
            daid2_gamma = utool.global_cache_read(cache_key, appname='smk')
            #print('gamma_dbg cache hit')
            return daid2_gamma
        except Exception:
            pass

    # Gropuing by aid and words
    mark, end_ = utool.log_progress('gamma grouping ',
                                    wx2_drvecs.shape[0],
                                    flushfreq=100)
    daid2_wx2_drvecs = utool.ddict(dict)
    for count, wx in enumerate(wx2_drvecs.index):
        mark(wx)
        group  = wx2_drvecs[wx].groupby(idx2_daid)
        for daid, vecs in group:
            daid2_wx2_drvecs[daid][wx] = vecs.values
    end_()

    # Summation over words for each aid
    mark, end_ = utool.log_progress('computing daid gamma: ', len(daid2_wx2_drvecs),
                                    flushfreq=100)
    daid2_gamma = pd.Series(np.empty(daids.shape[0]), index=daids, name='gamma')
    for count, (daid, wx2_drvecs) in enumerate(six.iteritems(daid2_wx2_drvecs)):
        mark(count)
        wx2_rvecs = wx2_drvecs
        daid2_gamma[daid] = smk_core.gamma_summation(wx2_rvecs, wx2_weight)

    # Cache save
    utool.global_cache_write(cache_key, daid2_gamma, appname='smk')
    return daid2_gamma


@six.add_metaclass(utool.ReloadingMetaclass)
class InvertedIndex(object):
    def __init__(invindex, words, wordflann, idx2_vec, idx2_aid, idx2_fx, _daids):
        invindex.wordflann  = wordflann
        invindex.words      = words     # visual word centroids
        invindex._daids     = _daids    # indexed annotation ids
        invindex.idx2_dvec  = idx2_vec  # stacked index -> descriptor vector
        invindex.idx2_daid  = idx2_aid  # stacked index -> annot id
        invindex.idx2_dfx   = idx2_fx   # stacked index -> feature index
        invindex.wx2_idxs   = None      # word index -> stacked indexes
        invindex.wx2_drvecs = None      # word index -> residual vectors
        invindex.wx2_weight = None      # word index -> idf
        invindex.daid2_gamma = None
        #invindex.compute_internals()

    def get_cfgstr(invindex):
        lbl = 'InvIndex'
        hashstr = utool.hashstr(repr(invindex.wx2_drvecs))
        return '_{lbl}({hashstr})'.format(lbl=lbl, hashstr=hashstr)

    def compute_internals(invindex):
        compute_internals_(invindex)

    def compute_word_weights(invindex, wx2_idxs):
        idx2_aid = invindex.idx2_daid
        _daids   = invindex._daids
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
        use_cache = USE_CACHE_GAMMA
        idx2_daid  = invindex.idx2_daid
        wx2_drvecs = invindex.wx2_drvecs
        wx2_weight = invindex.wx2_weight
        daids      = invindex._daids
        daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_weight,
                                          daids, use_cache=use_cache)
        return daid2_gamma


def compute_query_repr(annots_df, qaid, invindex):
    qfx2_vec = annots_df['vecs'][qaid]
    wx2_qfxs, qfx2_wx = invindex.inverted_assignments(qfx2_vec, idx_name='fx', dense=False)
    wx2_qrvecs = invindex.compute_residuals(qfx2_vec, wx2_qfxs)
    return wx2_qfxs, wx2_qrvecs


def query_inverted_index(annots_df, qaid, invindex):
    #if daid_subset is not None:
    #    idx2_daid = idx2_daid[idx2_daid.isin(daid_subset)]
    #    _daids = _daids[_daids.isin(daid_subset)]
    #    wx2_idxmask_ = {wx: np.in1d(idxs, idx2_daid.index) for wx, idxs in six.iteritems(wx2_idxs)}
    #    wx2_idxs   = {wx: idxs[wx2_idxmask_[wx]]   for wx, idxs in six.iteritems(wx2_idxs)}
    #    wx2_drvecs = {wx: drvecs[wx2_idxmask_[wx]] for wx, drvecs in six.iteritems(wx2_drvecs)}
    wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    daid2_totalscore, daid2_wx2_scoremat = smk_core.match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid)
    daid2_totalscore.sort(axis=1, ascending=False)
    chipmatch = build_chipmatch(daid2_wx2_scoremat, invindex.idx2_dfx)
    return daid2_totalscore, chipmatch


def build_chipmatch(daid2_wx2_scoremat, idx2_dfx):
    daid_fm = {}
    daid_fs = {}
    daid_fk = {}
    mark, end = utool.log_progress('accumulating match info: ', len(daid2_wx2_scoremat), flushfreq=100)
    for count, item in enumerate(daid2_wx2_scoremat.items()):
        daid, wx2_scoremat = item
        mark(count)
        fm_accum = []
        fs_accum = []
        fk_accum = []
        for wx, scoremat in wx2_scoremat.iteritems():
            qfxs = scoremat.index
            dfxs = idx2_dfx[scoremat.columns]
            if len(qfxs) == 0 or len(dfxs) == 0:
                continue
            fm_ = np.vstack(np.dstack(np.meshgrid(qfxs, dfxs, indexing='ij')))
            #try:
            #except Exception as ex:
            #    utool.printex(ex)
            #    utool.embed()
            fs_ = scoremat.values.flatten()
            lower_thresh = 0.01
            valid = [fs_ > lower_thresh]
            fm = fm_[valid]
            fs = fs_[valid]
            fk = np.ones(len(fm), dtype=np.int32)
            fm_accum.append(fm)
            fs_accum.append(fs)
            fk_accum.append(fk)
        daid_fm[daid] = np.vstack(fm_accum)
        daid_fs[daid] = np.hstack(fs_accum).T
        daid_fk[daid] = np.hstack(fk_accum).T
    chipmatch = (daid_fm, daid_fs, daid_fk,)
    return chipmatch
