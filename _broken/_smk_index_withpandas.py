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
from ibeis.model.hots.smk.hstypes import INTEGER_TYPE, FLOAT_TYPE, INDEX_TYPE
from ibeis.model.hots.smk.pandas_helpers import VEC_COLUMNS, KPT_COLUMNS
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_index]')

USE_CACHE_WORDS = not utool.get_argflag('--nocache-words')
#WITH_PANDAS = True
WITH_PANDAS = False


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
        invindex.idx2_fweight = None    # stacked index -> feature weight

    #def get_cfgstr(invindex):
    #    lbl = 'InvIndex'
    #    hashstr = utool.hashstr(repr(invindex.wx2_drvecs))
    #    return '_{lbl}({hashstr})'.format(lbl=lbl, hashstr=hashstr)


#@profile
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
    aid_list = ibs.get_valid_aids()  # 80us
    kpts_list = ibs.get_annot_kpts(aid_list)  # 40ms
    vecs_list = ibs.get_annot_desc(aid_list)  # 50ms
    aid_series = pdh.IntSeries(np.array(aid_list, dtype=INTEGER_TYPE), name='aid')
    kpts_df = pdh.pandasify_list2d(kpts_list, aid_series, KPT_COLUMNS, 'fx', 'kpts')  # 6.7ms
    vecs_df = pdh.pandasify_list2d(vecs_list, aid_series, VEC_COLUMNS, 'fx', 'vecs')  # 7.1ms
    # Pandas Annotation Dataframe
    annots_df = pd.concat([kpts_df, vecs_df], axis=1)  # 845 us
    return annots_df


#@profile
def learn_visual_words(annots_df, taids, nWords, use_cache=USE_CACHE_WORDS,
                       with_pandas=WITH_PANDAS):
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
    max_iters = 200
    flann_params = {}
    train_vecs_list = [pdh.ensure_values(vecs) for vecs in annots_df['vecs'][taids].values]
    train_vecs = np.vstack(train_vecs_list)
    print('Training %d word vocabulary with %d annots and %d descriptors' %
          (nWords, len(taids), len(train_vecs)))
    kwds = dict(max_iters=max_iters, use_cache=use_cache, appname='smk',
                flann_params=flann_params)
    _words = clustertool.cached_akmeans(train_vecs, nWords, **kwds)
    if with_pandas:
        # Pandasify
        wx_series = pdh.RangeIndex(len(_words), name='wx')
        #words = pd.DataFrame(_words, index=wx_series, columns=VEC_COLUMNS)
        words = pd.DataFrame(_words, index=wx_series)
    else:
        words = _words
    return words


def index_data_annots(annots_df, daids, words, with_internals=True,
                      aggregate=False, alpha=3, thresh=0, with_pandas=WITH_PANDAS):
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
    if utool.VERBOSE:
        print('[smk_index] index_data_annots')
    flann_params = {}
    _words = pdh.ensure_values(words)
    wordflann = nntool.flann_cache(_words, flann_params=flann_params,
                                   appname='smk')
    _daids = pdh.ensure_values(daids)
    _vecs_list = pdh.ensure_2d_values(annots_df['vecs'][_daids])
    _idx2_dvec, _idx2_daid, _idx2_dfx = nntool.invertable_stack(_vecs_list, _daids)

    # Pandasify
    if with_pandas:
        idx_series = pdh.IntIndex(np.arange(len(_idx2_daid)), name='idx')
        idx2_dfx   = pdh.IntSeries(_idx2_dfx, index=idx_series, name='fx')
        idx2_daid  = pdh.IntSeries(_idx2_daid, index=idx_series, name='aid')
        idx2_dvec  = pd.DataFrame(_idx2_dvec, index=idx_series, columns=VEC_COLUMNS)
    else:
        idx2_dfx = _idx2_dfx
        idx2_daid = _idx2_daid
        idx2_dvec = _idx2_dvec
        pass

    invindex = InvertedIndex(words, wordflann, idx2_dvec, idx2_daid, idx2_dfx, daids)
    if with_internals:
        compute_data_internals_(invindex, aggregate, alpha, thresh)  # 99%
    return invindex


@profile
def compute_data_internals_(invindex, aggregate=False, alpha=3, thresh=0):
    """

    Builds each of the inverted index internals.

    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_raw_internals0()
    >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    >>> alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> compute_data_internals_(invindex, aggregate, alpha, thresh)

    idx2_vec = idx2_dvec
    """
    if utool.VERBOSE:
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
    if WITH_PANDAS:
        wx_series = invindex.words.index
    else:
        wx_series = np.arange(len(words))
    # Compute word assignments
    wx2_idxs, idx2_wx = assign_to_words_(wordflann, words, idx2_vec, idx_name='idx', dense=True)  # 7.1 %
    # Compute word weights
    wx2_weight = compute_word_idf_(
        wx_series, wx2_idxs, idx2_daid, daids)  # 2.4 %
    # Compute residual vectors and inverse mappings
    wx2_drvecs, wx2_aids, wx2_fxs = compute_residuals_(
        words, wx2_idxs, idx2_vec, idx2_daid, idx2_dfx, aggregate)  # 42.9%
    # Compute annotation normalization factor
    daid2_gamma = compute_data_gamma_(idx2_daid, wx2_drvecs, wx2_aids,
                                      wx2_weight, alpha, thresh)  # 47.5%
    assert len(daid2_gamma) == len(daids)
    # Store information
    invindex.idx2_wx    = idx2_wx    # stacked index -> word index
    invindex.wx2_idxs   = wx2_idxs
    invindex.wx2_weight = wx2_weight
    invindex.wx2_drvecs = wx2_drvecs
    invindex.wx2_aids   = wx2_aids  # needed for asmk
    invindex.wx2_fxs    = wx2_fxs   # needed for asmk
    invindex.daid2_gamma = daid2_gamma

    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_invindex_wx2(invindex)


@profile
def assign_to_words_(wordflann, words, idx2_vec, idx_name='idx', dense=True,
                     nAssign=1, with_pandas=WITH_PANDAS):
    """
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
    idx2_vec_values = pdh.ensure_values(idx2_vec)
    # Find each vectors nearest word
    #TODO: multiple assignment
    _idx2_wx, _idx2_wdist = wordflann.nn_index(idx2_vec_values, nAssign)
    if nAssign > 1:
        #((words[_idx2_wx[:,0]].astype(np.float64) - idx2_vec_values.astype(np.float64)) ** 2).sum(axis=0)
        #_idx2_wdist[:,0]
        #np.sqrt(((words[_idx2_wx[:,0]].astype(np.float64) - idx2_vec_values.astype(np.float64)) ** 2).sum(axis=0))
        # mutli assignment filtering as in
        # http://lear.inrialpes.fr/pubs/2010/JDS10a/jegou_improvingbof_preprint.pdf
        alpha = 1.2
        thresh = alpha * _idx2_wdist.T[0:1].T
        invalid = _idx2_wdist >= thresh
        # Weighting as in Lost in Quantization
        sigma = 80
        unnorm_weight = np.exp(np.divide(-_idx2_wdist.astype(np.float64), 2 * (sigma ** 2)))
        masked_weight = np.ma.masked_array(unnorm_weight, mask=invalid)
        weight = masked_weight / masked_weight.sum(axis=1)[:, np.newaxis]
        masked_wxs = np.ma.masked_array(_idx2_wx, mask=invalid)
        idx2_wxs = map(utool.filter_Nones, masked_wxs.tolist())
        idx2_wx_weights = map(utool.filter_Nones, weight.tolist())

        #masked_weight1 = np.ma.masked_array(_idx2_wdist, mask=invalid)
        #weight1 = masked_weight1 / masked_weight1.sum(axis=1)[:, np.newaxis]

    # multiple assignment weight: expt(-(d ** 2) / (2 * sigma ** 2))
    # The distance d_0 is used to filter asignments with distance less than
    # alpha * d_0 where alpha = 1.2
    PANDAS_GROUP = True or with_pandas
    # Compute inverted index
    if PANDAS_GROUP:
        # Pandas grouping seems to be faster in this instance
        word_assignments = pd.DataFrame(_idx2_wx, columns=['wx'])  # 141 us
        word_group = word_assignments.groupby('wx')  # 34.5 us
        _wx2_idxs = word_group['wx'].indices  # 8.6 us
    else:
        idx2_idx = np.arange(len(idx2_vec))
        wx_list, groupxs = smk_speed.group_indicies(_idx2_wx)  # 5.52 ms
        idxs_list = smk_speed.apply_grouping(idx2_idx, groupxs)  # 2.9 ms
        _wx2_idxs = dict(zip(wx_list, idxs_list))  # 753 us
    #
    if with_pandas:
        idx_series = pdh.ensure_index(idx2_vec)
        wx_series  = pdh.ensure_index(words)
        wx2_idxs = pdh.pandasify_dict1d(
            _wx2_idxs, wx_series, idx_name, ('wx2_' + idx_name + 's'), dense=dense)  # 274 ms 97.4 %
        idx2_wx = pdh.IntSeries(_idx2_wx, index=idx_series, name='wx')
    else:
        if dense:
            wx2_idxs = {
                wx: _wx2_idxs[wx].astype(INDEX_TYPE)
                if wx in _wx2_idxs else
                np.empty(0, dtype=INDEX_TYPE)
                for wx in range(len(words))
            }
            #wx2_idxs = _wx2_idxs
            #for wx in range(len(words)):
            #    if wx not in wx2_idxs:
            #        wx2_idxs[wx] = np.empty(0, dtype=INDEX_TYPE)
        else:
            wx2_idxs = _wx2_idxs
        idx2_wx  = _idx2_wx
    return wx2_idxs, idx2_wx


#@utool.cached_func('idf_', appname='smk', key_argx=[1, 2, 3])
@profile
def compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids, with_pandas=WITH_PANDAS):
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
    if utool.VERBOSE:
        mark, end_ = utool.log_progress('[smk_index] Word IDFs: ', len(wx_series), flushfreq=500, writefreq=50)
        mark(0)
    wx_series_values = pdh.ensure_values(wx_series)
    idx2_aid_values = pdh.ensure_values(idx2_aid)
    wx2_idxs_values = pdh.ensure_values_subset(wx2_idxs, wx_series_values)
    #with utool.Timer('method 1'):  # 0.16s
    idxs_list = [pdh.ensure_values(idxs).astype(INDEX_TYPE) for idxs in wx2_idxs_values]  # 11%
    aids_list = [idx2_aid_values.take(idxs) if len(idxs) > 0 else [] for idxs in idxs_list]
    nTotalDocs = len(daids)
    nDocsWithWord_list = [len(set(aids)) for aids in aids_list]  # 68%
    # compute idf half of tf-idf weighting
    idf_list = [np.log(nTotalDocs / nDocsWithWord).astype(FLOAT_TYPE)
                if nDocsWithWord > 0 else 0.0
                for nDocsWithWord in nDocsWithWord_list]  # 17.8 ms   # 13%
    if utool.VERBOSE:
        end_()
    if with_pandas:
        wx2_idf = pdh.IntSeries(idf_list, index=wx_series, name='idf')
    else:
        wx2_idf = dict(zip(wx_series_values, idf_list))
    return wx2_idf


#@utool.cached_func('residuals', appname='smk')
@profile
def compute_residuals_(words, wx2_idxs, idx2_vec, idx2_aid, idx2_fx, aggregate,
                       with_pandas=WITH_PANDAS):
    """
    Computes residual vectors based on word assignments
    returns mapping from word index to a set of residual vectors

    Output:
        wx2_rvecs - [ ... [ rvec_i1, ...,  rvec_Mi ]_i ... ]
        wx2_aids  - [ ... [  aid_i1, ...,   aid_Mi ]_i ... ]
        wx2_fxs   - [ ... [[fxs]_i1, ..., [fxs]_Mi ]_i ... ]

    For every word:
        * list of aggvecs
        For every aggvec:
            * one parent aid, if aggregate is False: assert isunique(aids)
            * list of parent fxs, if aggregate is True: assert len(fxs) == 1

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
    words_values    = pdh.ensure_values(words)
    idx2_aid_values = pdh.ensure_values(idx2_aid)
    idx2_vec_values = pdh.ensure_values(idx2_vec)
    idx2_fx_values  = pdh.ensure_values(idx2_fx)
    wx_sublist      = pdh.ensure_index(wx2_idxs)
    # Build lists w.r.t. words
    idxs_list = pdh.ensure_values_subset(wx2_idxs, wx_sublist)
    aids_list = [idx2_aid_values.take(idxs) for idxs in idxs_list]
    #wx2_idxs_values = pdh.ensure_values_subset(wx2_idxs, wx_sublist)
    #idxs_list  = [pdh.ensure_values(idxsdf).astype(INDEX_TYPE) for idxsdf in wx2_idxs_values]   # 13 ms
    if utool.DEBUG2:
        #assert np.all(np.diff(wx_sublist) == 1), 'not dense'
        assert all([len(a) == len(b) for a, b in zip(idxs_list, aids_list)]), 'bad alignment'
        assert idx2_vec_values.shape[0] == idx2_fx_values.shape[0]
        assert idx2_vec_values.shape[0] == idx2_aid_values.shape[0]
    # Prealloc output
    if utool.VERBOSE:
        print('[smk_index] Residual Vectors for %d words. aggregate=%r' %
              (len(wx2_idxs), aggregate,))
    # Nonaggregated residuals
    #_args1 = (words_values, wx_sublist, idxs_list, idx2_vec_values)
    #rvecs_list = smk_speed.compute_nonagg_rvec_listcomp(*_args1)  # 125 ms  11%
    words_list = [words_values[wx:wx + 1] for wx in wx_sublist]  # 1 ms
    vecs_list  = [idx2_vec_values.take(idxs, axis=0) for idxs in idxs_list]  # 5.3 ms
    rvecs_list = [smk_core.get_norm_rvecs(vecs, word)
                  for vecs, word in zip(vecs_list, words_list)]  # 103 ms  # 90%
    if aggregate:
        # Aggregate over words of the same aid
        tup = smk_speed.compute_agg_rvecs(rvecs_list, idxs_list, aids_list)  # 38%
        (aggvecs_list, aggaids_list, aggidxs_list) = tup
        aggfxs_list = [[idx2_fx_values.take(idxs) for idxs in aggidxs]
                       for aggidxs in aggidxs_list]
        if with_pandas:
            _args2 = (wx_sublist, aggvecs_list, aggaids_list, aggfxs_list)
            # Make aggregate dataframes
            wx2_aggvecs, wx2_aggaids, wx2_aggfxs = pdh.pandasify_agg_list(*_args2)  # 617 ms  47%
        else:
            wx2_aggvecs = {wx: aggvecs for wx, aggvecs in zip(wx_sublist, aggvecs_list)}
            wx2_aggaids = {wx: aggaids for wx, aggaids in zip(wx_sublist, aggaids_list)}
            wx2_aggfxs  = {wx: aggfxs  for wx, aggfxs  in zip(wx_sublist, aggfxs_list)}
            if utool.DEBUG2:
                from ibeis.model.hots.smk import smk_debug
                smk_debug.check_wx2(words, wx2_aggvecs, wx2_aggaids, wx2_aggfxs)

        return wx2_aggvecs, wx2_aggaids, wx2_aggfxs
    else:
        # Make residuals dataframes
        # compatibility hack
        fxs_list  = [[idx2_fx_values[idx:idx + 1] for idx in idxs]  for idxs in idxs_list]
        if with_pandas:
            _args3 = (wx_sublist, idxs_list, rvecs_list, aids_list, fxs_list)
            wx2_rvecs, wx2_aids, wx2_fxs = pdh.pandasify_rvecs_list(*_args3)  # 405 ms
        else:
            wx2_rvecs = {wx: rvecs for wx, rvecs in zip(wx_sublist, rvecs_list)}
            wx2_aids  = {wx: aids  for wx, aids  in zip(wx_sublist, aids_list)}
            wx2_fxs   = {wx: fxs   for wx, fxs   in zip(wx_sublist, fxs_list)}
        if utool.DEBUG2:
            from ibeis.model.hots.smk import smk_debug
            smk_debug.check_wx2(words, wx2_rvecs, wx2_aids, wx2_fxs)
        return wx2_rvecs, wx2_aids, wx2_fxs


#@utool.cached_func('gamma', appname='smk', key_argx=[1, 2])
@profile
def compute_data_gamma_(idx2_daid, wx2_rvecs, wx2_aids, wx2_weight,
                        alpha=3, thresh=0):
    """
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
    wx_sublist = pdh.ensure_values(pdh.ensure_index(wx2_rvecs))
    if utool.VERBOSE:
        print('[smk_index] Compute Gamma alpha=%r, thresh=%r: ' % (alpha, thresh))
        mark1, end1_ = utool.log_progress(
            '[smk_index] Gamma Group: ', len(wx_sublist), flushfreq=100, writefreq=50)
    rvecs_list1 = pdh.ensure_values_subset(wx2_rvecs, wx_sublist)
    aids_list  = pdh.ensure_values_subset(wx2_aids, wx_sublist)
    daid2_wx2_drvecs = utool.ddict(lambda: utool.ddict(list))
    # Group by daids first and then by word index
    for wx, aids, rvecs in zip(wx_sublist, aids_list, rvecs_list1):
        group_aids, groupxs = smk_speed.group_indicies(aids)
        rvecs_group = smk_speed.apply_grouping(rvecs, groupxs)  # 2.9 ms
        for aid, rvecs_ in zip(group_aids, rvecs_group):
            daid2_wx2_drvecs[aid][wx] = rvecs_

    if utool.VERBOSE:
        end1_()

    # For every daid, compute its gamma using pregrouped rvecs
    # Summation over words for each aid
    if utool.VERBOSE:
        mark2, end2_ = utool.log_progress(
            '[smk_index] Gamma Sum: ', len(daid2_wx2_drvecs), flushfreq=100, writefreq=25)

    aid_list          = list(daid2_wx2_drvecs.keys())
    wx2_aidrvecs_list = list(daid2_wx2_drvecs.values())
    aidwxs_list    = [list(wx2_aidrvecs.keys()) for wx2_aidrvecs in wx2_aidrvecs_list]
    aidrvecs_list  = [list(wx2_aidrvecs.values()) for wx2_aidrvecs in wx2_aidrvecs_list]
    aidweight_list = [[wx2_weight[wx] for wx in aidwxs] for aidwxs in aidwxs_list]

    #gamma_list = []
    #for weight_list, rvecs_list in zip(aidweight_list, aidrvecs_list):
    #    assert len(weight_list) == len(rvecs_list), 'one list for each word'
    #    gamma = smk_core.gamma_summation2(rvecs_list, weight_list, alpha, thresh)  # 66.8 %
    #    #weight_list = np.ones(weight_list.size)
    #    gamma_list.append(gamma)
    gamma_list = [smk_core.gamma_summation2(rvecs_list, weight_list, alpha, thresh)
                  for weight_list, rvecs_list in zip(aidweight_list, aidrvecs_list)]

    daid2_gamma = pdh.IntSeries(gamma_list, index=aid_list, name='gamma')
    if utool.VERBOSE:
        end2_()
    return daid2_gamma


@profile
def compute_query_repr(annots_df, qaid, invindex, aggregate=False, alpha=3, thresh=0):
    """
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
    if WITH_PANDAS:
        qfx2_vec = annots_df['vecs'][qaid]
    else:
        qfx2_vec = pdh.ensure_values(annots_df['vecs'][qaid])
    # Assign query to words
    wx2_qfxs1, qfx2_wx = assign_to_words_(wordflann, words, qfx2_vec,
                                          idx_name='fx', dense=False)  # 71.9 %
    # Hack to make implementing asmk easier, very redundant
    #qfx2_aid = pdh.IntSeries([qaid] * len(qfx2_wx), index=qfx2_wx.index, name='qfx2_aid')
    #qfx2_aid = pdh.IntSeries([qaid] * len(qfx2_wx), name='qfx2_aid')
    qfx2_aid = np.array([qaid] * len(qfx2_wx), dtype=INTEGER_TYPE)
    if WITH_PANDAS:
        qfx2_qfx = qfx2_vec.index
    else:
        qfx2_qfx = np.arange(len(qfx2_vec))
    # Compute query residuals
    wx2_qrvecs, wx2_qaids, wx2_qfxs = compute_residuals_(
        words, wx2_qfxs1, qfx2_vec, qfx2_aid, qfx2_qfx, aggregate)  # 24.8
    # Compute query gamma
    if utool.VERBOSE:
        print('[smk_index] Query Gamma alpha=%r, thresh=%r' % (alpha, thresh))
    wx_sublist = pdh.ensure_index(wx2_qrvecs).astype(np.int32)
    weight_list = pdh.ensure_values_subset(wx2_weight, wx_sublist)
    rvecs_list  = pdh.ensure_values_subset(wx2_qrvecs, wx_sublist)
    query_gamma = smk_core.gamma_summation2(rvecs_list, weight_list, alpha, thresh)
    assert query_gamma > 0, 'query gamma is not positive!'
    return wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma
