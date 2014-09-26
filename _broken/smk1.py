"""
Todo:
    * Implement correct cfgstrs based on algorithm input
    for cached computations.

    * Go pandas all the way
Issues:
    * errors when there is a word without any database vectors.
    currently a weight of zero is hacked in

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
from plottool import draw_func2 as df2
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 7)
pd.set_option('display.max_columns', 7)
pd.set_option('isplay.notebook_repr_html', True)


VEC_DIM = 128
VEC_COLUMNS = pd.Int64Index(range(VEC_DIM), name='vec')


def pandasify_dict1d(dict_, keys, val_name, series_name):
    """ Turns dict into heirarchy of series """
    key2_series = pd.Series(
        {key: pd.Series(dict_[key], name=val_name,)
         for key in keys},
        index=keys, name=series_name)
    return key2_series


def pandasify_dict2d(dict_, keys, key2_index, columns, series_name):
    """ Turns dict into heirarchy of dataframes """
    key2_df = pd.Series(
        {key: pd.DataFrame(dict_[key], index=key2_index[key], columns=columns,)
         for key in keys},
        index=keys, name=series_name)
    return key2_df


def make_annot_df(ibs):
    """
    Creates a panda dataframe using an ibeis controller
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs = ibeis.opendb('PZ_MTEST')
    >>> annots_df = make_annot_df(ibs)
    """
    aid_list = ibs.get_valid_aids()
    kpts_list = ibs.get_annot_kpts(aid_list)
    vecs_list = ibs.get_annot_desc(aid_list)
    aid_series = pd.Series(aid_list, name='aid')
    # TODO: this could be more pandas
    kpts_df = pd.DataFrame(kpts_list, index=aid_series, columns=['kpts'])
    vecs_df = pd.DataFrame(vecs_list, index=aid_series, columns=['vecs'])
    # Pandas Annotation Dataframe
    annots_df = pd.concat([kpts_df, vecs_df], axis=1)
    return annots_df


def learn_visual_words(annots_df, taids, nWords, use_cache=True):
    """
    Computes visual words
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    """
    vecs_list = annots_df['vecs'][taids].as_matrix()
    train_vecs = np.vstack(vecs_list)
    print('Training %d word vocabulary with %d annots and %d descriptors' %
          (nWords, len(taids), len(train_vecs)))
    cache_dir = utool.get_app_resource_dir('smk')
    words = clustertool.cached_akmeans(train_vecs, nWords, max_iters=100,
                                       use_cache=use_cache, cache_dir=cache_dir)
    return words


def index_data_annots(annots_df, daids, words, with_internals=True):
    """
    Create inverted index for database annotations
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = True
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    """
    vecs_list = ensure_values(annots_df['vecs'][daids])
    flann_params = {}
    cache_dir = utool.get_app_resource_dir('smk')
    wordflann = nntool.flann_cache(words, flann_params=flann_params, cache_dir=cache_dir)
    _daids = ensure_values(daids)
    idx2_dvec, idx2_daid, idx2_dfx = nntool.invertable_stack(vecs_list, _daids)
    invindex = InvertedIndex(words, wordflann, idx2_dvec, idx2_daid, idx2_dfx, _daids)
    if with_internals:
        invindex.compute_internals()
    return invindex


def ensure_values(data):
    return data.values if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) else data


def assign_to_words_(wordflann, idx2_vec):
    """ Assigns vectors to nearest word
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> wordflann = invindex.wordflann
    >>> idx2_vec = invindex.idx2_dvec
    >>> wx2_idxs, idx2_wx = assign_to_words_(wordflann, idx2_vec)
    """
    #TODO: multiple assignment
    idx2_vec_ = ensure_values(idx2_vec)
    idx2_wx, _idx2_wdist = wordflann.nn_index(idx2_vec_, 1)
    assign_df = pd.DataFrame(idx2_wx, columns=['wx'])
    word_group = assign_df.groupby('wx')
    wx2_idxs = word_group['wx'].indices
    return wx2_idxs, idx2_wx


def compute_word_idf_(wx2_idxs, idx2_aid, _daids):
    """ Returns the inverse-document-frequency weighting for each word
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> wx2_idxs = invindex.wx2_idxs
    >>> idx2_aid = invindex.idx2_daid
    >>> _daids   = invindex.daids
    >>> wx2_idf = compute_word_idf_(wx2_idxs, idx2_aid, _daids)
    """
    nTotalDocs = _daids.shape[0]
    wx2_idf = {wx: np.log(nTotalDocs / len(pd.unique(idx2_aid[idxs])))
               for wx, idxs in six.iteritems(wx2_idxs)}
    return wx2_idf


def compute_residuals_(words, idx2_vec, wx2_idxs):
    """ Computes residual vectors based on word assignments
    returns mapping from word index to a set of residual vectors
    """
    wx2_rvecs = {}
    for wx in wx2_idxs.keys():
        # for each word
        idxs = wx2_idxs[wx]
        # Get vecs assigned to it
        vecs = idx2_vec[idxs].astype(dtype=np.float64)
        word = words[wx].astype(dtype=np.float64)
        # compute residuals of all vecs assigned to this word
        _words = np.tile(word, (vecs.shape[0], 1))
        residuals = _words - vecs
        # normalize residuals
        # TODO Check for 0 division
        residuals_n = vtool.linalg.normalize_rows(residuals)
        #residuals_n[np.isnan(residuals_n)] = 1.0 / VEC_DIM
        wx2_rvecs[wx] = residuals_n
    return wx2_rvecs


def compute_internals_(invindex):
    """
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> compute_internals_(invindex)
    """
    idx2_vec = invindex.idx2_dvec
    wx2_idxs, idx2_wx = invindex.assign_to_words(idx2_vec)
    wx2_weight = invindex.compute_word_weights(wx2_idxs)
    wx2_drvecs = invindex.compute_residuals(idx2_vec, wx2_idxs)
    invindex.idx2_wx    = idx2_wx    # stacked index -> word index
    invindex.wx2_idxs   = wx2_idxs
    invindex.wx2_weight = wx2_weight
    invindex.wx2_drvecs = wx2_drvecs
    assert not isinstance(invindex.daids, pd.Series)
    pandasify(invindex)
    invindex.daid2_gamma = invindex.compute_data_gamma()


def pandasify(invindex):
    """
    Transitions from numpy to pandas
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> idx2_vec = invindex.idx2_dvec
    >>> wx2_idxs, idx2_wx = invindex.assign_to_words(idx2_vec)
    >>> wx2_weight = invindex.compute_word_weights(wx2_idxs)
    >>> wx2_drvecs = invindex.compute_residuals(idx2_vec, wx2_idxs)
    >>> invindex.idx2_wx    = idx2_wx    # stacked index -> word index
    >>> invindex.wx2_idxs   = wx2_idxs
    >>> invindex.wx2_weight = wx2_weight
    >>> invindex.wx2_drvecs = wx2_drvecs
    """
    #TODO: Integrate better!!
    #assert isinstance(invindex.wx2_weight, dict)
    assert not isinstance(invindex.daids, pd.Series)
    def noprint(arg):
        pass
    print = utool.super_print
    #print = noprint

    # aid - annotation ids
    _daids = pd.Series(invindex.daids, name='aid')
    print(_daids)

    # wx - word index
    wx_series = pd.Series(invindex.wx2_weight.keys(), name='wx')
    wx2_weight = pd.Series(invindex.wx2_weight.values(), index=wx_series, name='idf')
    print(wx2_weight)
    wx2_idxs = pandasify_dict1d(invindex.wx2_idxs, wx_series, 'idx', 'wx2_idxs')
    wx2_drvecs = pandasify_dict2d(invindex.wx2_drvecs, wx_series, wx2_idxs, VEC_COLUMNS, 'wx2_drvecs')
    print(wx2_idxs)
    print(wx2_drvecs)

    # idx - stacked indicies
    idx_series = pd.Series(np.arange(len(invindex.idx2_daid)), name='idx')
    idx2_daid = pd.Series(invindex.idx2_daid, index=idx_series, name='daid')
    idx2_daid = pd.Series(invindex.idx2_daid, index=idx_series, name='aid')
    idx2_dvec = pd.DataFrame(invindex.idx2_dvec, index=idx_series, columns=VEC_COLUMNS)
    idx2_dfx  = pd.Series(invindex.idx2_dfx, index=idx_series, name='fx')
    print(idx2_dvec)
    print(idx2_dfx)

    invindex.daids = _daids
    invindex.idx2_daid = idx2_daid
    invindex.idx2_dvec = idx2_dvec
    invindex.idx2_dfx  = idx2_dfx
    invindex.wx2_idxs   = wx2_idxs
    invindex.wx2_drvecs = wx2_drvecs
    invindex.wx2_weight = wx2_weight
    #invindex.daid2_gamma = compute_data_gamma_()


def compute_data_gamma_(invindex, use_cache=True):
    """
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = True
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> daid2_gamma = compute_data_gamma_(invindex, use_cache=True)
    """
    cache_key = utool.hashstr(invindex.get_cfgstr())
    if use_cache:
        try:
            daid2_gamma = utool.global_cache_read(cache_key, appname='smk')
            #print('gamma_dbg cache hit')
            return daid2_gamma
        except Exception:
            pass

    # Gropuing by aid and words

    mark, end_ = utool.log_progress(('gamma grouping %s ' % (cache_key,)),
                                    invindex.wx2_drvecs.shape[0],
                                    flushfreq=100)
    daid2_wx2_drvecs = utool.ddict(dict)
    for count, wx in enumerate(invindex.wx2_drvecs.index):
        if count % 100 == 0:
            mark(wx)
        group  = invindex.wx2_drvecs[wx].groupby(invindex.idx2_daid)
        for daid, vecs in group:
            daid2_wx2_drvecs[daid][wx] = vecs.values
    end_()

    # Summation over words for each aid
    mark, end_ = utool.log_progress('gamma summation ', len(daid2_wx2_drvecs),
                                    flushfreq=100)
    daid2_gamma = pd.Series(
        np.zeros(invindex.daids.shape[0]),
        index=invindex.daids,
        name='gamma')
    wx2_weight = invindex.wx2_weight
    for count, (daid, wx2_drvecs) in enumerate(six.iteritems(daid2_wx2_drvecs)):
        if count % 100 == 0:
            mark(count)
        wx2_rvecs = wx2_drvecs
        daid2_gamma[daid] = gamma_summation(wx2_rvecs, wx2_weight)
    utool.global_cache_write(cache_key, daid2_gamma, appname='smk')
    return daid2_gamma


@six.add_metaclass(utool.ReloadingMetaclass)
class InvertedIndex(object):
    def __init__(invindex, words, wordflann, idx2_vec, idx2_aid, idx2_fx, _daids):
        invindex.wordflann  = wordflann
        invindex.words      = words     # visual word centroids
        invindex.daids     = _daids    # indexed annotation ids
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
        _daids   = invindex.daids
        wx2_weight = compute_word_idf_(wx2_idxs, idx2_aid, _daids)
        return wx2_weight

    def assign_to_words(invindex, idx2_vec):
        wordflann = invindex.wordflann
        wx2_idxs, idx2_wx = assign_to_words_(wordflann, idx2_vec)
        return wx2_idxs, idx2_wx

    def compute_residuals(invindex, idx2_vec, wx2_idxs):
        words = invindex.words
        wx2_rvecs = compute_residuals_(words, idx2_vec, wx2_idxs)
        return wx2_rvecs

    def compute_data_gamma(invindex):
        use_cache = True
        daid2_gamma = compute_data_gamma_(invindex, use_cache=use_cache)
        return daid2_gamma


def gamma_summation(wx2_rvecs, wx2_weight):
    r"""
    \begin{equation}
    \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
    \end{equation}
    """
    gamma_iter = (wx2_weight.get(wx, 0) * Match_N(vecs, vecs).sum()
                  for wx, vecs in six.iteritems(wx2_rvecs))
    return np.reciprocal(np.sqrt(sum(gamma_iter)))


def selectivity_function(rscore_mat, alpha=3, thresh=0):
    """ sigma from SMK paper rscore = residual score """
    scores = (np.sign(rscore_mat) * np.abs(rscore_mat)) ** alpha
    scores[scores <= thresh] = 0
    return scores


def Match_N(vecs1, vecs2):
    simmat = vecs1.dot(vecs2.T)
    # Nanvectors were equal to the cluster center.
    # This means that point was the only one in its cluster
    # Therefore it is distinctive and should have a high score
    simmat[np.isnan(simmat)] = 1.0
    return selectivity_function(simmat)


def compute_query_repr(annots_df, qaid, invindex):
    qfx2_vec = annots_df['vecs'][qaid]
    wx2_qfxs_, qfx2_wx_ = invindex.assign_to_words(qfx2_vec)
    wx2_qrvecs_ = invindex.compute_residuals(qfx2_vec, wx2_qfxs_)
    wx_series = pd.Series(wx2_qfxs_.keys(), name='wx')

    wx2_qfxs = pandasify_dict1d(wx2_qfxs_, wx_series, 'fx', 'wx2_qfxs')
    wx2_qrvecs = pandasify_dict2d(wx2_qrvecs_, wx_series, wx2_qfxs, VEC_COLUMNS, 'wx2_qrvecs')
    return wx2_qfxs, wx2_qrvecs


def query_inverted_index(annots_df, qaid, invindex):
    #if daid_subset is not None:
    #    idx2_daid = idx2_daid[idx2_daid.isin(daid_subset)]
    #    _daids = _daids[_daids.isin(daid_subset)]
    #    wx2_idxmask_ = {wx: np.in1d(idxs, idx2_daid.index) for wx, idxs in six.iteritems(wx2_idxs)}
    #    wx2_idxs   = {wx: idxs[wx2_idxmask_[wx]]   for wx, idxs in six.iteritems(wx2_idxs)}
    #    wx2_drvecs = {wx: drvecs[wx2_idxmask_[wx]] for wx, drvecs in six.iteritems(wx2_drvecs)}
    wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    return match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid)


def match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid):
    """
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    >>> daid2_totalscore = match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid)
    """
    _daids = invindex.daids
    idx2_daid = invindex.idx2_daid
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight
    daid2_gamma = invindex.daid2_gamma

    wx2_rvecs = wx2_qrvecs
    query_gamma = gamma_summation(wx2_rvecs, wx2_weight)

    # Accumulate scores over the entire database
    daid2_aggscore = pd.Series(np.zeros(len(_daids)), index=_daids, name='total_score')
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))

    daid2_wx2_scoremat = utool.ddict(lambda: utool.ddict(list))

    # for each word compute the pairwise scores between matches
    mark, end = utool.log_progress('query word: ', len(common_wxs), flushfreq=100)
    for count, wx in enumerate(common_wxs):
        if count % 100 == 0:
            mark(count)
        # Query and database vectors for wx-th word
        qrvecs = wx2_qrvecs[wx]
        drvecs = wx2_drvecs[wx]
        # Word Weight
        weight = wx2_weight[wx]
        # Compute score matrix
        qfx2_wscore = Match_N(qrvecs, drvecs)
        qfx2_wscore.groupby(idx2_daid)
        # Group scores by database annotation ids
        group = qfx2_wscore.groupby(idx2_daid, axis=1)
        for daid, scoremat in group:
            daid2_wx2_scoremat[daid][wx] = scoremat
        #qfx2_wscore = pd.DataFrame(qfx2_wscore_, index=qfxs, columns=_idxs)
        daid2_wscore = weight * qfx2_wscore.sum(axis=0).groupby(idx2_daid).sum()
        daid2_aggscore = daid2_aggscore.add(daid2_wscore, fill_value=0)
    daid2_totalscore = daid2_aggscore * daid2_gamma * query_gamma
    end()

    daid_fm = {}
    daid_fs = {}
    daid_fk = {}
    mark, end = utool.log_progress('accumulating match info: ', len(daid2_wx2_scoremat), flushfreq=100)
    for count, item in enumerate(daid2_wx2_scoremat.items()):
        daid, wx2_scoremat = item
        if count % 25 == 0:
            mark(count)
        fm_accum = []
        fs_accum = []
        fk_accum = []
        for wx, scoremat in wx2_scoremat.iteritems():
            qfxs = scoremat.index
            dfxs = invindex.idx2_dfx[scoremat.columns]
            fm_ = np.vstack(np.dstack(np.meshgrid(qfxs, dfxs, indexing='ij')))
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

    daid2_totalscore.sort(axis=1, ascending=False)
    return daid2_totalscore, chipmatch


def testdata():
    ibeis.ensure_pz_mtest()
    ibs = ibeis.opendb('PZ_MTEST')
    # Pandas Annotation Dataframe
    annots_df = make_annot_df(ibs)
    valid_aids = annots_df.index
    # Training set
    taids = valid_aids[:]
    # Database set
    daids  = valid_aids[1:]
    # Search set
    #qaids = valid_aids[0::2]
    qaids = valid_aids[0:1]
    #default = 1000
    default = 2E4
    #default=5)  # default=95000)
    nWords = utool.get_argval(('--nWords', '--nCentroids'), int, default=default)
    return ibs, annots_df, taids, daids, qaids, nWords


def main():
    """
    >>> from ibeis.model.hots.smk.smk import *  # NOQA
    """
    from ibeis.model.hots import pipeline
    from ibeis.model.hots import query_request as hsqreq
    ibs, annots_df, taids, daids, qaids, nWords = testdata()
    # Learn vocabulary
    words = learn_visual_words(annots_df, taids, nWords)
    # Index a database of annotations
    invindex = index_data_annots(annots_df, daids, words)
    # Query using SMK
    qaid = qaids[0]
    qreq_ = hsqreq.new_ibeis_query_request(ibs, [qaid], daids)
    # Smk Mach
    daid2_totalscore1, chipmatch = query_inverted_index(annots_df, qaid, invindex)
    # Pack into QueryResult
    qaid2_chipmatch = {qaid: chipmatch}
    qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, {}, qreq_)
    qres = qaid2_qres_[qaid]
    # Show match
    qres.show_top(ibs)
    print(daid2_totalscore1)

    #daid2_totalscore2, chipmatch = query_inverted_index(annots_df, daids[0], invindex)
    #print(daid2_totalscore2)
    #display_info(ibs, invindex, annots_df)
    print('finished main')
    return locals()


def display_info(ibs, invindex, annots_df):
    ################
    from ibeis.dev import dbinfo
    print(ibs.get_infostr())
    dbinfo.get_dbinfo(ibs, verbose=True)
    ################
    print('Inverted Index Stats: vectors per word')
    print(utool.stats_str(map(len, invindex.wx2_idxs.values())))
    ################
    #qfx2_vec     = annots_df['vecs'][1]
    centroids    = invindex.words
    num_pca_dims = 3  # 3
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


if __name__ == '__main__':
    main_locals = main()
    main_execstr = utool.execstr_dict(main_locals, 'main_locals')
    exec(main_execstr)
    exec(df2.present())
