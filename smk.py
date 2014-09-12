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


def make_annot_df(ibs):
    """
    >>> from smk import *  # NOQA
    >>> ibs = ibeis.opendb('PZ_MTEST')
    >>> annots_df = make_annot_df(ibs)
    """
    aid_list = ibs.get_valid_aids()
    kpts_list = ibs.get_annot_kpts(aid_list)
    vecs_list = ibs.get_annot_desc(aid_list)
    aid_series = pd.Series(aid_list, name='aid')
    kpts_df = pd.DataFrame(kpts_list, index=aid_series, columns=['kpts'])
    vecs_df = pd.DataFrame(vecs_list, index=aid_series, columns=['vecs'])
    # Pandas Annotation Dataframe
    annots_df = pd.concat([kpts_df, vecs_df], axis=1)
    return annots_df


def learn_visual_words(annots_df, taids, nWords):
    """
    >>> from smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    """
    vecs_list = annots_df['vecs'][taids].as_matrix()
    train_vecs = np.vstack(vecs_list)
    print('Training %d word vocabulary with %d annots and %d descriptors' %
          (nWords, len(taids), len(train_vecs)))
    words = clustertool.cached_akmeans(train_vecs, nWords, max_iters=100)
    return words


def index_data_annots(annots_df, daids, words, with_internals=True):
    """
    >>> from smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = True
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    """
    vecs_df = annots_df['vecs'][daids]
    flann_params = {}
    wordflann = vtool.nearest_neighbors.flann_cache(words, flann_params=flann_params)
    _daids = daids.values if isinstance(daids, pd.Index) else daids
    idx2_vec, idx2_aid, idx2_fx = nntool.invertable_stack(vecs_df.values, _daids)
    invindex = InvertedIndex(words, wordflann, idx2_vec, idx2_aid, idx2_fx, _daids)
    if with_internals:
        invindex.compute_internals()
    return invindex


def inverted_assignments_(wordflann, idx2_vec):
    """ Assigns vectors to nearest word

    >>> from smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> wordflann = invindex.wordflann
    >>> idx2_vec = invindex.idx2_vec
    >>> wx2_idxs, idx2_wx = inverted_assignments_(wordflann, idx2_vec)
    """
    idx2_wx, _idx2_wdist = wordflann.nn_index(idx2_vec, 1)
    assign_df = pd.DataFrame(idx2_wx, columns=['wordindex'])
    word_group = assign_df.groupby('wordindex')
    wx2_idxs = word_group.wordindex.indices
    return wx2_idxs, idx2_wx


def compute_word_weights_(wx2_idxs, idx2_aid, _daids):
    """
    >>> from smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> wx2_idxs = invindex.wx2_idxs
    >>> idx2_aid = invindex.idx2_aid
    >>> _daids   = invindex._daids
    >>> wx2_idf = compute_word_weights_(wx2_idxs, idx2_aid, _daids)
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
        residuals_n = vtool.linalg.normalize_rows(residuals)
        wx2_rvecs[wx] = residuals_n
    return wx2_rvecs


def compute_internals_(invindex):
    """
    >>> from smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> with_internals = False
    >>> invindex = index_data_annots(annots_df, daids, words, with_internals)
    >>> compute_internals_(invindex)
    """

    idx2_vec = invindex.idx2_vec
    wx2_idxs, idx2_wx = invindex.inverted_assignments(idx2_vec)
    wx2_weight = invindex.compute_word_weights(wx2_idxs)
    wx2_drvecs = invindex.compute_residuals(idx2_vec, wx2_idxs)
    invindex.idx2_wx    = idx2_wx    # stacked index -> word index
    invindex.wx2_idxs   = wx2_idxs
    invindex.wx2_weight = wx2_weight
    invindex.wx2_drvecs = wx2_drvecs


@six.add_metaclass(utool.ReloadingMetaclass)
class InvertedIndex(object):
    def __init__(invindex, words, wordflann, idx2_vec, idx2_aid, idx2_fx, _daids):
        invindex._daids     = _daids
        invindex.wordflann  = wordflann
        invindex.words      = words     # visual word centroids
        invindex.idx2_vec   = idx2_vec  # stacked index -> descriptor vector
        invindex.idx2_aid   = idx2_aid  # stacked index -> annot id
        invindex.idx2_fx    = idx2_fx   # stacked index -> feature index
        invindex.wx2_idxs   = None      # word index -> stacked indexes
        invindex.wx2_drvecs = None      # word index -> residual vectors
        #invindex.compute_internals()

    def compute_internals(invindex):
        compute_internals_(invindex)

    def compute_word_weights(invindex, wx2_idxs):
        idx2_aid = invindex.idx2_aid
        _daids   = invindex._daids
        wx2_weight = compute_word_weights_(wx2_idxs, idx2_aid, _daids)
        return wx2_weight

    def inverted_assignments(invindex, idx2_vec):
        wordflann = invindex.wordflann
        wx2_idxs, idx2_wx = inverted_assignments_(wordflann, idx2_vec)
        return wx2_idxs, idx2_wx

    def compute_residuals(invindex, idx2_vec, wx2_idxs):
        words = invindex.words
        wx2_rvecs = compute_residuals_(words, idx2_vec, wx2_idxs)
        return wx2_rvecs


def selectivity_function(rscore_mat, alpha=3, thresh=0):
    """ sigma from SMK paper rscore = residual score """
    scores = (np.sign(rscore_mat) * np.abs(rscore_mat)) ** alpha
    scores[scores <= thresh] = 0
    return scores


def query_inverted_index(annots_df, qaid, invindex):
    """
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> wx2_drvecs = invindex.wx2_drvecs
    >>> qaid = qaids[0]
    >>> wx2_qrvecs = query_inverted_index(annots_df, qaid, invindex)
    """

    qfx2_axs = []
    qfx2_fm = []
    qfx2_fs = []
    aid_fm = []
    aid_fs = []

    qfx2_vec = annots_df['vecs'][qaid]
    wx2_qfxs, qfx2_wx = invindex.inverted_assignments(qfx2_vec)
    wx2_qrvecs = invindex.compute_residuals(qfx2_vec, wx2_qfxs)

    # Entire database
    daid2_score = utool.ddict(lambda: 0)
    query_wxs = set(wx2_qrvecs.keys())
    data_wxs  = set(invindex.wx2_drvecs.keys())

    idx2_daid = pd.Series(invindex.idx2_aid, name='daid')
    idx2_dfx  = pd.Series(invindex.idx2_fx, name='dfx')
    idx2_wfx  = pd.Series(invindex.idx2_wx, name='dwx')
    invindex.idx_df = pd.concat((idx2_daid, idx2_dfx, idx2_wfx), axis=1, names=['idx'])

    wx2_idxs = {wx: pd.Series(idxs, name='idx') for wx, idxs in
                six.iteritems(invindex.wx2_idxs)}
    wx2_qfxs = {wx: pd.Series(qfx, name='qfx') for wx, qfx in
                six.iteritems(wx2_qfxs)}

    for wx in data_wxs.intersection(query_wxs):
        # all pairs of scores
        _idxs = wx2_idxs[wx]
        weight = invindex.wx2_weight[wx]
        qfxs = wx2_qfxs[wx]
        qfx2_idx = np.tile(_idxs, (len(qfxs), 1))
        qfx2_aid = np.tile(invindex.idx_df['daid'].take(_idxs), (len(qfxs), 1))
        qfx2_fx = np.tile(invindex.idx_df['dfx'].take(_idxs), (len(qfxs), 1))
        qrvecs = wx2_qrvecs[wx]
        drvecs = invindex.wx2_drvecs[wx]
        qfx2_wordscore_ = selectivity_function(qrvecs.dot(drvecs.T))
        qfx2_wordscore = pd.DataFrame(qfx2_wordscore_, index=qfxs, columns=_idxs)
        qfx2_datascore = qfx2_wordscore.groupby(invindex.idx_df['daid'], axis=1).sum()
        daid2_wordscore = qfx2_datascore.sum(axis=0)
        for aid in daid2_wordscore.index:
            daid2_score[aid] = daid2_wordscore[aid] * weight
    aidkeys = np.array(daid2_score.keys())
    totalscores = np.array(daid2_score.values())
    sortx = totalscores.argsort()[::-1]
    ranked_aids = aidkeys[sortx]
    ranked_scores = totalscores[sortx]
    score_df = pd.DataFrame(ranked_scores, index=ranked_aids, columns=['score'])
    score_df.fillna(0, inplace=True)
    print(score_df)
    print(utool.dict_str(daid2_score))
    return wx2_qrvecs


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
    nWords = utool.get_arg(('--nWords', '--nCentroids'), int, default=1000)
                           #default=5)  # default=95000)
    return ibs, annots_df, taids, daids, qaids, nWords


def main():
    ibs, annots_df, taids, daids, qaids, nWords = testdata()
    words = learn_visual_words(annots_df, taids, nWords)
    invindex = index_data_annots(annots_df, daids, words)
    wx2_drvecs = invindex.wx2_drvecs
    qaid = qaids[0]
    wx2_qrvecs = query_inverted_index(annots_df, qaid, invindex)
    display_info(ibs, invindex, annots_df)
    return locals()


def display_info(ibs, invindex, annots_df):
    np.set_printoptions(precision=2)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('isplay.notebook_repr_html', True)
    #################
    #from ibeis.dev import dbinfo
    #print(ibs.get_infostr())
    #dbinfo.get_dbinfo(ibs, verbose=True)
    #################
    #print('Inverted Index Stats: vectors per word')
    #print(utool.stats_str(map(len, invindex.wx2_idxs.values())))
    #################
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
    #makeplot_(1, 'centroid vecs', centroids)
    #makeplot_(2, 'database vecs', invindex.idx2_vec)
    #makeplot_(3, 'query vecs', qfx2_vec)
    #makeplot_(4, 'database vecs', invindex.idx2_vec)
    #makeplot_(5, 'query vecs', qfx2_vec)
    #################


if __name__ == '__main__':
    main_locals = main()
    main_execstr = utool.execstr_dict(main_locals, 'main_locals')
    exec(main_execstr)
    exec(df2.present())
