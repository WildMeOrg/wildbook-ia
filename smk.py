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


def pandasify(invindex):
    assert isinstance(invindex.wx2_weight, dict)
    wx_series = pd.Series(invindex.wx2_weight.keys(), name='wx')
    wx2_weight = pd.Series(invindex.wx2_weight.values(), index=wx_series, name='idf')
    print(wx2_weight)
    invindex.wx2_weight = wx2_weight

    assert not isinstance(invindex._daids, pd.Series)
    _daids = pd.Series(invindex._daids, name='daid')
    print(_daids)
    invindex._daids = _daids

    idx_series = pd.Series(np.arange(len(invindex.idx2_aid)), name='idx')
    idx2_daid = pd.Series(invindex.idx2_aid, index=idx_series, name='daid')

    wx_series = pd.Series(invindex.wx2_drvecs.keys(), name='wx')
    drvec_dflist = []
    for wx in wx_series:
        drvecs = invindex.wx2_drvecs[wx]
        _idxs = pd.Series(invindex.wx2_idxs[wx], name='idx')
        drvec_df = pd.DataFrame(drvecs, index=_idxs)
        drvec_dflist.append(drvec_df)
    wx2_drvecs = pd.Series(drvec_dflist, index=wx_series, name='drvecs')
    print(wx2_drvecs)

    drvecs = wx2_drvecs[wx]
    group = drvecs.groupby(idx2_daid)
    for aid, vecs in group:
        vecs.dot(vecs.T)
        # LEFT OFF HERE
        pass


def unpandasify(invindex):
    invindex.wx2_weight = invindex.wx2_weight.to_dict()


def selectivity_function(rscore_mat, alpha=3, thresh=0):
    """ sigma from SMK paper rscore = residual score """
    scores = (np.sign(rscore_mat) * np.abs(rscore_mat)) ** alpha
    scores[scores <= thresh] = 0
    return scores


def compute_gamma(annots_df, invindex):
    _daids = pd.Series(invindex._daids, name='daid')
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight
    daid2_gamma = pd.Series(np.zeros(len(invindex._daids)), index=_daids, name='gamma')
    for wx, _idxs in six.iteritems(invindex.wx2_idxs):
        aids_ = invindex.idx2_aid[_idxs]
        weight =  invindex.wx2_weight[wx]
        for aid in aids_:
            daid2_gamma[aid] += weight


def query_inverted_index(annots_df, qaid, invindex, daid_subset=None):
    return match_kernel(annots_df, invindex, qaid, daid_subset=daid_subset)


def match_kernel(annots_df, invindex, qaid, daid_subset=None):
    """
    >>> from smk import *  # NOQA
    >>> ibs, annots_df, taids, daids, qaids, nWords = testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> wx2_drvecs = invindex.wx2_drvecs
    >>> qaid = qaids[0]
    >>> wx2_qrvecs = query_inverted_index(annots_df, qaid, invindex)
    """
    #qfx2_axs = []
    #qfx2_fm = []
    #qfx2_fs = []
    #aid_fm = []
    #aid_fs = []

    #idx2_dfx  = invindex.idx2_fx
    #idx2_wx  = invindex.idx2_wx
    #wx2_idxs_series = {wx: pd.Series(idxs, name='idx') for wx, idxs in
    #                   six.iteritems(invindex.wx2_idxs)}
    #wx2_qfxs_series = {wx: pd.Series(qfx, name='qfx') for wx, qfx in
    #                   six.iteritems(wx2_qfxs)}
    #qfx2_idx = np.tile(_idxs, (len(qfxs), 1))
    #qfx2_aid = np.tile(idx2_daid.take(_idxs), (len(qfxs), 1))
    #qfx2_fx = np.tile(idx2_dfx.take(_idxs), (len(qfxs), 1))

    qfx2_vec = annots_df['vecs'][qaid]
    wx2_qfxs, qfx2_wx = invindex.inverted_assignments(qfx2_vec)
    wx2_qrvecs = invindex.compute_residuals(qfx2_vec, wx2_qfxs)

    _daids = pd.Series(invindex._daids, name='daid')
    idx2_daid = pd.Series(invindex.idx2_aid, name='daid')
    wx2_idxs   = invindex.wx2_idxs
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight

    query_gamma = 0
    for wx, qrvecs in six.iteritems(wx2_qrvecs):
        query_gamma += selectivity_function(qrvecs.dot(qrvecs.T)).sum() * wx2_weight[wx]
    query_gamma = 1 / np.sqrt(query_gamma)

    #if __debug__:
    #    for wx in wx2_drvecs.keys():
    #        assert wx2_drvecs[wx].shape[0] == wx2_idxs[wx].shape[0]

    if daid_subset is not None:
        idx2_daid = idx2_daid[idx2_daid.isin(daid_subset)]
        _daids = _daids[_daids.isin(daid_subset)]
        wx2_idxmask_ = {wx: np.in1d(idxs, idx2_daid.index) for wx, idxs in six.iteritems(wx2_idxs)}
        wx2_idxs   = {wx: idxs[wx2_idxmask_[wx]]   for wx, idxs in six.iteritems(wx2_idxs)}
        wx2_drvecs = {wx: drvecs[wx2_idxmask_[wx]] for wx, drvecs in six.iteritems(wx2_drvecs)}

    daid2_totalscore = match_nonagg(wx2_qrvecs, wx2_qfxs, wx2_drvecs, wx2_idxs,
                                    wx2_weight, idx2_daid, _daids)

    daid2_totalscore.sort(axis=1, ascending=False)
    return daid2_totalscore


def match_nonagg(wx2_qrvecs, wx2_qfxs, wx2_drvecs, wx2_idxs, wx2_weight, idx2_daid,
                 _daids):
    # Accumulate scores over the entire database
    daid2_totalscore = pd.Series(np.zeros(len(_daids)), index=_daids, name='total_score')
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))

    # for each word compute the pairwise scores between matches
    for wx in common_wxs:
        # Query information for this word
        qfxs   = wx2_qfxs[wx]
        qrvecs = wx2_qrvecs[wx]
        # Database information for this word
        _idxs  = wx2_idxs[wx]
        drvecs = wx2_drvecs[wx]
        # Word Weight
        weight = wx2_weight[wx]
        # Compute score matrix
        qfx2_wscore_ = selectivity_function(qrvecs.dot(drvecs.T))
        # Group scores by database annotation ids
        qfx2_wscore = pd.DataFrame(qfx2_wscore_, index=qfxs, columns=_idxs)
        daid2_wscore = qfx2_wscore.sum(axis=0).groupby(idx2_daid).sum()
        daid2_totalscore = daid2_totalscore.add(daid2_wscore * weight, fill_value=0)
    return daid2_totalscore


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
    daid2_totalscore1 = query_inverted_index(annots_df, qaid, invindex)
    daid2_totalscore2 = query_inverted_index(annots_df, daids[0], invindex)
    print(daid2_totalscore1)
    print(daid2_totalscore2)
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
