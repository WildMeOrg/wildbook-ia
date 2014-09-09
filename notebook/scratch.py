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
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('isplay.notebook_repr_html', True)
ibeis.ensure_pz_mtest()

#taids = ibs.get_valid_aids()
#tvecs_list = ibs.get_annot_desc(taids)
#tkpts_list = ibs.get_annot_kpts(taids)
#tvec_list = np.vstack(tvecs_list)
#print(idx2_vec)

#labels, words = vtool.clustering.precompute_akmeans(tvec_list, 1000, 30, cache_dir='.')
#tvecdf_list = [pd.DataFrame(vecs) for vecs in  tvecs_list]
#tvecs_df = pd.DataFrame(tvecdf_list, index=taids)
#kpts_col = pd.DataFrame(tkpts_list, index=taids, columns=['kpts'])
#vecs_col = pd.DataFrame(tvecs_list, index=taids, columns=['vecs'])
#tvecs_dflist = [pd.DataFrame(vecs, index=np.arange(len(vecs))) for vecs in tvecs_list]
#pd.concat(tvecs_dflist)
## Bui


#taids = ibs.get_valid_aids()
#tvecs_list = ibs.get_annot_desc(taids)
#tkpts_list = ibs.get_annot_kpts(taids)

#orig_idx2_vec, orig_idx2_ax, orig_idx2_fx = vtool.nearest_neighbors.invertable_stack(tvecs_list, taids)
#annots_df = pd.concat([vecs_col, kpts_col], axis=1)
#annots_df

#idx2_vec = np.vstack(annots_df['vecs'].values)
##idx2_ax =
#idx2_vec, idx2_ax, idx2_fx = vtool.nearest_neighbors.invertable_stack(tvecs_list, taids)


#labels, words = vtool.clustering2.precompute_akmeans(tvec_list, 1000, 30)
#words = centroids

def make_annot_df(ibs):
    aid_list = ibs.get_valid_aids()
    _kpts_col = pd.DataFrame(ibs.get_annot_kpts(aid_list),
                             index=aid_list, columns=['kpts'])
    _vecs_col = pd.DataFrame(ibs.get_annot_desc(aid_list),
                             index=aid_list, columns=['vecs'])
    annots_df = pd.concat([_vecs_col, _kpts_col], axis=1)
    return annots_df


def learn_visual_words(annots_df, train_aids, nCentroids):
    vecs_list = annots_df['vecs'][train_aids].as_matrix()
    train_vecs = np.vstack(vecs_list)
    print('Training %d word vocabulary with %d annots and %d descriptors' %
          (nCentroids, len(train_aids), len(train_vecs)))
    words = clustertool.precompute_akmeans(train_vecs, nCentroids, max_iters=100)
    return words


def index_data_annots(annots_df, daids, words):
    vecs_list = annots_df['vecs'][daids]
    flann_params = {}
    wordflann = vtool.nearest_neighbors.flann_cache(words, flann_params=flann_params)
    idx2_vec, idx2_ax, idx2_fx = nntool.invertable_stack(vecs_list, daids)
    ax2_aid = np.array(daids)
    invindex = InvertedIndex(words, wordflann, idx2_vec, idx2_ax, idx2_fx, ax2_aid)
    return invindex


def inverted_assignments(wordflann, idx2_vec):
    idx2_wx, _idx2_wdist = wordflann.nn_index(idx2_vec, 1)
    idx_list = list(range(len(idx2_wx)))
    # TODO: replace with pandas groupby
    wx2_idxs = utool.group_items(idx_list, idx2_wx.tolist())
    return wx2_idxs, idx2_wx


def compute_residuals(words, wx2_idxs, idx2_vec):
    wx2_rvecs = {}
    for word_index in wx2_idxs.keys():
        # for each word
        idxs = wx2_idxs[word_index]
        vecs = np.array(idx2_vec[idxs], dtype=np.float64)
        word = np.array(words[word_index], dtype=np.float64)
        # compute residuals of all vecs assigned to this word
        residuals = np.array([word - vec for vec in vecs])
        # normalize residuals
        residuals_n = vtool.linalg.normalize_rows(residuals)
        wx2_rvecs[word_index] = residuals_n
    return wx2_rvecs


#def smk_similarity(wx2_qrvecs, wx2_drvecs):
#    similarity_matrix = (rvecs1.dot(rvecs2.T))


def query_inverted_index(annots_df, qaid, invindex):
    qfx2_vec = annots_df['vecs'][qaid]
    wx2_qfxs, qfx2_wx = invindex.inverted_assignments(qfx2_vec)
    wx2_qrvecs = invindex.compute_residuals(qfx2_vec, wx2_qfxs)
    return wx2_qrvecs


@six.add_metaclass(utool.ReloadingMetaclass)
class InvertedIndex(object):
    def __init__(invindex, words, wordflann, idx2_vec, idx2_ax, idx2_fx, ax2_aid):
        invindex.wordflann = wordflann
        invindex.words     = words
        invindex.ax2_aid   = ax2_aid
        invindex.idx2_vec  = idx2_vec
        invindex.idx2_ax   = idx2_ax
        invindex.idx2_fx   = idx2_fx
        invindex.idx2_wx  = None
        invindex.wx2_idxs = None
        invindex.wx2_drvecs = None
        #invindex.compute_internals()

    def compute_internals(invindex):
        idx2_vec = invindex.idx2_vec
        wx2_idxs, idx2_wx = invindex.inverted_assignments(idx2_vec)
        wx2_drvecs = invindex.compute_residuals(idx2_vec, wx2_idxs)
        invindex.idx2_wx = idx2_wx
        invindex.wx2_idxs = wx2_idxs
        invindex.wx2_drvecs = wx2_drvecs

    def inverted_assignments(invindex, idx2_vec):
        wx2_idxs, idx2_wx = inverted_assignments(invindex.wordflann, idx2_vec)
        return wx2_idxs, idx2_wx

    def compute_residuals(invindex, idx2_vec, wx2_idxs):
        """ returns mapping from word index to a set of residual vectors """
        wx2_rvec = compute_residuals(invindex.words, wx2_idxs, idx2_vec)
        return wx2_rvec

    def get_annot_residuals(invindex, daid):
        """ daid = 4
        FIXME: Inefficient code
        """
        ax = np.where(invindex.ax2_aid == daid)[0]
        wx2_dfxs = {}
        wx2_drvecs = {}
        for wx, idxs in invindex.wx2_idxs.items():
            valid = invindex.idx2_ax[idxs] == ax
            dfxs = invindex.idx2_fx[idxs][valid]
            drvecs = invindex.idx2_drvecs[idxs][valid]
            wx2_dfxs[wx] = dfxs
            wx2_drvecs[wx] = drvecs


def display_info(invindex, annots_df):
    print('Inverted Index Stats: vectors per word')
    print(utool.stats_str(map(len, invindex.wx2_idxs.values())))
    qfx2_vec     = annots_df['vecs'][1]
    #data         = invindex.idx2_vec
    #fnum = 1
    centroids    = invindex.words
    num_pca_dims = 3  # 3
    whiten       = False
    #datax2_label = invindex.idx2_wx
    kwd = dict(
        num_pca_dims=num_pca_dims,
        whiten=whiten,
    )
    clustertool.rrr()
    def makeplot_(fnum, prefix, data, labels='centroids', centroids=centroids):
        return clustertool.plot_centroids(data, centroids, labels=labels,
                                          fnum=fnum, prefix=prefix + '\n', **kwd)

    makeplot_(1, 'centroid vecs', centroids)
    makeplot_(2, 'database vecs', invindex.idx2_vec)
    makeplot_(3, 'query vecs', qfx2_vec)
    #makeplot_(4, 'database vecs', invindex.idx2_vec)
    #makeplot_(5, 'query vecs', qfx2_vec)

    #makeplot_(centroids, centroids, labels='centroids', fnum=5, prefix='centroids vecs ', **kwd)

    #clustertool.plot_centroids(invindex.idx2_vec, centroids, labels=datax2_label, fnum=1, prefix='database vecs ', **kwd)

    #clustertool.plot_centroids(invindex.idx2_vec, centroids,
    #                           labels='centroids', fnum=2, **kwd)

    #clustertool.plot_centroids(invindex.idx2_vec, centroids,
    #                           labels=invindex.idx2_ax, fnum=3, **kwd)

    #clustertool.plot_centroids(qfx2_vec, centroids, labels='centroids', fnum=4, prefix='query vecs ', **kwd)


def main():
    ibs = ibeis.opendb('PZ_MTEST')
    # Pandas Annotation Dataframe
    annots_df = make_annot_df(ibs)
    valid_aids = annots_df.index
    # Training set
    train_aids = valid_aids[0:20:2]
    # Database set
    daids  = valid_aids[3:30:2]
    # Search set
    #qaids = valid_aids[0::2]
    qaids = valid_aids[0:1]
    qaid = qaids[0]
    nCentroids = 10
    words = learn_visual_words(annots_df, train_aids, nCentroids)
    invindex = index_data_annots(annots_df, daids, words)
    invindex.compute_internals()
    wx2_drvecs = invindex.wx2_drvecs
    wx2_qrvecs = query_inverted_index(annots_df, qaid, invindex)
    display_info(invindex, annots_df)
    return locals()


if __name__ == '__main__':
    main_locals = main()
    main_execstr = utool.execstr_dict(main_locals, 'main_locals')
    exec(main_execstr)
    exec(df2.present())
