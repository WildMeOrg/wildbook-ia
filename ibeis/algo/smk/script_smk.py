"""
Results so far without SV / fancyness
Using standard descriptors / vocabulary

proot=bow,nWords=1E6 -> .594
proot=asmk,nWords=1E6 -> .529
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
#from ibeis.algo.smk import match_chips5 as mc5
from ibeis.algo.smk import inverted_index
from ibeis.algo.smk import smk_funcs
from ibeis.algo.smk import smk_pipeline
from six.moves import zip, map
#from ibeis.algo.smk import smk_funcs
#from ibeis.algo.smk import inverted_index
#from ibeis import core_annots
#from ibeis.algo import Config as old_config
(print, rrr, profile) = ut.inject2(__name__)


class SMK(ut.NiceRepr):
    """
    smk = SMK(wx_to_weight, method='bow')
    smk.match_score(X, X)
    """
    def __nice__(smk):
        return smk.method

    def __init__(smk, wx_to_weight, method='asmk', **kwargs):
        smk.wx_to_weight = wx_to_weight
        smk.method = method
        if method == 'asmk':
            smk.match_score = smk.match_score_agg
        elif method == 'smk':
            smk.match_score = smk.match_score_sep
        elif method == 'bow':
            smk.match_score = smk.match_score_bow
        if method in ['asmk', 'smk']:
            smk.alpha = kwargs.pop('alpha', 0.0)
            smk.thresh = kwargs.pop('thresh', 0.0)

        if method == 'bow2':
            smk.kernel = smk.kernel_bow_tfidf
        else:
            smk.kernel = smk.kernel_smk

        assert len(kwargs) == 0, 'unexpected kwargs=%r' % (kwargs,)

    def gamma(smk, X):
        score = smk.match_score(X, X)
        sccw = np.reciprocal(np.sqrt(score))
        return sccw

    def kernel_bow_tfidf(smk, X, Y):
        return X.bow.dot(Y.bow)

    def kernel_smk(smk, X, Y):
        score = smk.match_score(X, Y)
        score = X.gamma * Y.gamma * score
        return score

    def word_isect(smk, X, Y):
        isect_wxs = X.wx_set.intersection(Y.wx_set)
        X_idx = ut.take(X.wx_to_idx, isect_wxs)
        Y_idx = ut.take(Y.wx_to_idx, isect_wxs)
        weights = ut.take(smk.wx_to_weight, isect_wxs)
        return X_idx, Y_idx, weights

    def match_score_agg(smk, X, Y):
        X_idx, Y_idx, weights = smk.word_isect(X, Y)
        PhisX, flagsX = X.Phis_flags(X_idx)
        PhisY, flagsY = Y.Phis_flags(Y_idx)
        scores = smk_funcs.match_scores_agg(
            PhisX, PhisY, flagsX, flagsY, smk.alpha, smk.thresh)
        scores = np.multiply(scores, weights, out=scores)
        score = scores.sum()
        return score

    def match_score_sep(smk, X, Y):
        X_idx, Y_idx, weights = smk.word_isect(X, Y)
        phisX_list, flagsY_list = X.phis_flags_list(X_idx)
        phisY_list, flagsX_list = Y.phis_flags_list(Y_idx)
        scores_list = smk_funcs.match_scores_sep(
            phisX_list, phisY_list, flagsX_list, flagsY_list, smk.alpha,
            smk.thresh)
        for scores, w in zip(scores_list, weights):
            np.multiply(scores, w, out=scores)
        score = np.sum([s.sum() for s in scores_list])
        return score

    def match_score_bow(smk, X, Y):
        isect_words = X.wx_set.intersection(Y.wx_set)
        weights = ut.take(smk.wx_to_weight, isect_words)
        score = np.sum(weights)
        return score


class SparseVector(ut.NiceRepr):
    def __init__(self, _dict):
        self._dict = _dict

    def __nice__(self):
        return '%d nonzero values' % (len(self._dict),)

    def __getitem__(self, keys):
        vals = ut.take(self._dict, keys)
        return vals

    def dot(self, other):
        keys1 = set(self._dict.keys())
        keys2 = set(other._dict.keys())
        keys = keys1.intersection(keys2)
        vals1 = np.array(self[keys])
        vals2 = np.array(other[keys])
        return np.multiply(vals1, vals2).sum()


def load_internal_data():
    from ibeis.algo.smk.smk_pipeline import *  # NOQA
    import ibeis
    qreq_ = ibeis.testdata_qreq_(
        defaultdb='Oxford', a='oxford',
        p='smk:nWords=[64000],nAssign=[1],SV=[False],can_match_sameimg=True')
    cm_list = qreq_.execute()
    ave_precisions = [cm.get_annot_ave_precision() for cm in cm_list]
    mAP = np.mean(ave_precisions)
    print('mAP = %.3f' % (mAP,))
    cm = cm_list[-1]
    return qreq_, cm


def oxford_conic_test():
    # Test that these are what the readme says
    A, B, C = [0.016682, 0.001693, 0.014927]
    A, B, C = [0.010141, -1.1e-05, 0.02863]
    Z = np.array([[A, B], [B, C]])

    import vtool as vt
    invV = vt.decompose_Z_to_invV_2x2(Z)  # NOQA
    invV = vt.decompose_Z_to_invV_mats2x2(np.array([Z]))  # NOQA
    # seems ok
    #invV = np.linalg.inv(V)


def load_external_oxford_features(config):
    """
    # TODO: root sift with centering

    Such hacks for reading external oxford

    config = {
        'root_sift': True,
    }
    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """
    relevant_params = [
        'root_sift'
    ]
    config = ut.dict_subset(config, relevant_params)

    from os.path import join, basename, splitext
    suffix = ut.get_cfg_lbl(config)
    dbdir = ut.truepath('/raid/work/Oxford/')
    data_fpath1 = join(dbdir, ut.augpath('oxford_data1.pkl', suffix))
    if ut.checkpath(data_fpath1):
        oxford_data1 = ut.load_data(data_fpath1)
        return oxford_data1
    else:
        word_dpath = join(dbdir, 'word_oxc1_hesaff_sift_16M_1M')
        _word_fpath_list = ut.ls(word_dpath)
        imgid_to_word_fpath = {
            splitext(basename(word_fpath))[0]: word_fpath
            for word_fpath in _word_fpath_list
        }
        readme_fpath = join(dbdir, 'README2.txt')
        imgid_order = ut.readfrom(readme_fpath).split('\n')[20:-1]

        imgid_order = imgid_order

        import pandas as pd
        imgid_to_df = {}
        for imgid in ut.ProgIter(imgid_order, lbl='reading kpts'):
            word_fpath = imgid_to_word_fpath[imgid]
            row_gen = (map(float, line.strip('\n').split(' '))
                       for line in ut.read_lines_from(word_fpath)[2:])
            rows = [(int(word_id), x, y, e11, e12, e22)
                    for (word_id, x, y, e11, e12, e22) in row_gen]
            df = pd.DataFrame(rows, columns=['word_id', 'x', 'y', 'e11', 'e12', 'e22'])
            imgid_to_df[imgid] = df

        # Convert ellipses (in Z format to invV format)
        import vtool as vt
        for imgid in ut.ProgIter(imgid_to_df.keys()):
            df = imgid_to_df[imgid]
            e11, e12, e22 = df.loc[:, ('e11', 'e12', 'e22')].values.T
            #import numpy as np
            Z_mats2x2 = np.array([[e11, e12],
                                 [e12, e22]])
            Z_mats2x2 = np.rollaxis(Z_mats2x2, 2)
            invV_mats2x2 = vt.decompose_Z_to_invV_mats2x2(Z_mats2x2)
            invV_mats2x2 = invV_mats2x2.astype(np.float32)
            a = invV_mats2x2[:, 0, 0]
            c = invV_mats2x2[:, 1, 0]
            d = invV_mats2x2[:, 1, 1]
            df = df.assign(a=a, c=c, d=d)
            imgid_to_df[imgid] = df

        df_list = ut.take(imgid_to_df, imgid_order)

        offset_list = [0] + ut.cumsum([len(df_) for df_ in df_list])
        shape = (offset_list[-1], 128)
        #shape = (16334970, 128)
        try:
            sift_fpath = join(dbdir, 'OxfordSIFTDescriptors',
                              'feat_oxc1_hesaff_sift.bin')
            file_ = open(sift_fpath, 'rb')
            with ut.Timer('Reading SIFT binary file'):
                nbytes = np.prod(shape)
                vecs = np.fromstring(file_.read(nbytes), dtype=np.uint8)
            vecs = vecs.reshape(shape)
        finally:
            file_.close()

        if config['root_sift']:
            # Have to do this in chunks to fit in memory
            import vtool as vt
            chunksize = shape[0] // 100
            slices = list(ut.ichunk_slices(shape[0], chunksize))
            fidelity = 512.0
            for sl in ut.ProgIter(slices, lbl='apply rootsift'):
                s = vecs[sl].astype(np.float32) / fidelity
                s = vt.normalize(s, ord=1, axis=1, out=s)
                s = np.sqrt(s, out=s)
                s = (s * (fidelity)).astype(np.uint8)
                vecs[sl] = s

        vecs_list = [vecs[l:r] for l, r in ut.itertwo(offset_list)]
        kpts_list = [df_.loc[:, ('x', 'y', 'a', 'c', 'd')].values
                     for df_ in df_list]
        wordid_list = [df_.loc[:, 'word_id'].values for df_ in df_list]

        oxford_data1 = {
            'imgid_order': imgid_order,
            'kpts_list': kpts_list,
            'vecs_list': vecs_list,
            'wordid_list': wordid_list,
        }
        ut.save_data(data_fpath1, oxford_data1)

        if False:
            imgid = imgid_order[0]
            imgdir = join(dbdir, 'oxbuild_images')
            gpath = join(imgdir,  imgid.replace('oxc1_', '') + '.jpg')
            image = vt.imread(gpath)
            import plottool as pt
            pt.qt4ensure()
            pt.imshow(image)
            kpts = kpts_list[0].copy()
            vecs = vecs_list[0]
            #h, w = image.shape[0:2]
            #kpts.T[1] = h - kpts.T[1]

            #pt.draw_kpts2(kpts, ell_alpha=.4, pts=True, ell=True)
            pt.interact_keypoints.ishow_keypoints(image, kpts, vecs,
                                                  ori=False, ell_alpha=.4,
                                                  color='distinct')

    return oxford_data1


def train_vocabulary(all_vecs, config):
    #oxford_data1 = load_external_oxford_features()
    #imgid_order = oxford_data1['imgid_order']
    #kpts_list = oxford_data1['kpts_list']
    #vecs_list = oxford_data1['vecs_list']
    #wordid_list = oxford_data1['wordid_list']

    #num_words = 8000
    num_words = config['num_words']
    from os.path import join
    dbdir = ut.truepath('/raid/work/Oxford/')
    fpath = join(dbdir, 'vocab_%d.pkl' % (num_words,))
    if ut.checkpath(fpath):
        return ut.load_data(fpath)

    #train_vecs = np.vstack(vecs_list)[::100].copy()
    train_vecs = all_vecs.astype(np.float32)

    rng = np.random.RandomState(13421421)
    import sklearn.cluster
    train_vecs = train_vecs.astype(np.float32)
    clusterer = sklearn.cluster.MiniBatchKMeans(
        num_words, random_state=rng,
        n_init=3, verbose=5)
    clusterer.fit(train_vecs)
    words = clusterer.cluster_centers_
    words = words.astype(np.uint8)
    ut.save_data(ut.augpath(fpath, 'words'), words)

    from ibeis.algo.smk import vocab_indexer
    vocab = vocab_indexer.VisualVocab(words)
    vocab.build()

    #tuned_params = vt.tune_flann(words, target_precision=.95)

    ut.save_data(fpath, vocab)
    return words

    #import pyflann
    #flann = pyflann.FLANN()
    #centers1 = flann.kmeans(train_vecs, num_words, max_iterations=1)
    #pass


def load_external_data2():
    """
    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """

    config = {
        #'num_words': 1E6
        'root_sift': True,
        #'num_words': 8000,
        'num_words': 65000,
        'checks': 128,
    }
    nAssign = 1

    from os.path import join, basename, splitext
    import ibeis
    oxford_data1 = load_external_oxford_features(config)
    imgid_order = oxford_data1['imgid_order']
    kpts_list = oxford_data1['kpts_list']
    vecs_list = oxford_data1['vecs_list']
    wordid_list = oxford_data1['wordid_list']
    uri_order = [x.replace('oxc1_', '') for x in imgid_order]
    assert len(ut.list_union(*wordid_list)) == 1E6

    ibs = ibeis.opendb('Oxford')

    def get_oxford_keys(_annots):
        _images = ibs.images(_annots.gids)
        intern_uris = [splitext(basename(uri))[0] for uri in _images.uris_original]
        return intern_uris

    # Load database annotations and reorder them to aggree with internals
    _dannots = ibs.annots(ibs.filter_annots_general(has_none='query'))
    intern_uris = get_oxford_keys(_dannots)
    lookup = ut.make_index_lookup(intern_uris)
    data_annots = _dannots.take(ut.take(lookup, uri_order))
    assert get_oxford_keys(data_annots) == uri_order

    offset_list = [0] + ut.cumsum([len(v) for v in vecs_list])

    dbdir = ut.truepath('/raid/work/Oxford/')
    #======================
    # Build/load database info
    daids = data_annots.aids

    #======================
    # Compute All Word Assignments
    all_vecs = np.vstack(vecs_list)

    if False:
        mean_vec = np.mean(all_vecs, axis=0)
        # FOR ROOT SIFT
        mean_vec = np.array([46.62654716,  31.21229356,  20.65256618,  19.79029916, 26.35339489,
                             24.89652947,  25.73929759,  28.1263965 , 49.90879457,  39.53755507,
                             35.60305786,  31.12662852, 32.59826317,  33.3061391 ,  34.21935008,
                             33.61522911, 48.37291706,  34.53814424,  36.66045441,  35.01652565,
                             32.9685514 ,  30.68111989,  33.24712142,  36.92554183, 44.53439247,
                             28.89821659,  27.61463817,  26.57542163, 26.52943323,  19.98019709,
                             20.12697373,  28.90332924, 60.3505775 ,  37.49316552,  23.39667829,
                             26.81768004, 36.64132753,  31.33051857,  28.19458628,  36.05361797,
                             62.16145931,  42.61728225,  33.45618204,  33.99012927, 39.2309187 ,
                             35.93958642,  33.56585326,  39.49449849, 59.98748556,  39.56601898,
                             35.1362548 ,  37.46382026, 40.00613604,  33.68649346,  32.18926408,
                             41.0287681 , 57.70276193,  36.14551181,  29.74180816,  33.18968385,
                             37.47753488,  27.27518575,  23.62592677,  36.28705048, 60.33918189,
                             35.70739481,  28.27889338,  31.46749679, 36.64511836,  26.68877727,
                             23.33377521,  37.78693368, 62.19426562,  39.23236523,  33.69995568,
                             36.08645177, 39.22344608,  33.82699613,  33.29949893,  42.88519226,
                             59.96339491,  40.77440589,  32.36521555,  33.88829536, 40.01698503,
                             37.32365447,  35.03381512,  39.83779101, 57.69014384,  35.99944353,
                             23.70292024,  27.42776124, 37.4634587 ,  33.06158922,  29.73344414,
                             36.50520264, 46.65942545,  27.96910916,  25.83571901,  24.99287902,
                             26.34822531,  19.6731151 ,  20.55268029,  31.34612668, 50.05258026,
                             33.50795135,  34.33290915,  33.37659794, 32.55366866,  30.99947609,
                             35.47599224,  39.74032484, 48.20281941,  36.76442332,  33.44254829,
                             30.84983456, 32.96748179,  34.94445824,  36.59958041,  34.64582225,
                             44.42460843,  28.77922249,  20.29438548,  20.14281416, 26.51264502,
                             26.49364339,  27.59002478,  29.05427252])

        #mean_vec = np.array([46, 31, 20, 19, 26, 24, 25, 28, 49, 39, 35, 31, 32, 33, 34, 33, 48,
        #                     34, 36, 35, 32, 30, 33, 36, 44, 28, 27, 26, 26, 19, 20, 28, 60, 37,
        #                     23, 26, 36, 31, 28, 36, 62, 42, 33, 33, 39, 35, 33, 39, 59, 39, 35,
        #                     37, 40, 33, 32, 41, 57, 36, 29, 33, 37, 27, 23, 36, 60, 35, 28, 31,
        #                     36, 26, 23, 37, 62, 39, 33, 36, 39, 33, 33, 42, 59, 40, 32, 33, 40,
        #                     37, 35, 39, 57, 35, 23, 27, 37, 33, 29, 36, 46, 27, 25, 24, 26, 19,
        #                     20, 31, 50, 33, 34, 33, 32, 30, 35, 39, 48, 36, 33, 30, 32, 34, 36,
        #                     34, 44, 28, 20, 20, 26, 26, 27, 29], dtype=np.uint8)

        arr1 = all_vecs
        arr2 = mean_vec[None, :].astype(np.float32)

        def batch_subtract(arr1, arr2):
            # Center the vectors
            out = np.empty(arr1.shape, dtype=np.int8)
            chunksize = int(1E5)
            slices = list(ut.ichunk_slices(arr1.shape[0], chunksize))
            for sl in ut.ProgIter(slices, lbl='apply centering'):
                s = arr1[sl].astype(np.float32)
                out[sl] = np.subtract(s, arr2).astype(np.int8)
            return out

    #num_words = 8000

    if config['num_words'] == 1E6:
        import vtool as vt
        print('Using oxford word assignments')
        unique_word, groupxs = vt.group_indices(np.hstack(wordid_list))
        assert ut.issorted(unique_word)
        wx_to_vecs = vt.apply_grouping(all_vecs, groupxs, axis=0)

        wx_to_word = np.array([
            np.round(np.mean(sift_group, axis=0)).astype(np.uint8)
            for sift_group in ut.ProgIter(wx_to_vecs, lbl='compute words')
        ])
        from ibeis.algo.smk import vocab_indexer
        vocab = vocab_indexer.VisualVocab(wx_to_word)

        wx_lists = [wids[:, None] - 1 for wids in wordid_list]
    else:
        vocab = train_vocabulary(all_vecs, config['num_words'])
        word_hash = ut.hashstr_arr27(vocab.wx_to_word, 'words')
        relevant_params = ['checks']
        cfglbl = ut.get_cfg_lbl(ut.dict_subset(config, relevant_params))
        assign_fpath = join(dbdir, 'assigns_' + ut.hashstr27(word_hash + cfglbl) + '.pkl')
        if ut.checkpath(assign_fpath):
            (idx_to_wxs, offset_list) = ut.load_data(assign_fpath)
        else:
            #query_hash = ut.hashstr_arr27(all_vecs, 'all_vecs')
            with ut.Timer('assign vocab neighbors'):
                #idx_to_wxs, idx_to_maws = smk_funcs.assign_to_words(vocab, all_vecs, nAssign)
                _idx_to_wx, _idx_to_wdist = vocab.nn_index(all_vecs, nAssign, checks=128)
                if nAssign > 1:
                    idx_to_wxs, idx_to_maws = smk_funcs.weight_multi_assigns(
                        _idx_to_wx, _idx_to_wdist, massign_alpha=1.2, massign_sigma=80.0,
                        massign_equal_weights=True)
                else:
                    idx_to_wxs = _idx_to_wx.tolist()
                    #idx_to_maws = [[1.0]] * len(idx_to_wxs)
            # Maybe masked arrays is just overall better here
            idx_to_wxs = [np.array(wxs, dtype=np.int32) for wxs in ut.ProgIter(idx_to_wxs, lbl='cast')]
            ut.save_data(assign_fpath, (idx_to_wxs, offset_list))
        wx_lists = [idx_to_wxs[l:r] for l, r in ut.itertwo(offset_list)]

    relevant_params = ['num_words', 'root_sift', 'checks']
    cfglbl = ut.get_cfg_lbl(ut.dict_subset(config, relevant_params))
    ydata_fpath = join(dbdir, 'ydata' + cfglbl + '.pkl')
    if not ut.checkpath(ydata_fpath):
        Y_list = []
        _prog = ut.ProgPartial(nTotal=len(daids), lbl='new Y', bs=True, adjust=True)
        for aid, fx_to_wxs in _prog(zip(daids, wx_lists)):
            Y = new_external_annot(aid, fx_to_wxs)
            Y_list.append(Y)
        ut.save_data(ydata_fpath, Y_list)
    else:
        Y_list = ut.load_data(ydata_fpath)

    #======================
    # Build/load query info
    _qannots = ibs.annots(ibs.filter_annots_general(has_any='query'))
    # Reorder by name
    unique_names, groupxs = ut.group_indices(_qannots.names)
    _qannots = _qannots.take(ut.flatten(groupxs))
    # Map each query annot to its corresponding data index
    dgid_to_dx = ut.make_index_lookup(data_annots.gids)
    qx_to_dx = ut.take(dgid_to_dx, _qannots.gids)
    # get_oxford_keys(_qannots)
    relevant_params = ['num_words', 'root_sift', 'checks']
    cfglbl = ut.get_cfg_lbl(ut.dict_subset(config, relevant_params))
    xquery_fpath = join(dbdir, 'xquery' + cfglbl + '.pkl')

    if not ut.checkpath(xquery_fpath):
        query_super_kpts = ut.take(kpts_list, qx_to_dx)
        query_super_vecs = ut.take(vecs_list, qx_to_dx)
        query_super_wxs = ut.take(wx_lists, qx_to_dx)
        import vtool as vt
        # Mark which keypoints are within the bbox of the query
        query_flags_list = []
        for kpts, bbox in zip(query_super_kpts, _qannots.bboxes):
            xys = kpts[:, 0:2]
            wh_list = vt.get_kpts_wh(kpts)
            radii = wh_list / 2
            pts1 = xys + radii * (-1, 1)
            pts2 = xys + radii * (-1, -1)
            pts3 = xys + radii * (1, -1)
            pts4 = xys + radii * (1, 1)
            flags = np.logical_and.reduce([
                vt.point_inside_bbox(pts1.T, bbox),
                vt.point_inside_bbox(pts2.T, bbox),
                vt.point_inside_bbox(pts3.T, bbox),
                vt.point_inside_bbox(pts4.T, bbox),
            ])
            query_flags_list.append(flags)

        qaids = _qannots.aids
        query_kpts = vt.zipcompress(query_super_kpts, query_flags_list, axis=0)
        query_vecs = vt.zipcompress(query_super_vecs, query_flags_list, axis=0)
        query_wxs = vt.zipcompress(query_super_wxs, query_flags_list, axis=0)

        X_list = []
        _prog = ut.ProgPartial(nTotal=len(qaids), lbl='new X', bs=True, adjust=True)
        for aid, fx_to_wxs in _prog(zip(qaids, query_wxs)):
            X = new_external_annot(aid, fx_to_wxs)
            X_list.append(X)

        ut.save_data(xquery_fpath, X_list)
    else:
        X_list = ut.load_data(xquery_fpath)

    #======================
    # Add in some groundtruth
    for Y, nid in zip(Y_list, ibs.get_annot_nids(daids)):
        Y.nid = nid

    for X, nid in zip(X_list, ibs.get_annot_nids(qaids)):
        X.nid = nid

    for Y, qual in zip(Y_list, ibs.get_annot_quality_texts(daids)):
        Y.qual = qual

    #======================
    # Add in other properties
    for Y, vecs, kpts in zip(Y_list, vecs_list, kpts_list):
        Y.vecs = vecs
        Y.kpts = kpts

    for X, vecs, kpts in zip(X_list, query_vecs, query_kpts):
        X.kpts = kpts
        X.vecs = vecs

    #======================
    # Build inverted list
    daids = [Y.aid for Y in Y_list]
    wx_list = sorted(ut.list_union(*[Y.wx_list for Y in Y_list]))
    assert daids == data_annots.aids
    assert len(wx_list) <= config['num_words']

    #wx_to_aids = smk_funcs.invert_lists(
    #    daids, [Y.wx_list for Y in Y_list], all_wxs=wx_list)

    # Compute IDF weights
    ndocs_total = len(daids)
    if False:
        # ndocs_per_word1 = np.array([len(set(wx_to_aids[wx])) for wx in wx_list])
        pass
    else:
        # use total count of words like in Video Google
        ndocs_per_word2 = np.bincount(ut.flatten([Y.wx_list for Y in Y_list]))
        ndocs_per_word = ndocs_per_word2
    idf_per_word = smk_funcs.inv_doc_freq(ndocs_total, ndocs_per_word)
    wx_to_weight = dict(zip(wx_list, idf_per_word))
    print('idf stats: ' + ut.repr4(ut.get_stats(wx_to_weight.values())))

    # Filter junk
    Y_list_ = [Y for Y in Y_list if Y.qual != 'junk']
    test_kernel(X_list, Y_list_, vocab, wx_to_weight)


def test_kernel(X_list, Y_list_, vocab, wx_to_weight):
    #======================
    # Choose Query Kernel

    params = {
        'asmk': dict(alpha=3.0, thresh=0.0),
        'bow': dict(),
        'bow2': dict(),
    }
    method = 'bow'
    method = 'bow2'
    method = 'asmk'
    smk = SMK(wx_to_weight, method=method, **params[method])

    # Specific info for the type of query
    if method == 'asmk':
        # Make residual vectors:
        _prog = ut.ProgPartial(lbl='agg Y rvecs', bs=True, adjust=True)
        for Y in _prog(Y_list_):
            make_agg_vecs(Y, vocab, Y.vecs)

        _prog = ut.ProgPartial(lbl='agg X rvecs', bs=True, adjust=True)
        for X in _prog(X_list):
            make_agg_vecs(X, vocab, X.vecs)

    if method == 'bow2':
        # Hack for orig tf-idf bow vector
        nwords = len(vocab)
        for X in ut.ProgIter(X_list, lbl='make bow vector'):
            ensure_tf(X)
            bow_vector(X, wx_to_weight, nwords)

        for Y in ut.ProgIter(Y_list_, lbl='make bow vector'):
            ensure_tf(X)
            bow_vector(Y, wx_to_weight, nwords)
    else:
        for X in ut.ProgIter(X_list, 'compute X gamma'):
            X.gamma = smk.gamma(X)
        for Y in ut.ProgIter(Y_list_, 'compute Y gamma'):
            Y.gamma = smk.gamma(Y)

    # Execute matches (could go faster by enumerating candidates)
    scores_list = []
    for X in ut.ProgIter(X_list, lbl='query %s' % (smk,)):
        scores = [smk.kernel(X, Y) for Y in Y_list_]
        scores_list.append(scores)

    import sklearn.metrics
    avep_list = []
    _iter = list(zip(scores_list, X_list))
    _iter = ut.ProgIter(_iter, lbl='evaluate %s' % (smk,))
    for scores, X in _iter:
        truth = [X.nid == Y.nid for Y in Y_list_]
        avep = sklearn.metrics.average_precision_score(truth, scores)
        avep_list.append(avep)

        if False:
            sortx = np.argsort(scores)[::-1]
            yx = np.arange(len(Y_list_))
            truth_ranked = np.array(truth).take(sortx)
            scores_ranked = np.array(scores).take(sortx)
            Y_ranked = ut.take(Y_list_, sortx)
            Y_gts = ut.compress(Y_ranked, truth_ranked)

            Y = Y_gts[-1]
            ibs.show_annot(Y_gts[-1].aid, annote=False)
            ibs.show_annot(Y_gts[-1].aid, annote=False)
            #yx[truth]
            #gt_yxs = yx.take(sortx)[truth]
            #Y = Y_list_[gt_yxs[-1]]

    avep_list = np.array(avep_list)
    mAP = np.mean(avep_list)
    print('mAP  = %r' % (mAP,))


def new_external_annot(aid, fx_to_wxs=None):
    #nAssign = 1
    int_rvec = True
    # Compute assignments
    #fx_to_vecs = vecs
    #if fx_to_wxs is None:
    #    fx_to_wxs, fx_to_maws = smk_funcs.assign_to_words(vocab, fx_to_vecs,
    #                                                      nAssign)
    #else:
    fx_to_maws = [np.ones(len(wxs), dtype=np.float32) for wxs in fx_to_wxs]
    """
    z = np.array(ut.take_column(fx_to_wxs, 0)) + 1
    y = np.array(wordid_list[0])
    float((z == y).sum()) / len(y)

    vocab.flann_params['checks'] = 5120
    vocab.flann_params['trees'] = 8
    vocab.build()
    """
    wx_to_fxs, wx_to_maws = smk_funcs.invert_assigns(fx_to_wxs, fx_to_maws)
    X = inverted_index.SingleAnnot()
    X.aid = aid
    # Build Aggregate Residual Vectors
    X.wx_list = np.array(sorted(wx_to_fxs.keys()), dtype=np.int32)
    X.wx_to_idx = ut.make_index_lookup(X.wx_list)
    X.int_rvec = int_rvec
    X.wx_set = set(X.wx_list)
    X.fxs_list = ut.take(wx_to_fxs, X.wx_list)
    X.maws_list = ut.take(wx_to_maws, X.wx_list)
    return X


def make_agg_vecs(X, vocab, fx_to_vecs):
    word_list = ut.take(vocab.wx_to_word, X.wx_list)
    if X.int_rvec:
        X.agg_rvecs = np.empty((len(X.wx_list), fx_to_vecs.shape[1]),
                               dtype=np.int8)
    else:
        X.agg_rvecs = np.empty((len(X.wx_list), fx_to_vecs.shape[1]),
                               dtype=np.float)
    X.agg_flags = np.empty((len(X.wx_list), 1), dtype=np.bool)
    for idx in range(len(X.wx_list)):
        word = word_list[idx]
        fxs = X.fxs_list[idx]
        maws = X.maws_list[idx]
        vecs = fx_to_vecs.take(fxs, axis=0)
        _rvecs, _flags = smk_funcs.compute_rvec(vecs, word)
        _agg_rvec, _agg_flag = smk_funcs.aggregate_rvecs(_rvecs, maws, _flags)
        if X.int_rvec:
            _agg_rvec = smk_funcs.cast_residual_integer(_agg_rvec)
        X.agg_rvecs[idx] = _agg_rvec
        X.agg_flags[idx] = _agg_flag
    return X


def ensure_tf(X):
    termfreq = ut.dict_hist(X.wx_list)
    # do what video google does
    termfreq = ut.map_dict_vals(lambda x: x / len(X.wx_list), termfreq)
    X.termfreq = termfreq


def bow_vector(X, wx_to_weight, nwords):
    """
    nwords = len(vocab)
    for X in ut.ProgIter(X_list):
        bow_vector(X, wx_to_weight, nwords)

    for Y in ut.ProgIter(Y_list):
        bow_vector(Y, wx_to_weight, nwords)

    """
    import vtool as vt
    wxs = sorted(list(X.wx_set))
    tf = np.array(ut.take(X.termfreq, wxs))
    idf = np.array(ut.take(wx_to_weight, wxs))
    bow_ = tf * idf
    bow_ = vt.normalize(bow_)
    bow = SparseVector(dict(zip(wxs, bow_)))
    X.bow = bow


def make_temporary_annot(aid, vocab, wx_to_weight, ibs, config):
    nAssign = config.get('nAssign', 1)
    alpha = config.get('smk_alpha', 3.0)
    thresh = config.get('smk_thresh', 3.0)
    # Compute assignments
    fx_to_vecs = ibs.get_annot_vecs(aid, config2_=config)
    fx_to_wxs, fx_to_maws = smk_funcs.assign_to_words(vocab, fx_to_vecs, nAssign)
    wx_to_fxs, wx_to_maws = smk_funcs.invert_assigns(fx_to_wxs, fx_to_maws)
    # Build Aggregate Residual Vectors
    wx_list = sorted(wx_to_fxs.keys())
    word_list = ut.take(vocab.wx_to_word, wx_list)
    fxs_list = ut.take(wx_to_fxs, wx_list)
    maws_list = ut.take(wx_to_maws, wx_list)
    agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.float)
    agg_flags = np.empty((len(wx_list), 1), dtype=np.bool)
    for idx in range(len(wx_list)):
        word = word_list[idx]
        fxs = fxs_list[idx]
        maws = maws_list[idx]
        vecs = fx_to_vecs.take(fxs, axis=0)
        _rvecs, _flags = smk_funcs.compute_rvec(vecs, word)
        _agg_rvec, _agg_flag = smk_funcs.aggregate_rvecs(_rvecs, maws, _flags)
        agg_rvecs[idx] = _agg_rvec
        agg_flags[idx] = _agg_flag
    X = inverted_index.SingleAnnot()
    X.aid = aid
    X.wx_list = wx_list
    X.fxs_list = fxs_list
    X.maws_list = maws_list
    X.agg_rvecs = agg_rvecs
    X.agg_flags = agg_flags
    X.wx_to_idx = ut.make_index_lookup(X.wx_list)
    X.int_rvec = False
    X.wx_set = set(X.wx_list)

    weight_list = np.array(ut.take(wx_to_weight, wx_list))
    X.gamma = smk_funcs.gamma_agg(X.agg_rvecs, X.agg_flags, weight_list,
                                  alpha, thresh)
    return X


def verify_score():
    """
    Recompute all SMK things for two annotations and compare scores.

    >>> from ibeis.algo.smk.script_smk import *  # NOQA

    cm.print_inspect_str(qreq_)
    cm.show_single_annotmatch(qreq_, daid1)
    cm.show_single_annotmatch(qreq_, daid2)
    """
    qreq_, cm = load_internal_data()
    qreq_.ensure_data()

    ibs = qreq_.ibs
    qaid = cm.qaid
    daid1 = cm.get_top_truth_aids(ibs, ibs.const.TRUTH_MATCH)[0]
    daid2 = cm.get_top_truth_aids(ibs, ibs.const.TRUTH_MATCH, invert=True)[0]

    vocab = ibs.depc['vocab'].get_row_data([qreq_.dinva.vocab_rowid], 'words')[0]
    wx_to_weight = qreq_.dinva.wx_to_weight

    aid = qaid  # NOQA
    config = qreq_.qparams

    alpha = config.get('smk_alpha', 3.0)
    thresh = config.get('smk_thresh', 3.0)
    X = make_temporary_annot(qaid, vocab, wx_to_weight, ibs, config)
    assert np.isclose(smk_pipeline.match_kernel_agg(X, X, wx_to_weight, alpha, thresh)[0], 1.0)

    Y1 = make_temporary_annot(daid1, vocab, wx_to_weight, ibs, config)
    item = smk_pipeline.match_kernel_agg(X, Y1, wx_to_weight, alpha, thresh)
    score = item[0]
    assert np.isclose(score, cm.aid2_annot_score[daid1])
    assert np.isclose(smk_pipeline.match_kernel_agg(Y1, Y1, wx_to_weight, alpha, thresh)[0], 1.0)

    Y2 = make_temporary_annot(daid2, vocab, wx_to_weight, ibs, config)
    item = smk_pipeline.match_kernel_agg(X, Y2, wx_to_weight, alpha, thresh)
    score = item[0]
    assert np.isclose(score, cm.aid2_annot_score[daid2])
    assert np.isclose(smk_pipeline.match_kernel_agg(Y2, Y2, wx_to_weight, alpha, thresh)[0], 1.0)
    #Y2 = make_temporary_annot(daid2, vocab, wx_to_weight, ibs, config)


#ap_list2 = []
## Sanity Check: calculate avep like in
## http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
## It ends up the same. Good.
#for truth, scores in zip(truths_list, scores_list):
#    # HACK THE SORTING SO GT WITH EQUAL VALUES COME FIRST
#    # (I feel like this is disingenuous, but lets see how it changes AP)
#    if True:
#        sortx = np.lexsort(np.vstack((scores, truth))[::-1])[::-1]
#        truth = truth.take(sortx)
#        scores = scores.take(sortx)
#    old_precision = 1.0
#    old_recall = 0.0
#    ap = 0.0
#    intersect_size = 0.0
#    npos = float(sum(truth))
#    pr = []
#    j = 0
#    for i in range(len(truth)):
#        score = scores[i]
#        if score > 0:
#            val = truth[i]
#            if val:
#                intersect_size += 1
#            recall = intersect_size / npos
#            precision = intersect_size / (j + 1)
#            ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
#            pr.append((precision, recall))
#            old_recall = recall
#            old_precision = precision
#            j += 1

#    ap_list2.append(ap)
#ap_list2 = np.array(ap_list2)
#print('mAP2 = %r' % (np.mean(ap_list2)))

# annots._internal_attrs['kpts'] = kpts_list
# annots._internal_attrs['vecs'] = vecs_list
# annots._internal_attrs['wordid'] = wordid_list
# annots._ibs = None
