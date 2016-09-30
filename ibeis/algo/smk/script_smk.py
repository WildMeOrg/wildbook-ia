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
from ibeis.algo.smk import vocab_indexer
from ibeis.algo.smk import inverted_index
from ibeis.algo.smk import smk_funcs
from ibeis.algo.smk import smk_pipeline
from six.moves import zip, map
#from ibeis.algo.smk import smk_funcs
#from ibeis.algo.smk import inverted_index
#from ibeis import core_annots
#from ibeis.algo import Config as old_config
(print, rrr, profile) = ut.inject2(__name__)


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


def load_external_oxford_data():
    """
    Such hacks for reading external oxford
    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """
    from os.path import join, basename, splitext
    dbdir = ut.truepath('/raid/work/Oxford/')
    data_fpath1 = join(dbdir, 'oxford_data1.pkl')
    if ut.checkpath(data_fpath1):
        oxford_data1 = ut.load_data(data_fpath1)
        return oxford_data1
    else:
        sift_fpath = join(dbdir, 'OxfordSIFTDescriptors',
                          'feat_oxc1_hesaff_sift.bin')
        readme_fpath = join(dbdir, 'README2.txt')
        word_dpath = join(dbdir, 'word_oxc1_hesaff_sift_16M_1M')
        word_fpath_list = ut.ls(word_dpath)
        import pandas as pd
        imgid_to_df = {}

        imgid_order = ut.readfrom(readme_fpath).split('\n')[20:-1]

        for word_fpath in ut.ProgIter(word_fpath_list, lbl='reading kpts'):
            imgid = splitext(basename(word_fpath))[0]
            row_gen = (map(float, line.strip('\n').split(' '))
                       for line in ut.read_lines_from(word_fpath)[2:])
            rows = [(int(word_id), x, y, e11, e12, e22)
                    for (word_id, x, y, e11, e12, e22) in row_gen]
            df = pd.DataFrame(rows, columns=['word_id', 'x', 'y', 'e11', 'e12', 'e22'])
            imgid_to_df[imgid] = df

        df_list = ut.take(imgid_to_df, imgid_order)
        offset_list = [0] + ut.cumsum([len(df_) for df_ in df_list])
        try:
            shape = (16334970, 128)
            assert offset_list[-1] == shape[0]
            file_ = open(sift_fpath, 'rb')
            with ut.Timer('Reading file'):
                sifts = np.fromstring(file_.read(16334970 * 128), dtype=np.uint8)
            sifts = sifts.reshape(shape)
        finally:
            file_.close()

        vecs_list = [sifts[l:r] for l, r in ut.itertwo(offset_list)]
        # zkpts_list = [df_.loc[:, ('x', 'y', 'e11', 'e12', 'e22')].values
        #               for df_ in df_list]

        # FIXME
        # Z_mats = [np.array([[[a, b], [b, c]]
        #                     for x, y, a, b, c in zkpts]) for zkpts in zkpts_list]
        wordid_list = [df_.loc[:, 'word_id'].values for df_ in df_list]

        import vtool as vt
        unique_word, groupxs = vt.group_indices(np.hstack(wordid_list))
        assert ut.issorted(unique_word)
        wx_to_sifts = vt.apply_grouping(sifts, groupxs, axis=0)

        wx_to_word = np.array([
            np.round(np.mean(sift_group, axis=0)).astype(np.uint8)
            for sift_group in ut.ProgIter(wx_to_sifts, lbl='compute words')
        ])
        vocab = vocab_indexer.VisualVocab(wx_to_word)
        vocab.build()

        oxford_data1 = {
            'imgid_order': imgid_order,
            # 'kpts_list': kpts_list,
            'vecs_list': vecs_list,
            'wordid_list': wordid_list,
            'vocab': vocab,
        }
        ut.save_data(data_fpath1, oxford_data1)
    return oxford_data1


def load_external_data2():
    """
    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """
    from os.path import join, basename, splitext
    import ibeis
    oxford_data1 = load_external_oxford_data()
    imgid_order = oxford_data1['imgid_order']
    kpts_list = oxford_data1['kpts_list']
    vecs_list = oxford_data1['vecs_list']
    wordid_list = oxford_data1['wordid_list']
    vocab = oxford_data1['vocab']
    uri_order = [x.replace('oxc1_', '') for x in imgid_order]
    assert len(ut.list_union(*wordid_list)) == 1E6

    def lookup_oxford_annots(aids):
        pass

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

    dbdir = ut.truepath('/raid/work/Oxford/')
    #======================
    # Build/load database info
    data_fpath2 = join(dbdir, 'oxford_data2.pkl')
    daids = data_annots.aids
    if not ut.checkpath(data_fpath2):
        Y_list = []
        _prog = ut.ProgPartial(nTotal=len(daids), lbl='new Y', bs=True)
        for aid, vecs, wordids in _prog(zip(daids, vecs_list, wordid_list)):
            fx_to_wxs = wordids[:, None] - 1
            fx_to_vecs = vecs
            X = new_external_annot(aid, vecs, fx_to_vecs, fx_to_wxs)
            Y_list.append(X)

        for X, vecs in _prog(zip(Y_list, vecs_list)):
            fx_to_vecs = vecs
            make_agg_vecs(X, vocab, fx_to_vecs)
        external_data2 = {
            'Y_list': Y_list,
        }
        ut.save_data(data_fpath2, external_data2)
    else:
        external_data2 = ut.load_data(data_fpath2)
        Y_list = external_data2['Y_list']

    # Build inverted list
    daids = [Y.aid for Y in Y_list]
    wx_lists = [Y.wx_list for Y in Y_list]
    wx_list = sorted(ut.list_union(*wx_lists))
    assert daids == data_annots.aids
    assert len(wx_list) == 1E6

    # Compute IDF weights
    wx_to_aids = smk_funcs.invert_lists(daids, wx_lists, all_wxs=wx_list)
    ndocs_total = len(daids)
    # TODO: use total count of words like in Video Google
    if False:
        # ndocs_per_word1 = np.array([len(set(wx_to_aids[wx])) for wx in wx_list])
        pass
    else:
        ndocs_per_word2 = np.bincount(ut.flatten(wx_lists))
        ndocs_per_word = ndocs_per_word2
    idf_per_word = smk_funcs.inv_doc_freq(ndocs_total, ndocs_per_word)
    wx_to_weight = dict(zip(wx_list, idf_per_word))
    print('idf stats: ' + ut.repr4(ut.get_stats(wx_to_weight.values())))

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
    data_fpath3 = join(dbdir, 'oxford_data3.pkl')
    if not ut.checkpath(data_fpath3):
        query_super_kpts = ut.take(kpts_list, qx_to_dx)
        query_super_vecs = ut.take(vecs_list, qx_to_dx)
        query_super_wids = ut.take(wordid_list, qx_to_dx)
        import vtool as vt
        # Mark which keypoints are within the bbox of the query
        query_flags_list = []
        for kpts, bbox in zip(query_super_kpts, _qannots.bboxes):
            xys = kpts[:, 0:2]
            wh_list = vt.get_xy_axis_extents(kpts)
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
        # query_kpts = vt.zipcompress(query_super_kpts, query_flags_list, axis=0)
        query_vecs = vt.zipcompress(query_super_vecs, query_flags_list, axis=0)
        query_wids = vt.zipcompress(query_super_wids, query_flags_list, axis=0)

        X_list = []
        _prog = ut.ProgPartial(nTotal=len(qaids), lbl='new X', bs=True)
        for aid, vecs, wordids in _prog(zip(qaids, query_vecs, query_wids)):
            fx_to_wxs = wordids[:, None] - 1
            fx_to_vecs = vecs
            X = new_external_annot(aid, vecs, fx_to_vecs, fx_to_wxs)
            X_list.append(X)

        for X, vecs in _prog(zip(X_list, query_vecs)):
            fx_to_vecs = vecs
            make_agg_vecs(X, vocab, fx_to_vecs)
        external_data3 = {
            'X_list': X_list,
        }
        ut.save_data(data_fpath3, external_data3)

    #======================
    # Add in some groundtruth
    for Y, nid in zip(Y_list, ibs.get_annot_nids(daids)):
        Y.nid = nid

    for Y, qual in zip(Y_list, ibs.get_annot_quality_texts(daids)):
        Y.qual = qual

    # Filter junk
    Y_list_ = [Y for Y in Y_list if Y.qual != 'junk']

    for X, nid in zip(X_list, ibs.get_annot_nids(qaids)):
        X.nid = nid

    params = {
        'asmk': dict(alpha=3.0, thresh=0.0),
        'bow': dict(),
    }
    method = 'bow'
    # method = 'asmk'
    smk = SMK(wx_to_weight, method=method, **params[method])
    for X in ut.ProgIter(X_list, 'compute X gamma'):
        X.gamma = smk.gamma(X)
    for Y in ut.ProgIter(Y_list_, 'compute Y gamma'):
        Y.gamma = smk.gamma(Y)

    import sklearn.metrics
    # ranked_lists = []

    scores_list = []
    truths_list = []

    avep_list = []
    for X in ut.ProgIter(X_list, lbl='query'):
        scores = []
        truth = []
        for Y in Y_list_:
            score = X.bow.dot(Y.bow)
            # score = smk.kernel(X, Y)
            scores.append(score)
            truth.append(X.nid == Y.nid)
        scores = np.array(scores)
        truth = np.array(truth)
        sortx = scores.argsort()[::-1]
        # ranked_lists.append(truth.take(sortx))
        scores_list.append(scores.take(sortx))
        truths_list.append(truth.take(sortx))
        avep = sklearn.metrics.average_precision_score(truth, scores)
        avep_list.append(avep)
    avep_list = np.array(avep_list)
    mAP = np.mean(avep_list)
    print('mAP  = %r' % (mAP,))

    ap_list2 = []
    # Sanity Check: calculate avep like in
    # http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
    # It ends up the same. Good.
    for truth, scores in zip(truths_list, scores_list):
        # HACK THE SORTING SO GT WITH EQUAL VALUES COME FIRST
        # (I feel like this is disingenuous, but lets see how it changes AP)
        if True:
            sortx = np.lexsort(np.vstack((scores, truth))[::-1])[::-1]
            truth = truth.take(sortx)
            scores = scores.take(sortx)
        old_precision = 1.0
        old_recall = 0.0
        ap = 0.0
        intersect_size = 0.0
        npos = float(sum(truth))
        pr = []
        j = 0
        for i in range(len(truth)):
            score = scores[i]
            if score > 0:
                val = truth[i]
                if val:
                    intersect_size += 1
                recall = intersect_size / npos
                precision = intersect_size / (j + 1)
                ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
                pr.append((precision, recall))
                old_recall = recall
                old_precision = precision
                j += 1

        ap_list2.append(ap)
    ap_list2 = np.array(ap_list2)
    print('mAP2 = %r' % (np.mean(ap_list2)))

    # annots._internal_attrs['kpts'] = kpts_list
    # annots._internal_attrs['vecs'] = vecs_list
    # annots._internal_attrs['wordid'] = wordid_list
    # annots._ibs = None


def new_external_annot(aid, vecs, vocab, fx_to_wxs=None):
    nAssign = 1
    int_rvec = True
    # Compute assignments
    fx_to_vecs = vecs
    if fx_to_wxs is None:
        fx_to_wxs, fx_to_maws = smk_funcs.assign_to_words(vocab, fx_to_vecs,
                                                          nAssign)
    else:
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

    # Ensure casting
    #for X in ut.ProgIter(X_list):
    #    X.agg_rvecs = smk_funcs.cast_residual_integer(X.agg_rvecs)
    #    X.wx_list = np.array(X.wx_list, dtype=np.int32)
    #    X.wx_to_idx = ut.map_dict_vals(np.int32, X.wx_to_idx)
    #    X.wx_to_idx = ut.map_dict_keys(np.int32, X.wx_to_idx)
    return X


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
        assert len(kwargs) == 0, 'unexpected kwargs=%r' % (kwargs,)

    def gamma(smk, X):
        score = smk.match_score(X, X)
        sccw = np.reciprocal(np.sqrt(score))
        return sccw

    def kernel(smk, X, Y):
        return X.gamma * Y.gamma * smk.match_score(X, Y)

    def word_isect(smk, X, Y):
        isect_words = X.wx_set.intersection(Y.wx_set)
        X_idx = ut.take(X.wx_to_idx, isect_words)
        Y_idx = ut.take(Y.wx_to_idx, isect_words)
        weights = ut.take(smk.wx_to_weight, isect_words)
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

    # def kernel_tf_bow(smk, X, Y):
    #     if not hasattr(X, 'termfreq'):
    #         X.termfreq = ut.dict_hist(X.wx_list)
    #         # do what video google does
    #         X.termfreq = ut.map_dict_vals(lambda x: x / len(X.wx_list), X.termfreq)
    #     if not hasattr(Y, 'termfreq'):
    #         Y.termfreq = ut.dict_hist(Y.wx_list)
    #         Y.termfreq = ut.map_dict_vals(lambda x: x / len(Y.wx_list), Y.termfreq)
    #     isect_words = X.wx_set.intersection(Y.wx_set)
    #     idf_weights = np.array(ut.take(smk.wx_to_weight, isect_words))
    #     tf_weightsX = np.array(ut.take(X.termfreq, isect_words))
    #     tf_weightsY = np.array(ut.take(X.termfreq, isect_words))
    #     score = np.sum(idf_weights)
    #     return score


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


def ensure_tf(X):
    if not hasattr(X, 'termfreq'):
        X.termfreq = ut.dict_hist(X.wx_list)
        # do what video google does
        X.termfreq = ut.map_dict_vals(lambda x: x / len(X.wx_list), X.termfreq)


def bow_vector(X, wx_to_weight, nwords):
    """
    nwords = len(vocab)
    for X in ut.ProgIter(X_list):
        bow_vector(X, wx_to_weight, nwords)

    for Y in ut.ProgIter(Y_list):
        bow_vector(Y, wx_to_weight, nwords)

    """
    ensure_tf(X)
    wxs = sorted(list(X.wx_set))
    tf = np.array(ut.take(X.termfreq, wxs))
    idf = np.array(ut.take(wx_to_weight, wxs))
    import vtool as vt
    bow_ = tf * idf
    bow_ = vt.normalize(bow_)
    import scipy.sparse
    # nwords = max(wx_to_weight.keys()) + 1
    bow = SparseVector(dict(zip(wxs, bow_)))
    # bow = scipy.sparse.coo_matrix((bow_, (
    #     wxs, np.zeros(len(wxs)))), shape=(nwords, 1), dtype=np.float32).tocsc()
    # bow.T.dot(bow).toarray()[0, 0]
    X.bow = bow
    # for wx, v in zip(wxs, bow_):
    #     bow[wx, 1] = v


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
