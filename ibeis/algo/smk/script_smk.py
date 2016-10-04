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
        """ gamma(X) = (M(X, X)) ** (-1/2) """
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


def load_external_oxford_features(config):
    """
    # TODO: root sift with centering

    Such hacks for reading external oxford

    config = {
        'root_sift': True,
    }
    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """
    # relevant_params = [
    #     'root_sift'
    # ]
    # config = ut.dict_subset(config, relevant_params)
    from os.path import join, basename, splitext
    # suffix = ut.get_cfg_lbl(config)
    dbdir = ut.truepath('/raid/work/Oxford/')
    data_fpath0 = join(dbdir, 'oxford_data0.pkl')
    if ut.checkpath(data_fpath0):
        oxford_data1 = ut.load_data(data_fpath0)
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
        for imgid in ut.ProgIter(imgid_to_df.keys(), lbl='convert to invV'):
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

        # if config['root_sift']:
        #     # Have to do this in chunks to fit in memory
        #     import vtool as vt
        #     chunksize = shape[0] // 100
        #     slices = list(ut.ichunk_slices(shape[0], chunksize))
        #     fidelity = 512.0
        #     for sl in ut.ProgIter(slices, lbl='apply rootsift'):
        #         s = vecs[sl].astype(np.float32) / fidelity
        #         s = vt.normalize(s, ord=1, axis=1, out=s)
        #         s = np.sqrt(s, out=s)
        #         s = (s * (fidelity)).astype(np.uint8)
        #         vecs[sl] = s

        # vecs_list = [vecs[l:r] for l, r in ut.itertwo(offset_list)]
        kpts_list = [df_.loc[:, ('x', 'y', 'a', 'c', 'd')].values
                     for df_ in df_list]
        wordid_list = [df_.loc[:, 'word_id'].values for df_ in df_list]
        kpts = np.vstack(kpts_list)
        wordids = np.hstack(wordid_list)

        oxford_data0 = {
            'imgid_order': imgid_order,
            'offset_list': offset_list,
            'kpts': kpts,
            'vecs': vecs,
            'wordids': wordids,
            # 'kpts_list': kpts_list,
            # 'vecs_list': vecs_list,
        }
        ut.save_data(data_fpath0, oxford_data0)

        # if False:
        #     imgid = imgid_order[0]
        #     imgdir = join(dbdir, 'oxbuild_images')
        #     gpath = join(imgdir,  imgid.replace('oxc1_', '') + '.jpg')
        #     image = vt.imread(gpath)
        #     import plottool as pt
        #     pt.qt4ensure()
        #     pt.imshow(image)
        #     kpts = kpts_list[0].copy()
        #     vecs = vecs_list[0]
        #     #h, w = image.shape[0:2]
        #     #kpts.T[1] = h - kpts.T[1]

        #     #pt.draw_kpts2(kpts, ell_alpha=.4, pts=True, ell=True)
        #     pt.interact_keypoints.ishow_keypoints(image, kpts, vecs,
        #                                           ori=False, ell_alpha=.4,
        #                                           color='distinct')

    return oxford_data1


def load_jegou_oxford_data():
    from os.path import join
    smk_2013_dir = '/raid/work/Oxford/smk_data_iccv_2013/data/'

    # with open(join(smk_2013_dir, 'paris_sift.uint8'), 'rb') as file_:
    #     X = np.fromstring(file_.read(), np.uint8)
    #     X = X.reshape(len(X) / 128, 128)
    #     X = X.astype(np.float32)

    with open(join(smk_2013_dir, 'oxford_sift.uint8'), 'rb') as file_:
        X = np.fromstring(file_.read(), np.uint8)
        X = X.reshape(len(X) / 128, 128)
        X = X.astype(np.float32)

    import vtool as vt

    # Root SIFT
    np.sqrt(X, out=X)
    vt.normalize(X, ord=2, axis=1, out=X)

    # Get mean after root-sift
    Xm = np.mean(X, axis=0)

    # Center and then re-normalize
    np.subtract(X, Xm[None, :], out=X)
    vt.normalize(X, ord=2, axis=1, out=X)


def load_external_data2():
   """
   >>> from ibeis.algo.smk.script_smk import *  # NOQA
   """
   # FIXME: new_external_annot was not populated in namespace for with embed
   with ut.embed_on_exception_context:  # NOQA
    config = {
        'dtype': 'float32',
        'root_sift': True,
        'centering': True,
        'num_words': 2 ** 16,
        'checks': 1024,
        #'num_words': 1E6
        #'num_words': 8000,
    }
    # Define which params are relevant for which operations
    relevance = {}
    relevance['vecs'] = ['dtype', 'root_sift', 'centering']
    relevance['words'] = relevance['vecs'] + ['num_words']
    relevance['assign'] = relevance['words'] + ['checks']
    relevance['ydata'] = relevance['assign']
    relevance['xdata'] = relevance['assign']

    nAssign = 1

    class SMKCacher(ut.Cacher):
        def __init__(self, fname):
            relevant_params = relevance[fname]
            relevant_cfg = ut.dict_subset(config, relevant_params)
            cfgstr = ut.get_cfg_lbl(relevant_cfg)
            dbdir = ut.truepath('/raid/work/Oxford/')
            super(SMKCacher, self).__init__(fname, cfgstr, cache_dir=dbdir)

    # ==============================================
    # LOAD DATASET, EXTRACT AND POSTPROCESS FEATURES
    # ==============================================
    from os.path import basename, splitext
    import ibeis
    oxford_data0 = load_external_oxford_features(config)
    imgid_order = oxford_data0['imgid_order']
    offset_list = oxford_data0['offset_list']
    all_kpts    = oxford_data0['kpts']
    raw_vecs = oxford_data0['vecs']
    # wordids     = oxford_data0['wordids']
    del oxford_data0

    data_uri_order = [x.replace('oxc1_', '') for x in imgid_order]
    # assert len(np.unique(wordids)) == 1E6

    # Reqd standard query order
    dbdir = ut.truepath('/raid/work/Oxford/')
    query_files = sorted(ut.glob(dbdir + '/oxford_groundtruth', '*_query.txt'))
    query_uri_order = []
    for qpath in query_files:
        text = ut.readfrom(qpath, verbose=0)
        query_uri = text.split(' ')[0].replace('oxc1_', '')
        query_uri_order.append(query_uri)

    # Open the ibeis version of oxford
    ibs = ibeis.opendb('Oxford')

    def reorder_annots(_annots, uri_order):
        _images = ibs.images(_annots.gids)
        intern_uris = [splitext(basename(uri))[0]
                       for uri in _images.uris_original]
        lookup = ut.make_index_lookup(intern_uris)
        _reordered = _annots.take(ut.take(lookup, uri_order))
        return _reordered

    # Load database annotations and reorder them to agree with internals
    _dannots = ibs.annots(ibs.filter_annots_general(has_none='query'))
    data_annots = reorder_annots(_dannots, data_uri_order)

    # Load query annototations and reorder to standard order
    _qannots = ibs.annots(ibs.filter_annots_general(has_any='query'))
    query_annots = reorder_annots(_qannots, query_uri_order)

    # Map each query annot to its corresponding data index
    dgid_to_dx = ut.make_index_lookup(data_annots.gids)
    qx_to_dx = ut.take(dgid_to_dx, query_annots.gids)

    daids = data_annots.aids
    qaids = query_annots.aids

    # ================
    # PRE-PROCESS
    # ================
    import vtool as vt

    # Alias names to avoid errors in interactive sessions
    proc_vecs = raw_vecs
    del raw_vecs

    if config['dtype'] == 'float32':
        proc_vecs = proc_vecs.astype(np.float32)
    else:
        proc_vecs = proc_vecs
        raise NotImplementedError('other dtype')

    if config['root_sift']:
        with ut.Timer('Apply root sift'):
            np.sqrt(proc_vecs, out=proc_vecs)
            vt.normalize(proc_vecs, ord=2, axis=1, out=proc_vecs)

    if config['centering']:
        # Apply Centering
        with ut.Timer('Apply centering'):
            mean_vec = np.mean(proc_vecs, axis=0)
            # Center and then re-normalize
            np.subtract(proc_vecs, mean_vec[None, :], out=proc_vecs)
            vt.normalize(proc_vecs, ord=2, axis=1, out=proc_vecs)

    all_vecs = proc_vecs
    del proc_vecs

    # =====================================
    # BUILD VISUAL VOCABULARY
    # =====================================
    word_cacher = SMKCacher('words')
    words = word_cacher.tryload()
    if words is None:
        init_size = int(config['num_words'] * 2.5)
        with ut.embed_on_exception_context:
            import sklearn.cluster
            rng = np.random.RandomState(13421421)
            clusterer = sklearn.cluster.MiniBatchKMeans(
                config['num_words'], init_size=init_size,
                batch_size=5000, compute_labels=False, random_state=rng,
                n_init=3, verbose=5)
            clusterer.fit(all_vecs)
            words = clusterer.cluster_centers_
            word_cacher.save(words)

    if False:
        # Refine visual words
        with ut.embed_on_exception_context:
            import sklearn.cluster
            rng = np.random.RandomState(194932)
            clusterer = sklearn.cluster.MiniBatchKMeans(
                config['num_words'], init=words,
                batch_size=5000, compute_labels=False, random_state=rng,
                n_init=3, verbose=5)
            clusterer.fit(all_vecs)
            words = clusterer.cluster_centers_
        word_cacher.save(words)

    from ibeis.algo.smk import vocab_indexer
    vocab = vocab_indexer.VisualVocab(words)
    # vocab.flann_params['algorithm'] = 'linear'
    vocab.build()

    # =====================================
    # ASSIGN EACH VECTOR TO ITS NEAREST WORD
    # =====================================
    dassign_cacher = SMKCacher('assign')
    idx_to_wxs, idx_to_maws = dassign_cacher.tryload()
    if idx_to_wxs is None:
        with ut.Timer('assign vocab neighbors'):
            _idx_to_wx, _idx_to_wdist = vocab.nn_index(all_vecs, nAssign,
                                                       checks=config['checks'])
            if nAssign > 1:
                idx_to_wxs, idx_to_maws = smk_funcs.weight_multi_assigns(
                    _idx_to_wx, _idx_to_wdist, massign_alpha=1.2, massign_sigma=80.0,
                    massign_equal_weights=True)
            else:
                idx_to_wxs = np.ma.masked_array(_idx_to_wx, fill_value=-1)
                idx_to_maws = np.ma.ones(idx_to_wxs.shape, fill_value=-1,
                                         dtype=np.float32)
                idx_to_maws.mask = idx_to_wxs.mask
        dassign_cacher.save((idx_to_wxs, idx_to_maws))

    # Breakup vectors, keypoints, and word assignments by annotation
    wx_lists = [idx_to_wxs[l:r] for l, r in ut.itertwo(offset_list)]
    maw_lists = [idx_to_maws[l:r] for l, r in ut.itertwo(offset_list)]
    vecs_list = [all_vecs[l:r] for l, r in ut.itertwo(offset_list)]
    kpts_list = [all_kpts[l:r] for l, r in ut.itertwo(offset_list)]

    # =======================
    # FIND QUERY SUBREGIONS
    # =======================

    query_super_kpts = ut.take(kpts_list, qx_to_dx)
    query_super_vecs = ut.take(vecs_list, qx_to_dx)
    query_super_wxs  = ut.take(wx_lists, qx_to_dx)
    query_super_maws = ut.take(maw_lists, qx_to_dx)
    # Mark which keypoints are within the bbox of the query
    query_flags_list = []
    for kpts_, bbox in zip(query_super_kpts, query_annots.bboxes):
        flags = kpts_inside_bbox_aggressive(kpts_, bbox)
        query_flags_list.append(flags)

    print('Queries are crops of existing database images.')
    print('Looking at average percents')
    percent_list = [flags.sum() / flags.shape[0] for flags in query_flags_list]
    percent_stats = ut.get_stats(percent_list)
    print('percent_stats = %s' % (ut.repr4(percent_stats),))

    import vtool as vt
    query_kpts = vt.zipcompress(query_super_kpts, query_flags_list, axis=0)
    query_vecs = vt.zipcompress(query_super_vecs, query_flags_list, axis=0)
    query_wxs = vt.zipcompress(query_super_wxs, query_flags_list, axis=0)
    query_maws = vt.zipcompress(query_super_maws, query_flags_list, axis=0)

    def jegou_redone_agg_all():
        # Assume single assignment
        offset_list = np.array(offset_list)

        idx_to_dx = (np.searchsorted(
            offset_list, np.arange(len(idx_to_wxs)), side='right') - 1).astype(np.int32)
        wx_list = idx_to_wxs.T[0].compressed()
        unique_wx, groupxs = vt.group_indices(wx_list)

        dx_to_wxs = [np.unique(wxs) for wxs in wx_lists]
        dx_to_nagg = [len(wxs) for wxs in dx_to_wxs]
        agg_offset_list = np.array([0] + ut.cumsum(dx_to_nagg))
        num_agg_vecs = sum(dx_to_nagg)
        # Preallocate agg residuals for all dxs
        all_agg_wxs = np.hstack(dx_to_wxs)
        all_agg_vecs = np.empty((num_agg_vecs, dim), dtype=np.float32)
        all_agg_vecs[:, :] = np.nan

        # precompute agg residual stack
        wx_to_dxs = vt.apply_grouping(idx_to_dx, groupxs)
        subgroup = [vt.group_indices(dxs) for dxs in ut.ProgIter(wx_to_dxs)]
        wx_to_unique_dxs = ut.take_column(subgroup, 0)
        wx_to_dx_groupxs = ut.take_column(subgroup, 1)
        num_words = len(unique_wx)

        for i in ut.ProgIter(range(num_words), 'agg'):
            wx = unique_wx[i]
            word = words[wx:wx + 1]
            xs = groupxs[i]

            dxs = wx_to_unique_dxs[wx]
            dx_groupxs = wx_to_dx_groupxs[wx]
            assert np.bincount(dxs).max() < 2

            offsets1 = agg_offset_list.take(dxs)
            offsets2 = np.array([np.where(dx_to_wxs[dx] == wx)[0][0] for dx in dxs])
            offsets = offsets1 + offsets2

            if __debug__:
                offset = agg_offset_list[dxs[0]]
                assert np.all(dx_to_wxs[dxs[0]] == all_agg_wxs[offset:offset +
                                                               dx_to_nagg[dxs[0]]])

            # Compute residual
            rvecs = all_vecs[xs] - word
            vt.normalize(rvecs, axis=1, out=rvecs)
            # Aggregate across same images
            grouped_rvecs = vt.apply_grouping(rvecs, dx_groupxs, axis=0)
            #
            agg_rvecs = np.array([rvec_group.sum(axis=0) for rvec_group in grouped_rvecs])
            vt.normalize(agg_rvecs, axis=1, out=agg_rvecs)


    def jegou_port_agg_all():
        """
        ...This really only works with the HE scheme

        %  d  input matrix with descriptors (concatenated for all images)
        %  v  input vector with visual words (concatenated for all images)
        %  n  input vector with number of feature per image
        %  da aggregated descriptors (concatenated for all images)
        %  va unique visual words for each image (concatenated for all images)
        %  na number of features per image after aggregation
        """

        def aggregate_port(v, d):
            # % aggregate descriptors per visual word for a single image
            # %  d   descriptors
            # %  v   visual words
            # %  da  aggregated descriptors
            # %  va  unique visual words
            va = np.unique(v);
            n = len(va);
            da = np.zeros((n, d.shape[1]), 'single');

            for i in range(n):
                f = np.where(v == va[i])[0]
                if len(f) == 1:
                    da[i, :] = d[f, :];
                else:
                    # compute mean descriptor here, median will be subtracted
                    # before binarizing that would be equal to the mean
                    # residual instead of aggregated residual but binarization
                    # of each produces the same binary vector
                    da[i, :] = np.mean(d[f, :], axis=0)
            return va, da

        n_ = np.diff(offset_list)
        d_ = all_vecs
        v_ = idx_to_wxs

        da = {} # agg descriptors per image
        va = {} # agg words per image
        na = {} # num agg

        cs = offset_list

        for i in ut.ProgIter(range(len(n_))):
            sl = slice(cs[i], cs[i + 1])
            v = v_[sl]
            d = d_[sl]
            va[i], da[i] = aggregate_port(v, d)
            na[i] = len(va[i])

    # =======================
    # CONSTRUCT QUERY / DATABASE REPR
    # =======================
    int_rvec = not config['dtype'].startswith('float')

    X_list = []
    _prog = ut.ProgPartial(nTotal=len(qaids), lbl='new X', bs=True, adjust=True)
    for aid, fx_to_wxs, fx_to_maws in _prog(zip(qaids, query_wxs, query_maws)):
        X = new_external_annot(aid, fx_to_wxs, fx_to_maws, int_rvec)
        X_list.append(X)

    ydata_cacher = SMKCacher('ydata')
    Y_list = ydata_cacher.tryload()
    if Y_list is None:
        Y_list = []
        _prog = ut.ProgPartial(nTotal=len(daids), lbl='new Y', bs=True, adjust=True)
        for aid, fx_to_wxs, fx_to_maws in _prog(zip(daids, wx_lists, maw_lists)):
            Y = new_external_annot(aid, fx_to_wxs, fx_to_maws, int_rvec)
            Y_list.append(Y)
        ydata_cacher.save(Y_list)

    #======================
    # Add in some groundtruth
    print('Add in some groundtruth')
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
    print('Building inverted list')
    daids = [Y.aid for Y in Y_list]
    # wx_list = sorted(ut.list_union(*[Y.wx_list for Y in Y_list]))
    wx_list = sorted(set.union(*[Y.wx_set for Y in Y_list]))
    assert daids == data_annots.aids
    assert len(wx_list) <= config['num_words']

    #wx_to_aids = smk_funcs.invert_lists(
    #    daids, [Y.wx_list for Y in Y_list], all_wxs=wx_list)

    # Compute IDF weights
    print('Compute IDF weights')
    ndocs_total = len(daids)
    if False:
        # ndocs_per_word1 = np.array([len(set(wx_to_aids[wx])) for wx in wx_list])
        pass
    else:
        # use total count of words like in Video Google
        ndocs_per_word2 = np.bincount(ut.flatten([Y.wx_list for Y in Y_list]))
        ndocs_per_word = ndocs_per_word2
    print('ndocs_perword stats: ' + ut.repr4(ut.get_stats(ndocs_per_word)))
    idf_per_word = smk_funcs.inv_doc_freq(ndocs_total, ndocs_per_word)
    wx_to_weight = dict(zip(wx_list, idf_per_word))
    print('idf stats: ' + ut.repr4(ut.get_stats(wx_to_weight.values())))

    # Filter junk
    Y_list_ = [Y for Y in Y_list if Y.qual != 'junk']
    test_kernel(ibs, X_list, Y_list_, vocab, wx_to_weight)


def sanity_checks():
    nfeat_list = np.diff(offset_list)
    for Y, nfeat in ut.ProgIter(zip(Y_list, nfeat_list), 'checking'):
        assert nfeat == sum(ut.lmap(len, Y.fxs_list))

    if False:
        # Visualize queries
        # Look at the standard query images here
        # http://www.robots.ox.ac.uk:5000/~vgg/publications/2007/Philbin07/philbin07.pdf
        from ibeis.viz import viz_chip
        import plottool as pt
        pt.qt4ensure()
        fnum = 1
        pnum_ = pt.make_pnum_nextgen(len(query_annots.aids) // 5, 5)
        for aid in ut.ProgIter(query_annots.aids):
            pnum = pnum_()
            viz_chip.show_chip(ibs, aid, in_image=True, annote=False,
                               notitle=True, draw_lbls=False,
                               fnum=fnum, pnum=pnum)


def test_kernel(ibs, X_list, Y_list_, vocab, wx_to_weight):
    #======================
    # Choose Query Kernel
    params = {
        'asmk': dict(alpha=3.0, thresh=0.0),
        'bow': dict(),
        'bow2': dict(),
    }
    method = 'bow'
    method = 'bow2'
    # method = 'asmk'
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
            ensure_tf(Y)
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
            # yx = np.arange(len(Y_list_))
            truth_ranked = np.array(truth).take(sortx)
            # scores_ranked = np.array(scores).take(sortx)
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


def new_external_annot(aid, fx_to_wxs, fx_to_maws, int_rvec):
    wx_to_fxs, wx_to_maws = smk_funcs.invert_assigns(fx_to_wxs, fx_to_maws)
    X = inverted_index.SingleAnnot()
    X.aid = aid
    # Build Aggregate Residual Vectors
    X.wx_list = np.array(sorted(wx_to_fxs.keys()), dtype=np.int32)
    X.wx_to_idx = ut.make_index_lookup(X.wx_list)
    X.int_rvec = int_rvec
    X.wx_set = set(X.wx_list)
    # TODO: maybe use offset list structure instead of heavy nesting
    X.fxs_list = ut.take(wx_to_fxs, X.wx_list)
    X.maws_list = ut.take(wx_to_maws, X.wx_list)
    return X


def make_agg_vecs(X, vocab, fx_to_vecs):
    word_list = ut.take(vocab.wx_to_word, X.wx_list)
    dtype = np.int8 if X.int_rvec else np.float32
    dim = fx_to_vecs.shape[1]
    X.agg_rvecs = np.empty((len(X.wx_list), dim), dtype=dtype)
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


def kpts_inside_bbox_aggressive(kpts, bbox):
    import vtool as vt
    # Be aggressive with what is allowed in the query
    xys = kpts[:, 0:2]
    flags = vt.point_inside_bbox(xys.T, bbox)
    return flags


def kpts_inside_bbox_natural(kpts, bbox):
    import vtool as vt
    # Use keypoint extent to filter out what is in query
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
    return flags


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

    # def batch_elemwise_unary_op(func, arr1, chunksize, dtype=None,
    #                             work_dtype=None, out=None):
    #     if dtype is None:
    #         dtype = arr1.dtype
    #     if work_dtype is None:
    #         work_dtype = arr1.dtype
    #     if out is None:
    #         out = np.empty(arr1.shape, dtype=dtype)
    #     slices = list(ut.ichunk_slices(arr1.shape[0], chunksize))
    #     for sl in ut.ProgIter(slices, lbl='unary op'):
    #         _tmp = arr1[sl].astype(work_dtype)
    #         out[sl] = func(_tmp)
    #     return out

    # beup = batch_elemwise_unary_op

    # chunksize = 1000
    # tmp = beup(root_sift, X, chunksize)
    # mean_vec = tmp.mean(axis=1)
    # tmp = beup(root_sift, X, chunksize)
    # def minus_mean(x):
    #     return np.subtract(x, mean_vec)
    # beup(minus_mean, tmp, chunksize)

    # words_fpath = join(smk_2013_dir, 'clust_preprocessed/oxford_train_vw.int32')
    # with open(join(smk_2013_dir, 'paris_sift.uint8'), 'rb') as file_:
    #     rawbytes = file_.read()
    #     nbytes = len(rawbytes)
    #     print('nbytes = %r' % (nbytes,))
    #     print(sorted(ut.factors(nbytes - 0)))
    #     print(sorted(ut.factors(nbytes - 4)))
    #     print(sorted(ut.factors(nbytes - 8)))
    #     data = np.fromstring(rawbytes, np.uint8)
    #     data = data.reshape(len(data) / 128, 128)
    #     # np.linalg.norm(data[0:100], axis=1)

    # with open(join(smk_2013_dir, 'clust_preprocessed/oxford_vw.int32'), 'rb') as file_:
    #     rawbytes = file_.read()
    #     nbytes = len(rawbytes)
    #     print('nbytes = %r' % (nbytes,))
    #     print(sorted(ut.factors(nbytes - 0)))
    #     print(sorted(ut.factors(nbytes - 4)))
    #     print(sorted(ut.factors(nbytes - 8)))
    #     # header = file_.read(4)
    #     # num = np.fromstring(header, np.int32)

    # with open(words_fpath, 'rb') as file_:
    #     header = file_.read(4)
    #     num = np.fromstring(header, np.int32)
    #     data = np.fromstring(file_.read(), np.int32)

    # from os.path import join
    # sift_fpath = join(smk_2013_dir, 'oxford_sift.uint8')
    # with open(words_fpath, 'rb') as file_:
    #     sift = np.fromstring(file_.read(), np.uint8)

    #     # sift /

    # with open(join(smk_2013_dir, 'oxford_geom_sift.float'), 'rb') as file_:
    #     geom = np.fromstring(file_.read(), np.float32)
    #     geom = geom.reshape(geom.size / 5, 5)

    # data[0:100]
    # simulate load_ext https://github.com/Erotemic/yael/blob/master/matlab/load_ext.m
    # filename = words_fpath
    # nrows = 1
    # def yael_load_ext_int32(filename, nrows=1, verbose=True):
    #     nmin = 1
    #     nmax = np.inf
    #     if verbose:
    #         print('< load int file %s\n', filename)
    #     fid = open(filename, 'rb')
    #     try:
    #         bof = 0  # begining of file
    #         offset = 4 * (nmin - 1) * nrows
    #         fid.seek(offset, bof)
    #         data = np.fromstring(fid.read(), np.int32)
    #     finally:
    #         fid.close()

    # words = np.fromstring(data)
    # vecs = vecs.reshape(shape)

    # if True:
    #     # 28.5 Hz using only one CPU, (can ctrl+c)
    #     import sklearn.cluster
    #     rng = np.random.RandomState(13421421)
    #     centers_init = vt.kmeans_plusplus_sklearn(
    #         vecs_subset, K,
    #         random_state=rng, n_init=n_init,
    #         init_size=init_size)
    # if False:
    #     # 32.9 Hz all CPUs, (can't ctrl+c)
    #     import cv2
    #     criteria_type = (cv2.TERM_CRITERIA_EPS |
    #                      cv2.TERM_CRITERIA_MAX_ITER)

    #     max_iter = 0
    #     epsilon = 100.
    #     criteria = (criteria_type, max_iter, epsilon)
    #     with ut.Timer('cv2km++'):
    #         loss, label, centers_init = cv2.kmeans(
    #             data=vecs_subset, K=K, bestLabels=None,
    #             criteria=criteria, attempts=n_init,
    #             flags=cv2.KMEANS_PP_CENTERS)
    # if False:


def ox_vocab():
    # if config['num_words'] == 1E6:
    #     import vtool as vt
    #     print('Using oxford word assignments')
    #     unique_word, groupxs = vt.group_indices(np.hstack(wordid_list))
    #     assert ut.issorted(unique_word)
    #     wx_to_vecs = vt.apply_grouping(vecs, groupxs, axis=0)

    #     wx_to_word = np.array([
    #         np.round(np.mean(sift_group, axis=0)).astype(np.uint8)
    #         for sift_group in ut.ProgIter(wx_to_vecs, lbl='compute words')
    #     ])
    #     from ibeis.algo.smk import vocab_indexer
    #     vocab = vocab_indexer.VisualVocab(wx_to_word)

    #     wx_lists = [wids[:, None] - 1 for wids in wordid_list]
    # else:
    pass


# def train_vocabulary(vecs, config):
#     #oxford_data1 = load_external_oxford_features()
#     #imgid_order = oxford_data1['imgid_order']
#     #kpts_list = oxford_data1['kpts_list']
#     #vecs_list = oxford_data1['vecs_list']
#     #wordid_list = oxford_data1['wordid_list']

#     #num_words = 8000
#     num_words = config['num_words']
#     from os.path import join
#     dbdir = ut.truepath('/raid/work/Oxford/')
#     fpath = join(dbdir, 'vocab_%d.pkl' % (num_words,))
#     if ut.checkpath(fpath):
#         return ut.load_data(fpath)

#     #train_vecs = np.vstack(vecs_list)[::100].copy()
#     train_vecs = vecs.astype(np.float32)

#     rng = np.random.RandomState(13421421)
#     import sklearn.cluster
#     train_vecs = train_vecs.astype(np.float32)

#     import utool
#     utool.embed()

#     import utool
#     with utool.embed_on_exception_context:

#         clusterer = sklearn.cluster.MiniBatchKMeans(
#             num_words, random_state=rng,
#             init_size=num_words * 3,
#             n_init=3, verbose=5)
#         clusterer.fit(train_vecs)

#         words = clusterer.cluster_centers_
#         words = words.astype(np.uint8)
#         ut.save_data(ut.augpath(fpath, 'words'), words)

#     # from ibeis.algo.smk import vocab_indexer
#     # vocab = vocab_indexer.VisualVocab(words)
#     # vocab.build()

#     #tuned_params = vt.tune_flann(words, target_precision=.95)
#     # ut.save_data(fpath, vocab)
#     return words

#tuned_params = vt.tune_flann(words, target_precision=.95)
# word_hash = ut.hashstr_arr27(vocab.wx_to_word, 'words')
# relevant_params = ['checks']
# cfglbl = ut.get_cfg_lbl(ut.dict_subset(config, relevant_params))
# assign_fpath = join(dbdir, 'assigns_' + ut.hashstr27(word_hash + cfglbl) + '.pkl')

#import pyflann
#flann = pyflann.FLANN()
#centers1 = flann.kmeans(train_vecs, num_words, max_iterations=1)
#pass

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.smk.script_smk
    """
    load_external_data2()
