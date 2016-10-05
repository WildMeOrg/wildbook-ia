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


def load_oxford_2007(config):
    """
    Loads data from
    http://www.robots.ox.ac.uk:5000/~vgg/publications/2007/Philbin07/philbin07.pdf

    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """
    from os.path import join, basename, splitext
    import pandas as pd
    import vtool as vt
    dbdir = ut.truepath('/raid/work/Oxford/')
    data_fpath0 = join(dbdir, 'data_2007.pkl')

    if ut.checkpath(data_fpath0):
        data = ut.load_data(data_fpath0)
        return data
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
        data_uri_order = [x.replace('oxc1_', '') for x in imgid_order]

        imgid_to_df = {}
        for imgid in ut.ProgIter(imgid_order, lbl='reading kpts'):
            word_fpath = imgid_to_word_fpath[imgid]
            row_gen = (map(float, line.strip('\n').split(' '))
                       for line in ut.read_lines_from(word_fpath)[2:])
            rows = [(int(word_id), x, y, e11, e12, e22)
                    for (word_id, x, y, e11, e12, e22) in row_gen]
            df = pd.DataFrame(rows, columns=['word_id', 'x', 'y', 'e11', 'e12', 'e22'])
            imgid_to_df[imgid] = df

        df_list = ut.take(imgid_to_df, imgid_order)

        nfeat_list = [len(df_) for df_ in df_list]
        offset_list = [0] + ut.cumsum(nfeat_list)
        shape = (offset_list[-1], 128)
        #shape = (16334970, 128)
        sift_fpath = join(dbdir, 'OxfordSIFTDescriptors',
                          'feat_oxc1_hesaff_sift.bin')
        try:
            file_ = open(sift_fpath, 'rb')
            with ut.Timer('Reading SIFT binary file'):
                nbytes = np.prod(shape)
                all_vecs = np.fromstring(file_.read(nbytes), dtype=np.uint8)
            all_vecs = all_vecs.reshape(shape)
        finally:
            file_.close()

        kpts_list = [df_.loc[:, ('x', 'y', 'e11', 'e12', 'e22')].values
                     for df_ in df_list]
        wordid_list = [df_.loc[:, 'word_id'].values for df_ in df_list]
        kpts_Z = np.vstack(kpts_list)
        idx_to_wx = np.hstack(wordid_list)

        # assert len(np.unique(idx_to_wx)) == 1E6

        # Reqd standard query order
        query_files = sorted(ut.glob(dbdir + '/oxford_groundtruth', '*_query.txt'))
        query_uri_order = []
        for qpath in query_files:
            text = ut.readfrom(qpath, verbose=0)
            query_uri = text.split(' ')[0].replace('oxc1_', '')
            query_uri_order.append(query_uri)

        print('converting to invV')
        all_kpts = vt.convert_kptsZ_to_kpts(kpts_Z)

        data = {
            'offset_list': offset_list,
            'all_kpts': all_kpts,
            'all_vecs': all_vecs,
            'idx_to_wx': idx_to_wx,
            'data_uri_order': data_uri_order,
            'query_uri_order': query_uri_order,
        }
        ut.save_data(data_fpath0, data)
    return data


def load_oxford_2013():
    """
    Found this data in README of SMK publication
    https://hal.inria.fr/hal-00864684/document
    http://people.rennes.inria.fr/Herve.Jegou/publications.html
    with download link
    wget -nH --cut-dirs=4 -r -Pdata/ ftp://ftp.irisa.fr/local/texmex/corpus/iccv2013/

    This dataset has 5063 images wheras 07 has 5062
    This dataset seems to contain an extra junk image:
        ashmolean_000214

    # Remember that matlab is 1 indexed!
    # DONT FORGET TO CONVERT TO 0 INDEXING!
    """
    from yael.ynumpy import fvecs_read
    from yael.yutils import load_ext
    import scipy.io
    import vtool as vt
    from os.path import join

    dbdir = ut.truepath('/raid/work/Oxford/')
    datadir = dbdir + '/smk_data_iccv_2013/data/'

    # we are not retraining, so this is unused
    # # Training data descriptors for Paris6k dataset
    # train_sift_fname = join(datadir, 'paris_sift.uint8')  # NOQA
    # # File storing visual words of Paris6k descriptors used in our ICCV paper
    # train_vw_fname = join(datadir, 'clust_preprocessed/oxford_train_vw.int32')

    # Pre-learned quantizer used in ICCV paper (used if docluster=false)
    codebook_fname = join(datadir, 'clust_preprocessed/oxford_codebook.fvecs')

    # Files storing descriptors/geometry for Oxford5k dataset
    test_sift_fname = join(datadir, 'oxford_sift.uint8')
    test_geom_fname = join(datadir, 'oxford_geom_sift.float')
    test_nf_fname = join(datadir, 'oxford_nsift.uint32')

    # File storing visual words of Oxford5k descriptors used in our ICCV paper
    test_vw_fname = join(datadir,  'clust_preprocessed/oxford_vw.int32')
    # Ground-truth for Oxford dataset
    gnd_fname =  join(datadir,  'gnd_oxford.mat')

    oxford_vecs   = load_ext(test_sift_fname, ndims=128, verbose=True)
    oxford_nfeats = load_ext(test_nf_fname, verbose=True)
    oxford_words  = fvecs_read(codebook_fname)
    oxford_wids   = load_ext(test_vw_fname, verbose=True) - 1

    test_geom_invV_fname = test_geom_fname + '.invV.pkl'
    try:
        all_kpts = ut.load_data(test_geom_invV_fname)
        print('loaded invV keypoints')
    except IOError:
        oxford_kptsZ  = load_ext(test_geom_fname, ndims=5, verbose=True)
        print('converting to invV keypoints')
        all_kpts = vt.convert_kptsZ_to_kpts(oxford_kptsZ)
        ut.save_data(test_geom_invV_fname, all_kpts)

    gnd_ox = scipy.io.loadmat(gnd_fname)
    imlist = [x[0][0] for x in gnd_ox['imlist']]
    qx_to_dx = gnd_ox['qidx'] - 1

    data_uri_order = imlist
    query_uri_order = ut.take(data_uri_order, qx_to_dx)

    offset_list = np.hstack(([0], oxford_nfeats.cumsum())).astype(np.int64)

    # query_gnd = gnd_ox['gnd'][0][0]
    # bboxes = query_gnd[0]
    # qx_to_ok_gtidxs1 = [x[0] for x in query_gnd[1][0]]
    # qx_to_junk_gtidxs2 = [x[0] for x in query_gnd[2][0]]
    # # ut.depth_profile(qx_to_gtidxs1)
    # # ut.depth_profile(qx_to_gtidxs2)

    assert sum(oxford_nfeats) == len(oxford_vecs)
    assert offset_list[-1] == len(oxford_vecs)
    assert len(oxford_wids) == len(oxford_vecs)
    assert oxford_wids.max() == len(oxford_words) - 1

    data = {
        'offset_list': offset_list,
        'all_kpts': all_kpts,
        'all_vecs': oxford_vecs,
        'words': oxford_words,
        'idx_to_wx': oxford_wids,
        'data_uri_order': data_uri_order,
        'query_uri_order': query_uri_order,
    }
    return data


def show_data_image(data_uri_order, i, offset_list, all_kpts, all_vecs):
    """
    i = 12
    """
    import vtool as vt
    from os.path import join
    imgdir = ut.truepath('/raid/work/Oxford/oxbuild_images')
    gpath = join(imgdir,  data_uri_order[i] + '.jpg')
    image = vt.imread(gpath)
    import plottool as pt
    pt.qt4ensure()
    # pt.imshow(image)
    l = offset_list[i]
    r = offset_list[i + 1]
    kpts = all_kpts[l:r]
    vecs = all_vecs[l:r]
    pt.interact_keypoints.ishow_keypoints(image, kpts, vecs,
                                          ori=False, ell_alpha=.4,
                                          color='distinct')


def check_image_sizes(data_uri_order, all_kpts, offset_list):
    """
    Check if any keypoints go out of bounds wrt their associated images
    """
    import vtool as vt
    from os.path import join
    imgdir = ut.truepath('/raid/work/Oxford/oxbuild_images')
    gpath_list = [join(imgdir, imgid + '.jpg') for imgid in data_uri_order]
    imgsize_list = [vt.open_image_size(gpath) for gpath in gpath_list]
    kpts_list = [all_kpts[l:r] for l, r in ut.itertwo(offset_list)]

    kpts_extent = [vt.get_kpts_image_extent(kpts, outer=False, only_xy=False)
                   for kpts in ut.ProgIter(kpts_list, 'kpts extent')]

    for i, (size, extent) in enumerate(zip(imgsize_list, kpts_extent)):
        w, h = size
        _, maxx, _, maxy = extent
        assert np.isnan(maxx) or maxx < w
        assert np.isnan(maxy) or maxy < h


def load_ordered_annots(data_uri_order, query_uri_order):
    # Open the ibeis version of oxford
    from os.path import basename, splitext
    import ibeis
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

    return ibs, query_annots, data_annots, qx_to_dx


def run_asmk_script():
   """   # NOQA
   >>> from ibeis.algo.smk.script_smk import *  # NOQA
   """  # NOQA
   # FIXME: new_external_annot was not populated in namespace for with embed   # NOQA
   with ut.embed_on_exception_context:  # NOQA

    # ==============================================
    # PREPROCESSING CONFIGURATION
    # ==============================================
    config = {
        'dtype': 'float32',
        'root_sift': True,
        'centering': True,
        'num_words': 2 ** 16,
        #'num_words': 1E6
        #'num_words': 8000,
        'checks': 1024,
        'docluster': False,
        'doassign': False,
        'data_year': 2013,
    }
    # Define which params are relevant for which operations
    relevance = {}
    relevance['all_vecs'] = ['dtype', 'root_sift', 'centering']
    relevance['words'] = relevance['all_vecs'] + ['num_words']
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
    if config['data_year'] == 2007:
        data = load_oxford_2007()
    elif config['data_year'] == 2013:
        data = load_oxford_2013()

    offset_list = data['offset_list']
    all_kpts = data['all_kpts']
    raw_vecs = data['all_vecs']
    query_uri_order = data['query_uri_order']
    data_uri_order = data['data_uri_order']
    # del data

    ibs, query_annots, data_annots, qx_to_dx = load_ordered_annots(
        data_uri_order, query_uri_order)
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
        print('Converting vecs to float32')
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
    if config['docluster']:
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
        # if False:
        #     # Refine visual words
        #     with ut.embed_on_exception_context:
        #         import sklearn.cluster
        #         rng = np.random.RandomState(194932)
        #         clusterer = sklearn.cluster.MiniBatchKMeans(
        #             config['num_words'], init=words,
        #             init_size=init_size,
        #             batch_size=1000, compute_labels=False, random_state=rng,
        #             n_init=1, verbose=5)
        #         clusterer.fit(all_vecs)
        #         words = clusterer.cluster_centers_
        #     word_cacher.save(words)
    else:
        words = data['words']
        assert config['num_words'] is None or len(words) == config['num_words']

    if config['doassign']:
        # =====================================
        # ASSIGN EACH VECTOR TO ITS NEAREST WORD
        # =====================================
        from ibeis.algo.smk import vocab_indexer
        vocab = vocab_indexer.VisualVocab(words)
        # vocab.flann_params['algorithm'] = 'linear'
        vocab.build()
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
    else:
        idx_to_wxs = vt.atleast_nd(data['idx_to_wx'], 2)
        idx_to_maws = np.ones(idx_to_wxs.shape, dtype=np.float32)
        idx_to_wxs = np.ma.array(idx_to_wxs)
        idx_to_maws = np.ma.array(idx_to_maws)

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
        flags = kpts_inside_bbox(kpts_, bbox, only_xy=True)
        query_flags_list.append(flags)

    print('Queries are crops of existing database images.')
    print('Looking at average percents')
    percent_list = [flags_.sum() / flags_.shape[0]
                    for flags_ in query_flags_list]
    percent_stats = ut.get_stats(percent_list)
    print('percent_stats = %s' % (ut.repr4(percent_stats),))

    import vtool as vt
    query_kpts = vt.zipcompress(query_super_kpts, query_flags_list, axis=0)
    query_vecs = vt.zipcompress(query_super_vecs, query_flags_list, axis=0)
    query_wxs = vt.zipcompress(query_super_wxs, query_flags_list, axis=0)
    query_maws = vt.zipcompress(query_super_maws, query_flags_list, axis=0)

    # =======================
    # CONSTRUCT QUERY / DATABASE REPR
    # =======================
    int_rvec = not config['dtype'].startswith('float')

    X_list = []
    _prog = ut.ProgPartial(nTotal=len(qaids), lbl='new X', bs=True,
                           adjust=True)
    for aid, fx_to_wxs, fx_to_maws in _prog(zip(qaids, query_wxs,
                                                query_maws)):
        X = new_external_annot(aid, fx_to_wxs, fx_to_maws, int_rvec)
        X_list.append(X)

    # ydata_cacher = SMKCacher('ydata')
    # Y_list = ydata_cacher.tryload()
    # if Y_list is None:
    Y_list = []
    _prog = ut.ProgPartial(nTotal=len(daids), lbl='new Y', bs=True, adjust=True)
    for aid, fx_to_wxs, fx_to_maws in _prog(zip(daids, wx_lists, maw_lists)):
        Y = new_external_annot(aid, fx_to_wxs, fx_to_maws, int_rvec)
        Y_list.append(Y)
    # ydata_cacher.save(Y_list)

    def jegou_redone_agg_all(flat_wxs_assign, flat_offsets, flat_vecs):
        # flat_wxs_assign = idx_to_wxs
        # flat_offsets = offset_list
        # flat_vecs = all_vecs
        grouped_wxs = [flat_wxs_assign[l:r]
                       for l, r in ut.itertwo(flat_offsets)]

        # Assume single assignment, aggregate everything
        # across the entire database
        flat_offsets = np.array(flat_offsets)

        idx_to_dx = (
            np.searchsorted(
                flat_offsets,
                np.arange(len(flat_wxs_assign)),
                side='right'
            ) - 1
        ).astype(np.int32)
        wx_list = flat_wxs_assign.T[0].compressed()
        unique_wx, groupxs = vt.group_indices(wx_list)

        dim = flat_vecs.shape[1]
        dx_to_wxs = [np.unique(wxs.compressed())
                     for wxs in grouped_wxs]
        dx_to_nagg = [len(wxs) for wxs in dx_to_wxs]
        num_agg_vecs = sum(dx_to_nagg)
        # all_agg_wxs = np.hstack(dx_to_wxs)
        agg_offset_list = np.array([0] + ut.cumsum(dx_to_nagg))
        # Preallocate agg residuals for all dxs
        all_agg_vecs = np.empty((num_agg_vecs, dim),
                                dtype=np.float32)
        all_agg_vecs[:, :] = np.nan

        # precompute agg residual stack
        i_to_dxs = vt.apply_grouping(idx_to_dx, groupxs)
        subgroup = [vt.group_indices(dxs)
                    for dxs in ut.ProgIter(i_to_dxs)]
        i_to_unique_dxs = ut.take_column(subgroup, 0)
        i_to_dx_groupxs = ut.take_column(subgroup, 1)
        num_words = len(unique_wx)

        # Overall this takes 5 minutes and 21 seconds
        # I think the other method takes about 12 minutes
        for i in ut.ProgIter(range(num_words), 'agg'):
            wx = unique_wx[i]
            xs = groupxs[i]
            dxs = i_to_unique_dxs[i]
            dx_groupxs = i_to_dx_groupxs[i]
            word = words[wx:wx + 1]

            offsets1 = agg_offset_list.take(dxs)
            offsets2 = [np.where(dx_to_wxs[dx] == wx)[0][0]
                        for dx in dxs]
            offsets = np.add(offsets1, offsets2, out=offsets1)

            # if __debug__:
            #     assert np.bincount(dxs).max() < 2
            #     offset = agg_offset_list[dxs[0]]
            #     assert np.all(dx_to_wxs[dxs[0]] == all_agg_wxs[offset:offset +
            #                                                    dx_to_nagg[dxs[0]]])

            # Compute residuals
            rvecs = flat_vecs[xs] - word
            vt.normalize(rvecs, axis=1, out=rvecs)
            rvecs[np.all(np.isnan(rvecs), axis=1)] = 0
            # Aggregate across same images
            grouped_rvecs = vt.apply_grouping(rvecs, dx_groupxs, axis=0)
            agg_rvecs_ = [rvec_group.sum(axis=0)
                          for rvec_group in grouped_rvecs]
            # agg_rvecs = np.vstack(agg_rvecs_)
            all_agg_vecs[offsets, :] = agg_rvecs_

        assert not np.any(np.isnan(all_agg_vecs))
        print('Apply normalization')
        vt.normalize(all_agg_vecs, axis=1, out=all_agg_vecs)
        all_error_flags = np.all(np.isnan(all_agg_vecs), axis=1)
        all_agg_vecs[all_error_flags, :] = 0

        # ndocs_per_word1 = np.array(ut.lmap(len, wx_to_unique_dxs))
        # ndocs_total1 = len(flat_offsets) - 1
        # idf1 = smk_funcs.inv_doc_freq(ndocs_total1, ndocs_per_word1)

        agg_rvecs_list = [all_agg_vecs[l:r] for l, r in ut.itertwo(agg_offset_list)]
        agg_flags_list = [all_error_flags[l:r] for l, r in ut.itertwo(agg_offset_list)]
        return agg_rvecs_list, agg_flags_list

        #     # Y.vecs = vecs
        # Y.kpts = kpts

    if False:
        flat_query_vecs = np.vstack(query_vecs)
        flat_query_wxs = np.vstack(query_wxs)
        flat_query_offsets = np.array([0] + ut.cumsum(ut.lmap(len, query_wxs)))

        flat_wxs_assign = flat_query_wxs
        flat_offsets =  flat_query_offsets
        flat_vecs = flat_query_vecs
        agg_rvecs_list, agg_flags_list = jegou_redone_agg_all(
            flat_wxs_assign, flat_offsets, flat_vecs)

        for X, agg_rvecs, agg_flags in zip(X_list, agg_rvecs_list, agg_flags_list):
            X.agg_rvecs = agg_rvecs
            X.agg_flags = agg_flags[:, None]

        flat_wxs_assign = idx_to_wxs
        flat_offsets = offset_list
        flat_vecs = all_vecs
        agg_rvecs_list, agg_flags_list = jegou_redone_agg_all(
            flat_wxs_assign, flat_offsets, flat_vecs)

        for Y, agg_rvecs, agg_flags in zip(Y_list, agg_rvecs_list, agg_flags_list):
            Y.agg_rvecs = agg_rvecs
            Y.agg_flags = agg_flags[:, None]

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

    wx_to_aids = smk_funcs.invert_lists(
        daids, [Y.wx_list for Y in Y_list], all_wxs=wx_list)

    # Compute IDF weights
    print('Compute IDF weights')
    ndocs_total = len(daids)
    # Use only the unique number of words
    ndocs_per_word = np.array([len(set(wx_to_aids[wx])) for wx in wx_list])
    print('ndocs_perword stats: ' + ut.repr4(ut.get_stats(ndocs_per_word)))
    idf_per_word = smk_funcs.inv_doc_freq(ndocs_total, ndocs_per_word)
    wx_to_weight = dict(zip(wx_list, idf_per_word))
    print('idf stats: ' + ut.repr4(ut.get_stats(wx_to_weight.values())))

    # Filter junk
    Y_list_ = [Y for Y in Y_list if Y.qual != 'junk']
    test_kernel(ibs, X_list, Y_list_, vocab, wx_to_weight)


def sanity_checks(offset_list, Y_list, query_annots, ibs):
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


def kpts_inside_bbox(kpts, bbox, only_xy=False):
    # Use keypoint extent to filter out what is in query
    import vtool as vt
    xys = kpts[:, 0:2]
    if only_xy:
        flags = vt.point_inside_bbox(xys.T, bbox)
    else:
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

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.smk.script_smk
    """
    run_asmk_script()
