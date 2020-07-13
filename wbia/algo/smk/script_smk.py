# -*- coding: utf-8 -*-
"""
Results so far without SV / fancyness
Using standard descriptors / vocabulary

proot=bow,nWords=1E6 -> .594
proot=asmk,nWords=1E6 -> .529

Note:
    * Results from SMK Oxford Paper (mAP)
    ASMK nAssign=1, SV=False: .78
    ASMK nAssign=5, SV=False: .82

    Philbin with tf-idf ranking SV=False
    SIFT: .636, RootSIFT: .683 (+.05)

    Philbin with tf-idf ranking SV=True
    SIFT: .672, RootSIFT: .720 (+.05)

    * My Results (WITH BAD QUERY BBOXES)
    smk:nAssign=1,SV=True,: .58
    smk:nAssign=1,SV=False,: .38

    Yesterday I got
    .22 when I fixed the bounding boxes
    And now I'm getting
    .08 and .32 (sv=[F,T]) after deleting and redoing everything (also removing junk images)
    After fix of normalization I get
    .38 and .44

    Using oxford descriptors I get .51ish
    Then changing to root-sift I
    smk-bow = get=0.56294936807700813
    Then using tfidf-bow2=0.56046968275748565
    asmk-gets 0.54146

    Going down to 8K words smk-BOW gets .153
    Going down to 8K words tfidf-BOW gets .128
    Going down to 8K words smk-asmk gets 0.374

    Ok the 65K vocab smk-asmk gets mAP=0.461...
    Ok, after recomputing a new 65K vocab with centered and root-sifted
        descriptors, using float32 precision (in most places), asmk
        gets a new map score of:
        mAP=.5275... :(
        This is with permissive query kpts and oxford vocab.
        Next step: ensure everything is float32.
        Ensured float32
        mAP=.5279, ... better but indiciative of real error

    After that try again at Jegou's data.
    Ensure there are no smk algo bugs. There must be one.

    FINALLY!
    Got Jegou's data working.
    With jegou percmopute oxford feats, words, and assignments
    And float32 version
    asmk = .78415
    bow = .545

    asmk got 0.78415 with float32 version
    bow got .545
    bot2 got .551


    vecs07, root_sift, approx assign, (either jegou or my words)
    mAP=.673

    Weird:
    vecs07, root_sift, exact assign,
    Maybe jegou words or maybe my words. Can't quite tell.
    Might have messed with a config.
    mAP=0.68487357885738664


    October 8
    Still using the same descriptors, but my own vocab with approx assign
    mAP  = 0.78032

    my own vocab approx assign, no center
    map = .793

    The problem was minibatch params. Need higher batch size and init size.
    Needed to modify sklearn to handle this requirement.

    Using my own descriptors I got 0.7460. Seems good.

    Now, back to the HS pipeline.
    Getting a 0.638, so there is an inconsistency.
    Should be getting .7460. Maybe I gotta root_sift it up?


    Turned off root_sift in script
    got .769, so there is a problem in system script
    minibatch  29566/270340... rate=0.86 Hz, eta=0:00:00, total=9:44:35, wall=05:24 EST inertia: mean batch=53730.923812, ewa=53853.439903
    now need to try turning off float32


Differences Between this and SMK:
   * No RootSIFT
   * No SIFT Centering
   * No Independent Vocab
   * Chip RESIZE

Differences between this and VLAD
   * residual vectors are normalized
   * larger default vocabulary size


Feat Info
==========
name     | num_vecs   | n_annots |
=================================
Oxford13 | 12,534,635 |          |
Oxford07 | 16,334,970 |          |
mine1    |  8,997,955 |          |
mine2    | 13,516,721 |   5063   |
mine3    |  8,371,196 |   4728   |
mine4    |  8,482,137 |   4783   |


Cluster Algo Config
===================
name       | algo             | init      | init_size      |  batch size  |
==========================================================================|
minibatch1 | minibatch kmeans | kmeans++  | num_words * 4  | 100          |
minibatch2 | minibatch kmeans | kmeans++  | num_words * 4  | 1000         |
given13    | Lloyd?           | kmeans++? | num_words * 8? | nan?         |


Assign Algo Config
==================
name   | algo   | trees | checks     |
======================================
approx | kdtree |  8    | 1024       |
exact  | linear |  nan  | nan        |
exact  | linear |  nan  | nan        |


SMK Results
===========
  tagid    | mAP   | train_feats | test_feats | center | rootSIFT | assign  | num_words | cluster methods | int | only_xy |
           =================================================================================================================
           | 0.38  |  mine1      |   mine1    |        |          | approx  |  64000    | minibatch1      |     |         |
           | 0.541 |  oxford07   |   oxford07 |        |    X     | approx  |  2 ** 16  | minibatch1      |     |    X    |
           | 0.673 |  oxford13   |   oxford13 |   X    |    X     | approx  |  2 ** 16  | minibatch1      |     |    X    |
           | 0.684 |  oxford13   |   oxford13 |   X    |    X     | exact   |  2 ** 16  | minibatch1      |     |    X    |
           ----------------------------------------------------------------------------------------------------------------
 mybest    | 0.793 |  oxford13   |   oxford13 |        |    X     | approx  |  2 ** 16  | minibatch2      |     |    X    |
           | 0.780 |  oxford13   |   oxford13 |   X    |    X     | approx  |  2 ** 16  | minibatch2      |     |    X    |
           | 0.788 |  paras13    |   oxford13 |   X    |    X     | approx  |  2 ** 16  | given13         |     |    X    |
  allgiven | 0.784 |  paras13    |   oxford13 |   X    |    X     | given13 |  2 ** 16  | given13         |     |    X    |
reported13 | 0.781 |  paras13    |   oxford13 |   X    |    X     | given13 |  2 ** 16  | given13         |     |    X    |
           -----------------------------------------------------------------------------------------------------------------
  inhouse1 | 0.746 |     mine2   |      mine2 |        |    X     | approx  |  2 ** 16  | minibatch2      |     |    X    |
  inhouse2 | 0.769 |     mine2   |      mine2 |        |          | approx  |  2 ** 16  | minibatch2      |     |    X    |
  inhouse3 | 0.769 |     mine2   |      mine2 |        |          | approx  |  2 ** 16  | minibatch2      | X   |    X    |
  inhouse4 | 0.751 |     mine2   |      mine2 |        |          | approx  |  2 ** 16  | minibatch2      | X   |         |
sysharn1   | 0.638 |     mine3   |      mine3 |        |          | approx  |  64000    | minibatch2      | X   |         |
sysharn2   | 0.713 |     mine3   |      mine4 |        |          | approx  |  64000    | minibatch2      | X   |         |


In the SMK paper they report 0.781 as shown in the table, but they also report a score of 0.820 when increasing
the number of features to from 12.5M to 19.2M by lowering feature detection thresholds.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
from wbia.algo.smk import inverted_index
from wbia.algo.smk import smk_funcs
from wbia.algo.smk import smk_pipeline
from six.moves import zip, map

(print, rrr, profile) = ut.inject2(__name__)


class SMK(ut.NiceRepr):
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
        """
        Compute gamma of X

        gamma(X) = (M(X, X)) ** (-1/2)
        """
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
            PhisX, PhisY, flagsX, flagsY, smk.alpha, smk.thresh
        )
        scores = np.multiply(scores, weights, out=scores)
        score = scores.sum()
        return score

    def match_score_sep(smk, X, Y):
        X_idx, Y_idx, weights = smk.word_isect(X, Y)
        phisX_list, flagsY_list = X.phis_flags_list(X_idx)
        phisY_list, flagsX_list = Y.phis_flags_list(Y_idx)
        scores_list = smk_funcs.match_scores_sep(
            phisX_list, phisY_list, flagsX_list, flagsY_list, smk.alpha, smk.thresh
        )
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


# class StackedLists(object):
#     def __init__(self, list_, offsets):
#         self.list_ = list_
#         self.offsets = offsets
#     def split(self):
#         return [self._list_[left: right] for left, right in ut.itertwo(self.offsets)]
#     stacked_vecs = StackedLists(all_vecs, offset_list)
#     vecs_list = stacked_vecs.split()


def load_oxford_2007():
    """
    Loads data from
    http://www.robots.ox.ac.uk:5000/~vgg/publications/2007/Philbin07/philbin07.pdf

    >>> from wbia.algo.smk.script_smk import *  # NOQA
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
        for imgid in ut.ProgIter(imgid_order, label='reading kpts'):
            word_fpath = imgid_to_word_fpath[imgid]
            row_gen = (
                map(float, line.strip('\n').split(' '))
                for line in ut.read_lines_from(word_fpath)[2:]
            )
            rows = [
                (int(word_id), x, y, e11, e12, e22)
                for (word_id, x, y, e11, e12, e22) in row_gen
            ]
            df = pd.DataFrame(rows, columns=['word_id', 'x', 'y', 'e11', 'e12', 'e22'])
            imgid_to_df[imgid] = df

        df_list = ut.take(imgid_to_df, imgid_order)

        nfeat_list = [len(df_) for df_ in df_list]
        offset_list = [0] + ut.cumsum(nfeat_list)
        shape = (offset_list[-1], 128)
        # shape = (16334970, 128)
        sift_fpath = join(dbdir, 'OxfordSIFTDescriptors', 'feat_oxc1_hesaff_sift.bin')
        try:
            file_ = open(sift_fpath, 'rb')
            with ut.Timer('Reading SIFT binary file'):
                nbytes = np.prod(shape)
                all_vecs = np.fromstring(file_.read(nbytes), dtype=np.uint8)
            all_vecs = all_vecs.reshape(shape)
        finally:
            file_.close()

        kpts_list = [
            df_.loc[:, ('x', 'y', 'e11', 'e12', 'e22')].values for df_ in df_list
        ]
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
    with download script

    CommandLine:
        # Download oxford13 data
        cd ~/work/Oxford
        mkdir -p smk_data_iccv_2013
        cd smk_data_iccv_2013
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
    test_vw_fname = join(datadir, 'clust_preprocessed/oxford_vw.int32')
    # Ground-truth for Oxford dataset
    gnd_fname = join(datadir, 'gnd_oxford.mat')

    oxford_vecs = load_ext(test_sift_fname, ndims=128, verbose=True)
    oxford_nfeats = load_ext(test_nf_fname, verbose=True)
    oxford_words = fvecs_read(codebook_fname)
    oxford_wids = load_ext(test_vw_fname, verbose=True) - 1

    test_geom_invV_fname = test_geom_fname + '.invV.pkl'
    try:
        all_kpts = ut.load_data(test_geom_invV_fname)
        print('loaded invV keypoints')
    except IOError:
        oxford_kptsZ = load_ext(test_geom_fname, ndims=5, verbose=True)
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


def load_oxford_wbia():
    import wbia

    ibs = wbia.opendb('Oxford')
    dim_size = None
    _dannots = ibs.annots(
        ibs.filter_annots_general(has_none='query'), config=dict(dim_size=dim_size)
    )
    _qannots = ibs.annots(
        ibs.filter_annots_general(has_any='query'), config=dict(dim_size=dim_size)
    )

    with ut.Timer('reading info'):
        vecs_list = _dannots.vecs
        kpts_list = _dannots.kpts
        nfeats_list = np.array(_dannots.num_feats)

    with ut.Timer('stacking info'):
        all_vecs = np.vstack(vecs_list)
        all_kpts = np.vstack(kpts_list)
        offset_list = np.hstack(([0], nfeats_list.cumsum())).astype(np.int64)
        # data_annots = reorder_annots(_dannots, data_uri_order)

    data_uri_order = get_annots_imgid(_dannots)
    query_uri_order = get_annots_imgid(_qannots)
    data = {
        'offset_list': offset_list,
        'all_kpts': all_kpts,
        'all_vecs': all_vecs,
        'data_uri_order': data_uri_order,
        'query_uri_order': query_uri_order,
    }
    return data


def get_annots_imgid(_annots):
    from os.path import basename, splitext

    _images = _annots._ibs.images(_annots.gids)
    intern_uris = [splitext(basename(uri))[0] for uri in _images.uris_original]
    return intern_uris


def load_ordered_annots(data_uri_order, query_uri_order):
    # Open the wbia version of oxford
    import wbia

    ibs = wbia.opendb('Oxford')

    def reorder_annots(_annots, uri_order):
        intern_uris = get_annots_imgid(_annots)
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
    with ut.embed_on_exception_context:  # NOQA
        """
    >>> from wbia.algo.smk.script_smk import *
    """  # NOQA

        # ==============================================
        # PREPROCESSING CONFIGURATION
        # ==============================================
        config = {
            # 'data_year': 2013,
            'data_year': None,
            'dtype': 'float32',
            # 'root_sift': True,
            'root_sift': False,
            # 'centering': True,
            'centering': False,
            'num_words': 2 ** 16,
            # 'num_words': 1E6
            # 'num_words': 8000,
            'kmeans_impl': 'sklearn.mini',
            'extern_words': False,
            'extern_assign': False,
            'assign_algo': 'kdtree',
            'checks': 1024,
            'int_rvec': True,
            'only_xy': False,
        }
        # Define which params are relevant for which operations
        relevance = {}
        relevance['feats'] = ['dtype', 'root_sift', 'centering', 'data_year']
        relevance['words'] = relevance['feats'] + [
            'num_words',
            'extern_words',
            'kmeans_impl',
        ]
        relevance['assign'] = relevance['words'] + [
            'checks',
            'extern_assign',
            'assign_algo',
        ]
        # relevance['ydata'] = relevance['assign'] + ['int_rvec']
        # relevance['xdata'] = relevance['assign'] + ['only_xy', 'int_rvec']

        nAssign = 1

        class SMKCacher(ut.Cacher):
            def __init__(self, fname, ext='.cPkl'):
                relevant_params = relevance[fname]
                relevant_cfg = ut.dict_subset(config, relevant_params)
                cfgstr = ut.get_cfg_lbl(relevant_cfg)
                dbdir = ut.truepath('/raid/work/Oxford/')
                super(SMKCacher, self).__init__(fname, cfgstr, cache_dir=dbdir, ext=ext)

        # ==============================================
        # LOAD DATASET, EXTRACT AND POSTPROCESS FEATURES
        # ==============================================
        if config['data_year'] == 2007:
            data = load_oxford_2007()
        elif config['data_year'] == 2013:
            data = load_oxford_2013()
        elif config['data_year'] is None:
            data = load_oxford_wbia()

        offset_list = data['offset_list']
        all_kpts = data['all_kpts']
        raw_vecs = data['all_vecs']
        query_uri_order = data['query_uri_order']
        data_uri_order = data['data_uri_order']
        # del data

        # ================
        # PRE-PROCESS
        # ================
        import vtool as vt

        # Alias names to avoid errors in interactive sessions
        proc_vecs = raw_vecs
        del raw_vecs

        feats_cacher = SMKCacher('feats', ext='.npy')
        all_vecs = feats_cacher.tryload()
        if all_vecs is None:
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
                with ut.Timer('Apply centering'):
                    mean_vec = np.mean(proc_vecs, axis=0)
                    # Center and then re-normalize
                    np.subtract(proc_vecs, mean_vec[None, :], out=proc_vecs)
                    vt.normalize(proc_vecs, ord=2, axis=1, out=proc_vecs)

            if config['dtype'] == 'int8':
                smk_funcs

            all_vecs = proc_vecs
            feats_cacher.save(all_vecs)
        del proc_vecs

        # =====================================
        # BUILD VISUAL VOCABULARY
        # =====================================
        if config['extern_words']:
            words = data['words']
            assert config['num_words'] is None or len(words) == config['num_words']
        else:
            word_cacher = SMKCacher('words')
            words = word_cacher.tryload()
            if words is None:
                with ut.embed_on_exception_context:
                    if config['kmeans_impl'] == 'sklearn.mini':
                        import sklearn.cluster

                        rng = np.random.RandomState(13421421)
                        # init_size = int(config['num_words'] * 8)
                        init_size = int(config['num_words'] * 4)
                        # converged after 26043 iterations
                        clusterer = sklearn.cluster.MiniBatchKMeans(
                            config['num_words'],
                            init_size=init_size,
                            batch_size=1000,
                            compute_labels=False,
                            max_iter=20,
                            random_state=rng,
                            n_init=1,
                            verbose=1,
                        )
                        clusterer.fit(all_vecs)
                        words = clusterer.cluster_centers_
                    elif config['kmeans_impl'] == 'yael':
                        from yael import ynumpy

                        centroids, qerr, dis, assign, nassign = ynumpy.kmeans(
                            all_vecs,
                            config['num_words'],
                            init='kmeans++',
                            verbose=True,
                            output='all',
                        )
                        words = centroids
                    word_cacher.save(words)

        # =====================================
        # ASSIGN EACH VECTOR TO ITS NEAREST WORD
        # =====================================
        if config['extern_assign']:
            assert config['extern_words'], 'need extern cluster to extern assign'
            idx_to_wxs = vt.atleast_nd(data['idx_to_wx'], 2)
            idx_to_maws = np.ones(idx_to_wxs.shape, dtype=np.float32)
            idx_to_wxs = np.ma.array(idx_to_wxs)
            idx_to_maws = np.ma.array(idx_to_maws)
        else:
            from wbia.algo.smk import vocab_indexer

            vocab = vocab_indexer.VisualVocab(words)
            dassign_cacher = SMKCacher('assign')
            assign_tup = dassign_cacher.tryload()
            if assign_tup is None:
                vocab.flann_params['algorithm'] = config['assign_algo']
                vocab.build()
                # Takes 12 minutes to assign jegous vecs to 2**16 vocab
                with ut.Timer('assign vocab neighbors'):
                    _idx_to_wx, _idx_to_wdist = vocab.nn_index(
                        all_vecs, nAssign, checks=config['checks']
                    )
                    if nAssign > 1:
                        idx_to_wxs, idx_to_maws = smk_funcs.weight_multi_assigns(
                            _idx_to_wx,
                            _idx_to_wdist,
                            massign_alpha=1.2,
                            massign_sigma=80.0,
                            massign_equal_weights=True,
                        )
                    else:
                        idx_to_wxs = np.ma.masked_array(_idx_to_wx, fill_value=-1)
                        idx_to_maws = np.ma.ones(
                            idx_to_wxs.shape, fill_value=-1, dtype=np.float32
                        )
                        idx_to_maws.mask = idx_to_wxs.mask
                assign_tup = (idx_to_wxs, idx_to_maws)
                dassign_cacher.save(assign_tup)

        idx_to_wxs, idx_to_maws = assign_tup

        # Breakup vectors, keypoints, and word assignments by annotation
        wx_lists = [idx_to_wxs[left:right] for left, right in ut.itertwo(offset_list)]
        maw_lists = [idx_to_maws[left:right] for left, right in ut.itertwo(offset_list)]
        vecs_list = [all_vecs[left:right] for left, right in ut.itertwo(offset_list)]
        kpts_list = [all_kpts[left:right] for left, right in ut.itertwo(offset_list)]

        # =======================
        # FIND QUERY SUBREGIONS
        # =======================

        ibs, query_annots, data_annots, qx_to_dx = load_ordered_annots(
            data_uri_order, query_uri_order
        )
        daids = data_annots.aids
        qaids = query_annots.aids

        query_super_kpts = ut.take(kpts_list, qx_to_dx)
        query_super_vecs = ut.take(vecs_list, qx_to_dx)
        query_super_wxs = ut.take(wx_lists, qx_to_dx)
        query_super_maws = ut.take(maw_lists, qx_to_dx)
        # Mark which keypoints are within the bbox of the query
        query_flags_list = []
        only_xy = config['only_xy']
        for kpts_, bbox in zip(query_super_kpts, query_annots.bboxes):
            flags = kpts_inside_bbox(kpts_, bbox, only_xy=only_xy)
            query_flags_list.append(flags)

        print('Queries are crops of existing database images.')
        print('Looking at average percents')
        percent_list = [flags_.sum() / flags_.shape[0] for flags_ in query_flags_list]
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

        # int_rvec = not config['dtype'].startswith('float')
        int_rvec = config['int_rvec']

        X_list = []
        _prog = ut.ProgPartial(length=len(qaids), label='new X', bs=True, adjust=True)
        for aid, fx_to_wxs, fx_to_maws in _prog(zip(qaids, query_wxs, query_maws)):
            X = new_external_annot(aid, fx_to_wxs, fx_to_maws, int_rvec)
            X_list.append(X)

        # ydata_cacher = SMKCacher('ydata')
        # Y_list = ydata_cacher.tryload()
        # if Y_list is None:
        Y_list = []
        _prog = ut.ProgPartial(length=len(daids), label='new Y', bs=True, adjust=True)
        for aid, fx_to_wxs, fx_to_maws in _prog(zip(daids, wx_lists, maw_lists)):
            Y = new_external_annot(aid, fx_to_wxs, fx_to_maws, int_rvec)
            Y_list.append(Y)
        # ydata_cacher.save(Y_list)

        # ======================
        # Add in some groundtruth

        print('Add in some groundtruth')
        for Y, nid in zip(Y_list, ibs.get_annot_nids(daids)):
            Y.nid = nid

        for X, nid in zip(X_list, ibs.get_annot_nids(qaids)):
            X.nid = nid

        for Y, qual in zip(Y_list, ibs.get_annot_quality_texts(daids)):
            Y.qual = qual

        # ======================
        # Add in other properties
        for Y, vecs, kpts in zip(Y_list, vecs_list, kpts_list):
            Y.vecs = vecs
            Y.kpts = kpts

        imgdir = ut.truepath('/raid/work/Oxford/oxbuild_images')
        for Y, imgid in zip(Y_list, data_uri_order):
            gpath = ut.unixjoin(imgdir, imgid + '.jpg')
            Y.gpath = gpath

        for X, vecs, kpts in zip(X_list, query_vecs, query_kpts):
            X.kpts = kpts
            X.vecs = vecs

        # ======================
        print('Building inverted list')
        daids = [Y.aid for Y in Y_list]
        # wx_list = sorted(ut.list_union(*[Y.wx_list for Y in Y_list]))
        wx_list = sorted(set.union(*[Y.wx_set for Y in Y_list]))
        assert daids == data_annots.aids
        assert len(wx_list) <= config['num_words']

        wx_to_aids = smk_funcs.invert_lists(
            daids, [Y.wx_list for Y in Y_list], all_wxs=wx_list
        )

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

        # =======================
        # CHOOSE QUERY KERNEL
        # =======================
        params = {
            'asmk': dict(alpha=3.0, thresh=0.0),
            'bow': dict(),
            'bow2': dict(),
        }
        # method = 'bow'
        method = 'bow2'
        method = 'asmk'
        smk = SMK(wx_to_weight, method=method, **params[method])

        # Specific info for the type of query
        if method == 'asmk':
            # Make residual vectors
            if True:
                # The stacked way is 50x faster
                # TODO: extend for multi-assignment and record fxs
                flat_query_vecs = np.vstack(query_vecs)
                flat_query_wxs = np.vstack(query_wxs)
                flat_query_offsets = np.array([0] + ut.cumsum(ut.lmap(len, query_wxs)))

                flat_wxs_assign = flat_query_wxs
                flat_offsets = flat_query_offsets
                flat_vecs = flat_query_vecs
                tup = smk_funcs.compute_stacked_agg_rvecs(
                    words, flat_wxs_assign, flat_vecs, flat_offsets
                )
                all_agg_vecs, all_error_flags, agg_offset_list = tup
                if int_rvec:
                    all_agg_vecs = smk_funcs.cast_residual_integer(all_agg_vecs)
                agg_rvecs_list = [
                    all_agg_vecs[left:right]
                    for left, right in ut.itertwo(agg_offset_list)
                ]
                agg_flags_list = [
                    all_error_flags[left:right]
                    for left, right in ut.itertwo(agg_offset_list)
                ]

                for X, agg_rvecs, agg_flags in zip(
                    X_list, agg_rvecs_list, agg_flags_list
                ):
                    X.agg_rvecs = agg_rvecs
                    X.agg_flags = agg_flags[:, None]

                flat_wxs_assign = idx_to_wxs
                flat_offsets = offset_list
                flat_vecs = all_vecs
                tup = smk_funcs.compute_stacked_agg_rvecs(
                    words, flat_wxs_assign, flat_vecs, flat_offsets
                )
                all_agg_vecs, all_error_flags, agg_offset_list = tup
                if int_rvec:
                    all_agg_vecs = smk_funcs.cast_residual_integer(all_agg_vecs)

                agg_rvecs_list = [
                    all_agg_vecs[left:right]
                    for left, right in ut.itertwo(agg_offset_list)
                ]
                agg_flags_list = [
                    all_error_flags[left:right]
                    for left, right in ut.itertwo(agg_offset_list)
                ]

                for Y, agg_rvecs, agg_flags in zip(
                    Y_list, agg_rvecs_list, agg_flags_list
                ):
                    Y.agg_rvecs = agg_rvecs
                    Y.agg_flags = agg_flags[:, None]
            else:
                # This non-stacked way is about 500x slower
                _prog = ut.ProgPartial(label='agg Y rvecs', bs=True, adjust=True)
                for Y in _prog(Y_list_):
                    make_agg_vecs(Y, words, Y.vecs)

                _prog = ut.ProgPartial(label='agg X rvecs', bs=True, adjust=True)
                for X in _prog(X_list):
                    make_agg_vecs(X, words, X.vecs)
        elif method == 'bow2':
            # Hack for orig tf-idf bow vector
            nwords = len(words)
            for X in ut.ProgIter(X_list, label='make bow vector'):
                ensure_tf(X)
                bow_vector(X, wx_to_weight, nwords)

            for Y in ut.ProgIter(Y_list_, label='make bow vector'):
                ensure_tf(Y)
                bow_vector(Y, wx_to_weight, nwords)

        if method != 'bow2':
            for X in ut.ProgIter(X_list, 'compute X gamma'):
                X.gamma = smk.gamma(X)
            for Y in ut.ProgIter(Y_list_, 'compute Y gamma'):
                Y.gamma = smk.gamma(Y)

        # Execute matches (could go faster by enumerating candidates)
        scores_list = []
        for X in ut.ProgIter(X_list, label='query %s' % (smk,)):
            scores = [smk.kernel(X, Y) for Y in Y_list_]
            scores = np.array(scores)
            scores = np.nan_to_num(scores)
            scores_list.append(scores)

        import sklearn.metrics

        avep_list = []
        _iter = list(zip(scores_list, X_list))
        _iter = ut.ProgIter(_iter, label='evaluate %s' % (smk,))
        for scores, X in _iter:
            truth = [X.nid == Y.nid for Y in Y_list_]
            avep = sklearn.metrics.average_precision_score(truth, scores)
            avep_list.append(avep)
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


def make_agg_vecs(X, words, fx_to_vecs):
    word_list = ut.take(words, X.wx_list)
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
    X.gamma = smk_funcs.gamma_agg(X.agg_rvecs, X.agg_flags, weight_list, alpha, thresh)
    return X


def verify_score():
    """
    Recompute all SMK things for two annotations and compare scores.

    >>> from wbia.algo.smk.script_smk import *  # NOQA

    cm.print_inspect_str(qreq_)
    cm.show_single_annotmatch(qreq_, daid1)
    cm.show_single_annotmatch(qreq_, daid2)
    """
    qreq_, cm = load_internal_data()
    qreq_.ensure_data()

    ibs = qreq_.ibs
    qaid = cm.qaid
    daid1 = cm.get_top_truth_aids(ibs, ibs.const.EVIDENCE_DECISION.POSITIVE)[0]
    daid2 = cm.get_top_truth_aids(ibs, ibs.const.EVIDENCE_DECISION.POSITIVE, invert=True)[
        0
    ]

    vocab = ibs.depc['vocab'].get_row_data([qreq_.dinva.vocab_rowid], 'words')[0]
    wx_to_weight = qreq_.dinva.wx_to_weight

    aid = qaid  # NOQA
    config = qreq_.qparams

    alpha = config.get('smk_alpha', 3.0)
    thresh = config.get('smk_thresh', 3.0)
    X = make_temporary_annot(qaid, vocab, wx_to_weight, ibs, config)
    assert np.isclose(
        smk_pipeline.match_kernel_agg(X, X, wx_to_weight, alpha, thresh)[0], 1.0
    )

    Y1 = make_temporary_annot(daid1, vocab, wx_to_weight, ibs, config)
    item = smk_pipeline.match_kernel_agg(X, Y1, wx_to_weight, alpha, thresh)
    score = item[0]
    assert np.isclose(score, cm.get_annot_scores([daid1])[0])
    assert np.isclose(
        smk_pipeline.match_kernel_agg(Y1, Y1, wx_to_weight, alpha, thresh)[0], 1.0
    )

    Y2 = make_temporary_annot(daid2, vocab, wx_to_weight, ibs, config)
    item = smk_pipeline.match_kernel_agg(X, Y2, wx_to_weight, alpha, thresh)
    score = item[0]
    assert np.isclose(score, cm.get_annot_scores([daid2])[0])
    assert np.isclose(
        smk_pipeline.match_kernel_agg(Y2, Y2, wx_to_weight, alpha, thresh)[0], 1.0
    )
    # Y2 = make_temporary_annot(daid2, vocab, wx_to_weight, ibs, config)


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
        flags = np.logical_and.reduce(
            [
                vt.point_inside_bbox(pts1.T, bbox),
                vt.point_inside_bbox(pts2.T, bbox),
                vt.point_inside_bbox(pts3.T, bbox),
                vt.point_inside_bbox(pts4.T, bbox),
            ]
        )
    return flags


def sanity_checks(offset_list, Y_list, query_annots, ibs):
    nfeat_list = np.diff(offset_list)
    for Y, nfeat in ut.ProgIter(zip(Y_list, nfeat_list), 'checking'):
        assert nfeat == sum(ut.lmap(len, Y.fxs_list))

    if False:
        # Visualize queries
        # Look at the standard query images here
        # http://www.robots.ox.ac.uk:5000/~vgg/publications/2007/Philbin07/philbin07.pdf
        from wbia.viz import viz_chip
        import wbia.plottool as pt

        pt.qt4ensure()
        fnum = 1
        pnum_ = pt.make_pnum_nextgen(len(query_annots.aids) // 5, 5)
        for aid in ut.ProgIter(query_annots.aids):
            pnum = pnum_()
            viz_chip.show_chip(
                ibs,
                aid,
                in_image=True,
                annote=False,
                notitle=True,
                draw_lbls=False,
                fnum=fnum,
                pnum=pnum,
            )


def oxford_conic_test():
    # Test that these are what the readme says
    A, B, C = [0.016682, 0.001693, 0.014927]
    A, B, C = [0.010141, -1.1e-05, 0.02863]
    Z = np.array([[A, B], [B, C]])
    import vtool as vt

    invV = vt.decompose_Z_to_invV_2x2(Z)  # NOQA
    invV = vt.decompose_Z_to_invV_mats2x2(np.array([Z]))  # NOQA
    # seems ok
    # invV = np.linalg.inv(V)


def load_internal_data():
    r"""
    wbia TestResult --db Oxford \
        -p smk:nWords=[64000],nAssign=[1],SV=[False],can_match_sameimg=True,dim_size=None \
        -a oxford \
        --dev-mode

    wbia TestResult --db GZ_Master1 \
        -p smk:nWords=[64000],nAssign=[1],SV=[False],fg_on=False \
        -a ctrl:qmingt=2 \
        --dev-mode
    """
    # from wbia.algo.smk.smk_pipeline import *  # NOQA
    import wbia

    qreq_ = wbia.testdata_qreq_(
        defaultdb='Oxford',
        a='oxford',
        p='smk:nWords=[64000],nAssign=[1],SV=[False],can_match_sameimg=True,dim_size=None',
    )
    cm_list = qreq_.execute()
    ave_precisions = [cm.get_annot_ave_precision() for cm in cm_list]
    mAP = np.mean(ave_precisions)
    print('mAP = %.3f' % (mAP,))
    cm = cm_list[-1]
    return qreq_, cm


def compare_data(Y_list_):
    import wbia

    qreq_ = wbia.testdata_qreq_(
        defaultdb='Oxford',
        a='oxford',
        p='smk:nWords=[64000],nAssign=[1],SV=[False],can_match_sameimg=True,dim_size=None',
    )
    qreq_.ensure_data()

    gamma1s = []
    gamma2s = []

    print(len(Y_list_))
    print(len(qreq_.daids))

    dinva = qreq_.dinva
    bady = []
    for Y in Y_list_:
        aid = Y.aid
        gamma1 = Y.gamma
        if aid in dinva.aid_to_idx:
            idx = dinva.aid_to_idx[aid]
            gamma2 = dinva.gamma_list[idx]
            gamma1s.append(gamma1)
            gamma2s.append(gamma2)
        else:
            bady += [Y]
            print(Y.nid)
            # print(Y.qual)

    # ibs = qreq_.ibs
    # z = ibs.annots([a.aid for a in bady])

    import wbia.plottool as pt

    ut.qtensure()
    gamma1s = np.array(gamma1s)
    gamma2s = np.array(gamma2s)
    sortx = gamma1s.argsort()
    pt.plot(gamma1s[sortx], label='script')
    pt.plot(gamma2s[sortx], label='pipe')
    pt.legend()


def show_data_image(data_uri_order, i, offset_list, all_kpts, all_vecs):
    """
    i = 12
    """
    import vtool as vt
    from os.path import join

    imgdir = ut.truepath('/raid/work/Oxford/oxbuild_images')
    gpath = join(imgdir, data_uri_order[i] + '.jpg')
    image = vt.imread(gpath)
    import wbia.plottool as pt

    pt.qt4ensure()
    # pt.imshow(image)
    left = offset_list[i]
    right = offset_list[i + 1]
    kpts = all_kpts[left:right]
    vecs = all_vecs[left:right]
    pt.interact_keypoints.ishow_keypoints(
        image, kpts, vecs, ori=False, ell_alpha=0.4, color='distinct'
    )


def check_image_sizes(data_uri_order, all_kpts, offset_list):
    """
    Check if any keypoints go out of bounds wrt their associated images
    """
    import vtool as vt
    from os.path import join

    imgdir = ut.truepath('/raid/work/Oxford/oxbuild_images')
    gpath_list = [join(imgdir, imgid + '.jpg') for imgid in data_uri_order]
    imgsize_list = [vt.open_image_size(gpath) for gpath in gpath_list]
    kpts_list = [all_kpts[left:right] for left, right in ut.itertwo(offset_list)]

    kpts_extent = [
        vt.get_kpts_image_extent(kpts, outer=False, only_xy=False)
        for kpts in ut.ProgIter(kpts_list, 'kpts extent')
    ]

    for i, (size, extent) in enumerate(zip(imgsize_list, kpts_extent)):
        w, h = size
        _, maxx, _, maxy = extent
        assert np.isnan(maxx) or maxx < w
        assert np.isnan(maxy) or maxy < h


def hyrule_vocab_test():
    from yael.yutils import load_ext
    from os.path import join
    import sklearn.cluster

    dbdir = ut.truepath('/raid/work/Oxford/')
    datadir = dbdir + '/smk_data_iccv_2013/data/'

    # Files storing descriptors/geometry for Oxford5k dataset
    test_sift_fname = join(datadir, 'oxford_sift.uint8')
    # test_nf_fname = join(datadir, 'oxford_nsift.uint32')
    all_vecs = load_ext(test_sift_fname, ndims=128, verbose=True).astype(np.float32)
    print(ut.print_object_size(all_vecs))
    # nfeats_list = load_ext(test_nf_fname, verbose=True)

    with ut.embed_on_exception_context:
        rng = np.random.RandomState(13421421)
        # init_size = int(config['num_words'] * 8)
        num_words = int(2 ** 16)
        init_size = num_words * 4
        # converged after 26043 iterations
        minibatch_params = dict(
            n_clusters=num_words,
            init='k-means++',
            # init='random',
            init_size=init_size,
            n_init=1,
            max_iter=100,
            batch_size=1000,
            tol=0.0,
            max_no_improvement=10,
            reassignment_ratio=0.01,
        )
        clusterer = sklearn.cluster.MiniBatchKMeans(
            compute_labels=False, random_state=rng, verbose=1, **minibatch_params
        )
        clusterer.fit(all_vecs)
        words = clusterer.cluster_centers_
        print(words.shape)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.smk.script_smk
    """
    run_asmk_script()
