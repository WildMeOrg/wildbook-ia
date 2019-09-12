# -*- coding: utf-8 -*-
"""
smk_index
This module contains functions for the SelectiveMatchKernels's inverted index.

TODO::
    * Test suit 1000k images
    * Extend for SMK with labels
    * Test get numbers and refine
    * Extrnal keypoint specific weighting
"""
from __future__ import absolute_import, division, print_function
#import six
import utool  # NOQA
import utool as ut
#import weakref
import numpy as np
import six  # NOQA
import pyflann
#import pandas as pd
from six.moves import zip, map, range  # NOQA
from vtool import clustering2 as clustertool
from ibeis.algo.hots import hstypes
from ibeis.algo.hots.smk import smk_scoring
from ibeis.algo.hots.smk import smk_residuals
(print, print_, printDBG, rrr, profile) = ut.inject2(__name__, '[smk_index]')

USE_CACHE_WORDS = not ut.get_argflag('--nocache-words')
WITH_TOTALTIME = True


#@ut.memprof
@profile
def learn_visual_words(ibs, config2_=None, use_cache=USE_CACHE_WORDS, memtrack=None):
    """
    Computes and caches visual words

    Args:
        ibs (?):
        qreq_ (QueryRequest):  query request object with hyper-parameters
        use_cache (bool):  turns on disk based caching(default = True)
        memtrack (None): (default = None)

    Returns:
        ndarray[uint8_t, ndim=2]: words -  aggregate descriptor cluster centers

    Returns:
        words

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, taids, daids, qaids, qreq_, nWords = smk_debug.testdata_dataframe()
        >>> use_cache = True
        >>> words = learn_visual_words(ibs, qreq_)
        >>> print(words.shape)
        (8000, 128)

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_Master1')
        >>> config2_ = ibs.new_query_params(cfgdict=dict(nWords=128000))
        >>> use_cache = True
        >>> words = learn_visual_words(ibs, config2_)
        >>> print(words.shape)
        (8000, 128)

    Auto:
        from ibeis.algo.hots.smk import smk_index
        import utool as ut
        argdoc = ut.make_default_docstr(smk_index.learn_visual_words)
        print(argdoc)
    """
    #if memtrack is None:
    #    memtrack = ut.MemoryTracker('[learn_visual_words]')
    #config2_ = qreq_.extern_data_config2
    nWords = config2_.nWords
    # TODO: Incorporated taids (vocab training ids) into qreq
    if config2_.vocab_taids == 'all':
        taids = ibs.get_valid_aids(species=ibs.get_primary_database_species())  # exemplar
    else:
        taids = config2_.vocab_taids
    initmethod   = config2_.vocab_init_method
    max_iters    = config2_.vocab_nIters
    flann_params = config2_.vocab_flann_params
    train_vecs_list = ibs.get_annot_vecs(taids, eager=True, config2_=config2_)
    #memtrack.track_obj(train_vecs_list[0], 'train_vecs_list[0]')
    #memtrack.report('loaded trainvecs')
    train_vecs = np.vstack(train_vecs_list)
    #memtrack.track_obj(train_vecs, 'train_vecs')
    #memtrack.report('stacked trainvecs')
    del train_vecs_list
    print('[smk_index] Train Vocab(nWords=%d) using %d annots and %d descriptors' %
          (nWords, len(taids), len(train_vecs)))
    kwds = dict(max_iters=max_iters, use_cache=use_cache,
                initmethod=initmethod, appname='smk',
                flann_params=flann_params)
    words = clustertool.cached_akmeans(train_vecs, nWords, **kwds)
    del train_vecs
    del kwds
    #memtrack.report('returning words')
    #del train_vecs_list
    return words


@profile
def assign_to_words_(wordflann, words, idx2_vec, nAssign, massign_alpha,
                     massign_sigma, massign_equal_weights):
    """
    Assigns descriptor-vectors to nearest word.

    Args:
        wordflann (FLANN): nearest neighbor index over words
        words (ndarray): vocabulary words
        idx2_vec (ndarray): descriptors to assign
        nAssign (int): number of words to assign each descriptor to
        massign_alpha (float): multiple-assignment ratio threshold
        massign_sigma (float): multiple-assignment gaussian variance
        massign_equal_weights (bool): assign equal weight to all multiassigned words

    Returns:
        tuple: inverted index, multi-assigned weights, and forward index
        formated as::

            * wx2_idxs - word index   -> vector indexes
            * wx2_maws - word index   -> multi-assignment weights
            * idf2_wxs - vector index -> assigned word indexes

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_raw_internals0()
        >>> words  = invindex.words
        >>> wordflann = invindex.wordflann
        >>> idx2_vec  = invindex.idx2_dvec
        >>> nAssign = qreq_.qparams.nAssign
        >>> massign_alpha = qreq_.qparams.massign_alpha
        >>> massign_sigma = qreq_.qparams.massign_sigma
        >>> massign_equal_weights = qreq_.qparams.massign_equal_weights
        >>> _dbargs = (wordflann, words, idx2_vec, nAssign, massign_alpha, massign_sigma, massign_equal_weights)
        >>> wx2_idxs, wx2_maws, idx2_wxs = assign_to_words_(*_dbargs)
    """
    if ut.VERBOSE:
        print('[smk_index.assign] +--- Start Assign vecs to words.')
        print('[smk_index.assign] * nAssign=%r' % nAssign)
    if not ut.QUIET:
        print('[smk_index.assign] assign_to_words_. len(idx2_vec) = %r' % len(idx2_vec))
    # Assign each vector to the nearest visual words
    assert nAssign > 0, 'cannot assign to 0 neighbors'
    try:
        _idx2_wx, _idx2_wdist = wordflann.nn_index(idx2_vec, nAssign)
    except pyflann.FLANNException as ex:
        ut.printex(ex, 'probably misread the cached flann_fpath=%r' % (wordflann.flann_fpath,))
        raise
    _idx2_wx.shape    = (idx2_vec.shape[0], nAssign)
    _idx2_wdist.shape = (idx2_vec.shape[0], nAssign)
    if nAssign > 1:
        idx2_wxs, idx2_maws = compute_multiassign_weights_(
            _idx2_wx, _idx2_wdist, massign_alpha, massign_sigma, massign_equal_weights)
    else:
        idx2_wxs = _idx2_wx.tolist()
        idx2_maws = [[1.0]] * len(idx2_wxs)

    # Invert mapping -- Group by word indexes
    jagged_idxs = ([idx] * len(wxs)for idx, wxs in enumerate(idx2_wxs))
    wx_keys, groupxs = clustertool.jagged_group(idx2_wxs)
    idxs_list = clustertool.apply_jagged_grouping(jagged_idxs, groupxs)
    maws_list = clustertool.apply_jagged_grouping(idx2_maws, groupxs)
    wx2_idxs = dict(zip(wx_keys, idxs_list))
    wx2_maws = dict(zip(wx_keys, maws_list))
    if ut.VERBOSE:
        print('[smk_index.assign] L___ End Assign vecs to words.')

    return wx2_idxs, wx2_maws, idx2_wxs


@profile
def compute_multiassign_weights_(_idx2_wx, _idx2_wdist, massign_alpha,
                                 massign_sigma, massign_equal_weights):
    r"""
    Multi Assignment Filtering from Improving Bag of Features

    Args:
        _idx2_wx ():
        _idx2_wdist ():
        massign_alpha ():
        massign_sigma ():
        massign_equal_weights (): Turns off soft weighting. Gives all assigned
            vectors weight 1

    Returns:
        tuple : (idx2_wxs, idx2_maws)

    References:
        (Improving Bag of Features)
        http://lear.inrialpes.fr/pubs/2010/JDS10a/jegou_improvingbof_preprint.pdf

        (Lost in Quantization)
        http://www.robots.ox.ac.uk/~vgg/publications/papers/philbin08.ps.gz

        (A Context Dissimilarity Measure for Accurate and Efficient Image Search)
        https://lear.inrialpes.fr/pubs/2007/JHS07/jegou_cdm.pdf

    Notes:
        sigma values from \cite{philbin_lost08}
        (70 ** 2) ~= 5000,
        (80 ** 2) ~= 6250,
        (86 ** 2) ~= 7500,

    Auto:
        from ibeis.algo.hots.smk import smk_index
        import utool as ut; print(ut.make_default_docstr(smk_index.compute_multiassign_weights_))
    """
    if not ut.QUIET:
        print('[smk_index.assign] compute_multiassign_weights_')
    # Valid word assignments are beyond fraction of distance to the nearest word
    massign_thresh = _idx2_wdist.T[0:1].T.copy()
    # HACK: If the nearest word has distance 0 then this threshold is too hard
    # so we should use the distance to the second nearest word.
    flag_too_close = (massign_thresh == 0)
    massign_thresh[flag_too_close] = _idx2_wdist.T[1:2].T[flag_too_close]
    # Compute the threshold fraction
    np.add(.001, massign_thresh, out=massign_thresh)
    np.multiply(massign_alpha, massign_thresh, out=massign_thresh)
    invalid = np.greater_equal(_idx2_wdist, massign_thresh)
    if ut.VERBOSE:
        _ = (invalid.size - invalid.sum(), invalid.size)
        print('[smk_index.assign] + massign_alpha = %r' % (massign_alpha,))
        print('[smk_index.assign] + massign_sigma = %r' % (massign_sigma,))
        print('[smk_index.assign] + massign_equal_weights = %r' % (massign_equal_weights,))
        print('[smk_index.assign] * Marked %d/%d assignments as invalid' % _)

    if massign_equal_weights:
        # Performance hack from jegou paper: just give everyone equal weight
        masked_wxs = np.ma.masked_array(_idx2_wx, mask=invalid)
        idx2_wxs  = list(map(ut.filter_Nones, masked_wxs.tolist()))
        #ut.embed()
        if ut.DEBUG2:
            assert all([isinstance(wxs, list) for wxs in idx2_wxs])
        idx2_maws = [np.ones(len(wxs), dtype=np.float32) for wxs in idx2_wxs]
    else:
        # More natural weighting scheme
        # Weighting as in Lost in Quantization
        gauss_numer = -_idx2_wdist.astype(np.float64)
        gauss_denom = 2 * (massign_sigma ** 2)
        gauss_exp   = np.divide(gauss_numer, gauss_denom)
        unnorm_maw = np.exp(gauss_exp)
        # Mask invalid multiassignment weights
        masked_unorm_maw = np.ma.masked_array(unnorm_maw, mask=invalid)
        # Normalize multiassignment weights from 0 to 1
        masked_norm = masked_unorm_maw.sum(axis=1)[:, np.newaxis]
        masked_maw = np.divide(masked_unorm_maw, masked_norm)
        masked_wxs = np.ma.masked_array(_idx2_wx, mask=invalid)
        # Remove masked weights and word indexes
        idx2_wxs  = list(map(ut.filter_Nones, masked_wxs.tolist()))
        idx2_maws = list(map(ut.filter_Nones, masked_maw.tolist()))
        #with ut.EmbedOnException():
        if ut.DEBUG2:
            checksum = [sum(maws) for maws in idx2_maws]
            for x in np.where([not ut.almost_eq(val, 1) for val in checksum])[0]:
                print(checksum[x])
                print(_idx2_wx[x])
                print(masked_wxs[x])
                print(masked_maw[x])
                print(massign_thresh[x])
                print(_idx2_wdist[x])
            #all([ut.almost_eq(x, 1) for x in checksum])
            assert all([ut.almost_eq(val, 1) for val in checksum]), 'weights did not break evenly'

    return idx2_wxs, idx2_maws


#@ut.cached_func('smk_idf', appname='smk', key_argx=[1, 2, 3], key_kwds=['daid2_label'])
@profile
def compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids, daid2_label=None,
                      vocab_weighting='idf', verbose=False):
    """
    Computes the inverse-document-frequency weighting for each word

    Args:
        wx_series ():
        wx2_idxs ():
        idx2_aid ():
        daids ():
        daid2_label ():
        vocab_weighting ():

    Returns:
        wx2_idf

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
        >>> wx_series = np.arange(len(invindex.words))
        >>> idx2_aid = invindex.idx2_daid
        >>> daid2_label = invindex.daid2_label
        >>> wx2_idf = compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
        >>> result = str(len(wx2_idf))
        >>> print(result)
        8000

    Ignore:
        #>>> wx2_idxs = invindex.wx2_idxs


    Auto:
        from ibeis.algo.hots.smk import smk_index
        import utool as ut; print(ut.make_default_docstr(smk_index.compute_word_idf_))

    """
    if not ut.QUIET:
        print('[smk_index.idf] +--- Start Compute IDF')
    if ut.VERBOSE or verbose:
        print('[smk_index.idf] Word IDFs: ')

    idxs_list, aids_list = helper_idf_wordgroup(wx2_idxs, idx2_aid, wx_series)

    # TODO: Integrate different idf measures
    if vocab_weighting == 'idf':
        idf_list = compute_idf_orig(aids_list, daids)
    elif vocab_weighting == 'negentropy':
        assert daid2_label is not None
        idf_list = compute_idf_label1(aids_list, daid2_label)
    else:
        raise AssertionError('unknown option vocab_weighting=%r' % vocab_weighting)
    if ut.VERBOSE or verbose:
        print('[smk_index.idf] L___ End Compute IDF')
    wx2_idf = dict(zip(wx_series, idf_list))
    return wx2_idf


@profile
def helper_idf_wordgroup(wx2_idxs, idx2_aid, wx_series):
    """ helper function """
    # idxs for each word
    idxs_list = [wx2_idxs[wx].astype(hstypes.INDEX_TYPE)
                 if wx in wx2_idxs
                 else np.empty(0, dtype=hstypes.INDEX_TYPE)
                 for wx in wx_series]
    # aids for each word
    aids_list = [idx2_aid.take(idxs)
                 if len(idxs) > 0
                 else np.empty(0, dtype=hstypes.INDEX_TYPE)
                 for idxs in idxs_list]
    return idxs_list, aids_list


@profile
def compute_idf_orig(aids_list, daids):
    """
    The standard tried and true idf measure
    """
    nTotalDocs = len(daids)
    # idf denominator
    nDocsWithWord_list = np.array([len(set(aids)) for aids in aids_list])
    # Typically for IDF, 1 is added to the denominator to prevent divide by 0
    # compute idf half of sccw-idf weighting
    idf_list = np.log(np.divide(nTotalDocs, np.add(nDocsWithWord_list, 1),
                                dtype=hstypes.FLOAT_TYPE), dtype=hstypes.FLOAT_TYPE)
    return idf_list


@profile
def compute_negentropy_names(aids_list, daid2_label):
    r"""
    One of our idf extensions
    Word weighting based on the negative entropy over all names of p(n_i | word)

    Args:
        aids_list (list of aids):
        daid2_label (dict from daid to label):

    Returns:
        negentropy_list (ndarray[float32]): idf-like weighting for each word based on the negative entropy

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
        >>> wx_series = np.arange(len(invindex.words))
        >>> idx2_aid = invindex.idx2_daid
        >>> daid2_label = invindex.daid2_label
        >>> _ = helper_idf_wordgroup(wx2_idxs, idx2_aid, wx_series)
        >>> idxs_list, aids_list = _

    Math::
        p(n_i | \word) = \sum_{\lbl \in L_i} p(\lbl | \word)

        p(\lbl | \word) = \frac{p(\word | \lbl) p(\lbl)}{p(\word)}

        p(\word) = \sum_{\lbl' \in L} p(\word | \lbl') p(\lbl')

        p(\word | \lbl) = NumAnnotOfLabelWithWord / NumAnnotWithLabel =
        \frac{\sum_{\X \in \DB_\lbl} b(\word, \X)}{\card{\DB_\lbl}}

        h(n_i | word) = -\sum_{i=1}^N p(n_i | \word) \log p(n_i | \word)

        word_weight = log(N) - h(n | word)

    CommandLine:
        python dev.py -t smk2 --allgt --db GZ_ALL
        python dev.py -t smk5 --allgt --db GZ_ALL

    Auto:
        python -c "import utool as ut; ut.print_auto_docstr('ibeis.algo.hots.smk.smk_index', 'compute_negentropy_names')"
    """
    nWords = len(aids_list)
    # --- LABEL MEMBERS w.r.t daids ---
    # compute mapping from label to daids
    # Translate tuples into scalars for efficiency
    label_list = list(daid2_label.values())
    lblindex_list = np.array(ut.tuples_to_unique_scalars(label_list))
    #daid2_lblindex = dict(zip(daid_list, lblindex_list))
    unique_lblindexes, groupxs = clustertool.group_indices(lblindex_list)
    daid_list = np.array(daid2_label.keys())
    daids_list = [daid_list.take(xs) for xs in groupxs]

    # --- DAID MEMBERS w.r.t. words ---
    # compute mapping from daid to word indexes
    # finds all the words that belong to an annotation
    daid2_wxs = ut.ddict(list)
    for wx, _daids in enumerate(aids_list):
        for daid in _daids:
            daid2_wxs[daid].append(wx)

    # --- \Pr(\word \given \lbl) for each label ---
    # Compute the number of annotations in a label with the word vs
    # the number of annotations in the label
    lblindex2_daids = list(zip(unique_lblindexes, daids_list))
    # Get num times word appears for each label
    probWordGivenLabel_list = []
    for lblindex, _daids in lblindex2_daids:
        nAnnotOfLabelWithWord = np.zeros(nWords, dtype=np.int32)
        for daid in _daids:
            wxs = np.unique(daid2_wxs[daid])
            nAnnotOfLabelWithWord[wxs] += 1
        probWordGivenLabel = nAnnotOfLabelWithWord.astype(np.float64) / len(_daids)
        probWordGivenLabel_list.append(probWordGivenLabel)
    # (nLabels, nWords)
    probWordGivenLabel_arr = np.array(probWordGivenLabel_list)
    # --- \Pr(\lbl \given \word) ---
    # compute partition function that approximates probability of a word
    # (1, nWords)
    probWord = probWordGivenLabel_arr.sum(axis=0)
    probWord.shape = (1, probWord.size)
    # (nLabels, nWords)
    probLabelGivenWord_arr = (probWordGivenLabel_arr / probWord)
    # --- \Pr(\name \given \lbl) ---
    # get names for each unique label
    nid_list = np.array([label_list[xs[0]][0] for xs in groupxs])
    unique_nids, groupxs_ = clustertool.group_indices(nid_list)
    # (nNames, nWords)
    # add a little wiggle room
    eps = 1E-9
    # http://stackoverflow.com/questions/872544/precision-of-floating-point
    #epsilon = 2^(E-52)    # For a 64-bit float (double precision)
    #epsilon = 2^(E-23)    # For a 32-bit float (single precision)
    #epsilon = 2^(E-10)    # For a 16-bit float (half precision)
    probNameGivenWord = eps + (1.0 - eps) * np.array([probLabelGivenWord_arr.take(xs, axis=0).sum(axis=0) for xs in groupxs_])
    logProbNameGivenWord = np.log(probNameGivenWord)
    wordNameEntropy = -(probNameGivenWord * logProbNameGivenWord).sum(0)
    # Compute negative entropy for weights
    nNames = len(nid_list)
    negentropy_list = np.log(nNames) - wordNameEntropy
    return negentropy_list


@profile
def compute_idf_label1(aids_list, daid2_label):
    """
    One of our idf extensions

    Example:
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
        >>> wx_series = np.arange(len(invindex.words))
        >>> idx2_aid = invindex.idx2_daid
        >>> daid2_label = invindex.daid2_label
        >>> _ = helper_idf_wordgroup(wx2_idxs, idx2_aid, wx_series)
        >>> idxs_list, aids_list = _
        >>> wx2_idf = compute_idf_label1(wx_series, wx2_idxs, idx2_aid, daids)
    """
    nWords = len(aids_list)
    # Computes our novel label idf weight
    lblindex_list = np.array(ut.tuples_to_unique_scalars(daid2_label.values()))
    #daid2_lblindex = dict(zip(daid_list, lblindex_list))
    unique_lblindexes, groupxs = clustertool.group_indices(lblindex_list)
    daid_list = np.array(daid2_label.keys())
    daids_list = [daid_list.take(xs) for xs in groupxs]
    daid2_wxs = ut.ddict(list)
    for wx, daids in enumerate(aids_list):
        for daid in daids:
            daid2_wxs[daid].append(wx)
    lblindex2_daids = list(zip(unique_lblindexes, daids_list))
    nLabels = len(unique_lblindexes)
    pcntLblsWithWord = np.zeros(nWords, np.float64)
    # Get num times word appears for eachlabel
    for lblindex, daids in lblindex2_daids:
        nWordsWithLabel = np.zeros(nWords)
        for daid in daids:
            wxs = daid2_wxs[daid]
            nWordsWithLabel[wxs] += 1
        pcntLblsWithWord += (1 - nWordsWithLabel.astype(np.float64) / len(daids))

    # Labels for each word
    idf_list = np.log(np.divide(nLabels, np.add(pcntLblsWithWord, 1),
                                dtype=hstypes.FLOAT_TYPE),
                      dtype=hstypes.FLOAT_TYPE)
    return idf_list


#@ut.cached_func('smk_rvecs_', appname='smk')
@profile
def compute_residuals_(words, wx2_idxs, wx2_maws, idx2_vec, idx2_aid,
                       idx2_fx, aggregate, verbose=False):
    """
    Computes residual vectors based on word assignments
    returns mapping from word index to a set of residual vectors

    Args:
        words (ndarray):
        wx2_idxs (dict):
        wx2_maws (dict):
        idx2_vec (dict):
        idx2_aid (dict):
        idx2_fx (dict):
        aggregate (bool):
        verbose (bool):

    Returns:
        tuple : (wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws) formatted as::
            * wx2_rvecs - [ ... [ rvec_i1, ...,  rvec_Mi ]_i ... ]
            * wx2_aids  - [ ... [  aid_i1, ...,   aid_Mi ]_i ... ]
            * wx2_fxs   - [ ... [[fxs]_i1, ..., [fxs]_Mi ]_i ... ]

        For every word::

            * list of aggvecs
            * For every aggvec:
                * one parent aid, if aggregate is False: assert isunique(aids)
                * list of parent fxs, if aggregate is True: assert len(fxs) == 1

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
        >>> words     = invindex.words
        >>> idx2_aid  = invindex.idx2_daid
        >>> idx2_fx   = invindex.idx2_dfx
        >>> idx2_vec  = invindex.idx2_dvec
        >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
        >>> wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws, wx2_flags = compute_residuals_(words, wx2_idxs, wx2_maws, idx2_vec, idx2_aid, idx2_fx, aggregate)
    """
    if not ut.QUIET:
        print('[smk_index.rvec] +--- Start Compute Residuals')

    wx_sublist = np.array(wx2_idxs.keys())
    # Build lists w.r.t. words

    idxs_list = [wx2_idxs[wx].astype(hstypes.INDEX_TYPE) for wx in wx_sublist]
    aids_list = [idx2_aid.take(idxs) for idxs in idxs_list]
    if ut.DEBUG2:
        #assert np.all(np.diff(wx_sublist) == 1), 'not dense'
        assert all([len(a) == len(b) for a, b in zip(idxs_list, aids_list)]), 'bad alignment'
        assert idx2_vec.shape[0] == idx2_fx.shape[0]
        assert idx2_vec.shape[0] == idx2_aid.shape[0]
    # Prealloc output
    if ut.VERBOSE or verbose:
        lbl = '[smk_index.rvec] agg rvecs' if aggregate else '[smk_index.rvec] nonagg rvecs'
        print(lbl)
    if ut.DEBUG2:
        from ibeis.algo.hots.smk import smk_debug
        smk_debug.check_wx2_idxs(wx2_idxs, len(words))
    # Compute Residuals
    rvecs_list, flags_list = smk_residuals.compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list)

    if ut.VERBOSE:
        print('Computed size(rvecs_list) = %r' % ut.get_object_size_str(rvecs_list))
        print('Computed size(flags_list) = %r' % ut.get_object_size_str(flags_list))
    if aggregate:
        maws_list = [wx2_maws[wx] for wx in wx_sublist]
        # Aggregate Residuals
        tup = smk_residuals.compute_agg_rvecs(rvecs_list, idxs_list, aids_list, maws_list)
        (aggvecs_list, aggaids_list, aggidxs_list, aggmaws_list, aggflags_list) = tup
        # Pack into common query structure
        aggfxs_list = [[idx2_fx.take(idxs) for idxs in aggidxs] for aggidxs in aggidxs_list]
        wx2_aggvecs  = dict(zip(wx_sublist, aggvecs_list))
        wx2_aggaids  = dict(zip(wx_sublist, aggaids_list))
        wx2_aggfxs   = dict(zip(wx_sublist, aggfxs_list))
        wx2_aggmaws  = dict(zip(wx_sublist, aggmaws_list))
        wx2_aggflags = dict(zip(wx_sublist, aggflags_list))
        (wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws, wx2_flags) = (
            wx2_aggvecs, wx2_aggaids, wx2_aggfxs, wx2_aggmaws, wx2_aggflags)
    else:
        # Hack non-aggregate residuals to have the same structure as aggregate
        # residuals for compatability: i.e. each rvec gets a list of fxs that
        # contributed to it, and for SMK this is a list of size 1
        fxs_list  = [[idx2_fx[idx:idx + 1] for idx in idxs]  for idxs in idxs_list]
        wx2_rvecs = dict(zip(wx_sublist, rvecs_list))
        wx2_aids  = dict(zip(wx_sublist, aids_list))
        wx2_fxs   = dict(zip(wx_sublist, fxs_list))
        wx2_flags = dict(zip(wx_sublist, flags_list))
    if ut.DEBUG2:
        from ibeis.algo.hots.smk import smk_debug
        smk_debug.check_wx2(words, wx2_rvecs, wx2_aids, wx2_fxs)
    if ut.VERBOSE or verbose:
        print('[smk_index.rvec] L___ End Compute Residuals')
    return wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws, wx2_flags


#@ut.cached_func('sccw', appname='smk', key_argx=[1, 2])
@profile
def compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_dflags, wx2_aids, wx2_idf,
                       wx2_dmaws, smk_alpha, smk_thresh, verbose=False):
    r"""
    Computes sccw normalization scalar for the database annotations.
    This is gamma from the SMK paper.
    sccw is a self consistency critiron weight --- a scalar which ensures
    the score of K(X, X) = 1

    Args:
        idx2_daid ():
        wx2_drvecs ():
        wx2_aids ():
        wx2_idf ():
        wx2_dmaws ():
        smk_alpha ():
        smk_thresh ():

    Returns:
        daid2_sccw

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_index
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> #tup = smk_debug.testdata_compute_data_sccw(db='testdb1')
        >>> tup = smk_debug.testdata_compute_data_sccw(db='PZ_MTEST')
        >>> ibs, annots_df, invindex, wx2_idxs, wx2_idf, wx2_drvecs, wx2_aids, qparams = tup
        >>> wx2_dflags = invindex.wx2_dflags
        >>> ws2_idxs = invindex.wx2_idxs
        >>> wx2_dmaws  = invindex.wx2_dmaws
        >>> idx2_daid  = invindex.idx2_daid
        >>> daids      = invindex.daids
        >>> smk_alpha  = qparams.smk_alpha
        >>> smk_thresh = qparams.smk_thresh
        >>> wx2_idf    = wx2_idf
        >>> verbose = True
        >>> invindex.invindex_dbgstr()
        >>> invindex.report_memory()
        >>> invindex.report_memsize()
        >>> daid2_sccw = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_dflags, wx2_aids, wx2_idf, wx2_dmaws, smk_alpha, smk_thresh, verbose)
    """
    #for wx in wx_sublist:
    #    print(len(wx2_dmaws

    verbose_ = ut.VERBOSE or verbose

    if ut.DEBUG2:
        from ibeis.algo.hots.smk import smk_debug
        smk_debug.check_wx2(wx2_rvecs=wx2_drvecs, wx2_aids=wx2_aids)
    if not ut.QUIET:
        print('\n[smk_index.sccw] +--- Start Compute Data Self Consistency Weight')
    if verbose_:
        print('[smk_index.sccw] Compute SCCW smk_alpha=%r, smk_thresh=%r: ' % (smk_alpha, smk_thresh))

    # Group by daids first and then by word index
    # Get list of aids and rvecs w.r.t. words (ie one item per word)
    wx_sublist = np.array(list(wx2_drvecs.keys()))
    aids_perword  = [wx2_aids[wx] for wx in wx_sublist]

    # wx_list1: Lays out word indexes for each annotation
    # tx_list1: Temporary within annotation subindex + wx uniquely identifies
    # item in wx2_drvecs, wx2_dflags, and wx2_dmaws

    # Flatten out indexes to perform grouping
    flat_aids = np.hstack(aids_perword)
    count = len(flat_aids)
    txs_perword = [np.arange(aids.size) for aids in aids_perword]
    flat_txs  = np.hstack(txs_perword)
    # fromiter is faster for flat_wxs because is not a list of numpy arrays
    wxs_perword = ([wx] * len(aids) for wx, aids in zip(wx_sublist, aids_perword))
    flat_wxs  = np.fromiter(ut.iflatten(wxs_perword), hstypes.INDEX_TYPE, count)

    # Group flat indexes by annotation id
    unique_aids, annot_groupxs = clustertool.group_indices(flat_aids)

    # Wxs and Txs grouped by annotation id
    wxs_perannot = clustertool.apply_grouping_iter(flat_wxs, annot_groupxs)
    txs_perannot = clustertool.apply_grouping_iter(flat_txs, annot_groupxs)

    # Group by word inside each annotation group
    wxsubgrouping_perannot = [clustertool.group_indices(wxs)
                              for wxs in wxs_perannot]
    word_groupxs_perannot = (groupxs for wxs, groupxs in wxsubgrouping_perannot)
    txs_perword_perannot = [clustertool.apply_grouping(txs, groupxs)
                            for txs, groupxs in
                            zip(txs_perannot, word_groupxs_perannot)]
    wxs_perword_perannot = [wxs for wxs, groupxs in wxsubgrouping_perannot]

    # Group relavent data for sccw measure by word for each annotation grouping

    def _vector_subgroup_by_wx(wx2_arr, wxs_perword_perannot, txs_perword_perannot):
        return [[wx2_arr[wx].take(txs, axis=0)
                 for wx, txs in zip(wx_perword_, txs_perword_)]
                for wx_perword_, txs_perword_ in
                zip(wxs_perword_perannot, txs_perword_perannot)]

    def _scalar_subgroup_by_wx(wx2_scalar, wxs_perword_perannot):
        return [[wx2_scalar[wx] for wx in wxs] for wxs in wxs_perword_perannot]

    subgrouped_drvecs = _vector_subgroup_by_wx(wx2_drvecs, wxs_perword_perannot, txs_perword_perannot)
    subgrouped_dmaws  = _vector_subgroup_by_wx(wx2_dmaws,  wxs_perword_perannot, txs_perword_perannot)
    # If we aren't using dmaws replace it with an infinite None iterator
    #subgrouped_dmaws  = iter(lambda: None, 1)
    subgrouped_dflags = _vector_subgroup_by_wx(wx2_dflags, wxs_perword_perannot, txs_perword_perannot)
    #subgrouped_dflags  = iter(lambda: None, 1)
    subgrouped_idfs   = _scalar_subgroup_by_wx(wx2_idf, wxs_perword_perannot)

    if verbose_:
        progiter = ut.ProgressIter(lbl='[smk_index.sccw] SCCW Sum (over daid): ',
                                   total=len(unique_aids), freq=10, with_time=WITH_TOTALTIME)
    else:
        progiter = ut.identity

    if ut.DEBUG2:
        from ibeis.algo.hots.smk import smk_debug
        smk_debug.check_data_smksumm(subgrouped_idfs, subgrouped_drvecs)

    sccw_list = [
        smk_scoring.sccw_summation(rvecs_list, flags_list, idf_list, maws_list, smk_alpha, smk_thresh)
        for rvecs_list, flags_list, maws_list, idf_list in
        progiter(zip(subgrouped_drvecs, subgrouped_dflags, subgrouped_dmaws, subgrouped_idfs))
    ]
    daid2_sccw = dict(zip(unique_aids, sccw_list))

    if verbose_:
        print('[smk_index.sccw] L___ End Compute Data SCCW\n')

    return daid2_sccw
