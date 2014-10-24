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
import utool
#import weakref
import numpy as np
#import pandas as pd
from six.moves import zip, map  # NOQA
from vtool import clustering2 as clustertool
from ibeis.model.hots import hstypes
from ibeis.model.hots.smk import smk_scoring
from ibeis.model.hots.smk import smk_residuals
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_index]')

USE_CACHE_WORDS = not utool.get_argflag('--nocache-words')
WITH_TOTALTIME = True


#@utool.memprof
@profile
def learn_visual_words(annots_df, taids, nWords, use_cache=USE_CACHE_WORDS, memtrack=None):
    """
    Computes and caches visual words

    Args:
        annots_df ():
        taids ():
        nWords ():
        use_cache ():
        memtrack ():

    Returns:
        words

    Example:
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, taids, daids, qaids, qreq_, nWords = smk_debug.testdata_dataframe()
        >>> use_cache = True
        >>> words = learn_visual_words(annots_df, taids, nWords)
        >>> print(words.shape)
        (8000, 128)

    Auto:
        from ibeis.model.hots.smk import smk_index
        import utool
        argdoc = utool.make_default_docstr(smk_index.learn_visual_words)
        print(argdoc)
    """
    #if memtrack is None:
    #    memtrack = utool.MemoryTracker('[learn_visual_words]')
    #max_iters = 200
    max_iters = 300
    flann_params = {}
    train_vecs_list = annots_df.ibs.get_annot_desc(taids, eager=True)
    #memtrack.track_obj(train_vecs_list[0], 'train_vecs_list[0]')
    #memtrack.report('loaded trainvecs')
    train_vecs = np.vstack(train_vecs_list)
    #memtrack.track_obj(train_vecs, 'train_vecs')
    #memtrack.report('stacked trainvecs')
    del train_vecs_list
    print('[smk_index] Train Vocab(nWords=%d) using %d annots and %d descriptors' %
          (nWords, len(taids), len(train_vecs)))
    kwds = dict(max_iters=max_iters, use_cache=use_cache, appname='smk',
                flann_params=flann_params)
    words = clustertool.cached_akmeans(train_vecs, nWords, **kwds)
    #annots_df.ibs.dbcache.squeeze()
    #annots_df.ibs.dbcache.reboot()
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
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_raw_internals0()
        >>> words  = invindex.words
        >>> wordflann = invindex.wordflann
        >>> idx2_vec  = invindex.idx2_dvec
        >>> nAssign = qreq_.qparams.nAssign
        >>> massign_alpha = qreq_.qparams.massign_alpha
        >>> massign_sigma = qreq_.qparams.massign_sigma
        >>> massign_equal_weights = qreq_.qparams.massign_equal_weights
        >>> _dbargs = wordflann, words, idx2_vec, nAssign, massign_alpha, massign_sigma, massign_equal_weights)
        >>> wx2_idxs, wx2_maws, idx2_wxs = assign_to_words_(*_dbargs)
    """
    if utool.VERBOSE:
        print('[smk_index.assign] +--- Start Assign vecs to words.')
        print('[smk_index.assign] * nAssign=%r' % nAssign)
    if not utool.QUIET:
        print('[smk_index.assign] assign_to_words_. len(idx2_vec) = %r' % len(idx2_vec))
    # Assign each vector to the nearest visual words
    assert nAssign > 0, 'cannot assign to 0 neighbors'
    _idx2_wx, _idx2_wdist = wordflann.nn_index(idx2_vec, nAssign)
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
    if utool.VERBOSE:
        print('[smk_index.assign] L___ End Assign vecs to words.')

    return wx2_idxs, wx2_maws, idx2_wxs


@profile
def compute_multiassign_weights_(_idx2_wx, _idx2_wdist, massign_alpha,
                                 massign_sigma, massign_equal_weights):
    """
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
        from ibeis.model.hots.smk import smk_index
        import utool; print(utool.make_default_docstr(smk_index.compute_multiassign_weights_))
    """
    if not utool.QUIET:
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
    if utool.VERBOSE:
        _ = (invalid.size - invalid.sum(), invalid.size)
        print('[smk_index.assign] + massign_alpha = %r' % (massign_alpha,))
        print('[smk_index.assign] + massign_sigma = %r' % (massign_sigma,))
        print('[smk_index.assign] + massign_equal_weights = %r' % (massign_equal_weights,))
        print('[smk_index.assign] * Marked %d/%d assignments as invalid' % _)

    if massign_equal_weights:
        # Performance hack from jegou paper: just give everyone equal weight
        masked_wxs = np.ma.masked_array(_idx2_wx, mask=invalid)
        idx2_wxs  = list(map(utool.filter_Nones, masked_wxs.tolist()))
        #utool.embed()
        if utool.DEBUG2:
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
        idx2_wxs  = list(map(utool.filter_Nones, masked_wxs.tolist()))
        idx2_maws = list(map(utool.filter_Nones, masked_maw.tolist()))
        #with utool.EmbedOnException():
        if utool.DEBUG2:
            checksum = [sum(maws) for maws in idx2_maws]
            for x in np.where([not utool.almost_eq(val, 1) for val in checksum])[0]:
                print(checksum[x])
                print(_idx2_wx[x])
                print(masked_wxs[x])
                print(masked_maw[x])
                print(massign_thresh[x])
                print(_idx2_wdist[x])
            #all([utool.almost_eq(x, 1) for x in checksum])
            assert all([utool.almost_eq(val, 1) for val in checksum]), 'weights did not break evenly'

    return idx2_wxs, idx2_maws


#@utool.cached_func('smk_idf', appname='smk', key_argx=[1, 2, 3], key_kwds=['daid2_label'])
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
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
        >>> wx_series = np.arange(len(invindex.words))
        >>> idx2_aid = invindex.idx2_daid
        >>> daid2_label = invindex.daid2_label
        >>> wx2_idf = compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
        >>> print(wx2_idf.shape)
        (8000,)

    Ignore:
        #>>> wx2_idxs = invindex.wx2_idxs


    Auto:
        from ibeis.model.hots.smk import smk_index
        import utool; print(utool.make_default_docstr(smk_index.compute_word_idf_))

    """
    if not utool.QUIET:
        print('[smk_index.idf] +--- Start Compute IDF')
    if utool.VERBOSE or verbose:
        mark, end_ = utool.log_progress('[smk_index.idf] Word IDFs: ',
                                        len(wx_series), freq=50,
                                        with_time=WITH_TOTALTIME)

    idxs_list, aids_list = helper_idf_wordgroup(wx2_idxs, idx2_aid, wx_series)

    # TODO: Integrate different idf measures
    if vocab_weighting == 'idf':
        idf_list = compute_idf_orig(aids_list, daids)
    elif vocab_weighting == 'negentropy':
        assert daid2_label is not None
        idf_list = compute_idf_label1(aids_list, daid2_label)
    else:
        raise AssertionError('unknown option vocab_weighting=%r' % vocab_weighting)
    if utool.VERBOSE or verbose:
        end_()
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
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
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
        python -c "import utool; utool.print_auto_docstr('ibeis.model.hots.smk.smk_index', 'compute_negentropy_names')"
    """
    nWords = len(aids_list)
    # --- LABEL MEMBERS w.r.t daids ---
    # compute mapping from label to daids
    # Translate tuples into scalars for efficiency
    label_list = list(daid2_label.values())
    lblindex_list = np.array(utool.tuples_to_unique_scalars(label_list))
    #daid2_lblindex = dict(zip(daid_list, lblindex_list))
    unique_lblindexes, groupxs = clustertool.group_indicies(lblindex_list)
    daid_list = np.array(daid2_label.keys())
    daids_list = [daid_list.take(xs) for xs in groupxs]

    # --- DAID MEMBERS w.r.t. words ---
    # compute mapping from daid to word indexes
    # finds all the words that belong to an annotation
    daid2_wxs = utool.ddict(list)
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
    unique_nids, groupxs_ = clustertool.group_indicies(nid_list)
    # (nNames, nWords)
    # add a little wiggle room
    eps = 1E-9
    # http://stackoverflow.com/questions/872544/precision-of-floating-point
    #epsilon = 2^(E-52)    % For a 64-bit float (double precision)
    #epsilon = 2^(E-23)    % For a 32-bit float (single precision)
    #epsilon = 2^(E-10)    % For a 16-bit float (half precision)
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
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
        >>> wx_series = np.arange(len(invindex.words))
        >>> idx2_aid = invindex.idx2_daid
        >>> daid2_label = invindex.daid2_label
        >>> _ = helper_idf_wordgroup(wx2_idxs, idx2_aid, wx_series)
        >>> idxs_list, aids_list = _
        >>> wx2_idf = compute_word_idf_(wx_series, wx2_idxs, idx2_aid, daids)
    """
    nWords = len(aids_list)
    # Computes our novel label idf weight
    lblindex_list = np.array(utool.tuples_to_unique_scalars(daid2_label.values()))
    #daid2_lblindex = dict(zip(daid_list, lblindex_list))
    unique_lblindexes, groupxs = clustertool.group_indicies(lblindex_list)
    daid_list = np.array(daid2_label.keys())
    daids_list = [daid_list.take(xs) for xs in groupxs]
    daid2_wxs = utool.ddict(list)
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


#@utool.cached_func('smk_rvecs_', appname='smk')
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
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, wx2_idxs, qparams = smk_debug.testdata_raw_internals1()
        >>> words     = invindex.words
        >>> idx2_aid  = invindex.idx2_daid
        >>> idx2_fx   = invindex.idx2_dfx
        >>> idx2_vec  = invindex.idx2_dvec
        >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
        >>> wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws, wx2_flags = compute_residuals_(words, wx2_idxs, wx2_maws, idx2_vec, idx2_aid, idx2_fx, aggregate)
    """
    if not utool.QUIET:
        print('[smk_index.rvec] +--- Start Compute Residuals')

    wx_sublist = np.array(wx2_idxs.keys())
    # Build lists w.r.t. words

    idxs_list = [wx2_idxs[wx].astype(hstypes.INDEX_TYPE) for wx in wx_sublist]
    aids_list = [idx2_aid.take(idxs) for idxs in idxs_list]
    if utool.DEBUG2:
        #assert np.all(np.diff(wx_sublist) == 1), 'not dense'
        assert all([len(a) == len(b) for a, b in zip(idxs_list, aids_list)]), 'bad alignment'
        assert idx2_vec.shape[0] == idx2_fx.shape[0]
        assert idx2_vec.shape[0] == idx2_aid.shape[0]
    # Prealloc output
    if utool.VERBOSE or verbose:
        #print('[smk_index.rvec] Residual Vectors for %d words. aggregate=%r' %
        #      (len(wx2_idxs), aggregate,))
        lbl = '[smk_index.rvec] agg rvecs' if aggregate else '[smk_index.rvec] nonagg rvecs'
        mark, end_ = utool.log_progress(lbl, len(wx2_idxs), freq=50, with_time=True)
    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_wx2_idxs(wx2_idxs, len(words))
    # Compute Residuals
    rvecs_list, flags_list = smk_residuals.compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list)

    if utool.VERBOSE:
        print('Computed size(rvecs_list) = %r' % utool.get_object_size_str(rvecs_list))
        print('Computed size(flags_list) = %r' % utool.get_object_size_str(flags_list))
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
    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_wx2(words, wx2_rvecs, wx2_aids, wx2_fxs)
    if utool.VERBOSE or verbose:
        end_()
        print('[smk_index.rvec] L___ End Compute Residuals')
    return wx2_rvecs, wx2_aids, wx2_fxs, wx2_maws, wx2_flags


#@utool.cached_func('sccw', appname='smk', key_argx=[1, 2])
@profile
def compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_dflags, wx2_aids, wx2_idf,
                       wx2_dmaws, smk_alpha, smk_thresh, verbose=False):
    """
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
        >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_index
        >>> from ibeis.model.hots.smk import smk_debug
        >>> tup = smk_debug.testdata_raw_internals2(db='testdb1')
        >>> ibs, annots_df, invindex, wx2_idxs, wx2_idf, wx2_drvecs, wx2_aids, qparams = tup
        >>> wx2_dflags = invindex.wx2_dflags
        >>> wx2_dmaws  = invindex.wx2_maws
        >>> idx2_daid  = invindex.idx2_daid
        >>> daids      = invindex.daids
        >>> smk_alpha  = qparams.smk_alpha
        >>> smk_thresh = qparams.smk_thresh
        >>> wx2_idf    = wx2_idf
        >>> verbose = False
        >>> daid2_sccw = smk_index.compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_aids, wx2_idf, wx2_dmaws, smk_alpha, smk_thresh)
    """

    """
    #Auto:
    #    from ibeis.model.hots.smk import smk_index
    #    import utool; print(utool.make_default_docstr(smk_index.compute_data_sccw_))
    """
    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_wx2(wx2_rvecs=wx2_drvecs, wx2_aids=wx2_aids)
    if not utool.QUIET:
        print('\n[smk_index.sccw] +--- Start Compute Data Self Consistency Weight')
    if utool.VERBOSE or verbose:
        print('[smk_index.sccw] Compute SCCW smk_alpha=%r, smk_thresh=%r: ' % (smk_alpha, smk_thresh))
        mark1, end1_ = utool.log_progress(
            '[smk_index.sccw] SCCW group (by present words): ', len(wx2_drvecs),
            freq=100, with_time=WITH_TOTALTIME)
    #
    # Get list of aids and rvecs w.r.t. words (ie one item per word)
    wx_sublist = np.array(list(wx2_drvecs.keys()))
    """
    wx_sublist = wx_sublist[0:4]
    """
    aids_list1  = [wx2_aids[wx] for wx in wx_sublist]
    rvecs_list1 = [wx2_drvecs[wx] for wx in wx_sublist]
    maws_list1  = [wx2_dmaws[wx] for wx in wx_sublist]
    flags_list1 = [wx2_dflags[wx] for wx in wx_sublist]
    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.assert_single_assigned_maws(maws_list1)
    # Group by daids first and then by word index

    if utool.VERBOSE or verbose:
        end1_()

    #with utool.Timer('time2'):
    flat_aids, cumlen_list = utool.invertable_flatten2(aids_list1)
    flat_aids = np.array(flat_aids)
    flat_wxs  = np.array(utool.flatten(([wx] * len(aids) for wx, aids in zip(wx_sublist, aids_list1))))
    unique_aids, groupxs = clustertool.group_indicies(flat_aids)
    aid_list = unique_aids   # noqa
    # wxs grouped by aid
    grouped_wxs   = clustertool.apply_grouping(flat_wxs, groupxs)
    # Make subgroups of wxs for each aid
    grouped_subgroups = [clustertool.group_indicies(wxs) for wxs in grouped_wxs]

    # Use subgroups to get sccw score for each aid
    # This is very inefficient. It should be using idxs to look it up in one
    # big numpy array.
    grouped_rvecs     = clustertool.apply_grouping(np.array(utool.flatten(rvecs_list1)), groupxs)
    grouped_maws      = clustertool.apply_grouping(np.array(utool.flatten(maws_list1)), groupxs)
    grouped_flags     = clustertool.apply_grouping(np.array(utool.flatten(flags_list1)), groupxs)

    # For every daid, compute its sccw using pregrouped rvecs
    # Summation over words for each aid
    if utool.VERBOSE or verbose:
        lbl = '[smk_index.sccw] SCCW Sum (over daid): '
        mark2, end2_ = utool.log_progress(lbl, len(flat_aids), freq=100,
                                          with_time=WITH_TOTALTIME)

    def make_subgrouplist(grouped_vals, grouped_subgroups):
        return (clustertool.apply_grouping(vals, sub_groupxs)
                for vals, (sub_keys, sub_groupxs) in zip(grouped_vals, grouped_subgroups))

    aidrvecs_list2 = make_subgrouplist(grouped_rvecs, grouped_subgroups)
    aidflags_list2 = make_subgrouplist(grouped_flags, grouped_subgroups)
    aidmaws_list2  = make_subgrouplist(grouped_maws, grouped_subgroups)
    aididf_list2   = ([wx2_idf[wx] for wx in sub_wxs] for (sub_wxs, _) in grouped_subgroups)

    sccw_list = [
        smk_scoring.sccw_summation(rvecs_list, flags_list, idf_list, maws_list, smk_alpha, smk_thresh)
        for rvecs_list, flags_list, maws_list, idf_list in
        zip(aidrvecs_list2, aidflags_list2, aidmaws_list2, aididf_list2)
    ]

    """
    assert sorted(list(daid2_wx2_drvecs.keys())) == sorted(unique_aids)
    unique_aids
    _wx2_aidrvecs_list = [daid2_wx2_drvecs[daid] for daid in unique_aids]
    _aidwxs_list   = [list(wx2_aidrvecs.keys()) for wx2_aidrvecs in _wx2_aidrvecs_list]
    aidrvecs_list  = [list(wx2_aidrvecs.values()) for wx2_aidrvecs in _wx2_aidrvecs_list]
    aididf_list   = [[wx2_idf[wx] for wx in aidwxs] for aidwxs in _aidwxs_list]
    aidflags_list = [[wx2_dflags[wx] for wx in aidwxs] for aidwxs in _aidwxs_list]
    aidmaws_list  = [None for aidwxs in _aidwxs_list]

    z = [all([np.all(x == y) for x, y in zip(xs, ys)]) for xs, ys in zip(aidrvecs_list2, aidrvecs_list)]
    assert all(z)
    """

    '''
    len(aids_list1)
    len(wx_sublist)
    unflat_list = aids_list1
    '''
    #with utool.Timer('time1'):
    #    daid2_wx2_drvecs = clustertool.double_group(wx_sublist, aids_list1, rvecs_list1)
    #    # Get lists w.r.t daids
    #    aid_list = list(daid2_wx2_drvecs.keys())
    #    # list of mappings from words to rvecs foreach daid
    #    # [wx2_aidrvecs_1, ..., wx2_aidrvecs_nDaids,]
    #    _wx2_aidrvecs_list = list(daid2_wx2_drvecs.values())
    #    _aidwxs_list   = [list(wx2_aidrvecs.keys()) for wx2_aidrvecs in _wx2_aidrvecs_list]
    #    aidrvecs_list  = [list(wx2_aidrvecs.values()) for wx2_aidrvecs in _wx2_aidrvecs_list]
    #    aididf_list   = [[wx2_idf[wx] for wx in aidwxs] for aidwxs in _aidwxs_list]
    #    aidflags_list = [[wx2_dflags[wx] for wx in aidwxs] for aidwxs in _aidwxs_list]
    #    aidmaws_list  = [None for aidwxs in _aidwxs_list]

    #    if utool.DEBUG2:
    #        from ibeis.model.hots.smk import smk_debug
    #        smk_debug.check_data_smksumm(aididf_list, aidrvecs_list)
    #    # TODO: implement database side soft-assign
    #    sccw_list = [
    #        smk_scoring.sccw_summation(rvecs_list, None, idf_list, maws_list, smk_alpha, smk_thresh)
    #        for idf_list, rvecs_list, flags_list, maws_list in
    #        zip(aididf_list, aidrvecs_list, aidflags_list, aidmaws_list)
    #    ]

    daid2_sccw = dict(zip(aid_list, sccw_list))
    if utool.VERBOSE or verbose:
        end2_()
        print('[smk_index.sccw] L___ End Compute Data SCCW\n')

    return daid2_sccw


def OLD_compute_data_sccw_(idx2_daid, wx2_drvecs, wx2_dflags, wx2_aids, wx2_idf,
                           wx2_dmaws, smk_alpha, smk_thresh, verbose=False):
    #
    # Get list of aids and rvecs w.r.t. words (ie one item per word)
    wx_sublist = np.array(list(wx2_drvecs.keys()))
    """
    wx_sublist = wx_sublist[0:4]
    """
    aids_list1  = [wx2_aids[wx] for wx in wx_sublist]
    rvecs_list1 = [wx2_drvecs[wx] for wx in wx_sublist]
    maws_list1  = [wx2_dmaws[wx] for wx in wx_sublist]
    flags_list1 = [wx2_dflags[wx] for wx in wx_sublist]
    # Group by daids first and then by word index
    daid2_wx2_drvecs = clustertool.double_group(wx_sublist, aids_list1, rvecs_list1)

    # Get lists w.r.t daids
    aid_list = list(daid2_wx2_drvecs.keys())
    # list of mappings from words to rvecs foreach daid
    # [wx2_aidrvecs_1, ..., wx2_aidrvecs_nDaids,]
    _wx2_aidrvecs_list = list(daid2_wx2_drvecs.values())
    _aidwxs_list   = [list(wx2_aidrvecs.keys()) for wx2_aidrvecs in _wx2_aidrvecs_list]
    aidrvecs_list  = [list(wx2_aidrvecs.values()) for wx2_aidrvecs in _wx2_aidrvecs_list]
    aididf_list   = [[wx2_idf[wx] for wx in aidwxs] for aidwxs in _aidwxs_list]
    aidflags_list = [[wx2_dflags[wx] for wx in aidwxs] for aidwxs in _aidwxs_list]
    aidmaws_list  = [None for aidwxs in _aidwxs_list]

    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_data_smksumm(aididf_list, aidrvecs_list)
    # TODO: implement database side soft-assign
    sccw_list = [smk_scoring.sccw_summation(rvecs_list, None, idf_list, maws_list, smk_alpha, smk_thresh)
                 for idf_list, rvecs_list, flags_list, maws_list in
                 zip(aididf_list, aidrvecs_list, aidflags_list, aidmaws_list)]

    daid2_sccw = dict(zip(aid_list, sccw_list))
    return daid2_sccw
