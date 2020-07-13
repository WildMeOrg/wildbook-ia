# -*- coding: utf-8 -*-
"""
References:
    Jegou's Source Code, Data, and Publications
    http://people.rennes.inria.fr/Herve.Jegou/publications.html

    To aggregate or not to aggregate: selective match kernels for image search
    https://hal.inria.fr/hal-00864684/document

    Image search with selective match kernels: aggregation across single and multiple images
    http://image.ntua.gr/iva/files/Tolias_ijcv15_iasmk.pdf

    Negative evidences and co-occurrences in image retrieval: the benefit of PCA and whitening
    https://hal.inria.fr/file/index/docid/722626/filename/jegou_chum_eccv2012.pdf

    Revisiting the VLAD image representation
    https://hal.inria.fr/file/index/docid/850249/filename/nextvlad_hal.pdf

    Aggregating local descriptors into a compact image representation
    https://lear.inrialpes.fr/pubs/2010/JDSP10/jegou_compactimagerepresentation.pdf

    Large-scale image retrieval with compressed Fisher vectors
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.401.9140&rep=rep1&type=pdf

    Improving Bag of Features
    http://lear.inrialpes.fr/pubs/2010/JDS10a/jegou_improvingbof_preprint.pdf

    Lost in Quantization
    http://www.robots.ox.ac.uk/~vgg/publications/papers/philbin08.ps.gz

    A Context Dissimilarity Measure for Accurate and Efficient Image Search
    https://lear.inrialpes.fr/pubs/2007/JHS07/jegou_cdm.pdf

    Video Google: A text retrieval approach to object matching in videos
    http://www.robots.ox.ac.uk/~vgg/publications/papers/sivic03.pdf

    Hamming embedding and weak geometric consistency for large scale image search
    https://lear.inrialpes.fr/pubs/2008/JDS08/jegou_hewgc08.pdf

    Three things everyone should know to improve object retrieval
    https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

    Object retrieval with large vocabularies and fast spatial matching
    http://www.robots.ox.ac.uk:5000/~vgg/publications/2007/Philbin07/philbin07.pdf

    Aggregating Local Descriptors into Compact Codes
    https://hal.inria.fr/file/index/docid/633013/filename/jegou_aggregate.pdf

    Local visual query expansion
    https://hal.inria.fr/hal-00840721/PDF/RR-8325.pdf

    Root SIFT technique
    https://hal.inria.fr/hal-00688169/document

    Fisher Kernel For Large Scale Classification
    https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf

    Orientation covariant aggregation of local descriptors with embeddings
    https://arxiv.org/pdf/1407.2170.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
import utool as ut
import vtool as vt
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)


def cast_residual_integer(rvecs):
    """
    quantize residual vectors to 8-bits using the same trunctation hack as in
    SIFT. values will typically not reach the maximum, so we can multiply by a
    higher number for better fidelity.

    Args:
        rvecs (ndarray[float64_t]):

    Returns:
        ndarray[uint8_t]:

    CommandLine:
        python -m wbia.algo.smk.smk_funcs cast_residual_integer --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> rvecs = testdata_rvecs(dim=128)['rvecs'][4:]
        >>> rvecs_int8 = cast_residual_integer(rvecs)
        >>> rvecs_float = uncast_residual_integer(rvecs_int8)
        >>> # Casting from float to int8 will result in a max quantization error
        >>> measured_error = np.abs(rvecs_float - rvecs)
        >>> # But there are limits on what this error can be
        >>> cutoff = 127  # np.iinfo(np.int8).max
        >>> fidelity = 255.0
        >>> theory_error_in = 1 / fidelity
        >>> theory_error_out = (fidelity - cutoff) / fidelity
        >>> # Determine if any component values exceed the cutoff
        >>> is_inside = (np.abs(rvecs * fidelity) < cutoff)
        >>> # Check theoretical maximum for values inside and outside cutoff
        >>> error_stats_in = ut.get_stats(measured_error[is_inside])
        >>> error_stats_out = ut.get_stats(measured_error[~is_inside])
        >>> print('inside cutoff error stats: ' + ut.repr4(error_stats_in, precision=8))
        >>> print('outside cutoff error stats: ' + ut.repr4(error_stats_out, precision=8))
        >>> assert rvecs_int8.dtype == np.int8
        >>> assert np.all(measured_error[is_inside] < theory_error_in)
        >>> assert np.all(measured_error[~is_inside] < theory_error_out)
    """
    # maybe don't round?
    # return np.clip(rvecs * 255.0, -127, 127).astype(np.int8)
    # TODO: -128, 127
    return np.clip(np.round(rvecs * 255.0), -127, 127).astype(np.int8)


def uncast_residual_integer(rvecs):
    """
    Args:
        rvecs (ndarray[uint8_t]):

    Returns:
        ndarray[float64_t]:

    """
    return rvecs.astype(np.float32) / 255.0


def compute_stacked_agg_rvecs(words, flat_wxs_assign, flat_vecs, flat_offsets):
    """
    More efficient version of agg on a stacked structure

    Args:
        words (ndarray): entire vocabulary of words
        flat_wxs_assign (ndarray): maps a stacked index to word index
        flat_vecs (ndarray): stacked SIFT descriptors
        flat_offsets (ndarray): offset positions per annotation

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> data = testdata_rvecs(dim=2, nvecs=1000, nannots=10)
        >>> words = data['words']
        >>> flat_offsets = data['offset_list']
        >>> flat_wxs_assign, flat_vecs = ut.take(data, ['idx_to_wx', 'vecs'])
        >>> tup = compute_stacked_agg_rvecs(words, flat_wxs_assign, flat_vecs, flat_offsets)
        >>> all_agg_vecs, all_error_flags, agg_offset_list = tup
        >>> agg_rvecs_list = [all_agg_vecs[l:r] for l, r in ut.itertwo(agg_offset_list)]
        >>> agg_flags_list = [all_error_flags[l:r] for l, r in ut.itertwo(agg_offset_list)]
        >>> assert len(agg_flags_list) == len(flat_offsets) - 1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> data = testdata_rvecs(dim=2, nvecs=100, nannots=5)
        >>> words = data['words']
        >>> flat_offsets = data['offset_list']
        >>> flat_wxs_assign, flat_vecs = ut.take(data, ['idx_to_wx', 'vecs'])
        >>> tup = compute_stacked_agg_rvecs(words, flat_wxs_assign, flat_vecs, flat_offsets)
        >>> all_agg_vecs, all_error_flags, agg_offset_list = tup
        >>> agg_rvecs_list = [all_agg_vecs[l:r] for l, r in ut.itertwo(agg_offset_list)]
        >>> agg_flags_list = [all_error_flags[l:r] for l, r in ut.itertwo(agg_offset_list)]
        >>> assert len(agg_flags_list) == len(flat_offsets) - 1
    """
    grouped_wxs = [
        flat_wxs_assign[left:right] for left, right in ut.itertwo(flat_offsets)
    ]

    # Assume single assignment, aggregate everything
    # across the entire database
    flat_offsets = np.array(flat_offsets)

    idx_to_dx = (
        np.searchsorted(flat_offsets, np.arange(len(flat_wxs_assign)), side='right') - 1
    ).astype(np.int32)

    if isinstance(flat_wxs_assign, np.ma.masked_array):
        wx_list = flat_wxs_assign.T[0].compressed()
    else:
        wx_list = flat_wxs_assign.T[0].ravel()
    unique_wx, groupxs = vt.group_indices(wx_list)

    dim = flat_vecs.shape[1]
    if isinstance(flat_wxs_assign, np.ma.masked_array):
        dx_to_wxs = [np.unique(wxs.compressed()) for wxs in grouped_wxs]
    else:
        dx_to_wxs = [np.unique(wxs.ravel()) for wxs in grouped_wxs]
    dx_to_nagg = [len(wxs) for wxs in dx_to_wxs]
    num_agg_vecs = sum(dx_to_nagg)
    # all_agg_wxs = np.hstack(dx_to_wxs)
    agg_offset_list = np.array([0] + ut.cumsum(dx_to_nagg))
    # Preallocate agg residuals for all dxs
    all_agg_vecs = np.empty((num_agg_vecs, dim), dtype=np.float32)
    all_agg_vecs[:, :] = np.nan

    # precompute agg residual stack
    i_to_dxs = vt.apply_grouping(idx_to_dx, groupxs)
    subgroup = [vt.group_indices(dxs) for dxs in ut.ProgIter(i_to_dxs)]
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
        word = words[wx : wx + 1]

        offsets1 = agg_offset_list.take(dxs)
        offsets2 = [np.where(dx_to_wxs[dx] == wx)[0][0] for dx in dxs]
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
        agg_rvecs_ = [rvec_group.sum(axis=0) for rvec_group in grouped_rvecs]
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

    tup = all_agg_vecs, all_error_flags, agg_offset_list
    return tup


def compute_rvec(vecs, word):
    """
    Compute residual vectors phi(x_c)

    Subtract each vector from its quantized word to get the resiudal, then
    normalize residuals to unit length.

    CommandLine:
        python -m wbia.algo.smk.smk_funcs compute_rvec --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> vecs, words = ut.take(testdata_rvecs(), ['vecs', 'words'])
        >>> word = words[-1]
        >>> rvecs, error_flags = compute_rvec(vecs, word)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.figure()
        >>> # recenter residuals for visualization
        >>> cvecs = (rvecs + word[None, :])
        >>> pt.plot(word[0], word[1], 'r*', markersize=12, label='word')
        >>> pt.plot(vecs.T[0], vecs.T[1], 'go', label='original vecs')
        >>> pt.plot(cvecs.T[0], cvecs.T[1], 'b.', label='re-centered rvec')
        >>> pt.draw_line_segments2(cvecs, [word] * len(cvecs), alpha=.5, color='black')
        >>> pt.gca().set_aspect('equal')
        >>> pt.legend()
        >>> ut.show_if_requested()
    """
    rvecs = np.subtract(vecs.astype(np.float32), word.astype(np.float32))
    # If a vec is a word then the residual is 0 and it cant be L2 noramlized.
    error_flags = np.all(rvecs == 0, axis=1)
    vt.normalize(rvecs, axis=1, out=rvecs)
    # reset these values to zero
    if np.any(error_flags):
        rvecs[error_flags, :] = 0
    return rvecs, error_flags


def aggregate_rvecs(rvecs, maws, error_flags):
    r"""
    Compute aggregated residual vectors Phi(X_c)

    CommandLine:
        python -m wbia.algo.smk.smk_funcs aggregate_rvecs --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> vecs, words = ut.take(testdata_rvecs(), ['vecs', 'words'])
        >>> word = words[-1]
        >>> rvecs, error_flags = compute_rvec(vecs, word)
        >>> maws = [1.0] * len(rvecs)
        >>> agg_rvec, agg_flag = aggregate_rvecs(rvecs, maws, error_flags)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.qt4ensure()
        >>> pt.figure()
        >>> # recenter residuals for visualization
        >>> agg_cvec = agg_rvec + word
        >>> cvecs = (rvecs + word[None, :])
        >>> pt.plot(word[0], word[1], 'r*', markersize=12, label='word')
        >>> pt.plot(agg_cvec[0], agg_cvec[1], 'ro', label='re-centered agg_rvec')
        >>> pt.plot(vecs.T[0], vecs.T[1], 'go', label='original vecs')
        >>> pt.plot(cvecs.T[0], cvecs.T[1], 'b.', label='re-centered rvec')
        >>> pt.draw_line_segments2([word] * len(cvecs), cvecs, alpha=.5, color='black')
        >>> pt.draw_line_segments2([word], [agg_cvec], alpha=.5, color='red')
        >>> pt.gca().set_aspect('equal')
        >>> pt.legend()
        >>> ut.show_if_requested()
    """
    # Propogate errors from previous step
    agg_flag = np.any(error_flags, axis=0)
    if rvecs.shape[0] == 0:
        raise ValueError('cannot compute without rvecs')
    if rvecs.shape[0] == 1:
        # Efficiency shortcut
        agg_rvec = rvecs
    else:
        # Prealloc residual vector, take the weighted sum and renormalize.
        agg_rvec = np.empty(rvecs.shape[1], dtype=np.float32)
        out = agg_rvec

        if False:
            # Take weighted average of multi-assigned vectors
            coeff = np.divide(maws, maws.sum())[:, None]
            agg_rvec = (coeff * rvecs).sum(axis=0, out=out)
        else:
            # Don't consider multiple assignment weights
            agg_rvec = rvecs.sum(axis=0, out=out)
        is_zero = np.all(agg_rvec == 0)

        vt.normalize(agg_rvec, axis=0, out=agg_rvec)

        if is_zero:
            agg_flag = True
    return agg_rvec, agg_flag


def weight_multi_assigns(
    _idx_to_wx,
    _idx_to_wdist,
    massign_alpha=1.2,
    massign_sigma=80.0,
    massign_equal_weights=False,
):
    r"""
    Multi Assignment Weight Filtering from Improving Bag of Features

    Args:
        massign_equal_weights (): Turns off soft weighting. Gives all assigned
            vectors weight 1

    Returns:
        tuple : (idx_to_wxs, idx_to_maws)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> _idx_to_wx = np.array([[0, 1], [2, 3], [4, 5], [2, 0]])
        >>> _idx_to_wdist = np.array([[.1, .11], [.2, .25], [.03, .25], [0, 1]])
        >>> massign_alpha = 1.2
        >>> massign_sigma = 80.0
        >>> massign_equal_weights = False
        >>> idx_to_wxs, idx_to_maws = weight_multi_assigns(
        >>>     _idx_to_wx, _idx_to_wdist, massign_alpha, massign_sigma,
        >>>     massign_equal_weights)
        >>> result = 'idx_to_wxs = %s' % (ut.repr2(idx_to_wxs.astype(np.float64)),)
        >>> result += '\nidx_to_maws = %s' % (ut.repr2(idx_to_maws, precision=2),)
        >>> print(result)
        idx_to_wxs = np.ma.MaskedArray([[0., 1.],
                           [2., inf],
                           [4., inf],
                           [2., 0.]])
        idx_to_maws = np.ma.MaskedArray([[0.5, 0.5],
                           [1. , inf],
                           [1. , inf],
                           [0.5, 0.5]])

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> _idx_to_wx = np.array([[0, 1], [2, 3], [4, 5], [2, 0]])
        >>> _idx_to_wdist = np.array([[.1, .11], [.2, .25], [.03, .25], [0, 1]])
        >>> _idx_to_wx = _idx_to_wx.astype(np.int32)
        >>> _idx_to_wdist = _idx_to_wdist.astype(np.float32)
        >>> massign_alpha = 1.2
        >>> massign_sigma = 80.0
        >>> massign_equal_weights = True
        >>> idx_to_wxs, idx_to_maws = weight_multi_assigns(
        >>>     _idx_to_wx, _idx_to_wdist, massign_alpha, massign_sigma,
        >>>     massign_equal_weights)
        >>> result = 'idx_to_wxs = %s' % (ut.repr2(idx_to_wxs.astype(np.float64)),)
        >>> result += '\nidx_to_maws = %s' % (ut.repr2(idx_to_maws, precision=2),)
        >>> print(result)
        idx_to_wxs = np.ma.MaskedArray([[0., 1.],
                           [2., inf],
                           [4., inf],
                           [2., 0.]])
        idx_to_maws = np.ma.MaskedArray([[1., 1.],
                           [1., inf],
                           [1., inf],
                           [1., 1.]])
    """
    if _idx_to_wx.shape[1] <= 1:
        idx_to_wxs = _idx_to_wx.tolist()
        idx_to_maws = [[1.0]] * len(idx_to_wxs)
    else:
        # Valid word assignments are beyond fraction of distance to the nearest word
        ma_thresh = _idx_to_wdist.T[0:1].T.copy()
        # If the nearest word has distance 0 then this threshold is too hard so
        # we should use the distance to the second nearest word.
        flag_too_close = np.isclose(ma_thresh, 0)
        ma_thresh[flag_too_close] = _idx_to_wdist.T[1:2].T[flag_too_close]
        # Compute a threshold based on the nearest assignment.
        eps = 0.001
        ma_thresh = np.add(eps, ma_thresh, out=ma_thresh)
        ma_thresh = np.multiply(massign_alpha, ma_thresh, out=ma_thresh)
        # Invalidate assignments that are too far away
        invalid = np.greater_equal(_idx_to_wdist, ma_thresh)
        if ut.VERBOSE:
            nInvalid = (invalid.size - invalid.sum(), invalid.size)
            print('[maw] + massign_alpha = %r' % (massign_alpha,))
            print('[maw] + massign_sigma = %r' % (massign_sigma,))
            print('[maw] + massign_equal_weights = %r' % (massign_equal_weights,))
            print('[maw] * Marked %d/%d assignments as invalid' % nInvalid)

        if massign_equal_weights:
            # Performance hack from jegou paper: just give everyone equal weight
            idx_to_wxs = np.ma.masked_array(_idx_to_wx, mask=invalid, fill_value=-1)
            # idx_to_wxs  = ut.lmap(ut.filter_Nones, masked_wxs.tolist())
            idx_to_maws = np.ma.ones(
                idx_to_wxs.shape, fill_value=np.nan, dtype=np.float32
            )
            idx_to_maws.mask = idx_to_wxs.mask
            # idx_to_maws = [np.ones(len(wxs), dtype=np.float32)
            #                for wxs in idx_to_wxs]
        else:
            # More natural weighting scheme
            # Weighting as in Lost in Quantization
            gauss_numer = np.negative(_idx_to_wdist)
            gauss_denom = 2 * (massign_sigma ** 2)
            gauss_exp = np.divide(gauss_numer, gauss_denom)
            unnorm_maw = np.exp(gauss_exp)
            # Mask invalid multiassignment weights
            masked_unorm_maw = np.ma.masked_array(unnorm_maw, mask=invalid)
            # Normalize multiassignment weights from 0 to 1
            masked_norm = masked_unorm_maw.sum(axis=1)[:, np.newaxis]
            masked_maw = np.divide(masked_unorm_maw, masked_norm)
            masked_wxs = np.ma.masked_array(_idx_to_wx, mask=invalid)
            # Just keep masked arrays as they are
            # (more efficient than python lists)
            idx_to_wxs = masked_wxs
            idx_to_maws = masked_maw
            # # Remove masked weights and word indexes
            # idx_to_wxs  = [np.array(ut.filter_Nones(wxs), dtype=np.int32)
            #                for wxs in masked_wxs.tolist()]
            # idx_to_maws = [np.array(ut.filter_Nones(maw), dtype=np.float32)
            #                for maw in masked_maw.tolist()]
    return idx_to_wxs, idx_to_maws


def assign_to_words(
    vocab,
    idx_to_vec,
    nAssign,
    massign_alpha=1.2,
    massign_sigma=80.0,
    massign_equal_weights=False,
    verbose=None,
):
    """
    Assigns descriptor-vectors to nearest word.

    Notes:
        Maybe move out of this file? The usage of vocab is out of this file
        scope.

    Args:
        wordflann (FLANN): nearest neighbor index over words
        words (ndarray): vocabulary words
        idx_to_vec (ndarray): descriptors to assign
        nAssign (int): number of words to assign each descriptor to
        massign_alpha (float): multiple-assignment ratio threshold
        massign_sigma (float): multiple-assignment gaussian variance
        massign_equal_weights (bool): assign equal weight to all multiassigned words

    Returns:
        tuple: inverted index, multi-assigned weights, and forward index
        formated as::

            * wx_to_idxs - word index   -> vector indexes
            * wx_to_maws - word index   -> multi-assignment weights
            * idx2_wxs - vector index -> assigned word indexes

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> idx_to_vec = depc.d.get_feat_vecs(aid_list)[0][0::300]
        >>> idx_to_vec = np.vstack((idx_to_vec, vocab.wx_to_word[0]))
        >>> nAssign = 2
        >>> massign_equal_weights = False
        >>> massign_alpha = 1.2
        >>> massign_sigma = 80.0
        >>> nAssign = 2
        >>> idx_to_wxs, idx_to_maws = assign_to_words(vocab, idx_to_vec, nAssign)
        >>> print('idx_to_maws = %s' % (ut.repr2(idx_to_wxs, precision=2),))
        >>> print('idx_to_wxs = %s' % (ut.repr2(idx_to_maws, precision=2),))
    """
    if verbose is None:
        verbose = ut.VERBOSE
    if verbose:
        print('[vocab.assign] +--- Start Assign vecs to words.')
        print('[vocab.assign] * nAssign=%r' % nAssign)
        print('[vocab.assign] assign_to_words_. len(idx_to_vec) = %r' % len(idx_to_vec))
    _idx_to_wx, _idx_to_wdist = vocab.nn_index(idx_to_vec, nAssign)
    if nAssign > 1:
        idx_to_wxs, idx_to_maws = weight_multi_assigns(
            _idx_to_wx,
            _idx_to_wdist,
            massign_alpha,
            massign_sigma,
            massign_equal_weights,
        )
    else:
        # idx_to_wxs = _idx_to_wx.tolist()
        # idx_to_maws = [[1.0]] * len(idx_to_wxs)
        idx_to_wxs = np.ma.masked_array(_idx_to_wx, fill_value=-1)
        idx_to_maws = np.ma.ones(idx_to_wxs.shape, fill_value=-1, dtype=np.float32)
        idx_to_maws.mask = idx_to_wxs.mask
    return idx_to_wxs, idx_to_maws


def invert_assigns_old(idx_to_wxs, idx_to_maws, verbose=False):
    r"""
    Inverts assignment of vectors to words into words to vectors.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> idx_to_wxs = [
        >>>     np.array([0, 4], dtype=np.int32),
        >>>     np.array([2], dtype=np.int32),
        >>>     np.array([2, 0], dtype=np.int32),
        >>> ]
        >>> idx_to_maws = [
        >>>     np.array([ 0.5,  0.5], dtype=np.float32),
        >>>     np.array([ 1.], dtype=np.float32),
        >>>     np.array([ 0.5,  0.5], dtype=np.float32),
        >>> ]
        >>> wx_to_idxs, wx_to_maws = invert_assigns_old(idx_to_wxs, idx_to_maws)
        >>> result = 'wx_to_idxs = %s' % (ut.repr4(wx_to_idxs, with_dtype=True),)
        >>> result += '\nwx_to_maws = %s' % (ut.repr4(wx_to_maws, with_dtype=True),)
        >>> print(result)
        wx_to_idxs = {
            0: np.array([0, 2], dtype=np.int32),
            2: np.array([1, 2], dtype=np.int32),
            4: np.array([0], dtype=np.int32),
        }
        wx_to_maws = {
            0: np.array([0.5, 0.5], dtype=np.float32),
            2: np.array([1. , 0.5], dtype=np.float32),
            4: np.array([0.5], dtype=np.float32),
        }
    """
    # Invert mapping -- Group by word indexes
    idx_to_nAssign = [len(wxs) for wxs in idx_to_wxs]
    jagged_idxs = [[idx] * num for idx, num in enumerate(idx_to_nAssign)]
    wx_keys, groupxs = vt.jagged_group(idx_to_wxs)
    idxs_list = vt.apply_jagged_grouping(jagged_idxs, groupxs)
    idxs_list = [np.array(idxs, dtype=np.int32) for idxs in idxs_list]
    wx_to_idxs = dict(zip(wx_keys, idxs_list))
    maws_list = vt.apply_jagged_grouping(idx_to_maws, groupxs)
    maws_list = [np.array(maws, dtype=np.float32) for maws in maws_list]
    wx_to_maws = dict(zip(wx_keys, maws_list))
    if verbose:
        print('[vocab] L___ End Assign vecs to words.')
    return (wx_to_idxs, wx_to_maws)


def invert_assigns(idx_to_wxs, idx_to_maws, verbose=False):
    r"""
    Inverts assignment of
    vectors->to->words into words->to->vectors.
    Invert mapping -- Group by word indexes

    This gives a HUGE speedup over the old invert_assigns

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> idx_to_wxs = np.ma.array([
        >>>     (0, 4),
        >>>     (2, -1),
        >>>     (2, 0)], dtype=np.int32)
        >>> idx_to_wxs[1, 1] = np.ma.masked
        >>> idx_to_maws = np.ma.array(
        >>>     [(.5, 1.), (1., np.nan), (.5, .5)], dtype=np.float32)
        >>> idx_to_maws[1, 1] = np.ma.masked
        >>> tup = invert_assigns(idx_to_wxs, idx_to_maws)
        >>> wx_to_idxs, wx_to_maws = tup
        >>> result = 'wx_to_idxs = %s' % (ut.repr4(wx_to_idxs, with_dtype=True),)
        >>> result += '\nwx_to_maws = %s' % (ut.repr4(wx_to_maws, with_dtype=True),)
        >>> print(result)
        wx_to_idxs = {
            0: np.array([0, 2], dtype=np.int32),
            2: np.array([1, 2], dtype=np.int32),
            4: np.array([0], dtype=np.int32),
        }
        wx_to_maws = {
            0: np.array([0.5, 0.5], dtype=np.float32),
            2: np.array([1. , 0.5], dtype=np.float32),
            4: np.array([1.], dtype=np.float32),
        }
    """
    assert isinstance(idx_to_wxs, np.ma.masked_array)
    assert isinstance(idx_to_maws, np.ma.masked_array)

    nrows, ncols = idx_to_wxs.shape
    if len(idx_to_wxs.mask.shape) == 0:
        valid_mask = np.ones((nrows, ncols), dtype=np.bool)
    else:
        valid_mask = ~idx_to_maws.mask
        # idx_to_nAssign = (valid_mask).sum(axis=1)

    _valid_x2d = np.flatnonzero(valid_mask)
    flat_idxs = np.floor_divide(_valid_x2d, ncols, dtype=np.int32)
    flat_wxs = idx_to_wxs.compressed()
    flat_maws = idx_to_maws.compressed()

    sortx = flat_wxs.argsort()
    flat_wxs = flat_wxs.take(sortx)
    flat_idxs = flat_idxs.take(sortx)
    flat_maws = flat_maws.take(sortx)

    wx_keys, groupxs = vt.group_indices(flat_wxs)
    idxs_list = vt.apply_grouping(flat_idxs, groupxs)
    maws_list = vt.apply_grouping(flat_maws, groupxs)

    wx_to_idxs = dict(zip(wx_keys, idxs_list))
    wx_to_maws = dict(zip(wx_keys, maws_list))

    if verbose:
        print('[vocab] L___ End Assign vecs to words.')
    return (wx_to_idxs, wx_to_maws)


def invert_lists(aids, wx_lists, all_wxs=None):
    """
    takes corresponding lists of (aids, wxs) and maps wxs to aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> aids = [1, 2, 3]
        >>> wx_lists = [[0, 1], [20, 0, 1], [3]]
        >>> wx_to_aids = invert_lists(aids, wx_lists)
        >>> result = ('wx_to_aids = %s' % (ut.repr2(wx_to_aids),))
        >>> print(result)
        wx_to_aids = {0: [1, 2], 1: [1, 2], 3: [3], 20: [2]}
    """
    if all_wxs is None:
        wx_to_aids = ut.ddict(list)
    else:
        wx_to_aids = {wx: [] for wx in all_wxs}
    for aid, wxs in zip(aids, wx_lists):
        for wx in wxs:
            wx_to_aids[wx].append(aid)
    return wx_to_aids


def inv_doc_freq(ndocs_total, ndocs_per_word):
    """
    Args:
        ndocs_total (int): numer of unique documents
        ndocs_per_word (ndarray): ndocs_per_word[i] should correspond to the
            number of unique documents containing word[i]

    Returns:
        ndarrary: idf_per_word

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> ndocs_total = 21
        >>> ndocs_per_word = [0, 21, 20, 2, 15, 8, 12, 1, 2]
        >>> idf_per_word = inv_doc_freq(ndocs_total, ndocs_per_word)
        >>> result = '%s' % (ut.repr2(idf_per_word, precision=2),)
        >>> print(result)
        np.array([0.  , 0.  , 0.05, 2.35, 0.34, 0.97, 0.56, 3.04, 2.35])
    """
    # We add epsilon to numer and denom to ensure recep is a probability
    out = np.empty(len(ndocs_per_word), dtype=np.float32)
    out[:] = ndocs_per_word
    # use jegou's version
    out = np.divide(ndocs_total, out, out=out)
    idf_per_word = np.log(out, out=out)
    idf_per_word[np.isinf(idf_per_word)] = 0
    return idf_per_word
    # if not adjust:
    #     # Typically for IDF, 1 is added to the denom to prevent divide by 0
    #     out[:] = ndocs_per_word
    #     ndocs_total += 1
    # else:
    #     # We add the 1 to the denominator and 2 to the numberator
    #     # to prevent words from receiving 0 weight
    #     out = np.add(ndocs_per_word, 1, out=out)
    #     ndocs_total += 2
    # out = np.divide(ndocs_total, out, out=out)
    # idf_per_word = np.log(out, out=out)
    # return idf_per_word


@profile
def match_scores_agg(PhisX, PhisY, flagsX, flagsY, alpha, thresh):
    """
    Scores matches to multiple words using aggregate residual vectors

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> PhisX = np.array([[ 0.        ,  0.        ],
        >>>                   [-1.        ,  0.        ],
        >>>                   [ 0.85085751,  0.52539652],
        >>>                   [-0.89795083, -0.4400958 ],
        >>>                   [-0.99934547,  0.03617512]])
        >>> PhisY = np.array([[ 0.88299408, -0.46938411],
        >>>                   [-0.12096522, -0.99265675],
        >>>                   [-0.99948266, -0.03216222],
        >>>                   [-0.08394916, -0.99647004],
        >>>                   [-0.96414952, -0.26535957]])
        >>> flagsX = np.array([True, False, False, True, False])[:, None]
        >>> flagsY = np.array([False, False, False, True, False])[:, None]
        >>> alpha = 3.0
        >>> thresh = 0.0
        >>> score_list = match_scores_agg(PhisX, PhisY, flagsX, flagsY, alpha, thresh)
        >>> result = 'score_list = ' + ut.repr2(score_list, precision=4)
        >>> print(result)
        score_list = np.array([1.    , 0.0018, 0.    , 1.    , 0.868 ])
    """
    # Can speedup aggregate with one vector per word assumption.
    # Take dot product between correponding VLAD vectors
    u = (PhisX * PhisY).sum(axis=1)
    # Propogate error flags
    flags = np.logical_or(flagsX.T[0], flagsY.T[0])
    assert len(flags) == len(u), 'mismatch'
    u[flags] = 1
    score_list = selectivity(u, alpha, thresh, out=u)
    return score_list


@profile
def match_scores_sep(phisX_list, phisY_list, flagsX_list, flagsY_list, alpha, thresh):
    """
    Scores matches to multiple words using lists of separeated residual vectors

    """
    scores_list = []
    _iter = zip(phisX_list, phisY_list, flagsX_list, flagsY_list)
    for phisX, phisY, flagsX, flagsY in _iter:
        u = phisX.dot(phisY.T)
        flags = np.logical_or(flagsX.T[0], flagsY.T[0])
        u[flags] = 1
        scores = selectivity(u, alpha, thresh, out=u)
        scores_list.append(scores)
    return scores_list


@profile
def build_matches_agg(X_fxs, Y_fxs, X_maws, Y_maws, score_list):
    r"""
    Builds explicit features matches. Break and distribute up each aggregate
    score amongst its contributing features.

    Returns:
        tuple: (fm, fs)

    CommandLine:
        python -m wbia.algo.smk.smk_funcs build_matches_agg --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> map_int = ut.partial(ut.lmap, ut.partial(np.array, dtype=np.int32))
        >>> map_float = ut.partial(ut.lmap, ut.partial(np.array, dtype=np.float32))
        >>> X_fxs = map_int([[0, 1], [2, 3, 4], [5]])
        >>> Y_fxs = map_int([[8], [0, 4], [99]])
        >>> X_maws = map_float([[1, 1], [1, 1, 1], [1]])
        >>> Y_maws = map_float([[1], [1, 1], [1]])
        >>> score_list = np.array([1, 2, 3], dtype=np.float32)
        >>> (fm, fs) = build_matches_agg(X_fxs, Y_fxs, X_maws, Y_maws, score_list)
        >>> print('fm = ' + ut.repr2(fm))
        >>> print('fs = ' + ut.repr2(fs))
        >>> assert len(fm) == len(fs)
        >>> assert score_list.sum() == fs.sum()
    """
    # Build feature matches
    # Spread word score according to contriubtion (maw) weight
    unflat_contrib = [
        maws1[:, None].dot(maws2[:, None].T).ravel()
        for maws1, maws2 in zip(X_maws, Y_maws)
    ]
    unflat_factor = [contrib / contrib.sum() for contrib in unflat_contrib]
    # factor_list = np.divide(score_list, factor_list, out=factor_list)
    for score, factor in zip(score_list, unflat_factor):
        np.multiply(score, factor, out=factor)
    unflat_fs = unflat_factor

    # itertools.product seems fastest for small arrays
    unflat_fm = (ut.product(fxs1, fxs2) for fxs1, fxs2 in zip(X_fxs, Y_fxs))

    fm = np.array(ut.flatten(unflat_fm), dtype=np.int32)
    fs = np.array(ut.flatten(unflat_fs), dtype=np.float32)
    isvalid = np.greater(fs, 0)
    fm = fm.compress(isvalid, axis=0)
    fs = fs.compress(isvalid, axis=0)
    return fm, fs


@profile
def build_matches_sep(X_fxs, Y_fxs, scores_list):
    r"""
    Just build matches. Scores have already been broken up. No need to do that.

    Returns:
        tuple: (fm, fs)

    CommandLine:
        python -m wbia.algo.smk.smk_funcs build_matches_agg --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> map_int = ut.partial(ut.lmap, ut.partial(np.array, dtype=np.int32))
        >>> map_float = ut.partial(ut.lmap, ut.partial(np.array, dtype=np.float32))
        >>> X_fxs = map_int([[0, 1], [2, 3, 4], [5]])
        >>> Y_fxs = map_int([[8], [0, 4], [99]])
        >>> scores_list = map_float([
        >>>     [[.1], [.2],],
        >>>     [[.3, .4], [.4, .6], [.5, .9],],
        >>>     [[.4]],
        >>> ])
        >>> (fm, fs) = build_matches_sep(X_fxs, Y_fxs, scores_list)
        >>> print('fm = ' + ut.repr2(fm))
        >>> print('fs = ' + ut.repr2(fs))
        >>> assert len(fm) == len(fs)
        >>> assert np.isclose(np.sum(ut.total_flatten(scores_list)), fs.sum())
    """
    fs = np.array(ut.total_flatten(scores_list), dtype=np.float32)
    unflat_fm = (ut.product(fxs1, fxs2) for fxs1, fxs2 in zip(X_fxs, Y_fxs))
    fm = np.array(ut.flatten(unflat_fm), dtype=np.int32)
    isvalid = np.greater(fs, 0)
    fm = fm.compress(isvalid, axis=0)
    fs = fs.compress(isvalid, axis=0)
    return fm, fs


@profile
def gamma_agg(phisX, flagsX, weight_list, alpha, thresh):
    r"""
    Computes gamma (self consistency criterion)
    It is a scalar which ensures K(X, X) = 1

    Returns:
        float: sccw self-consistency-criterion weight

    Math:
        gamma(X) = (sum_{c in C} w_c M(X_c, X_c))^{-.5}

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.smk_pipeline import *  # NOQA
        >>> ibs, smk, qreq_= testdata_smk()
        >>> X = qreq_.qinva.grouped_annots[0]
        >>> wx_to_weight = qreq_.wx_to_weight
        >>> print('X.gamma = %r' % (gamma(X),))
    """
    scores = match_scores_agg(phisX, phisX, flagsX, flagsX, alpha, thresh)
    sccw = sccw_normalize(scores, weight_list)
    return sccw


@profile
def gamma_sep(phisX_list, flagsX_list, weight_list, alpha, thresh):
    scores_list = match_scores_sep(
        phisX_list, phisX_list, flagsX_list, flagsX_list, alpha, thresh
    )
    scores = np.array([scores.sum() for scores in scores_list])
    sccw = sccw_normalize(scores)
    return sccw


def sccw_normalize(scores, weight_list):
    scores *= weight_list
    score = scores.sum()
    sccw = np.reciprocal(np.sqrt(score))
    return sccw


@profile
def selective_match_score(phisX, phisY, flagsX, flagsY, alpha, thresh):
    """
    computes the score of each feature match
    """
    u = phisX.dot(phisY.T)
    # Give error flags full scores. These are typically distinctive and
    # important cases without enough info to get residual data.
    flags = np.logical_or(flagsX[:, None], flagsY)
    u[flags] = 1
    score = selectivity(u, alpha, thresh, out=u)
    return score


@profile
def selectivity(u, alpha=3.0, thresh=0.0, out=None):
    r"""
    The selectivity function thresholds and applies a power law.

    This downweights weak matches.
    The following is the exact definition from SMK paper.
    sigma_alpha(u) = (sign(u) * (u ** alpha) if u > thresh else 0)

    Args:
        u (ndarray): input score between (-1, +1)
        alpha (float): power law (default = 3.0)
        thresh (float): number between 0 and 1 (default = 0.0)
        out (None): inplace output (default = None)

    Returns:
        float: score

    CommandLine:
        python -m wbia.plottool plot_func --show --range=-1,1  \
            --setup="import wbia" \
            --func wbia.algo.smk.smk_funcs.selectivity \
            "lambda u: sign(u) * abs(u)**3.0 * greater_equal(u, 0)"
        python -m wbia.algo.smk.smk_funcs selectivity --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> u = np.array([-1.0, -.5, -.1, 0, .1, .5, 1.0])
        >>> alpha = 3.0
        >>> thresh = 0
        >>> score = selectivity(u, alpha, thresh)
        >>> result = ut.repr2(score.tolist(), precision=4)
        >>> print(result)
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0010, 0.1250, 1.0000]
    """
    score = u
    flags = np.less(score, thresh)
    isign = np.sign(score)
    score = np.abs(score, out=out)
    score = np.power(score, alpha, out=out)
    score = np.multiply(isign, score, out=out)
    score[flags] = 0
    #
    # score = np.sign(u) * np.power(np.abs(u), alpha)
    # score *= flags
    return score


def testdata_rvecs(dim=2, nvecs=13, nwords=5, nannots=4):
    """
    two dimensional test data

    CommandLine:
        python -m wbia.algo.smk.smk_funcs testdata_rvecs --show

    Ignore:
        dim = 2
        nvecs = 13
        nwords = 5
        nannots = 5

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.smk_funcs import *  # NOQA
        >>> data = testdata_rvecs()
        >>> ut.quit_if_noshow()
        >>> exec(ut.execstr_dict(data))
        >>> import wbia.plottool as pt
        >>> from scipy.spatial import Voronoi, voronoi_plot_2d
        >>> pt.qt4ensure()
        >>> fig = pt.figure()
        >>> vor = Voronoi(words)
        >>> pt.plot(words.T[0], words.T[1], 'r*', label='words')
        >>> pt.plot(vecs.T[0], vecs.T[1], 'b.', label='vecs')
        >>> # lines showing assignments (and residuals)
        >>> pts1 = vecs
        >>> pts2 = words[idx_to_wx.T[0]]
        >>> pt.draw_line_segments2(pts1, pts2)
        >>> pt.plot(vecs.T[0], vecs.T[1], 'g.', label='vecs')
        >>> voronoi_plot_2d(vor, show_vertices=False, ax=pt.gca())
        >>> extents = vt.get_pointset_extents(np.vstack((vecs, words)))
        >>> extents = vt.scale_extents(extents, 1.1)
        >>> ax = pt.gca()
        >>> ax.set_aspect('equal')
        >>> ax.set_xlim(*extents[0:2])
        >>> ax.set_ylim(*extents[2:4])
        >>> ut.show_if_requested()
    """
    from sklearn.metrics.pairwise import euclidean_distances

    rng = np.random.RandomState(42)
    # dim = dim
    # nvecs = 13
    # nwords = 5
    words = rng.rand(nwords, dim)
    vecs = rng.rand(nvecs, dim)
    # Create vector = word special case
    words[0, :] = 0.0
    vecs[0, :] = 0.0
    # Create aggregate canceling special case
    # TODO: ensure no other words are closer
    words[1, :] = 2.0
    vecs[1:3, :] = 2.0
    vecs[1, 0] = 1.2
    vecs[2, 0] = 2.2
    # vt.normalize(words, axis=1, inplace=True)
    # vt.normalize(vecs, axis=1, inplace=True)
    dist_mat = euclidean_distances(vecs, words)
    nAssign = 1
    sortx2d = dist_mat.argsort(axis=1)
    row_offset = np.arange(sortx2d.shape[0])[:, None]
    sortx1d = (row_offset * sortx2d.shape[1] + sortx2d).ravel()
    idx_to_dist = dist_mat.take(sortx1d).reshape(sortx2d.shape)[:, :nAssign]
    idx_to_wx = sortx2d[:, :nAssign]
    rvecs, flags = compute_rvec(vecs, words[idx_to_wx.T[0]])

    if nannots is None:
        nannots = rng.randint(nvecs)
    offset_list = [0] + sorted(rng.choice(nvecs, nannots - 1)) + [nvecs]
    # nfeat_list = np.diff(offset_list)
    data = {
        'idx_to_dist': idx_to_dist,
        'idx_to_wx': idx_to_wx,
        'rvecs': rvecs,
        'vecs': vecs,
        'words': words,
        'flags': flags,
        'offset_list': offset_list,
    }

    return data


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.smk.smk_funcs
        python -m wbia.algo.smk.smk_funcs --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
