from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import zip
import dtool
import utool as ut
import vtool as vt
import six
import pyflann
import numpy as np
# import cv2
from ibeis.control.controller_inject import register_preprocs, register_subprops
(print, rrr, profile) = ut.inject2(__name__, '[new_annots]')


derived_attribute = register_preprocs['annot']
register_subprop = register_subprops['annot']
# dtool.Config.register_func = derived_attribute


@ut.memoize
def testdata_vocab():
    # from ibeis.new_annots import *  # NOQA
    import ibeis
    ibs, aid_list = ibeis.testdata_aids('testdb1')
    depc = ibs.depc_annot
    fid_list = depc.get_rowids('feat', aid_list)
    config = VocabConfig()
    vocab = compute_vocab(depc, fid_list, config)
    return ibs, aid_list, vocab


@six.add_metaclass(ut.ReloadingMetaclass)
class VisualVocab(ut.NiceRepr):

    def __init__(self, words=None):
        self.wx2_word = words
        self.wordflann = pyflann.FLANN()
        self.flann_params = vt.get_flann_params(random_seed=42)

    def reindex(self, verbose=True):
        num_vecs = len(self.wx2_word)
        if verbose:
            print('[nnindex] ...building kdtree over %d points (this may take a sec).' % num_vecs)
            tt = ut.tic(msg='Building vocab index')
        if num_vecs == 0:
            print('WARNING: CANNOT BUILD FLANN INDEX OVER 0 POINTS. THIS MAY BE A SIGN OF A DEEPER ISSUE')
        else:
            self.wordflann.build_index(self.wx2_word, **self.flann_params)
        if verbose:
            ut.toc(tt)

    def nn_index(self, idx2_vec, nAssign):
        """
            >>> idx2_vec = depc.d.get_feat_vecs(aid_list)[0]
            >>> self = vocab
            >>> nAssign = 1
        """
        # Assign each vector to the nearest visual words
        assert nAssign > 0, 'cannot assign to 0 neighbors'
        try:
            _idx2_wx, _idx2_wdist = self.wordflann.nn_index(idx2_vec, nAssign)
        except pyflann.FLANNException as ex:
            ut.printex(ex, 'probably misread the cached flann_fpath=%r' % (self.wordflann.flann_fpath,))
            raise
        else:
            _idx2_wx = vt.atleast_nd(_idx2_wx, 2)
            _idx2_wdist = vt.atleast_nd(_idx2_wdist, 2)
            return _idx2_wx, _idx2_wdist

    def assign_to_words(self, idx2_vec, nAssign, massign_alpha=1.2,
                        massign_sigma=80.0, massign_equal_weights=False):
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
            >>> idx2_vec = depc.d.get_feat_vecs(aid_list)[0][0::300]
            >>> idx2_vec = np.vstack((idx2_vec, self.wx2_word[0]))
            >>> nAssign = 2
            >>> massign_equal_weights = False
            >>> massign_alpha = 1.2
            >>> massign_sigma = 80.0
            >>> nAssign = 2
            >>> idx2_wxs, idx2_maws = self.assign_to_words(idx2_vec, nAssign)
            >>> print('idx2_maws = %s' % (ut.repr2(idx2_wxs, precision=2),))
            >>> print('idx2_wxs = %s' % (ut.repr2(idx2_maws, precision=2),))
        """
        if ut.VERBOSE:
            print('[smk_index.assign] +--- Start Assign vecs to words.')
            print('[smk_index.assign] * nAssign=%r' % nAssign)
        if not ut.QUIET:
            print('[smk_index.assign] assign_to_words_. len(idx2_vec) = %r' % len(idx2_vec))
        _idx2_wx, _idx2_wdist = self.nn_index(idx2_vec, nAssign)
        if nAssign > 1:
            idx2_wxs, idx2_maws = self._compute_multiassign_weights(
                _idx2_wx, _idx2_wdist, massign_alpha, massign_sigma, massign_equal_weights)
        else:
            idx2_wxs = _idx2_wx.tolist()
            idx2_maws = [[1.0]] * len(idx2_wxs)
        return idx2_wxs, idx2_maws

    def invert_assignment(self, idx2_wxs, idx2_maws, *other_idx2_prop):
        """
        Inverts assignment of vectors to words into words to vectors.

        Example:
            >>> idx2_idx = np.arange(len(idx2_wxs))
            >>> other_idx2_prop = (idx2_idx,)
            >>> wx2_idxs, wx2_maws, wx2_idxs_ = self.invert_assignment(idx2_wxs, idx2_maws, idx2_idx)
            >>> assert ut.dict_str(wx2_idxs) == ut.dict_str(wx2_idxs_)
        """
        # Invert mapping -- Group by word indexes
        idx2_nAssign = [len(wxs) for wxs in idx2_wxs]
        jagged_idxs = [[idx] * num for idx, num in enumerate(idx2_nAssign)]
        wx_keys, groupxs = vt.jagged_group(idx2_wxs)
        idxs_list = vt.apply_jagged_grouping(jagged_idxs, groupxs)
        wx2_idxs = dict(zip(wx_keys, idxs_list))
        maws_list = vt.apply_jagged_grouping(idx2_maws, groupxs)
        wx2_maws = dict(zip(wx_keys, maws_list))

        other_wx2_prop = []
        for idx2_prop in other_idx2_prop:
            # Props are assumed to be non-jagged, so make them jagged
            jagged_prop = [[prop] * num for prop, num in zip(idx2_prop, idx2_nAssign)]
            prop_list = vt.apply_jagged_grouping(jagged_prop, groupxs)
            wx2_prop = dict(zip(wx_keys, prop_list))
            other_wx2_prop.append(wx2_prop)
        if ut.VERBOSE:
            print('[smk_index.assign] L___ End Assign vecs to words.')
        assignment = (wx2_idxs, wx2_maws) + tuple(other_wx2_prop)
        return assignment

    @staticmethod
    def _compute_multiassign_weights(_idx2_wx, _idx2_wdist, massign_alpha=1.2,
                                     massign_sigma=80.0,
                                     massign_equal_weights=False):
        """
        Multi Assignment Weight Filtering from Improving Bag of Features

        Args:
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

        Example:
            >>> massign_alpha = 1.2
            >>> massign_sigma = 80.0
            >>> massign_equal_weights = False

        Notes:
            sigma values from \cite{philbin_lost08}
            (70 ** 2) ~= 5000, (80 ** 2) ~= 6250, (86 ** 2) ~= 7500,
        """
        if not ut.QUIET:
            print('[smk_index.assign] compute_multiassign_weights_')
        if _idx2_wx.shape[1] <= 1:
            idx2_wxs = _idx2_wx.tolist()
            idx2_maws = [[1.0]] * len(idx2_wxs)
        else:
            # Valid word assignments are beyond fraction of distance to the nearest word
            massign_thresh = _idx2_wdist.T[0:1].T.copy()
            # HACK: If the nearest word has distance 0 then this threshold is too hard
            # so we should use the distance to the second nearest word.
            EXACT_MATCH_HACK = True
            if EXACT_MATCH_HACK:
                flag_too_close = (massign_thresh == 0)
                massign_thresh[flag_too_close] = _idx2_wdist.T[1:2].T[flag_too_close]
            # Compute the threshold fraction
            epsilon = .001
            np.add(epsilon, massign_thresh, out=massign_thresh)
            np.multiply(massign_alpha, massign_thresh, out=massign_thresh)
            # Mark assignments as invalid if they are too far away from the nearest assignment
            invalid = np.greater_equal(_idx2_wdist, massign_thresh)
            if ut.VERBOSE:
                nInvalid = (invalid.size - invalid.sum(), invalid.size)
                print('[maw] + massign_alpha = %r' % (massign_alpha,))
                print('[maw] + massign_sigma = %r' % (massign_sigma,))
                print('[maw] + massign_equal_weights = %r' % (massign_equal_weights,))
                print('[maw] * Marked %d/%d assignments as invalid' % nInvalid)

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
                gauss_numer = np.negative(_idx2_wdist.astype(np.float64))
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

    def __nice__(self):
        return ' nW=%r' % (ut.safelen(self.wx2_word))

    def on_load(nnindexer, depc):
        pass

    def on_save(nnindexer, depc, fpath):
        pass

    def __getstate__(self):
        # TODO: Figure out how to make these play nice with the depcache
        state = self.__dict__
        del state['wordflann']
        return state

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)


@six.add_metaclass(ut.ReloadingMetaclass)
class InvertedIndex(ut.NiceRepr):
    def __init__(self):
        pass


class VocabConfig(dtool.Config):
    """
    Example:
        >>> from ibeis.core_annots import *  # NOQA
        >>> cfg = VocabConfig()
        >>> result = str(cfg)
        >>> print(result)
    """
    _param_info_list = [
        ut.ParamInfo('algorithm', 'kdtree', 'alg'),
        ut.ParamInfo('random_seed', 42, 'seed'),
        ut.ParamInfo('num_words', 1000, 'seed'),
        ut.ParamInfo('version', 1),
        # max iters
        # flann params
        # random seed
    ]
    _sub_config_list = [
    ]


@derived_attribute(
    tablename='vocab', parents=['feat*'],
    colnames=['words'], coltypes=[VisualVocab],
    configclass=VocabConfig,
    chunksize=1, fname='visual_vocab',
    # func_is_single=True,
    # _internal_parent_ids=False,  # Give the function nicer ids to work with
    # _internal_parent_ids=True,  # Give the function nicer ids to work with
)
def compute_vocab(depc, fid_list, config):
    r"""
    Args:
        depc (dtool.DependencyCache):
        fids_list (list):
        config (dtool.Config):

    CommandLine:
        python -m ibeis.core_annots --exec-compute_neighbor_index --show
        python -m ibeis.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=neighbor_index

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.new_annots import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> fid_list = depc.get_rowids('feat', aid_list)
        >>> config = VocabConfig()
        >>> vocab, train_vecs = ut.exec_func_src(compute_vocab, key_list=['vocab', 'train_vecs'])
        >>> idx2_vec = depc.d.get_feat_vecs(aid_list)[0]
        >>> self = vocab
        >>> ut.quit_if_noshow()
        >>> data = train_vecs
        >>> centroids = vocab.wx2_word
        >>> import plottool as pt
        >>> vt.plot_centroids(data, centroids, num_pca_dims=2)
        >>> ut.show_if_requested()
        >>> #config = ibs.depc_annot['vocab'].configclass()
        >>>
    """
    print('[IBEIS] COMPUTE_VOCAB:')
    vecs_list = depc.get_native('feat', fid_list, 'vecs')
    train_vecs = np.vstack(vecs_list)
    num_words = config['num_words']
    max_iters = 100
    print('[smk_index] Train Vocab(nWords=%d) using %d annots and %d descriptors' %
          (num_words, len(fid_list), len(train_vecs)))
    flann_params = vt.get_flann_params(random_seed=42)
    kwds = dict(
        max_iters=max_iters,
        flann_params=flann_params
    )
    words = vt.akmeans(train_vecs, num_words, **kwds)
    vocab = VisualVocab(words)
    vocab.reindex()
    return vocab


@six.add_metaclass(ut.ReloadingMetaclass)
class StackedFeatures(ut.NiceRepr):
    def __init__(fstack, ibs, aid_list, config=None):
        ax2_vecs = ibs.depc_annot.d.get_feat_vecs(aid_list, config=config)
        fstack.config = config
        fstack.ibs = ibs
        fstack.ax2_aid = aid_list
        fstack.ax2_nFeat = [len(vecs) for vecs in ax2_vecs]
        fstack.idx2_fxs = ut.flatten([list(range(num)) for num in fstack.ax2_nFeat])
        fstack.idx2_axs = ut.flatten([[ax] * num for ax, num in enumerate(fstack.ax2_nFeat)])
        fstack.idx2_vec = np.vstack(ax2_vecs)
        #fstack.idx2_fxs = vt.atleast_nd(fstack.idx2_fxs, 2)
        #fstack.idx2_axs = vt.atleast_nd(fstack.idx2_axs, 2)
        fstack.num_feat = sum(fstack.ax2_nFeat)

    def __nice__(fstack):
        return ' nA=%r nF=%r' % (ut.safelen(fstack.ax2_aid), fstack.num_feat)

    def inverted_assignment(fstack, vocab, nAssign=1):
        # do work
        idx2_wxs, idx2_maws = vocab.assign_to_words(fstack.idx2_vec, nAssign)
        # Pack into structure
        forward_assign = (idx2_wxs, idx2_maws, fstack.idx2_fxs, fstack.idx2_axs)
        invert_assign = vocab.invert_assignment(*forward_assign)
        (wx2_idxs, wx2_maws, wx2_fxs, wx2_axs) = invert_assign
        invassign = InvertedStackAssignment(fstack, vocab, wx2_idxs, wx2_maws, wx2_fxs, wx2_axs)
        return invassign


@six.add_metaclass(ut.ReloadingMetaclass)
class InvertedStackAssignment(ut.NiceRepr):
    def __init__(invassign, fstack, vocab, wx2_idxs, wx2_maws, wx2_fxs, wx2_axs):
        invassign.fstack = fstack
        invassign.vocab = vocab
        invassign.wx2_idxs = wx2_idxs
        invassign.wx2_maws = wx2_maws
        invassign.wx2_fxs = wx2_fxs
        invassign.wx2_axs = wx2_axs
        invassign.wx2_num = ut.map_dict_vals(len, invassign.wx2_axs)
        invassign.wx_list = sorted(invassign.wx2_num.keys())
        invassign.num_list = ut.take(invassign.wx2_num, invassign.wx_list)
        invassign.perword_stats = ut.get_stats(invassign.num_list)

    def __nice__(invassign):
        return ' nW=%r mean=%.2f' % (ut.safelen(invassign.wx2_idxs), invassign.perword_stats['mean'])

    def get_vecs(invassign, wx):
        vecs = invassign.fstack.idx2_vec.take(invassign.wx2_idxs[wx], axis=0)
        return vecs

    def get_patches(invassign, wx):
        ax_list = invassign.wx2_axs[wx]
        fx_list = invassign.wx2_fxs[wx]
        config = invassign.fstack.config
        ibs = invassign.fstack.ibs

        unique_axs, groupxs = vt.group_indices(ax_list)
        fxs_groups = vt.apply_grouping(fx_list, groupxs)

        unique_aids = ut.take(invassign.fstack.ax2_aid, unique_axs)

        all_kpts_list = ibs.depc.d.get_feat_kpts(unique_aids, config=config)
        sub_kpts_list = vt.ziptake(all_kpts_list, fxs_groups, axis=0)

        chip_list = ibs.depc_annot.d.get_chips_img(unique_aids)
        # convert to approprate colorspace
        #if colorspace is not None:
        #    chip_list = vt.convert_image_list_colorspace(chip_list, colorspace)
        # ut.print_object_size(chip_list, 'chip_list')
        patch_size = 64
        grouped_patches_list = [
            vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
            for chip, kpts in ut.ProgIter(zip(chip_list, sub_kpts_list),
                                          nTotal=len(unique_aids),
                                          lbl='warping patches')
        ]
        # Make it correspond with original fx_list and ax_list
        word_patches = vt.invert_apply_grouping(grouped_patches_list, groupxs)
        return word_patches

    def compute_nonagg_rvecs(invassign, wx, compress=False):
        """
        Driver function for nonagg residual computation

        Args:
            words (ndarray): array of words
            idx2_vec (dict): stacked vectors
            wx_sublist (list): words of interest
            idxs_list (list): list of idxs grouped by wx_sublist

        Returns:
            tuple : (rvecs_list, flags_list)
        """
        # Pick out corresonding lists of residuals and words
        vecs = invassign.get_vecs(wx)
        word = invassign.vocab.wx2_word[wx]
        # Compute nonaggregated normalized residuals
        arr_float = np.subtract(word.astype(np.float), vecs.astype(np.float))
        vt.normalize_rows(arr_float, out=arr_float)
        if compress:
            rvecs_list = np.clip(np.round(arr_float * 255.0), -127, 127).astype(np.int8)
        else:
            rvecs_list = arr_float
        # Extract flags (rvecs_list which are all zeros) and rvecs_list
        error_flags = ~np.any(rvecs_list, axis=1)
        return rvecs_list, error_flags

    def compute_agg_rvecs(invassign, wx):
        """
        Sums and normalizes all rvecs that belong to the same word and the same
        annotation id
        """
        rvecs_list, error_flags = invassign.compute_nonagg_rvecs(wx)
        ax_list = invassign.wx2_axs[wx]
        maw_list = invassign.wx2_maws[wx]
        # group members of each word by aid, we will collapse these groups
        unique_ax, groupxs = vt.group_indices(ax_list)
        # (weighted aggregation with multi-assign-weights)
        grouped_maws = vt.apply_grouping(maw_list, groupxs)
        grouped_rvecs = vt.apply_grouping(rvecs_list, groupxs)
        grouped_flags = vt.apply_grouping(~error_flags, groupxs)

        grouped_rvecs2_ = vt.zipcompress(grouped_rvecs, grouped_flags, axis=0)
        grouped_maws2_ = vt.zipcompress(grouped_maws, grouped_flags)
        is_good = [len(rvecs) > 0 for rvecs in grouped_rvecs2_]
        aggvecs = [aggregate_rvecs(rvecs, maws)[0] for rvecs, maws in zip(grouped_rvecs2_, grouped_maws2_)]
        unique_ax2_ = unique_ax.compress(is_good)
        ax2_aggvec = dict(zip(unique_ax2_, aggvecs))
        # Need to recompute flags for consistency
        # flag is true when aggvec is all zeros
        return ax2_aggvec


def aggregate_rvecs(rvecs, maws, compress=False):
    r"""
    helper for compute_agg_rvecs
    """
    if rvecs.shape[0] == 0:
        rvecs_agg = np.empty((0, rvecs.shape[1]), dtype=np.float)
    if rvecs.shape[0] == 1:
        rvecs_agg = rvecs
    else:
        # Prealloc sum output (do not assign the result of sum)
        arr_float = np.empty((1, rvecs.shape[1]), dtype=np.float)
        out = arr_float[0]
        # Take weighted average of multi-assigned vectors
        total_weight = maws.sum()
        weighted_sum = (maws[:, np.newaxis] * rvecs.astype(np.float)).sum(axis=0, out=out)
        np.divide(weighted_sum, total_weight, out=out)
        vt.normalize_rows(arr_float, out=arr_float)
        if compress:
            rvecs_agg = np.clip(np.round(arr_float * 255.0), -127, 127).astype(np.int8)
        else:
            rvecs_agg = arr_float
    return rvecs_agg


def visualize_vocab_word(ibs, invassign, wx, fnum=None):
    """

    Example:
        >>> from ibeis.new_annots import *  # NOQA
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> ibs, aid_list, vocab = testdata_vocab()
        >>> #aid_list = aid_list[0:1]
        >>> fstack = StackedFeatures(ibs, aid_list)
        >>> nAssign = 2
        >>> invassign = fstack.inverted_assignment(vocab, nAssign)
        >>> sortx = ut.argsort(invassign.num_list)[::-1]
        >>> wx_list = ut.take(invassign.wx_list, sortx)
        >>> wx = wx_list[0]
    """
    import plottool as pt
    pt.qt4ensure()
    vecs = invassign.get_vecs(wx)
    word = invassign.vocab.wx2_word[wx]

    word_patches = invassign.get_patches(wx)
    average_patch = np.mean(word_patches, axis=0)

    average_vec = vecs.mean(axis=0)
    average_vec = word

    word

    with_sift = True
    fnum = 2
    fnum = pt.ensure_fnum(fnum)
    if with_sift:
        patch_img = pt.render_sift_on_patch(average_patch, average_vec)
        #sift_word_patches = [pt.render_sift_on_patch(patch, vec) for patch, vec in ut.ProgIter(list(zip(word_patches, vecs)))]
        #stacked_patches = vt.stack_square_images(word_patches)
        #stacked_patches = vt.stack_square_images(sift_word_patches)
    else:
        patch_img = average_patch
    stacked_patches = vt.stack_square_images(word_patches)
    solidbar = np.zeros((patch_img.shape[0], int(patch_img.shape[1] * .1), 3), dtype=patch_img.dtype)
    border_color = (100, 10, 10)  # bgr, darkblue
    if ut.is_float(solidbar):
        solidbar[:, :, :] = (np.array(border_color) / 255)[None, None]
    else:
        solidbar[:, :, :] = np.array(border_color)[None, None]
    word_img = vt.stack_image_list([patch_img, solidbar, stacked_patches], vert=False, modifysize=True)
    pt.imshow(word_img, fnum=fnum)
    #pt.imshow(patch_img, pnum=(1, 2, 1), fnum=fnum)
    #patch_size = 64
    #half_size = patch_size / 2
    #pt.imshow(stacked_patches, pnum=(1, 2, 2), fnum=fnum)
    pt.iup()


def test_visualize_vocab_interact():
    """
    python -m ibeis.new_annots --exec-test_visualize_vocab_interact --show

    Example:
        >>> from ibeis.new_annots import *  # NOQA
        >>> test_visualize_vocab_interact()
        >>> ut.show_if_requested()
    """
    import plottool as pt
    pt.qt4ensure()
    ibs, aid_list, vocab = testdata_vocab()
    #aid_list = aid_list[0:1]
    fstack = StackedFeatures(ibs, aid_list)
    nAssign = 2
    invassign = fstack.inverted_assignment(vocab, nAssign)
    sortx = ut.argsort(invassign.num_list)[::-1]
    wx_list = ut.take(invassign.wx_list, sortx)
    wx = wx_list[0]
    fnum = 1
    for wx in ut.InteractiveIter(wx_list):
        visualize_vocab_word(ibs, invassign, wx, fnum)


def extract_patches(ibs, aid_list, fxs_list=None, patch_size=None, colorspace=None):
    """

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> ut.show_if_requested()
    """
    depc = ibs.depc
    kpts_list = depc.d.get_feat_kpts(aid_list)
    if fxs_list is None:
        fxs_list = [slice(None)] * len(kpts_list)
    kpts_list_ = ut.ziptake(kpts_list, fxs_list)
    chip_list = depc.d.get_chips_img(aid_list)
    # convert to approprate colorspace
    if colorspace is not None:
        chip_list = vt.convert_image_list_colorspace(chip_list, colorspace)
    # ut.print_object_size(chip_list, 'chip_list')
    patch_size = 64

    patches_list = [
        vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
        for chip, kpts in ut.ProgIter(zip(chip_list, kpts_list_),
                                      nTotal=len(aid_list),
                                      lbl='warping patches')
    ]
    return patches_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.new_annots
        python -m ibeis.new_annots --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
