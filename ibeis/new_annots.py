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
    from ibeis.new_annots import *  # NOQA
    import ibeis
    ibs, aid_list = ibeis.testdata_aids('testdb1')
    depc = ibs.depc_annot
    fid_list = depc.get_rowids('feat', aid_list)
    config = VocabConfig()
    vocab = compute_vocab(depc, fid_list, config)
    return vocab


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

    def invert_assignment(self, idx2_wxs, *other_idx2_prop):
        """
        Inverts assignment of vectors to words into words to vectors.

        Example:
            >>> wx2_idxs, wx2_maws = self.invert_assignment(idx2_wxs, idx2_maws)
        """
        # Invert mapping -- Group by word indexes
        jagged_idxs = ([idx] * len(wxs)for idx, wxs in enumerate(idx2_wxs))
        wx_keys, groupxs = vt.jagged_group(idx2_wxs)
        idxs_list = vt.apply_jagged_grouping(jagged_idxs, groupxs)
        wx2_idxs = dict(zip(wx_keys, idxs_list))
        other_wx2_prop = []
        for idx2_prop in other_idx2_prop:
            prop_list = vt.apply_jagged_grouping(idx2_prop, groupxs)
            wx2_prop = dict(zip(wx_keys, prop_list))
            other_wx2_prop.append(wx2_prop)
        if ut.VERBOSE:
            print('[smk_index.assign] L___ End Assign vecs to words.')
        assignment = (wx2_idxs,) + tuple(other_wx2_prop)
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


#@derived_attribute(
#    tablename='vocab', parents=['feat*'],
#    colnames=['words'], coltypes=[VisualVocab],
#    configclass=VocabConfig,
#    chunksize=1, fname='visual_vocab',
#    single=True,
#    # _internal_parent_ids=False,  # Give the function nicer ids to work with
#    _internal_parent_ids=True,  # Give the function nicer ids to work with
#)
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


def visualize_vocab(depc, words, aid_list):
    """

    Example:
        >>> vocab = testdata_vocab()
        >>> idx2_vec = depc.d.get_feat_vecs(aid_list[0:1])[0]
        >>> idx2_patch = extract_patches(ibs, aid_list[0:1])[0]
        >>> idx2_aid = [[aid_list[0]]] * len(idx2_vec)
        >>> nAssign = 1
        >>> idx2_wxs, idx2_maws = vocab.assign_to_words(idx2_vec, nAssign)
        >>> wx2_idxs, wx2_maws, wx2_aid = vocab.invert_assignment(idx2_wxs, idx2_maws, idx2_aid)
        >>> # convert idxs to aids and fxs
        >>> aid = aid_list[0]
        >>> wx2_assigned = {wx: [idx, aid for idx in idxs] for wx, idxs in wx2_idxs.items()}
    """
    #patches_list = extract_patches()
    #vecs_list = depc.d.get_feat_vecs(aid_list)
    #flann_params = {}
    #flann = vt.build_flann_index(words, flann_params)

    #wx2_assigned = ut.ddict(list)
    #for aid, vecs in zip(aid_list, vecs_list):
    #    vx2_wx, vx2_dist = flann.nn_index(vecs)
    #    vx2_wx = vt.atleast_nd(vx2_wx, 2)
    #    vx_list = np.arange(len(vx2_wx))
    #    for vx, wxs in zip(vx_list, vx2_wx):
    #        for wx in wxs:
    #            wx2_assigned[wx].append((aid, vx))

    aid2_ax = ut.make_index_lookup(aid_list)
    sortx = ut.argsort(list(map(len, wx2_assigned.values())))
    wxs = list(wx2_assigned.keys())

    import plottool as pt
    pt.qt4ensure()
    wx_list = ut.take(sortx, wxs)

    for wx in ut.InteractiveIter(wx_list):
        assigned = wx2_assigned[wx]
        aids = ut.take_column(assigned, 0)
        fxs = ut.take_column(assigned, 1)

        word_patches = []
        for aid, fxs in ut.group_items(fxs, aids).items():
            ax = aid2_ax[aid]
            word_patches.extend(ut.take(patches_list[ax], fxs))
            pass

        stacked_img = vt.stack_square_images(word_patches)
        pt.imshow(stacked_img)
        pt.iup()


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
