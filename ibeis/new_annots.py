# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, map
import six
import dtool
import utool as ut
import vtool as vt
import pyflann
import numpy as np
from ibeis.control.controller_inject import register_preprocs, register_subprops
(print, rrr, profile) = ut.inject2(__name__, '[new_annots]')


derived_attribute = register_preprocs['annot']
register_subprop = register_subprops['annot']
# dtool.Config.register_func = derived_attribute


@ut.memoize
def testdata_vocab(defaultdb='testdb1', **kwargs):
    """
    >>> from ibeis.new_annots import *  # NOQA
    """
    import ibeis
    ibs, aid_list = ibeis.testdata_aids(defaultdb=defaultdb)
    depc = ibs.depc_annot
    #fid_list = depc.get_rowids('feat', aid_list)
    config = VocabConfig(**kwargs)
    #vocab = compute_vocab(depc, fid_list, config)
    vocab = depc.get('vocab', [aid_list], 'words', config=config)[0]
    #vocab_fpath = depc.get('vocab', [aid_list], 'words', config=config, read_extern=False)[0]
    return ibs, aid_list, vocab


def testdata_inva(*args, **kwargs):
    ibs, aid_list, vocab = testdata_vocab(*args, **kwargs)
    fstack = ForwardIndex(ibs, aid_list)
    inva = fstack.inverted_assignment(vocab, nAssign=3)
    return ibs, aid_list, inva


def testdata_smk(*args, **kwargs):
    """
    >>> from ibeis.new_annots import *  # NOQA
    >>> kwargs = {}
    """
    import ibeis
    import sklearn
    # import sklearn.model_selection
    import sklearn.cross_validation
    ibs, aid_list = ibeis.testdata_aids(defaultdb='PZ_MTEST')
    nid_list = np.array(ibs.annots(aid_list).nids)
    rng = ut.ensure_rng(0)
    xvalkw = dict(n_folds=4, shuffle=False, random_state=rng)
    # skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
    # train_idx, test_idx = list(skf.split(nid_list, nid_list))[0]
    skf = sklearn.cross_validation.StratifiedKFold(nid_list, **xvalkw)
    train_idx, test_idx = six.next(iter(skf))
    daids = ut.take(aid_list, train_idx)
    qaids = ut.take(aid_list, test_idx)

    kwargs['num_words'] = kwargs.get('num_words', 10000)
    config = VocabConfig(**kwargs)
    vocab = ibs.depc_annot.get('vocab', [daids], 'words', config=config)[0]

    query_fstack = ForwardIndex(ibs, qaids, name='query')
    data_fstack = ForwardIndex(ibs, daids, name='data')
    qinva = query_fstack.inverted_assignment(vocab, nAssign=3)
    dinva = data_fstack.inverted_assignment(vocab, nAssign=3)
    smk = SMK(qinva, dinva)

    #qreq_ = ibs.new_query_request(qaids, daids, cfgdict={'pipeline_root': 'smk', 'proot': 'smk'})
    qreq_ = ibs.new_query_request(qaids, daids, cfgdict={})
    return ibs, smk, qreq_


@ut.reloadable_class
class VisualVocab(ut.NiceRepr):
    """
    Class that maintains a list of visual words (cluster centers)
    Also maintains a nearest neighbor index structure for finding words.
    This class is build using the depcache
    """

    def __init__(self, words=None):
        self.wx2_word = words
        self.wordflann = None
        self.flann_params = vt.get_flann_params(random_seed=42)

    def __len__(self):
        return len(self.wx2_word)

    def __getstate__(self):
        # TODO: Figure out how to make these play nice with the depcache
        state = self.__dict__.copy()
        del state['wordflann']
        # Make a special wordflann pickle
        # THIS WILL NOT WORK ON WINDOWS
        import tempfile
        assert not ut.WIN32, 'Need to fix this on WIN32. Cannot write to temp file'
        temp = tempfile.NamedTemporaryFile(delete=True)
        try:
            self.wordflann.save_index(temp.name)
            wordindex_bytes = temp.read()
            #print('wordindex_bytes = %r' % (len(wordindex_bytes),))
            state['wordindex_bytes'] = wordindex_bytes
        except Exception:
            raise
        finally:
            temp.close()

        return state

    def __setstate__(self, state):
        wordindex_bytes = state.pop('wordindex_bytes')
        self.__dict__.update(state)
        self.wordflann = pyflann.FLANN()
        import tempfile
        assert not ut.WIN32, 'Need to fix this on WIN32. Cannot write to temp file'
        temp = tempfile.NamedTemporaryFile(delete=True)
        try:
            temp.write(wordindex_bytes)
            temp.file.flush()
            self.wordflann.load_index(temp.name, self.wx2_word)
        except Exception:
            raise
        finally:
            temp.close()

    def reindex(self, verbose=True):
        num_vecs = len(self.wx2_word)
        if self.wordflann is None:
            self.wordflann = pyflann.FLANN()
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
            idx2_vec = idx2_vec.astype(self.wordflann._FLANN__curindex_data.dtype)
            _idx2_wx, _idx2_wdist = self.wordflann.nn_index(idx2_vec, nAssign)
        except pyflann.FLANNException as ex:
            ut.printex(ex, 'probably misread the cached flann_fpath=%r' % (getattr(self.wordflann, 'flann_fpath', None),))
            import utool
            utool.embed()
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

    def invert_assignment(self, idx2_wxs, idx2_maws):
        """
        Inverts assignment of vectors to words into words to vectors.

        Example:
            >>> idx2_idx = np.arange(len(idx2_wxs))
            >>> other_idx2_prop = (idx2_idx,)
            >>> wx2_idxs, wx2_maws = self.invert_assignment(idx2_wxs, idx2_maws)
        """
        # Invert mapping -- Group by word indexes
        idx2_nAssign = [len(wxs) for wxs in idx2_wxs]
        jagged_idxs = [[idx] * num for idx, num in enumerate(idx2_nAssign)]
        wx_keys, groupxs = vt.jagged_group(idx2_wxs)
        idxs_list = vt.apply_jagged_grouping(jagged_idxs, groupxs)
        wx2_idxs = dict(zip(wx_keys, idxs_list))
        maws_list = vt.apply_jagged_grouping(idx2_maws, groupxs)
        wx2_maws = dict(zip(wx_keys, maws_list))
        if ut.VERBOSE:
            print('[smk_index.assign] L___ End Assign vecs to words.')
        return (wx2_idxs, wx2_maws)

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
        return 'nW=%r' % (ut.safelen(self.wx2_word))


class VocabConfig(dtool.Config):
    """
    Example:
        >>> from ibeis.core_annots import *  # NOQA
        >>> cfg = VocabConfig()
        >>> result = str(cfg)
        >>> print(result)
    """
    _param_info_list = [
        ut.ParamInfo('algorithm', 'minibatch', 'alg'),
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
    # Let the depcache know that this function Can only compute a single value
    # at a time (ie it is not vectorized)
    vectorized=False,
    # _internal_parent_ids=False,  # Give the function nicer ids to work with
    # _internal_parent_ids=True,  # Give the function nicer ids to work with
)
def compute_vocab(depc, fid_list, config):
    r"""
    Depcache method for computing a new visual vocab

    Args:
        depc (dtool.DependencyCache):
        fids_list (list):
        config (dtool.Config):

    CommandLine:
        python -m ibeis.core_annots --exec-compute_neighbor_index --show
        python -m ibeis.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=neighbor_index

        python -m ibeis.new_annots --exec-compute_vocab:0

        # FIXME make util_tests register
        python -m ibeis.new_annots compute_vocab:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.new_annots import *  # NOQA
        >>> # Test depcache access
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> input_tuple = [aid_list]
        >>> rowid_kw = {}
        >>> tablename = 'vocab'
        >>> vocabid_list = depc.get_rowids(tablename, input_tuple, **rowid_kw)
        >>> vocab = depc.get(tablename, input_tuple, 'words')[0]
        >>> assert vocab.wordflann is not None
        >>> assert vocab.wordflann._FLANN__curindex_data is not None
        >>> assert vocab.wordflann._FLANN__curindex_data is vocab.wx2_word

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

    if config['algorithm'] == 'kdtree':
        words = vt.akmeans(train_vecs, num_words, **kwds)
    elif config['algorithm'] == 'minibatch':
        print('Using minibatch kmeans')
        import sklearn.cluster
        rng = np.random.RandomState(config['random_seed'])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clusterer = sklearn.cluster.MiniBatchKMeans(
                num_words, random_state=rng, verbose=5)
            clusterer.fit(train_vecs)
        words = clusterer.cluster_centers_
    if False:
        flann_params['checks'] = 64
        flann_params['trees'] = 4
        num_words = 128
        centroids = vt.initialize_centroids(num_words, train_vecs, 'akmeans++')
        words, hist = vt.akmeans_iterations(train_vecs, centroids, max_iters=1000, monitor=True,
                                            flann_params=flann_params)
        # words, hist = vt.akmeans_iterations(train_vecs, centroids, max_iters=max_iters, monitor=True,
        #                                     flann_params=flann_params)
        import plottool as pt
        ut.qt4ensure()
        pt.multi_plot(hist['epoch_num'], [hist['loss']], fnum=2, pnum=(1, 2, 1), label_list=['loss'])
        pt.multi_plot(hist['epoch_num'], [hist['ave_unchanged']], fnum=2, pnum=(1, 2, 2), label_list=['unchanged'])

    vocab = VisualVocab(words)
    vocab.reindex()
    return (vocab,)


@ut.reloadable_class
class ForwardIndex(ut.NiceRepr):
    """
    A Forward Index of Stacked Features

    A stack of features from multiple annotations.
    Contains a method to create an inverted index given a vocabulary.
    """
    def __init__(fstack, ibs, aid_list, config=None, name=None):
        ax2_vecs = ibs.depc_annot.d.get_feat_vecs(aid_list, config=config)
        fstack.config = config
        fstack.name = name
        fstack.ibs = ibs
        #-- Stack 1 --
        fstack.ax2_aid = aid_list
        fstack.ax2_nFeat = [len(vecs) for vecs in ax2_vecs]
        #-- Stack 2 --
        fstack.idx2_fx = np.array(ut.flatten([list(range(num)) for num in fstack.ax2_nFeat]))
        fstack.idx2_ax = np.array(ut.flatten([[ax] * num for ax, num in enumerate(fstack.ax2_nFeat)]))
        fstack.idx2_vec = np.vstack(ax2_vecs)
        #fstack.idx2_fx = vt.atleast_nd(fstack.idx2_fx, 2)
        #fstack.idx2_ax = vt.atleast_nd(fstack.idx2_ax, 2)
        fstack.num_feat = sum(fstack.ax2_nFeat)

    def __nice__(fstack):
        name = '' if fstack.name is None else fstack.name
        return '%s nA=%r nF=%r' % (name, ut.safelen(fstack.ax2_aid), fstack.num_feat)

    def inverted_assignment(fstack, vocab, nAssign=1):
        idx2_wxs, idx2_maws = vocab.assign_to_words(fstack.idx2_vec, nAssign)
        wx2_idxs, wx2_maws = vocab.invert_assignment(idx2_wxs, idx2_maws)
        inva = InvertedIndex(fstack, vocab, wx2_idxs, wx2_maws)
        return inva


@ut.reloadable_class
class InvertedIndex(ut.NiceRepr):
    """
    Maintains an inverted index of chip descriptors that are multi-assigned to
    a set of vocabulary words.

    This stack represents the database.
    It prorcesses the important information like

    * vocab - this is the quantizer

    Each word is has an inverted index to a list of:
        (annotation index, feature index, multi-assignment weight)
    """
    def __init__(inva, fstack, vocab, wx2_idxs, wx2_maws):
        inva.fstack = fstack
        inva.vocab = vocab
        # Corresponding arrays
        inva.wx2_idxs = wx2_idxs
        inva.wx2_maws = wx2_maws
        inva.wx2_num = ut.map_dict_vals(len, inva.wx2_idxs)
        #
        inva.wx_list = sorted(inva.wx2_num.keys())
        inva.num_list = ut.take(inva.wx2_num, inva.wx_list)

        inva.perword_stats = ut.get_stats(inva.num_list)

        # Extra stuff
        inva._word_patches = {}

    def wx2_fxs(inva, wx):
        return inva.fstack.idx2_fx.take(inva.wx2_idxs[wx], axis=0)

    def wx2_axs(inva, wx):
        return inva.fstack.idx2_ax.take(inva.wx2_idxs[wx], axis=0)

    def __nice__(inva):
        fstack = inva.fstack
        name = '' if fstack.name is None else fstack.name
        return '%s nW=%r mean=%.2f' % (name, ut.safelen(inva.wx2_idxs), inva.perword_stats['mean'])

    def get_patches(inva, wx, verbose=True):
        """
        Loads the patches assigned to a particular word in this stack
        """
        ax_list = inva.wx2_axs(wx)
        fx_list = inva.wx2_fxs(wx)
        config = inva.fstack.config
        ibs = inva.fstack.ibs

        # Group annotations with more than one assignment to this word, so we
        # only have to load a chip at most once
        unique_axs, groupxs = vt.group_indices(ax_list)
        fxs_groups = vt.apply_grouping(fx_list, groupxs)

        unique_aids = ut.take(inva.fstack.ax2_aid, unique_axs)

        all_kpts_list = ibs.depc.d.get_feat_kpts(unique_aids, config=config)
        sub_kpts_list = vt.ziptake(all_kpts_list, fxs_groups, axis=0)

        chip_list = ibs.depc_annot.d.get_chips_img(unique_aids)
        # convert to approprate colorspace
        #if colorspace is not None:
        #    chip_list = vt.convert_image_list_colorspace(chip_list, colorspace)
        # ut.print_object_size(chip_list, 'chip_list')
        patch_size = 64
        _prog = ut.ProgPartial(enabled=verbose, lbl='warping patches', bs=True)
        grouped_patches_list = [
            vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
            #vt.get_warped_patches(chip, kpts, patch_size=patch_size, use_cpp=True)[0]
            for chip, kpts in _prog(zip(chip_list, sub_kpts_list),
                                    nTotal=len(unique_aids))
        ]
        # Make it correspond with original fx_list and ax_list
        word_patches = vt.invert_apply_grouping(grouped_patches_list, groupxs)
        return word_patches

    def get_word_patch(inva, wx):
        if wx not in inva._word_patches:
            assigned_patches = inva.get_patches(wx, verbose=False)
            #print('assigned_patches = %r' % (len(assigned_patches),))
            average_patch = np.mean(assigned_patches, axis=0)
            average_patch = average_patch.astype(np.float)
            inva._word_patches[wx] = average_patch
        return inva._word_patches[wx]

    def compute_idf(inva):
        """
        Use idf (inverse document frequency) to weight each word.

        References:
            https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2

        Example:
            >>> # ENABLE_LATER
            >>> from ibeis.new_annots import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva(num_words=256)
            >>> wx2_idf = inva.compute_word_idf()
            >>> idf_list = np.array(wx2_idf.values())
            >>> assert np.all(idf_list >= 0)
            >>> prob_list = (1 / np.exp(idf_list))
            >>> assert np.all(prob_list >= 0)
            >>> assert np.all(prob_list <= 1)
        """
        num_docs_total = len(inva.fstack.ax2_aid)
        # idf denominator (the num of docs containing a word for each word)
        # The max(maws) to denote the probab that this word indexes an annot
        wx_to_num_docs = np.array([
            sum([maws.max() for maws in vt.apply_grouping(
                inva.wx2_maws[wx],
                vt.group_indices(inva.wx2_axs(wx))[1])])
            for wx in inva.wx_list
        ])
        # Typically for IDF, 1 is added to the denom to prevent divide by 0
        # We add epsilon to numer and denom to ensure recep is a probability
        idf_list = np.log(np.divide(num_docs_total + 1, wx_to_num_docs + 1))
        wx2_idf = dict(zip(inva.wx_list, idf_list))
        return wx2_idf

    def get_vecs(inva, wx):
        """
        Get raw vectors assigned to a word
        """
        vecs = inva.fstack.idx2_vec.take(inva.wx2_idxs[wx], axis=0)
        return vecs

    def compute_rvecs(inva, wx, asint=False):
        """
        Driver function for non-aggregated residual vectors to a specific word.

        Notes:
            rvec = phi(x)
            phi(x) = x - q(x)
            This function returns `[phi(x) for x in X_c]`
            Where c is a word `wx`
            IE: all residual vectors for the descriptors assigned to word c

        Returns:
            tuple : (rvecs_list, flags_list)

        Example:
            >>> # ENABLE_LATER
            >>> from ibeis.new_annots import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva(num_words=256)
            >>> wx = 1
            >>> asint = False
            >>> rvecs_list, error_flags = inva.compute_rvecs(1)
        """
        # Pick out corresonding lists of residuals and words
        vecs = inva.get_vecs(wx)
        word = inva.vocab.wx2_word[wx]
        return compute_rvec(vecs, word)

    def get_grouped_rvecs(inva, wx):
        """
        # Get all residual vectors assigned to a word grouped by annotation
        """
        rvecs_list, error_flags = inva.compute_rvecs(wx)
        ax_list = inva.wx2_axs(wx)
        maw_list = inva.wx2_maws[wx]
        # group residual vectors by annotation
        unique_ax, groupxs = vt.group_indices(ax_list)
        # (weighted aggregation with multi-assign-weights)
        grouped_maws  = vt.apply_grouping(maw_list, groupxs)
        grouped_rvecs = vt.apply_grouping(rvecs_list, groupxs)
        inner_flags = vt.apply_grouping(~error_flags, groupxs)

        # Remove vectors with errors
        grouped_rvecs2_ = vt.zipcompress(grouped_rvecs, inner_flags, axis=0)
        grouped_maws2_  = vt.zipcompress(grouped_maws, inner_flags)

        outer_flags = [len(rvecs) > 0 for rvecs in grouped_rvecs2_]
        grouped_rvecs3_ = ut.compress(grouped_rvecs2_, outer_flags)
        grouped_maws3_ = ut.compress(grouped_maws2_, outer_flags)
        unique_ax3_ = ut.compress(unique_ax, outer_flags)
        return unique_ax3_, grouped_rvecs3_, grouped_maws3_

    def compute_agg_rvecs(inva, wx):
        """
        Sums and normalizes all rvecs that belong to the same word and the same
        annotation id

        Returns:
            ax2_aggvec: A mapping from an annotation to its aggregated vector
                for this word.

        Notes:
            aggvec = Phi
            Phi(X_c) = psi(sum([phi(x) for x in X_c]))
            psi is the identify function currently.
            Phi is esentially a VLAD vector for a specific word.
            Returns Phi vectors wrt each annotation.

        Example:
            >>> # ENABLE_LATER
            >>> from ibeis.new_annots import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva(num_words=256)
            >>> wx = 1
            >>> asint = False
            >>> ax2_aggvec = inva.compute_agg_rvecs(1)
        """
        tup = inva.get_grouped_rvecs(wx)
        unique_ax3_, grouped_rvecs3_, grouped_maws3_ = tup
        ax2_aggvec = {
            ax: aggregate_rvecs(rvecs, maws)
            for ax, rvecs, maws in
            zip(unique_ax3_, grouped_rvecs3_, grouped_maws3_)
        }
        return ax2_aggvec

    def group_annots(inva):
        """
        Group by annotations first and then by words

        inva = smk.qinva
            >>> from ibeis.new_annots import *  # NOQA
            >>> ibs, smk, qreq_ = testdata_smk()
            >>> inva = smk.qinva
        """
        fstack = inva.fstack
        ax2_X = [IndexedAnnot(ax, inva) for ax in range(len(fstack.ax2_aid))]

        wx_list = list(inva.wx2_idxs.keys())

        for wx in wx_list:
            idxs = inva.wx2_idxs[wx]
            maws = inva.wx2_maws[wx]
            axs = fstack.idx2_ax.take(idxs, axis=0)

            unique_axs, groupxs = vt.group_indices(axs)
            grouped_maws = vt.apply_grouping(maws, groupxs)
            grouped_idxs = vt.apply_grouping(idxs, groupxs)

            # precompute here
            ax2_aggvec = inva.compute_agg_rvecs(wx)
            for ax, idxs_, maws_ in zip(unique_axs, grouped_idxs,
                                        grouped_maws):
                X = ax2_X[ax]
                X.wx2_idxs_[wx] = idxs_
                X.wx2_maws_[wx] = maws_
                X.wx2_aggvec[wx] = ax2_aggvec[ax]
        X_list = ax2_X
        return X_list


@ut.reloadable_class
class IndexedAnnot(ut.NiceRepr):
    def __init__(X, ax, inva):
        X.ax = ax
        X.inva = inva
        # only idxs that belong to ax
        X.wx2_idxs_ = {}
        X.wx2_maws_ = {}
        X.wx2_aggvec = {}

    def __nice__(X):
        return 'aid=%r, nW=%r' % (X.aid, len(X.wx2_idxs_),)

    @property
    def aid(X):
        return X.inva.fstack.ax2_aid[X.ax]

    def idxs(X, wx):
        return X.wx2_idxs_[wx]

    def maws(X, wx):
        return X.wx2_maws_[wx]

    def fxs(X, wx):
        fxs = X.inva.fstack.idx2_fx.take(X.idxs(wx), axis=0)
        return fxs

    def Phi(X, wx):
        return X.wx2_aggvec[wx]

    def phis(X, wx):
        vocab = X.inva.vocab
        idxs = X.idxs(wx)
        vecs = X.inva.fstack.idx2_vec.take(idxs, axis=0)
        word = vocab.wx2_word[wx]
        rvecs, flags = compute_rvec(vecs, word)
        return rvecs

    @property
    def words(X):
        return list(X.wx2_aggvec)

    def assert_self(X):
        assert len(X.wx2_idxs_) == len(X.wx2_aggvec)
        for wx in X.wx2_idxs_.keys():
            axs = X.inva.fstack.idx2_ax.take(X.idxs(wx), axis=0)
            assert np.all(axs == X.ax)

            agg_rvec = aggregate_rvecs(X.phis(wx), X.maws(wx))
            assert np.all(agg_rvec == X.Phi(wx))


@ut.reloadable_class
class SMK(ut.NiceRepr):
    """
    Harness class that controls the execution of the SMK algorithm

    K(X, Y) = gamma(X) * gamma(Y) * sum([Mc(Xc, Yc) for c in words])

    """
    def __init__(smk, qinva, dinva):
        smk.qinva = qinva
        smk.dinva = dinva
        smk.config = {
            'alpha': 3,
            'thresh': 0,
            # 'thresh': .5,
        }
        # Choose which version to use
        if True:
            smk.match_score = smk.match_score_agg
        else:
            smk.match_score = smk.match_score_sep

    @property
    @ut.memoize
    def wx2_idf(smk):
        wx2_idf = smk.dinva.compute_idf()
        return wx2_idf

    def weighted_match_score(smk, X, Y, c):
        """
        Just computes the total score of all feature matches
        """
        u = smk.match_score(X, Y, c)
        score = smk.selectivity(u).sum()
        score *= smk.wx2_idf[c]
        return score

    def weighted_matches(smk, X, Y, c):
        """
        Explicitly computes the feature matches that will be scored
        """
        u = smk.match_score(X, Y, c)
        score = smk.selectivity(u).sum()
        score *= smk.wx2_idf[c]
        word_fm = list(ut.product(X.fxs(c), Y.fxs(c)))
        if score.size != len(word_fm):
            # Spread word score according to contriubtion (maw) weight
            contribution = X.maws(c)[:, None].dot(Y.maws(c)[:, None].T)
            contrib_weight = (contribution / contribution.sum())
            word_fs = (contrib_weight * score).ravel()
        else:
            # Scores were computed separately, so dont spread
            word_fs = score.ravel()

        isvalid = word_fs > 0
        word_fs = word_fs.compress(isvalid)
        word_fm = ut.compress(word_fm, isvalid)
        return word_fm, word_fs

    def match_score_agg(smk, X, Y, c):
        u = X.Phi(c).dot(Y.Phi(c).T)
        return u

    def match_score_sep(smk, X, Y, c):
        u = X.phis(c).dot(Y.phis(c).T)
        return u

    def selectivity(smk, u):
        score = np.sign(u) * np.power(np.abs(u), smk.config['alpha'])
        score *= np.greater(u, smk.config['thresh'])
        return score

    def gamma(smk, X):
        r"""
        Computes gamma (self consistency criterion)
        It is a scalar which ensures K(X, X) = 1

        Returns:
            float: sccw self-consistency-criterion weight

        Math:
            gamma(X) = (sum_{c in C} w_c M(X_c, X_c))^{-.5}

            >>> from ibeis.new_annots import *  # NOQA
            >>> ibs, smk, qreq_= testdata_smk()
            >>> X = smk.qinva.group_annots()[0]
            >>> print('X.gamma = %r' % (smk.gamma(X),))
        """
        rawscores = [smk.weighted_match_score(X, X, c) for c in X.words]
        idf_list = np.array(ut.take(smk.wx2_idf, X.words))
        scores = np.array(rawscores) * idf_list
        score = scores.sum()
        sccw = np.reciprocal(np.sqrt(score))
        return sccw

    def execute(smk, qreq_):
        """
        >>> from ibeis.new_annots import *  # NOQA
        >>> ibs, smk, qreq_ = testdata_smk()
        """
        X_list = smk.qinva.group_annots()[0]
        Y_list = smk.dinva.group_annots()[0]

        cm_list = [
            smk.smk_chipmatch(X, Y_list)
            for X in ut.ProgIter(X_list, lbl='smk query')
        ]
        return cm_list

    def smk_chipmatch(smk, X, Y_list, qreq_):
        """
        CommandLine:
            python -m ibeis.new_annots smk_chipmatch --show

        Example:
            >>> # FUTURE_ENABLE
            >>> from ibeis.new_annots import *  # NOQA
            >>> ibs, smk, qreq_ = testdata_smk()
            >>> X = smk.qinva.group_annots()[0]
            >>> Y_list = smk.dinva.group_annots()
            >>> Y = Y_list[0]
            >>> cm = smk.smk_chipmatch(X, Y_list, qreq_)
            >>> ut.qt4ensure()
            >>> cm.ishow_analysis(qreq_)
            >>> ut.show_if_requested()

        """
        from ibeis.algo.hots import chip_match
        from ibeis.algo.hots import pipeline

        qaid = X.aid
        daid_list = []
        fm_list = []
        fsv_list = []
        fsv_col_lbls = ['smk']

        gammaX = smk.gamma(X)
        for Y in ut.ProgIter(Y_list, lbl='smk match qaid=%r' % (qaid,)):
            gammaY = smk.gamma(Y)
            gammaXY = gammaX * gammaY

            fm = []
            fs = []
            # Words in common define matches
            common_words = ut.isect(X.words, Y.words)
            for c in common_words:
                word_fm, word_fs = smk.weighted_matches(X, Y, c)
                word_fs *= gammaXY
                fm.extend(word_fm)
                fs.extend(word_fs)

            #if len(fm) > 0:
            daid = Y.aid
            fsv = np.array(fs)[:, None]
            daid_list.append(daid)
            fm = np.array(fm)
            fm_list.append(fm)
            fsv_list.append(fsv)

        progiter = iter(ut.ProgIter([0], lbl='smk sver qaid=%r' % (qaid,)))
        six.next(progiter)

        # Build initial matches
        cm = chip_match.ChipMatch(qaid=qaid, daid_list=daid_list,
                                  fm_list=fm_list, fsv_list=fsv_list,
                                  fsv_col_lbls=fsv_col_lbls)
        cm.arraycast_self()
        # Score matches and take a shortlist
        cm.score_maxcsum(qreq_)
        nNameShortList  = qreq_.qparams.nNameShortlistSVER
        nAnnotPerName   = qreq_.qparams.nAnnotPerNameSVER
        top_aids = cm.get_name_shortlist_aids(nNameShortList, nAnnotPerName)
        cm = cm.shortlist_subset(top_aids)
        # Spatially verify chip matches
        cm = pipeline.sver_single_chipmatch(qreq_, cm)
        # Rescore
        cm.score_maxcsum(qreq_)
        list(progiter)
        return cm


def render_vocab_word(ibs, inva, wx, fnum=None):
    """
    Creates a visualization of a visual word. This includes the average patch,
    the SIFT-like representation of the centroid, and some of the patches that
    were assigned to it.

    CommandLine:
        python -m ibeis.new_annots render_vocab_word --show

    Example:
        >>> from ibeis.new_annots import *  # NOQA
        >>> ibs, aid_list, inva = testdata_inva('PZ_MTEST', num_words=10000)
        >>> sortx = ut.argsort(inva.num_list)[::-1]
        >>> wx_list = ut.take(inva.wx_list, sortx)
        >>> wx = wx_list[0]
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.qt4ensure()
        >>> fnum = 2
        >>> fnum = pt.ensure_fnum(fnum)
        >>> # Interactive visualization of many words
        >>> for wx in ut.InteractiveIter(wx_list):
        >>>     word_img = render_vocab_word(ibs, inva, wx, fnum)
        >>>     pt.imshow(word_img, fnum=fnum, title='Word %r/%r' % (wx, len(inva.vocab)))
        >>>     pt.update()
        >>> ut.show_if_requested()
    """
    import plottool as pt
    # Create the contributing patch image
    word_patches = inva.get_patches(wx)
    stacked_patches = vt.stack_square_images(word_patches)

    # Create the average word image
    word = inva.vocab.wx2_word[wx]
    average_patch = np.mean(word_patches, axis=0)
    #vecs = inva.get_vecs(wx)
    #assert np.allclose(word, vecs.mean(axis=0))
    with_sift = True
    if with_sift:
        patch_img = pt.render_sift_on_patch(average_patch, word)
    else:
        patch_img = average_patch

    # Stack them together
    solidbar = np.zeros((patch_img.shape[0], int(patch_img.shape[1] * .1), 3), dtype=patch_img.dtype)
    border_color = (100, 10, 10)  # bgr, darkblue
    if ut.is_float(solidbar):
        solidbar[:, :, :] = (np.array(border_color) / 255)[None, None]
    else:
        solidbar[:, :, :] = np.array(border_color)[None, None]
    # word_img = vt.stack_image_list([patch_img, solidbar, stacked_patches], vert=False, modifysize=True)
    patch_img2 = vt.inverted_sift_patch(word)
    patch_img = vt.rectify_to_uint8(patch_img)
    patch_img2 = vt.rectify_to_uint8(patch_img2)
    solidbar = vt.rectify_to_uint8(solidbar)
    stacked_patches = vt.rectify_to_uint8(stacked_patches)
    patch_img2, patch_img = vt.make_channels_comparable(patch_img2, patch_img)
    word_img = vt.stack_image_list([patch_img, solidbar, patch_img2, solidbar, stacked_patches], vert=False, modifysize=True)
    return word_img


def render_vocab(inva):
    """
    Renders the average patch of each word.
    This is a quick visualization of the entire vocabulary.

    CommandLine:
        python -m ibeis.new_annots render_vocab --show

    Example:
        >>> from ibeis.new_annots import *  # NOQA
        >>> ibs, aid_list, inva = testdata_inva('PZ_MTEST', num_words=10000)
        >>> render_vocab(inva)
        >>> ut.show_if_requested()
    """
    sortx = ut.argsort(inva.num_list)[::-1]
    # Get words with the most assignments
    wx_list = ut.take(inva.wx_list, sortx)

    wx_list = ut.strided_sample(wx_list, 64)

    word_patch_list = []
    for wx in ut.ProgIter(wx_list, bs=True, lbl='building patches'):
        word = inva.vocab.wx2_word[wx]
        if False:
            word_patch = inva.get_word_patch(wx)
        else:
            word_patch = vt.inverted_sift_patch(word, 64)
        import plottool as pt
        word_patch = pt.render_sift_on_patch(word_patch, word)
        word_patch_list.append(word_patch)

    #for wx, p in zip(wx_list, word_patch_list):
    #    inva._word_patches[wx] = p

    all_words = vt.stack_square_images(word_patch_list)
    import plottool as pt
    pt.qt4ensure()
    pt.imshow(all_words)


def compute_rvec(vecs, word, asint=False):
    # Subtract each vector from its quantized word to get the resiudal
    rvecs_ = np.subtract(word.astype(np.float), vecs.astype(np.float))
    # If a word was quantized to itself then the residual is 0 and it cant
    # be L2 noramlized.
    is_zero = ~np.any(rvecs_, axis=1)
    # Normalize residuals to unit length
    vt.normalize_rows(rvecs_, out=rvecs_)
    if np.any(is_zero):
        # reset these values back to zero
        rvecs_[is_zero, :] = 0
    if asint:
        rvecs = np.clip(np.round(rvecs_ * 255.0), -127, 127).astype(np.int8)
    else:
        rvecs = rvecs_
    # Determine if any errors occurred
    # FIXME: zero will drive the score of a match to 0 even though if they
    # are both 0, then it is an exact match and should be scored as a 1.
    error_flags = ~np.any(rvecs_, axis=1)
    return rvecs, error_flags


def aggregate_rvecs(rvecs, maws, asint=False):
    r"""
    helper to compute Phi(X_c)
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
        if asint:
            rvecs_agg = np.clip(np.round(arr_float * 255.0), -127, 127).astype(np.int8)
        else:
            rvecs_agg = arr_float
    return rvecs_agg


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
