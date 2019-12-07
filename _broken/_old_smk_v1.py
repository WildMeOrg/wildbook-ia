import weakref
import six


@ut.reloadable_class
class SMKCacheable(object):
    """
    helper for depcache-less caching
    FIXME: Just use the depcache. Much less headache.
    """
    def get_fpath(self, cachedir, cfgstr=None, cfgaug=''):
        _args2_fpath = ut.util_cache._args2_fpath
        #prefix = self.get_prefix()
        prefix = self.__class__.__name__ + '_'
        cfgstr = self.get_hashid() + cfgaug
        #ext    = self.ext
        ext    = '.cPkl'
        fpath  = _args2_fpath(cachedir, prefix, cfgstr, ext)
        return fpath

    def get_hashid(self):
        cfgstr = self.get_cfgstr()
        version = six.text_type(getattr(self, '__version__', 0))
        hashstr = ut.hashstr27(cfgstr + version)
        return hashstr

    def ensure(self, cachedir):
        fpath = self.get_fpath(cachedir)
        needs_build = True
        if ut.checkpath(fpath):
            try:
                self.load(fpath)
                needs_build = False
            except ImportError:
                print('Need to recompute')
        if needs_build:
            self.build()
            ut.garbage_collect()
            self.save(fpath)
        return fpath

    def save(self, fpath):
        state = ut.dict_subset(self.__dict__, self.__columns__)
        ut.save_data(fpath, state)

    def load(self, fpath):
        state = ut.load_data(fpath)
        self.__dict__.update(**state)



@ut.reloadable_class
class ForwardIndex(ut.NiceRepr, SMKCacheable):
    """
    A Forward Index of Stacked Features

    A stack of features from multiple annotations.
    Contains a method to create an inverted index given a vocabulary.
    """
    __columns__ = ['ax_to_aid', 'ax_to_nFeat', 'idx_to_fx', 'idx_to_ax',
                   'idx_to_vec', 'aid_to_ax']
    __version__ = 1

    def __init__(fstack, ibs=None, aid_list=None, config=None, name=None):
        # Basic Info
        fstack.ibs = ibs
        fstack.config = config
        fstack.name = name
        #-- Stack 1 --
        fstack.ax_to_aid = aid_list
        fstack.ax_to_nFeat = None
        fstack.aid_to_ax = None
        #-- Stack 2 --
        fstack.idx_to_fx = None
        fstack.idx_to_ax = None
        fstack.idx_to_vec = None
        # --- Misc ---
        fstack.num_feat = None

    def build(fstack):
        print('building forward index')
        ibs = fstack.ibs
        config = fstack.config
        ax_to_vecs = ibs.depc.d.get_feat_vecs(fstack.ax_to_aid, config=config)
        #-- Stack 1 --
        fstack.ax_to_nFeat = [len(vecs) for vecs in ax_to_vecs]
        #-- Stack 2 --
        fstack.idx_to_fx = np.array(ut.flatten([
            list(range(num)) for num in fstack.ax_to_nFeat]))
        fstack.idx_to_ax = np.array(ut.flatten([
            [ax] * num for ax, num in enumerate(fstack.ax_to_nFeat)]))
        fstack.idx_to_vec = np.vstack(ax_to_vecs)
        # --- Misc ---
        fstack.num_feat = sum(fstack.ax_to_nFeat)
        fstack.aid_to_ax = ut.make_index_lookup(fstack.ax_to_aid)

    def __nice__(fstack):
        name = '' if fstack.name is None else fstack.name + ' '
        if fstack.num_feat is None:
            return '%snA=%r NotInitialized' % (name, ut.safelen(fstack.ax_to_aid))
        else:
            return '%snA=%r nF=%r' % (name, ut.safelen(fstack.ax_to_aid),
                                      fstack.num_feat)

    def get_cfgstr(self):
        depc = self.ibs.depc
        annot_cfgstr = self.ibs.get_annot_hashid_visual_uuid(self.aid_list, prefix='')
        stacked_cfg = depc.stacked_config('annotations', 'feat', self.config)
        config_cfgstr = stacked_cfg.get_cfgstr()
        cfgstr = '_'.join([annot_cfgstr, config_cfgstr])
        return cfgstr

    @property
    def aid_list(fstack):
        return fstack.ax_to_aid

    def assert_self(fstack):
        assert len(fstack.idx_to_fx) == len(fstack.idx_to_ax)
        assert len(fstack.idx_to_fx) == len(fstack.idx_to_vec)
        assert len(fstack.ax_to_aid) == len(fstack.ax_to_nFeat)
        assert fstack.idx_to_ax.max() < len(fstack.ax_to_aid)


@ut.reloadable_class
class InvertedIndex(ut.NiceRepr, SMKCacheable):
    """
    Maintains an inverted index of chip descriptors that are multi-assigned to
    a set of vocabulary words.

    This stack represents the database.
    It prorcesses the important information like

    * vocab - this is the quantizer

    Each word is has an inverted index to a list of:
        (annotation index, feature index, multi-assignment weight)

    CommandLine:
        python -m ibeis.algo.smk.vocab_indexer InvertedIndex --show

    Example:
        >>> # ENABLE_LATER
        >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
        >>> ibs, aid_list, inva = testdata_inva('testdb1', num_words=1000)
        >>> print(inva)
    """
    __columns__ = ['wx_to_idxs', 'wx_to_maws', 'wx_to_num', 'wx_list',
                   'perword_stats', 'wx_to_idf', 'grouped_annots']
    __version__ = 1

    def __init__(inva, fstack=None, vocab=None, config={}):
        inva.fstack = fstack
        inva.vocab = vocab
        inva.config = InvertedIndexConfig(**config)
        # Corresponding arrays
        inva.wx_to_idxs = None
        inva.wx_to_maws = None
        inva.wx_to_num = None
        #
        inva.wx_list = None
        # Extra stuff
        inva.perword_stats = None
        inva._word_patches = {}
        # Optional measures
        inva.wx_to_idf = None
        inva.grouped_annots = None

    def __nice__(inva):
        fstack = inva.fstack
        name = '' if fstack.name is None else fstack.name + ' '
        if inva.wx_to_idxs is None:
            return '%sNotInitialized' % (name,)
        else:
            return '%snW=%r mean=%.2f' % (name, ut.safelen(inva.wx_to_idxs),
                                          inva.perword_stats['mean'])

    def build(inva):
        print('building inverted index')
        nAssign = inva.config.get('nAssign', 1)
        fstack = inva.fstack
        vocab = inva.vocab
        idx_to_wxs, idx_to_maws = vocab.assign_to_words(
            fstack.idx_to_vec, nAssign, verbose=True)
        wx_to_idxs, wx_to_maws = vocab.invert_assignment(
            idx_to_wxs, idx_to_maws, verbose=True)
        # Corresponding arrays
        inva.wx_to_idxs = wx_to_idxs
        inva.wx_to_maws = wx_to_maws
        inva.wx_to_num = ut.map_dict_vals(len, inva.wx_to_idxs)
        #
        inva.wx_list = sorted(inva.wx_to_num.keys())
        # Extra stuff
        inva.perword_stats = ut.get_stats(list(inva.wx_to_num.values()))
        inva._word_patches = {}
        # Optional measures
        inva.grouped_annots = inva.compute_annot_groups()
        print('Done building inverted index')

    def inverted_annots(inva, aids):
        if inva.grouped_annots is None:
            raise ValueError('grouped annots not computed')
        ax_list = ut.take(inva.fstack.aid_to_ax, aids)
        return ut.take(inva.grouped_annots, ax_list)

    def get_cfgstr(inva):
        cfgstr = '_'.join([inva.config.get_cfgstr(),
                           inva.vocab.config.get_cfgstr(),
                           inva.fstack.get_cfgstr()])
        return cfgstr

    def wx_to_fxs(inva, wx):
        return inva.fstack.idx_to_fx.take(inva.wx_to_idxs[wx], axis=0)

    def wx_to_axs(inva, wx):
        return inva.fstack.idx_to_ax.take(inva.wx_to_idxs[wx], axis=0)

    def load(inva, fpath):
        super(InvertedIndex, inva).load(fpath)
        if inva.grouped_annots is not None:
            for X in inva.grouped_annots:
                X.inva = inva

    def get_patches(inva, wx, verbose=True):
        """
        Loads the patches assigned to a particular word in this stack
        """
        ax_list = inva.wx_to_axs(wx)
        fx_list = inva.wx_to_fxs(wx)
        config = inva.fstack.config
        ibs = inva.fstack.ibs

        # Group annotations with more than one assignment to this word, so we
        # only have to load a chip at most once
        unique_axs, groupxs = vt.group_indices(ax_list)
        fxs_groups = vt.apply_grouping(fx_list, groupxs)

        unique_aids = ut.take(inva.fstack.ax_to_aid, unique_axs)

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
            >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva(num_words=256)
            >>> wx_to_idf = inva.compute_word_idf()
            >>> assert np.all(wx_to_idf >= 0)
            >>> prob_list = (1 / np.exp(wx_to_idf))
            >>> assert np.all(prob_list >= 0)
            >>> assert np.all(prob_list <= 1)
        """
        print('Computing idf')
        num_docs_total = len(inva.fstack.ax_to_aid)
        # idf denominator (the num of docs containing a word for each word)
        # The max(maws) to denote the probab that this word indexes an annot
        num_docs_list = np.array([
            sum([maws.max() for maws in vt.apply_grouping(
                inva.wx_to_maws[wx],
                vt.group_indices(inva.wx_to_axs(wx))[1])])
            for wx in ut.ProgIter(inva.wx_list, lbl='Compute IDF', bs=True)
        ])
        # Typically for IDF, 1 is added to the denom to prevent divide by 0
        # We add epsilon to numer and denom to ensure recep is a probability
        idf_list = np.log(np.divide(num_docs_total + 1, num_docs_list + 1))
        wx_to_idf = dict(zip(inva.wx_list, idf_list))
        wx_to_idf = ut.DefaultValueDict(0, wx_to_idf)
        return wx_to_idf

    def print_name_consistency(inva, ibs):
        annots = ibs.annots(inva.fstack.ax_to_aid)
        ax_to_nid = annots.nids

        wx_to_consist = {}

        for wx in inva.wx_list:
            axs = inva.wx_to_axs(wx)
            nids = ut.take(ax_to_nid, axs)
            consist = len(ut.unique(nids)) / len(nids)
            wx_to_consist[wx] = consist
            #if len(nids) > 5:
            #    break

    def get_vecs(inva, wx):
        """
        Get raw vectors assigned to a word
        """
        vecs = inva.fstack.idx_to_vec.take(inva.wx_to_idxs[wx], axis=0)
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
            >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva(num_words=256)
            >>> wx = 1
            >>> asint = False
            >>> rvecs_list, error_flags = inva.compute_rvecs(1)
        """
        # Pick out corresonding lists of residuals and words
        vecs = inva.get_vecs(wx)
        word = inva.vocab.wx_to_word[wx]
        return compute_rvec(vecs, word)

    #def get_grouped_rvecs(inva, wx):
    #    """
    #    # Get all residual vectors assigned to a word grouped by annotation
    #    """
    #    rvecs_list, error_flags = inva.compute_rvecs(wx)
    #    ax_list = inva.wx_to_axs(wx)
    #    maw_list = inva.wx_to_maws[wx]
    #    # group residual vectors by annotation
    #    unique_ax, groupxs = vt.group_indices(ax_list)
    #    # (weighted aggregation with multi-assign-weights)
    #    grouped_maws   = vt.apply_grouping(maw_list, groupxs)
    #    grouped_rvecs  = vt.apply_grouping(rvecs_list, groupxs)
    #    grouped_errors = vt.apply_grouping(error_flags, groupxs)

    #    outer_flags = [len(rvecs) > 0 for rvecs in grouped_rvecs]
    #    if all(outer_flags):
    #        grouped_rvecs3_  = grouped_rvecs
    #        grouped_maws3_   = grouped_maws
    #        grouped_errors3_ = grouped_errors
    #        unique_ax3_      = unique_ax
    #    else:
    #        grouped_rvecs3_  = ut.compress(grouped_rvecs, outer_flags)
    #        grouped_maws3_   = ut.compress(grouped_maws, outer_flags)
    #        grouped_errors3_ = ut.compress(grouped_errors, outer_flags)
    #        unique_ax3_ = ut.compress(unique_ax, outer_flags)
    #    rvec_group_tup = (unique_ax3_, grouped_rvecs3_, grouped_maws3_,
    #                      grouped_errors3_)
    #    return rvec_group_tup

    #def compute_rvecs_agg(inva, wx):
    #    """
    #    Sums and normalizes all rvecs that belong to the same word and the same
    #    annotation id

    #    Returns:
    #        ax_to_aggs: A mapping from an annotation to its aggregated vector
    #            for this word and an error flag.

    #    Notes:
    #        aggvec = Phi
    #        Phi(X_c) = psi(sum([phi(x) for x in X_c]))
    #        psi is the identify function currently.
    #        Phi is esentially a VLAD vector for a specific word.
    #        Returns Phi vectors wrt each annotation.

    #    Example:
    #        >>> # ENABLE_LATER
    #        >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
    #        >>> ibs, aid_list, inva = testdata_inva(num_words=256)
    #        >>> wx = 1
    #        >>> asint = False
    #        >>> ax_to_aggs = inva.compute_rvecs_agg(1)
    #    """
    #    rvec_group_tup = inva.get_grouped_rvecs(wx)
    #    ax_to_aggs = {
    #        ax: aggregate_rvecs(rvecs, maws, errors)
    #        for ax, rvecs, maws, errors in
    #        zip(*rvec_group_tup)
    #    }
    #    return ax_to_aggs

    def compute_annot_groups(inva):
        """
        Group by annotations first and then by words

            >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
            >>> ibs, smk, qreq_ = testdata_smk()
            >>> inva = qreq_.qinva
        """
        print('Computing annot groups')
        fstack = inva.fstack
        ax_to_X = [IndexedAnnot(ax, inva) for ax in range(len(fstack.ax_to_aid))]

        for wx in ut.ProgIter(inva.wx_list, lbl='Group Annots', adjust=True,
                              freq=1, bs=True):
            idxs = inva.wx_to_idxs[wx]
            maws = inva.wx_to_maws[wx]
            axs = fstack.idx_to_ax.take(idxs, axis=0)

            unique_axs, groupxs = vt.group_indices(axs)
            grouped_maws = vt.apply_grouping(maws, groupxs)
            grouped_idxs = vt.apply_grouping(idxs, groupxs)

            # precompute here
            #ax_to_aggs = inva.compute_rvecs_agg(wx)
            for ax, idxs_, maws_ in zip(unique_axs, grouped_idxs,
                                        grouped_maws):
                X = ax_to_X[ax]
                X.wx_to_idxs_[wx] = idxs_
                X.wx_to_maws_[wx] = maws_
                #X.wx_to_aggs[wx] = ax_to_aggs[ax]
        X_list = ax_to_X

        #_prog = ut.ProgPartial(lbl='Compute Phi', adjust=True, freq=1, bs=True)
        #for X in _prog(X_list):
        #    X.wx_to_aggs = dict([(wx, X._make_Phi_flags(wx)) for wx in X.words])

        print('Precomputing agg rvecs')
        agg_gen = parallel_Phi(X_list)
        for X, wx_to_aggs in zip(X_list, agg_gen):
            X.wx_to_aggs = wx_to_aggs

        #for X, wx_to_aggs in zip(X_list, parallel_Phi2(X_list)):
        #    X.wx_to_aggs = wx_to_aggs

        #for X, wx_to_aggs in zip(X_list, serial_Phi(X_list)):
        #    X.wx_to_aggs = wx_to_aggs

        #for X in ut.ProgIter(X_list, lbl='Compute Phi', adjust=True,
        #                     freq=1, bs=True):
        #    for wx in X.words:
        #        X.wx_to_aggs[wx] = X._make_Phi_flags(wx)

        print('Finished grouping annots')
        return X_list

    def assert_self(inva):
        assert sorted(inva.wx_to_idxs.keys()) == sorted(inva.wx_to_maws.keys())
        assert sorted(inva.wx_list) == sorted(inva.wx_to_maws.keys())
        # Ensure that each descriptor received a combined weight of one
        idx_to_maws = ut.ddict(list)
        for wx in inva.wx_list:
            maws = inva.wx_to_maws[wx]
            idxs = inva.wx_to_idxs[wx]
            for idx, maw in zip(idxs, maws):
                idx_to_maws[idx].append(maw)
        checksum = map(sum, idx_to_maws.values())
        # This is ok if the jegou hack is on
        assert all(ut.almost_eq(val, 1) for val in checksum), (
            'weights did not break evenly')
        inva.fstack.assert_self()


#def serial_Phi(X_list):
#    # Build argument generator
#    args_gen = get_Phi_data(X_list)
#    _prog = ut.ProgPartial(nTotal=len(X_list), lbl='phi chunk', freq=1,
#                           adjust=True, bs=True)
#    for args in _prog(args_gen):
#        result = par_Phi_worker(args)
#        yield result


def parallel_Phi(X_list):
    from concurrent import futures
    # Build argument generator
    args_gen = get_Phi_data(X_list)
    worker = par_Phi_worker

    nprocs = 6
    executor = futures.ProcessPoolExecutor(nprocs)
    #executor = futures.ThreadPoolExecutor(nprocs)
    chunksize = nprocs * 1
    nTasks = len(X_list)
    nChunks = ut.get_num_chunks(nTasks, chunksize)
    chunk_iter = ut.ichunks(args_gen, chunksize)
    _prog = ut.ProgPartial(nTotal=nChunks, lbl='phi chunk', freq=1,
                           adjust=True, bs=True)
    print('nprocs = %r' % (nprocs,))
    print('chunksize = %r' % (chunksize,))
    for arg_chunk in _prog(chunk_iter):
        fs_chunk = [executor.submit(worker, args) for args in arg_chunk]
        for fs in fs_chunk:
            result = fs.result()
            yield result
    executor.shutdown(wait=True)

#def parallel_Phi2(X_list):
#    # Build argument generator
#    args_gen = get_Phi_data(X_list)
#    worker = par_Phi_worker
#    #nprocs = 8
#    #chunksize = nprocs * 1
#    nTasks = len(X_list)
#    #nChunks = ut.get_num_chunks(nTasks, chunksize)
#    #chunk_iter = ut.ichunks(args_gen, chunksize)
#    #_prog = ut.ProgPartial(nTotal=nChunks, lbl='phi chunk', freq=1,
#    #                       adjust=True, bs=True)
#    for result in ut.generate(worker, args_gen, nTasks=nTasks):
#        yield result

    #result_gen = executor.map(par_Phi_worker, args_gen)
    #results = [res for res in ut.ProgIter(result_gen, nTotal=len(X_list), adjust=False, freq=1)]
    #return results


def get_Phi_data(X_list):
    for X in X_list:
        data_tup = par_Phi_data(X)
        yield data_tup


def par_Phi_worker(data_tup):
    wx_list, maws_list, vecs_list, word_list = data_tup
    wx_to_aggs = {}
    for wx, maws, vecs, word in zip(wx_list, maws_list, vecs_list, word_list):
        rvecs, flags = compute_rvec(vecs, word)
        rvecs_agg, flags_agg = aggregate_rvecs(rvecs, maws, flags)
        wx_to_aggs[wx] = rvecs_agg, flags_agg
    return wx_to_aggs


def par_Phi_data(X):
    idx_to_vec = X.inva.fstack.idx_to_vec
    wx_to_word = X.inva.vocab.wx_to_word
    wx_list = X.words

    idxs_list = ut.take(X.wx_to_idxs_, wx_list)
    maws_list = ut.take(X.wx_to_maws_, wx_list)
    vecs_list = [idx_to_vec.take(idxs, axis=0)
                 for idxs in idxs_list]
    word_list = ut.take(wx_to_word, wx_list)
    data_tup = wx_list, maws_list, vecs_list, word_list
    return data_tup


@ut.reloadable_class
class IndexedAnnot(ut.NiceRepr):
    """
    Example:
        >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
        >>> ibs, aid_list, inva = testdata_inva(num_words=256)
        >>> X_list = inva.grouped_annots
        >>> X = X_list[0]
        >>> X.assert_self()
    """
    def __init__(X, ax, inva):
        X.ax = ax
        X._inva = None
        X.inva = inva
        # only idxs that belong to ax
        X.wx_to_idxs_ = {}
        X.wx_to_maws_ = {}
        X.wx_to_aggs = {}
        X.wx_to_phis = {}

    @property
    def inva(X):
        return X._inva()

    @inva.setter
    def inva(X, inva):
        X._inva = weakref.ref(inva)

    def __getstate__(self):
        state = ut.delete_dict_keys(self.__dict__.copy(), ['_inva'])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __nice__(X):
        return 'aid=%r, nW=%r' % (X.aid, len(X.wx_to_idxs_),)

    @property
    def aid(X):
        return X.inva.fstack.ax_to_aid[X.ax]

    def idxs(X, wx):
        return X.wx_to_idxs_[wx]

    def maws(X, wx):
        return X.wx_to_maws_[wx]

    def fxs(X, wx):
        fxs = X.inva.fstack.idx_to_fx.take(X.idxs(wx), axis=0)
        return fxs

    def VLAD(X):
        import scipy.sparse
        num_words, word_dim = X.inva.vocab.shape
        vlad_dim = num_words * word_dim
        vlad_vec = scipy.sparse.lil_matrix((vlad_dim, 1))
        for wx, aggvec in X.wx_to_aggs.items():
            start = wx * word_dim
            vlad_vec[start:start + word_dim] = aggvec.T
        vlad_vec = vlad_vec.tocsc()
        return vlad_vec

    def _make_Phi_flags(X, wx):
        rvecs, flags = X._make_phis_flags(wx)
        maws = X.wx_to_maws_[wx]
        rvecs_agg, flags_agg = aggregate_rvecs(rvecs, maws, flags)
        return rvecs_agg, flags_agg

    def _make_phis_flags(X, wx):
        idx_to_vec = X.inva.fstack.idx_to_vec
        wx_to_word = X.inva.vocab.wx_to_word
        idxs = X.wx_to_idxs_[wx]
        vecs = idx_to_vec.take(idxs, axis=0)
        word = wx_to_word[wx]
        rvecs, flags = compute_rvec(vecs, word)
        return rvecs, flags

    def Phi(X, wx):
        return X.wx_to_aggs[wx][0]

    def Phi_flags(X, wx):
        return X.wx_to_aggs[wx]

    def phis(X, wx):
        rvecs =  X.phis_flags(wx)[0]
        return rvecs

    def phis_flags(X, wx):
        if wx in X.wx_to_phis:
            return X.wx_to_phis[wx]
        else:
            return X._make_phis_flags(wx)

    @property
    def words(X):
        return list(X.wx_to_idxs_.keys())

    def assert_self(X):
        assert len(X.wx_to_idxs_) == len(X.wx_to_aggs)
        for wx in X.words:
            axs = X.inva.fstack.idx_to_ax.take(X.idxs(wx), axis=0)
            assert np.all(axs == X.ax)
            rvecs, flags = X.phis_flags(wx)
            rvecs_agg, flags_agg = X._make_Phi_flags(wx)
            assert np.all(rvecs_agg == X.Phi(wx))


def get_Phi_data2(X_list):
    for X in X_list:
        data_tup = par_Phi_data(X)
        yield data_tup




def par_Phi_workers2(data_tup):
    wx_list, maws_list, vecs_list, word_list = data_tup
    wx_to_aggs = {}
    for wx, maws, vecs, word in zip(wx_list, maws_list, vecs_list, word_list):
        rvecs, flags = compute_rvec(vecs, word)
        rvecs_agg, flags_agg = aggregate_rvecs(rvecs, maws, flags)
        wx_to_aggs[wx] = rvecs_agg, flags_agg
    return wx_to_aggs


def par_Phi_data2(X):
    idx_to_vec = X.inva.fstack.idx_to_vec
    wx_to_word = X.inva.vocab.wx_to_word
    wx_list = X.words

    idxs_list = ut.take(X.wx_to_idxs_, wx_list)
    maws_list = ut.take(X.wx_to_maws_, wx_list)
    vecs_list = [idx_to_vec.take(idxs, axis=0)
                 for idxs in idxs_list]
    word_list = ut.take(wx_to_word, wx_list)
    data_tup = wx_list, maws_list, vecs_list, word_list
    return data_tup



@profile
def assign_invert_phi(vocab, fx_to_vecs, nAssign, int_rvec):
    fx_to_wxs, fx_to_maws = vocab.assign_to_words(fx_to_vecs, nAssign)
    wx_to_fxs, wx_to_maws = vocab.invert_assignment(fx_to_wxs, fx_to_maws)
    wx_list = sorted(wx_to_fxs.keys())

    if int_rvec:
        agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.int8)
    else:
        agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.float)
    agg_flags = np.empty((len(wx_list), 1), dtype=np.bool)

    #rvecs_list = []
    #flags_list = []

    # For each word this annotation is assigned to
    # Compute the rvecs and the agg_rvecs
    for idx, wx in enumerate(wx_list):
        maws = wx_to_maws[wx]
        fxs = wx_to_fxs[wx]
        word = vocab.wx_to_word[wx]
        vecs = fx_to_vecs.take(fxs, axis=0)

        _rvecs, _flags = compute_rvec(vecs, word)
        _agg_rvecs, _agg_flags = aggregate_rvecs(_rvecs, maws, _flags)
        # Cast to integers for storage
        if int_rvec:
            _agg_rvecs = cast_residual_integer(_agg_rvecs)
            #_rvecs = cast_residual_integer(_rvecs)
        #rvecs_list.append(_rvecs)
        #flags_list.append(_flags)
        agg_rvecs[idx] = _agg_rvecs[0]
        agg_flags[idx] = _agg_flags[0]

    fxs_list = ut.take(wx_to_fxs, wx_list)
    maws_list = ut.take(wx_to_maws, wx_list)

    tup = (wx_list, fxs_list, maws_list, agg_rvecs, agg_flags)
    return tup


#if True:
#else:
#    #with ut.Timer():
#    for fx_to_vecs in vecs_list:
#        tup = assign_invert_phi(vocab, fx_to_vecs, nAssign, int_rvec)



def render_vocab(vocab, inva=None, use_data=False):
    """
    Renders the average patch of each word.
    This is a quick visualization of the entire vocabulary.

    CommandLine:
        python -m ibeis.algo.smk.vocab_indexer render_vocab --show
        python -m ibeis.algo.smk.vocab_indexer render_vocab --show --use-data
        python -m ibeis.algo.smk.vocab_indexer render_vocab --show --debug-depc

    Example:
        >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
        >>> ibs, aid_list, voab = testdata_vocab('PZ_MTEST', num_words=10000)
        >>> use_data = ut.get_argflag('--use-data')
        >>> vocab = inva.vocab
        >>> all_words = vocab.render_vocab(inva, use_data=use_data)
        >>> ut.quit_if_noshow()
        >>> import plottool_ibeis as pt
        >>> pt.qt4ensure()
        >>> pt.imshow(all_words)
        >>> ut.show_if_requested()
    """
    # Get words with the most assignments
    sortx = ut.argsort(list(inva.wx_to_num.values()))[::-1]
    wx_list = ut.take(list(inva.wx_to_num.keys()), sortx)

    wx_list = ut.strided_sample(wx_list, 64)

    word_patch_list = []
    for wx in ut.ProgIter(wx_list, bs=True, lbl='building patches'):
        word = inva.vocab.wx_to_word[wx]
        if use_data:
            word_patch = inva.get_word_patch(wx)
        else:
            word_patch = vt.inverted_sift_patch(word, 64)
        import plottool_ibeis as pt
        word_patch = pt.render_sift_on_patch(word_patch, word)
        word_patch_list.append(word_patch)

    #for wx, p in zip(wx_list, word_patch_list):
    #    inva._word_patches[wx] = p
    all_words = vt.stack_square_images(word_patch_list)
    return all_words
