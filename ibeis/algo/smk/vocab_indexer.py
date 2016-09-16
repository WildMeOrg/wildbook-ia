# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, map
import six
import dtool
import utool as ut
import vtool as vt
import pyflann
import numpy as np
#from ibeis import core_annots
from ibeis.control.controller_inject import register_preprocs
(print, rrr, profile) = ut.inject2(__name__)


derived_attribute = register_preprocs['annot']


class VocabConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('algorithm', 'minibatch', 'alg'),
        ut.ParamInfo('random_seed', 42, 'seed'),
        ut.ParamInfo('num_words', 1000, 'n'),
        #ut.ParamInfo('num_words', 64000),
        ut.ParamInfo('version', 1),
        #ut.ParamInfo('n_jobs', -1, hide=True),
    ]


class InvertedIndexConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('nAssign', 2),
        #massign_alpha=1.2,
        #massign_sigma=80.0,
        #massign_equal_weights=False
    ]


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

    def ensure(self, cachedir, **kwargs):
        if kwargs:
            cfgaug = ut.hashstr27(ut.repr2(kwargs, sorted_=True), hashlen=4)
        else:
            cfgaug = ''
        fpath = self.get_fpath(cachedir, cfgaug=cfgaug)
        needs_build = True
        if ut.checkpath(fpath):
            try:
                self.load(fpath)
                needs_build = False
            except ImportError:
                print('Need to recompute')
        if needs_build:
            self.build(**kwargs)
            self.save(fpath)
        return fpath

    def save(self, fpath):
        state = ut.dict_subset(self.__dict__, self.__columns__)
        ut.save_data(fpath, state)

    def load(self, fpath):
        state = ut.load_data(fpath)
        self.__dict__.update(**state)


@ut.reloadable_class
class VisualVocab(ut.NiceRepr):
    """
    Class that maintains a list of visual words (cluster centers)
    Also maintains a nearest neighbor index structure for finding words.
    This class is build using the depcache
    """

    def __init__(vocab, words=None):
        vocab.wx_to_word = words
        vocab.wordflann = None
        vocab.flann_params = vt.get_flann_params(random_seed=42)
        # TODO: grab the depcache rowid and maybe config?
        # make a dtool.Computable

    def __nice__(vocab):
        return 'nW=%r' % (ut.safelen(vocab.wx_to_word))

    def __len__(vocab):
        return len(vocab.wx_to_word)

    @property
    def shape(vocab):
        return vocab.wx_to_word.shape

    def __getstate__(vocab):
        """
        http://www.linuxscrew.com/2010/03/24/fastest-way-to-create-ramdisk-in-ubuntulinux/
        sudo mkdir /tmp/ramdisk; chmod 777 /tmp/ramdisk
        sudo mount -t tmpfs -o size=256M tmpfs /tmp/ramdisk/
        http://zeblog.co/?p=1588
        """
        # TODO: Figure out how to make these play nice with the depcache
        state = vocab.__dict__.copy()
        if 'wx2_word' in state:
            state['wx_to_word'] = state.pop('wx2_word')
        del state['wordflann']
        # Make a special wordflann pickle
        # THIS WILL NOT WORK ON WINDOWS
        import tempfile
        assert not ut.WIN32, 'Need to fix this on WIN32. Cannot write to temp file'
        temp = tempfile.NamedTemporaryFile(delete=True)
        try:
            vocab.wordflann.save_index(temp.name)
            wordindex_bytes = temp.read()
            #print('wordindex_bytes = %r' % (len(wordindex_bytes),))
            state['wordindex_bytes'] = wordindex_bytes
        except Exception:
            raise
        finally:
            temp.close()
        return state

    def __setstate__(vocab, state):
        wordindex_bytes = state.pop('wordindex_bytes')
        vocab.__dict__.update(state)
        vocab.wordflann = pyflann.FLANN()
        import tempfile
        assert not ut.WIN32, 'Need to fix this on WIN32. Cannot write to temp file'
        temp = tempfile.NamedTemporaryFile(delete=True)
        try:
            temp.write(wordindex_bytes)
            temp.file.flush()
            vocab.wordflann.load_index(temp.name, vocab.wx_to_word)
        except Exception:
            raise
        finally:
            temp.close()

    def build(vocab, verbose=True):
        num_vecs = len(vocab.wx_to_word)
        if vocab.wordflann is None:
            vocab.wordflann = pyflann.FLANN()
        if verbose:
            print('[nnindex] ...building kdtree over %d points (this may take a sec).' % num_vecs)
            tt = ut.tic(msg='Building vocab index')
        if num_vecs == 0:
            print('WARNING: CANNOT BUILD FLANN INDEX OVER 0 POINTS.')
            print('THIS MAY BE A SIGN OF A DEEPER ISSUE')
        else:
            vocab.wordflann.build_index(vocab.wx_to_word, **vocab.flann_params)
        if verbose:
            ut.toc(tt)

    def nn_index(vocab, idx_to_vec, nAssign):
        """
            >>> idx_to_vec = depc.d.get_feat_vecs(aid_list)[0]
            >>> vocab = vocab
            >>> nAssign = 1
        """
        # Assign each vector to the nearest visual words
        assert nAssign > 0, 'cannot assign to 0 neighbors'
        try:
            idx_to_vec = idx_to_vec.astype(vocab.wordflann._FLANN__curindex_data.dtype)
            _idx_to_wx, _idx_to_wdist = vocab.wordflann.nn_index(idx_to_vec, nAssign)
        except pyflann.FLANNException as ex:
            ut.printex(ex, 'probably misread the cached flann_fpath=%r' % (
                getattr(vocab.wordflann, 'flann_fpath', None),))
            raise
        else:
            _idx_to_wx = vt.atleast_nd(_idx_to_wx, 2)
            _idx_to_wdist = vt.atleast_nd(_idx_to_wdist, 2)
            return _idx_to_wx, _idx_to_wdist

    def assign_to_words(vocab, idx_to_vec, nAssign, massign_alpha=1.2,
                        massign_sigma=80.0, massign_equal_weights=False, verbose=None):
        """
        Assigns descriptor-vectors to nearest word.

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
                * idf2_wxs - vector index -> assigned word indexes

        Example:
            >>> # SLOW_DOCTEST
            >>> idx_to_vec = depc.d.get_feat_vecs(aid_list)[0][0::300]
            >>> idx_to_vec = np.vstack((idx_to_vec, vocab.wx_to_word[0]))
            >>> nAssign = 2
            >>> massign_equal_weights = False
            >>> massign_alpha = 1.2
            >>> massign_sigma = 80.0
            >>> nAssign = 2
            >>> idx_to_wxs, idx_to_maws = vocab.assign_to_words(idx_to_vec, nAssign)
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
                _idx_to_wx, _idx_to_wdist, massign_alpha, massign_sigma,
                massign_equal_weights)
        else:
            idx_to_wxs = _idx_to_wx.tolist()
            idx_to_maws = [[1.0]] * len(idx_to_wxs)
        return idx_to_wxs, idx_to_maws

    def invert_assignment(vocab, idx_to_wxs, idx_to_maws):
        """
        Inverts assignment of vectors to words into words to vectors.

        Example:
            >>> idx_to_idx = np.arange(len(idx_to_wxs))
            >>> other_idx_to_prop = (idx_to_idx,)
            >>> wx_to_idxs, wx_to_maws = vocab.invert_assignment(idx_to_wxs, idx_to_maws)
        """
        # Invert mapping -- Group by word indexes
        idx_to_nAssign = [len(wxs) for wxs in idx_to_wxs]
        jagged_idxs = [[idx] * num for idx, num in enumerate(idx_to_nAssign)]
        wx_keys, groupxs = vt.jagged_group(idx_to_wxs)
        idxs_list = vt.apply_jagged_grouping(jagged_idxs, groupxs)
        wx_to_idxs = dict(zip(wx_keys, idxs_list))
        maws_list = vt.apply_jagged_grouping(idx_to_maws, groupxs)
        wx_to_maws = dict(zip(wx_keys, maws_list))
        if ut.VERBOSE:
            print('[smk_index.assign] L___ End Assign vecs to words.')
        return (wx_to_idxs, wx_to_maws)

    def render_vocab_word(vocab, inva, wx, fnum=None):
        """
        Creates a visualization of a visual word. This includes the average patch,
        the SIFT-like representation of the centroid, and some of the patches that
        were assigned to it.

        CommandLine:
            python -m ibeis.algo.smk.vocab_indexer render_vocab_word --show

        Example:
            >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva('PZ_MTEST', num_words=10000)
            >>> vocab = inva.vocab
            >>> sortx = ut.argsort(list(inva.wx_to_num.values()))[::-1]
            >>> wx_list = ut.take(list(inva.wx_to_num.keys()), sortx)
            >>> wx = wx_list[0]
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> ut.qt4ensure()
            >>> fnum = 2
            >>> fnum = pt.ensure_fnum(fnum)
            >>> # Interactive visualization of many words
            >>> for wx in ut.InteractiveIter(wx_list):
            >>>     word_img = vocab.render_vocab_word(inva, wx, fnum)
            >>>     pt.imshow(word_img, fnum=fnum, title='Word %r/%r' % (wx, len(inva.vocab)))
            >>>     pt.update()
            >>> ut.show_if_requested()
        """
        import plottool as pt
        # Create the contributing patch image
        word_patches = inva.get_patches(wx)
        stacked_patches = vt.stack_square_images(word_patches)

        # Create the average word image
        word = inva.vocab.wx_to_word[wx]
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
        patch_img2 = vt.inverted_sift_patch(word)
        patch_img = vt.rectify_to_uint8(patch_img)
        patch_img2 = vt.rectify_to_uint8(patch_img2)
        solidbar = vt.rectify_to_uint8(solidbar)
        stacked_patches = vt.rectify_to_uint8(stacked_patches)
        patch_img2, patch_img = vt.make_channels_comparable(patch_img2, patch_img)
        img_list = [patch_img, solidbar, patch_img2, solidbar, stacked_patches]
        word_img = vt.stack_image_list(img_list, vert=False, modifysize=True)
        return word_img

    def render_vocab(vocab, inva, use_data=False):
        """
        Renders the average patch of each word.
        This is a quick visualization of the entire vocabulary.

        CommandLine:
            python -m ibeis.algo.smk.vocab_indexer render_vocab --show
            python -m ibeis.algo.smk.vocab_indexer render_vocab --show --use-data
            python -m ibeis.algo.smk.vocab_indexer render_vocab --show --debug-depc

        Example:
            >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva('PZ_MTEST', num_words=10000)
            >>> use_data = ut.get_argflag('--use-data')
            >>> vocab = inva.vocab
            >>> all_words = vocab.render_vocab(inva, use_data=use_data)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
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
            import plottool as pt
            word_patch = pt.render_sift_on_patch(word_patch, word)
            word_patch_list.append(word_patch)

        #for wx, p in zip(wx_list, word_patch_list):
        #    inva._word_patches[wx] = p
        all_words = vt.stack_square_images(word_patch_list)
        return all_words


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
            return '%snA=%r nF=%r' % (name, ut.safelen(fstack.ax_to_aid), fstack.num_feat)

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

    def build(inva):
        print('building inverted index')
        nAssign = inva.config.get('nAssign', 1)
        fstack = inva.fstack
        vocab = inva.vocab
        idx_to_wxs, idx_to_maws = vocab.assign_to_words(fstack.idx_to_vec, nAssign)
        wx_to_idxs, wx_to_maws = vocab.invert_assignment(idx_to_wxs, idx_to_maws)
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

    def inverted_annots(inva, aids):
        if inva.grouped_annots is None:
            raise ValueError('grouped annots not computed')
        ax_list = ut.take(inva.fstack.aid_to_ax, aids)
        return ut.take(inva.grouped_annots, ax_list)

    @classmethod
    def invert(cls, fstack, vocab, nAssign=1):
        inva = cls(fstack, vocab, config={'nAssign': nAssign})
        inva.build()
        return inva

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

    def __nice__(inva):
        fstack = inva.fstack
        name = '' if fstack.name is None else fstack.name + ' '
        if inva.wx_to_idxs is None:
            return '%sNotInitialized' % (name,)
        else:
            return '%snW=%r mean=%.2f' % (name, ut.safelen(inva.wx_to_idxs), inva.perword_stats['mean'])

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

    def get_grouped_rvecs(inva, wx):
        """
        # Get all residual vectors assigned to a word grouped by annotation
        """
        rvecs_list, error_flags = inva.compute_rvecs(wx)
        ax_list = inva.wx_to_axs(wx)
        maw_list = inva.wx_to_maws[wx]
        # group residual vectors by annotation
        unique_ax, groupxs = vt.group_indices(ax_list)
        # (weighted aggregation with multi-assign-weights)
        grouped_maws  = vt.apply_grouping(maw_list, groupxs)
        grouped_rvecs = vt.apply_grouping(rvecs_list, groupxs)
        grouped_errors = vt.apply_grouping(error_flags, groupxs)

        # ~~Remove vectors with errors~~
        #inner_flags = vt.apply_grouping(~error_flags, groupxs)
        #grouped_rvecs2_ = vt.zipcompress(grouped_rvecs, inner_flags, axis=0)
        #grouped_maws2_  = vt.zipcompress(grouped_maws, inner_flags)
        grouped_rvecs2_ = grouped_rvecs
        grouped_maws2_ = grouped_maws
        grouped_errors2_ = grouped_errors

        outer_flags = [len(rvecs) > 0 for rvecs in grouped_rvecs2_]
        grouped_rvecs3_ = ut.compress(grouped_rvecs2_, outer_flags)
        grouped_maws3_ = ut.compress(grouped_maws2_, outer_flags)
        grouped_errors3_ = ut.compress(grouped_errors2_, outer_flags)
        unique_ax3_ = ut.compress(unique_ax, outer_flags)
        return unique_ax3_, grouped_rvecs3_, grouped_maws3_, grouped_errors3_

    def compute_rvecs_agg(inva, wx):
        """
        Sums and normalizes all rvecs that belong to the same word and the same
        annotation id

        Returns:
            ax_to_aggs: A mapping from an annotation to its aggregated vector
                for this word and an error flag.

        Notes:
            aggvec = Phi
            Phi(X_c) = psi(sum([phi(x) for x in X_c]))
            psi is the identify function currently.
            Phi is esentially a VLAD vector for a specific word.
            Returns Phi vectors wrt each annotation.

        Example:
            >>> # ENABLE_LATER
            >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
            >>> ibs, aid_list, inva = testdata_inva(num_words=256)
            >>> wx = 1
            >>> asint = False
            >>> ax_to_aggs = inva.compute_rvecs_agg(1)
        """
        tup = inva.get_grouped_rvecs(wx)
        unique_ax3_, grouped_rvecs3_, grouped_maws3_, grouped_errors3_ = tup
        ax_to_aggs = {
            ax: aggregate_rvecs(rvecs, maws, errors)
            for ax, rvecs, maws, errors in
            zip(unique_ax3_, grouped_rvecs3_, grouped_maws3_, grouped_errors3_)
        }
        return ax_to_aggs

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

        for wx in ut.ProgIter(inva.wx_list, lbl='Group Annots', bs=True):
            idxs = inva.wx_to_idxs[wx]
            maws = inva.wx_to_maws[wx]
            axs = fstack.idx_to_ax.take(idxs, axis=0)

            unique_axs, groupxs = vt.group_indices(axs)
            grouped_maws = vt.apply_grouping(maws, groupxs)
            grouped_idxs = vt.apply_grouping(idxs, groupxs)

            # precompute here
            ax_to_aggs = inva.compute_rvecs_agg(wx)
            for ax, idxs_, maws_ in zip(unique_axs, grouped_idxs,
                                        grouped_maws):
                X = ax_to_X[ax]
                X.wx_to_idxs_[wx] = idxs_
                X.wx_to_maws_[wx] = maws_
                X.wx_to_aggs[wx] = ax_to_aggs[ax]
        X_list = ax_to_X
        return X_list

    # def group_annots(inva):
    #     if inva.X_list is None:
    #         inva.X_list = inva.compute_annot_groups()
    #     return inva.X_list

    def assert_self(inva):
        assert sorted(inva.wx_to_idxs.keys()) == sorted(inva.wx_to_maws.keys())
        assert sorted(inva.wx_list) == sorted(inva.wx_to_maws.keys())
        inva.fstack.assert_self()


@ut.reloadable_class
class IndexedAnnot(ut.NiceRepr):
    """
    Example:
        >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
        >>> ibs, smk, qreq_ = testdata_smk()
        >>> inva = qreq_.qinva
        >>> X_list = inva.grouped_annots
        >>> X = X_list[0]
        >>> X.assert_self()
    """
    def __init__(X, ax, inva):
        X.ax = ax
        X.inva = inva
        # only idxs that belong to ax
        X.wx_to_idxs_ = {}
        X.wx_to_maws_ = {}
        X.wx_to_aggs = {}
        X.wx_to_phis = {}

    def __getstate__(self):
        return ut.delete_dict_keys(self.__dict__.copy(), 'inva')

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
            vocab = X.inva.vocab
            idxs = X.idxs(wx)
            vecs = X.inva.fstack.idx_to_vec.take(idxs, axis=0)
            word = vocab.wx_to_word[wx]
            rvecs, flags = compute_rvec(vecs, word)
            return rvecs, flags

    @property
    def words(X):
        return list(X.wx_to_aggs.keys())

    def assert_self(X):
        assert len(X.wx_to_idxs_) == len(X.wx_to_aggs)
        for wx in X.wx_to_idxs_.keys():
            axs = X.inva.fstack.idx_to_ax.take(X.idxs(wx), axis=0)
            assert np.all(axs == X.ax)
            rvecs, flags = X.phis_flags(wx)
            maws = X.maws(wx)
            rvec_agg, flag_agg = aggregate_rvecs(rvecs, maws, flags)
            assert np.all(rvec_agg == X.Phi(wx))


def compute_rvec(vecs, word, asint=False):
    """
    Compute residual vectors phi(x_c)

    Subtract each vector from its quantized word to get the resiudal, then
    normalize residuals to unit length.
    """
    rvecs = np.subtract(word.astype(np.float), vecs.astype(np.float))
    # If a vec is a word then the residual is 0 and it cant be L2 noramlized.
    is_zero = np.all(rvecs == 0, axis=1)
    vt.normalize_rows(rvecs, out=rvecs)
    # reset these values back to zero
    if np.any(is_zero):
        rvecs[is_zero, :] = 0
    if asint:
        rvecs = np.clip(np.round(rvecs * 255.0), -127, 127).astype(np.int8)
    # Determine if any errors occurred
    # FIXME: zero will drive the score of a match to 0 even though if they
    # are both 0, then it is an exact match and should be scored as a 1.
    error_flags = np.any(np.isnan(rvecs), axis=1)
    return rvecs, error_flags


def aggregate_rvecs(rvecs, maws, error_flags, asint=False):
    r"""
    Compute aggregated residual vectors Phi(X_c)
    """
    # Propogate errors from previous step
    flags_agg = np.any(error_flags, axis=0, keepdims=True)
    if rvecs.shape[0] == 0:
        rvecs_agg = np.empty((0, rvecs.shape[1]), dtype=np.float)
    if rvecs.shape[0] == 1:
        rvecs_agg = rvecs
    else:
        # Prealloc sum output (do not assign the result of sum)
        rvecs_agg = np.empty((1, rvecs.shape[1]), dtype=np.float)
        out = rvecs_agg[0]
        # Take weighted average of multi-assigned vectors
        weighted_sum = (maws[:, None] * rvecs).sum(axis=0, out=out)
        total_weight = maws.sum()
        is_zero = np.all(rvecs_agg == 0, axis=1)
        rvecs_agg = np.divide(weighted_sum, total_weight, out=rvecs_agg)
        vt.normalize_rows(rvecs_agg, out=rvecs_agg)
        if np.any(is_zero):
            # Add in arrors from this step
            rvecs_agg[is_zero, :] = 0
            flags_agg[is_zero] = True
        if asint:
            rvecs_agg = np.clip(np.round(rvecs_agg * 255.0), -127, 127).astype(np.int8)
        else:
            rvecs_agg = rvecs_agg
    return rvecs_agg, flags_agg


def weight_multi_assigns(_idx_to_wx, _idx_to_wdist, massign_alpha=1.2,
                         massign_sigma=80.0, massign_equal_weights=False):
    r"""
    Multi Assignment Weight Filtering from Improving Bag of Features

    Args:
        massign_equal_weights (): Turns off soft weighting. Gives all assigned
            vectors weight 1

    Returns:
        tuple : (idx_to_wxs, idx_to_maws)

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
    if _idx_to_wx.shape[1] <= 1:
        idx_to_wxs = _idx_to_wx.tolist()
        idx_to_maws = [[1.0]] * len(idx_to_wxs)
    else:
        # Valid word assignments are beyond fraction of distance to the nearest word
        massign_thresh = _idx_to_wdist.T[0:1].T.copy()
        # HACK: If the nearest word has distance 0 then this threshold is too hard
        # so we should use the distance to the second nearest word.
        EXACT_MATCH_HACK = True
        if EXACT_MATCH_HACK:
            flag_too_close = (massign_thresh == 0)
            massign_thresh[flag_too_close] = _idx_to_wdist.T[1:2].T[flag_too_close]
        # Compute the threshold fraction
        epsilon = .001
        np.add(epsilon, massign_thresh, out=massign_thresh)
        np.multiply(massign_alpha, massign_thresh, out=massign_thresh)
        # Mark assignments as invalid if they are too far away from the nearest assignment
        invalid = np.greater_equal(_idx_to_wdist, massign_thresh)
        if ut.VERBOSE:
            nInvalid = (invalid.size - invalid.sum(), invalid.size)
            print('[maw] + massign_alpha = %r' % (massign_alpha,))
            print('[maw] + massign_sigma = %r' % (massign_sigma,))
            print('[maw] + massign_equal_weights = %r' % (massign_equal_weights,))
            print('[maw] * Marked %d/%d assignments as invalid' % nInvalid)

        if massign_equal_weights:
            # Performance hack from jegou paper: just give everyone equal weight
            masked_wxs = np.ma.masked_array(_idx_to_wx, mask=invalid)
            idx_to_wxs  = list(map(ut.filter_Nones, masked_wxs.tolist()))
            if ut.DEBUG2:
                assert all([isinstance(wxs, list) for wxs in idx_to_wxs])
            idx_to_maws = [np.ones(len(wxs), dtype=np.float32) for wxs in idx_to_wxs]
        else:
            # More natural weighting scheme
            # Weighting as in Lost in Quantization
            gauss_numer = np.negative(_idx_to_wdist.astype(np.float64))
            gauss_denom = 2 * (massign_sigma ** 2)
            gauss_exp   = np.divide(gauss_numer, gauss_denom)
            unnorm_maw = np.exp(gauss_exp)
            # Mask invalid multiassignment weights
            masked_unorm_maw = np.ma.masked_array(unnorm_maw, mask=invalid)
            # Normalize multiassignment weights from 0 to 1
            masked_norm = masked_unorm_maw.sum(axis=1)[:, np.newaxis]
            masked_maw = np.divide(masked_unorm_maw, masked_norm)
            masked_wxs = np.ma.masked_array(_idx_to_wx, mask=invalid)
            # Remove masked weights and word indexes
            idx_to_wxs  = list(map(ut.filter_Nones, masked_wxs.tolist()))
            idx_to_maws = list(map(ut.filter_Nones, masked_maw.tolist()))
            #with ut.EmbedOnException():
            if ut.DEBUG2:
                checksum = [sum(maws) for maws in idx_to_maws]
                for x in np.where([not ut.almost_eq(val, 1) for val in checksum])[0]:
                    print(checksum[x])
                    print(_idx_to_wx[x])
                    print(masked_wxs[x])
                    print(masked_maw[x])
                    print(massign_thresh[x])
                    print(_idx_to_wdist[x])
                #all([ut.almost_eq(x, 1) for x in checksum])
                assert all([ut.almost_eq(val, 1) for val in checksum]), (
                    'weights did not break evenly')
    return idx_to_wxs, idx_to_maws


@derived_attribute(
    tablename='vocab', parents=['feat*'],
    colnames=['words'], coltypes=[VisualVocab],
    configclass=VocabConfig, chunksize=1, fname='visual_vocab',
    vectorized=False,
)
def compute_vocab(depc, fid_list, config):
    r"""
    Depcache method for computing a new visual vocab

    CommandLine:
        python -m ibeis.core_annots --exec-compute_neighbor_index --show
        python -m ibeis.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=neighbor_index

        python -m ibeis.algo.smk.vocab_indexer --exec-compute_vocab:0
        python -m ibeis.algo.smk.vocab_indexer --exec-compute_vocab:1

        # FIXME make util_tests register
        python -m ibeis.algo.smk.vocab_indexer compute_vocab:0

    Ignore:
        ibs.depc['vocab'].print_table()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
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
        >>> assert vocab.wordflann._FLANN__curindex_data is vocab.wx_to_word

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> fid_list = depc.get_rowids('feat', aid_list)
        >>> config = VocabConfig()
        >>> vocab, train_vecs = ut.exec_func_src(compute_vocab, key_list=['vocab', 'train_vecs'])
        >>> idx_to_vec = depc.d.get_feat_vecs(aid_list)[0]
        >>> self = vocab
        >>> ut.quit_if_noshow()
        >>> data = train_vecs
        >>> centroids = vocab.wx_to_word
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
    vocab.build()
    return (vocab,)


def testdata_inva(defaultdb='testdb1', **kwargs):
    """
    >>> from ibeis.algo.smk.vocab_indexer import *  # NOQA
    >>> args, kwargs = tuple(), dict()
    """
    import ibeis
    ibs, aid_list = ibeis.testdata_aids(defaultdb=defaultdb)
    config = VocabConfig(**kwargs)
    vocab = ibs.depc_annot.get('vocab', [aid_list], 'words', config=config)[0]
    fstack = ForwardIndex(ibs, aid_list)
    inva = InvertedIndex(fstack, vocab, config={'nAssign': 3})
    fstack.build()
    inva.build()
    return ibs, aid_list, inva


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/ibeis/ibeis/algo/smk
        python ~/code/ibeis/ibeis/algo/smk/vocab_indexer.py
        python ~/code/ibeis/ibeis/algo/smk/vocab_indexer.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
