# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range
from wbia import dtool
import utool as ut
import vtool as vt
import numpy as np
from wbia.algo.smk import smk_funcs
from wbia.control.controller_inject import register_preprocs

(print, rrr, profile) = ut.inject2(__name__)


derived_attribute = register_preprocs['annot']


class InvertedIndexConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('nAssign', 1),
        # ut.ParamInfo('int_rvec', False, hideif=False),
        ut.ParamInfo('int_rvec', True, hideif=False),
        ut.ParamInfo('massign_equal,', False),
        ut.ParamInfo('massign_alpha,', 1.2),
        # ut.ParamInfo('massign_sigma,', 80.0, hideif=lambda cfg: cfg['massign_equal']),
        ut.ParamInfo('inva_version', 2),
        #
        # massign_sigma=80.0,
        # massign_equal_weights=False
    ]


class InvertedAnnotsExtras(object):
    def get_size_info(inva):
        import sys

        def get_homog_list_nbytes_scalar(list_scalar):
            if list_scalar is None:
                return 0
            if len(list_scalar) == 0:
                return 0
            else:
                val = list_scalar[0]
                item_nbytes = ut.get_object_nbytes(val)
                return item_nbytes * len(list_scalar)

        def get_homog_list_nbytes_nested(list_nested):
            if list_nested is None:
                return 0
            if len(list_nested) == 0:
                return 0
            else:
                val = list_nested[0]
                if isinstance(val, np.ndarray):
                    nbytes = sum(sys.getsizeof(v) for v in list_nested)
                    # item_nbytes = sum(v.nbytes for v in list_nested)
                else:
                    nest_nbytes = sys.getsizeof(val) * len(list_nested)
                    totals = sum(ut.lmap(len, list_nested))
                    item_nbytes = sys.getsizeof(val[0]) * totals
                    nbytes = nest_nbytes + item_nbytes
                return nbytes

        def get_homog_dict_nbytes_nested(dict_nested):
            if dict_nested is None:
                return 0
            wxkeybytes = get_homog_list_nbytes_scalar(list(dict_nested.keys()))
            wxvalbytes = get_homog_list_nbytes_nested(list(dict_nested.values()))
            wxbytes = wxkeybytes + wxvalbytes + sys.getsizeof(dict_nested)
            return wxbytes

        def get_homog_dict_nbytes_scalar(dict_scalar):
            if dict_scalar is None:
                return 0
            wxkeybytes = get_homog_list_nbytes_scalar(list(dict_scalar.keys()))
            wxvalbytes = get_homog_list_nbytes_scalar(list(dict_scalar.values()))
            wxbytes = wxkeybytes + wxvalbytes + sys.getsizeof(dict_scalar)
            return wxbytes

        sizes = {
            'aids': get_homog_list_nbytes_scalar(inva.aids),
            'wx_lists': get_homog_list_nbytes_nested(inva.wx_lists),
            'fxs_lists': get_homog_list_nbytes_nested(inva.fxs_lists),
            'maws_lists': get_homog_list_nbytes_nested(inva.maws_lists),
            'agg_rvecs': get_homog_list_nbytes_nested(inva.agg_rvecs),
            'agg_flags': get_homog_list_nbytes_nested(inva.agg_flags),
            'aid_to_idx': get_homog_dict_nbytes_scalar(inva.aid_to_idx),
            'gamma_list': get_homog_list_nbytes_scalar(inva.gamma_list),
            'wx_to_aids': get_homog_dict_nbytes_nested(inva.wx_to_aids),
            'wx_to_weight': get_homog_dict_nbytes_scalar(inva.wx_to_weight),
        }
        return sizes

    def print_size_info(inva):
        sizes = inva.get_size_info()
        sizes = ut.sort_dict(sizes, 'vals', ut.identity)
        total_nbytes = sum(sizes.values())
        print(
            ut.align(ut.repr3(ut.map_dict_vals(ut.byte_str2, sizes), strvals=True), ':')
        )
        print('total_nbytes = %r' % (ut.byte_str2(total_nbytes),))

    def get_nbytes(inva):
        sizes = inva.get_size_info()
        total_nbytes = sum(sizes.values())
        return total_nbytes

    def get_word_patch(inva, wx, ibs):
        if not hasattr(inva, 'word_patches'):
            inva._word_patches = {}
        if wx not in inva._word_patches:
            assigned_patches = inva.get_patches(wx, ibs, verbose=False)
            # print('assigned_patches = %r' % (len(assigned_patches),))
            average_patch = np.mean(assigned_patches, axis=0)
            average_patch = average_patch.astype(np.float)
            inva._word_patches[wx] = average_patch
        return inva._word_patches[wx]

    def get_patches(inva, wx, ibs, verbose=True):
        """
        Loads the patches assigned to a particular word in this stack

        >>> inva.wx_to_aids = inva.compute_inverted_list()
        >>> verbose=True
        """
        config = inva.config
        aid_list = inva.wx_to_aids[wx]
        X_list = [inva.get_annot(aid) for aid in aid_list]
        fxs_groups = [X.fxs(wx) for X in X_list]
        all_kpts_list = ibs.depc.d.get_feat_kpts(aid_list, config=config)
        sub_kpts_list = vt.ziptake(all_kpts_list, fxs_groups, axis=0)
        total_patches = sum(ut.lmap(len, fxs_groups))

        chip_list = ibs.depc_annot.d.get_chips_img(aid_list, config=config)
        # convert to approprate colorspace
        # if colorspace is not None:
        #    chip_list = vt.convert_image_list_colorspace(chip_list, colorspace)
        # ut.print_object_size(chip_list, 'chip_list')

        patch_size = 64
        shape = (total_patches, patch_size, patch_size, 3)
        _prog = ut.ProgPartial(enabled=verbose, lbl='warping patches', bs=True)
        _patchiter = ut.iflatten(
            [
                vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
                # vt.get_warped_patches(chip, kpts, patch_size=patch_size, use_cpp=True)[0]
                for chip, kpts in _prog(
                    zip(chip_list, sub_kpts_list), length=len(aid_list)
                )
            ]
        )
        word_patches = vt.fromiter_nd(_patchiter, shape, dtype=np.uint8)
        return word_patches

    def render_inverted_vocab_word(inva, wx, ibs, fnum=None):
        """
        Creates a visualization of a visual word. This includes the average patch,
        the SIFT-like representation of the centroid, and some of the patches that
        were assigned to it.

        CommandLine:
            python -m wbia.algo.smk.inverted_index render_inverted_vocab_word --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.smk.inverted_index import *  # NOQA
            >>> import wbia.plottool as pt
            >>> qreq_, inva = testdata_inva()
            >>> ibs = qreq_.ibs
            >>> wx_list = list(inva.wx_to_aids.keys())
            >>> wx = wx_list[0]
            >>> ut.qtensure()
            >>> fnum = 2
            >>> fnum = pt.ensure_fnum(fnum)
            >>> # Interactive visualization of many words
            >>> for wx in ut.InteractiveIter(wx_list):
            >>>     word_img = inva.render_inverted_vocab_word(wx, ibs, fnum)
            >>>     pt.imshow(word_img, fnum=fnum, title='Word %r/%r' % (wx, '?'))
            >>>     pt.update()
        """
        import wbia.plottool as pt

        # Create the contributing patch image
        word_patches = inva.get_patches(wx, ibs)
        word_patches_ = ut.strided_sample(word_patches, 64)
        stacked_patches = vt.stack_square_images(word_patches_)

        # Create the average word image
        vocab = ibs.depc['vocab'].get_row_data([inva.vocab_rowid], 'words')[0]
        word = vocab.wx_to_word[wx]

        average_patch = np.mean(word_patches, axis=0)
        # vecs = inva.get_vecs(wx)
        # assert np.allclose(word, vecs.mean(axis=0))
        with_sift = True
        if with_sift:
            patch_img = pt.render_sift_on_patch(average_patch, word)
        else:
            patch_img = average_patch

        # Stack them together
        solidbar = np.zeros(
            (patch_img.shape[0], int(patch_img.shape[1] * 0.1), 3), dtype=patch_img.dtype,
        )
        border_color = (100, 10, 10)  # bgr, darkblue
        if ut.is_float(solidbar):
            solidbar[:, :, :] = (np.array(border_color) / 255)[None, None]
        else:
            solidbar[:, :, :] = np.array(border_color)[None, None]
        patch_img2 = vt.inverted_sift_patch(word)
        # Fix types
        patch_img = vt.rectify_to_uint8(patch_img)
        patch_img2 = vt.rectify_to_uint8(patch_img2)
        solidbar = vt.rectify_to_uint8(solidbar)
        stacked_patches = vt.rectify_to_uint8(stacked_patches)
        # Stack everything together
        patch_img2, patch_img = vt.make_channels_comparable(patch_img2, patch_img)
        img_list = [patch_img, solidbar, patch_img2, solidbar, stacked_patches]
        word_img = vt.stack_image_list(img_list, vert=False, modifysize=True)
        return word_img

    def render_inverted_vocab(inva, ibs, use_data=False):
        """
        Renders the average patch of each word.
        This is a visualization of the entire vocabulary.

        CommandLine:
            python -m wbia.algo.smk.inverted_index render_inverted_vocab --show
            python -m wbia.algo.smk.inverted_index render_inverted_vocab --show --use-data
            python -m wbia.algo.smk.inverted_index render_inverted_vocab --show --debug-depc

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.smk.inverted_index import *  # NOQA
            >>> qreq_, inva = testdata_inva()
            >>> ibs = qreq_.ibs
            >>> all_words = inva.render_inverted_vocab(ibs)
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> pt.qt4ensure()
            >>> pt.imshow(all_words)
            >>> ut.show_if_requested()
        """
        import wbia.plottool as pt

        # Get words with the most assignments
        vocab = ibs.depc['vocab'].get_row_data([inva.vocab_rowid], 'words')[0]

        wx_list = ut.strided_sample(inva.wx_list, 64)

        word_patch_list = []
        for wx in ut.ProgIter(wx_list, bs=True, lbl='building patches'):
            word = vocab.wx_to_word[wx]
            word_patch = inva.get_word_patch(wx, ibs)
            word_patch = pt.render_sift_on_patch(word_patch, word)
            word_patch_list.append(word_patch)

        all_words = vt.stack_square_images(word_patch_list)
        return all_words


@ut.reloadable_class
class InvertedAnnots(InvertedAnnotsExtras):
    """
    CommandLine:
        python -m wbia.algo.smk.inverted_index InvertedAnnots --show

    Ignore:
        >>> from wbia.algo.smk.inverted_index import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='Oxford', a='oxford',
        >>>                              p='default:proot=smk,nAssign=1,num_words=64000')
        >>> config = qreq_.qparams
        >>> ibs = qreq_.ibs
        >>> depc = qreq_.ibs.depc
        >>> aids = qreq_.daids
        >>> aids = qreq_.qaids
        >>> input_tuple = (aids, [qreq_.daids])
        >>> inva = ut.DynStruct()
        >>> inva = InvertedAnnots(aids, qreq_)

    Example:
        >>> # DISABLE_DOCTEST
        >>> qreq_, inva = testdata_inva()
    """

    def __init__(inva):
        inva.aids = None
        inva.wx_lists = None
        inva.fxs_lists = None
        inva.agg_rvecs = None
        inva.agg_flags = None
        inva.aid_to_idx = None
        inva.gamma_list = None
        inva.wx_to_weight = None
        inva.wx_to_aids = None
        inva.int_rvec = None
        inva.config = None
        inva.vocab_rowid = None

    @property
    def wx_list(inva):
        wx = sorted(ut.flat_unique(*inva.wx_lists))
        return wx

    # @classmethod
    # def from_qreq(cls, aids, qreq_, isdata=False):
    #     print('Loading up inverted assigments')
    #     depc = qreq_.ibs.depc
    #     vocab_aids = qreq_.daids
    #     config = qreq_.qparams
    #     inva = cls.from_depc(depc, aids, vocab_aids, config)
    #     return inva

    @classmethod
    def from_depc(cls, depc, aids, vocab_aids, config):
        inva = cls()
        vocab_rowid = depc.get_rowids('vocab', (vocab_aids,), config=config)[0]
        inva.vocab_rowid = vocab_rowid
        tablename = 'inverted_agg_assign'
        table = depc[tablename]
        input_tuple = (aids, [vocab_rowid] * len(aids))
        tbl_rowids = depc.get_rowids(
            tablename, input_tuple, config=config, _hack_rootmost=True, _debug=False
        )
        # input_tuple = (aids, [vocab_aids])
        # tbl_rowids = depc.get_rowids(tablename, input_tuple, config=config)
        print('Reading data')
        inva.aids = aids
        inva.wx_lists = [
            np.array(wx_list_, dtype=np.int32)
            for wx_list_ in table.get_row_data(tbl_rowids, 'wx_list', showprog='load wxs')
        ]
        inva.fxs_lists = [
            [np.array(fxs, dtype=np.uint16) for fxs in fxs_list]
            for fxs_list in table.get_row_data(
                tbl_rowids, 'fxs_list', showprog='load fxs'
            )
        ]
        inva.maws_lists = [
            [np.array(m, dtype=np.float32) for m in maws]
            for maws in table.get_row_data(tbl_rowids, 'maws_list', showprog='load maws')
        ]
        inva.agg_rvecs = table.get_row_data(
            tbl_rowids, 'agg_rvecs', showprog='load agg_rvecs'
        )
        inva.agg_flags = table.get_row_data(
            tbl_rowids, 'agg_flags', showprog='load agg_flags'
        )
        # less memory hogs
        inva.aid_to_idx = ut.make_index_lookup(inva.aids)
        inva.int_rvec = config['int_rvec']
        inva.gamma_list = None
        # Inverted list
        inva.wx_to_weight = None
        inva.wx_to_aids = None
        inva.config = config
        return inva

    def _assert_self(inva, qreq_):
        ibs = qreq_.ibs
        assert len(inva.aids) == len(inva.wx_lists)
        assert len(inva.aids) == len(inva.fxs_lists)
        assert len(inva.aids) == len(inva.maws_lists)
        assert len(inva.aids) == len(inva.agg_rvecs)
        assert len(inva.aids) == len(inva.agg_flags)
        nfeat_list1 = ibs.get_annot_num_feats(inva.aids, config2_=qreq_.qparams)
        nfeat_list2 = [sum(ut.lmap(len, fx_list)) for fx_list in inva.fxs_lists]
        nfeat_list3 = [sum(ut.lmap(len, maws)) for maws in inva.maws_lists]
        ut.assert_lists_eq(nfeat_list1, nfeat_list2)
        ut.assert_lists_eq(nfeat_list1, nfeat_list3)

    def __getstate__(inva):
        state = inva.__dict__
        return state

    def __setstate__(inva, state):
        inva.__dict__.update(**state)

    @profile
    @ut.memoize
    def get_annot(inva, aid):
        idx = inva.aid_to_idx[aid]
        X = SingleAnnot.from_inva(inva, idx)
        return X

    def compute_inverted_list(inva):
        with ut.Timer('Building inverted list'):
            wx_to_aids = smk_funcs.invert_lists(inva.aids, inva.wx_lists)
            return wx_to_aids

    @profile
    def compute_word_weights(inva, method='idf'):
        """
        Compute a per-word weight like idf

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.smk.inverted_index import *  # NOQA
            >>> qreq_, inva = testdata_inva()
            >>> wx_to_weight = inva.compute_word_weights()
            >>> print('wx_to_weight = %r' % (wx_to_weight,))
        """
        wx_list = sorted(inva.wx_to_aids.keys())
        with ut.Timer('Computing %s weights' % (method,)):
            if method == 'idf':
                ndocs_total = len(inva.aids)
                # Unweighted documents
                ndocs_per_word = np.array(
                    [len(set(inva.wx_to_aids[wx])) for wx in wx_list]
                )
                weight_per_word = smk_funcs.inv_doc_freq(ndocs_total, ndocs_per_word)
            elif method == 'idf-maw':
                # idf denom (the num of docs containing a word for each word)
                # The max(maws) denote the prob that this word indexes an annot
                ndocs_total = len(inva.aids)
                # Weighted documents
                wx_to_ndocs = {wx: 0.0 for wx in wx_list}
                for wx, maws in zip(
                    ut.iflatten(inva.wx_lists), ut.iflatten(inva.maws_lists)
                ):
                    wx_to_ndocs[wx] += min(1.0, max(maws))
                ndocs_per_word = ut.take(wx_to_ndocs, wx_list)
                weight_per_word = smk_funcs.inv_doc_freq(ndocs_total, ndocs_per_word)
            elif method == 'uniform':
                weight_per_word = np.ones(len(wx_list))
            wx_to_weight = dict(zip(wx_list, weight_per_word))
            wx_to_weight = ut.DefaultValueDict(0, wx_to_weight)
        return wx_to_weight

    @profile
    def compute_gammas(inva, alpha, thresh):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.smk.inverted_index import *  # NOQA
            >>> qreq_, inva = testdata_inva()
            >>> inva.wx_to_weight = inva.compute_word_weights('uniform')
            >>> alpha = 3.0
            >>> thresh = 0.0
            >>> gamma_list = inva.compute_gammas(alpha, thresh)
        """
        # TODO: sep
        wx_to_weight = inva.wx_to_weight
        _prog = ut.ProgPartial(
            length=len(inva.wx_lists), bs=True, lbl='gamma', adjust=True
        )
        _iter = zip(inva.wx_lists, inva.agg_rvecs, inva.agg_flags)
        gamma_list = []
        for wx_list, phiX_list, flagsX_list in _prog(_iter):
            if inva.int_rvec:
                phiX_list = smk_funcs.uncast_residual_integer(phiX_list)
            weight_list = np.array(ut.take(wx_to_weight, wx_list))
            gammaX = smk_funcs.gamma_agg(
                phiX_list, flagsX_list, weight_list, alpha, thresh
            )
            gamma_list.append(gammaX)
        return gamma_list


@ut.reloadable_class
class SingleAnnot(ut.NiceRepr):
    def __init__(X):
        X.aid = None
        X.wx_list = None
        X.fxs_list = None
        X.maws_list = None
        X.agg_rvecs = None
        X.agg_flags = None
        X.gamma = None
        X.wx_to_idx = None
        X.int_rvec = None
        X.wx_set = None

    def __nice___(X):
        return '%s' % (X.aid,)

    @classmethod
    def from_inva(cls, inva, idx):
        X = cls()
        X.aid = inva.aids[idx]
        X.wx_list = inva.wx_lists[idx]
        X.fxs_list = inva.fxs_lists[idx]
        X.maws_list = inva.maws_lists[idx]
        X.agg_rvecs = inva.agg_rvecs[idx]
        X.agg_flags = inva.agg_flags[idx]
        if inva.gamma_list is not None:
            X.gamma = inva.gamma_list[idx]
        X.wx_to_idx = ut.make_index_lookup(X.wx_list)
        X.int_rvec = inva.int_rvec
        X.wx_set = set(X.wx_list)
        return X

    def to_dense(X, inva=None, out=None):
        if out is None:
            assert inva is not None
            n_words = inva.wx_list[-1] + 1
            n_dims = X.agg_rvecs.shape[1]
            out = np.zeros((n_words * n_dims), dtype=np.float32)
        # out[X.wx_list] = X.Phis_flags(range(len(X.wx_list)))[0]
        out[X.wx_list] = X.agg_rvecs
        return out

    @property
    def words(X):
        return X.wx_set

    @profile
    def fxs(X, c):
        idx = X.wx_to_idx[c]
        fxs = X.fxs_list[idx]
        return fxs

    @profile
    def maws(X, c):
        idx = X.wx_to_idx[c]
        maws = X.maws_list[idx]
        return maws

    def phis_flags_list(X, idxs):
        """ get subset of non-aggregated residual vectors """
        phis_list = ut.take(X.rvecs_list, idxs)
        flags_list = ut.take(X.flags_list, idxs)
        if X.int_rvec:
            phis_list = ut.lmap(smk_funcs.uncast_residual_integer, phis_list)
        return phis_list, flags_list

    def Phis_flags(X, idxs):
        """ get subset of aggregated residual vectors """
        Phis = X.agg_rvecs.take(idxs, axis=0)
        flags = X.agg_flags.take(idxs, axis=0)
        if X.int_rvec:
            Phis = smk_funcs.uncast_residual_integer(Phis)
        return Phis, flags

    def _assert_self(X, qreq_, vocab):
        import utool as ut

        all_fxs = sorted(ut.flatten(X.fxs_list))
        assert len(all_fxs) > all_fxs[-1]
        assert len(all_fxs) == qreq_.ibs.get_annot_num_feats(X.aid, qreq_.config)

        nAssign = qreq_.qparams['nAssign']
        int_rvec = qreq_.qparams['int_rvec']
        # vocab = new_load_vocab(qreq_.ibs, qreq_.daids, qreq_.config)
        annots = qreq_.ibs.annots([X.aid], config=qreq_.config)
        vecs = annots.vecs[0]

        argtup = residual_args(vocab, vecs, nAssign, int_rvec)
        wx_list, word_list, fxs_list, maws_list, fx_to_vecs, int_rvec = argtup
        assert np.all(X.wx_list == wx_list)
        assert np.all([all(a == b) for a, b in zip(X.fxs_list, fxs_list)])
        assert np.all([all(a == b) for a, b in zip(X.maws_list, maws_list)])
        tup = residual_worker(argtup)
        (wx_list, fxs_list, maws_list, agg_rvecs, agg_flags) = tup
        assert np.all(X.agg_rvecs == agg_rvecs)
        assert np.all(X.agg_flags == agg_flags)
        assert X.agg_rvecs is not agg_rvecs
        assert X.agg_flags is not agg_flags

    def nbytes_info(X):
        size_info = ut.map_vals(ut.get_object_nbytes, X.__dict__)
        return size_info

    def nbytes(X):
        size_info = X.nbytes_info()
        nbytes = sum(size_info.values())
        return nbytes


@derived_attribute(
    tablename='inverted_agg_assign',
    parents=['feat', 'vocab'],
    colnames=[
        'wx_list',
        'fxs_list',
        'maws_list',
        # 'rvecs_list',
        # 'flags_list',
        'agg_rvecs',
        'agg_flags',
    ],
    coltypes=[
        list,
        list,
        list,
        # list, list,
        np.ndarray,
        np.ndarray,
    ],
    configclass=InvertedIndexConfig,
    fname='smk/smk_agg_rvecs',
    chunksize=256,
)
def compute_residual_assignments(depc, fid_list, vocab_id_list, config):
    r"""
    CommandLine:
        python -m wbia.control.IBEISControl show_depc_annot_table_input \
                --show --tablename=residuals

    Ignore:
        ibs.depc['vocab'].print_table()

    Ignore:
        data = ibs.depc.get('inverted_agg_assign', ([1, 2473], qreq_.daids), config=qreq_.config)
        wxs1 = data[0][0]
        wxs2 = data[1][0]

        # Lev Example
        import wbia
        ibs = wbia.opendb('Oxford')
        depc = ibs.depc
        table = depc['inverted_agg_assign']
        table.print_table()
        table.print_internal_info()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.inverted_index import *  # NOQA
        >>> # Test depcache access
        >>> import wbia
        >>> ibs, aid_list = wbia.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> config = {'num_words': 1000, 'nAssign': 1}
        >>> #input_tuple = (aid_list, [aid_list] * len(aid_list))
        >>> daids = aid_list
        >>> input_tuple = (daids, [daids])
        >>> rowid_kw = {}
        >>> tablename = 'inverted_agg_assign'
        >>> target_tablename = tablename
        >>> input_ids = depc.get_parent_rowids(tablename, input_tuple, config)
        >>> fid_list = ut.take_column(input_ids, 0)
        >>> vocab_id_list = ut.take_column(input_ids, 1)
        >>> data = depc.get(tablename, input_tuple, config)
        >>> tup = dat[1]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.inverted_index import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='Oxford', a='oxford', p='default:proot=smk,nAssign=1,num_words=64000')
        >>> config = {'num_words': 64000, 'nAssign': 1, 'int_rvec': True}
        >>> depc = qreq_.ibs.depc
        >>> daids = qreq_.daids
        >>> input_tuple = (daids, [daids])
        >>> rowid_kw = {}
        >>> tablename = 'inverted_agg_assign'
        >>> target_tablename = tablename
        >>> input_ids = depc.get_parent_rowids(tablename, input_tuple, config)
        >>> fid_list = ut.take_column(input_ids, 0)
        >>> vocab_id_list = ut.take_column(input_ids, 1)
    """
    # print('[IBEIS] ASSIGN RESIDUALS:')
    assert ut.allsame(vocab_id_list)
    vocabid = vocab_id_list[0]

    # NEED HACK TO NOT LOAD INDEXER EVERY TIME
    this_table = depc['inverted_agg_assign']
    vocab_table = depc['vocab']
    if (
        this_table._hack_chunk_cache is not None
        and vocabid in this_table._hack_chunk_cache
    ):
        vocab = this_table._hack_chunk_cache[vocabid]
    else:
        vocab = vocab_table.get_row_data([vocabid], 'words')[0]
        if this_table._hack_chunk_cache is not None:
            this_table._hack_chunk_cache[vocabid] = vocab

    print('Grab Vecs')
    vecs_list = depc.get_native('feat', fid_list, 'vecs')
    nAssign = config['nAssign']
    int_rvec = config['int_rvec']

    from concurrent import futures

    print('Building residual args')
    worker = residual_worker
    args_gen = gen_residual_args(vocab, vecs_list, nAssign, int_rvec)
    args_gen = [
        args for args in ut.ProgIter(args_gen, length=len(vecs_list), lbl='building args')
    ]
    # nprocs = ut.num_unused_cpus(thresh=10) - 1
    nprocs = ut.num_cpus()
    print('Creating %d processes' % (nprocs,))
    executor = futures.ProcessPoolExecutor(nprocs)
    try:
        print('Submiting workers')
        fs_chunk = [
            executor.submit(worker, args)
            for args in ut.ProgIter(args_gen, lbl='submit proc')
        ]
        for fs in ut.ProgIter(fs_chunk, lbl='getting phi result'):
            tup = fs.result()
            yield tup
    except Exception:
        raise
    finally:
        executor.shutdown(wait=True)


def gen_residual_args(vocab, vecs_list, nAssign, int_rvec):
    for vecs in vecs_list:
        argtup = residual_args(vocab, vecs, nAssign, int_rvec)
        yield argtup


def residual_args(vocab, vecs, nAssign, int_rvec):
    fx_to_vecs = vecs
    fx_to_wxs, fx_to_maws = smk_funcs.assign_to_words(vocab, fx_to_vecs, nAssign)
    wx_to_fxs, wx_to_maws = smk_funcs.invert_assigns(fx_to_wxs, fx_to_maws)
    wx_list = sorted(wx_to_fxs.keys())

    word_list = ut.take(vocab.wx_to_word, wx_list)
    fxs_list = ut.take(wx_to_fxs, wx_list)
    maws_list = ut.take(wx_to_maws, wx_list)
    argtup = (wx_list, word_list, fxs_list, maws_list, fx_to_vecs, int_rvec)
    return argtup


def residual_worker(argtup):
    wx_list, word_list, fxs_list, maws_list, fx_to_vecs, int_rvec = argtup
    if int_rvec:
        agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.int8)
    else:
        agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.float)
    agg_flags = np.empty((len(wx_list), 1), dtype=np.bool)

    # for idx, wx in enumerate(wx_list):
    for idx in range(len(wx_list)):
        # wx = wx_list[idx]
        word = word_list[idx]
        fxs = fxs_list[idx]
        maws = maws_list[idx]
        vecs = fx_to_vecs.take(fxs, axis=0)

        _rvecs, _flags = smk_funcs.compute_rvec(vecs, word)
        # rvecs = _rvecs  # NOQA
        # error_flags = _flags  # NOQA
        _agg_rvec, _agg_flag = smk_funcs.aggregate_rvecs(_rvecs, maws, _flags)
        # Cast to integers for storage
        if int_rvec:
            _agg_rvec = smk_funcs.cast_residual_integer(_agg_rvec)
        agg_rvecs[idx] = _agg_rvec
        agg_flags[idx] = _agg_flag

    tup = (wx_list, fxs_list, maws_list, agg_rvecs, agg_flags)
    return tup


def testdata_inva():
    """
    from wbia.algo.smk.inverted_index import *  # NOQA
    """
    import wbia

    qreq_ = wbia.testdata_qreq_(
        defaultdb='PZ_MTEST', a='default', p='default:proot=smk,nAssign=1,num_words=64'
    )
    aids = qreq_.daids
    cls = InvertedAnnots
    depc = qreq_.ibs.depc
    vocab_aids = qreq_.daids
    config = qreq_.qparams
    inva = cls.from_depc(depc, aids, vocab_aids, config)
    inva.wx_to_aids = inva.compute_inverted_list()
    return qreq_, inva


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.smk.inverted_index
        python -m wbia.algo.smk.inverted_index --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
