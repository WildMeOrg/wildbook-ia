from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import zip
import dtool
import utool as ut
import vtool as vt
import six
import numpy as np
# import cv2
from ibeis.control.controller_inject import register_preprocs, register_subprops
(print, rrr, profile) = ut.inject2(__name__, '[new_annots]')


derived_attribute = register_preprocs['annot']
register_subprop = register_subprops['annot']
# dtool.Config.register_func = derived_attribute


@six.add_metaclass(ut.ReloadingMetaclass)
class VisualVocab(ut.NiceRepr):

    def __init__(self):
        self.wx2_word = None

    def __nice__(self):
        return ' nW=%r' % (ut.safelen(self.wx2_word))

    @staticmethod
    def get_support(depc, aid_list, config):
        vecs_list = depc.get('feat', aid_list, 'vecs', config)
        if False and config['fg_on']:
            fgws_list = depc.get('featweight', aid_list, 'fgw', config)
        else:
            fgws_list = None
        return vecs_list, fgws_list

    def on_load(nnindexer, depc):
        pass

    def on_save(nnindexer, depc, fpath):
        pass

    def __getstate__(self):
        # TODO: Figure out how to make these play nice with the depcache
        state = self.__dict__
        del state['flann']
        del state['idx2_fgw']
        del state['idx2_vec']
        del state['idx2_ax']
        del state['idx2_fx']
        return state

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)


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
    ]
    _sub_config_list = [
    ]


@derived_attribute(
    tablename='vocab', parents=['feat*'],
    colnames=['words'], coltypes=[VisualVocab],
    configclass=VocabConfig,
    chunksize=1, fname='visual_vocab',
    single=True,
    # _internal_parent_ids=False,  # Give the function nicer ids to work with
    _internal_parent_ids=True,  # Give the function nicer ids to work with
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
        >>> from ibeis.core_annots import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> fid_list = depc.get_rowids('feat', aid_list)
        >>> config = VocabConfig()
        >>> centroids = words
        >>> data = train_vecs
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
    return words


def visualize_vocab(depc, words, aid_list):
    patches_list = extract_patches()
    vecs_list = depc.d.get_feat_vecs(aid_list)
    flann_params = {}
    flann = vt.build_flann_index(words, flann_params)

    wx2_assigned = ut.ddict(list)
    for aid, vecs in zip(aid_list, vecs_list):
        vx2_wx, vx2_dist = flann.nn_index(vecs)
        vx2_wx = vt.atleast_nd(vx2_wx, 2)
        vx_list = np.arange(len(vx2_wx))
        for vx, wxs in zip(vx_list, vx2_wx):
            for wx in wxs:
                wx2_assigned[wx].append((aid, vx))

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
