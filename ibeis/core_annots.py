# -*- coding: utf-8 -*-
"""
IBEIS CORE
Defines the core dependency cache supported by the image analysis api

Extracts annotation chips from imaages and applies optional image
normalizations.

TODO:
    * Dependency Cache from flukes

    * make coltypes take imwrite and return just
     the image and let dtool save it where it wants

     * external write functions
     * interactive callback functions
     * detection interface
     * identificatin interface
     * table based registration

NOTES:
    HOW TO DESIGN INTERACTIVE PLOTS:
        decorate as interactive

        depc.get_property(recompute=True)

        instead of calling preproc as a generator and then adding,
        calls preproc and passes in a callback function.
        preproc spawns interaction and must call callback function when finished.

        callback function adds the rowids to the table.

Needed Tables:
    Chip
    NormChip
    Feats
    Keypoints
    Descriptors
    ProbChip

    IdentifyQuery
    NeighborIndex
    QualityClassifier
    ViewpointClassifier


CommandLine:
    python -m ibeis.control.IBEISControl --test-show_depc_graph --show

Setup:
    >>> from ibeis.core_annots import *  # NOQA
    >>> import ibeis
    >>> import plottool as pt
    >>> ibs = ibeis.opendb('testdb1')
    >>> depc = ibs.depc
    >>> aid_list = ibs.get_valid_aids()[0:2]
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
import dtool
import utool as ut
import vtool as vt
import numpy as np
import cv2
from ibeis.control.controller_inject import register_preproc, register_subprop
from ibeis.algo.hots.chip_match import ChipMatch
from ibeis.algo.hots import neighbor_index
(print, rrr, profile) = ut.inject2(__name__, '[core]')


# dtool.Config.register_func = register_preproc


def testdata_core(defaultdb='testdb1', size=2):
    import ibeis
    # import plottool as pt
    ibs = ibeis.opendb(defaultdb=defaultdb)
    depc = ibs.depc
    aid_list = ut.get_argval(('--aids', '--aid'), type_=list,
                             default=ibs.get_valid_aids()[0:size])
    return ibs, depc, aid_list


class ChipConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('resize_dim', 'width',
                     valid_values=['area', 'width', 'height', 'diag', 'maxwh'],
                     hideif=lambda cfg: cfg['dim_size'] is None),
        #ut.ParamInfo('dim_size', 128, 'sz', hideif=None),
        ut.ParamInfo('dim_size', 960, 'sz', hideif=None),
        ut.ParamInfo('preserve_aspect', True, hideif=True),
        ut.ParamInfo('histeq', False, hideif=False),
        ut.ParamInfo('pad', 0, hideif=0),
        ut.ParamInfo('ext', '.png'),
    ]


@register_preproc(
    tablename='chips', parents=['annotations'],
    colnames=['img', 'width', 'height', 'M'],
    coltypes=[('extern', vt.imread), int, int, np.ndarray],
    configclass=ChipConfig,
    fname='chipcache4',
    chunksize=256,
)
def compute_chip(depc, aid_list, config=None):
    r"""
    Extracts the annotation chip from the bounding box

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        python -m ibeis.core_annots --exec-compute_chip --show
        python -m ibeis.core_annots --exec-compute_chip --show --pad=64 --dim_size=256 --db PZ_MTEST
        python -m ibeis.core_annots --exec-compute_chip --show --db humpbacks

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> depc = ibs.depc
        >>> config = ChipConfig.from_argv_dict(dim_size=None)
        >>> aid_list = ibs.get_valid_aids()[0:8]
        >>> chips = depc.get_property('chips', aid_list, 'img', config)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(chips, nPerPage=4)
        >>> pt.show_if_requested()
    """
    print('Preprocess Chips')
    print('config = %r' % (config,))

    ibs = depc.controller
    chip_dpath = ibs.get_chipdir() + '2'

    ut.ensuredir(chip_dpath)

    ext = config['ext']
    pad = config['pad']
    dim_size = config['dim_size']
    resize_dim = config['resize_dim']

    cfghashid = config.get_hashid()
    avuuid_list = ibs.get_annot_visual_uuids(aid_list)

    # TODO: just hash everything together
    _fmt = 'chip_aid_{aid}_avuuid_{avuuid}_{cfghashid}{ext}'
    cfname_list = [_fmt.format(aid=aid, avuuid=avuuid, ext=ext, cfghashid=cfghashid)
                   for aid, avuuid in zip(aid_list, avuuid_list)]
    cfpath_list = [ut.unixjoin(chip_dpath, chip_fname)
                   for chip_fname in cfname_list]

    gfpath_list = ibs.get_annot_image_paths(aid_list)
    bbox_list   = ibs.get_annot_bboxes(aid_list)
    theta_list  = ibs.get_annot_thetas(aid_list)
    bbox_size_list = ut.take_column(bbox_list, [2, 3])

    # Checks
    invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
    invalid_aids = ut.compress(aid_list, invalid_flags)
    assert len(invalid_aids) == 0, 'invalid aids=%r' % (invalid_aids,)

    scale_func_dict = {
        'width': vt.get_scaled_size_with_width,
        'root_area': vt.get_scaled_size_with_area,
    }
    scale_func = scale_func_dict[resize_dim]

    if dim_size is None:
        newsize_list = bbox_size_list
    else:
        if resize_dim == 'root_area':
            dim_size = dim_size ** 2
        newsize_list = [scale_func(dim_size, w, h) for (w, h) in bbox_size_list]

    if pad > 0:
        halfoffset_ms = (pad, pad)
        extras_list = [vt.get_extramargin_measures(bbox, new_size, halfoffset_ms)
                       for bbox, new_size in zip(bbox_list, newsize_list)]

        # Overwrite bbox and new size with margined versions
        bbox_list = ut.take_column(extras_list, 0)
        newsize_list = ut.take_column(extras_list, 1)

    # Build transformation from image to chip
    M_list = [vt.get_image_to_chip_transform(bbox, new_size, theta) for
              bbox, theta, new_size in zip(bbox_list, theta_list, newsize_list)]

    arg_iter = zip(cfpath_list, gfpath_list, newsize_list, M_list)
    arg_list = list(arg_iter)

    flags = cv2.INTER_LANCZOS4
    borderMode = cv2.BORDER_CONSTANT
    warpkw = dict(flags=flags, borderMode=borderMode)

    for tup in ut.ProgIter(arg_list, lbl='computing chips'):
        cfpath, gfpath, new_size, M = tup
        # Read parent image
        imgBGR = vt.imread(gfpath)
        # Warp chip
        chipBGR = cv2.warpAffine(imgBGR, M[0:2], tuple(new_size), **warpkw)
        width, height = vt.get_size(chipBGR)
        #yield (chipBGR, width, height, M)
        # Write chip to disk
        vt.imwrite(cfpath, chipBGR)
        yield (cfpath, width, height, M)


@register_subprop('chips', 'dlen_sqrd')
def compute_dlen_sqrd(depc, aid_list, config=None):
    size_list = np.array(depc.get('chips', aid_list, ('width', 'height'), config))
    dlen_sqrt_list = (size_list ** 2).sum(axis=1).tolist()
    return dlen_sqrt_list


class AnnotMaskConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('manual', True)
    ]
    _sub_config_list = [
        ChipConfig
    ]


@register_preproc(
    tablename='annotmask', parents=['annotations'],
    colnames=['img', 'width', 'height'],
    coltypes=[('extern', vt.imread), int, int],
    configclass=AnnotMaskConfig,
    fname='../maskcache2',
    # isinteractive=True,
)
def compute_annotmask(depc, aid_list, config=None):
    r"""
    Interaction dispatcher for annotation masks.

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        aid_list (list):  list of annotation rowids
        config (AnnotMaskConfig): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        python -m ibeis.core_annots --exec-compute_annotmask --show
        python -m ibeis.core_annots --exec-compute_annotmask --show --edit

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> config = AnnotMaskConfig(dim_size=None)
        >>> chip_config = config.chip_cfg
        >>> edit = ut.get_argflag('--edit')
        >>> mask = depc.get_property('annotmask', aid_list, 'img', config, recompute=edit)[0]
        >>> chip = depc.get_property('chips', aid_list, 'img', config=chip_config)[0]
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> resized = vt.resize_mask(mask, chip)
        >>> blended = vt.blend_images_multiply(chip, resized)
        >>> pt.imshow(blended, title='mask')
        >>> pt.show_if_requested()
    """
    from plottool import interact_impaint
    # TODO: Ensure interactive required cache words
    # Keep manual things above the cache dir
    mask_dpath = ut.unixjoin(depc.cache_dpath, '../ManualChipMask')
    ut.ensuredir(mask_dpath)

    ibs = depc.controller
    chip_config = config.chip_cfg
    chip_imgs = depc.get('chips', aid_list, 'img', config=chip_config)

    cfghashid = config.get_hashid()
    avuuid_list = ibs.get_annot_visual_uuids(aid_list)

    # TODO: just hash everything together
    ext = '.png'
    _fmt = 'mask_aid_{aid}_avuuid_{avuuid}_{cfghashid}{ext}'
    fname_list = [_fmt.format(aid=aid, avuuid=avuuid, ext=ext, cfghashid=cfghashid)
                   for aid, avuuid in zip(aid_list, avuuid_list)]

    for img, fname, aid in zip(chip_imgs, fname_list, aid_list):
        mask_fpath = ut.unixjoin(mask_dpath, fname)
        if ut.checkpath(mask_fpath):
            # Allow for editing on recompute
            init_mask = vt.imread(mask_fpath)
        else:
            init_mask = None
        mask = interact_impaint.impaint_mask2(img, init_mask=init_mask)
        vt.imwrite(mask_fpath, mask)
        print('imwrite')
        w, h = vt.get_size(mask)

        yield mask_fpath, w, h
        # Remove the old chips
        #ibs.delete_annot_chips([aid])
        #ibs.delete_annot_chip_thumbs([aid])


class ProbchipConfig(dtool.Config):
    # TODO: incorporate into base
    _named_defaults = {
        'rf': {
            'detector': 'rf',
            'smooth_thresh': None,
            'smooth_ksize': None,
        }
    }
    _param_info_list = [
        #ut.ParamInfo('preserve_aspect', True, hideif=True),
        ut.ParamInfo('detector', 'cnn'),
        ut.ParamInfo('dim_size', 256),
        ut.ParamInfo('smooth_thresh', 20),
        ut.ParamInfo('smooth_ksize', 20, hideif=lambda cfg: cfg['smooth_thresh'] is None),
        #ut.ParamInfo('ext', '.png'),
    ]
    #_sub_config_list = [
    #    ChipConfig
    #]


@register_preproc(
    tablename='probchip', parents=['annotations'],
    colnames=['img'],
    coltypes=[('extern', vt.imread)],
    configclass=ProbchipConfig,
    fname='chipcache4',
    # isinteractive=True,
)
def compute_probchip(depc, aid_list, config=None):
    """ Computes probability chips using pyrf

    CommandLine:
        python -m ibeis.core_annots --test-compute_probchip --nocnn --show --db PZ_MTEST
        python -m ibeis.core_annots --test-compute_probchip --show --detector=cnn
        python -m ibeis.core_annots --test-compute_probchip --show --detector=rf --smooth_thresh=None

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> import ibeis
        >>> ibs, depc, aid_list = testdata_core()
        >>> aid_list = ibs.get_valid_aids(species='zebra_plains')[0:10]
        >>> config = ProbchipConfig.from_argv_dict(detector='rf', smooth_thresh=None)
        >>> probchip_fpath_list_ = ut.take_column(list(compute_probchip(depc, aid_list, config)), 0)
        >>> result = ut.list_str(probchip_fpath_list_)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> xlabel_list = list(map(str, [vt.image.open_image_size(p) for p in probchip_fpath_list_]))
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(probchip_fpath_list_, nPerPage=4, xlabel_list=xlabel_list)
        >>> ut.show_if_requested()
    """
    import vtool as vt

    ibs = depc.controller

    # Use the labeled species for the detector
    species_list = ibs.get_annot_species_texts(aid_list)

    detector = config['detector']
    dim_size = config['dim_size']
    smooth_thresh = config['smooth_thresh']
    smooth_ksize = config['smooth_ksize']

    if detector == 'rf':
        pad = 64
    else:
        pad = 0

    probchip_dir = ibs.get_probchip_dir() + '2'
    cfghashid = config.get_hashid()

    # TODO: just hash everything together
    ut.ensuredir(probchip_dir)
    _fmt = 'probchip_avuuid_{avuuid}_' + cfghashid + '.png'
    annot_visual_uuid_list  = ibs.get_annot_visual_uuids(aid_list)
    probchip_fpath_list = [ut.unixjoin(probchip_dir, _fmt.format(avuuid=avuuid))
                           for avuuid in annot_visual_uuid_list]

    chip_config = ChipConfig(pad=pad, dim_size=dim_size)
    mchip_path_list = depc.get('chips', aid_list, 'img', config=chip_config, read_extern=False)

    aid_list = np.array(aid_list)
    species_list = np.array(species_list)
    species_rowid = np.array(ibs.get_species_rowids_from_text(species_list))

    # Group by species
    unique_species_rowids, groupxs = vt.group_indices(species_rowid)
    grouped_aids    = vt.apply_grouping(aid_list, groupxs)
    grouped_species = vt.apply_grouping(species_list, groupxs)
    grouped_mpaths = ut.apply_grouping(mchip_path_list, groupxs)
    grouped_ppaths = ut.apply_grouping(probchip_fpath_list, groupxs)
    unique_species = ut.get_list_column(grouped_species, 0)

    if ut.VERBOSE:
        print('[preproc_probchip] +--------------------')
    print(('[preproc_probchip.compute_and_write_probchip] '
          'Preparing to compute %d probchips of %d species')
          % (len(aid_list), len(unique_species)))
    print(config)

    grouped_probchip_fpath_list = []
    _iter = zip(grouped_aids, unique_species, grouped_ppaths, grouped_mpaths)
    _iter = ut.ProgIter(_iter, nTotal=len(grouped_aids),
                        lbl='probchip for species', enabled=ut.VERBOSE)

    if detector == 'rf':
        for aids, species, probchip_fpaths, inputchip_fpaths in _iter:
            if len(aids) == 0:
                continue
            rf_probchips(ibs, aids, species, probchip_fpaths, inputchip_fpaths, pad,
                         smooth_thresh, smooth_ksize)
            grouped_probchip_fpath_list.append(probchip_fpaths)
    elif detector == 'cnn':
        for aids, species, probchip_fpaths, inputchip_fpaths in _iter:
            if len(aids) == 0:
                continue
            cnn_probchips(ibs, species, probchip_fpath_list, inputchip_fpaths,
                          smooth_thresh, smooth_ksize)
            grouped_probchip_fpath_list.append(probchip_fpaths)
    else:
        raise NotImplementedError('unknown detector=%r' % (detector,))

    if ut.VERBOSE:
        print('[preproc_probchip] Done computing probability images')
        print('[preproc_probchip] L_______________________')

    probchip_fpath_list = vt.invert_apply_grouping2(
        grouped_probchip_fpath_list, groupxs, dtype=object)

    for fpath in probchip_fpath_list:
        yield (fpath,)


def cnn_probchips(ibs, species, probchip_fpath_list, inputchip_fpaths, smooth_thresh, smooth_ksize):
    # dont use extrmargin here (for now)
    mask_gen = ibs.generate_species_background_mask(inputchip_fpaths, species)
    _iter = zip(probchip_fpath_list, mask_gen)
    for chunk in ut.ichunks(_iter, 64):
        _progiter = ut.ProgIter(chunk, lbl='write probchip chunk', adjust=True, time_thresh=30.0)
        for probchip_fpath, probchip in _progiter:
            if smooth_thresh is not None and smooth_ksize is not None:
                probchip = postprocess_mask(probchip, smooth_thresh, smooth_ksize)
            vt.imwrite(probchip_fpath, probchip)


def rf_probchips(ibs, aids, species, probchip_fpaths, inputchip_fpaths, pad,
                 smooth_thresh, smooth_ksize):
    from ibeis.algo.detect import randomforest
    extramargin_probchip_fpaths = [ut.augpath(path, '_margin')
                                   for path in probchip_fpaths]
    rfconfig = {'scale_list': [1.0], 'mode': 1,
                'output_gpath_list': extramargin_probchip_fpaths}
    probchip_generator = randomforest.detect_gpath_list_with_species(
        ibs, inputchip_fpaths, species, **rfconfig)
    # Evalutate genrator until completion
    ut.evaluate_generator(probchip_generator)
    extramargin_mask_gen = (vt.imread(fpath, grayscale=True)
                            for fpath in extramargin_probchip_fpaths)
    # Crop the extra margin off of the new probchips
    _iter = zip(probchip_fpaths, extramargin_mask_gen)
    for (probchip_fpath, extramargin_probchip) in _iter:
        half_w, half_h = (pad, pad)
        probchip = extramargin_probchip[half_h:-half_h, half_w:-half_w]
        if smooth_thresh is not None and smooth_ksize is not None:
            probchip = postprocess_mask(probchip, smooth_thresh, smooth_ksize)
        vt.imwrite(probchip_fpath, probchip)


def postprocess_mask(mask, thresh=20, kernel_size=20):
    r"""
    Args:
        mask (ndarray):

    Returns:
        ndarray: mask2

    CommandLine:
        python -m ibeis.core_annots --exec-postprocess_mask --cnn --show --aid=1 --db PZ_MTEST
        python -m ibeis --tf postprocess_mask --cnn --show --db PZ_MTEST --adapteq=True

    SeeAlso:
        python -m ibeis_cnn --tf generate_species_background_mask --show --db PZ_Master1 --aid 9970

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> import plottool as pt
        >>> from ibeis.algo.preproc.preproc_probchip import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> config = ChipConfig.from_argv_dict()
        >>> probchip_config = ProbchipConfig(smooth_thresh=None)
        >>> chip = ibs.depc.get('chips', aid_list, 'img', config)[0]
        >>> mask = ibs.depc.get('probchip', aid_list, 'img', probchip_config)[0]
        >>> mask2 = postprocess_mask(mask)
        >>> ut.quit_if_noshow()
        >>> fnum = 1
        >>> pt.imshow(chip, pnum=(1, 3, 1), fnum=fnum, xlabel=str(chip.shape))
        >>> pt.imshow(mask, pnum=(1, 3, 2), fnum=fnum, title='before', xlabel=str(mask.shape))
        >>> pt.imshow(mask2, pnum=(1, 3, 3), fnum=fnum, title='after', xlabel=str(mask2.shape))
        >>> ut.show_if_requested()
    """
    import cv2
    thresh = 20
    kernel_size = 20
    mask2 = mask.copy()
    # light threshold
    mask2[mask2 < thresh] = 0
    # open and close
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    return mask2


class FeatConfig(dtool.Config):
    r"""
    Example:
        >>> from ibeis.core_annots import *  # NOQA
        >>> feat_cfg = FeatConfig()
        >>> result = str(feat_cfg)
        >>> print(result)
        <FeatConfig(hesaff+sift,scale_max=40)>
    """

    def get_param_info_list(self):
        import pyhesaff
        default_keys = list(pyhesaff.get_hesaff_default_params().keys())
        default_items = list(pyhesaff.get_hesaff_default_params().items())
        param_info_list = [
            ut.ParamInfo('feat_type', 'hesaff+sift', ''),
            ut.ParamInfo('maskmethod', None, hideif=None)
        ]
        param_info_dict = {
            name: ut.ParamInfo(name, default, hideif=default)
            for name, default in default_items
        }
        param_info_dict['scale_max'].default = 50
        param_info_list += ut.dict_take(param_info_dict, default_keys)
        return param_info_list

    def get_hesaff_params(self):
        # Get subset of these params that correspond to hesaff
        import pyhesaff
        default_keys = list(pyhesaff.get_hesaff_default_params().keys())
        hesaff_param_dict = ut.dict_subset(self, default_keys)
        return hesaff_param_dict


@register_preproc(
    tablename='feat', parents=['chips'],
    colnames=['num_feats', 'kpts', 'vecs'],
    coltypes=[int, np.ndarray, np.ndarray],
    configclass=FeatConfig,
    fname='featcache',
)
def compute_feats(depc, cid_list, config=None):
    r"""
    Computes features and yields results asynchronously: TODO: Remove IBEIS from
    this equation. Move the firewall towards the controller

    Args:
        depc (dtool.DependencyCache):
        cid_list (list):
        config (None):

    Returns:
        generator : generates param tups

    SeeAlso:
        ~/code/ibeis_cnn/ibeis_cnn/_plugin.py

    CommandLine:
        python -m ibeis.core_annots --test-compute_feats:0 --show
        python -m ibeis.core_annots --test-compute_feats:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> chip_config = {}
        >>> config = FeatConfig()
        >>> cid_list = depc.get_rowids('chips', aid_list, config=chip_config)
        >>> featgen = compute_feats(depc, cid_list, config)
        >>> feat_list = list(featgen)
        >>> assert len(feat_list) == len(aid_list)
        >>> (nFeat, kpts, vecs) = feat_list[0]
        >>> assert nFeat == len(kpts) and nFeat == len(vecs)
        >>> assert kpts.shape[1] == 6
        >>> assert vecs.shape[1] == 128
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> chip = depc.get_native('chips', cid_list[0:1], 'img')[0]
        >>> pt.interact_keypoints.KeypointInteraction(chip, kpts, vecs, autostart=True)
        >>> ut.show_if_requested()

    Example:
        >>> # TIMING
        >>> from ibeis.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core('PZ_MTEST', 100)
        >>> config = {'dim_size': 450}
        >>> num_feats = depc.get('feat', aid_list, 'num_feats', config=config, recompute=True)

        ibs.delete_annot_feats(aid_list)
        ibs.get_annot_feat_rowids(aid_list)
    """
    nInput = len(cid_list)
    hesaff_params  = config.get_hesaff_params()
    feat_type      = config['feat_type']
    maskmethod       = config['maskmethod']

    ut.assert_all_not_None(cid_list, 'cid_list')
    chip_fpath_list = depc.get_native('chips', cid_list, 'img', read_extern=False)

    if maskmethod is not None:
        assert False
        #aid_list = ibs.get_chip_aids(cid_list)
        #probchip_fpath_list = ibs.get_annot_probchip_fpath(aid_list)
    else:
        probchip_fpath_list = (None for _ in range(nInput))

    if ut.NOT_QUIET:
        print('[preproc_feat] config = %s' % config)
        if ut.VERYVERBOSE:
            print('full_params = ' + ut.dict_str())

    if feat_type == 'hesaff+sift':
        # Multiprocessing parallelization
        dictargs_iter = (hesaff_params for _ in range(nInput))
        arg_iter = zip(chip_fpath_list, probchip_fpath_list, dictargs_iter)
        # eager evaluation.
        # TODO: Check if there is any benefit to just passing in the iterator.
        arg_list = list(arg_iter)
        featgen = ut.util_parallel.generate(gen_feat_worker, arg_list, nTasks=nInput, freq=10, ordered=True)
    elif feat_type == 'hesaff+siam128':
        from ibeis_cnn import _plugin
        assert maskmethod is None, 'not implemented'
        assert False, 'not implemented'
        ibs = depc.controller
        featgen = _plugin.generate_siam_l2_128_feats(ibs, cid_list, config=config)
    else:
        raise AssertionError('unknown feat_type=%r' % (feat_type,))

    for nFeat, kpts, vecs in featgen:
        yield (nFeat, kpts, vecs,)


def gen_feat_worker(tup):
    r"""
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (tuple):

    Returns:
        tuple: (None, kpts, vecs)

    CommandLine:
        python -m ibeis.core_annots --exec-gen_feat_worker --show
        python -m ibeis.core_annots --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1 --affine-invariance=False --scale_max=30
        python -m ibeis.core_annots --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1 --affine-invariance=False --maskmethod=None  --scale_max=30

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> aid = aid_list[0]
        >>> config = {}
        >>> feat_config = FeatConfig.from_argv_dict()
        >>> chip_fpath = ibs.depc.get('chips', aid_list[0], 'img', config=config, read_extern=False)
        >>> maskmethod = ut.get_argval('--maskmethod', type_=str, default='cnn')
        >>> probchip_fpath = ibs.depc.get('probchip', aid_list[0], 'img', config=config, read_extern=False) if feat_config['maskmethod'] == 'cnn' else None
        >>> hesaff_params = feat_config.asdict()
        >>> # Exec function source
        >>> tup = (chip_fpath, probchip_fpath, hesaff_params)
        >>> masked_chip, num_kpts, kpts, vecs = ut.exec_func_src(
        >>>     gen_feat_worker, key_list=['masked_chip', 'num_kpts', 'kpts', 'vecs'],
        >>>     sentinal='num_kpts = kpts.shape[0]')
        >>> result = ('(num_kpts, kpts, vecs) = %s' % (ut.repr2((num_kpts, kpts, vecs)),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> from plottool.interactions import ExpandableInteraction
        >>> interact = ExpandableInteraction()
        >>> interact.append_plot(pt.interact_keypoints.KeypointInteraction(masked_chip, kpts, vecs))
        >>> interact.append_plot(lambda **kwargs: pt.plot_score_histograms([vt.get_scales(kpts)], **kwargs))
        >>> interact.start()
        >>> ut.show_if_requested()
    """
    import pyhesaff
    #import numpy as np
    #import vtool as vt
    chip_fpath, probchip_fpath, hesaff_params = tup
    chip = vt.imread(chip_fpath)
    if probchip_fpath is not None:
        probchip = vt.imread(probchip_fpath, grayscale=True)
        probchip = vt.resize_mask(probchip, chip)
        #vt.blend_images_multiply(chip, probchip)
        masked_chip = (chip * (probchip[:, :, None].astype(np.float32) / 255)).astype(np.uint8)
    else:
        masked_chip = chip
    kpts, vecs = pyhesaff.detect_kpts_in_image(masked_chip, **hesaff_params)
    num_kpts = kpts.shape[0]
    return (num_kpts, kpts, vecs)


class FeatWeightConfig(dtool.Config):
    _param_info_list = []


@register_preproc(
    tablename='featweight', parents=['feat', 'probchip'],
    colnames=['fwg'],
    coltypes=[np.ndarray],
    configclass=FeatWeightConfig,
    fname='featcache',
)
def compute_fgweights(depc, fid_list, pcid_list, config=None):
    """
    Args:
        depc (dtool.DependencyCache): depc
        fid_list (list):
        config (None): (default = None)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> full_config = {}
        >>> config = FeatConfig()
        >>> fid_list = depc.get_rowids('feat', aid_list, config=full_config)
        >>> pcid_list = depc.get_rowids('probchip', aid_list, config=full_config)
        >>> prop_list = list(compute_fgweights(depc, fid_list, pcid_list))
        >>> featweight_list = ut.take_column(prop_list, 0)
        >>> result = np.array_str(featweight_list[0][0:3], precision=3)
        >>> print(result)
    """
    nTasks = len(fid_list)
    print('[compute_fgweights] Computing %d fgweights' % (nTasks,))
    #aid_list = depc.get_ancestor_rowids('feat', fid_list, 'annotations')
    #probchip_fpath_list = depc.get(aid_list, 'img', config={}, read_extern=False)
    probchip_list = depc.get_native('probchip', pcid_list, 'img')
    cid_list = depc.get_ancestor_rowids('feat', fid_list, 'chips')
    chipsize_list = depc.get_native('chips', cid_list, ('width', 'height'))
    kpts_list = depc.get_native('feat', fid_list, 'kpts')
    # Force grayscale reading of chips
    arg_iter = zip(kpts_list, probchip_list, chipsize_list)
    #featweight_gen = ut.generate(gen_featweight_worker, arg_iter,
    #nTasks=nTasks, ordered=True, freq=10)
    featweight_gen = ut.generate(gen_featweight_worker, arg_iter,
                                 nTasks=nTasks, ordered=True, freq=10,
                                 force_serial=True)
    featweight_list = list(featweight_gen)
    print('[compute_fgweights] Done computing %d fgweights' % (nTasks,))
    for fw in featweight_list:
        yield (fw,)


def gen_featweight_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (aid, tuple(kpts(ndarray), probchip_fpath )): keypoints and
            probability chip file path aid, kpts, probchip_fpath

    CommandLine:
        python -m ibeis.core_annots --test-gen_featweight_worker --show
        python -m ibeis.core_annots --test-gen_featweight_worker --show --dpath figures --save ~/latex/crall-candidacy-2015/figures/gen_featweight.jpg
        python -m ibeis.core_annots --test-gen_featweight_worker --show --db PZ_MTEST --qaid_list=1,2,3,4,5,6,7,8,9

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> #test_featweight_worker()
        >>> ibs, depc, aid_list = testdata_core()
        >>> aid_list = aid_list[0:1]
        >>> config = {}
        >>> probchip = depc.get('probchip', aid_list, 'img', config)[0]
        >>> chipsize = depc.get('chips', aid_list, ('width', 'height'), config)[0]
        >>> kpts = depc.get('feat', aid_list, 'kpts', config)[0]
        >>> tup = (kpts, probchip, chipsize)
        >>> weights = gen_featweight_worker(tup)
        >>> chip = depc.get('chips', aid_list, 'img', config)[0]
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> fnum = 1
        >>> pnum_ = pt.make_pnum_nextgen(1, 3)
        >>> pt.figure(fnum=fnum, doclf=True)
        >>> pt.imshow(chip, pnum=pnum_(0), fnum=fnum)
        >>> pt.imshow(probchip, pnum=pnum_(2), fnum=fnum)
        >>> pt.imshow(chip, pnum=pnum_(1), fnum=fnum)
        >>> color_list = pt.draw_kpts2(kpts, weights=weights, ell_alpha=.3)
        >>> cb = pt.colorbar(weights, color_list)
        >>> cb.set_label('featweights')
        >>> pt.show_if_requested()
    """
    (kpts, probchip, chipsize) = tup
    if probchip is None:
        # hack for undetected chips. SETS ALL FEATWEIGHTS TO .25 = 1/4
        weights = np.full(len(kpts), .25, dtype=np.float32)
    else:
        sfx, sfy = (probchip.shape[1] / chipsize[0], probchip.shape[0] / chipsize[1])
        kpts_ = vt.offset_kpts(kpts, (0, 0), (sfx, sfy))
        #vtpatch.get_warped_patches()
        # VERY SLOW
        patch_list  = [vt.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0
                       for kp in kpts_]
        weight_list = [vt.gaussian_average_patch(patch) for patch in patch_list]
        #weight_list = [patch.sum() / (patch.size) for patch in patch_list]
        weights = np.array(weight_list, dtype=np.float32)
    return weights


class VsOneRequest(dtool.base.VsOneSimilarityRequest):
    _tablename = 'vsone'

    def postprocess_execute(request, parent_rowids, result_list):
        import ibeis
        qaid_list, daid_list = list(zip(*parent_rowids))
        depc = request.depc
        cm_list = []
        unique_qaids, groupxs = ut.group_indices(qaid_list)
        grouped_daids = ut.apply_grouping(daid_list, groupxs)

        ibs = depc.controller
        unique_qnids = ibs.get_annot_nids(unique_qaids)
        single_cm_list = ut.take_column(result_list, 1)
        grouped_cms = ut.apply_grouping(single_cm_list, groupxs)

        _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_cms)

        for qaid, qnid, daids, cms in _iter:
            # Hacked in version of creating an annot match object
            match = ibeis.ChipMatch.combine_cms(cms)
            match.score_nsum(request)
            cm_list.append(match)

        #import utool
        #utool.embed()
        #cm = cm_list[0]
        #cm.print_inspect_str(request)
        #cm.assert_self(request, assert_feats=False)
        return cm_list


class VsOneConfig(dtool.Config):
    """
    Example:
        >>> from ibeis.core_annots import *  # NOQA
        >>> cfg = VsOneConfig()
        >>> result = str(cfg)
        >>> print(result)
    """
    _param_info_list = [
        #ut.ParamInfo('sver_xy_thresh', .01),
        ut.ParamInfo('sver_xy_thresh', .001),
        ut.ParamInfo('ratio_thresh', .625),
        ut.ParamInfo('refine_method', 'homog'),
        ut.ParamInfo('symmetric', False),
        ut.ParamInfo('K', 1),
        ut.ParamInfo('Knorm', 1),
        ut.ParamInfo('version', 0),
        ut.ParamInfo('augment_queryside_hack', False),
    ]
    _sub_config_list = [
        FeatConfig,
        ChipConfig,  # TODO: infer chip config from feat config
        FeatWeightConfig,
    ]


@register_preproc(
    tablename='vsone', parents=['annotations', 'annotations'],
    colnames=['score', 'match'], coltypes=[float, ChipMatch],
    requestclass=VsOneRequest,
    configclass=VsOneConfig,
    chunksize=128, fname='vsone',
)
def compute_one_vs_one(depc, qaids, daids, config):
    """
    CommandLine:
        python -m ibeis.core_annots --test-compute_one_vs_one --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> #ibs, depc, aid_list = testdata_core(size=5)
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids('wd_peter2', 'timectrl:pername=2,view=left,view_ext=0,exclude_reference=True')
        >>> depc = ibs.depc
        >>> request = depc.new_request('vsone', aid_list, aid_list, {'dim_size': 450})
        >>> config = request.config
        >>> qaids, daids = request.parent_rowids_T
        >>> # Compute using request
        >>> print('...Test vsone cache')
        >>> rawres_list2 = request.execute(postprocess=False)
        >>> score_list2 = ut.take_column(rawres_list2, 0)
        >>> res_list2 = request.execute()
        >>> print(res_list2)
        >>> # Compute using function
        >>> #print('...Test vsone function')
        >>> #rawres_list1 = list(compute_one_vs_one(depc, qaids, daids, config))
        >>> #score_list1 = ut.take_column(rawres_list1, 0)
        >>> #print(score_list1)
        >>> #assert np.all(score_list1 == score_list2)
        >>> ut.quit_if_noshow()
        >>> ut.ensure_pylab_qt4()
        >>> match = res_list2[0]
        >>> match.print_inspect_str(request)
        >>> #match.show_analysis(qreq_=request)
        >>> match.ishow_analysis(qreq_=request)
        >>> ut.show_if_requested()
    """
    import ibeis
    ibs = depc.controller
    qconfig2_ = config
    dconfig2_ = config
    unique_qaids = np.unique(qaids)
    unique_daids = np.unique(daids)

    # TODO: Ensure entire pipeline can use new dependencies
    if True:
        annot1_list = [ibs.get_annot_lazy_dict2(qaid, config=qconfig2_)
                       for qaid in unique_qaids]
        annot2_list = [ibs.get_annot_lazy_dict2(daid, config=dconfig2_)
                       for daid in unique_daids]
    else:
        #config.chip_cfgstr = config.chip_cfg.get_cfgstr()
        #config.chip_cfg_dict = config.chip_cfg.asdict()
        annot1_list = [ibs.get_annot_lazy_dict(qaid, config2_=qconfig2_)
                       for qaid in unique_qaids]
        annot2_list = [ibs.get_annot_lazy_dict(daid, config2_=dconfig2_)
                       for daid in unique_daids]

    # precache flann structures
    # TODO: Make depcache node
    flann_params = {'algorithm': 'kdtree', 'trees': 8}
    for annot1 in annot1_list:
        if 'flann' not in annot1:
            annot1['flann'] = lambda: vt.flann_cache(
                annot1['vecs'], flann_params=flann_params, quiet=True,
                verbose=False)

    qaid_to_annot = dict(zip(unique_qaids, annot1_list))
    daid_to_annot = dict(zip(unique_daids, annot2_list))

    #all_aids = np.unique(ut.flatten([qaids, daids]))
    verbose = False
    for qaid, daid  in ut.ProgIter(zip(qaids, daids), nTotal=len(qaids),
                                   lbl='compute vsone'):
        annot1 = qaid_to_annot[qaid]
        annot2 = daid_to_annot[daid]
        metadata = {
            'annot1': annot1,
            'annot2': annot2,
        }
        vt_match = vt.vsone_matching2(metadata, cfgdict=config, verbose=verbose)
        matchtup = vt_match.matches['TOP+SV']
        H = vt_match.metadata['H_TOP']
        score = matchtup.fs.sum()
        fm = matchtup.fm
        fs = matchtup.fs

        match = ibeis.ChipMatch(
            qaid=qaid,
            daid_list=[daid],
            fm_list=[fm],
            fsv_list=[vt.atleast_nd(fs, 2)],
            H_list=[H],
            fsv_col_lbls=['L2_SIFT'])
        match._update_daid_index()
        match.evaluate_dnids(ibs)
        match._update_daid_index()
        match.set_cannonical_name_score([score], [score])

        #import utool
        #utool.embed()
        if False:
            ut.ensure_pylab_qt4()
            ibs, depc, aid_list = testdata_core(size=3)
            request = depc.new_request('vsone', aid_list, aid_list, {'dim_size': 450})
            match.ishow_analysis(request)
        #match = SingleMatch_IBEIS(qaid, daid, score, fm)
        yield (score, match)


class IndexerConfig(dtool.Config):
    """
    Example:
        >>> from ibeis.core_annots import *  # NOQA
        >>> cfg = VsOneConfig()
        >>> result = str(cfg)
        >>> print(result)
    """
    _param_info_list = [
        ut.ParamInfo('algorithm', 'kdtree', 'alg'),
        ut.ParamInfo('random_seed', 42, 'seed'),
        ut.ParamInfo('trees', 4, hideif=lambda cfg: cfg['algorithm'] != 'kdtree'),
        ut.ParamInfo('version', 1),
    ]
    _sub_config_list = [
        #FeatConfig,
        #ChipConfig,  # TODO: infer chip config from feat config
        FeatWeightConfig,
    ]

    def get_flann_params(cfg):
        default_params = vt.get_flann_params(cfg['algorithm'])
        flann_params = ut.update_existing(default_params, cfg.asdict())
        return flann_params


@register_preproc(
    #tablename='neighbor_index', parents=['annotations'],
    #tablename='neighbor_index', parents=['feat*'],
    tablename='neighbor_index', parents=['feat'],
    colnames=['indexer'], coltypes=[neighbor_index.NeighborIndex2],
    ismulti=True,
    configclass=IndexerConfig,
    chunksize=1, fname='indexer',
)
def compute_neighbor_index(depc, fids_list, config):
    """
    Args:
        depc (dtool.DependencyCache):
        fids_list (list):
        config (dtool.Config):

    CommandLine:
        python -m ibeis.core_annots --exec-compute_neighbor_index --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core_annots import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids('testdb1')
        >>> depc = ibs.depc
        >>> fid_list = depc.get_rowids('feat', aid_list)
        >>> aids_list = [aid_list]
        >>> fids_list = [fid_list]
        >>> # Compute directly from function
        >>> config = ibs.depc['neighbor_index'].configclass()
        >>> result1 = list(compute_neighbor_index(depc, fids_list, config))
        >>> nnindexer1 = result1[0][0]
        >>> # Compute using depcache
        >>> result2 = ibs.depc.get('neighbor_index', [aids_list], 'indexer', config, recompute=False, _debug=True)
        >>> result3 = ibs.depc.get('neighbor_index', [aids_list], 'indexer', config, recompute=False)
        >>> print(result2)
        >>> print(result3)
        >>> assert result2[0] is not result3[0]
        >>> assert nnindexer1.knn(ibs.get_annot_vecs(1), 1) is not None
        >>> assert result3[0].knn(ibs.get_annot_vecs(1), 1) is not None
    """
    print('[IBEIS] COMPUTE_NEIGHBOR_INDEX:')
    # TODO: allow augment
    assert len(fids_list) == 1, 'only working with one indexer at a time'
    fid_list = fids_list[0]
    aid_list = depc.get_root_rowids('feat', fid_list)
    flann_params = config.get_flann_params()
    cfgstr = config.get_cfgstr()
    verbose = True
    nnindexer = neighbor_index.NeighborIndex2(flann_params, cfgstr)
    # Initialize neighbor with unindexed data
    support = nnindexer.get_support(depc, aid_list, config)
    nnindexer.init_support(aid_list, *support, verbose=verbose)
    nnindexer.config = config
    nnindexer.reindex()
    yield (nnindexer,)


#class FeatNeighborConfig(dtool.Config)


if False:
    # NOT YET READY
    @register_preproc(
        tablename='feat_neighbs', parents=['feat', 'neighbor_index'],
        colnames=['qfx2_idx', 'qfx2_dist'], coltypes=[np.ndarray, np.ndarray],
        #configclass=IndexerConfig,
        chunksize=1, fname='neighbors',
    )
    def compute_feature_neighbors(depc, fid_list, indexer_rowid_list, config):
        """
        Args:
            depc (dtool.DependencyCache):
            aids_list (list):
            config (dtool.Config):

        CommandLine:
            python -m ibeis.core_annots --exec-compute_neighbor_index --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.core_annots import *  # NOQA
            >>> #ibs, depc, aid_list = testdata_core(size=5)
            >>> import ibeis
            >>> ibs, qaid_list = ibeis.testdata_aids('seaturtles')
            >>> daid_list = qaid_list
            >>> depc = ibs.depc
            >>> index_config = ibs.depc['neighbor_index'].configclass()
            >>> fid_list = depc.get_rowids('feat', qaid_list)
            >>> indexer_rowid_list = ibs.depc.get_rowids('neighbor_index', [daid_list], index_config)
            >>> config = ibs.depc['feat_neighbs'].configclass()
            >>> compute_feature_neighbors(depc, fid_list, indexer_rowid_list, config)
        """
        print('[IBEIS] NEAREST NEIGHBORS')
        #assert False
        # do computation
        #num_neighbors = (config['K'] + config['Knorm'])
        ibs = depc.controller
        num_neighbors = 1

        #b = np.broadcast([1, 2, 3], [1])
        #list(b)
        #[(1, 1), (2, 1), (3, 1)]

        # FIXME: not sure how depc should handle this case
        # Maybe it groups by indexer_rowid_list and then goes from there.
        indexer = depc.get_native('neighbor_index', indexer_rowid_list, 'indexer')[0]
        qvecs_list = depc.get_native('feat', fid_list, 'vecs', eager=False, nInput=len(fid_list))
        #qvecs_list = depc.get('feat', qaid_list, 'vecs', config, eager=False, nInput=len(qaid_list))
        qaid_list = depc.get_ancestor_rowids('feat', fid_list)

        ax2_encid = np.array(ibs.get_annot_encounter_text(indexer.ax2_aid))

        for qaid, qfx2_vec in zip(qaid_list, qvecs_list):
            qencid = ibs.get_annot_encounter_text([qaid])[0]
            invalid_axs = np.where(ax2_encid == qencid)[0]
            #indexer.ax2_aid[invalid_axs]
            nnindxer = indexer
            qfx2_idx, qfx2_dist, iter_count = nnindxer.conditional_knn(qfx2_vec,
                                                                       num_neighbors,
                                                                       invalid_axs)
            yield qfx2_idx, qfx2_dist


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.core_annots
        python -m ibeis.core_annots --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
