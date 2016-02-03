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

     * move version to TableConfig
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
    python -m ibeis.control.IBEISControl --test-show_depc_digraph --show

Setup:
    >>> from ibeis.core import *  # NOQA
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
from ibeis.control.controller_inject import register_preproc
(print, rrr, profile) = ut.inject2(__name__, '[core]')


# dtool.TableConfig.register_func = register_preproc


def testdata_core():
    import ibeis
    # import plottool as pt
    ibs = ibeis.opendb('testdb1')
    depc = ibs.depc
    aid_list = ibs.get_valid_aids()[0:2]
    return ibs, depc, aid_list


class ChipConfig(dtool.TableConfig):
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
    coltypes=[('extern', vt.imread, vt.imwrite), int, int, np.ndarray],
    configclass=ChipConfig,
    fname='chipcache4',
    version=0
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
        python -m ibeis.core --exec-compute_chip --show
        python -m ibeis.core --exec-compute_chip --show --pad=64 --dim_size=256 --db PZ_MTEST
        python -m ibeis.core --exec-compute_chip --show --db humpbacks

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core import *  # NOQA
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
        # Write chip to disk
        vt.imwrite(cfpath, chipBGR)
        yield (cfpath, width, height, M)


class AnnotMaskConfig(dtool.TableConfig):
    _param_info_list = [
        ut.ParamInfo('manual', True)
    ]
    _sub_config_list = [
        ChipConfig
    ]


@register_preproc(
    tablename='annotmask', parents=['annotations'],
    colnames=['img', 'width', 'height'],
    coltypes=[('extern', vt.imread, vt.imwrite), int, int],
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
        python -m ibeis.core --exec-compute_annotmask --show
        python -m ibeis.core --exec-compute_annotmask --show --edit

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core import *  # NOQA
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


class ProbchipConfig(dtool.TableConfig):
    _param_info_list = [
        #ut.ParamInfo('preserve_aspect', True, hideif=True),
        ut.ParamInfo('detector', 'cnn'),
        ut.ParamInfo('dim_size', 256),
        #ut.ParamInfo('ext', '.png'),
    ]
    #_sub_config_list = [
    #    ChipConfig
    #]


def compute_probchip(depc, aid_list, config=None):
    """ Computes probability chips using pyrf

    CommandLine:
        python -m ibeis.core --test-compute_probchip --nocnn --show

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> depc = ibs.depc
        >>> config = ProbchipConfig.from_argv_dict(detector='rf')
        >>> aid_list = ibs.get_valid_aids(species='zebra_plains')
        >>> probchip_fpath_list_ = compute_probchip(depc, aid_list, config)
        >>> result = ut.list_str(probchip_fpath_list_)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(probchip_fpath_list_, nPerPage=4)
        >>> ut.show_if_requested()
    """
    import vtool as vt

    ibs = depc.controller

    # Use the labeled species for the detector
    species_list = ibs.get_annot_species_texts(aid_list)

    detector = config['detector']
    pad = 64

    if detector == 'rf':
        from ibeis.algo.detect import randomforest

    cfghashid = config.get_hashid()
    probchip_dir = ibs.get_probchip_dir() + '2'

    chip_config = ChipConfig(pad=pad, dim_size=config['dim_size'])
    mchip_path_list = depc.get('chips', aid_list, 'img', config=chip_config, read_extern=False)

    # TODO: just hash everything together
    ut.ensuredir(probchip_dir)
    _fmt = 'probchip_avuuid_{avuuid}_' + cfghashid + '.png'
    annot_visual_uuid_list  = ibs.get_annot_visual_uuids(aid_list)
    probchip_fpath_list = [ut.unixjoin(probchip_dir, _fmt.format(avuuid=avuuid))
                           for avuuid in annot_visual_uuid_list]

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

    nSpecies = len(unique_species)
    nTasks = len(aid_list)
    print(('[preproc_probchip.compute_and_write_probchip] '
          'Preparing to compute %d probchips of %d species')
          % (nTasks, nSpecies))

    grouped_probchip_fpath_list = []
    if ut.VERBOSE:
        print('[preproc_probchip] +--------------------')


    _iter = zip(grouped_aids, unique_species, grouped_ppaths, grouped_mpaths)
    for aids, species, probchip_fpaths, margin_fpaths in _iter:
        if ut.VERBOSE:
            print('[preproc_probchip] Computing probchips for species=%r' % species)
            print('[preproc_probchip] |--------------------')
        if len(aids) == 0:
            continue
        # No filtering
        if detector == 'rf':
            probchip_extramargin_fpath_list = [ut.augpath(path, 'margin') for path in probchip_fpaths]
            #dirty_cfpath_list  = ibs.get_annot_chip_fpath(aids, ensure=True, config2_=config2_)

            config = {
                'scale_list': [1.0],
                'output_gpath_list': probchip_extramargin_fpath_list,
                'mode': 1,
            }
            probchip_generator = randomforest.detect_gpath_list_with_species(
                ibs, margin_fpaths, species, **config)
            # Evalutate genrator until completion
            ut.evaluate_generator(probchip_generator)
            extramargin_mask_gen = (
                vt.imread(fpath, grayscale=True) for fpath in probchip_extramargin_fpath_list
            )
            # Crop the extra margin off of the new probchips
            _iter = zip(probchip_fpath_list,
                        extramargin_mask_gen)
            for (probchip_fpath, extramargin_probchip) in _iter:
                half_w, half_h = (pad, pad)
                probchip = extramargin_probchip[half_h:-half_h, half_w:-half_w]
                vt.imwrite(probchip_fpath, probchip)
        elif detector == 'cnn':
            # dont use extrmargin here (for now)
            mask_gen = ibs.generate_species_background_mask(margin_fpaths, species)
            _iter = zip(probchip_fpath_list, mask_gen)
            for chunk in ut.ichunks(_iter, 64):
                _progiter = ut.ProgIter(chunk, lbl='write probchip chunk', adjust=True, time_thresh=30.0)
                for probchip_fpath, probchip in _progiter:
                    probchip = postprocess_mask(probchip)
                    vt.imwrite(probchip_fpath, probchip)
        grouped_probchip_fpath_list.append(probchip_fpaths)
    if ut.VERBOSE:
        print('[preproc_probchip] Done computing probability images')
        print('[preproc_probchip] L_______________________')

    probchip_fpath_list = vt.invert_apply_grouping2(
        grouped_probchip_fpath_list, groupxs, dtype=object)
    return probchip_fpath_list


def postprocess_mask(mask):
    r"""
    Args:
        mask (ndarray):

    Returns:
        ndarray: mask2

    CommandLine:
        python -m ibeis.algo.preproc.preproc_probchip --exec-postprocess_mask --cnn --show --aid=1 --db PZ_MTEST
        python -m ibeis --tf postprocess_mask --cnn --show --db PZ_Master1 --aid 9970
        python -m ibeis --tf postprocess_mask --cnn --show --db PZ_Master1 --aid 9970 --adapteq=True
        python -m ibeis --tf postprocess_mask --cnn --show --db GIRM_Master1 --aid 9970 --adapteq=True
        python -m ibeis --tf postprocess_mask --cnn --show --db GIRM_Master1

    SeeAlso:
        python -m ibeis_cnn --tf generate_species_background_mask --show --db PZ_Master1 --aid 9970

    Example:
        >>> # DISABLE_DOCTEST
        >>> import ibeis_cnn
        >>> import ibeis
        >>> import vtool as vt
        >>> import plottool as pt
        >>> from ibeis.algo.preproc.preproc_probchip import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ut.get_argval(('--aids', '--aid'), type_=list, default=[10])
        >>> default_config = dict(ibs.cfg.chip_cfg.parse_items())
        >>> cfgdict = ut.argparse_dict(default_config)
        >>> config2_ = ibs.new_query_params(cfgdict=cfgdict)
        >>> chip_fpath = ibs.get_annot_chip_fpath(aid_list, config2_=config2_)[0]
        >>> chip = vt.imread(chip_fpath)
        >>> #species = ibs.const.TEST_SPECIES.ZEB_PLAIN
        >>> species = ibs.get_primary_database_species()
        >>> print('species = %r' % (species,))
        >>> mask_list = list(ibs.generate_species_background_mask([chip_fpath], species))
        >>> mask = mask_list[0]
        >>> mask2 = postprocess_mask(mask)
        >>> ut.quit_if_noshow()
        >>> fnum = 1
        >>> pt.imshow(chip, pnum=(1, 3, 1), fnum=fnum)
        >>> pt.imshow(mask, pnum=(1, 3, 2), fnum=fnum, title='before')
        >>> pt.imshow(mask2, pnum=(1, 3, 3), fnum=fnum, title='after')
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.core
        python -m ibeis.core --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
