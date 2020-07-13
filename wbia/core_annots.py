# -*- coding: utf-8 -*-

"""
IBEIS CORE
Defines the core dependency cache supported by the image analysis api

Extracts annotation chips from imaages and applies optional image
normalizations.

TODO:
     * interactive callback functions
     * detection interface
     * identification interface

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
    python -m wbia.control.IBEISControl --test-show_depc_annot_graph --show

Setup:
    >>> from wbia.core_annots import *  # NOQA
    >>> import wbia
    >>> import wbia.plottool as pt
    >>> ibs = wbia.opendb('testdb1')
    >>> depc = ibs.depc_annot
    >>> aid_list = ibs.get_valid_aids()[0:2]
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
from vtool import image_filters
from wbia import dtool
import utool as ut
import vtool as vt
import numpy as np
import cv2
import wbia.constants as const
from wbia.control.controller_inject import register_preprocs, register_subprops
from wbia.algo.hots.chip_match import ChipMatch
from wbia.algo.hots import neighbor_index

(print, rrr, profile) = ut.inject2(__name__)


derived_attribute = register_preprocs['annot']
register_subprop = register_subprops['annot']
# dtool.Config.register_func = derived_attribute


def testdata_core(defaultdb='testdb1', size=2):
    import wbia

    # import wbia.plottool as pt
    ibs = wbia.opendb(defaultdb=defaultdb)
    depc = ibs.depc_annot
    aid_list = ut.get_argval(
        ('--aids', '--aid'), type_=list, default=ibs.get_valid_aids()[0:size]
    )
    return ibs, depc, aid_list


class ChipThumbConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('thumbsize', 128, 'sz', type_=eval),
        ut.ParamInfo('pad', 0, hideif=0),
        ut.ParamInfo('in_image', False),
        ut.ParamInfo('version', 'dev'),
        ut.ParamInfo('grow', False, hideif=False),
    ]


@derived_attribute(
    tablename='chipthumb',
    parents=['annotations'],
    colnames=['img', 'width', 'height'],
    coltypes=[dtool.ExternType(vt.imread, vt.imwrite, extern_ext='.jpg'), int, int],
    configclass=ChipThumbConfig,
    fname='chipthumb',
    rm_extern_on_delete=True,
    chunksize=256,
)
def compute_chipthumb(depc, aid_list, config=None):
    """
    Yet another chip thumb computer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_annot
        >>> config = ChipThumbConfig.from_argv_dict(dim_size=None)
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> compute_chipthumb(depc, aid_list, config)
        >>> chips = depc.get_property('chips', aid_list, 'img', config={'dim_size': 256})
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> import wbia.viz.interact.interact_chip
        >>> interact_obj = wbia.viz.interact.interact_chip.interact_multichips(ibs, aid_list, config2_=config)
        >>> interact_obj.start()
        >>> pt.show_if_requested()
    """
    print('Chips Thumbs')
    print('config = %r' % (config,))

    ibs = depc.controller

    in_image = config['in_image']

    pad = config['pad']
    thumbsize = config['thumbsize']
    max_dsize = (thumbsize, thumbsize)

    gid_list = ibs.get_annot_gids(aid_list)
    bbox_list = ibs.get_annot_bboxes(aid_list)
    theta_list = ibs.get_annot_thetas(aid_list)
    interest_list = ibs.get_annot_interest(aid_list)

    if 0.0 < pad and pad < 1.0:
        for index in range(len(bbox_list)):
            xtl, ytl, width, height = bbox_list[index]
            padding_w = int(np.around(width * pad))
            padding_h = int(np.around(height * pad))

            xtl -= padding_w
            ytl -= padding_h
            width += 2.0 * padding_w
            height += 2.0 * padding_h

            bbox_list[index] = (xtl, ytl, width, height)

    bbox_size_list = ut.take_column(bbox_list, [2, 3])
    # Checks
    invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
    invalid_aids = ut.compress(aid_list, invalid_flags)
    assert len(invalid_aids) == 0, 'invalid aids=%r' % (invalid_aids,)

    if in_image:
        imgsz_list = ibs.get_image_sizes(gid_list)
        if config['grow']:
            newsize_list = [vt.ScaleStrat.width(thumbsize, wh) for wh in imgsz_list]
            newscale_list = [sz[0] / thumbsize for sz in newsize_list]
        else:
            newsize_scale_list = [
                vt.resized_clamped_thumb_dims((w, h), max_dsize) for (w, h) in imgsz_list
            ]
            newsize_list_ = ut.take_column(newsize_scale_list, 0)
            newscale_list = ut.take_column(newsize_scale_list, [1, 2])
        new_verts_list = [
            vt.scaled_verts_from_bbox(bbox, theta, sx, sy)
            for bbox, theta, (sx, sy) in zip(bbox_list, theta_list, newscale_list)
        ]
        M_list = [vt.scale_mat3x3(sx, sy) for (sx, sy) in newscale_list]
    else:
        if config['grow']:
            newsize_list = [vt.ScaleStrat.width(thumbsize, wh) for wh in bbox_size_list]
        else:
            newsize_scale_list = [
                vt.resized_clamped_thumb_dims(wh, max_dsize) for wh in bbox_size_list
            ]
            newsize_list = ut.take_column(newsize_scale_list, 0)
            # newscale_list = ut.take_column(newsize_scale_list, [1, 2])

        if 1.0 < pad:
            pad = (pad, pad)
            extras_list = []
            for bbox, new_size in zip(bbox_list, newsize_list):
                extras = vt.get_extramargin_measures(bbox, new_size, halfoffset_ms=pad)
                extras_list.append(extras)

            # Overwrite bbox and new size with margined versions
            bbox_list_ = ut.take_column(extras_list, 0)
            newsize_list_ = ut.take_column(extras_list, 1)
        else:
            newsize_list_ = newsize_list
            bbox_list_ = bbox_list
        # Build transformation from image to chip
        M_list = [
            vt.get_image_to_chip_transform(bbox, new_size, theta)
            for bbox, theta, new_size in zip(bbox_list_, theta_list, newsize_list_)
        ]

        new_verts_list = [
            np.round(
                vt.transform_points_with_homography(
                    M, np.array(vt.verts_from_bbox(bbox)).T
                ).T
            ).astype(np.int32)
            for M, bbox in zip(M_list, bbox_list)
        ]

    # arg_iter = zip(cfpath_list, gid_list, newsize_list_, M_list)
    arg_iter = zip(gid_list, newsize_list_, M_list, new_verts_list, interest_list)
    arg_list = list(arg_iter)

    warpkw = dict(flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    last_gid = None
    for tup in ut.ProgIter(arg_list, lbl='computing annot chipthumb', bs=True):
        gid, new_size, M, new_verts, interest = tup
        if gid != last_gid:
            imgBGR = ibs.get_images(gid)
            last_gid = gid

        thumbBGR = cv2.warpAffine(imgBGR, M[0:2], tuple(new_size), **warpkw)

        # -----------------
        if in_image or pad:

            orange_bgr = (0, 128, 255)
            blue_bgr = (255, 128, 0)
            color_bgr = blue_bgr if interest else orange_bgr
            thumbBGR = vt.draw_verts(thumbBGR, new_verts, color=color_bgr, thickness=2)

        width, height = vt.get_size(thumbBGR)
        yield (thumbBGR, width, height)


class ChipConfig(dtool.Config):
    _param_info_list = [
        # ut.ParamInfo('dim_size', 128, 'sz', hideif=None),
        # ut.ParamInfo('dim_size', 960, 'sz', hideif=None),
        ut.ParamInfo(
            'dim_size', 700, 'sz', hideif=None, type_=eval
        ),  # TODO: allow types to vary
        ut.ParamInfo(
            'resize_dim',
            'maxwh',
            '',
            # 'resize_dim', 'width', '',
            # 'resize_dim', 'area', '',
            valid_values=['area', 'width', 'height', 'diag', 'maxwh', 'wh'],
            hideif=lambda cfg: cfg['dim_size'] is None,
        ),
        ut.ParamInfo('dim_tol', 0, 'tol', hideif=0),
        # ut.ParamInfo('preserve_aspect', True, hideif=True),
        # ---
        ut.ParamInfo('histeq', False, hideif=False),
        ut.ParamInfo('greyscale', False, hideif=False),
        # ---
        ut.ParamInfo('adapteq', False, hideif=False),
        ut.ParamInfo('adapteq_ksize', 16, hideif=lambda cfg: not cfg['adapteq']),
        ut.ParamInfo('adapteq_limit', 2.0, hideif=lambda cfg: not cfg['adapteq']),
        # ---
        ut.ParamInfo('medianblur', False, hideif=False),
        ut.ParamInfo('medianblur_thresh', 50, hideif=lambda cfg: not cfg['medianblur']),
        ut.ParamInfo('medianblur_ksize1', 3, hideif=lambda cfg: not cfg['medianblur']),
        ut.ParamInfo('medianblur_ksize2', 5, hideif=lambda cfg: not cfg['medianblur']),
        ut.ParamInfo('axis_aligned', False, hideif=False),
        # ---
        ut.ParamInfo('pad', 0, hideif=0, type_=eval),
        ut.ParamInfo('ext', '.png', hideif='.png'),
    ]


ChipImgType = dtool.ExternType(vt.imread, vt.imwrite, extkey='ext')


@derived_attribute(
    tablename='chips',
    parents=['annotations'],
    colnames=['img', 'width', 'height', 'M'],
    coltypes=[ChipImgType, int, int, np.ndarray],
    configclass=ChipConfig,
    # depprops=['image_uuid', 'verts', 'theta'],
    fname='chipcache4',
    rm_extern_on_delete=True,
    chunksize=256,
)
def compute_chip(depc, aid_list, config=None):
    r"""
    Extracts the annotation chip from the bounding box

    Args:
        depc (wbia.depends_cache.DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        python -m wbia.core_annots --exec-compute_chip:0 --show
        python -m wbia.core_annots --exec-compute_chip:0 --show --greyscale
        wbia --tf compute_chip --show --pad=64 --dim_size=256 --db PZ_MTEST
        wbia --tf compute_chip --show --pad=64 --dim_size=None --db PZ_MTEST
        wbia --tf compute_chip --show --db humpbacks
        wbia --tf compute_chip:1 --show

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'testdb1'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_annot
        >>> config = ChipConfig.from_argv_dict(dim_size=None)
        >>> aid_list = ibs.get_valid_aids()[0:8]
        >>> chips = depc.get_property('chips', aid_list, 'img', config={'dim_size': 256})
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> #interact_obj = pt.interact_multi_image.MultiImageInteraction(chips, nPerPage=4)
        >>> import wbia.viz.interact.interact_chip
        >>> interact_obj = wbia.viz.interact.interact_chip.interact_multichips(ibs, aid_list, config2_=config)
        >>> interact_obj.start()
        >>> pt.show_if_requested()

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'testdb1'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_annot
        >>> config = ChipConfig(**{'dim_size': (256, 256), 'resize_dim': 'wh'})
        >>> #dlg = config.make_qt_dialog()
        >>> #config = dlg.widget.config
        >>> aid_list = ibs.get_valid_aids()[0:8]
        >>> chips = depc.get_property('chips', aid_list, 'img', config=config, recompute=True)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.imshow(vt.stack_image_recurse(chips))
        >>> pt.show_if_requested()
    """
    print('Preprocess Chips')
    print('config = %r' % (config,))

    ibs = depc.controller

    gid_list = ibs.get_annot_gids(aid_list)
    bbox_list = ibs.get_annot_bboxes(aid_list)
    theta_list = ibs.get_annot_thetas(aid_list)

    result_list = gen_chip_configure_and_compute(
        ibs, gid_list, aid_list, bbox_list, theta_list, config
    )
    for result in result_list:
        yield result
    print('Done Preprocessing Chips')


def gen_chip_configure_and_compute(
    ibs, gid_list, rowid_list, bbox_list, theta_list, config
):
    # ext = config['ext']
    pad = config['pad']
    dim_size = config['dim_size']
    dim_tol = config['dim_tol']
    resize_dim = config['resize_dim']
    axis_aligned = config['axis_aligned']
    greyscale = config['greyscale']
    # cfghashid = config.get_hashid()

    if axis_aligned:
        # Over-write bbox and theta with a friendlier, axis-aligned version
        bbox_list_ = []
        theta_list_ = []
        for bbox, theta in zip(bbox_list, theta_list):
            # Transformation matrix
            R = vt.rotation_around_bbox_mat3x3(theta, bbox)
            # Get verticies of the annotation polygon
            verts = vt.verts_from_bbox(bbox, close=True)
            # Rotate and transform vertices
            xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
            trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
            new_verts = np.round(trans_pts).astype(np.int).T.tolist()
            x_points = [pt[0] for pt in new_verts]
            y_points = [pt[1] for pt in new_verts]
            xtl = int(min(x_points))
            xbr = int(max(x_points))
            ytl = int(min(y_points))
            ybr = int(max(y_points))
            bbox_ = (xtl, ytl, xbr - xtl, ybr - ytl)
            theta_ = 0.0
            bbox_list_.append(bbox_)
            theta_list_.append(theta_)
        bbox_list = bbox_list_
        theta_list = theta_list_

    if 0.0 < pad and pad < 1.0:
        for index in range(len(bbox_list)):
            bbox = bbox_list[index]
            xtl, ytl, width, height = bbox
            padding_w = int(np.around(width * pad))
            padding_h = int(np.around(height * pad))

            xtl -= padding_w
            ytl -= padding_h
            width += 2.0 * padding_w
            height += 2.0 * padding_h

            bbox = (xtl, ytl, width, height)
            bbox_list[index] = bbox

    # Checks
    bbox_size_list = ut.take_column(bbox_list, [2, 3])
    invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
    invalid_rowids = ut.compress(rowid_list, invalid_flags)
    assert len(invalid_rowids) == 0, 'invalid rowids=%r' % (invalid_rowids,)

    if resize_dim == 'wh':
        assert isinstance(
            dim_size, tuple
        ), 'must specify both width and height in dim_size when resize_dim=wh'
        # Aspect ratio is not preserved. Use exact specifications.
        newsize_list = [dim_size for _ in range(len(bbox_size_list))]
    else:
        scale_func_dict = {
            'width': vt.ScaleStrat.width,
            'area': vt.ScaleStrat.area,  # actually root area
            'maxwh': vt.ScaleStrat.maxwh,
        }
        scale_func = scale_func_dict[resize_dim]

        if dim_size is None:
            newsize_list = bbox_size_list
        else:
            if resize_dim == 'area':
                dim_size = dim_size ** 2
                dim_tol = dim_tol ** 2
            newsize_list = [scale_func(dim_size, wh, dim_tol) for wh in bbox_size_list]

    if 1.0 < pad:
        pad = (pad, pad)
        extras_list = []
        for bbox, new_size in zip(bbox_list, newsize_list):
            extras = vt.get_extramargin_measures(bbox, new_size, halfoffset_ms=pad)
            extras_list.append(extras)
        # Overwrite bbox and new size with margined versions
        bbox_list = ut.take_column(extras_list, 0)
        newsize_list = ut.take_column(extras_list, 1)

    # Build transformation from image to chip
    M_list = [
        vt.get_image_to_chip_transform(bbox, new_size, theta)
        for bbox, theta, new_size in zip(bbox_list, theta_list, newsize_list)
    ]

    filter_list = []
    # new way
    if config['histeq']:
        filter_list.append(('histeq', {}))
    if config['medianblur']:
        filter_list.append(
            (
                'medianblur',
                {
                    'noise_thresh': config['medianblur_thresh'],
                    'ksize1': config['medianblur_ksize1'],
                    'ksize2': config['medianblur_ksize2'],
                },
            )
        )
    if config['adapteq']:
        ksize = config['adapteq_ksize']
        filter_list.append(
            (
                'adapteq',
                {'tileGridSize': (ksize, ksize), 'clipLimit': config['adapteq_limit']},
            )
        )
    ipreproc = image_filters.IntensityPreproc()

    warpkw = dict(flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    _parallel_chips = getattr(ibs, '_parallel_chips', True)

    if _parallel_chips:
        gpath_list = ibs.get_image_paths(gid_list)
        orient_list = ibs.get_image_orientation(gid_list)
        args_gen = zip(gpath_list, orient_list, M_list, newsize_list)

        gen_kw = {'filter_list': filter_list, 'warpkw': warpkw}
        gen = ut.generate2(
            gen_chip_worker,
            args_gen,
            gen_kw,
            nTasks=len(gpath_list),
            force_serial=ibs.force_serial,
        )
        for chipBGR, width, height, M in gen:
            if greyscale:
                chipBGR = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2GRAY)
            yield chipBGR, width, height, M
    else:
        # arg_iter = zip(cfpath_list, gid_list, newsize_list, M_list)
        arg_iter = zip(gid_list, M_list, newsize_list)
        arg_list = list(arg_iter)

        last_gid = None
        for tup in ut.ProgIter(arg_list, lbl='computing chips', bs=True):
            # FIXME: THE GPATH SHOULD BE PASSED HERE WITH AN ORIENTATION FLAG
            # cfpath, gid, new_size, M = tup
            gid, M, new_size = tup
            # Read parent image # TODO: buffer this?
            # If the gids are sorted, no need to load the image more than once, if so
            if gid != last_gid:
                imgBGR = ibs.get_images(gid)
                last_gid = gid
            # Warp chip
            new_size = tuple([int(np.around(val)) for val in new_size])
            chipBGR = cv2.warpAffine(imgBGR, M[0:2], new_size, **warpkw)
            # Do intensity normalizations
            if filter_list:
                chipBGR = ipreproc.preprocess(chipBGR, filter_list)
            width, height = vt.get_size(chipBGR)
            if greyscale:
                chipBGR = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2GRAY)
            yield (chipBGR, width, height, M)


def gen_chip_worker(gpath, orient, M, new_size, filter_list, warpkw):
    imgBGR = vt.imread(gpath, orient=orient)
    # Warp chip
    new_size = tuple([int(np.around(val)) for val in new_size])
    chipBGR = cv2.warpAffine(imgBGR, M[0:2], new_size, **warpkw)
    # Do intensity normalizations
    if filter_list:
        ipreproc = image_filters.IntensityPreproc()
        chipBGR = ipreproc.preprocess(chipBGR, filter_list)
    width, height = vt.get_size(chipBGR)
    return (chipBGR, width, height, M)


@register_subprop('chips', 'dlen_sqrd')
def compute_dlen_sqrd(depc, aid_list, config=None):
    size_list = np.array(depc.get('chips', aid_list, ('width', 'height'), config))
    dlen_sqrt_list = (size_list ** 2).sum(axis=1).tolist()
    return dlen_sqrt_list


class AnnotMaskConfig(dtool.Config):
    _param_info_list = [ut.ParamInfo('manual', True)]
    _sub_config_list = [ChipConfig]


@derived_attribute(
    tablename='annotmask',
    parents=['annotations'],
    colnames=['img', 'width', 'height'],
    coltypes=[('extern', vt.imread), int, int],
    configclass=AnnotMaskConfig,
    fname='maskcache2',
    # isinteractive=True,
)
def compute_annotmask(depc, aid_list, config=None):
    r"""
    Interaction dispatcher for annotation masks.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        aid_list (list):  list of annotation rowids
        config (AnnotMaskConfig): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        python -m wbia.core_annots --exec-compute_annotmask --show
        python -m wbia.core_annots --exec-compute_annotmask --show --edit

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> config = AnnotMaskConfig(dim_size=None)
        >>> chip_config = config.chip_cfg
        >>> edit = ut.get_argflag('--edit')
        >>> mask = depc.get_property('annotmask', aid_list, 'img', config, recompute=edit)[0]
        >>> chip = depc.get_property('chips', aid_list, 'img', config=chip_config)[0]
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> resized = vt.resize_mask(mask, chip)
        >>> blended = vt.blend_images_multiply(chip, resized)
        >>> pt.imshow(blended, title='mask')
        >>> pt.show_if_requested()
    """
    from wbia.plottool import interact_impaint

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
    fname_list = [
        _fmt.format(aid=aid, avuuid=avuuid, ext=ext, cfghashid=cfghashid)
        for aid, avuuid in zip(aid_list, avuuid_list)
    ]

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
        # ibs.delete_annot_chips([aid])
        # ibs.delete_annot_chip_thumbs([aid])


class ProbchipConfig(dtool.Config):
    # TODO: incorporate into base
    _named_defaults = {
        'rf': {'fw_detector': 'rf', 'smooth_thresh': None, 'smooth_ksize': None}
    }
    _param_info_list = [
        # ut.ParamInfo('preserve_aspect', True, hideif=True),
        # TODO: Need to specify hte hash of the CNN lastest detector
        ut.ParamInfo('fw_detector', 'cnn', 'detector='),
        ut.ParamInfo('fw_dim_size', 256, 'sz'),
        ut.ParamInfo('smooth_thresh', 20, 'thresh='),
        ut.ParamInfo(
            'smooth_ksize', 20, 'ksz=', hideif=lambda cfg: cfg['smooth_thresh'] is None
        ),
        # ut.ParamInfo('ext', '.png'),
    ]
    # _sub_config_list = [
    #    ChipConfig
    # ]


ProbchipImgType = dtool.ExternType(
    ut.partial(vt.imread, grayscale=True), vt.imwrite, extern_ext='.png'
)


@derived_attribute(
    tablename='probchip',
    parents=['annotations'],
    colnames=['img'],
    coltypes=[ProbchipImgType],
    configclass=ProbchipConfig,
    fname='chipcache4',
    # isinteractive=True,
)
def compute_probchip(depc, aid_list, config=None):
    """ Computes probability chips using pyrf

    CommandLine:
        python -m wbia.core_annots --test-compute_probchip --nocnn --show --db PZ_MTEST
        python -m wbia.core_annots --test-compute_probchip --show --fw_detector=cnn
        python -m wbia.core_annots --test-compute_probchip --show --fw_detector=rf --smooth_thresh=None

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia
        >>> ibs, depc, aid_list = testdata_core()
        >>> aid_list = ibs.get_valid_aids(species='zebra_plains')[0:10]
        >>> config = ProbchipConfig.from_argv_dict(fw_detector='cnn', smooth_thresh=None)
        >>> #probchip_fpath_list_ = ut.take_column(list(compute_probchip(depc, aid_list, config)), 0)
        >>> probchip_list_ = ut.take_column(list(compute_probchip(depc, aid_list, config)), 0)
        >>> #result = ut.repr2(probchip_fpath_list_)
        >>> #print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> #xlabel_list = list(map(str, [vt.image.open_image_size(p) for p in probchip_fpath_list_]))
        >>> #iteract_obj = pt.interact_multi_image.MultiImageInteraction(probchip_fpath_list_, nPerPage=4, xlabel_list=xlabel_list)
        >>> xlabel_list = [str(vt.get_size(img)) for img in probchip_list_]
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(probchip_list_, nPerPage=4, xlabel_list=xlabel_list)
        >>> iteract_obj.start()
        >>> ut.show_if_requested()
    """
    print('[core] COMPUTING FEATWEIGHTS')
    print('config = %r' % (config,))
    import vtool as vt

    ibs = depc.controller

    # Use the labeled species for the fw_detector
    species_list = ibs.get_annot_species_texts(aid_list)
    # print('aid_list = %r' % (aid_list,))
    # print('species_list = %r' % (species_list,))

    fw_detector = config['fw_detector']
    dim_size = config['fw_dim_size']
    smooth_thresh = config['smooth_thresh']
    smooth_ksize = config['smooth_ksize']

    if fw_detector == 'rf':
        pad = 64
    else:
        pad = 0

    probchip_dir = ibs.get_probchip_dir() + '2'
    # cfghashid = config.get_hashid()

    # FIXME: The depcache should make it so this doesn't matter anymore
    ut.ensuredir(probchip_dir)
    # _fmt = 'probchip_avuuid_{avuuid}_' + cfghashid + '.png'
    # annot_visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    # probchip_fpath_list = [ut.unixjoin(probchip_dir, _fmt.format(avuuid=avuuid))
    #                        for avuuid in annot_visual_uuid_list]

    chip_config = ChipConfig(pad=pad, dim_size=dim_size)
    mchip_path_list = depc.get(
        'chips', aid_list, 'img', config=chip_config, read_extern=False
    )

    aid_list = np.array(aid_list)
    species_list = np.array(species_list)
    species_rowid = np.array(ibs.get_species_rowids_from_text(species_list))

    # Group by species
    unique_species_rowids, groupxs = vt.group_indices(species_rowid)
    grouped_aids = vt.apply_grouping(aid_list, groupxs)
    grouped_species = vt.apply_grouping(species_list, groupxs)
    grouped_mpaths = ut.apply_grouping(mchip_path_list, groupxs)
    # grouped_ppaths = ut.apply_grouping(probchip_fpath_list, groupxs)
    unique_species = ut.get_list_column(grouped_species, 0)

    if ut.VERBOSE:
        print('[preproc_probchip] +--------------------')
    print(
        (
            '[preproc_probchip.compute_and_write_probchip] '
            'Preparing to compute %d probchips of %d species'
        )
        % (len(aid_list), len(unique_species))
    )
    print('unique_species = %r' % (unique_species,))
    print(config)

    # grouped_probchip_fpath_list = []
    grouped_probchips = []
    _iter = zip(grouped_aids, unique_species, grouped_mpaths)
    _iter = ut.ProgIter(
        _iter,
        length=len(grouped_aids),
        lbl='probchip for {} species'.format(len(unique_species)),
        enabled=ut.VERBOSE,
        bs=True,
    )

    if fw_detector == 'rf':
        for aids, species, inputchip_fpaths in _iter:
            if len(aids) == 0:
                continue
            if species == '____':
                gen = empty_probchips(inputchip_fpaths)
            else:
                gen = rf_probchips(
                    ibs,
                    aids,
                    species,
                    inputchip_fpaths,
                    pad,
                    smooth_thresh,
                    smooth_ksize,
                )
            grouped_probchips.append(list(gen))
    elif fw_detector == 'cnn':
        for aids, species, inputchip_fpaths in _iter:
            if len(aids) == 0:
                continue
            if species == '____':
                gen = empty_probchips(inputchip_fpaths)
            else:
                gen = cnn_probchips(
                    ibs, species, inputchip_fpaths, smooth_thresh, smooth_ksize
                )
            grouped_probchips.append(list(gen))
    else:
        raise NotImplementedError('unknown fw_detector=%r' % (fw_detector,))

    if ut.VERBOSE:
        print('[preproc_probchip] Done computing probability images')
        print('[preproc_probchip] L_______________________')

    probchip_result_list = vt.invert_apply_grouping2(
        grouped_probchips, groupxs, dtype=object
    )
    for probchip in probchip_result_list:
        yield (probchip,)


def empty_probchips(inputchip_fpaths):
    # HACK for unknown species
    for fpath in inputchip_fpaths:
        size = vt.open_image_size(fpath)
        probchip = np.ones(size[::-1], dtype=np.float)
        yield probchip


def cnn_probchips(ibs, species, inputchip_fpaths, smooth_thresh, smooth_ksize):
    # dont use extrmargin here (for now)
    mask_gen = ibs.generate_species_background_mask(inputchip_fpaths, species)
    for chunk in ut.ichunks(mask_gen, 256):
        _progiter = ut.ProgIter(
            chunk,
            lbl='compute {} probchip chunk'.format(species),
            adjust=True,
            time_thresh=30.0,
            bs=True,
        )
        for mask in _progiter:
            if smooth_thresh is not None and smooth_ksize is not None:
                probchip = postprocess_mask(mask, smooth_thresh, smooth_ksize)
            else:
                probchip = mask
            yield probchip


def rf_probchips(ibs, aids, species, inputchip_fpaths, pad, smooth_thresh, smooth_ksize):
    import ubelt as ub
    from os.path import join
    from wbia.algo.detect import randomforest

    cachedir = ub.ensure_app_cache_dir('wbia', ibs.dbname, 'rfchips')
    # Hack disk based output for RF detector.
    temp_output_fpaths = [
        # Give a reasonably distinctive name for parallel safety
        join(cachedir, 'rf_{}_{}_margin.png'.format(species, aid))
        for aid in aids
    ]
    rfconfig = {'scale_list': [1.0], 'mode': 1, 'output_gpath_list': temp_output_fpaths}
    probchip_generator = randomforest.detect_gpath_list_with_species(
        ibs, inputchip_fpaths, species, **rfconfig
    )
    # Evalutate genrator until completion
    ut.evaluate_generator(probchip_generator)
    # Read output of RF detector and crop the extra margin off of the new
    # probchips
    for fpath in temp_output_fpaths:
        extramargin_probchip = vt.imread(fpath, grayscale=True)
        half_w, half_h = (pad, pad)
        probchip = extramargin_probchip[half_h:-half_h, half_w:-half_w]
        if smooth_thresh is not None and smooth_ksize is not None:
            probchip = postprocess_mask(probchip, smooth_thresh, smooth_ksize)
        yield probchip
        # Delete the temporary file
        ut.delete(fpath, verbose=False)


def postprocess_mask(mask, thresh=20, kernel_size=20):
    r"""
    Args:
        mask (ndarray):

    Returns:
        ndarray: mask2

    CommandLine:
        python -m wbia.core_annots --exec-postprocess_mask --cnn --show --aid=1 --db PZ_MTEST
        python -m wbia --tf postprocess_mask --cnn --show --db PZ_MTEST --adapteq=True

    SeeAlso:
        python -m wbia_cnn --tf generate_species_background_mask --show --db PZ_Master1 --aid 9970

    Ignore:
        input_tuple = aid_list
        tablename = 'probchip'
        config = full_config
        rowid_kw = dict(config=config)

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia.plottool as pt
        >>> ibs, depc, aid_list = testdata_core()
        >>> config = ChipConfig.from_argv_dict()
        >>> probchip_config = ProbchipConfig(smooth_thresh=None)
        >>> chip = ibs.depc_annot.get('chips', aid_list, 'img', config)[0]
        >>> mask = ibs.depc_annot.get('probchip', aid_list, 'img', probchip_config)[0]
        >>> mask2 = postprocess_mask(mask)
        >>> ut.quit_if_noshow()
        >>> fnum = 1
        >>> pt.imshow(chip, pnum=(1, 3, 1), fnum=fnum, xlabel=str(chip.shape))
        >>> pt.imshow(mask, pnum=(1, 3, 2), fnum=fnum, title='before', xlabel=str(mask.shape))
        >>> pt.imshow(mask2, pnum=(1, 3, 3), fnum=fnum, title='after', xlabel=str(mask2.shape))
        >>> ut.show_if_requested()
    """
    import cv2

    # thresh = 20
    # kernel_size = 20
    mask2 = mask.copy()
    # light threshold
    mask2[mask2 < thresh] = 0
    # open and close
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    return mask2


class HOGConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('orientations', 8),
        ut.ParamInfo('pixels_per_cell', (16, 16)),
        ut.ParamInfo('cells_per_block', (1, 1)),
    ]


def make_hog_block_image(hog, config=None):
    """
    References:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py
    """
    from skimage import draw

    if config is None:
        config = HOGConfig()

    cx, cy = config['pixels_per_cell']

    normalised_blocks = hog
    (n_blocksy, n_blocksx, by, bx, orientations) = normalised_blocks.shape

    n_cellsx = (n_blocksx - 1) + bx
    n_cellsy = (n_blocksy - 1) + by

    # Undo the normalization step
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            norm_block = normalised_blocks[y, x, :]
            # hack, this only works right for block sizes of 1
            orientation_histogram[y : y + by, x : x + bx, :] = norm_block

    sx = n_cellsx * cx
    sy = n_cellsy * cy

    radius = min(cx, cy) // 2 - 1
    orientations_arr = np.arange(orientations)
    dx_arr = radius * np.cos(orientations_arr / orientations * np.pi)
    dy_arr = radius * np.sin(orientations_arr / orientations * np.pi)
    hog_image = np.zeros((sy, sx), dtype=float)
    for x in range(n_cellsx):
        for y in range(n_cellsy):
            for o, dx, dy in zip(orientations_arr, dx_arr, dy_arr):
                centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                rr, cc = draw.line(
                    int(centre[0] - dx),
                    int(centre[1] + dy),
                    int(centre[0] + dx),
                    int(centre[1] - dy),
                )
                hog_image[rr, cc] += orientation_histogram[y, x, o]
    return hog_image


@derived_attribute(
    tablename='hog',
    parents=['chips'],
    colnames=['hog'],
    coltypes=[np.ndarray],
    configclass=HOGConfig,
    fname='hogcache',
    chunksize=32,
)
def compute_hog(depc, cid_list, config=None):
    """
    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> chip_config = {}
        >>> config = HOGConfig()
        >>> cid_list = depc.get_rowids('chips', aid_list, config=chip_config)
        >>> hoggen = compute_hog(depc, cid_list, config)
        >>> hog = list(hoggen)[0]
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> hog_image = make_hog_block_image(hog, config)
        >>> ut.show_if_requested()
    """
    import skimage.feature

    orientations = config['orientations']

    ut.assert_all_not_None(cid_list, 'cid_list')
    chip_fpath_list = depc.get_native('chips', cid_list, 'img', read_extern=False)

    for chip_fpath in chip_fpath_list:
        chip = vt.imread(chip_fpath, grayscale=True) / 255.0
        hog = skimage.feature.hog(
            chip,
            feature_vector=False,
            orientations=orientations,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
        )
        yield (hog,)


class FeatConfig(dtool.Config):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
        >>> feat_cfg = FeatConfig()
        >>> result = str(feat_cfg)
        >>> print(result)
        <FeatConfig(hesaff+sift)>
    """

    # TODO: FIXME
    # _parents = [ChipConfig]

    def get_param_info_list(self):
        import pyhesaff

        default_keys = list(pyhesaff.get_hesaff_default_params().keys())
        default_items = list(pyhesaff.get_hesaff_default_params().items())
        param_info_list = [
            ut.ParamInfo('feat_type', 'hesaff+sift', ''),
            ut.ParamInfo('maskmethod', None, hideif=None),
        ]
        param_info_dict = {
            name: ut.ParamInfo(name, default, hideif=default)
            for name, default in default_items
        }
        # param_info_dict['scale_max'].default = -1
        # param_info_dict['scale_max'].default = 50
        param_info_list += ut.dict_take(param_info_dict, default_keys)
        return param_info_list

    def get_hesaff_params(self):
        # Get subset of these params that correspond to hesaff
        import pyhesaff

        default_keys = list(pyhesaff.get_hesaff_default_params().keys())
        hesaff_param_dict = ut.dict_subset(self, default_keys)
        return hesaff_param_dict


# class FeatureTask(object):
#     # TODO: make depcache more luigi-like?
#     tablename = 'feat'
#     parents = ['chips']
#     colnames = ['num_feats', 'kpts', 'vecs']
#     coltypes = [int, np.ndarray, np.ndarray]
#     configclass = FeatConfig
#     fname = 'featcache'
#     chunksize = 1024

#     def run(self):
#         pass


@derived_attribute(
    tablename='feat',
    parents=['chips'],
    colnames=['num_feats', 'kpts', 'vecs'],
    coltypes=[int, np.ndarray, np.ndarray],
    configclass=FeatConfig,
    rm_extern_on_delete=True,
    fname='featcache',
    chunksize=1024,
)
def compute_feats(depc, cid_list, config=None):
    """
    Computes features and yields results asynchronously: TODO: Remove IBEIS from
    this equation. Move the firewall towards the controller

    Args:
        depc (dtool.DependencyCache):
        cid_list (list):
        config (None):

    Returns:
        generator : generates param tups

    SeeAlso:
        ~/code/wbia_cnn/wbia_cnn/_plugin.py

    CommandLine:
        python -m wbia.core_annots --test-compute_feats:0 --show
        python -m wbia.core_annots --test-compute_feats:1

    Doctest:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
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
        >>> import wbia.plottool as pt
        >>> chip = depc.get_native('chips', cid_list[0:1], 'img')[0]
        >>> pt.interact_keypoints.KeypointInteraction(chip, kpts, vecs, autostart=True)
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> # TIMING
        >>> from wbia.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core('PZ_MTEST', 100)
        >>> #config = {'dim_size': 450}
        >>> config = {}
        >>> cid_list = depc.get_rowids('chips', aid_list, config=config)
        >>> config = FeatConfig()
        >>> featgen = compute_feats(depc, cid_list, config)
        >>> feat_list = list(featgen)
        >>> idx = 5
        >>> (nFeat, kpts, vecs) = feat_list[idx]
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> chip = depc.get_native('chips', cid_list[idx:idx + 1], 'img')[0]
        >>> pt.interact_keypoints.KeypointInteraction(chip, kpts, vecs, autostart=True)
        >>> ut.show_if_requested()

        >>> #num_feats = depc.get('feat', aid_list, 'num_feats', config=config, recompute=True)

        ibs.delete_annot_feats(aid_list)
        ibs.get_annot_feat_rowids(aid_list)
    """
    nInput = len(cid_list)
    hesaff_params = config.get_hesaff_params()
    feat_type = config['feat_type']
    maskmethod = config['maskmethod']

    ut.assert_all_not_None(cid_list, 'cid_list')
    chip_fpath_list = depc.get_native('chips', cid_list, 'img', read_extern=False)

    if maskmethod is not None:
        assert False
        # aid_list = ibs.get_chip_aids(cid_list)
        # probchip_fpath_list = ibs.get_annot_probchip_fpath(aid_list)
    else:
        probchip_fpath_list = (None for _ in range(nInput))

    if ut.NOT_QUIET:
        print('[preproc_feat] config = %s' % config)
        if ut.VERYVERBOSE:
            print('full_params = ' + ut.repr2())

    ibs = depc.controller
    if feat_type == 'hesaff+sift':
        # Multiprocessing parallelization
        dictargs_iter = (hesaff_params for _ in range(nInput))
        arg_iter = zip(chip_fpath_list, probchip_fpath_list, dictargs_iter)
        # eager evaluation.
        # TODO: Check if there is any benefit to just passing in the iterator.
        arg_list = list(arg_iter)
        featgen = ut.generate2(
            gen_feat_worker,
            arg_list,
            nTasks=nInput,
            ordered=True,
            force_serial=ibs.force_serial,
            progkw={'freq': 1},
            futures_threaded=True,
        )
    elif feat_type == 'hesaff+siam128':
        from wbia_cnn import _plugin

        assert maskmethod is None, 'not implemented'
        assert False, 'not implemented'
        featgen = _plugin.generate_siam_l2_128_feats(ibs, cid_list, config=config)
    else:
        raise AssertionError('unknown feat_type=%r' % (feat_type,))

    for nFeat, kpts, vecs in featgen:
        yield (
            nFeat,
            kpts,
            vecs,
        )


def gen_feat_worker(chip_fpath, probchip_fpath, hesaff_params):
    r"""
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        chip_fpath:
        probchip_fpath:
        hesaff_params:

    Returns:
        tuple: (None, kpts, vecs)

    CommandLine:
        python -m wbia.core_annots --exec-gen_feat_worker --show
        python -m wbia.core_annots --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1 --affine-invariance=False --scale_max=30
        python -m wbia.core_annots --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1 --affine-invariance=False --maskmethod=None  --scale_max=30

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> ibs, depc, aid_list = testdata_core()
        >>> aid = aid_list[0]
        >>> config = {}
        >>> feat_config = FeatConfig.from_argv_dict()
        >>> chip_fpath = ibs.depc_annot.get('chips', aid_list[0], 'img', config=config, read_extern=False)
        >>> maskmethod = ut.get_argval('--maskmethod', type_=str, default='cnn')
        >>> probchip_fpath = ibs.depc_annot.get('probchip', aid_list[0], 'img', config=config, read_extern=False) if feat_config['maskmethod'] == 'cnn' else None
        >>> hesaff_params = feat_config.asdict()
        >>> # Exec function source
        >>> masked_chip, num_kpts, kpts, vecs = ut.exec_func_src(
        >>>     gen_feat_worker, key_list=['masked_chip', 'num_kpts', 'kpts', 'vecs'],
        >>>     sentinal='num_kpts = kpts.shape[0]')
        >>> result = ('(num_kpts, kpts, vecs) = %s' % (ut.repr2((num_kpts, kpts, vecs)),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> from wbia.plottool.interactions import ExpandableInteraction
        >>> interact = ExpandableInteraction()
        >>> interact.append_plot(pt.interact_keypoints.KeypointInteraction(masked_chip, kpts, vecs))
        >>> interact.append_plot(lambda **kwargs: pt.plot_score_histograms([vt.get_scales(kpts)], **kwargs))
        >>> interact.start()
        >>> ut.show_if_requested()
    """
    import pyhesaff

    chip = vt.imread(chip_fpath)
    if probchip_fpath is not None:
        probchip = vt.imread(probchip_fpath, grayscale=True)
        probchip = vt.resize_mask(probchip, chip)
        # vt.blend_images_multiply(chip, probchip)
        masked_chip = (chip * (probchip[:, :, None].astype(np.float32) / 255)).astype(
            np.uint8
        )
    else:
        masked_chip = chip
    kpts, vecs = pyhesaff.detect_feats_in_image(masked_chip, **hesaff_params)
    num_kpts = kpts.shape[0]
    return (num_kpts, kpts, vecs)


class FeatWeightConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('featweight_enabled', True, 'enabled='),
    ]
    # FIXME: incorporate config dependencies in dtool
    # _parents = [FeatConfig, ProbchipConfig]


@derived_attribute(
    tablename='featweight',
    parents=['feat', 'probchip'],
    colnames=['fwg'],
    coltypes=[np.ndarray],
    configclass=FeatWeightConfig,
    rm_extern_on_delete=True,
    fname='featcache',
    chunksize=64 if const.CONTAINERIZED else 512,
)
def compute_fgweights(depc, fid_list, pcid_list, config=None):
    """
    Args:
        depc (dtool.DependencyCache): depc
        fid_list (list):
        config (None): (default = None)

    CommandLine:
        python -m wbia.core_annots compute_fgweights

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
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
    ibs = depc.controller
    nTasks = len(fid_list)
    print('[compute_fgweights] Computing %d fgweights' % (nTasks,))
    # aid_list = depc.get_ancestor_rowids('feat', fid_list, 'annotations')
    # probchip_fpath_list = depc.get(aid_list, 'img', config={}, read_extern=False)
    probchip_list = depc.get_native('probchip', pcid_list, 'img')
    cid_list = depc.get_ancestor_rowids('feat', fid_list, 'chips')
    chipsize_list = depc.get_native('chips', cid_list, ('width', 'height'))
    kpts_list = depc.get_native('feat', fid_list, 'kpts')
    # Force grayscale reading of chips
    arg_iter = zip(kpts_list, probchip_list, chipsize_list)
    featweight_gen = ut.generate2(
        gen_featweight_worker,
        arg_iter,
        nTasks=nTasks,
        ordered=True,
        force_serial=ibs.force_serial,
        progkw={'freq': 1},
        futures_threaded=True,
    )
    featweight_list = list(featweight_gen)
    print('[compute_fgweights] Done computing %d fgweights' % (nTasks,))
    for fw in featweight_list:
        yield (fw,)


def gen_featweight_worker(kpts, probchip, chipsize):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        kpts:
        probchip:
        chipsize:

    CommandLine:
        python -m wbia.core_annots --test-gen_featweight_worker --show
        python -m wbia.core_annots --test-gen_featweight_worker --show --dpath figures --save ~/latex/crall-candidacy-2015/figures/gen_featweight.jpg
        python -m wbia.core_annots --test-gen_featweight_worker --show --db PZ_MTEST --qaid_list=1,2,3,4,5,6,7,8,9

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> #test_featweight_worker()
        >>> ibs, depc, aid_list = testdata_core()
        >>> aid_list = aid_list[0:1]
        >>> config = {'dim_size': 450, 'resize_dim': 'area', 'smooth_thresh': 0, 'smooth_ksize': 0}
        >>> probchip = depc.get('probchip', aid_list, 'img', config=config)[0]
        >>> chipsize = depc.get('chips', aid_list, ('width', 'height'), config=config)[0]
        >>> kpts = depc.get('feat', aid_list, 'kpts', config=config)[0]
        >>> weights = gen_featweight_worker(kpts, probchip, chipsize)
        >>> assert np.all(weights <= 1.0), 'weights cannot be greater than 1'
        >>> chip = depc.get('chips', aid_list, 'img', config=config)[0]
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
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
    if probchip is None:
        # hack for undetected chips. SETS ALL FEATWEIGHTS TO .25 = 1/4
        assert False, 'should not be in this state'
        weights = np.full(len(kpts), 0.25, dtype=np.float32)
    else:
        sfx, sfy = (probchip.shape[1] / chipsize[0], probchip.shape[0] / chipsize[1])
        kpts_ = vt.offset_kpts(kpts, (0, 0), (sfx, sfy))
        # vtpatch.get_warped_patches()
        if False:
            # VERY SLOW
            patch_list1 = [
                vt.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0
                for kp in kpts_
            ]
            weight_list = [vt.gaussian_average_patch(patch1) for patch1 in patch_list1]
            # weight_list = [patch.sum() / (patch.size) for patch in patch_list]
        else:
            # New way
            weight_list = vt.patch_gaussian_weighted_average_intensities(probchip, kpts_)
        weights = np.array(weight_list, dtype=np.float32)
    return weights


class VsOneConfig(dtool.Config):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
        >>> cfg = VsOneConfig()
        >>> result = str(cfg)
        >>> print(result)
    """

    _param_info_list = vt.matching.VSONE_DEFAULT_CONFIG + [
        ut.ParamInfo('version', 8),
        ut.ParamInfo('query_rotation_heuristic', False),
    ]
    #     #ut.ParamInfo('sver_xy_thresh', .01),
    #     ut.ParamInfo('sver_xy_thresh', .001),
    #     ut.ParamInfo('ratio_thresh', .625),
    #     ut.ParamInfo('refine_method', 'homog'),
    #     ut.ParamInfo('symmetric', False),
    #     ut.ParamInfo('K', 1),
    #     ut.ParamInfo('Knorm', 1),

    _sub_config_list = [
        FeatConfig,
        ChipConfig,  # TODO: infer chip config from feat config
        FeatWeightConfig,
    ]


@derived_attribute(
    tablename='pairwise_match',
    parents=['annotations', 'annotations'],
    colnames=['match'],
    coltypes=[vt.PairwiseMatch],
    configclass=VsOneConfig,
    chunksize=512,
    fname='vsone2',
)
def compute_pairwise_vsone(depc, qaids, daids, config):
    """
    Executes one-vs-one matching between pairs of annotations using
    the vt.PairwiseMatch object.

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> match_config = ut.hashdict({})
        >>> qaids = [1, 4, 2]
        >>> daids = [2, 5, 3]
        >>> match_list = ibs.depc.get('pairwise_match', (qaids, daids),
        >>>                           'match', config=match_config)
        >>> m1, m2, m3 = match_list
        >>> assert (m1.annot1['aid'], m1.annot2['aid']) == (1, 2)
        >>> assert (m2.annot1['aid'], m2.annot2['aid']) == (4, 5)
        >>> assert m1.fs.sum() > m2.fs.sum()
    """
    ibs = depc.controller
    qannot_cfg = config
    dannot_cfg = config

    configured_lazy_annots = make_configured_annots(
        ibs, qaids, daids, qannot_cfg, dannot_cfg, preload=True
    )

    unique_lazy_annots = ut.flatten([x.values() for x in configured_lazy_annots.values()])

    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    for annot in unique_lazy_annots:
        vt.matching.ensure_metadata_flann(annot, flann_params)

    for annot in unique_lazy_annots:
        vt.matching.ensure_metadata_normxy(annot)
        # annot['norm_xys'] = (vt.get_xys(annot['kpts']) /
        #                      np.array(annot['chip_size'])[:, None])

    for qaid, daid in ut.ProgIter(
        zip(qaids, daids), length=len(qaids), lbl='compute vsone', bs=True, freq=1
    ):
        annot1 = configured_lazy_annots[qannot_cfg][qaid]
        annot2 = configured_lazy_annots[dannot_cfg][daid]
        match = vt.PairwiseMatch(annot1, annot2)
        match.apply_all(config)
        yield (match,)


def make_configured_annots(
    ibs, qaids, daids, qannot_cfg, dannot_cfg, preload=False, return_view_cache=False
):
    """
    Configures annotations so they can be sent to the vsone vt.matching
    procedure.

    CommandLine:
        python -m wbia.core_annots make_configured_annots

    Doctest:
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> qannot_cfg = dannot_cfg = ut.hashdict({})
        >>> qaids = [1, 2]
        >>> daids = [3, 4]
        >>> preload = True
        >>> configured_lazy_annots, configured_annot_views = make_configured_annots(
        >>>     ibs, qaids, daids, qannot_cfg, dannot_cfg, preload=False,
        >>>     return_view_cache=True,
        >>> )
        >>> aid_dict = configured_lazy_annots[qannot_cfg]
        >>> annot_views = configured_annot_views[qannot_cfg]
        >>> annot = aid_dict[1]
        >>> assert len(annot_views._cache) == 0
        >>> view = annot['view']
        >>> kpts = annot['kpts']
        >>> assert len(annot_views._cache) == 2
    """
    # Prepare lazy attributes for annotations
    unique_qaids = set(qaids)
    unique_daids = set(daids)

    # Determine a unique set of annots per config
    configured_aids = ut.ddict(set)
    configured_aids[qannot_cfg].update(unique_qaids)
    configured_aids[dannot_cfg].update(unique_daids)

    # Make efficient annot-view representation
    configured_annot_views = {}
    for config, aids in configured_aids.items():
        # Create a view of the annotations to efficiently preload data.
        annots = ibs.annots(sorted(aids), config=config)
        # Views are always caching
        configured_annot_views[config] = annots.view()

    if preload:
        precompute_weights = (
            qannot_cfg['weight'] == 'fgweights' or dannot_cfg['weight'] == 'fgweights'
        )
        unique_annot_views = list(configured_annot_views.values())
        for annots in unique_annot_views:
            annots.chip_size
            annots.vecs
            annots.kpts
            annots.yaw
            annots.viewpoint_int
            annots.qual
            annots.gps
            annots.time
            if precompute_weights:
                annots.fgweights

    configured_lazy_annots = ut.ddict(dict)
    for config, annots in configured_annot_views.items():
        annot_dict = configured_lazy_annots[config]
        for aid in ut.ProgIter(annots, label='make lazy dict'):
            # make a subview that points to the original view
            annot_view = annots.view(aid)
            annot = annot_view._make_lazy_dict()
            # hack for vsone to use the "view" feature
            annot['view'] = ut.partial(getattr, annot_view, 'viewpoint_int')
            annot_dict[aid] = annot

    if return_view_cache:
        # Return the underlying annot cache.  we will loose an explicit
        # reference to it if its not returned.  This is ok, because all created
        # annot dict objects do have a reference to it, its just hard to get
        # to. This is only for debuging.
        return configured_lazy_annots, configured_annot_views
    else:
        return configured_lazy_annots


class IndexerConfig(dtool.Config):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
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
        # FeatConfig,
        # ChipConfig,  # TODO: infer chip config from feat config
        # FeatWeightConfig,
    ]

    def get_flann_params(self):
        default_params = vt.get_flann_params(self['algorithm'])
        flann_params = ut.update_existing(default_params, self.asdict())
        return flann_params


testmode = ut.get_argflag('--testmode')


# if 1 or testmode:
@derived_attribute(
    # tablename='neighbor_index', parents=['annotations*'],
    # tablename='neighbor_index', parents=['annotations'],
    # tablename='neighbor_index', parents=['feat*'],
    tablename='neighbor_index',
    parents=['featweight*'],
    # tablename='neighbor_index', parents=['feat*'],
    # tablename='neighbor_index', parents=['feat'],
    colnames=['indexer'],
    coltypes=[neighbor_index.NeighborIndex2],
    configclass=IndexerConfig,
    chunksize=1,
    fname='indexer',
)
def compute_neighbor_index(depc, fids_list, config):
    r"""
    Args:
        depc (dtool.DependencyCache):
        fids_list (list):
        config (dtool.Config):

    CommandLine:
        python -m wbia.core_annots --exec-compute_neighbor_index --show
        python -m wbia.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=neighbor_index

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_annots import *  # NOQA
        >>> import wbia
        >>> ibs, aid_list = wbia.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> fid_list = depc.get_rowids('feat', aid_list)
        >>> aids_list = tuple([aid_list])
        >>> fids_list = tuple([fid_list])
        >>> # Compute directly from function
        >>> config = ibs.depc_annot['neighbor_index'].configclass()
        >>> result1 = list(compute_neighbor_index(depc, fids_list, config))
        >>> nnindexer1 = result1[0][0]
        >>> # Compute using depcache
        >>> result2 = ibs.depc_annot.get('neighbor_index', [aids_list], 'indexer', config, recompute=False, _debug=True)
        >>> #result3 = ibs.depc_annot.get('neighbor_index', [tuple(fids_list)], 'indexer', config, recompute=False)
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


# class FeatNeighborConfig(dtool.Config)


if testmode:
    # NOT YET READY
    @derived_attribute(
        tablename='feat_neighbs',
        parents=['featweight', 'neighbor_index'],
        colnames=['qfx2_idx', 'qfx2_dist'],
        coltypes=[np.ndarray, np.ndarray],
        # configclass=IndexerConfig,
        chunksize=1,
        fname='neighbors',
    )
    def compute_feature_neighbors(depc, fid_list, indexer_rowid_list, config):
        """
        Args:
            depc (dtool.DependencyCache):
            aids_list (list):
            config (dtool.Config):

        CommandLine:
            python -m wbia.core_annots --exec-compute_feature_neighbors --show
            python -m wbia.control.IBEISControl --test-show_depc_annot_table_input --show --tablename=feat_neighbs

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.core_annots import *  # NOQA
            >>> #ibs, depc, aid_list = testdata_core(size=5)
            >>> import wbia
            >>> ibs, qaid_list = wbia.testdata_aids('seaturtles')
            >>> daid_list = qaid_list
            >>> depc = ibs.depc_annot
            >>> index_config = ibs.depc_annot['neighbor_index'].configclass()
            >>> fid_list = depc.get_rowids('feat', qaid_list)
            >>> indexer_rowid_list = ibs.depc_annot.get_rowids('neighbor_index', [daid_list], index_config)
            >>> config = ibs.depc_annot['feat_neighbs'].configclass()
            >>> compute_feature_neighbors(depc, fid_list, indexer_rowid_list, config)
        """
        print('[IBEIS] NEAREST NEIGHBORS')
        # assert False
        # do computation
        # num_neighbors = (config['K'] + config['Knorm'])
        ibs = depc.controller
        num_neighbors = 1

        # b = np.broadcast([1, 2, 3], [1])
        # list(b)
        # [(1, 1), (2, 1), (3, 1)]

        # FIXME: not sure how depc should handle this case
        # Maybe it groups by indexer_rowid_list and then goes from there.
        indexer = depc.get_native('neighbor_index', indexer_rowid_list, 'indexer')[0]
        qvecs_list = depc.get_native(
            'feat', fid_list, 'vecs', eager=False, nInput=len(fid_list)
        )
        # qvecs_list = depc.get('feat', qaid_list, 'vecs', config, eager=False, nInput=len(qaid_list))
        qaid_list = depc.get_ancestor_rowids('feat', fid_list)

        ax2_encid = np.array(ibs.get_annot_encounter_text(indexer.ax2_aid))

        for qaid, qfx2_vec in zip(qaid_list, qvecs_list):
            qencid = ibs.get_annot_encounter_text([qaid])[0]
            invalid_axs = np.where(ax2_encid == qencid)[0]
            # indexer.ax2_aid[invalid_axs]
            nnindxer = indexer
            qfx2_idx, qfx2_dist, iter_count = nnindxer.conditional_knn(
                qfx2_vec, num_neighbors, invalid_axs
            )
            yield qfx2_idx, qfx2_dist

    # NOT YET READY
    @derived_attribute(
        tablename='sver',
        parents=['feat_neighbs'],
        colnames=['chipmatch'],
        coltypes=[ChipMatch],
        # configclass=IndexerConfig,
        chunksize=1,
        fname='vsmany',
    )
    def compute_sver(depc, fid_list, config):
        pass

    @derived_attribute(
        tablename='vsmany',
        parents=['sver'],
        colnames=['chipmatch'],
        coltypes=[ChipMatch],
        # configclass=IndexerConfig,
        chunksize=1,
        fname='vsmany',
    )
    def compute_vsmany(depc, fid_list, config):
        pass


class ClassifierConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('classifier_algo', 'cnn', valid_values=['cnn', 'densenet']),
        ut.ParamInfo('classifier_weight_filepath', None),
    ]
    _sub_config_list = [ChipConfig]


@derived_attribute(
    tablename='classifier',
    parents=['annotations'],
    colnames=['score', 'class'],
    coltypes=[float, str],
    configclass=ClassifierConfig,
    fname='chipcache4',
    chunksize=32 if const.CONTAINERIZED else 1024,
)
def compute_classifications(depc, aid_list, config=None):
    r"""
    Extracts the detections for a given input annotation

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        wbia compute_classifications

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> # depc.delete_property('classifier', gid_list)
        >>> results = depc.get_property('classifier', gid_list, None)
        >>> print(results)
    """
    print('[ibs] Process Image Classifications')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_annot
    if config['classifier_algo'] in ['cnn']:
        config2 = {
            'dim_size': (192, 192),
            'resize_dim': 'wh',
        }
        chip_list = depc.get_property('chips', aid_list, 'img', config=config2)
        result_list = ibs.generate_thumbnail_class_list(chip_list, **config)
    elif config['classifier_algo'] in ['densenet']:
        from wbia.algo.detect import densenet

        config2 = {
            'dim_size': (densenet.INPUT_SIZE, densenet.INPUT_SIZE),
            'resize_dim': 'wh',
        }
        chip_filepath_list = depc.get_property(
            'chips', aid_list, 'img', config=config2, read_extern=False, ensure=True
        )
        result_list = densenet.test(chip_filepath_list, **config)  # yield detections
    else:
        raise ValueError(
            'specified classifier algo is not supported in config = %r' % (config,)
        )

    # yield detections
    for result in result_list:
        yield result


class CanonicalConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('canonical_weight_filepath', None),
    ]
    _sub_config_list = [ChipConfig]


@derived_attribute(
    tablename='canonical',
    parents=['annotations'],
    colnames=['x0', 'y0', 'x1', 'y1'],
    coltypes=[float, float, float, float],
    configclass=CanonicalConfig,
    fname='canonicalcache4',
    chunksize=32 if const.CONTAINERIZED else 1024,
)
def compute_canonical(depc, aid_list, config=None):
    r"""
    Extracts the detections for a given input annotation

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        wbia compute_canonical

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> # depc.delete_property('canonical', gid_list)
        >>> results = depc.get_property('canonical', gid_list, None)
        >>> print(results)
    """
    print('[ibs] Process Annot Canonical')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_annot
    from wbia.algo.detect import canonical

    config2 = {
        'dim_size': (canonical.INPUT_SIZE, canonical.INPUT_SIZE),
        'resize_dim': 'wh',
    }
    chip_filepath_list = depc.get_property(
        'chips', aid_list, 'img', config=config2, read_extern=False, ensure=True
    )
    result_list = canonical.test(chip_filepath_list, **config)  # yield detections

    # yield detections
    for result in result_list:
        yield result


class LabelerConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo(
            'labeler_algo',
            'pipeline',
            valid_values=['azure', 'cnn', 'pipeline', 'densenet'],
        ),
        ut.ParamInfo('labeler_weight_filepath', None),
        ut.ParamInfo('labeler_axis_aligned', False, hideif=False),
    ]
    _sub_config_list = [ChipConfig]


@derived_attribute(
    tablename='labeler',
    parents=['annotations'],
    colnames=['score', 'species', 'viewpoint', 'quality', 'orientation', 'probs'],
    coltypes=[float, str, str, str, float, dict],
    configclass=LabelerConfig,
    fname='chipcache4',
    chunksize=8 if const.CONTAINERIZED else 128,
)
def compute_labels_annotations(depc, aid_list, config=None):
    r"""
    Extracts the detections for a given input image

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        python -m wbia.core_annots --exec-compute_labels_annotations

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_annot
        >>> aid_list = ibs.get_valid_aids()[0:8]
        >>> config = {'labeler_algo': 'densenet', 'labeler_weight_filepath': 'giraffe_v1'}
        >>> # depc.delete_property('labeler', aid_list)
        >>> results = depc.get_property('labeler', aid_list, None, config=config)
        >>> print(results)
        >>> config = {'labeler_weight_filepath': 'candidacy'}
        >>> # depc.delete_property('labeler', aid_list)
        >>> results = depc.get_property('labeler', aid_list, None, config=config)
        >>> print(results)
        >>> config = {'labeler_algo': 'azure'}
        >>> # depc.delete_property('labeler', aid_list)
        >>> results = depc.get_property('labeler', aid_list, None, config=config)
        >>> print(results)
        >>> # depc.delete_property('labeler', aid_list)
        >>> results = depc.get_property('labeler', aid_list, None)
        >>> print(results)
    """
    print('[ibs] Process Annotation Labels')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_annot

    if config['labeler_algo'] in ['pipeline', 'cnn']:
        print('[ibs] labeling using Detection Pipeline Labeler')
        config_ = {
            'dim_size': (128, 128),
            'resize_dim': 'wh',
            'axis_aligned': config['labeler_axis_aligned'],
        }
        chip_list = depc.get_property('chips', aid_list, 'img', config=config_)
        result_gen = ibs.generate_chip_label_list(chip_list, **config)
    elif config['labeler_algo'] in ['azure']:
        from wbia.algo.detect import azure

        print('[ibs] detecting using Azure AI for Earth Species Classification API')
        result_gen = azure.label_aid_list(ibs, aid_list, **config)
    elif config['labeler_algo'] in ['densenet']:
        from wbia.algo.detect import densenet

        config_ = {
            'dim_size': (densenet.INPUT_SIZE, densenet.INPUT_SIZE),
            'resize_dim': 'wh',
            'axis_aligned': config['labeler_axis_aligned'],
        }
        chip_filepath_list = depc.get_property(
            'chips', aid_list, 'img', config=config_, read_extern=False, ensure=True
        )
        config = dict(config)
        config['classifier_weight_filepath'] = config['labeler_weight_filepath']
        result_gen = densenet.test_dict(chip_filepath_list, return_dict=True, **config)
    else:
        raise ValueError(
            'specified labeler algo is not supported in config = %r' % (config,)
        )

    # yield detections
    for result in result_gen:
        yield result


class AoIConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('aoi_two_weight_filepath', None),
    ]


@derived_attribute(
    tablename='aoi_two',
    parents=['annotations'],
    colnames=['score', 'class'],
    coltypes=[float, str],
    configclass=AoIConfig,
    fname='chipcache4',
    chunksize=32 if const.CONTAINERIZED else 256,
)
def compute_aoi2(depc, aid_list, config=None):
    r"""
    Extracts the Annotation of Interest (AoI) for a given input annotation

    Args:
        depc (wbia.depends_cache.DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        wbia compute_aoi2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_annot
        >>> aid_list = ibs.get_valid_aids()[0:8]
        >>> # depc.delete_property('aoi_two', aid_list)
        >>> results = depc.get_property('aoi_two', aid_list, None)
        >>> print(results)
    """
    print('[ibs] Process Annotation AoI2s')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_image
    config_ = {
        'draw_annots': False,
        'thumbsize': (192, 192),
    }
    gid_list = ibs.get_annot_gids(aid_list)
    thumbnail_list = depc.get_property('thumbnails', gid_list, 'img', config=config_)
    bbox_list = ibs.get_annot_bboxes(aid_list)
    size_list = ibs.get_image_sizes(gid_list)
    result_list = ibs.generate_thumbnail_aoi2_list(
        thumbnail_list, bbox_list, size_list, **config
    )
    # yield detections
    for result in result_list:
        yield result


class OrienterConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('orienter_algo', 'deepsense', valid_values=['deepsense']),
        ut.ParamInfo('orienter_weight_filepath', None),
    ]
    _sub_config_list = [ChipConfig]


@derived_attribute(
    tablename='orienter',
    parents=['annotations'],
    colnames=['xtl', 'ytl', 'w', 'h', 'theta'],
    coltypes=[float, float, float, float, float],
    configclass=OrienterConfig,
    fname='detectcache',
    chunksize=8 if const.CONTAINERIZED else 128,
)
def compute_orients_annotations(depc, aid_list, config=None):
    r"""
    Extracts the detections for a given input image

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        python -m wbia.core_annots --exec-compute_orients_annotations --deepsense

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'testdb_identification'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_annot
        >>> aid_list = ibs.get_valid_aids()[-16:-8]
        >>> config = {'orienter_algo': 'deepsense'}
        >>> # depc.delete_property('orienter', aid_list)
        >>> result_list = depc.get_property('orienter', aid_list, None, config=config)
        >>> xtl_list    = list(map(int, map(np.around, ut.take_column(result_list, 0))))
        >>> ytl_list    = list(map(int, map(np.around, ut.take_column(result_list, 1))))
        >>> w_list      = list(map(int, map(np.around, ut.take_column(result_list, 2))))
        >>> h_list      = list(map(int, map(np.around, ut.take_column(result_list, 3))))
        >>> theta_list  = ut.take_column(result_list, 4)
        >>> bbox_list   = list(zip(xtl_list, ytl_list, w_list, h_list))
        >>> ibs.set_annot_bboxes(aid_list, bbox_list, theta_list=theta_list)
        >>> result_list = depc.get_property('orienter', aid_list, None, config=config)
        >>> print(result_list)
    """
    print('[ibs] Process Annotation Labels')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_annot

    if config['orienter_algo'] in ['deepsense']:
        print('[ibs] orienting using Deepsense Orienter')
        try:
            bbox_list = ibs.get_annot_bboxes(aid_list)
            annot_uuid_list = ibs.get_annot_uuids(aid_list)

            result_gen = []
            for bbox, annot_uuid in zip(bbox_list, annot_uuid_list):
                xtl, ytl, w, h = bbox

                cx = xtl + w // 2
                cy = ytl + h // 2
                diameter = max(w, h)
                radius = diameter // 2

                xtl = cx - radius
                ytl = cy - radius
                w = diameter
                h = diameter

                response = ibs.wbia_plugin_deepsense_keypoint(annot_uuid)
                angle = response['keypoints']['angle']
                angle -= 90
                theta = ut.deg_to_rad(angle)

                result = (
                    xtl,
                    ytl,
                    w,
                    h,
                    theta,
                )
                result_gen.append(result)
        except Exception:
            raise RuntimeError('Deepsense orienter not working!')
    else:
        raise ValueError(
            'specified orienter algo is not supported in config = %r' % (config,)
        )

    # yield detections
    for result in result_gen:
        yield result


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.core_annots
        python -m wbia.core_annots --allexamples
        utprof.py -m wbia.core_annots --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
