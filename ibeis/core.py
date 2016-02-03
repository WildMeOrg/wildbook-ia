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


class AnnotMaskConfig(dtool.TableConfig):
    _param_info_list = [
        ut.ParamInfo('dim_size', 960, 'sz', hideif=None),
        ut.ParamInfo('manual', True)
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
        >>> chip_config = ChipConfig(dim_size=None)
        >>> edit = ut.get_argflag('--edit')
        >>> mask = depc.get_property('annotmask', aid_list, 'img', recompute=edit)[0]
        >>> chip = depc.get_property(const.CHIP_TABLE, aid_list, 'img', config=chip_config)[0]
        >>> ut.quit_if_noshow()
        >>> pt.imshow(vt.blend_images_multiply(chip, vt.resize_mask(mask, chip)), title='mask')
        >>> pt.show_if_requested()
    """
    from plottool import interact_impaint
    # TODO: Ensure interactive required cache words
    # Keep manual things above the cache dir
    mask_dpath = ut.unixjoin(depc.cache_dpath, '../ManualChipMask')
    ut.ensuredir(mask_dpath)

    ibs = depc.controller
    #chip_config = ChipConfig(dim_size=None)
    chip_config = ChipConfig(dim_size=config['dim_size'])
    chip_imgs = depc.get_property('chips', aid_list, 'img',
                                  config=chip_config)

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


class ChipConfig(dtool.TableConfig):
    _param_info_list = [
        ut.ParamInfo('resize_dim', 'width',
                     valid_values=['area', 'width', 'height', 'diag', 'maxwh'],
                     hideif=lambda cfg: cfg['dim_size'] is None),
        #ut.ParamInfo('dim_size', 128, 'sz', hideif=None),
        ut.ParamInfo('dim_size', 960, 'sz', hideif=None),
        ut.ParamInfo('preserve_aspect', True, hideif=True),
        ut.ParamInfo('histeq', False, hideif=False),
        ut.ParamInfo('ext', '.png'),
    ]


@register_preproc(
    tablename='chips', parents=['annotations'],
    colnames=['img', 'width', 'height', 'M'],
    coltypes=[('extern', vt.imread, vt.imwrite), int, int, np.ndarray],
    configclass=ChipConfig,
    docstr='Used to store *processed* annots as chips',
    fname='chipcache4',
    version=0
)
def compute_chip(depc, aid_list, config=None):
    r"""
    Example of using the dependency cache.

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        python -m ibeis.core --exec-compute_chip --show --db humpbacks

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> depc = ibs.depc
        >>> config = ChipConfig(dim_size=None)
        >>> aid_list = ibs.get_valid_aids()[0:20]
        >>> chips = depc.get_property('chips', aid_list, 'img', {})
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(chips, nPerPage=4)
        >>> pt.show_if_requested()
    """
    print('Preprocess Chips')
    print('config = %r' % (config,))

    ibs = depc.controller
    chip_dpath = ibs.get_chipdir() + '2'
    #if config is None:
    #    config = ChipConfig()

    ut.ensuredir(chip_dpath)

    ext = config['ext']

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

    dim_size = config['dim_size']
    resize_dim = config['resize_dim']
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
