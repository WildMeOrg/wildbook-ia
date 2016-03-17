# -*- coding: utf-8 -*-
"""
Preprocess Chips

Extracts annotation chips from imaages and applies optional image
normalizations.

TODO:
    * Dependency Cache from flukes
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range, filter  # NOQA
# from os.path import exists
# import dtool
import utool as ut
# import vtool as vt
# import numpy as np
#from ibeis import depends_cache
# from ibeis.control.controller_inject import register_preproc
# from ibeis import constants as const
#ut.noinject('[preproc_chip]')
(print, rrr, profile) = ut.inject2(__name__, '[preproc_chip]')
from ibeis.algo.preproc.old_chip_preproc import *  # NOQA


#class AnnotMaskConfig(dtool.TableConfig):
#    def get_param_info_list(self):
#        return []
# DEFAULT_ANNOT_MASK_CONFIG = {
#     'dim_size': 500,
# }


# @register_preproc(
#     'annotmask',
#     parents=[const.ANNOTATION_TABLE],
#     colnames=['img', 'width', 'height'],
#     coltypes=[('extern', vt.imread), int, int],
#     configclass=DEFAULT_ANNOT_MASK_CONFIG,
#     #configclass=AnnotMaskConfig,
#     docstr='Used to store *processed* annots as chips',
#     fname='../maskcache2',
#     isinteractive=True,
# )
# def preproc_annotmask(depc, aid_list, config=None):
#     r"""
#     Example of using the dependency cache.

#     # TODO: make coltypes take imwrite and return just
#     # the image and let dtool save it where it wants

#     Args:
#         depc (ibeis.depends_cache.DependencyCache):
#         aid_list (list):  list of annotation rowids
#         config2_ (dict): (default = None)

#     Yields:
#         (uri, int, int): tup

#     CommandLine:
#         python -m ibeis.algo.preproc.preproc_chip --exec-preproc_annotmask --show
#         python -m ibeis.algo.preproc.preproc_chip --exec-preproc_annotmask --show --edit

#     NOTES:
#         HOW TO DESIGN INTERACTIVE PLOTS:
#             decorate as interactive

#             depc.get_property(recompute=True)

#             instead of calling preproc as a generator and then adding,
#             calls preproc and passes in a callback function.
#             preproc spawns interaction and must call callback function when finished.

#             callback function adds the rowids to the table.

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from ibeis.algo.preproc.preproc_chip import *  # NOQA
#         >>> import ibeis
#         >>> ibs = ibeis.opendb('testdb1')
#         >>> depc = ibs.depc_annot
#         >>> depc.print_all_tables()
#         >>> aid_list = ibs.get_valid_aids()[0:2]
#         >>> chip_config = ChipConfig(dim_size=None)
#         >>> edit = ut.get_argflag('--edit')
#         >>> mask = depc.get_property('annotmask', aid_list, 'img', recompute=edit)[0]
#         >>> chip = depc.get_property(const.CHIP_TABLE, aid_list, 'img', config=chip_config)[0]
#         >>> import plottool as pt
#         >>> ut.quit_if_noshow()
#         >>> pt.imshow(vt.blend_images_multiply(chip, vt.resize_mask(mask, chip)), title='mask')
#         >>> pt.show_if_requested()
#         >>> #depc.print_all_tables()
#     """
#     # TODO: Ensure interactive required cache words
#     # Keep manual things above the cache dir
#     mask_dpath = ut.unixjoin(depc.cache_dpath, '../ManualChipMask')
#     ut.ensuredir(mask_dpath)

#     ibs = depc.controller
#     #chip_config = ChipConfig(dim_size=None)
#     chip_config = ChipConfig(dim_size=config['dim_size'])
#     chip_imgs = depc.get_property(const.CHIP_TABLE, aid_list, 'img',
#                                   config=chip_config)

#     cfghashid = config.get_hashid()
#     avuuid_list = ibs.get_annot_visual_uuids(aid_list)

#     # TODO: just hash everything together
#     ext = '.png'
#     _fmt = 'mask_aid_{aid}_avuuid_{avuuid}_{cfghashid}{ext}'
#     fname_list = [_fmt.format(aid=aid, avuuid=avuuid, ext=ext, cfghashid=cfghashid)
#                    for aid, avuuid in zip(aid_list, avuuid_list)]

#     from plottool import interact_impaint
#     #import plottool as pt
#     #pt.ensure_pylab_qt4()
#     #for uri, w, h in generate_chip_properties(ibs, aid_list, config2_=config):

#     for img, fname, aid in zip(chip_imgs, fname_list, aid_list):
#         mask_fpath = ut.unixjoin(mask_dpath, fname)
#         if exists(mask_fpath):
#             # Allow for editing on recompute
#             init_mask = vt.imread(mask_fpath)
#         else:
#             init_mask = None
#         mask = interact_impaint.impaint_mask2(img, init_mask=init_mask)
#         #mask = interact_impaint.impaint_mask(img, init_mask=init_mask)
#         vt.imwrite(mask_fpath, mask)
#         print('imwrite')
#         w, h = vt.get_size(mask)

#         yield mask_fpath, w, h
#         # Remove the old chips
#         #ibs.delete_annot_chips([aid])
#         #ibs.delete_annot_chip_thumbs([aid])


# class ChipConfig(dtool.TableConfig):
#     def get_param_info_list(self):
#         return [
#             ut.ParamInfo('resize_dim', 'width',
#                          valid_values=['area', 'width', 'height', 'diag', 'maxwh'],
#                          hideif=lambda cfg: cfg['dim_size'] is None),
#             #ut.ParamInfo('dim_size', 128, 'sz', hideif=None),
#             ut.ParamInfo('dim_size', 960, 'sz', hideif=None),
#             ut.ParamInfo('preserve_aspect', True, hideif=True),
#             ut.ParamInfo('histeq', False, hideif=False),
#             ut.ParamInfo('ext', '.png'),
#         ]


# # NEW CHIP TABLE
# @register_preproc(
#     const.CHIP_TABLE,
#     parents=[const.ANNOTATION_TABLE],
#     colnames=['img', 'width', 'height', 'M'],
#     coltypes=[('extern', vt.imread), int, int, np.ndarray],
#     configclass=ChipConfig,
#     docstr='Used to store *processed* annots as chips',
#     fname='chipcache4',
#     version=0
# )
# def preproc_chip(depc, aid_list, config=None):
#     r"""
#     Example of using the dependency cache.

#     Args:
#         depc (ibeis.depends_cache.DependencyCache):
#         aid_list (list):  list of annotation rowids
#         config2_ (dict): (default = None)

#     Yields:
#         (uri, int, int): tup

#     CommandLine:
#         python -m ibeis.algo.preproc.preproc_chip --exec-preproc_chip --show --db humpbacks

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from ibeis.algo.preproc.preproc_chip import *  # NOQA
#         >>> import ibeis
#         >>> ibs = ibeis.opendb(defaultdb='testdb1')
#         >>> depc = ibs.depc_annot
#         >>> #config = ChipConfig(dim_size=None)
#         >>> config = None  # ChipConfig(dim_size=None)
#         >>> aid_list = ibs.get_valid_aids()[0:20]
#         >>> chips = depc.get_property(ibs.const.CHIP_TABLE, aid_list, 'img', {})
#         >>> #procgen = preproc_chip(depc, aid_list, config)
#         >>> #chip_props = list(procgen)
#         >>> #chips = ut.take_column(chip_props, 0)
#         >>> import plottool as pt
#         >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(chips, nPerPage=4)
#         >>> pt.show_if_requested()
#         >>> #depc[const.CHIP_TABLE].print_csv()
#     """
#     print('Preprocess Chips')
#     print(config)

#     ibs = depc.controller
#     chip_dpath = ibs.get_chipdir() + '2'
#     #if config is None:
#     #    config = ChipConfig()

#     ut.ensuredir(chip_dpath)

#     ext = config['ext']

#     cfghashid = config.get_hashid()
#     avuuid_list = ibs.get_annot_visual_uuids(aid_list)

#     # TODO: just hash everything together
#     _fmt = 'chip_aid_{aid}_avuuid_{avuuid}_{cfghashid}{ext}'
#     cfname_list = [_fmt.format(aid=aid, avuuid=avuuid, ext=ext, cfghashid=cfghashid)
#                    for aid, avuuid in zip(aid_list, avuuid_list)]
#     cfpath_list = [ut.unixjoin(chip_dpath, chip_fname)
#                    for chip_fname in cfname_list]

#     gfpath_list = ibs.get_annot_image_paths(aid_list)
#     bbox_list   = ibs.get_annot_bboxes(aid_list)
#     theta_list  = ibs.get_annot_thetas(aid_list)
#     bbox_size_list = ut.take_column(bbox_list, [2, 3])

#     # Checks
#     invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
#     invalid_aids = ut.compress(aid_list, invalid_flags)
#     assert len(invalid_aids) == 0, 'invalid aids=%r' % (invalid_aids,)

#     dim_size = config['dim_size']
#     resize_dim = config['resize_dim']
#     scale_func_dict = {
#         'width': vt.get_scaled_size_with_width,
#         'root_area': vt.get_scaled_size_with_area,
#     }
#     scale_func = scale_func_dict[resize_dim]

#     if dim_size is None:
#         newsize_list = bbox_size_list
#     else:
#         if resize_dim == 'root_area':
#             dim_size = dim_size ** 2
#         newsize_list = [scale_func(dim_size, w, h) for (w, h) in bbox_size_list]

#     # Build transformation from image to chip
#     M_list = [vt.get_image_to_chip_transform(bbox, new_size, theta) for
#               bbox, theta, new_size in zip(bbox_list, theta_list, newsize_list)]

#     arg_iter = zip(cfpath_list, gfpath_list, newsize_list, M_list)
#     arg_list = list(arg_iter)

#     import cv2

#     flags = cv2.INTER_LANCZOS4
#     borderMode = cv2.BORDER_CONSTANT
#     warpkw = dict(flags=flags, borderMode=borderMode)

#     for tup in ut.ProgIter(arg_list, lbl='computing chips'):
#         cfpath, gfpath, new_size, M = tup
#         # Read parent image
#         imgBGR = vt.imread(gfpath)
#         # Warp chip
#         chipBGR = cv2.warpAffine(imgBGR, M[0:2], tuple(new_size), **warpkw)
#         width, height = vt.get_size(chipBGR)
#         # Write chip to disk
#         vt.imwrite(cfpath, chipBGR)
#         yield (cfpath, width, height, M)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.preproc.preproc_chip
        python -m ibeis.algo.preproc.preproc_chip --allexamples --serial --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut  # NOQA
    ut.doctest_funcs()
