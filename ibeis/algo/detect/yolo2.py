# -*- coding: utf-8 -*-
"""
Interface to pytorch yolo2 object detection.
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip
import utool as ut
import tempfile
import os
(print, rrr, profile) = ut.inject2(__name__, '[yolo2]')

if not ut.get_argflag('--no-yolo2'):
    try:
        import yolo2
    except ImportError as ex:
        print('WARNING Failed to import yolo2. '
              'PyTorch YOLO v2 detection is unavailable')
        if ut.SUPER_STRICT:
            raise


VERBOSE_YOLO2 = ut.get_argflag('--verbyolo2') or ut.VERBOSE


def detect_gid_list(ibs, gid_list, **kwargs):
    """
    Args:
        ibs (ibeis.IBEISController):  image analysis api
        gid_list (list of int): the list of IBEIS image_rowids that need detection

    Kwargs:
        detector, config_filepath, weights_filepath, verbose

    Yields:
        tuple: (gid, gpath, result_list)

    CommandLine:
        python -m ibeis.algo.detect.yolo22 detect_gid_list --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.detect.yolo2 import *  # NOQA
        >>> from ibeis.core_images import LocalizerConfig
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='WS_ALL')
        >>> gid_list = ibs.images()._rowids[0:1]
        >>> kwargs = config = LocalizerConfig(**{
        >>>     'weights_filepath': '/media/raid/work/WS_ALL/localizer_backup/detect.yolo2.2.39000.weights',
        >>>     'config_filepath': '/media/raid/work/WS_ALL/localizer_backup/detect.yolo2.2.cfg',
        >>> })
        >>> exec(ut.execstr_dict(config), globals())
        >>> #classes_fpath = '/media/raid/work/WS_ALL/localizer_backup/detect.yolo2.2.cfg.classes'
        >>> downsample = False
        >>> (gid, gpath, result_list) = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> result = ('(gid, gpath, result_list) = %s' % (ut.repr2((gid, gpath, result_list)),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()

    Yields:
        results (list of dict)
    """
    # Get new gpaths if downsampling
    gpath_list = ibs.get_image_paths(gid_list)
    orient_list = ibs.get_image_orientation(gid_list)
    # Run detection
    results_iter = detect(gpath_list, **kwargs)
    # Upscale the results
    _iter = zip(gid_list, orient_list, results_iter)
    for gid, orient, (gpath, result_list) in _iter:
        # Upscale the results back up to the original image size
        for result in result_list:
            bbox = (result['xtl'], result['ytl'], result['width'], result['height'], )
            bbox_list = [ bbox ]
            bbox_list = ibs.fix_horizontal_bounding_boxes_to_orient(gid, bbox_list)
            bbox = bbox_list[0]
            result['xtl'], result['ytl'], result['width'], result['height'] = bbox
        yield (gid, gpath, result_list)


def detect(gpath_list, **kwargs):
    """
    Args:
        gpath_list (list of str): the list of image paths that need detection

    Kwargs (optional): refer to the PyTorch YOLO v2 documentation for configuration settings

    Returns:
        iter

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.detect.yolo2 import *  # NOQA
        >>> from ibeis.core_images import LocalizerConfig
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='WS_ALL')
        >>> gid_list = ibs.images()._rowids[0:1]
        >>> gpath_list = ibs.get_image_paths(gid_list)
        >>> dpath = '/media/raid/work/WS_ALL/localizer_backup/'
        >>> weights_filepath = join(dpath, 'detect.yolo2.2.39000.weights')
        >>> config_filepath = join(dpath, 'detect.yolo2.2.cfg')
        >>> config = LocalizerConfig(
        >>>     weights_filepath=weights_filepath,
        >>>     config_filepath=config_filepath,
        >>> )
        >>> kwargs = config.asdict()
        >>> ut.delete_dict_keys(kwargs, ['weights_filepath', 'config_filepath'])
        >>> ut.delete_dict_keys(kwargs, ['thumbnail_cfg', 'species', 'algo'])
    """
    # Run detection
    temp_file, temp_filepath = tempfile.mkstemp(suffix='.txt')
    os.close(temp_file)

    with open(temp_filepath, 'w') as temp_file:
        for gpath in gpath_list:
            temp_file.write('%s\n' % (gpath, ))

    verbose = VERBOSE_YOLO2
    results_iter = yolo2.valid.validate(temp_filepath, verbose=verbose,
                                        **kwargs)
    results_list = list(results_iter)
    os.remove(temp_filepath)

    return results_list
