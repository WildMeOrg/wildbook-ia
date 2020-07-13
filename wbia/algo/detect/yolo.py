# -*- coding: utf-8 -*-
"""
Interface to pydarknet yolo object detection.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import vtool as vt
from six.moves import zip

(print, rrr, profile) = ut.inject2(__name__, '[yolo]')

if not ut.get_argflag('--no-pydarknet'):
    try:
        import pydarknet
    except ImportError:
        print(
            'WARNING Failed to import pydarknet. '
            'PyDarknet YOLO detection is unavailable'
        )
        if ut.SUPER_STRICT:
            raise


VERBOSE_DARK = ut.get_argflag('--verbdark') or ut.VERBOSE


# def train_gid_list(ibs, gid_list, trees_path=None, species=None, setup=True,


def detect_gid_list(ibs, gid_list, downsample=False, **kwargs):
    """
    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        downsample (bool, optional): a flag to indicate if the original image
            sizes should be used; defaults to True

            True:  ibs.get_image_detectpaths() is used
            False: ibs.get_image_paths() is used

    Kwargs (optional): refer to the PyDarknet documentation for configuration settings

    Args:
        ibs (wbia.IBEISController):  image analysis api
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        downsample (bool, optional): a flag to indicate if the original image
                sizes should be used; defaults to True

    Kwargs:
        detector, config_filepath, weights_filepath, verbose

    Yields:
        tuple: (gid, gpath, result_list)

    CommandLine:
        python -m wbia.algo.detect.yolo detect_gid_list --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.detect.yolo import *  # NOQA
        >>> from wbia.core_images import LocalizerConfig
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='WS_ALL')
        >>> gid_list = ibs.images()._rowids[0:1]
        >>> kwargs = config = LocalizerConfig(**{
        >>>     'weights_filepath': '/media/raid/work/WS_ALL/localizer_backup/detect.yolo.2.39000.weights',
        >>>     'config_filepath': '/media/raid/work/WS_ALL/localizer_backup/detect.yolo.2.cfg',
        >>> })
        >>> exec(ut.execstr_dict(config), globals())
        >>> #classes_fpath = '/media/raid/work/WS_ALL/localizer_backup/detect.yolo.2.cfg.classes'
        >>> downsample = False
        >>> (gid, gpath, result_list) = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> result = ('(gid, gpath, result_list) = %s' % (ut.repr2((gid, gpath, result_list)),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()

    Yields:
        results (list of dict)
    """
    # Get new gpaths if downsampling
    if downsample:
        gpath_list = ibs.get_image_detectpaths(gid_list)
        neww_list = [vt.open_image_size(gpath)[0] for gpath in gpath_list]
        oldw_list = [oldw for (oldw, oldh) in ibs.get_image_sizes(gid_list)]
        downsample_list = [oldw / neww for oldw, neww in zip(oldw_list, neww_list)]
        orient_list = [1] * len(gid_list)
    else:
        gpath_list = ibs.get_image_paths(gid_list)
        downsample_list = [None] * len(gpath_list)
        orient_list = ibs.get_image_orientation(gid_list)
    # Run detection
    results_iter = detect(gpath_list, **kwargs)
    # Upscale the results
    _iter = zip(downsample_list, gid_list, orient_list, results_iter)
    for downsample, gid, orient, (gpath, result_list) in _iter:
        # Upscale the results back up to the original image size
        for result in result_list:
            if downsample is not None and downsample != 1.0:
                for key in ['xtl', 'ytl', 'width', 'height']:
                    result[key] = int(result[key] * downsample)
            bbox = (
                result['xtl'],
                result['ytl'],
                result['width'],
                result['height'],
            )
            bbox_list = [bbox]
            bbox = bbox_list[0]
            result['xtl'], result['ytl'], result['width'], result['height'] = bbox
        yield (gid, gpath, result_list)


def detect(
    gpath_list, detector=None, config_filepath=None, weights_filepath=None, **kwargs
):
    """
    Args:
        gpath_list (list of str): the list of image paths that need detection

    Kwargs (optional): refer to the PyDarknet documentation for configuration settings

    Returns:
        iter

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.detect.yolo import *  # NOQA
        >>> from wbia.core_images import LocalizerConfig
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='WS_ALL')
        >>> gid_list = ibs.images()._rowids[0:1]
        >>> gpath_list = ibs.get_image_paths(gid_list)
        >>> dpath = '/media/raid/work/WS_ALL/localizer_backup/'
        >>> weights_filepath = join(dpath, 'detect.yolo.2.39000.weights')
        >>> config_filepath = join(dpath, 'detect.yolo.2.cfg')
        >>> config = LocalizerConfig(
        >>>     weights_filepath=weights_filepath,
        >>>     config_filepath=config_filepath,
        >>> )
        >>> kwargs = config.asdict()
        >>> ut.delete_dict_keys(kwargs, ['weights_filepath', 'config_filepath'])
        >>> ut.delete_dict_keys(kwargs, ['thumbnail_cfg', 'species', 'algo'])
    """
    # Run detection
    if detector is None:
        classes_filepath = kwargs.pop('classes_filepath', None)
        verbose = kwargs.get('verbose', False)
        detector = pydarknet.Darknet_YOLO_Detector(
            config_filepath=config_filepath,
            weights_filepath=weights_filepath,
            classes_filepath=classes_filepath,
            verbose=verbose,
        )
    # dark = detector
    # input_gpath_list = gpath_list
    results_iter = detector.detect(gpath_list, **kwargs)
    results_list = list(results_iter)
    del detector
    return results_list
