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
    except ImportError as ex:
        if ut.SUPER_STRICT:
            print('WARNING Failed to import pydarknet. '
                  'PyDarknet YOLO detection is unavailable')
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

    Yields:
        results (list of dict)
    """
    # Get new gpaths if downsampling
    if downsample:
        gpath_list = ibs.get_image_detectpaths(gid_list)
        neww_list = [vt.open_image_size(gpath)[0] for gpath in gpath_list]
        oldw_list = [oldw for (oldw, oldh) in ibs.get_image_sizes(gid_list)]
        downsample_list = [oldw / neww for oldw, neww in zip(oldw_list, neww_list)]
    else:
        gpath_list = ibs.get_image_paths(gid_list)
        downsample_list = [None] * len(gpath_list)
    # Run detection
    results_iter = detect(gpath_list, **kwargs)
    # Upscale the results
    for downsample, gid, (gpath, result_list) in zip(downsample_list, gid_list, results_iter):
        # Upscale the results back up to the original image size
        if downsample is not None and downsample != 1.0:
            for result in result_list:
                for key in ['xtl', 'ytl', 'width', 'height']:
                    result[key] = int(result[key] * downsample)
        yield (gid, gpath, result_list)


def detect(gpath_list, detector=None, config_filepath=None, weight_filepath=None,
           **kwargs):
    """
    Args:
        gpath_list (list of str): the list of image paths that need detection

    Kwargs (optional): refer to the PyDarknet documentation for configuration settings

    Returns:
        iter
    """
    # Run detection
    if detector is None:
        verbose = kwargs.get('verbose', False)
        detector = pydarknet.Darknet_YOLO_Detector(config_filepath=config_filepath,
                                                   weight_filepath=weight_filepath,
                                                   verbose=verbose)
    results_iter = detector.detect(gpath_list, **kwargs)
    return results_iter
