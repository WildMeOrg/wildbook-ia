# -*- coding: utf-8 -*-
"""
Interface to Selective Search object proposals.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import vtool as vt
from six.moves import zip
import tempfile
import subprocess
import shlex
import os
from os.path import abspath, dirname, expanduser, join, exists  # NOQA
import numpy as np
import scipy.io

(print, rrr, profile) = ut.inject2(__name__, '[selective search]')

# SCRIPT_PATH = abspath(dirname(__file__))
SCRIPT_PATH = abspath(expanduser(join('~', 'code', 'selective_search_ijcv_with_python')))

if not ut.get_argflag('--no-selective-search'):
    try:
        assert exists(SCRIPT_PATH)
    except AssertionError:
        print(
            'WARNING Failed to find selective_search_ijcv_with_python. '
            'Selective Search is unavailable'
        )
        # if ut.SUPER_STRICT:
        #     raise


VERBOSE_SS = ut.get_argflag('--verbdss') or ut.VERBOSE


def detect_gid_list(ibs, gid_list, downsample=True, verbose=VERBOSE_SS, **kwargs):
    """
    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        downsample (bool, optional): a flag to indicate if the original image
            sizes should be used; defaults to True

            True:  ibs.get_image_detectpaths() is used
            False: ibs.get_image_paths() is used

    Kwargs (optional): refer to the Selective Search documentation for configuration settings

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
        python -m wbia.algo.detect.selectivesearch detect_gid_list --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.detect.selectivesearch import *  # NOQA
        >>> from wbia.core_images import LocalizerConfig
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> config = {'matlab_command': 'selective_search', 'verbose': True}
        >>> downsample = False
        >>> results_list = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> results_list = list(results_list)
        >>> print('result lens = %r' % (map(len, list(results_list))))
        >>> print('result[0] = %r' % (len(list(results_list[0][2]))))
        >>> config = {'matlab_command': 'selective_search_rcnn', 'verbose': True}
        >>> downsample = False
        >>> results_list = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> results_list = list(results_list)
        >>> print('result lens = %r' % (map(len, list(results_list))))
        >>> print('result[0] = %r' % (len(list(results_list[0][2]))))
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
    results_iter = detect(gpath_list, verbose=verbose, **kwargs)
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


def detect(gpath_list, matlab_command='selective_search', verbose=VERBOSE_SS, **kwargs):
    """
    Args:
        gpath_list (list of str): the list of image paths that need proposal candidates

    Kwargs (optional): refer to the Selective Search documentation for configuration settings

    Returns:
        iter
    """
    # Form the MATLAB script command that processes images and write to
    # temporary results file.
    temp_file, temp_filepath = tempfile.mkstemp(suffix='.mat')
    os.close(temp_file)
    gpath_str = '{%s}' % (','.join(["'%s'" % (gpath,) for gpath in gpath_list]))
    matlab_command_str = "%s(%s, '%s')" % (matlab_command, gpath_str, temp_filepath)
    if verbose:
        print('Calling: %s' % (matlab_command_str,))

    # Execute command in MATLAB.
    bash_command = 'matlab -nojvm -r "try; %s; catch; exit; end; exit"'
    bash_str = bash_command % (matlab_command_str,)
    bash_list = shlex.split(bash_str)
    with open('/dev/null', 'w') as null:
        process_id = subprocess.Popen(bash_list, stdout=null, cwd=SCRIPT_PATH)
        process_return_code = process_id.wait()
        if process_return_code != 0:
            raise RuntimeError('Matlab selective search did not exit successfully')

    # Read the results and undo Matlab's 1-based indexing.
    boxes_list = list(scipy.io.loadmat(temp_filepath)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0))[np.newaxis, :]
    results_list = [boxes - subtractor for boxes in boxes_list]

    if len(results_list) != len(gpath_list):
        raise ValueError('Matlab selective search did not return valid data')
    # Remove temporary file, and return.
    os.remove(temp_filepath)

    # Pack results
    results_list_ = []
    for result_list in results_list:
        result_list_ = []
        for result in result_list:
            xtl = int(np.around(result[0]))
            ytl = int(np.around(result[1]))
            xbr = int(np.around(result[2]))
            ybr = int(np.around(result[3]))
            result_dict = {
                'xtl': xtl,
                'ytl': ytl,
                'width': xbr - xtl,
                'height': ybr - ytl,
                'class': None,
                'confidence': 1.0,
            }
            result_list_.append(result_dict)
        results_list_.append(result_list_)

    results_list = zip(gpath_list, results_list_)
    return results_list
