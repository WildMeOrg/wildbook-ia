# -*- coding: utf-8 -*-
"""
Interface to Darknet object proposals.
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

(print, rrr, profile) = ut.inject2(__name__, '[darknet]')

# SCRIPT_PATH = abspath(dirname(__file__))
SCRIPT_PATH = abspath(expanduser(join('~', 'code', 'darknet')))

if not ut.get_argflag('--no-darknet'):
    try:
        assert exists(SCRIPT_PATH)
    except AssertionError:
        print('WARNING Failed to find darknet. ' 'Darknet is unavailable')
        # if ut.SUPER_STRICT:
        #     raise


VERBOSE_SS = ut.get_argflag('--verbdss') or ut.VERBOSE


CONFIG_URL_DICT = {
    # 'pretrained-v1-pascal'       : 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.v1.pascal.cfg',
    'pretrained-v2-pascal': 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.v2.pascal.cfg',
    'pretrained-v2-large-pascal': 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.v2.large.pascal.cfg',
    'pretrained-tiny-pascal': 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.tiny.pascal.cfg',
    'pretrained-v2-large-coco': 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.v2.large.coco.cfg',
    'pretrained-tiny-coco': 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.tiny.coco.cfg',
    'default': 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.v2.large.coco.cfg',
    None: 'https://wildbookiarepository.azureedge.net/models/pretrained.darknet.v2.large.coco.cfg',
}


def _parse_weight_from_cfg(url):
    return url.replace('.cfg', '.weights')


def _parse_data_from_cfg(url):
    return url.replace('.cfg', '.data')


def _parse_classes_from_cfg(url):
    return url.replace('.cfg', '.classes')


def _parse_class_list(classes_filepath):
    # Load classes from file into the class list
    assert exists(classes_filepath)
    class_list = []
    with open(classes_filepath) as classes:
        for line in classes.readlines():
            line = line.strip()
            if len(line) > 0:
                class_list.append(line)
    return class_list


def detect_gid_list(ibs, gid_list, downsample=True, verbose=VERBOSE_SS, **kwargs):
    """
    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        downsample (bool, optional): a flag to indicate if the original image
            sizes should be used; defaults to True

            True:  ibs.get_image_detectpaths() is used
            False: ibs.get_image_paths() is used

    Kwargs (optional): refer to the Darknet documentation for configuration settings

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
        python -m wbia.algo.detect.darknet detect_gid_list --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.detect.darknet import *  # NOQA
        >>> from wbia.core_images import LocalizerConfig
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> config = {'verbose': True}
        >>> downsample = False
        >>> results_list = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> results_list = list(results_list)
        >>> print('result lens = %r' % (map(len, list(results_list))))
        >>> print('result[0] = %r' % (len(list(results_list[0][2]))))
        >>> config = {'verbose': True}
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


def detect(
    gpath_list,
    config_filepath,
    weight_filepath,
    class_filepath,
    sensitivity,
    verbose=VERBOSE_SS,
    use_gpu=True,
    use_gpu_id=0,
    **kwargs,
):
    """
    Args:
        gpath_list (list of str): the list of image paths that need proposal candidates

    Kwargs (optional): refer to the Darknet documentation for configuration settings

    Returns:
        iter
    """
    # Get correct config if specified with shorthand
    config_url = None
    if config_filepath in CONFIG_URL_DICT:
        config_url = CONFIG_URL_DICT[config_filepath]
        config_filepath = ut.grab_file_url(config_url, appname='wbia', check_hash=True)

    # Get correct weights if specified with shorthand
    if weight_filepath in CONFIG_URL_DICT:
        if weight_filepath is None and config_url is not None:
            config_url_ = config_url
        else:
            config_url_ = CONFIG_URL_DICT[weight_filepath]
        weight_url = _parse_weight_from_cfg(config_url_)
        weight_filepath = ut.grab_file_url(weight_url, appname='wbia', check_hash=True)

    data_url = _parse_data_from_cfg(config_url)
    data_filepath = ut.grab_file_url(
        data_url, appname='wbia', check_hash=True, verbose=verbose
    )
    with open(data_filepath, 'r') as data_file:
        data_str = data_file.read()
    names_tag = '_^_NAMES_^_'
    if names_tag in data_str:
        class_url = _parse_classes_from_cfg(config_url)
        class_filepath = ut.grab_file_url(
            class_url, appname='wbia', check_hash=True, verbose=verbose
        )
        data_str = data_str.replace(names_tag, class_filepath)
        with open(data_filepath, 'w') as data_file:
            data_file.write(data_str)

    # Form the temporary results file that the code will write to
    temp_file, temp_filepath = tempfile.mkstemp(suffix='.txt')
    os.close(temp_file)

    # Execute command for each image
    results_list_ = []
    for gpath in gpath_list:
        # # Clear the contents of the file (C code should do this instead)
        # with open(temp_filepath, 'w'):
        #     pass

        # Run darknet on image
        bash_args = (
            data_filepath,
            config_filepath,
            weight_filepath,
            gpath,
            temp_filepath,
            sensitivity,
        )
        bash_str = './darknet detector test %s %s %s %s %s -thresh %0.5f' % bash_args
        if verbose:
            print('Calling: %s' % (bash_str,))
        bash_list = shlex.split(bash_str)
        with open('/dev/null', 'w') as null:
            process_id = subprocess.Popen(bash_list, stdout=null, cwd=SCRIPT_PATH)
            process_return_code = process_id.wait()
            if process_return_code != 0:
                raise RuntimeError('Darknet did not exit successfully')

        # Load the temporary file and load it's contents
        with open(temp_filepath, 'r') as temp_file:
            temps_str = temp_file.read()

        temps_list = temps_str.split('\n\n')
        # Parse results from output file
        result_list_ = []
        for temp_str in temps_list:
            temp_str = temp_str.strip()
            if len(temp_str) == 0:
                continue
            result = temp_str.split('\n')
            gpath_ = result[0]
            assert gpath == gpath_
            xtl = int(np.around(float(result[1])))
            ytl = int(np.around(float(result[2])))
            xbr = int(np.around(float(result[3])))
            ybr = int(np.around(float(result[4])))
            class_ = result[5]
            conf = float(result[6])
            result_dict = {
                'xtl': xtl,
                'ytl': ytl,
                'width': xbr - xtl,
                'height': ybr - ytl,
                'class': class_,
                'confidence': conf,
            }
            result_list_.append(result_dict)
        results_list_.append(result_list_)

    if len(results_list_) != len(gpath_list):
        raise ValueError('Darknet did not return valid data')

    # Remove temporary file, and return.
    os.remove(temp_filepath)

    results_list = zip(gpath_list, results_list_)
    return results_list
