# -*- coding: utf-8 -*-
"""Interface to Azure object proposals."""
from __future__ import absolute_import, division, print_function
import requests
import utool as ut
import numpy as np
from six.moves import zip
from os.path import abspath, dirname, expanduser, join, exists  # NOQA
(print, rrr, profile) = ut.inject2(__name__, '[azure]')


VERBOSE_AZURE = ut.get_argflag('--verbazure') or ut.VERBOSE


NPROC_MULTIPLIER = 2


PREDICTION_URL = 'https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/%s/image?iterationId=%s'
PREDICTION_HEADER = {
    'Prediction-Key': None,
    'Content-Type': 'application/octet-stream'
}
PREDICTION_DICT = {
    None  : ('9bb5790b-7f59-4c0b-b571-21e68d29f4b2', 'a4fb7280-b0be-4706-91c6-7651d116ac46', '34e5c511adfc449290e10868218906f9'),
}


def detect_gid_list(ibs, gid_list, verbose=VERBOSE_AZURE, **kwargs):
    """Detect gid_list with azure.

    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection

    Kwargs (optional): refer to the Azure documentation for configuration settings

    Args:
        ibs (ibeis.IBEISController):  image analysis api
        gid_list (list of int): the list of IBEIS image_rowids that need detection

    Kwargs:
        detector, config_filepath, weights_filepath, verbose

    Yields:
        tuple: (gid, gpath, result_list)
    """
    # Get new gpaths if downsampling
    config = {
        'draw_annots': False,
        'thumbsize': 900,
    }
    gpath_list = ibs.get_image_thumbpath(gid_list, ensure_paths=True, **config)
    size_list = ibs.get_image_sizes(gid_list)

    # Run detection
    results_iter = detect(gpath_list, verbose=verbose, **kwargs)

    # Upscale the results
    _iter = zip(gid_list, size_list, results_iter)
    for gid, size, (gpath, result_list) in _iter:
        width, height = size

        # Upscale the results back up to the original image size
        for result in result_list:
            result['xtl']    = int(np.around(result['xtl']    * width ))
            result['ytl']    = int(np.around(result['ytl']    * height))
            result['width']  = int(np.around(result['width']  * width ))
            result['height'] = int(np.around(result['height'] * height))

        yield (gid, gpath, result_list)


def _detect(gpath, prediction_project, prediction_iteration, prediction_model):
    with open(gpath, 'rb') as image_file:
        data = image_file.read()

    prediction_url = PREDICTION_URL % (prediction_project, prediction_iteration, )
    prediction_header = PREDICTION_HEADER.copy()
    prediction_header['Prediction-Key'] = prediction_model
    response = requests.post(url=prediction_url, data=data, headers=prediction_header)

    response_json = response.json()
    output_list = response_json['predictions']

    return output_list


def detect(gpath_list, config_filepath, verbose=VERBOSE_AZURE, **kwargs):
    """Detect image filepaths with azure.

    Args:
        gpath_list (list of str): the list of image paths that need proposal candidates

    Kwargs (optional): refer to the Azure documentation for configuration settings

    Returns:
        iter
    """
    # Get correct weight if specified with shorthand
    if config_filepath not in PREDICTION_DICT:
        config_filepath = None

    prediction = PREDICTION_DICT.get(config_filepath, None)
    assert prediction is not None, 'Azure needs to have a model configuration'
    prediction_project, prediction_iteration, prediction_model = prediction

    prediction_project_list   = [prediction_project] * len(gpath_list)
    prediction_iteration_list = [prediction_iteration] * len(gpath_list)
    prediction_model_list     = [prediction_model]   * len(gpath_list)
    arg_iter = list(zip(gpath_list, prediction_project_list, prediction_iteration_list, prediction_model_list))
    nprocs = ut.util_parallel.get_default_numprocs()
    nprocs *= NPROC_MULTIPLIER
    nprocs = min(nprocs, len(arg_iter))
    outputs_list = ut.util_parallel.generate2(_detect, arg_iter, nprocs=nprocs, ordered=True)

    # Execute detector for each image
    results_list_ = []
    for output_list in outputs_list:
        result_list_ = []
        for output in list(output_list):
            result_dict = {
                'xtl'        : output['boundingBox']['left'],
                'ytl'        : output['boundingBox']['top'],
                'width'      : output['boundingBox']['width'],
                'height'     : output['boundingBox']['height'],
                'class'      : output['tagName'],
                'confidence' : output['probability'],
            }
            result_list_.append(result_dict)
        results_list_.append(result_list_)

    if len(results_list_) != len(gpath_list):
        raise ValueError('Azure did not return valid data')

    results_list = zip(gpath_list, results_list_)
    return results_list
