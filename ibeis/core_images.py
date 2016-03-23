# -*- coding: utf-8 -*-
"""
IBEIS CORE
Defines the core dependency cache supported by the image analysis api

Extracts detection results from images and applies additional processing
automatically

TODO:

NOTES:
    HOW TO DESIGN INTERACTIVE PLOTS:
        decorate as interactive

        depc.get_property(recompute=True)

        instead of calling preproc as a generator and then adding,
        calls preproc and passes in a callback function.
        preproc spawns interaction and must call callback function when finished.

        callback function adds the rowids to the table.

Needed Tables:
    Detections
    QualityClassifier
    ViewpointClassifier

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import dtool
import utool as ut
import numpy as np
from ibeis.control.controller_inject import register_preprocs
(print, rrr, profile) = ut.inject2(__name__, '[core_images]')


class DetectionConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('algo', 'cnn'),
        ut.ParamInfo('species', 'zebra_plains', hideif='zebra_plains'),
    ]


register_preproc = register_preprocs['image']


@register_preproc(
    tablename='detections', parents=['images'],
    colnames=['score', 'bboxes', 'thetas', 'confs', 'classes'],
    coltypes=[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=DetectionConfig,
    fname='imgdetects',
    chunksize=32,
)
def compute_detections(depc, gid_list, config=None):
    r"""
    Extracts the detections for a given input image

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray): tup

    CommandLine:
        ibeis --tf compute_detections

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_images import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> print(depc.get_tablenames())
        >>> gid_list = ibs.get_valid_gids()[0:10]
        >>> config = {'algo': 'yolo'}
        >>> detects = depc.get_property('detections', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'pyrf'}
        >>> detects = depc.get_property('detections', gid_list, 'bboxes', config=config)
        >>> print(detects)
    """
    def package_to_numpy(key_list, result_list, score):
        temp = [
            [
                key[0] if isinstance(key, tuple) else result[key]
                for key in key_list
            ]
            for result in result_list
        ]
        return (
            score,
            np.array([ _[0:4] for _ in temp ]),
            np.array([ _[4]   for _ in temp ]),
            np.array([ _[5]   for _ in temp ]),
            np.array([ _[6]   for _ in temp ]),
        )

    print('[ibs] Preprocess Detections')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    ibs.assert_valid_gids(gid_list)
    base_key_list = ['xtl', 'ytl', 'width', 'height', 'theta', 'confidence', 'class']
    # Temporary for all detectors
    base_key_list[4] = (0.0, )  # Theta

    if config['algo'] in ['yolo']:
        from ibeis.algo.detect import yolo
        print('[ibs] detecting using CNN YOLO')
        detect_gen = yolo.detect_gid_list(ibs, gid_list, **config)
    elif config['algo'] in ['pyrf']:
        from ibeis.algo.detect import randomforest
        print('[ibs] detecting using Random Forests')
        base_key_list[6] = (config['species'], )  # class == species
        detect_gen = randomforest.detect_gid_list_with_species(ibs, gid_list, **config)
    else:
        raise ValueError('specified detection algo is not supported')

    # yield detections
    for gid, gpath, result_list in detect_gen:
        score = 0.0
        yield package_to_numpy(base_key_list, result_list, score)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.core_images
        python -m ibeis.core_images --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
