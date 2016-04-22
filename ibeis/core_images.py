# -*- coding: utf-8 -*-
"""
IBEIS CORE
Defines the core dependency cache supported by the image analysis api

Extracts detection results from images and applies additional processing
automatically


Ex
    python -m ibeis.control.IBEISControl --test-show_depc_image_graph --show
    python -m ibeis.control.IBEISControl --test-show_depc_image_graph --show --reduced


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
import vtool as vt
from ibeis.control.controller_inject import register_preprocs
(print, rrr, profile) = ut.inject2(__name__, '[core_images]')


register_preproc = register_preprocs['image']


class ThumbnailConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('draw_annots', True, hideif=True),
        ut.ParamInfo('thumbsize', None, hideif=None),
        ut.ParamInfo('ext', '.png', hideif='.png'),
        ut.ParamInfo('force_serial', False, hideif=False),
    ]


@register_preproc(
    tablename='thumbnails', parents=['images'],
    colnames=['img', 'width', 'height'],
    coltypes=[('extern', vt.imread, vt.imwrite), int, int],
    configclass=ThumbnailConfig,
    fname='thumbcache',
    rm_extern_on_delete=True,
    chunksize=256,
)
def compute_thumbnails(depc, gid_list, config=None):
    r"""
    Computers the thumbnail for a given input image

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        ibeis --tf compute_thumbnails --show --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_images import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'testdb1'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:10]
        >>> thumbs = depc.get_property('thumbnails', gid_list, 'img', config={'thumbsize': 221})
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(thumbs, nPerPage=4)
        >>> iteract_obj.start()
        >>> pt.show_if_requested()
    """

    ibs = depc.controller
    draw_annots = config['draw_annots']
    thumbsize = config['thumbsize']
    if thumbsize is None:
        cfg = ibs.cfg.other_cfg
        thumbsize = cfg.thumb_size if draw_annots else cfg.thumb_bare_size
    thumbsize_list = [thumbsize] * len(gid_list)
    gpath_list = ibs.get_image_paths(gid_list)
    orient_list = ibs.get_image_orientation(gid_list)
    aids_list = ibs.get_image_aids(gid_list)
    if draw_annots:
        bboxes_list = ibs.unflat_map(ibs.get_annot_bboxes, aids_list)
        thetas_list = ibs.unflat_map(ibs.get_annot_thetas, aids_list)
    else:
        bboxes_list = [ [] for aids in aids_list ]
        thetas_list = [ [] for aids in aids_list ]

    # Execute all tasks in parallel
    args_list = zip(thumbsize_list, gpath_list, orient_list, bboxes_list, thetas_list)
    genkw = {
        'ordered': False,
        'chunksize': 256,
        'freq': 50,
        #'adjust': True,
        'force_serial': ibs.force_serial or config['force_serial'],
    }
    gen = ut.generate(draw_thumb_helper, args_list, nTasks=len(args_list), **genkw)
    for val in gen:
        yield val


def draw_thumb_helper(tup):
    thumbsize, gpath, orient, bbox_list, theta_list = tup
    # time consuming
    # img = vt.imread(gpath, orient=orient)
    img = vt.imread(gpath)
    (gh, gw) = img.shape[0:2]
    img_size = (gw, gh)
    if isinstance(thumbsize, int):
        max_dsize = (thumbsize, thumbsize)
        dsize, sx, sy = vt.resized_clamped_thumb_dims(img_size, max_dsize)
    elif isinstance(thumbsize, tuple) and len(thumbsize) == 2:
        th, tw = thumbsize
        dsize, sx, sy = thumbsize, tw / gw, th / gh
    else:
        raise ValueError('Incompatible thumbsize')
    new_verts_list = list(vt.scaled_verts_from_bbox_gen(bbox_list, theta_list, sx, sy))
    # -----------------
    # Actual computation
    thumb = vt.resize(img, dsize)
    orange_bgr = (0, 128, 255)
    for new_verts in new_verts_list:
        thumb = vt.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
    width, height = dsize
    return thumb, width, height


class DetectionConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('algo', 'cnn'),
        ut.ParamInfo('sensitivity', 0.2),
        ut.ParamInfo('species', 'zebra_plains', hideif='zebra_plains'),
        ut.ParamInfo('config_filepath', None, hideif=None),
        ut.ParamInfo('weight_filepath', None, hideif=None),
        ut.ParamInfo('grid', False),
    ]
    _sub_config_list = [
        ThumbnailConfig
    ]


@register_preproc(
    tablename='detections', parents=['images'],
    colnames=['score', 'bboxes', 'thetas', 'confs', 'classes'],
    coltypes=[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=DetectionConfig,
    fname='detectcache',
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
        ibeis compute_detections

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.core_images import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> print(depc.get_tablenames())
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> config = {'algo': 'yolo'}
        >>> depc.delete_property('detections', gid_list, config=config)
        >>> detects = depc.get_property('detections', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'pyrf'}
        >>> depc.delete_property('detections', gid_list, config=config)
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


class ClassifierConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('classifier_sensitivity', 0.2),
    ]


@register_preproc(
    tablename='classifier', parents=['thumbnails'],
    colnames=['score', 'class'],
    coltypes=[float, str],
    configclass=ClassifierConfig,
    fname='detectcache',
    chunksize=32,
)
def compute_classifications(depc, gid_list, config=None):
    r"""
    Extracts the detections for a given input image

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        ibeis compute_classifications

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core_images import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> # depc.delete_property('classifier', gid_list)
        >>> results = depc.get_property('classifier', gid_list, None)
        >>> print(results)
    """
    from ibeis.algo.detect.classifier.classifier import classify_gid_list
    print('[ibs] Process Image Classifications')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    result_list = classify_gid_list(ibs, gid_list)
    # yield detections
    for result in result_list:
        yield result


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
