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
import cv2
from ibeis.control.controller_inject import register_preprocs
(print, rrr, profile) = ut.inject2(__name__, '[core_images]')


register_preproc = register_preprocs['image']


class ThumbnailConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('draw_annots', True, hideif=True),
        ut.ParamInfo('thumbsize', None, type_=None, hideif=None),
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


class ClassifierConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('classifier_sensitivity', None),
    ]
    _sub_config_list = [
        ThumbnailConfig
    ]


@register_preproc(
    tablename='classifier', parents=['images'],
    colnames=['score', 'class'],
    coltypes=[float, str],
    configclass=ClassifierConfig,
    fname='detectcache',
    chunksize=1024,
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
    from ibeis.algo.detect.classifier.classifier import classify_thumbnail_list
    print('[ibs] Process Image Classifications')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_image
    config = {
        'draw_annots' : False,
        'thumbsize'   : (192, 192),
    }
    thumbnail_list = depc.get_property('thumbnails', gid_list, 'img', config=config)
    result_list = classify_thumbnail_list(thumbnail_list)
    # yield detections
    for result in result_list:
        yield result


class LocalizerConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('algo', 'yolo', valid_values=['yolo', 'rf']),
        ut.ParamInfo('sensitivity', 0.0),
        ut.ParamInfo('species', 'zebra_plains', hideif='zebra_plains'),
        ut.ParamInfo('config_filepath', None),
        ut.ParamInfo('weight_filepath', None),
        ut.ParamInfo('class_filepath', None, hideif=lambda cfg: cfg['algo'] != 'yolo' or cfg['class_filepath']),
        ut.ParamInfo('grid', False),
    ]
    _sub_config_list = [
        ThumbnailConfig
    ]


@register_preproc(
    tablename='localizations', parents=['images'],
    colnames=['score', 'bboxes', 'thetas', 'confs', 'classes'],
    coltypes=[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=LocalizerConfig,
    fname='detectcache',
    chunksize=256,
)
def compute_localizations(depc, gid_list, config=None):
    r"""
    Extracts the localizations for a given input image

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray): tup

    CommandLine:
        ibeis compute_localizations

    CommandLine:
        python -m ibeis.core_images compute_localizations --show

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.core_images import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> print(depc.get_tablenames())
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> config = {'algo': 'yolo'}
        >>> depc.delete_property('localizations', gid_list, config=config)
        >>> detects = depc.get_property('localizations', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'pyrf'}
        >>> depc.delete_property('localizations', gid_list, config=config)
        >>> detects = depc.get_property('localizations', gid_list, 'bboxes', config=config)
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

    print('[ibs] Preprocess Localizations')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    ibs.assert_valid_gids(gid_list)
    base_key_list = ['xtl', 'ytl', 'width', 'height', 'theta', 'confidence', 'class']
    # Temporary for all detectors
    base_key_list[4] = (0.0, )  # Theta

    if config['algo'] in ['yolo', 'cnn']:
        from ibeis.algo.detect import yolo
        print('[ibs] detecting using CNN YOLO')
        detect_gen = yolo.detect_gid_list(ibs, gid_list, **config)
    elif config['algo'] in ['pyrf']:
        from ibeis.algo.detect import randomforest
        print('[ibs] detecting using Random Forests')
        base_key_list[6] = (config['species'], )  # class == species
        detect_gen = randomforest.detect_gid_list_with_species(ibs, gid_list, **config)
    else:
        raise ValueError('specified detection algo is not supported in config = %r' % (config, ))

    # yield detections
    for gid, gpath, result_list in detect_gen:
        score = 0.0
        yield package_to_numpy(base_key_list, result_list, score)


class LabelerConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('labeler_sensitivity', None),
    ]


@register_preproc(
    tablename='labeler', parents=['localizations'],
    colnames=['score', 'species', 'viewpoint', 'quality', 'orientation', 'probs'],
    coltypes=[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list],
    configclass=LabelerConfig,
    fname='detectcache',
    chunksize=1024,
)
def compute_labels_localizations(depc, loc_id_list, config=None):
    r"""
    Extracts the detections for a given input image

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        loc_id_list (list):  list of localization rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        ibeis compute_labels_localizations

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.core_images import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:100]
        >>> depc.delete_property('labeler', gid_list)
        >>> results = depc.get_property('labeler', gid_list, None)
        >>> results = depc.get_property('labeler', gid_list, 'species')
        >>> print(results)
    """
    from ibeis.algo.detect.labeler.labeler import label_chip_list
    print('[ibs] Process Localization Labels')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_image

    gid_list_ = depc.get_ancestor_rowids('localizations', loc_id_list, 'images')
    assert len(gid_list_) == len(loc_id_list)

    # Grab the localizations
    bboxes_list = depc.get_native('localizations', loc_id_list, 'bboxes')
    thetas_list = depc.get_native('localizations', loc_id_list, 'thetas')
    gids_list   = [
        np.array([gid] * len(bbox_list))
        for gid, bbox_list in zip(gid_list_, bboxes_list)
    ]

    # Flatten all of these lists for efficiency
    bbox_list      = ut.flatten(bboxes_list)
    theta_list     = ut.flatten(thetas_list)
    gid_list       = ut.flatten(gids_list)
    bbox_size_list = ut.take_column(bbox_list, [2, 3])
    newsize_list   = [(128, 128)] * len(bbox_list)
    # Checks
    invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
    invalid_bboxes = ut.compress(bbox_list, invalid_flags)
    assert len(invalid_bboxes) == 0, 'invalid bboxes=%r' % (invalid_bboxes,)

    # Build transformation from image to chip
    M_list = [
        vt.get_image_to_chip_transform(bbox, new_size, theta)
        for bbox, theta, new_size in zip(bbox_list, theta_list, newsize_list)
    ]

    # Extract "chips"
    flags = cv2.INTER_LANCZOS4
    borderMode = cv2.BORDER_CONSTANT
    warpkw = dict(flags=flags, borderMode=borderMode)

    last_gid = None
    chip_list = []
    for gid, new_size, M in zip(gid_list, newsize_list, M_list):
        if gid != last_gid:
            img = ibs.get_image_imgdata(gid)
            last_gid = gid
        chip = cv2.warpAffine(img, M[0:2], tuple(new_size), **warpkw)
        # cv2.imshow('', chip)
        # cv2.waitKey()
        assert chip.shape[0] == 128 and chip.shape[1] == 128
        chip_list.append(chip)

    # Get the results from the algorithm
    result_list = label_chip_list(chip_list)
    assert len(gid_list) == len(result_list)

    # Group the results
    group_dict = {}
    for gid, result in zip(gid_list, result_list):
        if gid not in group_dict:
            group_dict[gid] = []
        group_dict[gid].append(result)
    assert len(gid_list_) == len(group_dict.keys())

    # Return the results
    for gid in gid_list_:
        result_list = group_dict[gid]
        zipped_list = zip(*result_list)
        ret_tuple = (
            np.array(zipped_list[0]),
            np.array(zipped_list[1]),
            np.array(zipped_list[2]),
            np.array(zipped_list[3]),
            np.array(zipped_list[4]),
            list(zipped_list[5]),
        )
        # print(ret_tuple[:-1])
        # print('-------')
        yield ret_tuple


class DetectorConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('classifier_sensitivity',    0.82),
        ut.ParamInfo('localizer_config_filepath', None),
        ut.ParamInfo('localizer_weight_filepath', None),
        ut.ParamInfo('localizer_grid',            False),
        ut.ParamInfo('localizer_sensitivity',     0.16),
        ut.ParamInfo('labeler_sensitivity',       0.42),
    ]
    _sub_config_list = [
        ThumbnailConfig,
        LocalizerConfig,
    ]


@register_preproc(
    tablename='detections', parents=['images'],
    colnames=['score', 'bboxes', 'thetas', 'species', 'viewpoints', 'confs'],
    coltypes=[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=DetectorConfig,
    fname='detectcache',
    chunksize=1024,
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
        >>> # SLOW_DOCTEST
        >>> from ibeis.core_images import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> # dbdir = '/Users/bluemellophone/Desktop/GGR-IBEIS-TEST/'
        >>> # dbdir = '/media/danger/GGR/GGR-IBEIS-TEST/'
        >>> # ibs = ibeis.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> depc.delete_property('detections', gid_list)
        >>> detects = depc.get_property('detections', gid_list, None)
        >>> print(detects)
    """
    from ibeis.web.apis_detect import USE_LOCALIZATIONS
    print('[ibs] Preprocess Detections')
    print('config = %r' % (config,))
    # Get controller
    ibs = depc.controller
    ibs.assert_valid_gids(gid_list)

    if not USE_LOCALIZATIONS:
        # Filter the gids by annotations
        prediction_list = depc.get_property('classifier', gid_list, 'class')
        confidence_list = depc.get_property('classifier', gid_list, 'score')
        confidence_list = [
            confidence if prediction == 'positive' else 1.0 - confidence
            for prediction, confidence  in zip(prediction_list, confidence_list)
        ]
        gid_list_ = [
            gid
            for gid, confidence in zip(gid_list, confidence_list)
            if confidence >= config['classifier_sensitivity']
        ]
    else:
        gid_list_ = list(gid_list)

    gid_set_ = set(gid_list_)
    # Get the localizations for the good gids and add formal annotations
    localizer_config = {
        'config_filepath' : config['localizer_config_filepath'],
        'weight_filepath' : config['localizer_weight_filepath'],
        'grid'            : config['localizer_grid'],
    }
    bboxes_list  = depc.get_property('localizations', gid_list_, 'bboxes',    config=localizer_config)
    thetas_list  = depc.get_property('localizations', gid_list_, 'thetas',    config=localizer_config)
    confses_list = depc.get_property('localizations', gid_list_, 'confs',     config=localizer_config)

    if not USE_LOCALIZATIONS:
        # depc.delete_property('labeler', gid_list_, config=localizer_config)
        specieses_list     = depc.get_property('labeler', gid_list_, 'species',   config=localizer_config)
        viewpoints_list    = depc.get_property('labeler', gid_list_, 'viewpoint', config=localizer_config)
        scores_list        = depc.get_property('labeler', gid_list_, 'score',     config=localizer_config)
    else:
        specieses_list     = depc.get_property('localizations', gid_list_, 'classes',   config=localizer_config)
        viewpoints_list    = [
            [-1] * len(bbox_list)
            for bbox_list in bboxes_list
        ]
        scores_list        = depc.get_property('localizations', gid_list_, 'confs',     config=localizer_config)

    # Collect the detections, filtering by the localization confidence
    empty_list = [0.0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
    detect_dict = {}
    for index, gid in enumerate(gid_list_):
        bbox_list = bboxes_list[index]
        theta_list = thetas_list[index]
        species_list = specieses_list[index]
        # species_dict = {}
        # for species in species_list:
        #     if species not in species_dict:
        #         species_dict[species] = 0
        #     species_dict[species] += 1
        # for tup in species_dict.iteritems():
        #     print('\t%r' % (tup, ))
        # print('----')
        viewpoint_list = viewpoints_list[index]
        conf_list = confses_list[index]
        score_list = scores_list[index]
        zipped = zip(bbox_list, theta_list, species_list, viewpoint_list, conf_list, score_list)
        zipped = [
            [bbox, theta, species, viewpoint, conf * score]
            for bbox, theta, species, viewpoint, conf, score in zipped
            if conf >= config['localizer_sensitivity'] and score >= config['labeler_sensitivity'] and max(bbox[2], bbox[3]) / min(bbox[2], bbox[3]) < 20.0
        ]
        if len(zipped) == 0:
            detect_list = list(empty_list)
        else:
            detect_list = [0.0] + [np.array(_) for _ in zip(*zipped)]
        detect_dict[gid] = detect_list

    # Filter the annotations by the localizer operating point
    for gid in gid_list:
        if gid not in gid_set_:
            assert gid not in detect_dict
            result = list(empty_list)
        else:
            assert gid in detect_dict
            result = detect_dict[gid]
        # print(result)
        # raw_input()
        # print('')
        # image = ibs.get_image_imgdata(gid)
        # image = vt.resize(image, (500, 500))
        # cv2.imshow('', image)
        # cv2.waitKey(0)
        yield tuple(result)


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
