# -*- coding: utf-8 -*-
"""IBEIS CORE IMAGE.

Defines the core dependency cache supported by the image analysis api

Extracts detection results from images and applies additional processing
automatically


Ex
    python -m wbia.control.IBEISControl --test-show_depc_image_graph --show
    python -m wbia.control.IBEISControl --test-show_depc_image_graph --show --reduced


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
import logging
import sys
from os.path import exists, join, split

import cv2
import numpy as np
import tqdm
import utool as ut
import vtool as vt
from six.moves import zip

import wbia.constants as const
from wbia import dtool
from wbia.control.controller_inject import register_preprocs

(print, rrr, profile) = ut.inject2(__name__, '[core_images]')
logger = logging.getLogger('wbia')


register_preproc = register_preprocs['image']


class ThumbnailConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('draw_annots', True, hideif=True),
        ut.ParamInfo('thumbsize', None, type_=None, hideif=None),
        ut.ParamInfo('ext', '.png', hideif='.png'),
        ut.ParamInfo('force_serial', False, hideif=False),
    ]


@register_preproc(
    tablename='thumbnails',
    parents=['images'],
    colnames=['img', 'width', 'height'],
    coltypes=[('extern', vt.imread, vt.imwrite), int, int],
    configclass=ThumbnailConfig,
    fname='thumbcache',
    rm_extern_on_delete=True,
    chunksize=256,
)
def compute_thumbnails(depc, gid_list, config=None):
    r"""Compute the thumbnail for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        wbia --tf compute_thumbnails --show --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(--weird)
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'testdb1'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:10]
        >>> thumbs = depc.get_property('thumbnails', gid_list, 'img', config={'thumbsize': 221}, recompute=True)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import wbia.plottool as pt
        >>> pt.quit_if_noshow()
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
        interests_list = ibs.unflat_map(ibs.get_annot_interest, aids_list)
    else:
        bboxes_list = [[] for aids in aids_list]
        thetas_list = [[] for aids in aids_list]
        interests_list = [[] for aids in aids_list]

    # Execute all tasks in parallel
    args_list = list(
        zip(
            thumbsize_list,
            gpath_list,
            orient_list,
            bboxes_list,
            thetas_list,
            interests_list,
        )
    )

    genkw = {
        'ordered': True,
        'chunksize': 256,
        'progkw': {'freq': 50},
        # 'adjust': True,
        'futures_threaded': True,
        'force_serial': ibs.force_serial or config['force_serial'],
    }
    gen = ut.generate2(draw_thumb_helper, args_list, nTasks=len(args_list), **genkw)
    for val in gen:
        yield val


def draw_thumb_helper(thumbsize, gpath, orient, bbox_list, theta_list, interest_list):
    # time consuming
    img = vt.imread(gpath, orient=orient)
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
    blue_bgr = (255, 128, 0)
    color_bgr_list = [blue_bgr if interest else orange_bgr for interest in interest_list]
    for new_verts, color_bgr in zip(new_verts_list, color_bgr_list):
        thumb = vt.draw_verts(thumb, new_verts, color=color_bgr, thickness=2)
    width, height = dsize
    return thumb, width, height


def load_text(fpath):
    with open(fpath, 'r') as file:
        text = file.read()
    return text


def save_text(fpath, text):
    with open(fpath, 'w') as file:
        file.write(text)


class WebSrcConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('ext', '.txt', hideif='.txt'),
        ut.ParamInfo('force_serial', False, hideif=False),
    ]


@register_preproc(
    tablename='web_src',
    parents=['images'],
    colnames=['src'],
    coltypes=[('extern', load_text, save_text)],
    configclass=WebSrcConfig,
    fname='webcache',
    chunksize=1024,
)
def compute_web_src(depc, gid_list, config=None):
    r"""Compute the web src

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (str): tup

    CommandLine:
        wbia --tf compute_web_src --show --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'testdb1'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:10]
        >>> thumbs = depc.get_property('web_src', gid_list, 'src', recompute=True)
        >>> thumb = thumbs[0]
        >>> hash_str = ut.hash_data(thumb)
        >>> assert hash_str in ['yerctlgfqosrhmjpqvkbmnoocagfqsna', 'wcuppmpowkvhfmfcnrxdeedommihexfu', 'lerhyizhlignvvzmvqbbberyklzyfbzq'], 'Found %r' % (hash_str, )
    """
    ibs = depc.controller

    gpath_list = ibs.get_image_paths(gid_list)
    orient_list = ibs.get_image_orientation(gid_list)
    args_list = list(zip(gpath_list, orient_list))

    genkw = {
        'ordered': True,
        'futures_threaded': True,
        'force_serial': ibs.force_serial or config['force_serial'],
    }
    gen = ut.generate2(draw_web_src, args_list, nTasks=len(args_list), **genkw)
    for val in gen:
        yield (val,)


def draw_web_src(gpath, orient):
    from wbia.web.routes_ajax import image_src_path

    image_src = image_src_path(gpath, orient)
    return image_src


class ClassifierConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo(
            'classifier_algo',
            'cnn',
            valid_values=[
                'cnn',
                'svm',
                'densenet',
                'densenet+neighbors',
                'lightnet',
                'densenet+lightnet',
                'densenet+lightnet!',
                'tile_aggregation',
                'tile_aggregation_quick',
                'scout_detectnet',
                'scout_detectnet_csv',
                'scout_faster_rcnn_csv',
            ],
        ),
        ut.ParamInfo('classifier_weight_filepath', None),
    ]
    _sub_config_list = [ThumbnailConfig]


@register_preproc(
    tablename='classifier',
    parents=['images'],
    colnames=['score', 'class'],
    coltypes=[float, str],
    configclass=ClassifierConfig,
    fname='detectcache',
    chunksize=32 if const.CONTAINERIZED else 1024,
)
def compute_classifications(depc, gid_list, config=None):
    r"""Extract the detections for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        wbia compute_classifications

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> # depc.delete_property('classifier', gid_list)
        >>> results = depc.get_property('classifier', gid_list, None)
        >>> print(results)
        >>> depc = ibs.depc_image
        >>> config = {'classifier_algo': 'svm'}
        >>> depc.delete_property('classifier', gid_list, config=config)
        >>> results = depc.get_property('classifier', gid_list, None, config=config)
        >>> print(results)
        >>> config = {'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-10'}
        >>> depc.delete_property('classifier', gid_list, config=config)
        >>> results = depc.get_property('classifier', gid_list, None, config=config)
        >>> print(results)
    """
    logger.info('[ibs] Process Image Classifications')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_image
    if config['classifier_algo'] in ['cnn']:
        config_ = {
            'draw_annots': False,
            'thumbsize': (192, 192),
        }
        thumbnail_list = depc.get_property('thumbnails', gid_list, 'img', config=config_)
        result_list = ibs.generate_thumbnail_class_list(thumbnail_list, **config)
    elif config['classifier_algo'] in ['svm']:
        from wbia.algo.detect.svm import classify

        config_ = {'algo': 'resnet'}
        vector_list = depc.get_property('features', gid_list, 'vector', config=config_)
        classifier_weight_filepath = config['classifier_weight_filepath']
        result_list = classify(vector_list, weight_filepath=classifier_weight_filepath)
    elif config['classifier_algo'] in ['densenet']:
        from wbia.algo.detect import densenet

        config_ = {
            'draw_annots': False,
            'thumbsize': (densenet.INPUT_SIZE, densenet.INPUT_SIZE),
        }
        thumbpath_list = ibs.depc_image.get(
            'thumbnails', gid_list, 'img', config=config_, read_extern=False, ensure=True
        )
        result_list = densenet.test(thumbpath_list, ibs=ibs, gid_list=gid_list, **config)
    elif config['classifier_algo'] in ['tile_aggregation', 'tile_aggregation_quick']:
        classifier_weight_filepath = config['classifier_weight_filepath']
        classifier_weight_filepath = classifier_weight_filepath.strip().split(';')

        assert len(classifier_weight_filepath) == 2
        classifier_algo_, model_tag_ = classifier_weight_filepath

        include_grid2 = config['classifier_algo'] in ['tile_aggregation']
        tid_list = ibs.scout_get_valid_tile_rowids(
            gid_list=gid_list, include_grid2=include_grid2
        )
        ancestor_gid_list = ibs.get_tile_ancestor_gids(tid_list)
        confidence_list = ibs.scout_wic_test(
            tid_list, classifier_algo=classifier_algo_, model_tag=model_tag_
        )

        gid_dict = {}
        for ancestor_gid, tid, confidence in zip(
            ancestor_gid_list, tid_list, confidence_list
        ):
            if ancestor_gid not in gid_dict:
                gid_dict[ancestor_gid] = []
            gid_dict[ancestor_gid].append(confidence)

        result_list = []
        for gid in tqdm.tqdm(gid_list):
            gid_confidence_list = gid_dict.get(gid, None)
            assert gid_confidence_list is not None
            best_score = np.max(gid_confidence_list)

            if best_score >= 0.5:
                best_key = 'positive'
            else:
                best_key = 'negative'
                best_score = 1.0 - best_score

            result = (
                best_score,
                best_key,
            )
            result_list.append(result)
    elif config['classifier_algo'] in ['densenet+neighbors']:
        raise NotImplementedError
        # ut.embed()
        # classifier_weight_filepath = config['classifier_weight_filepath']

        # all_bbox_list = ibs.get_image_bboxes(gid_list)
        # wic_confidence_list = ibs.scout_wic_test(gid_list, classifier_algo='densenet',
        #                                           model_tag=classifier_weight_filepath)
        #
        # ancestor_gid_list = list(set(ibs.get_tile_ancestor_gids(gid_list)))
        # all_tile_list = list(set(ibs.scout_get_valid_tile_rowids(gid_list=ancestor_gid_list)))
        # all_bbox_list = ibs.get_image_bboxes(all_tile_list)
        # all_confidence_list = ibs.scout_wic_test(all_tile_list, classifier_algo='densenet',
        #                                           model_tag=classifier_weight_filepath)
        #
        # TODO: USE THRESHOLDED AVERAGE, NOT MAX
        # result_list = []
        # for gid, wic_confidence in zip(gid_list, wic_confidence_list):
        #     best_score = wic_confidence
        #     for aid in aid_list:
        #         wic_confidence_ = aid_conf_dict.get(aid, None)
        #         assert wic_confidence_ is not None
        #         best_score = max(best_score, wic_confidence_)
        #
        #     if wic_confidence < 0.5:
        #         best_key = 'negative'
        #         best_score = 1.0 - best_score
        #     else:
        #         best_key = 'positive'
        #     if best_score > wic_confidence:
        #         recovered += 1
        #     result = (best_score, best_key, )
        #     result_list.append(result)
    elif config['classifier_algo'] in ['scout_detectnet']:
        import json

        json_filepath = join(ibs.dbdir, config['classifier_weight_filepath'])
        assert exists(json_filepath)
        with open(json_filepath, 'r') as json_file:
            values = json.load(json_file)
        annotations = values.get('annotations', {})

        gpath_list = ibs.get_image_paths(gid_list)
        gname_list = [split(gpath)[1] for gpath in gpath_list]

        result_list = []
        for gname in gname_list:
            annotation = annotations.get(gname, None)
            assert annotation is not None

            best_score = 1.0
            if len(annotation) == 0:
                best_key = 'negative'
            else:
                best_key = 'positive'
            result = (
                best_score,
                best_key,
            )
            result_list.append(result)
    elif config['classifier_algo'] in ['scout_detectnet_csv', 'scout_faster_rcnn_csv']:
        uuid_str_list = list(map(str, ibs.get_image_uuids(gid_list)))

        manifest_filepath = join(ibs.dbdir, 'WIC_manifest_output.csv')
        csv_filepath = join(ibs.dbdir, config['classifier_weight_filepath'])

        assert exists(manifest_filepath)
        assert exists(csv_filepath)

        manifest_dict = {}
        with open(manifest_filepath, 'r') as manifest_file:
            manifest_file.readline()  # Discard column header row
            manifest_line_list = manifest_file.readlines()
            for manifest_line in manifest_line_list:
                manifest = manifest_line.strip().split(',')
                assert len(manifest) == 2
                manifest_filename, manifest_uuid = manifest
                manifest_dict[manifest_filename] = manifest_uuid

        csv_dict = {}
        with open(csv_filepath, 'r') as csv_file:
            csv_file.readline()  # Discard column header row
            csv_line_list = csv_file.readlines()
            for csv_line in csv_line_list:
                csv = csv_line.strip().split(',')
                assert len(csv) == 2
                csv_filename, csv_score = csv
                csv_uuid = manifest_dict.get(csv_filename, None)
                assert (
                    csv_uuid is not None
                ), 'Test image {!r} is not in the manifest'.format(
                    csv,
                )
                csv_dict[csv_uuid] = csv_score

        result_list = []
        for uuid_str in uuid_str_list:
            best_score = csv_dict.get(uuid_str, None)
            assert best_score is not None

            if config['classifier_algo'] in ['scout_detectnet_csv']:
                assert best_score in ['yes', 'no']
                best_key = 'positive' if best_score == 'yes' else 'negative'
                best_score = 1.0
            elif config['classifier_algo'] in ['scout_faster_rcnn_csv']:
                best_score = float(best_score)
                if best_score >= 0.5:
                    best_key = 'positive'
                else:
                    best_key = 'negative'
                    best_score = 1.0 - best_score
            else:
                raise ValueError

            result = (
                best_score,
                best_key,
            )
            result_list.append(result)
    elif config['classifier_algo'] in [
        'lightnet',
        'densenet+lightnet',
        'densenet+lightnet!',
    ]:
        min_area = 10

        classifier_weight_filepath = config['classifier_weight_filepath']
        classifier_weight_filepath = classifier_weight_filepath.strip().split(',')

        if config['classifier_algo'] in ['lightnet']:
            assert len(classifier_weight_filepath) == 2
            weight_filepath, nms_thresh = classifier_weight_filepath
            wic_thresh = 0.0
            nms_thresh = float(nms_thresh)
            wic_confidence_list = [np.inf] * len(gid_list)
            wic_filter = False
        elif config['classifier_algo'] in ['densenet+lightnet', 'densenet+lightnet!']:
            assert len(classifier_weight_filepath) == 4
            (
                wic_model_tag,
                wic_thresh,
                weight_filepath,
                nms_thresh,
            ) = classifier_weight_filepath
            wic_thresh = float(wic_thresh)
            nms_thresh = float(nms_thresh)
            wic_confidence_list = ibs.scout_wic_test(
                gid_list, classifier_algo='densenet', model_tag=wic_model_tag
            )
            wic_filter = config['classifier_algo'] in ['densenet+lightnet']
        else:
            raise ValueError

        flag_list = [
            wic_confidence >= wic_thresh for wic_confidence in wic_confidence_list
        ]
        if wic_filter:
            gid_list_ = ut.compress(gid_list, flag_list)
        else:
            gid_list_ = gid_list[:]
        config = {
            'grid': False,
            'algo': 'lightnet',
            'config_filepath': weight_filepath,
            'weight_filepath': weight_filepath,
            'nms': True,
            'nms_thresh': nms_thresh,
            'sensitivity': 0.0,
        }
        prediction_list = depc.get_property(
            'localizations', gid_list_, None, config=config
        )
        prediction_dict = dict(zip(gid_list_, prediction_list))

        result_list = []
        for gid, wic_confidence, flag in zip(gid_list, wic_confidence_list, flag_list):
            if not flag:
                best_key = 'negative'
                best_score = 1.0 - wic_confidence
            else:
                prediction = prediction_dict.get(gid, None)
                assert prediction is not None

                best_score = 0.0
                if prediction is not None:
                    score, bboxes, thetas, confs, classes = prediction
                    for bbox, conf in zip(bboxes, confs):
                        xtl, ytl, w, h = bbox
                        area = w * h
                        if area >= min_area:
                            best_score = max(best_score, conf)

                if best_score >= 0.5:
                    best_key = 'positive'
                else:
                    best_key = 'negative'
                    best_score = 1.0 - best_score
            result = (
                best_score,
                best_key,
            )
            result_list.append(result)
    else:
        raise ValueError(
            'specified classifier algo is not supported in config = {!r}'.format(config)
        )

    # yield detections
    for result in result_list:
        yield result


class Classifier2Config(dtool.Config):
    _param_info_list = [
        ut.ParamInfo(
            'classifier_two_algo', 'cnn', valid_values=['cnn', 'rf', 'densenet']
        ),
        ut.ParamInfo('classifier_two_weight_filepath', None),
    ]
    _sub_config_list = [ThumbnailConfig]


@register_preproc(
    tablename='classifier_two',
    parents=['images'],
    colnames=['scores', 'classes'],
    coltypes=[dict, list],
    configclass=Classifier2Config,
    fname='detectcache',
    chunksize=32 if const.CONTAINERIZED else 128,
)
def compute_classifications2(depc, gid_list, config=None):
    r"""Extract the multi-class classifications for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (np.ndarray, np.ndarray): tup

    CommandLine:
        wbia compute_classifications2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> # depc.delete_property('classifier_two', gid_list)
        >>> results = depc.get_property('classifier_two', gid_list, None)
        >>> print(results)
    """
    logger.info('[ibs] Process Image Classifications2')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller
    depc = ibs.depc_image
    if config['classifier_two_algo'] in ['cnn']:
        config_ = {
            'draw_annots': False,
            'thumbsize': (192, 192),
        }
        # depc.delete_property('thumbnails', gid_list, config=config_)
        thumbnail_list = depc.get_property('thumbnails', gid_list, 'img', config=config_)
        result_list = ibs.generate_thumbnail_class2_list(thumbnail_list, **config)
    elif config['classifier_two_algo'] in ['rf']:
        from wbia.algo.detect.rf import classify

        config_ = {'algo': 'resnet'}
        vector_list = depc.get_property('features', gid_list, 'vector', config=config_)
        classifier_weight_filepath = config['classifier_weight_filepath']
        result_list = classify(vector_list, weight_filepath=classifier_weight_filepath)
    elif config['classifier_two_algo'] in ['densenet']:
        from wbia.algo.detect import densenet

        config_ = {
            'draw_annots': False,
            'thumbsize': (densenet.INPUT_SIZE, densenet.INPUT_SIZE),
        }
        thumbpath_list = ibs.depc_image.get(
            'thumbnails', gid_list, 'img', config=config_, read_extern=False, ensure=True
        )
        config_ = {
            'classifier_weight_filepath': config['classifier_two_weight_filepath'],
        }
        result_list = densenet.test(
            thumbpath_list,
            ibs=ibs,
            gid_list=gid_list,
            return_dict=True,
            multiclass=True,
            **config_,
        )
        result_list = list(result_list)
        for index in range(len(result_list)):
            best_score, best_key, scores = result_list[index]
            classes = [best_key]
            result_list[index] = (
                scores,
                classes,
            )
    else:
        raise ValueError(
            'specified classifier_two algo is not supported in config = {!r}'.format(
                config
            )
        )

    # yield detections
    for result in result_list:
        yield result


class FeatureConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('framework', 'torch', valid_values=['keras', 'torch']),
        ut.ParamInfo(
            'model',
            'vgg16',
            valid_values=['vgg', 'vgg16', 'vgg19', 'resnet', 'inception', 'densenet'],
        ),
        ut.ParamInfo('flatten', True),
    ]
    _sub_config_list = [ThumbnailConfig]


@register_preproc(
    tablename='features',
    parents=['images'],
    colnames=['vector'],
    coltypes=[np.ndarray],
    configclass=FeatureConfig,
    fname='featcache',
    chunksize=256,
)
def compute_features(depc, gid_list, config=None):
    r"""Compute features on images using pre-trained state-of-the-art models in Keras.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (np.ndarray, ): tup

    CommandLine:
        wbia compute_features

    CommandLine:
        python -m wbia.core_images compute_features --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> print(depc.get_tablenames())
        >>> gid_list = ibs.get_valid_gids()[:16]
        >>> config = {'model': 'vgg16'}
        >>> depc.delete_property('features', gid_list, config=config)
        >>> features = depc.get_property('features', gid_list, 'vector', config=config)
        >>> print(features)
        >>> config = {'model': 'vgg19'}
        >>> depc.delete_property('features', gid_list, config=config)
        >>> features = depc.get_property('features', gid_list, 'vector', config=config)
        >>> print(features)
        >>> config = {'model': 'resnet'}
        >>> depc.delete_property('features', gid_list, config=config)
        >>> features = depc.get_property('features', gid_list, 'vector', config=config)
        >>> print(features)
        >>> config = {'model': 'inception'}
        >>> depc.delete_property('features', gid_list, config=config)
        >>> features = depc.get_property('features', gid_list, 'vector', config=config)
        >>> print(features)
    """
    logger.info('[ibs] Preprocess Features')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller
    ibs.assert_valid_gids(gid_list)
    ######################################################################################

    if config['framework'] in ['keras']:
        from keras.preprocessing import image as preprocess_image

        thumbnail_config = {
            'draw_annots': False,
            'thumbsize': (500, 500),
        }
        thumbpath_list = depc.get(
            'thumbnails',
            gid_list,
            'img',
            config=thumbnail_config,
            read_extern=False,
            ensure=True,
        )

        target_size = (224, 224)
        if config['model'] in ['vgg', 'vgg16']:
            from keras.applications.vgg16 import VGG16 as MODEL_CLASS
            from keras.applications.vgg16 import preprocess_input
        ######################################################################################
        elif config['model'] in ['vgg19']:
            from keras.applications.vgg19 import VGG19 as MODEL_CLASS
            from keras.applications.vgg19 import preprocess_input
        ######################################################################################
        elif config['model'] in ['resnet']:
            from keras.applications.resnet50 import ResNet50 as MODEL_CLASS  # NOQA
            from keras.applications.resnet50 import preprocess_input
        ######################################################################################
        elif config['model'] in ['inception']:
            from keras.applications.inception_v3 import InceptionV3 as MODEL_CLASS  # NOQA
            from keras.applications.inception_v3 import preprocess_input

            target_size = (299, 299)
        ######################################################################################
        else:
            raise ValueError(
                'specified feature model is not supported in config = {!r}'.format(config)
            )

        # Build model
        model = MODEL_CLASS(include_top=False)

        thumbpath_iter = ut.ProgIter(thumbpath_list, lbl='forward inference', bs=True)
        for thumbpath in thumbpath_iter:
            image = preprocess_image.load_img(thumbpath, target_size=target_size)
            image_array = preprocess_image.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)
            features = model.predict(image_array)
            if config['flatten']:
                features = features.flatten()
            yield (features,)
    elif config['framework'] in ['torch']:
        from wbia.algo.detect import densenet

        if config['model'] in ['densenet']:
            config_ = {
                'draw_annots': False,
                'thumbsize': (densenet.INPUT_SIZE, densenet.INPUT_SIZE),
            }
            thumbpath_list = ibs.depc_image.get(
                'thumbnails',
                gid_list,
                'img',
                config=config_,
                read_extern=False,
                ensure=True,
            )
            feature_list = densenet.features(thumbpath_list)
        else:
            raise ValueError(
                'specified feature model is not supported in config = {!r}'.format(config)
            )

        for feature in feature_list:
            if config['flatten']:
                feature = feature.flatten()
            yield (feature,)
    else:
        raise ValueError(
            'specified feature framework is not supported in config = {!r}'.format(config)
        )


class LocalizerOriginalConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo(
            'algo',
            'yolo',
            valid_values=[
                'azure',
                'yolo',
                'lightnet',
                'ssd',
                'darknet',
                'rf',
                'fast-rcnn',
                'faster-rcnn',
                'selective-search',
                'selective-search-rcnn',
                '_COMBINED',
                'tile_aggregation',
                'tile_aggregation_quick',
                'scout_detectnet_json',
                'scout_faster_rcnn_json',
            ],
        ),
        ut.ParamInfo('species', None),
        ut.ParamInfo('config_filepath', None),
        ut.ParamInfo('weight_filepath', None),
        ut.ParamInfo('class_filepath', None),
        ut.ParamInfo(
            'flip', False, hideif=False
        ),  # True will horizontally flip all images before being sent to the localizer and will flip the results back
        ut.ParamInfo('grid', False),
    ]
    _sub_config_list = [ThumbnailConfig]


@register_preproc(
    tablename='localizations_original',
    parents=['images'],
    colnames=['score', 'bboxes', 'thetas', 'confs', 'classes'],
    coltypes=[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=LocalizerOriginalConfig,
    fname='localizationscache',
    chunksize=8 if const.CONTAINERIZED else 128,
)
def compute_localizations_original(depc, gid_list, config=None):
    r"""Extract the localizations for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray): tup

    CommandLine:
        wbia compute_localizations_original

    CommandLine:
        python -m wbia.core_images compute_localizations_original --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> print(depc.get_tablenames())
        >>> gid_list = ibs.get_valid_gids()[:16]
        >>> config = {'algo': 'azure', 'config_filepath': None}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'darknet', 'config_filepath': 'pretrained-v2-pascal'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'darknet', 'config_filepath': 'pretrained-v2-large-pascal'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'darknet', 'config_filepath': 'pretrained-tiny-pascal'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'darknet', 'config_filepath': 'pretrained-v2-large-coco'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'darknet', 'config_filepath': 'pretrained-tiny-coco'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'yolo'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'lightnet'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'rf'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'selective-search'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'selective-search-rcnn'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'faster-rcnn', 'config_filepath': 'pretrained-vgg-pascal'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'faster-rcnn', 'config_filepath': 'pretrained-zf-pascal'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'faster-rcnn', 'config_filepath': 'pretrained-vgg-ilsvrc'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'faster-rcnn', 'config_filepath': 'pretrained-zf-ilsvrc'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal-plus'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal-plus'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-300-coco'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-512-coco'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-300-ilsvrc'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': 'ssd', 'config_filepath': 'pretrained-500-ilsvrc'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'algo': '_COMBINED'}
        >>> depc.delete_property('localizations_original', gid_list, config=config)
        >>> detects = depc.get_property('localizations_original', gid_list, 'bboxes', config=config)
        >>> print(detects)
    """

    def _package_to_numpy(key_list, result_list, score):
        temp = [
            [key[0] if isinstance(key, tuple) else result[key] for key in key_list]
            for result in result_list
        ]
        return (
            score,
            np.array([_[0:4] for _ in temp]),
            np.array([_[4] for _ in temp]),
            np.array([_[5] for _ in temp]),
            np.array([_[6] for _ in temp]),
        )

    def _combined(gid_list, config_dict_list):
        # Combined list of algorithm configs
        colname_list = ['score', 'bboxes', 'thetas', 'confs', 'classes']
        for gid in gid_list:
            accum_list = []
            for colname in colname_list:
                if colname == 'score':
                    accum_value = 0.0
                else:
                    len_list = []
                    temp_list = []
                    for config_dict_ in config_dict_list:
                        temp = depc.get_property(
                            'localizations_original', gid, colname, config=config_dict_
                        )
                        len_list.append(len(temp))
                        temp_list.append(temp)
                    if colname == 'bboxes':
                        accum_value = np.vstack(temp_list)
                    else:
                        accum_value = np.hstack(temp_list)
                    assert len(accum_value) == sum(len_list)
                accum_list.append(accum_value)
            yield tuple(accum_list)

    logger.info('[ibs] Preprocess Localizations')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller
    ibs.assert_valid_gids(gid_list)

    config = dict(config)
    config['sensitivity'] = 0.0

    flip = config.get('flip', False)
    if flip:
        assert (
            config['algo'] == 'lightnet'
        ), 'config "flip" is only supported by the "lightnet" algo'

    # Normal computations
    base_key_list = ['xtl', 'ytl', 'width', 'height', 'theta', 'confidence', 'class']
    # Temporary for all detectors
    base_key_list[4] = (0.0,)  # Theta

    ######################################################################################
    if config['algo'] in ['pydarknet', 'yolo', 'cnn']:
        from wbia.algo.detect import yolo

        logger.info('[ibs] detecting using PyDarknet CNN YOLO v1')
        detect_gen = yolo.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['lightnet']:
        from wbia.algo.detect import lightnet

        logger.info('[ibs] detecting using Lightnet CNN YOLO v2')
        detect_gen = lightnet.detect_gid_list(ibs, gid_list, **config)
    elif config['algo'] in ['azure']:
        from wbia.algo.detect import azure

        logger.info('[ibs] detecting using Azure CustomVision')
        detect_gen = azure.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['rf']:
        from wbia.algo.detect import randomforest

        logger.info('[ibs] detecting using Random Forests')
        assert config['species'] is not None
        base_key_list[6] = (config['species'],)  # class == species
        detect_gen = randomforest.detect_gid_list_with_species(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['selective-search']:
        from wbia.algo.detect import selectivesearch

        logger.info('[ibs] detecting using Selective Search')
        matlab_command = 'selective_search'
        detect_gen = selectivesearch.detect_gid_list(
            ibs, gid_list, matlab_command=matlab_command, **config
        )
    ######################################################################################
    elif config['algo'] in ['selective-search-rcnn']:
        from wbia.algo.detect import selectivesearch

        logger.info('[ibs] detecting using Selective Search (R-CNN)')
        matlab_command = 'selective_search_rcnn'
        detect_gen = selectivesearch.detect_gid_list(
            ibs, gid_list, matlab_command=matlab_command, **config
        )
    ######################################################################################
    # elif config['algo'] in ['fast-rcnn']:
    #     from wbia.algo.detect import fasterrcnn
    #     logger.info('[ibs] detecting using CNN Fast R-CNN')
    #     detect_gen = fasterrcnn.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['faster-rcnn']:
        from wbia.algo.detect import fasterrcnn

        logger.info('[ibs] detecting using CNN Faster R-CNN')
        detect_gen = fasterrcnn.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['darknet']:
        from wbia.algo.detect import darknet

        logger.info('[ibs] detecting using Darknet CNN YOLO')
        detect_gen = darknet.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['ssd']:
        from wbia.algo.detect import ssd

        logger.info('[ibs] detecting using CNN SSD')
        detect_gen = ssd.detect_gid_list(ibs, gid_list, **config)
    # ######################################################################################
    elif config['algo'] in ['_COMBINED']:
        # Combined computations
        config_dict_list = [
            # {'algo': 'selective-search', 'config_filepath': None},                          # SS1
            {'algo': 'darknet', 'config_filepath': 'pretrained-tiny-pascal'},  # YOLO1
            {'algo': 'darknet', 'config_filepath': 'pretrained-v2-pascal'},  # YOLO2
            {'algo': 'faster-rcnn', 'config_filepath': 'pretrained-zf-pascal'},  # FRCNN1
            {'algo': 'faster-rcnn', 'config_filepath': 'pretrained-vgg-pascal'},  # FRCNN2
            {'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal'},  # SSD1
            {'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal'},  # SSD1
            {'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal-plus'},  # SSD
            {'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal-plus'},  # SSD4
        ]
        detect_gen = _combined(gid_list, config_dict_list)
    elif config['algo'] in ['tile_aggregation', 'tile_aggregation_quick']:
        from wbia.other.detectfuncs import general_intersection_over_union

        include_grid2 = config['algo'] in ['tile_aggregation']
        tid_list = ibs.scout_get_valid_tile_rowids(
            gid_list=gid_list, include_grid2=include_grid2
        )

        ancestor_gid_list = ibs.get_tile_ancestor_gids(tid_list)
        bbox_list = ibs.get_tile_bboxes(tid_list)
        size_list = ibs.get_image_sizes(tid_list)

        config_filepath = config['config_filepath']
        assert config_filepath in [
            'variant1',
            'variant2',
            'variant2-32',
            'variant2-64',
            'variant3-32',
            'variant3-64',
            'variant4-32',
            'variant4-64',
        ]

        weight_filepath = config['weight_filepath']
        weight_filepath = weight_filepath.strip().split(';')

        assert len(weight_filepath) == 2
        algo_, model_tag_ = weight_filepath

        if algo_ in ['densenet+lightnet']:
            model_tag_ = model_tag_.strip().split(',')

            assert len(model_tag_) == 4
            wic_model_tag, wic_thresh, weight_filepath, nms_thresh = model_tag_
            wic_thresh = float(wic_thresh)
            nms_thresh = float(nms_thresh)
            wic_confidence_list = ibs.scout_wic_test(
                tid_list, classifier_algo='densenet', model_tag=wic_model_tag
            )

            flag_list = [
                wic_confidence >= wic_thresh for wic_confidence in wic_confidence_list
            ]
            tid_list_ = ut.compress(tid_list, flag_list)
            logger.info(
                '%d tiles passed WIC filter out of %d'
                % (
                    len(tid_list_),
                    len(tid_list),
                )
            )

            loc_config = {
                'grid': False,
                'algo': 'lightnet',
                'config_filepath': weight_filepath,
                'weight_filepath': weight_filepath,
                'nms': True,
                'nms_thresh': nms_thresh,
                'sensitivity': 0.0,
            }
            prediction_list = depc.get_property(
                'localizations', tid_list_, None, config=loc_config
            )
            prediction_dict = dict(zip(tid_list_, prediction_list))
        else:
            raise ValueError('Only "densenet+lightnet" is implemented')

        gid_dict = {}
        zipped = list(zip(ancestor_gid_list, tid_list, size_list, bbox_list))
        for ancestor_gid, tid, tile_size, tile_bbox in tqdm.tqdm(zipped):
            prediction = prediction_dict.get(tid, None)

            if prediction is None:
                continue

            score, bboxes, thetas, confs, classes = prediction
            tile_xtl, tile_ytl, tile_w, tile_h = tile_bbox

            if ancestor_gid not in gid_dict:
                gid_dict[ancestor_gid] = {
                    'score': [],
                    'bboxes': [],
                    'thetas': [],
                    'confs': [],
                    'classes': [],
                }

            bboxes_ = []
            confs_ = []
            for detect_bbox, detect_conf in zip(bboxes, confs):
                detect_xtl, detect_ytl, detect_w, detect_h = detect_bbox
                detect_xbr = detect_xtl + detect_w
                detect_ybr = detect_ytl + detect_h
                detect_annot = {
                    'xtl': detect_xtl / tile_w,
                    'ytl': detect_ytl / tile_h,
                    'xbr': detect_xbr / tile_w,
                    'ybr': detect_ybr / tile_h,
                    'width': detect_w / tile_w,
                    'height': detect_h / tile_h,
                }

                multiplier = None
                if config_filepath in ['variant1']:
                    multiplier = 1.0
                elif config_filepath in [
                    'variant2',
                    'variant2-32',
                    'variant2-64',
                    'variant3-32',
                    'variant3-64',
                    'variant4-32',
                    'variant4-64',
                ]:
                    margin = (
                        32.0
                        if config_filepath
                        in ['variant-2', 'variant2-32', 'variant3-32', 'variant4-32']
                        else 64.0
                    )
                    tile_w, tile_h = tile_size
                    margin_percent_w = margin / tile_w
                    margin_percent_h = margin / tile_h
                    xtl = margin_percent_w
                    ytl = margin_percent_h
                    xbr = 1.0 - margin_percent_w
                    ybr = 1.0 - margin_percent_h
                    width = xbr - xtl
                    height = ybr - ytl
                    center = {
                        'xtl': xtl,
                        'ytl': ytl,
                        'xbr': xbr,
                        'ybr': ybr,
                        'width': width,
                        'height': height,
                    }
                    intersection, union = general_intersection_over_union(
                        detect_annot, center, return_components=True
                    )
                    area = detect_annot['width'] * detect_annot['height']
                    overlap = 0.0 if area <= 0 else intersection / area
                    assert 0.0 <= overlap and overlap <= 1.0

                    if config_filepath in ['variant2', 'variant2-32', 'variant2-64']:
                        multiplier = overlap
                    elif config_filepath in ['variant3-32', 'variant3-64']:
                        multiplier = np.sqrt(overlap)
                    elif config_filepath in ['variant4-32', 'variant4-64']:
                        multiplier = overlap ** overlap
                    else:
                        raise ValueError
                else:
                    raise ValueError
                assert multiplier is not None

                bbox_ = (
                    tile_xtl + detect_xtl,
                    tile_ytl + detect_ytl,
                    detect_w,
                    detect_h,
                )
                bboxes_.append(bbox_)
                confs_.append(detect_conf * multiplier)
            thetas_ = list(thetas)
            classes_ = list(classes)

            gid_dict[ancestor_gid]['score'] += [score]
            gid_dict[ancestor_gid]['bboxes'] += bboxes_
            gid_dict[ancestor_gid]['thetas'] += thetas_
            gid_dict[ancestor_gid]['confs'] += confs_
            gid_dict[ancestor_gid]['classes'] += classes_

        detect_gen = []
        for gid in tqdm.tqdm(gid_list):
            if gid in gid_dict:
                score = gid_dict[gid]['score']
                score = sum(score) / len(score) if len(score) > 0 else 0.0
                bboxes = np.array(gid_dict[gid]['bboxes'])
                thetas = np.array(gid_dict[gid]['thetas'])
                confs = np.array(gid_dict[gid]['confs'])
                classes = np.array(gid_dict[gid]['classes'])
            else:
                score = 0.0
                bboxes = np.array([])
                thetas = np.array([])
                confs = np.array([])
                classes = np.array([])

            detect = (score, bboxes, thetas, confs, classes)
            detect_gen.append(detect)
    elif config['algo'] in ['scout_detectnet_json', 'scout_faster_rcnn_json']:
        import json

        uuid_str_list = list(map(str, ibs.get_image_uuids(gid_list)))

        manifest_filepath = join(ibs.dbdir, 'WIC_manifest_output.csv')
        assert config['config_filepath'] in ['variant1']
        json_filepath = join(ibs.dbdir, config['weight_filepath'])

        assert exists(manifest_filepath)
        assert exists(json_filepath)

        manifest_dict = {}
        with open(manifest_filepath, 'r') as manifest_file:
            manifest_file.readline()  # Discard column header row
            manifest_line_list = manifest_file.readlines()
            for manifest_line in manifest_line_list:
                manifest = manifest_line.strip().split(',')
                assert len(manifest) == 2
                manifest_filename, manifest_uuid = manifest
                manifest_dict[manifest_filename] = manifest_uuid

        with open(json_filepath, 'r') as json_file:
            json_values = json.load(json_file)

        images = json_values.get('images', None)
        annotations = json_values.get('annotations', None)

        assert images is not None
        assert annotations is not None

        image_dict = {}
        annotation_dict = {}
        for image in images:
            assert 'file_name' in image and 'id' in image, 'Incorrect COCO format'
            image_filename = image['file_name']
            image_id = image['id']
            image_uuid = manifest_dict.get(image_filename, None)
            assert image_uuid is not None
            image_dict[image_id] = image_uuid
            annotation_dict[image_uuid] = []

        for annotation in annotations:
            assert 'image_id' in annotation and 'bbox' in annotation
            image_id = annotation['image_id']
            image_uuid = image_dict[image_id]
            assert image_uuid is not None
            annotation_bbox = annotation['bbox']
            assert image_uuid in annotation_dict
            annotation_dict[image_uuid].append(annotation_bbox)

        detect_gen = []
        for uuid_str in tqdm.tqdm(uuid_str_list):
            assert uuid_str in annotation_dict
            bbox_list = annotation_dict[uuid_str]

            score = 0.0
            bboxes = np.array(bbox_list)
            thetas = np.array([0.0] * len(bbox_list))
            confs = np.array([1.0] * len(bbox_list))
            classes = np.array(['elephant_savanna'] * len(bbox_list))

            detect = (score, bboxes, thetas, confs, classes)
            detect_gen.append(detect)
    else:
        raise ValueError(
            'specified detection algo is not supported in config = {!r}'.format(config)
        )

    # yield detections
    for detect in detect_gen:
        if len(detect) == 3:
            gid, gpath, result_list = detect
            score = 0.0
            result = _package_to_numpy(base_key_list, result_list, score)
        else:
            result = detect
        yield result


class LocalizerConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('sensitivity', 0.0),
        ut.ParamInfo('nms', True),
        ut.ParamInfo('nms_thresh', 0.2),
        ut.ParamInfo('nms_aware', None, hideif=None),
        ut.ParamInfo('invalid', True),
        ut.ParamInfo('invalid_margin', 0.25),
        ut.ParamInfo('boundary', True),
        ut.ParamInfo(
            'squared', False, hideif=False
        ),  # True will convert the bounding box into a square using the existing center as a reference point and the radius as half the length of the longest side (width or height).
    ]


@register_preproc(
    tablename='localizations',
    parents=['localizations_original'],
    colnames=['score', 'bboxes', 'thetas', 'confs', 'classes'],
    coltypes=[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=LocalizerConfig,
    fname='detectcache',
    chunksize=64 if const.CONTAINERIZED else 256,
)
def compute_localizations(depc, loc_orig_id_list, config=None):
    r"""Extract the localizations for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray): tup

    CommandLine:
        wbia compute_localizations

    CommandLine:
        python -m wbia.core_images compute_localizations --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> print(depc.get_tablenames())
        >>> gid_list = ibs.get_valid_gids()[:16]
        >>> config = {'algo': 'lightnet', 'nms': True}
        >>> # depc.delete_property('localizations', gid_list, config=config)
        >>> detects = depc.get_property('localizations', gid_list, 'bboxes', config=config)
        >>> print(detects)
        >>> config = {'combined': True}
        >>> # depc.delete_property('localizations', gid_list, config=config)
        >>> detects = depc.get_property('localizations', gid_list, 'bboxes', config=config)
        >>> print(detects)
    """
    logger.info('[ibs] Preprocess Localizations')
    logger.info('config = {!r}'.format(config))

    VERBOSE = False

    ibs = depc.controller

    zipped = list(zip(depc.get_native('localizations_original', loc_orig_id_list, None)))
    zipped2 = list(zip(loc_orig_id_list, zipped))
    for loc_orig_id, detect in tqdm.tqdm(zipped2):
        score, bboxes, thetas, confs, classes = detect[0]

        # Apply Threshold
        if config['sensitivity'] > 0.0:
            count_old = len(bboxes)
            if count_old > 0:
                keep_list = np.array(
                    [
                        index
                        for index, conf in enumerate(confs)
                        if conf >= config['sensitivity']
                    ]
                )

                if len(keep_list) == 0:
                    bboxes = np.array([])
                    thetas = np.array([])
                    confs = np.array([])
                    classes = np.array([])
                else:
                    bboxes = bboxes[keep_list]
                    thetas = thetas[keep_list]
                    confs = confs[keep_list]
                    classes = classes[keep_list]

                count_new = len(bboxes)
                if VERBOSE:
                    logger.info(
                        'Filtered with sensitivity = %0.02f (%d -> %d)'
                        % (
                            config['sensitivity'],
                            count_old,
                            count_new,
                        )
                    )

        # Apply NMS
        if config['nms']:
            indices = np.array(list(range(len(bboxes))))
            indices, bboxes, thetas, confs, classes = ibs.nms_boxes(
                indices, bboxes, thetas, confs, classes, verbose=VERBOSE, **config
            )

        # Kill invalid images
        if config['invalid']:
            margin = config['invalid_margin']

            count_old = len(bboxes)
            if count_old > 0:
                gid = depc.get_ancestor_rowids(
                    'localizations_original', [loc_orig_id], 'images'
                )[0]
                w, h = ibs.get_image_sizes(gid)

                keep_list = []
                for (xtl, ytl, width, height) in bboxes:
                    xbr = xtl + width
                    ybr = ytl + height

                    x_margin = w * margin
                    y_margin = h * margin
                    x_min = 0 - x_margin
                    x_max = w + x_margin
                    y_min = 0 - y_margin
                    y_max = h + y_margin

                    keep = True
                    if xtl < x_min or x_max < xtl or xbr < x_min or x_max < xbr:
                        keep = False
                    if ytl < y_min or y_max < ytl or ybr < y_min or y_max < ybr:
                        keep = False
                    keep_list.append(keep)

                if len(keep_list) == 0:
                    bboxes = np.array([])
                    thetas = np.array([])
                    confs = np.array([])
                    classes = np.array([])
                else:
                    bboxes = bboxes[keep_list]
                    thetas = thetas[keep_list]
                    confs = confs[keep_list]
                    classes = classes[keep_list]

                count_new = len(bboxes)
                if VERBOSE:
                    logger.info(
                        'Filtered invalid images (%d -> %d)'
                        % (
                            count_old,
                            count_new,
                        )
                    )

        if config['boundary']:
            gid = depc.get_ancestor_rowids(
                'localizations_original', [loc_orig_id], 'images'
            )[0]
            w, h = ibs.get_image_sizes(gid)

            for index, (xtl, ytl, width, height) in enumerate(bboxes):
                xbr = xtl + width
                ybr = ytl + height

                xtl = min(max(0, xtl), w)
                xbr = min(max(0, xbr), w)
                ytl = min(max(0, ytl), h)
                ybr = min(max(0, ybr), h)

                width = xbr - xtl
                height = ybr - ytl

                bboxes[index][0] = xtl
                bboxes[index][1] = ytl
                bboxes[index][2] = width
                bboxes[index][3] = height

        if config['squared']:
            for index, (xtl, ytl, width, height) in enumerate(bboxes):
                cx = xtl + width // 2
                cy = ytl + height // 2
                radius = max(width, height) // 2

                xtl = cx - radius
                ytl = cy - radius
                width = 2 * radius
                height = 2 * radius

                bboxes[index][0] = xtl
                bboxes[index][1] = ytl
                bboxes[index][2] = width
                bboxes[index][3] = height

        yield (
            score,
            bboxes,
            thetas,
            confs,
            classes,
        )


def get_localization_chips_worker(
    gid, img, bbox_list, theta_list, target_size, axis_aligned=False
):
    target_size_list = [target_size] * len(bbox_list)

    if axis_aligned:
        # Over-write bbox and theta with a friendlier, axis-aligned version
        bbox_list_ = []
        theta_list_ = []
        for bbox, theta in zip(bbox_list, theta_list):
            # Transformation matrix
            R = vt.rotation_around_bbox_mat3x3(theta, bbox)
            # Get verticies of the annotation polygon
            verts = vt.verts_from_bbox(bbox, close=True)
            # Rotate and transform vertices
            xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
            trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
            new_verts = np.round(trans_pts).astype(np.int).T.tolist()
            x_points = [pt[0] for pt in new_verts]
            y_points = [pt[1] for pt in new_verts]
            xtl = int(min(x_points))
            xbr = int(max(x_points))
            ytl = int(min(y_points))
            ybr = int(max(y_points))
            bbox_ = (xtl, ytl, xbr - xtl, ybr - ytl)
            theta_ = 0.0
            bbox_list_.append(bbox_)
            theta_list_.append(theta_)
        bbox_list = bbox_list_
        theta_list = theta_list_

    # Build transformation from image to chip
    M_list = [
        vt.get_image_to_chip_transform(bbox, new_size, theta)
        for bbox, theta, new_size in zip(bbox_list, theta_list, target_size_list)
    ]

    # Extract "chips"
    flags = cv2.INTER_LANCZOS4
    borderMode = cv2.BORDER_CONSTANT
    warpkw = dict(flags=flags, borderMode=borderMode)

    def _compute_localiation_chip(tup):
        new_size, M = tup
        chip = cv2.warpAffine(img, M[0:2], tuple(new_size), **warpkw)
        # cv2.imshow('', chip)
        # cv2.waitKey()
        msg = 'Chip shape {!r} does not agree with target size {!r}'.format(
            chip.shape,
            target_size,
        )
        assert chip.shape[0] == new_size[0] and chip.shape[1] == new_size[1], msg
        return chip

    arg_list = list(zip(target_size_list, M_list))
    chip_list = [_compute_localiation_chip(tup_) for tup_ in arg_list]
    gid_list = [gid] * len(chip_list)
    return gid_list, chip_list


def get_localization_masks_worker(gid, img, bbox_list, theta_list, target_size):
    target_size_list = [target_size] * len(bbox_list)
    verts_list = vt.geometry.scaled_verts_from_bbox_gen(bbox_list, theta_list)

    # Extract "masks"
    interpolation = cv2.INTER_LANCZOS4
    warpkw = dict(interpolation=interpolation)
    fill_pixel_value = (128, 128, 128)  # Grey-scale medium

    def _compute_localiation_mask(tup):
        new_size, vert_list = tup
        # Copy the image, mask out the patch
        img_ = np.copy(img)
        vert_list_ = np.array(vert_list, dtype=np.int32)
        cv2.fillConvexPoly(img_, vert_list_, fill_pixel_value)
        # Resize the image
        mask = cv2.resize(img_, tuple(new_size), **warpkw)
        # cv2.imshow('', mask)
        # cv2.waitKey()
        msg = 'Chip shape {!r} does not agree with target size {!r}'.format(
            mask.shape, new_size
        )
        assert mask.shape[0] == new_size[0] and mask.shape[1] == new_size[1], msg
        return mask

    arg_list = list(zip(target_size_list, verts_list))
    mask_list = [_compute_localiation_mask(tup_) for tup_ in arg_list]
    gid_list = [gid] * len(mask_list)
    return gid_list, mask_list


def get_localization_chips(ibs, loc_id_list, target_size=(128, 128), axis_aligned=False):
    depc = ibs.depc_image
    gid_list_ = depc.get_ancestor_rowids('localizations', loc_id_list, 'images')
    assert len(gid_list_) == len(loc_id_list)

    # Grab the localizations
    bboxes_list = depc.get_native('localizations', loc_id_list, 'bboxes')
    len_list = [len(bbox_list) for bbox_list in bboxes_list]
    avg = sum(len_list) / len(len_list)
    args = (
        len(loc_id_list),
        min(len_list),
        avg,
        max(len_list),
        sum(len_list),
    )
    logger.info(
        'Extracting %d localization chips (min: %d, avg: %0.02f, max: %d, total: %d)'
        % args
    )
    thetas_list = depc.get_native('localizations', loc_id_list, 'thetas')

    OLD = True
    if OLD:
        gids_list = [
            np.array([gid] * len(bbox_list))
            for gid, bbox_list in zip(gid_list_, bboxes_list)
        ]
        # Flatten all of these lists for efficiency
        bbox_list = ut.flatten(bboxes_list)
        theta_list = ut.flatten(thetas_list)

        if axis_aligned:
            # Over-write bbox and theta with a friendlier, axis-aligned version
            bbox_list_ = []
            theta_list_ = []
            for bbox, theta in zip(bbox_list, theta_list):
                # Transformation matrix
                R = vt.rotation_around_bbox_mat3x3(theta, bbox)
                # Get verticies of the annotation polygon
                verts = vt.verts_from_bbox(bbox, close=True)
                # Rotate and transform vertices
                xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
                trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
                new_verts = np.round(trans_pts).astype(np.int).T.tolist()
                x_points = [pt[0] for pt in new_verts]
                y_points = [pt[1] for pt in new_verts]
                xtl = int(min(x_points))
                xbr = int(max(x_points))
                ytl = int(min(y_points))
                ybr = int(max(y_points))
                bbox_ = (xtl, ytl, xbr - xtl, ybr - ytl)
                theta_ = 0.0
                bbox_list_.append(bbox_)
                theta_list_.append(theta_)
            bbox_list = bbox_list_
            theta_list = theta_list_

        gid_list = ut.flatten(gids_list)
        bbox_size_list = ut.take_column(bbox_list, [2, 3])
        newsize_list = [target_size] * len(bbox_list)

        # Checks
        invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
        invalid_bboxes = ut.compress(bbox_list, invalid_flags)
        assert len(invalid_bboxes) == 0, 'invalid bboxes={!r}'.format(invalid_bboxes)

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
        arg_list = list(zip(gid_list, newsize_list, M_list))
        for tup in ut.ProgIter(arg_list, lbl='computing localization chips', bs=True):
            gid, new_size, M = tup
            if gid != last_gid:
                img = ibs.get_images(gid)
                last_gid = gid
            chip = cv2.warpAffine(img, M[0:2], tuple(new_size), **warpkw)
            # cv2.imshow('', chip)
            # cv2.waitKey()
            msg = 'Chip shape {!r} does not agree with target size {!r}'.format(
                chip.shape,
                target_size,
            )
            assert (
                chip.shape[0] == target_size[0] and chip.shape[1] == target_size[1]
            ), msg
            chip_list.append(chip)
    else:
        target_size_list = [target_size] * len(bboxes_list)
        axis_aligned_list = [axis_aligned] * len(bboxes_list)
        img_list = [ibs.get_images(gid) for gid in gid_list_]
        arg_iter = list(
            zip(
                gid_list_,
                img_list,
                bboxes_list,
                thetas_list,
                target_size_list,
                axis_aligned_list,
            )
        )
        result_list = ut.util_parallel.generate2(
            get_localization_chips_worker, arg_iter, ordered=True
        )
        # Compute results
        result_list = list(result_list)
        # Extract results
        gids_list = ut.take_column(result_list, 0)
        chips_list = ut.take_column(result_list, 1)
        # Explicitly garbage collect large list of chips
        result_list = None
        # Flatten results
        gid_list = ut.flatten(gids_list)
        chip_list = ut.flatten(chips_list)
        assert len(gid_list) == len(chip_list)

    return gid_list_, gid_list, chip_list


def get_localization_masks(ibs, loc_id_list, target_size=(128, 128)):
    depc = ibs.depc_image
    gid_list_ = depc.get_ancestor_rowids('localizations', loc_id_list, 'images')
    assert len(gid_list_) == len(loc_id_list)

    # Grab the localizations
    bboxes_list = depc.get_native('localizations', loc_id_list, 'bboxes')
    len_list = [len(bbox_list) for bbox_list in bboxes_list]
    avg = sum(len_list) / len(len_list)
    args = (
        len(loc_id_list),
        min(len_list),
        avg,
        max(len_list),
        sum(len_list),
    )
    logger.info(
        'Extracting %d localization masks (min: %d, avg: %0.02f, max: %d, total: %d)'
        % args
    )
    thetas_list = depc.get_native('localizations', loc_id_list, 'thetas')

    OLD = True
    if OLD:
        gids_list = [
            np.array([gid] * len(bbox_list))
            for gid, bbox_list in zip(gid_list_, bboxes_list)
        ]
        # Flatten all of these lists for efficiency
        bbox_list = ut.flatten(bboxes_list)
        theta_list = ut.flatten(thetas_list)
        verts_list = vt.geometry.scaled_verts_from_bbox_gen(bbox_list, theta_list)
        gid_list = ut.flatten(gids_list)
        bbox_size_list = ut.take_column(bbox_list, [2, 3])
        newsize_list = [target_size] * len(bbox_list)

        # Checks
        invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
        invalid_bboxes = ut.compress(bbox_list, invalid_flags)
        assert len(invalid_bboxes) == 0, 'invalid bboxes={!r}'.format(invalid_bboxes)

        # Extract "masks"
        interpolation = cv2.INTER_LANCZOS4
        warpkw = dict(interpolation=interpolation)

        last_gid = None
        mask_list = []
        arg_list = list(zip(gid_list, newsize_list, verts_list))
        for tup in ut.ProgIter(arg_list, lbl='computing localization masks', bs=True):
            gid, new_size, vert_list = tup
            if gid != last_gid:
                img = ibs.get_images(gid)
                last_gid = gid

            # Copy the image, mask out the patch
            img_ = np.copy(img)
            vert_list_ = np.array(vert_list, dtype=np.int32)
            cv2.fillConvexPoly(img_, vert_list_, (128, 128, 128))

            # Resize the image
            mask = cv2.resize(img_, tuple(new_size), **warpkw)
            # cv2.imshow('', mask)
            # cv2.waitKey()
            msg = 'Chip shape {!r} does not agree with target size {!r}'.format(
                mask.shape,
                target_size,
            )
            assert (
                mask.shape[0] == target_size[0] and mask.shape[1] == target_size[1]
            ), msg
            mask_list.append(mask)
    else:
        target_size_list = [target_size] * len(bboxes_list)
        img_list = [ibs.get_images(gid) for gid in gid_list_]
        arg_iter = list(
            zip(gid_list_, img_list, bboxes_list, thetas_list, target_size_list)
        )
        result_list = ut.util_parallel.generate2(
            get_localization_masks_worker, arg_iter, ordered=True
        )
        # Compute results
        result_list = list(result_list)
        # Extract results
        gids_list = ut.take_column(result_list, 0)
        chips_list = ut.take_column(result_list, 1)
        # Explicitly garbage collect large list of chips
        result_list = None
        # Flatten results
        gid_list = ut.flatten(gids_list)
        chip_list = ut.flatten(chips_list)
        assert len(gid_list) == len(chip_list)

    return gid_list_, gid_list, mask_list


def get_localization_aoi2(ibs, loc_id_list, target_size=(192, 192)):
    depc = ibs.depc_image
    gid_list_ = depc.get_ancestor_rowids('localizations', loc_id_list, 'images')
    assert len(gid_list_) == len(loc_id_list)

    # Grab the localizations
    bboxes_list = depc.get_native('localizations', loc_id_list, 'bboxes')
    gids_list = [
        np.array([gid] * len(bbox_list)) for gid, bbox_list in zip(gid_list_, bboxes_list)
    ]
    # Flatten all of these lists for efficiency
    gid_list = ut.flatten(gids_list)
    bbox_list = ut.flatten(bboxes_list)
    size_list = ibs.get_image_sizes(gid_list)

    config_ = {
        'draw_annots': False,
        'thumbsize': target_size,
    }
    thumbnail_list = depc.get_property('thumbnails', gid_list, 'img', config=config_)

    return gid_list_, gid_list, thumbnail_list, bbox_list, size_list


ChipListImgType = dtool.ExternType(
    ut.partial(ut.load_cPkl, verbose=False),
    ut.partial(ut.save_cPkl, verbose=False),
    extkey='ext',
)


class Chip2Config(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('localization_chip_target_size', (128, 128)),
        ut.ParamInfo('localization_chip_masking', False),
    ]
    _sub_config_list = [ThumbnailConfig]


@register_preproc(
    tablename='localizations_chips',
    parents=['localizations'],
    colnames=['chips'],
    coltypes=[ChipListImgType],
    configclass=Chip2Config,
    fname='chipcache4',
    chunksize=32 if const.CONTAINERIZED else 128,
)
def compute_localizations_chips(depc, loc_id_list, config=None):
    r"""Extract the detections for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        loc_id_list (list):  list of localization rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        wbia compute_localizations_chips

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> config = {'combined': True, 'localization_chip_masking': True}
        >>> # depc.delete_property('localizations_chips', gid_list, config=config)
        >>> results = depc.get_property('localizations_chips', gid_list, None, config=config)
        >>> print(results)
        >>> config = {'combined': True, 'localization_chip_masking': False}
        >>> # depc.delete_property('localizations_chips', gid_list, config=config)
        >>> results = depc.get_property('localizations_chips', gid_list, None, config=config)
        >>> print(results)
    """
    logger.info('[ibs] Process Localization Chips')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller

    masking = config['localization_chip_masking']
    target_size = config['localization_chip_target_size']
    target_size_list = [target_size] * len(loc_id_list)

    gid_list_ = depc.get_ancestor_rowids('localizations', loc_id_list, 'images')
    assert len(gid_list_) == len(loc_id_list)

    # Grab the localizations
    bboxes_list = depc.get_native('localizations', loc_id_list, 'bboxes')
    thetas_list = depc.get_native('localizations', loc_id_list, 'thetas')
    len_list = [len(bbox_list) for bbox_list in bboxes_list]
    avg = sum(len_list) / len(len_list)
    args = (
        len(loc_id_list),
        min(len_list),
        avg,
        max(len_list),
        sum(len_list),
    )

    # Create image iterator
    img_list = (ibs.get_images(gid) for gid in gid_list_)

    if masking:
        logger.info(
            'Extracting %d localization masks (min: %d, avg: %0.02f, max: %d, total: %d)'
            % args
        )
        worker_func = get_localization_masks_worker
    else:
        logger.info(
            'Extracting %d localization chips (min: %d, avg: %0.02f, max: %d, total: %d)'
            % args
        )
        worker_func = get_localization_chips_worker

    arg_iter = zip(gid_list_, img_list, bboxes_list, thetas_list, target_size_list)
    result_list = ut.util_parallel.generate2(
        worker_func, arg_iter, ordered=True, nTasks=len(gid_list_), force_serial=True
    )

    # Return the results
    for gid, chip_list in result_list:
        ret_tuple = (chip_list,)
        yield ret_tuple


class ClassifierLocalizationsConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('classifier_algo', 'cnn', valid_values=['cnn', 'svm']),
        ut.ParamInfo('classifier_weight_filepath', None),
        ut.ParamInfo(
            'classifier_masking', False, hideif=False
        ),  # True will classify localization chip as whole-image, False will classify whole image with localization masked out.
    ]
    _sub_config_list = [
        ThumbnailConfig,
    ]


@register_preproc(
    tablename='localizations_classifier',
    parents=['localizations'],
    colnames=['score', 'class'],
    coltypes=[np.ndarray, np.ndarray],
    configclass=ClassifierLocalizationsConfig,
    fname='detectcache',
    chunksize=2 if const.CONTAINERIZED else 8,
)
def compute_localizations_classifications(depc, loc_id_list, config=None):
    r"""Extract the detections for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        loc_id_list (list):  list of localization rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        wbia compute_localizations_classifications

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:8]
        >>> config = {'algo': 'yolo'}
        >>> # depc.delete_property('localizations_classifier', gid_list, config=config)
        >>> results = depc.get_property('localizations_classifier', gid_list, None, config=config)
        >>> print(results)
        >>> config = {'algo': 'yolo', 'classifier_masking': True}
        >>> # depc.delete_property('localizations_classifier', gid_list, config=config)
        >>> results = depc.get_property('localizations_classifier', gid_list, None, config=config)
        >>> print(results)
        >>>
        >>> depc = ibs.depc_image
        >>> gid_list = list(set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))))
        >>> config = {'combined': True, 'classifier_algo': 'svm', 'classifier_weight_filepath': None}
        >>> # depc.delete_property('localizations_classifier', gid_list, config=config)
        >>> results = depc.get_property('localizations_classifier', gid_list, None, config=config)
        >>> print(results)
        >>>
        >>> config = {'combined': True, 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-10'}
        >>> # depc.delete_property('localizations_classifier', gid_list, config=config)
        >>> results = depc.get_property('localizations_classifier', gid_list, None, config=config)
        >>> print(results)
        >>>
        >>> config = {'combined': True, 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-50'}
        >>> results = depc.get_property('localizations_classifier', gid_list, None, config=config)
        >>> print(results)
        >>>
        >>> config = {'combined': True, 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-100'}
        >>> results = depc.get_property('localizations_classifier', gid_list, None, config=config)
        >>> print(results)
    """
    logger.info('[ibs] Process Localization Classifications')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller

    masking = config.get('classifier_masking', False)

    # Get the results from the algorithm
    if config['classifier_algo'] in ['cnn']:
        if masking:
            gid_list_, gid_list, thumbnail_list = get_localization_masks(
                ibs, loc_id_list, target_size=(192, 192)
            )
        else:
            gid_list_, gid_list, thumbnail_list = get_localization_chips(
                ibs, loc_id_list, target_size=(192, 192)
            )

        # Generate thumbnail classifications
        result_list = ibs.generate_thumbnail_class_list(thumbnail_list, **config)

        # Assert the length is the same
        assert len(gid_list) == len(result_list)

        # Release thumbnails
        thumbnail_list = None

        # Group the results
        group_dict = {}
        for gid, result in zip(gid_list, result_list):
            if gid not in group_dict:
                group_dict[gid] = []
            group_dict[gid].append(result)
        assert len(gid_list_) == len(group_dict.keys())

        if masking:
            # We need to perform a difference calculation to see how much the masking
            # caused a deviation from the un-masked image
            config_ = dict(config)
            key_list = ['thumbnail_cfg', 'classifier_masking']
            for key in key_list:
                config_.pop(key)
            class_list_ = depc.get_property(
                'classifier', gid_list_, 'class', config=config_
            )
            score_list_ = depc.get_property(
                'classifier', gid_list_, 'score', config=config_
            )
        else:
            class_list_ = [None] * len(gid_list_)
            score_list_ = [None] * len(gid_list_)

        # Return the results
        for gid, class_, score_ in zip(gid_list_, class_list_, score_list_):
            result_list = group_dict[gid]
            zipped_list = list(zip(*result_list))
            score_list = np.array(zipped_list[0])
            class_list = np.array(zipped_list[1])
            if masking:
                score_ = score_ if class_ == 'positive' else 1.0 - score_
                score_list = score_ - score_list
                class_list = np.array(['positive'] * len(score_list))
            # Return tuple values
            ret_tuple = (
                score_list,
                class_list,
            )
            yield ret_tuple
    elif config['classifier_algo'] in ['svm']:
        from wbia.algo.detect.svm import classify

        # from localizations get gids
        config_ = {
            'combined': True,
            'feature2_algo': 'resnet',
            'feature2_chip_masking': masking,
        }
        gid_list_ = depc.get_ancestor_rowids('localizations', loc_id_list, 'images')
        assert len(gid_list_) == len(loc_id_list)

        # Get features
        vectors_list = depc.get_property(
            'localizations_features', gid_list_, 'vector', config=config_
        )
        vectors_list_ = np.vstack(vectors_list)
        # Get gid_list
        shape_list = [vector_list.shape[0] for vector_list in vectors_list]
        gids_list = [[gid_] * shape for gid_, shape in zip(gid_list_, shape_list)]
        gid_list = ut.flatten(gids_list)

        # Stack vectors and classify
        classifier_weight_filepath = config['classifier_weight_filepath']
        result_list = classify(
            vectors_list_, weight_filepath=classifier_weight_filepath, verbose=True
        )

        # Group the results
        score_dict = {}
        class_dict = {}
        for index, (gid, result) in enumerate(zip(gid_list, result_list)):
            if gid not in score_dict:
                score_dict[gid] = []
            if gid not in class_dict:
                class_dict[gid] = []
            score_, class_ = result
            score_dict[gid].append(score_)
            class_dict[gid].append(class_)
        assert len(gid_list_) == len(score_dict.keys())
        assert len(gid_list_) == len(class_dict.keys())

        if masking:
            # We need to perform a difference calculation to see how much the masking
            # caused a deviation from the un-masked image
            config_ = dict(config)
            key_list = ['thumbnail_cfg', 'classifier_masking']
            for key in key_list:
                config_.pop(key)
            class_list_ = depc.get_property(
                'classifier', gid_list_, 'class', config=config_
            )
            score_list_ = depc.get_property(
                'classifier', gid_list_, 'score', config=config_
            )
        else:
            class_list_ = [None] * len(gid_list_)
            score_list_ = [None] * len(gid_list_)

        # Return the results
        for gid_, class_, score_ in zip(gid_list_, class_list_, score_list_):
            score_list = score_dict[gid_]
            class_list = class_dict[gid_]
            if masking:
                score_ = score_ if class_ == 'positive' else 1.0 - score_
                score_list = score_ - np.array(score_list)
                class_list = np.array(['positive'] * len(score_list))
            ret_tuple = (
                np.array(score_list),
                np.array(class_list),
            )
            yield ret_tuple


class Feature2Config(dtool.Config):
    _param_info_list = [
        ut.ParamInfo(
            'feature2_algo',
            'vgg16',
            valid_values=['vgg', 'vgg16', 'vgg19', 'resnet', 'inception'],
        ),
        ut.ParamInfo('feature2_chip_masking', False, hideif=False),
        ut.ParamInfo('flatten', True),
    ]
    _sub_config_list = [ThumbnailConfig]


@register_preproc(
    tablename='localizations_features',
    parents=['localizations'],
    colnames=['vector'],
    coltypes=[np.ndarray],
    configclass=Feature2Config,
    fname='featcache',
    chunksize=2 if const.CONTAINERIZED else 4,
)
def compute_localizations_features(depc, loc_id_list, config=None):
    r"""Compute features on images using pre-trained state-of-the-art models in Keras.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (np.ndarray, ): tup

    CommandLine:
        wbia compute_localizations_features

    CommandLine:
        python -m wbia.core_images compute_localizations_features --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> print(depc.get_tablenames())
        >>> gid_list = ibs.get_valid_gids()[:16]
        >>> config = {'feature2_algo': 'vgg16', 'combined': True}
        >>> depc.delete_property('localizations_features', gid_list, config=config)
        >>> features = depc.get_property('localizations_features', gid_list, 'vector', config=config)
        >>> print(features)
        >>> config = {'feature2_algo': 'vgg19', 'combined': True}
        >>> depc.delete_property('localizations_features', gid_list, config=config)
        >>> features = depc.get_property('localizations_features', gid_list, 'vector', config=config)
        >>> print(features)
        >>> config = {'feature2_algo': 'resnet', 'combined': True}
        >>> depc.delete_property('localizations_features', gid_list, config=config)
        >>> features = depc.get_property('localizations_features', gid_list, 'vector', config=config)
        >>> print(features)
        >>> config = {'feature2_algo': 'inception', 'combined': True}
        >>> depc.delete_property('localizations_features', gid_list, config=config)
        >>> features = depc.get_property('localizations_features', gid_list, 'vector', config=config)
        >>> print(features)
    """
    from keras.preprocessing import image as preprocess_image
    from PIL import Image

    logger.info('[ibs] Preprocess Features')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller
    target_size = (224, 224)
    ######################################################################################
    if config['feature2_algo'] in ['vgg', 'vgg16']:
        from keras.applications.vgg16 import VGG16 as MODEL_CLASS
        from keras.applications.vgg16 import preprocess_input
    ######################################################################################
    elif config['feature2_algo'] in ['vgg19']:
        from keras.applications.vgg19 import VGG19 as MODEL_CLASS
        from keras.applications.vgg19 import preprocess_input
    ######################################################################################
    elif config['feature2_algo'] in ['resnet']:
        from keras.applications.resnet50 import ResNet50 as MODEL_CLASS  # NOQA
        from keras.applications.resnet50 import preprocess_input
    ######################################################################################
    elif config['feature2_algo'] in ['inception']:
        from keras.applications.inception_v3 import InceptionV3 as MODEL_CLASS  # NOQA
        from keras.applications.inception_v3 import preprocess_input

        target_size = (299, 299)
    ######################################################################################
    else:
        raise ValueError(
            'specified feature algo is not supported in config = {!r}'.format(config)
        )

    # Load chips
    if config['feature2_chip_masking']:
        gid_list_, gid_list, thumbnail_list = get_localization_masks(
            ibs, loc_id_list, target_size=target_size
        )
    else:
        gid_list_, gid_list, thumbnail_list = get_localization_chips(
            ibs, loc_id_list, target_size=target_size
        )

    # Build model
    model = MODEL_CLASS(include_top=False)

    # Define Preprocess
    def _preprocess(thumbnail):
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(thumbnail)
        # Process PIL image
        image_array = preprocess_image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        return image_array

    thumbnail_iter = ut.ProgIter(thumbnail_list, lbl='preprocessing chips', bs=True)
    image_array = [_preprocess(thumbnail) for thumbnail in thumbnail_iter]
    # Release thumbnails
    thumbnail_list = None

    inference_iter = ut.ProgIter(image_array, lbl='forward inference', bs=True)
    result_list = [model.predict(image_array_) for image_array_ in inference_iter]

    # Release image_array
    image_array = None

    # Group the results
    group_dict = {}
    for gid, result in zip(gid_list, result_list):
        if gid not in group_dict:
            group_dict[gid] = []
        group_dict[gid].append(result)
    assert len(gid_list_) == len(group_dict.keys())

    # Return the results
    group_iter = ut.ProgIter(gid_list_, lbl='grouping results', bs=True)
    for gid in group_iter:
        result_list = group_dict[gid]
        if config['flatten']:
            result_list = [_.flatten() for _ in result_list]
        result_list = np.vstack(result_list)
        # Return tuple values
        ret_tuple = (result_list,)
        yield ret_tuple


class LabelerConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo(
            'labeler_algo',
            'pipeline',
            valid_values=['azure', 'cnn', 'pipeline', 'densenet'],
        ),
        ut.ParamInfo('labeler_weight_filepath', None),
        ut.ParamInfo('labeler_axis_aligned', False, hideif=False),
    ]


@register_preproc(
    tablename='localizations_labeler',
    parents=['localizations'],
    colnames=['score', 'species', 'viewpoint', 'quality', 'orientation', 'probs'],
    coltypes=[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list],
    configclass=LabelerConfig,
    fname='detectcache',
    chunksize=32 if const.CONTAINERIZED else 128,
)
def compute_localizations_labels(depc, loc_id_list, config=None):
    r"""Extract the detections for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        loc_id_list (list):  list of localization rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        python -m wbia.core_images --exec-compute_localizations_labels

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:10]
        >>> config = {'labeler_algo': 'densenet', 'labeler_weight_filepath': 'giraffe_v1'}
        >>> # depc.delete_property('localizations_labeler', aid_list)
        >>> results = depc.get_property('localizations_labeler', gid_list, None, config=config)
        >>> print(results)
        >>> config = {'labeler_weight_filepath': 'candidacy'}
        >>> # depc.delete_property('localizations_labeler', aid_list)
        >>> results = depc.get_property('localizations_labeler', gid_list, None, config=config)
        >>> print(results)
    """
    from os.path import exists, join

    logger.info('[ibs] Process Localization Labels')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller

    if config['labeler_algo'] in ['pipeline', 'cnn']:
        gid_list_, gid_list, chip_list = get_localization_chips(
            ibs,
            loc_id_list,
            target_size=(128, 128),
            axis_aligned=config['labeler_axis_aligned'],
        )
        result_list = ibs.generate_chip_label_list(chip_list, **config)
    elif config['labeler_algo'] in ['azure']:
        raise NotImplementedError('Azure is not implemented for images')
    elif config['labeler_algo'] in ['densenet']:
        from wbia.algo.detect import densenet

        target_size = (
            densenet.INPUT_SIZE,
            densenet.INPUT_SIZE,
        )
        gid_list_, gid_list, chip_list = get_localization_chips(
            ibs,
            loc_id_list,
            target_size=target_size,
            axis_aligned=config['labeler_axis_aligned'],
        )
        config = dict(config)
        config['classifier_weight_filepath'] = config['labeler_weight_filepath']
        nonce = ut.random_nonce()[:16]
        cache_path = join(ibs.cachedir, 'localization_labels_{}'.format(nonce))
        assert not exists(cache_path)
        ut.ensuredir(cache_path)
        chip_filepath_list = []
        for index, chip in enumerate(chip_list):
            chip_filepath = join(cache_path, 'chip_%08d.png' % (index,))
            cv2.imwrite(chip_filepath, chip)
            assert exists(chip_filepath)
            chip_filepath_list.append(chip_filepath)
        result_gen = densenet.test_dict(chip_filepath_list, return_dict=True, **config)
        result_list = list(result_gen)
        ut.delete(cache_path)

    assert len(gid_list) == len(result_list)

    # Release chips
    chip_list = None

    # Group the results
    group_dict = {}
    for gid, result in zip(gid_list, result_list):
        if gid not in group_dict:
            group_dict[gid] = []
        group_dict[gid].append(result)

    # Return the results
    for gid in gid_list_:
        result_list = group_dict.get(gid, None)
        if result_list is None:
            ret_tuple = (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                [],
            )
        else:
            zipped_list = list(zip(*result_list))
            ret_tuple = (
                np.array(zipped_list[0]),
                np.array(zipped_list[1]),
                np.array(zipped_list[2]),
                np.array(zipped_list[3]),
                np.array(zipped_list[4]),
                list(zipped_list[5]),
            )
        yield ret_tuple


class AoIConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('aoi_two_weight_filepath', None),
    ]


@register_preproc(
    tablename='localizations_aoi_two',
    parents=['localizations'],
    colnames=['score', 'class'],
    coltypes=[np.ndarray, np.ndarray],
    configclass=AoIConfig,
    fname='detectcache',
    chunksize=32 if const.CONTAINERIZED else 128,
)
def compute_localizations_interest(depc, loc_id_list, config=None):
    r"""Extract the detections for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        loc_id_list (list):  list of localization rowids
        config (dict): (default = None)

    Yields:
        (float, str): tup

    CommandLine:
        wbia compute_localizations_labels

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:100]
        >>> depc.delete_property('labeler', gid_list)
        >>> results = depc.get_property('labeler', gid_list, None)
        >>> results = depc.get_property('labeler', gid_list, 'species')
        >>> print(results)
    """
    logger.info('[ibs] Process Localization AoI2s')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller

    values = get_localization_aoi2(ibs, loc_id_list, target_size=(192, 192))
    gid_list_, gid_list, thumbnail_list, bbox_list, size_list = values

    # Get the results from the algorithm
    size_list = ibs.get_image_sizes(gid_list)
    result_list = ibs.generate_thumbnail_aoi2_list(
        thumbnail_list, bbox_list, size_list, **config
    )
    assert len(gid_list) == len(result_list)

    # Release chips
    thumbnail_list = None

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
        zipped_list = list(zip(*result_list))
        ret_tuple = (
            np.array(zipped_list[0]),
            np.array(zipped_list[1]),
        )
        yield ret_tuple


class DetectorConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('classifier_weight_filepath', 'candidacy'),
        ut.ParamInfo('classifier_sensitivity', 0.10),
        #
        ut.ParamInfo('localizer_algo', 'yolo'),
        ut.ParamInfo('localizer_config_filepath', 'candidacy'),
        ut.ParamInfo('localizer_weight_filepath', 'candidacy'),
        ut.ParamInfo('localizer_grid', False),
        ut.ParamInfo('localizer_sensitivity', 0.10),
        #
        ut.ParamInfo('labeler_weight_filepath', 'candidacy'),
        ut.ParamInfo('labeler_sensitivity', 0.10),
    ]
    _sub_config_list = [
        ThumbnailConfig,
        LocalizerConfig,
    ]


@register_preproc(
    tablename='detections',
    parents=['images'],
    colnames=['score', 'bboxes', 'thetas', 'species', 'viewpoints', 'confs'],
    coltypes=[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=DetectorConfig,
    fname='detectcache',
    chunksize=32 if const.CONTAINERIZED else 256,
)
def compute_detections(depc, gid_list, config=None):
    r"""Extract the detections for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray): tup

    CommandLine:
        wbia compute_detections

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> # dbdir = '/Users/bluemellophone/Desktop/GGR-IBEIS-TEST/'
        >>> # dbdir = '/media/danger/GGR/GGR-IBEIS-TEST/'
        >>> # ibs = wbia.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_image
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> depc.delete_property('detections', gid_list)
        >>> detects = depc.get_property('detections', gid_list, None)
        >>> print(detects)
    """
    logger.info('[ibs] Preprocess Detections')
    logger.info('config = {!r}'.format(config))
    # Get controller
    ibs = depc.controller
    ibs.assert_valid_gids(gid_list)

    USE_CLASSIFIER = False

    if USE_CLASSIFIER:
        classifier_config = {
            'classifier_weight_filepath': config['classifier_weight_filepath'],
        }
        # Filter the gids by annotations
        prediction_list = depc.get_property(
            'classifier', gid_list, 'class', config=classifier_config
        )
        confidence_list = depc.get_property(
            'classifier', gid_list, 'score', config=classifier_config
        )
        confidence_list = [
            confidence if prediction == 'positive' else 1.0 - confidence
            for prediction, confidence in zip(prediction_list, confidence_list)
        ]
        gid_list_ = [
            gid
            for gid, confidence in zip(gid_list, confidence_list)
            if confidence >= config['classifier_sensitivity']
        ]
    else:
        classifier_config = {
            'classifier_two_weight_filepath': config['classifier_weight_filepath'],
        }
        # Filter the gids by annotations
        predictions_list = depc.get_property(
            'classifier_two', gid_list, 'classes', config=classifier_config
        )
        gid_list_ = [
            gid
            for gid, prediction_list in zip(gid_list, predictions_list)
            if len(prediction_list) > 0
        ]

    gid_set_ = set(gid_list_)
    # Get the localizations for the good gids and add formal annotations
    localizer_config = {
        'algo': config['localizer_algo'],
        'config_filepath': config['localizer_config_filepath'],
        'weight_filepath': config['localizer_weight_filepath'],
        'grid': config['localizer_grid'],
    }
    bboxes_list = depc.get_property(
        'localizations', gid_list_, 'bboxes', config=localizer_config
    )
    thetas_list = depc.get_property(
        'localizations', gid_list_, 'thetas', config=localizer_config
    )
    confses_list = depc.get_property(
        'localizations', gid_list_, 'confs', config=localizer_config
    )

    # Get the corrected species and viewpoints
    labeler_config = {
        'labeler_weight_filepath': config['labeler_weight_filepath'],
    }
    # depc.delete_property('localizations_labeler', gid_list_, config=labeler_config)
    specieses_list = depc.get_property(
        'localizations_labeler', gid_list_, 'species', config=labeler_config
    )
    viewpoints_list = depc.get_property(
        'localizations_labeler', gid_list_, 'viewpoint', config=labeler_config
    )
    scores_list = depc.get_property(
        'localizations_labeler', gid_list_, 'score', config=labeler_config
    )

    # Collect the detections, filtering by the localization confidence
    empty_list = [
        0.0,
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    ]
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
        #     logger.info('\t%r' % (tup, ))
        # logger.info('----')
        viewpoint_list = viewpoints_list[index]
        conf_list = confses_list[index]
        score_list = scores_list[index]
        zipped = list(
            zip(
                bbox_list,
                theta_list,
                species_list,
                viewpoint_list,
                conf_list,
                score_list,
            )
        )
        zipped_ = []
        for bbox, theta, species, viewpoint, conf, score in zipped:
            if (
                conf >= config['localizer_sensitivity']
                and score >= config['labeler_sensitivity']
            ):
                zipped_.append([bbox, theta, species, viewpoint, conf * score])
            else:
                logger.info(
                    'Localizer {:0.02f} {:0.02f}'.format(
                        conf, config['localizer_sensitivity']
                    )
                )
                logger.info(
                    'Labeler   {:0.02f} {:0.02f}'.format(
                        score, config['labeler_sensitivity']
                    )
                )
        if len(zipped_) == 0:
            detect_list = list(empty_list)
        else:
            detect_list = [0.0] + [np.array(_) for _ in zip(*zipped_)]
        detect_dict[gid] = detect_list

    # Filter the annotations by the localizer operating point
    for gid in gid_list:
        if gid not in gid_set_:
            assert gid not in detect_dict
            result = list(empty_list)
        else:
            assert gid in detect_dict
            result = detect_dict[gid]
        # logger.info(result)
        # raw_input()
        # logger.info('')
        # image = ibs.get_images(gid)
        # image = vt.resize(image, (500, 500))
        # cv2.imshow('', image)
        # cv2.waitKey(0)
        yield tuple(result)


class TileConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('tile_width', 512),
        ut.ParamInfo('tile_height', 512),
        ut.ParamInfo('tile_overlap', 64),
        ut.ParamInfo('tile_offset', 0, hideif=0),
        ut.ParamInfo('allow_borders', True),
        ut.ParamInfo('keep_extern', True),
        ut.ParamInfo('force_serial', False, hideif=False),
    ]


@register_preproc(
    tablename='tiles',
    parents=['images'],
    colnames=['paths', 'gids', 'num'],
    coltypes=[list, list, int],
    configclass=TileConfig,
    fname='tilecache',
    rm_extern_on_delete=True,
    chunksize=64,
)
def compute_tiles(depc, gid_list, config=None):
    r"""Compute the tile for a given input image.

    Args:
        depc (wbia.depends_cache.DependencyCache):
        gid_list (list):  list of image rowids
        config (dict): (default = None)

    Yields:
        (list, list, int): tup

    CommandLine:
        wbia --tf compute_tiles --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.core_images import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'testdb1'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_image
        >>> gid_list = sorted(ibs.get_valid_gids())[0:5]
        >>> config = {'tile_width': 128, 'tile_height': 128}
        >>> result = depc.get_property('tiles', gid_list, 'num', config=config)
        >>> nums = list(map(len, ibs.get_tile_children_gids(gid_list)))
        >>> nums_ = list(map(len, ibs.get_tile_descendants_gids(gid_list)))
        >>> assert result == nums
        >>> assert result == nums_
        >>> assert result == [204, 136, 221, 221, 221]
    """
    from os.path import abspath, join, relpath

    ibs = depc.controller

    tile_width = config['tile_width']
    tile_height = config['tile_height']
    tile_overlap = config['tile_overlap']
    tile_offset = config['tile_offset']
    allow_borders = config['allow_borders']
    keep_extern = config['keep_extern']

    if allow_borders:
        assert tile_offset == 0, 'Cannot use an offset with borders turned on'

    config_dict = dict(config)
    config_hashid = config.get_hashid()

    gpath_list = ibs.get_image_paths(gid_list)
    orient_list = ibs.get_image_orientation(gid_list)

    tile_size = (tile_width, tile_height)
    tile_size_list = [tile_size] * len(gid_list)
    tile_overlap_list = [tile_overlap] * len(gid_list)
    tile_offset_list = [tile_offset] * len(gid_list)

    tile_output_path = abspath(join(depc.cache_dpath, 'extern_tiles'))
    ut.ensuredir(tile_output_path)

    fmt_str = join(tile_output_path, 'tiles_gid_%d_w_%d_h_%d_ol_%d_os_%d_%s')
    output_path_list = [
        fmt_str
        % (
            gid,
            tile_width,
            tile_height,
            tile_overlap,
            tile_offset,
            config_hashid,
        )
        for gid in gid_list
    ]
    allow_border_list = [allow_borders] * len(gid_list)

    for output_path in output_path_list:
        ut.ensuredir(output_path)

    # Execute all tasks in parallel
    args_list = list(
        zip(
            gid_list,
            gpath_list,
            orient_list,
            tile_size_list,
            tile_overlap_list,
            tile_offset_list,
            output_path_list,
            allow_border_list,
        )
    )

    genkw = {
        'ordered': True,
        'chunksize': 256,
        'progkw': {'freq': 50},
        # 'adjust': True,
        'futures_threaded': True,
        'force_serial': ibs.force_serial or config['force_serial'],
    }
    gen = ut.generate2(compute_tile_helper, args_list, nTasks=len(args_list), **genkw)
    for val in gen:
        parent_gid, output_path, tile_filepath_list, bbox_list, border_list = val

        if keep_extern:
            gids = ibs.add_images(
                tile_filepath_list,
                auto_localize=False,
                ensure_loadable=False,
                ensure_exif=False,
            )
        else:
            gids = ibs.add_images(tile_filepath_list)

        if ut.duplicates_exist(gids):
            flag_list = []
            seen_set = set()
            for gid in gids:
                if gid is None:
                    flag = False
                else:
                    flag = gid not in seen_set
                    seen_set.add(gid)
                flag_list.append(flag)
            gids = ut.compress(gids, flag_list)
            bbox_list = ut.compress(bbox_list, flag_list)
            border_list = ut.compress(border_list, flag_list)

        num = len(gids)
        parent_gids = [parent_gid] * num
        config_dict_list = [config_dict] * num
        config_hashid_list = [config_hashid] * num

        ibs.set_tile_source(
            gids,
            parent_gids,
            bbox_list,
            border_list,
            config_dict_list,
            config_hashid_list,
        )

        if keep_extern:
            tile_relative_filepath_list_ = [
                relpath(tile_filepath, start=depc.cache_dpath)
                for tile_filepath in tile_filepath_list
            ]
        else:
            ut.delete(output_path)
            tile_relative_filepath_list_ = [None] * len(tile_filepath_list)

        yield tile_relative_filepath_list_, gids, num


def compute_tile_helper(gid, gpath, orient, size, overlap, offset, opath, borders):
    from os.path import join

    ext = '.jpg'
    w, h = size
    ol = overlap
    os = offset

    image = vt.imread(gpath, orient=orient)
    h_, w_ = image.shape[:2]

    y_ = int(np.floor((h_ - ol) / (h - ol)))
    x_ = int(np.floor((w_ - ol) / (w - ol)))
    iy = (h * y_) - (ol * (y_ - 1))
    ix = (w * x_) - (ol * (x_ - 1))
    oy = int(np.floor((h_ - iy) * 0.5))
    ox = int(np.floor((w_ - ix) * 0.5))

    miny = 0
    minx = 0
    maxy = h_ - h
    maxx = w_ - w

    ys = list(range(oy, h_ - h + 1, h - ol))
    yb = [False] * len(ys)
    xs = list(range(ox, w_ - w + 1, w - ol))
    xb = [False] * len(xs)

    if borders and oy > 0:
        ys = [miny] + ys + [maxy]
        yb = [True] + yb + [True]

    if borders and ox > 0:
        xs = [minx] + xs + [maxx]
        xb = [True] + xb + [True]

    tile_filepath_list = []
    bbox_list = []
    border_list = []

    for y0, yb_ in zip(ys, yb):
        y0 += os
        y1 = y0 + h
        for x0, xb_ in zip(xs, xb):
            x0 += os
            x1 = x0 + w

            # Sanity, mostly to check for offset
            valid = True
            try:
                assert x1 - x0 == w, '%d, %d' % (
                    x1 - x0,
                    w,
                )
                assert y1 - y0 == h, '%d, %d' % (
                    y1 - y0,
                    h,
                )
                assert 0 <= x0 and x0 <= w_, '%d, %d' % (
                    x0,
                    w_,
                )
                assert 0 <= x1 and x1 <= w_, '%d, %d' % (
                    x1,
                    w_,
                )
                assert 0 <= y0 and y0 <= h_, '%d, %d' % (
                    y0,
                    h_,
                )
                assert 0 <= y1 and y1 <= h_, '%d, %d' % (
                    y1,
                    h_,
                )
            except AssertionError:
                valid = False

            if valid:
                bbox = (x0, y0, w, h)
                border = yb_ or xb_

                args = (
                    gid,
                    x0,
                    y0,
                    w,
                    h,
                    ext,
                )
                tile_filename = 'tile_gid_%d_xtl_%d_ytl_%d_w_%d_h_%d%s' % args
                file_filepath = join(opath, tile_filename)

                tile = image[y0:y1, x0:x1]
                vt.imwrite(file_filepath, tile)

                tile_filepath_list.append(file_filepath)
                bbox_list.append(bbox)
                border_list.append(border)

    return gid, opath, tile_filepath_list, bbox_list, border_list


class CameraTrapEXIFConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('bottom', 80),
        ut.ParamInfo('psm', 7),
        ut.ParamInfo('oem', 1),
        ut.ParamInfo('whitelist', '0123456789°CF/:'),
    ]


@register_preproc(
    tablename='cameratrap_exif',
    parents=['images'],
    colnames=['raw'],
    coltypes=[str],
    configclass=CameraTrapEXIFConfig,
    fname='exifcache',
    chunksize=1024,
)
def compute_cameratrap_exif(depc, gid_list, config=None):
    ibs = depc.controller

    gpath_list = ibs.get_image_paths(gid_list)
    orient_list = ibs.get_image_orientation(gid_list)

    arg_iter = list(zip(gpath_list, orient_list))
    kwargs_iter = [config] * len(gid_list)
    raw_list = ut.util_parallel.generate2(
        compute_cameratrap_exif_worker, arg_iter, kwargs_iter
    )
    for raw in raw_list:
        yield (raw,)


def compute_cameratrap_exif_worker(
    gpath, orient, bottom=80, psm=7, oem=1, whitelist='0123456789°CF/:'
):
    import pytesseract

    img = vt.imread(gpath, orient=orient)
    # Crop
    img = img[-1 * bottom :, :, :]

    config = []
    if sys.platform.startswith('darwin'):
        config += ['--tessdata-dir', '"/opt/local/share/"']
    else:
        config += ['--tessdata-dir', '"/usr/share/tesseract-ocr/"']
    config += [
        '--psm',
        str(psm),
        '--oem',
        str(oem),
        '-c',
        'tessedit_char_whitelist={}'.format(whitelist),
    ]
    config = ' '.join(config)

    try:
        raw = pytesseract.image_to_string(img, config=config)
    except Exception:
        raw = None

    return raw
