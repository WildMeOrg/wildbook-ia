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
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
from wbia import dtool
import utool as ut
import numpy as np
import vtool as vt
import cv2
from wbia.control.controller_inject import register_preprocs
import sys
import wbia.constants as const

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
        >>> assert ut.hash_data(thumb) in ['wcuppmpowkvhfmfcnrxdeedommihexfu', 'wjkpjrsmqzdhmqdxjbgomdmqxaxsckxn']
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
        ut.ParamInfo('classifier_algo', 'cnn', valid_values=['cnn', 'svm', 'densenet']),
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
    chunksize=32 if const.CONTAINERIZED else 128,
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
    print('[ibs] Process Image Classifications')
    print('config = %r' % (config,))
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
            'thumbnails', gid_list, 'img', config=config_, read_extern=False, ensure=True,
        )
        result_list = densenet.test(thumbpath_list, ibs=ibs, gid_list=gid_list, **config)
    else:
        raise ValueError(
            'specified classifier algo is not supported in config = %r' % (config,)
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
    print('[ibs] Process Image Classifications2')
    print('config = %r' % (config,))
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
            'thumbnails', gid_list, 'img', config=config_, read_extern=False, ensure=True,
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
            'specified classifier_two algo is not supported in config = %r' % (config,)
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
    print('[ibs] Preprocess Features')
    print('config = %r' % (config,))
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
                'specified feature model is not supported in config = %r' % (config,)
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
                'specified feature model is not supported in config = %r' % (config,)
            )

        for feature in feature_list:
            if config['flatten']:
                feature = feature.flatten()
            yield (feature,)
    else:
        raise ValueError(
            'specified feature framework is not supported in config = %r' % (config,)
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

    print('[ibs] Preprocess Localizations')
    print('config = %r' % (config,))
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

        print('[ibs] detecting using PyDarknet CNN YOLO v1')
        detect_gen = yolo.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['lightnet']:
        from wbia.algo.detect import lightnet

        print('[ibs] detecting using Lightnet CNN YOLO v2')
        detect_gen = lightnet.detect_gid_list(ibs, gid_list, **config)
    elif config['algo'] in ['azure']:
        from wbia.algo.detect import azure

        print('[ibs] detecting using Azure CustomVision')
        detect_gen = azure.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['rf']:
        from wbia.algo.detect import randomforest

        print('[ibs] detecting using Random Forests')
        assert config['species'] is not None
        base_key_list[6] = (config['species'],)  # class == species
        detect_gen = randomforest.detect_gid_list_with_species(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['selective-search']:
        from wbia.algo.detect import selectivesearch

        print('[ibs] detecting using Selective Search')
        matlab_command = 'selective_search'
        detect_gen = selectivesearch.detect_gid_list(
            ibs, gid_list, matlab_command=matlab_command, **config
        )
    ######################################################################################
    elif config['algo'] in ['selective-search-rcnn']:
        from wbia.algo.detect import selectivesearch

        print('[ibs] detecting using Selective Search (R-CNN)')
        matlab_command = 'selective_search_rcnn'
        detect_gen = selectivesearch.detect_gid_list(
            ibs, gid_list, matlab_command=matlab_command, **config
        )
    ######################################################################################
    # elif config['algo'] in ['fast-rcnn']:
    #     from wbia.algo.detect import fasterrcnn
    #     print('[ibs] detecting using CNN Fast R-CNN')
    #     detect_gen = fasterrcnn.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['faster-rcnn']:
        from wbia.algo.detect import fasterrcnn

        print('[ibs] detecting using CNN Faster R-CNN')
        detect_gen = fasterrcnn.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['darknet']:
        from wbia.algo.detect import darknet

        print('[ibs] detecting using Darknet CNN YOLO')
        detect_gen = darknet.detect_gid_list(ibs, gid_list, **config)
    ######################################################################################
    elif config['algo'] in ['ssd']:
        from wbia.algo.detect import ssd

        print('[ibs] detecting using CNN SSD')
        detect_gen = ssd.detect_gid_list(ibs, gid_list, **config)
    # ######################################################################################
    elif config['algo'] in ['_COMBINED']:
        # Combined computations
        config_dict_list = [
            # {'algo': 'selective-search', 'config_filepath': None},                          # SS1
            {'algo': 'darknet', 'config_filepath': 'pretrained-tiny-pascal'},  # YOLO1
            {'algo': 'darknet', 'config_filepath': 'pretrained-v2-pascal'},  # YOLO2
            {'algo': 'faster-rcnn', 'config_filepath': 'pretrained-zf-pascal'},  # FRCNN1
            {
                'algo': 'faster-rcnn',
                'config_filepath': 'pretrained-vgg-pascal',
            },  # FRCNN2
            {'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal'},  # SSD1
            {'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal'},  # SSD1
            {'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal-plus'},  # SSD
            {'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal-plus'},  # SSD4
        ]
        detect_gen = _combined(gid_list, config_dict_list)
    else:
        raise ValueError(
            'specified detection algo is not supported in config = %r' % (config,)
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
    print('[ibs] Preprocess Localizations')
    print('config = %r' % (config,))

    VERBOSE = False

    ibs = depc.controller

    zipped = zip(depc.get_native('localizations_original', loc_orig_id_list, None))
    for loc_orig_id, detect in zip(loc_orig_id_list, zipped):
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
                    print(
                        'Filtered with sensitivity = %0.02f (%d -> %d)'
                        % (config['sensitivity'], count_old, count_new,)
                    )

        # Apply NMS
        if config['nms']:
            from wbia.other import detectcore

            count_old = len(bboxes)
            if count_old > 0:

                nms_dict = {}
                nms_class_set = set(classes)
                nms_aware = config['nms_aware']

                for nms_class in nms_class_set:
                    flag_list = classes == nms_class
                    nms_values = {
                        'bboxes': np.compress(flag_list, bboxes, axis=0),
                        'thetas': np.compress(flag_list, thetas, axis=0),
                        'confs': np.compress(flag_list, confs, axis=0),
                        'classes': np.compress(flag_list, classes, axis=0),
                    }

                    if nms_aware in ['byclass']:
                        nms_key = nms_class
                    elif nms_aware in ['ispart']:
                        nms_key = 'part' if '+' in nms_class else 'body'
                    else:
                        nms_key = None

                    if nms_key not in nms_dict:
                        nms_dict[nms_key] = {}

                    for value_key in nms_values:
                        nms_value = nms_values[value_key]
                        if value_key not in nms_dict[nms_key]:
                            nms_dict[nms_key][value_key] = []
                        nms_dict[nms_key][value_key].append(nms_value)

                for nms_key in nms_dict:
                    nms_values = nms_dict[nms_key]

                    nms_bboxes = np.vstack(nms_values['bboxes'])
                    nms_thetas = np.hstack(nms_values['thetas'])
                    nms_confs = np.hstack(nms_values['confs'])
                    nms_classes = np.hstack(nms_values['classes'])

                    nms_values = {
                        'bboxes': nms_bboxes,
                        'thetas': nms_thetas,
                        'confs': nms_confs,
                        'classes': nms_classes,
                    }
                    nms_dict[nms_key] = nms_values

                for nms_key in nms_dict:
                    nms_values = nms_dict[nms_key]

                    nms_bboxes = nms_values['bboxes']
                    nms_thetas = nms_values['thetas']
                    nms_confs = nms_values['confs']
                    nms_classes = nms_values['classes']

                    nms_count_old = len(nms_bboxes)
                    assert nms_count_old > 0

                    coord_list = []
                    for (xtl, ytl, width, height) in nms_bboxes:
                        xbr = xtl + width
                        ybr = ytl + height
                        coord_list.append([xtl, ytl, xbr, ybr])
                    coord_list = np.vstack(coord_list)
                    confs_list = np.array(nms_confs)

                    nms_thresh = 1.0 - config['nms_thresh']
                    keep_indices_list = detectcore.nms(coord_list, confs_list, nms_thresh)
                    keep_list = np.array(keep_indices_list)

                    if len(keep_list) == 0:
                        nms_bboxes = np.array([])
                        nms_thetas = np.array([])
                        nms_confs = np.array([])
                        nms_classes = np.array([])
                    else:
                        nms_bboxes = nms_bboxes[keep_list]
                        nms_thetas = nms_thetas[keep_list]
                        nms_confs = nms_confs[keep_list]
                        nms_classes = nms_classes[keep_list]

                    nms_values = {
                        'bboxes': nms_bboxes,
                        'thetas': nms_thetas,
                        'confs': nms_confs,
                        'classes': nms_classes,
                    }
                    nms_dict[nms_key] = nms_values

                    count_new = len(nms_bboxes)
                    if VERBOSE:
                        nms_args = (
                            nms_key,
                            nms_thresh,
                            nms_count_old,
                            count_new,
                        )
                        print(
                            'Filtered nms_key = %r with nms_thresh = %0.02f (%d -> %d)'
                            % nms_args
                        )

                bboxes = []
                thetas = []
                confs = []
                classes = []

                for nms_key in nms_dict:
                    nms_values = nms_dict[nms_key]
                    nms_bboxes = nms_values['bboxes']
                    nms_thetas = nms_values['thetas']
                    nms_confs = nms_values['confs']
                    nms_classes = nms_values['classes']

                    if len(nms_bboxes) > 0:
                        bboxes.append(nms_bboxes)
                        thetas.append(nms_thetas)
                        confs.append(nms_confs)
                        classes.append(nms_classes)

                bboxes = np.vstack(bboxes)
                thetas = np.hstack(thetas)
                confs = np.hstack(confs)
                classes = np.hstack(classes)

            count_new = len(bboxes)
            if VERBOSE:
                nms_args = (
                    nms_thresh,
                    count_old,
                    count_new,
                )
                print('Filtered with nms_thresh = %0.02f (%d -> %d)' % nms_args)

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
                    print('Filtered invalid images (%d -> %d)' % (count_old, count_new,))

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
        msg = 'Chip shape %r does not agree with target size %r' % (
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
        msg = 'Chip shape %r does not agree with target size %r' % (mask.shape, new_size,)
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
    print(
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
        arg_list = list(zip(gid_list, newsize_list, M_list))
        for tup in ut.ProgIter(arg_list, lbl='computing localization chips', bs=True):
            gid, new_size, M = tup
            if gid != last_gid:
                img = ibs.get_images(gid)
                last_gid = gid
            chip = cv2.warpAffine(img, M[0:2], tuple(new_size), **warpkw)
            # cv2.imshow('', chip)
            # cv2.waitKey()
            msg = 'Chip shape %r does not agree with target size %r' % (
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
    print(
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
        assert len(invalid_bboxes) == 0, 'invalid bboxes=%r' % (invalid_bboxes,)

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
            msg = 'Chip shape %r does not agree with target size %r' % (
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
    print('[ibs] Process Localization Chips')
    print('config = %r' % (config,))
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
        print(
            'Extracting %d localization masks (min: %d, avg: %0.02f, max: %d, total: %d)'
            % args
        )
        worker_func = get_localization_masks_worker
    else:
        print(
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
    print('[ibs] Process Localization Classifications')
    print('config = %r' % (config,))
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
    from PIL import Image
    from keras.preprocessing import image as preprocess_image

    print('[ibs] Preprocess Features')
    print('config = %r' % (config,))
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
            'specified feature algo is not supported in config = %r' % (config,)
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
    from os.path import join, exists

    print('[ibs] Process Localization Labels')
    print('config = %r' % (config,))
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
        cache_path = join(ibs.cachedir, 'localization_labels_%s' % (nonce,))
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
    print('[ibs] Process Localization AoI2s')
    print('config = %r' % (config,))
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
    print('[ibs] Preprocess Detections')
    print('config = %r' % (config,))
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
        #     print('\t%r' % (tup, ))
        # print('----')
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
                print(
                    'Localizer %0.02f %0.02f' % (conf, config['localizer_sensitivity'],)
                )
                print('Labeler   %0.02f %0.02f' % (score, config['labeler_sensitivity'],))
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
        # print(result)
        # raw_input()
        # print('')
        # image = ibs.get_images(gid)
        # image = vt.resize(image, (500, 500))
        # cv2.imshow('', image)
        # cv2.waitKey(0)
        yield tuple(result)


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

    arg_iter = list(zip(gpath_list, orient_list,))
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
        'tessedit_char_whitelist=%s' % (whitelist,),
    ]
    config = ' '.join(config)

    try:
        raw = pytesseract.image_to_string(img, config=config)
    except Exception:
        raw = None

    return raw


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.core_images
        python -m wbia.core_images --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
