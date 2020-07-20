# -*- coding: utf-8 -*-
"""Dependencies: flask, tornado."""
from __future__ import absolute_import, division, print_function
from wbia.control import accessor_decors, controller_inject
from wbia import constants as const
import utool as ut
import simplejson as json
from os.path import join, dirname, abspath, exists
from flask import url_for, request, current_app
from wbia.constants import KEY_DEFAULTS, SPECIES_KEY
from wbia.web import appfuncs as appf
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)

CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/wic/cnn/', methods=['PUT', 'GET', 'POST'])
def wic_cnn(ibs, gid_list, testing=False, algo='cnn', model_tag='candidacy', **kwargs):
    depc = ibs.depc_image
    config = {}

    if model_tag is not None:
        config['classifier_two_algo'] = algo
        config['classifier_two_weight_filepath'] = model_tag

    if testing:
        depc.delete_property('classifier_two', gid_list, config=config)

    result_list = depc.get_property('classifier_two', gid_list, None, config=config)

    output_list = []
    for result in result_list:
        scores, classes = result
        output_list.append(scores)

    return output_list


@register_ibs_method
@accessor_decors.getter_1to1
def wic_cnn_json(ibs, gid_list, config={}, **kwargs):
    return wic_cnn(ibs, gid_list, **config)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/detect/randomforest/', methods=['PUT', 'GET'])
def detect_random_forest(ibs, gid_list, species, commit=True, **kwargs):
    """Run animal detection in each image. Adds annotations to the database as they are found.

    Args:
        gid_list (list): list of image ids to run detection on
        species (str): string text of the species to identify

    Returns:
        aids_list (list): list of lists of annotation ids detected in each
            image

    CommandLine:
        python -m wbia.web.apis_detect --test-detect_random_forest --show

    RESTful:
        Method: PUT, GET
        URL:    /api/detect/randomforest/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_detect import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> species = wbia.const.TEST_SPECIES.ZEB_PLAIN
        >>> aids_list = ibs.detect_random_forest(gid_list, species)
        >>> # Visualize results
        >>> if ut.show_was_requested():
        >>>     import wbia.plottool as pt
        >>>     from wbia.viz import viz_image
        >>>     for fnum, gid in enumerate(gid_list):
        >>>         viz_image.show_image(ibs, gid, fnum=fnum)
        >>>     pt.show_if_requested()
        >>> # Remove newly detected annotations
        >>> ibs.delete_annots(ut.flatten(aids_list))
    """
    # TODO: Return confidence here as well
    depc = ibs.depc_image
    config = {
        'algo': 'rf',
        'species': species,
        'sensitivity': 0.2,
        'nms': True,
        'nms_thresh': 0.4,
    }
    results_list = depc.get_property('localizations', gid_list, None, config=config)
    if commit:
        aids_list = ibs.commit_localization_results(
            gid_list, results_list, note='pyrfdetect'
        )
        return aids_list

    # results_list = depc.get_property('detections', gid_list, None, config=config)
    # if commit:
    #     aids_list = ibs.commit_detection_results(gid_list, results_list, note='pyrfdetect')
    #     return aids_list


@register_route('/test/review/detect/cnn/yolo/', methods=['GET'])
def review_detection_test(
    image_uuid=None,
    result_list=None,
    callback_url=None,
    callback_method='POST',
    **kwargs,
):
    ibs = current_app.ibs
    if image_uuid is None or result_list is None:
        results_dict = ibs.detection_yolo_test()
        image_uuid = results_dict['image_uuid_list'][0]
        result_list = results_dict['results_list'][0]
    if callback_url is None:
        callback_url = request.args.get('callback_url', url_for('process_detection_html'))
    if callback_method is None:
        callback_method = request.args.get('callback_method', 'POST')
    template_html = review_detection_html(
        ibs, image_uuid, result_list, callback_url, callback_method, include_jquery=True
    )
    template_html = """
        <script src="http://code.jquery.com/jquery-2.2.1.min.js" ia-dependency="javascript"></script>
        %s
    """ % (
        template_html,
    )
    return template_html


@register_ibs_method
@register_api('/test/detect/cnn/yolo/', methods=['GET'])
def detection_yolo_test(ibs, config={}):
    from random import shuffle  # NOQA

    gid_list = ibs.get_valid_gids()
    shuffle(gid_list)
    gid_list = gid_list[:3]
    results_dict = ibs.detect_cnn_yolo_json(gid_list, config=config)
    return results_dict


@register_ibs_method
@register_api('/test/detect/cnn/lightnet/', methods=['GET'])
def detection_lightnet_test(ibs, config={}):
    from random import shuffle  # NOQA

    gid_list = ibs.get_valid_gids()
    shuffle(gid_list)
    gid_list = gid_list[:3]
    results_dict = ibs.detect_cnn_lightnet_json(gid_list, config=config)
    return results_dict


@register_api('/api/review/detect/cnn/yolo/', methods=['GET'])
def review_detection_html(
    ibs,
    image_uuid,
    result_list,
    callback_url,
    callback_method='POST',
    include_jquery=False,
    config=None,
):
    """
    Return the detection review interface for a particular image UUID and a list of results for that image.

    Args:
        image_uuid (UUID): the UUID of the image you want to review detections for
        result_list (list of dict): list of detection results returned by the detector
        callback_url (str): URL that the review form will submit to (action) when
            the user is complete with their review
        callback_method (str): HTTP method the review form will submit to (method).
            Defaults to 'POST'

    Returns:
        template (html): json response with the detection web interface in html

    RESTful:
        Method: GET
        URL:    /api/review/detect/cnn/yolo/
    """
    ibs.web_check_uuids(image_uuid_list=[image_uuid])
    gid = ibs.get_image_gids_from_uuid(image_uuid)

    if gid is None:
        return 'INVALID IMAGE UUID'

    default_config = {
        'autointerest': False,
        'interest_bypass': False,
        'metadata': True,
        'metadata_viewpoint': False,
        'metadata_quality': False,
        'metadata_flags': True,
        'metadata_flags_aoi': True,
        'metadata_flags_multiple': False,
        'metadata_species': True,
        'metadata_label': True,
        'metadata_quickhelp': True,
        'parts': False,
        'modes_rectangle': True,
        'modes_diagonal': True,
        'modes_diagonal2': True,
        'staged': False,
    }

    if config is not None:
        default_config.update(config)

    gpath = ibs.get_image_thumbpath(gid, ensure_paths=True, draw_annots=False)
    image = ibs.get_images(gid)
    image_src = appf.embed_image_html(image)
    width, height = ibs.get_image_sizes(gid)

    if width <= 0 or width is None or height <= 0 or height is None:
        vals = (
            image_uuid,
            width,
            height,
        )
        raise IOError(
            'Image %r for review has either no width or no height (w = %s, h = %s)' % vals
        )

    annotation_list = []
    for result in result_list:
        quality = result.get('quality', None)
        if quality in [-1, None]:
            quality = 0
        elif quality <= 2:
            quality = 1
        elif quality > 2:
            quality = 2

        viewpoint1 = result.get('viewpoint1', None)
        viewpoint2 = result.get('viewpoint2', None)
        viewpoint3 = result.get('viewpoint3', None)

        if viewpoint1 is None and viewpoint2 is None and viewpoint3 is None:
            viewpoint = result.get('viewpoint', None)
            viewpoint1, viewpoint2, viewpoint3 = appf.convert_viewpoint_to_tuple(
                viewpoint
            )

        annotation_list.append(
            {
                'id': result.get('id', None),
                'left': 100.0 * (result.get('left', result['xtl']) / width),
                'top': 100.0 * (result.get('top', result['ytl']) / height),
                'width': 100.0 * (result['width'] / width),
                'height': 100.0 * (result['height'] / height),
                'species': result.get('species', result['class']),
                'theta': result.get('theta', 0.0),
                'viewpoint1': viewpoint1,
                'viewpoint2': viewpoint2,
                'viewpoint3': viewpoint3,
                'quality': quality,
                'multiple': 'true' if result.get('multiple', None) == 1 else 'false',
                'interest': 'true' if result.get('interest', None) == 1 else 'false',
            }
        )

    species = KEY_DEFAULTS[SPECIES_KEY]

    root_path = dirname(abspath(__file__))
    css_file_list = [
        ['include', 'jquery-ui', 'jquery-ui.min.css'],
        ['include', 'jquery.ui.rotatable', 'jquery.ui.rotatable.css'],
        ['css', 'style.css'],
    ]
    json_file_list = [
        ['include', 'jquery-ui', 'jquery-ui.min.js'],
        ['include', 'jquery.ui.rotatable', 'jquery.ui.rotatable.min.js'],
        ['include', 'bbox_annotator_percent.js'],
        ['javascript', 'script.js'],
        ['javascript', 'turk-detection.js'],
    ]

    if include_jquery:
        json_file_list = [['javascript', 'jquery.min.js']] + json_file_list

    EMBEDDED_CSS = ''
    EMBEDDED_JAVASCRIPT = ''

    css_template_fmtstr = '<style type="text/css" ia-dependency="css">%s</style>\n'
    json_template_fmtstr = (
        '<script type="text/javascript" ia-dependency="javascript">%s</script>\n'
    )
    for css_file in css_file_list:
        css_filepath_list = [root_path, 'static'] + css_file
        with open(join(*css_filepath_list)) as css_file:
            EMBEDDED_CSS += css_template_fmtstr % (css_file.read(),)

    for json_file in json_file_list:
        json_filepath_list = [root_path, 'static'] + json_file
        with open(join(*json_filepath_list)) as json_file:
            EMBEDDED_JAVASCRIPT += json_template_fmtstr % (json_file.read(),)

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [combined[0] for combined in combined_list]
    species_rowids = [combined[1] for combined in combined_list]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_list = list(zip(species_nice_list, species_text_list))
    species_list = [('Unspecified', const.UNKNOWN)] + species_list

    # Collect mapping of species to parts
    aid_list = ibs.get_valid_aids()
    part_species_rowid_list = ibs.get_annot_species_rowids(aid_list)
    part_species_text_list = ibs.get_species_texts(part_species_rowid_list)
    part_rowids_list = ibs.get_annot_part_rowids(aid_list)
    part_types_list = map(ibs.get_part_types, part_rowids_list)

    zipped = list(zip(part_species_text_list, part_types_list))
    species_part_dict = {const.UNKNOWN: set([])}
    for part_species_text, part_type_list in zipped:
        if part_species_text not in species_part_dict:
            species_part_dict[part_species_text] = set([const.UNKNOWN])
        for part_type in part_type_list:
            species_part_dict[part_species_text].add(part_type)
            species_part_dict[const.UNKNOWN].add(part_type)
    # Add any images that did not get added because they aren't assigned any annotations
    for species_text in species_text_list:
        if species_text not in species_part_dict:
            species_part_dict[species_text] = set([const.UNKNOWN])
    for key in species_part_dict:
        species_part_dict[key] = sorted(list(species_part_dict[key]))
    species_part_dict_json = json.dumps(species_part_dict)

    orientation_flag = '0'
    if species is not None and 'zebra' in species:
        orientation_flag = '1'

    settings_key_list = [
        ('ia-detection-setting-orientation', orientation_flag),
        ('ia-detection-setting-parts-assignments', '1'),
        ('ia-detection-setting-toggle-annotations', '1'),
        ('ia-detection-setting-toggle-parts', '0'),
        ('ia-detection-setting-parts-show', '0'),
        ('ia-detection-setting-parts-hide', '0'),
    ]

    settings = {
        settings_key: request.cookies.get(settings_key, settings_default) == '1'
        for (settings_key, settings_default) in settings_key_list
    }

    return appf.template(
        'turk',
        'detection_insert',
        gid=gid,
        refer_aid=None,
        species=species,
        image_path=gpath,
        image_src=image_src,
        config=default_config,
        settings=settings,
        annotation_list=annotation_list,
        species_list=species_list,
        species_part_dict_json=species_part_dict_json,
        callback_url=callback_url,
        callback_method=callback_method,
        EMBEDDED_CSS=EMBEDDED_CSS,
        EMBEDDED_JAVASCRIPT=EMBEDDED_JAVASCRIPT,
    )


@register_api('/api/review/detect/cnn/yolo/', methods=['POST'])
def process_detection_html(ibs, **kwargs):
    """
    Process the return from the detection review interface.  Pass the POST result from the detection review form directly to this function unmodified.

    Returns:
        detection results (dict): Same format as `func:start_detect_image`

    RESTful:
        Method: POST
        URL:    /api/review/detect/cnn/yolo/
    """
    gid = int(request.form['detection-gid'])
    image_uuid = ibs.get_image_uuids(gid)
    width, height = ibs.get_image_sizes(gid)
    # Get aids
    annotation_list = json.loads(request.form['ia-detection-data'])

    viewpoint1_list = [
        int(annot['metadata'].get('viewpoint1', -1)) for annot in annotation_list
    ]
    viewpoint2_list = [
        int(annot['metadata'].get('viewpoint2', -1)) for annot in annotation_list
    ]
    viewpoint3_list = [
        int(annot['metadata'].get('viewpoint3', -1)) for annot in annotation_list
    ]
    zipped = list(zip(viewpoint1_list, viewpoint2_list, viewpoint3_list))
    viewpoint_list = [appf.convert_tuple_to_viewpoint(tup) for tup in zipped]

    result_list = [
        {
            'id': annot['label'],
            'xtl': int(width * (annot['percent']['left'] / 100.0)),
            'ytl': int(height * (annot['percent']['top'] / 100.0)),
            'left': int(width * (annot['percent']['left'] / 100.0)),
            'top': int(height * (annot['percent']['top'] / 100.0)),
            'width': int(width * (annot['percent']['width'] / 100.0)),
            'height': int(height * (annot['percent']['height'] / 100.0)),
            'theta': float(annot['angles']['theta']),
            'confidence': 1.0,
            'class': annot['label'],
            'species': annot['label'],
            'viewpoint': viewpoint,
            'quality': annot['metadata']['quality'],
            'multiple': annot['metadata']['multiple'],
            'interest': annot['highlighted'],
        }
        for annot, viewpoint in list(zip(annotation_list, viewpoint_list))
    ]
    result_dict = {
        'image_uuid_list': [image_uuid],
        'results_list': [result_list],
        'score_list': [1.0],
    }
    return result_dict


@register_ibs_method
@accessor_decors.getter_1to1
def detect_cnn_json(ibs, gid_list, detect_func, config={}, **kwargs):
    """
    Run animal detection in each image and returns json-ready formatted results, does not return annotations.

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        results_dict (list): dict of detection results (not annotations)

    CommandLine:
        python -m wbia.web.apis_detect --test-detect_cnn_yolo_json

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_detect import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> results_dict = ibs.detect_cnn_yolo_json(gid_list)
        >>> print(results_dict)
    """
    # TODO: Return confidence here as well
    image_uuid_list = ibs.get_image_uuids(gid_list)
    ibs.assert_valid_gids(gid_list)
    # Get detections from depc
    aids_list = detect_func(gid_list, **config)
    results_list = [
        [
            {
                'id': aid,
                'uuid': ibs.get_annot_uuids(aid),
                'xtl': ibs.get_annot_bboxes(aid)[0],
                'ytl': ibs.get_annot_bboxes(aid)[1],
                'left': ibs.get_annot_bboxes(aid)[0],
                'top': ibs.get_annot_bboxes(aid)[1],
                'width': ibs.get_annot_bboxes(aid)[2],
                'height': ibs.get_annot_bboxes(aid)[3],
                'theta': round(ibs.get_annot_thetas(aid), 4),
                'confidence': round(ibs.get_annot_detect_confidence(aid), 4),
                'class': ibs.get_annot_species_texts(aid),
                'species': ibs.get_annot_species_texts(aid),
                'viewpoint': ibs.get_annot_viewpoints(aid),
                'quality': ibs.get_annot_qualities(aid),
                'multiple': ibs.get_annot_multiple(aid),
                'interest': ibs.get_annot_interest(aid),
            }
            for aid in aid_list
        ]
        for aid_list in aids_list
    ]
    score_list = [0.0] * len(gid_list)
    # Wrap up results with other information
    results_dict = {
        'image_uuid_list': image_uuid_list,
        'results_list': results_list,
        'score_list': score_list,
    }
    return results_dict


@register_ibs_method
def detect_cnn_json_wrapper(ibs, image_uuid_list, detect_func, **kwargs):
    """
    Detect with CNN (general).

    REST:
        Method: GET
        URL: /api/detect/cnn/yolo/json/

    Args:
        image_uuid_list (list) : list of image uuids to detect on.
    """
    from wbia.web.apis_engine import ensure_uuid_list

    # Check UUIDs
    ibs.web_check_uuids(image_uuid_list=image_uuid_list)
    image_uuid_list = ensure_uuid_list(image_uuid_list)
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return detect_func(gid_list, **kwargs)


@register_ibs_method
@register_api('/api/detect/cnn/yolo/json/', methods=['POST'])
def detect_cnn_yolo_json_wrapper(ibs, image_uuid_list, **kwargs):
    return detect_cnn_json_wrapper(
        ibs, image_uuid_list, ibs.detect_cnn_yolo_json, **kwargs
    )


@register_ibs_method
@accessor_decors.getter_1to1
def detect_cnn_yolo_json(ibs, gid_list, config={}, **kwargs):
    return detect_cnn_json(ibs, gid_list, ibs.detect_cnn_yolo, config=config, **kwargs)


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/detect/cnn/yolo/', methods=['PUT', 'GET', 'POST'])
def detect_cnn_yolo(ibs, gid_list, model_tag=None, commit=True, testing=False, **kwargs):
    """
    Run animal detection in each image. Adds annotations to the database as they are found.

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        aids_list (list): list of lists of annotation ids detected in each
            image

    CommandLine:
        python -m wbia.web.apis_detect --test-detect_cnn_yolo --show

    RESTful:
        Method: PUT, GET
        URL:    /api/detect/cnn/yolo/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_detect import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> gid_list = ibs.get_valid_gids()[:5]
        >>> aids_list = ibs.detect_cnn_yolo(gid_list)
        >>> if ut.show_was_requested():
        >>>     import wbia.plottool as pt
        >>>     from wbia.viz import viz_image
        >>>     for fnum, gid in enumerate(gid_list):
        >>>         viz_image.show_image(ibs, gid, fnum=fnum)
        >>>     pt.show_if_requested()
        >>> # Remove newly detected annotations
        >>> ibs.delete_annots(ut.flatten(aids_list))
    """
    # TODO: Return confidence here as well
    depc = ibs.depc_image
    config = {
        'grid': False,
        'algo': 'yolo',
        'sensitivity': 0.2,
        'nms': True,
        'nms_thresh': 0.4,
    }
    if model_tag is not None:
        config['config_filepath'] = model_tag
        config['weight_filepath'] = model_tag

    config_str_list = ['config_filepath', 'weight_filepath'] + list(config.keys())
    for config_str in config_str_list:
        if config_str in kwargs:
            config[config_str] = kwargs[config_str]

    if testing:
        depc.delete_property('localizations', gid_list, config=config)

    results_list = depc.get_property('localizations', gid_list, None, config=config)

    if commit:
        aids_list = ibs.commit_localization_results(
            gid_list, results_list, note='cnnyolodetect', **kwargs
        )
        return aids_list
    else:
        return results_list

    # results_list = depc.get_property('detections', gid_list, None, config=config)
    # if commit:
    #     aids_list = ibs.commit_detection_results(gid_list, results_list, note='cnnyolodetect')
    #     return aids_list


@register_ibs_method
@register_api(
    '/api/models/cnn/lightnet/',
    methods=['PUT', 'GET', 'POST'],
    __api_plural_check__=False,
)
def models_cnn_lightnet(ibs, **kwargs):
    """
    Return the models (and their labels) for the YOLO CNN detector

    RESTful:
        Method: PUT, GET
        URL:    /api/labels/cnn/lightnet/
    """

    def identity(x):
        return x

    from wbia.algo.detect.lightnet import CONFIG_URL_DICT, _parse_class_list

    model_dict = ibs.models_cnn(CONFIG_URL_DICT, identity, _parse_class_list, **kwargs)
    return model_dict


@register_ibs_method
@register_api(
    '/api/models/cnn/yolo/', methods=['PUT', 'GET', 'POST'], __api_plural_check__=False
)
def models_cnn_yolo(ibs, **kwargs):
    """
    Return the models (and their labels) for the YOLO CNN detector

    RESTful:
        Method: PUT, GET
        URL:    /api/labels/cnn/yolo/
    """
    from pydarknet._pydarknet import (
        CONFIG_URL_DICT,
        _parse_classes_from_cfg,
        _parse_class_list,
    )

    model_dict = ibs.models_cnn(
        CONFIG_URL_DICT, _parse_classes_from_cfg, _parse_class_list, **kwargs
    )
    return model_dict


@register_ibs_method
def models_cnn(
    ibs,
    config_dict,
    parse_classes_func,
    parse_line_func,
    check_hash=False,
    hidden_models=[],
    **kwargs,
):
    import urllib

    model_dict = {}
    for config_tag in config_dict:
        if config_tag in hidden_models:
            continue

        try:
            config_url = config_dict[config_tag]
            classes_url = parse_classes_func(config_url)
            try:
                classes_filepath = ut.grab_file_url(
                    classes_url, appname='wbia', check_hash=check_hash
                )
                assert exists(classes_filepath)
            except (urllib.error.HTTPError, AssertionError):
                continue

            classes_filepath = ut.truepath(classes_filepath)
            line_list = parse_line_func(classes_filepath)
            model_dict[config_tag] = line_list
        except Exception:
            pass

    return model_dict


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/labeler/cnn/', methods=['PUT', 'GET', 'POST'])
def labeler_cnn(
    ibs, aid_list, testing=False, algo='pipeline', model_tag='candidacy', **kwargs
):
    depc = ibs.depc_annot
    config = {}

    if algo is not None:
        config['labeler_algo'] = algo
    if model_tag is not None:
        config['labeler_weight_filepath'] = model_tag

    if testing:
        depc.delete_property('labeler', aid_list, config=config)

    result_list = depc.get_property('labeler', aid_list, None, config=config)

    output_list = []
    for result in result_list:
        score, species, viewpoint, quality, orientation, probs = result
        output_list.append({'score': score, 'species': species, 'viewpoint': viewpoint})

    return output_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/aoi/cnn/', methods=['PUT', 'GET', 'POST'])
def aoi_cnn(ibs, aid_list, testing=False, model_tag='candidacy', **kwargs):
    depc = ibs.depc_annot
    config = {}

    if model_tag is not None:
        config['aoi_two_weight_filepath'] = model_tag

    if testing:
        depc.delete_property('aoi_two', aid_list, config=config)

    result_list = depc.get_property('aoi_two', aid_list, None, config=config)

    output_list = []
    for result in result_list:
        score, class_ = result
        output_list.append({'score': score, 'class': class_})

    return output_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/detect/cnn/yolo/exists/', methods=['GET'], __api_plural_check__=False)
def detect_cnn_yolo_exists(ibs, gid_list, testing=False):
    """
    Check to see if a detection has been completed.

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        flag_list (list): list of flags for if the detection has been run on
            the image

    CommandLine:
        python -m wbia.web.apis_detect --test-detect_cnn_yolo_exists

    RESTful:
        Method: GET
        URL:    /api/detect/cnn/yolo/exists/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_detect import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> gid_list = ibs.get_valid_gids()
        >>> depc = ibs.depc_image
        >>> aids_list = ibs.detect_cnn_yolo(gid_list[:3], testing=True)
        >>> result = ibs.detect_cnn_yolo_exists(gid_list[:5])
        >>> ibs.delete_annots(ut.flatten(aids_list))
        >>> print(result)
        [True, True, True, False, False]
    """
    depc = ibs.depc_image
    config = {
        'algo': 'yolo',
        'sensitivity': 0.2,
        'nms': True,
        'nms_thresh': 0.4,
    }
    score_list = depc.get_property(
        'localizations', gid_list, 'score', ensure=False, config=config
    )
    # score_list = depc.get_property('detections', gid_list, 'score', ensure=False, config=config)
    flag_list = [score is not None for score in score_list]
    return flag_list


@register_ibs_method
@register_api('/api/detect/cnn/lightnet/json/', methods=['POST'])
def detect_cnn_lightnet_json_wrapper(ibs, image_uuid_list, **kwargs):
    return detect_cnn_json_wrapper(
        ibs, image_uuid_list, ibs.detect_cnn_lightnet_json, **kwargs
    )


@register_ibs_method
@accessor_decors.getter_1to1
def detect_cnn_lightnet_json(ibs, gid_list, config={}, **kwargs):
    return detect_cnn_json(
        ibs, gid_list, ibs.detect_cnn_lightnet, config=config, **kwargs
    )


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/detect/cnn/lightnet/', methods=['PUT', 'GET', 'POST'])
def detect_cnn_lightnet(
    ibs, gid_list, model_tag=None, commit=True, testing=False, **kwargs
):
    """
    Run animal detection in each image. Adds annotations to the database as they are found.

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        aids_list (list): list of lists of annotation ids detected in each
            image

    CommandLine:
        python -m wbia.web.apis_detect --test-detect_cnn_lightnet --show

    RESTful:
        Method: PUT, GET
        URL:    /api/detect/cnn/lightnet/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_detect import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> gid_list = ibs.get_valid_gids()[:5]
        >>> aids_list = ibs.detect_cnn_lightnet(gid_list)
        >>> if ut.show_was_requested():
        >>>     import wbia.plottool as pt
        >>>     from wbia.viz import viz_image
        >>>     for fnum, gid in enumerate(gid_list):
        >>>         viz_image.show_image(ibs, gid, fnum=fnum)
        >>>     pt.show_if_requested()
        >>> # Remove newly detected annotations
        >>> ibs.delete_annots(ut.flatten(aids_list))
    """
    # TODO: Return confidence here as well
    depc = ibs.depc_image
    config = {
        'algo': 'lightnet',
        'sensitivity': 0.75,
        'nms': True,
        'nms_thresh': 0.4,
        'nms_aware': None,
    }

    if model_tag is not None:
        config['config_filepath'] = model_tag
        config['weight_filepath'] = model_tag

    config_str_list = ['config_filepath', 'weight_filepath'] + list(config.keys())
    for config_str in config_str_list:
        if config_str in kwargs:
            config[config_str] = kwargs[config_str]

    if testing:
        depc.delete_property('localizations', gid_list, config=config)

    results_list = depc.get_property('localizations', gid_list, None, config=config)

    if commit:
        aids_list = ibs.commit_localization_results(
            gid_list, results_list, note='cnnlightnetdetect', **kwargs
        )
        return aids_list
    else:
        return results_list


@register_ibs_method
def commit_localization_results(
    ibs,
    gid_list,
    results_list,
    note=None,
    labeler_algo='pipeline',
    labeler_model_tag=None,
    use_labeler_species=False,
    orienter_algo=None,
    orienter_model_tag=None,
    update_json_log=True,
    **kwargs,
):
    global_gid_list = []
    global_bbox_list = []
    global_theta_list = []
    global_class_list = []
    global_conf_list = []
    global_notes_list = []

    zipped_list = list(zip(gid_list, results_list))

    for gid, results in zipped_list:
        score, bbox_list, theta_list, conf_list, class_list = results
        num = len(bbox_list)
        gid_list_ = [gid] * num
        notes_list = [note] * num

        global_gid_list += list(gid_list_)
        global_bbox_list += list(bbox_list)
        global_theta_list += list(theta_list)
        global_class_list += list(class_list)
        global_conf_list += list(conf_list)
        global_notes_list += list(notes_list)

    assert len(global_gid_list) == len(global_bbox_list)
    assert len(global_gid_list) == len(global_theta_list)
    assert len(global_gid_list) == len(global_class_list)
    assert len(global_gid_list) == len(global_conf_list)
    assert len(global_gid_list) == len(global_notes_list)

    global_aid_list = ibs.add_annots(
        global_gid_list,
        global_bbox_list,
        global_theta_list,
        global_class_list,
        detect_confidence_list=global_conf_list,
        notes_list=global_notes_list,
        quiet_delete_thumbs=True,
        skip_cleaning=True,
    )

    global_aid_set = set(global_aid_list)
    aids_list = ibs.get_image_aids(gid_list)
    aids_list = [
        [aid for aid in aid_list_ if aid in global_aid_set] for aid_list_ in aids_list
    ]
    aid_list = ut.flatten(aids_list)

    if labeler_model_tag is not None:
        labeler_config = {}
        labeler_config['labeler_algo'] = labeler_algo
        labeler_config['labeler_weight_filepath'] = labeler_model_tag
        viewpoint_list = ibs.depc_annot.get_property(
            'labeler', aid_list, 'viewpoint', config=labeler_config
        )
        ibs.set_annot_viewpoints(aid_list, viewpoint_list)
        if use_labeler_species:
            species_list = ibs.depc_annot.get_property(
                'labeler', aid_list, 'species', config=labeler_config
            )
            ibs.set_annot_species(aid_list, species_list)

    if orienter_algo is not None:
        orienter_config = {}
        orienter_config['orienter_algo'] = orienter_algo
        orienter_config['orienter_weight_filepath'] = orienter_model_tag
        result_list = ibs.depc_annot.get_property(
            'orienter', aid_list, None, config=orienter_config
        )
        xtl_list = list(map(int, map(np.around, ut.take_column(result_list, 0))))
        ytl_list = list(map(int, map(np.around, ut.take_column(result_list, 1))))
        w_list = list(map(int, map(np.around, ut.take_column(result_list, 2))))
        h_list = list(map(int, map(np.around, ut.take_column(result_list, 3))))
        theta_list = ut.take_column(result_list, 4)
        bbox_list = list(zip(xtl_list, ytl_list, w_list, h_list))
        assert len(aid_list) == len(bbox_list)
        assert len(aid_list) == len(theta_list)
        if len(bbox_list) > 0:
            ibs.set_annot_bboxes(aid_list, bbox_list, theta_list=theta_list)

    ibs._clean_species()
    if update_json_log:
        ibs.log_detections(aid_list)

    return aids_list


@register_ibs_method
def commit_detection_results(
    ibs, gid_list, results_list, note=None, update_json_log=True
):
    zipped_list = list(zip(gid_list, results_list))
    aids_list = []
    for (
        gid,
        (score, bbox_list, theta_list, species_list, viewpoint_list, conf_list),
    ) in zipped_list:
        num = len(bbox_list)
        notes_list = None if note is None else [note] * num
        aid_list = ibs.add_annots(
            [gid] * num,
            bbox_list,
            theta_list,
            species_list,
            detect_confidence_list=conf_list,
            notes_list=notes_list,
            quiet_delete_thumbs=True,
            skip_cleaning=True,
        )
        ibs.set_annot_viewpoints(aid_list, viewpoint_list)
        # TODO ibs.set_annot_viewpoint_code(aid_list, viewpoint_list)
        aids_list.append(aid_list)
    ibs._clean_species()
    if update_json_log:
        aid_list = ut.flatten(aids_list)
        ibs.log_detections(aid_list)
    return aids_list


@register_ibs_method
def commit_detection_results_filtered(
    ibs,
    gid_list,
    filter_species_list=None,
    filter_viewpoint_list=None,
    note=None,
    update_json_log=True,
):
    depc = ibs.depc_image
    results_list = depc.get_property('detections', gid_list, None)
    zipped_list = list(zip(gid_list, results_list))
    aids_list = []
    for (
        gid,
        (score, bbox_list, theta_list, species_list, viewpoint_list, conf_list),
    ) in zipped_list:
        aid_list = []
        result_list = list(
            zip(bbox_list, theta_list, species_list, viewpoint_list, conf_list)
        )
        for bbox, theta, species, viewpoint, conf in result_list:
            if not (filter_species_list is None or species in filter_species_list):
                continue
            if not (filter_viewpoint_list is None or viewpoint in filter_viewpoint_list):
                continue
            note_ = None if note is None else [note]
            temp_list = ibs.add_annots(
                [gid],
                [bbox],
                [theta],
                [species],
                detect_confidence_list=[conf],
                notes_list=note_,
                quiet_delete_thumbs=True,
                skip_cleaning=True,
            )
            aid = temp_list[0]
            ibs.set_annot_viewpoints([aid], [viewpoint])
            # TODO ibs.set_annot_viewpoint_code([aid], [viewpoint])
            aid_list.append(aid)
        aids_list.append(aid_list)
    ibs._clean_species()
    if update_json_log:
        aid_list = ut.flatten(aids_list)
        ibs.log_detections(aid_list)
    return aids_list


@register_ibs_method
def log_detections(ibs, aid_list, fallback=True):
    import time
    import os

    json_log_path = ibs.get_logdir_local()
    json_log_filename = 'detections.json'
    json_log_filepath = os.path.join(json_log_path, json_log_filename)
    print('Logging detections added to: %r' % (json_log_filepath,))

    try:
        # Log has never been made, create one
        if not os.path.exists(json_log_filepath):
            json_dict = {
                'updates': [],
            }
            json_str = ut.to_json(json_dict, pretty=True)
            with open(json_log_filepath, 'w') as json_log_file:
                json_log_file.write(json_str)
        # Get current log state
        with open(json_log_filepath, 'r') as json_log_file:
            json_str = json_log_file.read()
        json_dict = ut.from_json(json_str)
        # Get values
        db_name = ibs.get_db_name()
        db_init_uuid = ibs.get_db_init_uuid()
        # Zip all the updates together and write to updates list in dictionary
        gid_list = ibs.get_annot_gids(aid_list)
        bbox_list = ibs.get_annot_bboxes(aid_list)
        theta_list = ibs.get_annot_thetas(aid_list)
        zipped = list(zip(aid_list, gid_list, bbox_list, theta_list))
        for aid, gid, bbox, theta in zipped:
            json_dict['updates'].append(
                {
                    'time_unixtime': time.time(),
                    'db_name': db_name,
                    'db_init_uuid': db_init_uuid,
                    'image_rowid': gid,
                    'annot_rowid': aid,
                    'annot_bbox': bbox,
                    'annot_theta': theta,
                }
            )
        # Write new log state
        json_str = ut.to_json(json_dict, pretty=True)
        with open(json_log_filepath, 'w') as json_log_file:
            json_log_file.write(json_str)
    except Exception:
        if fallback:
            print('WRITE DETECTION.JSON FAILED - ATTEMPTING FALLBACK')
            ut.delete(json_log_filepath)
            ibs.log_detections(aid_list, fallback=False)
        else:
            print('WRITE DETECTION.JSON FAILED - FALLBACK FAILED')


@register_ibs_method
@register_api('/api/detect/species/enabled/', methods=['GET'], __api_plural_check__=False)
def has_species_detector(ibs, species_text):
    """
    TODO: extend to use non-constant species.

    RESTful:
        Method: GET
        URL:    /api/detect/species/enabled/
    """
    # FIXME: infer this
    return species_text in const.SPECIES_WITH_DETECTORS


@register_ibs_method
@register_api('/api/detect/species/', methods=['GET'], __api_plural_check__=False)
def get_species_with_detectors(ibs):
    """
    Get valid species for detection.

    RESTful:
        Method: GET
        URL:    /api/detect/species/
    """
    # FIXME: infer this
    return const.SPECIES_WITH_DETECTORS


@register_ibs_method
@register_api('/api/detect/species/working/', methods=['GET'], __api_plural_check__=False)
def get_working_species(ibs):
    """
    Get working species for detection.

    RESTful:
        Method: GET
        URL:    /api/detect/species/working/
    """
    RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS = ut.get_argflag('--no-allspecies')

    species_nice_list = ibs.get_all_species_nice()
    species_text_list = ibs.get_all_species_texts()
    species_tup_list = list(zip(species_nice_list, species_text_list))
    if RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS:
        working_species_tups = [
            species_tup
            for species_tup in species_tup_list
            if ibs.has_species_detector(species_tup[1])
        ]
    else:
        working_species_tups = species_tup_list
    return working_species_tups


@register_ibs_method
@register_api('/api/detect/whaleSharkInjury/', methods=['PUT', 'GET'])
def detect_ws_injury(ibs, gid_list):
    """
    Classify if a whale shark is injured.

    Args:
        gid_list (list): list of image ids to run classification on

    Returns:
        result_list (dictionary): predictions is list of strings representing a possible tag.
            confidences is a list of floats of correspoinding cofidence to the prediction

    """
    from wbia.scripts import labelShark

    labels = labelShark.classifyShark(ibs, gid_list)
    return labels


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.app
        python -m wbia.web.app --allexamples
        python -m wbia.web.app --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
