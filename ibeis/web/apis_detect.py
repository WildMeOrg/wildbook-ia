# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from ibeis.control import accessor_decors, controller_inject
from ibeis import constants as const
import utool as ut
import simplejson as json
from os.path import join, dirname, abspath
from flask import url_for, request, current_app
from ibeis.constants import KEY_DEFAULTS, SPECIES_KEY
from ibeis.web import appfuncs as appf


try:
    import jpcnn  # NOQA
    USE_LOCALIZATIONS = False
    print('[apis_detect] USING DETECTIONS FOR DETECTIONS')
except ImportError:
    USE_LOCALIZATIONS = True
    print('[apis_detect] USING LOCALIZATIONS FOR DETECTIONS')


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_ibs_method
@accessor_decors.default_decorator
@accessor_decors.getter_1to1
@register_api('/api/detect/randomforest/', methods=['PUT', 'GET'])
def detect_random_forest(ibs, gid_list, species, commit=True, **kwargs):
    """
    Runs animal detection in each image. Adds annotations to the database
    as they are found.

    Args:
        gid_list (list): list of image ids to run detection on
        species (str): string text of the species to identify

    Returns:
        aids_list (list): list of lists of annotation ids detected in each
            image

    CommandLine:
        python -m ibeis.web.apis_detect --test-detect_random_forest --show

    RESTful:
        Method: PUT, GET
        URL:    /api/detect/randomforest/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.web.apis_detect import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> species = ibeis.const.TEST_SPECIES.ZEB_PLAIN
        >>> # execute function
        >>> aids_list = ibs.detect_random_forest(gid_list, species)
        >>> # Visualize results
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     from ibeis.viz import viz_image
        >>>     for fnum, gid in enumerate(gid_list):
        >>>         viz_image.show_image(ibs, gid, fnum=fnum)
        >>>     pt.show_if_requested()
        >>> # Remove newly detected annotations
        >>> ibs.delete_annots(ut.flatten(aids_list))
    """
    # TODO: Return confidence here as well
    depc = ibs.depc_image
    config = {
        'algo'                   : 'pyrf',
        'species'                : species,
        'sensitivity'            : 0.2,
        # 'classifier_sensitivity' : 0.64,
        # 'localizer_grid'         : False,
        # 'localizer_sensitivity'  : 0.16,
        # 'labeler_sensitivity'    : 0.42,
        # 'detector_sensitivity'   : 0.08,
    }
    if USE_LOCALIZATIONS:
        results_list = depc.get_property('localizations', gid_list, None, config=config)
        if commit:
            aids_list = ibs.commit_localization_results(gid_list, results_list, note='pyrfdetect')
            return aids_list
    else:
        results_list = depc.get_property('detections', gid_list, None, config=config)
        if commit:
            aids_list = ibs.commit_detection_results(gid_list, results_list, note='pyrfdetect')
            return aids_list


@register_route('/test/review/detect/cnn/yolo/', methods=['GET'])
def review_detection_test():
    ibs = current_app.ibs
    results_dict = ibs.detection_yolo_test()
    image_uuid = results_dict['image_uuid_list'][0]
    result_list = results_dict['results_list'][0]
    callback_url = request.args.get('callback_url', url_for('process_detection_html'))
    callback_method = request.args.get('callback_method', 'POST')
    template_html = review_detection_html(ibs, image_uuid, result_list, callback_url, callback_method, include_jquery=True)
    template_html = '''
        <script src="http://code.jquery.com/jquery-2.2.1.min.js" ia-dependency="javascript"></script>
        %s
    ''' % (template_html, )
    return template_html


@register_ibs_method
@register_api('/test/detect/cnn/yolo/', methods=['GET'])
def detection_yolo_test(ibs):
    from random import shuffle
    gid_list = ibs.get_valid_gids()
    shuffle(gid_list)
    gid_list = gid_list[:3]
    results_dict = ibs.detect_cnn_yolo_json(gid_list)
    return results_dict


@register_api('/api/review/detect/cnn/yolo/', methods=['GET'])
def review_detection_html(ibs, image_uuid, result_list, callback_url, callback_method='POST', include_jquery=False):
    """
    Returns the detection review interface for a particular image UUID and a list of
    results for that image.

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

    gpath = ibs.get_image_thumbpath(gid, ensure_paths=True, draw_annots=False)
    image = ibs.get_image_imgdata(gid)
    image_src = appf.embed_image_html(image)
    width, height = ibs.get_image_sizes(gid)

    if width <= 0 or width is None or height <= 0 or height is None:
        vals = (image_uuid, width, height, )
        raise IOError('Image %r for review has either no width or no height (w = %s, h = %s)' % vals)

    annotation_list = []
    for result in result_list:
        annotation_list.append({
            'left'   : 100.0 * (result['xtl'] / width),
            'top'    : 100.0 * (result['ytl'] / height),
            'width'  : 100.0 * (result['width'] / width),
            'height' : 100.0 * (result['height'] / height),
            'label'  : result['class'],
            'id'     : None,
            'theta'  : result.get('theta', 0.0),
        })

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
        json_file_list = [
            ['javascript', 'jquery.min.js'],
        ] + json_file_list

    EMBEDDED_CSS = ''
    EMBEDDED_JAVASCRIPT = ''

    css_template_fmtstr = '<style type="text/css" ia-dependency="css">%s</style>\n'
    json_template_fmtstr = '<script type="text/javascript" ia-dependency="javascript">%s</script>\n'
    for css_file in css_file_list:
        css_filepath_list = [root_path, 'static'] + css_file
        with open(join(*css_filepath_list)) as css_file:
            EMBEDDED_CSS += css_template_fmtstr % (css_file.read(), )

    for json_file in json_file_list:
        json_filepath_list = [root_path, 'static'] + json_file
        with open(join(*json_filepath_list)) as json_file:
            EMBEDDED_JAVASCRIPT += json_template_fmtstr % (json_file.read(), )

    return appf.template('turk', 'detection_insert',
                         gid=gid,
                         refer_aid=None,
                         species=species,
                         image_path=gpath,
                         image_src=image_src,
                         annotation_list=annotation_list,
                         callback_url=callback_url,
                         callback_method=callback_method,
                         EMBEDDED_CSS=EMBEDDED_CSS,
                         EMBEDDED_JAVASCRIPT=EMBEDDED_JAVASCRIPT)


@register_api('/api/review/detect/cnn/yolo/', methods=['POST'])
def process_detection_html(ibs, **kwargs):
    """
    Processes the return from the detection review interface.  Pass the POST
    result from the detection review form directly to this function unmodified

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
    annotation_list = json.loads(request.form['detection-annotations'])
    result_list = [
        {
            'xtl'        : int( width  * (annot['left']   / 100.0) ),
            'ytl'        : int( height * (annot['top']    / 100.0) ),
            'width'      : int( width  * (annot['width']  / 100.0) ),
            'height'     : int( height * (annot['height'] / 100.0) ),
            'theta'      : float(annot['theta']),
            'confidence' : 1.0,
            'class'      : annot['label'],
        }
        for annot in annotation_list
    ]
    result_dict = {
        'image_uuid_list' : [image_uuid],
        'results_list'    : [result_list],
        'score_list'      : [1.0],
    }
    return result_dict


@register_ibs_method
@accessor_decors.default_decorator
@accessor_decors.getter_1to1
def detect_cnn_yolo_json(ibs, gid_list, **kwargs):
    """
    Runs animal detection in each image and returns json-ready formatted
        results, does not return annotations

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        results_dict (list): dict of detection results (not annotations)

    CommandLine:
        python -m ibeis.web.apis_detect --test-detect_cnn_yolo_json

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.web.apis_detect import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> # execute function
        >>> results_dict = ibs.detect_cnn_yolo_json(gid_list)
        >>> print(results_dict)
    """
    # TODO: Return confidence here as well
    image_uuid_list = ibs.get_image_uuids(gid_list)
    ibs.assert_valid_gids(gid_list)
    # Get detections from depc
    aids_list = ibs.detect_cnn_yolo(gid_list, **kwargs)
    results_list = [
        [
            {
                'xtl'        : ibs.get_annot_bboxes(aid)[0],
                'ytl'        : ibs.get_annot_bboxes(aid)[1],
                'width'      : ibs.get_annot_bboxes(aid)[2],
                'height'     : ibs.get_annot_bboxes(aid)[3],
                'theta'      : round(ibs.get_annot_thetas(aid), 4),
                'confidence' : round(ibs.get_annot_detect_confidence(aid), 4),
                'class'      : ibs.get_annot_species_texts(aid),
            }
            for aid in aid_list
        ]
        for aid_list in aids_list
    ]
    score_list = [0.0] * len(gid_list)
    # Wrap up results with other information
    results_dict = {
        'image_uuid_list' : image_uuid_list,
        'results_list'    : results_list,
        'score_list'      : score_list,
    }
    return results_dict


@register_ibs_method
@accessor_decors.default_decorator
@accessor_decors.getter_1toM
@register_api('/api/detect/cnn/yolo/', methods=['PUT', 'GET'])
def detect_cnn_yolo(ibs, gid_list, commit=True, testing=False, **kwargs):
    """
    Runs animal detection in each image. Adds annotations to the database
    as they are found.

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        aids_list (list): list of lists of annotation ids detected in each
            image

    CommandLine:
        python -m ibeis.web.apis_detect --test-detect_cnn_yolo --show

    RESTful:
        Method: PUT, GET
        URL:    /api/detect/cnn/yolo/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.web.apis_detect import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> gid_list = ibs.get_valid_gids()[:5]
        >>> # execute function
        >>> aids_list = ibs.detect_cnn_yolo(gid_list)
        >>> # Visualize results
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     from ibeis.viz import viz_image
        >>>     for fnum, gid in enumerate(gid_list):
        >>>         viz_image.show_image(ibs, gid, fnum=fnum)
        >>>     pt.show_if_requested()
        >>> # Remove newly detected annotations
        >>> ibs.delete_annots(ut.flatten(aids_list))
    """
    # TODO: Return confidence here as well
    depc = ibs.depc_image
    config = {
        'algo'                   : 'yolo',
        'sensitivity'            : 0.2,
        # 'classifier_sensitivity' : 0.64,
        # 'localizer_grid'         : False,
        # 'localizer_sensitivity'  : 0.16,
        # 'labeler_sensitivity'    : 0.42,
        # 'detector_sensitivity'   : 0.08,
    }
    if USE_LOCALIZATIONS:
        if testing:
            depc.delete_property('localizations', gid_list, config=config)
        results_list = depc.get_property('localizations', gid_list, None, config=config)
        if commit:
            aids_list = ibs.commit_localization_results(gid_list, results_list, note='cnnyolodetect')
            return aids_list
    else:
        if testing:
            depc.delete_property('detections', gid_list, config=config)
        results_list = depc.get_property('detections', gid_list, None, config=config)
        if commit:
            aids_list = ibs.commit_detection_results(gid_list, results_list, note='cnnyolodetect')
            return aids_list


@register_ibs_method
@accessor_decors.default_decorator
@accessor_decors.getter_1to1
@register_api('/api/detect/cnn/yolo/exists/', methods=['GET'], __api_plural_check__=False)
def detect_cnn_yolo_exists(ibs, gid_list, testing=False):
    """
    Checks to see if a detection has been completed.

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        flag_list (list): list of flags for if the detection has been run on
            the image

    CommandLine:
        python -m ibeis.web.apis_detect --test-detect_cnn_yolo_exists

    RESTful:
        Method: GET
        URL:    /api/detect/cnn/yolo/exists/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.web.apis_detect import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('PZ_MTEST')
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
        'algo'                   : 'yolo',
        'sensitivity'            : 0.2,
        # 'classifier_sensitivity' : 0.64,
        # 'localizer_grid'         : False,
        # 'localizer_sensitivity'  : 0.16,
        # 'labeler_sensitivity'    : 0.42,
        # 'detector_sensitivity'   : 0.08,
    }
    if USE_LOCALIZATIONS:
        score_list = depc.get_property('localizations', gid_list, 'score', ensure=False, config=config)
    else:
        score_list = depc.get_property('detections', gid_list, 'score', ensure=False, config=config)
    flag_list = [ score is not None for score in score_list ]
    return flag_list


@register_ibs_method
def commit_localization_results(ibs, gid_list, results_list, note=None):
    zipped_list = zip(gid_list, results_list)
    aids_list = []
    for gid, (score, bbox_list, theta_list, conf_list, class_list) in zipped_list:
        num = len(bbox_list)
        notes_list = None if note is None else [note] * num
        aid_list = ibs.add_annots(
            [gid] * num,
            bbox_list,
            theta_list,
            class_list,
            detect_confidence_list=conf_list,
            notes_list=notes_list,
            quiet_delete_thumbs=True,
            skip_cleaning=True
        )
        # ibs.set_annot_yaw_texts(aid_list, viewpoint_list)
        aids_list.append(aid_list)
    ibs._clean_species()
    return aids_list


@register_ibs_method
def commit_detection_results(ibs, gid_list, results_list, note=None):
    zipped_list = zip(gid_list, results_list)
    aids_list = []
    for gid, (score, bbox_list, theta_list, species_list, viewpoint_list, conf_list) in zipped_list:
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
            skip_cleaning=True
        )
        ibs.set_annot_yaw_texts(aid_list, viewpoint_list)
        aids_list.append(aid_list)
    ibs._clean_species()
    return aids_list


@register_ibs_method
def commit_detection_results_filtered(ibs, gid_list, filter_species_list=None, filter_viewpoint_list=None, note=None):
    depc = ibs.depc_image
    results_list = depc.get_property('detections', gid_list, None)
    zipped_list = zip(gid_list, results_list)
    aids_list = []
    for gid, (score, bbox_list, theta_list, species_list, viewpoint_list, conf_list) in zipped_list:
        aid_list = []
        result_list = zip(bbox_list, theta_list, species_list, viewpoint_list, conf_list)
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
                skip_cleaning=True
            )
            aid = temp_list[0]
            ibs.set_annot_yaw_texts([aid], [viewpoint])
            aid_list.append(aid)
        aids_list.append(aid_list)
    ibs._clean_species()
    return aids_list


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/detect/species/enabled/', methods=['GET'], __api_plural_check__=False)
def has_species_detector(ibs, species_text):
    """
    TODO: extend to use non-constant species

    RESTful:
        Method: GET
        URL:    /api/detect/species/enabled/
    """
    # FIXME: infer this
    return species_text in const.SPECIES_WITH_DETECTORS


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/detect/species/', methods=['GET'], __api_plural_check__=False)
def get_species_with_detectors(ibs):
    """
    RESTful:
        Method: GET
        URL:    /api/detect/species/
    """
    # FIXME: infer this
    return const.SPECIES_WITH_DETECTORS


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/detect/species/working/', methods=['GET'], __api_plural_check__=False)
def get_working_species(ibs):
    """
    RESTful:
        Method: GET
        URL:    /api/detect/species/working/
    """
    RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS = ut.get_argflag('--no-allspecies')

    species_nice_list = ibs.get_all_species_nice()
    species_text_list = ibs.get_all_species_texts()
    species_tup_list = zip(species_nice_list, species_text_list)
    if RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS:
        working_species_tups = [
            species_tup
            for species_tup in species_tup_list
            if ibs.has_species_detector(species_tup[1])
        ]
    else:
        working_species_tups = species_tup_list
    return working_species_tups

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
