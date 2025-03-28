# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado

SeeAlso:
    routes.review_identification
"""
import logging
import traceback
import uuid
from datetime import datetime
from os.path import abspath, dirname, exists, join

import cv2
import numpy as np  # NOQA
import requests
import utool as ut
from flask import current_app, request, url_for  # NOQA

from wbia import constants as const
from wbia.algo.hots import pipeline
from wbia.control import controller_inject
from wbia.web import appfuncs as appf

# (print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)


GRAPH_CLIENT_PEEK = 100


RENDER_STATUS = None


@register_ibs_method
@register_api('/api/query/annot/rowid/', methods=['GET'])
def get_recognition_query_aids(ibs, is_known, species=None):
    """
    DEPCIRATE

    RESTful:
        Method: GET
        URL:    /api/query/annot/rowid/
    """
    qaid_list = ibs.get_valid_aids(is_known=is_known, species=species)
    return qaid_list


@register_ibs_method
@register_api('/api/query/chip/dict/simple/', methods=['GET'])
def query_chips_simple_dict(ibs, *args, **kwargs):
    r"""
    Runs query_chips, but returns a json compatible dictionary

    Args:
        same as query_chips

    RESTful:
        Method: GET
        URL:    /api/query/chip/dict/simple/

    SeeAlso:
        query_chips

    CommandLine:
        python -m wbia.web.apis_query --test-query_chips_simple_dict:0
        python -m wbia.web.apis_query --test-query_chips_simple_dict:1

        python -m wbia.web.apis_query --test-query_chips_simple_dict:0 --humpbacks

    Example:
        >>> # xdoctest: +REQUIRES(--web-tests)
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> #qaid = ibs.get_valid_aids()[0:3]
        >>> qaids = ibs.get_valid_aids()
        >>> daids = ibs.get_valid_aids()
        >>> dict_list = ibs.query_chips_simple_dict(qaids, daids)
        >>> qgids = ibs.get_annot_image_rowids(qaids)
        >>> qnids = ibs.get_annot_name_rowids(qaids)
        >>> for dict_, qgid, qnid in list(zip(dict_list, qgids, qnids)):
        >>>     dict_['qgid'] = qgid
        >>>     dict_['qnid'] = qnid
        >>>     dict_['dgid_list'] = ibs.get_annot_image_rowids(dict_['daid_list'])
        >>>     dict_['dnid_list'] = ibs.get_annot_name_rowids(dict_['daid_list'])
        >>>     dict_['dgname_list'] = ibs.get_image_gnames(dict_['dgid_list'])
        >>>     dict_['qgname'] = ibs.get_image_gnames(dict_['qgid'])
        >>> result  = ut.repr2(dict_list, nl=2, precision=2, hack_liststr=True)
        >>> result = result.replace('u\'', '"').replace('\'', '"')
        >>> print(result)

    Example:
        >>> # xdoctest: +SKIP
        >>> # FIXME failing-test (04-Aug-2020) This test hangs when running together with the test above
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia
        >>> # Start up the web instance
        >>> with wbia.opendb_bg_web(db='testdb1', managed=True) as web_ibs:
        ...     cmdict_list = web_ibs.send_wbia_request('/api/query/chip/dict/simple/', type_='get', qaid_list=[1], daid_list=[1, 2, 3])
        >>> print(cmdict_list)
        >>> assert 'score_list' in cmdict_list[0]

    """
    kwargs['return_cm_simple_dict'] = True
    return ibs.query_chips(*args, **kwargs)


@register_ibs_method
@register_api('/api/query/chip/dict/', methods=['GET'])
def query_chips_dict(ibs, *args, **kwargs):
    """
    Runs query_chips, but returns a json compatible dictionary

    RESTful:
        Method: GET
        URL:    /api/query/chip/dict/
    """
    kwargs['return_cm_dict'] = True
    return ibs.query_chips(*args, **kwargs)


@register_ibs_method
@register_api('/api/review/query/graph/', methods=['POST'])
def process_graph_match_html(ibs, **kwargs):
    """
    RESTful:
        Method: POST
        URL:    /api/review/query/graph/
    """

    def sanitize(state):
        state = state.strip().lower()
        state = ''.join(state.split())
        return state

    import uuid

    map_dict = {
        'sameanimal': const.EVIDENCE_DECISION.INT_TO_CODE[
            const.EVIDENCE_DECISION.POSITIVE
        ],
        'differentanimals': const.EVIDENCE_DECISION.INT_TO_CODE[
            const.EVIDENCE_DECISION.NEGATIVE
        ],
        'cannottell': const.EVIDENCE_DECISION.INT_TO_CODE[
            const.EVIDENCE_DECISION.INCOMPARABLE
        ],
        'unreviewed': const.EVIDENCE_DECISION.INT_TO_CODE[
            const.EVIDENCE_DECISION.UNREVIEWED
        ],
        'unknown': const.EVIDENCE_DECISION.INT_TO_CODE[const.EVIDENCE_DECISION.UNKNOWN],
        'photobomb': 'photobomb',
        'scenerymatch': 'scenerymatch',
        'excludetop': 'excludetop',
        'excludebottom': 'excludebottom',
        'excludeleft': 'excludeleft',
        'excluderight': 'excluderight',
    }
    annot_uuid_1 = uuid.UUID(request.form['identification-annot-uuid-1'])
    annot_uuid_2 = uuid.UUID(request.form['identification-annot-uuid-2'])
    state = request.form.get('identification-submit', '')
    state = sanitize(state)
    state = map_dict[state]
    tag_list = []
    if state in ['photobomb', 'scenerymatch']:
        tag_list.append(state)
        state = const.EVIDENCE_DECISION.NEGATIVE
    assert state in map_dict.values(), 'matching state is unrecognized'
    # Get checbox tags
    checbox_tag_list = ['photobomb', 'scenerymatch']
    for checbox_tag in checbox_tag_list:
        checkbox_name = 'ia-%s-value' % (checbox_tag)
        if checkbox_name in request.form:
            tag_list.append(checbox_tag)
    tag_list = sorted(set(tag_list))
    confidence_default = const.CONFIDENCE.INT_TO_CODE[const.CONFIDENCE.UNKNOWN]
    confidence = request.form.get('ia-review-confidence', confidence_default)
    if confidence not in const.CONFIDENCE.CODE_TO_INT.keys():
        confidence = confidence_default
    if len(tag_list) == 0:
        tag_str = ''
    else:
        tag_str = ';'.join(tag_list)
    user_times = {
        'server_time_start': request.form.get('server_time_start', None),
        'client_time_start': request.form.get('client_time_start', None),
        'client_time_end': request.form.get('client_time_end', None),
    }
    return (
        annot_uuid_1,
        annot_uuid_2,
        state,
        tag_str,
        'web-api',
        confidence,
        user_times,
    )


def ensure_review_image(
    ibs,
    aid,
    cm,
    qreq_,
    view_orientation='vertical',
    draw_matches=True,
    draw_heatmask=False,
    verbose=False,
    image=None,
    use_gradcam=False,
):
    r""" "
    Create the review image for a pair of annotations

    CommandLine:
        python -m wbia.web.apis_query ensure_review_image --show

    Example:
        >>> # SCRIPT
        >>> from wbia.web.apis_query import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm('PZ_MTEST', a='default:dindex=0:10,qindex=0:1')
        >>> ibs = qreq_.ibs
        >>> aid = cm.get_top_aids()[0]
        >>> tt = ut.tic('make image')
        >>> image, _ = ensure_review_image(ibs, aid, cm, qreq_)
        >>> ut.toc(tt)
        >>> ut.quit_if_noshow()
        >>> print('image.shape = %r' % (image.shape,))
        >>> print('image.dtype = %r' % (image.dtype,))
        >>> ut.print_object_size(image)
        >>> import wbia.plottool as pt
        >>> pt.imshow(image)
        >>> ut.show_if_requested()
    """
    from wbia.gui import id_review_api

    # Get thumb path
    match_thumb_path = ibs.get_match_thumbdir()
    match_thumb_filename = id_review_api.get_match_thumb_fname(
        cm,
        aid,
        qreq_,
        view_orientation=view_orientation,
        draw_matches=draw_matches,
        draw_heatmask=draw_heatmask,
    )
    match_thumb_filepath = join(match_thumb_path, match_thumb_filename)
    if verbose:
        logger.info('Checking: {!r}'.format(match_thumb_filepath))

    if exists(match_thumb_filepath):
        image = cv2.imread(match_thumb_filepath)
    else:
        render_config = {
            'dpi': 300,
            'overlay': True,
            'draw_fmatches': True,
            'draw_fmatch': draw_matches,
            'show_matches': draw_matches,
            'show_ell': draw_matches,
            'show_lines': draw_matches,
            'show_ori': False,
            'heatmask': draw_heatmask,
            'vert': view_orientation == 'vertical',
            'show_aidstr': False,
            'show_name': False,
            'show_exemplar': False,
            'show_num_gt': False,
            'show_timedelta': False,
            'show_name_rank': False,
            'show_score': False,
            'show_annot_score': False,
            'show_name_score': False,
            'draw_lbl': False,
            'draw_border': False,
            'white_background': True,
            'use_gradcam': use_gradcam,
        }

        if image is not None:
            pass
        elif hasattr(qreq_, 'render_single_result'):
            image = qreq_.render_single_result(cm, aid, **render_config)
        elif hasattr(cm, 'render_single_annotmatch'):
            image = cm.render_single_annotmatch(qreq_, aid, **render_config)
        else:
            image = ibs.get_annot_chips(aid)
        # image = vt.crop_out_imgfill(image, fillval=(255, 255, 255), thresh=64)
        cv2.imwrite(match_thumb_filepath, image)
    return image, match_thumb_filepath


@register_api(
    '/api/review/query/graph/alias/', methods=['POST'], __api_plural_check__=False
)
def review_graph_match_html_alias(*args, **kwargs):
    review_graph_match_html(*args, **kwargs)


@register_api('/api/review/query/graph/', methods=['GET'])
def review_graph_match_html(
    ibs,
    review_pair,
    cm_dict,
    query_config_dict,
    _internal_state,
    callback_url,
    callback_method='POST',
    view_orientation='vertical',
    include_jquery=False,
):
    r"""
    Args:
        ibs (wbia.IBEISController):  image analysis api
        review_pair (dict): pair of annot uuids
        cm_dict (dict):
        query_config_dict (dict):
        _internal_state (?):
        callback_url (str):
        callback_method (unicode): (default = u'POST')
        view_orientation (unicode): (default = u'vertical')
        include_jquery (bool): (default = False)

    CommandLine:
        python -m wbia.web.apis_query review_graph_match_html --show

        wbia --web
        python -m wbia.web.apis_query review_graph_match_html --show --domain=localhost

    Example:
        >>> # xdoctest: +REQUIRES(--web-tests)
        >>> # xdoctest: +REQUIRES(--job-engine-tests)
        >>> # DISABLE_DOCTEST
        >>> # Disabled because this test uses opendb_bg_web, which hangs the test runner and leaves zombie processes
        >>> from wbia.web.apis_query import *  # NOQA
        >>> import wbia
        >>> web_ibs = wbia.opendb_bg_web('testdb1')  # , domain='http://52.33.105.88')
        >>> aids = web_ibs.send_wbia_request('/api/annot/', 'get')[0:2]
        >>> uuid_list = web_ibs.send_wbia_request('/api/annot/uuid/', type_='get', aid_list=aids)
        >>> quuid_list = uuid_list[0:1]
        >>> duuid_list = uuid_list
        >>> query_config_dict = {
        >>>    # 'pipeline_root' : 'BC_DTW'
        >>> }
        >>> data = dict(
        >>>     query_annot_uuid_list=quuid_list, database_annot_uuid_list=duuid_list,
        >>>     query_config_dict=query_config_dict,
        >>> )
        >>> jobid = web_ibs.send_wbia_request('/api/engine/query/graph/', **data)
        >>> print('jobid = %r' % (jobid,))
        >>> status_response = web_ibs.wait_for_results(jobid)
        >>> result_response = web_ibs.read_engine_results(jobid)
        >>> inference_result = result_response['json_result']
        >>> print('inference_result = %r' % (inference_result,))
        >>> auuid2_cm = inference_result['cm_dict']
        >>> quuid = quuid_list[0]
        >>> class_dict = auuid2_cm[str(quuid)]
        >>> # Get information in frontend
        >>> #ibs = wbia.opendb('testdb1')
        >>> #cm = match_obj = wbia.ChipMatch.from_dict(class_dict, ibs=ibs)
        >>> #match_obj.print_rawinfostr()
        >>> # Make the dictionary a bit more managable
        >>> #match_obj.compress_top_feature_matches(num=2)
        >>> #class_dict = match_obj.to_dict(ibs=ibs)
        >>> cm_dict = class_dict
        >>> # Package for review
        >>> review_pair = {'annot_uuid_1': quuid, 'annot_uuid_2': duuid_list[1]}
        >>> callback_method = u'POST'
        >>> view_orientation = u'vertical'
        >>> include_jquery = False
        >>> kw = dict(
        >>>     review_pair=review_pair,
        >>>     cm_dict=cm_dict,
        >>>     query_config_dict=query_config_dict,
        >>>     _internal_state=None,
        >>>     callback_url = None,
        >>> )
        >>> html_str = web_ibs.send_wbia_request('/api/review/query/graph/', type_='get', **kw)
        >>> web_ibs.terminate2()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.render_html(html_str)
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(--job-engine-tests)
        >>> # This starts off using web to get information, but finishes the rest in python
        >>> from wbia.web.apis_query import *  # NOQA
        >>> import wbia
        >>> ut.exec_funckw(review_graph_match_html, globals())
        >>> web_ibs = wbia.opendb_bg_web('testdb1')  # , domain='http://52.33.105.88')
        >>> aids = web_ibs.send_wbia_request('/api/annot/', 'get')[0:2]
        >>> uuid_list = web_ibs.send_wbia_request('/api/annot/uuid/', type_='get', aid_list=aids)
        >>> quuid_list = uuid_list[0:1]
        >>> duuid_list = uuid_list
        >>> query_config_dict = {
        >>>    # 'pipeline_root' : 'BC_DTW'
        >>> }
        >>> data = dict(
        >>>     query_annot_uuid_list=quuid_list, database_annot_uuid_list=duuid_list,
        >>>     query_config_dict=query_config_dict,
        >>> )
        >>> jobid = web_ibs.send_wbia_request('/api/engine/query/graph/', **data)
        >>> status_response = web_ibs.wait_for_results(jobid)
        >>> result_response = web_ibs.read_engine_results(jobid)
        >>> web_ibs.terminate2()
        >>> # NOW WORK IN THE FRONTEND
        >>> inference_result = result_response['json_result']
        >>> auuid2_cm = inference_result['cm_dict']
        >>> quuid = quuid_list[0]
        >>> class_dict = auuid2_cm[str(quuid)]
        >>> # Get information in frontend
        >>> ibs = wbia.opendb('testdb1')
        >>> cm = wbia.ChipMatch.from_dict(class_dict, ibs=ibs)
        >>> cm.print_rawinfostr()
        >>> # Make the dictionary a bit more managable
        >>> cm.compress_top_feature_matches(num=1)
        >>> cm.print_rawinfostr()
        >>> class_dict = cm.to_dict(ibs=ibs)
        >>> cm_dict = class_dict
        >>> # Package for review ( CANT CALL DIRECTLY BECAUSE OF OUT OF CONTEXT )
        >>> review_pair = {'annot_uuid_1': quuid, 'annot_uuid_2': duuid_list[1]}
        >>> x = review_graph_match_html(ibs, review_pair, cm_dict,
        >>>                             query_config_dict, _internal_state=None,
        >>>                             callback_url=None)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.render_html(html_str)
        >>> ut.show_if_requested()
    """
    from wbia.algo.hots import chip_match

    # from wbia.algo.hots.query_request import QueryRequest

    proot = query_config_dict.get('pipeline_root', 'vsmany')
    proot = query_config_dict.get('proot', proot)
    if proot.lower() in (
        'bc_dtw',
        'oc_wdtw',
        'curvrankdorsal',
        'curvrankfinfindrhybriddorsal',
        'curvrankfluke',
        'curvranktwodorsal',
        'curvranktwofluke',
        'deepsense',
        'finfindr',
        'kaggle7',
        'kaggleseven',
        'pie',
        'pietwo',
        'miewid',
        'whaleridgefindr'
    ):
        cls = chip_match.AnnotMatch  # ibs.depc_annot.requestclass_dict['BC_DTW']
    else:
        cls = chip_match.ChipMatch

    view_orientation = view_orientation.lower()
    if view_orientation not in ['vertical', 'horizontal']:
        view_orientation = 'horizontal'

    # unpack info
    try:
        annot_uuid_1 = review_pair['annot_uuid_1']
        annot_uuid_2 = review_pair['annot_uuid_2']
    except Exception:
        # ??? HACK
        # FIXME:
        logger.info('[!!!!] review_pair = {!r}'.format(review_pair))
        review_pair = review_pair[0]
        annot_uuid_1 = review_pair['annot_uuid_1']
        annot_uuid_2 = review_pair['annot_uuid_2']

    ibs.web_check_uuids(qannot_uuid_list=[annot_uuid_1], dannot_uuid_list=[annot_uuid_2])

    aid_1 = ibs.get_annot_aids_from_uuid(annot_uuid_1)
    aid_2 = ibs.get_annot_aids_from_uuid(annot_uuid_2)

    cm = cls.from_dict(cm_dict, ibs=ibs)
    qreq_ = ibs.new_query_request([aid_1], [aid_2], cfgdict=query_config_dict)

    # Get score
    idx = cm.daid2_idx[aid_2]
    match_score = cm.name_score_list[idx]
    # match_score = cm.aid2_score[aid_2]

    try:
        image_matches, _ = ensure_review_image(
            ibs, aid_2, cm, qreq_, view_orientation=view_orientation
        )
    except KeyError:
        image_matches = np.zeros((100, 100, 3), dtype=np.uint8)
        traceback.print_exc()
    try:
        image_clean, _ = ensure_review_image(
            ibs, aid_2, cm, qreq_, view_orientation=view_orientation, draw_matches=False
        )
    except KeyError:
        image_clean = np.zeros((100, 100, 3), dtype=np.uint8)
        traceback.print_exc()

    image_matches_src = appf.embed_image_html(image_matches)
    image_clean_src = appf.embed_image_html(image_clean)

    confidence_dict = const.CONFIDENCE.NICE_TO_CODE
    confidence_nice_list = confidence_dict.keys()
    confidence_text_list = confidence_dict.values()
    confidence_selected_list = [
        confidence_text == 'unspecified' for confidence_text in confidence_text_list
    ]
    confidence_list = list(
        zip(confidence_nice_list, confidence_text_list, confidence_selected_list)
    )

    if False:
        from wbia.web import apis_query

        root_path = dirname(abspath(apis_query.__file__))
    else:
        root_path = dirname(abspath(__file__))
    css_file_list = [
        ['css', 'style.css'],
        ['include', 'bootstrap', 'css', 'bootstrap.css'],
    ]
    json_file_list = [
        ['javascript', 'script.js'],
        ['include', 'bootstrap', 'js', 'bootstrap.js'],
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

    annot_uuid_1 = str(annot_uuid_1)
    annot_uuid_2 = str(annot_uuid_2)

    embedded = dict(globals(), **locals())
    return appf.template('review', 'identification_insert', **embedded)


@register_route('/test/review/query/chip/', methods=['GET'])
def review_query_chips_test(**kwargs):
    ibs = current_app.ibs

    # the old block curvature dtw
    if 'use_bc_dtw' in request.args:
        query_config_dict = {'pipeline_root': 'BC_DTW'}
    # the new oriented curvature dtw
    elif 'use_oc_wdtw' in request.args:
        query_config_dict = {'pipeline_root': 'OC_WDTW'}
    elif 'use_curvrank_dorsal' in request.args:
        query_config_dict = {'pipeline_root': 'CurvRankDorsal'}
    elif 'use_curvrank_finfindr_hybrid_dorsal' in request.args:
        query_config_dict = {'pipeline_root': 'CurvRankFinfindrHybridDorsal'}
    elif 'use_curvrank_fluke' in request.args:
        query_config_dict = {'pipeline_root': 'CurvRankFluke'}
    elif 'use_curvrank_v2_dorsal' in request.args:
        query_config_dict = {'pipeline_root': 'CurvRankTwoDorsal'}
    elif 'use_curvrank_v2_fluke' in request.args:
        query_config_dict = {'pipeline_root': 'CurvRankTwoFluke'}
    elif 'use_deepsense' in request.args:
        query_config_dict = {'pipeline_root': 'Deepsense'}
    elif 'use_finfindr' in request.args:
        query_config_dict = {'pipeline_root': 'Finfindr'}
    elif 'use_kaggle7' in request.args or 'use_kaggleseven' in request.args:
        query_config_dict = {'pipeline_root': 'KaggleSeven'}
    elif 'use_pie' in request.args:
        query_config_dict = {'pipeline_root': 'Pie'}
    elif 'use_pie_v2' in request.args:
        query_config_dict = {'pipeline_root': 'PieTwo'}
    elif 'use_miew_id' in request.args:
        query_config_dict = {'pipeline_root': 'MiewId'}
    elif 'use_whaleridgefindr' in request.args:
        query_config_dict = {'pipeline_root': 'whaleridgefindr'}
    else:
        query_config_dict = {}
    result_dict = ibs.query_chips_test(query_config_dict=query_config_dict)

    review_pair = result_dict['inference_dict']['annot_pair_dict']['review_pair_list'][0]
    annot_uuid_key = str(review_pair['annot_uuid_key'])
    cm_dict = result_dict['cm_dict'][annot_uuid_key]
    query_config_dict = result_dict['query_config_dict']
    _internal_state = result_dict['inference_dict']['_internal_state']
    callback_url = request.args.get('callback_url', url_for('process_graph_match_html'))
    callback_method = request.args.get('callback_method', 'POST')
    # view_orientation = request.args.get('view_orientation', 'vertical')
    view_orientation = request.args.get('view_orientation', 'horizontal')

    template_html = review_graph_match_html(
        ibs,
        review_pair,
        cm_dict,
        query_config_dict,
        _internal_state,
        callback_url,
        callback_method,
        view_orientation,
        include_jquery=True,
    )
    template_html = """
        <script src="http://code.jquery.com/jquery-2.2.1.min.js" ia-dependency="javascript"></script>
        {}
    """.format(
        template_html,
    )
    return template_html
    return 'done'


@register_ibs_method
@register_api('/api/review/query/chip/best/', methods=['GET'])
def review_query_chips_best(ibs, aid, database_imgsetid=None, **kwargs):
    # from wbia.algo.hots import chip_match

    # query_config_dict = {}

    # Compile test data
    qaid_list = [aid]
    if database_imgsetid is not None:
        all_imgsetids = set(ibs.get_valid_imgsetids())
        if database_imgsetid not in all_imgsetids:
            database_imgsetid = None

    if database_imgsetid is None:
        daid_list = ibs.get_valid_aids()
    else:
        daid_list = ibs.get_imageset_aids(database_imgsetid)

    result_dict = ibs.query_chips_graph(qaid_list, daid_list, **kwargs)

    uuid = list(result_dict['cm_dict'].keys())[0]
    result = result_dict['cm_dict'][uuid]

    scores = result['name_score_list']
    names = result['unique_name_list']

    if len(scores) == 0:
        response = {
            'name': None,
            'sex': None,
            'age': None,
            'extern': {
                'reference': None,
                'qannot_uuid': None,
                'dannot_uuid': None,
            },
        }
    else:
        best_index = np.argmax(scores)
        best_name = names[best_index]
        best_nid = ibs.get_name_rowids_from_text(best_name)
        best_sex = ibs.get_name_sex_text(best_nid)
        best_age_list = ibs.get_name_age_months_est_min(best_nid)
        best_age_list = [
            int(best_age)
            for best_age in best_age_list
            if best_age is not None and isinstance(best_age, int)
        ]
        best_age = min(best_age_list)

        if best_age in [0, 3, 6]:
            best_age_str = 'Newborn'
        elif best_age in [12]:
            best_age_str = 'Yearling'
        elif best_age in [24]:
            best_age_str = '2-Year-Old'
        elif best_age in [36]:
            best_age_str = 'Adult'
        else:
            best_age_str = 'Unknown'

        # ut.embed()
        dannot_score_list = result['annot_score_list']
        dannot_uuid_list = result['dannot_uuid_list']
        dannot_flag_list = result['dannot_extern_list']
        daid_list = ibs.get_annot_aids_from_uuid(dannot_uuid_list)
        zipped = zip(dannot_score_list, daid_list, dannot_uuid_list, dannot_flag_list)
        zipped = sorted(list(zipped), reverse=True)

        aid_set = set(ibs.get_name_aids(best_nid))
        dannot_uuid = None
        for item in zipped:
            score, daid, dannot_uuid_local, flag = item
            if flag and daid in aid_set:
                dannot_uuid = dannot_uuid_local
                break
        assert dannot_uuid is not None

        response = {
            'name': best_name,
            'sex': best_sex.title(),
            'age': best_age_str,
            'extern': {
                'reference': result['dannot_extern_reference'],
                'qannot_uuid': str(result['qannot_uuid']),
                'dannot_uuid': str(dannot_uuid),
            },
        }

    return response

    # review_pair = result_dict['inference_dict']['annot_pair_dict']['review_pair_list'][0]

    # annot_uuid_key = str(review_pair['annot_uuid_key'])
    # cm_dict = result_dict['cm_dict'][annot_uuid_key]
    # query_config_dict = result_dict['query_config_dict']

    # proot = query_config_dict.get('pipeline_root', 'vsmany')
    # proot = query_config_dict.get('proot', proot)

    # view_orientation = 'horizontal'

    # # unpack info
    # try:
    #     annot_uuid_1 = review_pair['annot_uuid_1']
    #     annot_uuid_2 = review_pair['annot_uuid_2']
    # except Exception:
    #     #??? HACK
    #     # FIXME:
    #     logger.info('[!!!!] review_pair = %r' % (review_pair,))
    #     review_pair = review_pair[0]
    #     annot_uuid_1 = review_pair['annot_uuid_1']
    #     annot_uuid_2 = review_pair['annot_uuid_2']

    # ibs.web_check_uuids(qannot_uuid_list=[annot_uuid_1],
    #                     dannot_uuid_list=[annot_uuid_2])

    # aid_1 = ibs.get_annot_aids_from_uuid(annot_uuid_1)
    # aid_2 = ibs.get_annot_aids_from_uuid(annot_uuid_2)

    # cm = chip_match.ChipMatch.from_dict(cm_dict, ibs=ibs)
    # qreq_ = ibs.new_query_request([aid_1], [aid_2],
    #                               cfgdict=query_config_dict)

    # # Get score
    # # idx = cm.daid2_idx[aid_2]
    # # match_score = cm.name_score_list[idx]
    # match_score = cm.aid2_score[aid_2]

    # try:
    #     image_matches, _ = ensure_review_image(ibs, aid_2, cm, qreq_,
    #                                         view_orientation=view_orientation)
    # except KeyError:
    #     image_matches = np.zeros((100, 100, 3), dtype=np.uint8)
    #     traceback.print_exc()
    # try:
    #     image_clean, _ = ensure_review_image(ibs, aid_2, cm, qreq_,
    #                                       view_orientation=view_orientation,
    #                                       draw_matches=False)
    # except KeyError:
    #     image_clean = np.zeros((100, 100, 3), dtype=np.uint8)
    #     traceback.print_exc()

    # image_matches_src = appf.embed_image_html(image_matches)
    # image_clean_src = appf.embed_image_html(image_clean)

    # return_dict = {
    #     'match': image_matches_src,
    #     'clean': image_clean_src,
    #     'score': match_score,
    # }
    # return return_dict


@register_ibs_method
@register_api('/test/query/chip/', methods=['GET'])
def query_chips_test(ibs, aid=None, limited=False, census_annotations=True, **kwargs):
    """
    CommandLine:
        python -m wbia.web.apis_query query_chips_test

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='testdb1')
        >>> ibs = qreq_.ibs
        >>> result_dict = ibs.query_chips_test()
        >>> print(result_dict)
    """
    from random import shuffle  # NOQA

    # Compile test data
    aid_list = ibs.get_valid_aids()
    if aid is None:
        shuffle(aid_list)
        qaid_list = aid_list[:1]
    else:
        qaid_list = [aid]
    daid_list = aid_list

    if limited:
        daid_list = daid_list[-4:]

    if census_annotations:
        flag_list = ibs.get_annot_canonical(daid_list)
        daid_list = ut.compress(daid_list, flag_list)

    result_dict = ibs.query_chips_graph(qaid_list, daid_list, **kwargs)
    return result_dict


@register_ibs_method
@register_api('/api/query/graph/complete/', methods=['GET', 'POST'])
def query_chips_graph_complete(ibs, aid_list, query_config_dict={}, k=5, **kwargs):
    import theano  # NOQA

    cm_list, qreq_ = ibs.query_chips(
        qaid_list=aid_list,
        daid_list=aid_list,
        cfgdict=query_config_dict,
        return_request=True,
    )

    result_dict = {}
    for cm in cm_list:
        key = str(ibs.get_annot_uuids(cm.qaid))
        id_list = ibs.get_annot_uuids(cm.daid_list)
        score_list = cm.annot_score_list
        value_list = sorted(zip(score_list, id_list), reverse=True)
        k_ = min(k, len(value_list))
        value_list = value_list[:k_]
        result = {str(id_): score for score, id_ in value_list}
        result_dict[key] = result

    return result_dict


@register_ibs_method
def log_render_status(ibs, *args):
    import os

    global RENDER_STATUS

    if RENDER_STATUS is None:
        RENDER_STATUS = ibs._init_render_status()
    assert RENDER_STATUS is not None

    json_log_path = ibs.get_logdir_local()
    json_log_filename = 'render.log'
    json_log_filepath = os.path.join(json_log_path, json_log_filename)
    logger.info('Logging renders added to: {!r}'.format(json_log_filepath))

    status = '{}'.format(args[-1])
    if status not in RENDER_STATUS:
        RENDER_STATUS[status] = 0
    RENDER_STATUS[status] += 1

    try:
        with open(json_log_filepath, 'a') as json_log_file:
            line = ','.join(['{}'.format(arg) for arg in args])
            line = '{}\n'.format(line)
            json_log_file.write(line)
    except Exception:
        logger.info('WRITE RENDER.LOG FAILED')


@register_ibs_method
def _init_render_status(ibs):
    import os

    json_log_path = ibs.get_logdir_local()
    json_log_filename = 'render.log'
    json_log_filepath = os.path.join(json_log_path, json_log_filename)

    status_dict = {}
    try:
        with open(json_log_filepath, 'r') as json_log_file:
            for line in json_log_file.readlines():
                try:
                    line = line.strip().split(',')
                    status = line[-1]
                    if status not in status_dict:
                        status_dict[status] = 0
                    status_dict[status] += 1
                except Exception:
                    pass
    except Exception:
        pass

    return status_dict


@register_ibs_method
@register_api('/api/query/graph/', methods=['GET', 'POST'])
def query_chips_graph(
    ibs,
    qaid_list,
    daid_list,
    user_feedback=None,
    query_config_dict={},
    echo_query_params=True,
    cache_images=True,
    n=20,
    view_orientation='horizontal',
    return_summary=True,
    **kwargs,
):
    import uuid

    import theano  # NOQA

    from wbia.unstable.orig_graph_iden import OrigAnnotInference

    def convert_to_uuid(nid):
        try:
            text = ibs.get_name_texts(nid)
            uuid_ = uuid.UUID(text)
        except ValueError:
            uuid_ = nid
        return uuid_

    proot = query_config_dict.get('pipeline_root', 'vsmany')
    proot = query_config_dict.get('proot', proot)
    use_gradcam = query_config_dict.get('use_gradcam', False)
    logger.info('query_config_dict = {!r}'.format(query_config_dict))

    curvrank_daily_tag = query_config_dict.get('curvrank_daily_tag', None)
    if curvrank_daily_tag is not None:
        if len(curvrank_daily_tag) > 144:
            curvrank_daily_tag_ = ut.hashstr27(curvrank_daily_tag)
            curvrank_daily_tag_ = 'wbia-shortened-{}'.format(curvrank_daily_tag_)
            logger.info('[WARNING] curvrank_daily_tag too long (Probably an old job)')
            logger.info('[WARNING] Original: {!r}'.format(curvrank_daily_tag))
            logger.info('[WARNING] Shortened: {!r}'.format(curvrank_daily_tag_))
            query_config_dict['curvrank_daily_tag'] = curvrank_daily_tag_

    num_qaids = len(qaid_list)
    num_daids = len(daid_list)

    valid_aid_list = ibs.get_valid_aids()
    qaid_list = list(set(qaid_list) & set(valid_aid_list))
    daid_list = list(set(daid_list) & set(valid_aid_list))

    num_qaids_ = len(qaid_list)
    num_daids_ = len(daid_list)

    if num_qaids != num_qaids_:
        logger.info('len(qaid_list)  = %d' % (num_qaids,))
        logger.info('len(qaid_list_) = %d' % (num_qaids_,))

    if num_daids != num_daids_:
        logger.info('len(daid_list)  = %d' % (num_daids,))
        logger.info('len(daid_list_) = %d' % (num_daids_,))

    if num_qaids_ == 0:
        raise ValueError('There are 0 valid query aids, %d were provided' % (num_qaids,))

    if num_daids_ == 0:
        raise ValueError(
            'There are 0 valid database aids, %d were provided' % (num_daids,)
        )

    cm_list, qreq_ = ibs.query_chips(
        qaid_list=qaid_list,
        daid_list=daid_list,
        cfgdict=query_config_dict,
        return_request=True,
    )

    annot_inference = OrigAnnotInference(qreq_, cm_list, user_feedback)
    inference_dict = annot_inference.make_annot_inference_dict()

    cache_dir = join(ibs.cachedir, 'query_match')
    ut.ensuredir(cache_dir)

    cache_path = None
    reference = None

    if cache_images:
        reference = ut.hashstr27(qreq_.get_cfgstr())
        cache_path = join(cache_dir, 'qreq_cfgstr_{}'.format(reference))
        ut.ensuredir(cache_path)

    cm_dict = {}
    for cm in cm_list:
        quuid = ibs.get_annot_uuids(cm.qaid)
        cm_key = str(quuid)

        if cache_images:
            args = (quuid,)
            qannot_cache_filepath = join(cache_path, 'qannot_uuid_%s' % args)
            ut.ensuredir(qannot_cache_filepath)

            assert cache_path is not None

            score_list = cm.score_list
            daid_list_ = cm.daid_list

            # Get best names
            logger.info('Visualizing name matches')
            DEEPSENSE_NUM_TO_VISUALIZE_PER_NAME = 3

            unique_nids = cm.unique_nids
            name_score_list = cm.name_score_list

            zipped = sorted(zip(name_score_list, unique_nids), reverse=True)
            n_ = min(n, len(zipped))
            zipped = zipped[:n_]
            logger.info('Top %d names: %r' % (n_, zipped))
            dnid_set = set(ut.take_column(zipped, 1))
            aids_list = ibs.get_name_aids(dnid_set)

            daid_set = []
            for nid, aid_list in zip(dnid_set, aids_list):
                # Assume that the highest AIDs are the newest sightings, visualize the most recent X sightings per name
                aid_list = sorted(aid_list, reverse=True)
                num_per_name = (
                    len(aid_list)
                    if DEEPSENSE_NUM_TO_VISUALIZE_PER_NAME is None
                    else DEEPSENSE_NUM_TO_VISUALIZE_PER_NAME
                )
                limit = min(len(aid_list), num_per_name)
                args = (
                    nid,
                    len(aid_list),
                    limit,
                )
                logger.info('\tFiltering NID %d from %d -> %d' % args)
                aid_list = aid_list[:limit]
                daid_set += aid_list

            args = (
                len(daid_set),
                daid_set,
            )
            logger.info('Found %d candidate name aids: %r' % args)
            daid_set = set(daid_set)
            name_daid_set = daid_set & set(daid_list_)  # Filter by supplied query daids

            args = (
                len(name_daid_set),
                len(dnid_set),
                name_daid_set,
            )
            logger.info('Found %d overlapping aids for %d nids: %r' % args)

            args = (
                len(name_daid_set),
                len(dnid_set),
            )
            logger.info('Visualizing %d annotations for best %d names' % args)

            # Get best annotations
            logger.info('Visualizing annotation matches')
            zipped = sorted(zip(score_list, daid_list_), reverse=True)
            n_ = min(n, len(zipped))
            zipped = zipped[:n_]
            logger.info('Top %d annots: %r' % (n_, zipped))
            annot_daid_set = set(ut.take_column(zipped, 1))

            # Combine names and annotations
            daid_set = list(name_daid_set) + list(annot_daid_set)
            daid_set = list(set(daid_set))
            logger.info('Visualizing %d annots: %r' % (len(daid_set), daid_set))

            extern_flag_list = []
            for daid in daid_list_:
                extern_flag = daid in daid_set

                if extern_flag:
                    logger.info('Rendering match images to disk for daid=%d' % (daid,))
                    duuid = ibs.get_annot_uuids(daid)

                    args = (duuid,)
                    dannot_cache_filepath = join(
                        qannot_cache_filepath, 'dannot_uuid_%s' % args
                    )
                    ut.ensuredir(dannot_cache_filepath)

                    cache_filepath_fmtstr = join(
                        dannot_cache_filepath, 'version_%s_orient_%s.png'
                    )

                    try:
                        _, filepath_matches = ensure_review_image(
                            ibs,
                            daid,
                            cm,
                            qreq_,
                            view_orientation=view_orientation,
                            draw_matches=True,
                            draw_heatmask=False,
                            use_gradcam=use_gradcam,
                        )
                    except Exception as ex:
                        filepath_matches = None
                        extern_flag = 'error'
                        ut.printex(ex, iswarning=True)
                    log_render_status(
                        ibs,
                        ut.timestamp(),
                        cm.qaid,
                        daid,
                        quuid,
                        duuid,
                        cm,
                        qreq_,
                        view_orientation,
                        True,
                        False,
                        filepath_matches,
                        extern_flag,
                    )
                    try:
                        _, filepath_heatmask = ensure_review_image(
                            ibs,
                            daid,
                            cm,
                            qreq_,
                            view_orientation=view_orientation,
                            draw_matches=False,
                            draw_heatmask=True,
                            use_gradcam=use_gradcam,
                        )
                    except Exception as ex:
                        filepath_heatmask = None
                        extern_flag = 'error'
                        ut.printex(ex, iswarning=True)
                    log_render_status(
                        ibs,
                        ut.timestamp(),
                        cm.qaid,
                        daid,
                        quuid,
                        duuid,
                        cm,
                        qreq_,
                        view_orientation,
                        False,
                        True,
                        filepath_heatmask,
                        extern_flag,
                    )
                    try:
                        _, filepath_clean = ensure_review_image(
                            ibs,
                            daid,
                            cm,
                            qreq_,
                            view_orientation=view_orientation,
                            draw_matches=False,
                            draw_heatmask=False,
                            use_gradcam=use_gradcam,
                        )
                    except Exception as ex:
                        filepath_clean = None
                        extern_flag = 'error'
                        ut.printex(ex, iswarning=True)
                    log_render_status(
                        ibs,
                        ut.timestamp(),
                        cm.qaid,
                        daid,
                        quuid,
                        duuid,
                        cm,
                        qreq_,
                        view_orientation,
                        False,
                        False,
                        filepath_clean,
                        extern_flag,
                    )

                    if filepath_matches is not None:
                        args = (
                            'matches',
                            view_orientation,
                        )
                        cache_filepath = cache_filepath_fmtstr % args
                        ut.symlink(filepath_matches, cache_filepath, overwrite=True)

                    if filepath_heatmask is not None:
                        args = (
                            'heatmask',
                            view_orientation,
                        )
                        cache_filepath = cache_filepath_fmtstr % args
                        ut.symlink(filepath_heatmask, cache_filepath, overwrite=True)

                    if filepath_clean is not None:
                        args = (
                            'clean',
                            view_orientation,
                        )
                        cache_filepath = cache_filepath_fmtstr % args
                        ut.symlink(filepath_clean, cache_filepath, overwrite=True)

                extern_flag_list.append(extern_flag)
        else:
            extern_flag_list = None
        cm_dict[cm_key] = {
            # 'qaid'                    : cm.qaid,
            'qannot_uuid': ibs.get_annot_uuids(cm.qaid),
            # 'qnid'                    : cm.qnid,
            'qname_uuid': convert_to_uuid(cm.qnid),
            'qname': ibs.get_name_texts(cm.qnid),
            # 'daid_list'               : cm.daid_list,
            'dannot_uuid_list': ibs.get_annot_uuids(cm.daid_list),
            # 'dnid_list'               : cm.dnid_list,
            'dname_uuid_list': [convert_to_uuid(nid) for nid in cm.dnid_list],
            # FIXME: use qreq_ state not wbia state
            'dname_list': ibs.get_name_texts(cm.dnid_list),
            'score_list': cm.score_list,
            'annot_score_list': cm.annot_score_list,
            'fm_list': cm.fm_list if hasattr(cm, 'fm_list') else None,
            'fsv_list': cm.fsv_list if hasattr(cm, 'fsv_list') else None,
            # Non-corresponding lists to above
            # 'unique_nids'             : cm.unique_nids,
            'unique_name_uuid_list': [convert_to_uuid(nid) for nid in cm.unique_nids],
            # FIXME: use qreq_ state not wbia state
            'unique_name_list': ibs.get_name_texts(cm.unique_nids),
            'name_score_list': cm.name_score_list,
            # Placeholders for the reinitialization of the ChipMatch object
            'fk_list': None,
            'H_list': None,
            'fsv_col_lbls': None,
            'filtnorm_aids': None,
            'filtnorm_fxs': None,
            'dannot_extern_reference': reference,
            'dannot_extern_list': extern_flag_list,
        }

        if hasattr(cm, 'model_url'):
            cm_dict[cm_key]['model_url'] = cm.model_url
        if hasattr(cm, 'config_url'):
            cm_dict[cm_key]['config_url'] = cm.config_url

    result_dict = {
        'cm_dict': cm_dict,
        'inference_dict': inference_dict,
    }

    if echo_query_params:
        result_dict['query_annot_uuid_list'] = ibs.get_annot_uuids(qaid_list)
        result_dict['database_annot_uuid_list'] = ibs.get_annot_uuids(daid_list)
        result_dict['query_config_dict'] = query_config_dict

    if return_summary:
        summary_list = [
            ('summary_annot', cm.annot_score_list),
            ('summary_name', cm.name_score_list),
        ]
        for summary_key, summary_score_list in summary_list:
            value_list = sorted(list(zip(summary_score_list, cm.daid_list)), reverse=True)
            n_ = min(len(value_list), n)
            value_list = value_list[:n_]

            dscore_list = ut.take_column(value_list, 0)
            daid_list_ = ut.take_column(value_list, 1)
            duuid_list = ibs.get_annot_uuids(daid_list_)
            dnid_list = ibs.get_annot_nids(daid_list_)
            species_list = ibs.get_annot_species(daid_list_)
            viewpoint_list = ibs.get_annot_viewpoints(daid_list_)
            values_list = list(
                zip(
                    dscore_list,
                    duuid_list,
                    daid_list_,
                    dnid_list,
                    species_list,
                    viewpoint_list,
                )
            )
            key_list = ['score', 'duuid', 'daid', 'dnid', 'species', 'viewpoint']
            result_dict[summary_key] = [
                dict(zip(key_list, value_list_)) for value_list_ in values_list
            ]

    return result_dict


@register_route(
    '/api/query/graph/match/thumb/',
    methods=['GET'],
    __route_prefix_check__=False,
    __route_authenticate__=False,
)
def query_chips_graph_match_thumb(
    extern_reference, query_annot_uuid, database_annot_uuid, version
):
    from io import BytesIO

    import vtool as vt
    from flask import send_file
    from PIL import Image  # NOQA

    ibs = current_app.ibs
    args = (
        extern_reference,
        query_annot_uuid,
        database_annot_uuid,
        version,
    )

    cache_dir = join(ibs.cachedir, 'query_match')

    reference_path = join(cache_dir, 'qreq_cfgstr_{}'.format(extern_reference))
    if not exists(reference_path):
        message = 'dannot_extern_reference is unknown'
        raise controller_inject.WebMatchThumbException(*args, message=message)

    qannot_path = join(reference_path, 'qannot_uuid_{}'.format(query_annot_uuid))
    if not exists(qannot_path):
        message = 'query_annot_uuid is unknown for the given reference {}'.format(
            extern_reference,
        )
        raise controller_inject.WebMatchThumbException(*args, message=message)

    dannot_path = join(qannot_path, 'dannot_uuid_{}'.format(database_annot_uuid))
    if not exists(dannot_path):
        message = (
            'database_annot_uuid is unknown for the given reference %s and query_annot_uuid %s'
            % (extern_reference, query_annot_uuid)
        )
        raise controller_inject.WebMatchThumbException(*args, message=message)

    version = version.lower()

    alias_dict = {
        None: 'clean',
        'match': 'matches',
        'mask': 'heatmask',
    }
    for alias in alias_dict:
        version = alias_dict.get(version, version)

    assert version in ['clean', 'matches', 'heatmask']

    version_path = join(dannot_path, 'version_{}_orient_horizontal.png'.format(version))
    if not exists(version_path):
        message = (
            'match thumb version is unknown for the given reference %s, query_annot_uuid %s, database_annot_uuid %s'
            % (extern_reference, query_annot_uuid, database_annot_uuid)
        )
        raise controller_inject.WebMatchThumbException(*args, message=message)

    # Load image
    image = vt.imread(version_path, orient='auto')
    image = image[:, :, ::-1]

    # Encode image
    image_pil = Image.fromarray(image)
    img_io = BytesIO()
    image_pil.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@register_ibs_method
@register_api('/api/query/chip/', methods=['GET'])
def query_chips(
    ibs,
    qaid_list=None,
    daid_list=None,
    cfgdict=None,
    use_cache=None,
    use_bigcache=None,
    qreq_=None,
    return_request=False,
    verbose=pipeline.VERB_PIPELINE,
    save_qcache=None,
    prog_hook=None,
    return_cm_dict=False,
    return_cm_simple_dict=False,
):
    r"""
    Submits a query request to the hotspotter recognition pipeline. Returns
    a list of QueryResult objects.

    Args:
        qaid_list (list): a list of annotation ids to be submitted as
            queries
        daid_list (list): a list of annotation ids used as the database
            that will be searched
        cfgdict (dict): dictionary of configuration options used to create
            a new QueryRequest if not already specified
        use_cache (bool): turns on/off chip match cache (default: True)
        use_bigcache (bool): turns one/off chunked chip match cache (default:
            True)
        qreq\_ (QueryRequest): optional, a QueryRequest object that
            overrides all previous settings
        return_request (bool): returns the request which will be created if
            one is not already specified
        verbose (bool): default=False, turns on verbose printing

    Returns:
        list: a list of ChipMatch objects containing the matching
            annotations, scores, and feature matches

    Returns(2):
        tuple: (cm_list, qreq\_) - a list of query results and optionally the
            QueryRequest object used

    RESTful:
        Method: PUT
        URL:    /api/query/chip/

    CommandLine:
        python -m wbia.web.apis_query --test-query_chips

        # Test speed of single query
        python -m wbia --tf IBEISController.query_chips --db PZ_Master1 \
        -a default:qindex=0:1,dindex=0:500 --nocache-hs

        python -m wbia --tf IBEISController.query_chips --db PZ_Master1 \
        -a default:qindex=0:1,dindex=0:3000 --nocache-hs

        python -m wbia.web.apis_query --test-query_chips:1 --show
        python -m wbia.web.apis_query --test-query_chips:2 --show

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_()
        >>> ibs = qreq_.ibs
        >>> cm_list = qreq_.execute()
        >>> cm = cm_list[0]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> import wbia
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> ibs = wbia.opendb_test(db='testdb1')
        >>> qreq_ = ibs.new_query_request(qaid_list, daid_list)
        >>> cm = ibs.query_chips(qaid_list, daid_list, use_cache=False, qreq_=qreq_)[0]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> import wbia
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> ibs = wbia.opendb_test(db='testdb1')
        >>> cfgdict = {'pipeline_root':'BC_DTW'}
        >>> qreq_ = ibs.new_query_request(qaid_list, daid_list, cfgdict=cfgdict, verbose=True)
        >>> cm = ibs.query_chips(qreq_=qreq_)[0]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()
    """
    # The qaid and daid objects are allowed to be None if qreq_ is
    # specified
    if qreq_ is None:
        assert qaid_list is not None, 'Need to specify qaids'
        assert daid_list is not None, 'Need to specify daids'
        qaid_list, was_scalar = ut.wrap_iterable(qaid_list)
        if daid_list is None:
            daid_list = ibs.get_valid_aids()
        qreq_ = ibs.new_query_request(
            qaid_list, daid_list, cfgdict=cfgdict, verbose=verbose
        )
    else:
        assert qaid_list is None, 'do not specify qreq and qaids'
        assert daid_list is None, 'do not specify qreq and daids'
        was_scalar = False
    cm_list = qreq_.execute()
    assert isinstance(cm_list, list), 'Chip matches were not returned as a list'

    # Convert to cm_list
    if return_cm_simple_dict:
        for cm in cm_list:
            cm.qauuid = ibs.get_annot_uuids(cm.qaid)
            cm.dauuid_list = ibs.get_annot_uuids(cm.daid_list)
        keys = ['qaid', 'daid_list', 'score_list', 'qauuid', 'dauuid_list']
        cm_list = [ut.dict_subset(cm.to_dict(), keys) for cm in cm_list]
    elif return_cm_dict:
        cm_list = [cm.to_dict() for cm in cm_list]

    if was_scalar:
        # hack for scalar input
        assert len(cm_list) == 1
        cm_list = cm_list[0]

    if return_request:
        return cm_list, qreq_
    else:
        return cm_list


##########################################################################################


@register_ibs_method
def get_graph_client_query_chips_graph_v2(ibs, graph_uuid):
    if isinstance(graph_uuid, str):
        try:
            graph_uuid = uuid.UUID(graph_uuid)
        except Exception:
            pass

    graph_client = current_app.GRAPH_CLIENT_DICT.get(graph_uuid, None)
    # We could be redirecting to a newer graph_client
    graph_uuid_chain = [graph_uuid]
    while isinstance(graph_client, str):
        graph_uuid_chain.append(graph_client)
        graph_client = current_app.GRAPH_CLIENT_DICT.get(graph_client, None)
    if graph_client is None:
        raise controller_inject.WebUnknownUUIDException(['graph_uuid'], [graph_uuid])
    return graph_client, graph_uuid_chain


def ensure_review_image_v2(
    ibs,
    match,
    draw_matches=False,
    draw_heatmask=False,
    view_orientation='vertical',
    overlay=True,
):
    import wbia.plottool as pt

    render_config = {
        'overlay': overlay,
        'show_ell': draw_matches,
        'show_lines': draw_matches,
        'show_ori': False,
        'heatmask': draw_heatmask,
        'vert': view_orientation == 'vertical',
    }
    with pt.RenderingContext(dpi=150) as ctx:
        match.show(**render_config)
    image = ctx.image
    return image


def query_graph_v2_callback(graph_client, callback_type):
    from wbia.web.graph_server import ut_to_json_encode

    assert callback_type in ['review', 'finished', 'error']
    callback_tuple = graph_client.callbacks.get(callback_type, None)
    if callback_tuple is not None:
        callback_url, callback_method = callback_tuple
        if callback_url is not None:
            callback_method = callback_method.lower()
            data_dict = ut_to_json_encode({'graph_uuid': graph_client.graph_uuid})
            if callback_method == 'post':
                requests.post(callback_url, data=data_dict)
            elif callback_method == 'get':
                requests.get(callback_url, params=data_dict)
            elif callback_method == 'put':
                requests.put(callback_url, data=data_dict)
            elif callback_method == 'delete':
                requests.delete(callback_url, data=data_dict)
            else:
                raise KeyError('Unsupported HTTP callback method')


@register_ibs_method
@register_api('/api/query/graph/v2/', methods=['POST'])
def query_chips_graph_v2(
    ibs,
    annot_uuid_list=None,
    query_config_dict={},
    review_callback_url=None,
    review_callback_method='POST',
    finished_callback_url=None,
    finished_callback_method='POST',
    creation_imageset_rowid_list=None,
    backend='graph_algorithm',
    **kwargs,
):
    """
    CommandLine:
        python -m wbia.web.apis_query --test-query_chips_graph_v2:0

        python -m wbia reset_mtest_graph

        python -m wbia --db PZ_MTEST --web --browser --url=/review/identification/hardcase/
        python -m wbia --db PZ_MTEST --web --browser --url=/review/identification/graph/

    Example:
        >>> # xdoctest: +REQUIRES(--web-tests)
        >>> from wbia.web.apis_query import *
        >>> import wbia
        >>> # Open local instance
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> uuid_list = ibs.annots().uuids[0:10]
        >>> data = dict(annot_uuid_list=uuid_list)
        >>> # Start up the web instance
        >>> with wbia.opendb_with_web(db='PZ_MTEST') as (ibs, client):
        ...     resp = client.post('/api/query/graph/v2/', data=data)
        >>> resp.json
        {'status': {'success': False, 'code': 608, 'message': 'Invalid image and/or annotation UUIDs (0, 1)', 'cache': -1}, 'response': {'invalid_image_uuid_list': [], 'invalid_annot_uuid_list': [[0, '...']]}}

    Example:
        >>> # DEBUG_SCRIPT
        >>> from wbia.web.apis_query import *
        >>> # Hack a flask context
        >>> current_app = ut.DynStruct()
        >>> current_app.GRAPH_CLIENT_DICT = {}
        >>> old = query_chips_graph_v2.__globals__.get('current_app', None)
        >>> query_chips_graph_v2.__globals__['current_app'] = current_app
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> #ut.exec_funckw(query_chips_graph_v2, globals())
        >>> # Run function in main process
        >>> query_chips_graph_v2(ibs)
        >>> # Reset context
        >>> query_chips_graph_v2.__globals__['current_app'] = old
    """

    backend = backend.lower()
    if backend in ['graph_algorithm', 'graph', 'ga']:
        from wbia.web.graph_server import GraphAlgorithmClient as GraphClient
    elif backend in ['lca', 'clusters']:
        from wbia_lca._plugin import LCAClient as GraphClient
    else:
        raise NotImplementedError('Requested backend {!r} not supported'.format(backend))

    logger.info('[apis_query] Creating GraphClient')

    if annot_uuid_list is None:
        annot_uuid_list = ibs.get_annot_uuids(ibs.get_valid_aids())

    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)

    # FILTER FOR GGR2
    if False:
        aid_list = ibs.check_ggr_valid_aids(aid_list, **kwargs)

    graph_uuid = ut.hashable_to_uuid(sorted(aid_list))
    if graph_uuid not in current_app.GRAPH_CLIENT_DICT:
        for graph_uuid_ in current_app.GRAPH_CLIENT_DICT:
            graph_client_ = current_app.GRAPH_CLIENT_DICT[graph_uuid_]
            aid_list_ = graph_client_.aids
            assert aid_list_ is not None
            overlap_aid_set = set(aid_list_) & set(aid_list)
            if len(overlap_aid_set) > 0:
                overlap_aid_list = list(overlap_aid_set)
                overlap_annot_uuid_list = ibs.get_annot_uuids(overlap_aid_list)
                raise controller_inject.WebUnavailableUUIDException(
                    overlap_annot_uuid_list, graph_uuid_
                )

        callback_dict = {
            'review': (review_callback_url, review_callback_method),
            'finished': (finished_callback_url, finished_callback_method),
        }

        config = {
            'manual.n_peek': GRAPH_CLIENT_PEEK,
            'manual.autosave': True,
            'redun.pos': 2,
            'redun.neg': 2,
            'algo.quickstart': False,
        }
        config.update(query_config_dict)
        logger.info(
            '[apis_query] graph_client.actor_config = {}'.format(ut.repr3(config))
        )

        graph_client = GraphClient(
            aids=aid_list,
            actor_config=config,
            imagesets=creation_imageset_rowid_list,
            graph_uuid=graph_uuid,
            callbacks=callback_dict,
            autoinit=True,
        )

        # Ensure no race-conditions
        current_app.GRAPH_CLIENT_DICT[graph_uuid] = graph_client

        # Start (create the Graph Inference object)
        payload = {
            'action': 'start',
            'dbdir': ibs.dbdir,
            'aids': graph_client.aids,
            'config': graph_client.actor_config,
            'graph_uuid': graph_uuid,
        }
        future = graph_client.post(payload)
        future.result()  # Guarantee that this has happened before calling refresh

        future = graph_client.post({'action': 'logs'})
        future.graph_client = graph_client
        future.add_done_callback(query_graph_v2_latest_logs)

        # Refresh metadata (place to store information provided by the actor)
        graph_client.refresh_metadata()

        # Start main loop
        future = graph_client.post({'action': 'resume'})
        future.graph_client = graph_client
        future.add_done_callback(query_graph_v2_on_request_review)

        future = graph_client.post({'action': 'logs'})
        future.graph_client = graph_client
        future.add_done_callback(query_graph_v2_latest_logs)

    return graph_uuid


@register_ibs_method
def review_graph_match_config_v2(
    ibs, graph_uuid, aid1=None, aid2=None, view_orientation='vertical', view_version=1
):
    from flask import session

    from wbia.algo.verif import pairfeat

    if isinstance(graph_uuid, str):
        try:
            graph_uuid = uuid.UUID(graph_uuid)
        except Exception:
            pass

    EDGES_KEY = '_EDGES_'
    EDGES_MAX = 10

    user_id = controller_inject.get_user()
    graph_client, _ = ibs.get_graph_client_query_chips_graph_v2(graph_uuid)

    if aid1 is not None and aid2 is not None:
        previous_edge_list = None
        if aid1 > aid2:
            aid1, aid2 = aid2, aid1
        edge = (aid1, aid2)
        data = graph_client.check(edge)
        if data is None:
            data = (
                edge,
                np.nan,
                {},
            )
    else:
        if EDGES_KEY not in session:
            session[EDGES_KEY] = []
        previous_edge_list = session[EDGES_KEY]
        logger.info(
            'Using previous_edge_list\n\tUser: %s\n\tList: %r'
            % (user_id, previous_edge_list)
        )

        data = graph_client.sample(
            previous_edge_list=previous_edge_list, max_previous_edges=EDGES_MAX
        )
        if data is None:
            raise controller_inject.WebReviewNotReadyException(graph_uuid)

    edge, priority, data_dict = data
    edge = [
        int(edge[0]),
        int(edge[1]),
    ]
    aid_1, aid_2 = edge
    annot_uuid_1 = str(ibs.get_annot_uuids(aid_1))
    annot_uuid_2 = str(ibs.get_annot_uuids(aid_2))

    if previous_edge_list is not None:
        previous_edge_list.append(edge)
        if len(previous_edge_list) > EDGES_MAX:
            cutoff = int(-1.0 * EDGES_MAX)
            previous_edge_list = previous_edge_list[cutoff:]
        session[EDGES_KEY] = previous_edge_list
        logger.info(
            'Updating previous_edge_list\n\tUser: %s\n\tList: %r'
            % (user_id, previous_edge_list)
        )

    args = (
        edge,
        priority,
    )
    logger.info('Sampled edge %r with priority %0.02f' % args)
    logger.info('Data: ' + ut.repr4(data_dict))

    extr_ = graph_client.metadata.get('extr', None)
    feat_extract_config = {
        'match_config': {} if extr_ is None else extr_.match_config,
    }
    extr = pairfeat.PairwiseFeatureExtractor(ibs, config=feat_extract_config)

    match = extr._exec_pairwise_match([edge])[0]

    image_clean = ensure_review_image_v2(
        ibs, match, view_orientation=view_orientation, overlay=False
    )
    # image_matches = ensure_review_image_v2(ibs, match, draw_matches=True,
    #                                        view_orientation=view_orientation)

    logger.info('Using View Version: {!r}'.format(view_version))
    if view_version == 1:
        image_heatmask = ensure_review_image_v2(
            ibs, match, draw_heatmask=True, view_orientation=view_orientation
        )
    else:
        image_heatmask = ensure_review_image_v2(
            ibs, match, draw_matches=True, view_orientation=view_orientation
        )

    image_clean_src = appf.embed_image_html(image_clean)
    # image_matches_src = appf.embed_image_html(image_matches)
    image_heatmask_src = appf.embed_image_html(image_heatmask)

    now = datetime.utcnow()
    server_time_start = float(now.strftime('%s.%f'))

    return (
        edge,
        priority,
        data_dict,
        aid_1,
        aid_2,
        annot_uuid_1,
        annot_uuid_2,
        image_clean_src,
        image_heatmask_src,
        image_heatmask_src,
        server_time_start,
    )


@register_api('/api/review/query/graph/v2/', methods=['GET'])
def review_graph_match_html_v2(
    ibs,
    graph_uuid,
    callback_url=None,
    callback_method='POST',
    view_orientation='vertical',
    view_version=1,
    include_jquery=False,
):
    values = ibs.review_graph_match_config_v2(
        graph_uuid, view_orientation=view_orientation, view_version=view_version
    )

    (
        edge,
        priority,
        data_dict,
        aid1,
        aid2,
        annot_uuid_1,
        annot_uuid_2,
        image_clean_src,
        image_matches_src,
        image_heatmask_src,
        server_time_start,
    ) = values

    confidence_dict = const.CONFIDENCE.NICE_TO_CODE
    confidence_nice_list = confidence_dict.keys()
    confidence_text_list = confidence_dict.values()
    confidence_selected_list = [
        confidence_text == 'unspecified' for confidence_text in confidence_text_list
    ]
    confidence_list = list(
        zip(confidence_nice_list, confidence_text_list, confidence_selected_list)
    )

    if False:
        from wbia.web import apis_query

        root_path = dirname(abspath(apis_query.__file__))
    else:
        root_path = dirname(abspath(__file__))
    css_file_list = [
        ['css', 'style.css'],
        ['include', 'bootstrap', 'css', 'bootstrap.css'],
    ]
    json_file_list = [
        ['javascript', 'script.js'],
        ['include', 'bootstrap', 'js', 'bootstrap.js'],
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

    embedded = dict(globals(), **locals())
    return appf.template('review', 'identification_insert', **embedded)


@register_api('/api/status/query/graph/v2/', methods=['GET'], __api_plural_check__=False)
def view_graphs_status(ibs):
    graph_dict = {}
    for graph_uuid in current_app.GRAPH_CLIENT_DICT:
        graph_client = current_app.GRAPH_CLIENT_DICT.get(graph_uuid, None)
        if graph_client is None:
            continue
        graph_status, graph_exception = graph_client.refresh_status()
        if graph_client.review_dict is None:
            num_edges = None
        else:
            edge_list = list(graph_client.review_dict.keys())
            num_edges = len(edge_list)
        graph_uuid = str(graph_uuid)
        graph_dict[graph_uuid] = {
            'status': graph_status,
            'num_aids': len(graph_client.aids),
            'num_reviews': num_edges,
        }
    return graph_dict


@register_ibs_method
@register_api('/api/review/query/graph/v2/', methods=['POST'])
def process_graph_match_html_v2(ibs, graph_uuid, **kwargs):
    if isinstance(graph_uuid, str):
        try:
            graph_uuid = uuid.UUID(graph_uuid)
        except Exception:
            pass

    graph_client, _ = ibs.get_graph_client_query_chips_graph_v2(graph_uuid)
    response_tuple = process_graph_match_html(ibs, **kwargs)
    (
        annot_uuid_1,
        annot_uuid_2,
        decision,
        tags,
        user_id,
        confidence,
        user_times,
    ) = response_tuple
    aid1 = ibs.get_annot_aids_from_uuid(annot_uuid_1)
    aid2 = ibs.get_annot_aids_from_uuid(annot_uuid_2)
    edge = (
        aid1,
        aid2,
    )
    user_id = controller_inject.get_user()
    now = datetime.utcnow()

    if decision in ['excludetop', 'excludebottom']:
        aid = aid1 if decision == 'excludetop' else aid2

        metadata_dict = ibs.get_annot_metadata(aid)
        assert 'excluded' not in metadata_dict
        metadata_dict['excluded'] = True
        ibs.set_annot_metadata([aid], [metadata_dict])

        payload = {
            'action': 'remove_aids',
            'aids': [aid],
        }
    elif decision in ['excludeleft', 'excluderight']:
        aid = aid1 if decision == 'excludeleft' else aid2

        metadata_dict = ibs.get_annot_metadata(aid)
        assert 'excluded' not in metadata_dict
        metadata_dict['excluded'] = True
        ibs.set_annot_metadata([aid], [metadata_dict])

        payload = {
            'action': 'remove_aids',
            'aids': [aid],
        }
    else:
        payload = {
            'action': 'feedback',
            'edge': edge,
            'evidence_decision': decision,
            # TODO: meta_decision should come from the html resp.  When generating
            # the html page, the default value should be its previous value. If the
            # user changes it to be something incompatible them perhaps just reset
            # it to null.
            'meta_decision': 'null',
            'tags': [] if len(tags) == 0 else tags.split(';'),
            'user_id': 'user:web:{}'.format(user_id),
            'confidence': confidence,
            'timestamp_s1': user_times['server_time_start'],
            'timestamp_c1': user_times['client_time_start'],
            'timestamp_c2': user_times['client_time_end'],
            'timestamp': float(now.strftime('%s.%f')),
        }
    logger.info('POSTING GRAPH CLIENT REVIEW:')
    logger.info(ut.repr4(payload))
    graph_client.post(payload)

    # Clean any old resumes
    graph_client.cleanup()

    # Continue review
    future = graph_client.post({'action': 'resume'})
    future.graph_client = graph_client
    future.add_done_callback(query_graph_v2_on_request_review)

    future = graph_client.post({'action': 'logs'})
    future.graph_client = graph_client
    future.add_done_callback(query_graph_v2_latest_logs)
    return (
        annot_uuid_1,
        annot_uuid_2,
    )


@register_ibs_method
@register_api('/api/query/graph/v2/', methods=['GET'])
def sync_query_chips_graph_v2(ibs, graph_uuid):
    if isinstance(graph_uuid, str):
        try:
            graph_uuid = uuid.UUID(graph_uuid)
        except Exception:
            pass

    graph_client, _ = ibs.get_graph_client_query_chips_graph_v2(graph_uuid)
    ret_dict = graph_client.sync(ibs)
    return ret_dict


@register_ibs_method
@register_api('/api/query/graph/v2/', methods=['PUT'])
def add_annots_query_chips_graph_v2(ibs, graph_uuid, annot_uuid_list):
    if isinstance(graph_uuid, str):
        try:
            graph_uuid = uuid.UUID(graph_uuid)
        except Exception:
            pass

    graph_client, _ = ibs.get_graph_client_query_chips_graph_v2(graph_uuid)
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)

    for graph_uuid_ in current_app.GRAPH_CLIENT_DICT:
        graph_client_ = current_app.GRAPH_CLIENT_DICT[graph_uuid_]
        aid_list_ = graph_client_.aids
        assert aid_list_ is not None
        overlap_aid_set = set(aid_list_) & set(aid_list)
        if len(overlap_aid_set) > 0:
            overlap_aid_list = list(overlap_aid_set)
            overlap_annot_uuid_list = ibs.get_annot_uuids(overlap_aid_list)
            raise controller_inject.WebUnavailableUUIDException(
                overlap_annot_uuid_list, graph_uuid_
            )

    aid_list_ = graph_client.aids + aid_list
    graph_uuid_ = ut.hashable_to_uuid(sorted(aid_list_))
    assert graph_uuid_ not in current_app.GRAPH_CLIENT_DICT
    graph_client.graph_uuid = graph_uuid_

    payload = {
        'action': 'add_aids',
        'dbdir': ibs.dbdir,
        'aids': aid_list,
    }
    future = graph_client.post(payload)
    future.result()  # Guarantee that this has happened before calling refresh

    # Start main loop
    future = graph_client.post({'action': 'resume'})
    future.graph_client = graph_client
    future.add_done_callback(query_graph_v2_on_request_review)

    current_app.GRAPH_CLIENT_DICT[graph_uuid_] = graph_client
    current_app.GRAPH_CLIENT_DICT[graph_uuid] = graph_uuid_
    return graph_uuid_


@register_ibs_method
def remove_annots_query_chips_graph_v2(ibs, graph_uuid, annot_uuid_list):
    if isinstance(graph_uuid, str):
        try:
            graph_uuid = uuid.UUID(graph_uuid)
        except Exception:
            pass

    graph_client, _ = ibs.get_graph_client_query_chips_graph_v2(graph_uuid)
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)

    aid_list_ = list(set(graph_client.aids) - set(aid_list))
    graph_uuid_ = ut.hashable_to_uuid(sorted(aid_list_))
    assert graph_uuid_ not in current_app.GRAPH_CLIENT_DICT
    graph_client.graph_uuid = graph_uuid_

    payload = {
        'action': 'remove_aids',
        'dbdir': ibs.dbdir,
        'aids': aid_list,
    }
    future = graph_client.post(payload)
    future.result()  # Guarantee that this has happened before calling refresh

    # Start main loop
    future = graph_client.post({'action': 'resume'})
    future.graph_client = graph_client
    future.add_done_callback(query_graph_v2_on_request_review)

    current_app.GRAPH_CLIENT_DICT[graph_uuid_] = graph_client
    current_app.GRAPH_CLIENT_DICT[graph_uuid] = graph_uuid_
    return graph_uuid_


@register_ibs_method
@register_api('/api/query/graph/v2/', methods=['DELETE'])
def delete_query_chips_graph_v2(ibs, graph_uuid):
    if isinstance(graph_uuid, str):
        try:
            graph_uuid = uuid.UUID(graph_uuid)
        except Exception:
            pass

    values = ibs.get_graph_client_query_chips_graph_v2(graph_uuid)
    graph_client, graph_uuid_chain = values
    del graph_client
    for graph_uuid_ in graph_uuid_chain:
        if graph_uuid_ in current_app.GRAPH_CLIENT_DICT:
            current_app.GRAPH_CLIENT_DICT[graph_uuid_] = None
            current_app.GRAPH_CLIENT_DICT.pop(graph_uuid_)
    return True


def query_graph_v2_latest_logs(future):
    if not future.cancelled():
        logs = future.result()
        if logs is not None:
            logger.info('--- <LOG DUMP> ---')
            for msg, color in logs:
                ut.cprint('[web.infr] ' + msg, color)
            logger.info('--- <\\LOG DUMP> ---')


def query_graph_v2_on_request_review(future):
    if not future.cancelled():
        graph_client = future.graph_client
        try:
            data_list = future.result()
            finished = graph_client.update(data_list)
            callback_type = 'finished' if finished else 'review'
        except ValueError:
            callback_type = 'error'
        query_graph_v2_callback(graph_client, callback_type)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.detect.train_assigner --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
