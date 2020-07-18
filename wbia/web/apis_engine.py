# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

# if False:
#    import os
#    os.environ['UTOOL_NOCNN'] = 'True'
import six
import utool as ut
import uuid  # NOQA
from wbia.control import controller_inject
import wbia.constants as const

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_api = controller_inject.get_wbia_flask_api(__name__)


def ensure_simple_server(port=5832):
    r"""
    CommandLine:
        python -m wbia.web.apis_engine --exec-ensure_simple_server
        python -m utool.util_web --exec-start_simple_webserver

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_engine import *  # NOQA
        >>> result = ensure_simple_server()
        >>> print(result)
    """
    if ut.is_local_port_open(port):
        bgserver = ut.spawn_background_process(ut.start_simple_webserver, port=port)
        return bgserver
    else:
        bgserver = ut.DynStruct()
        bgserver.terminate2 = lambda: None
        print('server is running elsewhere')
    return bgserver


def ensure_uuid_list(list_):
    if list_ is not None and len(list_) > 0 and isinstance(list_[0], six.string_types):
        list_ = list(map(uuid.UUID, list_))
    return list_


def web_check_annot_uuids_with_names(annot_uuid_list, name_list):
    assert None not in [annot_uuid_list, name_list]
    assert len(annot_uuid_list) == len(name_list)

    bad_uuid_set = set([])

    annot_dict = {}
    zipped = zip(annot_uuid_list, name_list)
    for index, (annot_uuid, name) in enumerate(zipped):
        if not isinstance(annot_uuid, (uuid.UUID, six.string_types)):
            raise ValueError(
                'Received UUID %r that is not a UUID or string at index %s'
                % (annot_uuid, index,)
            )

        if annot_uuid not in annot_dict:
            annot_dict[annot_uuid] = set([])

        if name not in ['', const.UNKNOWN, None]:
            annot_dict[annot_uuid].add(name)

        if len(annot_dict[annot_uuid]) > 1:
            bad_uuid_set.add(annot_uuid)

    if len(bad_uuid_set) > 0:
        bad_dict = {bad_uuid: annot_dict[bad_uuid] for bad_uuid in bad_uuid_set}
        raise controller_inject.WebMultipleNamedDuplicateException(bad_dict)

    annot_uuid_list_ = []
    name_list_ = []
    for annot_uuid in annot_dict:
        names = list(annot_dict.get(annot_uuid, []))
        if len(names) == 0:
            name = const.UNKNOWN
        elif len(names) == 1:
            name = names[0]
        else:
            raise RuntimeError('Sanity check failed in web_check_annot_uuids_with_names')

        annot_uuid_list_.append(annot_uuid)
        name_list_.append(name)

    return annot_uuid_list_, name_list_


@register_ibs_method
@register_api('/api/engine/uuid/check/', methods=['GET', 'POST'])
def web_check_uuids(ibs, image_uuid_list=[], qannot_uuid_list=[], dannot_uuid_list=[]):
    r"""
    Args:
        ibs (wbia.IBEISController):  image analysis api
        image_uuid_list (list): (default = [])
        qannot_uuid_list (list): (default = [])
        dannot_uuid_list (list): (default = [])

    CommandLine:
        python -m wbia.web.apis_engine --exec-web_check_uuids --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_engine import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> image_uuid_list = []
        >>> qannot_uuid_list = ibs.get_annot_uuids([1, 1, 2, 3, 2, 4])
        >>> dannot_uuid_list = ibs.get_annot_uuids([1, 2, 3])
        >>> try:
        >>>     web_check_uuids(ibs, image_uuid_list, qannot_uuid_list,
        >>>                     dannot_uuid_list)
        >>> except controller_inject.WebDuplicateUUIDException:
        >>>     pass
        >>> else:
        >>>     raise AssertionError('Should have gotten WebDuplicateUUIDException')
        >>> try:
        >>>     web_check_uuids(ibs, [1, 2, 3], qannot_uuid_list,
        >>>                     dannot_uuid_list)
        >>> except controller_inject.WebMissingUUIDException as ex:
        >>>     pass
        >>> else:
        >>>     raise AssertionError('Should have gotten WebMissingUUIDException')
        >>> print('Successfully reported errors')
    """
    import uuid

    # Unique list
    if qannot_uuid_list is None:
        qannot_uuid_list = []
    if dannot_uuid_list is None:
        dannot_uuid_list = []

    annot_uuid_list = qannot_uuid_list + dannot_uuid_list

    # UUID parse check
    invalid_image_uuid_list = []
    for index, image_uuid in enumerate(image_uuid_list):
        try:
            assert isinstance(image_uuid, uuid.UUID)
        except Exception:
            value = (
                index,
                image_uuid,
            )
            invalid_image_uuid_list.append(value)

    invalid_annot_uuid_list = []
    for index, annot_uuid in enumerate(annot_uuid_list):
        try:
            assert isinstance(annot_uuid, uuid.UUID)
        except Exception:
            value = (
                index,
                annot_uuid,
            )
            invalid_annot_uuid_list.append(value)

    if len(invalid_image_uuid_list) > 0 or len(invalid_annot_uuid_list) > 0:
        kwargs = {
            'invalid_image_uuid_list': invalid_image_uuid_list,
            'invalid_annot_uuid_list': invalid_annot_uuid_list,
        }
        raise controller_inject.WebInvalidUUIDException(**kwargs)

    image_uuid_list = list(set(image_uuid_list))
    annot_uuid_list = list(set(annot_uuid_list))
    # Check for all annot UUIDs exist
    missing_image_uuid_list = ibs.get_image_missing_uuid(image_uuid_list)
    missing_annot_uuid_list = ibs.get_annot_missing_uuid(annot_uuid_list)
    if len(missing_image_uuid_list) > 0 or len(missing_annot_uuid_list) > 0:
        kwargs = {
            'missing_image_uuid_list': missing_image_uuid_list,
            'missing_annot_uuid_list': missing_annot_uuid_list,
        }
        raise controller_inject.WebMissingUUIDException(**kwargs)
    ddup_pos_map = ut.find_duplicate_items(qannot_uuid_list)
    qdup_pos_map = ut.find_duplicate_items(dannot_uuid_list)
    if len(ddup_pos_map) + len(qdup_pos_map) > 0:
        raise controller_inject.WebDuplicateUUIDException(qdup_pos_map, qdup_pos_map)


@register_ibs_method
@register_api('/api/engine/query/annot/rowid/', methods=['GET', 'POST'])
def start_identify_annots(
    ibs,
    qannot_uuid_list,
    dannot_uuid_list=None,
    pipecfg={},
    callback_url=None,
    callback_method=None,
):
    r"""
    REST:
        Method: GET
        URL: /api/engine/query/annot/rowid/

    Args:
        qannot_uuid_list (list) : specifies the query annotations to
            identify.
        dannot_uuid_list (list) : specifies the annotations that the
            algorithm is allowed to use for identification.  If not
            specified all annotations are used.   (default=None)
        pipecfg (dict) : dictionary of pipeline configuration arguments
            (default=None)

    CommandLine:
        # Run as main process
        python -m wbia.web.apis_engine --exec-start_identify_annots:0
        # Run using server process
        python -m wbia.web.apis_engine --exec-start_identify_annots:1

        # Split into multiple processes
        python -m wbia.web.apis_engine --main --bg
        python -m wbia.web.apis_engine --exec-start_identify_annots:1 --fg

        python -m wbia.web.apis_engine --exec-start_identify_annots:1 --domain http://52.33.105.88

        python -m wbia.web.apis_engine --exec-start_identify_annots:1 --duuids=[]
        python -m wbia.web.apis_engine --exec-start_identify_annots:1 --domain http://52.33.105.88 --duuids=03a17411-c226-c960-d180-9fafef88c880


    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_engine import *  # NOQA
        >>> from wbia.web import apis_engine
        >>> import wbia
        >>> ibs, qaids, daids = wbia.testdata_expanded_aids(
        >>>     defaultdb='PZ_MTEST', a=['default:qsize=2,dsize=10'])
        >>> qannot_uuid_list = ibs.get_annot_uuids(qaids)
        >>> dannot_uuid_list = ibs.get_annot_uuids(daids)
        >>> pipecfg = {}
        >>> ibs.initialize_job_manager()
        >>> jobid = ibs.start_identify_annots(qannot_uuid_list, dannot_uuid_list, pipecfg)
        >>> result = ibs.wait_for_job_result(jobid, timeout=None, freq=2)
        >>> print(result)
        >>> import utool as ut
        >>> #print(ut.to_json(result))
        >>> ibs.close_job_manager()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_engine import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')  # , domain='http://52.33.105.88')
        >>> aids = ibs.get_valid_aids()[0:2]
        >>> qaids = aids[0:1]
        >>> daids = aids
        >>> query_config_dict = {
        >>>     #'pipeline_root' : 'BC_DTW'
        >>> }
        >>> qreq_ = ibs.new_query_request(qaids, daids, cfgdict=query_config_dict)
        >>> cm_list = qreq_.execute()

    Example:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.web.apis_engine import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', managed=True) as web_ibs:  # , domain='http://52.33.105.88')
        ...     aids = web_ibs.send_wbia_request('/api/annot/', 'get')[0:2]
        ...     uuid_list = web_ibs.send_wbia_request('/api/annot/uuid/', type_='get', aid_list=aids)
        ...     quuid_list = ut.get_argval('--quuids', type_=list, default=uuid_list)
        ...     duuid_list = ut.get_argval('--duuids', type_=list, default=uuid_list)
        ...     data = dict(
        ...         qannot_uuid_list=quuid_list, dannot_uuid_list=duuid_list,
        ...         pipecfg={},
        ...         callback_url='http://127.0.1.1:5832'
        ...     )
        ...     # Start callback server
        ...     bgserver = ensure_simple_server()
        ...     # --
        ...     jobid = web_ibs.send_wbia_request('/api/engine/query/annot/rowid/', **data)
        ...     status_response = web_ibs.wait_for_results(jobid, delays=[1, 5, 30])
        ...     print('status_response = %s' % (status_response,))
        ...     result_response = web_ibs.read_engine_results(jobid)
        ...     print('result_response = %s' % (result_response,))
        ...     cm_dict = result_response['json_result'][0]
        ...     print('Finished test')
        ...     bgserver.terminate2()

    Ignore:
        qaids = daids = ibs.get_valid_aids()
        jobid = ibs.start_identify_annots(**payload)
    """
    # Check UUIDs
    ibs.web_check_uuids([], qannot_uuid_list, dannot_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)
    qannot_uuid_list = ensure_uuid_list(qannot_uuid_list)
    dannot_uuid_list = ensure_uuid_list(dannot_uuid_list)

    qaid_list = ibs.get_annot_aids_from_uuid(qannot_uuid_list)
    if dannot_uuid_list is None:
        daid_list = ibs.get_valid_aids()
    else:
        if len(dannot_uuid_list) == 1 and dannot_uuid_list[0] is None:
            # VERY HACK
            daid_list = ibs.get_valid_aids()
        else:
            daid_list = ibs.get_annot_aids_from_uuid(dannot_uuid_list)

    ibs.assert_valid_aids(
        qaid_list, msg='error in start_identify qaids', auuid_list=qannot_uuid_list
    )
    ibs.assert_valid_aids(
        daid_list, msg='error in start_identify daids', auuid_list=dannot_uuid_list
    )
    args = (qaid_list, daid_list, pipecfg)
    jobid = ibs.job_manager.jobiface.queue_job(
        'query_chips_simple_dict', callback_url, callback_method, *args
    )

    # if callback_url is not None:
    #    #import requests
    #    #requests.
    #    #callback_url
    return jobid


@register_ibs_method
@register_api('/api/engine/query/graph/complete/', methods=['GET', 'POST'])
def start_identify_annots_query_complete(
    ibs,
    annot_uuid_list=None,
    annot_name_list=None,
    matching_state_list=[],
    query_config_dict={},
    k=5,
    echo_query_params=True,
    callback_url=None,
    callback_method=None,
):
    r"""
    REST:
        Method: GET
        URL: /api/engine/query/complete/

    Args:
        annot_uuid_list (list) : specifies the query annotations to
            identify.
        annot_name_list (list) : specifies the query annotation names
        matching_state_list (list of tuple) : the list of matching state
            3-tuples corresponding to the query_annot_uuid_list (default=None)
        query_config_dict (dict) : dictionary of algorithmic configuration
            arguments.  (default=None)
        echo_query_params (bool) : flag for if to return the original query
            parameters with the result
    """
    # Check inputs
    if annot_uuid_list is not None and annot_name_list is not None:
        assert len(annot_uuid_list) == len(annot_name_list)

    # Check UUIDs
    if annot_name_list is not None:
        vals = web_check_annot_uuids_with_names(annot_uuid_list, annot_name_list)
        annot_uuid_list, annot_name_list = vals
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)

    annot_uuid_list = ensure_uuid_list(annot_uuid_list)

    # Ensure annotations
    if annot_uuid_list is None or (
        len(annot_uuid_list) == 1 and annot_uuid_list[0] is None
    ):
        aid_list = ibs.get_valid_aids()
    else:
        aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)

    # Ensure names
    # FIXME: THE QREQ STRUCTURE SHOULD HANDLE NAMES.
    if annot_name_list is not None:
        # Set names for query annotations
        nid_list = ibs.add_names(annot_name_list)
        ibs.set_annot_name_rowids(aid_list, nid_list)

    ibs.assert_valid_aids(
        aid_list, msg='error in start_identify qaids', auuid_list=annot_uuid_list
    )

    args = (
        aid_list,
        query_config_dict,
        k,
    )
    jobid = ibs.job_manager.jobiface.queue_job(
        'query_chips_graph_complete', callback_url, callback_method, *args
    )
    return jobid


@register_ibs_method
@register_api('/api/engine/query/graph/', methods=['GET', 'POST'])
def start_identify_annots_query(
    ibs,
    query_annot_uuid_list=None,
    # query_annot_name_uuid_list=None,
    query_annot_name_list=None,
    database_annot_uuid_list=None,
    # database_annot_name_uuid_list=None,
    database_annot_name_list=None,
    matching_state_list=[],
    query_config_dict={},
    echo_query_params=True,
    include_qaid_in_daids=True,
    callback_url=None,
    callback_method=None,
):
    r"""
    REST:
        Method: GET
        URL: /api/engine/query/graph/

    Args:
        query_annot_uuid_list (list) : specifies the query annotations to
            identify.
        query_annot_name_list (list) : specifies the query annotation names
        database_annot_uuid_list (list) : specifies the annotations that the
            algorithm is allowed to use for identification.  If not
            specified all annotations are used.   (default=None)
        database_annot_name_list (list) : specifies the database annotation
            names (default=None)
        matching_state_list (list of tuple) : the list of matching state
            3-tuples corresponding to the query_annot_uuid_list (default=None)
        query_config_dict (dict) : dictionary of algorithmic configuration
            arguments.  (default=None)
        echo_query_params (bool) : flag for if to return the original query
            parameters with the result

    CommandLine:
        # Normal mode
        python -m wbia.web.apis_engine start_identify_annots_query
        # Split mode
        wbia --web
        python -m wbia.web.apis_engine start_identify_annots_query --show --domain=localhost

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_engine import *  # NOQA
        >>> import wbia
        >>> #domain = 'localhost'
        >>> domain = None
        >>> with wbia.opendb_bg_web('testdb1', domain=domain, managed=True) as web_ibs:  # , domain='http://52.33.105.88')
        ...     aids = web_ibs.send_wbia_request('/api/annot/', 'get')[0:3]
        ...     uuid_list = web_ibs.send_wbia_request('/api/annot/uuid/', type_='get', aid_list=aids)
        ...     quuid_list = ut.get_argval('--quuids', type_=list, default=uuid_list)[0:1]
        ...     duuid_list = ut.get_argval('--duuids', type_=list, default=uuid_list)
        ...     query_config_dict = {
        ...        #'pipeline_root' : 'BC_DTW'
        ...     }
        ...     data = dict(
        ...         query_annot_uuid_list=quuid_list, database_annot_uuid_list=duuid_list,
        ...         query_config_dict=query_config_dict,
        ...     )
        ...     jobid = web_ibs.send_wbia_request('/api/engine/query/graph/', **data)
        ...     print('jobid = %r' % (jobid,))
        ...     status_response = web_ibs.wait_for_results(jobid)
        ...     result_response = web_ibs.read_engine_results(jobid)
        ...     print('result_response = %s' % (ut.repr3(result_response),))
        ...     inference_result = result_response['json_result']
        ...     if isinstance(inference_result, six.string_types):
        ...        print(inference_result)
        ...     cm_dict = inference_result['cm_dict']
        ...     quuid = quuid_list[0]
        ...     cm = cm_dict[str(quuid)]

    """
    valid_states = {
        'match': ['matched'],  # ['match', 'matched'],
        'nomatch': [
            'notmatched',
            'nonmatch',
        ],  # ['nomatch', 'notmatched', 'nonmatched', 'notmatch', 'non-match', 'not-match'],
        'notcomp': ['notcomparable'],
    }
    prefered_states = ut.take_column(valid_states.values(), 0)
    flat_states = ut.flatten(valid_states.values())

    def sanitize(state):
        state = state.strip().lower()
        state = ''.join(state.split())
        assert state in flat_states, (
            'matching_state_list has unrecognized states. Should be one of %r'
            % (prefered_states,)
        )
        return state

    # HACK
    # if query_annot_uuid_list is None:
    #     if True:
    #         query_annot_uuid_list = []
    #     else:
    #         query_annot_uuid_list = ibs.get_annot_uuids(ibs.get_valid_aids()[0:1])

    dname_list = database_annot_name_list
    qname_list = query_annot_name_list

    # Check inputs
    assert len(query_annot_uuid_list) == 1, (
        'Can only identify one query annotation at a time. Got %d '
        % (len(query_annot_uuid_list),)
    )
    assert (
        database_annot_uuid_list is None or len(database_annot_uuid_list) > 0
    ), 'Cannot specify an empty database_annot_uuid_list (e.g. []), specify or use None instead'

    if qname_list is not None:
        assert len(query_annot_uuid_list) == len(qname_list)
    if database_annot_uuid_list is not None and dname_list is not None:
        assert len(database_annot_uuid_list) == len(dname_list)

    proot = query_config_dict.get('pipeline_root', 'vsmany')
    proot = query_config_dict.get('proot', proot)
    if proot.lower() in (
        'curvrankdorsal',
        'curvrankfinfindrhybriddorsal',
        'curvrankfluke',
    ):
        curvrank_daily_tag = query_config_dict.get('curvrank_daily_tag', '')
        if len(curvrank_daily_tag) > 144:
            message = 'The curvrank_daily_tag cannot have more than 128 characters, please shorten and try again.'
            key = 'query_config_dict["curvrank_daily_tag"]'
            value = curvrank_daily_tag
            raise controller_inject.WebInvalidInput(message, key, value)
    else:
        print(ut.repr3(query_config_dict))
        pop_key_list = ['curvrank_daily_tag']
        for pop_key in pop_key_list:
            print('Popping irrelevant key for config: %r' % (pop_key,))
            query_config_dict.pop(pop_key, None)

    # Check UUIDs
    if dname_list is not None:
        vals = web_check_annot_uuids_with_names(database_annot_uuid_list, dname_list)
        database_annot_uuid_list, dname_list = vals
    ibs.web_check_uuids([], query_annot_uuid_list, database_annot_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)

    qannot_uuid_list = ensure_uuid_list(query_annot_uuid_list)
    dannot_uuid_list = ensure_uuid_list(database_annot_uuid_list)

    # Ensure annotations
    qaid_list = ibs.get_annot_aids_from_uuid(qannot_uuid_list)

    if dannot_uuid_list is None or (
        len(dannot_uuid_list) == 1 and dannot_uuid_list[0] is None
    ):
        qaid = qaid_list[0]
        species = ibs.get_annot_species_texts(qaid)
        assert (
            species != const.UNKNOWN
        ), 'Cannot query an annotation with an empty (unset) species against an unspecified database_annot_uuid_list'
        daid_list = ibs.get_valid_aids()
        daid_list = ibs.filter_annotation_set(daid_list, species=species)
    else:
        daid_list = ibs.get_annot_aids_from_uuid(dannot_uuid_list)

    # Ensure that the daid_list is non-empty after filtering out qaid_list
    daid_remaining_set = set(daid_list) - set(qaid_list)
    if len(daid_remaining_set) == 0:
        raise controller_inject.WebInvalidMatchException(qaid_list, daid_list)

    # Ensure names
    # FIXME: THE QREQ STRUCTURE SHOULD HANDLE NAMES.
    if qname_list is not None:
        # Set names for query annotations
        qnid_list = ibs.add_names(qname_list)
        ibs.set_annot_name_rowids(qaid_list, qnid_list)

    if dname_list is not None:
        # Set names for database annotations
        dnid_list = ibs.add_names(dname_list)
        ibs.set_annot_name_rowids(daid_list, dnid_list)

    # For caching reasons, let's add the qaids to the daids
    if include_qaid_in_daids:
        daid_list = daid_list + qaid_list
        daid_list = sorted(daid_list)

    # Convert annot UUIDs to aids for matching_state_list into user_feedback for query
    state_list = map(sanitize, ut.take_column(matching_state_list, 2))
    # {'aid1': [1], 'aid2': [2], 'p_match': [1.0], 'p_nomatch': [0.0], 'p_notcomp': [0.0]}
    user_feedback = {
        'aid1': ibs.get_annot_aids_from_uuid(ut.take_column(matching_state_list, 0)),
        'aid2': ibs.get_annot_aids_from_uuid(ut.take_column(matching_state_list, 1)),
        'p_match': [
            1.0 if state in valid_states['match'] else 0.0 for state in state_list
        ],
        'p_nomatch': [
            1.0 if state in valid_states['nomatch'] else 0.0 for state in state_list
        ],
        'p_notcomp': [
            1.0 if state in valid_states['notcomp'] else 0.0 for state in state_list
        ],
    }

    ibs.assert_valid_aids(
        qaid_list, msg='error in start_identify qaids', auuid_list=qannot_uuid_list
    )
    ibs.assert_valid_aids(
        daid_list, msg='error in start_identify daids', auuid_list=dannot_uuid_list
    )
    args = (qaid_list, daid_list, user_feedback, query_config_dict, echo_query_params)
    jobid = ibs.job_manager.jobiface.queue_job(
        'query_chips_graph', callback_url, callback_method, *args
    )
    return jobid


@register_ibs_method
@register_api('/api/engine/wic/cnn/', methods=['POST'])
def start_wic_image(
    ibs, image_uuid_list, callback_url=None, callback_method=None, **kwargs
):
    """
    REST:
        Method: GET
        URL: /api/engine/wic/cnn/

    Args:
        image_uuid_list (list) : list of image uuids to detect on.
        callback_url (url) : url that will be called when detection succeeds or fails
    """
    # Check UUIDs
    ibs.web_check_uuids(image_uuid_list=image_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)
    image_uuid_list = ensure_uuid_list(image_uuid_list)
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    args = (
        gid_list,
        kwargs,
    )
    jobid = ibs.job_manager.jobiface.queue_job(
        'wic_cnn_json', callback_url, callback_method, *args
    )

    # if callback_url is not None:
    #    #import requests
    #    #requests.
    #    #callback_url
    return jobid


@register_ibs_method
@register_api('/api/engine/detect/cnn/yolo/', methods=['POST'])
def start_detect_image_yolo(
    ibs, image_uuid_list, callback_url=None, callback_method=None, **kwargs
):
    """
    REST:
        Method: GET
        URL: /api/engine/detect/cnn/yolo/

    Args:
        image_uuid_list (list) : list of image uuids to detect on.
        callback_url (url) : url that will be called when detection succeeds or fails
    """
    # Check UUIDs
    ibs.web_check_uuids(image_uuid_list=image_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)
    image_uuid_list = ensure_uuid_list(image_uuid_list)
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    args = (
        gid_list,
        kwargs,
    )
    jobid = ibs.job_manager.jobiface.queue_job(
        'detect_cnn_yolo_json', callback_url, callback_method, *args
    )

    # if callback_url is not None:
    #    #import requests
    #    #requests.
    #    #callback_url
    return jobid


@register_ibs_method
@register_api('/api/engine/labeler/cnn/', methods=['POST'])
def start_labeler_cnn(
    ibs, annot_uuid_list, callback_url=None, callback_method=None, **kwargs
):
    # Check UUIDs
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)
    annot_uuid_list = ensure_uuid_list(annot_uuid_list)
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    args = (
        aid_list,
        kwargs,
    )
    jobid = ibs.job_manager.jobiface.queue_job(
        'labeler_cnn', callback_url, callback_method, *args
    )

    # if callback_url is not None:
    #    #import requests
    #    #requests.
    #    #callback_url
    return jobid


@register_ibs_method
@register_api('/api/engine/review/query/chip/best/', methods=['POST'])
def start_review_query_chips_best(
    ibs, annot_uuid, callback_url=None, callback_method=None, **kwargs
):
    annot_uuid_list = [annot_uuid]

    # Check UUIDs
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)
    annot_uuid_list = ensure_uuid_list(annot_uuid_list)
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    aid = aid_list[0]
    args = (aid,)
    jobid = ibs.job_manager.jobiface.queue_job(
        'review_query_chips_best', callback_url, callback_method, *args
    )

    # if callback_url is not None:
    #    #import requests
    #    #requests.
    #    #callback_url
    return jobid


@register_ibs_method
@register_api('/test/engine/detect/cnn/yolo/', methods=['GET'])
def start_detect_image_test_yolo(ibs):
    from random import shuffle  # NOQA

    gid_list = ibs.get_valid_gids()
    shuffle(gid_list)
    gid_list = gid_list[:3]
    image_uuid_list = ibs.get_image_uuids(gid_list)
    jobid = ibs.start_detect_image_yolo(image_uuid_list)
    return jobid


@register_ibs_method
@register_api('/api/engine/detect/cnn/lightnet/', methods=['POST', 'GET'])
def start_detect_image_lightnet(
    ibs, image_uuid_list, callback_url=None, callback_method=None, **kwargs
):
    """
    REST:
        Method: GET/api/engine/detect/cnn/lightnet/
        URL:

    Args:
        image_uuid_list (list) : list of image uuids to detect on.
        callback_url (url) : url that will be called when detection succeeds or fails
    """
    # Check UUIDs
    ibs.web_check_uuids(image_uuid_list=image_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)
    image_uuid_list = ensure_uuid_list(image_uuid_list)
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    args = (
        gid_list,
        kwargs,
    )
    jobid = ibs.job_manager.jobiface.queue_job(
        'detect_cnn_lightnet_json', callback_url, callback_method, *args
    )

    # if callback_url is not None:
    #    #import requests
    #    #requests.
    #    #callback_url
    return jobid


@register_ibs_method
@register_api('/test/engine/detect/cnn/lightnet/', methods=['GET'])
def start_detect_image_test_lightnet(ibs):
    from random import shuffle  # NOQA

    gid_list = ibs.get_valid_gids()
    shuffle(gid_list)
    gid_list = gid_list[:3]
    image_uuid_list = ibs.get_image_uuids(gid_list)
    jobid = ibs.start_detect_image_lightnet(image_uuid_list)
    return jobid


@register_ibs_method
@register_api('/api/engine/classify/whaleshark/injury/', methods=['POST'])
def start_predict_ws_injury_interim_svm(
    ibs, annot_uuid_list, callback_url=None, callback_method=None, **kwargs
):
    """
    REST:
        Method: POST
        URL: /api/engine/classify/whaleshark/injury/

    Args:
        annot_uuid_list (list) : list of annot uuids to detect on.
        callback_url (url) : url that will be called when detection succeeds or fails

    CommandLine:
        python -m wbia.web.apis_engine start_predict_ws_injury_interim_svm


    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.apis_engine import *  # NOQA
        >>> from wbia.web import apis_engine
        >>> import wbia
        >>> ibs, qaids, daids = wbia.testdata_expanded_aids(
        >>>     defaultdb='WS_ALL', a=['default:qsize=2,dsize=10'])
        >>> annot_uuid_list = ibs.get_annot_uuids(qaids)
        >>> ibs.initialize_job_manager()
        >>> jobid = ibs.start_predict_ws_injury_interim_svm(annot_uuid_list)
        >>> result = ibs.wait_for_job_result(jobid, timeout=None, freq=2)
        >>> print(result)
        >>> import utool as ut
        >>> #print(ut.to_json(result))
        >>> ibs.close_job_manager()
    """
    # Check UUIDs
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)

    # import wbia
    # from wbia.web import apis_engine
    # ibs.load_plugin_module(apis_engine)
    annot_uuid_list = ensure_uuid_list(annot_uuid_list)
    annots = ibs.annots(uuids=annot_uuid_list)
    args = (annots.aids,)
    jobid = ibs.job_manager.jobiface.queue_job(
        'predict_ws_injury_interim_svm', callback_url, callback_method, *args
    )

    # if callback_url is not None:
    #    #import requests
    #    #requests.
    #    #callback_url
    return jobid


@register_ibs_method
@register_api('/api/engine/query/web/', methods=['GET'])
def start_web_query_all(ibs):
    """
    REST:
        Method: GET
        URL: /api/engine/query/web/
    """
    jobid = ibs.job_manager.jobiface.queue_job('load_identification_query_object_worker')
    return jobid


@register_ibs_method
@register_api('/api/engine/flukebook/sync/', methods=['GET'])
def start_flukebook_sync(ibs, **kwargs):
    """
    REST:
        Method: GET
        URL: /api/engine/flukebook/sync/
    """
    jobid = ibs.job_manager.jobiface.queue_job('flukebook_sync')
    return jobid


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.web.apis_engine
        python -m wbia.web.apis_engine --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
