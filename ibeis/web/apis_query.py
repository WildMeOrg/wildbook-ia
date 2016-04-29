# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ibeis.control import accessor_decors, controller_inject
from ibeis.algo.hots import pipeline
from flask import url_for, request, current_app  # NOQA
import utool as ut
import cv2
import dtool


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/query/recognition_query_aids/', methods=['GET'])
def get_recognition_query_aids(ibs, is_known, species=None):
    """
    DEPCIRATE

    RESTful:
        Method: GET
        URL:    /api/query/recognition_query_aids/
    """
    qaid_list = ibs.get_valid_aids(is_known=is_known, species=species)
    return qaid_list


@register_ibs_method
@register_api('/api/query/chips/simple_dict/', methods=['GET'])
def query_chips_simple_dict(ibs, *args, **kwargs):
    r"""
    Runs query_chips, but returns a json compatible dictionary

    Args:
        same as query_chips

    RESTful:
        Method: GET
        URL:    /api/query/chips/simple_dict/

    SeeAlso:
        query_chips

    CommandLine:
        python -m ibeis.web.apis_query --test-query_chips_simple_dict:0
        python -m ibeis.web.apis_query --test-query_chips_simple_dict:1

        python -m ibeis.web.apis_query --test-query_chips_simple_dict:0 --humpbacks

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> #qaid = ibs.get_valid_aids()[0:3]
        >>> qaids = ibs.get_valid_aids()
        >>> daids = ibs.get_valid_aids()
        >>> dict_list = ibs.query_chips_simple_dict(qaids, daids)
        >>> qgids = ibs.get_annot_image_rowids(qaids)
        >>> qnids = ibs.get_annot_name_rowids(qaids)
        >>> for dict_, qgid, qnid in zip(dict_list, qgids, qnids):
        >>>     dict_['qgid'] = qgid
        >>>     dict_['qnid'] = qnid
        >>>     dict_['dgid_list'] = ibs.get_annot_image_rowids(dict_['daid_list'])
        >>>     dict_['dnid_list'] = ibs.get_annot_name_rowids(dict_['daid_list'])
        >>>     dict_['dgname_list'] = ibs.get_image_gnames(dict_['dgid_list'])
        >>>     dict_['qgname'] = ibs.get_image_gnames(dict_['qgid'])
        >>> result  = ut.list_str(dict_list, nl=2, precision=2, hack_liststr=True)
        >>> result = result.replace('u\'', '"').replace('\'', '"')
        >>> print(result)

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import time
        >>> import ibeis
        >>> import requests
        >>> # Start up the web instance
        >>> web_instance = ibeis.opendb_in_background(db='testdb1', web=True, browser=False)
        >>> time.sleep(.5)
        >>> baseurl = 'http://127.0.1.1:5000'
        >>> data = dict(qaid_list=[1])
        >>> resp = requests.get(baseurl + '/api/query/chips/simple_dict/', data=data)
        >>> print(resp)
        >>> web_instance.terminate()
        >>> json_dict = resp.json()
        >>> cmdict_list = json_dict['response']
        >>> assert 'score_list' in cmdict_list[0]

    """
    kwargs['return_cm_simple_dict'] = True
    return ibs.query_chips(*args, **kwargs)


@register_ibs_method
@register_api('/api/query/chips/dict/', methods=['GET'])
def query_chips_dict(ibs, *args, **kwargs):
    """
    Runs query_chips, but returns a json compatible dictionary

    RESTful:
        Method: GET
        URL:    /api/query/chips/dict/
    """
    kwargs['return_cm_dict'] = True
    return ibs.query_chips(*args, **kwargs)


@register_route('/test/review/query/chips/', methods=['GET'])
def review_query_chips_test():
    from ibeis.algo.hots.chip_match import ChipMatch
    from ibeis.algo.hots.query_request import QueryRequest
    ibs = current_app.ibs
    result_list = ibs.query_chips_test()
    result = result_list[0]

    state_dict = result.pop('qreq_').__getstate__()
    cm = ChipMatch(**result)
    qreq_ = QueryRequest()
    qreq_.__setstate__(state_dict)
    aid = cm.get_top_aids()[0]
    image = cm.render_single_annotmatch(qreq_, aid)
    # callback_url = request.args.get('callback_url', url_for('process_detection_html'))
    # callback_method = request.args.get('callback_method', 'POST')
    # template_html = review_detection_html(ibs, image_uuid, result_list, callback_url, callback_method, include_jquery=True)
    # template_html = '''
    #     <script src="http://code.jquery.com/jquery-2.2.1.min.js" ia-dependency="javascript"></script>
    #     %s
    # ''' % (template_html, )
    # return template_html
    return 'done'


@register_ibs_method
@register_api('/test/query/chips/', methods=['GET'])
def query_chips_test(ibs):
    from random import shuffle
    aid_list = ibs.get_valid_aids()
    shuffle(aid_list)
    qaid_list = aid_list[:3]
    daid_list = aid_list[-10:]
    query_resut_list, qreq_ = ibs.query_chips(qaid_list=qaid_list, daid_list=daid_list, return_request=True)
    result_list = [
        {
            'qaid'              : qr.qaid,
            'score_list'        : qr.score_list,
            'daid_list'         : qr.daid_list,
            'annot_score_list'  : qr.annot_score_list,
            'name_score_list'   : qr.name_score_list,
            'fm_list'           : qr.fm_list,
            'fsv_list'          : qr.fsv_list,
            'qreq_'             : qreq_,
        }
        for qr in query_resut_list
    ]
    return result_list


@register_ibs_method
@register_api('/api/query/chips/', methods=['GET'])
def query_chips(ibs, qaid_list=None,
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

    Note:
        In the future the QueryResult objects will be replaced by ChipMatch
        objects

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
        qreq_ (QueryRequest): optional, a QueryRequest object that
            overrides all previous settings
        return_request (bool): returns the request which will be created if
            one is not already specified
        verbose (bool): default=False, turns on verbose printing

    Returns:
        list: a list of ChipMatch objects containing the matching
            annotations, scores, and feature matches

    Returns(2):
        tuple: (cm_list, qreq_) - a list of query results and optionally the
            QueryRequest object used

    RESTful:
        Method: PUT
        URL:    /api/query/chips/

    CommandLine:
        python -m ibeis.web.apis_query --test-query_chips

        # Test speed of single query
        python -m ibeis --tf IBEISController.query_chips --db PZ_Master1 \
            -a default:qindex=0:1,dindex=0:500 --nocache-hs

        python -m ibeis --tf IBEISController.query_chips --db PZ_Master1 \
            -a default:qindex=0:1,dindex=0:3000 --nocache-hs

        python -m ibeis.web.apis_query --test-query_chips:1 --show
        python -m ibeis.web.apis_query --test-query_chips:2 --show

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> qreq_ = ibeis.testdata_qreq_()
        >>> ibs = qreq_.ibs
        >>> cm_list = qreq_.execute()
        >>> cm = cm_list[0]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()

    Example:
        >>> # SLOW_DOCTEST
        >>> #from ibeis.all_imports import *  # NOQA
        >>> import ibeis
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> ibs = ibeis.test_main(db='testdb1')
        >>> qreq_ = ibs.new_query_request(qaid_list, daid_list)
        >>> cm = ibs.query_chips(qaid_list, daid_list, use_cache=False, qreq_=qreq_)[0]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()

    Example1:
        >>> # SLOW_DOCTEST
        >>> #from ibeis.all_imports import *  # NOQA
        >>> import ibeis
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> ibs = ibeis.test_main(db='testdb1')
        >>> cfgdict = {'pipeline_root':'BC_DTW'}
        >>> qreq_ = ibs.new_query_request(qaid_list, daid_list, cfgdict=cfgdict, verbose=True)
        >>> cm = ibs.query_chips(qaid_list, daid_list, use_cache=False, qreq_=qreq_)[0]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()
    """
    from ibeis.algo.hots import match_chips4 as mc4
    # The qaid and daid objects are allowed to be None if qreq_ is
    # specified
    if qaid_list is None:
        qaid_list = qreq_.qaids
    if daid_list is None:
        if qreq_ is not None:
            daid_list = qreq_.daids
        else:
            daid_list = ibs.get_valid_aids()

    qaid_list, was_scalar = ut.wrap_iterable(qaid_list)

    # Check fo empty queries
    try:
        assert len(daid_list) > 0, 'there are no database chips'
        assert len(qaid_list) > 0, 'there are no query chips'
    except AssertionError as ex:
        ut.printex(ex, 'Impossible query request', iswarning=True,
                   keys=['qaid_list', 'daid_list'])
        if ut.SUPER_STRICT:
            raise
        cm_list = [None for qaid in qaid_list]
    else:
        # Check for consistency
        if qreq_ is not None:
            ut.assert_lists_eq(
                qreq_.qaids, qaid_list,
                'qaids do not agree with qreq_', verbose=True)
            ut.assert_lists_eq(
                qreq_.daids, daid_list,
                'daids do not agree with qreq_', verbose=True)
        if qreq_ is None:
            qreq_ = ibs.new_query_request(qaid_list, daid_list,
                                          cfgdict=cfgdict, verbose=verbose)

        if isinstance(qreq_, dtool.BaseRequest):
            # Dtool has a new-ish way of doing requests.  Eventually requests
            # will be depricated and all of this will go away though.
            cm_list = qreq_.execute()
        else:
            # Send query to hotspotter (runs the query)
            cm_list = mc4.submit_query_request(
                ibs,  qaid_list, daid_list, use_cache, use_bigcache,
                cfgdict=cfgdict, qreq_=qreq_,
                verbose=verbose, save_qcache=save_qcache, prog_hook=prog_hook)

    if return_cm_dict or return_cm_simple_dict:
        # Convert to cm_list
        if return_cm_simple_dict:
            for cm in cm_list:
                cm.qauuid = ibs.get_annot_uuids(cm.qaid)
                cm.dauuid_list = ibs.get_annot_uuids(cm.daid_list)
            keys = ['qauuid', 'dauuid_list']
            cm_list = [cm.as_simple_dict(keys) for cm in cm_list]
        elif return_cm_dict:
            cm_list = [cm.as_dict() for cm in cm_list]

    if was_scalar:
        # hack for scalar input
        assert len(cm_list) == 1
        cm_list = cm_list[0]

    if return_request:
        return cm_list, qreq_
    else:
        return cm_list


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
