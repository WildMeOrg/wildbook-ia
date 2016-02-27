# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from ibeis.control import accessor_decors, controller_inject
from ibeis.algo.hots import pipeline
import utool as ut


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)


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
        python -m ibeis.control.IBEISControl --test-query_chips_simple_dict:0
        python -m ibeis.control.IBEISControl --test-query_chips_simple_dict:1

        python -m ibeis.control.IBEISControl --test-query_chips_simple_dict:0 --humpbacks

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> #qaid = ibs.get_valid_aids()[0:3]
        >>> qaids = ibs.get_valid_aids()
        >>> daids = ibs.get_valid_aids()
        >>> dict_list = ibs.query_chips_simple_dict(qaids, daids, return_cm=True)
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
                return_cm=None,
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
        use_bigcache (bool): turns one/off chunked chip match cache (default: True)
        qreq_ (QueryRequest): optional, a QueryRequest object that
            overrides all previous settings
        return_request (bool): returns the request which will be created if
            one is not already specified
        verbose (bool): default=False, turns on verbose printing
        return_cm (bool): default=True, if true converts QueryResult
            objects into serializable ChipMatch objects (in the future
            this will be defaulted to True)

    Returns:
        list: a list of ChipMatch objects containing the matching
            annotations, scores, and feature matches

    Returns(2):
        tuple: (cm_list, qreq_) - a list of query results and optionally the QueryRequest object used

    CommandLine:
        python -m ibeis.control.IBEISControl --test-query_chips

        # Test speed of single query
        python -m ibeis --tf IBEISController.query_chips --db PZ_Master1 \
            -a default:qindex=0:1,dindex=0:500 --nocache-hs

        python -m ibeis --tf IBEISController.query_chips --db PZ_Master1 \
            -a default:qindex=0:1,dindex=0:3000 --nocache-hs

    RESTful:
        Method: PUT
        URL:    /api/query/chips/

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> qreq_ = ibeis.testdata_qreq_()
        >>> ibs = qreq_.ibs
        >>> cm_list = ibs.query_chips(qreq_=qreq_)
        >>> cm = cm_list[0]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()
    """

    if return_cm is None:
        return_cm = True
    # The qaid and daid objects are allowed to be None if qreq_ is
    # specified
    if qaid_list is None:
        qaid_list = qreq_.get_external_qaids()
    if daid_list is None:
        if qreq_ is not None:
            daid_list = qreq_.get_external_daids()
        else:
            daid_list = ibs.get_valid_aids()

    qaid_list, was_scalar = ut.wrap_iterable(qaid_list)

    # Wrapped call to the main entrypoint in the API to the hotspotter
    # pipeline
    qaid2_cm, qreq_ = ibs._query_chips4(
        qaid_list, daid_list, cfgdict=cfgdict, use_cache=use_cache,
        use_bigcache=use_bigcache, qreq_=qreq_,
        return_request=True, verbose=verbose,
        save_qcache=save_qcache,
        prog_hook=prog_hook)

    # Return a list of query results instead of that awful dictionary
    # that will be depricated in future version of hotspotter.
    cm_list = [qaid2_cm[qaid] for qaid in qaid_list]

    if return_cm or return_cm_dict or return_cm_simple_dict:
        # Convert to cm_list
        if return_cm_simple_dict:
            for cm in cm_list:
                cm.qauuid = ibs.get_annot_uuids(cm.qaid)
                cm.dauuid_list = ibs.get_annot_uuids(cm.daid_list)
            keys = ['qauuid', 'dauuid_list']
            cm_list = [cm.as_simple_dict(keys) for cm in cm_list]
        elif return_cm_dict:
            cm_list = [cm.as_dict() for cm in cm_list]
        else:
            cm_list = cm_list
    #else:
    #    cm_list = [
    #        cm.as_qres2(qreq_)
    #        for cm in cm_list
    #    ]

    if was_scalar:
        # hack for scalar input
        assert len(cm_list) == 1
        cm_list = cm_list[0]

    if return_request:
        return cm_list, qreq_
    else:
        return cm_list


@register_ibs_method
@register_api('/api/query/chips4/', methods=['PUT'])
def _query_chips4(ibs, qaid_list, daid_list,
                  use_cache=None,
                  use_bigcache=None,
                  return_request=False,
                  cfgdict=None,
                  qreq_=None,
                  verbose=pipeline.VERB_PIPELINE,
                  save_qcache=None,
                  prog_hook=None):
    """
    submits a query request
    main entrypoint in the IBIES API to the hotspotter pipeline

    CommandLine:
        python -m ibeis.control.IBEISControl --test-_query_chips4 --show

    RESTful:
        Method: PUT
        URL:    /api/query/chips4/

    Example:
        >>> # SLOW_DOCTEST
        >>> #from ibeis.all_imports import *  # NOQA
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> ibs = ibeis.test_main(db='testdb1')
        >>> qreq_ = ibs.new_query_request(qaid_list, daid_list)
        >>> cm = ibs._query_chips4(qaid_list, daid_list, use_cache=False)[1]
        >>> ut.quit_if_noshow()
        >>> cm.ishow_analysis(qreq_)
        >>> ut.show_if_requested()
    """
    from ibeis.algo.hots import match_chips4 as mc4
    # Check fo empty queries
    try:
        assert len(daid_list) > 0, 'there are no database chips'
        assert len(qaid_list) > 0, 'there are no query chips'
    except AssertionError as ex:
        ut.printex(ex, 'Impossible query request', iswarning=True,
                   keys=['qaid_list', 'daid_list'])
        if ut.SUPER_STRICT:
            raise
        qaid2_cm = {qaid: None for qaid in qaid_list}
    else:
        # Check for consistency
        if qreq_ is not None:
            ut.assert_lists_eq(
                qreq_.get_external_qaids(), qaid_list,
                'qaids do not agree with qreq_', verbose=True)
            ut.assert_lists_eq(
                qreq_.get_external_daids(), daid_list,
                'daids do not agree with qreq_', verbose=True)
        if qreq_ is None:
            qreq_ = ibs.new_query_request(qaid_list, daid_list,
                                          cfgdict=cfgdict, verbose=verbose)

        # Send query to hotspotter (runs the query)
        qaid2_cm = mc4.submit_query_request(
            ibs,  qaid_list, daid_list, use_cache, use_bigcache,
            cfgdict=cfgdict, qreq_=qreq_,
            verbose=verbose, save_qcache=save_qcache, prog_hook=prog_hook)

    if return_request:
        return qaid2_cm, qreq_
    else:
        return qaid2_cm


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
