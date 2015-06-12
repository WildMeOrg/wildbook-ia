"""
not really used
most things in here can be depricated
"""
from __future__ import absolute_import, division, print_function
import utool
import utool as ut
from ibeis.experiments import results_organizer
from ibeis.experiments import results_analyzer
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[results_all]', DEBUG=False)


class AllResults(utool.DynStruct):
    """
    Data container for all compiled results
    """
    def __init__(allres):
        super(AllResults, allres).__init__(child_exclude_list=['qaid2_qres'])
        allres.ibs = None
        allres.qaid2_qres = None
        allres.allorg = None
        allres.cfgstr = None
        allres.dbname = None
        allres.qreq_ = None

    def get_orgtype(allres, orgtype):
        orgres = allres.allorg.get(orgtype)
        return orgres

    def get_cfgstr(allres):
        return allres.cfgstr

    def make_title(allres):
        return allres.dbname + '\n' + ut.packstr(allres.get_cfgstr(), textwidth=80, break_words=False, breakchars='_', wordsep='_')

    def get_qres(allres, qaid):
        return allres.qaid2_qres[qaid]

    def get_orgres_desc_match_dists(allres, orgtype_list):
        return results_analyzer.get_orgres_desc_match_dists(allres, orgtype_list)

    def get_orgres_annotationmatch_scores(allres, orgtype_list):
        return results_analyzer.get_orgres_annotationmatch_scores(allres, orgtype_list)


def init_allres(ibs, qaid2_qres, qreq_=None):
    if qreq_ is not None:
        allres_cfgstr = qreq_.get_cfgstr()
    else:
        allres_cfgstr = '???'
    print('Building allres')
    allres = AllResults()
    allres.qaid2_qres = qaid2_qres
    allres.allorg = results_organizer.organize_results(ibs, qaid2_qres)
    allres.cfgstr = allres_cfgstr
    allres.dbname = ibs.get_dbname()
    allres.ibs = ibs
    allres.qreq_ = qreq_
    return allres


# ALL RESULTS CACHE


__ALLRES_CACHE__ = {}
__QRESREQ_CACHE__ = {}


def build_cache_key(ibs, qaid_list, daid_list, cfgdict):
    # a little overconstrained
    cfgstr = ibs.cfg.query_cfg.get_cfgstr()
    query_hashid = ibs.get_annot_hashid_semantic_uuid(qaid_list, prefix='Q')
    data_hashid  = ibs.get_annot_hashid_semantic_uuid(daid_list, prefix='D')
    key = (query_hashid, data_hashid, cfgstr, str(cfgdict))
    return key


def get_qres_and_qreq_(ibs, qaid_list, daid_list=None, cfgdict=None):
    if daid_list is None:
        daid_list = ibs.get_valid_aids()

    qres_cache_key = build_cache_key(ibs, qaid_list, daid_list, cfgdict)

    try:
        (qaid2_qres, qreq_) = __QRESREQ_CACHE__[qres_cache_key]
    except KeyError:
        qaid2_qres, qreq_ = ibs._query_chips4(qaid_list, daid_list,
                                              return_request=True,
                                              cfgdict=cfgdict)
        # Cache save
        __QRESREQ_CACHE__[qres_cache_key] = (qaid2_qres, qreq_)
    return (qaid2_qres, qreq_)


def get_allres(ibs, qaid_list, daid_list=None, cfgdict=None):
    """
    get_allres

    Args:
        ibs (IBEISController):
        qaid_list (int): query annotation id
        daid_list (list):

    Returns:
        AllResults: allres

    Example:
        >>> from dev import *  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_list = ibs.get_valid_aids()
        >>> daid_list = None
        >>> allres = get_allres(ibs, qaid_list, daid_list)
        >>> print(allres)
        >>> allres.allorg['top_true'].printme3()
    """
    print('[dev] get_allres')
    if daid_list is None:
        daid_list = ibs.get_valid_aids()
    allres_key = build_cache_key(ibs, qaid_list, daid_list, cfgdict)
    try:
        allres = __ALLRES_CACHE__[allres_key]
    except KeyError:
        qaid2_qres, qreq_ = get_qres_and_qreq_(ibs, qaid_list, daid_list, cfgdict)
        allres = init_allres(ibs, qaid2_qres, qreq_)
        # Cache save
        __ALLRES_CACHE__[allres_key] = allres
    return allres


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.experiments.results_all; utool.doctest_funcs(ibeis.experiments.results_all, allexamples=True)"
        python -c "import utool, ibeis.experiments.results_all; utool.doctest_funcs(ibeis.experiments.results_all)"
        python -m ibeis.experiments.results_all --allexamples
        python -m ibeis.experiments.results_all --test-learn_score_normalization --enableall
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
    import plottool as pt  # NOQA
    exec(pt.present())
