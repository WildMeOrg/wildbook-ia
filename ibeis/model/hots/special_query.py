from __future__ import absolute_import, division, print_function
import six
import utool as ut
import vtool as vt
import numpy as np
from six.moves import filter
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[special_query]')


def choose_vsmany_K(ibs, qaids, daids):
    K = ibs.cfg.query_cfg.nn_cfg.K
    if len(daids) < 20:
        K = 1
    if len(ut.intersect_ordered(qaids, daids)) > 0:
        # if self is in query bump k
        K += 1
    return K


def build_vsone_shortlist(ibs, qaid2_qres_vsmany):
    vsone_query_pairs = []
    nNameShortlistVsone = 3
    for qaid, qres_vsmany in six.iteritems(qaid2_qres_vsmany):
        sorted_nids, sorted_nscores = qres_vsmany.get_sorted_nids_and_scores(ibs=ibs)
        nNameShortlistVsone = min(len(sorted_nids), nNameShortlistVsone)
        top_nids = sorted_nids[0:nNameShortlistVsone]
        # get top annotations beloning to the database query
        # TODO: allow annots not in daids to be included
        top_unflataids = ibs.get_name_aids(top_nids, enable_unknown_fix=True)
        flat_top_aids = ut.flatten(top_unflataids)
        top_aids = ut.intersect_ordered(flat_top_aids, qres_vsmany.daids)
        vsone_query_pairs.append((qaid, top_aids))
    print('built %d pairs' % (len(vsone_query_pairs),))
    return vsone_query_pairs


def query_vsone_pairs(ibs, vsone_query_pairs, use_cache):
    qaid2_qres_vsone = {}
    for qaid, top_aids in vsone_query_pairs:
        vsone_cfgdict = dict(codename='vsone_norm')
        qaid2_qres_vsone_, qreq_ = ibs._query_chips4(
            [qaid], top_aids, cfgdict=vsone_cfgdict, return_request=True,
            use_cache=use_cache)
        qaid2_qres_vsone.update(qaid2_qres_vsone_)
    return qaid2_qres_vsone


def get_new_qres_filter_scores(qres_vsone, qres_vsmany, top_aids, filtkey):
    """
    applies scores of type ``filtkey`` from qaid2_qres_vsmany to qaid2_qres_vsone
    """
    newfsv_list = []
    newscore_aids = []
    for daid in top_aids:
        if (daid not in qres_vsone.aid2_fm or
             daid not in qres_vsmany.aid2_fm):
            # no matches to work with
            continue
        fm_vsone      = qres_vsone.aid2_fm[daid]
        fm_vsmany     = qres_vsmany.aid2_fm[daid]

        scorex_vsone  = ut.listfind(qres_vsone.filtkey_list, filtkey)
        scorex_vsmany = ut.listfind(qres_vsmany.filtkey_list, filtkey)
        if scorex_vsone is None:
            shape = (qres_vsone.aid2_fsv[daid].shape[0], 1)
            new_filtkey_list = qres_vsone.filtkey_list[:]
            #new_scores_vsone = np.full(shape, np.nan)
            new_scores_vsone = np.ones(shape)
            fsv = np.hstack((qres_vsone.aid2_fsv[daid], new_scores_vsone))
            new_filtkey_list.append(filtkey)
            assert len(new_filtkey_list) == len(fsv.T), 'filter length is not consistent'
            new_fsv_vsone = fsv.T[-1].T
        else:
            assert False, 'scorex_vsone should be None'
            new_fsv_vsone  = qres_vsone.aid2_fsv[daid].T[scorex_vsone].T
        scores_vsmany = qres_vsmany.aid2_fsv[daid].T[scorex_vsmany].T

        # find intersecting matches
        # (should we just take the scores from the pre-spatial verification
        #  part of the pipeline?)
        common, fmx_vsone, fmx_vsmany = vt.intersect2d_numpy(fm_vsone, fm_vsmany, return_indicies=True)
        mutual_scores = scores_vsmany.take(fmx_vsmany)
        new_fsv_vsone[fmx_vsone] = mutual_scores

        newfsv_list.append(new_fsv_vsone)
        newscore_aids.append(daid)
    return newfsv_list, newscore_aids


def apply_new_qres_filter_scores(qres_vsone, newfsv_list, newscore_aids, filtkey):
    assert ut.listfind(qres_vsone.filtkey_list, filtkey) is None
    qres_vsone.filtkey_list.append(filtkey)
    for new_fsv_vsone, daid in zip(newscore_aids, newscore_aids):
        scorex_vsone  = ut.listfind(qres_vsone.filtkey_list, filtkey)
        if scorex_vsone is None:
            # TODO: add spatial verification as a filter score
            # augment the vsone scores
            qres_vsone.aid2_fsv[daid] = new_fsv_vsone
            qres_vsone.aid2_fs[daid] = qres_vsone.aid2_fsv[daid].prod(axis=1)
            qres_vsone.aid2_score[daid] = qres_vsone.aid2_fs[daid].sum()
            if qres_vsone.aid2_prob is not None:
                qres_vsone.aid2_prob[daid] = qres_vsone.aid2_score[daid]


def empty_query(ibs, qaids):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (?):

    Returns:
        tuple: (qaid2_qres, qreq_)

    CommandLine:
        python -m ibeis.model.hots.special_query --test-empty_query

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaids = ibs.get_valid_aids()
        >>> # execute function
        >>> (qaid2_qres, qreq_) = empty_query(ibs, qaids)
        >>> # verify results
        >>> result = str((qaid2_qres, qreq_))
        >>> print(result)
        >>> qres = qaid2_qres[1]
        >>> qres.ishow_top(ibs, update=True, make_figtitle=True, sidebyside=False)
    """
    daids = []
    qreq_ = ibs.new_query_request(qaids, daids)
    qres_list = qreq_.make_empty_query_results()
    for qres in qres_list:
        qres.aid2_score = {}
    qaid2_qres = dict(zip(qaids, qres_list))
    return qaid2_qres, qreq_


#@ut.indent_func
def query_vsone_verified(ibs, qaids, daids):
    """
    A hacked in vsone-reranked pipeline
    Actually just two calls to the pipeline

    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (list):
        daids (list):

    Returns:
        tuple: qaid2_qres, qreq_

    CommandLine:
        python -m ibeis.model.hots.special_query --test-query_vsone_verified

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> valid_aids = ibs.get_valid_aids()
        >>> qaids = valid_aids[0:1]
        >>> daids = valid_aids[1:]
        >>> qaid = qaids[0]
        >>> # execute function
        >>> qaid2_qres, qreq_ = query_vsone_verified(ibs, qaids, daids)
        >>> qres = qaid2_qres[qaid]

    Ignore:
        qres = qaid2_qres_vsmany[qaid]

        qres = qaid2_qres[qaid]
        ibs.delete_qres_cache()
        qres.show_top(ibs, update=True)
    """
    if len(daids) == 0:
        qreq_ = ibs.new_query_request(qaids, daids)
        qres_list = qreq_.make_empty_query_results()
        qaid2_qres = dict(zip(qaids, qres_list))
        return qaid2_qres, qreq_
    use_cache = True
    #use_cache = False
    print('issuing vsmany part')
    cfgdict = {
        #'pipeline_root': 'vsmany',
        'K': choose_vsmany_K(ibs, qaids, daids),
        'index_method': 'multi',
    }
    qaid2_qres_vsmany, qreq_vsmany_ = ibs._query_chips4(
        qaids, daids, cfgdict=cfgdict, return_request=True, use_cache=use_cache)
    print('finished vsmany part')
    isnsum = qreq_vsmany_.qparams.score_method == 'nsum'
    assert isnsum
    assert qreq_vsmany_.qparams.pipeline_root != 'vsone'
    #qreq_vsmany_.qparams.prescore_method

    # build vs one list
    print('[query_vsone_verified] building vsone pairs')
    vsone_query_pairs = build_vsone_shortlist(ibs, qaid2_qres_vsmany)

    # vs-one reranking
    print('running vsone queries')
    qaid2_qres_vsone = query_vsone_pairs(ibs, vsone_query_pairs, use_cache)

    # Apply vsmany distinctiveness scores to vsone
    for qaid, top_aids in vsone_query_pairs:
        filtkey = 'lnbnn'
        qres_vsone = qaid2_qres_vsone[qaid]
        qres_vsmany = qaid2_qres_vsmany[qaid]
        #with ut.EmbedOnException():
        if len(top_aids) == 0:
            print('Warning: top_aids is len 0')
            qaid = qres_vsmany.qaid
            continue
        qres_vsone.assert_self()
        qres_vsmany.assert_self()
        newfsv_list, newscore_aids = get_new_qres_filter_scores(
            qres_vsone, qres_vsmany, top_aids, filtkey)
        apply_new_qres_filter_scores(
            qres_vsone, newfsv_list, newscore_aids, filtkey)

    print('finished vsone queries')
    if ut.VERBOSE:
        for qaid in qaids:
            qres_vsone = qaid2_qres_vsone[qaid]
            qres_vsmany = qaid2_qres_vsmany[qaid]
            if qres_vsmany is not None:
                vsmanyinspectstr = qres_vsmany.get_inspect_str(ibs=ibs, name_scoring=True)
                print(ut.msgblock('VSMANY-INITIAL-RESULT qaid=%r' % (qaid,), vsmanyinspectstr))
            if qres_vsone is not None:
                vsoneinspectstr = qres_vsone.get_inspect_str(ibs=ibs, name_scoring=True)
                print(ut.msgblock('VSONE-VERIFIED-RESULT qaid=%r' % (qaid,), vsoneinspectstr))

    # FIXME: returns the last qreq_. There should be a notion of a query
    # request for a vsone reranked query
    qaid2_qres = qaid2_qres_vsone
    qreq_ = qreq_vsmany_
    return qaid2_qres, qreq_


def test_vsone_verified(ibs):
    """
    hack in vsone-reranking

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> #reload_all()
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> test_vsone_verified(ibs)
    """
    import plottool as pt
    #qaids = ibs.get_easy_annot_rowids()
    nids = ibs.get_valid_nids(filter_empty=True)
    grouped_aids_ = ibs.get_name_aids(nids)
    grouped_aids = list(filter(lambda x: len(x) > 1, grouped_aids_))
    items_list = grouped_aids

    sample_aids = ut.flatten(ut.sample_lists(items_list, num=2, seed=0))
    qaid2_qres, qreq_ = query_vsone_verified(ibs, sample_aids, sample_aids)
    for qres in ut.InteractiveIter(list(six.itervalues(qaid2_qres))):
        pt.close_all_figures()
        fig = qres.ishow_top(ibs)
        fig.show()
    #return qaid2_qres


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.special_query
        python -m ibeis.model.hots.special_query --allexamples
        python -m ibeis.model.hots.special_query --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
