from __future__ import absolute_import, division, print_function
import six
import utool as ut
import numpy as np
from six.moves import filter
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[special_query]')


def choose_vsmany_K(num_names, qaids, daids):
    """
    method for choosing K in the initial vsmany queries

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> from ibeis.all_imports import *  # NOQA
        >>> ibs, valid_aids = testdata_special_query()
        >>> num_names = np.arange(0, 1000)
        >>> num_names_slope = .1
        >>> K_min / num_names_slope
        >>> K_max / num_names_slope
        >>> K_list = np.floor(num_names_slope * num_names)
        >>> K_list[K_list > 10] = 10
        >>> K_list[K_list < 1] = 1
        >>> pt.plot2(num_names, K_list, x_label='num_names', y_label='K', equal_aspect=False, marker='-')
        >>> pt.update()
    """
    #K = ibs.cfg.query_cfg.nn_cfg.K
    # TODO: paramaterize in config
    num_names_slope = .1  # increase K every fifty names
    K_max = 10
    K_min = 1
    num_names_lower = K_min / num_names_slope
    num_names_upper = K_max / num_names_slope
    if num_names < num_names_lower:
        K = K_min
    elif num_names < num_names_upper:
        K = num_names_slope * num_names
    else:
        K  = K_max

    if len(ut.intersect_ordered(qaids, daids)) > 0:
        # if self is in query bump k
        K += 1
    return K


@profile
def query_vsmany_initial(ibs, qaids, daids, use_cache=True):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (?):
        daids (?):
        use_cache (bool):

    Returns:
        tuple: (newfsv_list, newscore_aids)

    CommandLine:
        python -m ibeis.model.hots.special_query --test-query_vsmany_initial

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> ibs, valid_aids = testdata_special_query()
        >>> qaids = valid_aids[0:1]
        >>> daids = valid_aids[1:]
        >>> use_cache = False
        >>> # execute function
        >>> qaid2_qres_vsmany, qreq_vsmany_ = query_vsmany_initial(ibs, qaids, daids, use_cache)
        >>> qres_vsmany = qaid2_qres_vsmany[qaids[0]]
        >>> # verify results
        >>> result = qres_vsmany.get_top_aids(ibs=ibs, name_scoring=True).tolist()
        >>> print(result)
        [2, 6, 4]
    """
    num_names = len(set(ibs.get_annot_nids(daids)))
    vsmany_cfgdict = {
        #'pipeline_root': 'vsmany',
        'K': choose_vsmany_K(num_names, qaids, daids),
        'index_method': 'multi',
        'return_expanded_nns': True
    }
    qaid2_qres_vsmany, qreq_vsmany_ = ibs._query_chips4(
        qaids, daids, cfgdict=vsmany_cfgdict, return_request=True, use_cache=use_cache)
    isnsum = qreq_vsmany_.qparams.score_method == 'nsum'
    assert isnsum
    assert qreq_vsmany_.qparams.pipeline_root != 'vsone'
    return qaid2_qres_vsmany, qreq_vsmany_


def build_vsone_shortlist(ibs, qaid2_qres_vsmany):
    """
    looks that the top N names in a vsmany query to apply vsone reranking

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid2_qres_vsmany (dict):  dict of query result objects

    Returns:
        list: vsone_query_pairs

    CommandLine:
        python -m ibeis.model.hots.special_query --test-build_vsone_shortlist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> ibs, valid_aids = testdata_special_query()
        >>> qaids = valid_aids[0:1]
        >>> daids = valid_aids[1:]
        >>> qaid2_qres_vsmany, qreq_vsmany_ = query_vsmany_initial(ibs, qaids, daids)
        >>> # execute function
        >>> vsone_query_pairs = build_vsone_shortlist(ibs, qaid2_qres_vsmany)
        >>> # verify results
        >>> result = str(vsone_query_pairs)
        >>> print(result)
        [(1, [2, 3, 5, 6, 4])]
    """
    vsone_query_pairs = []
    nNameShortlistVsone = 3
    for qaid, qres_vsmany in six.iteritems(qaid2_qres_vsmany):
        sorted_nids, sorted_nscores = qres_vsmany.get_sorted_nids_and_scores(ibs=ibs)
        top_nids = ut.listclip(sorted_nids, nNameShortlistVsone)
        # get top annotations beloning to the database query
        # TODO: allow annots not in daids to be included
        top_unflataids = ibs.get_name_aids(top_nids, enable_unknown_fix=True)
        flat_top_aids = ut.flatten(top_unflataids)
        top_aids = ut.intersect_ordered(flat_top_aids, qres_vsmany.daids)
        vsone_query_pairs.append((qaid, top_aids))
    print('built %d pairs' % (len(vsone_query_pairs),))
    return vsone_query_pairs


@profile
def query_vsone_pairs(ibs, vsone_query_pairs, use_cache):
    """
    does vsone queries to rerank the top few vsmany querys

    Returns:
        tuple: qaid2_qres_vsone, qreq_vsone_

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> ibs, valid_aids = testdata_special_query()
        >>> qaids = valid_aids[0:1]
        >>> daids = valid_aids[1:]
        >>> qaid = qaids[0]
        >>> filtkey = 'distinctivness'
        >>> use_cache = False
        >>> # execute function
        >>> qaid2_qres_vsmany, qreq_vsmany_ = query_vsmany_initial(ibs, qaids, daids, use_cache)
        >>> vsone_query_pairs = build_vsone_shortlist(ibs, qaid2_qres_vsmany)
        >>> qaid2_qres_vsone, qreq_vsone_ = query_vsone_pairs(ibs, vsone_query_pairs, use_cache)
        >>> qres_vsone = qaid2_qres_vsone[qaid]
        >>> result = qres_vsone.get_top_aids(ibs=ibs, name_scoring=True).tolist()
        >>> print(result)
        [2, 5]

    """
    vsone_cfgdict = dict(codename='vsone_unnorm')
    #------------------------
    # METHOD 1:
    qaid2_qres_vsone = {}
    for qaid, top_aids in vsone_query_pairs:
        # Perform a query request for each
        qaid2_qres_vsone_, __qreq_vsone_ = ibs._query_chips4(
            [qaid], top_aids, cfgdict=vsone_cfgdict, return_request=True,
            use_cache=use_cache)
        qaid2_qres_vsone.update(qaid2_qres_vsone_)
    #------------------------
    # METHOD 2:
    # doesn't work because daids are not the same for each run
    #qaid2_qres_vsone_, vsone_qreq_ = ibs._query_chips4(
    #    [qaid], top_aids, cfgdict=vsone_cfgdict, return_request=True,
    #    use_cache=use_cache)
    #------------------------
    # Create pseudo query request because there is no good way to
    # represent the vsone reranking as a single query request and
    # we need one for the score normalizer
    pseudo_vsone_cfgdict = dict(codename='vsone_norm')
    pseudo_qaids = ut.get_list_column(vsone_query_pairs, 0)
    pseudo_daids = ut.unique_ordered(ut.flatten(ut.get_list_column(vsone_query_pairs, 1)))
    pseudo_qreq_vsone_ = ibs.new_query_request(pseudo_qaids, pseudo_daids,
                                               cfgdict=pseudo_vsone_cfgdict,
                                               verbose=ut.VERBOSE)
    qreq_vsone_ = pseudo_qreq_vsone_
    # Hack in a special config name
    qreq_vsone_.qparams.query_cfgstr = '_special' + qreq_vsone_.qparams.query_cfgstr
    return qaid2_qres_vsone, qreq_vsone_


@profile
def get_new_qres_distinctiveness(qres_vsone, qres_vsmany, top_aids, filtkey):
    """
    gets the distinctivenss score from vsmany and applies it to vsone

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> ibs, valid_aids = testdata_special_query()
        >>> qaids = valid_aids[0:1]
        >>> daids = valid_aids[1:]
        >>> qaid = qaids[0]
        >>> filtkey = 'distinctivness'
        >>> use_cache = False
        >>> # execute function
        >>> qaid2_qres_vsmany, qreq_vsmany_ = query_vsmany_initial(ibs, qaids, daids, use_cache)
        >>> vsone_query_pairs = build_vsone_shortlist(ibs, qaid2_qres_vsmany)
        >>> qaid2_qres_vsone, qreq_vsone_ = query_vsone_pairs(ibs, vsone_query_pairs, use_cache)
        >>> qreq_vsone_.load_score_normalizer()
        >>> qres_vsone = qaid2_qres_vsone[qaid]
        >>> qres_vsmany = qaid2_qres_vsmany[qaid]
        >>> top_aids = vsone_query_pairs[0][1]
        >>> # verify results
        >>> newfsv_list, newscore_aids = get_new_qres_distinctiveness(qres_vsone, qres_vsmany, top_aids, filtkey)
    """
    newfsv_list = []
    newscore_aids = []

    for daid in top_aids:
        # Distinctiveness is mostly independent of the vsmany database results
        if daid not in qres_vsone.aid2_fm:  # or daid not in qres_vsmany.aid2_fm):
            # no matches to work with
            continue
        scorex_vsone  = ut.listfind(qres_vsone.filtkey_list, filtkey)
        if scorex_vsone is None:
            shape = (qres_vsone.aid2_fsv[daid].shape[0], 1)
            new_filtkey_list = qres_vsone.filtkey_list[:]
            new_scores_vsone = np.full(shape, np.nan)
            #new_scores_vsone = np.ones(shape)
            new_fsv_vsone = np.hstack((qres_vsone.aid2_fsv[daid], new_scores_vsone))
            new_filtkey_list.append(filtkey)
            assert len(new_filtkey_list) == len(new_fsv_vsone.T), 'filter length is not consistent'
            new_scores = new_fsv_vsone.T[-1].T
        fm_vsone  = qres_vsone.aid2_fm[daid]
        qfx_vsone = fm_vsone.T[0]
        # Get the distinctiveness score from the neighborhood
        # around each query point in the vsmany query result
        norm_dist = qres_vsmany.qfx2_dist.T[-1].take(qfx_vsone)
        p = 1.0  # expondent to augment distinctivness scores.  # TODO: paramaterize
        distinctiveness_scores = norm_dist ** p
        new_scores[:] = distinctiveness_scores  #
        newfsv_list.append(new_fsv_vsone)
        newscore_aids.append(daid)
    return newfsv_list, newscore_aids


@profile
def apply_new_qres_filter_scores(qreq_vsone_, qres_vsone, newfsv_list, newscore_aids, filtkey):
    r"""
    applies the new filter scores vectors to a query result and updates other
    scores

    Args:
        qres_vsone (QueryResult):  object of feature correspondences and scores
        newfsv_list (list):
        newscore_aids (?):
        filtkey (?):

    CommandLine:
        python -m ibeis.model.hots.special_query --test-apply_new_qres_filter_scores

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.special_query import *  # NOQA
        >>> ibs, valid_aids = testdata_special_query()
        >>> qaids = valid_aids[0:1]
        >>> daids = valid_aids[1:]
        >>> qaid = qaids[0]
        >>> filtkey = 'distinctivness'
        >>> use_cache = False
        >>> # execute function
        >>> qaid2_qres_vsmany, qreq_vsmany_ = query_vsmany_initial(ibs, qaids, daids, use_cache)
        >>> vsone_query_pairs = build_vsone_shortlist(ibs, qaid2_qres_vsmany)
        >>> qaid2_qres_vsone, qreq_vsone_ = query_vsone_pairs(ibs, vsone_query_pairs, use_cache)
        >>> qreq_vsone_.load_score_normalizer()
        >>> qres_vsone = qaid2_qres_vsone[qaid]
        >>> qres_vsmany = qaid2_qres_vsmany[qaid]
        >>> top_aids = vsone_query_pairs[0][1]
        >>> # verify results
        >>> newfsv_list, newscore_aids = get_new_qres_distinctiveness(qres_vsone, qres_vsmany, top_aids, filtkey)
        >>> apply_new_qres_filter_scores(qres_vsone, newfsv_list, newscore_aids, filtkey)
        >>> # verify results
        >>> print(result)

    Ignore:
        qres_vsone.show_top(ibs, name_scoring=True)
        print(qres_vsone.get_inspect_str(ibs=ibs, name_scoring=True))

        print(qres_vsmany.get_inspect_str(ibs=ibs, name_scoring=True))

    """
    assert ut.listfind(qres_vsone.filtkey_list, filtkey) is None
    # HACK to update result cfgstr
    qres_vsone.filtkey_list.append(filtkey)
    qres_vsone.cfgstr = qreq_vsone_.get_cfgstr()
    # Find positions of weight filters and score filters
    # so we can apply a weighted average
    #numer_filters  = ['lnbnn', 'ratio']
    def index_partition(item_list, part1_items):
        """
        returns two lists. The first are the indecies of items in item_list that
        are in part1_items. the second is the indicies in item_list that are not
        in part1_items. items in part1_items that are not in item_list are
        ignored
        """
        part1_indexes = np.array([item_list.index(item)
                                  for item in part1_items
                                  if item in item_list])
        part2_indexes = np.setdiff1d(np.arange(len(item_list)), part1_indexes)
        return part1_indexes, part2_indexes

    weight_filters = ['fg', 'distinctivness']
    weight_filtxs, nonweight_filtxs = index_partition(qres_vsone.filtkey_list, weight_filters)

    def weighted_average_scoring(new_fsv_vsone, weight_filtxs, nonweight_filtxs):
        r"""
        does \frac{\sum_i w^f_i * w^d_i * r_i}{\sum_i w^f_i, w^d_i}
        to get a weighed average of ratio scores

        If we normalize the weight part to sum to 1 then we can get per-feature
        scores.

        References:
            http://en.wikipedia.org/wiki/Weighted_arithmetic_mean

        Ignore:
            # Show that the formulat is the same
            new_fsv_vsone_numer = np.multiply(weight_fs, nonweight_fs)
            new_fsv_vsone_denom = weight_fs
            assert new_fs_vsone.sum() == new_fsv_vsone_numer.sum() / new_fsv_vsone_denom.sum()
        """
        weight_fs    = new_fsv_vsone.T.take(weight_filtxs, axis=0).T.prod(axis=1)
        nonweight_fs = new_fsv_vsone.T.take(nonweight_filtxs, axis=0).T.prod(axis=1)
        weight_fs_norm01 = weight_fs / weight_fs.sum()
        new_fs_vsone = np.multiply(nonweight_fs, weight_fs_norm01)
        return new_fs_vsone

    def product_scoring(new_fsv_vsone):
        """ product of all weights """
        new_fs_vsone = new_fsv_vsone.prod(axis=1)
        return new_fs_vsone

    for new_fsv_vsone, daid in zip(newfsv_list, newscore_aids):
        #scorex_vsone  = ut.listfind(qres_vsone.filtkey_list, filtkey)
        #if scorex_vsone is None:
        # TODO: add spatial verification as a filter score
        # augment the vsone scores
        # TODO: paramaterize
        weighted_ave_score = True
        if weighted_ave_score:
            # weighted average scoring
            new_fs_vsone = weighted_average_scoring(new_fsv_vsone, weight_filtxs, nonweight_filtxs)
        else:
            # product scoring
            new_fs_vsone = product_scoring(new_fsv_vsone)
        new_score_vsone = new_fs_vsone.sum()
        qres_vsone.aid2_fsv[daid]   = new_fsv_vsone
        qres_vsone.aid2_fs[daid]    = new_fs_vsone
        qres_vsone.aid2_score[daid] = new_score_vsone
        # FIXME: this is not how to compute new probability
        #if qres_vsone.aid2_prob is not None:
        #    qres_vsone.aid2_prob[daid] = qres_vsone.aid2_score[daid]

    # This is how to compute new probability
    if qreq_vsone_.qparams.score_normalization:
        normalizer = qreq_vsone_.normalizer
        daid2_score = qres_vsone.aid2_score
        score_list = list(six.itervalues(daid2_score))
        daid_list  = list(six.iterkeys(daid2_score))
        prob_list = normalizer.normalize_score_list(score_list)
        daid2_prob = dict(zip(daid_list, prob_list))
        qres_vsone.aid2_prob = daid2_prob


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
        >>> ibs, valid_aids = testdata_special_query()
        >>> # execute function
        >>> (qaid2_qres, qreq_) = empty_query(ibs, qaids)
        >>> # verify results
        >>> result = str((qaid2_qres, qreq_))
        >>> print(result)
        >>> qres = qaid2_qres[1]
        >>> qres.ishow_top(ibs, update=True, make_figtitle=True, show_query=True, sidebyside=False)
    """
    daids = []
    qreq_ = ibs.new_query_request(qaids, daids)
    qres_list = qreq_.make_empty_query_results()
    for qres in qres_list:
        qres.aid2_score = {}
    qaid2_qres = dict(zip(qaids, qres_list))
    return qaid2_qres, qreq_


#@ut.indent_func
@profile
def query_vsone_verified(ibs, qaids, daids):
    """
    main special query

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
        >>> ibs, valid_aids = testdata_special_query('PZ_MTEST')
        >>> qaids = valid_aids[0:1]
        >>> daids = valid_aids[1:]
        >>> qaid = qaids[0]
        >>> # execute function
        >>> qaid2_qres, qreq_ = query_vsone_verified(ibs, qaids, daids)
        >>> qres = qaid2_qres[qaid]

    Ignore:
        from ibeis.model.hots import score_normalization

        qres = qaid2_qres_vsmany[qaid]

        ibs.delete_qres_cache()
        qres = qaid2_qres[qaid]
        qres.show_top(ibs, update=True, name_scoring=True)

        qres_vsmany = qaid2_qres_vsmany[qaid]
        qres_vsmany.show_top(ibs, update=True, name_scoring=True)

        qres_vsone = qaid2_qres_vsone[qaid]
        qres_vsone.show_top(ibs, update=True, name_scoring=True)

    """
    if len(daids) == 0:
        return empty_query(ibs, qaids)
    use_cache = True
    use_cache = False

    # vs-many initial scoring
    print('issuing vsmany part')
    qaid2_qres_vsmany, qreq_vsmany_ = query_vsmany_initial(ibs, qaids, daids, use_cache=use_cache)
    print('finished vsmany part')
    #qreq_vsmany_.qparams.prescore_method

    # build vs one list
    print('[query_vsone_verified] building vsone pairs')
    vsone_query_pairs = build_vsone_shortlist(ibs, qaid2_qres_vsmany)

    # vs-one reranking
    print('running vsone queries')
    qaid2_qres_vsone, qreq_vsone_ = query_vsone_pairs(ibs, vsone_query_pairs, use_cache)
    # hack in score normalization
    if qreq_vsone_.qparams.score_normalization:
        qreq_vsone_.load_score_normalizer()

    # AUGMENT VSONE QUERIES (BIG HACKS AFTER THIS POINT)
    # Apply vsmany distinctiveness scores to vsone
    for qaid, top_aids in vsone_query_pairs:
        qres_vsone = qaid2_qres_vsone[qaid]
        qres_vsmany = qaid2_qres_vsmany[qaid]
        #with ut.EmbedOnException():
        if len(top_aids) == 0:
            print('Warning: top_aids is len 0')
            qaid = qres_vsmany.qaid
            continue
        qres_vsone.assert_self()
        qres_vsmany.assert_self()
        filtkey = 'distinctiveness'
        newfsv_list, newscore_aids = get_new_qres_distinctiveness(
            qres_vsone, qres_vsmany, top_aids, filtkey)
        apply_new_qres_filter_scores(
            qreq_vsone_, qres_vsone, newfsv_list, newscore_aids, filtkey)

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
    qreq_ = qreq_vsone_
    all_failed_qres = all([qres is None for qres in six.itervalues(qaid2_qres)])
    any_failed_qres = any([qres is None for qres in six.itervalues(qaid2_qres)])
    if any_failed_qres:
        assert all_failed_qres, "Needs to finish implemetation"
        return empty_query(ibs, qaids)
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


def testdata_special_query(dbname='testdb1'):
    import ibeis
    from ibeis import constants as const
    # build test data
    ibs = ibeis.opendb(dbname)
    #ibs = ibeis.opendb('PZ_MTEST')
    valid_aids = ibs.get_valid_aids(species=const.Species.ZEB_PLAIN)
    return ibs, valid_aids


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
