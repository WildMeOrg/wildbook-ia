from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import utool as ut
import six
from ibeis.dev import results_organizer
from ibeis.dev import results_analyzer
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
    return allres


def learn_score_normalization(ibs, qaid2_qres):
    """
    Args:
        qaid2_qres (int): query annotation id

    Example:
        >>> from ibeis.dev.results_all import *   # NOQA
        >>> from ibeis.dev import results_all
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = daid_list = ibs.get_valid_aids()
        >>> cfgdict = dict(codename='nsum')
        >>> qaid2_qres, qreq_ = results_all.get_qres_and_qreq_(ibs, qaid_list, daid_list, cfgdict)
        >>> results_all.learn_score_normalization(ibs, qaid2_qres)


    valid_aids = ibs.get_valid_aids()
    hard_aids = ut.filter_items(valid_aids, ibs.get_annot_is_hard(valid_aids))
    qaid2_qres = ibs._query_chips4(hard_aids, daid_list, custom_qparams=cfgdict)
    """
    #ut.embed()
    #unflat_xdata = []
    unflat_ydata = []

    import scipy.stats as spstats
    def estimate_pdf(data, bw_factor):
        try:
            data_pdf = spstats.gaussian_kde(data, bw_factor)
            data_pdf.covariance_factor = bw_factor
        except Exception as ex:
            ut.printex(ex, '! Exception while estimating kernel density',
                       keys=['data'])
            raise
        return data_pdf

    scores_given_good_tp = []
    scores_given_good_fp = []
    score_diff_given_good = []

    # http://en.wikipedia.org/wiki/Statistical_hypothesis_testing
    # http://en.wikipedia.org/wiki/Type_I_and_type_II_errors
    # http://en.wikipedia.org/wiki/P-value
    # ftp://ftp.stat.duke.edu/pub/WorkingPapers/10-13.pdf

    for qx, (qaid, qres) in enumerate(six.iteritems(qaid2_qres)):
        if not ibs.get_annot_has_groundtruth(qaid):
            continue
        is_nsum = qres.is_nsum()
        if is_nsum:
            qnid = ibs.get_annot_nids(qaid)
            sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)

            if len(sorted_nids) == 0:
                continue

            if len(sorted_nids) > 3:
                # suppose this score is true.
                # then what should we see
                hypothesis_score = sorted_nscores[0]

                (hypothesis_score - mu) / sigma

                min_sample = 8
                pval_list = [spstats.normaltest(sorted_nscores[ix:])[1]
                             for ix in range(len(sorted_nscores) - min_sample)]

                kurt, pval = spstats.normaltest(sorted_nscores[1:])
                kurt, pval = spstats.normaltest(sorted_nscores)

                if pval < .01:
                    # very strong
                    pass
                elif pval < .05:
                    # strong
                    pass
                elif pval < .1:
                    # low
                    pass
                else:
                    # none
                    pass

                reference_measurements = sorted_nscores[1:]
                mu = reference_measurements.mean()
                sigma = reference_measurements.std()

                std_list  = np.array([sorted_nscores[ix:].std() for ix in range(len(sorted_nscores))])
                mean_list = np.array([sorted_nscores[ix:].mean() for ix in range(len(sorted_nscores))])
                (sorted_nscores - mean_list) / (std_list + 1E-9)
                significance_list = -np.diff(std_list)

            if sorted_nids[0] == qnid:
                pass

            gt_aids = qres.get_groundtruth_aids(ibs)
            gf_aids = qres.get_groundfalse_aids(ibs)
            gt_scores = np.array(qres.get_aid_scores(gt_aids, fillvalue=-1))
            gf_scores = np.array(qres.get_aid_scores(gf_aids, fillvalue=-1))
            #false_nscoretup = get_one_score_per_name(ibs, gf_aids, gf_scores)
            #(sorted_nids, sorted_aids, sorted_nscore, _sorted_scores) = false_nscoretup
            gt_score = gt_scores.max()
            gf_score = gf_scores.max()
            if gt_score > gf_score:
                scores_given_good_tp.append(gt_score)
                scores_given_good_fp.append(gf_score)
                if len(score_diff_given_good) == 0:
                    pass

        if len(gt_scores) == 0:
            pass

    unflat_max_y = map(sorted, unflat_ydata)
    unflat_ydata2 = ut.sortedby2(unflat_ydata, unflat_max_y)
    unflat_xdata2 = [[qx] * len(ydata) for qx, ydata in enumerate(unflat_ydata2)]
    y_data = ut.flatten(unflat_ydata2)
    x_data = ut.flatten(unflat_xdata2)
    #unflat_ydata2 = ut.sortedby2(unflat_ydata, unflat_max_y)
    #x_data = ut.flatten(unflat_xdata)
    #x_data = ut.sortedby2(x_data, y_data)
    #y_data = ut.sortedby2(y_data, y_data)
    return x_data, y_data


def get_stem_data(ibs, qaid2_qres):
    """
    returns data for pt.plot_stems

    data is sorted by result ranks. nsum is taken into acount if it exists

    get_stem_data

    Args:
        qaid2_qres (int): query annotation id

    Example:
        >>> from ibeis.dev.results_all import *   # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_list = qaid_list = ibs.get_valid_aids()
        >>> qaid2_qres, qreq_ = results_all.get_qres_and_qreq_(ibs, qaid_list, daid_list)
    """
    #ut.embed()
    import numpy as np
    #unflat_xdata = []
    unflat_ydata = []

    for qx, (qaid, qres) in enumerate(six.iteritems(qaid2_qres)):
        #qres.rrr(verbose=False)
        is_nsum = qres.is_nsum()
        worst_possible_rank = qres.get_worse_possible_rank()
        gt_ranks  = np.array(qres.get_gt_ranks(ibs=ibs, fillvalue=worst_possible_rank))
        if len(gt_ranks) == 0:
            continue
        if is_nsum:
            #gt_scores = np.array(qres.get_gt_scores(ibs=ibs))
            argx = gt_ranks.argmin()
            best_rank = gt_ranks[argx:argx + 1]
            qres_ydata = best_rank
        else:
            qres_ydata = gt_ranks
        #qres_xdata  [qx] * len(qres_ydata)
        #unflat_xdata.append(qres_xdata)
        unflat_ydata.append(qres_ydata)

    unflat_max_y = map(sorted, unflat_ydata)
    unflat_ydata2 = ut.sortedby2(unflat_ydata, unflat_max_y)
    unflat_xdata2 = [[qx] * len(ydata) for qx, ydata in enumerate(unflat_ydata2)]
    y_data = ut.flatten(unflat_ydata2)
    x_data = ut.flatten(unflat_xdata2)
    #unflat_ydata2 = ut.sortedby2(unflat_ydata, unflat_max_y)
    #x_data = ut.flatten(unflat_xdata)
    #x_data = ut.sortedby2(x_data, y_data)
    #y_data = ut.sortedby2(y_data, y_data)
    return x_data, y_data


# ALL RESULTS CACHE


__ALLRES_CACHE__ = {}
__QRESREQ_CACHE__ = {}


def build_cache_key(ibs, qaid_list, daid_list, custom_qparams):
    # a little overconstrained
    cfgstr = ibs.cfg.query_cfg.get_cfgstr()
    query_hashid = ibs.get_annot_uuid_hashid(qaid_list, '_QAUUID')
    data_hashid  = ibs.get_annot_uuid_hashid(daid_list, '_DAUUID')
    key = (query_hashid, data_hashid, cfgstr, str(custom_qparams))
    return key


def get_qres_and_qreq_(ibs, qaid_list, daid_list=None, custom_qparams=None):
    if daid_list is None:
        daid_list = ibs.get_valid_aids()

    qres_cache_key = build_cache_key(ibs, qaid_list, daid_list, custom_qparams)

    try:
        (qaid2_qres, qreq_) = __QRESREQ_CACHE__[qres_cache_key]
    except KeyError:
        qaid2_qres, qreq_ = ibs._query_chips(qaid_list, daid_list,
                                             return_request=True,
                                             custom_qparams=custom_qparams)
        # Cache save
        __QRESREQ_CACHE__[qres_cache_key] = (qaid2_qres, qreq_)
    return (qaid2_qres, qreq_)


def get_allres(ibs, qaid_list, daid_list=None, custom_qparams=None):
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
    allres_key = build_cache_key(ibs, qaid_list, daid_list, custom_qparams)
    try:
        allres = __ALLRES_CACHE__[allres_key]
    except KeyError:
        qaid2_qres, qreq_ = get_qres_and_qreq_(ibs, qaid_list, daid_list, custom_qparams)
        allres = init_allres(ibs, qaid2_qres, qreq_)
        # Cache save
        __ALLRES_CACHE__[allres_key] = allres
    return allres
