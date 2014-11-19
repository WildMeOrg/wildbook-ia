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


def test_confidence_measures(ibs, qres_list, qaid_list):
    import scipy.stats as spstats

    def find_tscore(qres, ibs):
        # H0: null hypothesis nid_list[0] is not he same animal
        # H1: alt hypothesis nid_list[0] is the same animal
        # assuming the null hypothesis is true, how likely is it
        # that we got the sample that we did.
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        alt_score = sorted_nscores[0]
        sample_measurements = sorted_nscores[1:]
        num_samples = len(sample_measurements)
        sample_mean = sample_measurements.mean()
        sample_std  = sample_measurements.std()

        population_mean = sample_mean
        population_std = sample_std / np.sqrt(num_samples)
        # t-score is like the z-score but for a sample
        tscore = (alt_score - population_mean) / population_std
        return tscore

    def find_prob_tscore(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        alt_score = sorted_nscores[0]
        sample_measurements = sorted_nscores[1:]
        tscore, pval = spstats.ttest_1samp(sample_measurements, alt_score)
        return pval

    def find_prob_tscore_next(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        tscore, pval1 = spstats.ttest_1samp(sorted_nscores[1:], sorted_nscores[0])
        tscore, pval2 = spstats.ttest_1samp(sorted_nscores[2:], sorted_nscores[1])
        return pval1 - pval2

    def find_prob_densitity(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        alt_score = sorted_nscores[0]
        sample_measurements = sorted_nscores[1:]
        data_pdf = ut.estimate_pdf(sample_measurements)
        density = data_pdf.evaluate(alt_score)[0]
        prob_correct = (1 - density)
        return '%.2f' % prob_correct

    def find_prob_normtest1(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        sample_measurements = sorted_nscores[1:]
        kurt, pval = spstats.normaltest(sample_measurements)
        return '%.2f' % pval

    def find_prob_normtest2(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        kurt, pval = spstats.normaltest(sorted_nscores)
        return '%.2f' % pval

    def find_scorediff(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        return -np.diff(sorted_nscores)[0]

    def find_prob_scorediff(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        diff_scores = -np.diff(sorted_nscores)
        tscore, pval1 = spstats.ttest_1samp(diff_scores[1:], diff_scores[0])
        tscore, pval2 = spstats.ttest_1samp(diff_scores[2:], diff_scores[1])
        return '%.2E' % ((1 - (pval1)) - ((1 - pval2)))

    if False:
        true_nid_list = ibs.get_annot_nids(qaid_list)
        bestaidrank_list  = [qres.get_best_gt_rank(ibs=ibs) for qres in qres_list]
        decision_tup_list = [qres.get_name_decisiontup(ibs) for qres in qres_list]
        decision_nid_list = ut.get_list_column(decision_tup_list, 0)
        decision_score_list = ut.get_list_column(decision_tup_list, 1)
        iscorrect_list = [nid_target == nid_res
                          for nid_target, nid_res in zip(decision_nid_list, true_nid_list)]

        t_list          = [find_tscore(qres, ibs) for qres in qres_list]
        pval_t_list     = [find_prob_tscore(qres, ibs) for qres in qres_list]
        pval_tnext_list = [find_prob_tscore_next(qres, ibs) for qres in qres_list]
        norm1_list      = [find_prob_normtest1(qres, ibs) for qres in qres_list]
        norm2_list      = [find_prob_normtest2(qres, ibs) for qres in qres_list]
        pden_list       = [find_prob_densitity(qres, ibs) for qres in qres_list]
        scorediff       = [find_scorediff(qres, ibs) for qres in qres_list]
        prob_scorediffs = [find_prob_scorediff(qres, ibs) for qres in qres_list]

        column_list = [iscorrect_list, decision_score_list, scorediff, prob_scorediffs, t_list, pval_t_list, pval_tnext_list, pden_list, norm1_list, norm2_list]
        column_labels  = ['Correct', 'score_list', 'scorediff', 'pscorediff', 't', 'pval_t', 'pnext', 'pden_list', 'norm1', 'norm2']

        print(ut.make_csv_table(column_list, column_labels))


def learn_score_normalization(ibs, qres_list, qaid_list):
    """
    Args:
        qaid2_qres (int): query annotation id

    Example:
        >>> from ibeis.dev.results_all import *   # NOQA
        >>> from ibeis.dev import results_all
        >>> import ibeis
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> ibs = ibeis.opendb('GZ_ALL')
        >>> qaid_list = daid_list = ibs.get_valid_aids()
        >>> hard_aids = ibs.get_hard_annot_rowids()
        >>> easy_aids = ibs.get_easy_annot_rowids()
        >>> qaid_list = hard_aids + easy_aids[::1]
        >>> cfgdict = dict(codename='nsum')
        >>> qaid2_qres, qreq_ = results_all.get_qres_and_qreq_(ibs, qaid_list, daid_list, cfgdict)
        >>> qres_list = [qaid2_qres[aid] for aid in qaid_list]
        >>>
        >>> results_all.learn_score_normalization(ibs, qres_list, qaid_list)


    valid_aids = ibs.get_valid_aids()
    hard_aids = ibs.get_hard_annot_rowids()
    qaid2_qres = ibs._query_chips4(hard_aids, daid_list, custom_qparams=cfgdict)
    """
    qres = qres_list[0]
    #sorted_nids, sorted_scores = qres.get_sorted_nids_and_scores(ibs)
    #data = sorted_scores
    #qaid_list = [aid for aid in six.iterkeys(qaid2_qres)]
    #qres_list = [qaid2_qres[qaid] for qaid in qaid_list]
    #for qres in qres_list:
    #    qres.rrr(verbose=False)

    # http://en.wikipedia.org/wiki/Statistical_hypothesis_testing
    # http://en.wikipedia.org/wiki/Type_I_and_type_II_errors
    # http://en.wikipedia.org/wiki/P-value
    # ftp://ftp.stat.duke.edu/pub/WorkingPapers/10-13.pdf
    good_tp_nscores = []
    good_fp_nscores = []

    good_tp_ndiff = []
    good_fp_ndiff = []
    for qx, qres in enumerate(qres_list):
        qaid = qres.get_qaid()
        if not ibs.get_annot_has_groundtruth(qaid):
            continue
        is_nsum = qres.is_nsum()
        if is_nsum:
            qnid = ibs.get_annot_nids(qres.get_qaid())
            sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)

            sorted_ndiff = -np.diff(sorted_nscores.tolist())
            sorted_nids = np.array(sorted_nids)
            gt_rank = np.where([sorted_nids == qnid])[0]
            if gt_rank == 0 and len(sorted_nscores) > 1:
                good_tp_nscores.append(sorted_nscores[0])
                good_fp_nscores.append(sorted_nscores[1])
                if len(sorted_ndiff) > 1:
                    good_tp_ndiff.append(sorted_ndiff[0])
                    good_fp_ndiff.append(sorted_ndiff[1])
                #if len(score_diff_given_good) == 0:
                #    pass

    import plottool as pt  # NOQA
    import functools
    statstr_ = functools.partial(ut.get_stats_str, exclude_keys=['nMin', 'nMax'])

    good_tp_nscores = np.array(good_tp_nscores)
    good_fp_nscores = np.array(good_fp_nscores)
    good_tp_ndiff = np.array(good_tp_ndiff)
    good_fp_ndiff = np.array(good_fp_ndiff)
    good_scores = np.hstack((good_tp_nscores, good_fp_nscores))
    good_ndiff = np.hstack((good_tp_ndiff, good_fp_ndiff))

    #print(statstr_(good_tp_nscores))
    #print(statstr_(good_fp_nscores))
    #print(statstr_(good_scores))
    #print(statstr_(good_tp_ndiff))
    #print(statstr_(good_fp_ndiff))
    #print(statstr_(good_ndiff))
    #import statsmodels.nonparametric.kde

    #ut.rrrr(verbose=False)

    #print(ut.stats_str(score_fp_pdf.support))
    #print(ut.stats_str(score_tp_pdf.support))

    pt.plots.plot_densities(
        (score_tp_pdf.evaluate(xdata) , score_fp_pdf.evaluate(xdata) , score_pdf.evaluate(xdata)),
        ('score | fp', 'score | tp', 'score'),
        xdata=xdata,
    )

    def bayes_rule(a_given_b, prob_a, prob_b):
        b_given_a = (a_given_b * prob_b) / prob_a
        return b_given_a

    def inspect_pdfs(good_tp, good_fp, lbl):
        good_all = np.hstack((good_tp, good_fp))


        score_tp_pdf = ut.estimate_pdf(good_tp, gridsize=512)
        score_fp_pdf = ut.estimate_pdf(good_fp, gridsize=512)
        score_pdf = ut.estimate_pdf(good_all, gridsize=512)
        xdata = np.linspace(0, 2000, 1024)

        p_score_given_tp = score_tp_pdf.evaluate(xdata)
        p_score_given_fp = score_fp_pdf.evaluate(xdata)
        p_score = score_pdf.evaluate(xdata)
        fp_bw = score_fp_pdf.bw
        tp_bw = score_tp_pdf.bw
        s_bw = score_pdf.bw
        p_tp = .04
        p_fp = (1 - p_tp)

        # Apply bayes
        p_tp_given_score = bayes_rule(p_score_given_tp, p_tp, p_score)
        p_fp_given_score = bayes_rule(p_score_given_fp, p_fp, p_score)

        w = xdata[1] - xdata[0]
        p_tp_given_score_cdf = (p_tp_given_score / p_tp_given_score.sum()).cumsum()
        p_fp_given_score_cdf = (p_fp_given_score / p_fp_given_score.sum()).cumsum()

        pt.plots.plot_densities(
            (p_fp_given_score_cdf,  p_tp_given_score_cdf),
            ('fp given ' + lbl, 'tp given ' + lbl),
            figtitle='cdf',
            xdata=xdata)

        pt.plots.plot_densities(
            (1 - p_fp_given_score_cdf, 1 - p_tp_given_score_cdf),
            ('fp given ' + lbl, 'tp given ' + lbl),
            figtitle='1 - cdf ' + lbl,
            xdata=xdata)

        pt.plots.plot_densities(
            (p_fp_given_score, p_tp_given_score),
            ('fp given ' + lbl, 'tp given ' + lbl),
            figtitle='pdf ' + lbl,
            xdata=xdata)

    pt.close_all_figures()
    #import imp
    #imp.reload(pt.plots)
    #imp.reload(pt)

    pt.plots.plot_sorted_scores(
        (good_fp_nscores, good_tp_nscores),
        ('score | fp', 'score | tp'),
        figtitle='sorted nscores'
    )
    pt.plots.plot_sorted_scores(
        (good_fp_ndiff, good_tp_ndiff),
        ('diff | fp', 'diff | tp'),
        figtitle='sorted ndiff'
    )
    inspect_pdfs(good_tp_nscores, good_fp_nscores, 'score')
    inspect_pdfs(good_tp_ndiff, good_fp_ndiff, 'diff')
    pt.present()


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
