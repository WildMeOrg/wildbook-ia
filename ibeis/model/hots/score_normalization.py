"""
GOALS:
    1) vsmany
       * works resaonable for very few and very many
       * stars with small k and then k becomes a percent or log percent
       * distinctiveness from different location

    2) 1-vs-1
       * uses distinctivness and foreground when available
       * start with ratio test and ransac

    3) First N decision are interactive until we learn a good threshold

    4) Always show numbers between 0 and 1 spatial verification is based on
    single best exemplar

       x - build normalizer
       x - test normalizer
       x - monotonicity (both nondecreasing and strictly increasing)
       x - cache normalizer
       x - cache maitainance (deleters and listers)
       o - Incemental learning
       o - Spceies sensitivity


    5) Add exemplars that are distinct from exiting (matches below threshold)

    (no rebuilding ing kd-tree for each image)



"""
from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import utool as ut
import vtool as vt
import six  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[scorenorm]', DEBUG=False)


@six.add_metaclass(ut.ReloadingMetaclass)
class ScoreNormalizer(ut.Cachable):
    prefix = 'normalizer_'

    def __init__(normalizer, cfgstr=None, score_domain=None, p_tp_given_score=None):
        super(ScoreNormalizer, normalizer).__init__()
        normalizer.cfgstr = cfgstr
        normalizer.set_values(score_domain, p_tp_given_score)

    def get_prefix(normalizer):
        return 'normalizer_'

    def get_cfgstr(normalizer):
        assert normalizer.cfgstr is not None
        return normalizer.cfgstr

    def set_values(normalizer, score_domain, p_tp_given_score):
        normalizer.score_domain = score_domain
        normalizer.p_tp_given_score = p_tp_given_score

    #def load(normalizer, *args, **kwargs):
    #    # Inherited method
    #    super(ScoreNormalizer, normalizer).load(*args, **kwargs)

    #def save(normalizer, *args, **kwargs):
    #    # Inherited method
    #    super(ScoreNormalizer, normalizer).save(*args, **kwargs)

    def normalize_score(normalizer, score):
        if score < normalizer.score_domain[0]:
            prob = 0.0
        elif score > normalizer.score_domain[-1]:
            prob = (normalizer.p_tp_given_score[-1] + 1.0) / 2.0
        else:
            indexes = np.where(normalizer.score_domain <= score)[0]
            index = indexes[-1]
            prob = normalizer.p_tp_given_score[index]
        #if prob >= 1:
        #    ut.embed()
        return prob

    def normalize_score_list(normalizer, score_list):
        prob_list = [normalizer.normalize_score(score) for score in score_list]
        return prob_list

    def visualize(normalizer, update=True):
        """
            >>> from ibeis.model.hots.score_normalization import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb('PZ_MTEST')
            >>> normalizer = load_precomputed_normalizer(ibs, 0)
            >>> normalizer.visualize()
        """
        import plottool as pt
        p_tp_given_score = normalizer.p_tp_given_score
        p_tn_given_score = 1 - p_tp_given_score
        score_domain = normalizer.score_domain
        cfgstr = normalizer.get_cfgstr()
        true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
        false_color = pt.FALSE_RED

        pt.plots.plot_probabilities(
            (p_tn_given_score, p_tp_given_score),
            ('p(tn | score)', 'p(tp | score)'),
            prob_colors=(false_color, true_color,),
            figtitle='post_bayes pdf score ' + cfgstr,
            xdata=score_domain)

        if update:
            pt.update()

    def __call__(normalizer, score_list):
        return normalizer.normalize_score_list(score_list)


def parse_available_normalizers(ibs):
    import parse
    normalizers_fpaths = list_available_score_normalizers(ibs)
    parsestr = '{cachedir}/' + ScoreNormalizer.prefix + '{cfgstr}' + ScoreNormalizer.ext
    result_list = [parse.parse(parsestr, path) for path in normalizers_fpaths]
    cfgstr_list = [result['cfgstr'] for result in result_list]
    cachedir_list = [result['cachedir'] for result in result_list]
    return cfgstr_list, cachedir_list


def load_precomputed_normalizer(ibs, choice):
    """
    Example:
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> normalizer = load_precomputed_normalizer(ibs, 0)
    """
    cfgstr_list, cachedir_list = parse_available_normalizers(ibs)
    cfgstr = cfgstr_list[0]
    cachedir = cachedir_list[0]
    normalizer = ScoreNormalizer(cfgstr=cfgstr)
    normalizer.load(cachedir)
    return normalizer


def list_available_score_normalizers(with_global=True, with_local=True):
    r"""
    CommandLine:
        python ibeis/model/hots/score_normalization.py --test-list_available_score_normalizers --enableall

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> import ibeis
        >>> local_normalizers_fpaths = list_available_score_normalizers(with_global=False)
        >>> global_normalizers_fpaths = list_available_score_normalizers(with_local=False)
        >>> # quote them
        >>> # local_normalizers_fpaths = ['"%s"' % fpath for fpath in local_normalizers_fpaths]
        >>> # global_normalizers_fpaths = ['"%s"' % fpath for fpath in global_normalizers_fpaths]
        >>> print('Available LOCAL normalizers: ' + ut.indentjoin(local_normalizers_fpaths, '\n  '))
        >>> print('Available GLOBAL normalizers: ' + ut.indentjoin(global_normalizers_fpaths, '\n  '))
        >>> #  [ut.delete(fpath) for fpath in local_normalizers_fpaths]
        >>> #  [ut.delete(fpath) for fpath in global_normalizers_fpaths]

    """
    from ibeis.dev import sysres
    from ibeis import constants
    from os.path import join
    pattern = ScoreNormalizer.prefix + '*' + ScoreNormalizer.ext
    ibeis_resdir = sysres.get_ibeis_resource_dir()
    workdir = sysres.get_workdir()

    normalizer_fpaths = []
    if with_global:
        global_normalizers = ut.glob(ibeis_resdir, pattern, recursive=True)
        normalizer_fpaths += global_normalizers
    if with_local:
        ibsdbdir_list = sysres.get_ibsdb_list(workdir)
        searchdirs = [join(ibsdbdir, constants.PATH_NAMES._ibsdb, constants.PATH_NAMES.cache)
                      for ibsdbdir in ibsdbdir_list]
        local_normalizers_list = [ut.glob(path, pattern, recursive=True) for path in  searchdirs]
        local_normalizers = ut.flatten(local_normalizers_list)
        normalizer_fpaths += local_normalizers
    # Just search localdb cachedirs (otherwise it will take forever)
    return normalizer_fpaths


#def test_normalizer(with_indexer=True):
#    """
#    Example:
#        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
#        >>> nnindexer, qreq_, ibs = test_normalizer() # doctest: +ELLIPSIS
#    """


def delete_all_learned_normalizers():
    """
    CommandLine:
        python ibeis/model/hots/score_normalization.py --test-delete_all_learned_normalizers --enableall

    Example:
        >>> from ibeis.model.hots import score_normalization
        >>> score_normalization.delete_all_learned_normalizers()
    """
    from ibeis.model.hots import score_normalization
    import utool as ut
    normalizer_fpath_list = score_normalization.list_available_score_normalizers()
    for path in normalizer_fpath_list:
        ut.delete(path)


def train_baseline_for_all_dbs():
    """
    Runs unnormalized queries to compute normalized queries

    CommandLine:
        python ibeis/model/hots/score_normalization.py --test-train_baseline_for_all_dbs --enableall

    Example:
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> train_baseline_for_all_dbs()
    """
    import ibeis
    #from ibeis.model.hots import score_normalization
    dbname = 'GZ_ALL'
    dbname = 'PZ_MTEST'
    learnkw = dict()

    for dbname in ['GZ_ALL', 'PZ_MTEST']:
        ibs = ibeis.opendb(dbname)
        train_baseline_ibeis_normalizer(ibs, use_cache=False, **learnkw)


def train_baseline_ibeis_normalizer(ibs, use_cache=True, **learnkw):
    """
    Runs unnormalized queries to compute normalized queries

    Args:
        ibs (IBEISController):

    Returns:
        ScoreNormalizer: normalizer

    CommandLine:
        python ibeis/model/hots/score_normalization.py --test-train_baseline_ibeis_normalizer --enableall
        python ibeis/model/hots/score_normalization.py --test-train_baseline_ibeis_normalizer --enableall --noshow

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> import ibeis
        >>> from ibeis.model.hots import score_normalization
        >>> score_normalization.rrr()
        >>> dbname = 'GZ_ALL'
        >>> dbname = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(dbname)
        >>> learnkw = dict()
        >>> normalizer = score_normalization.train_baseline_ibeis_normalizer(ibs, use_cache=False, **learnkw)
        >>> normalizer.visualize()
        >>> result = str(normalizer)
        >>> print(result)
        >>> import plottool as pt
        >>> exec(pt.present())
    """
    # TRAIN BASELINE
    tag = '<TRAINING> '
    print(utool.msgblock(tag, 'Begning Training'))
    with utool.Timer(tag):
        #with utool.Indenter('TRAIN >>> '):
        from ibeis.model.hots import query_request
        qaid_list = ibs.get_valid_aids()
        daid_list = ibs.get_valid_aids()
        cfgdict = {'codename': 'nsum_unnorm'}
        qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list, cfgdict)
        qres_list = ibs.query_chips(qaid_list, daid_list, qreq_=qreq_, cfgdict=None)
        normalizer = cached_ibeis_score_normalizer(ibs, qaid_list,
                                                   qres_list,
                                                   use_cache=use_cache, **learnkw)
        # Save as baseline for this species
        species_text = '_'.join(qreq_.species_list)  # HACK
        baseline_cfgstr = 'baseline_' + species_text
        cachedir = ibs.get_species_cachedir(species_text)
        normalizer.save(cachedir, cfgstr=baseline_cfgstr)
        #print(fpath)
        #learn_ibeis_score_normalizer(ibs, qaid_list, qres_list, cfgstr)
    print('\n' + utool.msgblock(tag, 'Finished Training'))
    return normalizer


def test():
    """
    >>> from ibeis.model.hots.score_normalization import *  # NOQA
    """
    #from ibeis.model.hots import query_request
    import ibeis
    ibs = ibeis.opendb(db='PZ_MTEST')
    qaid_list = [1, 2, 3, 4, 5]
    daid_list = [1, 2, 3, 4, 5]
    cfgdict = {'codename': 'nsum'}
    qres_list, qreq_ = ibs.query_chips(qaid_list, daid_list, use_cache=False, cfgdict=cfgdict, return_request=True)
    qreq_.load_score_normalizer(qreq_.ibs)
    normalizer = qreq_.normalizer

    for qres in qres_list:
        aid_list = list(six.iterkeys(qres.aid2_score))
        score_list = list(six.itervalues(qres.aid2_score))
        #normalizer  = normalizer
        prob_list = [normalizer.normalize_score(score) for score in score_list]
        qres.qaid2_score = dict(zip(aid_list, prob_list))
    for qres in qres_list:
        print(list(six.itervalues(qres.qaid2_score)))

        #aid2_score = {aid: normalizer.no(score) for aid, score in }
        pass


def request_ibeis_normalizer(ibs, qreq_):
    """
    Example:
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> from ibeis.model.hots import query_request
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='PZ_MTEST')
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5n
        >>> cfgdict = {'codename': 'nsum_unnorm'}
        >>> qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list, cfgdict=cfgdict)
        >>> normalizer = request_ibeis_normalizer(ibs, qreq_)
    """
    species_text = qreq_.species_list[0]  # HACK
    cfgstr = 'baseline_' + species_text
    cachedir = ibs.get_species_cachedir(species_text)
    try:
        normalizer = ScoreNormalizer(cfgstr)
        normalizer.load(cachedir)
        print('returning baseline normalizer')
        return normalizer
    except Exception:
        try:
            print('Baseline does not exist. Training baseline')
            normalizer = train_baseline_ibeis_normalizer(ibs)
            return normalizer
        except Exception as ex:
            ut.printex(ex)
            raise


def cached_ibeis_score_normalizer(ibs, qaid_list, qres_list, use_cache=True, **learnkw):
    """
    Args:
        qaid2_qres (int): query annotation id

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *   # NOQA
        >>> import ibeis
        >>> dbname = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(dbname)
        >>> qaid_list = daid_list = ibs.get_valid_aids()
        >>> cfgdict = dict(codename='nsum_unnorm')
        >>> qres_list = ibs.query_chips(qaid_list, daid_list, cfgdict)
        >>> score_normalizer = cached_ibeis_score_normalizer(ibs, qaid_list, qres_list)
        >>> result = score_normalizer.get_fname()
        >>> print(result)
        normalizer_PZ_MTEST_UUIDS((119)htc%i42+w9plda&d).cPkl
    """
    # Collect training data
    cfgstr = ibs.get_dbname() + ibs.get_annot_uuid_hashid(qaid_list)
    try:
        if use_cache is False:
            raise Exception('forced cache miss')
        normalizer = ScoreNormalizer(cfgstr)
        normalizer.load(ibs.cachedir)
        print('returning cached normalizer')
    except Exception as ex:
        ut.printex(ex, iswarning=True)
        normalizer = learn_ibeis_score_normalizer(ibs, qaid_list, qres_list, cfgstr, **learnkw)
        normalizer.save(ibs.cachedir)
    return normalizer


def learn_ibeis_score_normalizer(ibs, qaid_list, qres_list, cfgstr, **learnkw):
    print('learning normalizer')
    (truepos_scores, trueneg_scores) = get_ibeis_score_training_data(ibs, qaid_list, qres_list)
    (score_domain, p_tp_given_score) = learn_score_normalization(truepos_scores, trueneg_scores, **learnkw)
    normalizer = ScoreNormalizer(cfgstr, score_domain, p_tp_given_score)
    return normalizer


def get_ibeis_score_training_data(ibs, qaid_list, qres_list):
    """
    Returns "good" taining examples
    """
    good_tp_nscores = []
    good_tn_nscores = []
    good_tp_aidnid_pairs = []
    good_tn_aidnid_pairs = []
    for qx, qres in enumerate(qres_list):
        qaid = qres.get_qaid()
        if not qres.is_nsum():
            raise AssertionError('must be nsum')
        if not ibs.get_annot_has_groundtruth(qaid):
            continue
        qnid = ibs.get_annot_nids(qres.get_qaid())

        nscoretup = qres.get_nscoretup(ibs)
        (sorted_nids, sorted_nscores, sorted_aids, sorted_scores) = nscoretup

        sorted_ndiff = -np.diff(sorted_nscores.tolist())
        sorted_nids = np.array(sorted_nids)
        is_positive  = sorted_nids == qnid
        is_negative = np.logical_and(~is_positive, sorted_nids > 0)
        if not np.any(is_positive) or not np.any(is_negative):
            continue
        gt_rank = np.where(is_positive)[0][0]
        gf_rank = np.nonzero(is_negative)[0][0]
        if gt_rank == 0 and len(sorted_nscores) > gf_rank:
            if len(sorted_ndiff) > gf_rank:
                good_tp_nscores.append(sorted_nscores[gt_rank])
                good_tn_nscores.append(sorted_nscores[gf_rank])
                good_tp_aidnid_pairs.append((qaid, sorted_nids[gt_rank]))
                good_tn_aidnid_pairs.append((qaid, sorted_nids[gf_rank]))
    truepos_scores = np.array(good_tp_nscores)
    trueneg_scores = np.array(good_tn_nscores)
    return (truepos_scores, trueneg_scores)


def learn_score_normalization(truepos_scores, trueneg_scores, gridsize=1024,
                              adjust=8, return_all=False, monotonize=True,
                              clip_factor=(ut.PHI + 1)):
    r"""
    Takes collected data and applys parzen window density estimation and bayes rule.

    learn_score_normalization

    Args:
        truepos_scores (ndarray):
        trueneg_scores (ndarray):
        gridsize       (int): default 512
        adjust         (int): default 8
        return_all     (bool): default False
        monotonize     (bool): default True
        clip_factor    (float): default phi ** 2

    Returns:
        tuple: (score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score, clip_score)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> truepos_scores = np.linspace(100, 10000, 512)
        >>> trueneg_scores = np.linspace(0, 120, 512)
        >>> (score_domain, p_tp_given_score) = learn_score_normalization(truepos_scores, trueneg_scores)
        >>> result = int(p_tp_given_score.sum())
        >>> print(result)
        92
    """
    #clip_score = 2000
    # Find good maximum score
    clip_score = find_score_maxclip(truepos_scores, trueneg_scores, clip_factor)
    score_domain = np.linspace(0, clip_score, 1024)
    # Estimate true positive density
    score_tp_pdf = ut.estimate_pdf(truepos_scores, gridsize=gridsize, adjust=adjust)
    score_tn_pdf = ut.estimate_pdf(trueneg_scores, gridsize=gridsize, adjust=adjust)
    # Evaluate true negative density
    p_score_given_tp = score_tp_pdf.evaluate(score_domain)
    p_score_given_tn = score_tn_pdf.evaluate(score_domain)
    # Average to get probablity of any score
    p_score = (np.array(p_score_given_tp) + np.array(p_score_given_tn)) / 2.0
    # Apply bayes
    p_tp = .5
    p_tp_given_score = ut.bayes_rule(p_score_given_tp, p_tp, p_score)
    if monotonize:
        #p_tp_given_score = vt.ensure_monotone_increasing(p_tp_given_score)
        p_tp_given_score = vt.ensure_monotone_strictly_increasing(p_tp_given_score, zerohack=True, onehack=True)
    if return_all:
        #p_tn = 1.0 - p_tp  # NOT SURE WHY THIS CANT BE .5
        #p_tn_given_score = ut.bayes_rule(p_score_given_tn, 1.0, p_score)
        p_tn_given_score = 1 - p_tp_given_score
        #if monotonize:
        #    p_tn_given_score = vt.ensure_monotone_decreasing(p_tn_given_score)
        return (score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score, clip_score)
    else:
        return (score_domain, p_tp_given_score)


def find_score_maxclip(truepos_scores, trueneg_scores, clip_factor=ut.PHI + 1):
    """
    returns score to clip true positives past.

    Args:
        truepos_scores (ndarray):
        trueneg_scores (ndarray):

    Returns:
        float: clip_score

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *  # NOQA
        >>> truepos_scores = np.array([100, 200, 50000])
        >>> trueneg_scores = np.array([10, 30, 110])
        >>> clip_score = find_score_maxclip(truepos_scores, trueneg_scores)
        >>> result = str(clip_score)
        >>> print(result)
        287.983738762
    """
    max_true_positive_score = truepos_scores.max()
    max_true_negative_score = trueneg_scores.max()
    if clip_factor is None:
        clip_score = max_true_positive_score
    else:
        overshoot_factor = max_true_positive_score / max_true_negative_score
        if overshoot_factor > clip_factor:
            clip_score = max_true_negative_score * clip_factor
        else:
            clip_score = max_true_positive_score
    return clip_score


def test_score_normalization():
    """

    CommandLine:
        python ibeis/model/hots/score_normalization.py --test-test_score_normalization --enableall

    Example:
        >>> # DISABLE_DOCTEST
        >>> #from ibeis.model.hots import score_normalization
        >>> #score_normalization.rrr()
        >>> from ibeis.model.hots.score_normalization import *   # NOQA
        >>> locals_ = test_score_normalization()
        >>> execstr = ut.execstr_dict(locals_)
        >>> #print(execstr)
        >>> exec(execstr)
        >>> import plottool as pt
        >>> exec(pt.present())

    """
    import ibeis

    # Load IBEIS database
    dbname = 'PZ_MTEST'
    #dbname = 'GZ_ALL'

    ibs = ibeis.opendb(dbname)
    qaid_list = daid_list = ibs.get_valid_aids()

    # Get unnormalized query results
    cfgdict = dict(codename='nsum_unnorm')
    qres_list = ibs.query_chips(qaid_list, daid_list, cfgdict)

    # Get a training sample
    (truepos_scores, trueneg_scores) = get_ibeis_score_training_data(ibs, qaid_list, qres_list)

    # Print raw score statistics
    ut.print_stats(truepos_scores, lbl='truepos_scores')
    ut.print_stats(trueneg_scores, lbl='trueneg_scores')

    normkw_list = ut.util_dict.all_dict_combinations(
        {
            'monotonize': [True, False],
            'adjust': [1, 4, 8],
        }
    )

    if len(normkw_list) > 32:
        raise AssertionError('Too many plots to test!')

    for normkw in normkw_list:
        # Learn the appropriate normalization
        #normkw = {}  # dict(gridsize=1024, adjust=8, clip_factor=ut.PHI + 1, return_all=True)
        (score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn,
         p_score, clip_score) = learn_score_normalization(truepos_scores, trueneg_scores, return_all=True, **normkw)

        assert clip_score > trueneg_scores.max()
        import plottool as pt  # NOQA

        inspect_pdfs(trueneg_scores, truepos_scores, score_domain,
                     p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score)

        pt.set_figtitle('ScoreNorm ' + ibs.get_dbname() + ' ' + ut.dict_str(normkw))
    locals_ = locals()
    return locals_


def inspect_pdfs(trueneg_scores, truepos_scores, score_domain, p_tp_given_score,
                 p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score):
    import plottool as pt  # NOQA

    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED
    unknown_color = pt.UNKNOWN_PURP

    fnum = pt.next_fnum()
    pnum_ = pt.get_pnum_func(nRows=3, nCols=1)
    pt.figure(fnum=fnum, pnum=pnum_(0))

    pt.plots.plot_sorted_scores(
        (trueneg_scores, truepos_scores),
        ('true negative scores', 'true positive scores'),
        score_colors=(false_color, true_color),
        logscale=True,
        figtitle='sorted nscores',
        fnum=fnum,
        pnum=pnum_(0))

    pt.plots.plot_probabilities(
        (p_score_given_tn,  p_score_given_tp, p_score),
        ('p(score | tn)', 'p(score | tp)', 'p(score)'),
        prob_colors=(false_color, true_color, unknown_color),
        figtitle='pre_bayes pdf score',
        xdata=score_domain,
        fnum=fnum,
        pnum=pnum_(1))

    pt.plots.plot_probabilities(
        (p_tn_given_score, p_tp_given_score),
        ('p(tn | score)', 'p(tp | score)'),
        prob_colors=(false_color, true_color,),
        figtitle='post_bayes pdf score',
        xdata=score_domain,
        fnum=fnum,
        pnum=pnum_(2))

if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.model.hots.score_normalization; utool.doctest_funcs(ibeis.model.hots.score_normalization, allexamples=True)"
        python -c "import utool, ibeis.model.hots.score_normalization; utool.doctest_funcs(ibeis.model.hots.score_normalization)"
        python ibeis/model/hots/score_normalization.py
        python ibeis/model/hots/score_normalization.py --allexamples
        python ibeis/model/hots/score_normalization.py --allexamples --noface --nosrc

    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
