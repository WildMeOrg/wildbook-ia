"""
Reports decisions and confidences about names (identifications) and
exemplars using query results objects.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import random
from collections import namedtuple
#print, print_, printDBG, rrr, profile = ut.inject(__name__, '[suggest]')


# ---- GLOBALS ----

ChoiceTuple = namedtuple(
    'ChoiceTuple',
    ('sorted_nids', 'sorted_nscore', 'sorted_rawscore',
     'sorted_aids', 'sorted_ascores')
)


# An exmplar decision adds and removes exemplars
ExemplarDecision = namedtuple(
    'ExemplarDecision',
    ('new_exemplar_aids', 'remove_exemplar_aids')
)


def get_qres_name_choices(ibs, qres):
    r"""
    returns all possible decision a user could make

    TODO: Return the possiblity of a merge.
    TODO: Ensure that the total probability of each possible choice sums to 1.
    This will define a probability density function that we can take advantage
    of

    Args:
        ibs (IBEISController):  ibeis controller object
        qres (QueryResult):  object of feature correspondences and scores

    Returns:
        ChoiceTuple: choicetup

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-get_qres_name_choices

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automatch_suggestor import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict())[1]
        >>> choicetup = get_qres_name_choices(ibs, qres)
        >>> print(choicetup)
        >>> result = ut.numpy_str(choicetup.sorted_nids[0:1], force_dtype=False)
        >>> print(result)
        np.array([1])

    """
    if qres is None:
        nscoretup = list(map(np.array, ([], [], [], [])))
        (sorted_nids, sorted_nscore, sorted_aids, sorted_ascores) = nscoretup
    else:
        nscoretup = qres.get_nscoretup(ibs)

    (sorted_nids, sorted_nscore, sorted_aids, sorted_ascores) = nscoretup
    sorted_rawscore = [qres.get_aid_scores(aids, rawscore=True) for aids in sorted_aids]

    choicetup = ChoiceTuple(sorted_nids, sorted_nscore, sorted_rawscore,
                            sorted_aids, sorted_ascores)
    return choicetup


def get_system_name_suggestion(ibs, choicetup):
    """
    Suggests a decision based on the current choices

    Args:
        ibs      (IBEISController):
        qaid      (int):  query annotation id
        qres      (QueryResult):  object of feature correspondences and scores
        metatup   (None):

    Returns:
        tuple: (autoname_msg, autoname_func)

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:0
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:1
        python -m ibeis.model.hots.automated_matcher --test-get_system_name_suggestion

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automatch_suggestor import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid2_qres, qreq_ = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict(),
        ...            return_request=True)
        >>> qres = qaid2_qres[1]
        >>> choicetup = get_qres_name_choices(ibs, qres)
        >>> (autoname_msg, name, name_confidence) = get_system_name_suggestion(ibs, choicetup)
        >>> # verify results
        >>> result = str((autoname_msg, name, name_confidence))
        >>> print(result)

    """
    threshold = ut.get_sys_maxfloat()
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup

    autoname_msg_list = []
    if len(sorted_nids) == 0:
        autoname_msg_list.append('Unable to find any matches')
        # if we can't find any matches then we must be sure it is a
        # new name. A false negative here can only be fixed with a merge
        name_confidence = 1.0
        nid, score, rank = None, None, None
    else:
        name_confidence = 0
        candidate_indexes = np.where(sorted_nscore > threshold)[0]
        if len(candidate_indexes) == 0:
            rank = None
            autoname_msg_list.append('No candidates above threshold')
        else:
            sortx = sorted_nscore[candidate_indexes].argsort()[::-1]
            rank = sortx[0]
            multiple_candidates = len(candidate_indexes) > 1
            if multiple_candidates:
                autoname_msg_list.append('Multiple candidates above threshold')
            else:
                autoname_msg_list.append('Single candidate above threshold')

    # Get system suggested nid and message
    if rank is not None:
        score = sorted_nscore[rank]
        nid = sorted_nids[rank]
        rawscore = sorted_rawscore[rank][0]
        msg_fmtstr = 'suggesting nid=%r, score=%.2f, rank=%r, rawscore=%.2f'
        msg = msg_fmtstr % (nid, score, rank, rawscore)
        autoname_msg_list.append(msg)
    else:
        nid, score, rawscore = None, None, None
        autoname_msg_list.append('suggesting new name')
    autoname_msg = '\n'.join(autoname_msg_list)

    name = ibs.get_name_texts(nid) if nid is not None else None
    if name is None:
        chosen_names = []
    else:
        chosen_names = [name]
    system_name_suggest_tup = (autoname_msg, chosen_names, name_confidence,)
    return system_name_suggest_tup


def get_system_exemplar_suggestion(ibs, qaid):
    """ hotspotter returns an exemplar suggestion

    TODO:
        do a vsone query between all of the exemplars to see if this one is good
        enough to be added.

    TODO:
        build a complete graph of exemplar scores and only add if this one is
        lower than any other edge

    SeeAlso:
        ibsfuncs.prune_exemplars

    CommandLine:
        python -m ibeis.model.hots.automatch_suggestor --test-get_system_exemplar_suggestion

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automatch_suggestor import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid = 2
        >>> # execute function
        >>> (autoexmplr_msg, exemplar_decision, exemplar_confidence) = get_system_exemplar_suggestion(ibs, qaid)
        >>> # verify results
        >>> result = str((autoexmplr_msg, exemplar_decision, exemplar_confidence))
        >>> print(result)
    """
    if ut.VERBOSE:
        print('[suggest_exemplar] Deciding if adding qaid=%r as an exemplar' % (qaid,))
    # Need a good criteria here
    max_exemplars = ibs.cfg.other_cfg.max_exemplars
    #max_exemplars = 2
    print('[suggest_exemplar] max_exemplars = %r' % max_exemplars)
    other_exemplars = ibs.get_annot_groundtruth(qaid, is_exemplar=True)
    num_other_exemplars = len(other_exemplars)
    #
    is_already_exemplar = ibs.get_annot_exemplar_flags(qaid)
    have_max_exemplars = num_other_exemplars >= max_exemplars
    if ut.VERBOSE:
        print('[suggest_exemplar] num_other_exemplars = %r' % num_other_exemplars)
        print('[suggest_exemplar] is_already_exemplar = %r' % is_already_exemplar)
    if is_already_exemplar:
        #ut.embed()
        print('[suggest_exemplar] WARNING is_already_exemplar = %r' % is_already_exemplar)

    exemplar_confidence = 0

    if is_already_exemplar:
        exemplar_decision = ExemplarDecision([], [])
        autoexmplr_msg = 'WARNING exemplar suggestion should not be in this state'
    elif num_other_exemplars == 0:
        # Always add the first exemplar
        exemplar_decision = ExemplarDecision([qaid], [])
        exemplar_confidence = 1.0
        autoexmplr_msg = 'First exemplar of this name.'
    elif not have_max_exemplars:
        exemplar_decision = ExemplarDecision([qaid], [])
        exemplar_confidence = 1.0
        autoexmplr_msg = 'This name has room for more exemplars'
    elif have_max_exemplars:
        METHOD = 2
        if METHOD == 1:
            exemplar_decision = exemplar_method1_distinctiveness(ibs, qaid, other_exemplars)
            autoexmplr_msg = '[suggest_exemplar] computed bsed on distinctivness '
        elif METHOD == 2:
            exemplar_decision = exemplar_method2_randomness(qaid, other_exemplars)
            autoexmplr_msg = '[suggest_exemplar] randomly chose '
        elif METHOD == 3:
            exemplar_decision = ExemplarDecision([], [])
            autoexmplr_msg = '[suggest_exemplar] defaulting  '
        else:
            raise AssertionError('Impossible State')
    else:
        raise AssertionError('Impossible State')

    autoexmplr_msg += 'exemplar_decision=%r' % (exemplar_decision,)

    return autoexmplr_msg, exemplar_decision, exemplar_confidence


def exemplar_method2_randomness(qaid, other_exemplars):
    r"""
    CommandLine:
        python -m ibeis.model.hots.automatch_suggestor --test-exemplar_method2_randomness

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automatch_suggestor import *  # NOQA
        >>> # build test data
        >>> random.seed(0)
        >>> qaid = 4
        >>> other_exemplars = [1, 2, 3, 5, 6, 9]
        >>> # execute function
        >>> exemplar_decision = exemplar_method2_randomness(qaid, other_exemplars)
        >>> exemplar_decision_list = [exemplar_method2_randomness(qaid, other_exemplars) for _ in range(1000)]
        >>> # verify results
        >>> flat_others = ut.flatten(ut.get_list_column(exemplar_decision_list, 1))
        >>> result = str(flat_others)
        >>> print(result)
        [3, 2, 6, 1, 3, 2, 9, 3, 6, 2, 1, 5, 1, 6, 1]

    Ignore:
        ibs = ut.search_stack_for_localvar('ibs')

    """
    rand_thresh = .01
    #rand_thresh = .1
    #rand_thresh = .9
    rand_float = random.random()
    if  rand_float < rand_thresh:
        exemplar_decision = ExemplarDecision([qaid], [random.choice(other_exemplars)])
        #ut.embed()
    else:
        exemplar_decision = ExemplarDecision([], [])
    return exemplar_decision


def exemplar_method1_distinctiveness(ibs, qaid, other_exemplars):
    """
    choose as exemplar if it is distinctive with respect to other exemplars
    """
    # FIXME ExemplarDecision
    print('[suggest_exemplar] Testing exemplar disinctiveness')
    exemplar_distinctiveness_thresh = ibs.cfg.other_cfg.exemplar_distinctiveness_thresh
    # Logic to choose query based on exemplar score distance
    qaid_list = [qaid]
    daid_list = other_exemplars
    cfgdict = dict(codename='vsone_norm_csum')
    qres = ibs.query_chips(qaid_list, daid_list, cfgdict=cfgdict, verbose=False)[0]
    if qres is None:
        exemplar_decision = True
    else:
        #ut.embed()
        aid_arr, score_arr = qres.get_aids_and_scores()
        exemplar_decision = np.all(aid_arr < exemplar_distinctiveness_thresh)
    return exemplar_decision


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.automatch_suggestor
        python -m ibeis.model.hots.automatch_suggestor --allexamples
        python -m ibeis.model.hots.automatch_suggestor --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
