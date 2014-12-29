from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
from collections import namedtuple
#print, print_, printDBG, rrr, profile = ut.inject(__name__, '[suggest]')


# ---- GLOBALS ----

ChoiceTuple = namedtuple('ChoiceTuple', ('sorted_nids', 'sorted_nscore',
                                         'sorted_rawscore', 'sorted_aids',
                                         'sorted_ascores'))


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
        >>> from ibeis.model.hots.system_suggestor import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict())[1]
        >>> choicetup = get_qres_name_choices(ibs, qres)
        >>> print(choicetup)
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
    """ hotspotter returns an name suggestion
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
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid2_qres, qreq_ = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict(),
        ...            return_request=True)
        >>> qres = qaid2_qres[1]
        >>> choicetup = get_qres_choices(ibs, qres)
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
        autoname_msg_list.append('suggesting nid=%r, score=%.2f, rank=%r, rawscore=%.2f' % (nid, score, rank, rawscore))
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
    """
    print('Deciding if adding qaid=%r as an exemplar' % (qaid,))
    # Need a good criteria here
    max_exemplars = ibs.cfg.other_cfg.max_exemplars
    other_exemplars = ibs.get_annot_groundtruth(qaid, is_exemplar=True)
    num_other_exemplars = len(other_exemplars)
    #
    is_non_exemplar = not ibs.get_annot_exemplar_flags(qaid)
    can_add_more = num_other_exemplars < max_exemplars
    print('num_other_exemplars = %r' % num_other_exemplars)
    print('max_exemplars = %r' % max_exemplars)
    print('is_non_exemplar = %r' % is_non_exemplar)

    exemplar_confidence = 0

    if num_other_exemplars == 0:
        autoexmplr_msg = 'First exemplar of this name.'
        exemplar_decision = True
        exemplar_confidence = 1.0
        return autoexmplr_msg, exemplar_decision, exemplar_confidence
    elif is_non_exemplar and can_add_more:
        print('Testing exemplar disinctiveness')
        with ut.Indenter('[exemplar_test]'):
            exemplar_distinctiveness_thresh = ibs.cfg.other_cfg.exemplar_distinctiveness_thresh
            # Logic to choose query based on exemplar score distance
            qaid_list = [qaid]
            daid_list = other_exemplars
            cfgdict = dict(codename='vsone_norm_csum')
            qres = ibs.query_chips(qaid_list, daid_list, cfgdict=cfgdict, verbose=False)[0]
            if qres is None:
                is_distinctive = True
            else:
                #ut.embed()
                aid_arr, score_arr = qres.get_aids_and_scores()
                is_distinctive = np.all(aid_arr < exemplar_distinctiveness_thresh)
    else:
        is_distinctive = True
        print('Not testing exemplar disinctiveness')

    do_exemplar_add = (can_add_more and is_distinctive and is_non_exemplar)
    if do_exemplar_add:
        autoexmplr_msg = ('marking as qaid=%r exemplar' % (qaid,))
        exemplar_decision = True
    else:
        exemplar_decision = False
        autoexmplr_msg = 'annotation is not marked as exemplar'

    return autoexmplr_msg, exemplar_decision, exemplar_confidence
