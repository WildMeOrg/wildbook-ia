"""
Idea:
    what about the probability of a descriptor match being a score like in SIFT.
    we can learn that too.

Have:
    * semantic and visual uuids
    * Test that accepts unknown annotations one at a time and
      for each runs query, makes decision about name, and executes decision.
    * As a placeholder for exemplar decisions  an exemplar is added if
      number of exemplars per name is less than threshold.
    * vs-one reranking query mode
    * test harness but start with larger test set
    * vs-one score normalizer ~~/ score normalizer for different values of K * / different params~~
      vs-many score normalization doesnt actually matter. We just need the ranking.
    * need to add in the multi-indexer code into the pipeline. Need to
      decide which subindexers to load given a set of daids
    * need to use set query as an exemplar if its vs-one reranking scores
      are below a threshold
    * flip the vsone ratio score so its < .8 rather than > 1.2 or whatever
    * start from nothing and let the system make the first few decisions correctly
    * tell me the correct answer in the automated test
    * turn on multi-indexing. (should just work..., probably bugs though. Just need to throw the switch)
    * paramater to only add exemplar if post-normlized score is above a threshold
    * ensure vsone ratio test is happening correctly
    * normalization gets a cfgstr based on the query
    * need to allow for scores to be un-invalidatd post spatial verification
      e.g. when the first match initially is invalidated through
      spatial verification but the next matches survive.
    * keep distinctiveness weights from vsmany for vsone weighting
      basically involves keeping weights from different filters and not
      aggregating match weights until the end.


TODO:
    * update normalizer (have setup the datastructure to allow for it need to integrate it seemlessly)
    * Improve vsone scoring.
    * score normalization update. on add the new support data, reapply bayes
     rule, and save to the current cache for a given algorithm configuration.
    * test case where there is a 360 view that is linkable from the tests case
    * Put test query mode into the main application and work on the interface for it.
    * spawn background process to reindex chunks of data
    * ~~Remember name_confidence of decisions for manual review~~ Defer
"""
from __future__ import absolute_import, division, print_function
import six
from collections import namedtuple
import utool as ut
import numpy as np
import sys
from ibeis.model.hots import automated_oracle as ao
from ibeis.model.hots import automated_helpers as ah
from ibeis.model.hots import special_query
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')


def incremental_test(ibs_gt, num_initial=0):
    """
    Plots the scores/ranks of correct matches while varying the size of the
    database.

    Args:
        ibs       (list) : IBEISController object
        qaid_list (list) : list of annotation-ids to query

    CommandLine:
        python dev.py -t inc --db PZ_MTEST --qaid 1:30:3 --cmd

        python dev.py --db PZ_MTEST --allgt --cmd

        python dev.py --db PZ_MTEST --allgt -t inc

        python dev.py --db PZ_MTEST --allgt -t inc

        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1

        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0 --interact-after 444440 --noqcache
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1 --interact-after 444440 --noqcache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('PZ_MTEST')
        >>> #num_initial = 0
        >>> num_initial = 0
        >>> incremental_test(ibs_gt, num_initial)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('GZ_ALL')
        >>> num_initial = 100
        >>> incremental_test(ibs_gt, num_initial)
    """

    ibs2, aid_list1, aid1_to_aid2 = ah.setup_incremental_test(ibs_gt, num_initial=num_initial)

    #interact_after = 100
    #interact_after = None
    interact_after = ut.get_argval(('--interactive-after', '--interact-after',), type_=int, default=0)
    threshold = ut.get_sys_maxfloat()  # 1.99

    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    aids_chunk1_iter = ut.progress_chunks(aid_list1, chunksize, lbl='TEST QUERY')
    metatup = (ibs_gt, aid1_to_aid2)
    for count, aids_chunk1 in enumerate(aids_chunk1_iter):
        interactive = (interact_after is not None and count > interact_after)
        # ensure new annot is added (most likely it will have been preadded)
        aids_chunk2 = ah.add_annot_chunk(ibs_gt, ibs2, aids_chunk1, aid1_to_aid2)
        qaid_chunk = aids_chunk2
        execute_teststep(ibs2, qaid_chunk, count, threshold, interactive, metatup)

    #if True:
        #ut.embed()


def generate_incremental_queries(ibs, qaid_list, daid_list):
    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    qaid_chunk_iter = ut.progress_chunks(qaid_list, chunksize, lbl='TEST QUERY')

    # FULL INCREMENT
    #aids_chunk1_iter = ut.ichunks(aid_list1, 1)
    interactive = True
    ibs2 = ibs
    metatup = None
    threshold = ut.get_sys_maxfloat()  # 1.99
    for count, qaid_chunk in enumerate(qaid_chunk_iter):
        yield execute_teststep(ibs2, qaid_chunk, count, threshold, interactive, metatup)


def execute_teststep(ibs, qaid_chunk, count, threshold, interactive,
                     metatup=None):
    """ Add an unseen annotation and run a query """
    sys.stdout.write('\n')
    print('\n==== EXECUTING TESTSTEP %d ====' % (count,))
    exemplar_aids = ibs.get_valid_aids(is_exemplar=True)
    qaid2_qres, qreq_ = special_query.query_vsone_verified(ibs, qaid_chunk, exemplar_aids)
    for qaid, qres in six.iteritems(qaid2_qres):
        make_decisions(ibs, qaid, qres, qreq_, threshold, interactive, metatup)


def get_system_exemplar_suggestion(ibs2, qaid):
    """
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
    max_exemplars = ibs2.cfg.other_cfg.max_exemplars
    other_exemplars = ibs2.get_annot_groundtruth(qaid, is_exemplar=True)
    num_other_exemplars = len(other_exemplars)
    #
    is_non_exemplar = not ibs2.get_annot_exemplar_flags(qaid)
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
            exemplar_distinctivness_thresh = ibs2.cfg.other_cfg.exemplar_distinctivness_thresh
            # Logic to choose query based on exemplar score distance
            qaid_list = [qaid]
            daid_list = other_exemplars
            cfgdict = dict(codename='vsone_norm_csum')
            qres = ibs2.query_chips(qaid_list, daid_list, cfgdict=cfgdict, verbose=False)[0]
            if qres is None:
                is_distinctive = True
            else:
                #ut.embed()
                aid_arr, score_arr = qres.get_aids_and_scores()
                is_distinctive = np.all(aid_arr < exemplar_distinctivness_thresh)
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


def execute_name_decision(ibs2, qaid, name):
    print('setting name to name=%r' % (name,))
    if name is None:
        assert ibs2.is_aid_unknown(qaid), 'animal is already known'
        newname = ibs2.make_next_name()
        print('Adding qaid=%r as newname=%r' % (qaid, newname))
        ibs2.set_annot_names([qaid], [newname])
    else:
        ibs2.set_annot_names([qaid], [name])


def get_qres_choices(ibs2, qres):
    if qres is None:
        nscoretup = list(map(np.array, ([], [], [], [])))
        (sorted_nids, sorted_nscore, sorted_aids, sorted_ascores) = nscoretup
    else:
        nscoretup = qres.get_nscoretup(ibs2)

    (sorted_nids, sorted_nscore, sorted_aids, sorted_ascores) = nscoretup
    sorted_rawscore = [qres.get_aid_scores(aids, rawscore=True) for aids in sorted_aids]

    ChoiceTuple = namedtuple('ChoiceTuple', ('sorted_nids', 'sorted_nscore',
                                             'sorted_rawscore', 'sorted_aids',
                                             'sorted_ascores'))
    choicetup = ChoiceTuple(sorted_nids, sorted_nscore, sorted_rawscore,
                            sorted_aids, sorted_ascores)
    return choicetup


def get_system_name_suggestion(ibs2, choicetup, threshold):
    r"""
    Args:
        ibs2      (IBEISController):
        qaid      (int):  query annotation id
        qres      (QueryResult):  object of feature correspondences and scores
        threshold (float):
        metatup   (None):

    Returns:
        tuple: (autoname_msg, autoname_func)

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1
    """
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup

    autoname_msg_list = []
    if len(sorted_nids) == 0:
        autoname_msg_list.append('Unable to find any matches')
        nid, score, rank = None, None, None
    else:
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

    name = ibs2.get_name_texts(nid) if nid is not None else None
    name_confidence = 0
    return autoname_msg, name, name_confidence


def update_normalizer(ibs2, qreq_, choicetup, name):
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup
    # Get new True Negative support data for score normalization
    rank = ut.listfind(ibs2.get_name_texts(sorted_nids), name)
    tp_rawscore = sorted_rawscore[rank]
    valid_falseranks = set(range(len(sorted_rawscore))) - set([rank])
    if len(valid_falseranks) > 0:
        tn_rank = min(valid_falseranks)
        tn_rawscore = sorted_rawscore[tn_rank][0]
    else:
        tn_rawscore = None
    return tp_rawscore, tn_rawscore

    if tp_rawscore is not None and tn_rawscore is not None:
        # UPDATE SCORE NORMALIZER HERE
        print('new normalization example: tp_rawscore={}, tn_rawscore={}'.format(tp_rawscore, tn_rawscore))
    else:
        print('cannot update score normalization')


def get_user_name_decision(ibs2, qres, qreq_, autoname_msg, name, name_confidence, choicetup):
    r"""
    Prompts the user for input

    Args:
        ibs2 (IBEISController):
        qres (QueryResult):  object of feature correspondences and scores
        autoname_func (function):
    """
    if qres is None:
        print('WARNING: qres is None')
    import plottool as pt

    mplshowtop = True and qres is not None
    qtinspect = False and qres is not None
    if mplshowtop:
        fnum = 513
        print('Showing matplotlib window')
        pt.figure(fnum=fnum, pnum=(2, 3, 1), doclf=True, docla=True)
        fig = qres.ishow_top(ibs2, name_scoring=True, fnum=fnum, in_image=False,
                             annot_mode=0, sidebyside=False, show_query=True)
        fig.show()
        #fig.canvas.raise_()
        #from plottool import fig_presenter
        #fig_presenter.bring_to_front(fig)
    if qtinspect:
        print('Showing qt inspect window')
        qres_wgt = qres.qt_inspect_gui(ibs2, name_scoring=True)
        qres_wgt.show()
        qres_wgt.raise_()
    if qreq_ is not None:
        if qreq_.normalizer is None:
            print('normalizer is None!!')
        else:
            qreq_.normalizer.visualize(update=False, fnum=2)

    newname = ibs2.make_next_name()
    newname_prefix = 'New Name:\n'
    if name is None:
        name = newname_prefix + newname

    aid_list = ut.get_list_column(choicetup.sorted_aids, 0)
    name_options = ibs2.get_annot_names(aid_list) + [newname_prefix + newname]
    import guitool
    msg = 'Decide on query name. System suggests; ' + str(name)
    title = 'name decision'
    options = name_options[::-1]
    user_chosen_name = guitool.user_option(None, msg, title, options)  # NOQA
    if user_chosen_name is None:
        raise AssertionError('User Canceled Query')
    user_chosen_name = user_chosen_name.replace(newname_prefix, '')
    # Prompt the user (this could be swaped out with a qt or web interface)
    if qtinspect:
        qres_wgt.close()
    return user_chosen_name


def get_user_exemplar_decision(autoexemplar_msg, exemplar_decision, exemplar_condience):
    import guitool
    msg = 'Decide if exemplar. System suggests; ' + str(exemplar_decision)
    title = 'exemplar decision'
    options = ['True', 'False'][::-1]
    result = guitool.user_option(None, msg, title, options)  # NOQA
    if result is None:
        raise AssertionError('User Canceled Query')
    if result == 'True':
        return True


@ut.indent_func
def make_decisions(ibs2, qaid, qres, qreq_, threshold, interactive=False,
                   metatup=None):
    r"""
    Either makes automatic decision or asks user for feedback.

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> threshold = 3432432
        >>> # build test data
        >>> ibs2 = ibeis.opendb('testdb1')
        >>> qaid_chunk = ibs2.get_valid_aids()[0:1]
        >>> # execute function
        >>> exemplar_aids = ibs2.get_valid_aids(is_exemplar=True)
        >>> qaid2_qres, qreq_ = special_query.query_vsone_verified(ibs2, qaid_chunk, exemplar_aids)
        >>> qaid = qaid_chunk[0]
        >>> qres = qaid2_qres[qaid_chunk[0]]
        >>> # verify results
        >>> metatup = None
        >>> interactive = True

    qres.ishow_top(ibs, sidebyside=False, show_query=True)
    """
    #ut.embed()
    name_confidence_thresh = 99999999
    exemplar_confidence_thresh = 99999999
    # print query result info
    #if qres is not None;
    #    inspectstr = qres.get_inspect_str(ibs=ibs2, name_scoring=True)
    #    print(ut.msgblock('QUERY RESULT', inspectstr))
    #else:
    #    print('WARNING: qres is None')
    # Get system suggested name
    choicetup = get_qres_choices(ibs2, qres)
    autoname_msg, name, name_confidence = get_system_name_suggestion(ibs2, choicetup, threshold)
    # ---------------------------------------------
    # Get oracle suggestion if we have the metadata
    # override the system suggestion
    print(autoname_msg)
    if metatup is not None:
        autoname_msg, name, name_confidence = ao.get_oracle_name_suggestion(
            ibs2, autoname_msg, qaid, choicetup, metatup)
        #ut.embed()
    # ---------------------------------------------
    if interactive and name_confidence < name_confidence_thresh:
        name = get_user_name_decision(ibs2, qres, qreq_, autoname_msg, name, name_confidence, choicetup)
        #if name is not None:
    execute_name_decision(ibs2, qaid, name)
    #update_normalizer(ibs, qreq_, choicetup, name)
    autoexemplar_msg, exemplar_decision, exemplar_condience = get_system_exemplar_suggestion(ibs2, qaid)
    print(autoexemplar_msg)
    if interactive and exemplar_condience < exemplar_confidence_thresh:
        exemplar_decision = get_user_exemplar_decision(autoexemplar_msg, exemplar_decision, exemplar_condience)
    if exemplar_decision:
        ibs2.set_annot_exemplar_flags((qaid,), [1])


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.automated_matcher
        python -m ibeis.model.hots.automated_matcher --allexamples
        python -m ibeis.model.hots.automated_matcher --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut
    ut.doctest_funcs()
