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
    * add matches to multiple animals (merge)

Tasks:

    Algorithm::
        * Incremental query needs to handle
            - test mode and live mode
            - normalizer update
            - use correct distinctivenes score in vsone
            - tested application of distinctiveness, foreground, ratio,
                spatial_verification, vsone verification, and score
                normalization.

        * Mathematically formal description of the space of choices
            - getting the proability of each choice will give us a much better
                confidence measure for our decision. An example of a probability
                partition might be .2 - merge with rank1.  .2 merge with rank 2, .5
                merge with rank1 and rank2, .1 others

        * Improved automated exemplar decision mechanism

        * Improved automated name decision mechanism

     SQL::
         * New Image Columns
             - image_posix_timedelta

         * New Name Columns
             - name_temp_flag
             - name_alias_text

             - name_uuid
             - name_visual_uuid
             - name_member_annot_rowids_evalstr
             - name_member_num_annot_rowids

         * New Encounter Columns
             - encounter_start_time
             - encounter_end_time
             - encounter_lat
             - encounter_lon
             - encounter_processed_flag
             - encounter_shipped_flag

    Decision UIs::
        * Query versus top N results
            - ability to draw an undirected edge between the query and any number of
                results. ie create a match any of the top results
            - a match to more than one results should by default merge the two names
                (this involves a name enhancement subtask). trigger a split / merge dialog
        * Is Exemplar
            - allows for user to set the exemplars for a given name
        * Name Progress
            - Shows the current name matching progress
        * Split
            - Allows a user to split off some images from a name into a new name
              or some other name.
        * Merge
            - Allows a user to join two names.


    GUI::
        * NameTree needs to not refresh unless absolutely necessary
        * Time Sync
        * Encounter metadata sync from the SMART
        * Hide shipped encounters
            - put flag to turn them on
        * Mark processed encounters
        * Gui naturally ensures that all annotations in the query belong
           to the same species
        * Garbage collection function that removes all non-exemplar
          information from encounters that have been shipped.
        * Spawn process that reindexes large chunks of descriptors as the
          database grows.


LONG TERM TASKS:

    Architecture:
        * Pipeline needs
            - DEFER: a move from dict based representation to list based
            - DEFER: spatial verification cyth speedup
            - DEFER: nearest neighbor (based on visual uuid caching) caching

    Controller:
         * LONGTERM: AutogenController
             - register data convertors for verts / other eval columns. Make
               several convertors standard and we can tag those columns to
               autogenerate their functions.
             - be able to mark a column as determined by the aggregate of other
               columns. Then the data is either generated on the fly, or it is
               cached and the necessary book-keeping functions are
               autogenerated.

    Decision UIs::
        * Is Exemplar
            - LONG TERM: it would be cool if they were visualized by using
              networkx or some gephi like program and clustered by match score.

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
from ibeis.model.hots import system_suggestor
from ibeis.model.hots import user_dialogs
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')


# ---- GLOBALS ----

ChoiceTuple = namedtuple('ChoiceTuple', ('sorted_nids', 'sorted_nscore',
                                         'sorted_rawscore', 'sorted_aids',
                                         'sorted_ascores'))


# ---- ENTRY POINT ----


def test_incremental_queries(ibs_gt, num_initial=0):
    """ Adds and queries new annotations one at a time with oracle guidance

    Args:
        ibs       (list) : IBEISController object
        qaid_list (list) : list of annotation-ids to query

    CommandLine:
        python dev.py -t inc --db PZ_MTEST --qaid 1:30:3 --cmd
        python dev.py --db PZ_MTEST --allgt --cmd
        python dev.py --db PZ_MTEST --allgt -t inc
        python dev.py --db PZ_MTEST --allgt -t inc
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:0
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:1
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:0 --interact-after 444440 --noqcache
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:1 --interact-after 444440 --noqcache

        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:2
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('PZ_MTEST')
        >>> #num_initial = 0
        >>> num_initial = 90
        >>> test_incremental_queries(ibs_gt, num_initial)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('GZ_ALL')
        >>> num_initial = 100
        >>> test_incremental_queries(ibs_gt, num_initial)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('testdb1')
        >>> #num_initial = 0
        >>> num_initial = 0
        >>> test_incremental_queries(ibs_gt, num_initial)
    """
    ibs, aid_list1, aid1_to_aid2 = ah.setup_incremental_test(ibs_gt, num_initial=num_initial)
    #interact_after = 100
    #interact_after = None
    interact_after = ut.get_argval(('--interactive-after', '--interact-after',), type_=int, default=0)
    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    aids_chunk1_iter = ut.progress_chunks(aid_list1, chunksize, lbl='TEST QUERY')
    metatup = (ibs_gt, aid1_to_aid2)
    print('begin interactive iter')
    incinfo = {
        'metatup': metatup,
        'interactive': False,
        'dry': False,
        #'name_confidence_thresh':
    }
    VSEXEMPLAR = True
    if VSEXEMPLAR:
        daid_list = ibs.get_valid_aids(is_exemplar=True)
    #else:
    #    # Hacky half-written code to get daids from an encounter that have
    #    # been given a temporary name already.
    #    eids_list = ibs.get_image_eids(ibs.get_annot_gids(qaid_chunk))
    #    eid = eids_list[0][0]
    #    daid_list = ibs.get_valid_aids(eid=eid)
    #    daid_list = ut.filterfalse_items(daid_list, ibs.is_aid_unknown(daid_list))

    for count, aids_chunk1 in enumerate(aids_chunk1_iter):
        sys.stdout.write('\n')
        print('\n==== EXECUTING TESTSTEP %d ====' % (count,))
        incinfo['interactive'] = (interact_after is not None and count > interact_after)
        # ensure new annot is added (most likely it will have been preadded)
        aids_chunk2 = ah.add_annot_chunk(ibs_gt, ibs, aids_chunk1, aid1_to_aid2)
        qaid_chunk = aids_chunk2
        for item in generate_subquery_steps(ibs, qaid_chunk, daid_list, incinfo=incinfo):
            (ibs, qres, qreq_, choicetup, incinfo) = item
            run_until_name_decision_signal(ibs, qres, qreq_, choicetup, incinfo=incinfo)
    print('ending interactive iter')
    ah.check_results(ibs_gt, ibs, aid1_to_aid2)


def generate_incremental_queries(ibs, qaid_list, incinfo=None):
    r"""

    qt entry point

    generates incremental queries that completely new to the system

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid_list (list):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-generate_incremental_queries

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_list = '?'
        >>> result = generate_incremental_queries(ibs, qaid_list, daid_list)
        >>> # verify results
        >>> print(result)
    """
    # Execute each query as a test
    chunksize = 1

    # FIXME, Dont take in daid list here. Its impossible
    # Need to update daids for every query with new exemplars

    #assert daid_list is None, 'fixme take in daid_list'
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    qaid_chunk_iter = ut.progress_chunks(qaid_list, chunksize, lbl='TEST QUERY')

    # FULL INCREMENT
    #aids_chunk1_iter = ut.ichunks(aid_list1, 1)
    ibs = ibs
    for count, qaid_chunk in enumerate(qaid_chunk_iter):
        sys.stdout.write('\n')
        print('\n==== EXECUTING TESTSTEP %d ====' % (count,))
        for item in generate_subquery_steps(ibs, qaid_chunk, incinfo=incinfo):
            yield item


# ---- QUERY ----

def generate_subquery_steps(ibs, qaid_chunk, incinfo=None):
    """ Add an unseen annotation and run a query

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid_chunk (?):
        count (?):
        metatup (None):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-generate_subquery_steps

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_chunk = [1]
        >>> result = generate_subquery_steps(ibs, qaid_chunk)
        >>> # verify results
        >>> print(result)
    """
    # Execute actual queries
    qaid2_qres, qreq_ = special_query.query_vsone_verified(ibs, qaid_chunk)

    #try_decision_callback = incinfo.get('try_decision_callback', None)
    for qaid, qres in six.iteritems(qaid2_qres):
        choicetup = get_qres_name_choices(ibs, qres)
        item = [ibs, qres, qreq_, choicetup, incinfo]
        yield item


# ---- PRE DECISION ---

def get_qres_name_choices(ibs, qres):
    r""" returns all possible decision a user could make

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
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
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


# ---- DECISION ---


def run_until_name_decision_signal(ibs, qres, qreq_, choicetup, incinfo=None):
    r""" Either makes automatic decision or asks user for feedback.

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:0
        python -m ibeis.model.hots.automated_matcher --test-test_incremental_queries:1
        python -m ibeis.model.hots.automated_matcher --test-run_until_name_decision_signal

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_chunk = ibs.get_valid_aids()[0:1]
        >>> exemplar_aids = ibs.get_valid_aids(is_exemplar=True)
        >>> dry = True
        >>> qaid2_qres, qreq_ = special_query.query_vsone_verified(ibs, qaid_chunk, exemplar_aids)
        >>> qaid = qaid_chunk[0]
        >>> qres = qaid2_qres[qaid]
        >>> # verify results
        >>> run_until_name_decision_signal(ibs, qres, qreq_, choicetup, incinfo)

    qres.ishow_top(ibs, sidebyside=False, show_query=True)
    """
    qaid = qres.get_qaid()
    #ut.embed()
    # print query result info
    #if qres is not None;
    #    inspectstr = qres.get_inspect_str(ibs=ibs, name_scoring=True)
    #    print(ut.msgblock('QUERY RESULT', inspectstr))
    #else:
    #    print('WARNING: qres is None')
    # Get system suggested name
    autoname_msg, name, name_confidence = system_suggestor.get_system_name_suggestion(ibs, choicetup)
    # ---------------------------------------------
    # Get oracle suggestion if we have the metadata
    # override the system suggestion
    print(autoname_msg)
    if incinfo.get('metatup', None) is not None:
        metatup = incinfo['metatup']
        autoname_msg, name, name_confidence = ao.get_oracle_name_suggestion(
            ibs, autoname_msg, qaid, choicetup, metatup)
        #ut.embed()
    # ---------------------------------------------
    # Either make the name signal callback if confident or ask
    # user input
    name_confidence_thresh = incinfo.get('name_confidence_thresh', ut.get_sys_maxfloat())
    if incinfo.get('interactive', False) and name_confidence < name_confidence_thresh:
        user_dialogs.wait_for_user_name_decision(ibs, qres, qreq_, autoname_msg, name,
                                                 name_confidence, choicetup,
                                                 incinfo=incinfo)
    else:
        # May need to execute callback whereas whatever the interaction was
        # would issue it otherwise Noncallback version
        if name_confidence < name_confidence_thresh:
            assert False, 'TODO: make no decision, but continue looping'
        else:
            exec_name_decision_and_continue(name, choicetup, ibs, qres, qreq_, incinfo=incinfo)
        #if name is not None:


def exec_name_decision_and_continue(name, choicetup, ibs, qres, qreq_,
                                    incinfo=None):
    if not incinfo.get('dry', False):
        qaid = qres.get_qaid()
        execute_name_decision(ibs, qaid, name)
        #update_normalizer(ibs, qreq_, choicetup, name)
    run_until_exemplar_decision_signal(ibs, qres, qreq_, incinfo=incinfo)


def run_until_exemplar_decision_signal(ibs, qres, qreq_, incinfo=None):
    qaid = qres.get_qaid()
    exemplar_confidence_thresh = ut.get_sys_maxfloat()
    exmplr_suggestion = system_suggestor.get_system_exemplar_suggestion(ibs, qaid)
    (autoexemplar_msg, exemplar_decision, exemplar_condience) = exmplr_suggestion
    print(autoexemplar_msg)
    need_user_input = (False and
                       incinfo.get('interactive', False) and
                       exemplar_condience < exemplar_confidence_thresh)
    if need_user_input:
        user_dialogs.wait_for_user_exemplar_decision(
            autoexemplar_msg, exemplar_decision, exemplar_condience)
    else:
        # May need to execute callback whereas whatever the interaction was
        # would issue it otherwise
        exec_exemplar_decision_and_continue(exemplar_decision, ibs, qres, qreq_, incinfo=incinfo)


def exec_exemplar_decision_and_continue(exemplar_decision, ibs, qres, qreq_,
                                        incinfo=None):
    qaid = qres.get_qaid()
    if exemplar_decision:
        if not incinfo.get('dry', False):
            ibs.set_annot_exemplar_flags((qaid,), [1])
    if incinfo is not None:
        # This query run as eneded
        incinfo['next_query_callback']()


# ---- POST DECISION ---


def execute_name_decision(ibs, qaid, name):
    r""" sets the name of qaid to be name

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid (int):  query annotation id
        name (?):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-execute_name_decision

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = '?'
        >>> qaid = '?'
        >>> name = '?'
        >>> result = execute_name_decision(ibs, qaid, name)
        >>> # verify results
        >>> print(result)
    """
    print('setting name to name=%r' % (name,))
    return
    if name is None:
        assert ibs.is_aid_unknown(qaid), 'animal is already known'
        newname = ibs.make_next_name()
        print('Adding qaid=%r as newname=%r' % (qaid, newname))
        ibs.set_annot_names([qaid], [newname])
    else:
        ibs.set_annot_names([qaid], [name])


def update_normalizer(ibs, qreq_, choicetup, name):
    r""" adds new support data to the current normalizer

    FIXME: a miss-save in vim will trigger module unloading

    Args:
        ibs (IBEISController):  ibeis controller object
        qreq_ (QueryRequest):  query request object with hyper-parameters
        choicetup (?):
        name (?):

    Returns:
        tuple: (tp_rawscore, tn_rawscore)

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-update_normalizer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = '?'
        >>> qreq_ = '?'
        >>> choicetup = '?'
        >>> name = '?'
        >>> (tp_rawscore, tn_rawscore) = update_normalizer(ibs, qreq_, choicetup, name)
        >>> # verify results
        >>> result = str((tp_rawscore, tn_rawscore))
        >>> print(result)
    """
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup
    # Get new True Negative support data for score normalization
    rank = ut.listfind(ibs.get_name_texts(sorted_nids), name)
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


# --- TESTING ---


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
