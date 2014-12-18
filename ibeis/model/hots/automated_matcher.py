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
from six.moves import builtins  # NOQA
import utool as ut
import numpy as np
import sys
from ibeis.model.hots import automated_oracle as ao
from ibeis.model.hots import automated_helpers as ah
from ibeis.model.hots import special_query
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')


# ---- GLOBALS ----

ChoiceTuple = namedtuple('ChoiceTuple', ('sorted_nids', 'sorted_nscore',
                                         'sorted_rawscore', 'sorted_aids',
                                         'sorted_ascores'))


# ---- ENTRY POINT ----


def incremental_test(ibs_gt, num_initial=0):
    """ Adds and queries new annotations one at a time with oracle guidance

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

        python -m ibeis.model.hots.automated_matcher --test-incremental_test:2
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('PZ_MTEST')
        >>> #num_initial = 0
        >>> num_initial = 90
        >>> incremental_test(ibs_gt, num_initial)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('GZ_ALL')
        >>> num_initial = 100
        >>> incremental_test(ibs_gt, num_initial)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('testdb1')
        >>> #num_initial = 0
        >>> num_initial = 0
        >>> incremental_test(ibs_gt, num_initial)
    """
    ibs, aid_list1, aid1_to_aid2 = ah.setup_incremental_test(ibs_gt, num_initial=num_initial)
    #interact_after = 100
    #interact_after = None
    interact_after = ut.get_argval(('--interactive-after', '--interact-after',), type_=int, default=0)
    threshold = ut.get_sys_maxfloat()  # 1.99
    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    aids_chunk1_iter = ut.progress_chunks(aid_list1, chunksize, lbl='TEST QUERY')
    metatup = (ibs_gt, aid1_to_aid2)
    print('begin interactive iter')
    for count, aids_chunk1 in enumerate(aids_chunk1_iter):
        sys.stdout.write('\n')
        print('\n==== EXECUTING TESTSTEP %d ====' % (count,))
        interactive = (interact_after is not None and count > interact_after)
        # ensure new annot is added (most likely it will have been preadded)
        aids_chunk2 = ah.add_annot_chunk(ibs_gt, ibs, aids_chunk1, aid1_to_aid2)
        qaid_chunk = aids_chunk2
        item = execute_teststep(ibs, qaid_chunk, threshold, interactive, metatup)
        (ibs, qres, qreq_, choicetup, metatup, callbacks, threshold) = item
        try_automatic_decision(ibs, qres, qreq_, choicetup, threshold, interactive,
                               metatup, callbacks=callbacks)
    print('ending interactive iter')
    ah.check_results(ibs_gt, ibs, aid1_to_aid2)


import guitool


def incremental_test_qt(ibs, num_initial=0):
    """
    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-incremental_test_qt

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> main_locals = ibeis.main(db='testdb1')
        >>> ibs = main_locals['ibs']
        >>> #num_initial = 0
        >>> num_initial = 0
        >>> incremental_test_qt(ibs, num_initial)
        >>> execstr = ibeis.main_loop(main_locals)
        >>> print(execstr)
    """
    qaid_list = ibs.get_valid_aids()
    daid_list = ibs.get_valid_aids()
    self = WaitForInputQtLoop()
    self = self.request_nonblocking_inc_query(ibs, qaid_list, daid_list)
    pass


INC_LOOP_BASE = guitool.__PYQT__.QtCore.QObject


class WaitForInputQtLoop(INC_LOOP_BASE):
    next_query_signal = guitool.signal_()
    name_decision_signal = guitool.signal_(list)

    def __init__(self):
        INC_LOOP_BASE.__init__(self)
        self.inc_query_gen = None
        self.ibs = None
        self.dry = False
        self.interactive = True
        # connect signals to slots
        self.next_query_signal.connect(self.next_query_slot)
        self.name_decision_signal.connect(self.name_decision_slot)

    def request_nonblocking_inc_query(self, ibs, qaid_list, daid_list):
        self.ibs = ibs

        def emit_name_decision(sorted_aids):
            print(sorted_aids)
            print(';)')
            self.name_decision_signal.emit(sorted_aids)
        callbacks = {
            'next_query_callback': self.next_query_signal.emit,
            #'name_decision_callback': self.name_decision_signal.emit,
            'name_decision_callback': emit_name_decision,
            #'try_decision_callback': self.try_decision_signal.emit
        }
        self.inc_query_gen = generate_incremental_queries(ibs, qaid_list,
                                                          daid_list,
                                                          callbacks=callbacks)
        callbacks['next_query_callback']()
        #pass

    @guitool.slot_()
    def next_query_slot(self):
        try:
            dry = self.dry
            interactive = self.interactive
            item = six.next(self.inc_query_gen)
            (ibs, qres, qreq_, choicetup, metatup, callbacks, threshold) = item
            self.choicetup = choicetup
            self.qres      = qres
            self.qreq_     = qreq_
            self.metatup   = metatup
            self.callbacks = callbacks
            self.threshold = threshold
            try_automatic_decision(ibs, qres, qreq_, choicetup, threshold, interactive=interactive,
                                   metatup=metatup, dry=dry, callbacks=callbacks)
        except StopIteration:
            print('NO MORE QUERIES. CLOSE DOWN WINDOWS AND DISPLAY DONE MESSAGE')
            pass

    @guitool.slot_(list)
    def name_decision_slot(self, sorted_aids):
        print('[QT] name_decision_slot')
        try:
            ibs = self.ibs
            choicetup   = self.choicetup
            qres        = self.qres
            qreq_       = self.qreq_
            metatup     = self.metatup
            callbacks   = self.callbacks
            threshold   = self.threshold
            interactive = self.interactive
            dry         = self.dry
            if sorted_aids is None or len(sorted_aids) == 0:
                name = None
            else:
                name = ibs.get_annot_names(sorted_aids[0])
            make_name_decision(name, choicetup, ibs, qres, qreq_, threshold,
                               interactive=interactive, metatup=metatup,
                               dry=dry, callbacks=callbacks)
        except StopIteration:
            print('NO MORE QUERIES. CLOSE DOWN WINDOWS AND DISPLAY DONE MESSAGE')
            pass


def generate_incremental_queries(ibs, qaid_list, daid_list, callbacks=None):
    r""" generates incremental queries that completely new to the system

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid_list (list):
        daid_list (list):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-generate_incremental_queries

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_list = '?'
        >>> daid_list = '?'
        >>> # execute function
        >>> result = generate_incremental_queries(ibs, qaid_list, daid_list)
        >>> # verify results
        >>> print(result)
    """
    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    qaid_chunk_iter = ut.progress_chunks(qaid_list, chunksize, lbl='TEST QUERY')

    # FULL INCREMENT
    #aids_chunk1_iter = ut.ichunks(aid_list1, 1)
    interactive = True
    ibs = ibs
    metatup = None
    threshold = ut.get_sys_maxfloat()  # 1.99
    for count, qaid_chunk in enumerate(qaid_chunk_iter):
        sys.stdout.write('\n')
        print('\n==== EXECUTING TESTSTEP %d ====' % (count,))
        for item in execute_teststep(ibs, qaid_chunk, threshold, interactive,
                                     metatup, callbacks=callbacks):
            yield item


# ---- QUERY ----

def execute_teststep(ibs, qaid_chunk, threshold, interactive, metatup=None,
                     callbacks=None):
    """ Add an unseen annotation and run a query

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid_chunk (?):
        count (?):
        threshold (?):
        interactive (?):
        metatup (None):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-execute_teststep

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_chunk = [1]
        >>> threshold = '?'
        >>> interactive = '?'
        >>> metatup = None
        >>> # execute function
        >>> result = execute_teststep(ibs, qaid_chunk, threshold, interactive, metatup)
        >>> # verify results
        >>> print(result)
    """
    VSEXEMPLAR = True
    if VSEXEMPLAR:
        daid_list = ibs.get_valid_aids(is_exemplar=True)
    else:
        # Hacky half-written code to get daids from an encounter that have
        # been given a temporary name already.
        eids_list = ibs.get_image_eids(ibs.get_annot_gids(qaid_chunk))
        eid = eids_list[0][0]
        daid_list = ibs.get_valid_aids(eid=eid)
        daid_list = ut.filterfalse_items(daid_list, ibs.is_aid_unknown(daid_list))
    qaid2_qres, qreq_ = special_query.query_vsone_verified(ibs, qaid_chunk, daid_list)

    #try_decision_callback = callbacks.get('try_decision_callback', None)
    for qaid, qres in six.iteritems(qaid2_qres):
        choicetup = get_qres_choices(ibs, qres)
        yield (ibs, qres, qreq_, choicetup, metatup, callbacks, threshold)
        builtins.print('[TRY 3] NOTHING ELSE SHOULD HAPPEN')


# ---- PRE DECISION ---

def get_qres_choices(ibs, qres):
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
        python -m ibeis.model.hots.automated_matcher --test-get_qres_choices

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict())[1]
        >>> # execute function
        >>> choicetup = get_qres_choices(ibs, qres)
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


def try_automatic_decision(ibs, qres, qreq_, choicetup, threshold, interactive=False,
                            metatup=None, dry=False, callbacks=None):
    r""" Either makes automatic decision or asks user for feedback.

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1
        python -m ibeis.model.hots.automated_matcher --test-try_automatic_decision

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_chunk = ibs.get_valid_aids()[0:1]
        >>> # execute function
        >>> exemplar_aids = ibs.get_valid_aids(is_exemplar=True)
        >>> dry = True
        >>> qaid2_qres, qreq_ = special_query.query_vsone_verified(ibs, qaid_chunk, exemplar_aids)
        >>> qaid = qaid_chunk[0]
        >>> qres = qaid2_qres[qaid]
        >>> # verify results
        >>> metatup = None
        >>> threshold = -1
        >>> interactive = False
        >>> metatup = None
        >>> dry = False
        >>> # execute function
        >>> result = try_automatic_decision(ibs, qres, qreq_, threshold,
        ...                                 choicetup, interactive, metatup, dry)
        >>> # verify results
        >>> print(result)

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
    autoname_msg, name, name_confidence = get_system_name_suggestion(ibs, choicetup)
    # ---------------------------------------------
    # Get oracle suggestion if we have the metadata
    # override the system suggestion
    print(autoname_msg)
    if metatup is not None:
        autoname_msg, name, name_confidence = ao.get_oracle_name_suggestion(
            ibs, autoname_msg, qaid, choicetup, metatup)
        #ut.embed()
    # ---------------------------------------------
    # WE MAY NEED TO DO CALLBACKS HERE
    try_automatic_name_decision(autoname_msg, name, name_confidence, choicetup,
                                ibs, qres, qreq_, threshold, interactive=interactive,
                                metatup=metatup, dry=dry, callbacks=callbacks)
    builtins.print('[TRY 2] NOTHING ELSE SHOULD HAPPEN')


def try_automatic_name_decision(autoname_msg, name, name_confidence, choicetup,
                                ibs, qres, qreq_, threshold, interactive=False,
                                metatup=None, dry=False, callbacks=None):
    name_confidence_thresh = ut.get_sys_maxfloat()
    if interactive and name_confidence < name_confidence_thresh:
        get_user_name_decision(ibs, qres, qreq_, autoname_msg, name,
                               name_confidence, choicetup, callbacks=callbacks)
        builtins.print('[TRY 1] NOTHING ELSE SHOULD HAPPEN')
    else:
        # May need to execute callback whereas whatever the interaction was
        # would issue it otherwise
        make_name_decision()
        #if name is not None:


def make_name_decision(name, choicetup, ibs, qres, qreq_, threshold,
                       interactive=True, metatup=None,
                       dry=False, callbacks=None):
    if not dry:
        qaid = qres.get_qaid()
        execute_name_decision(ibs, qaid, name)
    try_automatic_exemplar_decision(choicetup, ibs, qres, qreq_, threshold,
                                    interactive=interactive, metatup=metatup,
                                    dry=dry, callbacks=callbacks)


def try_automatic_exemplar_decision(choicetup, ibs, qres, qreq_, threshold,
                                    interactive=False, metatup=None, dry=False,
                                    callbacks=None):
    qaid = qres.get_qaid()
    exemplar_confidence_thresh = ut.get_sys_maxfloat()
    #update_normalizer(ibs, qreq_, choicetup, name)
    autoexemplar_msg, exemplar_decision, exemplar_condience = get_system_exemplar_suggestion(ibs, qaid)
    print(autoexemplar_msg)
    if False and interactive and exemplar_condience < exemplar_confidence_thresh:
        exemplar_decision = get_user_exemplar_decision(autoexemplar_msg, exemplar_decision, exemplar_condience)
    else:
        # May need to execute callback whereas whatever the interaction was
        # would issue it otherwise
        pass
    if exemplar_decision:
        if not dry:
            ibs.set_annot_exemplar_flags((qaid,), [1])


# ---- ALGORITHM / USER INPUT -----

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
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1
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
        >>> # execute function
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

    name = ibs.get_name_texts(nid) if nid is not None else None
    name_confidence = 0
    return autoname_msg, name, name_confidence


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
            exemplar_distinctivness_thresh = ibs.cfg.other_cfg.exemplar_distinctivness_thresh
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


def get_user_name_decision(ibs, qres, qreq_, autoname_msg, name,
                           name_confidence, choicetup, callbacks=None):
    r""" hooks into to some method of getting user input for names

    TODO: really good interface

    Prompts the user for input

    Args:
        ibs (IBEISController):
        qres (QueryResult):  object of feature correspondences and scores
        autoname_func (function):
    """
    if qres is None:
        print('WARNING: qres is None')
    import plottool as pt

    new_mplshow = True and qres is not None
    mplshowtop = False and qres is not None
    qtinspect = False and qres is not None

    if new_mplshow:
        from ibeis.viz.interact import interact_query_decision
        print('Showing matplotlib window')
        comp_aids_all = ut.get_list_column(choicetup.sorted_aids, 0)
        comp_aids     = comp_aids_all[0:min(3, len(comp_aids_all))]
        suggestx      = ut.listfind(ibs.get_annot_names(comp_aids), name)
        suggest_aid   = None if suggestx is None else comp_aids[suggestx]
        name_decision_callback = callbacks['name_decision_callback']
        builtins.print('Calling interact_query_decision')
        qvi = interact_query_decision.QueryVerificationInteraction(
            ibs, qres, comp_aids, suggest_aid, decision_callback=name_decision_callback)
        qvi.fig.show()
    if mplshowtop:
        fnum = 513
        pt.figure(fnum=fnum, pnum=(2, 3, 1), doclf=True, docla=True)
        fig = qres.ishow_top(ibs, name_scoring=True, fnum=fnum, in_image=False,
                             annot_mode=0, sidebyside=False, show_query=True)
        fig.show()
        #fig.canvas.raise_()
        #from plottool import fig_presenter
        #fig_presenter.bring_to_front(fig)
        newname = ibs.make_next_name()
        newname_prefix = 'New Name:\n'
        if name is None:
            name = newname_prefix + newname

        aid_list = ut.get_list_column(choicetup.sorted_aids, 0)
        name_options = ibs.get_annot_names(aid_list) + [newname_prefix + newname]
        import guitool
        msg = 'Decide on query name. System suggests; ' + str(name)
        title = 'name decision'
        options = name_options[::-1]
        user_chosen_name = guitool.user_option(None, msg, title, options)  # NOQA
        if user_chosen_name is None:
            raise AssertionError('User Canceled Query')
        user_chosen_name = user_chosen_name.replace(newname_prefix, '')
    if qtinspect:
        print('Showing qt inspect window')
        qres_wgt = qres.qt_inspect_gui(ibs, name_scoring=True)
        qres_wgt.show()
        qres_wgt.raise_()
    if qreq_ is not None:
        if qreq_.normalizer is None:
            print('normalizer is None!!')
        else:
            qreq_.normalizer.visualize(update=False, fnum=2)

    # Prompt the user (this could be swaped out with a qt or web interface)
    #if qtinspect:
    #    qres_wgt.close()
    #return user_chosen_name


def get_user_exemplar_decision(autoexemplar_msg, exemplar_decision,
                               exemplar_condience, callbacks=None):
    r""" hooks into to some method of getting user input for exemplars

    TODO: really good interface

    Args:
        autoexemplar_msg (?):
        exemplar_decision (?):
        exemplar_condience (?):

    Returns:
        ?: True

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-get_user_exemplar_decision

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> autoexemplar_msg = '?'
        >>> exemplar_decision = '?'
        >>> exemplar_condience = '?'
        >>> # execute function
        >>> True = get_user_exemplar_decision(autoexemplar_msg, exemplar_decision, exemplar_condience)
        >>> # verify results
        >>> result = str(True)
        >>> print(result)
    """
    import guitool
    options = ['No', 'Yes']
    msg = 'Decide if exemplar. System suggests; ' + options[exemplar_decision]
    title = 'exemplar decision'
    responce = guitool.user_option(None, msg, title, options)  # NOQA
    if responce is None:
        raise AssertionError('User Canceled Query')
    if responce == 'Yes':
        return True
    elif responce == 'No':
        return False
    else:
        return None

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
        >>> # execute function
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
        >>> # execute function
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
