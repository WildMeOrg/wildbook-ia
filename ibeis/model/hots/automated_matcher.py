"""
Idea:
    what about the probability of a descriptor match being a score.

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

TODO:
    * ~~Remember confidence of decisions for manual review~~
      Defer
    * need to use set query as an exemplar if its vs-one reranking scores
      are below a threshold

New TODO:
    * update normalizer (have setup the datastructure to allow for it need to integrate it seemlessly)
    * Improve vsone scoring.
    * Put this query mode into the main application and work on the interface for it.
    * normalization gets a cfgstr based on the query

TODO:
    * need to allow for scores to be re-added post spatial verification
      e.g. when the first match initially is invalidated through
      spatial verification but the next matches survive.

    * score normalization update. on a decision add the point and redo score
      normalization

    * test case where there is a 360 view that is linkable from the tests case

    * spawn background process to reindex chunks of data

    * keep distinctiveness weights from vsmany for vsone weighting

    * ensure vsone ratio test is happening correctly

HAVEDONE:
    * flip the vsone ratio score so its < .8 rather than > 1.2 or whatever
    * start from nothing and let the system make the first few decisions
      correctly
    * tell me the correct answer in the automated test
    * turn on multi-indexing. (should just work..., probably bugs though. Just need to throw the switch)
    * paramater to only add exemplar if post-normlized score is above a
      threshold
"""
from __future__ import absolute_import, division, print_function
import ibeis
import six
import utool as ut
import numpy as np
import functools
import sys
from six.moves import input
from ibeis.model.hots import automated_helpers as ah
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')


def incremental_test(ibs1, num_initial=0):
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

        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1

    Example:
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs1 = ibeis.opendb('PZ_MTEST')
        >>> #num_initial = 0
        >>> num_initial = 20
        >>> incremental_test(ibs1, num_initial)

    Example2:
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs1 = ibeis.opendb('GZ_ALL')
        >>> num_initial = 100
        >>> incremental_test(ibs1, num_initial)
    """

    def execute_teststep(ibs1, ibs2, aids_chunk1, aid1_to_aid2, interactive):
        """ Add an unseen annotation and run a query """
        sys.stdout.write('\n')
        print('\n==== EXECUTING TESTSTEP ====')
        # ensure new annot is added (most likely it will have been preadded)
        aids_chunk2 = add_annot_chunk(ibs1, ibs2, aids_chunk1, aid1_to_aid2)
        threshold = 1.99
        exemplar_aids = ibs2.get_valid_aids(is_exemplar=True)
        qaid2_qres, qreq_ = ah.query_vsone_verified(ibs2, aids_chunk2, exemplar_aids)
        metatup = (ibs1, ibs2, aid1_to_aid2)
        make_decisions(ibs2, qaid2_qres, qreq_, threshold,
                       interactive=interactive,
                       metatup=metatup)

    ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs1, num_initial=num_initial)

    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    aids_chunk1_iter = ut.progress_chunks(aid_list1, chunksize, lbl='TEST QUERY')

    #interactive = DEFAULT_INTERACTIVE
    interact_after = 10
    #interact_after = 100
    #interact_after = None

    # FULL INCREMENT
    #aids_chunk1_iter = ut.ichunks(aid_list1, 1)
    for count, aids_chunk1 in enumerate(aids_chunk1_iter):
        try:
            interactive = (interact_after is not None and count > interact_after)
            execute_teststep(ibs1, ibs2, aids_chunk1, aid1_to_aid2, interactive)
        except KeyboardInterrupt:
            print('Caught keyboard interupt')
            print('interact_after is currently=%r' % (interact_after))
            print('What would you like to change it to?')
            #you like to become interactive?')
            ans = input('...enter new interact after val')
            if ans == 'None':
                interact_after = None
            else:
                interact_after = int(ans)


def setup_incremental_test(ibs1, num_initial=0):
    r"""
    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-setup_incremental_test:0

        python dev.py -t custom --cfg codename:vsone_unnorm --db PZ_MTEST --allgt --vf --va
        python dev.py -t custom --cfg codename:vsone_unnorm --db PZ_MTEST --allgt --vf --va --index 0 4 8 --verbose

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> ibs1 = ibeis.opendb('PZ_MTEST')
        >>> num_initial = 0
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs1, num_initial)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> ibs1 = ibeis.opendb('GZ_ALL')
        >>> num_initial = 100
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs1, num_initial)
    """
    # Take a known dataase
    # Create an empty database to test in
    aid_list1 = ibs1.get_aids_with_groundtruth()
    reset = False
    #reset = True

    aid1_to_aid2 = {}  # annotation mapping

    def make_incremental_test_database(ibs1, aid_list1, reset):
        """
        Makes test database. adds image and annotations but does not transfer names.
        if reset is true the new database is gaurenteed to be built from a fresh
        start.

        Args:
            ibs1      (IBEISController):
            aid_list1 (list):
            reset     (bool):

        Returns:
            IBEISController: ibs2
        """
        print('make_incremental_test_database. reset=%r' % (reset,))
        dbname2 = '_INCREMENTALTEST_' + ibs1.get_dbname()
        ibs2 = ibeis.opendb(dbname2, allow_newdir=True, delete_ibsdir=reset, use_cache=False)

        # reset if flag specified or no data in ibs2
        if reset or len(ibs2.get_valid_gids()) == 0:
            assert len(ibs2.get_valid_aids())  == 0
            assert len(ibs2.get_valid_gids())  == 0
            assert len(ibs2.get_valid_nids())  == 0

            # Get annotations and their images from database 1
            gid_list1 = ibs1.get_annot_gids(aid_list1)
            gpath_list1 = ibs1.get_image_paths(gid_list1)

            # Add all images from database 1 to database 2
            gid_list2 = ibs2.add_images(gpath_list1, auto_localize=False)

            # Image UUIDS should be consistent between databases
            image_uuid_list1 = ibs1.get_image_uuids(gid_list1)
            image_uuid_list2 = ibs2.get_image_uuids(gid_list2)
            assert image_uuid_list1 == image_uuid_list2
            ut.assert_lists_eq(image_uuid_list1, image_uuid_list2)
        return ibs2

    ibs2 = make_incremental_test_database(ibs1, aid_list1, reset)

    # Add the annotations without names

    aids_chunk1 = aid_list1
    aid_list2 = add_annot_chunk(ibs1, ibs2, aids_chunk1, aid1_to_aid2)

    # Assert annotation visual uuids are in agreement
    ah.annot_consistency_checks(ibs1, ibs2, aid_list1, aid_list2)

    # Remove name exemplars
    ah.ensure_clean_data(ibs1, ibs2, aid_list1, aid_list2)

    # Preprocess features and such
    ibs2.ensure_annotation_data(aid_list2, featweights=True)

    print('Transfer %d initial test annotations' % (num_initial,))
    if num_initial > 0:
        # Transfer some initial data
        aid_sublist1 = aid_list1[0:num_initial]
        aid_sublist2 = aid_list2[0:num_initial]
        name_list = ibs1.get_annot_names(aid_sublist1)
        ibs2.set_annot_names(aid_sublist2, name_list)
        ibs2.set_annot_exemplar_flags(aid_sublist2, [True] * len(aid_sublist2))
        aid_list1 = aid_list1[num_initial:]
    print(ibs2.get_dbinfo_str())
    return ibs2, aid_list1, aid1_to_aid2


def add_annot_chunk(ibs1, ibs2, aids_chunk1, aid1_to_aid2):
    """
    adds annotations to the tempoarary database and prevents duplicate
    additions.

    aids_chunk1 = aid_list1

    Args:
        ibs1         (IBEISController):
        ibs2         (IBEISController):
        aids_chunk1  (list):
        aid1_to_aid2 (dict):

    Returns:
        list: aids_chunk2
    """
    # Visual info
    guuids_chunk1 = ibs1.get_annot_image_uuids(aids_chunk1)
    verts_chunk1  = ibs1.get_annot_verts(aids_chunk1)
    thetas_chunk1 = ibs1.get_annot_thetas(aids_chunk1)
    # Non-name semantic info
    species_chunk1 = ibs1.get_annot_species(aids_chunk1)
    gids_chunk2 = ibs2.get_image_gids_from_uuid(guuids_chunk1)
    ut.assert_all_not_None(gids_chunk2, 'gids_chunk2')
    # Add this new unseen test case to the database
    aids_chunk2 = ibs2.add_annots(gids_chunk2,
                                  species_list=species_chunk1,
                                  vert_list=verts_chunk1,
                                  theta_list=thetas_chunk1,
                                  prevent_visual_duplicates=True)
    def register_annot_mapping(aids_chunk1, aids_chunk2, aid1_to_aid2):
        """
        called by add_annot_chunk
        """
        # Should be 1 to 1
        for aid1, aid2 in zip(aids_chunk1, aids_chunk2):
            if aid1 in aid1_to_aid2:
                assert aid1_to_aid2[aid1] == aid2
            else:
                aid1_to_aid2[aid1] = aid2
    # Register the mapping from ibs1 to ibs2
    register_annot_mapping(aids_chunk1, aids_chunk2, aid1_to_aid2)
    print('Added: aids_chunk2=%s' % (ut.truncate_str(repr(aids_chunk2), maxlen=60),))
    return aids_chunk2


def autodecide_newname(ibs2, qaid):
    if ibs2.is_aid_unknown(qaid):
        #if metatup is not None:
        #    #ut.embed()
        #    (ibs1, ibs2, aid1_to_aid2) = metatup
        #    aid2_to_aid1 = ut.invert_dict(aid1_to_aid2)
        #    qaid1 = aid2_to_aid1[qaid]
        #    # Wow, the program just happend to choose a new
        #    # name that was the same as the other database
        #    # I wonder how it did that...
        #    newname = ibs1.get_annot_names(qaid1)
        #else:
        #    # actual new name
        newname = ibs2.make_next_name()
        print('Adding qaid=%r as newname=%r' % (qaid, newname))
        ibs2.set_annot_names([qaid], [newname])
    else:
        print('already has name')
    return get_suggested_exemplar_decision(ibs2, qaid)


def autodecide_match(ibs2, qaid, nid):
    print('setting nameid to nid=%r' % nid)
    ibs2.set_annot_name_rowids([qaid], [nid])
    return get_suggested_exemplar_decision(ibs2, qaid)


def get_suggested_exemplar_decision(ibs2, qaid):
    """
    TODO:
        do a vsone query between all of the exemplars to see if this one is good
        enough to be added.

    SeeAlso:
        ibsfuncs.prune_exemplars
    """
    print('Deciding if adding qaid=%r as an exemplar' % (qaid,))
    # Need a good criteria here
    max_exemplars = ibs2.cfg.other_cfg.max_exemplars
    other_exemplars = ibs2.get_annot_groundtruth(qaid, is_exemplar=True)
    num_other_exemplars = len(other_exemplars)
    #
    is_non_exemplar = ibs2.get_annot_exemplar_flags(qaid)
    can_add_more = num_other_exemplars < max_exemplars

    if is_non_exemplar and can_add_more:
        ENABLE_DISTINCTIVENESS_METHOD = True
        if ENABLE_DISTINCTIVENESS_METHOD:
            exemplar_distinctivness_thresh = ibs2.cfg.other_cfg.exemplar_distinctivness_thresh
            # Logic to choose query based on exemplar score distance
            qres = ibs2.query_chips([qaid], other_exemplars, cfgdict=dict(codename='vsone_norm_csum'))[0]
            aid_arr, score_arr = qres.get_aids_and_scores()
            is_distinctive = np.all(aid_arr < exemplar_distinctivness_thresh)
        else:
            is_distinctive = True
    else:
        is_distinctive = True

    do_exemplar_add = (can_add_more and is_distinctive and not is_non_exemplar)
    if do_exemplar_add:
        autoexmplr_msg = ('marking as qaid=%r exemplar' % (qaid,))
        autoexmplr_func = functools.partial(ibs2.set_annot_exemplar_flags, [qaid], [1])
    else:
        autoexmplr_msg = 'annotation is not marked as exemplar'
        autoexmplr_func = lambda: None

    return autoexmplr_msg, autoexmplr_func


def get_suggested_decision(ibs2, qaid, qres, threshold, metatup=None):
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
        python -m ibeis.model.hots.automated_matcher --test-get_suggested_decision
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1
    """
    if qres is None:
        nscoretup = list(map(np.array, ([], [], [], [])))
    else:
        nscoretup = qres.get_nscoretup(ibs2)
    # Get System Responce
    (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscoretup
    if len(sorted_nids) == 0:
        autoname_msg = '\n'.join((
            'Unable to find any matches',
            'suggesting new name',))
        nid = None
        score = None
        rank = None
    else:
        candidate_indexes = np.where(sorted_nscore > threshold)[0]
        if len(candidate_indexes) == 0:
            autoname_msg = '\n'.join((
                'No candidates above threshold',
                'suggesting new name',))
            nid = None
            score = None
            rank = None
        #elif len(candidate_indexes) == 1:
        #    nid = sorted_nids[candidate_indexes[0]]
        #    score = sorted_nscore[candidate_indexes[0]]
        #    autoname_msg = '\n'.join((
        #        'Single candidates above threshold',
        #        'suggesting score=%r, nid=%r' % (score, nid),
        #    ))
        else:
            nids = sorted_nids[candidate_indexes]
            scores = sorted_nscore[candidate_indexes]
            sortx = scores.argsort()[::-1]
            rank = sortx[0]
            score = scores[rank]
            nid = nids[rank]
            autoname_msg = '\n'.join((
                ('Multiple candidates above threshold'
                    if len(candidate_indexes) > 1 else
                    'Single candidate above threshold'),
                #'with scores=%r, nids=%r' % (scores, nids),
                'suggesting score=%r, nid=%r' % (score, nid),
            ))
    # If we have metainformation use the oracle to make a decision
    if metatup is not None:
        name2 = ah.get_oracle_decision(metatup, qaid, sorted_nids, sorted_aids)
        system_msg = 'The overrided system responce was:\n%s\n' % (ut.indent(autoname_msg),)
        if name2 is not None:
            nid = ibs2.get_name_rowids_from_text(name2)
            sorted_rawscore = [qres.get_aid_scores(aids, rawscore=True) for aids in sorted_aids]
            rank = np.where(sorted_nids == nid)[0]
            if rank is None:
                score = None
                rawscore = None
            else:
                score = sorted_nscore[rank]
                rawscore = sorted_rawscore[rank][0]
            #ut.embed()
            autoname_msg = system_msg + 'The oracle suggests score=%r nid=%r at rank=%r' % (score, nid, rank)
        else:
            autoname_msg = system_msg + 'The oracle suggests a new name'
            score = None
            nid = None
        #print(autoname_msg)
        #ut.embed()
    # Get new support data for score normalization
    if rank is not None:
        if rank > 0:
            rawscore_false = sorted_rawscore[0][0]
        else:
            if len(sorted_rawscore) > rank + 1:
                rawscore_false = sorted_rawscore[rank + 1][0]
            else:
                rawscore_false = None
    # Build decision function
    if nid is not None:
        autoname_func = functools.partial(autodecide_match, ibs2, qaid, nid)
    else:
        autoname_func = functools.partial(autodecide_newname, ibs2, qaid)
    return autoname_msg, autoname_func


def interactive_prompt(msg, decisiontype):
    prompt_fmtstr = ut.codeblock(
        '''
        Accept system {decisiontype} decision?
        ==========
        {msg}
        ==========
        Enter {no_phrase} to reject
        Enter {embed_phrase} to embed into ipython
        Any other inputs accept system decision
        (input is case insensitive)
        '''
    )
    ans_list_embed = ['cmd', 'ipy', 'embed']
    ans_list_no = ['no', 'n']
    #ans_list_yes = ['yes', 'y']
    prompt_str = prompt_fmtstr.format(
        no_phrase=ut.cond_phrase(ans_list_no),
        embed_phrase=ut.cond_phrase(ans_list_embed),
        msg=msg,
        decisiontype=decisiontype,
    )
    prompt_block = ut.msgblock('USER_INPUT', prompt_str)
    ans = input(prompt_block).lower()
    if ans in ans_list_embed:
        ut.embed()
        #print(ibs2.get_dbinfo_str())
        #qreq_ = ut.search_stack_for_localvar('qreq_')
        #qreq_.normalizer
    elif ans in ans_list_no:
        return False
    else:
        return True


def interactive_decision(ibs2, qres, autoname_msg, autoname_func):
    r"""
    Prompts the user for input

    Args:
        ibs2 (IBEISController):
        qres (QueryResult):  object of feature correspondences and scores
        autoname_func (function):
    """
    if qres is None:
        print('WARNING: qres is None')
    mplshowtop = True and qres is not None
    qtinspect = False and qres is not None
    if mplshowtop:
        fnum = 1
        fig = qres.ishow_top(ibs2, name_scoring=True, fnum=fnum)
        fig.show()
    if qtinspect:
        qres_wgt = qres.qt_inspect_gui(ibs2, name_scoring=True)
    # Ask the user if they like the
    if interactive_prompt(autoname_msg, 'name'):
        autoexmplr_msg, autoexmplr_func = autoname_func()
    else:
        autoexmplr_msg = 'ERROR: Need to build method for user name decision'
        autoexmplr_func = lambda: None

    if interactive_prompt(autoexmplr_msg, 'exemplar'):
        autoexmplr_func()
    else:
        autoexmplr_msg = 'Need to build method for user exemplar decision'
    # Prompt the user (this could be swaped out with a qt or web interface)
    if qtinspect:
        qres_wgt.close()


@ut.indent_func
def make_decisions(ibs2, qaid2_qres, qreq_, threshold, interactive=False,
                   metatup=None):
    r"""
    Either makes automatic decision or asks user for feedback.

    Args:
        ibs2        (IBEISController):
        qaid        (int):  query annotation id
        qres        (QueryResult):  object of feature correspondences and scores
        threshold   (float): threshold for automatic decision
        interactive (bool):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:0
        python -m ibeis.model.hots.automated_matcher --test-incremental_test:1

    """
    if qreq_ is not None:
        qreq_.normalizer.visualize(update=False)

    for qaid, qres in six.iteritems(qaid2_qres):
        #inspectstr = qres.get_inspect_str(ibs=ibs2, name_scoring=True)
        #print(ut.msgblock('VSONE-VERIFIED-RESULT', inspectstr))
        #print(inspectstr)
        autoname_msg, autoname_func = get_suggested_decision(
            ibs2, qaid, qres, threshold, metatup)
        if interactive:
            interactive_decision(ibs2, qres, autoname_msg, autoname_func)
        else:
            print(autoname_msg)
            autoexmplr_msg, autoexmplr_func = autoname_func()
            print(autoexmplr_msg)
            autoexmplr_func()


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
