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
    * Put test query mode into the main application and work on the interface for it.
    * add matches to multiple animals (merge)
    * update normalizer (have setup the datastructure to allow for it need to integrate it seemlessly)
    * score normalization update. on add the new support data, reapply bayes
     rule, and save to the current cache for a given algorithm configuration.
    * spawn background process to reindex chunks of data


TODO:
    * Improve vsone scoring.
    * test case where there is a 360 view that is linkable from the tests case
    * ~~Remember name_confidence of decisions for manual review~~ Defer

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
import ibeis
import utool as ut
from six.moves import input
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inchelp]')


def assert_testdb_annot_consistency(ibs_gt, ibs2, aid_list1, aid_list2):
    """
    just tests uuids

    if anything goes wrong this should fix it:
        from ibeis import ibsfuncs
        aid_list1 = ibs_gt.get_valid_aids()
        ibs_gt.update_annot_visual_uuids(aid_list1)
        ibs2.update_annot_visual_uuids(aid_list2)
        ibsfuncs.fix_remove_visual_dupliate_annotations(ibs_gt)
    """
    assert len(aid_list2) == len(aid_list1)
    visualtup1 = ibs_gt.get_annot_visual_uuid_info(aid_list1)
    visualtup2 = ibs2.get_annot_visual_uuid_info(aid_list2)

    _visual_uuid_list1 = [ut.augment_uuid(*tup) for tup in zip(*visualtup1)]
    _visual_uuid_list2 = [ut.augment_uuid(*tup) for tup in zip(*visualtup2)]

    assert ut.hashstr(visualtup1) == ut.hashstr(visualtup2)
    ut.assert_lists_eq(visualtup1[0], visualtup2[0])
    ut.assert_lists_eq(visualtup1[1], visualtup2[1])
    ut.assert_lists_eq(visualtup1[2], visualtup2[2])
    #semantic_uuid_list1 = ibs_gt.get_annot_semantic_uuids(aid_list1)
    #semantic_uuid_list2 = ibs2.get_annot_semantic_uuids(aid_list2)

    visual_uuid_list1 = ibs_gt.get_annot_visual_uuids(aid_list1)
    visual_uuid_list2 = ibs2.get_annot_visual_uuids(aid_list2)

    # make sure visual uuids are still determenistic
    ut.assert_lists_eq(visual_uuid_list1, visual_uuid_list2)
    ut.assert_lists_eq(_visual_uuid_list1, visual_uuid_list1)
    ut.assert_lists_eq(_visual_uuid_list2, visual_uuid_list2)

    ibs1_dup_annots = ut.debug_duplicate_items(visual_uuid_list1)
    ibs2_dup_annots = ut.debug_duplicate_items(visual_uuid_list2)

    # if these fail try ibsfuncs.fix_remove_visual_dupliate_annotations
    assert len(ibs1_dup_annots) == 0
    assert len(ibs2_dup_annots) == 0


def ensure_testdb_clean_data(ibs_gt, ibs2, aid_list1, aid_list2):
    """
    removes previously set names and exemplars
    """
    # Make sure that there are not any names in this database
    nid_list2 = ibs2.get_annot_name_rowids(aid_list2, distinguish_unknowns=False)
    print('Removing names from the incremental test database')
    if not ut.list_all_eq_to(nid_list2, 0):
        ibs2.set_annot_name_rowids(aid_list2, [ibs2.UNKNOWN_NAME_ROWID] * len(aid_list2))
    ibs2.delete_names(ibs2._get_all_known_name_rowids())

    #exemplarflag_list2 = ibs2.get_annot_exemplar_flags(aid_list2)
    #if not ut.list_all_eq_to(exemplarflag_list2, 0):
    print('Unsetting all exemplars from database')
    ibs2.set_annot_exemplar_flags(aid_list2, [False] * len(aid_list2))

    # this test is for plains
    #assert  ut.list_all_eq_to(ibs2.get_annot_species_texts(aid_list2), 'zebra_plains')
    ibs2.delete_invalid_nids()


def annot_testdb_consistency_checks(ibs_gt, ibs2, aid_list1, aid_list2):
    try:
        assert_testdb_annot_consistency(ibs_gt, ibs2, aid_list1, aid_list2)
    except Exception as ex:
        # update and try again on failure
        ut.printex(ex, ('warning: consistency check failed.'
                        'updating and trying once more'), iswarning=True)
        ibs_gt.update_annot_visual_uuids(aid_list1)
        ibs2.update_annot_visual_uuids(aid_list2)
        assert_testdb_annot_consistency(ibs_gt, ibs2, aid_list1, aid_list2)


def interactive_commandline_prompt(msg, decisiontype):
    prompt_fmtstr = ut.codeblock(
        '''
        Accept system {decisiontype} decision?
        ==========

        {msg}

        ==========
        * press ENTER to ACCEPT
        * enter {no_phrase} to REJECT
        * enter {embed_phrase} to embed into ipython
        * any other inputs ACCEPT system decision
        * (input is case insensitive)
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


def setup_incremental_test(ibs_gt, num_initial=0, clear_names=True):
    r"""
    CommandLine:
        python -m ibeis.model.hots.automated_helpers --test-setup_incremental_test:0

        python dev.py -t custom --cfg codename:vsone_unnorm --db PZ_MTEST --allgt --vf --va
        python dev.py -t custom --cfg codename:vsone_unnorm --db PZ_MTEST --allgt --vf --va --index 0 4 8 --verbose

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_helpers import *  # NOQA
        >>> import ibeis # NOQA
        >>> ibs_gt = ibeis.opendb('PZ_MTEST')
        >>> num_initial = 0
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs_gt, num_initial)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_helpers import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs_gt = ibeis.opendb('GZ_ALL')
        >>> num_initial = 100
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs_gt, num_initial)
    """
    # Take a known dataase
    # Create an empty database to test in
    #aid_list1 = ibs_gt.get_aids_with_groundtruth()
    aid_list1 = ibs_gt.get_valid_aids()

    # If reset is true the test database is started completely from scratch
    reset = ut.get_argflag('--reset')
    #reset = True

    aid1_to_aid2 = {}  # annotation mapping

    def make_incremental_test_database(ibs_gt, aid_list1, reset):
        """
        Makes test database. adds image and annotations but does not transfer names.
        if reset is true the new database is gaurenteed to be built from a fresh
        start.

        Args:
            ibs_gt      (IBEISController):
            aid_list1 (list):
            reset     (bool):

        Returns:
            IBEISController: ibs2
        """
        print('make_incremental_test_database. reset=%r' % (reset,))
        dbname2 = '_INCREMENTALTEST_' + ibs_gt.get_dbname()
        ibs2 = ibeis.opendb(dbname2, allow_newdir=True, delete_ibsdir=reset, use_cache=False)

        # reset if flag specified or no data in ibs2
        if reset or len(ibs2.get_valid_gids()) == 0:
            assert len(ibs2.get_valid_aids())  == 0
            assert len(ibs2.get_valid_gids())  == 0
            assert len(ibs2.get_valid_nids())  == 0

            # Get annotations and their images from database 1
            gid_list1 = ibs_gt.get_annot_gids(aid_list1)
            gpath_list1 = ibs_gt.get_image_paths(gid_list1)

            # Add all images from database 1 to database 2
            gid_list2 = ibs2.add_images(gpath_list1, auto_localize=False)

            # Image UUIDS should be consistent between databases
            image_uuid_list1 = ibs_gt.get_image_uuids(gid_list1)
            image_uuid_list2 = ibs2.get_image_uuids(gid_list2)
            assert image_uuid_list1 == image_uuid_list2
            ut.assert_lists_eq(image_uuid_list1, image_uuid_list2)
        return ibs2

    ibs2 = make_incremental_test_database(ibs_gt, aid_list1, reset)

    # Add the annotations with names
    aids_chunk1 = aid_list1
    aid_list2 = add_annot_chunk(ibs_gt, ibs2, aids_chunk1, aid1_to_aid2)

    #ut.embed()
    # Assert annotation visual uuids are in agreement
    annot_testdb_consistency_checks(ibs_gt, ibs2, aid_list1, aid_list2)

    # Remove name exemplars
    if clear_names:
        ensure_testdb_clean_data(ibs_gt, ibs2, aid_list1, aid_list2)

    # Preprocess features and such
    ibs2.ensure_annotation_data(aid_list2, featweights=True)

    print('Transfer %d initial test annotations' % (num_initial,))
    if num_initial > 0:
        # Transfer some initial data
        aid_sublist1 = aid_list1[0:num_initial]
        aid_sublist2 = aid_list2[0:num_initial]
        name_list = ibs_gt.get_annot_names(aid_sublist1)
        ibs2.set_annot_names(aid_sublist2, name_list)
        ibs2.set_annot_exemplar_flags(aid_sublist2, [True] * len(aid_sublist2))
        aid_list1 = aid_list1[num_initial:]
    print(ibs2.get_dbinfo_str())
    return ibs2, aid_list1, aid1_to_aid2


def check_results(ibs_gt, ibs2, aid1_to_aid2):
    """
    reports how well the incremental query ran when the oracle was calling the
    shots.
    """
    import six
    #aid_list1 = ibs_gt.get_valid_aids()
    aid_list1 = ibs_gt.get_aids_with_groundtruth()
    aid_list2 = ibs2.get_valid_aids()

    nid_list1 = ibs_gt.get_annot_nids(aid_list1)
    nid_list2 = ibs2.get_annot_nids(aid_list2)

    grouped_aids1 = list(six.itervalues(ut.group_items(aid_list1, nid_list1)))
    grouped_aids2 = list(map(tuple, six.itervalues(ut.group_items(aid_list2, nid_list2))))

    grouped_aids12 = [tuple(ut.dict_take_list(aid1_to_aid2, aids1)) for aids1 in grouped_aids1]

    set_grouped_aids2 = set(grouped_aids2)
    set_grouped_aids12 = set(grouped_aids12)

    # What we got right
    perfect_groups = set_grouped_aids2.intersection(set_grouped_aids12)

    # What we got wrong
    missed_groups  = set_grouped_aids2.difference(perfect_groups)
    # What we should have got
    missed_truth_groups = set_grouped_aids12.difference(perfect_groups)

    truth_group_sets = list(map(set, missed_truth_groups))
    missed_groups_set  = list(map(set, missed_groups))

    failed_links = []
    wrong_links = []
    for missed_set in missed_groups_set:
        if any([missed_set.issubset(truth_set) for truth_set in truth_group_sets]):
            failed_links.append(missed_set)
        else:
            wrong_links.append(missed_set)

    print('# Name with failed links = %r' % len(failed_links))
    print('# Name with wrong links = %r' % len(wrong_links))
    print('# Name correct names = %r' % len(perfect_groups))


def add_annot_chunk(ibs_gt, ibs2, aids_chunk1, aid1_to_aid2):
    """
    adds annotations to the tempoarary database and prevents duplicate
    additions.

    aids_chunk1 = aid_list1

    Args:
        ibs_gt         (IBEISController):
        ibs2         (IBEISController):
        aids_chunk1  (list):
        aid1_to_aid2 (dict):

    Returns:
        list: aids_chunk2
    """
    # Visual info
    guuids_chunk1 = ibs_gt.get_annot_image_uuids(aids_chunk1)
    verts_chunk1  = ibs_gt.get_annot_verts(aids_chunk1)
    thetas_chunk1 = ibs_gt.get_annot_thetas(aids_chunk1)
    # Non-name semantic info
    species_chunk1 = ibs_gt.get_annot_species_texts(aids_chunk1)
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
    # Register the mapping from ibs_gt to ibs2
    register_annot_mapping(aids_chunk1, aids_chunk2, aid1_to_aid2)
    print('Added: aids_chunk2=%s' % (ut.truncate_str(repr(aids_chunk2), maxlen=60),))
    return aids_chunk2


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.automated_helpers
        python -m ibeis.model.hots.automated_helpers --allexamples
        python -m ibeis.model.hots.automated_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
