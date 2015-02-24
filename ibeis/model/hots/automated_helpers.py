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
import six
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


@profile
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
    ibs2.delete_empty_nids()


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


def make_incremental_test_database(ibs_gt, aid_list1, reset):
    """
    Makes test database. adds image and annotations but does not transfer names.
    if reset is true the new database is gaurenteed to be built from a fresh
    start.

    Args:
        ibs_gt    (IBEISController):
        aid_list1 (list):
        reset     (bool): if True the test database is completely rebuilt

    Returns:
        IBEISController: ibs2
    """
    print('make_incremental_test_database. reset=%r' % (reset,))
    aids1_hashid = ut.hashstr_arr(aid_list1)
    prefix = '_INCTEST_' + aids1_hashid + '_'
    dbname2 = prefix + ibs_gt.get_dbname()
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


@profile
def setup_incremental_test(ibs_gt, clear_names=True):
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
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs_gt)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_helpers import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs_gt = ibeis.opendb('GZ_ALL')
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs_gt)
    """
    print('\n\n---- SETUP INCREMENTAL TEST ---\n\n')
    # Take a known dataase
    # Create an empty database to test in

    ONLY_GT = True
    if ONLY_GT:
        # use only annotations that will have matches in test
        aid_list1_ = ibs_gt.get_aids_with_groundtruth()
    else:
        # use every annotation in test
        aid_list1_ = ibs_gt.get_valid_aids()

    if ut.get_argflag('--gzdev'):
        # Use a custom selection of gzall
        from ibeis.model.hots import devcases
        assert ibs_gt.get_dbname() == 'GZ_ALL', 'not gzall'
        vuuid_list, ignore_vuuids = devcases.get_gzall_small_test()
        # TODO; include all names of these annots too
        aid_list = ibs_gt.get_annot_aids_from_visual_uuid(vuuid_list)
        ignore_aid_list = ibs_gt.get_annot_aids_from_visual_uuid(ignore_vuuids)
        ignore_nid_list = ibs_gt.get_annot_nids(ignore_aid_list)
        ut.assert_all_not_None(aid_list)
        other_aids = ut.flatten(ibs_gt.get_annot_groundtruth(aid_list))
        aid_list.extend(other_aids)
        aid_list = sorted(set(aid_list))
        nid_list = ibs_gt.get_annot_nids(aid_list)
        isinvalid_list = [nid in ignore_nid_list for nid in nid_list]
        print('Filtering %r annots specified to ignore' % (sum(isinvalid_list),))
        aid_list = ut.filterfalse_items(aid_list, isinvalid_list)
        #ut.embed()
        aid_list1_ = aid_list
        #ut.embed()

    # Add aids in a random order
    VALID_ORDERS = ['shuffle', 'stagger', 'same']
    #AID_ORDER = 'shuffle'
    AID_ORDER = 'stagger'
    assert VALID_ORDERS.index(AID_ORDER) > -1

    if AID_ORDER == 'shuffle':
        aid_list1 = ut.deterministic_shuffle(aid_list1_[:])
    elif AID_ORDER == 'stagger':
        from six.moves import zip_longest, filter
        aid_groups, unique_nid_list = ibs_gt.group_annots_by_name(aid_list1_)
        def stagger_group(list_):
            return ut.filter_Nones(ut.iflatten(zip_longest(*list_)))
        aid_multiton_group = list(filter(lambda aids: len(aids) > 1, aid_groups))
        aid_list1 = stagger_group(aid_multiton_group)
        #aid_list1 = ibs_gt.get_annot_rowid_sample(per_name=10, aid_list=aid_list1_, stagger_names=True)
        pass
    elif AID_ORDER == 'same':
        aid_list1 = aid_list1_

    # If reset is true the test database is started completely from scratch
    reset = ut.get_argflag('--reset')

    aid1_to_aid2 = {}  # annotation mapping

    ibs2 = make_incremental_test_database(ibs_gt, aid_list1, reset)

    # Preadd all annotatinos to the test database
    aids_chunk1 = aid_list1
    aid_list2 = add_annot_chunk(ibs_gt, ibs2, aids_chunk1, aid1_to_aid2)

    #ut.embed()
    # Assert annotation visual uuids are in agreement
    if ut.DEBUG2:
        annot_testdb_consistency_checks(ibs_gt, ibs2, aid_list1, aid_list2)

    # Remove names and exemplar information from test database
    if clear_names:
        ensure_testdb_clean_data(ibs_gt, ibs2, aid_list1, aid_list2)

    # Preprocess features before testing
    ibs2.ensure_annotation_data(aid_list2, featweights=True)

    return ibs2, aid_list1, aid1_to_aid2


def check_results(ibs_gt, ibs2, aid1_to_aid2, aids_list1_, incinfo):
    """
    reports how well the incremental query ran when the oracle was calling the
    shots.
    """
    print('--------- CHECKING RESULTS ------------')
    testcases = incinfo.get('testcases')
    if testcases is not None:
        count_dict = ut.count_dict_vals(testcases)
        print('+--')
        #print(ut.dict_str(testcases))
        print('---')
        print(ut.dict_str(count_dict))
        print('L__')
    # TODO: dont include initially added aids in the result reporting
    aid_list1 = aids_list1_  # ibs_gt.get_valid_aids()
    #aid_list1 = ibs_gt.get_aids_with_groundtruth()
    aid_list2 = ibs2.get_valid_aids()

    nid_list1 = ibs_gt.get_annot_nids(aid_list1)
    nid_list2 = ibs2.get_annot_nids(aid_list2)

    # Group annotations from test and gt database by their respective names
    grouped_dict1 = ut.group_items(aid_list1, nid_list1)
    grouped_dict2 = ut.group_items(aid_list2, nid_list2)
    grouped_aids1 = list(six.itervalues(grouped_dict1))
    grouped_aids2 = list(map(tuple, six.itervalues(grouped_dict2)))
    #group_nids1 = list(six.iterkeys(grouped_dict1))
    #group_nids2 = list(six.iterkeys(grouped_dict2))

    # Transform annotation ids from database1 space to database2 space
    grouped_aids1_t = [tuple(ut.dict_take_list(aid1_to_aid2, aids1)) for aids1 in grouped_aids1]

    set_grouped_aids1_t = set(grouped_aids1_t)
    set_grouped_aids2   = set(grouped_aids2)

    # Find names we got right. (correct groupings of annotations)
    # these are the annotation groups that are intersecting between
    # the test database and groundtruth database
    perfect_groups = set_grouped_aids2.intersection(set_grouped_aids1_t)
    # Find names we got wrong. (incorrect groupings of annotations)
    # The test database sets that were not perfect
    nonperfect_groups = set_grouped_aids2.difference(perfect_groups)
    # What we should have got
    # The ground truth database sets that were not fully identified
    missed_groups = set_grouped_aids1_t.difference(perfect_groups)

    # Mark non perfect groups by their error type
    false_negative_groups = []  # failed to link enough
    false_positive_groups = []  # linked too much
    for nonperfect_group in nonperfect_groups:
        if ut.is_subset_of_any(nonperfect_group, missed_groups):
            false_negative_groups.append(nonperfect_group)
        else:
            false_positive_groups.append(nonperfect_group)

    # Get some more info on the nonperfect groups
    # find which groups should have been linked
    aid2_to_aid1 = ut.invert_dict(aid1_to_aid2)
    false_negative_groups_t = [tuple(ut.dict_take_list(aid2_to_aid1, aids2)) for aids2 in false_negative_groups]
    false_negative_group_nids_t = ibs_gt.unflat_map(ibs_gt.get_annot_nids, false_negative_groups_t)
    assert all(map(ut.list_allsame, false_negative_group_nids_t)), 'inconsistent nids'
    false_negative_group_nid_t = ut.get_list_column(false_negative_group_nids_t, 0)
    # These are the links that should have been made
    missed_links = ut.group_items(false_negative_groups, false_negative_group_nid_t)

    print(ut.dict_str(missed_links))

    print('# Name with failed links (FN) = %r' % len(false_negative_groups))
    print('... should have reduced to %d names.' % (len(missed_links)))
    print('# Name with wrong links (FP)  = %r' % len(false_positive_groups))
    print('# Name correct names (TP)     = %r' % len(perfect_groups))
    #ut.embed()


@profile
def add_annot_chunk(ibs_gt, ibs2, aids_chunk1, aid1_to_aid2):
    """
    adds annotations to the tempoarary database and prevents duplicate
    additions.

    aids_chunk1 = aid_list1

    Args:
        ibs_gt       (IBEISController):
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
    ut.assert_all_not_None(aids_chunk1, 'aids_chunk1')
    ut.assert_all_not_None(guuids_chunk1, 'guuids_chunk1')
    try:
        ut.assert_all_not_None(gids_chunk2, 'gids_chunk2')
    except Exception as ex:
        #index = ut.get_first_None_position(gids_chunk2)
        #set(ibs2.get_valid_gids()).difference(set(gids_chunk2))
        ut.printex(ex, keys=['gids_chunk2'])
        #ut.embed()
        #raise
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
