from __future__ import absolute_import, division, print_function
import ibeis
import six
import utool as ut
import numpy as np
from six.moves import input
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')


def register_annot_mapping(aids_chunk1, aids_chunk2, aid1_to_aid2):
    # Should be 1 to 1
    for aid1, aid2 in zip(aids_chunk1, aids_chunk2):
        if aid1 in aid1_to_aid2:
            assert aid1_to_aid2[aid1] == aid2
        else:
            aid1_to_aid2[aid1] = aid2


def add_annot_chunk(ibs1, ibs2, aids_chunk1, aid1_to_aid2):
    """
    aids_chunk1 = aid_list1
    """
    # Visual info
    guuids_chunk1  = ibs1.get_annot_image_uuids(aids_chunk1)
    verts_chunk1   = ibs1.get_annot_verts(aids_chunk1)
    thetas_chunk1  = ibs1.get_annot_thetas(aids_chunk1)
    # Non-name semantic info
    species_chunk1 = ibs1.get_annot_species(aids_chunk1)

    gids_chunk2 = ibs2.get_image_gids_from_uuid(guuids_chunk1)
    # Add this new unseen test case to the database
    aids_chunk2 = ibs2.add_annots(gids_chunk2,
                                  species_list=species_chunk1,
                                  vert_list=verts_chunk1,
                                  theta_list=thetas_chunk1,
                                  prevent_visual_duplicates=True)
    register_annot_mapping(aids_chunk1, aids_chunk2, aid1_to_aid2)
    print('Added: aids_chunk2=%r' % (aids_chunk2,))
    return aids_chunk2


def make_incremental_test_database(ibs1, aid_list1, reset):
    """ makes test db """
    print('make_incremental_test_database. reset=%r' % (reset,))
    dbname2 = '_INCREMENTALTEST_' + ibs1.get_dbname()
    ibs2 = ibeis.opendb(dbname2, allow_newdir=True, delete_ibsdir=reset, use_cache=False)

    if reset:
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


def assert_annot_consistency(ibs1, ibs2, aid_list1, aid_list2):
    """ just tests uuids

    if anything goes wrong this should fix it:
        ibs1.update_annot_visual_uuids(aid_list1)
        ibs2.update_annot_visual_uuids(aid_list2)
    """
    assert len(aid_list2) == len(aid_list1)
    visualtup1 = ibs1.get_annot_visual_uuid_info(aid_list1)
    visualtup2 = ibs2.get_annot_visual_uuid_info(aid_list2)

    [ut.augment_uuid(*tup) for tup in zip(*visualtup1)]
    [ut.augment_uuid(*tup) for tup in zip(*visualtup2)]

    assert ut.hashstr(visualtup1) == ut.hashstr(visualtup2)
    ut.assert_lists_eq(visualtup1[0], visualtup2[0])
    ut.assert_lists_eq(visualtup1[1], visualtup2[1])
    ut.assert_lists_eq(visualtup1[2], visualtup2[2])
    #semantic_uuid_list1 = ibs1.get_annot_semantic_uuids(aid_list1)
    #semantic_uuid_list2 = ibs2.get_annot_semantic_uuids(aid_list2)

    visual_uuid_list1 = ibs1.get_annot_visual_uuids(aid_list1)
    visual_uuid_list2 = ibs2.get_annot_visual_uuids(aid_list2)
    ut.assert_lists_eq(visual_uuid_list1, visual_uuid_list2)


def ensure_clean_data(ibs1, ibs2, aid_list1, aid_list2):
    """
    removes previously set names and exemplars
    """
    # Make sure that there are not any names in this database
    nid_list2 = ibs2.get_annot_name_rowids(aid_list2, distinguish_unknowns=False)
    if not ut.list_all_eq_to(nid_list2, 0):
        print('Removing names from database')
        ibs2.set_annot_name_rowids(aid_list2, [0] * len(aid_list2))

    exemplarflag_list2 = ibs2.get_annot_exemplar_flags(aid_list2)
    if not ut.list_all_eq_to(exemplarflag_list2, 0):
        print('Unsetting all exemplars from database')
        ibs2.set_annot_exemplar_flags(aid_list2, [0] * len(aid_list2))

    ibs2.delete_invalid_nids()

    # this test is for plains
    assert  ut.list_all_eq_to(ibs2.get_annot_species(aid_list2), 'zebra_plains')


def autodecide_newname(ibs2, aid):
    if ibs2.is_aid_unknown(aid):
        print('adding as new name')
        newname = ibs2.make_next_name()
        ibs2.set_annot_names([aid], [newname])
    else:
        print('already has name')
    if not ibs2.get_annot_exemplar_flags(aid):
        print('marking as exemplar')
        ibs2.set_annot_exemplar_flags([aid], [1])
    else:
        print('already is exemplar')


def autodecide_match(ibs2, aid, nid):
    print('setting nameid to nid=%r' % nid)
    ibs2.set_annot_name_rowids([aid], [nid])
    if not ibs2.get_annot_exemplar_flags(aid):
        print('marking as exemplar')
        ibs2.set_annot_exemplar_flags([aid], [1])
    else:
        print('already is exemplar')
        ibs2.get_name_exemplar_aids(nid)


def make_decision(ibs2, qaid, qres, threshold, interactive=True):
    if interactive:
        qres_wgt = qres.qt_inspect_gui(ibs2, name_scoring=True)  # NOQA
        #fig = qres.ishow_top(ibs2, name_scoring=True)
        #fig.show()
        inspectstr = qres.get_inspect_str(ibs=ibs2, name_scoring=True)
        print(inspectstr)
        ans = input('waiting\n')
        if ans in ['cmd', 'ipy', 'embed']:
            ut.embed()
        qres_wgt.close()
    nid_list, score_list = qres.get_sorted_nids_and_scores(ibs2)
    if len(nid_list) == 0:
        print('No matches made')
        autodecide_newname(ibs2, qaid)
    else:
        candidate_indexes = np.where(score_list > threshold)[0]
        if len(candidate_indexes) == 0:
            print('No candidates above threshold')
            autodecide_newname(ibs2, qaid)
        elif len(candidate_indexes) == 1:
            nid = nid_list[candidate_indexes[0]]
            score = score_list[candidate_indexes[0]]
            print('One candidate above threshold with score=%r' % (score,))
            autodecide_match(ibs2, qaid, nid)
        else:
            print('Multiple candidates above threshold')
            nids = nid_list[candidate_indexes]  # NOQA
            scores = score_list[candidate_indexes]
            print('One candidate above threshold with scores=%r' % (scores,))
            nid = nids[scores.argsort()[::-1]]
            autodecide_match(ibs2, qaid, nid)


def incremental_test(ibs1):
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

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-incremental_test

    Example:
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs1 = ibeis.opendb('PZ_MTEST')
        >>> incremental_test(ibs1)

    TODO:
        *
        * vs-one score normalizer / score normalizer for different values of K / different params
          vs-many score normalization doesnt actually matter. We just need the ranking.
        * Remember confidence of decisions for manual review
        * need to add in the multi-indexer code into the pipeline. Need to
          decide which subindexers to load given a set of daids
        * need a vs-one reranking pipeline node
        * need to use set query as an exemplar if its vs-one reranking scores
          are below a threshold

    Have:
        * semantic and visual uuids
        * test that accepts unknown annotations one at a time and
          for each runs query, makes decision about name, and executes decision
        * exemplar added if number of exemplars per name is less than threshold
    """
    # Take a known dataase
    # Create an empty database to test in
    aid_list1 = ibs1.get_aids_with_groundtruth()
    #reset = True
    reset = False
    # Helper functions

    aid1_to_aid2 = {}

    def execute_teststep(ibs1, ibs2, aids_chunk1):
        """ Add an unseen annotation and run a query """
        print('\n\n==== EXECUTING TESTSTEP ====')
        aids_chunk2 = add_annot_chunk(ibs1, ibs2, aids_chunk1, aid1_to_aid2)

        threshold = .96
        exemplar_aids = ibs2.get_valid_aids(is_exemplar=True)

        K = ibs2.cfg.query_cfg.nn_cfg.K
        if len(exemplar_aids) < 10:
            K = 1
        if len(ut.intersect_ordered(aids_chunk1, exemplar_aids)) > 0:
            # if self is in query bump k
            K += 1
        cfgdict = {
            'K': K
        }
        interactive = True

        if len(exemplar_aids) > 0:
            qaid2_qres = ibs2.query_exemplars(aids_chunk2, cfgdict=cfgdict)
            for qaid, qres in six.iteritems(qaid2_qres):
                make_decision(ibs2, qaid, qres, threshold, interactive=interactive)
        else:
            print('No exemplars in database')
            for aid in aids_chunk2:
                autodecide_newname(ibs2, aid)

    ibs2 = make_incremental_test_database(ibs1, aid_list1, reset)

    # Add the annotations without names
    aid_list2 = add_annot_chunk(ibs1, ibs2, aid_list1, aid1_to_aid2)

    # Assert visual uuids
    assert_annot_consistency(ibs1, ibs2, aid_list1, aid_list2)

    # Remove name exemplars
    ensure_clean_data(ibs1, ibs2, aid_list1, aid_list2)

    # Preprocess features and such
    ibs2.ensure_annotation_data(aid_list2, featweights=True)

    # TESTING
    chunksize = 1
    aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    #ut.embed()

    aids_chunk1 = six.next(aids_chunk1_iter)
    execute_teststep(ibs1, ibs2, aids_chunk1)

    for _ in range(2):
        aids_chunk1 = six.next(aids_chunk1_iter)
        execute_teststep(ibs1, ibs2, aids_chunk1)

    aids_chunk1 = six.next(aids_chunk1_iter)
    execute_teststep(ibs1, ibs2, aids_chunk1)

    # FULL INCREMENT
    #aids_chunk1_iter = ut.ichunks(aid_list1, 1)
    for aids_chunk1 in aids_chunk1_iter:
        execute_teststep(ibs1, ibs2, aids_chunk1)
    #    break
    #    #pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.automated_matcher
        python -m ibeis.model.hots.automated_matcher --allexamples
        python -m ibeis.model.hots.automated_matcher --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
