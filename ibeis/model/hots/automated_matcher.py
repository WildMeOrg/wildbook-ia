"""

Have:
    * semantic and visual uuids
    * Test that accepts unknown annotations one at a time and
      for each runs query, makes decision about name, and executes decision.
    * As a placeholder for exemplar decisions  an exemplar is added if
      number of exemplars per name is less than threshold.
    * vs-one reranking query mode

    * test harness but start with larger test set

TODO:
    * vs-one score normalizer ~~/ score normalizer for different values of K * / different params~~
      vs-many score normalization doesnt actually matter. We just need the ranking.
    * ~~Remember confidence of decisions for manual review~~
      Defer
    * need to add in the multi-indexer code into the pipeline. Need to
      decide which subindexers to load given a set of daids
    * need to use set query as an exemplar if its vs-one reranking scores
      are below a threshold

New TODO:

    * update normalizer (have setup the datastructure to allow for it need to integrate it seemlessly)
    * turn on multi-indexing. (should just work..., probably bugs though. Just need to throw the switch)
    * Improve vsone scoring.
    * Put this query mode into the main application and work on the interface for it.
"""
from __future__ import absolute_import, division, print_function
import ibeis
import six
import utool as ut
import numpy as np
from six.moves import input, filter  # NOQA
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')


DEFAULT_INTERACTIVE = True


def test_vsone_verified(ibs):
    """
    hack in vsone-reranking

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> #reload_all()
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> test_vsone_verified(ibs)
    """
    import plottool as pt
    #qaids = ibs.get_easy_annot_rowids()
    nids = ibs.get_valid_nids(filter_empty=True)
    grouped_aids_ = ibs.get_name_aids(nids)
    grouped_aids = list(filter(lambda x: len(x) > 1, grouped_aids_))
    items_list = grouped_aids

    sample_aids = ut.flatten(ut.sample_lists(items_list, num=2, seed=0))
    qaid2_qres = query_vsone_verified(ibs, sample_aids, sample_aids, cfgdict=None, return_request=False)
    for qres in ut.InteractiveIter(list(six.itervalues(qaid2_qres))):
        pt.close_all_figures()
        fig = qres.ishow_top(ibs)
        fig.show()
    #return qaid2_qres


def setup_incremental_test(ibs1, num_initial=0):
    r"""
    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-setup_incremental_test

        python dev.py -t custom --cfg codename:vsone_unnorm --db PZ_MTEST --allgt --vf --va
        python dev.py -t custom --cfg codename:vsone_unnorm --db PZ_MTEST --allgt --vf --va --index 0 4 8 --verbose

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> ibs1 = ibeis.opendb('PZ_MTEST')
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs1)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> ibs1 = ibeis.opendb('GZ_ALL')
        >>> ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs1, num_initial=100)
    """
    # Take a known dataase
    # Create an empty database to test in
    aid_list1 = ibs1.get_aids_with_groundtruth()
    reset = False
    #reset = True
    # Helper functions

    aid1_to_aid2 = {}

    ibs2 = make_incremental_test_database(ibs1, aid_list1, reset)

    # Add the annotations without names

    aids_chunk1 = aid_list1
    aid_list2 = add_annot_chunk(ibs1, ibs2, aids_chunk1, aid1_to_aid2)

    # Assert visual uuids
    try:
        assert_annot_consistency(ibs1, ibs2, aid_list1, aid_list2)
    except Exception as ex:
        ut.printex(ex, ('warning: consistency check failed.'
                        'updating and trying once more'), iswarning=True)
        ibs1.update_annot_visual_uuids(aid_list1)
        ibs2.update_annot_visual_uuids(aid_list2)
        assert_annot_consistency(ibs1, ibs2, aid_list1, aid_list2)

    # Remove name exemplars
    ensure_clean_data(ibs1, ibs2, aid_list1, aid_list2)

    # Preprocess features and such
    ibs2.ensure_annotation_data(aid_list2, featweights=True)

    if num_initial > 0:
        # Transfer some initial data
        aid_sublist1 = aid_list1[0:num_initial]
        aid_sublist2 = aid_list2[0:num_initial]
        name_list = ibs1.get_annot_names(aid_sublist1)
        ibs2.set_annot_names(aid_sublist2, name_list)
        ibs2.set_annot_exemplar_flags(aid_sublist2, [True] * len(aid_sublist2))
    return ibs2, aid_list1, aid1_to_aid2


def query_vsone_verified(ibs, qaids, daids, cfgdict=None, return_request=False):
    """
    hack in vsone-reranking
    """
    from ibeis import ibsfuncs  # NOQA
    qaid2_qres_vsmany, qreq_ = ibs._query_chips4(qaids, daids, cfgdict=cfgdict, return_request=True)
    #qaid2_qres, qreq_ = ibs._query_chips4(sample_aids, [], return_request=True)

    isnsum = qreq_.qparams.score_method == 'nsum'
    assert isnsum
    assert qreq_.qparams.pipeline_root != 'vsone'
    #qreq_.qparams.prescore_method

    # build vs one list
    vsone_query_pairs = []
    nShortlistVsone = 5
    for qaid, qres in six.iteritems(qaid2_qres_vsmany):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs=ibs)
        nShortlistVsone_ = min(len(sorted_nids), nShortlistVsone)
        top_nids = sorted_nids[0:nShortlistVsone_]
        # get top annotations beloning to the database query
        # TODO: allow annots not in daids to be included
        top_aids = ut.intersect_ordered(ut.flatten(ibs.get_name_aids(top_nids)), qres.daids)
        vsone_query_pairs.append((qaid, top_aids))

    # vs-one reranking
    qaid2_qres_vsone = {}
    for qaid, top_aids in vsone_query_pairs:
        cfgdict = dict(codename='vsone_norm')
        qaid2_qres_vsone_, qreq_ = ibs._query_chips4([qaid], top_aids, cfgdict=cfgdict, return_request=True)
        qres = qaid2_qres_vsone_[qaid]
        qaid2_qres_vsone[qaid] = qres

    qaid2_qres = qaid2_qres_vsone
    if return_request:
        return qaid2_qres, qreq_
    return qaid2_qres
    #ibsfuncs.unflat_map(ibs.get_annot_is_hard, grouped_aids)


def choose_vsmany_K(ibs, qaids, daids):
    K = ibs.cfg.query_cfg.nn_cfg.K
    if len(daids) < 10:
        K = 1
    if len(ut.intersect_ordered(qaids, daids)) > 0:
        # if self is in query bump k
        K += 1
    return K
    pass


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

        python -m ibeis.model.hots.automated_matcher --test-incremental_test

    Example:
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs1 = ibeis.opendb('PZ_MTEST')
        >>> incremental_test(ibs1)
    """

    def execute_teststep(ibs1, ibs2, aids_chunk1, aid1_to_aid2):
        """ Add an unseen annotation and run a query """
        print('\n\n==== EXECUTING TESTSTEP ====')
        # ensure new annot is added (most likely it will have been preadded)
        aids_chunk2 = add_annot_chunk(ibs1, ibs2, aids_chunk1, aid1_to_aid2)

        threshold = 1.99
        exemplar_aids = ibs2.get_valid_aids(is_exemplar=True)

        interactive = DEFAULT_INTERACTIVE
        #interactive = False
        cfgdict = {
            'K': choose_vsmany_K(ibs2, aids_chunk2, exemplar_aids)
        }

        if len(exemplar_aids) > 0:
            #qaid2_qres, qreq_ = ibs2.query_exemplars(aids_chunk2, cfgdict=cfgdict, return_request=True)
            qaid2_qres, qreq_ = query_vsone_verified(ibs2, aids_chunk2,
                                                     exemplar_aids,
                                                     cfgdict=cfgdict,
                                                     return_request=True)
            for qaid, qres in six.iteritems(qaid2_qres):
                make_decision(ibs2, qaid, qres, threshold, interactive=interactive)
        else:
            print('No exemplars in database')
            for aid in aids_chunk2:
                autodecide_newname(ibs2, aid)

    ibs2, aid_list1, aid1_to_aid2 = setup_incremental_test(ibs1, num_initial=0)

    # TESTING
    chunksize = 1
    aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    #ut.embed()

    aids_chunk1 = six.next(aids_chunk1_iter)
    execute_teststep(ibs1, ibs2, aids_chunk1, aid1_to_aid2)

    for _ in range(2):
        aids_chunk1 = six.next(aids_chunk1_iter)
        execute_teststep(ibs1, ibs2, aids_chunk1, aid1_to_aid2)

    aids_chunk1 = six.next(aids_chunk1_iter)
    execute_teststep(ibs1, ibs2, aids_chunk1, aid1_to_aid2)

    # FULL INCREMENT
    #aids_chunk1_iter = ut.ichunks(aid_list1, 1)
    for aids_chunk1 in aids_chunk1_iter:
        execute_teststep(ibs1, ibs2, aids_chunk1, aid1_to_aid2)
    #    break
    #    #pass


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
    ut.assert_all_not_None(gids_chunk2, 'gids_chunk2')
    # Add this new unseen test case to the database
    aids_chunk2 = ibs2.add_annots(gids_chunk2,
                                  species_list=species_chunk1,
                                  vert_list=verts_chunk1,
                                  theta_list=thetas_chunk1,
                                  prevent_visual_duplicates=True)
    register_annot_mapping(aids_chunk1, aids_chunk2, aid1_to_aid2)
    print('Added: aids_chunk2=%s' % (ut.truncate_str(repr(aids_chunk2)),))
    return aids_chunk2


def make_incremental_test_database(ibs1, aid_list1, reset):
    """
    makes test db
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
    #assert  ut.list_all_eq_to(ibs2.get_annot_species(aid_list2), 'zebra_plains')


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


def make_decision(ibs2, qaid, qres, threshold, interactive=False):
    inspectstr = qres.get_inspect_str(ibs=ibs2, name_scoring=True)
    print(inspectstr)
    if interactive:
        #fig = qres.ishow_top(ibs2, name_scoring=True)
        #fig.show()

        qres_wgt = qres.qt_inspect_gui(ibs2, name_scoring=True)

        ans = input('waiting\n')
        if ans in ['cmd', 'ipy', 'embed']:
            ut.embed()
        qres_wgt.close()
    else:
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
                nid = nids[scores.argsort()[::-1][0]]
                autodecide_match(ibs2, qaid, nid)


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
