#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
from vtool import geometry
from six.moves import zip, range
from ibeis import constants
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_IBS_CONTROL]')


def TEST_IBS_CONTROL(ibs):
    ibs.delete_all_encounters()
    ibs.compute_encounters()

    """ get_image_eids / get_encounter_gids """
    eid_list = ibs.get_valid_eids()
    assert eid_list, 'eid_list is empty'
    gids_list = ibs.get_encounter_gids(eid_list)
    # print('[TEST] gids_list = %r' % gids_list)
    assert gids_list, 'gids_list is empty'
    for gids, eid in zip(gids_list, eid_list):
        eid_list2 = ibs.get_image_eids(gids)
        try:
            assert ([[eid]] * len(eid_list2)) == eid_list2
        except AssertionError as ex:
            utool.printex(ex, key_list=['eid_list2', 'eid'])
            raise

    """ set_annot_notes / get_annot_notes """
    aid_list = ibs.get_valid_aids()
    annotation_notes_list = len(aid_list) * ["test text"]
    assert aid_list, 'aid_list is empty'
    ibs.set_annot_notes(aid_list, annotation_notes_list)
    annotation_notes_list2 = ibs.get_annot_notes(aid_list)
    assert annotation_notes_list2, 'get_annot_notes returned an empty list'
    #print('[TEST] annotation_notes_list = %r' % annotation_notes_list)
    #print('[TEST] annotation_notes_list2 = %r' % annotation_notes_list2)
    assert annotation_notes_list == annotation_notes_list2, 'annotation notes lists do not match'

    """ set_name_notes / get_name_notes """
    nid_list = ibs.get_valid_nids()
    assert nid_list, 'nid_list is empty'
    nid_notes_list = len(nid_list) * ['nid notes test']
    ibs.set_name_notes(nid_list, nid_notes_list)
    nid_notes_list2 = ibs.get_name_notes(nid_list)
    print('[TEST] nid_notes_list = %r' % nid_notes_list)
    print('[TEST] nid_notes_list2 = %r' % nid_notes_list2)
    utool.assert_lists_eq(nid_notes_list, nid_notes_list2, 'nid notes lists do not match', verbose=True)
    assert nid_notes_list == nid_notes_list2, 'nid notes lists do not match'

    """ set_image_notes / get_image_notes """
    gid_list = ibs.get_valid_gids()
    assert gid_list, 'gid_list is empty'
    gid_notes_list = len(gid_list) * ['image note test']
    ibs.set_image_notes(gid_list, gid_notes_list)
    gid_notes_list2 = ibs.get_image_notes(gid_list)
    print('[TEST] gid_notes_list = %r' % gid_notes_list)
    print('[TEST] gid_notes_list2 = %r' % gid_notes_list2)
    assert gid_notes_list == gid_notes_list2, 'images notes lists do not match'

    """ set_annot_bboxes / get_annot_bboxes """
    aid_list = ibs.get_valid_aids()
    assert aid_list, 'aid_list is empty'
    bbox_list_orig = ibs.get_annot_bboxes(aid_list)
    bbox_list = [(1, 2, 3, 4)] * len(aid_list)
    ibs.set_annot_bboxes(aid_list, bbox_list)
    bbox_list2 = ibs.get_annot_bboxes(aid_list)
    print('[TEST] aid_list = %r' % (aid_list,))
    print('[TEST] bbox_list = %r' % (bbox_list,))
    print('[TEST] bbox_list2 = %r' % (bbox_list2,))
    assert bbox_list == bbox_list2, 'bbox lists do not match'
    # put bboxes back to original state
    # (otherwise other tests will fail on the second run of run_tests.sh)
    ibs.set_annot_bboxes(aid_list, bbox_list_orig)

    """ set_annot_verts / get_annot_verts """
    aid_list = ibs.get_valid_aids()
    assert aid_list, 'aid_list is empty'
    bbox_list_orig = ibs.get_annot_bboxes(aid_list)
    vert_list_orig = ibs.get_annot_verts(aid_list)
    vert_list = [((1, 2), (3, 4), (5, 6), (7, 8))] * len(aid_list)
    print('[TEST] vert_list = %r' % vert_list)
    assert len(aid_list) == len(vert_list), 'lengths do not match, malformed input'
    ibs.set_annot_verts(aid_list, vert_list)
    vert_list2 = ibs.get_annot_verts(aid_list)
    assert vert_list == vert_list2, 'vert lists do not match'

    """ set_annot_verts / get_annot_bboxes """
    bbox_list = ibs.get_annot_bboxes(aid_list)
    bbox_list2 = geometry.bboxes_from_vert_list(vert_list2)
    assert bbox_list == bbox_list2, 'bbox lists do not match'
    vert_list = [((10, 10), (120, 10), (120, 120), (10, 120))] * len(aid_list)
    ibs.set_annot_verts(aid_list, vert_list)
    bbox_list3 = [(10, 10, 110, 110)] * len(aid_list)
    bbox_list4 = ibs.get_annot_bboxes(aid_list)
    assert bbox_list3 == bbox_list4, 'bbox lists do not match'
    # finish this test here

    """ set_annot_bboxes / get_annot_verts  """
    bbox_list = [(10, 10, 110, 110)] * len(aid_list)
    ibs.set_annot_bboxes(aid_list, bbox_list)
    # test that setting the bounding boxes overrides the vertices
    vert_list = [((10, 10), (120, 10), (120, 120), (10, 120))] * len(aid_list)
    vert_list2 = ibs.get_annot_verts(aid_list)
    assert vert_list == vert_list2, 'vert lists do not match'

    # put verts back to original state
    # (otherwise other tests will fail on the second run of run_tests.sh)
    ibs.set_annot_verts(aid_list, vert_list_orig)
    assert vert_list_orig == ibs.get_annot_verts(aid_list), 'Verts were not reset to original state'
    assert bbox_list_orig == ibs.get_annot_bboxes(aid_list), 'Bboxes were not reset to original state'

    """ set_image_gps / get_image_gps """
    gid_list = ibs.get_valid_gids()
    assert gids, 'gid_list is empty'
    gps_list_orig = ibs.get_image_gps(gid_list)
    gps_list = [(x, y) for (x, y) in zip(range(len(gid_list)), range(len(gid_list)))]
    ibs.set_image_gps(gid_list, gps_list)
    gps_list2 = ibs.get_image_gps(gid_list)
    assert gps_list == gps_list2, 'gps lists do not match'
    ibs.set_image_gps(gid_list, gps_list_orig)
    assert gps_list_orig == ibs.get_image_gps(gid_list), 'gps was not reset to original state'

    """ set encounter enctext / get encounter enctext """
    eid_list = ibs.get_valid_eids()
    enc_text_list_orig = ibs.get_encounter_text(eid_list)
    enc_text_list = [str(x) for x in range(len(eid_list))]
    assert eid_list, 'eid_list is empty'
    print('len eid_list: %d' % len(eid_list))
    ibs.set_encounter_text(eid_list, enc_text_list)
    enc_text_list2 = ibs.get_encounter_text(eid_list)
    print('enc_text_list = %r' % enc_text_list)
    print('enc_text_list2 = %r' % enc_text_list2)
    assert enc_text_list == enc_text_list2, 'encounter text lists do not match'
    ibs.set_encounter_text(eid_list, enc_text_list_orig)
    assert enc_text_list_orig == ibs.get_encounter_text(eid_list), 'enc text was not reset'

    """ set annotation names / get_annot_names """
    aid_list = ibs.get_valid_aids()
    assert aid_list, 'aid_list is empty'

    orig_names = ibs.get_annot_names(aid_list)
    new_names = ['TESTNAME_' + str(x) for x in range(len(aid_list))]
    ibs.set_annot_names(aid_list, new_names)
    new_names2 = ibs.get_annot_names(aid_list)
    try:
        assert new_names == new_names2, 'new_names == new_names2 failed!'
    except AssertionError as ex:
        utool.printex(ex, key_list=['new_names', 'new_names2'])
        raise

    ibs.set_annot_names(aid_list, orig_names)
    try:
        test_names = ibs.get_annot_names(aid_list)
        assert orig_names == test_names
    except AssertionError as ex:
        utool.printex(ex, key_list=['orig_names', 'test_names'])
        raise

    """ set annotation species / get annotation species """
    aid_list = ibs.get_valid_aids()
    assert aid_list, 'aid_list is empty'
    orig_species = ibs.get_annot_species_texts(aid_list)
    print('orig_species = %r' % (orig_species,))
    new_species = [constants.VALID_SPECIES[x % len(constants.VALID_SPECIES)] for x in range(len(aid_list))]
    print('new_species = %r' % (new_species,))
    ibs.set_annot_species(aid_list, new_species)
    try:
        new_species2 = ibs.get_annot_species_texts(aid_list)
        assert new_species == new_species2, 'new_species == new_species2 failed!'
    except AssertionError as ex:
        utool.printex(ex, key_list=['new_species', 'new_species2'])
        raise
    ibs.set_annot_species(aid_list, orig_species)
    assert orig_species == ibs.get_annot_species_texts(aid_list), 'species were not reset'

    """ set alr confidence / get alr confidence """
    if False:
        # NOT USING ALR TABLE CURRENTLY
        aid_list = ibs.get_valid_aids()
        assert aid_list, 'aid_list is empty'
        alrids_list = ibs.get_annot_alrids(aid_list)
        assert alrids_list, 'alrids_list is empty'
        alrid_list = utool.flatten(alrids_list)
        orig_confidences = ibs.get_alr_confidence(alrid_list)
        new_confidences = list(range(len(alrid_list)))
        #ibs.print_alr_table()
        ibs.set_alr_confidence(alrid_list, new_confidences)
        #ibs.print_alr_table()
        new_confidences2 = ibs.get_alr_confidence(alrid_list)
        assert new_confidences == new_confidences2, 'new_confidences == new_confidences2 failed'
        ibs.set_alr_confidence(alrid_list, orig_confidences)
        assert orig_confidences == ibs.get_alr_confidence(alrid_list), 'alr confidences were not reset'

    """ test metadata  """
    #ibs.print_tables()
    #ibs.print_lblannot_table()
    #ibs.print_alr_table()

    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For win32
    import ibeis
    # Initialize database
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_IBS_CONTROL, ibs)
    #execstr = utool.execstr_dict(test_locals, 'test_locals')
    #exec(execstr)
