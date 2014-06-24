#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.dev import ibsfuncs
#from itertools import izip
# Python
import multiprocessing
#import numpy as np
from uuid import UUID
# Tools
import utool
from ibeis.control.IBEISControl import IBEISController
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_ENCOUNTERS]')


def TEST_ENCOUNTERS(ibs):
    print('[TEST_ENCOUNTERS]')
    assert isinstance(ibs, IBEISController), 'type enforment'
    # Delete all encounters
    _eid_list1 = ibs.get_valid_eids()
    ibs.delete_encounters(_eid_list1)
    _eid_list2 = ibs.get_valid_eids()
    assert len(_eid_list2) == 0, 'eids should have been deleted'
    # Recompute encounters
    ibs.compute_encounters()

    eid_list = sorted(ibs.get_valid_eids())
    gids_list = ibs.get_encounter_gids(eid_list)
    rids_list = ibs.get_encounter_rids(eid_list)
    nids_list = ibs.get_encounter_nids(eid_list)

    enctext_list   = ibs.get_encounter_enctext(eid_list)
    gid_uuids_list = ibsfuncs.unflat_map(ibs.get_image_uuids, gids_list)
    roi_uuids_list = ibsfuncs.unflat_map(ibs.get_roi_uuids, rids_list)
    names_list     = ibsfuncs.unflat_map(ibs.get_names, nids_list)

    target_enctexts = ['_ENC(1,agg,sec_60)0', '_ENC(1,agg,sec_60)1']

    target_gid_uuids = [(UUID('08c76203-3614-cada-aace-01d22b3378e2'),
                         UUID('f409ae19-e64c-c40b-75bb-30bb18f5a1a3'),
                         UUID('383dda50-f26b-200a-8baf-548e3ef88f9c'),
                         UUID('70821b89-f34f-ab57-4575-5384380c592a'),
                         UUID('9d3ae88a-8c33-43ef-530e-62dbf916fff2'),
                         UUID('f2c4658f-a722-793b-8578-c392fd550888'),
                         UUID('aed981a2-4116-9936-6311-e46bd17e25de')),
                        (UUID('d35757a5-8f1b-80d5-8b35-9eb8661cb8df'),
                         UUID('9dd931ae-04d3-4996-3a3a-c6ca4a375ba4'),
                         UUID('145d74ce-b0e0-dce3-9bf5-365599e86b56'),
                         UUID('3c06d3d3-1073-5f28-6439-d7edac4c1893'),
                         UUID('672b1bd6-1516-d5fa-14f9-b39594447e23'),
                         UUID('a4597ee8-9e11-c704-efdc-f0d8a1d755b5'))]

    target_names = [
    ('polar', '____8', 'easy', 'hard', '____6', 'lena', 'jeff'),
    ('occl', 'zebra', '____1', 'easy', '____5', 'hard'),
]

    ibs.print_label_table()
    ibs.print_egpairs_table()
    ibs.print_encounter_table()
    ibs.print_rlr_table()
    gids_test_list = ibsfuncs.unflat_map(ibs.get_image_gids_from_uuid, gid_uuids_list)
    gids_target_list = ibsfuncs.unflat_map(ibs.get_image_gids_from_uuid, target_gid_uuids)

    rids_test_list = ibsfuncs.unflat_map(ibs.get_image_rids, gids_test_list)
    rids_target_list = ibsfuncs.unflat_map(ibs.get_image_rids, gids_target_list)
    print('a) rids_test_list = %s' % (utool.list_str(rids_test_list),))
    print('a) rids_target_list = %s' % (utool.list_str(rids_target_list),))
    try:
        print('0) gid_uuids_list = %s' % (utool.list_str(gids_test_list),))
        print('0) target_gid_uuids = %s' % (utool.list_str(gids_target_list),))
        print('')
        assert gids_test_list == gids_target_list, 'gids_test_list does not match gids_target_list'

        print('1) gid_uuids_list = %s' % (utool.list_str(gid_uuids_list),))
        print('1) target_gid_uuids = %s' % (utool.list_str(target_gid_uuids),))
        print('')
        assert gid_uuids_list == target_gid_uuids, 'gid_uuids_list does not match target_gid_uuids'

        print('2) enctext_list = %r' % (enctext_list,))
        print('2) target_enctexts = %r' % (target_enctexts,))
        print('')
        assert enctext_list == target_enctexts, 'enctext_list does not match target_enctexts'
        
        print('3a) rids_list = %s' % (utool.list_str(rids_list),))
        print('3a) nids_list = %s' % (utool.list_str(nids_list),))
        nids_listb = ibsfuncs.unflat_map(ibs.get_roi_nids, rids_list)
        print('3a) nids_listb = %s' % (utool.list_str(nids_listb),))
        
        print('3b) names_list = %s' % (utool.list_str(names_list),))
        print('3b) target_names = %s' % (utool.list_str(target_names),))
        print('')
        assert names_list == target_names, 'names_list does not match target_names'

    except AssertionError as ex:
        utool.printex(ex, 'failed test_encounter')
        raise

    gids_list2 = ibsfuncs.unflat_lookup(ibs.get_roi_gids, rids_list)
    assert gids_list2 == map(tuple, gids_list)

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_ENCOUNTERS, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
