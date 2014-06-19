#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.dev import ibsfuncs
from itertools import izip
# Python
import multiprocessing
#import numpy as np
from uuid import UUID
# Tools
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_ENCOUNTERS]')


def TEST_ENCOUNTERS(ibs):
    print('[TEST_ENCOUNTERS]')
    ibs.compute_encounters()

    eid_list = ibs.get_valid_eids()
    gids_list = ibs.get_encounter_gids(eid_list)
    rids_list = ibs.get_encounter_rids(eid_list)
    nids_list = ibs.get_encounter_nids(eid_list)

    enctext_list   = ibs.get_encounter_enctext(eid_list)
    gid_uuids_list = ibs.get_unflat_image_uuids(rids_list)
    roi_uuids_list = ibs.get_unflat_roi_uuids(rids_list)
    names_list     = ibs.get_unflat_names(nids_list)

    target_enctexts = ['_ENC(1,ms,quant_0.01)0',
                       '_ENC(1,ms,quant_0.01)0-2',
                       '_ENC(1,ms,quant_0.01)0-3',
                       '_ENC(1,ms,quant_0.01)0-4',
                       '_ENC(1,ms,quant_0.01)0-5',
                       '_ENC(1,ms,quant_0.01)0-6',
                       '_ENC(1,ms,quant_0.01)0-7',
                       '_ENC(1,ms,quant_0.01)0-8',
                       '_ENC(1,ms,quant_0.01)0-9',
                       '_ENC(1,ms,quant_0.01)0-10',
                       '_ENC(1,ms,quant_0.01)0-11',
                       '_ENC(1,ms,quant_0.01)0-12',
                       '_ENC(1,ms,quant_0.01)0-13']

    target_gid_uuids = [(UUID('70821b89-f34f-ab57-4575-5384380c592a'),
                         UUID('f2c4658f-a722-793b-8578-c392fd550888'),
                         UUID('08c76203-3614-cada-aace-01d22b3378e2'),
                         UUID('9d3ae88a-8c33-43ef-530e-62dbf916fff2'),
                         UUID('9dd931ae-04d3-4996-3a3a-c6ca4a375ba4'),
                         UUID('145d74ce-b0e0-dce3-9bf5-365599e86b56'),
                         UUID('d35757a5-8f1b-80d5-8b35-9eb8661cb8df'),
                         UUID('5485daf4-3a9a-fa68-196d-79c996f11181'),
                         UUID('aed981a2-4116-9936-6311-e46bd17e25de'),
                         UUID('383dda50-f26b-200a-8baf-548e3ef88f9c'),
                         UUID('672b1bd6-1516-d5fa-14f9-b39594447e23'),
                         UUID('a4597ee8-9e11-c704-efdc-f0d8a1d755b5'),
                         UUID('96735673-dce9-f43a-2cdf-4dc229a0627c'))]

    # roi_uuids are no longer testable as they are random
    #target_roi_uuids = [(UUID('42a3e2c3-bf75-8f01-6bd1-303a7b30eec2'),
    #                     UUID('ba47db63-439c-e300-0c13-80a40816a13d'),
    #                     UUID('0db42bc8-0ee1-8dee-b528-292f87a6d681'),
    #                     UUID('4bf7a461-6892-352d-e8cd-9d71864df380'),
    #                     UUID('e0cc89cc-fe0d-b2e9-2a8f-6b19ed03a566'),
    #                     UUID('cd338bea-16cc-c8e4-e7e3-7ffd92814024'),
    #                     UUID('9a9b1c0a-dc6a-868d-0858-e5588cad52e8'),
    #                     UUID('1d24701e-c85a-41ab-a4c6-0559ae00e370'),
    #                     UUID('10678bf2-597d-a25e-3fff-86929529cfb9'),
    #                     UUID('27dc2758-0e81-f337-fefb-b75e88581b1a'),
    #                     UUID('f0ea15bf-5d38-b65f-d157-0c4a3f34755f'),
    #                     UUID('ccabdf13-798d-152a-b6b1-6b7f106bbff3'),
    #                     UUID('0e3da848-f9f7-2bca-54cd-8e25ae1403b6'))]

    target_names = [('hard', 'lena', 'polar', 'zebra', 'occl', 'easy', 'jeff')]
    print(gid_uuids_list)
    print(target_gid_uuids)
    def assert_unflat_eq(unflat_list1, unflat_list2, lbl=''):
        print('testing %r' % lbl)
        print('unflat_list1 %r' % unflat_list1)
        print('unflat_list2 %r' % unflat_list2)
        passed_unsorted = True
        for ix, (list1, list2) in enumerate(izip(unflat_list1, unflat_list2)):
            try:
                for jx, (item1, item2) in enumerate(izip(list1, list2)):
                    if item1 != item2:
                        msg = (('Failed unsorted at pos ix=%r, jx=%r\n%r != %r') %
                               (ix, jx, item1, item2))
                        print(msg)
                        passed_unsorted = False
                        raise AssertionError(msg)
            except AssertionError:
                sorted1 = sorted(list1)
                sorted2 = sorted(list2)
                print('sorted1 %r' % sorted1)
                print('sorted2 %r' % sorted2)
                for jx, (item1, item2) in enumerate(izip(sorted1, sorted2)):
                    if item1 != item2:
                        msg = (('Failed sorted at pos ix=%r, jx=%r\n%r != %r') %
                               (ix, jx, item1, item2))
                        print(msg)
                        raise AssertionError(msg)
        if passed_unsorted:
            print('%r passed unsorted' % lbl)
        else:
            print('%r passed sorted' % lbl)

    if ibs.get_dbname() == 'testdb1':
        print('testing assertions')
        try:
            #assert_unflat_eq(roi_uuids_list, target_roi_uuids)
            assert_unflat_eq(names_list, target_names)
            assert_unflat_eq(enctext_list, target_enctexts)
            assert_unflat_eq(gid_uuids_list, target_gid_uuids)
            #assert map(set, roi_uuids_list) == map(set, target_roi_uuids)
            #assert map(set, names_list) == map(set, target_names)

            #ut1, ut2 = ibs.get_unflat_image_unixtime(gids_list)
            #ut1 = np.array(ut1)
            #ut2 = np.array(ut2)
            #assert(np.all(ut1 > 9000))
            #assert(np.all(ut2 < 9000))
        except AssertionError as ex:
            utool.printex(ex, key_list=['eid_list', 'gid_uuids_list', 'roi_uuids_list', 'names_list'])
            raise
        print('assertions passed')

    ibs.delete_encounters(eid_list)
    ibs.compute_encounters()
    #ibs.print_egpairs_table()
    #ibs.print_encounter_table()

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
