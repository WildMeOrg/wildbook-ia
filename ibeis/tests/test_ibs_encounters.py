#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from six.moves import map
from ibeis import ibsfuncs
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
    aids_list = ibs.get_encounter_aids(eid_list)
    nids_list = ibs.get_encounter_nids(eid_list)

    enctext_list   = ibs.get_encounter_text(eid_list)
    gid_uuids_list = list(map(list, ibsfuncs.unflat_map(ibs.get_image_uuids, gids_list)))
    annotation_uuids_list = list(map(list, ibsfuncs.unflat_map(ibs.get_annot_uuids, aids_list)))
    names_list     = list(map(list, ibsfuncs.unflat_map(ibs.get_name_texts, nids_list)))

    #target_enctexts = ['E0_ENC(agg,sec_60,1)', 'E1_ENC(agg,sec_60,1)']
    target_enctexts = [u'Encounter 0', u'Encounter 1']

    target_gid_uuids = [[UUID('66ec193a-1619-b3b6-216d-1784b4833b61'),
                         UUID('d8903434-942f-e0f5-d6c2-0dcbe3137bf7'),
                         UUID('b73b72f4-4acb-c445-e72c-05ce02719d3d'),
                         UUID('0cd05978-3d83-b2ee-2ac9-798dd571c3b3'),
                         UUID('0a9bc03d-a75e-8d14-0153-e2949502aba7'),
                         UUID('2deeff06-5546-c752-15dc-2bd0fdb1198a'),
                         UUID('a9b70278-a936-c1dd-8a3b-bc1e9a998bf0')],
                        [UUID('42fdad98-369a-2cbc-67b1-983d6d6a3a60'),
                         UUID('c459d381-fd74-1d99-6215-e42e3f432ea9'),
                         UUID('33fd9813-3a2b-774b-3fcc-4360d1ae151b'),
                         UUID('97e8ea74-873f-2092-b372-f928a7be30fa'),
                         UUID('588bc218-83a5-d400-21aa-d499832632b0'),
                         UUID('163a890c-36f2-981e-3529-c552b6d668a3')],
                        ]

    target_name_texts = [
        ['easy', 'hard', 'jeff'],
        ['lena', 'occl', 'polar', 'zebra'],
    ]

    ibs.print_lblannot_table()
    ibs.print_egpairs_table()
    ibs.print_encounter_table()
    ibs.print_alr_table()
    gids_test_list = ibsfuncs.unflat_map(ibs.get_image_gids_from_uuid, gid_uuids_list)
    gids_target_list = ibsfuncs.unflat_map(ibs.get_image_gids_from_uuid, target_gid_uuids)
    try:
        print('0) gid_uuids_list = %s' % (utool.list_str(gids_test_list),))
        print('0) target_gid_uuids = %s' % (utool.list_str(gids_target_list),))
        print('')
        print('1) gid_uuids_list = %s' % (utool.list_str(gid_uuids_list),))
        print('1) target_gid_uuids = %s' % (utool.list_str(target_gid_uuids),))
        print('')
        print('2) enctext_list = %r' % (enctext_list,))
        print('2) target_enctexts = %r' % (target_enctexts,))
        print('')

        aids_test_list = ibsfuncs.unflat_map(ibs.get_image_aids, gids_test_list)
        aids_target_list = ibsfuncs.unflat_map(ibs.get_image_aids, gids_target_list)
        print('a) aids_test_list = %s' % (utool.list_str(aids_test_list),))
        print('a) aids_target_list = %s' % (utool.list_str(aids_target_list),))

        print('3a) aids_list = %s' % (utool.list_str(aids_list),))
        print('3a) nids_list = %s' % (utool.list_str(nids_list),))
        nids_listb = [ ibs.get_annot_name_rowids(aid_list) for aid_list in aids_list ]
        print('3a) nids_listb = %s' % (utool.list_str(nids_listb),))
        print('3b) names_list = %s' % (utool.list_str(names_list),))
        print('3b) target_name_texts = %s' % (utool.list_str(target_name_texts),))
        print('')

        assert gids_test_list == gids_target_list, 'gids_test_list does not match gids_target_list'

        assert gid_uuids_list == target_gid_uuids, 'gid_uuids_list does not match target_gid_uuids'

        assert enctext_list == target_enctexts, 'enctext_list does not match target_enctexts'

        assert names_list == target_name_texts, 'names_list does not match target_name_texts'

    except AssertionError as ex:
        utool.printex(ex, 'failed test_encounter')
        raise

    gids_list2 = list(map(list, ibsfuncs.unflat_map(ibs.get_annot_gids, aids_list)))
    assert gids_list2 == list(map(list, gids_list))

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_ENCOUNTERS, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
