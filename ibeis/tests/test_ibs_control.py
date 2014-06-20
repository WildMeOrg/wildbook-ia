#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
from ibeis import viz
from vtool.tests import grabdata
from itertools import izip
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_IBS_CONTROL]')


def TEST_IBS_CONTROL(ibs):
	ibs.compute_encounters()

	""" get_image_eids / get_encounter_gids """
	eid_list = ibs.get_valid_eids()
	assert eid_list, 'eid_list is empty'
	gids_list = ibs.get_encounter_gids(eid_list)
	#print('[TEST] gids_list = %r' % gids_list)
	assert gids_list, 'gids_list is empty'
	for gids, eid in izip(gids_list, eid_list):
		eid_list2 = ibs.get_image_eids(gids)
		assert ([[eid]] * len(eid_list2)) == eid_list2
		#print('[TEST] eid_list2 = %r' % eid_list2)
		#print('[TEST] eid = %r' % eid)

	""" set_roi_notes / get_roi_notes """
	rid_list = ibs.get_valid_rids()
	roi_notes_list = len(rid_list) * ["test text"]
	assert rid_list, 'rid_list is empty'
	ibs.set_roi_notes(rid_list, roi_notes_list)
	roi_notes_list2 = ibs.get_roi_notes(rid_list)
	assert roi_notes_list2, 'get_roi_notes returned an empty list'
	#print('[TEST] roi_notes_list = %r' % roi_notes_list)
	#print('[TEST] roi_notes_list2 = %r' % roi_notes_list2)
	assert roi_notes_list == roi_notes_list2, 'roi notes lists do not match'

	""" set_name_notes / get_name_notes """
	nid_list = ibs.get_valid_nids()
	assert nid_list, 'nid_list is empty'
	nid_notes_list = len(nid_list) * ['nid notes test']
	ibs.set_name_notes(nid_list, nid_notes_list)
	nid_notes_list2 = ibs.get_name_notes(nid_list)
	print('[TEST] nid_notes_list = %r' % nid_notes_list)
	print('[TEST] nid_notes_list2 = %r' % nid_notes_list2)
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

	""" set_roi_bboxes / get_roi_bboxes """
	rid_list = ibs.get_valid_rids()
	assert rid_list, 'rid_list is empty'
	bbox_list = [(1,2,3,4)] * len(rid_list)
	ibs.set_roi_bboxes(rid_list, bbox_list)
	bbox_list2 = ibs.get_roi_bboxes(rid_list)
	print('[TEST] bbox_list = %r' % bbox_list)
	print('[TEST] bbox_list2 = %r' % bbox_list2)
	assert bbox_list == bbox_list2, 'bbox lists do not match'

	""" set_roi_verts / get_roi_verts """
	rid_list = ibs.get_valid_rids()
	assert rid_list, 'rid_list is empty'
	vert_list = [((1,2),(3,4),(5,6),(7,8))] * len(rid_list)
	print('[TEST] vert_list = %r' % vert_list)
	assert len(rid_list) == len(vert_list), 'lengths do not match, malformed input'
	ibs.set_roi_verts(rid_list, vert_list)
	vert_list2 = ibs.get_roi_verts(rid_list)
	assert vert_list == vert_list2, 'vert lists do not match'
	
	# eid_list = ibs.get_valid_eids()
	# enc_text_list = len(eid_list) * ["test encounter text"]
	# assert eid_list, 'eid_list is empty'
	# print('len eid_list: %d' % len(eid_list))
	# ibs.set_encounter_enctext(eid_list, enc_text_list)
	# enc_text_list2 = ibs.get_encounter_enctext(eid_list)
	# print('enc_text_list = %r' % enc_text_list)
	# print('enc_text_list2 = %r' % enc_text_list2)
	# assert enc_text_list == enc_text_list2, 'encounter text lists do not match'
	
	return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For win32
    import ibeis
    # Initialize database
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_IBS_CONTROL, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)