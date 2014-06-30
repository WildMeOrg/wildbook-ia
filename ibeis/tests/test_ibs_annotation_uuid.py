from __future__ import absolute_import, division, print_function
from ibeis.dev import ibsfuncs


def test_annotion_uuid(ibs):
    """ Consistency test """
    aid_list        = ibs.get_valid_aids()
    bbox_list       = ibs.get_annotion_bboxes(aid_list)
    theta_list      = ibs.get_annotion_thetas(aid_list)
    image_uuid_list = ibs.get_annotion_image_uuids(aid_list)

    annotion_uuid_list1 = ibs.get_annotion_uuids(aid_list)
    annotion_uuid_list2 = ibsfuncs.make_annotion_uuids(image_uuid_list, bbox_list, theta_list)

    assert annotion_uuid_list1 == annotion_uuid_list2
