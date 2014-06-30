from __future__ import absolute_import, division, print_function
from ibeis.dev import ibsfuncs


def test_annotation_uuid(ibs):
    """ Consistency test """
    aid_list        = ibs.get_valid_aids()
    bbox_list       = ibs.get_annotation_bboxes(aid_list)
    theta_list      = ibs.get_annotation_thetas(aid_list)
    image_uuid_list = ibs.get_annotation_image_uuids(aid_list)

    annotation_uuid_list1 = ibs.get_annotation_uuids(aid_list)
    annotation_uuid_list2 = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list, theta_list)

    assert annotation_uuid_list1 == annotation_uuid_list2
