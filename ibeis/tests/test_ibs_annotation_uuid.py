from __future__ import absolute_import, division, print_function
from ibeis import ibsfuncs


def test_annotation_uuid(ibs):
    """ Consistency test """
    aid_list        = ibs.get_valid_aids()
    bbox_list       = ibs.get_annot_bboxes(aid_list)
    theta_list      = ibs.get_annot_thetas(aid_list)
    image_uuid_list = ibs.get_annot_image_uuids(aid_list)

    annotation_uuid_list1 = ibs.get_annot_uuids(aid_list)
    annotation_uuid_list2 = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list, theta_list)

    assert annotation_uuid_list1 == annotation_uuid_list2
