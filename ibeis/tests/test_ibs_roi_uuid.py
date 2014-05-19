from __future__ import absolute_import, division, print_function
from ibeis.dev import ibsfuncs


def test_roi_uuid(ibs):
    """ Consistency test """
    rid_list        = ibs.get_valid_rids()
    bbox_list       = ibs.get_roi_bboxes(rid_list)
    theta_list      = ibs.get_roi_thetas(rid_list)
    image_uuid_list = ibs.get_roi_image_uuids(rid_list)

    roi_uuid_list1 = ibs.get_roi_uuids(rid_list)
    roi_uuid_list2 = ibsfuncs.make_roi_uuids(image_uuid_list, bbox_list, theta_list)

    assert roi_uuid_list1 == roi_uuid_list2
