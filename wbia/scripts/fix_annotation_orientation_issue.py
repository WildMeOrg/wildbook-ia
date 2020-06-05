# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)


def fix_annotation_orientation(ibs, min_percentage=0.95):
    """
    Fixes the annotations that are outside the bounds of the image due to a
    changed image orientation flag in the database

    CommandLine:
        python -m wbia.scripts.fix_annotation_orientation_issue fix_annotation_orientation

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> from wbia.scripts.fix_annotation_orientation_issue import *  # NOQA
        >>> ibs = wbia.opendb()
        >>> unfixable_gid_list = fix_annotation_orientation(ibs)
        >>> assert len(unfixable_gid_list) == 0
    """
    from vtool import exif

    def bbox_overlap(bbox1, bbox2):
        ax1 = bbox1[0]
        ay1 = bbox1[1]
        ax2 = bbox1[0] + bbox1[2]
        ay2 = bbox1[1] + bbox1[3]
        bx1 = bbox2[0]
        by1 = bbox2[1]
        bx2 = bbox2[0] + bbox2[2]
        by2 = bbox2[1] + bbox2[3]
        overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
        overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
        return overlap_x * overlap_y

    orient_dict = exif.ORIENTATION_DICT_INVERSE
    good_orient_list = [
        exif.ORIENTATION_UNDEFINED,
        exif.ORIENTATION_000,
    ]
    good_orient_key_list = [
        orient_dict.get(good_orient) for good_orient in good_orient_list
    ]
    assert None not in good_orient_key_list

    gid_list = ibs.get_valid_gids()
    orient_list = ibs.get_image_orientation(gid_list)
    flag_list = [orient not in good_orient_key_list for orient in orient_list]

    # Filter based on based gids
    unfixable_gid_list = []
    gid_list = ut.filter_items(gid_list, flag_list)
    if len(gid_list) > 0:
        args = (len(gid_list),)
        print('Found %d images with non-standard orientations' % args)
        aids_list = ibs.get_image_aids(gid_list, is_staged=None, __check_staged__=False)
        size_list = ibs.get_image_sizes(gid_list)
        invalid_gid_list = []
        zipped = zip(gid_list, orient_list, aids_list, size_list)
        for gid, orient, aid_list, (w, h) in zipped:
            image = ibs.get_images(gid)
            h_, w_ = image.shape[:2]
            if h != h_ or w != w_:
                ibs._set_image_sizes([gid], [w_], [h_])
            orient_str = exif.ORIENTATION_DICT[orient]
            image_bbox = (0, 0, w, h)
            verts_list = ibs.get_annot_rotated_verts(aid_list)
            invalid = False
            for aid, vert_list in zip(aid_list, verts_list):
                annot_bbox = vt.bbox_from_verts(vert_list)
                overlap = bbox_overlap(image_bbox, annot_bbox)
                area = annot_bbox[2] * annot_bbox[3]
                percentage = overlap / area
                if percentage < min_percentage:
                    args = (gid, orient_str, aid, overlap, area, percentage)
                    print(
                        '\tInvalid GID %r, Orient %r, AID %r: Overlap %0.2f, Area %0.2f (%0.2f %%)'
                        % args
                    )
                    invalid = True
                    # break
            if invalid:
                invalid_gid_list.append(gid)

        invalid_gid_list = list(set(invalid_gid_list))
        if len(invalid_gid_list) > 0:
            args = (
                len(invalid_gid_list),
                len(gid_list),
                invalid_gid_list,
            )
            print('Found %d / %d images with invalid annotations = %r' % args)
            orient_list = ibs.get_image_orientation(invalid_gid_list)
            aids_list = ibs.get_image_aids(
                invalid_gid_list, is_staged=None, __check_staged__=False
            )
            size_list = ibs.get_image_sizes(invalid_gid_list)
            zipped = zip(invalid_gid_list, orient_list, aids_list, size_list)
            for invalid_gid, orient, aid_list, (w, h) in zipped:
                orient_str = exif.ORIENTATION_DICT[orient]
                image_bbox = (0, 0, w, h)
                args = (
                    invalid_gid,
                    len(aid_list),
                )
                print('Fixing GID %r with %d annotations' % args)
                theta = np.pi / 2.0
                tx = 0.0
                ty = 0.0
                if orient == orient_dict.get(exif.ORIENTATION_090):
                    theta *= 1.0
                    tx = w
                elif orient == orient_dict.get(exif.ORIENTATION_180):
                    theta *= 2.0
                    tx = w
                    ty = h
                elif orient == orient_dict.get(exif.ORIENTATION_270):
                    theta *= -1.0
                    ty = h
                else:
                    raise ValueError('Unrecognized invalid orientation')
                H = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), tx],
                        [np.sin(theta), np.cos(theta), ty],
                        [0.0, 0.0, 1.0],
                    ]
                )
                # print(H)
                verts_list = ibs.get_annot_rotated_verts(aid_list)
                for aid, vert_list in zip(aid_list, verts_list):
                    vert_list = np.array(vert_list)
                    # print(vert_list)
                    vert_list = vert_list.T
                    transformed_vert_list = vt.transform_points_with_homography(
                        H, vert_list
                    )
                    transformed_vert_list = transformed_vert_list.T
                    # print(transformed_vert_list)

                    ibs.set_annot_verts(
                        [aid], [transformed_vert_list], update_visual_uuids=False
                    )
                    current_theta = ibs.get_annot_thetas(aid)
                    new_theta = current_theta + theta
                    ibs.set_annot_thetas(aid, new_theta, update_visual_uuids=False)

                    fixed_vert_list = ibs.get_annot_rotated_verts(aid)
                    fixed_annot_bbox = vt.bbox_from_verts(fixed_vert_list)
                    fixed_overlap = bbox_overlap(image_bbox, fixed_annot_bbox)
                    fixed_area = fixed_annot_bbox[2] * fixed_annot_bbox[3]
                    fixed_percentage = fixed_overlap / fixed_area
                    args = (
                        invalid_gid,
                        orient_str,
                        aid,
                        fixed_overlap,
                        fixed_area,
                        fixed_percentage,
                    )
                    print(
                        '\tFixing GID %r, Orient %r, AID %r: Overlap %0.2f, Area %0.2f (%0.2f %%)'
                        % args
                    )
                    if fixed_percentage < min_percentage:
                        print('\tWARNING: FIXING DID NOT CORRECT AID %r' % (aid,))
                        unfixable_gid_list.append(gid)
    print('Un-fixable gid_list = %r' % (unfixable_gid_list,))
    return unfixable_gid_list


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
