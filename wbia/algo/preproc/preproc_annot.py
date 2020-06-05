# -*- coding: utf-8 -*-
"""
helpers for controller manual_annot_funcs
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range, filter, map  # NOQA
import six
import utool as ut
import uuid
from vtool import geometry

(print, rrr, profile) = ut.inject2(__name__, '[preproc_annot]')


def make_annotation_uuids(image_uuid_list, bbox_list, theta_list, deterministic=True):
    try:
        # Check to make sure bbox input is a tuple-list, not a list-list
        if len(bbox_list) > 0:
            try:
                assert isinstance(
                    bbox_list[0], tuple
                ), 'Bounding boxes must be tuples of ints!'
                assert isinstance(
                    bbox_list[0][0], int
                ), 'Bounding boxes must be tuples of ints!'
            except AssertionError as ex:
                ut.printex(ex)
                print('bbox_list = %r' % (bbox_list,))
                raise
        annotation_uuid_list = [
            ut.augment_uuid(img_uuid, bbox, theta)
            for img_uuid, bbox, theta in zip(image_uuid_list, bbox_list, theta_list)
        ]
        if not deterministic:
            # Augment determenistic uuid with a random uuid to ensure randomness
            # (this should be ensured in all hardward situations)
            annotation_uuid_list = [
                ut.augment_uuid(ut.random_uuid(), _uuid) for _uuid in annotation_uuid_list
            ]
    except Exception as ex:
        ut.printex(
            ex,
            'Error building annotation_uuids',
            '[add_annot]',
            key_list=['image_uuid_list'],
        )
        raise
    return annotation_uuid_list


def generate_annot_properties(
    ibs,
    gid_list,
    bbox_list=None,
    theta_list=None,
    species_list=None,
    nid_list=None,
    name_list=None,
    detect_confidence_list=None,
    notes_list=None,
    vert_list=None,
    annot_uuid_list=None,
    yaw_list=None,
    quiet_delete_thumbs=False,
):
    # annot_uuid_list = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list,
    #                                                      theta_list, deterministic=False)
    image_uuid_list = ibs.get_image_uuids(gid_list)
    if annot_uuid_list is None:
        annot_uuid_list = [uuid.uuid4() for _ in range(len(image_uuid_list))]
    # Prepare the SQL input
    assert name_list is None or nid_list is None, 'cannot specify both names and nids'
    # For import only, we can specify both by setting import_override to True
    assert bool(bbox_list is None) != bool(
        vert_list is None
    ), 'must specify exactly one of bbox_list or vert_list'

    if theta_list is None:
        theta_list = [0.0 for _ in range(len(gid_list))]
    if name_list is not None:
        nid_list = ibs.add_names(name_list)
    if detect_confidence_list is None:
        detect_confidence_list = [0.0 for _ in range(len(gid_list))]
    if notes_list is None:
        notes_list = ['' for _ in range(len(gid_list))]
    if vert_list is None:
        vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
    elif bbox_list is None:
        bbox_list = geometry.bboxes_from_vert_list(vert_list)

    len_bbox = len(bbox_list)
    len_vert = len(vert_list)
    len_gid = len(gid_list)
    len_notes = len(notes_list)
    len_theta = len(theta_list)
    try:
        assert len_vert == len_bbox, 'bbox and verts are not of same size'
        assert len_gid == len_bbox, 'bbox and gid are not of same size'
        assert len_gid == len_theta, 'bbox and gid are not of same size'
        assert len_notes == len_gid, 'notes and gids are not of same size'
    except AssertionError as ex:
        ut.printex(
            ex, key_list=['len_vert', 'len_gid', 'len_bbox' 'len_theta', 'len_notes']
        )
        raise

    if len(gid_list) == 0:
        # nothing is being added
        print('[ibs] WARNING: 0 annotations are beign added!')
        print(ut.repr2(locals()))
        return []

    # Build ~~deterministic?~~ random and unique ANNOTATION ids
    image_uuid_list = ibs.get_image_uuids(gid_list)
    # annot_uuid_list = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list,
    #                                                      theta_list, deterministic=False)
    if annot_uuid_list is None:
        annot_uuid_list = [uuid.uuid4() for _ in range(len(image_uuid_list))]
    if yaw_list is None:
        yaw_list = [-1.0] * len(image_uuid_list)
    nVert_list = [len(verts) for verts in vert_list]
    vertstr_list = [six.text_type(verts) for verts in vert_list]
    xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
    assert len(nVert_list) == len(vertstr_list)
    # Define arguments to insert


def testdata_preproc_annot():
    import wbia

    ibs = wbia.opendb('testdb1')
    aid_list = ibs.get_valid_aids()
    return ibs, aid_list


def postget_annot_verts(vertstr_list):
    # TODO: Sanatize input for eval
    # print('vertstr_list = %r' % (vertstr_list,))
    locals_ = {}
    globals_ = {}
    vert_list = [eval(vertstr, globals_, locals_) for vertstr in vertstr_list]
    return vert_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.template_generator --tbls annotations --Tflags getters native

        python -m wbia.algo.preproc.preproc_annot
        python -m wbia.algo.preproc.preproc_annot --allexamples
        python -m wbia.algo.preproc.preproc_annot --allexamples --noface --nosrc

    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
