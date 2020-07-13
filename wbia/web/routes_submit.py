# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from flask import request, redirect, url_for, current_app
from wbia.control import controller_inject
from wbia.web import appfuncs as appf
from wbia import constants as const
from wbia.web.routes import THROW_TEST_AOI_TURKING
import utool as ut
import numpy as np
import uuid
import six

(print, rrr, profile) = ut.inject2(__name__)

register_route = controller_inject.get_wbia_flask_route(__name__)


@register_route('/submit/login/', methods=['POST'], __route_authenticate__=False)
def submit_login(name, organization, refer=None, *args, **kwargs):
    # Return HTML
    if refer is None:
        refer = url_for('root')
    else:
        refer = appf.decode_refer_url(refer)

    if name == '_new_':
        first = kwargs['new_name_first']
        last = kwargs['new_name_last']
        name = '%s.%s' % (first, last,)
        name = name.replace(' ', '')

    if organization == '_new_':
        organization = kwargs['new_organization']
        organization = organization.replace(' ', '')

    name = name.lower()
    organization = organization.lower()

    username = '%s@%s' % (name, organization,)
    controller_inject.authenticate(
        username=username, name=name, organization=organization
    )

    return redirect(refer)


@register_route('/submit/cameratrap/', methods=['POST'])
def submit_cameratrap(**kwargs):
    ibs = current_app.ibs
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    gid = int(request.form['cameratrap-gid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)
    flag = request.form.get('cameratrap-toggle', 'off') == 'on'
    ibs.set_image_cameratrap([gid], [flag])
    print('[web] user_id: %s, gid: %d, flag: %r' % (user_id, gid, flag,))

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_cameratrap', imgsetid=imgsetid, previous=gid))


@register_route('/submit/detection/', methods=['POST'])
def submit_detection(**kwargs):
    with ut.Timer('submit'):
        is_staged = kwargs.get('staged', False)
        is_canonical = kwargs.get('canonical', False)

        is_staged = is_staged and appf.ALLOW_STAGED

        ibs = current_app.ibs
        method = request.form.get('detection-submit', '')
        imgsetid = request.args.get('imgsetid', '')
        imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
        gid = int(request.form['detection-gid'])
        only_aid = request.form['detection-only-aid']
        only_aid = None if only_aid == 'None' or only_aid == '' else int(only_aid)
        user = controller_inject.get_user()
        if user is None:
            user = {}
        user_id = user.get('username', None)

        if is_canonical:
            assert only_aid is not None
            assert not is_staged
        if only_aid is not None:
            assert is_canonical

        poor_boxes = method.lower() == 'poor boxes'
        if poor_boxes:
            imgsetid_ = ibs.get_imageset_imgsetids_from_text('POOR BOXES')
            ibs.set_image_imgsetids([gid], [imgsetid_])
            method = 'accept'

        if method.lower() == 'delete':
            # ibs.delete_images(gid)
            # print('[web] (DELETED) user_id: %s, gid: %d' % (user_id, gid, ))
            pass
        elif method.lower() == 'clear':
            aid_list = ibs.get_image_aids(gid)
            ibs.delete_annots(aid_list)
            print('[web] (CLEAERED) user_id: %s, gid: %d' % (user_id, gid,))
            redirection = request.referrer
            if 'gid' not in redirection:
                # Prevent multiple clears
                if '?' in redirection:
                    redirection = '%s&gid=%d' % (redirection, gid,)
                else:
                    redirection = '%s?gid=%d' % (redirection, gid,)
            return redirect(redirection)
        elif method.lower() == 'rotate left':
            ibs.update_image_rotate_left_90([gid])
            print('[web] (ROTATED LEFT) user_id: %s, gid: %d' % (user_id, gid,))
            redirection = request.referrer
            if 'gid' not in redirection:
                # Prevent multiple clears
                if '?' in redirection:
                    redirection = '%s&gid=%d' % (redirection, gid,)
                else:
                    redirection = '%s?gid=%d' % (redirection, gid,)
            return redirect(redirection)
        elif method.lower() == 'rotate right':
            ibs.update_image_rotate_right_90([gid])
            print('[web] (ROTATED RIGHT) user_id: %s, aid: %d' % (user_id, gid,))
            redirection = request.referrer
            if 'gid' not in redirection:
                # Prevent multiple clears
                if '?' in redirection:
                    redirection = '%s&gid=%d' % (redirection, gid,)
                else:
                    redirection = '%s?gid=%d' % (redirection, gid,)
            return redirect(redirection)
        else:
            with ut.Timer('submit...update'):
                current_aid_list = ibs.get_image_aids(gid, is_staged=is_staged)

                if is_canonical:
                    assert (
                        only_aid in current_aid_list
                    ), 'Specified only_aid is not in this image'
                    current_aid_list = [only_aid]

                current_part_rowid_list = ut.flatten(
                    ibs.get_annot_part_rowids(current_aid_list, is_staged=is_staged)
                )

                if is_canonical:
                    current_part_type_list = ibs.get_part_types(current_part_rowid_list)
                    zipped = zip(current_part_rowid_list, current_part_type_list)

                    current_part_rowid_list_ = []
                    for current_part_rowid, current_part_type in zipped:
                        if current_part_type != appf.CANONICAL_PART_TYPE:
                            continue
                        current_part_rowid_list_.append(current_part_rowid)
                    current_part_rowid_list = current_part_rowid_list_

                    assert (
                        len(current_aid_list) == 1
                    ), 'Must have an canonical annotation to focus on in this mode'
                    assert (
                        len(current_part_rowid_list) <= 1
                    ), 'An annotation cannot have more than one canonical part (for now)'

                if is_staged:
                    staged_uuid = uuid.uuid4()
                    staged_user = controller_inject.get_user()
                    if staged_user is None:
                        staged_user = {}
                    staged_user_id = staged_user.get('username', None)

                    # Filter aids for current user
                    current_annot_user_id_list = ibs.get_annot_staged_user_ids(
                        current_aid_list
                    )
                    current_aid_list = [
                        current_aid
                        for current_aid, current_annot_user_id in zip(
                            current_aid_list, current_annot_user_id_list
                        )
                        if current_annot_user_id == staged_user_id
                    ]

                    # Filter part_rowids for current user
                    current_part_user_id_list = ibs.get_part_staged_user_ids(
                        current_part_rowid_list
                    )
                    current_part_rowid_list = [
                        current_part_rowid
                        for current_part_rowid, current_part_user_id in zip(
                            current_part_rowid_list, current_part_user_id_list
                        )
                        if current_part_user_id == staged_user_id
                    ]
                else:
                    staged_uuid = None
                    staged_user = None
                    staged_user_id = None

                # Make new annotations
                width, height = ibs.get_image_sizes(gid)

                if THROW_TEST_AOI_TURKING:
                    # Separate out annotations vs parts
                    raw_manifest = request.form['ia-detection-manifest'].strip()
                    try:
                        manifest_list = ut.from_json(raw_manifest)
                    except ValueError:
                        manifest_list = []
                    test_truth = len(manifest_list) > 0
                    test_challenge_list = [{'gid': gid, 'manifest_list': manifest_list}]
                    test_response_list = [{'poor_boxes': poor_boxes}]
                    test_result_list = [test_truth == poor_boxes]
                    test_user_id_list = [None]
                    ibs.add_test(
                        test_challenge_list,
                        test_response_list,
                        test_result_list=test_result_list,
                        test_user_identity_list=test_user_id_list,
                    )
                else:
                    test_truth = False

                if not test_truth:
                    data_list = ut.from_json(request.form['ia-detection-data'])

                    annotation_list = []
                    part_list = []
                    mapping_dict = {}
                    for outer_index, data in enumerate(data_list):
                        # Check for invalid NaN boxes, filter them out
                        try:
                            assert data['percent']['left'] is not None
                            assert data['percent']['top'] is not None
                            assert data['percent']['width'] is not None
                            assert data['percent']['height'] is not None
                        except Exception:
                            continue

                        parent_index = data['parent']
                        if parent_index is None:
                            inner_index = len(annotation_list)
                            annotation_list.append(data)
                            mapping_dict[outer_index] = inner_index
                        else:
                            assert parent_index in mapping_dict
                            part_list.append(data)

                    ##################################################################################
                    # Get primatives
                    bbox_list = [
                        (
                            int(np.round(width * (annot['percent']['left'] / 100.0))),
                            int(np.round(height * (annot['percent']['top'] / 100.0))),
                            int(np.round(width * (annot['percent']['width'] / 100.0))),
                            int(np.round(height * (annot['percent']['height'] / 100.0))),
                        )
                        for annot in annotation_list
                    ]
                    theta_list = [
                        float(annot['angles']['theta']) for annot in annotation_list
                    ]

                    # Get metadata
                    viewpoint1_list = [
                        int(annot['metadata'].get('viewpoint1', -1))
                        for annot in annotation_list
                    ]
                    viewpoint2_list = [
                        int(annot['metadata'].get('viewpoint2', -1))
                        for annot in annotation_list
                    ]
                    viewpoint3_list = [
                        int(annot['metadata'].get('viewpoint3', -1))
                        for annot in annotation_list
                    ]
                    zipped = zip(viewpoint1_list, viewpoint2_list, viewpoint3_list)
                    viewpoint_list = [
                        appf.convert_tuple_to_viewpoint(tup) for tup in zipped
                    ]

                    quality_list = [
                        int(annot['metadata'].get('quality', 0))
                        for annot in annotation_list
                    ]
                    # Fix qualities
                    for index, quality in enumerate(quality_list):
                        if quality == 0:
                            quality_list[index] = None
                        elif quality == 1:
                            quality_list[index] = 2
                        elif quality == 2:
                            quality_list[index] = 4
                        else:
                            raise ValueError('quality must be 0, 1 or 2')

                    multiple_list = [
                        annot['metadata'].get('multiple', False)
                        for annot in annotation_list
                    ]
                    interest_list = [annot['highlighted'] for annot in annotation_list]
                    species_list = [
                        annot['metadata'].get('species', const.UNKNOWN)
                        for annot in annotation_list
                    ]

                    # Process annotations
                    survived_aid_list = [
                        None if annot['label'] in [None, 'None'] else int(annot['label'])
                        for annot in annotation_list
                    ]

                    # Delete annotations that didn't survive
                    kill_aid_list = list(set(current_aid_list) - set(survived_aid_list))

                    if is_canonical:
                        assert (
                            len(survived_aid_list) == 1
                        ), 'Cannot add or delete annotations in this mode'
                        assert (
                            len(kill_aid_list) == 0
                        ), 'Cannot kill a canonical annotation in this mode'

                    ibs.delete_annots(kill_aid_list)

                    local_add_gid_list = []
                    local_add_bbox_list = []
                    local_add_theta_list = []
                    local_add_interest_list = []
                    local_add_viewpoint_list = []
                    local_add_quality_list = []
                    local_add_multiple_list = []
                    local_add_species_list = []
                    local_add_staged_uuid_list = []
                    local_add_staged_user_id_list = []

                    local_update_aid_list = []
                    local_update_bbox_list = []
                    local_update_theta_list = []
                    local_update_interest_list = []
                    local_update_viewpoint_list = []
                    local_update_quality_list = []
                    local_update_multiple_list = []
                    local_update_species_list = []
                    local_update_staged_uuid_list = []
                    local_update_staged_user_id_list = []

                    zipped = zip(
                        survived_aid_list,
                        bbox_list,
                        theta_list,
                        interest_list,
                        viewpoint_list,
                        quality_list,
                        multiple_list,
                        species_list,
                    )

                    survived_flag_list = []

                    for values in zipped:
                        (
                            aid,
                            bbox,
                            theta,
                            interest,
                            viewpoint,
                            quality,
                            multiple,
                            species,
                        ) = values
                        flag = aid is None
                        survived_flag_list.append(flag)
                        if flag:
                            local_add_gid_list.append(gid)
                            local_add_bbox_list.append(bbox)
                            local_add_theta_list.append(theta)
                            local_add_interest_list.append(interest)
                            local_add_viewpoint_list.append(viewpoint)
                            local_add_quality_list.append(quality)
                            local_add_multiple_list.append(multiple)
                            local_add_species_list.append(species)
                            local_add_staged_uuid_list.append(staged_uuid)
                            local_add_staged_user_id_list.append(staged_user_id)
                        else:
                            local_update_aid_list.append(aid)
                            local_update_bbox_list.append(bbox)
                            local_update_theta_list.append(theta)
                            local_update_interest_list.append(interest)
                            local_update_viewpoint_list.append(viewpoint)
                            local_update_quality_list.append(quality)
                            local_update_multiple_list.append(multiple)
                            local_update_species_list.append(species)
                            local_update_staged_uuid_list.append(staged_uuid)
                            local_update_staged_user_id_list.append(staged_user_id)

                    if len(local_add_gid_list) > 0:
                        add_aid_list = ibs.add_annots(
                            local_add_gid_list,
                            bbox_list=local_add_bbox_list,
                            theta_list=local_add_theta_list,
                            interest_list=local_add_interest_list,
                            viewpoint_list=local_add_viewpoint_list,
                            quality_list=local_add_quality_list,
                            multiple_list=local_add_multiple_list,
                            species_list=local_add_species_list,
                            staged_uuid_list=local_add_staged_uuid_list,
                            staged_user_id_list=local_add_staged_user_id_list,
                            delete_thumb=False,
                        )
                    else:
                        add_aid_list = []

                    if len(local_update_aid_list) > 0:
                        with ut.Timer('submit...update...updating'):
                            ibs.set_annot_bboxes(
                                local_update_aid_list,
                                local_update_bbox_list,
                                theta_list=local_update_theta_list,
                                interest_list=local_update_interest_list,
                            )

                        with ut.Timer('submit...update...metadata0'):
                            ibs.set_annot_viewpoints(
                                local_update_aid_list, local_update_viewpoint_list
                            )
                        with ut.Timer('submit...update...metadata1'):
                            ibs.set_annot_qualities(
                                local_update_aid_list, local_update_quality_list
                            )
                        with ut.Timer('submit...update...metadata2'):
                            ibs.set_annot_multiple(
                                local_update_aid_list, local_update_multiple_list
                            )
                        with ut.Timer('submit...update...metadata3'):
                            ibs.set_annot_species(
                                local_update_aid_list, local_update_species_list
                            )
                        with ut.Timer('submit...update...metadata4'):
                            ibs.set_annot_staged_uuids(
                                local_update_aid_list, local_update_staged_uuid_list
                            )
                        with ut.Timer('submit...update...metadata5'):
                            ibs.set_annot_staged_user_ids(
                                local_update_aid_list, local_update_staged_user_id_list
                            )

                    # Set the mapping dict to use aids now
                    aid_list = []
                    counter_add = 0
                    counter_update = 0
                    for flag in survived_flag_list:
                        if flag:
                            aid_list.append(add_aid_list[counter_add])
                            counter_add += 1
                        else:
                            aid_list.append(local_update_aid_list[counter_update])
                            counter_update += 1

                    assert counter_add == len(add_aid_list)
                    assert counter_update == len(local_update_aid_list)
                    assert counter_add + counter_update == len(survived_aid_list)

                    mapping_dict = {
                        key: aid_list[index] for key, index in mapping_dict.items()
                    }

                    ##################################################################################
                    # Process parts
                    survived_part_rowid_list = [
                        None if part['label'] is None else int(part['label'])
                        for part in part_list
                    ]

                    # Get primatives
                    aid_list = [mapping_dict[part['parent']] for part in part_list]
                    bbox_list = [
                        (
                            int(np.round(width * (part['percent']['left'] / 100.0))),
                            int(np.round(height * (part['percent']['top'] / 100.0))),
                            int(np.round(width * (part['percent']['width'] / 100.0))),
                            int(np.round(height * (part['percent']['height'] / 100.0))),
                        )
                        for part in part_list
                    ]
                    theta_list = [float(part['angles']['theta']) for part in part_list]

                    # Get metadata
                    viewpoint1_list = [
                        int(part['metadata'].get('viewpoint1', -1)) for part in part_list
                    ]
                    viewpoint2_list = [-1] * len(part_list)
                    viewpoint3_list = [-1] * len(part_list)
                    zipped = zip(viewpoint1_list, viewpoint2_list, viewpoint3_list)
                    viewpoint_list = [
                        appf.convert_tuple_to_viewpoint(tup) for tup in zipped
                    ]

                    quality_list = [
                        int(part['metadata'].get('quality', 0)) for part in part_list
                    ]
                    # Fix qualities
                    for index, quality in enumerate(quality_list):
                        if quality == 0:
                            quality_list[index] = None
                        elif quality == 1:
                            quality_list[index] = 2
                        elif quality == 2:
                            quality_list[index] = 4
                        else:
                            raise ValueError('quality must be 0, 1 or 2')

                    type_list = [
                        part['metadata'].get('type', const.UNKNOWN) for part in part_list
                    ]

                    # Delete annotations that didn't survive
                    kill_part_rowid_list = list(
                        set(current_part_rowid_list) - set(survived_part_rowid_list)
                    )

                    if is_canonical:
                        assert (
                            len(survived_part_rowid_list) <= 1
                        ), 'Cannot add more than one canonical part in this mode'
                        assert (
                            len(kill_part_rowid_list) <= 1
                        ), 'Cannot delete two or more canonical parts in this mode'
                        assert len(survived_part_rowid_list) == len(type_list)
                        type_list = [appf.CANONICAL_PART_TYPE] * len(
                            survived_part_rowid_list
                        )

                    ibs.delete_parts(kill_part_rowid_list)

                    staged_uuid_list = [staged_uuid] * len(survived_part_rowid_list)
                    staged_user_id_list = [staged_user_id] * len(survived_part_rowid_list)

                    part_rowid_list = []
                    zipped = zip(
                        survived_part_rowid_list,
                        aid_list,
                        bbox_list,
                        staged_uuid_list,
                        staged_user_id_list,
                    )
                    for part_rowid, aid, bbox, staged_uuid, staged_user_id in zipped:
                        staged_uuid_list_ = None if staged_uuid is None else [staged_uuid]
                        staged_user_id_list_ = (
                            None if staged_user_id is None else [staged_user_id]
                        )

                        if part_rowid is None:
                            part_rowid_ = ibs.add_parts(
                                [aid],
                                [bbox],
                                staged_uuid_list=staged_uuid_list_,
                                staged_user_id_list=staged_user_id_list_,
                            )
                            part_rowid_ = part_rowid_[0]
                        else:
                            ibs._set_part_aid([part_rowid], [aid])
                            ibs.set_part_bboxes([part_rowid], [bbox])
                            if staged_uuid_list_ is not None:
                                ibs.set_part_staged_uuids([part_rowid], staged_uuid_list_)
                            if staged_user_id_list_ is not None:
                                ibs.set_part_staged_user_ids(
                                    [part_rowid], staged_user_id_list_
                                )

                            part_rowid_ = part_rowid
                        part_rowid_list.append(part_rowid_)

                    # Set part metadata
                    print('part_rowid_list = %r' % (part_rowid_list,))
                    ibs.set_part_thetas(part_rowid_list, theta_list)
                    ibs.set_part_viewpoints(part_rowid_list, viewpoint_list)
                    ibs.set_part_qualities(part_rowid_list, quality_list)
                    ibs.set_part_types(part_rowid_list, type_list)

                    ##################################################################################
                    # Process image

                    if is_staged:
                        # Set image reviewed flag
                        metadata_dict = ibs.get_image_metadata(gid)
                        if 'staged' not in metadata_dict:
                            metadata_dict['staged'] = {
                                'sessions': {'uuids': [], 'user_ids': []}
                            }
                        metadata_dict['staged']['sessions']['uuids'].append(
                            str(staged_uuid)
                        )
                        metadata_dict['staged']['sessions']['user_ids'].append(
                            staged_user_id
                        )
                        ibs.set_image_metadata([gid], [metadata_dict])
                    else:
                        ibs.set_image_reviewed([gid], [1])

                        print(
                            '[web] user_id: %s, gid: %d, annots: %d, parts: %d'
                            % (user_id, gid, len(annotation_list), len(part_list),)
                        )

        default_list = [
            'autointerest',
            'interest_bypass',
            'metadata',
            'metadata_viewpoint',
            'metadata_quality',
            'metadata_flags',
            'metadata_flags_aoi',
            'metadata_flags_multiple',
            'metadata_species',
            'metadata_label',
            'metadata_quickhelp',
            'parts',
            'modes_rectangle',
            'modes_diagonal',
            'modes_diagonal2',
            'staged',
            'canonical',
        ]
        config = {
            default: kwargs[default] for default in default_list if default in kwargs
        }

        # Return HTML
        refer = request.args.get('refer', '')
        if len(refer) > 0:
            return redirect(appf.decode_refer_url(refer))
        else:
            signature = 'turk_detection_canonical' if is_canonical else 'turk_detection'
            return redirect(
                url_for(
                    signature,
                    imgsetid=imgsetid,
                    previous=gid,
                    previous_only_aid=only_aid,
                    **config,
                )
            )


@register_route('/submit/viewpoint/', methods=['POST'])
def submit_viewpoint(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('viewpoint-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    src_ag = request.args.get('src_ag', '')
    src_ag = None if src_ag == 'None' or src_ag == '' else int(src_ag)
    dst_ag = request.args.get('dst_ag', '')
    dst_ag = None if dst_ag == 'None' or dst_ag == '' else int(dst_ag)

    aid = int(request.form['viewpoint-aid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)
    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) user_id: %s, aid: %d' % (user_id, aid,))
        aid = None  # Reset AID to prevent previous
    if method.lower() == 'make junk':
        ibs.set_annot_quality_texts([aid], [const.QUAL_JUNK])
        print('[web] (SET AS JUNK) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    if method.lower() == 'rotate left':
        ibs.update_annot_rotate_left_90([aid])
        print('[web] (ROTATED LEFT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    if method.lower() == 'rotate right':
        ibs.update_annot_rotate_right_90([aid])
        print('[web] (ROTATED RIGHT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)
        viewpoint = int(request.form['viewpoint-value'])
        viewpoint_text = appf.VIEWPOINT_MAPPING.get(viewpoint, None)
        species_text = request.form['viewpoint-species']
        ibs.set_annot_viewpoints([aid], [viewpoint_text])
        # TODO ibs.set_annot_viewpoint_code([aid], [viewpoint_text])
        ibs.set_annot_species([aid], [species_text])
        print(
            '[web] user_id: %s, aid: %d, viewpoint_text: %s'
            % (user_id, aid, viewpoint_text)
        )
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for(
                'turk_viewpoint',
                imgsetid=imgsetid,
                src_ag=src_ag,
                dst_ag=dst_ag,
                previous=aid,
            )
        )


@register_route('/submit/viewpoint2/', methods=['POST'])
def submit_viewpoint2(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('viewpoint-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    src_ag = request.args.get('src_ag', '')
    src_ag = None if src_ag == 'None' or src_ag == '' else int(src_ag)
    dst_ag = request.args.get('dst_ag', '')
    dst_ag = None if dst_ag == 'None' or dst_ag == '' else int(dst_ag)

    aid = int(request.form['viewpoint-aid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)
    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) user_id: %s, aid: %d' % (user_id, aid,))
        aid = None  # Reset AID to prevent previous
    if method.lower() == 'make junk':
        ibs.set_annot_quality_texts([aid], [const.QUAL_JUNK])
        print('[web] (SET AS JUNK) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    if method.lower() == 'rotate left':
        ibs.update_annot_rotate_left_90([aid])
        print('[web] (ROTATED LEFT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    if method.lower() == 'rotate right':
        ibs.update_annot_rotate_right_90([aid])
        print('[web] (ROTATED RIGHT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)
        if method.lower() == 'ignore':
            viewpoint = 'unknown'
        else:
            # Get metadata
            viewpoint1 = int(kwargs.get('ia-viewpoint-value-1', None))
            viewpoint2 = int(kwargs.get('ia-viewpoint-value-2', None))
            viewpoint3 = int(kwargs.get('ia-viewpoint-value-3', None))
            viewpoint_tup = (
                viewpoint1,
                viewpoint2,
                viewpoint3,
            )
            viewpoint = appf.convert_tuple_to_viewpoint(viewpoint_tup)
        ibs.set_annot_viewpoints([aid], [viewpoint])
        species_text = request.form['viewpoint-species']
        # TODO ibs.set_annot_viewpoint_code([aid], [viewpoint_text])
        ibs.set_annot_species([aid], [species_text])
        print(
            '[web] user_id: %s, aid: %d, viewpoint_text: %s' % (user_id, aid, viewpoint)
        )
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for(
                'turk_viewpoint2',
                imgsetid=imgsetid,
                src_ag=src_ag,
                dst_ag=dst_ag,
                previous=aid,
            )
        )


@register_route('/submit/viewpoint3/', methods=['POST'])
def submit_viewpoint3(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('viewpoint-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    src_ag = request.args.get('src_ag', '')
    src_ag = None if src_ag == 'None' or src_ag == '' else int(src_ag)
    dst_ag = request.args.get('dst_ag', '')
    dst_ag = None if dst_ag == 'None' or dst_ag == '' else int(dst_ag)

    aid = int(request.form['viewpoint-aid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)

    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) user_id: %s, aid: %d' % (user_id, aid,))
        aid = None  # Reset AID to prevent previous
    if method.lower() == 'make junk':
        ibs.set_annot_quality_texts([aid], [const.QUAL_JUNK])
        print('[web] (SET AS JUNK) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    if method.lower() == 'rotate left':
        ibs.update_annot_rotate_left_90([aid])
        print('[web] (ROTATED LEFT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    if method.lower() == 'rotate right':
        ibs.update_annot_rotate_right_90([aid])
        print('[web] (ROTATED RIGHT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)

        if method.lower() == 'ignore':
            ibs.set_annot_viewpoints(
                [aid], ['ignore'], only_allow_known=False, _code_update=False
            )
            viewpoint = 'ignore'
        else:
            # Get metadata
            viewpoint_str = kwargs.get('viewpoint-text-code', '')
            if not isinstance(viewpoint_str, six.string_types) or len(viewpoint_str) == 0:
                viewpoint_str = None

            if viewpoint_str is None:
                viewpoint = const.VIEW.UNKNOWN
            else:
                viewpoint = getattr(const.VIEW, viewpoint_str, const.VIEW.UNKNOWN)
            ibs.set_annot_viewpoint_int([aid], [viewpoint])

        species_text = request.form['viewpoint-species']
        ibs.set_annot_species([aid], [species_text])
        ibs.set_annot_reviewed([aid], [1])
        print('[web] user_id: %s, aid: %d, viewpoint: %s' % (user_id, aid, viewpoint))

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        retval = redirect(appf.decode_refer_url(refer))
    else:
        retval = redirect(
            url_for(
                'turk_viewpoint3',
                imgsetid=imgsetid,
                src_ag=src_ag,
                dst_ag=dst_ag,
                previous=aid,
            )
        )

    return retval


@register_route('/submit/annotation/', methods=['POST'])
def submit_annotation(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('ia-annotation-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    src_ag = request.args.get('src_ag', '')
    src_ag = None if src_ag == 'None' or src_ag == '' else int(src_ag)
    dst_ag = request.args.get('dst_ag', '')
    dst_ag = None if dst_ag == 'None' or dst_ag == '' else int(dst_ag)

    aid = int(request.form['ia-annotation-aid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)
    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) user_id: %s, aid: %d' % (user_id, aid,))
        aid = None  # Reset AID to prevent previous
    elif method.lower() == 'make junk':
        ibs.set_annot_quality_texts([aid], [const.QUAL_JUNK])
        print('[web] (SET AS JUNK) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    elif method.lower() == 'rotate left':
        ibs.update_annot_rotate_left_90([aid])
        print('[web] (ROTATED LEFT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    elif method.lower() == 'rotate right':
        ibs.update_annot_rotate_right_90([aid])
        print('[web] (ROTATED RIGHT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)
        try:
            viewpoint = int(request.form['ia-annotation-viewpoint-value'])
        except ValueError:
            viewpoint = int(float(request.form['ia-annotation-viewpoint-value']))
        viewpoint_text = appf.VIEWPOINT_MAPPING.get(viewpoint, None)
        species_text = request.form['ia-annotation-species']
        try:
            quality = int(request.form['ia-quality-value'])
        except ValueError:
            quality = int(float(request.form['ia-quality-value']))
        if quality in [-1, None]:
            quality = None
        elif quality == 0:
            quality = 2
        elif quality == 1:
            quality = 4
        else:
            raise ValueError('quality must be -1, 0 or 1')
        ibs.set_annot_viewpoints([aid], [viewpoint_text])
        # TODO ibs.set_annot_viewpoint_code([aid], [viewpoint_text])
        ibs.set_annot_species([aid], [species_text])
        ibs.set_annot_qualities([aid], [quality])
        multiple = 1 if 'ia-multiple-value' in request.form else 0
        ibs.set_annot_multiple([aid], [multiple])
        ibs.set_annot_reviewed([aid], [1])
        print(
            '[web] user_id: %s, aid: %d, viewpoint: %r, quality: %r, multiple: %r'
            % (user_id, aid, viewpoint_text, quality, multiple)
        )
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for(
                'turk_annotation',
                imgsetid=imgsetid,
                src_ag=src_ag,
                dst_ag=dst_ag,
                previous=aid,
            )
        )


@register_route('/submit/annotation/canonical/', methods=['POST'])
def submit_annotation_canonical(samples=200, species=None, version=1, **kwargs):
    ibs = current_app.ibs

    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    assert version in [1, 2, 3]

    aid_list = kwargs['annotation-canonical-aids']
    canonical_list = kwargs['annotation-canonical-highlighted']
    assert len(aid_list) == len(canonical_list)

    # metadata_list = ibs.get_annot_metadata(aid_list)
    # metadata_list_ = []
    # for metadata, highlight in zip(metadata_list, highlight_list):
    #     if 'turk' not in metadata:
    #         metadata['turk'] = {}

    #     if version == 1:
    #         value = highlight
    #     elif version == 2:
    #         value = not highlight
    #     elif version == 3:
    #         value = highlight

    #     metadata['turk']['canonical'] = value
    #     metadata_list_.append(metadata)

    # ibs.set_annot_metadata(aid_list, metadata_list_)

    value_list = []
    for canonical in canonical_list:
        if version == 1:
            value = canonical
        elif version == 2:
            value = not canonical
        elif version == 3:
            value = canonical
        value_list.append(value)

    ibs.set_annot_canonical(aid_list, value_list)

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for(
                'turk_annotation_canonical',
                imgsetid=imgsetid,
                samples=samples,
                species=species,
                version=version,
            )
        )


@register_route('/submit/splits/', methods=['POST'])
def submit_splits(**kwargs):
    # ibs = current_app.ibs

    aid_list = kwargs['annotation-splits-aids']
    highlight_list = kwargs['annotation-splits-highlighted']
    assert len(aid_list) == len(highlight_list)

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_splits', aid=None))


@register_route('/submit/species/', methods=['POST'])
def submit_species(**kwargs):
    ibs = current_app.ibs

    method = request.form.get('ia-species-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    previous_species_rowids = request.form.get('ia-species-rowids', None)
    print('Using previous_species_rowids = %r' % (previous_species_rowids,))

    src_ag = request.args.get('src_ag', '')
    src_ag = None if src_ag == 'None' or src_ag == '' else int(src_ag)
    dst_ag = request.args.get('dst_ag', '')
    dst_ag = None if dst_ag == 'None' or dst_ag == '' else int(dst_ag)

    aid = int(request.form['ia-species-aid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)

    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) user_id: %s, aid: %d' % (user_id, aid,))
        aid = None  # Reset AID to prevent previous
    elif method.lower() == 'skip':
        species_text = const.UNKNOWN
        ibs.set_annot_species([aid], [species_text])
        ibs.set_annot_reviewed([aid], [1])
        print('[web] (SKIP) user_id: %s' % (user_id,))
        return redirect(
            url_for(
                'turk_species',
                imgsetid=imgsetid,
                src_ag=src_ag,
                dst_ag=dst_ag,
                previous=aid,
                previous_species_rowids=previous_species_rowids,
            )
        )
    elif method.lower() in 'refresh':
        print('[web] (REFRESH) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        if '?' in redirection:
            redirection = '%s&refresh=true' % (redirection,)
        else:
            redirection = '%s?refresh=true' % (redirection,)
        return redirect(redirection)
    elif method.lower() == 'rotate left':
        ibs.update_annot_rotate_left_90([aid])
        print('[web] (ROTATED LEFT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    elif method.lower() == 'rotate right':
        ibs.update_annot_rotate_right_90([aid])
        print('[web] (ROTATED RIGHT) user_id: %s, aid: %d' % (user_id, aid,))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid,)
            else:
                redirection = '%s?aid=%d' % (redirection, aid,)
        return redirect(redirection)
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)
        # species_text = request.form['ia-species-species']
        species_text = kwargs.get('ia-species-value', '')
        if len(species_text) == 0:
            species_text = const.UNKNOWN
        ibs.set_annot_species([aid], [species_text])
        ibs.set_annot_reviewed([aid], [1])

        metadata_dict = ibs.get_annot_metadata(aid)
        if 'turk' not in metadata_dict:
            metadata_dict['turk'] = {}
        metadata_dict['turk']['species'] = user_id
        ibs.set_annot_metadata([aid], [metadata_dict])

        print(
            '[web] user_id: %s, aid: %d, species: %r (%r)'
            % (user_id, aid, species_text, metadata_dict,)
        )
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for(
                'turk_species',
                imgsetid=imgsetid,
                src_ag=src_ag,
                dst_ag=dst_ag,
                previous=aid,
                previous_species_rowids=previous_species_rowids,
            )
        )


@register_route('/submit/part/type/', methods=['POST'])
def submit_part_types(**kwargs):
    ibs = current_app.ibs

    import utool as ut

    ut.embed()

    method = request.form.get('ia-part-type-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    # IF DECISION NOT IN previous_part_types, REFRESH = TRUE
    previous_part_types = request.form.get('ia-part-types', None)
    print('Using previous_part_types = %r' % (previous_part_types,))

    part_rowid = int(request.form['ia-part-type-part-rowid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)

    refresh = False
    if method.lower() in 'refresh':
        print('[web] (REFRESH) user_id: %s, part_rowid: %d' % (user_id, part_rowid,))
        redirection = request.referrer
        if 'part_rowid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&part_rowid=%d' % (redirection, part_rowid,)
            else:
                redirection = '%s?part_rowid=%d' % (redirection, part_rowid,)
        if '?' in redirection:
            redirection = '%s&refresh=true' % (redirection,)
        else:
            redirection = '%s?refresh=true' % (redirection,)
        return redirect(redirection)
    elif method.lower() == 'rotate left':
        ibs.update_part_rotate_left_90([part_rowid])
        print('[web] (ROTATED LEFT) user_id: %s, part_rowid: %d' % (user_id, part_rowid,))
        redirection = request.referrer
        if 'part_rowid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&part_rowid=%d' % (redirection, part_rowid,)
            else:
                redirection = '%s?part_rowid=%d' % (redirection, part_rowid,)
        return redirect(redirection)
    elif method.lower() == 'rotate right':
        ibs.update_part_rotate_right_90([part_rowid])
        print(
            '[web] (ROTATED RIGHT) user_id: %s, part_rowid: %d' % (user_id, part_rowid,)
        )
        redirection = request.referrer
        if 'part_rowid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&part_rowid=%d' % (redirection, part_rowid,)
            else:
                redirection = '%s?part_rowid=%d' % (redirection, part_rowid,)
        return redirect(redirection)
    else:
        part_type_text = kwargs.get('ia-part-type-value', '')
        ibs.set_part_types([part_rowid], [part_type_text])
        ibs.set_part_reviewed([part_rowid], [1])

        refresh = part_type_text not in previous_part_types
        print(
            '[web] user_id: %s, part_rowid: %d, type: %r'
            % (user_id, part_rowid, part_type_text,)
        )
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for(
                'turk_part_types',
                imgsetid=imgsetid,
                previous=part_rowid,
                previous_part_types=previous_part_types,
                refresh=refresh,
            )
        )


@register_route('/submit/quality/', methods=['POST'])
def submit_quality(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('quality-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    aid = int(request.form['quality-aid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)

    src_ag = request.args.get('src_ag', '')
    src_ag = None if src_ag == 'None' or src_ag == '' else int(src_ag)
    dst_ag = request.args.get('dst_ag', '')
    dst_ag = None if dst_ag == 'None' or dst_ag == '' else int(dst_ag)

    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) user_id: %s, aid: %d' % (user_id, aid,))
        aid = None  # Reset AID to prevent previous
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)
        quality = int(request.form['quality-value'])
        ibs.set_annot_qualities([aid], [quality])
        print('[web] user_id: %s, aid: %d, quality: %d' % (user_id, aid, quality))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for(
                'turk_quality',
                imgsetid=imgsetid,
                src_ag=src_ag,
                dst_ag=dst_ag,
                previous=aid,
            )
        )


@register_route('/submit/demographics/', methods=['POST'])
def submit_demographics(species='zebra_grevys', **kwargs):
    ibs = current_app.ibs

    GGR_UPDATE_AGE_FOR_ALL_NAMED_ANOTATIONS = False
    if ibs.dbname in ['GGR-IBEIS', 'GGR2-IBEIS']:
        GGR_UPDATE_AGE_FOR_ALL_NAMED_ANOTATIONS = True

    method = request.form.get('demographics-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    aid = int(request.form['demographics-aid'])
    user = controller_inject.get_user()
    if user is None:
        user = {}
    user_id = user.get('username', None)

    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) user_id: %s, aid: %d' % (user_id, aid,))
        aid = None  # Reset AID to prevent previous
    else:
        sex = int(request.form['demographics-sex-value'])
        age = int(request.form['demographics-age-value'])
        age_min = None
        age_max = None
        # Sex
        if sex >= 2:
            sex -= 2
        else:
            sex = -1

        if age == 1:
            age_min = None
            age_max = None
        elif age == 2:
            age_min = None
            age_max = 2
        elif age == 3:
            age_min = 3
            age_max = 5
        elif age == 4:
            age_min = 6
            age_max = 11
        elif age == 5:
            age_min = 12
            age_max = 23
        elif age == 6:
            age_min = 24
            age_max = 35
        elif age == 7:
            age_min = 36
            age_max = None

        ibs.set_annot_sex([aid], [sex])
        nid = ibs.get_annot_name_rowids(aid)
        if nid is not None and GGR_UPDATE_AGE_FOR_ALL_NAMED_ANOTATIONS:
            aid_list = ibs.get_name_aids(nid)
        else:
            aid_list = [aid]

        ibs.set_annot_age_months_est_min(aid_list, [age_min] * len(aid_list))
        ibs.set_annot_age_months_est_max(aid_list, [age_max] * len(aid_list))
        print(
            '[web] Updating %d demographics with user_id: %s\n\taid_list : %r\n\tsex: %r\n\tage_min: %r\n\tage_max: %r'
            % (len(aid_list), user_id, aid_list, sex, age_min, age_max,)
        )
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for('turk_demographics', imgsetid=imgsetid, previous=aid, species=species)
        )


@register_route('/submit/identification/', methods=['POST'])
def submit_identification(**kwargs):
    from wbia.web.apis_query import process_graph_match_html

    ibs = current_app.ibs

    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    aid1 = int(request.form['identification-aid1'])
    aid2 = int(request.form['identification-aid2'])
    replace_review_rowid = int(
        request.form.get('identification-replace-review-rowid', -1)
    )

    # Process form data
    annot_uuid_1, annot_uuid_2, state, tag_list = process_graph_match_html(ibs)

    # Add state to staging database
    # FIXME:
    # photobomb and scenerymatch tags should be disjoint from match-state
    if state == 'matched':
        decision = const.EVIDENCE_DECISION.POSITIVE
    elif state == 'notmatched':
        decision = const.EVIDENCE_DECISION.NEGATIVE
    elif state == 'notcomparable':
        decision = const.EVIDENCE_DECISION.INCOMPARABLE
    elif state == 'photobomb':
        decision = const.EVIDENCE_DECISION.NEGATIVE
        tag_list = ['photobomb']
    elif state == 'scenerymatch':
        decision = const.EVIDENCE_DECISION.NEGATIVE
        tag_list = ['scenerymatch']
    else:
        raise ValueError()

    # Replace a previous decision
    if replace_review_rowid > 0:
        print('REPLACING OLD EVIDENCE_DECISION ID = %r' % (replace_review_rowid,))
        ibs.delete_review([replace_review_rowid])

    # Add a new review row for the new decision (possibly replacing the old one)
    print('ADDING EVIDENCE_DECISION: %r %r %r %r' % (aid1, aid2, decision, tag_list,))
    tags_list = None if tag_list is None else [tag_list]
    review_rowid = ibs.add_review([aid1], [aid2], [decision], tags_list=tags_list)
    review_rowid = review_rowid[0]
    previous = '%s;%s;%s' % (aid1, aid2, review_rowid,)

    # Notify any attached web QUERY_OBJECT
    try:
        state = const.EVIDENCE_DECISION.INT_TO_CODE[decision]
        feedback = (aid1, aid2, state, tags_list)
        print('Adding %r to QUERY_OBJECT_FEEDBACK_BUFFER' % (feedback,))
        current_app.QUERY_OBJECT_FEEDBACK_BUFFER.append(feedback)
    except ValueError:
        pass

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for('turk_identification', imgsetid=imgsetid, previous=previous)
        )


@register_route('/submit/identification/v2/', methods=['POST'])
def submit_identification_v2(graph_uuid, **kwargs):
    ibs = current_app.ibs

    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    # Process form data
    annot_uuid_1, annot_uuid_2 = ibs.process_graph_match_html_v2(graph_uuid, **kwargs)
    aid1 = ibs.get_annot_aids_from_uuid(annot_uuid_1)
    aid2 = ibs.get_annot_aids_from_uuid(annot_uuid_2)

    hogwild = kwargs.get('identification-hogwild', False)
    hogwild_species = kwargs.get('identification-hogwild-species', None)
    hogwild_species = (
        None if hogwild_species == 'None' or hogwild_species == '' else hogwild_species
    )
    print('Using hogwild: %r' % (hogwild,))

    previous = '%s;%s;-1' % (aid1, aid2,)

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        base = url_for('turk_identification_graph')
        sep = '&' if '?' in base else '?'
        args = (
            base,
            sep,
            ut.to_json(graph_uuid),
            previous,
            hogwild,
            hogwild_species,
        )
        url = '%s%sgraph_uuid=%s&previous=%s&hogwild=%s&hogwild_species=%s' % args
        url = url.replace(': ', ':')
        return redirect(url)


@register_route('/submit/identification/v2/kaia/', methods=['POST'])
def submit_identification_v2_kaia(graph_uuid, **kwargs):
    ibs = current_app.ibs

    # Process form data
    annot_uuid_1, annot_uuid_2 = ibs.process_graph_match_html_v2(graph_uuid, **kwargs)
    aid1 = ibs.get_annot_aids_from_uuid(annot_uuid_1)
    aid2 = ibs.get_annot_aids_from_uuid(annot_uuid_2)

    age1 = kwargs.get('age-annot-1', None)
    age2 = kwargs.get('age-annot-2', None)
    # sex1 = kwargs.get('sex-annot-1', None)
    # sex2 = kwargs.get('sex-annot-2', None)
    condition1 = kwargs.get('condition-annot-1', None)
    condition2 = kwargs.get('condition-annot-2', None)
    comment1 = kwargs.get('comment-annot-1', None)
    comment2 = kwargs.get('comment-annot-2', None)
    comment_match = kwargs.get('comment-match', None)

    assert age1 in ['age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'unknown']
    assert age2 in ['age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'unknown']
    # assert sex1 in ['male', 'female', 'unknown']
    # assert sex2 in ['male', 'female', 'unknown']
    assert 0 <= condition1 and condition1 <= 5
    assert 0 <= condition2 and condition2 <= 5

    # if sex1 == 'male':
    #     sex1 = 1
    # elif sex1 == 'female':
    #     sex1 = 0
    # else:
    #     sex1 = -1

    # if sex2 == 'male':
    #     sex2 = 1
    # elif sex2 == 'female':
    #     sex2 = 0
    # else:
    #     sex2 = -1

    age1_min = None
    age1_max = None
    if age1 == 'age1':
        age1_min = None
        age1_max = 2
    elif age1 == 'age2':
        age1_min = 3
        age1_max = 5
    elif age1 == 'age3':
        age1_min = 6
        age1_max = 11
    elif age1 == 'age4':
        age1_min = 12
        age1_max = 23
    elif age1 == 'age5':
        age1_min = 24
        age1_max = 35
    elif age1 == 'age6':
        age1_min = 36
        age1_max = None
    elif age1 == 'unknown':
        age1_min = None
        age1_max = None
    else:
        raise ValueError()

    age2_min = None
    age2_max = None
    if age2 == 'age1':
        age2_min = None
        age2_max = 2
    elif age2 == 'age2':
        age2_min = 3
        age2_max = 5
    elif age2 == 'age3':
        age2_min = 6
        age2_max = 11
    elif age2 == 'age4':
        age2_min = 12
        age2_max = 23
    elif age2 == 'age5':
        age2_min = 24
        age2_max = 35
    elif age2 == 'age6':
        age2_min = 36
        age2_max = None
    elif age2 == 'unknown':
        age2_min = None
        age2_max = None
    else:
        raise ValueError()

    if condition1 in [0]:
        condition1 = None
    if condition2 in [0]:
        condition2 = None

    # ibs.set_annot_sex([aid1], [sex1])
    # ibs.set_annot_sex([aid2], [sex2])
    ibs.set_annot_age_months_est_min([aid1, aid2], [age1_min, age2_min])
    ibs.set_annot_age_months_est_max([aid1, aid2], [age1_max, age2_max])
    ibs.set_annot_qualities([aid1, aid2], [condition1, condition2])

    metadata1, metadata2 = ibs.get_annot_metadata([aid1, aid2])
    if 'turk' not in metadata1:
        metadata1['turk'] = {}
    if 'turk' not in metadata2:
        metadata2['turk'] = {}
    if 'match' not in metadata1['turk']:
        metadata1['turk']['match'] = {}
    if 'match' not in metadata2['turk']:
        metadata2['turk']['match'] = {}
    metadata1['turk']['match']['comment'] = comment1
    metadata2['turk']['match']['comment'] = comment2
    ibs.set_annot_metadata([aid1, aid2], [metadata1, metadata2])

    edge = (
        aid1,
        aid2,
    )
    review_rowid_list = ibs.get_review_rowids_from_edges([edge])[0]
    if len(review_rowid_list) > 0:
        review_rowid = review_rowid_list[-1]
        metadata_match = ibs.get_review_metadata(review_rowid)
        if 'turk' not in metadata_match:
            metadata_match['turk'] = {}
        if 'match' not in metadata_match['turk']:
            metadata_match['turk']['match'] = {}
        existing_comment = metadata_match.get('comment', '')
        updated_comment = '\n'.join([comment_match, existing_comment])
        updated_comment = updated_comment.strip()
        metadata_match['turk']['match']['comment'] = updated_comment
        ibs.set_review_metadata([review_rowid], [metadata_match])

    hogwild = kwargs.get('identification-hogwild', False)
    hogwild_species = kwargs.get('identification-hogwild-species', None)
    hogwild_species = (
        None if hogwild_species == 'None' or hogwild_species == '' else hogwild_species
    )
    print('Using hogwild: %r' % (hogwild,))

    previous = '%s;%s;-1' % (aid1, aid2,)

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        base = url_for('turk_identification_graph')
        sep = '&' if '?' in base else '?'
        args = (
            base,
            sep,
            ut.to_json(graph_uuid),
            previous,
            hogwild,
            hogwild_species,
        )
        url = (
            '%s%sgraph_uuid=%s&previous=%s&hogwild=%s&hogwild_species=%s&kaia=true' % args
        )
        url = url.replace(': ', ':')
        return redirect(url)


@register_route('/submit/group_review/', methods=['POST'])
def group_review_submit(**kwargs):
    """
    CommandLine:
        python -m wbia.web.app --exec-group_review_submit

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> import wbia.web
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::2]
        >>> ibs.start_web_annot_groupreview(aid_list)
    """
    ibs = current_app.ibs
    method = request.form.get('group-review-submit', '')
    if method.lower() == 'populate':
        redirection = request.referrer
        if 'prefill' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&prefill=true' % (redirection,)
            else:
                redirection = '%s?prefill=true' % (redirection,)
        return redirect(redirection)
    aid_list = request.form.get('aid_list', '')
    if len(aid_list) > 0:
        aid_list = aid_list.replace('[', '')
        aid_list = aid_list.replace(']', '')
        aid_list = aid_list.strip().split(',')
        aid_list = [int(aid_.strip()) for aid_ in aid_list]
    else:
        aid_list = []
    src_ag, dst_ag = ibs.prepare_annotgroup_review(aid_list)
    valid_modes = ut.get_list_column(appf.VALID_TURK_MODES, 0)
    mode = request.form.get('group-review-mode', None)
    assert mode in valid_modes
    return redirect(url_for(mode, src_ag=src_ag, dst_ag=dst_ag))


@register_route('/submit/part/contour/', methods=['POST'])
def submit_contour(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('contour-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    part_rowid = int(request.form['contour-part-rowid'])

    if method.lower() == 'accept':
        data_dict = ut.from_json(request.form['ia-contour-data'])

        if data_dict is None:
            data_dict = {}

        contour_dict = ibs.get_part_contour(part_rowid)
        contour_dict['contour'] = data_dict
        ibs.set_part_contour([part_rowid], [contour_dict])

        segment = data_dict.get('segment', [])
        num_contours = 1
        num_points = len(segment)
        ibs.set_part_reviewed([part_rowid], [1])

        print(
            '[web] part_rowid: %d, contours: %d, points: %d'
            % (part_rowid, num_contours, num_points,)
        )

    default_list = ['temp']
    config = {default: kwargs[default] for default in default_list if default in kwargs}

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(
            url_for('turk_contour', imgsetid=imgsetid, previous=part_rowid, **config)
        )


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.app
        python -m wbia.web.app --allexamples
        python -m wbia.web.app --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
