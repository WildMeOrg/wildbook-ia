# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
import simplejson as json
from flask import request, redirect, url_for, current_app
from ibeis.control import controller_inject
from ibeis.web import appfuncs as appf
from ibeis import constants as const
from ibeis.constants import PI, TAU
import utool as ut
import numpy as np


register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/submit/detection/', methods=['POST'])
def submit_detection(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('detection-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    gid = int(request.form['detection-gid'])
    turk_id = request.cookies.get('ia-turk_id', -1)

    if method.lower() == 'delete':
        # ibs.delete_images(gid)
        # print('[web] (DELETED) turk_id: %s, gid: %d' % (turk_id, gid, ))
        pass
    elif method.lower() == 'clear':
        aid_list = ibs.get_image_aids(gid)
        ibs.delete_annots(aid_list)
        print('[web] (CLEAERED) turk_id: %s, gid: %d' % (turk_id, gid, ))
        redirection = request.referrer
        if 'gid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&gid=%d' % (redirection, gid, )
            else:
                redirection = '%s?gid=%d' % (redirection, gid, )
        return redirect(redirection)
    else:
        current_aid_list = ibs.get_image_aids(gid)
        current_part_rowid_list = ut.flatten(ibs.get_annot_part_rowids(current_aid_list))
        # Make new annotations
        width, height = ibs.get_image_sizes(gid)

        # Separate out annotations vs parts
        data_list = json.loads(request.form['ia-detection-data'])
        annotation_list = []
        part_list = []
        mapping_dict = {}
        for outer_index, data in enumerate(data_list):
            parent_index = data['parent']
            if parent_index is None:
                inner_index = len(annotation_list)
                annotation_list.append(data)
                mapping_dict[outer_index] = inner_index
            else:
                assert parent_index in mapping_dict
                part_list.append(data)

        ##################################################################################
        # Process annotations
        survived_aid_list = [
            None if annot['label'] is None else int(annot['label'])
            for annot in annotation_list
        ]

        # Get primatives
        bbox_list = [
            (
                int(np.round(width  * (annot['percent']['left']   / 100.0) )),
                int(np.round(height * (annot['percent']['top']    / 100.0) )),
                int(np.round(width  * (annot['percent']['width']  / 100.0) )),
                int(np.round(height * (annot['percent']['height'] / 100.0) )),
            )
            for annot in annotation_list
        ]
        theta_list = [
            float(annot['angles']['theta'])
            for annot in annotation_list
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
        viewpoint_list = [ appf.convert_tuple_to_viewpoint(tup) for tup in zipped ]

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

        multiple_list =  [
            annot['metadata'].get('multiple', False)
            for annot in annotation_list
        ]
        interest_list =  [
            annot['highlighted']
            for annot in annotation_list
        ]
        species_list = [
            annot['metadata'].get('species', const.UNKNOWN)
            for annot in annotation_list
        ]

        # Delete annotations that didn't survive
        kill_aid_list = list(set(current_aid_list) - set(survived_aid_list))
        ibs.delete_annots(kill_aid_list)

        aid_list = []
        for aid, bbox in zip(survived_aid_list, bbox_list):
            if aid is None:
                aid_ = ibs.add_annots([gid], [bbox])[0]
            else:
                ibs.set_annot_bboxes([aid], [bbox])
                aid_ = aid
            aid_list.append(aid_)

        print('aid_list = %r' % (aid_list, ))
        # Set annotation metadata
        ibs.set_annot_thetas(aid_list, theta_list)
        ibs.set_annot_viewpoints(aid_list, viewpoint_list)
        ibs.set_annot_qualities(aid_list, quality_list)
        ibs.set_annot_multiple(aid_list, multiple_list)
        ibs.set_annot_interest(aid_list, interest_list)
        ibs.set_annot_species(aid_list, species_list)

        # Set the mapping dict to use aids now
        mapping_dict = { key: aid_list[index] for key, index in mapping_dict.items() }

        ##################################################################################
        # Process parts
        survived_part_rowid_list = [
            None if part['label'] is None else int(part['label'])
            for part in part_list
        ]

        # Get primatives
        aid_list = [
            mapping_dict[part['parent']]
            for part in part_list
        ]
        bbox_list = [
            (
                int(np.round(width  * (part['percent']['left']   / 100.0) )),
                int(np.round(height * (part['percent']['top']    / 100.0) )),
                int(np.round(width  * (part['percent']['width']  / 100.0) )),
                int(np.round(height * (part['percent']['height'] / 100.0) )),
            )
            for part in part_list
        ]
        theta_list = [
            float(part['angles']['theta'])
            for part in part_list
        ]

        # Get metadata
        viewpoint1_list = [
            int(part['metadata'].get('viewpoint1', -1))
            for part in part_list
        ]
        viewpoint2_list = [-1] * len(part_list)
        viewpoint3_list = [-1] * len(part_list)
        zipped = zip(viewpoint1_list, viewpoint2_list, viewpoint3_list)
        viewpoint_list = [ appf.convert_tuple_to_viewpoint(tup) for tup in zipped ]

        quality_list = [
            int(part['metadata'].get('quality', 0))
            for part in part_list
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
            part['metadata'].get('type', const.UNKNOWN)
            for part in part_list
        ]

        # Delete annotations that didn't survive
        kill_part_rowid_list = list(set(current_part_rowid_list) - set(survived_part_rowid_list))
        ibs.delete_parts(kill_part_rowid_list)

        part_rowid_list = []
        for part_rowid, aid, bbox in zip(survived_part_rowid_list, aid_list, bbox_list):
            if part_rowid is None:
                part_rowid_ = ibs.add_parts([aid], [bbox])
                part_rowid_ = part_rowid_[0]
            else:
                ibs._set_part_aid([part_rowid], [aid])
                ibs.set_part_bboxes([part_rowid], [bbox])
                part_rowid_ = part_rowid
            part_rowid_list.append(part_rowid_)

        # Set annotation metadata
        ibs.set_part_thetas(part_rowid_list, theta_list)
        ibs.set_part_viewpoints(part_rowid_list, viewpoint_list)
        ibs.set_part_qualities(part_rowid_list, quality_list)
        ibs.set_part_types(part_rowid_list, type_list)

        # Set image reviewed flag
        ibs.set_image_reviewed([gid], [1])
        print('[web] turk_id: %s, gid: %d, annots: %d, parts: %d' % (turk_id, gid, len(annotation_list), len(part_list), ))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_detection', imgsetid=imgsetid, previous=gid))


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
    turk_id = request.cookies.get('ia-turk_id', -1)
    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) turk_id: %s, aid: %d' % (turk_id, aid, ))
        aid = None  # Reset AID to prevent previous
    if method.lower() == 'make junk':
        ibs.set_annot_quality_texts([aid], [const.QUAL_JUNK])
        print('[web] (SET AS JUNK) turk_id: %s, aid: %d' % (turk_id, aid, ))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid, )
            else:
                redirection = '%s?aid=%d' % (redirection, aid, )
        return redirect(redirection)
    if method.lower() == 'rotate left':
        theta = ibs.get_annot_thetas(aid)
        theta = (theta + PI / 2) % TAU
        ibs.set_annot_thetas(aid, theta)
        (xtl, ytl, w, h) = ibs.get_annot_bboxes(aid)
        diffx = int(round((w / 2.0) - (h / 2.0)))
        diffy = int(round((h / 2.0) - (w / 2.0)))
        xtl, ytl, w, h = xtl + diffx, ytl + diffy, h, w
        ibs.set_annot_bboxes([aid], [(xtl, ytl, w, h)])
        print('[web] (ROTATED LEFT) turk_id: %s, aid: %d' % (turk_id, aid, ))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid, )
            else:
                redirection = '%s?aid=%d' % (redirection, aid, )
        return redirect(redirection)
    if method.lower() == 'rotate right':
        theta = ibs.get_annot_thetas(aid)
        theta = (theta - PI / 2) % TAU
        ibs.set_annot_thetas(aid, theta)
        (xtl, ytl, w, h) = ibs.get_annot_bboxes(aid)
        diffx = int(round((w / 2.0) - (h / 2.0)))
        diffy = int(round((h / 2.0) - (w / 2.0)))
        xtl, ytl, w, h = xtl + diffx, ytl + diffy, h, w
        ibs.set_annot_bboxes([aid], [(xtl, ytl, w, h)])
        print('[web] (ROTATED RIGHT) turk_id: %s, aid: %d' % (turk_id, aid, ))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid, )
            else:
                redirection = '%s?aid=%d' % (redirection, aid, )
        return redirect(redirection)
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)
        viewpoint = int(request.form['viewpoint-value'])
        viewpoint_text = appf.VIEWPOINT_MAPPING.get(viewpoint, None)
        species_text = request.form['viewpoint-species']
        ibs.set_annot_viewpoints([aid], [viewpoint_text])
        ibs.set_annot_species([aid], [species_text])
        print('[web] turk_id: %s, aid: %d, viewpoint_text: %s' % (turk_id, aid, viewpoint_text))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_viewpoint', imgsetid=imgsetid, src_ag=src_ag,
                                dst_ag=dst_ag, previous=aid))


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
    turk_id = request.cookies.get('ia-turk_id', -1)
    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) turk_id: %s, aid: %d' % (turk_id, aid, ))
        aid = None  # Reset AID to prevent previous
    elif method.lower() == 'make junk':
        ibs.set_annot_quality_texts([aid], [const.QUAL_JUNK])
        print('[web] (SET AS JUNK) turk_id: %s, aid: %d' % (turk_id, aid, ))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid, )
            else:
                redirection = '%s?aid=%d' % (redirection, aid, )
        return redirect(redirection)
    elif method.lower() == u'left 90\xb0':
        theta = ibs.get_annot_thetas(aid)
        theta = (theta + PI / 2) % TAU
        ibs.set_annot_thetas(aid, theta)
        (xtl, ytl, w, h) = ibs.get_annot_bboxes(aid)
        diffx = int(round((w / 2.0) - (h / 2.0)))
        diffy = int(round((h / 2.0) - (w / 2.0)))
        xtl, ytl, w, h = xtl + diffx, ytl + diffy, h, w
        ibs.set_annot_bboxes([aid], [(xtl, ytl, w, h)])
        print('[web] (ROTATED LEFT) turk_id: %s, aid: %d' % (turk_id, aid, ))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid, )
            else:
                redirection = '%s?aid=%d' % (redirection, aid, )
        return redirect(redirection)
    elif method.lower() == u'right 90\xb0':
        theta = ibs.get_annot_thetas(aid)
        theta = (theta - PI / 2) % TAU
        ibs.set_annot_thetas(aid, theta)
        (xtl, ytl, w, h) = ibs.get_annot_bboxes(aid)
        diffx = int(round((w / 2.0) - (h / 2.0)))
        diffy = int(round((h / 2.0) - (w / 2.0)))
        xtl, ytl, w, h = xtl + diffx, ytl + diffy, h, w
        ibs.set_annot_bboxes([aid], [(xtl, ytl, w, h)])
        print('[web] (ROTATED RIGHT) turk_id: %s, aid: %d' % (turk_id, aid, ))
        redirection = request.referrer
        if 'aid' not in redirection:
            # Prevent multiple clears
            if '?' in redirection:
                redirection = '%s&aid=%d' % (redirection, aid, )
            else:
                redirection = '%s?aid=%d' % (redirection, aid, )
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
        ibs.set_annot_species([aid], [species_text])
        ibs.set_annot_qualities([aid], [quality])
        multiple = 1 if 'ia-multiple-value' in request.form else 0
        ibs.set_annot_multiple([aid], [multiple])
        ibs.set_annot_reviewed([aid], [1])
        print('[web] turk_id: %s, aid: %d, viewpoint: %r, quality: %r, multiple: %r' % (turk_id, aid, viewpoint_text, quality, multiple))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_annotation', imgsetid=imgsetid, src_ag=src_ag,
                                dst_ag=dst_ag, previous=aid))


@register_route('/submit/quality/', methods=['POST'])
def submit_quality(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('quality-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    aid = int(request.form['quality-aid'])
    turk_id = request.cookies.get('ia-turk_id', -1)

    src_ag = request.args.get('src_ag', '')
    src_ag = None if src_ag == 'None' or src_ag == '' else int(src_ag)
    dst_ag = request.args.get('dst_ag', '')
    dst_ag = None if dst_ag == 'None' or dst_ag == '' else int(dst_ag)

    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) turk_id: %s, aid: %d' % (turk_id, aid, ))
        aid = None  # Reset AID to prevent previous
    else:
        if src_ag is not None and dst_ag is not None:
            appf.movegroup_aid(ibs, aid, src_ag, dst_ag)
        quality = int(request.form['quality-value'])
        ibs.set_annot_qualities([aid], [quality])
        print('[web] turk_id: %s, aid: %d, quality: %d' % (turk_id, aid, quality))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_quality', imgsetid=imgsetid, src_ag=src_ag,
                                dst_ag=dst_ag, previous=aid))


@register_route('/submit/demographics/', methods=['POST'])
def submit_demographics(**kwargs):
    ibs = current_app.ibs
    method = request.form.get('demographics-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    aid = int(request.form['demographics-aid'])
    turk_id = request.cookies.get('ia-turk_id', -1)

    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) turk_id: %s, aid: %d' % (turk_id, aid, ))
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
        DAN_SPECIAL_WRITE_AGE_TO_ALL_ANOTATIONS = True
        if nid is not None and DAN_SPECIAL_WRITE_AGE_TO_ALL_ANOTATIONS:
            aid_list = ibs.get_name_aids(nid)
            ibs.set_annot_age_months_est_min(aid_list, [age_min] * len(aid_list))
            ibs.set_annot_age_months_est_max(aid_list, [age_max] * len(aid_list))
        else:
            ibs.set_annot_age_months_est_min([aid], [age_min])
            ibs.set_annot_age_months_est_max([aid], [age_max])
        print('[web] turk_id: %s, aid: %d, sex: %r, age: %r' % (turk_id, aid, sex, age))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_demographics', imgsetid=imgsetid, previous=aid))


@register_route('/submit/identification/', methods=['POST'])
def submit_identification(**kwargs):
    from ibeis.web.apis_query import process_graph_match_html
    ibs = current_app.ibs

    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    aid1 = int(request.form['identification-aid1'])
    aid2 = int(request.form['identification-aid2'])
    replace_review_rowid = int(request.form.get('identification-replace-review-rowid', -1))

    # Process form data
    annot_uuid_1, annot_uuid_2, state, tag_list = process_graph_match_html(ibs)

    # Add state to staging database
    # FIXME:
    # photobomb and scenerymatch tags should be disjoint from match-state
    if state == 'matched':
        decision = const.REVIEW.POSITIVE
    elif state == 'notmatched':
        decision = const.REVIEW.NEGATIVE
    elif state == 'notcomparable':
        decision = const.REVIEW.INCOMPARABLE
    elif state == 'photobomb':
        decision = const.REVIEW.NEGATIVE
        tag_list = ['photobomb']
    elif state == 'scenerymatch':
        decision = const.REVIEW.NEGATIVE
        tag_list = ['scenerymatch']
    else:
        raise ValueError()

    # Replace a previous decision
    if replace_review_rowid > 0:
        print('REPLACING OLD REVIEW ID = %r' % (replace_review_rowid, ))
        ibs.delete_review([replace_review_rowid])

    # Add a new review row for the new decision (possibly replacing the old one)
    print('ADDING REVIEW: %r %r %r %r' % (aid1, aid2, decision, tag_list, ))
    tags_list = None if tag_list is None else [tag_list]
    review_rowid = ibs.add_review([aid1], [aid2], [decision], tags_list=tags_list)
    review_rowid = review_rowid[0]
    previous = '%s;%s;%s' % (aid1, aid2, review_rowid, )

    # Notify any attached web QUERY_OBJECT
    try:
        state = const.REVIEW.INT_TO_CODE[decision]
        feedback = (aid1, aid2, state, tags_list)
        print('Adding %r to QUERY_OBJECT_FEEDBACK_BUFFER' % (feedback, ))
        current_app.QUERY_OBJECT_FEEDBACK_BUFFER.append(feedback)
    except ValueError:
        pass

    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_identification', imgsetid=imgsetid, previous=previous))


@register_route('/submit/group_review/', methods=['POST'])
def group_review_submit(**kwargs):
    """
    CommandLine:
        python -m ibeis.web.app --exec-group_review_submit

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from ibeis.web.app import *  # NOQA
        >>> import ibeis
        >>> import ibeis.web
        >>> ibs = ibeis.opendb('testdb1')
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
                redirection = '%s&prefill=true' % (redirection, )
            else:
                redirection = '%s?prefill=true' % (redirection, )
        return redirect(redirection)
    aid_list = request.form.get('aid_list', '')
    if len(aid_list) > 0:
        aid_list = aid_list.replace('[', '')
        aid_list = aid_list.replace(']', '')
        aid_list = aid_list.strip().split(',')
        aid_list = [ int(aid_.strip()) for aid_ in aid_list ]
    else:
        aid_list = []
    src_ag, dst_ag = ibs.prepare_annotgroup_review(aid_list)
    valid_modes = ut.get_list_column(appf.VALID_TURK_MODES, 0)
    mode = request.form.get('group-review-mode', None)
    assert mode in valid_modes
    return redirect(url_for(mode, src_ag=src_ag, dst_ag=dst_ag))


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
