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


register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/submit/detection/', methods=['POST'])
def submit_detection():
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
        # Make new annotations
        width, height = ibs.get_image_sizes(gid)
        # Get aids
        annotation_list = json.loads(request.form['detection-annotations'])
        bbox_list = [
            (
                int( width  * (annot['left']   / 100.0) ),
                int( height * (annot['top']    / 100.0) ),
                int( width  * (annot['width']  / 100.0) ),
                int( height * (annot['height'] / 100.0) ),
            )
            for annot in annotation_list
        ]
        theta_list = [
            float(annot['theta'])
            for annot in annotation_list
        ]
        survived_aid_list = [
            None if annot['id'] is None else int(annot['id'])
            for annot in annotation_list
        ]
        species_list = [
            annot['label']
            for annot in annotation_list
        ]
        # Delete annotations that didn't survive
        kill_aid_list = list(set(current_aid_list) - set(survived_aid_list))
        ibs.delete_annots(kill_aid_list)
        for aid, bbox, theta, species in zip(survived_aid_list, bbox_list, theta_list, species_list):
            if aid is None:
                ibs.add_annots([gid], [bbox], theta_list=[theta], species_list=[species])
            else:
                ibs.set_annot_bboxes([aid], [bbox])
                ibs.set_annot_thetas([aid], [theta])
                ibs.set_annot_species([aid], [species])
        ibs.set_image_reviewed([gid], [1])
        print('[web] turk_id: %s, gid: %d, bbox_list: %r, species_list: %r' % (turk_id, gid, annotation_list, species_list))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_detection', imgsetid=imgsetid, previous=gid))


@register_route('/submit/viewpoint/', methods=['POST'])
def submit_viewpoint():
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
        value = int(request.form['viewpoint-value'])
        yaw = appf.convert_old_viewpoint_to_yaw(value)
        species_text = request.form['viewpoint-species']
        ibs.set_annot_yaws([aid], [yaw], input_is_degrees=False)
        ibs.set_annot_species([aid], [species_text])
        print('[web] turk_id: %s, aid: %d, yaw: %d' % (turk_id, aid, yaw))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_viewpoint', imgsetid=imgsetid, src_ag=src_ag,
                                dst_ag=dst_ag, previous=aid))


@register_route('/submit/annotation/', methods=['POST'])
def submit_annotation():
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
    elif method.lower() == 'rotate left':
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
    elif method.lower() == 'rotate right':
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
            viewpoint = int(request.form['ia-viewpoint-value'])
        except ValueError:
            viewpoint = int(float(request.form['ia-viewpoint-value']))
        yaw = appf.convert_old_viewpoint_to_yaw(viewpoint)
        species_text = request.form['ia-annotation-species']
        ibs.set_annot_yaws([aid], [yaw], input_is_degrees=False)
        ibs.set_annot_species([aid], [species_text])
        try:
            quality = int(request.form['ia-quality-value'])
        except ValueError:
            quality = int(float(request.form['ia-quality-value']))
        if quality == 1:
            quality = 2
        elif quality == 2:
            quality = 4
        else:
            raise ValueError('quality must be 1 or 2')
        ibs.set_annot_qualities([aid], [quality])
        multiple = 1 if 'ia-multiple-value' in request.form else 0
        ibs.set_annot_multiple([aid], [multiple])
        ibs.set_annot_reviewed([aid], [1])
        print('[web] turk_id: %s, aid: %d, yaw: %d, quality: %d, multiple: %r' % (turk_id, aid, yaw, quality, multiple))
    # Return HTML
    refer = request.args.get('refer', '')
    if len(refer) > 0:
        return redirect(appf.decode_refer_url(refer))
    else:
        return redirect(url_for('turk_annotation', imgsetid=imgsetid, src_ag=src_ag,
                                dst_ag=dst_ag, previous=aid))


@register_route('/submit/quality/', methods=['POST'])
def submit_quality():
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


@register_route('/submit/additional/', methods=['POST'])
def submit_additional():
    ibs = current_app.ibs
    method = request.form.get('additional-submit', '')
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    aid = int(request.form['additional-aid'])
    turk_id = request.cookies.get('ia-turk_id', -1)

    if method.lower() == 'delete':
        ibs.delete_annots(aid)
        print('[web] (DELETED) turk_id: %s, aid: %d' % (turk_id, aid, ))
        aid = None  # Reset AID to prevent previous
    else:
        sex = int(request.form['additional-sex-value'])
        age = int(request.form['additional-age-value'])
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
        return redirect(url_for('turk_additional', imgsetid=imgsetid, previous=aid))


@register_route('/submit/group_review/', methods=['POST'])
def group_review_submit():
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
